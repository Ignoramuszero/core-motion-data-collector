#!/usr/bin/env python3
"""Offline activity-state labelling for a Sensor Logger motion recording.

Takes the app's Motion.csv export (columns: time, accelerationX..Z [g],
rotationRateX..Z [rad/s], gravityX..Z [g], ...) and runs the SAME
event-driven policy as the iOS app / WISDM experiments:

    every 1 s : 1 s burst  -> BurstClassifier  (cheap estimate)
    every 3 s : 3 s window -> ActivityClassifier (committed label)
    every 6 s : burst history -> EventDetector (RF >= 0.5 OR rule z >= 4.0)
    on event  : force immediate re-classification, reset burst history

Unit handling: Sensor Logger stores acceleration in g, the models were
trained on WISDM data in m/s^2 -> total accel = (userAccel + gravity) * 9.80665.
The 100 Hz stream is resampled to the models' native 20 Hz.

Outputs (into --out-dir):
    timeline.csv    per-second decisions (cheap + committed + event info)
    segments.csv    contiguous activity segments (start, end, duration, label)
    summary.json    totals per activity + event log
    timeline.html   self-contained visual timeline (no dependencies)

Usage:
    python scripts/process_recording.py Motion.csv --out-dir artifacts/timeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import (  # noqa: E402
    ACTIVITY_LABEL_CN,
    ACTIVITIES,
)
from swn.features import burst_stats, burst_z  # noqa: E402
from swn.models.activity_classifier import ActivityClassifier  # noqa: E402
from swn.models.event_detector import EventDetector  # noqa: E402

FS_OUT = 20.0            # models' native rate
G_TO_MS2 = 9.80665
EVENT_THRESHOLD = 0.50   # RF branch
RULE_Z_THRESHOLD = 4.0   # rule branch
CONF_HIGH = 0.85
CONF_LOW = 0.60
MAIN_PERIOD_S = 3
EVENT_PERIOD_S = 6
MAX_HISTORY = 4

COLORS = ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6", "#10b981"]  # walking/jogging/stairs/sitting/standing


# ----------------------------------------------------------------- loading ----

def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, acc_total_ms2[N,3], gyro[N,3]) resampled to 20 Hz."""
    t, acc, grav, gyro = [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        cols = {c.strip().lower(): c.strip() for c in (reader.fieldnames or [])}
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            t.append(float(row["time"]))
            a = np.array([float(row[f"acceleration{ax}"]) for ax in "XYZ"])
            g = np.array([float(row[f"gravity{ax}"]) for ax in "XYZ"])
            w = np.array([float(row[f"rotationRate{ax}"]) for ax in "XYZ"])
            acc.append(a)
            grav.append(g)
            gyro.append(w)

    t = np.asarray(t, dtype=np.float64)
    acc = np.asarray(acc)
    grav = np.asarray(grav)
    gyro = np.asarray(gyro)

    # WISDM-trained models expect total acceleration in m/s^2.
    total = (acc + grav) * G_TO_MS2

    # Resample the (irregular) native stream onto a uniform 20 Hz grid.
    t_max = float(t[-1])
    grid = np.arange(0.0, t_max, 1.0 / FS_OUT)
    out_acc = np.stack([np.interp(grid, t, total[:, i]) for i in range(3)], axis=1)
    out_gyro = np.stack([np.interp(grid, t, gyro[:, i]) for i in range(3)], axis=1)
    native_rate = len(t) / t_max
    print(f"[load] {path.name}: {len(t)} samples over {t_max:.1f} s "
          f"(native ~{native_rate:.0f} Hz) -> {len(grid)} samples @ 20 Hz")
    return grid, out_acc, out_gyro


# ------------------------------------------------------------------ policy ----

def run_policy(t: np.ndarray, acc: np.ndarray, gyro: np.ndarray,
               main_clf: ActivityClassifier, burst_clf: ActivityClassifier,
               detector: EventDetector | None) -> list[dict]:
    rows: list[dict] = []
    history: list[np.ndarray] = []
    burst_count = 0
    committed: tuple[int, float] | None = None
    force_main = False
    last_log_t: float | None = None

    n = len(t)
    for s in range(0, n // int(FS_OUT)):
        b0, b1 = s * 20, s * 20 + 20
        if b1 > n:
            break
        burst_acc = acc[b0:b1]
        burst_gyro = gyro[b0:b1]

        # cheap 1 s classification
        k, kconf, _ = burst_clf.predict_window(burst_acc, burst_gyro)

        # high-fidelity 3 s window (periodic or forced by an event)
        due_main = (s % MAIN_PERIOD_S == 0) or force_main
        changed = False
        if due_main:
            force_main = False
            if s * 20 + 60 <= n:
                label, conf, _ = main_clf.predict_window(acc[s * 20:s * 20 + 60],
                                                         gyro[s * 20:s * 20 + 60])
                prev = committed[0] if committed else None
                should_commit = conf >= CONF_HIGH or (
                    conf >= CONF_LOW and (prev is None or prev == label))
                if should_commit:
                    committed = (label, conf)
                    if label != prev:
                        changed = True
                        last_log_t = float(t[b0])

        # event detection every 6 s: bursts spaced 6 s apart, matching the
        # detector's training distribution (IDLE_PERIOD_S); the 1 s cheap
        # classification above runs independently and much more often.
        p_change, z, fired = 0.0, 0.0, False
        if s % EVENT_PERIOD_S == 0:
            history.append(burst_stats(burst_acc, FS_OUT))
            if len(history) > MAX_HISTORY:
                history.pop(0)
            burst_count += 1
            if len(history) >= 2:
                z = float(burst_z(np.stack(history)))
                if detector is not None:
                    p_change = float(detector.prob(np.stack(history)))
                fired = p_change >= EVENT_THRESHOLD or z >= RULE_Z_THRESHOLD
                if fired:
                    force_main = True
                    history.clear()

        rows.append({"t": float(t[b0]), "cheap": k, "cheap_conf": kconf,
                     "committed": committed[0] if committed else None,
                     "conf": committed[1] if committed else 0.0,
                     "p_change": p_change, "z": z, "event": fired, "changed": changed})
    return rows


def build_segments(rows: list[dict]) -> list[dict]:
    """Per-second final label = committed if available else cheap estimate."""
    per_second = []
    for r in rows:
        label = r["committed"] if r["committed"] is not None else r["cheap"]
        per_second.append((r["t"], int(label)))

    segments: list[dict] = []
    for t, label in per_second:
        if segments and segments[-1]["label"] == label and t - segments[-1]["end"] <= 1.5:
            segments[-1]["end"] = t + 1.0
        else:
            segments.append({"start": t, "end": t + 1.0, "label": label})
    # drop slivers < 2 s (single-window misfires)
    return [s for s in segments if s["end"] - s["start"] >= 2.0]


# ----------------------------------------------------------------- outputs ----

def write_outputs(rows: list[dict], segments: list[dict], out_dir: Path, motion_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "cheap_label", "cheap_conf", "committed_label",
                    "committed_conf", "p_change", "rule_z", "event_fired", "label_used"])
        for r in rows:
            label = r["committed"] if r["committed"] is not None else r["cheap"]
            w.writerow([f"{r['t']:.2f}", ACTIVITIES[r["cheap"]], f"{r['cheap_conf']:.3f}",
                        ACTIVITIES[r["committed"]] if r["committed"] is not None else "",
                        f"{r['conf']:.3f}", f"{r['p_change']:.3f}", f"{r['z']:.2f}",
                        int(r["event"]), ACTIVITIES[label]])

    with open(out_dir / "segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "duration_s", "label", "label_cn"])
        for s in segments:
            w.writerow([f"{s['start']:.1f}", f"{s['end']:.1f}",
                        f"{s['end'] - s['start']:.1f}",
                        ACTIVITIES[s["label"]], ACTIVITY_LABEL_CN[ACTIVITIES[s["label"]]]])

    duration = rows[-1]["t"] + 1.0 if rows else 0.0
    per_activity = {a: 0.0 for a in ACTIVITIES}
    for s in segments:
        per_activity[ACTIVITIES[s["label"]]] += s["end"] - s["start"]
    events = [r for r in rows if r["event"]]
    summary = {
        "source": str(motion_csv),
        "duration_s": round(duration, 1),
        "per_activity_s": {a: round(v, 1) for a, v in per_activity.items()},
        "per_activity_pct": {a: round(100 * v / duration, 1) if duration else 0.0
                             for a, v in per_activity.items()},
        "n_events": len(events),
        "event_times_s": [round(r["t"], 1) for r in events],
        "segments": [{"start": round(s["start"], 1), "end": round(s["end"], 1),
                      "label": ACTIVITIES[s["label"]]} for s in segments],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out_dir / "timeline.html").write_text(render_html(rows, segments, summary), encoding="utf-8")
    return summary


def render_html(rows: list[dict], segments: list[dict], summary: dict) -> str:
    duration = summary["duration_s"] or 1.0
    W, H, BAND_Y, BAND_H, MAG_H = 900, 320, 60, 34, 90

    # rule-branch z scores as the activity line
    zs = np.array([r["z"] for r in rows], dtype=float)
    zs_scaled = np.clip(zs / max(zs.max(), 1e-6), 0, 1)

    band = []
    for s in segments:
        x0 = s["start"] / duration * W
        x1 = s["end"] / duration * W
        label = ACTIVITIES[s["label"]]
        band.append(
            f'<rect x="{x0:.1f}" y="{BAND_Y}" width="{max(x1 - x0, 2):.1f}" height="{BAND_H}" '
            f'fill="{COLORS[s["label"]]}" opacity="0.9"/>'
            + (f'<text x="{(x0 + x1) / 2:.1f}" y="{BAND_Y + 22}" text-anchor="middle" '
               f'font-size="11" fill="#fff">{escape(ACTIVITY_LABEL_CN[label])}</text>'
               if x1 - x0 > 44 else "")
        )
    for r in rows:
        if r["event"]:
            x = r["t"] / duration * W
            band.append(f'<line x1="{x:.1f}" y1="{BAND_Y - 12}" x2="{x:.1f}" '
                        f'y2="{BAND_Y + BAND_H + 12}" stroke="#f97316" stroke-width="2"/>')

    spark = []
    step = max(1, len(zs_scaled) // 400)
    pts = [(rows[i]["t"] / duration * W, MAG_H - zs_scaled[i] * (MAG_H - 8)) for i in range(0, len(rows), step)]
    if pts:
        spark.append('<polyline fill="none" stroke="#6366f1" stroke-width="1.5" points="'
                     + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">'
        f'<span style="width:12px;height:12px;background:{c};border-radius:2px;display:inline-block;"></span>'
        f'{escape(ACTIVITY_LABEL_CN[a])}</span>'
        for a, c in zip(ACTIVITIES, COLORS)
    )
    bars = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
        f'<div style="width:64px;text-align:right;color:#555;font-size:13px;">{escape(ACTIVITY_LABEL_CN[a])}</div>'
        f'<div style="flex:1;background:#eef1f5;border-radius:4px;height:18px;position:relative;">'
        f'<div style="width:{summary["per_activity_pct"][a]:.1f}%;height:100%;background:{c};'
        f'border-radius:4px;"></div></div>'
        f'<div style="width:110px;font-size:13px;color:#333;">{summary["per_activity_pct"][a]:.1f}%'
        f'（{summary["per_activity_s"][a]:.0f}s）</div></div>'
        for a, c in zip(ACTIVITIES, COLORS)
    )
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>活动时间线</title></head>
<body style="font-family:-apple-system,'Segoe UI',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1f2937;">
<h2>活动识别时间线（离线策略回放）</h2>
<p style="color:#555;font-size:14px;">数据源：{escape(summary["source"])} · 时长 {summary["duration_s"]:.0f}s ·
事件触发 {summary["n_events"]} 次（橙色竖线） · 折线为规则分支 z 分数（归一化）</p>
<div>{legend}</div>
<svg viewBox="0 0 {W} {H}" width="100%" style="background:#fafbfc;border:1px solid #e5e7eb;border-radius:8px;margin-top:12px;">
  <text x="8" y="{BAND_Y - 16}" font-size="12" fill="#666">识别状态</text>
  <text x="8" y="{MAG_H - 74}" font-size="12" fill="#666">z 分数</text>
  {''.join(spark)}
  {''.join(band)}
  <line x1="0" y1="{BAND_Y + BAND_H + 14}" x2="{W}" y2="{BAND_Y + BAND_H + 14}" stroke="#d1d5db"/>
  <text x="2" y="{BAND_Y + BAND_H + 30}" font-size="11" fill="#888">0s</text>
  <text x="{W - 34}" y="{BAND_Y + BAND_H + 30}" font-size="11" fill="#888">{duration:.0f}s</text>
</svg>
<h3>各活动占比</h3>
<div>{bars}</div>
</body></html>"""


# --------------------------------------------------------------------- main ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("motion_csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "timeline")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "artifacts" / "models")
    args = ap.parse_args()

    main_clf = ActivityClassifier.load(args.models_dir / "activity_classifier.pkl")
    burst_clf = ActivityClassifier.load(args.models_dir / "burst_classifier.pkl")
    try:
        detector = EventDetector.load(args.models_dir / "event_detector.pkl")
    except Exception as exc:  # rule branch still works without the RF
        print(f"[warn] EventDetector unavailable ({exc}); rule branch only")
        detector = None

    t, acc, gyro = load_motion(args.motion_csv)
    rows = run_policy(t, acc, gyro, main_clf, burst_clf, detector)
    segments = build_segments(rows)
    summary = write_outputs(rows, segments, args.out_dir, args.motion_csv)

    print(f"\n[done] {len(rows)} decisions, {len(segments)} segments, "
          f"{summary['n_events']} events -> {args.out_dir}")
    print("占比：")
    for a in ACTIVITIES:
        print(f"  {ACTIVITY_LABEL_CN[a]:<4} {summary['per_activity_pct'][a]:5.1f}%  "
              f"({summary['per_activity_s'][a]:.0f}s)")
    print("分段：")
    for s in summary["segments"]:
        print(f"  {s['start']:6.1f} - {s['end']:6.1f}s  {ACTIVITY_LABEL_CN[s['label']]}")


if __name__ == "__main__":
    main()
