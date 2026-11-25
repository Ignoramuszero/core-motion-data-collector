#!/usr/bin/env python3
"""Personalize the SenseWhenNeeded models with a labelled self-recording.

The WISDM models were trained on wrist-watch IMU data; a phone recording
(Sensor Logger export) lives in a different feature distribution and the
classifier locks onto the wrong label (e.g. walking -> stairs).  This script
retrains all three models on WISDM *train* subjects + the personal windows
(the user's ground-truth annotated recording), then verifies:

  1. no regression on the untouched WISDM *test* subjects;
  2. the personal recording is now classified correctly (in-sample fit).

Ground truth intervals for the recording (user-provided):
     35 -  95 s   stairs   (downstairs)
     95 - 180 s   walking  (flat ground)
    180 - end s   sitting
      0 -  35 s   unknown  -> excluded from classifier training, used as
                                 "standing" only for the event-detector stream

Usage:
    python scripts/personalize.py Motion.csv \
        --sessions-npz <path>/SenseWhenNeeded/data/processed/sessions.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import ACTIVITIES, BURST_WINDOW_S  # noqa: E402
from swn.data.sessions import Session, split_subjects  # noqa: E402
from swn.features import burst_stats, window_features  # noqa: E402
from swn.models.activity_classifier import ActivityClassifier, build_windows  # noqa: E402
from swn.models.event_detector import EventDetector, build_event_dataset, burst_histories  # noqa: E402

# user-provided ground truth (seconds -> activity code in ACTIVITIES order)
TRUTH_INTERVALS = [(35.0, 95.0, 2),   # stairs
                   (95.0, 180.0, 0),  # walking
                   (180.0, 1e9, 3)]   # sitting

FS_OUT = 20.0


# ------------------------------------------------------------- data loading ----

def load_wisdm_sessions(npz_path: Path) -> list[Session]:
    """Rebuild Session objects from the cached processed sessions."""
    d = np.load(npz_path, allow_pickle=True)
    sessions = []
    start = 0
    for sid, subj, length in zip(d["ids"], d["subjects"], d["lengths"]):
        sl = slice(start, start + int(length))
        labels = d["labels"][sl].astype(np.int8)
        change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        transitions = [(float(i) / FS_OUT, int(labels[i - 1]), int(labels[i]))
                       for i in change if i / FS_OUT >= 1.0]
        sessions.append(Session(session_id=str(sid), subject=int(subj),
                                acc=d["acc"][sl].astype(np.float32),
                                gyro=d["gyro"][sl].astype(np.float32),
                                labels=labels, transitions=transitions))
        start += int(length)
    return sessions


def load_personal(motion_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sensor Logger CSV -> (t, acc, gyro) at 20 Hz, m/s^2 + rad/s."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from process_recording import load_motion  # reuse the loader + unit conversion
    return load_motion(motion_csv)


def personal_session(t: np.ndarray, acc: np.ndarray, gyro: np.ndarray) -> Session:
    labels = np.full(len(t), 4, dtype=np.int8)   # 0-35 s assumed standing
    for lo, hi, code in TRUTH_INTERVALS:
        labels[(t >= lo) & (t < hi)] = code
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    transitions = [(float(i) / FS_OUT, int(labels[i - 1]), int(labels[i]))
                   for i in change if i / FS_OUT >= 1.0]
    return Session(session_id="personal", subject=999,
                   acc=acc.astype(np.float32), gyro=gyro.astype(np.float32),
                   labels=labels, transitions=transitions)


def personal_windows(t: np.ndarray, acc: np.ndarray, gyro: np.ndarray,
                     window_s: float, hop_s: float = 1.0):
    """Cut pure windows inside the ground-truth intervals only."""
    win = int(round(window_s * FS_OUT))
    X, y = [], []
    for lo, hi, code in TRUTH_INTERVALS:
        hi = min(hi, t[-1])
        for start in np.arange(lo, hi - window_s, hop_s):
            i0 = int(round(start * FS_OUT))
            i1 = i0 + win
            if i1 > len(t):
                break
            feat, _ = window_features(acc[i0:i1], gyro[i0:i1], FS_OUT)
            X.append(feat)
            y.append(code)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


# --------------------------------------------------------------------- main ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("motion_csv", type=Path)
    ap.add_argument("--sessions-npz", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "artifacts" / "models")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "models_personalized")
    ap.add_argument("--repeat", type=int, default=5,
                    help="how many times to repeat the personal windows when mixing")
    ap.add_argument("--max-per-class", type=int, default=20000,
                    help="cap on WISDM train windows per class (speed knob)")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data ----
    sessions = load_wisdm_sessions(args.sessions_npz)
    subjects = sorted({s.subject for s in sessions})
    split = split_subjects(subjects)          # same seed as scripts/train.py
    tr = [s for s in sessions if s.subject in split["train"]]
    te = [s for s in sessions if s.subject in split["test"]]
    print(f"[data] wisdm sessions train/test = {len(tr)}/{len(te)}; "
          f"personal subject 999 added to TRAIN only")

    t, acc, gyro = load_personal(args.motion_csv)
    p_session = personal_session(t, acc, gyro)
    print(f"[data] personal recording {p_session.duration:.1f}s, "
          f"transitions at {[round(x[0], 1) for x in p_session.transitions]}")

    # ------------------------------------------------------------- windows ----
    print("[windows] building WISDM train/test windows ...")
    X_tr, y_tr, names = build_windows(tr, max_per_class=args.max_per_class)
    X_te, y_te, _ = build_windows(te)
    print(f"[windows] wisdm train {X_tr.shape}, test {X_te.shape}")

    print("[windows] building personal windows (ground-truth intervals) ...")
    Xp, yp = personal_windows(t, acc, gyro, window_s=3.0)
    Xp_b, yp_b = personal_windows(t, acc, gyro, window_s=BURST_WINDOW_S)
    print(f"[windows] personal 3s {Xp.shape}, 1s {Xp_b.shape}")

    Xb_tr, yb_tr, _ = build_windows(tr, window_s=BURST_WINDOW_S, hop_s=1.0,
                                    max_per_class=args.max_per_class)
    Xb_te, yb_te, _ = build_windows(te, window_s=BURST_WINDOW_S, hop_s=1.0)

    old_main = ActivityClassifier.load(args.models_dir / "activity_classifier.pkl")
    old_burst = ActivityClassifier.load(args.models_dir / "burst_classifier.pkl")
    old_det = EventDetector.load(args.models_dir / "event_detector.pkl")

    report: dict = {"wisdm_test": {}, "personal": {}, "events": {}}

    # --------------------------------------------------- before: baselines ----
    report["wisdm_test"]["main_before"] = old_main.evaluate(X_te, y_te)
    report["wisdm_test"]["burst_before"] = old_burst.evaluate(Xb_te, yb_te)
    report["personal"]["main_before"] = old_main.evaluate(Xp, yp)
    report["personal"]["burst_before"] = old_burst.evaluate(Xp_b, yp_b)

    # --------------------------------------------------- retrain classifiers ----
    X_mix = np.concatenate([X_tr] + [Xp] * args.repeat)
    y_mix = np.concatenate([y_tr] + [yp] * args.repeat)
    print(f"[train] main classifier on {X_mix.shape[0]} windows "
          f"(personal x{args.repeat}) ...")
    new_main = ActivityClassifier(n_estimators=300)
    new_main.fit(X_mix, y_mix, names)
    new_main.save(out_dir / "activity_classifier.pkl")

    Xb_mix = np.concatenate([Xb_tr] + [Xp_b] * args.repeat)
    yb_mix = np.concatenate([yb_tr] + [yp_b] * args.repeat)
    print(f"[train] burst classifier on {Xb_mix.shape[0]} windows ...")
    new_burst = ActivityClassifier(n_estimators=200)
    new_burst.fit(Xb_mix, yb_mix, names)
    new_burst.save(out_dir / "burst_classifier.pkl")

    report["wisdm_test"]["main_after"] = new_main.evaluate(X_te, y_te)
    report["wisdm_test"]["burst_after"] = new_burst.evaluate(Xb_te, yb_te)
    report["personal"]["main_after"] = new_main.evaluate(Xp, yp)
    report["personal"]["burst_after"] = new_burst.evaluate(Xp_b, yp_b)

    # ------------------------------------------------------ event detector ----
    print("[train] event detector: wisdm train bursts + personal bursts ...")
    Xe_tr, ye_tr = build_event_dataset(tr)
    Xe_p, ye_p = build_event_dataset([p_session])
    Xe_mix = np.concatenate([Xe_tr] + [Xe_p] * args.repeat)
    ye_mix = np.concatenate([ye_tr] + [ye_p] * args.repeat)
    print(f"[train] event dataset {Xe_mix.shape}, pos_rate={ye_mix.mean():.3f}")

    new_det = EventDetector(n_estimators=300, mode="hybrid", threshold=0.5, rule_z=4.0)
    new_det.fit(Xe_mix, ye_mix)
    new_det.save(out_dir / "event_detector.pkl")

    report["wisdm_test"]["event_before"] = old_det.evaluate(
        *build_event_dataset(te))
    report["wisdm_test"]["event_after"] = new_det.evaluate(
        *build_event_dataset(te))

    # detector behaviour at the 3 real personal transitions
    hist_diag = {"true_transitions_s": [round(x[0], 1) for x in p_session.transitions]}
    for name, det in (("before", old_det), ("after", new_det)):
        per_tr = []
        for tt, a, b in p_session.transitions:
            best_p, best_z = 0.0, 0.0
            for t_end, hist, _ in burst_histories(p_session):
                if abs(t_end - tt) > 12.0:
                    continue
                feat_det = det.decide(hist)
                best_p = max(best_p, feat_det[1]["p_rf"])
                best_z = max(best_z, feat_det[1]["rule_z"])
            fired_p = best_p >= det.threshold
            fired_z = best_z >= det.rule.z_threshold
            per_tr.append({"t_s": round(tt, 1), "kind": f"{ACTIVITIES[a]}->{ACTIVITIES[b]}",
                           "max_p_rf": round(best_p, 3), "max_z": round(best_z, 2),
                           "rf_fired": fired_p, "rule_fired": fired_z})
        hist_diag[name] = per_tr
    report["events"]["transition_diagnosis"] = hist_diag

    (out_dir / "personalization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ summary ----
    def row(tag, before, after):
        print(f"  {tag:<22} macroF1 {before['macro_f1']:.3f} -> {after['macro_f1']:.3f}   "
              f"acc {before['accuracy']:.3f} -> {after['accuracy']:.3f}")

    print(f"\n[done] {time.time() - t0:.0f}s -> {out_dir}")
    print("WISDM test subjects (must not regress):")
    row("main 3s", report["wisdm_test"]["main_before"], report["wisdm_test"]["main_after"])
    row("burst 1s", report["wisdm_test"]["burst_before"], report["wisdm_test"]["burst_after"])
    print("Personal recording (in-sample):")
    row("main 3s", report["personal"]["main_before"], report["personal"]["main_after"])
    row("burst 1s", report["personal"]["burst_before"], report["personal"]["burst_after"])
    print("Personal transitions (detector):")
    for tr_d in hist_diag["after"]:
        print(f"  {tr_d['t_s']:6.1f}s {tr_d['kind']:<20} p_rf={tr_d['max_p_rf']:.3f} "
              f"z={tr_d['max_z']:.2f} rf_fired={tr_d['rf_fired']} rule_fired={tr_d['rule_fired']}")


if __name__ == "__main__":
    main()
