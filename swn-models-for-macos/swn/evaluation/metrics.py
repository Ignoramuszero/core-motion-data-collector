"""Metrics: continuous activity tracking, event detection, responsiveness."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..config import ACTIVITIES
from ..models.event_detector import describe_transition


def tracking_metrics(true: np.ndarray, pred: np.ndarray, warmup_s: float = 5.0) -> dict:
    """Per-second tracking accuracy of the system's *current* estimate.

    ``pred == -1`` (nothing committed yet) counts as an error, which is the
    honest thing to do: the user is looking at a dashboard that says nothing.
    """
    n = len(true)
    mask = np.arange(n) >= warmup_s
    mask &= true >= 0
    y_true = true[mask].astype(int)
    y_pred = pred[mask].astype(int)
    y_pred = np.where(y_pred < 0, -1, y_pred)  # "no estimate yet" = wrong, not a class
    present = [k for k in range(len(ACTIVITIES)) if (y_true == k).any()]
    per_class_vals = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    per_class = {ACTIVITIES[k]: float(v) for k, v in zip(present, per_class_vals)}
    cm = confusion_matrix(y_true, y_pred, labels=present)
    return {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(per_class_vals)) if len(per_class_vals) else 0.0,
        "per_class_f1": per_class,
        "classes_present": present,
        "confusion": cm.tolist(),
        "coverage": float(np.mean(y_pred >= 0)),
    }


def transition_latency(true: np.ndarray, pred: np.ndarray, transitions: list[tuple[float, int, int]],
                       max_wait_s: float = 120.0) -> list[dict]:
    """How long until the system's estimate switches to the *new* activity."""
    out = []
    n = len(true)
    for t_tr, a, b in transitions:
        start = int(np.ceil(t_tr))
        found = None
        for t in range(start, min(n, int(start + max_wait_s))):
            if pred[t] == b:
                # require the new label to stick for 3 s (avoid flicker credit)
                stick = pred[t : min(n, t + 3)]
                if len(stick) and np.all(stick == b):
                    found = t
                    break
        out.append(
            {
                "t": float(t_tr),
                "from": ACTIVITIES[a],
                "to": ACTIVITIES[b],
                "type": describe_transition(a, b),
                "latency_s": None if found is None else float(found - t_tr),
            }
        )
    return out


def event_metrics(triggers: list[dict], transitions: list[tuple[float, int, int]], tol_s: float = 8.0) -> dict:
    """Event-level precision / recall with a tolerance window.

    The default tolerance (8 s) matches the detector's temporal granularity: it
    can only fire on a 6 s burst tick, so a fair match window is one burst
    period plus jitter.
    """
    det_t = [float(x["t"]) for x in triggers]
    tru = [(float(t), a, b) for t, a, b in transitions]
    used = set()
    latencies = []
    per_type: dict[str, dict] = {}
    matched_true = set()
    for dt in sorted(det_t):
        best = None
        for i, (tt, a, b) in enumerate(tru):
            if i in used:
                continue
            d = abs(tt - dt)
            if d <= tol_s and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            d, i = best
            used.add(i)
            matched_true.add(i)
            latencies.append(dt - tru[i][0])
            key = describe_transition(tru[i][1], tru[i][2])
            slot = per_type.setdefault(key, {"tp": 0, "n": 0})
            slot["tp"] += 1
    for i, (tt, a, b) in enumerate(tru):
        key = describe_transition(a, b)
        per_type.setdefault(key, {"tp": 0, "n": 0})["n"] += 1

    tp = len(used)
    fp = len(det_t) - tp
    fn = len(tru) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "n_true": len(tru),
        "n_detected": len(det_t),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_latency_s": float(np.median(latencies)) if latencies else None,
        "mean_abs_latency_s": float(np.mean(np.abs(latencies))) if latencies else None,
        "recall_by_type": {
            k: {"recall": v["tp"] / max(1, v["n"]), "n": v["n"]} for k, v in per_type.items()
        },
    }


def merge_event_metrics(per_session: list[dict]) -> dict:
    """Pool event metrics across sessions."""
    keys = ("tp", "fp", "fn", "n_true", "n_detected")
    tot = {k: sum(m[k] for m in per_session) for k in keys}
    precision = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    recall = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    lats = [m["median_latency_s"] for m in per_session if m["median_latency_s"] is not None]
    by_type: dict[str, dict] = {}
    for m in per_session:
        for k, v in m["recall_by_type"].items():
            slot = by_type.setdefault(k, {"tp": 0, "n": 0})
            slot["n"] += v["n"]
            slot["tp"] += int(round(v["recall"] * v["n"]))
    return {
        **tot,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_latency_s": float(np.median(lats)) if lats else None,
        "recall_by_type": {k: {"recall": v["tp"] / max(1, v["n"]), "n": v["n"]} for k, v in by_type.items()},
    }
