"""Train the two models and tune the event-detection threshold on validation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import ACTIVITIES, BURST_WINDOW_S, DEFAULT_POLICY, MODEL_DIR, RESULT_DIR, config_snapshot  # noqa: E402
from swn.data.sessions import dataset_stats, load_sessions, split_subjects  # noqa: E402
from swn.evaluation.metrics import event_metrics, merge_event_metrics  # noqa: E402
from swn.models.activity_classifier import ActivityClassifier, build_windows  # noqa: E402
from swn.models.event_detector import EventDetector, build_event_dataset  # noqa: E402
from swn.runtime import run_session  # noqa: E402


def tune_threshold(det: EventDetector, val_sessions, clf: ActivityClassifier,
                   burst_clf: ActivityClassifier) -> tuple[dict, list[dict]]:
    """Grid-search both branch operating points on validation subjects."""
    grid_rf = [0.4, 0.5, 0.6, 0.7]
    grid_z = [3.0, 4.0, 5.0]
    table = []
    best = None
    best_f1 = -1.0
    for rf_t in grid_rf:
        for z_t in grid_z:
            det.threshold = rf_t
            det.rule.z_threshold = z_t
            metrics = []
            for s in val_sessions:
                res = run_session(s, "event_driven", clf, det, policy=DEFAULT_POLICY, run_llm=False,
                                  burst_classifier=burst_clf)
                trig = [x for x in res.triggers if not x.get("bootstrap")]
                metrics.append(event_metrics(trig, res.true_transitions))
            agg = merge_event_metrics(metrics)
            row = {"rf_threshold": rf_t, "rule_z": z_t,
                   **{k: agg[k] for k in ("precision", "recall", "f1", "median_latency_s")}}
            table.append(row)
            print(f"[tune] rf_thr={rf_t:.1f} z={z_t:.0f} P={agg['precision']:.2f} "
                  f"R={agg['recall']:.2f} F1={agg['f1']:.2f}", flush=True)
            if agg["f1"] > best_f1:
                best_f1 = agg["f1"]
                best = {"rf_threshold": rf_t, "rule_z": z_t}
    det.threshold = best["rf_threshold"]
    det.rule.z_threshold = best["rule_z"]
    return best, table


def main() -> None:
    t_start = time.time()
    sessions = load_sessions()
    subjects = sorted({s.subject for s in sessions})
    split = split_subjects(subjects)
    tr = [s for s in sessions if s.subject in split["train"]]
    va = [s for s in sessions if s.subject in split["val"]]
    te = [s for s in sessions if s.subject in split["test"]]
    print(f"[train] sessions train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # -------------------------------------------------- activity classifier ----
    print("[train] building windows ...")
    X_tr, y_tr, names = build_windows(tr, max_per_class=60000)
    X_va, y_va, _ = build_windows(va)
    X_te, y_te, _ = build_windows(te)
    print(f"[train] windows: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    clf = ActivityClassifier(n_estimators=300)
    info = clf.fit(X_tr, y_tr, names)
    print(f"[train] classifier {info}")
    clf_metrics = {
        "train": clf.evaluate(X_tr, y_tr),
        "val": clf.evaluate(X_va, y_va),
        "test": clf.evaluate(X_te, y_te),
        "n_features": len(names),
        "top_features": clf.top_features(12),
    }
    clf.save(MODEL_DIR / "activity_classifier.pkl")

    # ------------------------------------------------ cheap burst classifier ----
    print("[train] training 1 s burst classifier (low-power monitoring) ...")
    Xb_tr, yb_tr, _ = build_windows(tr, window_s=BURST_WINDOW_S, hop_s=1.0, max_per_class=60000)
    Xb_te, yb_te, _ = build_windows(te, window_s=BURST_WINDOW_S, hop_s=1.0)
    burst_clf = ActivityClassifier(n_estimators=200)
    burst_clf.fit(Xb_tr, yb_tr, names)
    burst_metrics = {"train": burst_clf.evaluate(Xb_tr, yb_tr), "test": burst_clf.evaluate(Xb_te, yb_te)}
    burst_clf.save(MODEL_DIR / "burst_classifier.pkl")
    print(f"[train] burst classifier test acc={burst_metrics['test']['accuracy']:.3f} "
          f"macroF1={burst_metrics['test']['macro_f1']:.3f}")

    # ------------------------------------------------------- event detector ----
    print("[train] building event dataset ...")
    Xe_tr, ye_tr = build_event_dataset(tr)
    Xe_va, ye_va = build_event_dataset(va)
    Xe_te, ye_te = build_event_dataset(te)
    print(f"[train] event windows {Xe_tr.shape}, pos_rate={ye_tr.mean():.3f}")

    det = EventDetector(n_estimators=300, mode="hybrid", threshold=0.5)
    det_info = det.fit(Xe_tr, ye_tr)
    print(f"[train] detector {det_info}")

    event_window_metrics = {"train": det_info}
    for mode in ("rule", "rf", "hybrid"):
        det.mode = mode
        event_window_metrics[mode] = {"val": det.evaluate(Xe_va, ye_va), "test": det.evaluate(Xe_te, ye_te)}

    # closed-loop threshold tuning on validation subjects
    det.mode = "hybrid"
    print("[train] tuning event threshold on validation ...")
    best, thr_table = tune_threshold(det, va, clf, burst_clf)
    det.save(MODEL_DIR / "event_detector.pkl")

    metrics = {
        "config": config_snapshot(),
        "dataset": dataset_stats(sessions),
        "split": {k: [int(x) for x in v] for k, v in split.items()},
        "classifier": clf_metrics,
        "burst_classifier": burst_metrics,
        "event_detector": {
            "window_level": event_window_metrics,
            "threshold_tuning": thr_table,
            "selected": best,
        },
        "train_seconds": round(time.time() - t_start, 1),
    }
    (RESULT_DIR / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(clf_metrics["test"], indent=2))
    print(f"[train] selected {best} in {metrics['train_seconds']}s")


if __name__ == "__main__":
    main()
