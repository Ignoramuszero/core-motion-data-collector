#!/usr/bin/env python3
"""Convert the trained sklearn models to Apple Core ML (.mlmodel).

Run this ON THE MAC (or any machine with coremltools) before integrating
the models into the iOS app:

    pip install coremltools joblib numpy scikit-learn
    python swn/coreml_export.py

Outputs into artifacts/coreml/:
    ActivityClassifier.mlmodel   3 s window, 5 activities
    BurstClassifier.mlmodel      1 s window, 5 activities (low-power monitoring)
    EventDetector.mlmodel        burst history -> change / no_change (binary)

The Swift side must compute the *same* features in the *same* order as
swn/features.py — see MACOS_INTEGRATION.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import MODEL_DIR  # noqa: E402
from swn.models.activity_classifier import ActivityClassifier  # noqa: E402
from swn.models.event_detector import EventDetector  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "coreml"


def export_sklearn(model, feature_names: list[str], out_path: Path, description: str):
    import inspect

    import coremltools as ct

    input_features = [(name, float) for name in feature_names]
    # coremltools >= 9 renamed output_feature_name -> output_feature_names and
    # changed the classifier convention: a list of TWO names
    # [class-label-name, probability-dict-name]. Old versions took a single
    # output_feature_name="output" and produced "classLabel" + "output".
    # Either way the app expects the outputs to be named classLabel/output.
    kwargs: dict = {}
    try:
        params = inspect.signature(ct.converters.sklearn.convert).parameters
        if "output_feature_names" in params:
            kwargs["output_feature_names"] = ["classLabel", "output"]
        elif "output_feature_name" in params:
            kwargs["output_feature_name"] = "output"
    except (ValueError, TypeError):
        kwargs["output_feature_name"] = "output"

    coreml = ct.converters.sklearn.convert(model, input_features, **kwargs)
    coreml.author = "SenseWhenNeeded"
    coreml.short_description = description
    coreml.save(str(out_path))
    print(f"[export] {out_path.name}  ({len(feature_names)} features)")

    # On macOS, sanity-check the converted model against sklearn itself.
    # This catches BOTH converter errors and corrupted unpickling (e.g. a
    # 1.9-trained pkl loaded under sklearn 1.5) — the two must agree.
    if sys.platform == "darwin":
        try:
            import numpy as np

            coreml_model = ct.models.MLModel(str(out_path))
            rng = np.random.default_rng(0)
            X = rng.normal(size=(12, len(feature_names))).astype(np.float32)
            sk_labels = model.predict(X)
            agree = 0
            for row, expected in zip(X, sk_labels):
                out = coreml_model.predict({n: float(v) for n, v in zip(feature_names, row)})
                got = out.get("classLabel")
                if isinstance(got, str):
                    got = int(got) if got.isdigit() else got
                if int(got) == int(expected):
                    agree += 1
            print(f"[verify] {out_path.name}: Core ML vs sklearn = {agree}/12 agree")
            if agree < 12:
                print("[warn] predictions disagree — check the sklearn version "
                      "used to unpickle (see InconsistentVersionWarning above)")
        except Exception as exc:  # verification is best-effort
            print(f"[verify] skipped ({exc})")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Convert trained sklearn models to Core ML.")
    ap.add_argument("--models-dir", type=Path, default=MODEL_DIR,
                    help="directory containing the three .pkl files "
                         "(default: artifacts/models; use artifacts/models_personalized "
                         "for the device-personalized models)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    src = args.models_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) main activity classifier (3 s window @ 20 Hz, 5 activities)
    clf = ActivityClassifier.load(src / "activity_classifier.pkl")
    export_sklearn(
        clf.model,
        clf.feature_names,
        args.out_dir / "ActivityClassifier.mlmodel",
        "5-activity recognition from a 3 s IMU window (WISDM-trained).",
    )

    # 2) cheap burst classifier (1 s window @ 20 Hz, 5 activities)
    burst = ActivityClassifier.load(src / "burst_classifier.pkl")
    export_sklearn(
        burst.model,
        burst.feature_names,
        args.out_dir / "BurstClassifier.mlmodel",
        "5-activity recognition from a 1 s IMU burst (low-power monitoring).",
    )

    # 3) event detector (burst-history features -> binary change/no_change)
    det = EventDetector.load(src / "event_detector.pkl")
    export_sklearn(
        det.model,
        det.feature_names,
        args.out_dir / "EventDetector.mlmodel",
        "Binary change-point detector over a history of 1 s burst statistics.",
    )

    print(f"[done] models written to {args.out_dir}")


if __name__ == "__main__":
    main()
