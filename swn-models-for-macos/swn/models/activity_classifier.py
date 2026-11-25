"""Window-based activity recognition (Random Forest).

Deliberately small and interpretable: the contribution of this project is the
sensing policy, not the classifier.  A Random Forest gives us calibrated-ish
per-window probabilities (needed by the confidence-driven policy), trains in
seconds, and runs on a CPU with no accelerator.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from ..config import ACTIVITIES, FS, HOP_S, WINDOW_S
from ..features import window_features
from ..data.sessions import Session


def build_windows(sessions: list[Session], hop_s: float = HOP_S, window_s: float = WINDOW_S,
                  pure_only: bool = True, max_per_class: int | None = None, seed: int = 0):
    """Cut labelled windows out of sessions."""
    win = int(round(window_s * FS))
    hop = int(round(hop_s * FS))
    X, y = [], []
    for s in sessions:
        n = len(s.labels)
        for end in range(win, n + 1, hop):
            seg_lab = s.labels[end - win : end]
            if pure_only and seg_lab.min() != seg_lab.max():
                continue
            acc = s.acc[end - win : end]
            gyro = s.gyro[end - win : end]
            feat, names = window_features(acc, gyro, FS)
            X.append(feat)
            y.append(int(seg_lab[0]))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        keep = []
        for k in range(len(ACTIVITIES)):
            idx = np.flatnonzero(y == k)
            if len(idx) > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        X, y = X[keep], y[keep]
    return X, y, names


class ActivityClassifier:
    def __init__(self, n_estimators: int = 300, seed: int = 0):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        self.feature_names: list[str] = []
        self.window_s = WINDOW_S

    def fit(self, X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
        self.feature_names = names
        self.model.fit(X, y)
        self.model.set_params(n_jobs=1)
        train_acc = float(accuracy_score(y, self.model.predict(X)))
        return {"train_accuracy": train_acc, "n_train": int(len(y))}

    def predict_proba(self, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        feat, _ = window_features(acc, gyro, FS)
        return self.model.predict_proba(feat.reshape(1, -1))[0]

    def predict_window(self, acc: np.ndarray, gyro: np.ndarray) -> tuple[int, float, np.ndarray]:
        proba = self.predict_proba(acc, gyro)
        k = int(np.argmax(proba))
        return k, float(proba[k]), proba

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        pred = self.model.predict(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class_f1": {
                ACTIVITIES[k]: float(v)
                for k, v in enumerate(f1_score(y, pred, average=None, labels=np.arange(len(ACTIVITIES))))
            },
        }

    # ------------------------------------------------------------ io ----
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names, "window_s": self.window_s}, f)
        return path

    @classmethod
    def load(cls, path: Path) -> "ActivityClassifier":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        obj = cls()
        obj.model = blob["model"]
        obj.feature_names = blob["feature_names"]
        obj.window_s = blob.get("window_s", WINDOW_S)
        return obj

    def top_features(self, n: int = 12) -> list[tuple[str, float]]:
        imp = self.model.feature_importances_
        order = np.argsort(imp)[::-1][:n]
        return [(self.feature_names[i], float(imp[i])) for i in order]
