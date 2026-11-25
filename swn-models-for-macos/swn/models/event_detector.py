"""Event detection: *when* do we need more data?

The low-power stage does not stream the IMU.  It takes a 1 s micro-burst at
20 Hz every 4 s (a 25 % duty cycle), and compares each burst with the recent
history of bursts.  Two detectors run on that history:

* ``rule``   - the largest robust z-score of level / energy change, the classic
  "variance threshold" baseline;
* ``rf``     - a small Random Forest over the same normalised features, which
  can learn the difference between a real regime change and natural amplitude
  modulation (e.g. arm swing during walking).

``hybrid`` (the default) fires when either branch is confident.  Keeping the
rule branch lets us report a clean ablation.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from ..config import ACTIVITIES, FS, IDLE_BURST_S, IDLE_HISTORY, IDLE_PERIOD_S
from ..features import (
    BURST_EVENT_FEATURE_NAMES,
    BURST_STAT_NAMES,
    burst_event_features,
    burst_stats,
    burst_z,
)
from ..data.sessions import Session

TOL_S = 2.5  # a burst ending this close to a boundary is a positive


def burst_histories(session: Session, period_s: float = IDLE_PERIOD_S, burst_s: float = IDLE_BURST_S,
                    history: int = IDLE_HISTORY):
    """Yield (burst_end_time, burst_history, burst_stats) for a session."""
    burst_n = int(round(burst_s * FS))
    step = int(round(period_s * FS))
    stats: list[np.ndarray] = []
    for b in range(1, len(session.acc) // step):
        end = b * step
        seg = session.acc[end - burst_n : end]
        stats.append(burst_stats(seg, FS))
        hist = np.stack(stats[-history:])
        yield float(end) / FS, hist, stats[-1]


def build_event_dataset(sessions: list[Session], tol_s: float = TOL_S, history: int = IDLE_HISTORY,
                        max_neg_ratio: float | None = None, seed: int = 0):
    X, y = [], []
    for s in sessions:
        tr = np.array([t for t, _, _ in s.transitions], dtype=np.float64)
        for t, hist, _ in burst_histories(s, history=history):
            feat = burst_event_features(hist)
            pos = bool(len(tr) and np.min(np.abs(tr - t)) <= tol_s)
            X.append(feat)
            y.append(1 if pos else 0)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if max_neg_ratio is not None:
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        keep = min(len(neg_idx), int(max(len(pos_idx) * max_neg_ratio, 1500)))
        if keep < len(neg_idx):
            rng = np.random.default_rng(seed)
            neg_idx = rng.choice(neg_idx, keep, replace=False)
        idx = np.sort(np.concatenate([pos_idx, neg_idx]))
        X, y = X[idx], y[idx]
    return X, y


class RuleChangeDetector:
    """Robust z-score rule on burst level / energy (the 'variance threshold')."""

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold

    def score_features(self, feat: np.ndarray) -> float:
        n = len(BURST_STAT_NAMES)
        z = np.asarray(feat).ravel()[2 * n : 3 * n]
        zmax = float(np.max(np.abs(z[[0, 1, 2]])))
        return float(min(1.0, zmax / (2.0 * self.z_threshold)))

    def score(self, history: np.ndarray) -> float:
        return self.score_features(burst_event_features(history))

    def zmax(self, history: np.ndarray) -> float:
        return burst_z(history)


class EventDetector:
    def __init__(self, n_estimators: int = 300, seed: int = 0, mode: str = "hybrid",
                 threshold: float = 0.5, rule_z: float = 4.0):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        self.rule = RuleChangeDetector(rule_z)
        self.mode = mode
        self.threshold = threshold
        self.feature_names = list(BURST_EVENT_FEATURE_NAMES)
        self.feature_slice: slice | None = None

    # -------------------------------------------------------------- train ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.model.fit(X, y)
        # single-sample, once-per-burst inference: parallel overhead dominates
        self.model.set_params(n_jobs=1)
        p = self.model.predict_proba(X)[:, 1]
        return {
            "n_train": int(len(y)),
            "pos_rate": float(y.mean()),
            "train_ap": float(average_precision_score(y, p)),
        }

    # ------------------------------------------------------------ scoring ----
    def prob_features(self, feat: np.ndarray) -> float:
        if self.mode == "rule":
            return self.rule.score_features(feat)
        x = np.asarray(feat).ravel()
        if self.feature_slice is not None:
            x = x[self.feature_slice]
        p_rf = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        if self.mode == "rf":
            return p_rf
        p_rule = self.rule.score_features(feat)
        if self.mode == "hybrid_and":
            return min(p_rf, p_rule)
        return max(p_rf, p_rule)

    def prob(self, history: np.ndarray) -> float:
        return self.prob_features(burst_event_features(history))

    # Each branch keeps its own operating point: a probability threshold for the
    # learned detector and a z-score threshold for the rule.  They are not on
    # the same scale, and a single shared threshold silently disables one branch
    # (which is exactly what the first version of this code did).
    def decide(self, history: np.ndarray) -> tuple[bool, dict]:
        feat = burst_event_features(history)
        p_rf = float(self.model.predict_proba(feat.reshape(1, -1))[0, 1])
        z = burst_z(history)
        fire_rf = p_rf >= self.threshold
        fire_rule = z >= self.rule.z_threshold
        if self.mode == "rf":
            fired = fire_rf
        elif self.mode == "rule":
            fired = fire_rule
        else:
            fired = fire_rf or fire_rule
        return fired, {
            "p_change": round(self._display_prob(p_rf, z), 3),
            "p_rf": round(p_rf, 3),
            "rule_z": round(z, 2),
        }

    def _display_prob(self, p_rf: float, z: float) -> float:
        return max(p_rf, min(1.0, z / (2.0 * self.rule.z_threshold)))

    def fired(self, history: np.ndarray) -> tuple[bool, float]:
        ok, info = self.decide(history)
        return ok, info["p_change"]

    # ----------------------------------------------------------- evaluate ----
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if self.mode == "rule":
            p = np.array([self.rule.score_features(x) for x in X])
        else:
            p = self.model.predict_proba(X)[:, 1]
        pred = (p >= self.threshold).astype(int)
        return {
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "average_precision": float(average_precision_score(y, p)),
        }

    # ------------------------------------------------------------------ io ----
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "mode": self.mode,
                    "threshold": self.threshold,
                    "rule_z": self.rule.z_threshold,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: Path, mode: str | None = None, threshold: float | None = None) -> "EventDetector":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        obj = cls(mode=mode or blob["mode"], threshold=threshold or blob["threshold"], rule_z=blob["rule_z"])
        obj.model = blob["model"]
        return obj


def describe_transition(a: int, b: int) -> str:
    if a >= 3 and b >= 3:
        return "static2static"
    if a >= 3:
        return "static2moving"
    if b >= 3:
        return "moving2static"
    return "moving2moving"


__all__ = [
    "EventDetector",
    "RuleChangeDetector",
    "build_event_dataset",
    "burst_histories",
    "describe_transition",
    "ACTIVITIES",
]
