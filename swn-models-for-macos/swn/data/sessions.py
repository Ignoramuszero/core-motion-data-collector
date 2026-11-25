"""Replay sessions: continuous multi-activity streams built from real runs.

A session is what the adaptive system actually sees in the simulator: a single
20 Hz IMU stream that walks through several activity blocks, with real
transition boundaries between them.  Sessions are built per subject by
concatenating that subject's real activity runs in acquisition order, so the
transitions are real behaviour changes, not synthetic segment joins inside an
activity block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ACTIVITIES, FS

MAX_SESSION_S = 620.0
MIN_SESSION_S = 45.0
WARMUP_S = 5.0


@dataclass
class Session:
    session_id: str
    subject: int
    acc: np.ndarray
    gyro: np.ndarray
    labels: np.ndarray
    transitions: list[tuple[float, int, int]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.labels) / FS

    def label_at(self, t: float) -> int:
        i = int(min(max(t * FS, 0), len(self.labels) - 1))
        return int(self.labels[i])

    def summary(self) -> dict:
        counts = {ACTIVITIES[k]: int((self.labels == k).sum() / FS) for k in range(len(ACTIVITIES))}
        return {
            "session_id": self.session_id,
            "subject": self.subject,
            "duration_s": round(self.duration, 1),
            "n_transitions": len(self.transitions),
            "activity_seconds": {k: v for k, v in counts.items() if v > 0},
        }


def split_subjects(subjects: list[int], seed: int = 7) -> dict[str, list[int]]:
    subs = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    perm = [subs[i] for i in rng.permutation(len(subs))]
    n_test = max(3, int(round(0.22 * len(subs))))
    n_val = max(2, int(round(0.16 * len(subs))))
    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_sessions(runs: list[dict], max_len_s: float = MAX_SESSION_S) -> list[Session]:
    """Concatenate each subject's runs into replay sessions."""
    by_subject: dict[int, list[dict]] = {}
    for r in runs:
        by_subject.setdefault(r["subject"], []).append(r)

    sessions: list[Session] = []
    for subj, subj_runs in sorted(by_subject.items()):
        subj_runs = sorted(subj_runs, key=lambda r: r["t0_ns"])
        chunk_acc: list[np.ndarray] = []
        chunk_gyro: list[np.ndarray] = []
        chunk_labels: list[np.ndarray] = []
        chunk_len = 0.0
        chunk_idx = 0

        def flush() -> None:
            nonlocal chunk_acc, chunk_gyro, chunk_labels, chunk_len, chunk_idx
            if chunk_len >= MIN_SESSION_S:
                acc = np.concatenate(chunk_acc)
                gyro = np.concatenate(chunk_gyro)
                labels = np.concatenate(chunk_labels)
                transitions: list[tuple[float, int, int]] = []
                # recompute transition times from the label stream
                change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
                for idx in change:
                    t = float(idx) / FS
                    if t < 1.0:
                        continue
                    transitions.append((t, int(labels[idx - 1]), int(labels[idx])))
                sessions.append(
                    Session(
                        session_id=f"s{subj}_{chunk_idx}",
                        subject=subj,
                        acc=acc.astype(np.float32),
                        gyro=gyro.astype(np.float32),
                        labels=labels.astype(np.int8),
                        transitions=transitions,
                    )
                )
                chunk_idx += 1
            chunk_acc, chunk_gyro, chunk_labels, chunk_len = [], [], [], 0.0

        for r in subj_runs:
            dur = len(r["acc"]) / FS
            if chunk_len + dur > max_len_s and chunk_len > 0:
                flush()
            chunk_acc.append(np.clip(r["acc"], -160.0, 160.0))
            chunk_gyro.append(np.clip(r["gyro"], -40.0, 40.0))
            chunk_labels.append(np.full(len(r["acc"]), r["label"], dtype=np.int8))
            chunk_len += dur
        flush()

    sessions.sort(key=lambda s: (s.subject, s.session_id))
    return sessions


def load_sessions() -> list[Session]:
    """Convenience loader: parsed runs -> replay sessions."""
    from .wisdm import load_runs

    return build_sessions(load_runs())


def dataset_stats(sessions: list[Session]) -> dict:
    total_s = sum(s.duration for s in sessions)
    per_class = np.zeros(len(ACTIVITIES))
    per_class_s = np.zeros(len(ACTIVITIES))
    for s in sessions:
        for k in range(len(ACTIVITIES)):
            per_class_s[k] += float((s.labels == k).sum()) / FS
            per_class[k] += float((s.labels == k).sum())
    n_tr = sum(len(s.transitions) for s in sessions)
    static2static = 0
    moving_types = {"static2moving": 0, "moving2static": 0, "moving2moving": 0}
    for s in sessions:
        for _, a, b in s.transitions:
            if a >= 3 and b >= 3:
                static2static += 1
            elif a >= 3:
                moving_types["static2moving"] += 1
            elif b >= 3:
                moving_types["moving2static"] += 1
            else:
                moving_types["moving2moving"] += 1
    return {
        "n_sessions": len(sessions),
        "n_subjects": len({s.subject for s in sessions}),
        "total_minutes": round(total_s / 60.0, 1),
        "n_transitions": n_tr,
        "transition_types": {"static2static": static2static, **moving_types},
        "samples_per_class": {ACTIVITIES[k]: int(per_class[k]) for k in range(len(ACTIVITIES))},
        "minutes_per_class": {ACTIVITIES[k]: round(per_class_s[k] / 60.0, 1) for k in range(len(ACTIVITIES))},
    }
