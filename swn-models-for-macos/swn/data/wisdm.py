"""WISDM raw-data parsing and preprocessing.

Dataset: WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
(UCI ML Repository id 507), 51 subjects, smartwatch + smartphone, 20 Hz,
accelerometer + gyroscope.

Raw line format (``raw/watch/accel/data_1600_accel_watch.txt``)::

    1600,A,252207666810782,-0.36476135,8.793503,1.0550842;

=> subject, activity code, timestamp (ns), x, y, z
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FS, PROCESSED_DIR, RAW_DIR, WISDM_CODES

WATCH_DIR = RAW_DIR / "wisdm" / "watch"
MIN_RUN_S = 20.0       # discard activity runs shorter than this
MAX_GAP_S = 0.5        # a timestamp gap larger than this starts a new run


def parse_raw_file(path: Path) -> pd.DataFrame:
    """Parse one WISDM raw txt file into a typed dataframe."""
    df = pd.read_csv(
        path,
        header=None,
        names=["subject", "code", "ts", "x", "y", "z"],
        dtype={0: "int32", 1: "str", 2: "int64", 3: "float64", 4: "float64", 5: "str"},
    )
    df["z"] = df["z"].str.rstrip(";").astype("float32")
    df["x"] = df["x"].astype("float32")
    df["y"] = df["y"].astype("float32")
    df["code"] = df["code"].str.strip()
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return df


def _runs(df: pd.DataFrame):
    """Split a dataframe into contiguous (same activity, small gap) runs."""
    code = df["code"].to_numpy()
    ts = df["ts"].to_numpy()
    if len(df) == 0:
        return []
    breaks = np.flatnonzero((code[1:] != code[:-1]) | (np.diff(ts) > MAX_GAP_S * 1e9))
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(df)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _resample(df: pd.DataFrame, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    t = (df["ts"].to_numpy() - df["ts"].to_numpy()[0]) / 1e9
    dur = float(t[-1])
    n = int(np.floor(dur * fs)) + 1
    grid = np.arange(n) / fs
    out = np.empty((n, 3), dtype=np.float32)
    for i, c in enumerate(("x", "y", "z")):
        out[:, i] = np.interp(grid, t, df[c].to_numpy())
    return out, grid


def extract_runs(subjects: list[int] | None = None, limit_subjects: int | None = None) -> list[dict]:
    """Extract aligned (acc, gyro) activity runs for the selected subjects."""
    acc_files = sorted(WATCH_DIR.glob("accel/*.txt"))
    gyro_files = {p.name.replace("accel", "gyro"): p for p in WATCH_DIR.glob("gyro/*.txt")}
    runs: list[dict] = []

    for acc_path in acc_files:
        subj = int(acc_path.stem.split("_")[1])
        if subjects is not None and subj not in subjects:
            continue
        gyro_path = gyro_files.get(acc_path.name.replace("accel", "gyro"))
        if gyro_path is None:
            continue
        acc_df = parse_raw_file(acc_path)
        gyro_df = parse_raw_file(gyro_path)
        acc_df = acc_df[acc_df["code"].isin(WISDM_CODES)]
        gyro_df = gyro_df[gyro_df["code"].isin(WISDM_CODES)]

        for s, e in _runs(acc_df):
            seg = acc_df.iloc[s:e]
            code = seg["code"].iloc[0]
            t0 = int(seg["ts"].iloc[0])
            t1 = int(seg["ts"].iloc[-1])
            if (t1 - t0) / 1e9 < MIN_RUN_S:
                continue
            gseg = gyro_df[(gyro_df["ts"] >= t0) & (gyro_df["ts"] <= t1)]
            if len(gseg) < 0.5 * max(1, len(seg)):
                continue
            gseg = gseg.sort_values("ts")
            acc, grid = _resample(seg)
            gyro, _ = _resample(gseg)
            if len(gyro) < len(acc):
                pad = np.repeat(gyro[-1:], len(acc) - len(gyro), axis=0)
                gyro = np.vstack([gyro, pad])
            gyro = gyro[: len(acc)]
            runs.append(
                {
                    "subject": subj,
                    "code": code,
                    "label": WISDM_CODES[code],
                    "t0_ns": t0,
                    "acc": acc,
                    "gyro": gyro,
                }
            )
    runs.sort(key=lambda r: (r["subject"], r["t0_ns"]))
    return runs


def save_runs(runs: list[dict], out_path: Path | None = None) -> Path:
    """Persist runs as a compact ragged npz."""
    out_path = out_path or (PROCESSED_DIR / "wisdm_watch_runs.npz")
    acc = np.concatenate([r["acc"] for r in runs]).astype(np.float32)
    gyro = np.concatenate([r["gyro"] for r in runs]).astype(np.float32)
    lengths = np.asarray([len(r["acc"]) for r in runs], dtype=np.int32)
    labels = np.asarray([r["label"] for r in runs], dtype=np.int8)
    subjects = np.asarray([r["subject"] for r in runs], dtype=np.int16)
    t0 = np.asarray([r["t0_ns"] for r in runs], dtype=np.int64)
    np.savez_compressed(
        out_path, acc=acc, gyro=gyro, lengths=lengths, labels=labels,
        subjects=subjects, t0_ns=t0, fs=np.float32(FS),
    )
    return out_path


def load_runs(path: Path | None = None) -> list[dict]:
    path = path or (PROCESSED_DIR / "wisdm_watch_runs.npz")
    z = np.load(path)
    acc, gyro = z["acc"], z["gyro"]
    lengths, labels = z["lengths"], z["labels"]
    subjects, t0 = z["subjects"], z["t0_ns"]
    runs, pos = [], 0
    for i, n in enumerate(lengths):
        n = int(n)
        runs.append(
            {
                "subject": int(subjects[i]),
                "label": int(labels[i]),
                "t0_ns": int(t0[i]),
                "acc": acc[pos : pos + n],
                "gyro": gyro[pos : pos + n],
            }
        )
        pos += n
    return runs
