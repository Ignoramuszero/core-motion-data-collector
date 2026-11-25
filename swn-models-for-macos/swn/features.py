"""Feature extraction for the two model families.

* ``window_features``  - 3 s / 20 Hz windows for activity recognition.
* ``low_rate_features`` - 1 Hz magnitude stream for event detection.
"""
from __future__ import annotations

import numpy as np

AXES = ("x", "y", "z")

ACC_STATS = ("mean", "std", "min", "max", "rms", "median", "iqr", "skew", "kurt", "energy")
GYRO_STATS = ("mean", "std", "rms", "max")


def _spectral(mag: np.ndarray, fs: float) -> tuple[float, float, float, float, float]:
    n = len(mag)
    if n < 8:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x = (mag - mag.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    total = spec.sum()
    if total < 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    p = spec / total
    dom = float(freqs[1:][np.argmax(spec[1:])]) if len(spec) > 2 else 0.0
    e_low = float(p[(freqs >= 0.4) & (freqs < 3.0)].sum())
    e_mid = float(p[(freqs >= 3.0) & (freqs < 8.0)].sum())
    e_high = float(p[freqs >= 8.0].sum())
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return dom, e_low, e_mid, e_high, ent


def _acc_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    mean = x.mean(0)
    std = x.std(0)
    q75 = np.percentile(x, 75, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    return {
        "mean": mean,
        "std": std,
        "min": x.min(0),
        "max": x.max(0),
        "rms": np.sqrt((x ** 2).mean(0)),
        "median": np.median(x, 0),
        "iqr": q75 - q25,
        "skew": (((x - mean) ** 3).mean(0)) / (std ** 3 + 1e-9),
        "kurt": (((x - mean) ** 4).mean(0)) / (std ** 4 + 1e-9),
        "energy": (x ** 2).sum(0),
    }


def _gyro_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": x.mean(0),
        "std": x.std(0),
        "rms": np.sqrt((x ** 2).mean(0)),
        "max": np.abs(x).max(0),
    }


def window_features(acc: np.ndarray, gyro: np.ndarray, fs: float = 20.0):
    """Return (feature_vector, feature_names) for one IMU window."""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 3)
    if gyro.ndim == 1:
        gyro = gyro.reshape(-1, 3)

    feats: list[float] = []
    names: list[str] = []

    st = _acc_stats(acc)
    for stat in ACC_STATS:
        for i, ax in enumerate(AXES):
            feats.append(float(st[stat][i]))
            names.append(f"acc_{ax}_{stat}")

    mag = np.linalg.norm(acc, axis=1)
    mag_vals = [
        mag.mean(),
        mag.std(),
        mag.max(),
        mag.min(),
        float(np.percentile(mag, 90)),
        float(np.percentile(mag, 10)),
    ]
    feats += [float(v) for v in mag_vals]
    names += ["mag_mean", "mag_std", "mag_max", "mag_min", "mag_p90", "mag_p10"]

    dom, e_low, e_mid, e_high, ent = _spectral(mag, fs)
    feats += [dom, e_low, e_mid, e_high, ent]
    names += ["mag_dom_freq", "mag_e_low", "mag_e_mid", "mag_e_high", "mag_entropy"]

    gst = _gyro_stats(gyro)
    for stat in GYRO_STATS:
        for i, ax in enumerate(AXES):
            feats.append(float(gst[stat][i]))
            names.append(f"gyro_{ax}_{stat}")

    gmag = np.linalg.norm(gyro, axis=1)
    feats += [float(gmag.mean()), float(gmag.std()), float(gmag.max())]
    names += ["gyromag_mean", "gyromag_std", "gyromag_max"]

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(acc.T)
    corr = np.nan_to_num(corr, nan=0.0)
    feats += [float(corr[0, 1]), float(corr[0, 2]), float(corr[1, 2])]
    names += ["acc_corr_xy", "acc_corr_xz", "acc_corr_yz"]

    return np.asarray(feats, dtype=np.float32), names


BURST_STAT_NAMES = ["mean", "std", "range", "mad", "dom", "diff"]

# Absolute floors per burst statistic.  Without them, a perfectly still
# subject has a near-zero MAD and every jitter looks like a huge z-score.
BURST_FLOORS = np.array([0.80, 0.60, 1.20, 0.40, 0.35, 0.30], dtype=np.float32)
BURST_REF_SCALE = np.array([1.0, 0.10, 0.10, 0.10, 0.10, 0.10], dtype=np.float32)

BURST_EVENT_FEATURE_NAMES = (
    [f"cur_{n}" for n in BURST_STAT_NAMES]
    + [f"prev_{n}" for n in BURST_STAT_NAMES]
    + [f"z_{n}" for n in BURST_STAT_NAMES]
    + [f"ratio_{n}" for n in BURST_STAT_NAMES]
    + [f"spread_{n}" for n in BURST_STAT_NAMES]
)


def burst_stats(acc: np.ndarray, fs: float = 20.0) -> np.ndarray:
    """Compact per-burst descriptor of a short 20 Hz acceleration burst."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 3)
    mag = np.linalg.norm(acc, axis=1)
    n = len(mag)
    dom = 0.0
    if n >= 8:
        spec = np.abs(np.fft.rfft((mag - mag.mean()) * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        if spec.sum() > 1e-9:
            dom = float(freqs[1:][np.argmax(spec[1:])])
    diff = float(np.abs(np.diff(mag)).mean()) if n > 1 else 0.0
    return np.asarray(
        [
            mag.mean(),
            mag.std(),
            mag.max() - mag.min(),
            np.abs(mag - mag.mean()).mean(),
            dom,
            diff,
        ],
        dtype=np.float32,
    )


def burst_event_features(history: np.ndarray) -> np.ndarray:
    """Change-detection features from the recent burst history.

    ``history`` is (n_bursts, 6); the last row is the burst that just ended.
    The reference is a robust (median / MAD) summary of the earlier bursts, so
    the representation is *relative* and transfers across subjects.
    """
    hist = np.asarray(history, dtype=np.float32)
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)
    cur = hist[-1]
    prev = hist[-2] if len(hist) > 1 else cur
    ref_hist = hist[:-1] if len(hist) > 1 else hist
    ref = np.median(ref_hist, axis=0)
    mad = 1.4826 * np.median(np.abs(ref_hist - ref), axis=0)
    z = (cur - ref) / (mad + BURST_FLOORS)
    ratio = cur / (ref + BURST_REF_SCALE)
    spread = ref_hist.std(axis=0)
    return np.concatenate([cur, prev, z, ratio, spread]).astype(np.float32)


def burst_z(history: np.ndarray) -> float:
    """Interpretable rule branch: largest robust z-score on level/energy."""
    feat = burst_event_features(history)
    n = len(BURST_STAT_NAMES)
    z = feat[2 * n : 3 * n]
    return float(np.max(np.abs(z[[0, 1, 2]])))
