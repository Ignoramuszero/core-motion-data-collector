"""Central configuration for the Sense When Needed system."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
RESULT_DIR = ARTIFACT_DIR / "results"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TIMELINE_DIR = ARTIFACT_DIR / "timelines"

for _d in (PROCESSED_DIR, MODEL_DIR, RESULT_DIR, FIGURE_DIR, TIMELINE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- sensing ----
FS = 20.0                # native replay rate of the WISDM smartwatch stream
WINDOW_S = 3.0           # activity-recognition window (same for all strategies)
HOP_S = 1.0              # decision hop
IDLE_BURST_S = 1.0       # low-power mode: 1 s micro-burst ...
IDLE_PERIOD_S = 6.0      # ... every 6 s  (16.7 % IMU duty cycle)
IDLE_HISTORY = 4         # bursts kept for the change detector (24 s of context)
BURST_WINDOW_S = 1.0     # cheap 1 s window classifier used while idle

# --------------------------------------------------------------- activity ----
WISDM_CODES = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
ACTIVITIES = ["walking", "jogging", "stairs", "sitting", "standing"]
ACTIVITY_LABEL_CN = {
    "walking": "行走",
    "jogging": "慢跑",
    "stairs": "上下楼",
    "sitting": "静坐",
    "standing": "站立",
}
LOCOMOTION = {0, 1, 2}
STATIC = {3, 4}
NUM_CLASSES = len(ACTIVITIES)

# Latent speed used *only* by the simulated GPS modality.  WISDM contains no
# GPS; the GPS channel is therefore a synthetic modality whose output the
# policy cannot see directly - it can only decide to pay for it.
LATENT_SPEED = {0: 1.30, 1: 3.20, 2: 0.70, 3: 0.02, 4: 0.02}


@dataclass
class PolicyConfig:
    """Parameters of the event-driven adaptive policy (Method 3)."""

    burst_s: float = IDLE_BURST_S
    period_s: float = IDLE_PERIOD_S
    history: int = IDLE_HISTORY
    event_threshold: float = 0.50      # P(change) trigger threshold
    rule_z: float = 3.0                # rule-based branch of the hybrid detector
    conf_high: float = 0.85            # confident enough to commit a label
    conf_low: float = 0.60             # uncertainty that requests more context
    confirm_windows: int = 2           # consecutive agreeing windows to commit
    min_active_s: float = 4.0
    max_active_s: float = 15.0
    cooldown_s: float = 8.0
    gps_enabled: bool = True
    gps_duty_s: float = 5.0
    llm_on_transition: bool = True
    bootstrap: bool = True
    burst_vote_window: int = 3   # majority vote over the last N cheap windows
    uncertainty_trigger: bool = False  # also trigger when the cheap model keeps disagreeing


@dataclass
class PowerModel:
    """Relative power model for the energy proxy (mW)."""

    imu_on_mw: float = 1.00     # 20 Hz accel + gyro while a window is acquired
    imu_sleep_mw: float = 0.02  # idle current between bursts
    gps_mw: float = 30.0
    cpu_watt: float = 0.50


@dataclass
class BaselineConfig:
    periodic_period_s: float = 10.0
    periodic_burst_s: float = WINDOW_S


DEFAULT_POLICY = PolicyConfig()
DEFAULT_POWER = PowerModel()
DEFAULT_BASELINE = BaselineConfig()


def config_snapshot() -> dict:
    return {
        "fs": FS,
        "window_s": WINDOW_S,
        "activities": ACTIVITIES,
        "policy": asdict(DEFAULT_POLICY),
        "power": asdict(DEFAULT_POWER),
        "baseline": asdict(DEFAULT_BASELINE),
    }
