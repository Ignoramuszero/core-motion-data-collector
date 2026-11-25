"""Runtime: replay a session under one of three sensing strategies.

All strategies share the same simulator and the same 3 s recognition window.
What differs is *when* the IMU is on, and when inference / GPS / LLM are paid
for:

* ``always_on``    - 20 Hz continuous, classify every second.
* ``periodic``     - a 3 s burst at 20 Hz every 10 s, classify at each burst.
* ``event_driven`` - 1 s micro-burst every 6 s.  Each burst is classified by a
  cheap 1 s-window model (multi-fidelity monitoring) and compared with the
  recent burst history.  On a detected change the system switches to 20 Hz,
  confirms with the accurate 3 s model, optionally fires GPS on a movement
  onset, calls the LLM only on real transitions, then returns to low power.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .config import (
    ACTIVITIES,
    BURST_WINDOW_S,
    FS,
    LOCOMOTION,
    WINDOW_S,
    BaselineConfig,
    PolicyConfig,
    PowerModel,
)
from .data.sessions import Session
from .features import burst_stats
from .llm.context_builder import build_transition_context
from .models.activity_classifier import ActivityClassifier
from .models.event_detector import EventDetector
from .sensing.sensor_simulator import SensorSimulator

STRATEGIES = ("always_on", "periodic", "event_driven")
STRATEGY_LABEL = {
    "always_on": "Always-on (20 Hz)",
    "periodic": "Periodic (3 s / 10 s)",
    "event_driven": "Event-driven (ours)",
}


@dataclass
class RunResult:
    strategy: str
    session_id: str
    subject: int
    duration: float
    t: np.ndarray
    true: np.ndarray
    pred: np.ndarray
    conf: np.ndarray
    state: list[str]
    event_prob: np.ndarray
    gps: np.ndarray
    imu_on: np.ndarray
    triggers: list[dict] = field(default_factory=list)
    true_transitions: list[tuple[float, int, int]] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=dict)

    def timeline(self) -> list[dict]:
        out = []
        for i, t in enumerate(self.t):
            if t <= 0:
                continue
            out.append(
                {
                    "t": float(t),
                    "true": int(self.true[i]),
                    "true_name": ACTIVITIES[int(self.true[i])] if self.true[i] >= 0 else "unknown",
                    "pred": int(self.pred[i]),
                    "pred_name": ACTIVITIES[int(self.pred[i])] if self.pred[i] >= 0 else "unknown",
                    "conf": round(float(self.conf[i]), 3),
                    "state": self.state[i],
                    "event_prob": None if np.isnan(self.event_prob[i]) else round(float(self.event_prob[i]), 3),
                    "gps": bool(self.gps[i]),
                    "imu_on": bool(self.imu_on[i]),
                }
            )
        return out


def _new_arrays(T: int):
    return (
        np.arange(T + 1, dtype=np.float32),
        np.full(T + 1, -1, dtype=np.int8),
        np.full(T + 1, -1, dtype=np.int8),
        np.zeros(T + 1, dtype=np.float32),
        np.full(T + 1, np.nan, dtype=np.float32),
        np.zeros(T + 1, dtype=bool),
        np.zeros(T + 1, dtype=bool),
    )


def run_session(
    session: Session,
    strategy: str,
    classifier: ActivityClassifier,
    detector: EventDetector | None,
    policy: PolicyConfig | None = None,
    power: PowerModel | None = None,
    baseline: BaselineConfig | None = None,
    llm=None,
    run_llm: bool = True,
    burst_classifier: ActivityClassifier | None = None,
    seed: int = 0,
) -> RunResult:
    policy = policy or PolicyConfig()
    power = power or PowerModel()
    baseline = baseline or BaselineConfig()
    T = int(session.duration)
    sim = SensorSimulator(session, power, seed=seed)
    t_arr, true_arr, pred_arr, conf_arr, prob_arr, gps_arr, imu_arr = _new_arrays(T)

    state = np.empty(T + 1, dtype=object)
    state[:] = "idle"
    triggers: list[dict] = []
    llm_calls: list[dict] = []
    inferences_window = 0
    inferences_burst = 0
    active_s = 0.0
    cur_pred, cur_conf = -1, 0.0

    def record(t: int, st: str) -> None:
        t_arr[t] = t
        true_arr[t] = session.label_at(t)
        pred_arr[t] = cur_pred
        conf_arr[t] = cur_conf
        state[t] = st
        gps_arr[t] = sim.gps_on
        imu_arr[t] = sim.mode == "active"

    # ------------------------------------------------------------ always-on ----
    if strategy == "always_on":
        sim.configure("active")
        for t in range(1, T + 1):
            sim.advance_to(t)
            active_s += 1.0
            if t >= WINDOW_S:
                w = sim.window(t, WINDOW_S)
                if w is not None:
                    lab, cf, _ = classifier.predict_window(*w)
                    cur_pred, cur_conf = lab, cf
                    inferences_window += 1
            record(t, "active")

    # ------------------------------------------------------------- periodic ----
    elif strategy == "periodic":
        sim.configure("burst", burst_s=baseline.periodic_burst_s, period_s=baseline.periodic_period_s)
        for t in range(1, T + 1):
            sim.advance_to(t)
            if t % int(baseline.periodic_period_s) == 0:
                w = sim.window(t, baseline.periodic_burst_s)
                if w is not None:
                    lab, cf, _ = classifier.predict_window(*w)
                    cur_pred, cur_conf = lab, cf
                    inferences_window += 1
            record(t, "sampling" if sim.mode == "burst" else "sleep")

    # ---------------------------------------------------------- event-driven ----
    elif strategy == "event_driven":
        if detector is None:
            raise ValueError("event_driven requires a detector")
        sim.configure("burst", burst_s=policy.burst_s, period_s=policy.period_s)
        idle_hist: deque = deque(maxlen=policy.history)
        votes: deque = deque(maxlen=policy.burst_vote_window)
        recent_burst_labels: deque = deque(maxlen=2)
        committed = -1
        last_commit_t = 0.0
        cooldown_until = 0.0
        active_start = 0.0
        pending: list[tuple[int, float]] = []
        trigger_prob = 0.0
        need_bootstrap = policy.bootstrap
        min_ref_bursts = max(4, policy.history - 1)
        st = "idle"

        for t in range(1, T + 1):
            sim.advance_to(t)
            state_now = st

            if st == "idle":
                state_now = "idle"
                if t % int(policy.period_s) == 0:
                    pair = sim.burst_pair(t, policy.burst_s)
                    if pair is not None:
                        b_acc, b_gyro = pair
                        idle_hist.append(burst_stats(b_acc, FS))

                        # (1) cheap multi-fidelity update: 1 s window model
                        burst_label = None
                        if burst_classifier is not None:
                            lab_b, cf_b, _ = burst_classifier.predict_window(b_acc, b_gyro)
                            votes.append(lab_b)
                            recent_burst_labels.append(lab_b)
                            majority = Counter(votes).most_common(1)[0][0]
                            cur_pred, cur_conf = int(majority), float(cf_b)
                            inferences_burst += 1
                            burst_label = int(lab_b)

                        # (2) change detection on the burst history
                        boot = need_bootstrap and len(idle_hist) >= 1
                        ready = len(idle_hist) >= min_ref_bursts and t >= cooldown_until
                        prob = 0.0
                        fired_dec = False
                        info = {"p_change": 0.0, "p_rf": 0.0, "rule_z": 0.0}
                        if boot or ready:
                            if boot:
                                prob = 0.0
                            else:
                                fired_dec, info = detector.decide(np.stack(idle_hist))
                                prob = info["p_change"]
                            prob_arr[t] = prob

                        # (3) uncertainty trigger: the cheap model keeps
                        # disagreeing with the last confirmed state (2 of the
                        # last 3 bursts agree, with non-trivial confidence)
                        voted, voted_n = Counter(votes).most_common(1)[0] if votes else (committed, 0)
                        disagree = policy.uncertainty_trigger and (
                            burst_classifier is not None
                            and committed >= 0
                            and len(votes) >= 2
                            and int(voted) != committed
                            and voted_n >= 2
                            and cur_conf >= 0.5
                            and t >= cooldown_until
                        )

                        if boot or disagree or (ready and fired_dec):
                            st = "active"
                            active_start = t
                            pending = []
                            trigger_prob = prob
                            reason = "bootstrap" if boot else ("label_change" if disagree else "change_point")
                            need_bootstrap = False
                            triggers.append(
                                {
                                    "t": float(t),
                                    "p_change": round(float(prob), 3),
                                    "rule_z": round(float(detector.rule.zmax(np.stack(idle_hist))), 2),
                                    "prev_activity": ACTIVITIES[committed] if committed >= 0 else "unknown",
                                    "bootstrap": bool(boot),
                                    "reason": reason,
                                }
                            )
                            sim.configure("active")
                            state_now = "active"
                record(t, state_now)

            elif st == "active":
                active_s += 1.0
                state_now = "active"
                elapsed = t - active_start
                if elapsed >= WINDOW_S:
                    w = sim.window(t, WINDOW_S)
                    if w is not None:
                        lab, cf, _ = classifier.predict_window(*w)
                        inferences_window += 1
                        pending.append((lab, float(cf)))
                        cur_conf = float(cf)
                        confident = [l for l, c in pending if c >= policy.conf_high]
                        ready = (
                            len(confident) >= policy.confirm_windows
                            and len(set(confident[-policy.confirm_windows:])) == 1
                        )
                        timeout = elapsed >= policy.max_active_s
                        if ready or timeout:
                            if ready:
                                new_label, new_conf = confident[-1], float(pending[-1][1])
                            else:
                                best = max(range(len(pending)), key=lambda i: pending[i][1])
                                new_label, new_conf = pending[best][0], float(pending[best][1])

                            cur_pred, cur_conf = new_label, new_conf
                            if new_label != committed:
                                boot = committed < 0
                                onset = new_label in LOCOMOTION and committed >= 3
                                if onset and policy.gps_enabled and not sim.gps_on:
                                    sim.gps_activate(t, policy.gps_duty_s)
                                gps_ctx = sim.gps_context(t) if sim.gps_on else None
                                ctx = build_transition_context(
                                    t=t,
                                    previous_activity=ACTIVITIES[committed] if not boot else None,
                                    current_activity=ACTIVITIES[new_label],
                                    confidence=new_conf,
                                    duration_in_previous_s=t - last_commit_t,
                                    gps=gps_ctx,
                                    event_prob=trigger_prob,
                                )
                                if run_llm and llm is not None and policy.llm_on_transition and not boot:
                                    state_now = "context"
                                    res = llm.summarize_transition(ctx)
                                    llm_calls.append({"t": float(t), **ctx, **res})
                                committed = new_label
                                last_commit_t = t

                            # return to low-power monitoring with a fresh reference
                            seed_win = sim.window(t, WINDOW_S)
                            st = "idle"
                            sim.configure("burst", burst_s=policy.burst_s, period_s=policy.period_s)
                            cooldown_until = t + policy.cooldown_s
                            idle_hist.clear()
                            votes.clear()
                            recent_burst_labels.clear()
                            votes.extend([new_label] * policy.burst_vote_window)
                            if seed_win is not None:
                                n_b = int(round(FS))
                                for k in range(int(round(WINDOW_S))):
                                    chunk = seed_win[0][k * n_b : (k + 1) * n_b]
                                    if len(chunk) == n_b:
                                        idle_hist.append(burst_stats(chunk, FS))
                            state_now = "context" if state_now == "context" else "active"
                record(t, state_now)
            else:
                record(t, st)

    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    counters = {
        "samples": sim.counters.samples,
        "samples_possible": int(T * FS),
        "sensor_on_s": round(sim.counters.sensor_on_s, 2),
        "gps_on_s": round(sim.counters.gps_on_s, 2),
        "energy_mj": round(sim.counters.energy_mj, 1),
        "imu_energy_mj": round(power.imu_on_mw * sim.counters.sensor_on_s + power.imu_sleep_mw * T, 1),
        "inferences": inferences_window + inferences_burst,
        "inferences_window": inferences_window,
        "inferences_burst": inferences_burst,
        "llm_calls": len(llm_calls),
        "active_s": round(active_s, 2),
        "duty_cycle": round(sim.counters.sensor_on_s / max(1e-9, T), 4),
    }

    return RunResult(
        strategy=strategy,
        session_id=session.session_id,
        subject=session.subject,
        duration=float(T),
        t=t_arr,
        true=true_arr,
        pred=pred_arr,
        conf=conf_arr,
        state=list(state),
        event_prob=prob_arr,
        gps=gps_arr,
        imu_on=imu_arr,
        triggers=triggers,
        true_transitions=list(session.transitions),
        llm_calls=llm_calls,
        counters=counters,
    )
