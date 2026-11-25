"""Run the three strategies on held-out subjects and aggregate results."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np

from ..config import (
    ACTIVITIES,
    DEFAULT_BASELINE,
    DEFAULT_POLICY,
    DEFAULT_POWER,
    FS,
    RESULT_DIR,
    TIMELINE_DIR,
)
from ..data.sessions import Session
from ..llm.context_builder import build_daily_summary
from ..llm.llm_assistant import LLMAssistant
from ..models.activity_classifier import ActivityClassifier
from ..models.event_detector import EventDetector
from ..runtime import STRATEGIES, RunResult, run_session
from .metrics import event_metrics, merge_event_metrics, tracking_metrics, transition_latency


def run_strategy(
    sessions: list[Session],
    strategy: str,
    classifier: ActivityClassifier,
    detector: EventDetector | None,
    llm_factory=None,
    seed: int = 0,
    burst_classifier: ActivityClassifier | None = None,
) -> dict:
    per_session = []
    cpu_total = 0.0
    latencies = []
    llm_stats = {"calls": 0, "failures": 0, "total_latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0} if llm_factory else None
    peak_mb = 0.0
    results: list[RunResult] = []

    for i, s in enumerate(sessions):
        llm = llm_factory() if llm_factory is not None else None
        t0 = time.process_time()
        res = run_session(
            s, strategy, classifier, detector,
            policy=DEFAULT_POLICY, power=DEFAULT_POWER, baseline=DEFAULT_BASELINE,
            llm=llm, run_llm=llm is not None, seed=seed + i,
            burst_classifier=burst_classifier,
        )
        cpu_total += time.process_time() - t0
        results.append(res)

        m = tracking_metrics(res.true, res.pred)
        m["session_id"] = s.session_id
        m["subject"] = s.subject
        m["duration"] = res.duration
        m["counters"] = res.counters
        m["llm"] = llm.stats() if llm is not None else None
        if llm is not None:
            st = llm.stats()
            for k in ("calls", "failures", "prompt_tokens", "completion_tokens"):
                llm_stats[k] += st[k]
            llm_stats["total_latency_s"] += st["total_latency_s"]
            llm_stats["backend"] = st["backend"]
            llm_stats["model"] = st["model"]
        if strategy == "event_driven":
            # bootstrap is a cold-start classification, not a detected change
            m["events"] = event_metrics(
                [x for x in res.triggers if not x.get("bootstrap")], res.true_transitions
            )
        m["switches"] = transition_latency(res.true, res.pred, res.true_transitions)
        per_session.append(m)
        latencies.extend([x["latency_s"] for x in m["switches"] if x["latency_s"] is not None])

    # peak memory of one representative run (tracemalloc, outside the timing loop)
    if sessions:
        tracemalloc.start()
        run_session(
            sessions[0], strategy, classifier, detector,
            policy=DEFAULT_POLICY, power=DEFAULT_POWER, baseline=DEFAULT_BASELINE,
            llm=None, run_llm=False, seed=seed, burst_classifier=burst_classifier,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1e6

    # ------------------------------------------------------------ aggregate ----
    total_n = sum(m["n"] for m in per_session)
    total_s = sum(m["duration"] for m in per_session)
    acc = sum(m["accuracy"] * m["n"] for m in per_session) / max(1, total_n)
    f1 = sum(m["macro_f1"] * m["n"] for m in per_session) / max(1, total_n)
    per_class = {}
    for a in ACTIVITIES:
        vals = [(m["per_class_f1"][a], m["n"]) for m in per_session if a in m["per_class_f1"]]
        if vals:
            per_class[a] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
        else:
            per_class[a] = float("nan")
    counters = {
        k: sum(m["counters"][k] for m in per_session) / max(1e-9, total_s) * 3600.0
        for k in ("samples", "sensor_on_s", "gps_on_s", "energy_mj", "imu_energy_mj",
                  "inferences", "inferences_window", "inferences_burst", "llm_calls")
    }
    counters["duty_cycle"] = sum(m["counters"]["sensor_on_s"] for m in per_session) / max(1e-9, total_s)
    counters["samples_per_hour"] = counters["samples"]
    counters["energy_j_per_hour"] = counters["energy_mj"] / 1000.0
    counters["cpu_ms_per_hour"] = cpu_total / max(1e-9, total_s) * 3600 * 1000
    counters["cpu_ms_per_min"] = cpu_total / max(1e-9, total_s) * 60 * 1000
    counters["peak_memory_mb"] = peak_mb
    counters["llm_calls_per_hour"] = counters["llm_calls"]

    out = {
        "strategy": strategy,
        "n_sessions": len(per_session),
        "duration_s": round(total_s, 1),
        "accuracy": acc,
        "macro_f1": f1,
        "per_class_f1": per_class,
        "coverage": float(np.average([m["coverage"] for m in per_session], weights=[max(1, m["n"]) for m in per_session])),
        "switch_latency_s": {
            "n": len(latencies),
            "mean": float(np.mean(latencies)) if latencies else None,
            "median": float(np.median(latencies)) if latencies else None,
        },
        "counters": counters,
        "llm": llm_stats,
        "sessions": per_session,
        "results": results,
    }
    if strategy == "event_driven":
        out["events"] = merge_event_metrics([m["events"] for m in per_session])
    return out


def _strip(results: dict) -> dict:
    return {k: v for k, v in results.items() if k != "results"}


def compare_strategies(sessions: list[Session], classifier: ActivityClassifier,
                       detector: EventDetector | None, backend: str = "auto",
                       demo_session_id: str | None = None,
                       burst_classifier: ActivityClassifier | None = None) -> dict:
    """Run all strategies and persist comparison + demo timelines."""
    import json

    all_res = {}
    for strategy in STRATEGIES:
        llm_factory = (lambda: LLMAssistant(backend=backend)) if strategy == "event_driven" else None
        print(f"[eval] {strategy} ...", flush=True)
        all_res[strategy] = run_strategy(sessions, strategy, classifier, detector, llm_factory,
                                          burst_classifier=burst_classifier)

    # pick a demo session: the one with most transitions under event_driven
    if demo_session_id is None:
        best = max(all_res["event_driven"]["results"], key=lambda r: len(r.true_transitions))
        demo_session_id = best.session_id

    demo = {}
    for strategy, res in all_res.items():
        target = next((r for r in res["results"] if r.session_id == demo_session_id), None)
        if target is None:
            continue
        demo[strategy] = {
            "session_id": target.session_id,
            "subject": target.subject,
            "duration": target.duration,
            "timeline": target.timeline(),
            "triggers": target.triggers,
            "transitions": [
                {"t": t, "from": ACTIVITIES[a], "to": ACTIVITIES[b]}
                for t, a, b in target.true_transitions
            ],
            "llm_calls": target.llm_calls,
            "counters": target.counters,
        }
    (TIMELINE_DIR / f"demo_{demo_session_id}.json").write_text(
        json.dumps(demo, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    comparison = {
        "demo_session_id": demo_session_id,
        "strategies": {k: _strip(v) for k, v in all_res.items()},
    }
    (RESULT_DIR / "experiments.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # csv summary
    lines = ["strategy,accuracy,macro_f1,duty_cycle,energy_j_per_hour,samples_per_hour,"
             "inferences_per_hour,gps_on_s_per_hour,llm_calls_per_hour,median_switch_latency_s"]
    for k, v in all_res.items():
        c = v["counters"]
        lat = v["switch_latency_s"]["median"]
        lines.append(
            f"{k},{v['accuracy']:.4f},{v['macro_f1']:.4f},{c['duty_cycle']:.4f},"
            f"{c['energy_j_per_hour']:.1f},{c['samples_per_hour']:.0f},"
            f"{c['inferences']:.0f},{c['gps_on_s']:.0f},{c['llm_calls']:.1f},"
            f"{'' if lat is None else round(lat, 2)}"
        )
    (RESULT_DIR / "strategy_comparison.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def demo_qa(session_timeline: list[dict], question: str, backend: str = "auto") -> dict:
    """Answer a natural-language question over a recorded timeline."""
    records = [
        {"activity": r["pred_name"], "dt": 1.0, "transition": None}
        for r in session_timeline
        if r["pred"] >= 0
    ]
    summary = build_daily_summary(records)
    llm = LLMAssistant(backend=backend)
    return llm.answer(question, summary)


__all__ = ["run_strategy", "compare_strategies", "demo_qa", "FS", "DEFAULT_POLICY"]
