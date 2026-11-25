"""Turn raw sensor events into compact structured context for the LLM.

The LLM never sees raw signals.  It sees a small JSON object - that is the
whole point of "understand only when needed": the expensive reasoning module
is fed a summary that a cheap policy decided was worth producing.
"""
from __future__ import annotations

from ..config import ACTIVITIES


def build_transition_context(
    *,
    t: float,
    previous_activity: str | None,
    current_activity: str,
    confidence: float,
    duration_in_previous_s: float,
    gps: dict | None,
    event_prob: float,
    state_history: list[str] | None = None,
) -> dict:
    ctx = {
        "event": "activity_transition",
        "timestamp_s": round(float(t), 1),
        "previous_activity": previous_activity or "unknown",
        "current_activity": current_activity,
        "confidence": round(float(confidence), 3),
        "duration_in_previous_activity_s": round(float(duration_in_previous_s), 1),
        "event_probability": round(float(event_prob), 3),
        "gps_context": gps or {"source": "not_activated"},
    }
    if state_history:
        ctx["recent_states"] = state_history[-5:]
    return ctx


def build_daily_summary(records: list[dict]) -> dict:
    """Aggregate a timeline into per-activity durations + transitions."""
    totals: dict[str, float] = {}
    transitions = []
    for r in records:
        lab = r.get("activity")
        if lab:
            totals[lab] = totals.get(lab, 0.0) + float(r.get("dt", 1.0))
        if r.get("transition"):
            transitions.append(r["transition"])
    return {
        "total_seconds": round(sum(totals.values()), 1),
        "activity_seconds": {k: round(v, 1) for k, v in totals.items()},
        "n_transitions": len(transitions),
        "transitions": transitions[-12:],
    }


def render_context_text(ctx: dict) -> str:
    """Compact textual form of a context object (used in prompts)."""
    gps = ctx.get("gps_context", {})
    gps_txt = "no GPS"
    if gps.get("source") == "simulated_gps":
        gps_txt = f"GPS speed {gps['speed_mps']} m/s ({'moving' if gps['moving'] else 'stationary'})"
    return (
        f"previous={ctx['previous_activity']} -> current={ctx['current_activity']} "
        f"(confidence {ctx['confidence']:.2f}) after {ctx['duration_in_previous_activity_s']}s; {gps_txt}"
    )


def activity_cn(name: str) -> str:
    from ..config import ACTIVITY_LABEL_CN

    return ACTIVITY_LABEL_CN.get(name, name)


__all__ = ["build_transition_context", "build_daily_summary", "render_context_text", "activity_cn", "ACTIVITIES"]
