"""On-demand LLM reasoning.

Two backends:

* ``openai``  - any OpenAI-compatible chat-completions endpoint.  Enabled when
  ``OPENAI_API_KEY`` is present (``OPENAI_BASE_URL`` / ``SWN_LLM_MODEL`` are
  optional).  No SDK dependency, just ``urllib``.
* ``offline`` - a deterministic template writer.  Used when no key is
  configured so that the whole pipeline, including the call accounting, runs
  end-to-end and reproducibly.  Every offline call is labelled as such in the
  results, so no experiment can silently pretend an LLM was used.

The module never touches sensors.  It is called by the policy, not by the
sensing loop.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .context_builder import render_context_text


SYSTEM_PROMPT = (
    "You are the context-understanding module of a wearable sensing system. "
    "You receive a small structured event produced by an activity-recognition "
    "pipeline. Write one or two short, factual sentences describing what the "
    "user is doing and how their activity changed. Do not invent data, do not "
    "give medical advice, and keep it under 45 words."
)

QA_PROMPT = (
    "You answer questions about a user's day using only the structured activity "
    "summary provided. Be concise and factual. If the summary does not contain "
    "the answer, say so."
)


class LLMAssistant:
    def __init__(self, backend: str = "auto", model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 timeout_s: float = 20.0, max_calls: int | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("SWN_LLM_MODEL") or "gpt-4o-mini"
        if backend == "auto":
            backend = "openai" if self.api_key else "offline"
        self.backend = backend
        self.timeout_s = timeout_s
        self.max_calls = max_calls
        self.calls = 0
        self.total_latency_s = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0

    # ------------------------------------------------------------- public ----
    @property
    def enabled(self) -> bool:
        return self.calls < (self.max_calls if self.max_calls is not None else 10 ** 9)

    def summarize_transition(self, ctx: dict) -> dict:
        prompt = (
            "Structured sensing event:\n"
            + json.dumps(ctx, ensure_ascii=False, indent=2)
            + "\n\nWrite the summary."
        )
        return self._generate(SYSTEM_PROMPT, prompt, kind="transition", ctx=ctx)

    def answer(self, question: str, summary: dict) -> dict:
        prompt = (
            "Structured activity summary:\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + f"\n\nQuestion: {question}"
        )
        return self._generate(QA_PROMPT, prompt, kind="qa", summary=summary, question=question)

    # ----------------------------------------------------------- internals ----
    def _generate(self, system: str, user: str, kind: str, **meta) -> dict:
        if not self.enabled:
            return {"text": "[LLM budget exhausted]", "backend": self.backend, "latency_s": 0.0, "skipped": True}
        t0 = time.perf_counter()
        if self.backend == "openai":
            try:
                text, usage = self._call_openai(system, user)
            except Exception as exc:  # network / quota / auth
                self.failures += 1
                text, usage = self._offline_fallback(kind, **meta), {}
                backend = "offline_fallback"
            else:
                backend = "openai"
        else:
            text, usage = self._offline_fallback(kind, **meta), {}
            backend = "offline"
        latency = time.perf_counter() - t0
        self.calls += 1
        self.total_latency_s += latency
        self.prompt_tokens += int(usage.get("prompt_tokens", len(system + user) // 4))
        self.completion_tokens += int(usage.get("completion_tokens", len(text) // 4))
        return {
            "text": text,
            "backend": backend,
            "model": self.model if backend == "openai" else "template",
            "latency_s": round(latency, 4),
            "kind": kind,
            **{k: v for k, v in meta.items() if k != "summary"},
        }

    def _call_openai(self, system: str, user: str) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 160,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return text, body.get("usage", {})

    def _offline_fallback(self, kind: str, **meta) -> str:
        if kind == "transition":
            ctx = meta.get("ctx", {})
            gps = ctx.get("gps_context", {})
            gps_txt = ""
            if gps.get("source") == "simulated_gps":
                gps_txt = (
                    f" GPS reports {gps['speed_mps']} m/s, "
                    f"{'so the user is moving' if gps['moving'] else 'consistent with a stationary period'}."
                )
            return (
                f"The user transitioned from {ctx.get('previous_activity', 'unknown')} to "
                f"{ctx.get('current_activity', 'unknown')} after about "
                f"{ctx.get('duration_in_previous_activity_s', 0):.0f} s "
                f"(model confidence {ctx.get('confidence', 0):.2f}).{gps_txt}"
            )
        summary = meta.get("summary", {})
        q = meta.get("question", "").lower()
        acts = summary.get("activity_seconds", {})
        if not acts:
            return "No activities were recorded in this window."
        if "how long" in q or "duration" in q:
            parts = [f"{k}: {v / 60:.1f} min" for k, v in sorted(acts.items(), key=lambda kv: -kv[1])]
            return "Recorded activity: " + ", ".join(parts) + "."
        if "when" in q or "change" in q or "transition" in q:
            trs = summary.get("transitions", [])
            if not trs:
                return "No activity transitions were detected in this window."
            return "Transitions: " + "; ".join(
                f"t={tr['t']:.0f}s {tr['from']}->{tr['to']}" for tr in trs
            ) + "."
        top = max(acts.items(), key=lambda kv: kv[1])
        return (
            f"In this window the user spent most time {top[0]} ({top[1] / 60:.1f} min out of "
            f"{summary.get('total_seconds', 0) / 60:.1f} min), with "
            f"{summary.get('n_transitions', 0)} detected transitions."
        )

    # ---------------------------------------------------------------- stats ----
    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model if self.backend == "openai" else "template",
            "calls": self.calls,
            "failures": self.failures,
            "total_latency_s": round(self.total_latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
