"""
ThoughtStream — concise reflections on recorded application context.

A background service produces useful observations about focus, conversation,
deadlines and activity. These are generated summaries of supplied evidence,
not access to a provider's hidden reasoning. They never claim that a recorded
action request succeeded without a separate result.

Strictly read-only introspection: a thought can never dispatch a tool, speak
aloud, or mutate state.  Every thought is

    • broadcast on ``bus.thought`` (command deck panel, phone mirror),
    • appended to ``config/thought_journal.jsonl`` for later reading.

Routine notes and an unavailable local-only model do not enter the activity
log. Real provider failures are reported once until generation recovers, with
a retry cooldown. Observed activity can still be shown without a model call.

Thinking is metered on its own provider slot ('thoughts' in api_keys.json —
its own key and model, usage task='thought' in the token ledger) and is
deliberately frugal: a reflection only runs when something actually changed
since the last one, and decision commentary is rate-limited.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .utils import first_line
from .reflections import concise_text, reflection_entry

THOUGHT_JOURNAL_PATH = CONFIG_DIR / "thought_journal.jsonl"


class ThoughtStream:
    """Periodic reflections plus rate-limited decision commentary."""

    DEFAULT_INTERVAL_S = 300.0     # reflection cadence when things are moving
    FIRST_RUN_DELAY_S = 90.0
    DECISION_GAP_S = 60.0          # at most one decision thought per minute
    JOURNAL_MAX_BYTES = 2_000_000  # rotate the journal before it grows silly
    FAILURE_RETRY_S = 60.0
    MAX_FAILURE_RETRY_S = 600.0
    DIAGNOSTIC_GAP_S = 300.0

    def __init__(
        self,
        bus: OrionBus,
        memory: Any,
        router: Any,
        cognitive_loop: Any | None = None,
        telemetry: Any | None = None,
        interval_s: float | None = None,
        journal_path: Path | str | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.router = router
        self.cognitive_loop = cognitive_loop
        self.telemetry = telemetry
        self.interval_s = float(
            interval_s
            or os.getenv("ORION_THOUGHT_INTERVAL_S", "")
            or self.DEFAULT_INTERVAL_S
        )
        self.enabled = os.getenv("ORION_THOUGHTS", "1").strip().lower() not in {
            "0", "false", "no", "off"}
        self.journal_path = Path(journal_path) if journal_path else THOUGHT_JOURNAL_PATH
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._pending_decisions: deque[str] = deque(maxlen=8)
        self._last_thought_at = 0.0
        self._last_decision_at = 0.0
        self._last_context_digest = ""
        self._thought_seq = 0
        self._retry_at = 0.0
        self._failure_count = 0
        self._last_diagnostic = ""
        self._diagnostic_at = 0.0
        self._external_note_connected = False
        self.recent: deque[dict[str, Any]] = deque(maxlen=50)
        if self.telemetry is not None:
            try:
                self.telemetry.health.register("thought_stream")
            except Exception:
                pass
        # Decisions worth explaining: tool dispatches and control-layer moves.
        for signal in ("agent_activity", "control_activity"):
            try:
                getattr(bus, signal).connect(self._note_decision)
            except Exception:
                pass
        # Maintenance and Forge can also publish factual observations on the
        # existing signal. Keep them available when Cognition opens later.
        try:
            bus.thought.connect(self._remember_external_note)
            self._external_note_connected = True
        except Exception:
            pass

    # ── decision intake (signal side — must stay non-async and cheap) ────────

    def _note_decision(self, *payload: Any) -> None:
        if self._stop.is_set():
            return
        summary = " ".join(str(p)[:120] for p in payload if p).strip()
        if summary:
            self._pending_decisions.append(summary)
            self._wake.set()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Cancel-safe background loop; every failure is contained."""
        if not self.enabled:
            return
        try:
            await asyncio.sleep(self.FIRST_RUN_DELAY_S)
            while not self._stop.is_set():
                try:
                    # Act on what is measurably true BEFORE thinking about it.
                    # This half costs nothing — no model, no tokens — and it
                    # runs even when thinking itself is rate-limited or the
                    # provider is down, which is exactly when ORION tidying up
                    # after himself matters most.
                    await self._act_without_tokens()
                    if self._pending_decisions and self._decision_gap_open():
                        await self._decision_thought()
                    elif time.monotonic() - self._last_thought_at >= self.interval_s:
                        await self._reflection_thought()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._diagnose(f"cycle recovered - {first_line(exc)}")
                # Sleep until the next cadence tick, but wake early for a
                # fresh decision so commentary lands while it is relevant.
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    async def _act_without_tokens(self) -> None:
        """Do the things ORION can settle for himself, for free.

        Deliberately keyed to measurements rather than to the WORDS of a
        thought: a thought can be wrong, a file size cannot, and reading a
        thought back to a model to decide what it meant would cost the tokens
        this is required to avoid. See thought_actions.
        """
        actor = getattr(self, "actor", None)
        if actor is None:
            return
        try:
            await asyncio.to_thread(actor.run_once)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._diagnose(f"self-maintenance skipped - {first_line(exc)}")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._external_note_connected:
            try:
                self.bus.thought.disconnect(self._remember_external_note)
            except Exception:
                pass
            self._external_note_connected = False

    def _decision_gap_open(self) -> bool:
        return time.monotonic() - self._last_decision_at >= self.DECISION_GAP_S

    # ── context assembly ──────────────────────────────────────────────────────

    def _observe(self) -> dict[str, Any]:
        """A compact, redaction-safe picture of the moment (no secrets)."""
        context: dict[str, Any] = {}
        if self.cognitive_loop is not None:
            try:
                digest = self.cognitive_loop.last_digest()
                context.update({k: digest[k] for k in
                                ("focus", "active_project", "deadlines", "fresh_turns")
                                if digest.get(k)})
            except Exception:
                pass
        try:
            turns = self.memory.recent_turns(limit=6)
            context["recent_conversation"] = [
                f"{t.get('role', '?')}: {str(t.get('content') or '')[:160]}"
                for t in turns if t.get("content")
            ]
        except Exception:
            pass
        try:
            status = self.router.status()
            context["providers"] = {
                "mode": status.get("mode"),
                "degraded": status.get("degraded"),
                "cooling_down": status.get("cooling_down"),
            }
        except Exception:
            pass
        return context

    @staticmethod
    def _digest(context: dict[str, Any]) -> str:
        try:
            return json.dumps(context, sort_keys=True, default=str)[:2000]
        except Exception:
            return str(context)[:2000]

    # ── thinking ──────────────────────────────────────────────────────────────

    async def _reflection_thought(self) -> None:
        context = self._observe()
        digest = self._digest(context)
        if digest == self._last_context_digest:
            return   # nothing changed since the last reflection — save tokens
        prompt = (
            "Write a concise application reflection. Context (observations "
            "only, may be sparse):\n" + digest +
            "\nSummarise the useful observation and, if appropriate, one next "
            "check. Use at most two sentences and 80 words, in UK English. "
            "Do not expose private reasoning, invent motivations or events, "
            "or claim completion without a recorded outcome."
        )
        if await self._think(prompt, kind="reflection", context=context):
            self._last_context_digest = digest

    async def _decision_thought(self) -> None:
        decisions = []
        while self._pending_decisions:
            decisions.append(self._pending_decisions.popleft())
        recent = "; ".join(decisions[-4:])
        prompt = (
            f"Recorded application activity (requests, not verified outcomes): {recent}. "
            "Summarise the activity and one useful follow-up check in at most "
            "two sentences and 80 words, in UK English. Do not expose private "
            "reasoning, invent motivations or assert that an action succeeded."
        )
        if await self._think(prompt, kind="decision", context=self._observe(),
                             evidence=decisions[-4:]):
            self._last_decision_at = time.monotonic()

    async def _think(self, prompt: str, kind: str, *,
                     context: dict[str, Any] | None = None,
                     evidence: list[str] | None = None) -> bool:
        if self._stop.is_set():
            return False
        self._thought_seq += 1
        seq = self._thought_seq
        at = datetime.now().strftime("%H:%M:%S")
        if time.monotonic() < self._retry_at:
            return self._observed_note(kind, seq, context, evidence)

        # The live "start" marker is emitted LAZILY, on the first real token, so
        # a provider that cannot stream never leaves a dangling header line — its
        # whole thought is typewriter-revealed through the full-thought path
        # instead.  ``got_delta`` records whether any token actually streamed.
        state = {"started": False, "got_delta": False}

        def _on_delta(piece: str) -> None:
            # Runs on the event-loop thread (inside the streaming read), so the
            # Qt signal emit is marshalled onto the GUI thread safely.
            piece = str(piece or "")
            if not piece or self._stop.is_set():
                return
            if not state["started"]:
                state["started"] = True
                self._emit_delta(seq, "start", "", kind=kind, model="", at=at)
            state["got_delta"] = True
            self._emit_delta(seq, "delta", piece, kind=kind, model="", at=at)

        profile: Any = None
        text = ""
        stream_ok = False
        from .providers import ThoughtGenerationUnavailable
        stream_fn = getattr(self.router, "generate_thought_stream", None)
        if stream_fn is not None:
            try:
                profile, text = await stream_fn(prompt, _on_delta)
                stream_ok = True
            except ThoughtGenerationUnavailable as exc:
                if state["started"]:
                    self._emit_delta(seq, "abort", "", kind=kind, model="", at=at)
                self._pause_generation(first_line(exc, 90), quiet=True)
                return self._observed_note(kind, seq, context, evidence)
            except Exception:
                stream_ok = False

        if not stream_ok or not state["got_delta"]:
            # Streaming was unavailable, failed, or produced no incremental
            # tokens.  Drop any partial line the GUI opened, fetch the full
            # thought, and reveal it through the typewriter (never truncated).
            if state["started"]:
                self._emit_delta(seq, "abort", "", kind=kind, model="", at=at)
                state["got_delta"] = False
            if not text:
                try:
                    profile, text = await self.router.generate_thought(prompt)
                except ThoughtGenerationUnavailable as exc:
                    self._pause_generation(first_line(exc, 90), quiet=True)
                    return self._observed_note(kind, seq, context, evidence)
                except Exception as exc:
                    self._pause_generation(first_line(exc, 90))
                    return self._observed_note(kind, seq, context, evidence)

        if self._stop.is_set():
            if state["started"]:
                self._emit_delta(seq, "abort", "", kind=kind, model="", at=at)
            return False
        text = str(text or "").strip()
        model = getattr(profile, "name", "model")
        if state["got_delta"]:
            # Streamed live token-by-token: close the stream so the GUI marks it
            # rendered; _emit's full thought is then de-duplicated by seq id.
            self._emit_delta(seq, "end", text, kind=kind, model=model, at=at)
        if not text:
            self._pause_generation("provider returned no reflection text")
            return self._observed_note(kind, seq, context, evidence)
        self._retry_at = 0.0
        self._failure_count = 0
        self._last_diagnostic = ""
        self._last_thought_at = time.monotonic()
        self._emit(text, kind=kind, model=model, seq=seq,
                   context=context, evidence=evidence)
        return True

    def _pause_generation(self, reason: str, *, quiet: bool = False) -> None:
        if self._stop.is_set():
            return
        self._failure_count = min(self._failure_count + 1, 8)
        self._retry_at = time.monotonic() + min(
            self.MAX_FAILURE_RETRY_S,
            self.FAILURE_RETRY_S * 2 ** (self._failure_count - 1))
        if not quiet:
            self._diagnose(f"skipped ({reason})")

    def _diagnose(self, message: str) -> None:
        """Keep real failures visible without repeating the same terminal line."""
        now = time.monotonic()
        if message != self._last_diagnostic:
            self.bus.log.emit(f"THOUGHT: {message}")
            self._last_diagnostic = message
            self._diagnostic_at = now

    def _observed_note(self, kind: str, seq: int, context: dict[str, Any] | None,
                       evidence: list[str] | None) -> bool:
        """Useful factual fallback; never spend tokens or pretend to infer intent."""
        if self._stop.is_set():
            return False
        context = context or {}
        notes: list[str] = []
        if evidence:
            activity = "; ".join(filter(None, (
                concise_text(item, 160) for item in evidence[-4:])))
            if activity:
                notes.append("Recent activity: " + activity)
        else:
            for field, label in (("focus", "Current focus"), ("active_project", "Active project")):
                value = concise_text(context.get(field), 120)
                if value:
                    notes.append(f"{label}: {value}.")
            providers = context.get("providers")
            if isinstance(providers, dict) and providers.get("degraded"):
                notes.append("AI availability is degraded; review provider diagnostics.")
        if not notes:
            return False
        self._last_thought_at = time.monotonic()
        self._emit(" ".join(notes), kind, "", seq, context=context,
                   evidence=evidence, origin="observation")
        return True

    def _emit_delta(self, seq: int, phase: str, text: str, *,
                    kind: str, model: str, at: str) -> None:
        try:
            self.bus.thought_delta.emit({
                "id": seq, "phase": phase, "text": text,
                "kind": kind, "model": model, "at": at,
            })
        except Exception:
            pass

    # ── surfacing ─────────────────────────────────────────────────────────────

    def _remember_external_note(self, payload: Any) -> None:
        """Cache externally recorded notes without duplicating this source's notes."""
        if self._stop.is_set():
            return
        if any(item is payload for item in self.recent):
            return
        entry = reflection_entry(payload)
        if entry is None:
            return
        fingerprint = hashlib.sha256(entry.event_id.encode("utf-8")).hexdigest()[:24]
        external_id = f"external:{fingerprint}"
        if any(item.get("id") == external_id for item in self.recent):
            return
        context = payload.get("context")
        safe_context = {
            field: concise_text(context.get(field), 120)
            for field in ("focus", "active_project")
            if isinstance(context, dict) and concise_text(context.get(field), 120)
        }
        note = {
            "at": entry.at or datetime.now().strftime("%H:%M:%S"),
            "observed_at": entry.observed_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": entry.kind,
            "text": entry.text,
            "model": concise_text(payload.get("model"), 80),
            "id": external_id,
            "context": safe_context,
            "evidence": list(entry.evidence),
            "origin": entry.origin,
        }
        self.recent.append(note)
        try:
            self.bus.dashboard_event.emit("reflections", {"entries": list(self.recent)})
        except Exception:
            pass
        self._journal(note)

    def _emit(self, text: str, kind: str, model: str, seq: int | None = None, *,
              context: dict[str, Any] | None = None,
              evidence: list[str] | None = None,
              origin: str = "generated_reflection") -> None:
        safe_context = {
            field: concise_text((context or {}).get(field), 120)
            for field in ("focus", "active_project")
            if concise_text((context or {}).get(field), 120)
        }
        payload = {
            "at": datetime.now().strftime("%H:%M:%S"),
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "text": concise_text(text),
            "model": model,
            "id": seq,
            "context": safe_context,
            "evidence": list(filter(None, (
                concise_text(item, 180) for item in (evidence or [])[-4:]))),
            "origin": origin,
        }
        self.recent.append(payload)
        try:
            self.bus.thought.emit(payload)
        except Exception:
            pass
        try:
            self.bus.dashboard_event.emit("thought", payload)
            self.bus.dashboard_event.emit("reflections", {"entries": list(self.recent)})
        except Exception:
            pass
        self._journal(payload)
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("thoughts.formed")
                self.telemetry.health.beat("thought_stream", "OK", f"last: {kind}")
            except Exception:
                pass

    def _journal(self, payload: dict[str, Any]) -> None:
        """Append-only JSONL journal with a simple size-based rotation."""
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            if self.journal_path.exists() \
                    and self.journal_path.stat().st_size > self.JOURNAL_MAX_BYTES:
                self.journal_path.replace(
                    self.journal_path.with_suffix(".jsonl.1"))
            entry = {"date": datetime.now().strftime("%Y-%m-%d"), **payload}
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass   # journalling must never break thinking
