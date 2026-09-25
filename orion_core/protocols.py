"""
ProtocolManager — JARVIS-style named protocols (macros).

A *protocol* is a named sequence of dispatcher tool calls that ORION runs on a
single command — "run my morning protocol", "engage focus protocol",
"wind-down protocol".  Protocols can be built-in or created by the user at
runtime (by voice or the dashboard) and are persisted to the PROJECT memory
tier so they survive restarts.

Each step is ``{"tool": <dispatcher tool>, "args": {...}}`` and is executed
through the same dispatcher every other capability uses, so protocols can do
anything ORION can do: open apps, deliver the briefing, list tasks, save the
workspace, set reminders, control media, and so on.

The manager announces start and completion through the proactive-voice channel
so a protocol *feels* like JARVIS running one.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from .bus import OrionBus
from .data import ToolResult
from .time_service import TIME
from .utils import first_line

# Built-in protocols shipped with ORION.  Steps are real dispatcher tool calls.
#
# Time awareness (brief §7-§9)
# ----------------------------
# A protocol must NOT assume the part of the day from the fact that the user
# ran it.  Running the morning protocol at 11pm should still be a real,
# useful run — it just has to know, and say, that it is not the morning.  Each
# time-of-day protocol therefore declares the period it is FOR; the manager
# compares that against the shared TimeService at run time and prefixes the
# announcement accordingly, rather than the protocol asserting a time.
BUILTIN_PROTOCOLS: dict[str, dict[str, Any]] = {
    "morning": {
        "description": "Morning start-up: overnight catch-up, today's tasks, "
                       "priority email, calendar, reminders and system health.",
        "period": "morning",
        # compose_source_material() already runs its own tasks/email sections
        # (_tasks_section/_email_section) — the separate notion_workspace/
        # outlook_mail steps that used to follow it just duplicated that
        # output. period="morning" is what makes the briefing itself an
        # overnight catch-up instead of a generic recent-news pull.
        "steps": [
            {"tool": "morning_briefing", "args": {"period": "morning"}},
            {"tool": "system_health", "args": {}},
        ],
    },
    "afternoon": {
        "description": "Afternoon check-in: what has developed since the "
                       "morning, outstanding tasks and system health.",
        "period": "afternoon",
        # period="afternoon" makes the briefing a "since this morning" delta
        # rather than a re-read of what was already delivered at breakfast —
        # the brief asks specifically that it avoid repeating itself.
        "steps": [
            {"tool": "morning_briefing", "args": {"period": "afternoon"}},
            {"tool": "system_health", "args": {}},
        ],
    },
    "evening": {
        "description": "Evening wind-down: the day's developments, unfinished "
                       "tasks, tomorrow's reminders and system status.",
        "period": "evening",
        "steps": [
            {"tool": "morning_briefing", "args": {"period": "evening"}},
            {"tool": "system_health", "args": {}},
            {"tool": "workspace_control", "args": {"action": "save"}},
        ],
    },
    "focus": {
        "description": "Focus mode: silence distractions and open the workspace.",
        "steps": [
            {"tool": "system_notify", "args": {"message": "FOCUS PROTOCOL ENGAGED", "priority": 2}},
            {"tool": "open_app", "args": {"app_name": "code"}},
            {"tool": "media_control", "args": {"action": "pause"}},
        ],
    },
    "wind_down": {
        "description": "End of day: save the workspace and stand down.",
        "steps": [
            {"tool": "workspace_control", "args": {"action": "save"}},
            {"tool": "system_notify", "args": {"message": "WIND-DOWN PROTOCOL — workspace saved", "priority": 2}},
        ],
    },
    "emergency": {
        "description": "Check for genuinely urgent events affecting you — "
                       "severe weather, infrastructure, transport, security, "
                       "cyber and ORION's own systems — and say what to do.",
        # No "steps": this one is not a macro.  Deciding whether something is
        # an emergency, how reliable the source is and whether it actually
        # reaches the user is a judgement, not a sequence of tool calls, so it
        # has its own pipeline (orion_core/emergency.py) bound at start-up.
        "native": "emergency",
        "steps": [],
    },
    "situation_report": {
        "description": "Full status: system health, connectivity mode and open tasks.",
        "steps": [
            {"tool": "ai_mode", "args": {}},
            {"tool": "sentinel", "args": {"action": "status"}},
            {"tool": "notion_workspace", "args": {"action": "list_tasks", "limit": 5}},
        ],
    },
}


class ProtocolManager:
    """Stores, resolves and runs named protocols through the dispatcher."""

    def __init__(self, bus: OrionBus, memory: Any, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        # Set post-construction to break the dispatcher↔manager cycle.
        self._dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]] | None = None
        self._user: dict[str, dict[str, Any]] = {}
        # Rate limiting + re-entrancy: a protocol double-triggered by a
        # repeated wake word or a double click must not run twice at once.
        self._last_run: dict[str, float] = {}
        self._running: set[str] = set()
        # Protocols that run their own pipeline instead of dispatcher steps.
        self._native: dict[str, Callable[[], Awaitable[Any]]] = {}
        self._load_user_protocols()

    def bind_dispatch(self, dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]]) -> None:
        self._dispatch = dispatch

    def bind_native(self, name: str, handler: Callable[[], Awaitable[Any]]) -> None:
        """Attach a protocol that runs its own pipeline rather than tool steps.

        The emergency protocol is the reason this exists: it has to weigh
        source reliability and whether an event actually reaches the user,
        which no list of tool calls can express.
        """
        self._native[str(name)] = handler

    # ── registry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(name: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")[:40]

    def all_protocols(self) -> dict[str, dict[str, Any]]:
        merged = dict(BUILTIN_PROTOCOLS)
        merged.update(self._user)   # user protocols override built-ins of same name
        return merged

    def resolve(self, phrase: str) -> str | None:
        """Find a protocol name from a spoken phrase like 'run the morning protocol'."""
        low = str(phrase or "").lower()
        protocols = self.all_protocols()
        # Exact slug match first.
        slug = self._slug(low.replace("protocol", "").replace("run", "").replace("engage", ""))
        if slug in protocols:
            return slug
        for name in protocols:
            if name.replace("_", " ") in low or name in low:
                return name
        return None

    def list_text(self) -> str:
        lines = ["Available protocols:"]
        for name, spec in self.all_protocols().items():
            tag = "" if name in self._user else " (built-in)"
            lines.append(f"- {name.replace('_', ' ')}{tag}: {spec.get('description', '')}")
        return "\n".join(lines)

    # ── execution ─────────────────────────────────────────────────────────────

    #: A single step may not hold the whole protocol up for ever.  Without
    #: this, one hung feed meant the protocol never finished and never said so.
    STEP_TIMEOUT_S = 90.0
    #: The emergency protocol is the one you cannot afford to wait on.
    EMERGENCY_STEP_TIMEOUT_S = 30.0
    #: Minimum gap between two runs of the same protocol (rate limiting).
    MIN_RERUN_S = 20.0

    def time_context(self, spec: dict[str, Any]) -> str:
        """Say the real time of day when it disagrees with the protocol's own.

        The brief is explicit: "The protocol must NOT assume it is morning
        simply because the user manually activates it."  So the protocol still
        runs in full — it just opens by naming the actual hour instead of
        greeting you with the wrong half of the day.
        """
        wanted = str(spec.get("period") or "").strip().lower()
        if not wanted:
            return ""
        actual = TIME.time_of_day()
        if actual == wanted:
            return ""
        return (f"Note: it is currently {actual}, not {wanted} — "
                f"{TIME.spoken_now()}. Running it anyway.")

    async def run(self, name: str) -> ToolResult:
        if self._dispatch is None:
            return ToolResult("Protocols are not wired to the dispatcher.", ok=False)
        protocols = self.all_protocols()
        key = name if name in protocols else (self.resolve(name) or "")
        spec = protocols.get(key)
        if spec is None:
            return ToolResult(
                f"I have no protocol matching '{name}'. {self.list_text()}", ok=False
            )
        # ── rate limiting: a protocol re-fired within seconds is a double
        #    trigger (a repeated wake word, a double click), not a request to
        #    run the whole thing twice concurrently.
        now = time.monotonic()
        last = self._last_run.get(key, 0.0)
        if now - last < self.MIN_RERUN_S:
            wait = self.MIN_RERUN_S - (now - last)
            return ToolResult(
                f"The {key.replace('_', ' ')} protocol ran {now - last:.0f} seconds "
                f"ago; ignoring a repeat for another {wait:.0f} seconds.", ok=False)
        if key in self._running:
            return ToolResult(
                f"The {key.replace('_', ' ')} protocol is already running.", ok=False)
        self._running.add(key)
        self._last_run[key] = now

        steps = spec.get("steps") or []
        timeout = (self.EMERGENCY_STEP_TIMEOUT_S if key == "emergency"
                   else self.STEP_TIMEOUT_S)
        context = self.time_context(spec)
        opening = f"Engaging the {key.replace('_', ' ')} protocol."
        if context:
            opening += f" {context}"
        self.bus.speak_request.emit(opening)
        self.bus.banner.emit(f"PROTOCOL: {key.upper()}", 3)
        results: list[str] = []
        ok_count = 0
        try:
            # A native protocol owns its own execution (see bind_native).
            native_key = str(spec.get("native") or "")
            handler = self._native.get(native_key) if native_key else None
            if native_key and handler is None:
                return ToolResult(
                    f"The {key.replace('_', ' ')} protocol is not wired up on "
                    "this instance, so I will not pretend to have run it.",
                    ok=False)
            if handler is not None:
                try:
                    outcome = await asyncio.wait_for(handler(), timeout=timeout)
                except asyncio.TimeoutError:
                    return ToolResult(
                        f"The {key.replace('_', ' ')} protocol timed out after "
                        f"{timeout:.0f} seconds.", ok=False)
                spoken = ""
                if isinstance(outcome, dict):
                    spoken = str(outcome.get("spoken") or "")
                elif outcome is not None:
                    spoken = str(getattr(outcome, "text", outcome))
                if spoken:
                    self.bus.speak_request.emit(spoken)
                return ToolResult((f"{context}\n" if context else "") + (spoken or
                                  f"{key.replace('_', ' ').title()} protocol complete."))

            for i, step in enumerate(steps, 1):
                tool = str(step.get("tool") or "")
                args = step.get("args") if isinstance(step.get("args"), dict) else {}
                if not tool:
                    continue
                try:
                    result = await asyncio.wait_for(self._dispatch(tool, args),
                                                    timeout=timeout)
                    ok_count += 1 if result.ok else 0
                    results.append(f"{i}. {tool}: {'ok' if result.ok else 'failed'} — "
                                   f"{first_line(result.text, 80)}")
                except asyncio.TimeoutError:
                    # A step that hangs must not take the protocol with it.
                    results.append(f"{i}. {tool}: TIMEOUT after {timeout:.0f}s — "
                                   "skipped, protocol continued")
                    self.bus.log.emit(
                        f"PROTOCOL: {key}/{tool} timed out after {timeout:.0f}s.")
                except asyncio.CancelledError:
                    results.append(f"{i}. {tool}: cancelled")
                    raise
                except Exception as exc:
                    results.append(f"{i}. {tool}: error — {first_line(exc, 80)}")
        finally:
            self._running.discard(key)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("protocol.run")
        summary = (f"{key.replace('_', ' ').title()} protocol complete — "
                   f"{ok_count}/{len(steps)} steps succeeded.")
        self.bus.speak_request.emit(summary)
        body = "\n".join(results)
        return ToolResult((f"{context}\n" if context else "") + summary + "\n" + body,
                          ok=ok_count > 0 or not steps)

    # ── authoring ─────────────────────────────────────────────────────────────

    def create(self, name: str, steps: list[dict[str, Any]], description: str = "") -> ToolResult:
        slug = self._slug(name)
        if not slug:
            return ToolResult("A protocol needs a name.", ok=False)
        if not isinstance(steps, list) or not steps:
            return ToolResult("A protocol needs at least one step (tool + args).", ok=False)
        clean = [
            {"tool": str(s.get("tool")), "args": s.get("args") if isinstance(s.get("args"), dict) else {}}
            for s in steps if isinstance(s, dict) and s.get("tool")
        ]
        if not clean:
            return ToolResult("None of the supplied steps had a valid tool.", ok=False)
        self._user[slug] = {"description": description or f"User protocol '{slug}'.", "steps": clean}
        self._save_user_protocols()
        return ToolResult(f"Protocol '{slug}' saved with {len(clean)} step(s).")

    def delete(self, name: str) -> ToolResult:
        slug = self._slug(name)
        if slug in self._user:
            del self._user[slug]
            self._save_user_protocols()
            return ToolResult(f"Protocol '{slug}' removed.")
        if slug in BUILTIN_PROTOCOLS:
            return ToolResult(f"'{slug}' is a built-in protocol and cannot be deleted.", ok=False)
        return ToolResult(f"No protocol named '{slug}'.", ok=False)

    # ── persistence (PROJECT memory tier) ─────────────────────────────────────

    def _save_user_protocols(self) -> None:
        try:
            self.memory.remember("project", "user_protocols", json.dumps(self._user), project="orion_system")
        except Exception:
            pass

    def _load_user_protocols(self) -> None:
        try:
            rows = self.memory.recall("project", project="orion_system", limit=50)
            for row in rows:
                if row.get("key_ref") == "user_protocols":
                    self._user = json.loads(row.get("value") or "{}")
                    break
        except Exception:
            self._user = {}
