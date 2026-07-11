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

import json
from typing import Any, Awaitable, Callable

from .bus import OrionBus
from .data import ToolResult
from .utils import first_line

# Built-in protocols shipped with ORION.  Steps are real dispatcher tool calls.
BUILTIN_PROTOCOLS: dict[str, dict[str, Any]] = {
    "morning": {
        "description": "Morning start-up: briefing, today's tasks and priority email.",
        "steps": [
            {"tool": "morning_briefing", "args": {}},
            {"tool": "notion_workspace", "args": {"action": "list_tasks", "limit": 6}},
            {"tool": "outlook_mail", "args": {"action": "priority", "limit": 5}},
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
        self._load_user_protocols()

    def bind_dispatch(self, dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolResult]]) -> None:
        self._dispatch = dispatch

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
        lines = ["Available protocols, sir:"]
        for name, spec in self.all_protocols().items():
            tag = "" if name in self._user else " (built-in)"
            lines.append(f"- {name.replace('_', ' ')}{tag}: {spec.get('description', '')}")
        return "\n".join(lines)

    # ── execution ─────────────────────────────────────────────────────────────

    async def run(self, name: str) -> ToolResult:
        if self._dispatch is None:
            return ToolResult("Protocols are not wired to the dispatcher.", ok=False)
        protocols = self.all_protocols()
        key = name if name in protocols else (self.resolve(name) or "")
        spec = protocols.get(key)
        if spec is None:
            return ToolResult(
                f"I have no protocol matching '{name}', sir. {self.list_text()}", ok=False
            )
        steps = spec.get("steps") or []
        self.bus.speak_request.emit(f"Engaging the {key.replace('_', ' ')} protocol, sir.")
        self.bus.banner.emit(f"PROTOCOL: {key.upper()}", 3)
        results: list[str] = []
        ok_count = 0
        for i, step in enumerate(steps, 1):
            tool = str(step.get("tool") or "")
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            if not tool:
                continue
            try:
                result = await self._dispatch(tool, args)
                ok_count += 1 if result.ok else 0
                results.append(f"{i}. {tool}: {'ok' if result.ok else 'failed'} — "
                               f"{first_line(result.text, 80)}")
            except Exception as exc:
                results.append(f"{i}. {tool}: error — {first_line(exc, 80)}")
        if self.telemetry is not None:
            self.telemetry.metrics.incr("protocol.run")
        summary = f"{key.replace('_', ' ').title()} protocol complete — {ok_count}/{len(steps)} steps succeeded."
        self.bus.speak_request.emit(summary)
        return ToolResult(summary + "\n" + "\n".join(results))

    # ── authoring ─────────────────────────────────────────────────────────────

    def create(self, name: str, steps: list[dict[str, Any]], description: str = "") -> ToolResult:
        slug = self._slug(name)
        if not slug:
            return ToolResult("A protocol needs a name, sir.", ok=False)
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
        return ToolResult(f"Protocol '{slug}' saved with {len(clean)} step(s), sir.")

    def delete(self, name: str) -> ToolResult:
        slug = self._slug(name)
        if slug in self._user:
            del self._user[slug]
            self._save_user_protocols()
            return ToolResult(f"Protocol '{slug}' removed, sir.")
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
