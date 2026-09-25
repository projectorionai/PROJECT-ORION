"""
CommandRouter — one place where a named command becomes a real action.

Why
---
The Command Palette listed ~100 dispatcher tools and every Command Deck page,
and choosing any of them did the same thing: typed its name into the message
box.  A palette entry called "System Scan" that types the words "system scan"
into a text field is a label pretending to be a control.  The brief is explicit
about this — no dead buttons, no decorative controls, and every action must map:

    UI Action -> Command ID -> Command Router -> Handler -> Subsystem
              -> Result -> Event -> UI

This module is the "Command Router -> Handler" link.  A command has:

    id            stable, dotted, e.g. "system.scan", "protocol.emergency"
    title         what the user reads
    subtitle      what it actually does, in one line
    handler       an async callable returning a ToolResult-shaped object
    destructive   whether it must be confirmed before running
    subsystem     which part of ORION owns it (shown in the result, and used
                  by the health model)

Everything is registered against a real backend.  A command whose backend is
not available is not registered at all, so the palette can never list something
that cannot run — an unavailable capability is reported honestly rather than
being offered and then failing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from .data import ToolResult
from .utils import first_line

Handler = Callable[[], Awaitable[Any]]


@dataclass(frozen=True)
class Command:
    """One executable command."""

    id: str
    title: str
    subtitle: str
    subsystem: str
    handler: Handler
    destructive: bool = False

    async def run(self) -> ToolResult:
        result = await self.handler()
        if isinstance(result, ToolResult):
            return result
        if result is None:
            return ToolResult(f"{self.title} completed.")
        return ToolResult(str(result))


class CommandRouter:
    """Registry + execution for the palette's real commands.

    Construction takes the live subsystems rather than reaching for globals, so
    the router is testable and so a missing subsystem simply means fewer
    commands rather than a crash at click time.
    """

    def __init__(
        self,
        bus: Any,
        *,
        dispatcher: Any = None,
        protocols: Any = None,
        health: Any = None,
        recovery: Any = None,
        worker: Any = None,
        deck: Any = None,
    ) -> None:
        self.bus = bus
        self.dispatcher = dispatcher
        self.protocols = protocols
        self.health = health
        self.recovery = recovery
        self.worker = worker
        self.deck = deck
        self._commands: dict[str, Command] = {}
        self._register_builtins()

    # ── registry ──────────────────────────────────────────────────────────────

    def register(self, command: Command) -> None:
        self._commands[command.id] = command

    def commands(self) -> list[Command]:
        return sorted(self._commands.values(), key=lambda c: c.id)

    def get(self, command_id: str) -> Command | None:
        return self._commands.get(command_id)

    def ids(self) -> list[str]:
        return sorted(self._commands)

    # ── execution ─────────────────────────────────────────────────────────────

    async def run(self, command_id: str) -> ToolResult:
        """Execute *command_id*, publishing the outcome on the bus.

        Never raises: a palette entry that throws a traceback at the user is a
        worse failure than one that reports what went wrong.
        """
        command = self._commands.get(command_id)
        if command is None:
            return ToolResult(f"No command with id '{command_id}'.", ok=False)
        self._log(f"COMMAND: {command.id} ({command.subsystem}) started.")
        self._emit("command_started", {"id": command.id, "title": command.title,
                                       "subsystem": command.subsystem})
        try:
            result = await command.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = first_line(exc, 200)
            self._log(f"COMMAND: {command.id} failed - {detail}")
            self._emit("command_result", {"id": command.id, "ok": False,
                                          "text": detail,
                                          "subsystem": command.subsystem})
            return ToolResult(f"{command.title} failed: {detail}", ok=False)
        self._log(f"COMMAND: {command.id} -> {'ok' if result.ok else 'failed'}.")
        self._emit("command_result", {"id": command.id, "ok": result.ok,
                                      "text": first_line(result.text, 300),
                                      "subsystem": command.subsystem})
        return result

    def _log(self, message: str) -> None:
        try:
            self.bus.log.emit(message)
        except Exception:
            pass

    def _emit(self, channel: str, payload: dict[str, Any]) -> None:
        """Results reach the UI as an EVENT, never by touching a widget."""
        try:
            self.bus.dashboard_event.emit(channel, payload)
        except Exception:
            pass

    # ── built-in commands (each wired to a real subsystem) ────────────────────

    def _register_builtins(self) -> None:
        self._register_protocol_commands()
        self._register_system_commands()
        self._register_audio_commands()
        self._register_diagnostic_commands()

    def _register_protocol_commands(self) -> None:
        protocols = self.protocols
        if protocols is None:
            return
        available = {}
        try:
            available = protocols.all_protocols()
        except Exception:
            return
        for name, spec in available.items():
            self.register(Command(
                id=f"protocol.{name}",
                title=f"{name.replace('_', ' ').title()} Protocol",
                subtitle=str(spec.get("description") or "Run this protocol."),
                subsystem="ProtocolManager",
                handler=(lambda n=name: protocols.run(n)),
                # A protocol runs real tools; the emergency one also speaks.
                destructive=bool(spec.get("destructive", False)),
            ))

    def _register_system_commands(self) -> None:
        dispatcher = self.dispatcher
        if dispatcher is None:
            return

        def tool(name: str, **args: Any) -> Handler:
            async def _run() -> Any:
                return await dispatcher.dispatch_chain(name, dict(args))
            return _run

        known = set()
        try:
            known = {str(d.get("name")) for d in
                     getattr(dispatcher, "declarations", None) or []}
        except Exception:
            known = set()

        # Only register what the dispatcher genuinely exposes.  Guessing at a
        # tool name would put an entry in the palette that fails when clicked.
        candidates = [
            ("system.scan", "System Scan",
             "Scan hardware, services and resource usage now.",
             "SystemMonitoring", "system_health", {}),
            ("system.diagnostics", "Diagnostics Report",
             "Full self-diagnostic across every subsystem.",
             "Diagnostics", "diagnostics", {}),
            ("security.status", "Security Status",
             "Current security posture from the sentinel.",
             "SecuritySentinel", "sentinel", {"action": "status"}),
            ("memory.search", "Memory Overview",
             "What ORION currently holds in memory.",
             "MemoryManager", "memory_search", {"query": ""}),
            ("briefing.now", "Intelligence Briefing",
             "Compose and deliver the current briefing.",
             "BriefingService", "morning_briefing", {}),
            ("ai.mode", "Provider Status",
             "Which model provider is live, and the fallback chain.",
             "ProviderRouter", "ai_mode", {}),
        ]
        for cid, title, subtitle, subsystem, tool_name, args in candidates:
            if known and tool_name not in known:
                continue
            self.register(Command(
                id=cid, title=title, subtitle=subtitle, subsystem=subsystem,
                handler=tool(tool_name, **args),
            ))

    def _register_audio_commands(self) -> None:
        recovery = self.recovery
        if recovery is not None:
            async def restore_audio() -> ToolResult:
                report = recovery.wake("command palette", force=True)
                return ToolResult(report.summary(), ok=report.ok or report.skipped)

            self.register(Command(
                id="audio.restore",
                title="Restore Audio Devices",
                subtitle="Re-verify and reopen ORION's speaker and microphone.",
                subsystem="AudioRecovery",
                handler=restore_audio,
            ))

        async def audio_devices() -> ToolResult:
            from . import audio_devices as ad
            out = ad.verify("output")
            inp = ad.verify("input")
            return ToolResult(
                f"{ad.describe()}\n\nOutput check: {out.detail}\nInput check: {inp.detail}",
                ok=out.ok and inp.ok)

        self.register(Command(
            id="audio.devices",
            title="Audio Device Check",
            subtitle="Prove ORION's speaker and microphone actually open.",
            subsystem="AudioDevices",
            handler=audio_devices,
        ))

    def _register_diagnostic_commands(self) -> None:
        async def why_ignored() -> ToolResult:
            from .request_trace import TRACES
            return ToolResult(TRACES.explain_last_silence())

        self.register(Command(
            id="diagnostics.last_silence",
            title="Why Did You Ignore Me?",
            subtitle="Name the subsystem that dropped the last request.",
            subsystem="RequestTrace",
            handler=why_ignored,
        ))

        async def trace_report() -> ToolResult:
            from .request_trace import TRACES
            return ToolResult(TRACES.report())

        self.register(Command(
            id="diagnostics.traces",
            title="Request Trace History",
            subtitle="Recent requests and where each one got to.",
            subsystem="RequestTrace",
            handler=trace_report,
        ))

        health = self.health
        if health is not None:
            async def health_report() -> ToolResult:
                snapshot = health.snapshot()
                return ToolResult(health.render(snapshot),
                                  ok=health.overall(snapshot) != "OFFLINE")

            self.register(Command(
                id="system.health",
                title="System Health",
                subtitle="Live state of every ORION subsystem.",
                subsystem="HealthModel",
                handler=health_report,
            ))

    # ── palette integration ───────────────────────────────────────────────────

    def palette_entries(self) -> list[tuple[str, str, str, bool]]:
        """(id, title, subtitle, destructive) for every registered command."""
        return [(c.id, c.title, c.subtitle, c.destructive) for c in self.commands()]


def run_soon(coro: Awaitable[Any], loop: asyncio.AbstractEventLoop | None) -> None:
    """Schedule *coro* on ORION's event loop from the Qt GUI thread.

    The GUI thread must never await, and it must never call into asyncio
    directly — both block the event loop the brief requires stay responsive.
    A reference to the task is kept until it finishes, because a task with no
    live reference can be garbage-collected mid-flight (one of the ways a
    command silently did nothing).
    """
    if loop is None or loop.is_closed():
        return
    def _spawn() -> None:
        task = asyncio.ensure_future(coro)
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
    loop.call_soon_threadsafe(_spawn)


_PENDING: set[Any] = set()


__all__ = ["Command", "CommandRouter", "run_soon"]
