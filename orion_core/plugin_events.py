"""
Plugin event hooks (Mark XXVI) — plugins that REACT, not just get called.

Until now a plugin was purely passive: it existed as a tool and did nothing until
the model invoked it. That rules out the whole class of plugin people actually
want — "when ORION goes offline, log it", "when a focus block ends, flash my
lamp", "when a security alert fires, push it to my phone".

A plugin can now declare which bus events it wants:

    { "name": "lamp", "module": "lamp_tool.py", "events": ["state", "speaking"] }

and export an optional handler beside ``run()``::

    def on_event(event: str, payload) -> None: ...

``PluginEventBridge`` connects the declared signals and fans them out. The rules
are the ones the rest of ORION lives by:

  * **A plugin can never break ORION.** Every handler call is wrapped; an
    exception is counted, logged once, and after ``MAX_FAULTS`` that plugin is
    muted for the session rather than allowed to fault on every event.
  * **Only declared events are delivered** — a plugin cannot quietly subscribe to
    everything, and the declaration is visible in its manifest.
  * **Unknown signal names are ignored**, so a manifest naming an event this
    build does not have is skipped with a log line, never a crash.
  * **Nothing here imports a plugin.** It binds modules the loader already
    imported, so subscribing adds no new execution surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

#: Signals a plugin may subscribe to. Deliberately a WHITELIST of observational,
#: non-destructive events — a plugin can watch what ORION is doing, but this is
#: not a hook into command execution or confirmation.
SUBSCRIBABLE: frozenset[str] = frozenset({
    "state",              # pipeline state changed (LISTENING/SPEAKING/…)
    "speaking",           # ORION started/stopped speaking
    "log",                # a log line
    "banner",             # a HUD banner
    "safety_alert",       # a safety/security alert
    "connection_state",   # online/offline transitions
    "emotion_changed",    # the face's emotional state
    "telemetry_sample",   # a telemetry sample
    "dashboard_event",    # a dashboard/diagnostics event
    "paused",             # paused/resumed
})

#: Consecutive faults before a plugin's handler is muted for the session.
MAX_FAULTS = 3

#: A hook runs on the SIGNAL's thread — under qasync that is the GUI/event loop
#: thread, so a slow handler stalls rendering AND the audio deadline. Anything
#: above this is reported; a plugin that keeps doing it is muted, because a
#: third-party plugin must never be able to make ORION stutter.
SLOW_MS = 50.0
MAX_SLOW = 5


@dataclass
class HookStats:
    """What a plugin's hooks have actually done — visible, not guessed."""
    delivered: int = 0
    faults: int = 0
    muted: bool = False
    last_error: str = ""
    events: tuple[str, ...] = ()
    #: Timing — a hook runs on the GUI thread, so its cost is ORION's cost.
    slow_calls: int = 0
    max_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def average_ms(self) -> float:
        return round(self.total_ms / self.delivered, 2) if self.delivered else 0.0


@dataclass
class PluginEventBridge:
    """Fans declared bus events out to plugin ``on_event`` handlers."""

    bus: Any = None
    log: Callable[[str], None] | None = None
    stats: dict[str, HookStats] = field(default_factory=dict)
    _bound: dict[str, list[str]] = field(default_factory=dict)
    _handlers: dict[str, Callable[[str, Any], Any]] = field(default_factory=dict)
    #: The live (signal, slot) pairs per plugin, so a hot-reload can DISCONNECT
    #: the previous binding instead of stacking a second one on top of it.
    _connections: dict[str, list[tuple[Any, Any]]] = field(default_factory=dict)
    #: Bumped on every (re)subscribe. A slot captures the generation it was made
    #: in and goes inert when superseded — so even a signal whose disconnect does
    #: not take (a stub, an exotic signal object) can never deliver twice.
    _generation: dict[str, int] = field(default_factory=dict)

    # ── wiring ────────────────────────────────────────────────────────────────

    def subscribe(self, name: str, module: Any, events) -> list[str]:
        """Bind one plugin's declared events. Returns the events actually bound.

        A module with no ``on_event`` is fine — it simply has no hooks, which is
        every plugin that shipped before this existed.
        """
        handler = getattr(module, "on_event", None)
        if not callable(handler):
            # A reload that removed on_event should also remove the old hooks.
            self.forget(name)
            return []
        # Re-subscribing (a hot-reload) must REPLACE the previous binding: two
        # live connections would deliver every event twice.
        self.forget(name)
        generation = self._generation[name] = self._generation.get(name, 0) + 1
        wanted = [str(e).strip() for e in (events or []) if str(e).strip()]
        bound: list[str] = []
        for event in wanted:
            if event not in SUBSCRIBABLE:
                self._emit(f"PLUGIN: '{name}' asked for unknown/forbidden event "
                           f"'{event}' — ignored.")
                continue
            signal = getattr(self.bus, event, None) if self.bus is not None else None
            if signal is None:
                self._emit(f"PLUGIN: '{name}' wants '{event}', which this build "
                           "does not emit — ignored.")
                continue
            try:
                slot = self._make_slot(name, event, generation)
                signal.connect(slot)
                self._connections.setdefault(name, []).append((signal, slot))
            except Exception as exc:
                self._emit(f"PLUGIN: could not bind '{name}' to '{event}' - {exc}")
                continue
            bound.append(event)
        if bound:
            self._handlers[name] = handler
            self._bound[name] = bound
            self.stats[name] = HookStats(events=tuple(bound))
            self._emit(f"PLUGIN: '{name}' hooked to {', '.join(bound)}.")
        return bound

    def _make_slot(self, name: str, event: str, generation: int):
        def _slot(*args: Any) -> None:
            if self._generation.get(name) != generation:
                return          # superseded by a later subscribe (hot-reload)
            payload = args[0] if len(args) == 1 else (args or None)
            self.dispatch(name, event, payload)
        return _slot

    # ── delivery ──────────────────────────────────────────────────────────────

    def dispatch(self, name: str, event: str, payload: Any = None) -> bool:
        """Deliver one event to one plugin. Never raises. Returns delivered?"""
        stats = self.stats.setdefault(name, HookStats())
        if stats.muted:
            return False
        handler = self._handlers.get(name)
        if handler is None:
            return False
        started = time.perf_counter()
        try:
            handler(event, payload)
        except Exception as exc:
            stats.faults += 1
            stats.last_error = f"{type(exc).__name__}: {exc}"[:160]
            if stats.faults >= MAX_FAULTS:
                stats.muted = True
                self._emit(f"PLUGIN: '{name}' muted after {stats.faults} hook "
                           f"faults — last: {stats.last_error}")
            else:
                self._emit(f"PLUGIN: '{name}' hook fault on '{event}' - "
                           f"{stats.last_error}")
            return False
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats.delivered += 1
        stats.total_ms += elapsed_ms
        stats.max_ms = max(stats.max_ms, elapsed_ms)
        if elapsed_ms > SLOW_MS:
            stats.slow_calls += 1
            if stats.slow_calls >= MAX_SLOW:
                stats.muted = True
                self._emit(
                    f"PLUGIN: '{name}' muted — {stats.slow_calls} hook calls over "
                    f"{SLOW_MS:.0f} ms (worst {stats.max_ms:.0f} ms). A hook runs on "
                    "ORION's own thread; move slow work to a background job.")
            else:
                self._emit(f"PLUGIN: '{name}' hook on '{event}' took "
                           f"{elapsed_ms:.0f} ms — that is on ORION's thread.")
        return True

    # ── introspection ─────────────────────────────────────────────────────────

    def forget(self, name: str) -> bool:
        """Disconnect a plugin's hooks (used on hot-reload and on disable)."""
        pairs = self._connections.pop(name, [])
        for signal, slot in pairs:
            try:
                signal.disconnect(slot)
            except Exception:
                pass            # already gone, or a stub signal in a test
        had = bool(pairs) or name in self._bound
        # Invalidate any slot still connected somewhere we could not reach.
        if name in self._generation:
            self._generation[name] += 1
        self._bound.pop(name, None)
        self._handlers.pop(name, None)
        return had

    def hooked(self) -> dict[str, list[str]]:
        return {name: list(events) for name, events in self._bound.items()}

    def unmute(self, name: str) -> bool:
        stats = self.stats.get(name)
        if stats is None or not stats.muted:
            return False
        stats.muted = False
        stats.faults = 0
        return True

    def report(self) -> str:
        if not self._bound:
            return "No plugin is subscribed to any event."
        lines = []
        for name, events in sorted(self._bound.items()):
            stats = self.stats.get(name, HookStats())
            line = f"  {name} ← {', '.join(events)}  ({stats.delivered} delivered"
            if stats.delivered:
                line += f", avg {stats.average_ms:.1f} ms, worst {stats.max_ms:.0f} ms"
            if stats.slow_calls:
                line += f", {stats.slow_calls} slow"
            if stats.faults:
                line += f", {stats.faults} fault(s)"
            if stats.muted:
                line += ", MUTED"
            lines.append(line + ")")
        return "Plugin event hooks:\n" + "\n".join(lines)

    def _emit(self, message: str) -> None:
        try:
            if self.log is not None:
                self.log(message)
            elif self.bus is not None:
                self.bus.log.emit(message)
        except Exception:
            pass


__all__ = ["SUBSCRIBABLE", "MAX_FAULTS", "SLOW_MS", "MAX_SLOW", "HookStats",
           "PluginEventBridge"]
