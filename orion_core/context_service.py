"""Volatile context from the existing, explicitly started signal producers.

Session tokens prevent a late classifier result from reviving a stopped source.
Only accepted typed metadata is emitted on OrionBus; no device, disk, model or
network is owned here. Consumers read a fresh snapshot instead of caching it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from .context_events import (
    AudioObservation, ContextBroker, ContextEvent, ContextPayload, PublishResult,
    Sensitivity, SystemObservation, VisualObservation,
)

_SOURCES = {"camera.local": 5.0, "audio.local": 5.0, "system.monitor": 15.0}
_VISUAL_LABELS = {"appeared": "motion appeared", "left": "motion left",
                  "returned": "motion returned", "scene_changed": "scene changed",
                  "named": "object classified locally"}


@dataclass
class _SourceState:
    token: str = ""
    state: str = "off"
    last_signal_at: float | None = None
    last_event_at: float | None = None


class ContextSession:
    def __init__(self, service: ContextService, source: str, token: str) -> None:
        self.service, self.source, self.token = service, source, token

    def available(self, available: bool) -> None:
        self.service._available(self, available)

    def close(self, *, failed: bool = False) -> None:
        self.service._close(self, failed)

    def publish(self, payload: ContextPayload, *, confidence: float,
                ttl: float = 30.0, observed_at: float | None = None) -> bool:
        return self.service._publish(self, payload, confidence, ttl, observed_at)

    def visual(self, kind: str, *, observed_at: float | None = None) -> bool:
        label = _VISUAL_LABELS.get(kind)
        return bool(label and self.publish(VisualObservation(label), confidence=0.5,
                                            observed_at=observed_at))

    def audio(self, category: str, score: float, *, observed_at: float | None = None) -> bool:
        return self.publish(AudioObservation(category), confidence=score,
                            observed_at=observed_at)


class ContextService:
    def __init__(self, bus: Any = None, *, clock: Callable[[], float] = time.time) -> None:
        self.bus, self.clock = bus, clock
        self.broker = ContextBroker(clock=clock)
        self._sources = {name: _SourceState() for name in _SOURCES}
        self._lock = RLock()
        self._closed = False
        self._last_notify_at = float("-inf")
        self._system: ContextSession | None = None

    def open_source(self, source: str) -> ContextSession:
        if source not in _SOURCES:
            raise ValueError("unknown context source")
        with self._lock:
            if self._closed:
                raise RuntimeError("context service is closed")
            token = uuid4().hex
            self.broker.clear_source(source)
            self._sources[source] = _SourceState(token=token, state="starting")
            session = ContextSession(self, source, token)
            self._notify()
            return session

    def _current(self, session: ContextSession) -> _SourceState | None:
        state = self._sources.get(session.source)
        return state if not self._closed and state and state.token == session.token else None

    def _available(self, session: ContextSession, available: bool) -> None:
        with self._lock:
            state = self._current(session)
            if state is None:
                return
            new_state = "active" if available else "degraded"
            changed = state.state != new_state
            state.state = new_state
            if available:
                state.last_signal_at = self.clock()
            else:
                self.broker.clear_source(session.source)
            if changed or self.clock() - self._last_notify_at >= 1.0:
                self._notify()

    def _close(self, session: ContextSession, failed: bool) -> None:
        with self._lock:
            if self._current(session) is None:
                return
            self.broker.clear_source(session.source)
            self._sources[session.source] = _SourceState(state="degraded" if failed else "off")
            self._notify()

    def _publish(self, session: ContextSession, payload: ContextPayload,
                 confidence: float, ttl: float, observed_at: float | None) -> bool:
        with self._lock:
            state = self._current(session)
            if state is None or state.state != "active":
                return False
            expected = {"camera.local": VisualObservation, "audio.local": AudioObservation,
                        "system.monitor": SystemObservation}[session.source]
            if not isinstance(payload, expected):
                raise ValueError("context payload does not match its source")
            observed = self.clock() if observed_at is None else observed_at
            event = ContextEvent(
                event_id=uuid4().hex, source=session.source, observed_at=observed,
                expires_at=observed + ttl, confidence=confidence,
                sensitivity=(Sensitivity.PUBLIC if session.source == "system.monitor"
                             else Sensitivity.PERSONAL), payload=payload,
            )
            outcome = self.broker.publish(event)
            if outcome not in {PublishResult.ACCEPTED, PublishResult.COALESCED}:
                return False
            state.last_event_at = observed
            self._emit("context_event", event)
            self._notify()
            return True

    def observe_resources(self, governor: Any) -> None:
        """Consume the governor's latest sample, without doing another probe."""
        with self._lock:
            if self._closed:
                return
            if self._system is None:
                self._system = self.open_source("system.monitor")
            status = governor.status()
            # A failed probe must not renew the age of its preceding reading.
            fresh = getattr(governor, "memory_sample_fresh", False)
            self._system.available(fresh)
            if not fresh:
                return
            for key in ("memory_used_percent", "cpu_percent", "gpu_memory_percent"):
                value = status.get(key)
                if (not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) and 0 <= value <= 100):
                    self._system.publish(SystemObservation(key, value, "%"),
                                         confidence=1.0, ttl=15.0)

    def snapshot(self, *, max_items: int = 12) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            sources = {}
            for name, state in self._sources.items():
                status = state.state
                if (status == "active" and (state.last_signal_at is None
                        or now - state.last_signal_at > _SOURCES[name])):
                    status = "degraded"
                    self.broker.clear_source(name)
                sources[name] = {"state": status, "last_signal_at": state.last_signal_at,
                                 "last_event_at": state.last_event_at,
                                 "fresh_until": (state.last_signal_at + _SOURCES[name]
                                                 if state.last_signal_at is not None else None)}
            events = [e.to_dict() for e in self.broker.summary(max_items=max_items).events
                      if sources[e.source]["state"] == "active"]
            return {"generated_at": now, "sources": sources, "events": events}

    def report(self) -> str:
        snapshot = self.snapshot()
        lines = ["Current local context:"]
        for name, state in snapshot["sources"].items():
            last = state["last_event_at"]
            age = (f"; last event {max(0, snapshot['generated_at'] - last):.0f}s ago"
                   if last is not None else "; no event this session")
            lines.append(f"- {name}: {state['state']}{age}")
        for event in snapshot["events"]:
            payload = event["payload"]
            value = (f"{payload['metric']} {payload['value']:g}{payload['unit'] or ''}"
                     if payload["kind"] == "system" else payload["label"])
            age = max(0, snapshot["generated_at"] - event["observed_at"])
            lines.append(f"- {event['source']}: {value} (observed {age:.0f}s ago; "
                         f"confidence {event['confidence']:.2f})")
        if any(event["payload"]["kind"] == "visual" for event in snapshot["events"]):
            lines.append("Visual confidence is a fixed heuristic, not a calibrated probability.")
        return "\n".join(lines)

    def _emit(self, signal: str, *args: Any) -> None:
        try:
            getattr(self.bus, signal).emit(*args)
        except Exception:
            pass

    def _notify(self) -> None:
        self._last_notify_at = self.clock()
        self._emit("dashboard_event", "context", self.snapshot())

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.broker.clear()
            self._sources = {name: _SourceState() for name in _SOURCES}
            self._notify()
