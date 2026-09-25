"""
Connection state machine for the live channel and provider failover (Section 4).

A server ``go_away`` (or any drop) must never leave ORION silent or frozen.
This module gives failover an explicit, inspectable state machine so every
transition — reconnect, credential rotation, provider switch, local fallback,
degraded/offline, shutdown — is deliberate and observable, rather than implied
by scattered booleans.

Pure and deterministic (no Qt, no networking): the live worker drives it and
publishes each transition; the diagnostics UI renders ``describe()``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    COOLING_DOWN = "cooling_down"
    ROTATING_CREDENTIAL = "rotating_credential"
    SWITCHING_PROVIDER = "switching_provider"
    LOCAL_FALLBACK = "local_fallback"
    DEGRADED_OFFLINE = "degraded_offline"
    SHUTTING_DOWN = "shutting_down"


class GoAwayClass(str, Enum):
    """How a ``go_away`` (or disconnect) should be handled."""

    RECOVERABLE = "recoverable"      # transient — reconnect on the same credential
    RATE_LIMIT = "rate_limit"        # backing off / cool the credential
    AUTH = "auth"                    # invalid/expired key — rotate credential
    MAINTENANCE = "maintenance"      # server maintenance — reconnect with backoff
    TERMINAL = "terminal"            # do not retry this channel


# Legal transitions.  SHUTTING_DOWN is reachable from anywhere; every active
# state can drop back toward reconnect/cooldown so failover never dead-ends.
_TRANSITIONS: dict[ConnectionState, set[ConnectionState]] = {
    ConnectionState.DISCONNECTED: {
        ConnectionState.CONNECTING, ConnectionState.DEGRADED_OFFLINE,
        ConnectionState.LOCAL_FALLBACK,
    },
    ConnectionState.CONNECTING: {
        ConnectionState.CONNECTED, ConnectionState.RECONNECTING,
        ConnectionState.COOLING_DOWN, ConnectionState.ROTATING_CREDENTIAL,
        ConnectionState.SWITCHING_PROVIDER, ConnectionState.LOCAL_FALLBACK,
        ConnectionState.DEGRADED_OFFLINE, ConnectionState.DISCONNECTED,
    },
    ConnectionState.CONNECTED: {
        ConnectionState.RECONNECTING, ConnectionState.COOLING_DOWN,
        ConnectionState.ROTATING_CREDENTIAL, ConnectionState.SWITCHING_PROVIDER,
        ConnectionState.LOCAL_FALLBACK, ConnectionState.DISCONNECTED,
    },
    ConnectionState.RECONNECTING: {
        ConnectionState.CONNECTING, ConnectionState.CONNECTED,
        ConnectionState.COOLING_DOWN, ConnectionState.ROTATING_CREDENTIAL,
        ConnectionState.SWITCHING_PROVIDER, ConnectionState.LOCAL_FALLBACK,
        ConnectionState.DEGRADED_OFFLINE,
    },
    ConnectionState.COOLING_DOWN: {
        ConnectionState.CONNECTING, ConnectionState.RECONNECTING,
        ConnectionState.ROTATING_CREDENTIAL, ConnectionState.SWITCHING_PROVIDER,
        ConnectionState.LOCAL_FALLBACK, ConnectionState.DEGRADED_OFFLINE,
    },
    ConnectionState.ROTATING_CREDENTIAL: {
        ConnectionState.CONNECTING, ConnectionState.RECONNECTING,
        ConnectionState.SWITCHING_PROVIDER, ConnectionState.LOCAL_FALLBACK,
        ConnectionState.DEGRADED_OFFLINE, ConnectionState.COOLING_DOWN,
    },
    ConnectionState.SWITCHING_PROVIDER: {
        ConnectionState.CONNECTING, ConnectionState.RECONNECTING,
        ConnectionState.LOCAL_FALLBACK, ConnectionState.DEGRADED_OFFLINE,
        ConnectionState.COOLING_DOWN,
    },
    ConnectionState.LOCAL_FALLBACK: {
        ConnectionState.CONNECTING, ConnectionState.RECONNECTING,
        ConnectionState.CONNECTED, ConnectionState.DEGRADED_OFFLINE,
        ConnectionState.SWITCHING_PROVIDER,
    },
    ConnectionState.DEGRADED_OFFLINE: {
        ConnectionState.CONNECTING, ConnectionState.RECONNECTING,
        ConnectionState.LOCAL_FALLBACK, ConnectionState.CONNECTED,
    },
    ConnectionState.SHUTTING_DOWN: set(),
}

_ACTIVE_STATES = {ConnectionState.CONNECTED, ConnectionState.LOCAL_FALLBACK}

# Reason classification patterns (checked against a server-supplied reason).
_AUTH_RE = re.compile(r"(?i)auth|unauthenti|unauthori[sz]|api[ _-]?key|token|forbidden|401|403|permission")
_RATE_RE = re.compile(r"(?i)rate.?limit|quota|resource exhausted|429|too many|overloaded")
_MAINT_RE = re.compile(r"(?i)maintenance|deprecat|going away|shutting down|restart|drain|migrat")
_TERM_RE = re.compile(r"(?i)terminated|banned|revoked|closed permanently|not supported|invalid model|404")


@dataclass
class StateTransition:
    at: float
    from_state: ConnectionState
    to_state: ConnectionState
    reason: str = ""


@dataclass
class ConnectionStateMachine:
    """Explicit failover state with validated transitions and bounded history."""

    state: ConnectionState = ConnectionState.DISCONNECTED
    provider: str = ""
    on_change: Callable[[StateTransition], None] | None = None
    clock: Callable[[], float] = time.monotonic
    history: list[StateTransition] = field(default_factory=list)
    _max_history: int = 64
    # Bounded reconnect-loop detection.
    reconnect_attempts: int = 0
    _reconnect_window_start: float = 0.0

    def can_transition(self, to: ConnectionState) -> bool:
        if to is ConnectionState.SHUTTING_DOWN:
            return True
        if self.state is ConnectionState.SHUTTING_DOWN:
            return False
        return to in _TRANSITIONS.get(self.state, set()) or to is self.state

    def transition(self, to: ConnectionState, reason: str = "") -> StateTransition:
        if not self.can_transition(to):
            raise ValueError(
                f"illegal connection transition {self.state.value} → {to.value}")
        record = StateTransition(self.clock(), self.state, to, reason)
        self.state = to
        self.history.append(record)
        if len(self.history) > self._max_history:
            del self.history[: len(self.history) - self._max_history]
        if to in (ConnectionState.RECONNECTING, ConnectionState.CONNECTING):
            self._note_reconnect()
        elif to is ConnectionState.CONNECTED:
            self.reconnect_attempts = 0
        if self.on_change is not None:
            try:
                self.on_change(record)
            except Exception:
                pass
        return record

    def set(self, to: ConnectionState, reason: str = "") -> StateTransition:
        """Tolerant driver entry point: transition when legal, otherwise record a
        forced transition rather than raising — a real-world channel may move in
        ways the strict graph does not enumerate, and observability must never
        crash the worker.  ``transition`` stays strict for callers that want it."""
        if self.state is ConnectionState.SHUTTING_DOWN and to is not ConnectionState.SHUTTING_DOWN:
            return self.history[-1] if self.history else StateTransition(
                self.clock(), self.state, self.state, reason)
        try:
            return self.transition(to, reason)
        except ValueError:
            record = StateTransition(self.clock(), self.state, to, reason + " (forced)")
            self.state = to
            self.history.append(record)
            if len(self.history) > self._max_history:
                del self.history[: len(self.history) - self._max_history]
            if to in (ConnectionState.RECONNECTING, ConnectionState.CONNECTING):
                self._note_reconnect()
            elif to is ConnectionState.CONNECTED:
                self.reconnect_attempts = 0
            if self.on_change is not None:
                try:
                    self.on_change(record)
                except Exception:
                    pass
            return record

    def _note_reconnect(self) -> None:
        now = self.clock()
        if now - self._reconnect_window_start > 120.0:
            self._reconnect_window_start = now
            self.reconnect_attempts = 1
        else:
            self.reconnect_attempts += 1

    def in_reconnect_loop(self, threshold: int = 6) -> bool:
        """True when reconnects are churning — the caller should escalate to a
        provider switch / local fallback rather than retry the same channel."""
        return self.reconnect_attempts >= threshold

    @property
    def is_active(self) -> bool:
        return self.state in _ACTIVE_STATES

    @property
    def is_shutting_down(self) -> bool:
        return self.state is ConnectionState.SHUTTING_DOWN

    @staticmethod
    def classify_go_away(reason: str | None) -> GoAwayClass:
        """Classify a ``go_away``/disconnect reason.  Order matters: auth and
        rate-limit are the actionable ones; maintenance and terminal are less
        common.  An empty/unknown reason is treated as recoverable so failover
        prefers a plain reconnect over giving up."""
        text = str(reason or "").strip()
        if not text:
            return GoAwayClass.RECOVERABLE
        if _AUTH_RE.search(text):
            return GoAwayClass.AUTH
        if _RATE_RE.search(text):
            return GoAwayClass.RATE_LIMIT
        if _TERM_RE.search(text):
            return GoAwayClass.TERMINAL
        if _MAINT_RE.search(text):
            return GoAwayClass.MAINTENANCE
        return GoAwayClass.RECOVERABLE

    # Recommended next state for a classified go_away (advisory — the worker
    # still decides based on what providers/credentials are configured).
    @staticmethod
    def next_state_for(go_away: GoAwayClass) -> ConnectionState:
        return {
            GoAwayClass.RECOVERABLE: ConnectionState.RECONNECTING,
            GoAwayClass.MAINTENANCE: ConnectionState.RECONNECTING,
            GoAwayClass.RATE_LIMIT: ConnectionState.COOLING_DOWN,
            GoAwayClass.AUTH: ConnectionState.ROTATING_CREDENTIAL,
            GoAwayClass.TERMINAL: ConnectionState.SWITCHING_PROVIDER,
        }[go_away]

    _USER_MESSAGE = {
        ConnectionState.CONNECTING: "Connecting to the live channel…",
        ConnectionState.CONNECTED: "Live channel connected.",
        ConnectionState.RECONNECTING: "Connection interrupted — reconnecting…",
        ConnectionState.COOLING_DOWN: "Provider is rate-limited — cooling down before retry…",
        ConnectionState.ROTATING_CREDENTIAL: "Selecting an alternate credential…",
        ConnectionState.SWITCHING_PROVIDER: "Switching to an alternate provider…",
        ConnectionState.LOCAL_FALLBACK: "Using the local model while the live channel recovers.",
        ConnectionState.DEGRADED_OFFLINE: (
            "No model provider is currently available — I can still run "
            "deterministic tools. Configure another API/provider or a local LLM."),
        ConnectionState.SHUTTING_DOWN: "Shutting down…",
        ConnectionState.DISCONNECTED: "Disconnected.",
    }

    def user_message(self) -> str:
        return self._USER_MESSAGE.get(self.state, self.state.value)

    def describe(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "provider": self.provider,
            "active": self.is_active,
            "reconnect_attempts": self.reconnect_attempts,
            "message": self.user_message(),
            "recent": [
                {"from": t.from_state.value, "to": t.to_state.value,
                 "reason": t.reason} for t in self.history[-8:]
            ],
        }
