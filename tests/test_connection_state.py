"""
Tests for the connection state machine (Section 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.connection_state import (
    ConnectionState,
    ConnectionStateMachine,
    GoAwayClass,
)


def test_happy_path_connect():
    sm = ConnectionStateMachine()
    sm.transition(ConnectionState.CONNECTING)
    sm.transition(ConnectionState.CONNECTED, "synchronised")
    assert sm.state is ConnectionState.CONNECTED
    assert sm.is_active


def test_illegal_transition_rejected():
    sm = ConnectionStateMachine(state=ConnectionState.DISCONNECTED)
    with pytest.raises(ValueError):
        sm.transition(ConnectionState.CONNECTED)  # must go via CONNECTING


def test_shutdown_reachable_from_anywhere_and_terminal():
    sm = ConnectionStateMachine(state=ConnectionState.CONNECTED)
    sm.transition(ConnectionState.SHUTTING_DOWN)
    assert sm.is_shutting_down
    with pytest.raises(ValueError):
        sm.transition(ConnectionState.CONNECTING)  # nothing leaves shutdown


@pytest.mark.parametrize("reason,expected", [
    ("api key expired", GoAwayClass.AUTH),
    ("unauthenticated request", GoAwayClass.AUTH),
    ("rate limit exceeded (429)", GoAwayClass.RATE_LIMIT),
    ("resource exhausted", GoAwayClass.RATE_LIMIT),
    ("server maintenance, please reconnect", GoAwayClass.MAINTENANCE),
    ("session terminated", GoAwayClass.TERMINAL),
    ("", GoAwayClass.RECOVERABLE),
    (None, GoAwayClass.RECOVERABLE),
    ("transient network blip", GoAwayClass.RECOVERABLE),
])
def test_go_away_classification(reason, expected):
    assert ConnectionStateMachine.classify_go_away(reason) is expected


def test_next_state_for_each_class():
    m = ConnectionStateMachine
    assert m.next_state_for(GoAwayClass.AUTH) is ConnectionState.ROTATING_CREDENTIAL
    assert m.next_state_for(GoAwayClass.RATE_LIMIT) is ConnectionState.COOLING_DOWN
    assert m.next_state_for(GoAwayClass.TERMINAL) is ConnectionState.SWITCHING_PROVIDER
    assert m.next_state_for(GoAwayClass.RECOVERABLE) is ConnectionState.RECONNECTING


def test_reconnect_loop_detection():
    ticks = {"t": 0.0}
    sm = ConnectionStateMachine(clock=lambda: ticks["t"])
    sm.transition(ConnectionState.CONNECTING)
    for _ in range(6):
        sm.transition(ConnectionState.RECONNECTING)
        sm.transition(ConnectionState.CONNECTING)
        ticks["t"] += 1.0
    assert sm.in_reconnect_loop()
    # A successful connect resets the counter.
    sm.transition(ConnectionState.CONNECTED)
    assert not sm.in_reconnect_loop()


def test_transition_callback_and_history():
    seen = []
    sm = ConnectionStateMachine(on_change=lambda t: seen.append((t.from_state, t.to_state)))
    sm.transition(ConnectionState.CONNECTING)
    sm.transition(ConnectionState.CONNECTED)
    assert seen == [
        (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING),
        (ConnectionState.CONNECTING, ConnectionState.CONNECTED),
    ]
    d = sm.describe()
    assert d["state"] == "connected"
    assert d["message"]


def test_failover_chain_auth_to_rotate_to_switch_to_local():
    sm = ConnectionStateMachine(state=ConnectionState.CONNECTED)
    # go_away: auth → rotate credential → still failing → switch provider →
    # cloud exhausted → local fallback.
    sm.transition(ConnectionState.ROTATING_CREDENTIAL, "auth")
    sm.transition(ConnectionState.SWITCHING_PROVIDER, "no more keys")
    sm.transition(ConnectionState.LOCAL_FALLBACK, "cloud exhausted")
    assert sm.state is ConnectionState.LOCAL_FALLBACK
    assert sm.is_active   # local fallback still serves the user
