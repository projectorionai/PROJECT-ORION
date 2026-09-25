"""
Construction regression test for GenAILiveWorker (guards the "stuck on
INITIALISING" bug).

A structurally-valid but logically-broken __init__ (a method accidentally
inserted mid-body, absorbing the rest of the initialisation) compiles and
imports cleanly, so unit tests that bypass __init__ never catch it — yet the
real app dies on the first missing attribute inside run(), never leaving the
INITIALISING state.

This test constructs a real worker with the heavy audio/recogniser dependencies
mocked and asserts every attribute run() relies on is actually set during
construction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.live_worker as lw
from orion_core.live_worker import GenAILiveWorker


def _construct(monkeypatch):
    # Mock the heavy, hardware-touching constructors used inside __init__.
    for name in ("AudioStateMachine", "AudioPlaybackThread", "SpeechSynthesiser",
                 "SpeechQueueManager", "SileroVADGatekeeper", "LocalSpeechRecogniser",
                 "VoiceInterruptManager"):
        monkeypatch.setattr(lw, name, MagicMock(), raising=True)

    bus = MagicMock()
    memory = MagicMock()
    dispatcher = SimpleNamespace()          # __init__ sets on_briefing_request on it
    router = MagicMock()
    settings = MagicMock()
    return GenAILiveWorker(settings, bus, memory, dispatcher, router)


def test_init_sets_every_attribute_run_depends_on(monkeypatch):
    worker = _construct(monkeypatch)
    # These were the attributes absorbed by the mid-__init__ method bug; run()
    # touches all of them, so each must exist after construction.
    for attr in ("microphone_enabled", "connected", "tool_busy", "vad", "recogniser",
                 "interrupts", "wake_mode_enabled", "_loop", "_turn_watchdog_task",
                 "conn_state", "_send_lock", "_send_closed", "_session_loop_active",
                 "_pending_turn", "_turn_active", "active_live_provider",
                 "_no_live_notice_sent"):
        assert hasattr(worker, attr), f"__init__ did not set self.{attr}"


def test_init_default_values_are_sane(monkeypatch):
    worker = _construct(monkeypatch)
    assert worker.microphone_enabled is True
    assert worker.connected is False
    assert worker._turn_watchdog_task is None       # run() checks this is None
    assert worker._send_closed is True              # no transport until connected
    assert worker._session_loop_active is False
    assert worker.conn_state is not None
    # The dispatcher hook and proactive-speech wiring must have run.
    assert worker.dispatcher.on_briefing_request == worker.deliver_briefing_on_demand
    worker.bus.speak_request.connect.assert_called_once_with(worker.announce)


def test_on_conn_state_is_a_lean_method_not_the_initialiser(monkeypatch):
    worker = _construct(monkeypatch)
    # Calling it must ONLY publish on the bus — it must NOT (re)initialise state.
    worker.conn_state.set  # exists
    worker._on_conn_state(None)
    worker.bus.connection_state.emit.assert_called()
