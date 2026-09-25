"""
Manual audio-device selection tests (new ask).

The user picks ORION's microphone/speaker directly when he locks onto the wrong
one.  These cover the specific-pick backend methods with the device layer and
stream I/O stubbed — no PortAudio, no hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.audio_devices as ad
from orion_core.audio import MicrophoneEngine, SpeechQueueManager
from orion_core.bus import OrionBus


class _Sig:
    def __init__(self):
        self.calls = []

    def emit(self, *payload):
        self.calls.append(payload)


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


class _FakePlayback:
    def __init__(self):
        self.requested = []

    def request_output_device(self, index):
        self.requested.append(index)


_DEVICES = [
    {"name": "Default Mic", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "USB Headset", "max_input_channels": 1, "max_output_channels": 2},
    {"name": "Studio Monitors", "max_input_channels": 0, "max_output_channels": 2},
]


@pytest.fixture
def stub_devices(monkeypatch):
    monkeypatch.setattr(ad, "_devices", lambda: list(_DEVICES))
    persisted = {}
    monkeypatch.setattr(ad, "set_device",
                        lambda kind, spec: persisted.__setitem__(kind, spec) or "ok")
    monkeypatch.setattr(ad, "resolve", lambda kind: None)
    return persisted


# ── output selection (SpeechQueueManager) ─────────────────────────────────────

def _speech():
    sqm = SpeechQueueManager.__new__(SpeechQueueManager)
    sqm.bus = _Bus()
    sqm.playback = _FakePlayback()
    return sqm


def test_set_output_device_picks_specific_speaker(stub_devices):
    sqm = _speech()
    name = sqm.set_output_device("Studio")          # name fragment
    assert name == "Studio Monitors"
    assert sqm.playback.requested == [2]            # moved the live renderer
    assert stub_devices["output"] == "2"            # persisted


def test_set_output_device_by_index(stub_devices):
    sqm = _speech()
    assert sqm.set_output_device("1") == "USB Headset"
    assert sqm.playback.requested == [1]


def test_set_output_device_default_resets(stub_devices):
    sqm = _speech()
    assert sqm.set_output_device("default") == "system default"
    assert sqm.playback.requested == [None]         # follow the default now
    assert stub_devices["output"] == "default"


def test_set_output_device_unknown_returns_none(stub_devices):
    sqm = _speech()
    assert sqm.set_output_device("no such speaker") is None
    assert sqm.playback.requested == []             # nothing changed


# ── input selection (MicrophoneEngine) ────────────────────────────────────────

def _mic():
    mic = MicrophoneEngine.__new__(MicrophoneEngine)
    mic.bus = _Bus()
    mic._forced_input = None
    mic._rotation_attempts = 3
    mic._gatekeeper = None
    mic._restarted = []
    mic._restart_stream = lambda idx: mic._restarted.append(idx)   # type: ignore
    return mic


def test_switch_to_input_picks_specific_mic(stub_devices):
    mic = _mic()
    name = mic.switch_to_input("Headset")
    assert name == "USB Headset"
    assert mic._forced_input == 1
    assert mic._restarted == [1]
    assert stub_devices["input"] == "1"
    assert mic._rotation_attempts == 0              # cycle guard reset


def test_switch_to_input_default_resets(stub_devices):
    mic = _mic()
    mic._forced_input = 1
    assert mic.switch_to_input("default") == "system default"
    assert mic._forced_input is None
    assert stub_devices["input"] == "default"


def test_switch_to_input_unknown_returns_none(stub_devices):
    mic = _mic()
    assert mic.switch_to_input("ghost mic") is None
    assert mic._restarted == []                     # stream untouched


# ── bus contract ──────────────────────────────────────────────────────────────

def test_bus_exposes_audio_device_request():
    bus = OrionBus()
    seen = []
    bus.audio_device_request.connect(lambda kind, spec: seen.append((kind, spec)))
    bus.audio_device_request.emit("output", "2")
    assert seen == [("output", "2")]
