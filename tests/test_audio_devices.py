"""
Tests for the audio device live-swap fix: audio_devices.resolve_effective()/
next_index()/note_live_device() (previously missing, silently breaking the
Command Centre's device picker and the "I can't hear you" round-robin), and
the audio_devices dispatcher tool now live-swapping instead of only
persisting config for next start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import audio_devices as ad
from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher


_FAKE_DEVICES = [
    {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "USB Headset Mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Headphones", "max_input_channels": 0, "max_output_channels": 2},
]


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "AUDIO_CONFIG_PATH", tmp_path / "audio.json")
    monkeypatch.setattr(ad, "_devices", lambda: _FAKE_DEVICES)
    ad._live.clear()
    monkeypatch.delenv("ORION_AUDIO_INPUT", raising=False)
    monkeypatch.delenv("ORION_AUDIO_OUTPUT", raising=False)


# ── resolve_effective / note_live_device ────────────────────────────────────

def test_resolve_effective_falls_back_to_resolve_when_no_live_override(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ad.resolve_effective("input") is None   # system default, nothing saved
    ad.set_device("input", "1")
    assert ad.resolve_effective("input") == 1       # falls back to saved config


def test_resolve_effective_prefers_live_override_over_saved_config(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ad.set_device("output", "2")          # persisted choice: index 2
    ad.note_live_device("output", 3)      # but the live stream is actually on index 3
    assert ad.resolve_effective("output") == 3


def test_set_device_persists_the_name_spec_not_the_index(tmp_path, monkeypatch):
    """Windows renumbers a device's port label as it's re-enumerated at boot
    (observed on the real machine: the same physical Fifine mic moved from
    index [16] "2- fifine Microphone" to the same index later relabelled
    "3- fifine Microphone") — persisting a raw index once can silently point
    at a DIFFERENT device after a renumbering. A name fragment re-matches
    fresh every resolve() call instead."""
    _isolate(tmp_path, monkeypatch)
    ad.set_device("input", "Headset Mic")   # a name fragment, not a digit
    cfg = json.loads((tmp_path / "audio.json").read_text(encoding="utf-8"))
    assert cfg["input"] == "Headset Mic"    # stored as the spec, not resolved to "1"


def test_resolve_follows_a_renumbered_device_by_name(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ad.set_device("input", "Headset Mic")
    assert ad.resolve("input") == 1     # "USB Headset Mic" at its current index

    # Simulate Windows re-enumerating: a new device is inserted first, so
    # every subsequent index shifts up by one.
    renumbered = [{"name": "New Webcam Mic", "max_input_channels": 1, "max_output_channels": 0}] + _FAKE_DEVICES
    monkeypatch.setattr(ad, "_devices", lambda: renumbered)
    assert ad.resolve("input") == 2     # re-matched by name at its NEW index


def test_explicit_numeric_spec_is_still_honoured_literally(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ad.set_device("output", "3")
    cfg = json.loads((tmp_path / "audio.json").read_text(encoding="utf-8"))
    assert cfg["output"] == "3"
    assert ad.resolve("output") == 3


def test_note_live_device_none_means_system_default_live(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ad.set_device("output", "2")
    ad.note_live_device("output", None)   # e.g. a failed swap reverted to default
    assert ad.resolve_effective("output") is None


# ── next_index (round-robin) ─────────────────────────────────────────────────

def test_next_index_rotates_through_devices_of_the_right_kind(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    result = ad.next_index("output", 2)
    assert result == (3, "Headphones")
    result = ad.next_index("output", 3)
    assert result == (2, "Speakers")      # wraps around


def test_next_index_none_when_only_one_device_of_that_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_devices", lambda: [_FAKE_DEVICES[2]])  # one output only
    ad._live.clear()
    assert ad.next_index("output", 2) is None


def test_next_index_starts_from_first_when_current_unknown(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert ad.next_index("input", 99) == (0, "Built-in Microphone")


# ── dispatcher tool: live-swap on set_input / set_output ───────────────────

class _Signal:
    def __init__(self) -> None:
        self.emitted: list[tuple] = []

    def emit(self, *args) -> None:
        self.emitted.append(args)

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __init__(self) -> None:
        self.audio_device_request = _Signal()

    def __getattr__(self, name):
        return _Signal()


def _dispatcher() -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    return d


def test_audio_devices_tool_set_output_live_swaps(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.audio_devices_tool({"action": "set_output", "device": "Headphones"})
    assert result.ok
    assert d.bus.audio_device_request.emitted == [("output", "Headphones")]


def test_audio_devices_tool_set_input_live_swaps(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.audio_devices_tool({"action": "set_input", "device": "USB Headset Mic"})
    assert result.ok
    assert d.bus.audio_device_request.emitted == [("input", "USB Headset Mic")]


def test_audio_devices_tool_no_match_does_not_live_swap(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.audio_devices_tool({"action": "set_output", "device": "Nonexistent Device"})
    assert not result.ok
    assert d.bus.audio_device_request.emitted == []
