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
    """The number picks the device NOW; what is saved is that device's name,
    because the index moves when anything is plugged in (10 Oct: "change to
    fifine" was saved as "4")."""
    _isolate(tmp_path, monkeypatch)
    ad.set_device("output", "3")
    cfg = json.loads((tmp_path / "audio.json").read_text(encoding="utf-8"))
    assert cfg["output"] == "Headphones"
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


def _webcam_guidance(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(ad, "_devices", lambda: [
        {"name": "Microphone (HD Pro Webcam C920)", "max_input_channels": 1},
        {"name": "Microphone (HyperX Cloud III)", "max_input_channels": 1},
        {"name": "Microphone (fifine USB)", "max_input_channels": 1}])
    monkeypatch.setattr(ad, "_suggestion_pending_until", 0.0, raising=False)
    return tmp_path / "audio.json"


def test_webcam_suggestion_is_once_only_and_never_changes_device(tmp_path, monkeypatch):
    config = _webcam_guidance(tmp_path, monkeypatch)
    suggestion = ad.microphone_suggestion(0)
    assert "webcam" in suggestion.lower() and "HyperX" in suggestion
    assert ad.resolve("input") is None
    assert ad._live == {}
    assert "input" not in json.loads(config.read_text())
    assert ad.microphone_suggestion(0) == ""


def test_no_to_a_pending_microphone_suggestion_is_remembered(tmp_path, monkeypatch):
    config = _webcam_guidance(tmp_path, monkeypatch)
    assert not ad.decline_microphone_suggestion("no")
    assert ad.microphone_suggestion(0)
    assert not ad.decline_microphone_suggestion("no, remind me to call mum")
    assert ad.decline_microphone_suggestion("No thanks.")
    assert json.loads(config.read_text())["microphone_guidance"] == "declined"
    assert ad.microphone_suggestion(0) == ""
    assert ad.resolve("input") is None


def test_guidance_does_not_claim_headset_quality_or_offer_virtual_devices(tmp_path, monkeypatch):
    _webcam_guidance(tmp_path, monkeypatch)
    assert ad.microphone_suggestion(1) == ""  # Already using the close microphone.
    monkeypatch.setattr(ad, "_devices", lambda: [
        {"name": "Webcam microphone", "max_input_channels": 1},
        {"name": "Virtual USB cable", "max_input_channels": 1}])
    assert ad.microphone_suggestion(0) == ""


def test_reconnecting_microphones_cannot_offer_the_suggestion_twice(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    _webcam_guidance(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(ad.microphone_suggestion, [0, 0]))
    assert sum(bool(reply) for reply in replies) == 1


# ── closing the microphone off the event loop (owner's log, 7 Oct 2026) ──────

def test_a_reconnect_closes_the_microphone_without_blocking(monkeypatch):
    """`PERF: event loop blocked 250 ms by close()` on every live-channel drop:
    the stream is now closed on a short-lived thread; shutdown still waits."""
    import time
    from types import SimpleNamespace

    from orion_core import audio

    closed = []

    class _SlowStream:
        def stop(self):
            pass

        def close(self):
            time.sleep(0.3)                  # what PortAudio's close() costs
            closed.append(True)

    mic = audio.MicrophoneEngine.__new__(audio.MicrophoneEngine)
    mic._stream, mic._gatekeeper = _SlowStream(), None
    mic.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None))
    started = time.perf_counter()
    mic.stop(wait=False)
    assert time.perf_counter() - started < 0.1, "the caller waited for the close"
    assert mic._stream is None and closed == []
    audio.finish_pending_closes()              # what a shutdown stop() does first
    assert closed == [True]


def test_a_microphone_switch_says_it_is_happening_now(tmp_path, monkeypatch):
    """10 Oct: after "change to fifine" ORION said it would apply "after I
    restart" (set_device's line) although the live swap was also requested,
    and the owner believed he was still on the webcam microphone."""
    import types
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    _isolate(tmp_path, monkeypatch)
    sent = []
    me = types.SimpleNamespace(bus=types.SimpleNamespace(
        audio_device_request=types.SimpleNamespace(emit=lambda *a: sent.append(a))))
    result = DesktopDispatchMixin.audio_devices_tool(me, {"action": "set_input", "device": "Headset Mic"})
    text = str(result.response_payload())
    assert "Switching to it now" in text and "next time ORION starts" not in text
    assert sent == [("input", "Headset Mic")]


def test_the_live_swap_is_queued_off_the_voice_task():
    import inspect
    from orion_core.live_worker import GenAILiveWorker
    source = inspect.getsource(GenAILiveWorker)
    assert "self._on_audio_device_request, Qt.ConnectionType.QueuedConnection" in source


# Windows endpoint changes must not reopen PortAudio's cached old default.
def _default_endpoint_fixture(monkeypatch, devices, name):
    from types import SimpleNamespace
    monkeypatch.setattr(ad, "_system_default_name", lambda _kind: name)
    monkeypatch.setattr(ad, "_devices", lambda: devices)
    monkeypatch.setattr(ad, "_chosen_api", {})
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}]))


def test_system_default_tracks_current_endpoint_without_portaudio_refresh(monkeypatch):
    devices = [
        {"name": "Headset output", "max_output_channels": 2, "hostapi": 0},
        {"name": "Monitor output", "max_output_channels": 2, "hostapi": 0},
    ]
    _default_endpoint_fixture(monkeypatch, devices, "Headset output")
    assert ad.system_default_index("output") == 0
    monkeypatch.setattr(ad, "_system_default_name", lambda _kind: "Monitor output")
    assert ad.system_default_index("output") == 1


def test_system_default_uses_measured_api_then_known_resampling_order(monkeypatch):
    name = "Speakers (USB output)"
    devices = [
        {"name": name, "max_output_channels": 2, "hostapi": 1},
        {"name": name, "max_output_channels": 2, "hostapi": 0},
    ]
    _default_endpoint_fixture(monkeypatch, devices, name)
    assert ad.system_default_index("output") == 1  # Reliable MME resampling.
    ad._chosen_api["output"] = "wasapi"
    assert ad.system_default_index("output") == 0  # A measured result wins.


def test_system_default_accepts_unambiguous_mme_truncated_name(monkeypatch):
    name = "XV252Q (NVIDIA High Definition Audio)"
    devices = [
        {"name": name[:31], "max_output_channels": 2, "hostapi": 0},
        {"name": name, "max_output_channels": 2, "hostapi": 1},
    ]
    _default_endpoint_fixture(monkeypatch, devices, name)
    assert ad.system_default_index("output") == 0


def test_system_default_does_not_guess_an_ambiguous_truncated_endpoint(monkeypatch):
    shared = "Speakers (Identical USB Audio Device "
    first, second = shared + "One)", shared + "Two)"
    devices = [
        {"name": first[:31], "max_output_channels": 2, "hostapi": 0},
        {"name": second, "max_output_channels": 2, "hostapi": 1},
    ]
    _default_endpoint_fixture(monkeypatch, devices, first)
    assert ad.system_default_index("output") is None
    devices.append({"name": first, "max_output_channels": 2, "hostapi": 1})
    assert ad.system_default_index("output") == 2  # Full exact identity survives.


def test_system_default_rejects_duplicate_names_in_the_best_api(monkeypatch):
    record = {"name": "USB speakers", "max_output_channels": 2, "hostapi": 0}
    _default_endpoint_fixture(monkeypatch, [dict(record), dict(record)], record["name"])
    assert ad.system_default_index("output") is None


def test_system_default_does_not_use_fuzzy_or_wrong_direction_matches(monkeypatch):
    devices = [
        {"name": "Headset speakers", "max_output_channels": 2, "hostapi": 0},
        {"name": "New Headset speakers", "max_input_channels": 1, "hostapi": 0},
    ]
    _default_endpoint_fixture(monkeypatch, devices, "New Headset speakers")
    assert ad.system_default_index("output") is None
    assert ad.system_default_index("input") == 1
    assert ad.system_default_index("unknown") is None
    monkeypatch.setattr(ad, "_system_default_name", lambda _kind: "headphones")
    assert ad.system_default_index("output") is None


def test_system_default_missing_endpoint_or_portaudio_is_safe(monkeypatch):
    _default_endpoint_fixture(monkeypatch, [], None)
    assert ad.system_default_index("output") is None
    monkeypatch.setattr(ad, "_system_default_name", lambda _kind: "USB output")
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert ad.system_default_index("output") is None


def _fake_endpoint_com(monkeypatch, utilities, *, fail_initialize=False):
    from types import SimpleNamespace
    events = []

    def initialize():
        events.append("initialize")
        if fail_initialize:
            raise OSError("COM apartment initialization failed")

    monkeypatch.setattr(ad, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "comtypes", SimpleNamespace(
        CoInitialize=initialize, CoUninitialize=lambda: events.append("uninitialize")))
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", SimpleNamespace(AudioUtilities=utilities))
    return events


def test_current_windows_output_name_balances_its_com_initialization(monkeypatch):
    from types import SimpleNamespace
    utilities = SimpleNamespace(GetSpeakers=lambda: SimpleNamespace(FriendlyName="Monitor output"))
    events = _fake_endpoint_com(monkeypatch, utilities)
    assert ad._system_default_name("output") == "Monitor output"
    assert events == ["initialize", "uninitialize"]


def test_current_windows_input_wraps_the_raw_microphone_endpoint(monkeypatch):
    from types import SimpleNamespace
    raw = object()
    wrapped = []

    def wrap(endpoint):
        wrapped.append(endpoint)
        return SimpleNamespace(FriendlyName="Desk microphone")

    utilities = SimpleNamespace(GetMicrophone=lambda: raw, CreateDevice=wrap)
    events = _fake_endpoint_com(monkeypatch, utilities)
    assert ad._system_default_name("input") == "Desk microphone"
    assert wrapped == [raw]
    assert events == ["initialize", "uninitialize"]


def test_endpoint_lookup_failure_still_balances_com_and_failed_init_does_not(monkeypatch):
    from types import SimpleNamespace

    def missing():
        raise OSError("endpoint disappeared")

    utilities = SimpleNamespace(GetSpeakers=missing)
    events = _fake_endpoint_com(monkeypatch, utilities)
    assert ad._system_default_name("output") is None
    assert events == ["initialize", "uninitialize"]
    events = _fake_endpoint_com(monkeypatch, utilities, fail_initialize=True)
    assert ad._system_default_name("output") is None
    assert events == ["initialize"]


def test_windows_endpoint_lookup_does_nothing_on_other_platforms(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(ad, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setitem(sys.modules, "comtypes", None)
    assert ad._system_default_name("output") is None
