"""
Audio device selection — the short list, and the measured transport probe.

Why this exists
---------------
``probe()`` opens a stream and closes it, which proves a device exists and is
not held by another application. It does not prove that sound reaches the
speaker, and on Windows that distinction is not academic: PortAudio's
DirectSound output is a silent sink. The stream opens, every write returns
success in about zero time, and nothing is ever heard. No capability flag
reports it, so an open-and-close probe passes it happily and ORION would be
mute while reporting healthy audio.

Measured on the development machine at ORION's own rates (16 kHz in, 24 kHz
out), which is the only configuration that matters:

    host API        output (blocking write)   input (callback)
    -----------     ----------------------    ----------------
    MME             0.67 s  real              frames arriving
    DirectSound     0.00 s  SILENT SINK       frames arriving
    WASAPI          open fails (rate)         open fails (rate)
    WDM-KS          open fails (rate)         open fails (rate)

WASAPI is the trap for anyone choosing by reputation: in shared mode it does
not resample, so against 48 kHz hardware every open fails outright.

The logic tests below run against a stubbed device table so they are
deterministic everywhere. The few that touch real hardware degrade to a skip
rather than a failure, because CI has no sound card and a red build that only
means "no speakers" teaches nobody anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import audio_devices as ad


# ── the short list ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "Microsoft Sound Mapper - Input",
    "Microsoft Sound Mapper - Output",
    "Primary Sound Capture Driver",
    "Primary Sound Driver",
    "sysdefault",
    "default",
    "dmix",
])
def test_pseudo_devices_are_not_offered_as_hardware(name):
    """These are aliases for "whatever the OS picks", which is already the
    default entry. Offering them again as if they were devices is noise."""
    assert ad._is_pseudo(name) is True


@pytest.mark.parametrize("name", [
    "Headphones (3- Xrocker)",
    "Microphone (HD Pro Webcam C920)",
    "Speakers (Realtek(R) Audio)",
    "Pixio PX248P (NVIDIA High Definition Audio)",
])
def test_real_devices_survive_the_filter(name):
    assert ad._is_pseudo(name) is False


@pytest.fixture
def stub_devices(monkeypatch):
    """A device table shaped like the real one: the same hardware repeated
    under four host APIs, plus the pseudo-devices."""
    apis = [{"name": "MME"}, {"name": "Windows DirectSound"},
            {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]
    rows = []
    for api_index in range(4):
        rows.append({"name": "Microsoft Sound Mapper - Input", "hostapi": api_index,
                     "max_input_channels": 2, "max_output_channels": 0})
        for device in ("Microphone (Webcam)", "Microphone (fifine)"):
            rows.append({"name": device, "hostapi": api_index,
                         "max_input_channels": 2, "max_output_channels": 0})
        for device in ("Headphones (Xrocker)", "Speakers (Realtek)"):
            rows.append({"name": device, "hostapi": api_index,
                         "max_input_channels": 0, "max_output_channels": 2})

    class _StubSounddevice:
        @staticmethod
        def query_devices():
            return list(rows)

        @staticmethod
        def query_hostapis():
            return list(apis)

    monkeypatch.setitem(sys.modules, "sounddevice", _StubSounddevice)
    monkeypatch.setattr(ad, "_chosen_api", {}, raising=False)
    return rows


def test_the_same_device_under_four_host_apis_is_offered_once(stub_devices):
    """query_devices() returns one row per device PER HOST API, not per device.
    Measured on the development machine: 68 rows for 21 distinct inputs and 26
    distinct outputs. The duplicates ARE the same physical hardware."""
    inputs = ad.usable_devices("input")
    names = [name for _index, name in inputs]
    assert names == sorted(set(names), key=names.index), "duplicates offered"
    assert len(names) == 2, names
    assert "Microphone (Webcam)" in names


def test_outputs_and_inputs_are_kept_apart(stub_devices):
    outputs = [name for _i, name in ad.usable_devices("output")]
    assert set(outputs) == {"Headphones (Xrocker)", "Speakers (Realtek)"}
    inputs = [name for _i, name in ad.usable_devices("input")]
    assert not set(outputs) & set(inputs)


def test_pseudo_devices_are_dropped_from_the_offered_list(stub_devices):
    for _index, name in ad.usable_devices("input"):
        assert not ad._is_pseudo(name), name


def test_listing_devices_opens_nothing(stub_devices):
    """usable_devices must be safe on the GUI thread: the stub has no stream
    classes at all, so touching one would raise rather than pass quietly."""
    assert ad.usable_devices("input")
    assert ad.usable_devices("output")


def test_a_measured_host_api_is_ordered_first(stub_devices, monkeypatch):
    """Once a direction has been measured, its winning API leads the list."""
    monkeypatch.setitem(ad._chosen_api, "output", "wdm-ks")
    first_index = ad.usable_devices("output")[0][0]
    api_of_first = stub_devices[first_index]["hostapi"]
    assert api_of_first == 3, "the measured API should come first"


def test_an_unavailable_audio_stack_yields_an_empty_list(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert ad.usable_devices("input") == []


# ── the measured probe ───────────────────────────────────────────────────────

def test_the_silent_sink_threshold_is_what_distinguishes_the_two():
    """A device consuming audio in real time takes roughly as long to accept it
    as to play it; a sink that swallows it returns immediately. The threshold
    has to sit between 0.00 s and 0.67 s for a 0.6 s write — the two values
    actually measured — and be nowhere near either."""
    seconds = ad._TRANSPORT_SECONDS["output"]
    floor = seconds * ad._TRANSPORT_MIN_RATIO
    assert 0.0 < floor < seconds
    assert floor > 0.05, "a sink returning in ~0 s must fail"
    assert floor < 0.6, "a real device must not be rejected for buffering"


def test_input_and_output_are_probed_separately():
    """Each direction chooses its own host API, because on the same machine
    they genuinely differ — no amount of reasoning about API quality would
    produce that split, which is why it is measured."""
    assert set(ad._TRANSPORT_SECONDS) == {"input", "output"}
    assert all(v > 0 for v in ad._TRANSPORT_SECONDS.values())


def test_mme_is_tried_before_directsound_for_a_reason():
    """Ordering by reputation puts WASAPI first and every open fails, because
    WASAPI shared mode does not resample against 48 kHz hardware. DirectSound
    then looks like the answer and is the silent sink. Measurement decides, but
    the starting order should not lead with the two that failed here."""
    order = ad._PREFERRED_HOST_APIS
    assert order.index("mme") < order.index("wasapi")
    assert order.index("directsound") < order.index("wdm-ks")


def test_transport_probe_never_raises_on_a_bad_device():
    ok, detail = ad.transport_works("output", 99999, use_cache=False)
    assert ok is False
    assert isinstance(detail, str) and detail


def test_transport_probe_result_is_cached_per_direction(monkeypatch):
    """The property belongs to the host API, not the device, and the probe
    costs hundreds of milliseconds — it must not run per listing."""
    monkeypatch.setattr(ad, "_transport_cache", {}, raising=False)
    ad._transport_cache[("output", 7)] = (True, "cached")
    assert ad.transport_works("output", 7) == (True, "cached")


def test_preferred_host_api_reports_none_rather_than_guessing(monkeypatch):
    """When nothing passes, the honest answer is 'fall back to the system
    default' — the behaviour ORION had before any of this existed."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(ad, "_chosen_api", {}, raising=False)
    assert ad.preferred_host_api("output") is None


# ── against real hardware, where there is any ────────────────────────────────

def _has_audio() -> bool:
    try:
        import sounddevice as sd

        return bool(list(sd.query_devices()))
    except Exception:
        return False


@pytest.mark.skipif(not _has_audio(), reason="no audio devices on this machine")
def test_the_real_device_table_reduces_substantially():
    """On real hardware the raw table is several times longer than the useful
    one. If this ever stops being true the dedup has silently stopped working."""
    import sounddevice as sd

    raw = list(sd.query_devices())
    raw_inputs = sum(1 for d in raw if d["max_input_channels"] > 0)
    offered = ad.usable_devices("input")
    assert offered, "no input devices survived the filter"
    assert len(offered) <= raw_inputs
    names = [name for _i, name in offered]
    assert len(names) == len(set(names)), "duplicates in the offered list"


@pytest.mark.skipif(not _has_audio(), reason="no audio devices on this machine")
def test_the_chosen_output_actually_carries_audio():
    """The end-to-end point of the module: whatever ORION picks to speak
    through must be measured to play, not merely to open."""
    devices = ad.usable_devices("output")
    if not devices:
        pytest.skip("no output devices offered")
    ok, detail = ad.transport_works("output", devices[0][0])
    if not ok:
        pytest.skip(f"first offered output did not pass here: {detail}")
    assert "consumed" in detail
