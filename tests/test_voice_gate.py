"""
The voice gate — acoustic echo rejection, and push-to-talk.

Echo
----
ORION's defence against answering himself was a flat 0.9 s of deafness after
every reply. It cannot fail, and it costs exactly what it looks like: for most
of a second after each answer he cannot hear you at all, so replying the instant
he stops — which is how people actually talk — does not work.

The acoustic guard lifts that window early whenever the microphone is carrying
something that demonstrably is not ORION. It does not test loudness: how loud
the echo is depends on the speaker volume, the room, the microphone's position
and whether headphones are plugged in, so any single threshold is wrong for
almost everyone. It subtracts the recent output from the microphone in the band
domain instead. Echo cancels to nearly nothing; another voice survives, because
its formants sit in bands where ORION's were weak — which holds even when the
two arrive at the same loudness, the case a loudness test gets wrong.

Push-to-talk
------------
ORION had none. The tests below cover what can be asserted without a keyboard:
the chord spelling, the honest reporting of global versus window scope, and the
lifecycle guarantees — above all that stopping never leaves the microphone
latched open.

Offline: numpy only, no audio device, no key presses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import echo_guard as eg
from orion_core.echo_guard import EchoGuard, band_energies, rms_level
from orion_core.push_to_talk import (
    DEFAULT_CHORD,
    PushToTalk,
    chord_label,
    qt_key_sequence,
)

SR_OUT, SR_IN = 24000, 16000


def synth_voice(f0: float, formants: tuple[float, ...], sample_rate: int,
                seconds: float = 0.12, amplitude: float = 0.5,
                seed: int = 3) -> np.ndarray:
    """A harmonic stack shaped by formants — a cheap, honest stand-in for a
    voice. Two speakers differ by pitch AND by where their formants sit, which
    is exactly the distinction the guard relies on."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    wave = np.zeros_like(t)
    for harmonic in range(1, 26):
        gain = sum(np.exp(-((harmonic * f0 - f) / 180.0) ** 2) for f in formants)
        wave += gain * np.sin(2 * np.pi * harmonic * f0 * t + rng.uniform(0, 6.28))
    wave /= (np.abs(wave).max() + 1e-9)
    return (wave * amplitude).astype(np.float32)


ORION_VOICE = (110.0, (600.0, 1100.0, 2500.0))
OTHER_VOICE = (210.0, (800.0, 1900.0, 3100.0))
SIMILAR_VOICE = (125.0, (650.0, 1200.0, 2600.0))


@pytest.fixture
def guard_with_output():
    guard = EchoGuard()
    guard.note_output(synth_voice(*ORION_VOICE, SR_OUT), SR_OUT, level=0.55)
    return guard


# ── band reduction ───────────────────────────────────────────────────────────

def test_band_energies_are_normalised_so_volume_does_not_matter():
    """The microphone runs at 16 kHz and playback at 24 kHz, at different
    volumes. Normalised bands make those comparable at all."""
    loud = band_energies(synth_voice(*ORION_VOICE, SR_IN, amplitude=0.9), SR_IN)
    quiet = band_energies(synth_voice(*ORION_VOICE, SR_IN, amplitude=0.1), SR_IN)
    assert abs(float(loud.sum()) - 1.0) < 1e-4
    assert np.allclose(loud, quiet, atol=0.02), "level changed the band shape"


def test_the_same_voice_at_two_sample_rates_reduces_alike():
    at16 = band_energies(synth_voice(*ORION_VOICE, SR_IN), SR_IN)
    at24 = band_energies(synth_voice(*ORION_VOICE, SR_OUT), SR_OUT)
    assert float(np.linalg.norm(at16 - at24)) < 0.25


def test_band_reduction_survives_rubbish():
    for bad in (None, "not audio", [], np.zeros(4)):
        out = band_energies(bad, SR_IN)
        assert out.shape == (len(eg._BAND_EDGES) - 1,)


def test_rms_accepts_int16_as_well_as_float():
    quiet = synth_voice(*ORION_VOICE, SR_IN, amplitude=0.05)
    loud = synth_voice(*ORION_VOICE, SR_IN, amplitude=0.9)
    assert rms_level(loud) > rms_level(quiet)
    assert rms_level((loud * 20000).astype("<i2")) > 0.05


# ── the decision ─────────────────────────────────────────────────────────────

def test_orions_own_echo_is_recognised_and_dropped(guard_with_output):
    echo = synth_voice(*ORION_VOICE, SR_IN, amplitude=0.22)
    is_echo, residual = guard_with_output.classify(echo, SR_IN)
    assert is_echo is True
    assert residual < 0.1, residual


def test_a_different_voice_survives_the_subtraction(guard_with_output):
    other = synth_voice(*OTHER_VOICE, SR_IN, amplitude=0.30)
    is_echo, residual = guard_with_output.classify(other, SR_IN)
    assert is_echo is False
    assert residual > eg._VOICE_RESIDUAL


def test_even_a_similar_sounding_voice_survives(guard_with_output):
    """The case a loudness test gets wrong: a second speaker arriving at about
    the same level, with a similar pitch. Only the CONTENT separates them."""
    similar = synth_voice(*SIMILAR_VOICE, SR_IN, amplitude=0.28)
    is_echo, residual = guard_with_output.classify(similar, SR_IN)
    assert is_echo is False, f"a real person was dropped (residual {residual:.3f})"


def test_the_gap_between_echo_and_voice_is_wide(guard_with_output):
    """The threshold should not be load-bearing. If these ever converge, the
    guard has become a tuned constant again and will be wrong on other rooms."""
    echo_residual = guard_with_output.classify(
        synth_voice(*ORION_VOICE, SR_IN, amplitude=0.22), SR_IN)[1]
    voice_residual = guard_with_output.classify(
        synth_voice(*OTHER_VOICE, SR_IN, amplitude=0.30), SR_IN)[1]
    assert voice_residual - echo_residual > 0.3


def test_nothing_playing_means_nothing_can_be_echo():
    guard = EchoGuard()
    is_echo, _residual = guard.classify(
        synth_voice(*ORION_VOICE, SR_IN, amplitude=0.4), SR_IN)
    assert is_echo is False


def test_room_noise_makes_no_claim_either_way(guard_with_output):
    quiet = synth_voice(*OTHER_VOICE, SR_IN, amplitude=0.001)
    is_echo, residual = guard_with_output.classify(quiet, SR_IN)
    assert is_echo is False
    assert residual == 0.0


def test_old_output_stops_being_a_candidate():
    """An echo cannot come from something played two seconds ago; keeping it
    would let ancient audio cancel a live voice."""
    guard = EchoGuard(history_seconds=0.5)
    guard.note_output(synth_voice(*ORION_VOICE, SR_OUT), SR_OUT, level=0.5, at=100.0)
    echo = synth_voice(*ORION_VOICE, SR_IN, amplitude=0.22)
    assert guard.classify(echo, SR_IN, at=100.1)[0] is True
    assert guard.classify(echo, SR_IN, at=101.0)[0] is False


def test_the_guard_learns_the_rooms_gain(guard_with_output):
    guard_with_output.classify(synth_voice(*ORION_VOICE, SR_IN, amplitude=0.22), SR_IN)
    assert guard_with_output.calibrated is True
    assert guard_with_output.gain > 0.0


def test_reset_forgets_the_room():
    guard = EchoGuard()
    guard.note_output(synth_voice(*ORION_VOICE, SR_OUT), SR_OUT, level=0.5)
    guard.classify(synth_voice(*ORION_VOICE, SR_IN, amplitude=0.22), SR_IN)
    guard.reset()
    assert guard.calibrated is False and guard.gain == 0.0


def test_classify_never_raises(guard_with_output):
    for bad in (None, "not audio", [], np.zeros((2, 2, 2))):
        is_echo, residual = guard_with_output.classify(bad, SR_IN)
        assert isinstance(is_echo, bool) and isinstance(residual, float)


def test_the_guard_is_shared_between_playback_and_capture():
    """They live in different objects on different threads and neither owns the
    other, so the guard is reached rather than injected."""
    assert eg.shared() is eg.shared()
    assert isinstance(eg.shared(), EchoGuard)


# ── the early release in audio.py ────────────────────────────────────────────

def _listener_stub(deaf_until: float):
    from orion_core.audio import AudioGateThread

    listener = AudioGateThread.__new__(AudioGateThread)
    listener._deaf_until = deaf_until
    return listener


def test_the_timed_window_still_releases_on_its_own():
    listener = _listener_stub(deaf_until=10.0)
    assert listener._past_echo_guard(b"", now=11.0) is True


def test_orions_own_echo_does_not_open_the_gate_early(monkeypatch):
    """The whole point of the window: ORION must not answer himself."""
    guard = EchoGuard()
    guard.note_output(synth_voice(*ORION_VOICE, SR_OUT), SR_OUT, level=0.55)
    monkeypatch.setattr(eg, "_shared", guard, raising=False)

    listener = _listener_stub(deaf_until=10.0)
    echo = (synth_voice(*ORION_VOICE, SR_IN, amplitude=0.22) * 20000).astype("<i2")
    assert listener._past_echo_guard(echo.tobytes(), now=9.0) is False


def test_a_real_voice_inside_the_window_opens_the_gate_early(monkeypatch):
    """The improvement: you can answer the instant ORION stops talking."""
    guard = EchoGuard()
    guard.note_output(synth_voice(*ORION_VOICE, SR_OUT), SR_OUT, level=0.55)
    monkeypatch.setattr(eg, "_shared", guard, raising=False)

    listener = _listener_stub(deaf_until=10.0)
    person = (synth_voice(*OTHER_VOICE, SR_IN, amplitude=0.30) * 20000).astype("<i2")
    assert listener._past_echo_guard(person.tobytes(), now=9.0) is True
    # And having decided, it stays open for the rest of the window.
    assert listener._deaf_until == 0.0


def test_a_broken_block_keeps_waiting_rather_than_opening(monkeypatch):
    """Any doubt resolves to 'stay deaf'. A guard that opened the microphone on
    an error would have ORION answering himself, which is the bug it exists to
    prevent."""
    listener = _listener_stub(deaf_until=10.0)
    assert listener._past_echo_guard(b"\x01", now=9.0) is False
    assert listener._past_echo_guard(b"", now=9.0) is False


def test_the_acoustic_release_can_be_switched_off():
    from orion_core.audio import AudioGateThread

    assert hasattr(AudioGateThread, "_ACOUSTIC_ECHO")
    assert AudioGateThread._ECHO_GUARD_SECONDS > 0, "the timed floor must remain"


# ── push-to-talk ─────────────────────────────────────────────────────────────

def test_the_default_chord_is_layout_independent():
    """Ctrl+Space is the same finger shape on every keyboard layout, which
    matters because ORION is not an English-only assistant."""
    assert DEFAULT_CHORD == ("ctrl", "space")
    assert chord_label(DEFAULT_CHORD) == "Ctrl+Space"
    assert qt_key_sequence(DEFAULT_CHORD) == "Ctrl+Space"


def test_scope_is_reported_honestly():
    """On platforms with no dependency-free global key state the chord is bound
    inside ORION's window, and the app should say so rather than implying a
    global binding that does not exist."""
    ptt = PushToTalk(lambda: None, lambda: None)
    assert ptt.scope in ("global", "window")
    described = ptt.describe()
    if ptt.scope == "window":
        assert "focus" in described
    else:
        assert "any window" in described


def test_stopping_never_leaves_the_microphone_latched_open():
    """If the watch stops while the key happens to be down, the release must
    still fire — otherwise the microphone stays open with nothing watching it."""
    events: list[str] = []
    ptt = PushToTalk(lambda: events.append("press"),
                     lambda: events.append("release"))
    ptt._set_held(True)
    assert events == ["press"]
    ptt.stop()
    assert events == ["press", "release"]
    assert ptt.held is False


def test_hold_and_release_fire_once_each():
    events: list[str] = []
    ptt = PushToTalk(lambda: events.append("press"),
                     lambda: events.append("release"))
    ptt._set_held(True)
    ptt._set_held(True)          # auto-repeat must not re-fire
    ptt._set_held(False)
    ptt._set_held(False)
    assert events == ["press", "release"]


def test_a_failing_consumer_does_not_kill_the_watch():
    """A microphone that silently stopped responding is the worst outcome."""
    def explode() -> None:
        raise RuntimeError("consumer blew up")

    ptt = PushToTalk(explode, explode)
    ptt._set_held(True)          # must not raise
    ptt._set_held(False)
    assert ptt.held is False


def test_the_debounce_is_short_enough_to_feel_instant():
    from orion_core.push_to_talk import _DEBOUNCE_S, _POLL_HZ

    assert 0.0 < _DEBOUNCE_S <= 0.12, "a hold must not feel laggy"
    assert _POLL_HZ >= 20, "a release must be noticed promptly"


def test_start_and_stop_are_idempotent():
    ptt = PushToTalk(lambda: None, lambda: None)
    ptt.start()
    ptt.start()
    ptt.stop()
    ptt.stop()
    assert ptt.running is False


# ── the capture gate ─────────────────────────────────────────────────────────

@pytest.fixture
def ptt_state():
    """Restore the module-level gate whatever a test does to it."""
    from orion_core import push_to_talk as ptt

    before = (ptt._enabled, ptt._watcher)
    yield ptt
    ptt._enabled, ptt._watcher = before


def _gate_stub(raw: bool):
    from orion_core.audio import AudioGateThread

    gate = AudioGateThread.__new__(AudioGateThread)
    gate._raw_can_capture = lambda: raw
    return gate


def test_with_push_to_talk_off_capture_is_unchanged(ptt_state):
    ptt_state.disable()
    assert ptt_state.gate_allows_capture() is True
    assert _gate_stub(raw=True)._capture_allowed() is True


def test_with_push_to_talk_on_the_microphone_is_closed_until_held(ptt_state):
    """The privacy property that makes the feature worth having: nothing leaves
    the machine while you are not holding the key."""
    class _Held:
        held = False

    ptt_state._enabled = True
    ptt_state._watcher = _Held()
    assert _gate_stub(raw=True)._capture_allowed() is False
    _Held.held = True
    assert _gate_stub(raw=True)._capture_allowed() is True


def test_push_to_talk_only_ever_subtracts(ptt_state):
    """Every existing reason not to capture — paused, muted, no device — must
    still win. The gate wraps the caller's answer, it does not replace it."""
    class _Held:
        held = True

    ptt_state._enabled = True
    ptt_state._watcher = _Held()
    assert _gate_stub(raw=False)._capture_allowed() is False


def test_a_raising_caller_gate_closes_rather_than_opens(ptt_state):
    from orion_core.audio import AudioGateThread

    ptt_state.disable()
    gate = AudioGateThread.__new__(AudioGateThread)

    def explode():
        raise RuntimeError("no device")

    gate._raw_can_capture = explode
    assert gate._capture_allowed() is False


def test_disabling_never_raises_on_teardown(ptt_state):
    class _Broken:
        held = False

    ptt_state._enabled = True
    ptt_state._watcher = _Broken()      # no stop() at all
    ptt_state.disable()                 # must not raise
    assert ptt_state.enabled() is False
    assert ptt_state.gate_allows_capture() is True
