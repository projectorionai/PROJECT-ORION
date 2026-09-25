"""
Streaming silence compression — smoother speech without gabbling it.

Adapted from MARK XL's `_compress_silence`, which works on a complete
utterance. ORION streams, so the run length has to carry across chunk
boundaries — a per-chunk implementation would leave a full-length pause
every time one straddled a boundary, which is most of them.
"""

from __future__ import annotations

import array

import pytest

from orion_core.audio_smoothing import FRAME_SAMPLES, SilenceCompressor

RATE = 24_000


def silence(ms: float) -> bytes:
    n = int(RATE * ms / 1000)
    return array.array("h", [0] * n).tobytes()


def tone(ms: float, amp: int = 9000) -> bytes:
    n = int(RATE * ms / 1000)
    return array.array("h", [amp if i % 2 else -amp for i in range(n)]).tobytes()


def ms_of(data: bytes) -> float:
    return len(data) / 2 / RATE * 1000.0


@pytest.fixture
def comp():
    return SilenceCompressor(sample_rate=RATE, max_silence_ms=400)


# ── it caps long pauses ───────────────────────────────────────────────────────

def test_a_long_pause_is_capped(comp):
    out = comp.feed(tone(200) + silence(1500) + tone(200))
    out += comp.flush()
    # 200 speech + <=400 silence + 200 speech, within a frame's tolerance
    assert ms_of(out) < 900
    assert comp.trimmed_ms > 900


def test_a_short_pause_is_left_completely_alone(comp):
    payload = tone(150) + silence(180) + tone(150)
    out = comp.feed(payload) + comp.flush()
    assert out == payload, "natural phrasing must survive untouched"
    assert comp.trimmed_ms == 0


def test_speech_is_never_removed(comp):
    payload = tone(600)
    out = comp.feed(payload) + comp.flush()
    assert out == payload


def test_it_never_returns_more_than_it_was_given(comp):
    payload = tone(100) + silence(2000) + tone(100)
    out = comp.feed(payload) + comp.flush()
    assert len(out) <= len(payload)


# ── the streaming part: state must carry across chunks ───────────────────────

def test_a_pause_split_across_chunks_is_still_capped(comp):
    """The whole reason this is not MARK XL's function copied over."""
    payload = silence(1600)
    out = b""
    step = FRAME_SAMPLES * 2 * 3          # three frames per chunk
    for i in range(0, len(payload), step):
        out += comp.feed(payload[i:i + step])
    out += comp.flush()
    assert ms_of(out) < 500, (
        "a pause straddling chunk boundaries must not reset the run")


def test_chunking_gives_the_same_result_as_one_pass():
    payload = tone(120) + silence(1200) + tone(120) + silence(900) + tone(120)

    whole = SilenceCompressor(sample_rate=RATE, max_silence_ms=400)
    one_pass = whole.feed(payload) + whole.flush()

    streamed = SilenceCompressor(sample_rate=RATE, max_silence_ms=400)
    out = b""
    step = FRAME_SAMPLES * 2
    for i in range(0, len(payload), step):
        out += streamed.feed(payload[i:i + step])
    out += streamed.flush()

    assert out == one_pass


def test_a_partial_frame_is_held_then_flushed(comp):
    partial = tone(2)                      # shorter than one frame
    assert comp.feed(partial) == b""       # held back
    assert comp.flush() == partial         # never lost


def test_reset_starts_a_new_utterance(comp):
    comp.feed(silence(1500))
    comp.reset()
    out = comp.feed(silence(200)) + comp.flush()
    assert ms_of(out) == pytest.approx(200, abs=20), (
        "a new utterance must not inherit the previous one's silence run")


# ── robustness on the audio path ─────────────────────────────────────────────

def test_empty_input_is_safe(comp):
    assert comp.feed(b"") == b""
    assert comp.flush() == b""


def test_an_odd_byte_count_never_raises(comp):
    assert isinstance(comp.feed(b"\x01\x02\x03"), bytes)


def test_quiet_speech_is_not_mistaken_for_silence():
    comp = SilenceCompressor(sample_rate=RATE, max_silence_ms=400)
    quiet = tone(1200, amp=700)            # soft, but clearly speech
    out = comp.feed(quiet) + comp.flush()
    assert out == quiet, "clipping quiet speech is worse than a long pause"


def test_it_reports_what_it_removed(comp):
    comp.feed(tone(80) + silence(1400) + tone(80))
    comp.flush()
    assert "dead air removed" in comp.describe()


# ── it is actually wired into the voice ──────────────────────────────────────

def test_the_elevenlabs_stream_uses_it():
    import inspect

    from orion_core.audio import SpeechSynthesiser

    source = inspect.getsource(SpeechSynthesiser._speak_elevenlabs)
    assert "SilenceCompressor" in source
    assert "smoother.flush()" in source, (
        "the final partial frame must still reach the renderer")
