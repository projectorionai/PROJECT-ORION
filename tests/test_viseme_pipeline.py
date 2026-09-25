"""
Lip-sync pipeline — language independence, formants, fusion, and a live producer.

Three separate things were wrong before this pipeline existed, and each has a
test here because each was invisible from the outside:

1. ``bus.viseme`` had NO PRODUCER. The signal was declared in bus.py, the Core
   Window subscribed to it, and orion_core/viseme.py implemented a scheduler
   for it — but nothing in ORION ever emitted one. Every face was driven by
   loudness alone while a complete lip-sync module sat inert beside it. A unit
   test of the module would have passed the whole time.

2. ``text_to_visemes`` matched ``[a-z]+``, so every non-English script produced
   an EMPTY timeline. ORION's mouth did not move when he spoke Turkish,
   Russian or Greek, and nothing reported it.

3. Mouth shape came from three broad energy bands at 20 Hz — enough for an
   amplitude orb, too coarse for a mouth, and unable to tell an open /a/ from
   a rounded /u/ at the same volume.

All offline: numpy only, no Qt, no audio device.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.viseme import (
    VISEMES,
    VisemeStream,
    pcm_shapes,
    script_coverage,
    text_to_visemes,
    to_latin,
)

SR = 24000


def vowel_pcm(f1: float, f2: float, seconds: float = 0.25) -> np.ndarray:
    """A two-formant synthetic vowel — the cheapest honest test signal."""
    t = np.arange(int(SR * seconds)) / SR
    wave = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)
    return (wave * 0.5).astype(np.float32)


# ── 1. language independence ─────────────────────────────────────────────────

@pytest.mark.parametrize("language,text", [
    ("English", "hello there"),
    ("Turkish", "günaydın efendim"),
    ("German", "schön über alles"),
    ("Spanish", "buenos días señor"),
    ("Polish", "dzień dobry"),
    ("Vietnamese", "xin chào bạn"),
    ("Russian", "привет мир"),
    ("Ukrainian", "добрий день"),
    ("Greek", "καλημέρα κόσμε"),
])
def test_every_latin_cyrillic_and_greek_script_moves_the_mouth(language, text):
    """All of these produced an EMPTY timeline before the Unicode reduction."""
    timeline = text_to_visemes(text)
    assert timeline, f"{language} produced no mouth movement at all"
    assert all(v in VISEMES for v, _ in timeline), language


@pytest.mark.parametrize("script,text", [
    ("Chinese", "你好世界"),
    ("Japanese", "こんにちは"),
    ("Arabic", "مرحبا بالعالم"),
    ("Hebrew", "שלום עולם"),
    ("Hindi", "नमस्ते दुनिया"),
    ("Thai", "สวัสดีชาวโลก"),
])
def test_scripts_that_hide_pronunciation_fall_back_cleanly(script, text):
    """Their spelling does not reveal pronunciation, so miming shapes onto them
    would be worse than not trying. The caller falls back to the audio-only
    mouth, which is physics and therefore already language-independent: less
    detail, never wrong."""
    assert script_coverage(text) < 0.55, script
    assert text_to_visemes(text) == [], script


def test_diacritics_reduce_to_their_base_letter():
    for accented, base in [("é", "e"), ("ü", "u"), ("ş", "s"), ("ğ", "g"),
                           ("ế", "e"), ("ñ", "n"), ("å", "a"), ("ı", "i"),
                           ("ł", "l"), ("ø", "o")]:
        assert to_latin(accented) == base, accented


def test_cyrillic_and_greek_transliterate():
    assert to_latin("м") == "m" and to_latin("а") == "a"
    assert to_latin("π") == "p" and to_latin("ω") == "o"


def test_the_english_contract_is_unchanged_by_the_reduction():
    """The reduction is identity for a-z, so existing behaviour must survive."""
    assert [v for v, _ in text_to_visemes("mama")] == ["PP", "aa", "PP", "aa", "sil"]
    assert [v for v, _ in text_to_visemes("the")] == ["TH", "E", "sil"]


def test_a_lip_closure_is_produced_for_the_bilabials():
    """/m/, /b/ and /p/ are the reason the transcript is consulted at all."""
    for word in ("mama", "baba", "papa"):
        assert "PP" in [v for v, _ in text_to_visemes(word)], word


# ── 2. formants ──────────────────────────────────────────────────────────────

def test_an_open_vowel_reads_open_and_a_close_vowel_reads_closed():
    """F1 climbs as the jaw drops. This is what loudness could never tell you."""
    open_frames = pcm_shapes(vowel_pcm(800, 1200), SR)
    close_frames = pcm_shapes(vowel_pcm(300, 2600), SR)
    open_mean = float(np.mean([f[1] for f in open_frames]))
    close_mean = float(np.mean([f[1] for f in close_frames]))
    assert open_mean > 0.6, open_mean
    assert close_mean < 0.3, close_mean
    assert open_mean - close_mean > 0.4


def test_spread_and_rounded_vowels_are_distinguishable():
    """F2 is high for spread vowels and low for rounded ones."""
    spread = float(np.mean([f[2] for f in pcm_shapes(vowel_pcm(300, 2600), SR)]))
    rounded = float(np.mean([f[2] for f in pcm_shapes(vowel_pcm(320, 800), SR)]))
    assert spread > 0.3, spread
    assert rounded < 0.0, rounded


def test_the_analysis_rate_is_fast_enough_for_a_plosive():
    """50 shapes a second. At the old 20 Hz a plosive can begin and end inside
    one frame, so the closure that distinguishes /m/ from /n/ never arrives."""
    frames = pcm_shapes(vowel_pcm(600, 1500, seconds=1.0), SR)
    assert 45 <= len(frames) <= 55, len(frames)


def test_every_sample_of_a_block_is_covered():
    """Stepping only while a full window fits stopped short of the end, so a
    200 ms batch yielded 160 ms of mouth and each batch drifted further behind
    the voice than the last."""
    seconds = 0.2
    frames = pcm_shapes(vowel_pcm(600, 1500, seconds=seconds), SR)
    assert len(frames) >= int(seconds / 0.020) - 1


def test_silence_produces_a_closed_resting_mouth():
    frames = pcm_shapes(np.zeros(SR // 4, dtype=np.float32), SR)
    assert frames
    assert all(f == (0.0, 0.0, 0.0) for f in frames)


def test_int16_input_is_accepted_as_well_as_float():
    """The audio path hands over raw int16 straight off the device."""
    as_int16 = (vowel_pcm(800, 1200) * 20000).astype("<i2")
    frames = pcm_shapes(as_int16, SR)
    assert frames and max(f[0] for f in frames) > 0.05


def test_malformed_audio_never_raises():
    for bad in (None, "not audio", [], np.array([]), np.zeros((3, 3, 3))):
        assert pcm_shapes(bad, SR) == [] or isinstance(pcm_shapes(bad, SR), list)


# ── 3. fusion ────────────────────────────────────────────────────────────────

def test_the_transcript_supplies_a_closure_the_spectrum_cannot_see():
    """THE point of fusing two sources.

    The audio here is a constantly-open vowel. Only the transcript knows the
    lips close on the /m/ of "mama", and no filter bank could ever tell you.
    """
    stream = VisemeStream()
    stream.feed_text("mama")
    audio = [(0.8, 0.9, 0.0)] * 40
    fused = stream.frames(audio, 0.020)
    assert min(o for _, o, _ in fused) < 0.05, "no closure reached the mouth"
    assert max(o for _, o, _ in fused) > 0.5, "the mouth never opened either"


def test_with_no_transcript_the_audio_shape_passes_through():
    """Feeding text is optional; an empty queue must not mute the mouth."""
    stream = VisemeStream()
    audio = [(0.7, 0.62, 0.25)] * 10
    fused = stream.frames(audio, 0.020)
    assert all(abs(o - 0.62) < 1e-6 for _, o, _ in fused)


def test_silence_does_not_burn_through_the_queued_shapes():
    """During a pause the queue must WAIT, or the mouth ends up ahead of the
    voice for the rest of the sentence."""
    stream = VisemeStream()
    stream.feed_text("hello there friend")
    before = stream.pending
    stream.frames([(0.0, 0.0, 0.0)] * 50, 0.020)
    assert stream.pending == before


def test_a_stalled_turn_cannot_grow_an_unbounded_backlog():
    stream = VisemeStream()
    for _ in range(400):
        stream.feed_text("the quick brown fox jumps over the lazy dog")
    assert stream.pending <= 600


def test_reset_clears_the_mouth_between_turns():
    stream = VisemeStream()
    stream.feed_text("hello")
    stream.reset()
    assert stream.pending == 0


def test_fused_output_stays_in_range():
    stream = VisemeStream()
    stream.feed_text("mama papa baba")
    fused = stream.frames([(1.0, 1.0, 1.0)] * 60, 0.020)
    for level, openness, width in fused:
        assert 0.0 <= openness <= 1.0
        assert -1.0 <= width <= 1.0
        assert 0.0 <= level <= 1.0


# ── 4. the producer actually exists ──────────────────────────────────────────

def test_the_audio_path_emits_a_viseme():
    """The regression that matters most: this signal had no producer at all.

    Driven through the real ``_VisualiserThread._emit_viseme`` with a stub bus,
    so it fails if the emission is removed, renamed or silently swallowed.
    """
    from orion_core.audio import RECEIVE_SAMPLE_RATE, _VisualiserThread

    seen: list = []

    class _Signal:
        def __init__(self, name: str) -> None:
            self.name = name

        def emit(self, *payload) -> None:
            seen.append((self.name, payload))

    bus = types.SimpleNamespace(amplitude=_Signal("amplitude"),
                                voice_spectrum=_Signal("spectrum"),
                                viseme=_Signal("viseme"))
    thread = _VisualiserThread.__new__(_VisualiserThread)
    thread.renderer = types.SimpleNamespace(bus=bus, _viseme_stream=None)

    pcm = (vowel_pcm(800, 1200, seconds=0.06) * 20000).astype("<i2").tobytes()
    thread._emit_viseme(pcm)

    postures = [payload[0] for name, payload in seen if name == "viseme"]
    assert postures, "bus.viseme still has no producer"
    assert "open" in postures[-1] and "width" in postures[-1]
    assert 0.0 <= postures[-1]["open"] <= 1.0
    assert RECEIVE_SAMPLE_RATE == SR


def test_the_visualiser_runs_fast_enough_for_a_mouth():
    from orion_core.audio import _VisualiserThread

    assert _VisualiserThread.INTERVAL_S <= 0.03, (
        "a mouth needs at least ~33 Hz; a plosive fits inside a 50 ms frame"
    )


def test_a_broken_chunk_never_takes_audio_down():
    """The mouth is cosmetic. Audio is not."""
    from orion_core.audio import _VisualiserThread

    thread = _VisualiserThread.__new__(_VisualiserThread)
    thread.renderer = types.SimpleNamespace(bus=None, _viseme_stream=None)
    thread._emit_viseme(b"\x01")          # too short, and bus is None
    thread._emit_viseme(b"")
