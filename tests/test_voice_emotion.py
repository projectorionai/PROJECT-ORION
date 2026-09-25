"""
Reading emotion from HOW something is said.

"I want ORION to be able to distinguish between my different emotions such as
anger, frustration, excitement, happiness, scared and etc."

What is actually being tested
-----------------------------
Synthesised utterances with known acoustics, not recordings. That bounds what
these tests can honestly claim: they establish that the CONTRASTS the design
rests on are really being measured and really drive the answer — loud-and-flat
versus loud-and-swooping, quiet-and-high versus loud, slow-and-low versus fast
— and that the machinery around them behaves. They do not establish accuracy on
real human speech, and no test here should be read as claiming it.

The contrasts are the right thing to pin down regardless, because every defect
found while building this was a contrast collapsing rather than a threshold
being slightly off:

  * peak-normalising the waveform deleted loudness, the strongest single cue —
    five deliberately different utterances measured within 1 dB of each other;
  * symmetric scoring made a shout match a mild profile better than an intense
    one, so genuine anger scored below a mild reading of it;
  * one-sided scoring alone then let an extreme utterance match every milder
    profile for free — a whisper overshot all of calm's targets and read as
    calm;
  * the baseline's spread estimate measured deviation from the ALREADY-UPDATED
    mean, which subtracts out most of what it is trying to measure, so after 50
    varied utterances every scale had collapsed to its floor and ordinary
    speech saturated the clamp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from orion_core.voice_emotion import (  # noqa: E402
    NOMINAL_SR, EmotionBaseline, EmotionReader, Features, extract,
)


def utterance(f0: float = 140.0, seconds: float = 1.4, amp: float = 0.30,
              vibrato: float = 0.0, rate: float = 3.0, bright: float = 0.0,
              sr: int = NOMINAL_SR):
    """A voice-like signal with controllable prosody.

    A glottal-ish pulse train (fundamental + two harmonics) gated into
    syllables. Crude, but every feature the reader measures is present and
    independently controllable, which is what these tests need.
    """
    t = np.arange(int(seconds * sr)) / sr
    instantaneous = f0 * (1 + vibrato * np.sin(2 * np.pi * 0.9 * t))
    phase = 2 * np.pi * np.cumsum(instantaneous) / sr
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    signal += bright * (0.35 * np.sin(6 * phase) + 0.3 * np.sin(9 * phase))
    syllables = (np.sin(2 * np.pi * rate * t) > -0.25).astype(float)
    return (signal * syllables * amp).astype(np.float32)


#: Prosody for each state, in the terms the design is written in.
VOICES = {
    "neutral":    dict(),
    "angry":      dict(amp=1.60, f0=205, rate=3.6, bright=0.7),          # loud, flat, hard
    "excited":    dict(amp=1.50, f0=235, vibrato=0.25, rate=5.2),        # loud, swooping
    "scared":     dict(amp=0.09, f0=265, bright=1.2, rate=5.0),          # quiet, high, bright
    "sad":        dict(amp=0.07, f0=95, rate=1.1),                       # quiet, low, slow
    "frustrated": dict(amp=0.34, f0=152, vibrato=0.0, rate=6.5),         # normal volume, fast, tight
    "happy":      dict(amp=0.36, f0=168, vibrato=0.20, rate=3.3),        # mildly up, lilting
    "tired":      dict(amp=0.16, f0=133, rate=1.5),                      # quiet, slow, pitch barely down
}


@pytest.fixture
def reader(tmp_path):
    """A reader that has heard fifty utterances of ordinary, VARIED speech.

    The variation matters: an earlier fixture trained on fifty identical
    utterances, which drove every spread estimate to its floor and made
    ordinary speech saturate the clamp — the fixture, not the model, was
    producing the wrong answers.
    """
    rng = np.random.default_rng(7)
    engine = EmotionReader(EmotionBaseline(tmp_path / "emotion.json"))
    for _ in range(50):
        engine.baseline.observe(extract(utterance(
            f0=140 * rng.normal(1, 0.10), amp=0.30 * rng.normal(1, 0.28),
            vibrato=abs(rng.normal(0, 0.05)), rate=3.0 * rng.normal(1, 0.18),
            bright=abs(rng.normal(0, 0.12)))), "you")
    return engine


def read(engine, name):
    return engine.read(utterance(**VOICES[name]), speaker="you", learn=False)


# ── the features are real measurements ───────────────────────────────────────

def test_loudness_is_actually_measured():
    """The regression: peak-normalising deleted the strongest cue entirely."""
    quiet = extract(utterance(amp=0.05))
    loud = extract(utterance(amp=1.5))
    assert loud.energy_db > quiet.energy_db + 15, (
        "a shout and a whisper measured the same loudness")


def test_pitch_is_tracked_not_guessed_from_the_loudest_harmonic():
    """Autocorrelation, because the fundamental is often weaker than its
    harmonics — an FFT peak lands on a harmonic and reports double."""
    for f0 in (110, 180, 260):
        measured = extract(utterance(f0=f0)).pitch_hz
        assert abs(measured - f0) < f0 * 0.12, f"{f0} Hz measured as {measured:.0f}"


def test_pitch_variation_separates_flat_from_swooping():
    """The single distinction between anger and excitement."""
    flat = extract(utterance(f0=210, vibrato=0.0))
    swooping = extract(utterance(f0=210, vibrato=0.25))
    assert swooping.pitch_spread > flat.pitch_spread * 4


def test_speech_rate_is_measured():
    assert extract(utterance(rate=6.0)).rate > extract(utterance(rate=1.2)).rate * 2


def test_brightness_is_measured():
    assert (extract(utterance(bright=1.2)).centroid_hz
            > extract(utterance(bright=0.0)).centroid_hz * 1.4)


def test_silence_and_nonsense_do_not_raise():
    for bad in (np.zeros(8000, dtype=np.float32), np.array([]), [0.0] * 10):
        assert isinstance(extract(bad), Features)


# ── the contrasts the design rests on ────────────────────────────────────────

def test_ordinary_speech_reads_as_neutral(reader):
    """Nothing is more corrosive than being told you sound angry when you
    aren't. An utterance inside the speaker's own normal range is level."""
    assert read(reader, "neutral").emotion == "neutral"


def test_loud_and_flat_is_anger_not_excitement(reader):
    assert read(reader, "angry").emotion == "angry"


def test_loud_and_swooping_is_excitement_not_anger(reader):
    assert read(reader, "excited").emotion == "excited"


def test_quiet_and_high_is_fear_not_excitement(reader):
    """Both are high and fast; only loudness tells them apart."""
    assert read(reader, "scared").emotion == "scared"


def test_quiet_low_and_slow_is_sadness(reader):
    assert read(reader, "sad").emotion == "sad"


def test_fast_and_tight_at_normal_volume_is_frustration_not_anger(reader):
    """Frustration is bounded by loudness — shouting has become anger."""
    assert read(reader, "frustrated").emotion == "frustrated"


def test_every_state_is_reachable(reader):
    """A profile no utterance can ever win is dead weight pretending to be a
    capability."""
    reached = {read(reader, name).emotion for name in VOICES}
    for required in ("angry", "excited", "scared", "sad", "frustrated"):
        assert required in reached, f"nothing ever reads as {required}"


def test_arousal_is_never_confused_with_its_opposite(reader):
    """The failure that would matter most: reading distress as enthusiasm."""
    assert read(reader, "scared").emotion not in {"excited", "happy"}
    assert read(reader, "sad").emotion not in {"excited", "happy", "angry"}
    assert read(reader, "excited").emotion not in {"sad", "scared", "tired"}


# ── it explains itself ───────────────────────────────────────────────────────

def test_a_reading_always_says_what_drove_it(reader):
    """An unexplained verdict about someone's mood is unfalsifiable — the user
    has to be able to say 'no, that's just how I talk'."""
    for name in VOICES:
        assert read(reader, name).because, f"{name} came back with no reason"


def test_the_reason_names_the_feature_that_moved(reader):
    assert "louder" in read(reader, "angry").because
    assert "quieter" in read(reader, "sad").because


def test_describe_is_speakable(reader):
    text = read(reader, "angry").describe()
    assert text.startswith("You sound")
    assert "angry" in text


# ── confidence is honest ─────────────────────────────────────────────────────

def test_a_stranger_gets_low_confidence(tmp_path):
    """With no baseline there is no 'normal' to compare against, so a reading
    is a guess and must not be voiced as anything else."""
    fresh = EmotionReader(EmotionBaseline(tmp_path / "e.json"))
    reading = fresh.read(utterance(**VOICES["angry"]), speaker="nobody", learn=False)
    assert not reading.certain_enough_to_mention


def test_neutral_is_never_worth_mentioning(reader):
    assert not read(reader, "neutral").certain_enough_to_mention


def test_a_clear_reading_from_a_known_speaker_is_mentionable(reader):
    assert read(reader, "angry").certain_enough_to_mention


def test_too_short_to_judge_says_so(reader):
    reading = reader.read(utterance(seconds=0.15), speaker="you", learn=False)
    assert reading.emotion == "neutral"
    assert "too little" in reading.because


# ── baselines are per speaker ────────────────────────────────────────────────

def test_baselines_do_not_leak_between_speakers(reader):
    """'Loud for you' is the only loud that means anything — and mum's voice
    measured against the user's baseline would read as permanently odd."""
    before = reader.baseline.observations("you")
    reader.baseline.observe(extract(utterance(f0=210)), "mum")
    assert reader.baseline.observations("you") == before
    assert reader.baseline.observations("mum") == 1


def test_the_spread_estimate_does_not_collapse(tmp_path):
    """The regression: measuring deviation from the already-updated mean
    subtracts out most of the deviation, so every scale sank to its floor and
    ordinary speech saturated the clamp."""
    rng = np.random.default_rng(3)
    baseline = EmotionBaseline(tmp_path / "e.json")
    for _ in range(60):
        baseline.observe(extract(utterance(f0=140 * rng.normal(1, 0.22),
                                           amp=0.3 * rng.normal(1, 0.5))), "you")
    entry = baseline._data["you"]
    floor = EmotionBaseline.DEFAULT_SCALE["pitch_hz"] * 0.4
    assert entry["pitch_hz_scale"] > floor * 1.05, (
        "the spread estimate collapsed to its floor despite varied input")


def test_learning_only_happens_on_level_utterances(reader):
    """Learning 'normal' from shouting would gradually redefine shouting as
    normal."""
    before = reader.baseline.observations("you")
    reader.read(utterance(**VOICES["angry"]), speaker="you", learn=True)
    assert reader.baseline.observations("you") == before


def test_a_level_utterance_is_learned_from(reader):
    before = reader.baseline.observations("you")
    reader.read(utterance(), speaker="you", learn=True)
    assert reader.baseline.observations("you") == before + 1


def test_baselines_survive_a_restart(tmp_path):
    path = tmp_path / "e.json"
    first = EmotionBaseline(path)
    for _ in range(12):
        first.observe(extract(utterance()), "mum")
    first.save()
    assert EmotionBaseline(path).observations("mum") == 12


def test_forgetting_a_speaker_works(tmp_path):
    baseline = EmotionBaseline(tmp_path / "e.json")
    baseline.observe(extract(utterance()), "mum")
    assert baseline.forget("mum")
    assert baseline.observations("mum") == 0


def test_a_corrupt_state_file_is_survivable(tmp_path):
    path = tmp_path / "e.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert EmotionBaseline(path).observations("you") == 0
