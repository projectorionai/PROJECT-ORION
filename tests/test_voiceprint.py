"""Telling one voice from another, in the application people actually install.

ORION had a speaker encoder and could not use it. resemblyzer runs on torch,
torch is 527 MB, and the packaged build excludes it — so recognising voices
worked on a developer machine and was absent from the installed application.
And nothing acted on the result anyway: who spoke was noted, then ignored.

This covers the module that fixes both: the same published network exported to
ONNX so it runs without torch, and a gate that lets ORION decline to answer a
voice that is not his owner's.

The synthetic voices below are built from different fundamentals and formant
structures — which is what vocal tracts of different sizes produce. They are
not human, so the absolute numbers matter less than the ORDER: the same voice
must score above the threshold and a different one below it, with room to
spare. A test that only checked "an embedding came back" would have passed
while the encoder returned the same point for everybody.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import voiceprint  # noqa: E402

np = pytest.importorskip("numpy")

SR = voiceprint.SAMPLE_RATE

#: (fundamental, formants) — a small larynx and a large one sound different
#: because both the pitch and the resonances move.
VOICES = {
    "owner": (112.0, [(620, 1.0, 90), (1180, 0.55, 120), (2500, 0.28, 180)]),
    "other": (208.0, [(780, 1.0, 100), (1950, 0.60, 140), (2900, 0.30, 200)]),
    "third": (145.0, [(500, 1.0, 80), (1500, 0.50, 130), (2650, 0.25, 190)]),
}


def speak(which: str, seconds: float = 3.2, seed: int = 0):
    """A crude vocal tract: a buzzing source shaped by fixed resonances."""
    f0, formants = VOICES[which]
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    phase = 2 * np.pi * f0 * t
    source = sum(np.sin(h * phase) / h for h in range(1, 40))
    freqs = np.fft.rfftfreq(t.size, 1 / SR)
    spectrum = np.fft.rfft(source)
    signal = np.zeros_like(t)
    for centre, gain, width in formants:
        shaped = spectrum * np.exp(-((freqs - centre) ** 2) / (2 * width ** 2))
        signal += gain * np.fft.irfft(shaped, n=t.size)
    signal *= 0.55 + 0.45 * np.sin(2 * np.pi * 3.7 * t + rng.random() * 6)
    signal += 0.002 * rng.standard_normal(t.size)
    peak = float(np.abs(signal).max())
    return (signal / peak * 0.6).astype(np.float32) if peak else signal


def pcm(which: str, **kwargs) -> bytes:
    return (speak(which, **kwargs) * 32767).astype(np.int16).tobytes()


needs_encoder = pytest.mark.skipif(
    not voiceprint.available(),
    reason="the exported encoder or onnxruntime is unavailable")


# ── the front end ────────────────────────────────────────────────────────────

def test_the_mel_spectrogram_has_the_shape_the_encoder_expects():
    """40 channels at a 10 ms hop. The encoder was trained on exactly this;
    anything else produces an embedding that means nothing in particular."""
    mels = voiceprint.mel_spectrogram(speak("owner", seconds=1.0))
    assert mels.shape[1] == voiceprint.N_MELS
    assert mels.shape[0] == pytest.approx(100, abs=2)


def test_the_filterbank_is_area_normalised():
    """Slaney normalisation, not equal peaks.

    With equal peaks the low filters dominate every frame, which raises
    nothing anywhere — it just quietly makes everyone sound alike.
    """
    filters = voiceprint._mel_filters()
    areas = filters.sum(axis=1)
    peaks = filters.max(axis=1)
    assert float(peaks.max() / peaks.min()) > 3.0, (
        "the peaks are uniform, so these are not area-normalised filters")
    assert float(areas.max() / areas.min()) < 2.5


# ── the encoder ──────────────────────────────────────────────────────────────

@needs_encoder
def test_an_embedding_is_a_unit_vector_of_the_right_size():
    vector = voiceprint.embed(speak("owner"), SR)
    assert vector is not None
    assert vector.shape == (256,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


@needs_encoder
def test_the_same_voice_scores_far_above_a_different_one():
    """The property everything else rests on.

    Checked as a comparison rather than against a fixed number, because a
    threshold that happened to sit between two nearly-equal scores would pass
    a test and fail a room.
    """
    mine = [voiceprint.embed(speak("owner", seed=s), SR) for s in (1, 2, 3)]
    theirs = voiceprint.embed(speak("other", seed=4), SR)
    same = min(voiceprint.similarity(mine[0], other) for other in mine[1:])
    different = max(voiceprint.similarity(vector, theirs) for vector in mine)
    assert same > voiceprint.DEFAULT_THRESHOLD
    assert different < voiceprint.DEFAULT_THRESHOLD
    assert same - different > 0.15, (
        f"only {same - different:.2f} between the same voice and a different "
        "one; the threshold has nowhere safe to sit")


@needs_encoder
def test_silence_is_not_a_speaker():
    """It scored as "a different voice" until this was fixed.

    Which would have made ORION stop listening in a quiet room and stay that
    way — the exact failure the gate must never cause.
    """
    assert voiceprint.embed(np.zeros(SR * 2, dtype=np.float32), SR) is None


@needs_encoder
def test_a_clip_too_short_to_judge_returns_nothing():
    assert voiceprint.embed(speak("owner", seconds=0.2), SR) is None


@needs_encoder
def test_int16_handed_in_as_floats_is_understood():
    """Audio arrives from several places in this codebase and not all of them
    have divided by 32768 yet. Treating a ±32767 signal as if it were ±1
    clips it into noise and the voiceprint becomes meaningless."""
    quiet = speak("owner")
    loud = (quiet * 32767).astype(np.float32)
    assert voiceprint.similarity(voiceprint.embed(quiet, SR),
                                 voiceprint.embed(loud, SR)) > 0.95


@needs_encoder
def test_a_different_sample_rate_gives_the_same_person():
    """The microphone does not always run at 16 kHz."""
    at_16k = voiceprint.embed(speak("owner", seed=9), SR)
    upsampled = np.repeat(speak("owner", seed=9), 3)      # crude 48 kHz
    assert voiceprint.similarity(at_16k,
                                 voiceprint.embed(upsampled, SR * 3)) > 0.85


# ── enrolment and the gate ───────────────────────────────────────────────────

@pytest.fixture()
def prints(tmp_path):
    """A store on scratch paths, so no test can touch a real enrolment."""
    return voiceprint.Voiceprints(path=tmp_path / "settings.json",
                                  profiles_path=tmp_path / "profiles.json")


@needs_encoder
def test_enrolling_then_recognising(prints):
    ok, message = prints.enrol("SampleUser", [speak("owner", seed=s)
                                           for s in (1, 2, 3)], SR)
    assert ok, message
    assert prints.names() == ["SampleUser"]
    assert prints.owner == "SampleUser"

    verdict = prints.is_owner(speak("owner", seed=77), SR)
    assert verdict.is_owner and verdict.confident
    assert verdict.score > voiceprint.DEFAULT_THRESHOLD


@needs_encoder
def test_a_clearly_different_voice_is_refused(prints):
    prints.enrol("SampleUser", [speak("owner", seed=s) for s in (1, 2, 3)], SR)
    verdict = prints.is_owner(speak("other", seed=88), SR)
    assert verdict.is_owner is False
    assert verdict.score < voiceprint.DEFAULT_THRESHOLD


@needs_encoder
def test_enrolment_survives_being_written_and_read_back(prints, tmp_path):
    prints.enrol("SampleUser", [speak("owner", seed=1)], SR)
    fresh = voiceprint.Voiceprints(path=prints.path,
                                   profiles_path=prints.profiles_path)
    assert fresh.names() == ["SampleUser"]
    assert fresh.owner == "SampleUser"


@needs_encoder
def test_the_embeddings_are_shared_with_the_older_service(tmp_path,
                                                          monkeypatch):
    """One enrolment, not two.

    The obvious way to write this module was with its own profile store, and
    it would have been wrong in a way nobody would notice for weeks: you
    enrol your voice, ORION confirms it, and the gate still has nothing to
    compare against because the enrolment went somewhere else.
    """
    from orion_core import voice_speaker_id

    profiles = tmp_path / "voice_profiles.json"
    monkeypatch.setattr(voice_speaker_id, "PROFILES_PATH", profiles)

    service = voice_speaker_id.SpeakerIdentificationService()
    result = service.enroll("SampleUser", [speak("owner", seed=s)
                                        for s in (1, 2)], SR, consent=True)
    assert result.ok, result.text

    gate = voiceprint.Voiceprints(path=tmp_path / "settings.json",
                                  profiles_path=profiles)
    assert gate.names() == ["SampleUser"], (
        "the gate cannot see a voice enrolled through the tool")
    assert gate.is_owner(speak("owner", seed=55), SR).is_owner


# ── failing open ─────────────────────────────────────────────────────────────

def test_with_nobody_enrolled_everyone_is_the_owner(prints):
    """Otherwise installing this update makes ORION deaf until you notice."""
    assert prints.is_owner(speak("other"), SR).is_owner is True


def test_the_gate_is_off_until_it_is_asked_for(prints):
    assert prints.only_owner is False
    assert voiceprint.should_listen(pcm("other"), SR).is_owner is True


def test_the_gate_refuses_to_turn_on_with_nothing_enrolled(prints):
    """It would ignore everybody, including its owner."""
    ok, message = prints.set_only_owner(True)
    assert ok is False
    assert "enrol" in message.lower()
    assert prints.only_owner is False


@needs_encoder
def test_an_uncertain_score_resolves_towards_listening(prints, monkeypatch):
    """The whole safety property.

    Wrongly ignoring the owner looks like broken hardware and is miserable to
    diagnose; wrongly answering a stranger costs one unwanted reply. They are
    not comparable, so the margin always resolves towards listening.
    """
    prints.enrol("SampleUser", [speak("owner", seed=1)], SR)
    # A threshold just above the owner's own score puts them in the margin.
    # Capped at 1.0 rather than 0.99: these voices score around 0.99 against
    # themselves, and a 0.99 cap landed BELOW the score, so the owner was
    # simply recognised and the margin was never exercised.
    score = prints.is_owner(speak("owner", seed=2), SR).score
    monkeypatch.setenv("ORION_VOICE_THRESHOLD", f"{min(1.0, score + 0.04):.3f}")
    verdict = prints.is_owner(speak("owner", seed=2), SR)
    assert verdict.is_owner is True
    assert verdict.confident is False
    assert "uncertain" in verdict.reason


def test_unreadable_audio_does_not_silence_him():
    assert voiceprint.should_listen(b"\x01", SR).is_owner is True


def test_a_verdict_is_truthy_in_the_obvious_way():
    assert bool(voiceprint.Verdict(True)) is True
    assert bool(voiceprint.Verdict(False)) is False


# ── it is reachable where it matters ─────────────────────────────────────────

def test_the_model_ships_with_the_application():
    """assets/ is bundled wholesale, so the encoder travels with the build —
    which is the entire point of exporting it."""
    assert voiceprint.model_path().is_file(), (
        "the exported encoder is missing; speaker recognition would be "
        "absent from the packaged application exactly as before")
    assert voiceprint.model_path().stat().st_size < 40_000_000


def test_torch_is_not_needed():
    """The reason this module exists.

    resemblyzer is excluded from the build because torch is 527 MB. If the
    torch-free path quietly depended on it again, recognition would vanish
    from the installed application and nothing would say so.
    """
    source = (ROOT / "orion_core" / "voiceprint.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "import resemblyzer", "import librosa"):
        assert forbidden not in source


def test_the_transcriber_consults_the_gate():
    """Identification that changes nothing is what was already there."""
    source = (ROOT / "orion_core" / "speech_offline.py").read_text(
        encoding="utf-8")
    assert "_owner_is_speaking" in source
    body = source[source.index("def transcribe_pcm"):]
    body = body[:body.index("\n    def ", 10)]
    assert "_owner_is_speaking" in body, (
        "the gate is defined but never consulted on the path that turns an "
        "utterance into a command")


# ── who counts as the owner ──────────────────────────────────────────────────

def test_a_single_enrolled_voice_is_the_owner(prints, tmp_path):
    """Enrolling through the older voice_speaker_id tool nominates nobody.

    Without this you could enrol your voice, be told it worked, ask ORION to
    answer only you, and be told to enrol a voice first — which you just had.
    """
    import json

    prints.profiles_path.write_text(
        json.dumps({"Jordan": {"embedding": [0.0] * 256}}), encoding="utf-8")
    assert prints.owner == "Jordan"
    assert prints.enrolled() is True


def test_with_several_voices_nobody_is_assumed(prints):
    """Guessing which of several people the machine belongs to is not a guess
    worth making silently — the owner action exists to say."""
    import json

    prints.profiles_path.write_text(
        json.dumps({"Jordan": {"embedding": [0.0] * 256},
                    "Sam": {"embedding": [0.0] * 256}}), encoding="utf-8")
    assert prints.owner == ""


def test_a_nomination_beats_the_single_voice_rule(prints):
    import json

    prints.profiles_path.write_text(
        json.dumps({"Jordan": {"embedding": [0.0] * 256},
                    "Sam": {"embedding": [0.0] * 256}}), encoding="utf-8")
    assert prints.set_owner("Sam")[0] is True
    assert prints.owner == "Sam"


def test_the_gate_works_for_a_voice_enrolled_through_the_older_tool(prints):
    """The bug this nearly shipped with.

    `set_only_owner` resolved the owner through the property, which treats a
    single enrolled voice as yours. `is_owner` — the one method the gate
    actually calls — read the raw field instead. So enrolling through
    voice_speaker_id and saying "only listen to my voice" would have turned
    the gate ON and left it doing nothing: every utterance came back "no owner
    voice is enrolled", which fails open.

    A gate that reports success and changes nothing is precisely the state
    this whole capability was built to replace.
    """
    import json

    prints.profiles_path.write_text(
        json.dumps({"SampleUser": {"embedding": [0.1] * 256,
                                "backend": "voiceprint-onnx"}}),
        encoding="utf-8")
    assert prints.set_only_owner(True)[0] is True

    verdict = prints.is_owner(speak("owner"), SR)
    assert verdict.reason != "no owner voice is enrolled", (
        "the gate cannot see the voice that was enrolled")


def test_describe_names_the_same_owner_the_gate_uses(prints):
    """Two different answers to "whose voice is this set to" is how the
    previous bug stayed invisible."""
    import json

    prints.profiles_path.write_text(
        json.dumps({"SampleUser": {"embedding": [0.1] * 256}}), encoding="utf-8")
    assert "SampleUser" in prints.describe()
    assert prints.owner == "SampleUser"
