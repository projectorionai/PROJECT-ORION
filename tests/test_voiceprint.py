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
def test_the_encoder_runs_on_one_cpu_thread_even_with_a_gpu(monkeypatch):
    """accelerator.POLICY keeps it on the CPU: 16.3 ms there against 13.7 ms
    on the GPU, whose session build holds Python's lock ~300 ms beside live
    audio. One thread, as before."""
    from orion_core import accelerator
    monkeypatch.setattr(voiceprint, "_SESSION", None)
    monkeypatch.delenv("ORION_GPU", raising=False)
    session = voiceprint._session()
    assert isinstance(session, accelerator.AcceleratedSession)
    assert session.model == "voiceprint" and session.device == "CPU"
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert session.get_session_options().intra_op_num_threads == 1
    assert voiceprint._embed_uncached(speak("owner"), SR).shape == (256,)


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


def test_guided_session_deadline_releases_model_without_another_utterance():
    from orion_core import voice_enrolment as learning

    clock = [0.0]
    session = learning.EnrolmentSession("SampleUser", clock=lambda: clock[0], time_limit=5)
    clock[0] = 6.0
    assert not session.listening
    assert session.progress().state == "failed"


def test_guided_cancel_erases_samples(monkeypatch):
    from orion_core import voice_enrolment as learning

    monkeypatch.setattr(learning, "assess", lambda *a: learning.Assessment(True))
    session = learning.EnrolmentSession("SampleUser")
    assert session.offer(b"\x01\x00" * 24000, embedding=np.array([1., 0., 0.])).accepted
    vector = session._embeddings[0]
    session.cancel()
    assert session._embeddings == [] and session._styles == []
    assert not np.any(vector)


def test_guided_forget_erases_a_completed_or_cancelled_registry(monkeypatch, prints):
    from orion_core import voice_enrolment as learning

    monkeypatch.setattr(learning, "assess", lambda *a: learning.Assessment(True))
    monkeypatch.setattr(learning, "_SESSION", None)
    session = learning.start("SampleUser", store=prints)
    session.offer(b"\x01\x00" * 24000, embedding=np.array([1., 0., 0.]))
    session.cancel()
    learning.forget_my_voice(prints)
    assert learning.current() is None
    assert not session._embeddings


def test_guided_minimum_applies_after_outliers(monkeypatch, prints):
    from orion_core import voice_enrolment as learning

    session = learning.EnrolmentSession("SampleUser", store=prints, minimum=12)
    session._embeddings = [np.array([1., 0., 0.]) for _ in range(12)]
    session._styles = ["normal"] * 12
    monkeypatch.setattr(learning, "_drop_outliers", lambda e, s: (e[:9], s[:9], 3))
    ok, message = session.finish()
    assert not ok and "12" in message
    assert prints.names() == []


def test_guided_save_failure_never_reports_success(monkeypatch, prints):
    from orion_core import voice_enrolment as learning

    session = learning.EnrolmentSession("SampleUser", store=prints, minimum=3,
                                       owner_only_after=False)
    session._embeddings = [np.array([1., 0., 0.]) for _ in range(3)]
    session._styles = ["normal"] * 3

    def locked(*_args, **_kwargs):
        raise PermissionError("locked voice profile")

    monkeypatch.setattr(voiceprint, "_write_json", locked)
    ok, message = session.finish()
    assert not ok and "locked" in message
    assert not session._embeddings


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_guided_nonfinite_audio_is_rejected(bad):
    from orion_core import voice_enrolment as learning

    result = learning.assess(np.full(40000, bad, dtype=np.float32))
    assert not result.ok
    assert voiceprint._unit([bad, 0., 0.]) is None


def test_guided_complete_persists_only_embeddings_and_erases_working_samples(monkeypatch, prints):
    from orion_core import voice_enrolment as learning

    monkeypatch.setattr(learning, "assess", lambda *a: learning.Assessment(True))
    session = learning.EnrolmentSession("SampleUser", store=prints, target=3, minimum=3,
                                       owner_only_after=False)
    for _ in range(3):
        session.offer(b"\x01\x00" * 24000, embedding=np.array([1., 0., 0.]))
    assert session.state == "complete"
    assert not session._embeddings and not session._styles
    assert session.progress().good == 3
    assert prints.names() == ["SampleUser"]
    assert prints._load()["profiles"]["SampleUser"]["method"] == "guided"


def test_voice_turns_keep_names_and_reply_gates_distinct():
    from orion_core import voice_enrolment as learning

    clock = [0.0]
    turns = learning.SpeakerTurns(clock=lambda: clock[0])
    turns.record(voiceprint.Verdict(True, .9, "SampleUser", confident=True, kind="owner"))
    assert turns.address_by_name()
    turns.record(voiceprint.Verdict(False, .2, "Other", confident=True, kind="known"))
    assert not turns.address_by_name() and not turns.reply_allowed(only_owner=True)
    turns.note_unjudged(.2)
    assert turns.reply_allowed(only_owner=True)
    assert not turns.address_by_name()
    clock[0] = 91.0
    assert turns.listener_note() == ""


# Explicit conversation recordings share identification, never authorisation.
def _conversation_recording(tmp_path, **kwargs):
    from orion_core.conversation_recorder import ConversationRecorder

    defaults = {"clock": lambda: 100.0,
                "transcribe": lambda audio, rate: "Synthetic recorded words",
                "identify": lambda audio, rate: voiceprint.Verdict(
                    True, .91, "SyntheticOwner", confident=True, kind="owner")}
    defaults.update(kwargs)
    return ConversationRecorder(tmp_path / "recordings", **defaults)


@pytest.mark.parametrize("consent", [False, None, "true", 1])
def test_room_recording_needs_explicit_consent_for_each_session(tmp_path, consent):
    service = _conversation_recording(tmp_path)
    assert not service.start(consent=consent).ok
    assert not service.active and not service.root.exists()
    assert not service.feed(pcm("owner", seconds=.1), captured_at=100.1)


def test_room_recording_keeps_wav_and_named_local_transcript_without_guard_changes(tmp_path):
    import json
    import wave
    from orion_core import speaker_gate, voice_enrolment

    before = voice_enrolment.turns().history()
    guarded_before = speaker_gate._last
    service = _conversation_recording(tmp_path)
    assert service.start(consent=True).ok
    audio = pcm("owner", seconds=.5)
    assert service.feed(audio, captured_at=100.5)
    result = service.stop()
    assert result.ok and not service.active and not service.record_only
    assert result.receipt["state"] == "stopped"
    assert result.receipt["audio_seconds"] == .5
    path = service.root / result.receipt["id"]
    with wave.open(str(path / "recording.wav"), "rb") as saved:
        assert saved.getframerate() == SR and saved.getnchannels() == 1
        assert saved.readframes(saved.getnframes()) == audio
    row = json.loads((path / "transcript.jsonl").read_text(encoding="utf-8").strip())
    assert row["speaker"] == "SyntheticOwner" and row["text"] == "Synthetic recorded words"
    assert row["start"] == 0 and row["end"] == .5
    assert voice_enrolment.turns().history() == before
    assert speaker_gate._last == guarded_before
    recalled = service.recall(query="recorded words")
    assert recalled.ok and "SyntheticOwner" in recalled.text
    assert "Synthetic recorded words" in recalled.text


@pytest.mark.parametrize("verdict", [
    voiceprint.Verdict(True, .9, "LooksLikeOwner", confident=False, kind="uncertain"),
    voiceprint.Verdict(False, .2, "Unknown", confident=True, kind="unknown"),
    voiceprint.Verdict(True, .0, "", confident=False, kind="nobody"),
    voiceprint.Verdict(True, .0, "", confident=False, kind="unreadable"),
])
def test_room_recording_never_labels_uncertain_or_unknown_audio_as_a_person(tmp_path, verdict):
    import json

    service = _conversation_recording(tmp_path, identify=lambda *_: verdict)
    assert service.start(consent=True).ok
    service.feed(pcm("owner", seconds=.5), captured_at=100.5)
    service.stop()
    row = json.loads((service._session.path / "transcript.jsonl").read_text(encoding="utf-8").strip())
    assert row["speaker"] == "Unknown speaker" and row["speaker_confidence"] == 0


def test_room_recording_never_resumes_saved_consent_on_restart(tmp_path):
    service = _conversation_recording(tmp_path)
    assert service.start(consent=True).ok
    service.stop()
    restarted = _conversation_recording(tmp_path)
    assert restarted.status().receipt["state"] == "off"
    assert not restarted.active and not restarted.record_only
    assert not restarted.start().ok
    assert len(restarted.list_sessions().receipt["recordings"]) == 1


def test_room_recording_retains_audio_when_local_transcription_is_unavailable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orion_core import speech_offline

    service = _conversation_recording(tmp_path, transcribe=None)
    monkeypatch.setattr(speech_offline, "shared", lambda: SimpleNamespace(available=False))
    assert service.start(consent=True).ok
    service.feed(pcm("owner", seconds=.5), captured_at=100.5)
    result = service.stop()
    assert result.receipt["audio_seconds"] == .5
    assert result.receipt["transcribed_segments"] == 0
    assert result.receipt["transcription_status"] == "unavailable"
    assert "No local transcription engine" in result.text
    assert "No transcript text" in service.recall().text


def test_room_transcription_bypasses_interaction_gate_without_updating_it(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orion_core import speech_offline

    calls = []
    fake = SimpleNamespace(available=True, transcribe_pcm=lambda *a, **kw: calls.append(kw) or "room words")
    monkeypatch.setattr(speech_offline, "shared", lambda: fake)
    service = _conversation_recording(tmp_path, transcribe=None)
    assert service.start(consent=True).ok
    service.feed(pcm("other", seconds=.5), captured_at=100.5)
    service.stop()
    assert calls == [{"identity_checked": True}]
    assert "room words" in service.recall().text


def test_room_recording_identity_failure_does_not_lose_available_words(tmp_path):
    def unavailable(*_):
        raise RuntimeError("synthetic identity outage")

    service = _conversation_recording(tmp_path, identify=unavailable)
    assert service.start(consent=True).ok
    service.feed(pcm("owner", seconds=.5), captured_at=100.5)
    service.stop()
    result = service.recall()
    assert "Unknown speaker" in result.text and "Synthetic recorded words" in result.text


def test_room_recording_forget_refuses_when_recycle_bin_cannot_preserve_evidence(tmp_path, monkeypatch):
    from orion_core import recycle_bin

    service = _conversation_recording(tmp_path)
    assert service.start(consent=True).ok
    session_id = service.snapshot()["id"]
    assert not service.forget(session_id).ok
    service.stop()
    monkeypatch.setattr(recycle_bin, "refusal", lambda path: "bin disabled")
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", lambda path: pytest.fail("refused recycle ran"))
    result = service.forget(session_id)
    assert not result.ok and "bin disabled" in result.text
    assert (service.root / session_id / "recording.wav").exists()
    assert not service.forget("../outside").ok


def test_room_recording_forget_recycles_only_the_selected_finished_session(tmp_path, monkeypatch):
    from orion_core import recycle_bin

    service = _conversation_recording(tmp_path)
    assert service.start(consent=True).ok
    service.stop()
    selected = service._session.path
    assert service.start(consent=True).ok
    service.stop()
    kept = service._session.path
    targets = []
    monkeypatch.setattr(recycle_bin, "refusal", lambda path: None)
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", lambda path: targets.append(path) or True)
    result = service.forget(selected.name)
    assert result.ok and targets == [selected.resolve()]
    assert kept.exists()


def test_room_recording_anonymous_voice_numbers_are_tentative_and_session_only(tmp_path):
    import json

    vectors = iter([np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                    np.array([.99, .01, 0.]), np.array([.7, .7, 0.])])
    service = _conversation_recording(tmp_path,
        identify=lambda *_: voiceprint.Verdict(True, kind="nobody"),
        embed=lambda *_: next(vectors))
    assert service.start(consent=True).ok
    at = 100.0
    for _ in range(4):
        for _ in range(3):
            at += .4
            assert service.feed(b"\x00\x10" * 6400, at)
        at += .7
        assert service.feed(b"\x00\x00" * 11200, at)
    result = service.stop()
    rows = [json.loads(line) for line in (service._session.path / "transcript.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [row["speaker"] for row in rows] == ["Speaker 1", "Speaker 2", "Speaker 1", "Unknown speaker"]
    assert rows[0]["speaker_source"] == "tentative session voice cluster"
    assert result.receipt["anonymous_speakers"] == 2
    assert not service._session.anonymous
    assert "embedding" not in (service._session.path / "manifest.json").read_text(encoding="utf-8")
    assert service.start(consent=True).ok
    assert not service._session.anonymous
    service.close()


def test_room_recording_uncertain_enrolled_match_never_becomes_anonymous_person(tmp_path):
    service = _conversation_recording(tmp_path,
        identify=lambda *_: voiceprint.Verdict(True, .65, "PossibleOwner", kind="uncertain"),
        embed=lambda *_: np.array([1., 0., 0.]))
    assert service.start(consent=True).ok
    for i in range(3):
        assert service.feed(b"\x00\x10" * 6400, 100.4 + .4 * i)
    service.stop()
    assert service.snapshot()["anonymous_speakers"] == 0
    assert "Unknown speaker" in service.recall().text


def test_room_recording_mixed_long_segment_does_not_inherit_one_persons_name(tmp_path, monkeypatch):
    from types import SimpleNamespace

    service = _conversation_recording(tmp_path, identify=None)
    vectors = iter([np.array([1., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])])
    monkeypatch.setattr(voiceprint, "embed", lambda *_: next(vectors))
    monkeypatch.setattr(voiceprint, "store", lambda: SimpleNamespace(verdict_for=lambda *_:
        voiceprint.Verdict(True, .91, "SyntheticOwner", confident=True, kind="owner")))
    text, status, verdict, vector = service._read_audio(b"\x00\x10" * (SR * 3))
    assert text == "Synthetic recorded words" and status == "local"
    assert verdict is None and vector is None


def test_room_recording_closed_shutdown_service_cannot_start_again(tmp_path):
    service = _conversation_recording(tmp_path)
    service.close()
    assert not service.start(consent=True).ok
    assert not service.root.exists()


def test_room_recording_thread_start_failure_stays_closed_and_retains_failed_manifest(tmp_path, monkeypatch):
    import json
    import threading

    service = _conversation_recording(tmp_path)
    real_start = threading.Thread.start

    def unavailable(thread):
        if thread.name.startswith("orion-recording-"):
            raise RuntimeError("synthetic thread-start fault")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", unavailable)
    assert not service.start(consent=True).ok
    assert not service.active
    assert service._session.analysed.is_set()
    manifest = json.loads((service._session.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "failed" and manifest["errors"]
    service.close()
