"""ORION hears who is speaking on the LIVE path too, not only the offline one.

``voice_presence`` — the enrolled voiceprint, the tone reading and the
only-my-voice gate — was fed from exactly one place: ``speech_offline``, which
is handed a whole utterance by an upstream VAD. The Gemini Live path has no
such moment. It forwards every chunk, voice and silence alike, because the
server's VAD is what finds end-of-turn; filtering to voiced chunks is what once
made the live channel deaf.

So in a live session ``voice_tone`` answered from a stale reading or nothing at
all, the enrolled voiceprint was never compared against anything, and the gate
could not apply. Half of ORION's hearing worked only in the half of the
pipeline most conversations do not take.

These drive the assembly with bytes. No microphone, no ONNX session, no
QApplication — the point is that the seam exists and behaves, and the pieces
either side of it are already covered by their own tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import live_presence  # noqa: E402
from orion_core.live_presence import LivePresenceBridge, LiveUtterance  # noqa: E402

RATE = live_presence.SAMPLE_RATE


def chunk(seconds: float) -> bytes:
    """Silence of a given length, in the format the gate captures."""
    return b"\x01\x00" * int(RATE * seconds)


# ── assembling an utterance ──────────────────────────────────────────────────

def test_chunks_accumulate_into_one_utterance():
    utterance = LiveUtterance()
    for _ in range(4):
        utterance.feed(chunk(0.5))
    assert utterance.seconds == pytest.approx(2.0, abs=0.01)
    assert len(utterance.audio()) == len(chunk(2.0))


def test_a_finished_utterance_is_handed_over_whole_and_resets():
    utterance = LiveUtterance()
    utterance.feed(chunk(2.0))
    assert len(utterance.end()) == len(chunk(2.0))
    assert utterance.seconds == 0.0, "the next utterance inherited this one"


def test_something_too_short_to_read_is_discarded():
    """A cough, a chair, one word of agreement. Reading a speaker from a
    quarter of a second produces a noisy embedding, and acting on it is worse
    than not acting."""
    utterance = LiveUtterance()
    utterance.feed(chunk(0.3))
    assert utterance.end() == b""


def test_a_monologue_is_capped_rather_than_accumulated():
    """Five minutes without a pause should not hold five minutes of PCM. The
    encoder learns nothing in the fourth minute it did not know in the
    second."""
    utterance = LiveUtterance(max_seconds=3.0)
    for _ in range(20):
        utterance.feed(chunk(1.0))
    assert utterance.seconds <= 3.0 + 1.0
    assert utterance.truncated is True


def test_the_owner_is_asked_about_once_per_utterance():
    """Once, part-way through — not on every chunk after the threshold, which
    would start an ONNX inference thirty times a second."""
    utterance = LiveUtterance(decide_after=1.0)
    utterance.feed(chunk(0.5))
    assert utterance.ready_to_decide() is False, "asked before anyone spoke"
    utterance.feed(chunk(0.8))
    assert utterance.ready_to_decide() is True
    assert utterance.ready_to_decide() is False, "asked twice for one utterance"


def test_ending_an_utterance_clears_the_mute_and_the_question():
    """Each utterance is judged on its own. A stranger speaking once must not
    silence whoever talks next."""
    utterance = LiveUtterance(decide_after=1.0)
    utterance.feed(chunk(1.5))
    utterance.ready_to_decide()
    utterance.muted = True
    utterance.end()
    assert utterance.muted is False
    assert utterance.asked is False


# ── handing it off ───────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.seen = []

    def observe_pcm(self, pcm, rate):
        self.seen.append((len(pcm), rate))


def test_a_finished_utterance_reaches_voice_presence():
    """The whole point: a live session updates the same reader the offline
    transcriber updates."""
    presence = Recorder()
    work = []
    bridge = LivePresenceBridge(presence=presence, schedule=work.append)
    bridge.observe(chunk(2.0))
    assert len(work) == 1, "nothing was scheduled"
    work[0]()
    assert presence.seen == [(len(chunk(2.0)), RATE)]


def test_nothing_is_run_on_the_calling_thread():
    """The capture thread has a 32 ms deadline. A speaker embedding is 13 ms
    of it, and the pitch tracking more, so both belong somewhere else."""
    presence = Recorder()
    bridge = LivePresenceBridge(presence=presence, schedule=lambda work: None)
    bridge.observe(chunk(2.0))
    assert presence.seen == [], "the work ran inline on the capture thread"


def test_no_presence_reader_is_survivable():
    bridge = LivePresenceBridge(presence=None, schedule=lambda work: None)
    bridge.observe(chunk(2.0))          # must not raise


def test_a_scheduler_that_throws_never_reaches_the_capture_thread():
    """An exception here costs ORION his hearing, and a tone reading is not
    worth that."""
    logged = []

    def broken(work):
        raise RuntimeError("the loop is gone")

    bridge = LivePresenceBridge(presence=Recorder(), schedule=broken,
                                log=logged.append)
    bridge.observe(chunk(2.0))          # must not raise
    assert logged and "presence" in logged[0]


# ── it is actually wired in ──────────────────────────────────────────────────

AUDIO = (ROOT / "orion_core" / "audio.py").read_text(encoding="utf-8")
WORKER = (ROOT / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
APP = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")


def test_the_capture_loop_assembles_the_utterance():
    assert "presence.utterance.feed(chunk)" in AUDIO, (
        "the live path still has no complete utterance to read")


def test_the_capture_loop_survives_having_no_presence_reader():
    """Reading who is speaking is a nice-to-have. Hearing "ORION resume"
    while paused is not, and the first must never take out the second.

    Caught by test_pause_hearing, which builds a gate with __new__ and sets
    only the fields the capture loop touches — a reasonable test of the
    capture loop, which should not have to know this was added later.
    """
    assert "\n    _presence = None" in AUDIO, (
        "there is no class-level default, so an instance built by __new__ "
        "raises in the capture loop")
    body = AUDIO[AUDIO.index("presence = self._presence"):]
    body = body[:body.index("command_listening")]
    assert "presence is not None" in body


def test_it_is_read_on_the_same_terms_as_the_gender_tracker():
    """Voiced, not ORION's own output, past the echo guard, capture allowed.

    Feeding it ORION's own voice would enrol him as a speaker and then
    recognise him as one.
    """
    assert "own_voice = (" in AUDIO
    body = AUDIO[AUDIO.index("own_voice = ("):]
    body = body[:body.index("if own_voice and self.speaker_tracker")]
    for condition in ("voiced", "not speaking", "_past_echo_guard",
                      "can_capture()"):
        assert condition in body, f"{condition} is no longer required"


def test_a_muted_utterance_is_not_forwarded_to_the_model():
    """Otherwise the gate records a verdict and changes nothing, which is
    exactly the state this replaced."""
    guard = "self._presence is not None and self._presence.utterance.muted"
    assert guard in AUDIO
    assert AUDIO.index(guard) < AUDIO.index('media = {"data": chunk')


def test_only_a_confident_miss_mutes_anything():
    """Every uncertain case already resolves to listening inside voiceprint.
    A second place that can decide to go deaf is a second place to get it
    wrong."""
    body = AUDIO[AUDIO.index("def _on_speaker_verdict"):]
    body = body[:body.index("\n    def ", 10)]
    assert 'getattr(verdict, "is_owner", True)' in body, (
        "an unreadable verdict would silence him")


def test_the_worker_carries_it_to_both_microphone_engines():
    """There are two — the reconnect path builds its own — and a reader
    attached to one of them works until the first reconnection."""
    assert WORKER.count("voice_presence=self.voice_presence,") == 2


def test_app_hands_the_reader_to_the_worker():
    assert "worker.voice_presence = voice_presence" in APP
    assert APP.index("voice_presence = VoicePresence") < APP.index(
        "worker.voice_presence = voice_presence"), (
        "the worker is handed the reader before it exists")


def test_the_offline_path_still_has_its_own():
    """This adds a second feeder; it must not have moved the first."""
    assert "offline_stt.presence = voice_presence" in APP
