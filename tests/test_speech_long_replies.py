"""The fallback voice must finish long answers and retain queue controls."""

from queue import Empty
from threading import Event, Thread

from orion_core.audio import SpeechSynthesiser


class _Signal:
    def connect(self, _callback):
        pass


class _Bus:
    emotion_changed = _Signal()


def _queued(synth):
    parts = []
    while True:
        try:
            parts.append(synth.queue.get_nowait())
        except Empty:
            return parts


def test_long_reply_is_queued_in_order_through_its_final_words():
    synth = SpeechSynthesiser(_Bus())
    reply = "".join(f"Section {index} explains the result. " for index in range(260))
    reply += "The final conclusion is preserved."

    synth.speak(reply)

    parts = _queued(synth)
    assert len(parts) > 1
    assert all(len(part) <= synth.MAX_UTTERANCE_CHARS for part in parts)
    assert "".join(parts) == reply
    assert parts[-1].endswith("The final conclusion is preserved.")
    assert all(part.rstrip().endswith(".") for part in parts[:-1])


def test_long_unbroken_word_is_kept_and_interruption_clears_every_part():
    synth = SpeechSynthesiser(_Bus())
    reply = "x" * (synth.MAX_UTTERANCE_CHARS * 2 + 37)

    synth.speak(reply)
    assert synth.queue_depth() == 3
    assert "".join(_queued(synth)) == reply

    synth.speak(reply)
    assert synth.is_busy()
    synth.interrupt()
    assert _queued(synth) == []
    assert not synth.is_busy()


def test_interruption_during_long_reply_submission_clears_every_part():
    synth = SpeechSynthesiser(_Bus())
    paused, resume = Event(), Event()

    def slow_parts(_text):
        yield "first part"
        paused.set()
        assert resume.wait(2)
        yield "final part"

    synth._split_utterances = slow_parts
    submitter = Thread(target=synth.speak, args=("long reply",))
    submitter.start()
    assert paused.wait(2)

    canceller = Thread(target=synth.interrupt)
    canceller.start()
    assert synth._interrupted.wait(2)
    resume.set()
    submitter.join(2)
    canceller.join(2)

    assert not submitter.is_alive() and not canceller.is_alive()
    assert _queued(synth) == []


def test_cloud_sections_are_fetched_before_the_previous_section_finishes(monkeypatch):
    synth = SpeechSynthesiser(_Bus())
    requests, drains, states = [], [], []

    class Voice:
        SAMPLE_RATE = 24000

        def available(self):
            return True

        def stream_pcm(self, text, emotion, on_chunk, should_stop):
            requests.append(text)
            pcm = b"\x70\x17" * 240
            on_chunk(pcm)
            return pcm

    class Renderer:
        def enqueue(self, _pcm):
            pass

    synth._initialise_engine = lambda: None
    synth._elevenlabs = Voice()
    synth.attach_renderer(Renderer())
    synth.state_cb = states.append
    reply = "An explanation with all final words. " * 350
    synth.speak(reply)
    expected = synth.queue_depth()

    def drained(_renderer):
        drains.append(len(requests))
        synth.stop_event.set()

    monkeypatch.setattr(synth, "_await_renderer", drained)
    synth.start()
    synth.join(2)
    assert not synth.is_alive()
    assert "".join(requests) == reply.strip()
    assert len(requests) == expected and expected > 1
    assert drains == [expected], "The next section must synthesise before playback drains"
    assert states == [True, False]


def test_shared_cloud_hold_preserves_pcm_without_repeating_the_whole_sentence(monkeypatch):
    synth = SpeechSynthesiser(_Bus())
    pcm, can_continue, fetching = [], Event(), Event()
    device_stops = []

    class Voice:
        SAMPLE_RATE = 24000

        def stream_pcm(self, text, emotion, on_chunk, should_stop):
            on_chunk(b"\x70\x17" * 240)
            fetching.set()
            assert can_continue.wait(2)
            assert not should_stop(), "A hold must finish fetching preserved PCM"
            on_chunk(b"\x90\xe8" * 240)
            return b"done"

    class Renderer:
        def enqueue(self, chunk):
            pcm.append(chunk)

    synth._elevenlabs = Voice()
    synth.attach_renderer(Renderer())
    synth._active_backend = "elevenlabs"
    synth._speaking.set()
    monkeypatch.setattr("orion_core.audio.sd.stop", lambda: device_stops.append(True))
    producer = Thread(target=synth._speak_elevenlabs, args=("The whole sentence.",))
    producer.start()
    assert fetching.wait(2)
    synth.hold()
    can_continue.set()
    producer.join(2)
    assert not producer.is_alive()
    assert synth._resume_text == ""
    assert b"".join(pcm) == b"\x70\x17" * 240 + b"\x90\xe8" * 240
    assert device_stops == []
