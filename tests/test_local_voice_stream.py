"""
Local brain, stage 2: the streaming local voice loop.

The local agent's final answer streams token by token; each sentence goes to
the voice as soon as it exists; barge-in and cancel stop the generation and
flush the queued speech; the offline microphone ends an utterance on a short
silence; and a dry benchmark measures the loop.

Fakes only: a scripted streaming model (on_delta fed word by word), fake
speech managers and synthesis voices, a fake transcriber. Nothing here starts
Ollama, ORION, a microphone or the speakers.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(autouse=True)
def _streaming_default(monkeypatch):
    """Every test runs with streaming at its default (on), whatever the
    developer's own environment says; one test switches it off itself."""
    monkeypatch.delenv("ORION_LOCAL_STREAMING", raising=False)


from orion_core import brain_mode  # noqa: E402
from orion_core import provider_transport as pt  # noqa: E402
from orion_core import voice_stream as vs  # noqa: E402
from orion_core.audio import (  # noqa: E402
    AudioGateThread, AudioPlaybackThread, SpeechQueueManager, SpeechSynthesiser)
from orion_core.audio_state import AudioStateMachine  # noqa: E402
from orion_core.constants import VOICE_HANGOVER_SECONDS  # noqa: E402
from orion_core.data import ToolResult  # noqa: E402
from orion_core.live_worker import GenAILiveWorker  # noqa: E402
from orion_core.local_agent import AgentReply, LocalAgent  # noqa: E402
from orion_core.voice_interrupt import VoiceInterruptManager  # noqa: E402

# ── fakes ───────────────────────────────────────────────────────────────────

DECLS = [
    {"name": "open_app", "description": "Launch a host application.",
     "parameters": {"type": "OBJECT", "properties": {"app_name": {"type": "STRING"}},
                    "required": ["app_name"]}},
    {"name": "resource_status", "description": "Report CPU and memory.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "shutdown_orion", "description": "Shut ORION down.",
     "parameters": {"type": "OBJECT", "properties": {}}},
]


def _scorer(text, declarations):
    return {d["name"]: 10.0 - i * 0.1 for i, d in enumerate(declarations)}


class StreamChat:
    """A scripted model that STREAMS: each text reply is fed to on_delta word
    by word, yielding to the loop between words (as NDJSON chunks arrive);
    a ChatResult reply (tool calls) comes back whole. A cancelled call is
    counted as a closed stream - what stops Ollama generating."""

    def __init__(self, *replies, delay: float = 0.0, hang_after: int | None = None):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.closed = 0
        self.sent = 0
        self.total = 0
        self.delay = delay
        self.hang_after = hang_after

    async def __call__(self, messages, tools, **kwargs):
        self.calls.append({"tools": [t["function"]["name"] for t in tools],
                           "streamed": "on_delta" in kwargs,
                           **{k: v for k, v in kwargs.items() if k != "on_delta"}})
        reply = self.replies.pop(0) if self.replies else "Nothing more to add."
        if isinstance(reply, pt.ChatResult):
            return reply
        on_delta = kwargs.get("on_delta")
        pieces = re.findall(r"\S+\s*", reply)
        self.sent, self.total = 0, len(pieces)
        try:
            for index, piece in enumerate(pieces):
                if self.hang_after is not None and index >= self.hang_after:
                    await asyncio.Event().wait()          # generation in progress...
                self.sent = index + 1
                if on_delta is not None:
                    on_delta(piece)
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.closed += 1
            raise
        return pt.ChatResult(text=reply, first_token_ms=1.0)


def _call(tool_name, **args):
    return pt.ChatResult(text="", tool_calls=[{"name": tool_name, "args": args, "id": ""}],
                         first_token_ms=1.0)


class FakeDispatch:
    def __init__(self, results=None):
        self.calls: list[tuple[str, dict]] = []
        self.results = results or {}

    async def __call__(self, name, args):
        self.calls.append((name, dict(args)))
        result = self.results.get(name)
        return result or ToolResult(f"[{name}] {name} done.")


def _agent(chat, dispatch=None, **kw):
    return LocalAgent(None, None, chat=chat, dispatch=dispatch or FakeDispatch(),
                      declarations=lambda: DECLS, scorer=_scorer, **kw)


class Ear:
    """A speak callback that also notes how far generation had got."""

    def __init__(self, chat=None):
        self.chat = chat
        self.chunks: list[str] = []
        self.at: list[int] = []

    def __call__(self, chunk):
        self.chunks.append(chunk)
        self.at.append(getattr(self.chat, "sent", 0))


def run(coro):
    return asyncio.run(coro)


# ── 1. sentence splitting ───────────────────────────────────────────────────

def test_a_sentence_ends_only_once_the_next_character_shows_it_has():
    splitter = vs.SentenceSplitter()
    assert splitter.feed("It costs 3.") == []          # "3." may still become "3.5"
    assert splitter.feed("5 pounds today, I think.") == []
    assert splitter.feed(" Next") == ["It costs 3.5 pounds today, I think."]
    assert splitter.flush() == ["Next"]


def test_abbreviations_initials_and_dotted_names_do_not_end_a_sentence():
    chunks = vs.split_text("Dr. Smith met J. R. Tolkien in the U.S. last year. "
                           "I am O.R.I.O.N. at your service.")
    assert chunks == ["Dr. Smith met J. R. Tolkien in the U.S. last year.",
                      "I am O.R.I.O.N. at your service."]


def test_a_long_run_is_cut_at_a_clause_after_about_twelve_words():
    splitter = vs.SentenceSplitter()
    out = []
    for word in ("The forecast for London tomorrow is mostly cloudy, with light rain in "
                 "the afternoon and a cold breeze from the north").split():
        out += splitter.feed(word + " ")
    assert out == ["The forecast for London tomorrow is mostly cloudy,"]
    assert splitter.flush() == ["with light rain in the afternoon and a cold breeze from the north"]


def test_a_run_with_no_clause_is_cut_between_whole_words():
    splitter = vs.SentenceSplitter()
    out = []
    words = [f"word{i}" for i in range(30)]
    for word in words:
        out += splitter.feed(word + " ")
    assert out and all(chunk.split()[-1] in words for chunk in out)
    assert " ".join(out + splitter.flush()).split() == words


def test_a_short_opening_is_held_for_the_sentence_after_it():
    assert vs.split_text("Sure. I've opened Notepad for you.") == [
        "Sure. I've opened Notepad for you."]
    assert vs.split_text("Yes.") == ["Yes."]             # alone, it is still said


def test_chunks_are_cleaned_and_keep_every_word_in_order():
    text = "**Right**, here it is: `notepad` is open.\nAnything else? <|eot_id|>"
    chunks = vs.split_text(text)
    assert "*" not in " ".join(chunks) and "`" not in " ".join(chunks)
    assert "<|" not in " ".join(chunks)
    assert " ".join(chunks).split()[:3] == ["Right,", "here", "it"]


def test_a_call_written_as_text_is_recognised():
    assert vs.looks_like_call('{"name": "open_app"')
    assert vs.looks_like_call("  ```json")
    assert vs.looks_like_call("<|python_tag|>")
    assert not vs.looks_like_call("Opening it now.")


# ── 2. the agent streams its answer ─────────────────────────────────────────

def test_the_answer_streams_in_order_and_starts_before_generation_ends():
    chat = StreamChat("Hello there, it is good to hear from you. The time is three o'clock. "
                      "Anything else I can do?")
    ear = Ear(chat)
    reply = run(_agent(chat).respond("hello", speak=ear))
    assert ear.chunks == ["Hello there, it is good to hear from you.",
                          "The time is three o'clock.", "Anything else I can do?"]
    assert ear.at[0] < chat.total, "the first sentence was spoken while generation went on"
    assert reply.spoken and reply.handled and reply.text == " ".join(ear.chunks)
    assert chat.calls[0]["streamed"] is True
    assert reply.timings["first_chunk_ms"] <= reply.timings["total_ms"]


def test_without_a_speak_callback_nothing_streams_and_the_reply_is_unchanged():
    chat = StreamChat("Hello there, it is good to hear from you.")
    reply = run(_agent(chat).respond("hello"))
    assert reply.text == "Hello there, it is good to hear from you."
    assert not reply.spoken and chat.calls[0]["streamed"] is False


def test_a_claim_with_no_tool_behind_it_is_never_spoken_and_stops_generation():
    chat = StreamChat("Sure. I've opened Notepad for you. It is ready now.")
    ear = Ear(chat)
    reply = run(_agent(chat).respond("open notepad", speak=ear))
    assert "opened" not in " ".join(ear.chunks).lower()
    assert ear.chunks and "haven't" in ear.chunks[0]
    assert chat.closed == 1, "the stream was closed: generation stopped"
    assert reply.spoken and reply.text == " ".join(ear.chunks)


def test_in_hybrid_a_claim_escalates_with_nothing_said():
    chat = StreamChat("I've opened Notepad for you.")
    ear = Ear(chat)
    reply = run(_agent(chat).respond("open notepad", speak=ear, allow_escalate=True))
    assert reply.escalate and not reply.handled and ear.chunks == []


def test_a_later_sentence_that_claims_an_action_is_dropped_and_corrected():
    chat = StreamChat("It looks dry and bright in London this afternoon. "
                      "I've set a reminder to take your umbrella anyway. Enjoy it.")
    ear = Ear(chat)
    reply = run(_agent(chat).respond("what's the weather like", speak=ear))
    assert ear.chunks[0] == "It looks dry and bright in London this afternoon."
    assert not any("set a reminder" in chunk for chunk in ear.chunks)
    assert "haven't actually done that" in ear.chunks[-1]
    assert chat.closed == 1 and reply.spoken


def test_an_unsure_opening_escalates_in_hybrid_before_anything_is_said():
    chat = StreamChat("I'm not sure about that, but it might be Tuesday. Or Wednesday.")
    ear = Ear(chat)
    reply = run(_agent(chat).respond("is the bin collected today", speak=ear,
                                     allow_escalate=True))
    assert reply.escalate and ear.chunks == [] and chat.closed == 1


def test_an_action_speaks_its_own_line_through_the_voice():
    chat = StreamChat(_call("open_app", app_name="notepad"))
    dispatch = FakeDispatch({"open_app": ToolResult("[open_app] Opening Notepad.")})
    ear = Ear(chat)
    reply = run(_agent(chat, dispatch).respond("open notepad", speak=ear))
    assert dispatch.calls == [("open_app", {"app_name": "notepad"})]
    assert ear.chunks == ["Opening Notepad."] and reply.spoken


def test_a_lookup_streams_the_answer_after_the_tool_and_claims_are_backed():
    chat = StreamChat(_call("resource_status"),
                      "I've checked: your CPU is at twenty three percent. Memory is fine.")
    dispatch = FakeDispatch({"resource_status": ToolResult("CPU 23%\nRAM 61%")})
    ear = Ear(chat)
    reply = run(_agent(chat, dispatch).respond("how busy is my pc", speak=ear))
    assert ear.chunks == ["I've checked: your CPU is at twenty three percent.", "Memory is fine."]
    assert chat.calls[1]["streamed"] is True and reply.spoken


def test_after_every_tool_failed_the_model_text_is_not_streamed():
    chat = StreamChat(_call("resource_status"), "Your PC is perfectly healthy right now.")
    dispatch = FakeDispatch({"resource_status": ToolResult("sensor offline", ok=False)})
    ear = Ear(chat)
    reply = run(_agent(chat, dispatch).respond("how busy is my pc", speak=ear))
    assert chat.calls[1]["streamed"] is False
    assert ear.chunks and ear.chunks[0].startswith("I couldn't do that.")
    assert "healthy" not in reply.text


def test_a_high_risk_call_still_asks_first_and_the_question_is_spoken():
    chat = StreamChat(_call("shutdown_orion"))
    dispatch = FakeDispatch()
    ear = Ear(chat)
    agent = _agent(chat, dispatch)
    reply = run(agent.respond("shut yourself down", speak=ear))
    assert reply.pending and agent.pending is not None and dispatch.calls == []
    assert " ".join(ear.chunks) == reply.text and "Shall I go ahead?" in reply.text


def test_a_call_written_as_json_text_is_never_spoken():
    chat = StreamChat('{"name": "open_app", "parameters": {"app_name": "notepad"}}')
    dispatch = FakeDispatch({"open_app": ToolResult("[open_app] Opening Notepad.")})
    ear = Ear(chat)
    reply = run(_agent(chat, dispatch).respond("open notepad", speak=ear))
    assert dispatch.calls == [("open_app", {"app_name": "notepad"})]
    assert ear.chunks == ["Opening Notepad."] and "{" not in reply.text


def test_cancelling_the_turn_closes_the_stream_and_nothing_more_is_said():
    chat = StreamChat("First sentence of a long answer here. Second sentence that "
                      "never comes.", hang_after=8)
    ear = Ear(chat)

    async def flow():
        task = asyncio.ensure_future(_agent(chat).respond("tell me", speak=ear))
        # The shortlist runs on a worker thread, so a fixed number of bare
        # yields can run out under load before the first sentence exists.
        # The stream hangs after that sentence, so waiting on a clock is safe.
        deadline = time.monotonic() + 10.0
        while not ear.chunks and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(flow())
    assert ear.chunks == ["First sentence of a long answer here."]
    assert chat.closed == 1


def test_a_cancelled_confirmation_notice_is_said_first():
    chat = StreamChat(_call("shutdown_orion"), "The weather looks lovely and bright today.")
    agent = _agent(chat)
    run(agent.respond("shut down", speak=Ear()))
    ear = Ear(chat)
    reply = run(agent.respond("what's the weather like", speak=ear))
    assert ear.chunks[0].startswith("I've cancelled the earlier request")
    assert ear.chunks[1] == "The weather looks lovely and bright today."
    assert reply.text == " ".join(ear.chunks)


def test_warming_asks_for_one_token_with_the_turns_own_prompt():
    chat = StreamChat("Hi.")
    clock = SimpleNamespace(t=1000.0)
    agent = _agent(chat, keep_alive="10m", clock=lambda: clock.t)
    assert agent.warm_due()
    result = run(agent.warm())
    assert result["ok"] and chat.calls[0]["max_tokens"] == 1 and chat.calls[0]["tools"] == []
    assert chat.calls[0]["keep_alive"] == "10m"
    assert not agent.warm_due()
    clock.t += 580.0                                      # keep_alive (600 s) nearly out
    assert agent.warm_due()


def test_the_router_hands_streamed_text_to_on_delta(monkeypatch):
    from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter

    class Content:
        def __init__(self, lines):
            self.lines = [json.dumps(line).encode() + b"\n" for line in lines]

        def __aiter__(self):
            async def gen():
                for line in self.lines:
                    yield line
            return gen()

        async def read(self, _n=-1):
            return b""

    class Response:
        status = 200
        headers = {"Content-Type": "application/x-ndjson"}

        def __init__(self, lines):
            self.content = Content(lines)

        def close(self):
            pass

        def release(self):
            pass

    profile = AIProviderProfile(name="local_ollama", kind="openai_compatible",
                                model="llama3.1:latest", base_url="http://127.0.0.1:11434/v1",
                                api_key="local", enabled=True)
    settings = OrionProviderSettings(active_provider="local_ollama",
                                     provider_order=["local_ollama"],
                                     providers={"local_ollama": profile})
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None),
                          dashboard_event=SimpleNamespace(emit=lambda *_a: None))
    router = ProviderRouter(settings, bus, memory=SimpleNamespace(prompt_context=lambda *_: ""))

    class Session:
        async def request(self, method, url, **kwargs):
            return Response([{"message": {"content": "Hello "}, "done": False},
                             {"message": {"content": "there."}, "done": False},
                             {"message": {"content": ""}, "done": True, "done_reason": "stop"}])

    async def is_ollama(root, scope=None):
        return True

    async def pool_get():
        return Session()

    monkeypatch.setattr(router, "_is_ollama", is_ollama)
    monkeypatch.setattr(router._pool, "get", pool_get)
    pieces = []

    async def flow():
        try:
            return await router.local_tool_chat([{"role": "user", "content": "hi"}], [],
                                                deadline_s=10.0, on_delta=pieces.append)
        finally:
            await router.aclose()

    _profile, result = run(flow())
    assert pieces == ["Hello ", "there."] and result.text == "Hello there."


# ── 3. the voice: one batch per streamed reply, the earcon ──────────────────

class _Signal:
    def emit(self, *_a):
        pass

    def connect(self, *_a):
        pass


class _AudioBus:
    def __getattr__(self, name):
        signal = _Signal()
        setattr(self, name, signal)
        return signal


def _speech_manager():
    bus = _AudioBus()
    machine = AudioStateMachine(bus)
    playback = AudioPlaybackThread(bus, machine)
    tts = SpeechSynthesiser(bus, machine)
    tts._initialise_engine = lambda: None
    tts._start_prewarm = lambda: None
    tts._start_kokoro_watch = lambda: None
    return SpeechQueueManager(bus, playback, tts, machine), playback, tts, machine


def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_an_open_stream_hands_over_queued_late_and_no_more_sentences():
    _manager, _playback, tts, _machine = _speech_manager()
    token = tts.open_stream()
    tts.speak("Second sentence.")
    assert tts._next_streamed() == "Second sentence."
    threading.Timer(0.05, tts.speak, args=("Third sentence.",)).start()
    assert tts._next_streamed() == "Third sentence."
    tts.close_stream(token)
    started = time.monotonic()
    assert tts._next_streamed() is None and time.monotonic() - started < 1.0
    tts.open_stream()
    tts.interrupt()
    assert not tts.streaming(), "an interruption ends every stream"


def _batches_for_two_sentences(stream: bool) -> tuple[int, list[str]]:
    _manager, _playback, tts, machine = _speech_manager()
    starts: list[int] = []
    original = machine.tts_started
    machine.tts_started = lambda: (starts.append(1), original())
    spoken: list[str] = []
    tts._speak_one = spoken.append
    tts.acquire_output = lambda: True
    tts.release_output = lambda: None
    token = tts.open_stream() if stream else None
    tts.speak("First sentence.")
    tts.start()
    try:
        assert _wait(lambda: spoken == ["First sentence."])
        time.sleep(0.3)                     # the model still writing sentence two
        tts.speak("Second sentence.")
        assert _wait(lambda: len(spoken) == 2)
        if token is not None:
            tts.close_stream(token)
        assert _wait(lambda: not tts.is_busy())
    finally:
        tts.stop_event.set()
        tts.queue.put_nowait(None)
        tts._utterance_ready.set()
        tts.join(2)
    return len(starts), spoken


def test_a_streamed_reply_is_one_utterance_not_one_per_sentence():
    batches, spoken = _batches_for_two_sentences(stream=True)
    assert spoken == ["First sentence.", "Second sentence."] and batches == 1
    batches, _spoken = _batches_for_two_sentences(stream=False)
    assert batches == 2                      # what happened without a stream


def test_the_earcon_is_soft_and_never_plays_over_speech():
    from array import array
    pcm = SpeechQueueManager.earcon_pcm()
    samples = array("h")
    samples.frombytes(pcm)
    assert 0.2 < len(samples) / 24000 < 0.4
    assert max(abs(s) for s in samples) <= int(32767 * SpeechQueueManager.EARCON_LEVEL) + 1
    assert abs(samples[0]) < 200 and abs(samples[-1]) < 200          # faded, no click
    manager, playback, _tts, machine = _speech_manager()
    machine.tts_started()
    assert manager.play_earcon() is False
    machine.tts_stopped()
    manager.muted = True
    assert manager.play_earcon() is False
    manager.muted = False
    assert manager.play_earcon() is True and not playback.queue.empty()


# ── 4. the spoken reply object ──────────────────────────────────────────────

class _Loop:
    """call_later that runs nothing until fired by hand."""

    def __init__(self):
        self.timers = []

    def call_later(self, delay, callback):
        handle = SimpleNamespace(delay=delay, callback=callback, cancelled=False)
        handle.cancel = lambda: setattr(handle, "cancelled", True)
        self.timers.append(handle)
        return handle

    def fire(self):
        for handle in list(self.timers):
            if not handle.cancelled:
                handle.cancelled = True
                handle.callback()


def test_a_reply_holds_while_the_owner_talks_and_releases_in_order():
    said, loop = [], _Loop()
    reply = vs.SpokenReply(speak=said.append, loop=loop, earcon_after=0).start()
    assert reply.hold()
    reply.say("One sentence here.")
    reply.say("Another one here.")
    assert said == []
    reply.close()                                         # the agent finished meanwhile
    assert not reply.closed, "held words keep the reply open"
    assert reply.release() == 2 and said == ["One sentence here.", "Another one here."]
    assert reply.closed


def test_a_cancelled_or_superseded_reply_says_nothing_more():
    said, live = [], {"ok": True}
    reply = vs.SpokenReply(speak=said.append, live=lambda: live["ok"], loop=_Loop(),
                           earcon_after=0).start()
    reply.say("Spoken.")
    live["ok"] = False                                     # the turn's epoch moved on
    reply.say("Dropped.")
    reply.hold()
    reply.cancel()
    assert said == ["Spoken."] and reply.release() == 0


def test_slow_first_words_get_the_earcon_once_and_fast_ones_get_none():
    loop, earcons = _Loop(), []
    slow = vs.SpokenReply(speak=lambda _t: None, earcon=lambda: earcons.append(1),
                          loop=loop, earcon_after=2.0, filler="earcon").start()
    assert loop.timers[0].delay == 2.0
    loop.fire()
    assert earcons == [1] and slow.signalled == "earcon"
    loop2, earcons2 = _Loop(), []
    fast = vs.SpokenReply(speak=lambda _t: None, earcon=lambda: earcons2.append(1),
                          loop=loop2, earcon_after=2.0).start()
    fast.say("Quick answer.")
    loop2.fire()
    assert earcons2 == []
    words, loop3 = [], _Loop()
    vs.SpokenReply(speak=words.append, loop=loop3, earcon_after=2.0, filler="words").start()
    loop3.fire()
    assert words == ["One moment."]


def test_merging_an_utterance_cut_short():
    assert vs.merge_requests("what's the weather tomorrow", "in London") == \
        "what's the weather tomorrow in London"
    assert vs.merge_requests("", "open notepad") == "open notepad"


# ── 5. endpointing ──────────────────────────────────────────────────────────

def test_the_local_endpoint_is_short_unless_the_words_plainly_continue(monkeypatch):
    assert vs.endpoint_hangover("open notepad", 0.65, 1.5) == 0.65
    assert vs.endpoint_hangover("open the", 0.65, 1.5) == 1.5
    assert vs.endpoint_hangover("check my email and", 0.65, 1.5) == 1.5
    assert vs.endpoint_hangover("", 0.65, 1.5) == 0.65
    monkeypatch.setenv("ORION_LOCAL_ENDPOINT_MS", "700")
    assert vs.local_endpoint_s() == 0.7
    monkeypatch.setenv("ORION_LOCAL_ENDPOINT_MS", "5")
    assert vs.local_endpoint_s() == 0.3


def test_the_worker_endpoint_follows_the_brain(monkeypatch):
    # The quick local endpoint is stretched by the owner's reply pause (Mark
    # XL.1); "brisk" keeps the measured 0.65 s.
    monkeypatch.setenv("ORION_REPLY_PAUSE", "brisk")
    w = GenAILiveWorker.__new__(GenAILiveWorker)
    w.recogniser = SimpleNamespace(last_words="open notepad")
    GenAILiveWorker._set_endpoint_for(w, "cloud")
    assert w._offline_endpoint_s() == VOICE_HANGOVER_SECONDS
    GenAILiveWorker._set_endpoint_for(w, "local")
    assert w._offline_endpoint_s() == pytest.approx(vs.LOCAL_ENDPOINT_S)
    w.recogniser.last_words = "remind me to"
    assert w._offline_endpoint_s() == VOICE_HANGOVER_SECONDS
    GenAILiveWorker._set_endpoint_for(w, "hybrid")
    w.recogniser.last_words = "what time is it"
    assert w._offline_endpoint_s() == pytest.approx(vs.LOCAL_ENDPOINT_S)


def test_the_gate_uses_the_callers_hangover_and_falls_back_safely():
    gate = AudioGateThread.__new__(AudioGateThread)
    assert gate._hangover() == VOICE_HANGOVER_SECONDS
    gate.hangover_s = lambda: 0.65
    assert gate._hangover() == 0.65
    gate.hangover_s = lambda: 1 / 0
    assert gate._hangover() == VOICE_HANGOVER_SECONDS


def test_the_gate_reports_sustained_owner_speech_and_its_end():
    gate = AudioGateThread.__new__(AudioGateThread)
    events = []
    gate.loop = SimpleNamespace(call_soon_threadsafe=lambda cb, *a: cb(*a))
    gate.on_speech_activity = events.append
    chunk = b"\x00\x01" * 512                              # 32 ms at 16 kHz
    for _ in range(9):
        gate._note_speech_activity(chunk, True, True, True)
    assert events == []                                    # 288 ms: a cough, not a turn
    gate._note_speech_activity(chunk, True, True, True)
    assert events == [True]
    gate._note_speech_activity(chunk, False, True, True)   # a pause between words
    assert events == [True]
    gate._note_speech_activity(chunk, False, False, True)  # the utterance ended
    assert events == [True, False]
    for _ in range(20):
        gate._note_speech_activity(chunk, True, True, False)   # ORION speaking: never
    assert events == [True, False]


def test_recogniser_partials_are_kept_for_endpointing():
    from orion_core.audio import LocalSpeechRecogniser
    rec = LocalSpeechRecogniser.__new__(LocalSpeechRecogniser)
    rec.available = True
    rec._lock = threading.RLock()
    rec.last_words = ""

    class Kaldi:
        def __init__(self):
            self.n = 0

        def AcceptWaveform(self, _chunk):
            self.n += 1
            return self.n == 3

        def PartialResult(self):
            return json.dumps({"partial": "open the" if self.n == 1 else "open the door"})

        def Result(self):
            return json.dumps({"text": "open the door please"})

    rec._recogniser = Kaldi()
    rec.feed(b"\x00\x00")
    assert rec.last_words == "open the"
    rec.feed(b"\x00\x00")
    rec.feed(b"\x00\x00")
    assert rec.last_words.startswith("open the door")


# ── 6. the live worker: streaming, barge-in, cancel, routes unchanged ───────

class FakeSpeech:
    def __init__(self):
        self.said: list[str] = []
        self.streams: list[int] = []
        self.closed: list[int] = []
        self.interrupts = 0
        self.earcons = 0
        self.muted = False

    def speak_text(self, text):
        self.said.append(text)

    def open_stream(self):
        self.streams.append(len(self.streams) + 1)
        return self.streams[-1]

    def close_stream(self, token):
        self.closed.append(token)

    def play_earcon(self):
        self.earcons += 1
        return True

    def interrupt_all(self):
        self.interrupts += 1
        return True

    def output_active(self):
        return False


class _Log:
    def __init__(self):
        self.lines: list[str] = []

    def emit(self, *args):
        self.lines.append(" ".join(str(a) for a in args))


def _worker(agent, *, mode="local", speech=None, cloud_reply="A cloud answer."):
    brain_mode.set_mode(mode)
    w = GenAILiveWorker.__new__(GenAILiveWorker)
    w.bus = SimpleNamespace(log=_Log(), banner=_Signal(), paused=_Signal(), state=_Signal())
    w.paused = w.standby_mode = False
    w.stop_event = threading.Event()
    w._fallback_epoch = 0
    w.connected = False
    w.mic = None
    w.tool_busy = False
    w._turn_active = False
    w._pending_turn = None
    w._held_tool_request = None
    w._partial_spoken_request = ""
    w._pending_turn_retries = 0
    w._turn_request_id = 0
    w.cloud_calls = []

    async def generate_text(prompt, *_a, **_k):
        w.cloud_calls.append(prompt)
        return SimpleNamespace(name="gemini"), cloud_reply

    async def rules(_text):
        return "A rules answer."

    async def known(_text):
        return None

    w.router = SimpleNamespace(is_online=lambda: True, has_text_fallback=lambda: True,
                               generate_text=generate_text)
    w.local_brain = SimpleNamespace(respond=rules, try_known_response=known)
    w.local_agent = agent
    w.session = None
    w.dispatch_calls = []

    async def dispatch_chain(name, args, **_k):
        w.dispatch_calls.append(name)
        return ToolResult(f"[{name}] Opening {args.get('app_name', 'it')}.")

    w.dispatcher = SimpleNamespace(dispatch_chain=dispatch_chain)
    w._emit_state = lambda *_: None
    w._persist_episode = lambda *_: None
    w._overheard_filler = lambda *_: False
    w._offline_voice_ready = lambda: False
    w._refresh_wake_window = lambda: None
    w.observe_for_reflex = lambda *a, **k: None
    w._last_spoken_norm, w._spoken_at = "", 0.0
    w.speech = speech or FakeSpeech()
    w.whole = []
    w._say = w.whole.append                    # the old, whole-reply voice path
    return w


async def _until(predicate, timeout=3.0):
    # The shortlist runs in a worker thread, so progress takes real time.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.002)
    assert predicate(), "expected async progress did not occur"


def test_a_local_voice_turn_reaches_the_voice_sentence_by_sentence_once():
    chat = StreamChat("Good afternoon, it is nice to hear from you. How can I help today?")
    w = _worker(_agent(chat))
    run(GenAILiveWorker._run_text_fallback(w, "hello"))
    assert w.speech.said == ["Good afternoon, it is nice to hear from you.",
                             "How can I help today?"]
    assert w.whole == [], "a streamed reply is never said a second time"
    assert w.speech.streams == [1] and w.speech.closed == [1]
    assert "good afternoon" in w._last_spoken_norm


def test_an_agent_that_cannot_stream_keeps_the_whole_reply_path():
    class OldAgent:
        pending = None

        def __init__(self):
            self.kwargs = []

        async def respond(self, text, *, origin="text", allow_escalate=False, dispatch=None):
            self.kwargs.append(allow_escalate)
            return AgentReply("Opening Notepad.")

    agent = OldAgent()
    w = _worker(agent)
    run(GenAILiveWorker._run_text_fallback(w, "open notepad"))
    assert w.whole == ["Opening Notepad."] and w.speech.said == []


def test_streaming_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ORION_LOCAL_STREAMING", "0")
    chat = StreamChat("Good afternoon, it is nice to hear from you.")
    w = _worker(_agent(chat))
    run(GenAILiveWorker._run_text_fallback(w, "hello"))
    assert w.whole == ["Good afternoon, it is nice to hear from you."] and w.speech.said == []


def test_cloud_mode_still_answers_from_the_cloud_without_the_local_agent():
    chat = StreamChat("A local answer that must not be used.")
    w = _worker(_agent(chat), mode="cloud")
    run(GenAILiveWorker._run_text_fallback(w, "hello"))
    assert w.whole == ["A cloud answer."] and chat.calls == [] and w.speech.said == []


def test_hybrid_still_hands_open_ended_turns_to_the_cloud_with_nothing_streamed():
    chat = StreamChat("A local poem that must not be used.")
    w = _worker(_agent(chat), mode="hybrid")
    run(GenAILiveWorker._run_text_fallback(w, "write me a poem about the sea"))
    assert len(w.cloud_calls) == 1 and w.whole == ["A cloud answer."]
    assert chat.calls == [] and w.speech.said == []


def test_hybrid_local_turns_stream():
    chat = StreamChat("Your next meeting is at three this afternoon. Nothing else is booked.")
    w = _worker(_agent(chat), mode="hybrid")
    run(GenAILiveWorker._run_text_fallback(w, "when is my next meeting"))
    assert w.speech.said == ["Your next meeting is at three this afternoon.",
                             "Nothing else is booked."] and w.cloud_calls == []


def test_a_confirmation_question_is_still_spoken_and_nothing_runs():
    chat = StreamChat(_call("shutdown_orion"))
    agent = _agent(chat, dispatch=None)
    w = _worker(agent)
    agent._dispatch_override = None
    run(GenAILiveWorker._run_text_fallback(w, "shut yourself down"))
    assert agent.pending is not None and w.dispatch_calls == []
    assert w.speech.said and "Shall I go ahead?" in " ".join(w.speech.said)


def _start_hanging_turn(w, text="tell me about the weather"):
    return asyncio.ensure_future(GenAILiveWorker._submit_text_fallback(w, text))


def test_barge_in_stops_generation_and_flushes_the_queued_speech():
    chat = StreamChat("The first part of a long answer is here. The rest never comes.",
                      hang_after=9)
    w = _worker(_agent(chat))

    async def flow():
        task = _start_hanging_turn(w)
        await _until(lambda: w.speech.said)
        GenAILiveWorker._on_barge_in(w)
        with pytest.raises(asyncio.CancelledError):
            await task

    run(flow())
    assert w.speech.said == ["The first part of a long answer is here."]
    assert chat.closed == 1, "the Ollama stream was closed: generation stopped"
    assert w.speech.interrupts >= 1, "queued sentences flushed"
    assert w._local_reply.cancelled and w._fallback_epoch == 1


def test_a_spoken_cancel_during_generation_stops_the_stream():
    chat = StreamChat("Thinking about it for a while here.", hang_after=0)
    w = _worker(_agent(chat))
    w.interrupts = SimpleNamespace(note_trigger=lambda: None)

    async def flow():
        task = _start_hanging_turn(w)
        await _until(lambda: chat.calls)
        await asyncio.sleep(0)
        GenAILiveWorker._on_interrupt_phrase(w, VoiceInterruptManager.ACTION_CANCEL)
        with pytest.raises(asyncio.CancelledError):
            await task

    run(flow())
    assert chat.closed == 1 and w.speech.said == []


def test_orion_cancel_is_a_whole_utterance_command():
    assert VoiceInterruptManager.classify("Orion, cancel that.") == VoiceInterruptManager.ACTION_CANCEL
    assert VoiceInterruptManager.classify("orion never mind") == VoiceInterruptManager.ACTION_CANCEL
    assert VoiceInterruptManager.classify("orion cancel my dentist reminder") != \
        VoiceInterruptManager.ACTION_CANCEL
    assert VoiceInterruptManager.classify("orion stop") == VoiceInterruptManager.ACTION_PAUSE


def test_talking_over_a_silent_reply_holds_it_then_his_words_replace_it():
    chat = StreamChat("Tomorrow will be wet everywhere.", hang_after=0)
    w = _worker(_agent(chat))

    async def flow():
        task = _start_hanging_turn(w, "what's the weather tomorrow")
        await _until(lambda: chat.calls)
        GenAILiveWorker._on_speech_activity(w, True)
        assert w._local_reply.holding
        GenAILiveWorker._on_speech_activity(w, False)
        merged = GenAILiveWorker._supersede_local_reply(w, "in London")
        with pytest.raises(asyncio.CancelledError):
            await task
        return merged

    merged = run(flow())
    assert merged == "what's the weather tomorrow in London"
    assert chat.closed == 1 and w.speech.said == [] and w._local_reply.cancelled


def test_noise_over_a_held_reply_releases_it(monkeypatch):
    chat = StreamChat("Tomorrow will be wet everywhere, I am afraid.", delay=0.02)
    w = _worker(_agent(chat))
    monkeypatch.setattr(GenAILiveWorker, "_route_local_transcript", lambda self, text: None)

    async def flow():
        task = _start_hanging_turn(w, "what's the weather tomorrow")
        await _until(lambda: chat.calls)                    # generating, nothing said yet
        GenAILiveWorker._on_speech_activity(w, True)          # a cough
        await task                                          # the answer is ready, held
        assert w.speech.said == []
        GenAILiveWorker._on_local_transcript(w, "uh")       # nothing he asked for
        return task

    run(flow())
    assert w.speech.said == ["Tomorrow will be wet everywhere, I am afraid."]


def test_a_turn_that_has_acted_is_never_superseded():
    chat = StreamChat(_call("open_app", app_name="notepad"), hang_after=0)
    agent = _agent(chat)
    w = _worker(agent)
    agent.turn_calls = 1
    reply = vs.SpokenReply(speak=lambda _t: None, loop=_Loop(), earcon_after=0,
                           request="open notepad").start()
    reply.hold()
    w._local_reply = reply
    assert GenAILiveWorker._supersede_local_reply(w, "and calculator") == "and calculator"
    assert not reply.cancelled


def test_onset_rewarms_a_model_that_may_have_been_unloaded():
    brain_mode.set_mode("local")
    warmed = []

    class Agent:
        def warm_due(self):
            return True

        async def warm(self):
            warmed.append(True)
            return {"ok": True}

    w = _worker(Agent())

    async def flow():
        GenAILiveWorker._rewarm_local_model(w)
        for _ in range(5):
            await asyncio.sleep(0)

    run(flow())
    assert warmed == [True]


# ── 7. the dry benchmark ────────────────────────────────────────────────────

def test_utterances_are_written_as_16k_mono_wavs(tmp_path):
    import math
    from tools import eval_local_voice as ev

    def make(text):
        return [0.2 * math.sin(i / 7.0) for i in range(2400)], 24000

    rows = ev.synthesise_utterances(make, tmp_path, ev.UTTERANCES[:2])
    pcm, rate = ev.read_wav(Path(rows[0]["path"]))
    assert rate == 16000 and len(rows) == 2
    assert abs(len(pcm) / 2 / 16000 - (0.3 + 0.1)) < 0.01      # lead silence + 0.1 s of words


class FakeVoice:
    """Synthesis that takes a little time and returns a little PCM."""

    def __init__(self):
        self.lines = []

    def stream_pcm(self, text, on_chunk=None):
        self.lines.append(text)
        time.sleep(0.005)
        if on_chunk is not None:
            on_chunk(b"\x01\x00" * 480)
        return b"\x01\x00" * 480


def test_the_benchmark_measures_first_audio_before_the_whole_reply():
    from tools import eval_local_voice as ev
    text = ("Hello there, it is good to hear from you this afternoon. I hope the day is "
            "treating you well. Let me know if you need anything at all.")

    def new_agent():
        return _agent(StreamChat(text, delay=0.01))

    bench = ev.VoiceBench(transcribe=lambda pcm, rate: "hello", agent=new_agent,
                          tts=lambda: ev.SynthesisWorker(FakeVoice()), endpoint_ms=650.0)

    async def flow():
        streamed = await bench.case("hello", "chat", b"\x00\x00" * 160, 16000, streamed=True)
        whole = await bench.case("hello", "chat", b"\x00\x00" * 160, 16000, streamed=False)
        return streamed, whole

    streamed, whole = run(flow())
    assert streamed["spoken"] and streamed["chunks"] == 3 and whole["chunks"] == 1
    assert 650.0 <= streamed["ttfa_ms"] <= streamed["total_ms"]
    assert streamed["ttfa_ms"] < whole["ttfa_ms"], "streaming starts speaking sooner"
    summary = ev.summarise(bench.rows)
    assert summary["streamed"]["chat"]["ttfa_ms"]["p50"] == streamed["ttfa_ms"]
    report = {"model": "fake", "created": "now", "summary": summary, "cases": 1, "runs": 1,
              "endpoint_ms": 650.0}
    assert "First audio p50" in ev.markdown(report)


def test_the_benchmark_says_plainly_what_is_missing_and_starts_nothing(monkeypatch):
    from orion_core import speech_offline, voice_kokoro
    from tools import eval_local_voice as ev

    def no_ollama(_root):
        raise ConnectionError("connection refused")

    class NoVoice:
        def __init__(self, *a, **k):
            pass

        def available(self):
            return False

        def describe(self):
            return "Kokoro is not available: model missing"

    class NoHearing:
        def __init__(self, *a, **k):
            self.cached_available = False

    monkeypatch.setattr(ev, "ollama_models", no_ollama)
    monkeypatch.setattr(voice_kokoro, "enable", lambda: None)
    monkeypatch.setattr(voice_kokoro, "KokoroVoice", NoVoice)
    monkeypatch.setattr(speech_offline, "OfflineTranscriber", NoHearing)
    problems = ev._preflight(SimpleNamespace(ollama="http://127.0.0.1:1", model="llama3.1:latest"))
    assert any("Ollama is not answering" in p and "never starts or pulls" in p for p in problems)
    assert any("Kokoro" in p for p in problems) and any("Whisper" in p for p in problems)
    assert ev.main(["--ollama", "http://127.0.0.1:1"]) == 2
