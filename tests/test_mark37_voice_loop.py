"""Mark XXXVII: the smarter voice loop.

Each test reproduces a failure from the owner's own sessions or the audit:
a running tool counted as a stall (17 Oct 11:31), a request heard locally and
never answered (21 Oct 17:10), cancellation that only muted the answer, a
45 s tool reported as a success, gateway envelopes taught as labels, noise
answered, spoken turns with no memory, and "done" said about nothing done
(15 Oct 13:30, 17 Oct 11:13). No network, audio device or provider.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace

import pytest

import orion_core.live_worker as lw
from orion_core.live_worker import (
    GenAILiveWorker,
    _effective_call,
    is_cancel_command,
    junk_transcript_reason,
)
from orion_core.research_commitments import ResearchCommitments, claimed_completions
from orion_core.session_context import ActivityLedger


class _Log:
    def __init__(self):
        self.lines: list[str] = []

    def emit(self, *args):
        self.lines.append(" ".join(str(a) for a in args))


class _Session:
    def __init__(self):
        self.notes: list[dict] = []
        self.responses: list[dict] = []

    async def send_client_content(self, **kwargs):
        self.notes.append(kwargs)

    async def send_tool_response(self, **kwargs):
        self.responses.append(kwargs)


class _Call:
    def __init__(self, name, args=None, call_id=None):
        self.name, self.args, self.id = name, args or {}, call_id or f"id-{name}"


def _worker() -> GenAILiveWorker:
    w = GenAILiveWorker.__new__(GenAILiveWorker)
    w.bus = SimpleNamespace(log=_Log(), banner=SimpleNamespace(emit=lambda *a: None),
                            paused=SimpleNamespace(emit=lambda *a: None))
    w.session = _Session()
    w._send_closed = False
    w._send_lock = asyncio.Lock()
    w.stop_event = asyncio.Event()
    w.paused = w.standby_mode = False
    w.connected = True
    w.mic = None
    w.tool_busy = False
    w._turn_active = False
    w._turn_said = ""
    w._pending_turn = None
    w._request_serial = w._turn_request_id = 0
    w._held_tool_request = None
    w._partial_spoken_request = ""
    w._pending_turn_retries = 0
    w._drop_live_output = False
    w._turn_progress_at = 0.0
    w._trace = None
    w._context_sessions = 0
    w._typed_live_turn = True
    w.wake_mode_enabled = False
    w._wake_gate_open = lambda: True
    w._last_spoken_norm, w._spoken_at = "", 0.0
    w._last_cmd_norm, w._last_cmd_at = "", 0.0
    w.speech = SimpleNamespace(output_active=lambda: False, interrupt_all=lambda: None, muted=False)
    w._emit_state = lambda *_: None
    w._refresh_wake_window = lambda: None
    w._offline_voice_ready = lambda: False
    w._said_by_me = lambda *_: False
    w._voice_reply_allowed = lambda: True
    w._activity_ledger = ActivityLedger()
    w._pending_context_notes = deque(maxlen=24)
    w.announcements = []
    w.announce = w.announcements.append
    w.learned = []
    w.observe_for_reflex = lambda u, t, a=None, ok=True: w.learned.append((u, t, dict(a or {}), ok))
    w.shadow = []
    w.observe_resolver_shadow = lambda q, t: w.shadow.append((q, t))
    w._function_response = lambda call_id, name, payload: (call_id, name, payload)
    return w


async def _settle(predicate=lambda: True, rounds=200):
    for _ in range(rounds):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate(), "expected async progress did not occur"


# ── 1. a running tool is progress, not a stall ───────────────────────────────

def test_a_long_single_tool_restarts_the_stall_clock_when_it_finishes(monkeypatch):
    """17 Oct 11:31: find_files took 27 s; the single-call path skipped the
    progress mark, the watchdog saw a 29 s 'stall' and re-ran the request."""
    w = _worker()
    fallbacks = []

    async def fallback(*args, **kwargs):
        fallbacks.append(args)

    w._submit_text_fallback = fallback
    w.router = SimpleNamespace(has_text_fallback=lambda: True)

    async def scenario():
        w._mark_turn("find the files I worked on")
        w._turn_progress_at = time.monotonic() - 27.0     # the tool started 27 s ago
        progress_during = []

        async def dispatch(name, args):
            progress_during.append(time.monotonic() - w._turn_progress_at)
            return {"ok": True, "result": "3 files"}, None

        w._dispatch_live_tool = dispatch
        await w._handle_tool_call(SimpleNamespace(function_calls=[_Call("find_files")]))
        assert progress_during[0] < 1.0, "starting the tool is progress"
        assert time.monotonic() - w._turn_progress_at < 1.0, "finishing it is progress"

        async def tick(_seconds):
            w.stop_event.set()

        monkeypatch.setattr(lw.asyncio, "sleep", tick)
        await w._turn_watchdog()

    asyncio.run(scenario())
    assert fallbacks == [], "a turn whose tool just answered is not stalled"
    assert w._pending_turn == "find the files I worked on"


# ── 2. heard here, never answered by Live ─────────────────────────────────────

def _heard_worker(monkeypatch):
    w = _worker()
    w._typed_live_turn = False
    monkeypatch.setattr(GenAILiveWorker, "HEARD_FALLBACK_S", 0.01)
    w.submitted = []

    async def submit(text):
        w.submitted.append(text)

    w.submit_text = submit
    return w


def test_a_request_live_never_heard_is_sent_once_as_text(monkeypatch):
    """21 Oct 17:10: "explain functions to me please" transcribed locally,
    and Live started no turn for it - nobody answered."""
    w = _heard_worker(monkeypatch)

    async def scenario():
        w._arm_heard_fallback("explain functions to me please", False)
        w._arm_heard_fallback("explain functions to me please", False)   # delivered twice
        await asyncio.sleep(0.05)
        await _settle(lambda: bool(w.submitted))
        await asyncio.sleep(0.02)

    asyncio.run(scenario())
    assert w.submitted == ["explain functions to me please"]
    assert any("Live never answered" in line for line in w.bus.log.lines)


@pytest.mark.parametrize("live_did", ["heard", "answered", "new_turn", "busy"])
def test_the_fallback_gives_way_to_any_sign_live_took_the_turn(monkeypatch, live_did):
    w = _heard_worker(monkeypatch)

    async def scenario():
        w._arm_heard_fallback("what is the capital of Peru", False)
        if live_did == "heard":          # Live's own input transcription
            w._live_heard_at = time.monotonic()
            w._cancel_heard_fallback()
        elif live_did == "answered":
            w._live_output_at = time.monotonic()
        elif live_did == "new_turn":
            w._mark_turn("something else")
            w._turn_active = False
        else:
            w.speech = SimpleNamespace(output_active=lambda: True)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert w.submitted == []


def test_nothing_is_armed_when_live_already_heard_or_for_noise(monkeypatch):
    w = _heard_worker(monkeypatch)

    async def scenario():
        w._live_heard_at = time.monotonic() - 3.0      # Live transcribed it mid-utterance
        w._arm_heard_fallback("explain functions to me please", False)
        w._live_heard_at = 0.0
        w._arm_heard_fallback("う う う 。", False)
        w._arm_heard_fallback("um uh", False)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert w.submitted == []
    assert sum("SR: ignored" in line for line in w.bus.log.lines) == 2



def test_a_missed_request_with_its_audio_delivered_refreshes_the_channel(monkeypatch):
    """28 Oct 16:56: Live had the owner's audio and produced nothing, twice in
    a row. The miss now says what reached Live, and the channel is refreshed."""
    w = _heard_worker(monkeypatch)
    loud = (6000).to_bytes(2, "little", signed=True) * 512
    for _ in range(200):
        w._note_audio_sent(loud)

    async def scenario():
        w._arm_heard_fallback("how much have you progressed since version one", False)
        await asyncio.sleep(0.05)
        await _settle(lambda: bool(w.submitted))

    asyncio.run(scenario())
    line = next(line for line in w.bus.log.lines if "Live never answered" in line)
    assert "Live was sent 6.4 s of the microphone" in line
    assert w._rotate_reason == "it stopped hearing the microphone"


def test_a_missed_request_whose_audio_never_left_names_why_and_repairs_it(monkeypatch):
    w = _heard_worker(monkeypatch)
    w.microphone_enabled = True
    w.mic = SimpleNamespace(enabled=False)
    w.mic.set_enabled = lambda on: setattr(w.mic, "enabled", on)

    async def scenario():
        w._arm_heard_fallback("what is on my calendar today", False)
        await asyncio.sleep(0.05)
        await _settle(lambda: bool(w.submitted))

    asyncio.run(scenario())
    assert any("the microphone was not reaching Live (the microphone engine was left "
               "disabled)" in line for line in w.bus.log.lines)
    assert w.mic.enabled is True
    assert getattr(w, "_rotate_reason", "") == "", "no refresh when Live never had the audio"

# ── 3. cancellation stops the work, not just the answer ──────────────────────

def test_a_server_cancellation_stops_the_running_tool_and_its_answer(monkeypatch):
    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)
    w = _worker()
    seen = []

    async def scenario():
        entered = asyncio.Event()

        async def dispatch(name, args):
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                seen.append("cancelled")
                raise
            return {"ok": True, "result": "late"}, None

        w._dispatch_live_tool = dispatch
        task = asyncio.create_task(w._handle_tool_call(
            SimpleNamespace(function_calls=[_Call("find_files")])))
        await entered.wait()
        w._note_cancelled_tool_calls(["id-find_files"])
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert seen == ["cancelled"]
    assert w.session.responses == []
    assert w.announcements == []


def test_a_detached_tool_cancelled_later_is_never_announced(monkeypatch):
    """A worker thread cannot be interrupted, so the work may finish anyway:
    'The X task has finished' must not follow a cancellation."""
    w = _worker()
    monkeypatch.setattr(GenAILiveWorker, "LIVE_TOOL_ANSWER_S", 0.01)
    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)

    async def scenario():
        release = asyncio.Event()

        async def stubborn(name, args):
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue                      # ignores cancellation, like a thread
            return {"ok": True, "result": "Everything deleted."}, None

        w._dispatch_live_tool = stubborn
        payload, _ = await w._run_tool_call(_Call("file_controller"))
        assert payload["status"] == "running"
        w._note_cancelled_tool_calls(["id-file_controller"])
        release.set()
        await _settle(lambda: not w._live_tool_tasks)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert w.announcements == []
    assert not w._pending_context_notes
    assert any("not announced" in line for line in w.bus.log.lines)


def test_spoken_cancel_stops_this_turns_tools_revokes_scope_and_answers_cancelled():
    from orion_core.execution_policy import background_scope

    w = _worker()
    scope_active_at_cancel = []

    async def scenario():
        entered = asyncio.Event()

        async def dispatch(name, args):
            with background_scope("tool", run_id=name) as scope:
                entered.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    scope_active_at_cancel.append(scope.active)
                    raise
            return {"ok": True, "result": "never"}, None

        w._dispatch_live_tool = dispatch
        w._mark_turn("open notepad and write my letter")
        w._start_tool_call(SimpleNamespace(function_calls=[
            _Call("open_app"), _Call("desktop_control")]))
        await entered.wait()
        w.cancel_turn()
        await asyncio.wait_for(asyncio.gather(*list(w._tool_call_tasks)), 2)

    asyncio.run(scenario())
    assert scope_active_at_cancel == [False], "the scope is revoked before the task unwinds"
    sent = [r for batch in w.session.responses for r in batch["function_responses"]]
    assert [name for _id, name, _p in sent] == ["open_app", "desktop_control"]
    assert all(p["status"] == "cancelled" for _id, _n, p in sent), "answered, so the turn cannot hang"
    assert w.learned == [] and w.announcements == []


def test_spoken_cancel_leaves_an_earlier_requests_background_work_alone(monkeypatch):
    w = _worker()
    monkeypatch.setattr(GenAILiveWorker, "LIVE_TOOL_ANSWER_S", 0.01)
    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)

    async def scenario():
        release = asyncio.Event()

        async def dispatch(name, args):
            await release.wait()
            return {"ok": True, "result": "Dossier saved."}, None

        w._dispatch_live_tool = dispatch
        w._mark_turn("research fusion energy")
        await w._run_tool_call(_Call("research"))          # detached, request 1
        w._mark_turn("what time is it")                    # request 2
        w.cancel_turn()
        release.set()
        await _settle(lambda: bool(w.announcements))

    asyncio.run(scenario())
    assert w.announcements == ["The research task has finished. Dossier saved."]


@pytest.mark.parametrize("said", ["cancel", "Cancel that, I meant Tuesday", "Orion, cancel.",
                                  "never mind", "never mind what time is it", "stop",
                                  "stop it please", "okay stop", "that's enough", "forget it",
                                  "scratch that, make it six"])
def test_a_spoken_cancel_is_the_whole_utterance_or_its_lead(said):
    assert is_cancel_command(said)


@pytest.mark.parametrize("said", ["cancel my dentist reminder", "forget that meeting",
                                  "stop the music", "stop listening", "cancel the timer",
                                  "can you cancel my subscription", "I said cancel", "",
                                  "enough about that, tell me the news"])
def test_a_command_that_merely_contains_a_cancel_word_goes_through(said):
    assert not is_cancel_command(said)


def test_cancel_my_reminder_while_speaking_reaches_orion_instead_of_aborting(monkeypatch):
    from tests.test_live_worker_construction import _construct

    worker = _construct(monkeypatch)
    worker.wake_mode_enabled = False
    worker.standby_mode = False
    worker.local_brain = None
    worker.speech = SimpleNamespace(output_active=lambda: True, interrupt_all=lambda: None)
    worker._turn_active = True
    worker._is_own_echo = lambda _: False
    worker._is_duplicate_command = lambda _: False
    for name in ("_handle_quiet_command", "_handle_focus_command",
                 "_handle_finish_speaking_command", "_handle_device_command"):
        setattr(worker, name, lambda _: False)
    monkeypatch.setattr(lw.VoiceInterruptManager, "classify", lambda _: "")
    cancelled, received = [], []
    worker.cancel_turn = lambda: cancelled.append(True)

    async def submit(text):
        received.append(text)

    worker.submit_text = submit
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_local_transcript("cancel my dentist reminder")
    assert cancelled == [] and received == ["cancel my dentist reminder"]
    worker._on_local_transcript("cancel")
    assert cancelled == [True]


# ── 4 + 5. running is not success; labels are the tool that ran ──────────────

def test_a_detached_tool_is_learned_only_from_how_it_really_ended(monkeypatch):
    w = _worker()
    monkeypatch.setattr(GenAILiveWorker, "LIVE_TOOL_ANSWER_S", 0.01)
    monkeypatch.setattr(lw, "_live_diag", lambda *a, **k: None)

    async def scenario():
        release = asyncio.Event()

        async def dispatch(name, args):
            await release.wait()
            return {"ok": False, "result": "drive not found"}, None

        w._dispatch_live_tool = dispatch
        w._mark_turn("find my tax return")
        w._turn_said = "find my tax return"
        await w._handle_tool_call(SimpleNamespace(function_calls=[_Call(
            "use_tool", {"name": "find_files", "arguments": '{"query": "tax return"}'})]))
        sent = w.session.responses[0]["function_responses"][0][2]
        assert sent["status"] == "running" and "ok" not in sent
        assert w.learned == [] and w.shadow == [], "nothing is learned from 'still running'"
        release.set()
        await _settle(lambda: bool(w.learned))

    asyncio.run(scenario())
    assert w.learned == [("find my tax return", "find_files", {"query": "tax return"}, False)]
    assert w.shadow == [("find my tax return", "find_files")]


def test_gateway_calls_teach_the_inner_tool_and_never_find_tool():
    w = _worker()

    async def dispatch(name, args):
        return {"ok": True, "result": "done"}, None

    w._dispatch_live_tool = dispatch
    w._turn_said = "what's the weather in Leeds"
    w._turn_learned = False

    async def scenario():
        await w._handle_tool_call(SimpleNamespace(function_calls=[
            _Call("find_tool", {"need": "weather"})]))
        assert w.learned == [] and w.shadow == []
        assert not w._turn_learned, "the search must not use up the turn's lesson"
        await w._handle_tool_call(SimpleNamespace(function_calls=[_Call(
            "use_tool", {"name": "weather", "arguments": {"location": "Leeds"}})]))

    asyncio.run(scenario())
    assert w.learned == [("what's the weather in Leeds", "weather", {"location": "Leeds"}, True)]
    assert w.shadow == [("what's the weather in Leeds", "weather")]
    assert _effective_call("use_tool", {"tool": "x", "args": "not json"}) == ("x", {})


def test_a_multi_tool_turn_scores_every_unwrapped_tool_but_teaches_none():
    w = _worker()

    async def dispatch(name, args):
        return {"ok": True, "result": "done"}, None

    w._dispatch_live_tool = dispatch
    w._turn_said = "check the weather and my calendar"
    w._turn_learned = False
    asyncio.run(w._handle_tool_call(SimpleNamespace(function_calls=[
        _Call("weather", {"location": "Leeds"}),
        _Call("use_tool", {"name": "calendar", "arguments": {"action": "today"}}, "c2"),
        _Call("find_tool", {"need": "x"}, "c3")])))
    assert w.learned == []
    assert w.shadow == [("check the weather and my calendar", "weather"),
                        ("check the weather and my calendar", "calendar")]


# ── 6. noise is not a request ────────────────────────────────────────────────

@pytest.mark.parametrize("said,reason", [
    ("다요.", "not English"), ("う う う 。", "not English"), ("um uh hmm", "filler"),
    ("so", "filler"), ("yeah", "too short"),
])
def test_noise_from_the_owners_logs_is_dropped(said, reason):
    assert reason in junk_transcript_reason(said)


@pytest.mark.parametrize("said", ["what time is it", "the weather", "explain functions to me please",
                                  "ballpark be"])
def test_ordinary_requests_pass_the_conservative_gate(said):
    assert junk_transcript_reason(said) == ""


def test_addressed_short_requests_and_a_learned_non_latin_language_pass():
    assert junk_transcript_reason("Orion, weather", addressed=True) == ""
    assert junk_transcript_reason("다요 다요 안녕하세요", language="ko") == ""


def test_offline_noise_is_logged_as_ignored_and_never_submitted(monkeypatch):
    from tests.test_live_worker_construction import _construct

    worker = _construct(monkeypatch)
    worker.wake_mode_enabled = False
    worker.standby_mode = False
    worker._is_own_echo = lambda _: False
    worker._is_duplicate_command = lambda _: False
    for name in ("_handle_quiet_command", "_handle_focus_command",
                 "_handle_finish_speaking_command", "_handle_device_command"):
        setattr(worker, name, lambda _: False)
    monkeypatch.setattr(lw.VoiceInterruptManager, "classify", lambda _: "")
    received = []

    async def submit(text):
        received.append(text)

    worker.submit_text = submit
    monkeypatch.setattr(lw.background, "spawn", lambda coro: asyncio.run(coro))
    worker._on_local_transcript("う う う 。")
    assert received == []
    logged = [str(c.args[0]) for c in worker.bus.log.emit.call_args_list]
    assert any("SR: ignored" in line for line in logged)


# ── 7. spoken turns meet what memory knows ───────────────────────────────────

def _recall_worker(note="[Background from ORION's memory: exam on 3 Nov.]"):
    w = _worker()
    asked = []

    def relevant_note(text, limit=3):
        asked.append(text)
        return note

    w.memory = SimpleNamespace(relevant_note=relevant_note)
    w.asked = asked
    return w


def test_a_spoken_request_gets_its_relevant_facts_once_while_the_channel_is_idle():
    w = _recall_worker()

    async def scenario():
        w._turn_request_id = 4
        w._recall_for_spoken_turn("book me in after my exam")
        await _settle(lambda: bool(w.session.notes))
        w._turn_request_id = 5
        w._recall_for_spoken_turn("when is my exam again")
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert w.asked == ["book me in after my exam", "when is my exam again"]
    assert len(w.session.notes) == 1, "the same facts are not repeated"
    assert w.session.notes[0]["turn_complete"] is False
    assert "exam on 3 Nov" in w.session.notes[0]["turns"]["parts"][0]["text"]


def test_spoken_recall_never_cuts_across_a_newer_turn():
    w = _recall_worker()

    async def scenario():
        w._turn_request_id = 4
        w._recall_for_spoken_turn("book me in after my exam")
        w._mark_turn("and play some music")          # the next turn began meanwhile
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert w.session.notes == []


def test_the_receive_loop_recalls_for_the_final_spoken_transcript(monkeypatch):
    from orion_core.emotion import SentimentAnalyser

    monkeypatch.setattr(SentimentAnalyser, "broadcast", lambda *_a, **_k: None)
    w = _recall_worker()
    w._language = SimpleNamespace(observe=lambda _t: False, code="en")
    w._persist_episode = lambda *_: None
    w._record_live_usage = lambda *_: None
    w._note_user_present = lambda: None
    w._handle_mute_command = lambda *_a, **_k: False
    w._typed_live_turn = False
    session = w.session

    class Receiving:
        async def receive(self):
            yield SimpleNamespace(server_content=SimpleNamespace(
                input_transcription=SimpleNamespace(text="book me in after my exam"),
                output_transcription=None, turn_complete=False, interrupted=False))
            yield SimpleNamespace(server_content=SimpleNamespace(
                input_transcription=None, output_transcription=SimpleNamespace(text="Of course."),
                turn_complete=True, interrupted=False))
            await _settle(lambda: bool(session.notes))
            w.session = None

    async def scenario():
        w.session = Receiving()
        await w._receive_realtime_stream([])

    async def send_note(self, text):
        # The real one checks the live transport; this one only records.
        await session.send_client_content(turns={"parts": [{"text": text}]}, turn_complete=False)
        return True

    monkeypatch.setattr(GenAILiveWorker, "_send_context_note", send_note)
    asyncio.run(scenario())
    assert w.asked == ["book me in after my exam"]
    assert len(session.notes) == 1


def test_cold_memory_volunteers_only_facts_the_request_names(tmp_path, monkeypatch):
    from orion_core import semantic
    from orion_core.memory import OrionMemoryMatrix

    class _NotReady:
        def ready(self):
            return False

    monkeypatch.setattr(semantic, "ENCODER", _NotReady())
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *a: None))
    matrix = OrionMemoryMatrix(tmp_path / "cold.db", tmp_path, bus)
    try:
        matrix.save("home", "rent", "The user pays 700 pounds rent on the 1st")
        matrix.save("travel", "window_seat", "The user prefers a window seat")
        matrix.save("study", "exam_date", "The exam is on 3 November")
        assert [r["key_ref"] for r in matrix.relevant("remind me to pay the rent")] == ["rent"]
        assert matrix.relevant("minimise all the windows") == []
        assert matrix.relevant("book me in after my exam") == []      # label needs "date" too
        assert [r["key_ref"] for r in matrix.relevant("what date is my exam")] == ["exam_date"]
    finally:
        matrix.close()


# ── 8. claimed done, wasn't ──────────────────────────────────────────────────

S15 = ("An extensive investigation into the recent developments in AI and AGI, semiconductor "
       "industry dynamics, robotics innovation, and their global market and economic impacts "
       "is now underway, sir. I will compile a detailed dossier for your review.")
S17 = ("Allow me to clear those overdue tasks for you now. Oddly, sir, the system is not "
       "reflecting any overdue tasks at present. It seems they are no longer listed for your "
       "attention.")


def test_the_two_real_false_claims_are_caught():
    guard = ResearchCommitments()
    looked = guard.begin(7, "awareness", {"action": "list"})
    guard.finish(looked, ok=True)
    cleared = guard.begin(7, "awareness", {"action": "clear_completed"})
    guard.finish(cleared, ok=False, result="rejected: nothing matched")
    [finding] = guard.audit_claims(7, S17)
    assert finding.domain == "removal" and "rejected" in finding.reason
    [finding] = guard.audit_claims(8, S15)
    assert finding.domain == "research" and "no tool" in finding.reason
    assert "NOT been done" in finding.note()
    assert guard.audit_claims(8, S15) == [], "once per request"


@pytest.mark.parametrize("reply", [
    "Done.", "I've sent it.", "I can open Spotify if you like.", "Shall I set a reminder?",
    "I couldn't send the message.", "Would you like me to delete the file?",
    "Here is what an investigation means.",
])
def test_vague_offered_or_negated_replies_claim_nothing(reply):
    assert claimed_completions(reply) == []


def test_a_successful_or_running_tool_of_the_kind_backs_the_claim():
    guard = ResearchCommitments()
    sent = guard.begin(1, "use_tool", {"name": "discord_message", "arguments": {"text": "hi"}})
    guard.finish(sent, ok=True)
    assert guard.audit_claims(1, "I've sent the message to Kieran on Discord.") == []
    guard.begin(2, "research", {"action": "dossier"})                 # still running
    assert guard.audit_claims(2, S15) == []
    news = guard.begin(3, "morning_briefing", {})
    guard.finish(news, ok=True)
    assert guard.audit_claims(3, "A police investigation is now underway.") == []
    # "Yes, I've sent it" about the previous turn's real message stands...
    earlier = guard.begin(10, "send_message", {"text": "hi"})
    guard.finish(earlier, ok=True)
    assert guard.audit_claims(11, "Yes, I've sent the message.") == []
    # ...but not after this turn's own attempt failed.
    failed = guard.begin(12, "send_message", {"text": "hi"})
    guard.finish(failed, ok=False, result="not delivered")
    assert guard.audit_claims(12, "I've sent the message.")[0].domain == "message"
    guard.dismiss(9)
    assert guard.audit_claims(9, "I've opened YouTube for you.") == [], "cancelled turn"


def test_a_false_claim_after_turn_complete_places_a_correction_note():
    w = _worker()
    w._flush_context_notes = lambda: asyncio.sleep(0)

    async def scenario():
        w._mark_turn("remind me to call mum at five")
        w._turn_said = w._pending_turn
        assert w._queue_claim_audit("Of course. I'll remind you at five.")
        await asyncio.gather(*list(w._claim_audit_tasks))
        w._mark_turn("open youtube")

        async def ok(name, args):
            return {"ok": True, "result": "opened"}, None

        w._dispatch_live_tool = ok
        await w._tracked_live_tool("open_app", {"app_name": "YouTube"})
        assert w._queue_claim_audit("I've opened YouTube for you.")
        await asyncio.gather(*list(w._claim_audit_tasks))

    asyncio.run(scenario())
    assert len(w._pending_context_notes) == 1
    assert "remind you at five" in w._pending_context_notes[0]
    assert any(line.startswith("CLAIM:") for line in w.bus.log.lines)


def test_no_audit_for_interrupted_or_cancelled_turns():
    w = _worker()
    w._mark_turn("send the report to Sam")
    w._turn_said = w._pending_turn
    assert not w._queue_claim_audit("I've sent the email to Sam.", interrupted=True)
    w._drop_live_output = True
    assert not w._queue_claim_audit("I've sent the email to Sam.")
