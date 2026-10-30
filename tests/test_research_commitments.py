"""A spoken promise starts no work without consent and never starts it twice."""

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.live_worker import _LIVE_TOOL_ORIGIN, GenAILiveWorker
from orion_core.research_commitments import (
    ResearchCommitments,
    is_research_start,
    promised_research,
    user_research_request,
)
from orion_core.session_context import ActivityLedger

PROMISE = ("An extensive investigation into AI developments and cryptocurrency markets "
           "is now underway. I will compile a detailed dossier.")


@pytest.mark.parametrize("user", [
    "Research quantum computing", "Could you investigate battery recycling?",
    "Please explore all the topics in more depth", "Orion, look into fusion energy",
])
def test_explicit_research_authorisation(user):
    assert user_research_request(user)[0]


@pytest.mark.parametrize("user", [
    "What did you find in your research?", "Have you started the investigation?",
    "Open the research dossier", "Find my research file", "Don't research this",
    "Do not investigate fusion", "Stop researching", "Tell me about quantum computing",
    "Research fusion tomorrow", "Explore the folder", "Yes", "",
])
def test_status_file_requests_negation_and_future_schedules_do_not_authorise_recovery(user):
    assert not user_research_request(user)[0]


@pytest.mark.parametrize("reply", [
    PROMISE, "I'm researching quantum computing now.", "I am investigating fusion energy.",
    "I'm doing research on battery recycling.", "I'll compile a detailed dossier on fusion energy.",
])
def test_affirmative_started_or_starting_research_is_recognised(reply):
    assert promised_research(reply)[0]


@pytest.mark.parametrize("reply", [
    "I can research quantum computing.", "I would investigate fusion if you wanted.",
    "I am not researching anything.", "I haven't started the research yet.",
    "Would you like a dossier?", "Here is what research means.",
])
def test_conditional_offers_and_truthful_non_start_are_not_promises(reply):
    assert not promised_research(reply)[0]


@pytest.mark.parametrize("name,args", [
    ("research", {"action": "dossier"}), ("research", {}),
    ("antigravity_research", {"topic": "fusion"}),
    ("use_tool", {"name": "research", "arguments": '{"action":"paper"}'}),
    ("job", {"action": "run", "tool": "research", "args": {"topic": "fusion"}}),
    ("use_tool", {"name": "job", "arguments": {"action": "start", "tool": "research", "args": {}}}),
])
def test_actual_research_and_supported_queued_wrappers_count_as_attempts(name, args):
    assert is_research_start(name, args)


@pytest.mark.parametrize("name,args", [
    ("find_tool", {"query": "research"}), ("research", {"action": "status"}),
    ("research", {"action": "queue"}), ("job", {"action": "status", "tool": "research"}),
    ("job", {"action": "run", "tool": "weather"}),
    ("job", {"action": "run", "tool": "research", "args": {"action": "status"}}),
])
def test_status_discovery_and_unrelated_jobs_do_not_count(name, args):
    assert not is_research_start(name, args)


def test_deictic_request_uses_real_recent_headlines_before_the_claimed_topic_once():
    guard = ResearchCommitments()
    decision = guard.decide(1, "Explore all the topics in more depth", PROMISE,
                            briefing="Battery recycling advances; Fusion reactor engineering")
    assert decision.topic == "Battery recycling advances; Fusion reactor engineering"
    assert guard.decide(1, "Explore all the topics in more depth", PROMISE) is None


def test_exact_observed_failure_recovers_descriptive_claimed_topics():
    decision = ResearchCommitments().decide(1, "Explore all the topics in more depth", PROMISE)
    assert decision.topic == "AI developments and cryptocurrency markets"


def test_ambiguous_authorised_research_corrects_without_inventing_a_topic():
    decision = ResearchCommitments().decide(1, "Research those topics", "I'll start research now.")
    assert not decision.topic and "no investigation had started" in decision.correction
    assert ResearchCommitments().decide(1, "Hello", PROMISE) is None


def worker():
    value = GenAILiveWorker.__new__(GenAILiveWorker)
    value.session = object()
    value.stop_event = asyncio.Event()
    value._turn_request_id = 1
    value._context_sessions = 2
    value._turn_active = True
    value._pending_turn = "Research quantum computing"
    value._turn_said = value._pending_turn
    value._typed_live_turn = True
    value._drop_live_output = False
    value._voice_reply_allowed = lambda: True
    value._said_by_me = lambda *_: False
    value.speech = SimpleNamespace(muted=False)
    value._activity_ledger = ActivityLedger()
    value._pending_context_notes = deque(maxlen=24)
    value.announcements, value.logs, value.calls = [], [], []
    value.announce = value.announcements.append
    value.bus = SimpleNamespace(log=SimpleNamespace(emit=value.logs.append))

    async def flush():
        pass

    async def dispatch(name, args):
        value.calls.append((name, args))
        return {"ok": True, "result": "Dossier saved.",
                "receipt": {"file_paths": [r"C:\Research\quantum.md"]}}, None

    value._flush_context_notes = flush
    value._dispatch_live_tool = dispatch
    return value


async def settle(value):
    tasks = list(getattr(value, "_research_recovery_tasks", ()))
    if tasks:
        await asyncio.gather(*tasks)
    await asyncio.sleep(0)


def test_live_recovery_calls_the_real_tracked_dossier_once_and_keeps_exact_result_context():
    async def scenario():
        value = worker()
        value._queue_research_commitment(PROMISE)
        value._queue_research_commitment(PROMISE)
        await settle(value)
        assert value.calls == [("research", {"action": "dossier", "topic": "quantum computing"})]
        assert value._activity_ledger.snapshot()[0].ok
        assert any(r"C:\Research\quantum.md" in note for note in value._pending_context_notes)
        origin = dict(value._tool_origin(), research_recovery=False)
        payload, _ = await value._tracked_live_tool("research", {"topic": "quantum computing"}, origin=origin)
        assert "duplicate" in payload["result"] and len(value.calls) == 1
    asyncio.run(scenario())


def test_a_tool_queued_after_final_words_prevents_duplicate_recovery():
    async def scenario():
        value = worker()
        value._transport_ready = lambda: True
        value._release_completed_tool_request = lambda: None

        async def handle(*_args, **_kwargs):
            pass

        value._handle_tool_call = handle
        value._queue_research_commitment(PROMISE)
        value._start_tool_call(SimpleNamespace(function_calls=[
            SimpleNamespace(name="research", args={"action": "paper", "topic": "quantum computing"})]))
        await settle(value)
        assert not value.calls
    asyncio.run(scenario())


@pytest.mark.parametrize("condition", ["interrupted", "muted", "owner", "stale", "cancelled", "retired", "stopping"])
def test_recovery_never_runs_a_rejected_cancelled_stale_or_muted_turn(condition):
    async def scenario():
        value = worker()
        value._queue_research_commitment(PROMISE, interrupted=condition == "interrupted")
        if condition == "muted":
            value.speech.muted = True
        elif condition == "owner":
            value._voice_reply_allowed = lambda: False
        elif condition == "stale":
            value._turn_request_id = 2
        elif condition == "cancelled":
            value._research_commitments().dismiss(1)
        elif condition == "retired":
            value.session = object()
        elif condition == "stopping":
            value.stop_event.set()
        await settle(value)
        assert not value.calls
    asyncio.run(scenario())


def test_discovery_or_status_tools_do_not_satisfy_a_promised_investigation():
    async def scenario():
        value = worker()
        value._research_commitments().observe(1, "find_tool", {"query": "research"})
        value._research_commitments().observe(1, "research", {"action": "status"})
        value._queue_research_commitment(PROMISE)
        await settle(value)
        assert len(value.calls) == 1
    asyncio.run(scenario())


def test_recovery_failure_is_truthful_and_preserved_in_continuity():
    async def scenario():
        value = worker()

        async def fail(*_args, **_kwargs):
            raise RuntimeError("provider unavailable")

        value._dispatch_live_tool = fail
        value._queue_research_commitment(PROMISE)
        await settle(value)
        assert "could not start or finish" in value.announcements[-1]
        assert "provider unavailable" in value.announcements[-1]
        assert not value._activity_ledger.snapshot()[0].ok
        assert "provider unavailable" in value._pending_context_notes[-1]
    asyncio.run(scenario())


def test_background_callbacks_keep_actual_running_then_completed_or_failed_state_once():
    async def scenario():
        value = worker()
        receipt = {"run_id": "actual-run", "action": "paper", "research_state": "running"}
        token = _LIVE_TOOL_ORIGIN.set(value._tool_origin())
        try:
            value._on_research_started("quantum computing", receipt)
        finally:
            _LIVE_TOOL_ORIGIN.reset(token)
        assert value._activity_ledger.snapshot()[0].running
        result = ToolResult("Research failed: no sources were available.", ok=False,
                            receipt=dict(receipt, research_state="failed"))
        value._on_research_completion("quantum computing", result)
        value._on_research_completion("quantum computing", result)
        assert not value._activity_ledger.snapshot()[0].running
        assert not value._activity_ledger.snapshot()[0].ok
        assert len(value.announcements) == 1 and "failed" in value.announcements[0]
        await asyncio.sleep(0)
    asyncio.run(scenario())


def test_completed_live_transcript_schedules_recovery_before_clearing_the_user_request(monkeypatch):
    from orion_core.emotion import SentimentAnalyser

    async def scenario():
        value = worker()
        value._research_briefing = (0.0, "")
        value._own_lines = deque()
        value._persist_episode = lambda *_: None
        value._refresh_wake_window = lambda: None
        value._record_live_usage = lambda *_: None
        value._emit_state = lambda *_: None
        value.paused, value.standby_mode, value.connected = False, False, True
        value.speech = SimpleNamespace(muted=False, output_active=lambda: False)
        monkeypatch.setattr(SentimentAnalyser, "broadcast", lambda *_a, **_k: None)

        class Session:
            async def receive(self):
                yield SimpleNamespace(server_content=SimpleNamespace(
                    output_transcription=SimpleNamespace(text=PROMISE),
                    input_transcription=None, turn_complete=True, interrupted=False))
                await settle(value)
                value.session = None

        value.session = Session()
        await value._receive_realtime_stream([])
        assert value._pending_turn is None
        assert value.calls == [("research", {"action": "dossier", "topic": "quantum computing"})]

    asyncio.run(scenario())


def test_registered_dossier_failure_is_announced_and_added_to_context_once():
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    async def scenario():
        value = worker()

        class Dispatch(KnowledgeDispatchMixin):
            research = object()
            research_started = value._on_research_started
            research_completion = value._on_research_completion

            async def _dossier(self, topic, args, **_kwargs):
                return ToolResult("No external sources were retrieved.", ok=False)

        dispatch = Dispatch()

        async def through_dispatch(name, args):
            value.calls.append((name, args))
            result = await dispatch.research_tool(args)
            return result.response_payload(), None

        value._dispatch_live_tool = through_dispatch
        value._queue_research_commitment(PROMISE)
        await settle(value)
        assert len(value.announcements) == 2  # initial truthful correction, then the one actual failure
        assert value.announcements[-1] == "No external sources were retrieved."
        assert len(value._pending_context_notes) == 1
        assert all(not item.ok and not item.running for item in value._activity_ledger.snapshot())

    asyncio.run(scenario())


def test_stopping_an_actual_awaited_dossier_cancels_recovery_and_reports_once():
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    async def scenario():
        value = worker()
        entered = asyncio.Event()

        class Dispatch(KnowledgeDispatchMixin):
            research = object()
            research_started = value._on_research_started
            research_completion = value._on_research_completion

            async def _dossier(self, topic, args, **_kwargs):
                entered.set()
                await asyncio.Event().wait()

        dispatch = Dispatch()

        async def through_dispatch(name, args):
            value.calls.append((name, args))
            result = await dispatch.research_tool(args)
            return result.response_payload(), None

        value._dispatch_live_tool = through_dispatch
        value._queue_research_commitment(PROMISE)
        await asyncio.wait_for(entered.wait(), 2)
        status = await dispatch.research_tool({"action": "status"})
        assert status.receipt["research_state"] == "running"
        stopping = await dispatch.research_tool({"action": "stop"})
        assert stopping.receipt["research_state"] == "cancelling"
        outcomes = await asyncio.gather(*list(value._research_recovery_tasks), return_exceptions=True)
        assert len(outcomes) == 1 and isinstance(outcomes[0], asyncio.CancelledError)
        await asyncio.sleep(0)
        final = await dispatch.research_tool({"action": "status"})
        assert not final.ok and final.receipt["research_state"] == "cancelled"
        assert len(value.calls) == 1
        assert len(value.announcements) == 2 and "cancelled" in value.announcements[-1]
        assert len(value._pending_context_notes) == 1
        assert all(not item.ok and not item.running for item in value._activity_ledger.snapshot())

    asyncio.run(scenario())



def test_finished_research_is_told_in_one_short_line_without_its_path():
    """30 Oct 2026: "I don't want ORION to narrate each time, I just want him to
    tell me it's done" - he read out the topic and the full path."""
    from orion_core.live_worker import _research_ready_line
    done = ToolResult("Research on 'x' is complete. Location: C:/Users/x/SUMMARY.md")
    line = _research_ready_line("circadian rhythm phase delay and practical strategies for severe "
                                "sleep schedule disruption", done, {"research_state": "completed"})
    assert line == ("Your research on circadian rhythm phase delay and practical... is ready. "
                    "Just ask and I'll open it.")
    assert ":" not in line and "SUMMARY" not in line


def test_a_call_with_no_action_is_recorded_as_its_default_and_backs_no_open_claim():
    """A file tool called without an action LISTS; it was recorded as action ""
    and counted as proof that a file had been opened."""
    from orion_core.research_commitments import CLAIM_DOMAINS
    receipt = ResearchCommitments().begin(1, "file_controller", {"path": "notes.md"})
    assert receipt.action == "list" and receipt.reads_only
    opened = next(domain for domain in CLAIM_DOMAINS if domain.name == "open")
    assert not opened.backed_by(receipt)
