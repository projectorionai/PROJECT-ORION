"""
Request tracing — turning "ORION ignored me" into a named subsystem.

Every test here is a way a request can die silently.  The point is not that
tracing prevents the failure; it is that after the failure there is a single
sentence saying which subsystem dropped it.
"""

from __future__ import annotations

import time

import pytest

from orion_core.request_trace import STAGE_ORDER, Stage, TraceStore


@pytest.fixture
def store():
    return TraceStore(capacity=50)


def _run_to(trace, *stages):
    for stage in stages:
        trace.stage(stage)
    return trace


# ── ids ───────────────────────────────────────────────────────────────────────

def test_ids_are_sequential_and_formatted_as_specified(store):
    a = store.begin()
    b = store.begin()
    assert a.request_id == "ORION_REQ_000001"
    assert b.request_id == "ORION_REQ_000002"


def test_ids_are_unique_across_many_requests(store):
    ids = {store.begin().request_id for _ in range(200)}
    assert len(ids) == 200


# ── the happy path ────────────────────────────────────────────────────────────

def test_a_complete_request_is_marked_complete(store):
    trace = store.begin()
    _run_to(trace, *STAGE_ORDER)
    assert trace.complete is True
    assert "completed" in trace.diagnosis()
    assert store.stalled(older_than=0.0) == []


def test_stage_timings_show_where_the_latency_went(store):
    trace = store.begin()
    trace.stage(Stage.CAPTURE_START)
    time.sleep(0.02)
    trace.stage(Stage.API_REQUEST_START)
    timings = dict(trace.stage_timings())
    assert timings[Stage.API_REQUEST_START.value] >= 15.0


# ── each way a request dies names its owner ───────────────────────────────────

@pytest.mark.parametrize("stages,expect_owner", [
    ((Stage.CAPTURE_START,), "microphone"),
    ((Stage.CAPTURE_START, Stage.CAPTURE_COMPLETE), "speech recognition"),
    ((Stage.CAPTURE_START, Stage.CAPTURE_COMPLETE,
      Stage.TRANSCRIPTION_START), "speech recognition"),
    ((Stage.TRANSCRIPTION_COMPLETE,), "command routing"),
    ((Stage.COMMAND_RECEIVED,), "agent routing"),
    ((Stage.AGENT_SELECTED,), "model provider"),
    ((Stage.API_REQUEST_START,), "model provider"),
    ((Stage.API_RESPONSE_RECEIVED,), "tool execution"),
    ((Stage.TOOL_EXECUTION_START,), "tool execution"),
    ((Stage.TOOL_EXECUTION_COMPLETE,), "speech synthesis"),
    ((Stage.TTS_START,), "speech synthesis"),
    ((Stage.PLAYBACK_START,), "audio playback"),
    ((Stage.PLAYBACK_COMPLETE,), "microphone gate"),
])
def test_the_stage_it_stopped_at_names_the_subsystem(store, stages, expect_owner):
    trace = store.begin()
    _run_to(trace, *stages)
    assert trace.complete is False
    assert expect_owner in trace.diagnosis()


def test_the_worst_case_is_called_out_explicitly(store):
    """Playback finished, listening never restored — ORION alive and deaf."""
    trace = store.begin()
    _run_to(trace, Stage.API_RESPONSE_RECEIVED, Stage.PLAYBACK_START,
            Stage.PLAYBACK_COMPLETE)
    diagnosis = trace.diagnosis()
    assert "listening was never restored" in diagnosis
    assert "ignores everything you say" in diagnosis


def test_a_request_that_never_started_is_distinguished(store):
    trace = store.begin()
    assert "never started" in trace.diagnosis()
    assert "microphone disabled" in trace.diagnosis()


# ── explicit failures, timeouts and recoveries ────────────────────────────────

def test_an_explicit_failure_records_stage_error_and_recovery(store):
    trace = store.begin()
    trace.stage(Stage.API_REQUEST_START)
    trace.failed(Stage.API_REQUEST_START, "websocket 1011",
                 recovery="rotated to the fallback provider")
    assert trace.failed_stage == Stage.API_REQUEST_START.value
    assert "websocket 1011" in trace.diagnosis()
    assert "rotated to the fallback provider" in trace.diagnosis()


def test_a_timeout_is_recorded_as_a_failure_with_its_duration(store):
    trace = store.begin()
    trace.timeout(Stage.TOOL_EXECUTION_START, 30.0, recovery="cancelled the tool")
    assert "TIMEOUT after 30.0s" in trace.error
    assert trace.failed_stage == Stage.TOOL_EXECUTION_START.value


def test_a_failed_request_is_counted_in_the_summary(store):
    ok = store.begin()
    _run_to(ok, *STAGE_ORDER)
    bad = store.begin()
    bad.failed(Stage.TTS_START, "no voice engine")
    summary = store.summary()
    assert summary["total"] == 2
    assert summary["complete"] == 1
    assert summary["failed"] == 1


# ── stall detection ───────────────────────────────────────────────────────────

def test_stalled_finds_incomplete_requests_only(store):
    done = store.begin()
    _run_to(done, *STAGE_ORDER)
    stuck = store.begin()
    _run_to(stuck, Stage.API_REQUEST_START)

    stalled = store.stalled(older_than=0.0)
    assert [t.request_id for t in stalled] == [stuck.request_id]


def test_explain_last_silence_answers_the_users_actual_question(store):
    stuck = store.begin()
    _run_to(stuck, Stage.API_REQUEST_START)
    answer = store.explain_last_silence()
    assert stuck.request_id in answer
    assert "model provider" in answer


def test_explain_last_silence_is_honest_when_nothing_was_dropped(store):
    done = store.begin()
    _run_to(done, *STAGE_ORDER)
    assert "nothing was dropped" in store.explain_last_silence()


def test_explain_last_silence_with_no_history(store):
    assert "No requests" in store.explain_last_silence()


# ── the store stays bounded ───────────────────────────────────────────────────

def test_the_store_is_bounded(store):
    for _ in range(500):
        store.begin()
    assert len(store.all()) == 50


def test_bottlenecks_rank_the_slowest_stages(store):
    trace = store.begin()
    trace.stage(Stage.CAPTURE_START)
    time.sleep(0.03)
    trace.stage(Stage.API_RESPONSE_RECEIVED)
    ranked = store.bottlenecks()
    assert ranked[0][0] == Stage.API_RESPONSE_RECEIVED.value


# ── instrumentation must never break the thing it measures ────────────────────

def test_a_broken_bus_never_propagates(store):
    class Exploding:
        class log:
            @staticmethod
            def emit(_m):
                raise RuntimeError("Qt is gone")

    store.attach_bus(Exploding())
    trace = store.begin()
    trace.stage(Stage.CAPTURE_START)          # must not raise
    trace.failed(Stage.TTS_START, "boom")     # must not raise
    assert trace.failed_stage == Stage.TTS_START.value


def test_stage_accepts_a_plain_string(store):
    trace = store.begin()
    trace.stage("CUSTOM_STAGE")
    assert trace.last_stage == "CUSTOM_STAGE"
    assert "CUSTOM_STAGE" in trace.diagnosis()


def test_report_renders_without_error(store):
    _run_to(store.begin(), *STAGE_ORDER)
    store.begin().failed(Stage.TTS_START, "no engine")
    text = store.report()
    assert "REQUEST TRACE" in text
    assert "ORION_REQ_" in text
