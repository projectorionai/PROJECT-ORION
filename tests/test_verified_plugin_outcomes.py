"""A plugin failure stays a failure; successful writes must survive reopening."""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.plugin_results import plugin_result
from orion_core import local_tool_outcomes as operations


@pytest.mark.parametrize("value", [None, {"ok": "false", "result": "done"},
                                    {"ok": False, "result": "declined"},
                                    "Error: failed", "Failed to save", object()])
def test_unverified_or_explicit_failures_stay_failures(value):
    assert not plugin_result(value).ok


def test_structured_result_preserves_evidence_and_media():
    value = ToolResult("declined", ok=False, evidence=[{"repaired": False}])
    assert plugin_result(value) is value
    assert plugin_result("legacy calculation result").ok


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dispatch_preserves_plugin_failure_and_records_it(asynchronous):
    from orion_core.dispatcher import OrionDispatcher
    expected = ToolResult("write rejected", ok=False, evidence=[{"persisted": False}])
    async def async_tool():
        return expected
    calls = []
    worker = SimpleNamespace(handler_table=lambda: {},
        _tool_handlers={"sample": async_tool if asynchronous else lambda: expected},
        plugins=SimpleNamespace(record_call=lambda name: None,
            mark_loaded=lambda *args: calls.append(args)))
    result = asyncio.run(OrionDispatcher.dispatch(worker, "sample", {}))
    assert result is expected
    assert calls == [("sample", True, "write rejected")]


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "outcomes.db"
    monkeypatch.setattr(operations, "STORE_PATH", path)
    return path


def test_goal_is_persisted_and_updated_without_duplicate(store):
    assert operations.save_goal("Walk", "2027-01-01", 20).ok
    result = operations.save_goal("Walk", "2027-01-01", 100, "Completed")
    assert result.ok and result.evidence[0]["persisted"]
    with sqlite3.connect(store) as db:
        rows = db.execute("SELECT payload FROM records").fetchall()
    assert len(rows) == 1 and '"progress": 100' in rows[0][0]


@pytest.mark.parametrize("progress", [True, -1, 101, 1.2])
def test_goal_rejects_invalid_progress(store, progress):
    assert not operations.save_goal("Walk", "2027-01-01", progress).ok
    assert not store.exists()


def test_goal_does_not_claim_completion_for_inconsistent_input(store):
    assert not operations.save_goal("Walk", "not a date").ok
    assert not operations.save_goal("Walk", "2027-01-01", 20, "Completed").ok


def test_persistence_failure_does_not_report_success(store):
    store.mkdir()
    assert not operations.save_goal("Walk", "2027-01-01").ok
    assert not operations.record_error("example", "ExampleError").ok
    assert not operations.save_tool_spec("sample", "a plan").ok


def test_records_distinguish_requested_actions_from_performed_actions(store):
    draft = operations.save_tool_spec("sample", "a plan")
    error = operations.record_error("sample error", "ExampleError")
    report = operations.save_progress("example", [{"minutes": 10}], ["tomorrow"], {})
    assert all(result.ok for result in [draft, error, report])
    assert draft.evidence[0]["created_tool"] is False
    assert error.evidence[0]["repaired"] is False
    assert report.evidence[0]["reminders_scheduled"] == 0
    with sqlite3.connect(store) as db:
        assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 3


def test_emotional_reading_depends_on_text_and_has_no_claim_of_certainty():
    positive = operations.analyse_text("Congratulations, excellent work!", 0, 1)
    worried = operations.analyse_text("I am worried, stressed and anxious", 0, 1)
    assert positive.ok and worried.ok
    assert positive.evidence[0]["reading"]["sentiment"] != worried.evidence[0]["reading"]["sentiment"]
    assert not positive.evidence[0]["verified_emotion"]
    assert not operations.analyse_text("hello", float("nan"), 1).ok
