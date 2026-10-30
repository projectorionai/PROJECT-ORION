"""Removed remote tasks stay removed, including across restarts."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from orion_core import atomic_io, cognition as cognition_mod
from orion_core.cognition import CognitiveStateManager
from orion_core.notion import NotionService
from orion_core.data import ToolResult
from orion_core.dispatch_web import WebDispatchMixin
from orion_core.utils import utc_stamp


def task_dispatch(tmp_path):
    dispatcher = WebDispatchMixin()
    dispatcher.bus = bus()
    dispatcher.cognition = CognitiveStateManager(
        dispatcher.bus, Mock(), path=tmp_path / "continuity.json")
    dispatcher.notion = SimpleNamespace(available=True, tasks_db="db",
        open_tasks=AsyncMock(return_value=[]), complete_task=AsyncMock(),
        update_task=AsyncMock(), format_tasks=NotionService.format_tasks)
    return dispatcher


def test_full_task_title_identity_survives_completion_and_restart(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "long-tasks.json")
    prefix = "Shared title " + "x" * 207
    finished, sibling = prefix + " finished", prefix + " still open"
    for title in (prefix, finished, sibling):
        assert state.add_task(title)["title"] == title
    state.note_mentioned_tasks([prefix, finished, sibling], "briefing")
    assert state.last_mentioned_tasks() == [prefix, finished, sibling]
    state.dismiss_task(finished, "completed")
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert restored.is_dismissed(finished)
    assert not restored.is_dismissed(prefix) and not restored.is_dismissed(sibling)
    assert {row["title"] for row in restored.list_tasks()} == {prefix, sibling}
    assert restored.last_mentioned_tasks() == [prefix, sibling]


async def test_completion_sends_full_long_title_and_keeps_prefix_task(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    title = "Shared title " + "x" * 260 + " finished"
    prefix = title[:220]
    dispatcher.cognition.add_task(prefix)
    dispatcher.cognition.note_mentioned_tasks([title], "briefing")
    dispatcher.notion.complete_task.return_value = ToolResult("Done", receipt={
        "task": title, "completed": True})
    result = await dispatcher.complete_owner_tasks(all_listed=True)
    assert result.ok
    dispatcher.notion.complete_task.assert_awaited_once_with(title)
    restored = CognitiveStateManager(bus(), Mock(), path=dispatcher.cognition.path)
    restored._load_locked()
    assert restored.is_dismissed(title) and not restored.is_dismissed(prefix)
    assert [row["title"] for row in restored.list_tasks()] == [prefix]


async def test_empty_board_listing_clears_previous_all_those_referent(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.note_mentioned_tasks(["Old task"], "briefing")
    assert await dispatcher.open_owner_tasks(source="list_tasks") == []
    assert dispatcher.cognition.last_mentioned_tasks() == []
    restored = CognitiveStateManager(bus(), Mock(), path=dispatcher.cognition.path)
    restored._load_locked()
    assert restored.last_mentioned_tasks() == []
    result = await dispatcher.complete_owner_tasks(all_listed=True)
    assert not result.ok
    dispatcher.notion.complete_task.assert_not_called()


async def test_listing_filters_dismissed_rows_before_limiting_visible_tasks(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    for index in range(30):
        dispatcher.cognition.dismiss_task(f"Finished {index}")
    rows = [{"title": f"Finished {index}"} for index in range(30)] + [{"title": "Still open"}]
    dispatcher.notion.open_tasks.return_value = rows
    assert await dispatcher.open_owner_tasks(limit=1) == [{"title": "Still open"}]
    dispatcher.notion.open_tasks.assert_awaited_once_with(limit=None)


async def test_ambiguous_completion_never_changes_local_or_remote_tasks(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.add_task("Write report")
    dispatcher.cognition.add_task("Review report")
    before = dispatcher.cognition.path.read_bytes()
    result = await dispatcher.complete_owner_tasks("report")
    assert not result.ok and "multiple" in result.text
    assert dispatcher.cognition.path.read_bytes() == before
    dispatcher.notion.complete_task.assert_not_called()


async def test_verified_remote_title_never_completes_a_shorter_local_title(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.add_task("Build")
    dispatcher.notion.complete_task.return_value = ToolResult("Done", receipt={
        "task": "Build report", "completed": True})
    result = await dispatcher.complete_owner_tasks("report")
    assert result.ok
    assert [row["title"] for row in dispatcher.cognition.list_tasks()] == ["Build"]
    assert dispatcher.cognition.is_dismissed("Build report")
    assert not dispatcher.cognition.is_dismissed("report")


async def test_board_failure_retains_exact_user_completion_but_reports_partial_failure(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.add_task("Write report")
    dispatcher.notion.complete_task.return_value = ToolResult("Notion permission denied", ok=False)
    result = await dispatcher.complete_owner_tasks("Write report")
    assert not result.ok
    assert result.receipt["completed_on_board"] == []
    assert result.receipt["recorded_done"] == ["Write report"]
    assert result.receipt["board_problems"]
    restored = CognitiveStateManager(bus(), Mock(), path=dispatcher.cognition.path)
    restored._load_locked()
    assert restored.is_dismissed("Write report") and restored.list_tasks() == []


async def test_board_failure_for_unknown_fragment_does_not_invent_completion(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.notion.complete_task.return_value = ToolResult("Multiple tasks match", ok=False)
    result = await dispatcher.complete_owner_tasks("report")
    assert not result.ok
    assert not dispatcher.cognition.is_dismissed("report")
    assert result.receipt["recorded_done"] == []


async def test_completion_keeps_verified_receipt_when_local_save_fails(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.notion.complete_task.return_value = ToolResult("Done", receipt={
        "task": "Write report", "completed": True})
    dispatcher.cognition.dismiss_task = Mock(side_effect=OSError("disk unavailable"))
    result = await dispatcher.complete_owner_tasks("Write report")
    assert not result.ok
    assert result.receipt["completed_on_board"] == ["Write report"]
    assert result.receipt["memory_problems"]


async def test_completion_rejects_oversized_batch_without_partial_silent_truncation(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    result = await dispatcher.complete_owner_tasks([f"Task {index}" for index in range(31)])
    assert not result.ok and "30" in result.text
    dispatcher.notion.complete_task.assert_not_called()
    assert dispatcher.cognition.snapshot()["dismissed_tasks"] == {}


async def test_verified_edit_reopens_task_and_survives_restart(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.dismiss_task("Old title", "completed")
    dispatcher.notion.update_task.return_value = ToolResult("Updated", receipt={
        "original_task": "Old title", "task": "New title", "completed": False,
        "status": "To do", "date": "2026-10-20", "page_id": "one"})
    result = await dispatcher.notion_workspace({"action": "update_task", "query": "Old title",
        "title": "New title", "status": "To do", "due": "2026-10-20"})
    assert result.ok
    dispatcher.notion.update_task.assert_awaited_once_with("Old title", title="New title",
        status="To do", due="2026-10-20")
    restored = CognitiveStateManager(bus(), Mock(), path=dispatcher.cognition.path)
    restored._load_locked()
    assert [row["title"] for row in restored.list_tasks()] == ["New title"]
    assert not restored.is_dismissed("New title")
    assert restored.is_dismissed("Old title")


async def test_renaming_a_locally_completed_task_preserves_completion_knowledge(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.dismiss_task("Old title", "completed")
    dispatcher.notion.update_task.return_value = ToolResult("Updated", receipt={
        "original_task": "Old title", "task": "New title", "completed": False,
        "status": "To do", "date": "", "page_id": "one"})
    assert (await dispatcher.notion_workspace({"action": "update_task", "query": "Old title",
        "title": "New title"})).ok
    assert dispatcher.cognition.list_tasks() == []
    assert dispatcher.cognition.is_dismissed("New title")


async def test_edit_board_success_memory_failure_preserves_partial_receipt(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.notion.update_task.return_value = ToolResult("Updated", receipt={
        "original_task": "Task", "task": "Task", "completed": False,
        "status": "To do", "date": "", "page_id": "one"})
    dispatcher.cognition.add_task = Mock(side_effect=OSError("disk full"))
    result = await dispatcher.notion_workspace({"action": "update_task", "query": "Task", "status": "To do"})
    assert not result.ok and result.receipt["page_id"] == "one"
    assert result.receipt["memory_problem"] == "disk full"


def bus():
    return SimpleNamespace(log=SimpleNamespace(emit=Mock()),
                           dashboard_event=SimpleNamespace(emit=Mock()))


def test_remote_only_dismissal_survives_restart(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    assert state.dismiss_task("Blog post") == "Blog post"
    assert state.list_tasks() == []
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert restored.is_dismissed("BLOG POST")
    restored.add_task("Blog post", due="2020-01-01")
    assert restored.list_tasks() == []
    assert not restored.is_dismissed("Blog post follow-up")


@pytest.mark.parametrize("titles", [
    ("A" * 90 + " first report", "A" * 90 + " second report"),
    ("Study C++", "Study C#"),
    ("東京", "大阪"),
])
def test_colliding_title_slugs_keep_independent_tasks_markers_and_references(tmp_path, titles):
    first, second = titles
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    first_id = state.add_task(first)["id"]
    second_id = state.add_task(second)["id"]
    assert first_id != second_id
    assert len(first_id) <= 80 and len(second_id) <= 80
    assert {row["title"] for row in state.list_tasks()} == set(titles)
    assert state.add_task(second)["id"] == second_id
    state.note_mentioned_tasks(list(titles), "briefing")
    assert state.complete_task(first)
    assert state.is_dismissed(first)
    assert not state.is_dismissed(second)
    assert state.last_mentioned_tasks() == [second]
    assert [row["title"] for row in state.list_tasks()] == [second]
    assert state.add_task(first)["dismissed"] is True
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert [row["title"] for row in restored.list_tasks()] == [second]
    assert restored.add_task(second)["id"] == second_id
    assert restored.is_dismissed(first) and not restored.is_dismissed(second)
    assert restored.last_mentioned_tasks() == [second]
    restored.add_task(first, restore=True)
    assert not restored.is_dismissed(first)
    assert {row["title"] for row in restored.list_tasks()} == set(titles)
    assert restored.add_task(first)["id"] == first_id


def test_task_identity_normalises_unicode_case_and_whitespace_without_changing_simple_ids(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    simple_id = state.add_task("Write report")["id"]
    assert simple_id == "write_report"
    assert state.add_task("  ＷＲＩＴＥ\u3000 report  ")["id"] == simple_id
    assert len(state.list_tasks()) == 1
    assert state.complete_task("write  REPORT")
    assert state.is_dismissed("  WRITE\tREPORT ")
    assert not state.is_dismissed("write_report")
    assert state.add_task("Write report")["dismissed"] is True
    state.add_task("  Write   report ", restore=True)
    assert len(state.list_tasks()) == 1 and not state.is_dismissed("Write report")


@pytest.mark.parametrize("operation", ["complete_task", "remove_task"])
def test_ambiguous_partial_task_name_never_removes_an_arbitrary_task(tmp_path, operation):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Write report")
    state.add_task("Review report")
    assert not getattr(state, operation)("report")
    assert {row["title"] for row in state.list_tasks()} == {"Write report", "Review report"}
    assert state.snapshot()["dismissed_tasks"] == {}


def test_exact_title_wins_over_partial_match_and_gui_ids_still_select_exactly(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Report follow-up")
    state.add_task("Report")
    gui_id = state.add_task("Write report")["id"]
    assert state.complete_task("REPORT")
    assert {row["title"] for row in state.list_tasks()} == {"Report follow-up", "Write report"}
    assert state.complete_task(gui_id, exact=True)
    assert [row["title"] for row in state.list_tasks()] == ["Report follow-up"]


def test_legacy_colliding_dismissal_migration_preserves_other_live_title_and_marker_bytes(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    marker = {"title": "Study C++", "reason": "completed", "at": "2026-01-01T00:00:00Z"}
    live = {"id": "study_c", "title": "Study C#", "status": "pending"}
    state.path.write_text(json.dumps({
        "pending_tasks": {"study_c": live},
        "dismissed_tasks": {"study_c": marker},
        "last_mentioned_tasks": {"titles": ["Study C++", "Study C#"], "at": utc_stamp()},
    }), encoding="utf-8")
    state._load_locked()
    assert state.snapshot()["pending_tasks"] == {"study_c": live}
    assert list(state.snapshot()["dismissed_tasks"].values()) == [marker]
    assert "study_c" not in state.snapshot()["dismissed_tasks"]
    assert state.is_dismissed("Study C++") and not state.is_dismissed("Study C#")
    assert state.last_mentioned_tasks() == ["Study C#"]
    assert state.add_task("Study C#")["id"] == "study_c"
    state.add_task("Study C++", restore=True)
    assert {row["title"] for row in state.list_tasks()} == {"Study C++", "Study C#"}
    assert not state.is_dismissed("Study C++")
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert {row["title"] for row in restored.list_tasks()} == {"Study C++", "Study C#"}


def test_legacy_terminal_rows_with_same_slug_create_separate_minimal_markers(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.path.write_text(json.dumps({"pending_tasks": {
        "old-id-one": {"title": "Study C++", "status": "done"},
        "old-id-two": {"title": "Study C#", "status": "done"},
    }}), encoding="utf-8")
    state._load_locked()
    assert state.list_tasks() == []
    assert len(state.snapshot()["dismissed_tasks"]) == 2
    assert state.is_dismissed("Study C++") and state.is_dismissed("Study C#")
    assert all(set(marker) == {"title", "reason", "at"}
               for marker in state.snapshot()["dismissed_tasks"].values())
    state.add_task("Study C#", restore=True)
    assert [row["title"] for row in state.list_tasks()] == ["Study C#"]
    assert state.is_dismissed("Study C++") and not state.is_dismissed("Study C#")


def test_remote_only_marker_cannot_remove_or_suppress_a_colliding_local_task(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Study C#")
    state.dismiss_task("Study C++", "completed by user")
    assert [row["title"] for row in state.list_tasks()] == ["Study C#"]
    assert state.is_dismissed("Study C++") and not state.is_dismissed("Study C#")
    state.note_mentioned_tasks(["Study C++", "Study C#"], "briefing")
    assert state.last_mentioned_tasks() == ["Study C#"]


def test_cognitive_save_retries_transient_replace_locks_and_keeps_other_temp_files(tmp_path, monkeypatch):
    signals = bus()
    telemetry = SimpleNamespace(health=SimpleNamespace(register=Mock()),
                                metrics=SimpleNamespace(incr=Mock()))
    state = CognitiveStateManager(signals, Mock(), path=tmp_path / "state.json", telemetry=telemetry)
    state.add_task("Original task")
    signals.dashboard_event.emit.reset_mock()
    telemetry.metrics.incr.reset_mock()
    other_temp = state.path.with_suffix(".tmp")
    other_temp.write_text("another writer owns this file", encoding="utf-8")
    real_replace = atomic_io.os.replace
    attempts = []
    pauses = []

    def temporarily_locked(source, target):
        if target == state.path:
            attempts.append(source)
            if len(attempts) < 3:
                raise PermissionError("transient OneDrive sharing lock")
        return real_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", temporarily_locked)
    monkeypatch.setattr(atomic_io.time, "sleep", pauses.append)
    state.add_task("Saved after retry")
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert pauses == [atomic_io.REPLACE_BACKOFF_S, atomic_io.REPLACE_BACKOFF_S * 2]
    stored = json.loads(state.path.read_text(encoding="utf-8"))
    assert set(stored["pending_tasks"]) == {"original_task", "saved_after_retry"}
    assert other_temp.read_text(encoding="utf-8") == "another writer owns this file"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []
    telemetry.metrics.incr.assert_called_once_with("cognition.state_saves")
    signals.dashboard_event.emit.assert_called_once()


def test_cognitive_save_permanent_lock_preserves_file_and_cleans_temp_without_success_event(tmp_path, monkeypatch):
    signals = bus()
    telemetry = SimpleNamespace(health=SimpleNamespace(register=Mock()),
                                metrics=SimpleNamespace(incr=Mock()))
    state = CognitiveStateManager(signals, Mock(), path=tmp_path / "state.json", telemetry=telemetry)
    state.add_task("Keep original task")
    before = state.path.read_bytes()
    before_state = state.snapshot()
    signals.dashboard_event.emit.reset_mock()
    telemetry.metrics.incr.reset_mock()
    real_replace = atomic_io.os.replace
    attempts = []
    pauses = []

    def persistently_locked(source, target):
        if target == state.path:
            attempts.append(source)
            raise PermissionError("persistent sharing lock")
        return real_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", persistently_locked)
    monkeypatch.setattr(atomic_io.time, "sleep", pauses.append)
    with pytest.raises(PermissionError, match="persistent sharing lock"):
        state.add_task("Uncommitted task")
    assert len(attempts) == atomic_io.REPLACE_ATTEMPTS
    assert len(pauses) == atomic_io.REPLACE_ATTEMPTS - 1
    assert state.path.read_bytes() == before
    assert state.snapshot() == before_state
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".state.json.*.tmp")) == []
    telemetry.metrics.incr.assert_not_called()
    signals.dashboard_event.emit.assert_not_called()
    monkeypatch.setattr(atomic_io.os, "replace", real_replace)
    state.add_priority("A later successful save")
    assert "uncommitted_task" not in json.loads(state.path.read_text(encoding="utf-8"))["pending_tasks"]


def test_failed_completion_restores_pending_row_mentions_and_dismissals(tmp_path, monkeypatch):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Still pending")
    state.note_mentioned_tasks(["Still pending"], "briefing")
    before, before_bytes = state.snapshot(), state.path.read_bytes()
    state.bus.dashboard_event.emit.reset_mock()
    with monkeypatch.context() as locked:
        locked.setattr(cognition_mod, "atomic_write_text", Mock(side_effect=PermissionError("locked")))
        with pytest.raises(PermissionError, match="locked"):
            state.complete_task("Still pending")
    assert state.snapshot() == before
    assert state.path.read_bytes() == before_bytes
    assert state.last_mentioned_tasks() == ["Still pending"]
    assert not state.is_dismissed("Still pending")
    state.bus.dashboard_event.emit.assert_not_called()
    state.add_priority("Next successful operation")
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert [row["title"] for row in restored.list_tasks()] == ["Still pending"]
    assert not restored.is_dismissed("Still pending")


def test_mutator_exception_rolls_back_nested_state_without_save(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Keep task")
    before, before_bytes = state.snapshot(), state.path.read_bytes()
    state.bus.dashboard_event.emit.reset_mock()

    def partially_mutate_then_fail(data):
        data["pending_tasks"]["keep_task"]["title"] = "Failed edit"
        data["user_priorities"].append("Failed priority")
        raise ValueError("invalid operation")

    with pytest.raises(ValueError, match="invalid operation"):
        state._mutate(partially_mutate_then_fail)
    assert state.snapshot() == before
    assert state.path.read_bytes() == before_bytes
    state.bus.dashboard_event.emit.assert_not_called()


@pytest.mark.parametrize("load_method", ["_load_locked", "_restore_sync"])
def test_unreadable_cognitive_state_never_becomes_default_or_corrupt(tmp_path, monkeypatch, load_method):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Keep history")
    before, before_bytes = state.snapshot(), state.path.read_bytes()
    original_read = type(state.path).read_text
    state.bus.dashboard_event.emit.reset_mock()

    def locked_read(path, *args, **kwargs):
        if path == state.path:
            raise PermissionError("temporary read lock")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(type(state.path), "read_text", locked_read)
    with pytest.raises(PermissionError, match="temporary read lock"):
        getattr(state, load_method)()
    assert state.snapshot() == before
    assert state.path.read_bytes() == before_bytes
    assert list(tmp_path.glob("*.corrupt-*.json")) == []
    state.bus.dashboard_event.emit.assert_not_called()


def test_corrupt_state_backup_failure_cannot_overwrite_history(tmp_path, monkeypatch):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Retain memory")
    before = state.snapshot()
    state.path.write_bytes(b'{"pending_tasks": {"recoverable": ')
    before_bytes = state.path.read_bytes()
    original_replace = type(state.path).replace

    def locked_backup(path, target):
        if path == state.path:
            raise PermissionError("backup locked")
        return original_replace(path, target)

    monkeypatch.setattr(type(state.path), "replace", locked_backup)
    with pytest.raises(PermissionError, match="backup locked"):
        state._load_locked()
    assert state.snapshot() == before
    assert state.path.read_bytes() == before_bytes
    assert list(tmp_path.glob("*.corrupt-*.json")) == []


def test_corrupt_state_recovers_only_after_retaining_exact_bytes(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    before_bytes = b'{"pending_tasks": {"recoverable": '
    state.path.write_bytes(before_bytes)
    state._load_locked()
    backups = list(tmp_path.glob("*.corrupt-*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == before_bytes
    assert json.loads(state.path.read_text(encoding="utf-8"))["pending_tasks"] == {}


def test_failed_startup_pruning_preserves_loaded_memory_and_can_retry(tmp_path, monkeypatch):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    stored = {"pending_tasks": {"done": {"title": "Finished", "status": "done"}},
              "last_mentioned_tasks": {"titles": ["Finished"], "at": utc_stamp()}}
    state.path.write_text(json.dumps(stored), encoding="utf-8")
    before_bytes = state.path.read_bytes()
    with monkeypatch.context() as locked:
        locked.setattr(cognition_mod, "atomic_write_text", Mock(side_effect=PermissionError("migration locked")))
        with pytest.raises(PermissionError, match="migration locked"):
            state._load_locked()
    assert state.snapshot()["pending_tasks"] == stored["pending_tasks"]
    assert state.snapshot()["last_mentioned_tasks"] == stored["last_mentioned_tasks"]
    assert state.snapshot()["dismissed_tasks"] == {}
    assert state.path.read_bytes() == before_bytes
    state._load_locked()
    assert state.list_tasks() == [] and state.is_dismissed("Finished")
    assert state.last_mentioned_tasks() == []


def test_startup_completion_knowledge_survives_more_than_five_hundred_tasks(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.path.write_text(json.dumps({"pending_tasks": {
        f"legacy-{index}": {"title": f"Finished {index}", "status": "done"}
        for index in range(501)
    }}), encoding="utf-8")
    state._load_locked()
    assert state.snapshot()["pending_tasks"] == {}
    assert len(state.snapshot()["dismissed_tasks"]) == 501
    assert all(state.is_dismissed(f"Finished {index}") for index in range(501))
    state.add_task("Finished 0")
    assert state.list_tasks() == []
    state.dismiss_task("New completion", "completed by user")
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert len(restored.snapshot()["dismissed_tasks"]) == 502
    assert restored.is_dismissed("Finished 0")
    assert restored.is_dismissed("New completion")


def test_optional_save_metric_failure_does_not_undo_a_committed_task(tmp_path):
    telemetry = SimpleNamespace(health=SimpleNamespace(register=Mock()),
                                metrics=SimpleNamespace(incr=Mock(side_effect=RuntimeError("metric offline"))))
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json", telemetry=telemetry)
    state.add_task("Committed task")
    assert [row["title"] for row in state.list_tasks()] == ["Committed task"]
    assert "committed_task" in json.loads(state.path.read_text(encoding="utf-8"))["pending_tasks"]
    state.bus.dashboard_event.emit.assert_called_once()


@pytest.mark.parametrize("status", [
    "done", "Complete", " COMPLETED ", "finished", "closed", "resolved",
    "cancelled", "canceled", "archived", "removed", "deleted",
])
def test_launch_physically_removes_legacy_terminal_tasks_and_references(tmp_path, status):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "pending_tasks": {
            "legacy-id": {"title": "Finished report", "status": status,
                          "notes": "Old completed row details", "due": "2020-01-01"},
            "open": {"id": "open", "title": "Still open", "status": "pending"},
        },
        "last_mentioned_tasks": {"titles": ["Finished report", "Still open"],
                                 "source": "briefing", "at": utc_stamp()},
    }), encoding="utf-8")
    state = CognitiveStateManager(bus(), Mock(), path=path)
    summary = state._restore_sync()
    assert [task["title"] for task in state.list_tasks()] == ["Still open"]
    assert set(state.snapshot()["pending_tasks"]) == {"open"}
    assert "Pending tasks: 1." in summary
    assert state.last_mentioned_tasks() == ["Still open"]
    assert state.is_dismissed("Finished report")
    marker = state.snapshot()["dismissed_tasks"]["finished_report"]
    assert set(marker) == {"title", "reason", "at"}
    assert marker["title"] == "Finished report"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored["pending_tasks"]) == {"open"}
    assert stored["last_mentioned_tasks"]["titles"] == ["Still open"]
    assert "Old completed row details" not in path.read_text(encoding="utf-8")
    state.add_task("Finished report")
    assert [task["title"] for task in state.list_tasks()] == ["Still open"]
    restored = CognitiveStateManager(bus(), Mock(), path=path)
    restored._load_locked()
    assert restored.snapshot()["dismissed_tasks"]["finished_report"] == marker
    assert set(restored.snapshot()["pending_tasks"]) == {"open"}


@pytest.mark.parametrize("flag", ["done", "completed", "terminal", "archived", "cancelled", "canceled"])
def test_launch_retires_legacy_checked_tasks_with_pending_status(tmp_path, flag):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"pending_tasks": {
        "legacy": {"title": "Checked task", "status": "pending", flag: True},
    }}), encoding="utf-8")
    state = CognitiveStateManager(bus(), Mock(), path=path)
    summary = state._restore_sync()
    assert state.snapshot()["pending_tasks"] == {}
    assert state.list_tasks() == []
    assert "Pending tasks" not in summary
    assert state.is_dismissed("Checked task")
    assert json.loads(path.read_text(encoding="utf-8"))["pending_tasks"] == {}


@pytest.mark.parametrize("status", [
    "pending", "open", "in progress", "blocked", "waiting", "not completed",
    "done later", "delivered to reviewer",
])
def test_launch_does_not_infer_completion_from_open_or_unknown_status(tmp_path, status):
    path = tmp_path / "state.json"
    row = {"title": "Keep this task", "status": status, "done": "false", "completed": False}
    path.write_text(json.dumps({"pending_tasks": {"keep": row}}), encoding="utf-8")
    state = CognitiveStateManager(bus(), Mock(), path=path)
    state._load_locked()
    assert state.snapshot()["pending_tasks"] == {"keep": row}
    assert not state.is_dismissed("Keep this task")


def test_launch_removes_pending_copy_of_previously_dismissed_task(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.dismiss_task("Already finished", "completed by user")
    stored = state.snapshot()
    stored["pending_tasks"]["legacy"] = {"title": "Already finished", "status": "pending"}
    stored["last_mentioned_tasks"] = {"titles": ["Already finished"], "at": utc_stamp()}
    state.path.write_text(json.dumps(stored), encoding="utf-8")
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert restored.snapshot()["pending_tasks"] == {}
    assert restored.last_mentioned_tasks() == []
    assert json.loads(state.path.read_text(encoding="utf-8"))["last_mentioned_tasks"]["titles"] == []


def test_completing_local_task_removes_row_and_reference_across_restarts(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Finish report")
    state.add_task("Next task")
    state.note_mentioned_tasks(["Finish report", "Next task"], "list_tasks")
    assert state.complete_task("Finish report")
    assert state.last_mentioned_tasks() == ["Next task"]
    state.note_mentioned_tasks(["Finish report", "Next task"], "stale listing")
    assert state.last_mentioned_tasks() == ["Next task"]
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._restore_sync()
    assert [task["title"] for task in restored.list_tasks()] == ["Next task"]
    assert set(restored.snapshot()["pending_tasks"]) == {"next_task"}
    assert restored.last_mentioned_tasks() == ["Next task"]
    assert json.loads(state.path.read_text(encoding="utf-8"))["pending_tasks"].keys() == {"next_task"}
    assert restored.add_task("Finish report")["dismissed"] is True


def test_explicit_restore_clears_legacy_completion_flags_and_survives_launch(tmp_path):
    state = CognitiveStateManager(bus(), Mock(), path=tmp_path / "state.json")
    state.add_task("Bring back task")
    state._state["pending_tasks"]["bring_back_task"].update({
        "done": True, "completed": True, "completed_at": utc_stamp()})
    state._state["dismissed_tasks"]["bring_back_task"] = {
        "title": "Bring back task", "reason": "completed", "at": utc_stamp()}
    state.add_task("Bring back task", restore=True)
    restored = CognitiveStateManager(bus(), Mock(), path=state.path)
    restored._load_locked()
    assert [task["title"] for task in restored.list_tasks()] == ["Bring back task"]
    assert not restored.is_dismissed("Bring back task")
    row = restored.snapshot()["pending_tasks"]["bring_back_task"]
    assert "done" not in row and "completed" not in row and "completed_at" not in row


async def test_archive_refuses_duplicate_titles():
    service = NotionService(bus(), {"token": "test", "tasks_database_id": "db"})
    service._database_schema = AsyncMock(return_value={"title": "Name"})
    service._query_database = AsyncMock(return_value=[
        {"title": "Gym", "id": "one"}, {"title": "Gym", "id": "two"},
    ])
    service._request = AsyncMock()
    assert not (await service.archive_task_exact("Gym")).ok
    service._request.assert_not_called()


async def test_archive_is_exact_and_requires_server_confirmation():
    service = NotionService(bus(), {"token": "test", "tasks_database_id": "db"})
    service._database_schema = AsyncMock(return_value={"title": "Name"})
    service._query_database = AsyncMock(return_value=[{"title": "Gym buffer", "id": "wrong"}])
    service._request = AsyncMock()
    await service.archive_task_exact("Gym")
    service._request.assert_not_called()
    service._query_database = AsyncMock(return_value=[{"title": "Gym", "id": "one"}])
    service._request = AsyncMock(return_value={"archived": False})
    assert not (await service.archive_task_exact("Gym")).ok
    service._request = AsyncMock(return_value={"archived": True})
    assert (await service.archive_task_exact("Gym")).ok
    service._request.assert_awaited_once_with("PATCH", "/pages/one", {"archived": True})


async def test_retirement_collects_pages_before_archiving_and_filters_exact_titles():
    service = NotionService(bus(), {"token": "test", "tasks_database_id": "db"})
    schema = {"title": "Name", "date": "Due", "status": "", "status_kind": ""}
    service._database_schema = AsyncMock(return_value=schema)
    def page(title, key):
        return {"id": key, "properties": {"Name": {"title": [{"plain_text": title}]}}}
    service._request = AsyncMock(side_effect=[
        {"results": [page("Gym", "one"), page("Gym buffer", "wrong")],
         "has_more": True, "next_cursor": "next"},
        {"results": [page("Gym", "two")], "has_more": False},
        {"archived": True}, {"archived": True},
    ])
    result = await service.archive_overdue_tasks_exact("Gym")
    assert result.ok and "2 overdue" in result.text
    calls = service._request.await_args_list
    assert [c.args[0] for c in calls] == ["POST", "POST", "PATCH", "PATCH"]
    assert calls[2].args[1] == "/pages/one" and calls[3].args[1] == "/pages/two"
    assert calls[0].args[2]["filter"]["and"][0]["title"] == {"equals": "Gym"}
    assert "before" in calls[0].args[2]["filter"]["and"][1]["date"]


def notion_board(kind="status", options=None, groups=None):
    properties = {
        "Name": {"type": "title", "title": {}},
        # A category preceding Status used to be selected as the task state.
        "Category": {"type": "select", "select": {"options": [{"name": "Work"}]}},
        "Created": {"type": "date", "date": {}},
        "Due": {"type": "date", "date": {}},
    }
    field = "Done" if kind == "checkbox" else "Status"
    properties[field] = {"type": kind, kind: {} if kind == "checkbox" else {
        "options": options if options is not None else [
            {"id": "todo", "name": "To do"}, {"id": "done", "name": "Done"}],
        "groups": groups or [],
    }}
    return {"properties": properties}


def notion_page(title, key="one", status="To do", kind="status", due=None):
    field = "Done" if kind == "checkbox" else "Status"
    return {"id": key, "url": f"https://notion.example/{key}", "properties": {
        "Name": {"title": [{"plain_text": title}]},
        field: {kind: status if kind == "checkbox" else {"name": status}},
        "Due": {"date": {"start": due} if due else None},
    }}


def notion_service(*responses):
    service = NotionService(bus(), {"token": "synthetic", "tasks_database_id": "db"})
    service.tasks_db = "db"
    service._request = AsyncMock(side_effect=responses)
    return service


async def test_board_schema_prefers_task_status_and_due_date():
    service = notion_service(notion_board())
    schema = await service._database_schema("db")
    assert schema["status"] == "Status"
    assert schema["date"] == "Due"
    assert schema["done_options"] == ["Done"]


async def test_custom_completed_status_is_excluded_and_pagination_finds_open_rows():
    schema = notion_board(options=[{"id": "todo", "name": "To do"}, {"id": "sent", "name": "Delivered"}],
                          groups=[{"name": "Complete", "option_ids": ["sent"]}])
    service = notion_service(schema, {
        "results": [notion_page("Finished work", status="Delivered")],
        "has_more": True, "next_cursor": "second-page",
    }, {"results": [notion_page("Next work", key="two")], "has_more": False})
    rows = await service.open_tasks(limit=1)
    assert [row["title"] for row in rows] == ["Next work"]
    assert service._request.await_args_list[2].args[2]["start_cursor"] == "second-page"
    assert service._request.await_args_list[1].args[2]["page_size"] == 100


async def test_checkbox_board_uses_completion_instead_of_category():
    service = notion_service(notion_board(kind="checkbox"), {"results": [
        notion_page("Completed", status=True, kind="checkbox"),
        notion_page("Remaining", key="two", status=False, kind="checkbox"),
    ]})
    assert [row["title"] for row in await service.open_tasks()] == ["Remaining"]


async def test_task_read_refreshes_a_renamed_status_property():
    board = notion_board()
    board["properties"]["Progress"] = board["properties"].pop("Status")
    page = notion_page("Finished", status="Done")
    page["properties"]["Progress"] = page["properties"].pop("Status")
    service = notion_service(board, {"results": [page]})
    service._schema_cache["db"] = {"title": "Name", "status": "Status", "status_kind": "status", "date": "Due"}
    assert await service.open_tasks() == []
    assert service._schema_cache["db"]["status"] == "Progress"


@pytest.mark.parametrize("response", [
    {}, {"results": [], "has_more": True, "next_cursor": None},
])
async def test_incomplete_board_reads_cannot_claim_the_board_is_clear(response):
    service = notion_service(notion_board(), response)
    result = await service.list_tasks()
    assert not result.ok
    assert "board is clear" not in result.text


async def test_cyclic_query_cursor_fails_without_returning_partial_tasks():
    service = notion_service(notion_board(),
        {"results": [], "has_more": True, "next_cursor": "a"},
        {"results": [], "has_more": True, "next_cursor": "b"},
        {"results": [], "has_more": True, "next_cursor": "a"})
    result = await service.list_tasks()
    assert not result.ok and "pagination cursor" in result.text


async def test_completion_refuses_ambiguous_fragment_without_writing():
    service = notion_service(notion_board(), {"results": [
        notion_page("Build dashboard"), notion_page("Build reports", key="two"),
    ]})
    result = await service.complete_task("Build")
    assert not result.ok and "Multiple" in result.text
    assert all(call.args[0] != "PATCH" for call in service._request.await_args_list)


async def test_completion_of_an_already_done_exact_title_returns_verified_receipt():
    service = notion_service(notion_board(), {"results": [
        notion_page("Build", status="Done"), notion_page("Build reports", key="two"),
    ]})
    result = await service.complete_task("Build")
    assert result.ok and result.receipt["completed"] is True
    assert result.receipt["page_id"] == "one"
    assert all(call.args[0] != "PATCH" for call in service._request.await_args_list)


async def test_completion_never_guesses_the_last_status_option():
    board = notion_board(options=[{"name": "Queued"}, {"name": "In progress"}])
    service = notion_service(board, {"results": [notion_page("Build", status="Queued")]})
    result = await service.complete_task("Build")
    assert not result.ok and "completion option" in result.text
    assert all(call.args[0] != "PATCH" for call in service._request.await_args_list)


async def test_custom_completion_option_updates_actual_status_and_returns_receipt():
    board = notion_board(options=[{"id": "todo", "name": "To do"}, {"id": "sent", "name": "Delivered"}],
                         groups=[{"name": "Complete", "option_ids": ["sent"]}])
    service = notion_service(board, {"results": [notion_page("Build dashboard")]},
                             notion_page("Build dashboard", status="Delivered"))
    result = await service.complete_task("dashboard")
    assert result.ok and result.receipt["completed"] is True
    assert result.receipt["task"] == "Build dashboard"
    assert service._request.await_args_list[-1].args == ("PATCH", "/pages/one", {
        "properties": {"Status": {"status": {"name": "Delivered"}}}})


async def test_checkbox_completion_updates_boolean_and_confirms_response():
    service = notion_service(notion_board(kind="checkbox"),
        {"results": [notion_page("Build", status=False, kind="checkbox")]},
        notion_page("Build", status=True, kind="checkbox"))
    result = await service.complete_task("Build")
    assert result.ok and result.receipt["completed"] is True
    assert service._request.await_args_list[-1].args[2] == {"properties": {"Done": {"checkbox": True}}}


async def test_completion_reads_back_sparse_response_and_refuses_unconfirmed_state():
    service = notion_service(notion_board(), {"results": [notion_page("Build")]},
                             {"id": "one"}, notion_page("Build", status="To do"))
    result = await service.complete_task("Build")
    assert not result.ok and "did not confirm" in result.text
    assert result.receipt["verification_pending"] is True
    assert result.receipt["page_id"] == "one" and "completed" not in result.receipt
    assert service._request.await_args_list[-1].args == ("GET", "/pages/one")


async def test_completion_finds_matching_task_after_the_first_query_page():
    service = notion_service(notion_board(), {
        "results": [], "has_more": True, "next_cursor": "later"},
        {"results": [notion_page("Build")]}, notion_page("Build", status="Done"))
    result = await service.complete_task("Build")
    assert result.ok
    assert service._request.await_args_list[2].args[2]["start_cursor"] == "later"


async def test_update_task_confirms_rename_date_and_reopen_before_returning_receipt():
    service = notion_service(notion_board(), {"results": [notion_page("Old title", status="Done")]},
                             notion_page("New title", status="To do", due="2026-10-20"))
    result = await service.update_task("Old title", title="New title", due="2026-10-20", status="To do")
    assert result.ok
    assert result.receipt["original_task"] == "Old title"
    assert result.receipt["task"] == "New title"
    assert result.receipt["date"] == "2026-10-20"
    assert result.receipt["completed"] is False


async def test_update_task_can_clear_due_date_without_affecting_status():
    service = notion_service(notion_board(), {"results": [notion_page("Build", due="2026-10-20")]},
                             notion_page("Build"))
    result = await service.update_task("Build", due="")
    assert result.ok and result.receipt["date"] == ""
    assert service._request.await_args_list[-1].args[2] == {"properties": {"Due": {"date": None}}}


async def test_update_task_rejects_unknown_status_without_creating_an_option():
    service = notion_service(notion_board())
    result = await service.update_task("Build", status="Imaginary option")
    assert not result.ok and "existing task status" in result.text
    assert service._request.await_count == 1


async def test_clearing_due_date_requires_the_property_in_the_confirmed_page():
    response = notion_page("Build")
    response["properties"].pop("Due")
    service = notion_service(notion_board(), {"results": [notion_page("Build")]}, response, response)
    result = await service.update_task("Build", due="")
    assert not result.ok and "did not return" in result.text


async def test_create_task_refuses_to_silently_discard_a_requested_due_date():
    board = notion_board()
    board["properties"].pop("Due")
    board["properties"].pop("Created")
    service = notion_service(board)
    result = await service.create_task("Build", due="2026-10-20")
    assert not result.ok and "no due-date property" in result.text
    assert service._request.await_count == 1


async def test_completion_never_uses_cancelled_as_a_completion_option():
    board = notion_board(options=[{"id": "todo", "name": "To do"},
                                 {"id": "cancel", "name": "Cancelled"}],
                         groups=[{"name": "Complete", "option_ids": ["cancel"]}])
    service = notion_service(board, {"results": [notion_page("Build")]})
    result = await service.complete_task("Build")
    assert not result.ok
    assert all(call.args[0] != "PATCH" for call in service._request.await_args_list)


async def test_cancelled_task_is_terminal_without_being_reported_completed():
    board = notion_board(options=[{"id": "cancel", "name": "Cancelled"}])
    service = notion_service(board, {"results": [notion_page("Build", status="Cancelled")]})
    result = await service.complete_task("Build")
    assert not result.ok and result.receipt["terminal"] is True
    assert result.receipt["completed"] is False
    assert all(call.args[0] != "PATCH" for call in service._request.await_args_list)


async def test_custom_completion_group_ignores_cancelled_target():
    board = notion_board(options=[{"id": "todo", "name": "To do"},
                                 {"id": "cancel", "name": "Cancelled"},
                                 {"id": "sent", "name": "Delivered"}],
                         groups=[{"name": "Complete", "option_ids": ["cancel", "sent"]}])
    service = notion_service(board, {"results": [notion_page("Build")]},
                             notion_page("Build", status="Delivered"))
    result = await service.complete_task("Build")
    assert result.ok and result.receipt["completed"] is True
    assert service._request.await_args_list[-1].args[2]["properties"] == {
        "Status": {"status": {"name": "Delivered"}}}


async def test_cancelled_and_archived_tasks_stay_out_of_open_listing():
    service = notion_service(notion_board(), {"results": [
        notion_page("Cancelled task", status="Cancelled"),
        notion_page("Archived task", key="two", status="Archived"),
        notion_page("Open task", key="three", status="To do")]})
    assert [row["title"] for row in await service.open_tasks()] == ["Open task"]


async def test_cancelling_task_does_not_reopen_it_in_local_memory(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.cognition.add_task("Task")
    dispatcher.notion.update_task.return_value = ToolResult("Cancelled", receipt={
        "original_task": "Task", "task": "Task", "completed": False,
        "terminal": True, "status": "Cancelled", "date": "", "page_id": "one"})
    result = await dispatcher.notion_workspace({"action": "update_task", "query": "Task", "status": "Cancelled"})
    assert result.ok
    assert dispatcher.cognition.list_tasks() == []
    assert dispatcher.cognition.is_dismissed("Task")


async def test_explicit_completion_checkbox_wins_over_progress_select():
    board = notion_board(kind="checkbox")
    board["properties"] = {"Progress": {"type": "select", "select": {"options": [
        {"name": "Not started"}, {"name": "In progress"}, {"name": "Complete"}]}},
        **board["properties"]}
    service = notion_service(board, {"results": [notion_page("Build", status=False, kind="checkbox")]},
                             notion_page("Build", status=True, kind="checkbox"))
    result = await service.complete_task("Build")
    assert result.ok and result.receipt["completed"] is True
    assert service._request.await_args_list[-1].args[2]["properties"] == {"Done": {"checkbox": True}}


def test_schema_exposes_verified_task_edit_and_explicit_mission_clear_all():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tools = {tool["name"]: tool for tool in TOOL_DECLARATIONS}
    properties = tools["notion_workspace"]["parameters"]["properties"]
    assert {"query", "title", "due", "status", "include_done"} <= properties.keys()
    assert "update_task" in properties["action"]["description"]
    assert "clear_all" in tools["mission"]["parameters"]["properties"]["action"]["description"]


async def test_inspection_of_completed_tasks_does_not_label_them_open():
    service = notion_service(notion_board(), {"results": [notion_page("Build", status="Done")]})
    result = await service.list_tasks(include_done=True)
    assert result.ok and "Tasks — 1" in result.text
    assert "Open tasks" not in result.text


async def test_completion_retains_uncertain_remote_identity_without_claiming_completion(tmp_path):
    dispatcher = task_dispatch(tmp_path)
    dispatcher.notion.complete_task.return_value = ToolResult("Accepted; final state unverified", ok=False,
        receipt={"page_id": "one", "original_task": "Write report", "requested_task": "Write report",
                 "board_write_accepted": True, "verification_pending": True})
    result = await dispatcher.complete_owner_tasks("Write report")
    assert not result.ok and result.receipt["completed_on_board"] == []
    assert result.receipt["recorded_done"] == []
    assert result.receipt["unverified_on_board"][0]["page_id"] == "one"
    assert not dispatcher.cognition.is_dismissed("Write report")
