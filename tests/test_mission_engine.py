"""
MissionEngine tests — the mission-based operating model.

All offline: stub bus, temporary JSON store, no Qt.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import missions as missions_mod
from orion_core import user_profile
from orion_core.missions import MissionEngine

# The starter board is operator-configurable (config/profile.json), so asserting
# against whatever this machine happens to carry would make the suite pass in CI
# and fail on a developer's box.  These tests pin the *publishable* defaults and
# the fixture below forces the engine to seed from them.
DEFAULT_MISSIONS = tuple(
    (str(name), str(brief)) for name, brief in user_profile.DEFAULTS["missions"]
)


@pytest.fixture(autouse=True)
def _pinned_default_missions(monkeypatch):
    monkeypatch.setattr(missions_mod.user_profile, "missions", lambda: DEFAULT_MISSIONS)


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))

    def connect(self, *_):
        pass


class StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


@pytest.fixture()
def engine(tmp_path):
    return MissionEngine(StubBus(), path=tmp_path / "missions.json")


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_default_missions_seed_once(engine, tmp_path):
    board = engine.overview().text
    for name, _brief in DEFAULT_MISSIONS:
        assert name in board
    assert "Develop ORION" in board
    # Re-construction over the same file must not re-impose defaults.
    engine.create("Extra Mission")
    again = MissionEngine(StubBus(), path=tmp_path / "missions.json")
    assert "Extra Mission" in again.overview().text


def test_an_explicitly_empty_board_does_not_resurrect_starters(tmp_path):
    path = tmp_path / "missions.json"
    stored = {"current": "", "missions": {}, "operator_note": "cleared deliberately"}
    path.write_text(json.dumps(stored), encoding="utf-8")
    engine = MissionEngine(StubBus(), path=path)
    assert engine.load() == stored
    assert engine.panel_snapshot() == {"current": "", "missions": []}
    assert "No missions" in engine.overview().text


def test_deleting_the_last_mission_stays_empty_after_restart(engine):
    for name in list(engine.load()["missions"]):
        assert engine.delete(name).ok
    again = MissionEngine(StubBus(), path=engine.path)
    assert again.load() == {"current": "", "missions": {}}
    assert again.current() is None


@pytest.mark.parametrize("stored", ["", "{broken", "[]", "{}"])
def test_startup_preserves_existing_mission_storage_for_recovery(tmp_path, stored):
    path = tmp_path / "missions.json"
    path.write_text(stored, encoding="utf-8")
    MissionEngine(StubBus(), path=path)
    assert path.read_text(encoding="utf-8") == stored


def test_a_profile_without_starters_can_seed_an_empty_board(tmp_path, monkeypatch):
    monkeypatch.setattr(missions_mod.user_profile, "missions", lambda: ())
    engine = MissionEngine(StubBus(), path=tmp_path / "missions.json")
    assert engine.load() == {"current": "", "missions": {}}


def test_clear_all_reports_removed_work_and_persists_empty_board(engine):
    engine.add_task("Develop ORION", "Remove stale mission")
    engine.add_goal("Develop ORION", "Keep it removed")
    count = len(engine.load()["missions"])
    result = engine.clear_all()
    assert result.ok
    assert f"Deleted all {count} mission(s)" in result.text
    assert "1 goal(s) and 1 task(s)" in result.text
    assert engine.load() == {"current": "", "missions": {}}
    assert engine.bus.emitted[-1] == (
        "dashboard_event", ("mission", {"name": "*", "status": "deleted"}))
    assert MissionEngine(path=engine.path).panel_snapshot()["missions"] == []
    assert "No missions" in engine.clear_all().text


@pytest.mark.parametrize("stored", [
    "", "{broken", "[]", "{}", '{"missions": []}',
    '{"missions": {"Keep": null}}',
    '{"missions": {"Keep": {"tasks": "recover this"}}}',
])
def test_bulk_removal_refuses_invalid_storage_without_overwriting(tmp_path, stored):
    path = tmp_path / "missions.json"
    path.write_text(stored, encoding="utf-8")
    engine = MissionEngine(StubBus(), path=path)
    result = engine.clear_all()
    assert not result.ok
    assert "No missions were deleted" in result.text
    assert path.read_text(encoding="utf-8") == stored
    assert engine.bus.emitted == []


def test_bulk_removal_refuses_unreadable_storage_without_overwriting(engine, monkeypatch):
    before = engine.path.read_bytes()
    original_read_text = Path.read_text

    def locked_read(path, *args, **kwargs):
        if path == engine.path:
            raise PermissionError("storage is temporarily locked")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", locked_read)
    result = engine.clear_all()
    assert not result.ok
    assert "could not be read" in result.text
    assert engine.path.read_bytes() == before
    assert engine.bus.emitted == []


_MISSION_MUTATIONS = [
    {"action": "create", "mission": "Fresh"},
    {"action": "current", "mission": "Develop ORION"},
    {"action": "goal", "mission": "Develop ORION", "text": "A goal"},
    {"action": "task", "mission": "Develop ORION", "text": "A task"},
    {"action": "complete", "mission": "Develop ORION", "text": "Old task"},
    {"action": "attach", "mission": "Develop ORION", "kind": "note", "text": "A note"},
    {"action": "remove_item", "mission": "Develop ORION", "text": "Old task"},
    {"action": "delete_mission", "mission": "Develop ORION"},
    {"action": "archive", "mission": "Develop ORION"},
    {"action": "clear_completed", "mission": "Develop ORION"},
    {"action": "clear_completed"},
    {"action": "clear_all"},
]


@pytest.mark.parametrize("args", _MISSION_MUTATIONS, ids=lambda args: args["action"] + args.get("mission", "_all"))
def test_every_mission_mutation_refuses_locked_reads_without_overwrite(engine, monkeypatch, args):
    engine.add_task("Develop ORION", "Old task")
    engine.complete("Develop ORION", "Old task")
    before = engine.path.read_bytes()
    engine.bus.emitted.clear()
    original_read = Path.read_text

    def locked_read(path, *positional, **keywords):
        if path == engine.path:
            raise PermissionError("temporarily unreadable mission board")
        return original_read(path, *positional, **keywords)

    monkeypatch.setattr(Path, "read_text", locked_read)
    result = asyncio.run(engine.handle(args))
    assert not result.ok and "could not be read" in result.text
    assert engine.path.read_bytes() == before
    assert engine.bus.emitted == []


@pytest.mark.parametrize("args", _MISSION_MUTATIONS, ids=lambda args: args["action"] + args.get("mission", "_all"))
@pytest.mark.parametrize("stored", [
    "{broken", "[]", "{}", '{"missions": null}',
    '{"missions": {"Keep": {"tasks": [42]}}}',
    '{"missions": {"Keep": {"notes": "retain this malformed history"}}}',
])
def test_every_mission_mutation_preserves_malformed_history(engine, args, stored):
    engine.path.write_text(stored, encoding="utf-8")
    result = asyncio.run(engine.handle(args))
    assert not result.ok and "storage" in result.text
    assert engine.path.read_text(encoding="utf-8") == stored
    assert engine.bus.emitted == []


@pytest.mark.parametrize("action", ["clear_all", "delete_all", "remove_all"])
def test_explicit_bulk_removal_actions_clear_every_mission(engine, action):
    assert asyncio.run(engine.handle({"action": action})).ok
    assert engine.load()["missions"] == {}


def test_new_mission_on_empty_board_becomes_current_and_updates_deck(engine):
    engine.clear_all()
    assert engine.create("Fresh start").ok
    assert engine.current()["name"] == "Fresh start"
    assert engine.bus.emitted[-1] == (
        "dashboard_event", ("mission", {"name": "Fresh start", "status": "created"}))


def test_removal_uses_case_insensitive_exact_name_before_partial_matches(engine):
    engine.create("Upgrade camera")
    engine.create("Upgrade")
    assert engine.delete("upgrade").ok
    assert "Upgrade camera" in engine.load()["missions"]
    assert "Upgrade" not in engine.load()["missions"]


def test_ambiguous_mission_removal_changes_nothing(engine):
    engine.create("Upgrade camera")
    engine.create("Upgrade memory")
    before = engine.path.read_bytes()
    result = engine.delete("Upgrade")
    assert not result.ok
    assert "More than one" in result.text
    assert "Upgrade camera" in result.text and "Upgrade memory" in result.text
    assert engine.path.read_bytes() == before
    assert not engine.add_task("Upgrade", "Do not guess the destination").ok
    assert engine.path.read_bytes() == before


def test_archiving_hides_mission_but_preserves_history_and_updates_current(engine):
    engine.add_task("Develop ORION", "Keep historical task")
    assert engine.archive("Develop ORION").ok
    assert "Develop ORION" not in {row["name"] for row in engine.panel_snapshot()["missions"]}
    assert "Develop ORION" not in engine.overview().text
    record = engine.load()["missions"]["Develop ORION"]
    assert record["status"] == "archived"
    assert record["tasks"][0]["task"] == "Keep historical task"
    assert engine.current()["name"] != "Develop ORION"
    assert engine.status("Develop ORION").ok
    assert engine.bus.emitted[-1] == (
        "dashboard_event", ("mission", {"name": "Develop ORION", "status": "archived"}))


def test_archived_mission_cannot_be_selected_as_current(engine):
    engine.archive("Develop ORION")
    before = engine.path.read_bytes()
    events_before = list(engine.bus.emitted)
    result = engine.set_current("Develop ORION")
    assert not result.ok
    assert "archived" in result.text
    assert engine.path.read_bytes() == before
    assert engine.bus.emitted == events_before
    assert engine.current()["name"] != "Develop ORION"


@pytest.mark.parametrize("action", ["add_task", "add_goal"])
@pytest.mark.parametrize("text", ["", "   ", None])
def test_blank_task_and_goal_are_rejected_without_mutation(engine, action, text):
    before = engine.path.read_bytes()
    result = getattr(engine, action)("Develop ORION", text)
    assert not result.ok
    assert engine.path.read_bytes() == before
    assert engine.bus.emitted == []


@pytest.mark.parametrize("operation", ["complete", "remove_item"])
def test_failed_item_match_preserves_activity_timestamp_stall_risk_and_events(engine, monkeypatch, operation):
    engine.add_task("Develop ORION", "Existing task")
    data = engine.load()
    data["missions"]["Develop ORION"]["updated_at"] = "2000-01-01T00:00:00Z"
    engine._save(data)
    before = engine.path.read_bytes()
    engine.bus.emitted.clear()
    monkeypatch.setattr(missions_mod, "utc_stamp", lambda: "2026-10-17T12:00:00Z")
    result = getattr(engine, operation)("Develop ORION", "No such task")
    assert not result.ok
    assert engine.path.read_bytes() == before
    assert engine.bus.emitted == []
    record = engine.load()["missions"]["Develop ORION"]
    assert record["updated_at"] == "2000-01-01T00:00:00Z"
    assert any("stalled" in risk for risk in engine.risks(record))


@pytest.mark.parametrize("mission", ["Develop ORION", ""])
def test_empty_completed_sweep_leaves_mission_activity_unchanged(engine, mission):
    engine.add_task("Develop ORION", "Still open")
    before = engine.path.read_bytes()
    engine.bus.emitted.clear()
    result = engine.clear_completed(mission)
    assert result.ok and "Nothing completed" in result.text
    assert engine.path.read_bytes() == before
    assert engine.bus.emitted == []


def test_whole_board_sweep_updates_only_missions_with_finished_items(engine, monkeypatch):
    engine.add_task("Develop ORION", "Finished task")
    engine.complete("Develop ORION", "Finished task")
    engine.add_task("University Studies", "Still pending")
    data = engine.load()
    for record in data["missions"].values():
        record["updated_at"] = "2000-01-01T00:00:00Z"
    engine._save(data)
    monkeypatch.setattr(missions_mod, "utc_stamp", lambda: "2026-10-17T12:00:00Z")
    assert engine.clear_completed().ok
    stored = engine.load()["missions"]
    assert stored["Develop ORION"]["updated_at"] == "2026-10-17T12:00:00Z"
    assert stored["Develop ORION"]["tasks"] == []
    assert all(record["updated_at"] == "2000-01-01T00:00:00Z"
               for name, record in stored.items() if name != "Develop ORION")


def test_deleting_current_never_selects_archived_mission(engine):
    engine.clear_all()
    engine.create("Archived first")
    engine.create("Current")
    engine.create("Available")
    engine.archive("Archived first")
    engine.set_current("Current")
    engine.delete("Current")
    assert engine.current()["name"] == "Available"


def test_goal_task_complete_progress_cycle(engine):
    assert engine.add_goal("Demo Game", "Ship the vertical slice").text
    assert engine.add_task("Demo Game", "Block out level one").ok
    assert engine.add_task("Demo Game", "Implement enemy AI").ok
    record = engine._resolve(engine.load()["missions"], "Demo Game")
    assert engine.progress(record) == 0.0
    done = engine.complete("Demo Game", "level one")
    assert done.ok and "level one" in done.text
    record = engine._resolve(engine.load()["missions"], "Demo Game")
    # 1 of 2 tasks done, goal (double weight) open → 1/4.
    assert engine.progress(record) == 25.0
    assert not engine.complete("Demo Game", "no such thing").ok


def test_current_mission_switch_partial_match(engine):
    result = engine.set_current("creator")
    assert result.ok and "Creator Studio" in result.text
    assert engine.current()["name"] == "Creator Studio"
    assert not engine.set_current("nonexistent").ok


def test_status_reports_risks_and_recommendations(engine):
    engine.add_goal("Neuroscience", "Publish a literature review")
    status = engine.status("Neuroscience").text
    assert "Risks:" in status                     # goals without research
    assert "Recommended:" in status
    assert "decompose" in status                  # goal with no tasks


def test_attach_kinds_and_validation(engine):
    assert engine.attach("ORION", "file", "docs/ULTRON_ANALYSIS.md").ok
    assert engine.attach("ORION", "note", "Crimson identity locked").ok
    assert not engine.attach("ORION", "bogus_kind", "x").ok
    assert not engine.attach("missing mission", "file", "x").ok
    status = engine.status("Develop ORION").text
    assert "Files: 1 linked" in status


def test_panel_snapshot_shape(engine):
    engine.add_task("University", "Submit assignment", due="2026-01-01T00:00:00Z")
    snap = engine.panel_snapshot()
    assert snap["current"] == "Develop ORION"
    names = {m["name"] for m in snap["missions"]}
    assert "University Studies" in names
    uni = next(m for m in snap["missions"] if m["name"] == "University Studies")
    assert uni["open_tasks"] == 1
    assert any("past due" in r for r in uni["risks"])


def test_handle_tool_surface(engine):
    loop = asyncio.new_event_loop()
    try:
        assert "Mission board" in loop.run_until_complete(
            engine.handle({"action": "overview"})).text
        assert loop.run_until_complete(engine.handle(
            {"action": "task", "mission": "Demo Game",
             "text": "Design the HUD"})).ok
        assert loop.run_until_complete(engine.handle(
            {"action": "status", "mission": "Demo Game"})).ok
        assert not loop.run_until_complete(
            engine.handle({"action": "warp_drive"})).ok
    finally:
        loop.close()
