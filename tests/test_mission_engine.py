"""
MissionEngine tests — the mission-based operating model.

All offline: stub bus, temporary JSON store, no Qt.
"""

from __future__ import annotations

import asyncio
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
