"""
Muscle memory (CAP-07) — record once, generalise, replay with new inputs.

    "record a sequence once, ORION generalises it into a reusable workflow."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.muscle_memory import (  # noqa: E402
    MuscleMemory, RecordedAction, SkillStore, generalise, plan,
)


@pytest.fixture()
def store(tmp_path):
    return SkillStore(path=tmp_path / "skills.json")


@pytest.fixture()
def mm(store):
    return MuscleMemory(store=store)


def _search_recording():
    return [
        RecordedAction("navigate", "https://shop.example.com"),
        RecordedAction("fill", "search", "wireless headphones"),
        RecordedAction("submit"),
        RecordedAction("click", "Add to basket"),
    ]


# ── generalisation ───────────────────────────────────────────────────────────

def test_typed_values_become_parameters():
    skill = generalise("buy a thing", _search_recording())
    assert skill.params == ["search"]
    fill = next(a for a in skill.actions if a.op == "fill")
    assert fill.value == "{search}"


def test_structure_is_preserved():
    skill = generalise("buy", _search_recording())
    ops = [a.op for a in skill.actions]
    assert ops == ["navigate", "click"] or ops == ["navigate", "fill", "submit", "click"]
    # origin captured from the first navigate
    assert skill.origin == "https://shop.example.com"


def test_two_fields_get_distinct_parameters():
    rec = [
        RecordedAction("fill", "email", "me@example.com"),
        RecordedAction("fill", "email", "second@example.com"),  # same label twice
        RecordedAction("fill", "password", "hunter2"),
    ]
    skill = generalise("login", rec)
    assert skill.params == ["email", "email_2", "password"]


def test_the_skill_name_is_slugged():
    skill = generalise("Buy A Thing!", _search_recording())
    assert skill.name == "buy_a_thing"


# ── planning ─────────────────────────────────────────────────────────────────

def test_plan_substitutes_parameters():
    skill = generalise("buy", _search_recording())
    steps, missing = plan(skill, {"search": "running shoes"})
    assert not missing
    fill = next(s for s in steps if s["op"] == "fill")
    assert fill["value"] == "running shoes"


def test_plan_reports_missing_parameters():
    skill = generalise("buy", _search_recording())
    steps, missing = plan(skill, {})       # nothing supplied
    assert missing == ["search"]


def test_plan_steps_use_the_copilot_vocabulary():
    skill = generalise("buy", _search_recording())
    steps, _ = plan(skill, {"search": "x"})
    for s in steps:
        assert s["op"] in {"navigate", "click", "fill", "submit", "scroll", "read", "wait"}


# ── record → store → replay ──────────────────────────────────────────────────

def test_record_stop_saves_a_skill(mm):
    mm.start_recording("do it")
    for a in _search_recording():
        mm.record(a.op, a.target, a.value)
    skill = mm.stop_recording()
    assert skill is not None
    assert mm.store.get("do_it") is not None


def test_replay_a_saved_skill_with_new_params(mm):
    mm.start_recording("buy")
    for a in _search_recording():
        mm.record(a.op, a.target, a.value)
    mm.stop_recording()
    skill, steps, missing = mm.run("buy", {"search": "kettle"})
    assert skill is not None and not missing
    assert any(s["value"] == "kettle" for s in steps)


def test_running_an_unknown_skill_returns_none(mm):
    skill, steps, missing = mm.run("nope")
    assert skill is None


def test_recording_nothing_saves_nothing(mm):
    mm.start_recording("empty")
    assert mm.stop_recording() is None


def test_skills_persist_across_store_instances(tmp_path):
    path = tmp_path / "skills.json"
    a = MuscleMemory(store=SkillStore(path=path))
    a.start_recording("persisted")
    a.record("navigate", "https://x.com")
    a.record("fill", "q", "hello")
    a.stop_recording()
    # a fresh engine over the same file sees it
    b = MuscleMemory(store=SkillStore(path=path))
    assert any(s.name == "persisted" for s in b.skills())


def test_forget_removes_a_skill(mm):
    mm.start_recording("temp")
    mm.record("navigate", "https://x.com")
    mm.stop_recording()
    assert mm.forget("temp")
    assert mm.store.get("temp") is None


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"muscle_memory"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "muscle_memory")
    assert "action" in tool["parameters"]["properties"]
