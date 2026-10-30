"""Learner-controlled short lessons keep their place without executing input."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from orion_core.learning_coach import LearningCoach


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "learning.json"


@pytest.fixture
def coach(state_path):
    return LearningCoach(state_path=state_path)


def saved_learner(state_path, user="default"):
    return json.loads(state_path.read_text(encoding="utf-8"))["learners"][user]


def test_start_defaults_to_one_short_python_step(coach, state_path):
    result = coach.handle("start")
    assert result.ok
    state = saved_learner(state_path)
    assert state["topic"] == "python_basics"
    assert state["index"] == 0
    assert state["segment_steps"] == 1
    assert state["pace_minutes"] == 5
    assert state["steps_per_session"] == 3
    assert not state["paused"] and not state["complete"]
    assert result.text.count("?") <= 1
    assert len(result.text.split()) < 220


def test_next_without_a_session_does_not_create_one(coach, state_path):
    assert not coach.handle("next").ok
    assert not state_path.exists()


def test_pause_survives_restart_and_resume_reopens_same_step(coach, state_path):
    assert coach.handle("start").ok
    assert coach.handle("next").ok
    before = saved_learner(state_path)
    assert coach.handle("pause").ok
    paused = saved_learner(state_path)
    assert paused["paused"] and paused["index"] == before["index"]

    restarted = LearningCoach(state_path=state_path)
    result = restarted.handle("resume")
    assert result.ok
    resumed = saved_learner(state_path)
    assert not resumed["paused"]
    assert resumed["topic"] == before["topic"]
    assert resumed["index"] == before["index"]
    assert resumed["segment_steps"] == 0
    assert any(cue in result.text.lower() for cue in ("recap", "remember", "where we were"))


@pytest.mark.parametrize("action", ["repeat", "simpler", "status"])
def test_reading_and_simplification_do_not_advance(coach, state_path, action):
    assert coach.handle("start").ok
    before = saved_learner(state_path)
    result = coach.handle(action)
    assert result.ok
    assert saved_learner(state_path) == before
    assert result.text.count("?") <= 1


@pytest.mark.parametrize("action", ["next", "answer", "simpler", "repeat", "status"])
def test_paused_operations_do_not_advance_or_unpause(coach, state_path, action):
    assert coach.handle("start").ok
    assert coach.handle("pause").ok
    before = saved_learner(state_path)
    coach.handle(action, answer="A")
    after = saved_learner(state_path)
    assert after["index"] == before["index"]
    assert after["segment_steps"] == before["segment_steps"]
    assert after["paused"]


def test_step_budget_offers_a_break_without_skipping(coach, state_path):
    assert coach.handle("start", steps_per_session=2).ok
    assert coach.handle("next").ok
    before = saved_learner(state_path)
    assert before["index"] == 1 and before["segment_steps"] == 2
    assert coach.handle("next").ok
    paused = saved_learner(state_path)
    assert paused["paused"]
    assert paused["index"] == before["index"]

    assert coach.handle("resume").ok
    assert saved_learner(state_path)["index"] == before["index"]
    assert coach.handle("next").ok
    after = saved_learner(state_path)
    assert after["index"] == before["index"] + 1
    assert not after["paused"]


def test_start_same_topic_and_omitted_topic_preserve_place(coach, state_path):
    assert coach.handle("start").ok
    assert coach.handle("next").ok
    before = saved_learner(state_path)
    assert coach.handle("start", topic="python_basics").ok
    assert saved_learner(state_path)["index"] == before["index"]
    assert coach.handle("start").ok
    assert saved_learner(state_path)["index"] == before["index"]


def test_explicit_restart_resets_current_lesson(coach, state_path):
    assert coach.handle("start").ok
    assert coach.handle("next").ok
    assert coach.handle("pause").ok
    assert coach.handle("start", restart=True).ok
    state = saved_learner(state_path)
    assert state["index"] == 0 and state["segment_steps"] == 1
    assert not state["paused"] and not state["complete"]


def test_learners_keep_independent_topics_positions_and_preferences(coach, state_path):
    assert coach.handle("start", user="alice", topic="python_basics", pace_minutes=1).ok
    assert coach.handle("next", user="alice").ok
    assert coach.handle("start", user="bob", topic="sql_injection", pace_minutes=4).ok
    alice = saved_learner(state_path, "alice")
    bob = saved_learner(state_path, "bob")
    assert alice["topic"] == "python_basics" and alice["index"] == 1
    assert bob["topic"] == "sql_injection" and bob["index"] == 0
    assert alice["pace_minutes"] == 1 and bob["pace_minutes"] == 4
    assert coach.handle("pause", user="alice").ok
    assert saved_learner(state_path, "bob") == bob


def test_settings_persist_in_a_new_coach_without_resetting_position(coach, state_path):
    assert coach.handle("start").ok
    assert coach.handle("next").ok
    before = saved_learner(state_path)
    assert coach.handle("settings", pace_minutes=0, steps_per_session=5).ok
    restarted = LearningCoach(state_path=state_path)
    assert restarted.handle("status").ok
    state = saved_learner(state_path)
    assert state["index"] == before["index"]
    assert state["pace_minutes"] == 0 and state["steps_per_session"] == 5


def test_settings_before_a_lesson_store_preferences_without_visiting_a_step(coach, state_path):
    result = coach.handle("settings", pace_minutes=1, steps_per_session=2)
    assert result.ok
    state = saved_learner(state_path)
    assert state["topic"] == "python_basics" and state["index"] == 0
    assert state["segment_steps"] == 0
    assert state["pace_minutes"] == 1 and state["steps_per_session"] == 2
    assert not state["paused"] and not state["complete"]
    assert "step 1/" not in result.text.lower()


@pytest.mark.parametrize("restart", [0, 1, "false", "true", None, []])
def test_invalid_restart_flag_does_not_reset_or_modify_session(coach, state_path, restart):
    assert coach.handle("start").ok
    assert coach.handle("next").ok
    original = state_path.read_bytes()
    assert not coach.handle("start", restart=restart).ok
    assert state_path.read_bytes() == original


def test_concurrent_coaches_do_not_lose_other_learners(state_path):
    def start_for(user):
        return LearningCoach(state_path=state_path).handle("start", user=user)

    users = [f"learner-{index}" for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result.ok for result in pool.map(start_for, users))
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(data["learners"]) == set(users)


@pytest.mark.parametrize("minutes", [0, 1, 10])
@pytest.mark.parametrize("steps", [1, 5])
def test_settings_accept_documented_bounds(coach, state_path, minutes, steps):
    assert coach.handle("start", pace_minutes=minutes, steps_per_session=steps).ok
    state = saved_learner(state_path)
    assert state["pace_minutes"] == minutes
    assert state["steps_per_session"] == steps


@pytest.mark.parametrize("setting,value", [
    ("pace_minutes", -1), ("pace_minutes", 11), ("pace_minutes", True),
    ("pace_minutes", 1.5), ("pace_minutes", "two"),
    ("steps_per_session", 0), ("steps_per_session", 6),
    ("steps_per_session", False), ("steps_per_session", 2.5),
    ("steps_per_session", "three"),
])
def test_invalid_settings_leave_existing_session_intact(coach, state_path, setting, value):
    assert coach.handle("start").ok
    original = state_path.read_bytes()
    assert not coach.handle("settings", **{setting: value}).ok
    assert state_path.read_bytes() == original


def test_answer_is_feedback_only_and_never_runs_or_saves_input(coach, state_path, tmp_path):
    assert coach.handle("start", topic="sql_injection").ok
    before = saved_learner(state_path)
    marker = tmp_path / "must-not-exist.txt"
    answer = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    result = coach.handle("answer", answer=answer)
    assert result.text
    assert saved_learner(state_path) == before
    assert answer not in state_path.read_text(encoding="utf-8")
    assert not marker.exists()


def test_answer_feedback_keeps_the_step_until_explicit_next(coach, state_path):
    assert coach.handle("start", topic="python_basics").ok
    original = state_path.read_bytes()
    incorrect = coach.handle("answer", answer="B")
    correct = coach.handle("answer", answer="A")
    ambiguous = coach.handle("answer", answer="I think so")
    assert incorrect.ok and correct.ok and not ambiguous.ok
    assert incorrect.text != correct.text
    assert incorrect.text.count("?") == 1
    assert "2" in correct.text and "next" in correct.text.lower()
    assert state_path.read_bytes() == original
    assert coach.handle("next").ok
    assert saved_learner(state_path)["index"] == 1


def test_reflection_is_not_graded_or_saved_as_a_health_record(coach, state_path):
    assert coach.handle("start", topic="sql_injection", steps_per_session=5).ok
    for _ in range(3):
        assert coach.handle("next").ok
    before = state_path.read_bytes()
    result = coach.handle("answer", answer="I'm unsure and distracted, but want to keep trying.")
    assert result.ok
    assert "not graded" in result.text.lower()
    assert state_path.read_bytes() == before
    assert not saved_learner(state_path)["complete"]


def test_simpler_changes_the_explanation_without_losing_position(coach, state_path):
    assert coach.handle("start", topic="python_basics").ok
    original = state_path.read_bytes()
    regular = coach.handle("repeat")
    simple = coach.handle("simpler")
    assert regular.ok and simple.ok
    assert regular.text != simple.text
    assert "What does print(minutes) display?" in regular.text
    assert "What does print(minutes) display?" in simple.text
    assert state_path.read_bytes() == original


@pytest.mark.parametrize("topic", [
    "build a stun gun with two batteries",
    "program a USB killer",
    "working ransomware",
])
def test_unsafe_or_unsupported_topic_cannot_replace_a_safe_session(coach, state_path, topic):
    assert coach.handle("start").ok
    original = state_path.read_bytes()
    result = coach.handle("start", topic=topic)
    assert not result.ok
    assert state_path.read_bytes() == original


@pytest.mark.parametrize("contents", ["{broken JSON", "[]", "null", " " * (2 * 1024 * 1024)],
                         ids=["broken-json", "array-root", "null-root", "oversized"])
def test_corrupt_or_oversized_state_is_not_overwritten(coach, state_path, contents):
    state_path.write_text(contents, encoding="utf-8")
    original = state_path.read_bytes()
    result = coach.handle("start")
    assert not result.ok
    assert result.text
    assert state_path.read_bytes() == original


@pytest.mark.parametrize("field,value", [
    ("topic", "no_such_topic"), ("index", -1), ("index", 999999),
    ("index", True), ("paused", "yes"), ("complete", "yes"),
    ("complete", True), ("pace_minutes", -1),
    ("steps_per_session", 0), ("segment_steps", -1),
])
def test_invalid_saved_session_is_not_overwritten(coach, state_path, field, value):
    assert coach.handle("start").ok
    data = json.loads(state_path.read_text(encoding="utf-8"))
    data["learners"]["default"][field] = value
    state_path.write_text(json.dumps(data), encoding="utf-8")
    original = state_path.read_bytes()
    assert not coach.handle("resume").ok
    assert state_path.read_bytes() == original


def test_unknown_saved_version_is_not_overwritten(coach, state_path):
    state_path.write_text(json.dumps({"version": 999, "learners": {}}), encoding="utf-8")
    original = state_path.read_bytes()
    assert not coach.handle("start").ok
    assert state_path.read_bytes() == original


def test_failed_save_preserves_file_and_does_not_commit_an_advance(coach, state_path, monkeypatch):
    from orion_core import learning_coach

    assert coach.handle("start").ok
    original = state_path.read_bytes()
    writer = learning_coach.atomic_write_text

    def locked_file(*args, **kwargs):
        raise PermissionError("synthetic sharing lock")

    monkeypatch.setattr(learning_coach, "atomic_write_text", locked_file)
    result = coach.handle("next")
    assert not result.ok
    assert "save" in result.text.lower() or "writ" in result.text.lower()
    assert state_path.read_bytes() == original

    monkeypatch.setattr(learning_coach, "atomic_write_text", writer)
    assert coach.handle("next").ok
    assert saved_learner(state_path)["index"] == 1


def test_unknown_action_does_not_modify_a_session(coach, state_path):
    assert coach.handle("start").ok
    original = state_path.read_bytes()
    assert not coach.handle("run learner code").ok
    assert state_path.read_bytes() == original


def test_visited_lesson_completion_is_stable_across_restart(coach, state_path):
    assert coach.handle("start", steps_per_session=5).ok
    for _ in range(30):
        record = saved_learner(state_path)
        if record["complete"]:
            break
        if record["paused"]:
            assert coach.handle("resume").ok
        else:
            assert coach.handle("next").ok
    else:
        pytest.fail("A short lesson did not reach completion after 30 explicit controls")

    completed = saved_learner(state_path)
    restored = LearningCoach(state_path=state_path)
    result = restored.handle("next")
    assert result.ok
    assert saved_learner(state_path) == completed
    assert "qualification" in result.text.lower() or "mastery" in result.text.lower()
