"""Normal requests choose short lessons; full detail and scope remain explicit."""

from types import SimpleNamespace

import pytest

from orion_core.cyber_curriculum import CyberCurriculum
from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
from orion_core.dispatch_schema import TOOL_DECLARATIONS
from orion_core.learning_coach import LearningCoach


@pytest.fixture
def teacher(tmp_path):
    return SimpleNamespace(
        _cyber_curriculum=CyberCurriculum(progress_path=tmp_path / "modules.json"),
        _learning_coach=LearningCoach(state_path=tmp_path / "lesson.json"),
        cyber=None,
    )


def ask(teacher, action, **args):
    return KnowledgeDispatchMixin.cyber_curriculum_tool(teacher, {"action": action, **args})


@pytest.mark.parametrize("action,topic", [
    ("path", ""), ("teach", "sql_injection"), ("code", "xss"),
    ("exercise", "access_control"), ("teach", "electronics_fundamentals"),
])
def test_normal_teaching_starts_with_one_small_step(teacher, action, topic):
    result = ask(teacher, action, topic=topic)
    assert result.ok and "step 1/" in result.text and "Goal:" in result.text
    assert "step 2/" not in result.text
    assert len(result.text.split()) < 220


def test_full_reference_lesson_still_available_by_request(teacher):
    result = ask(teacher, "teach", topic="sql_injection", format="full")
    assert result.ok and "Key identifiers" in result.text and ":memory:" in result.text
    assert not teacher._learning_coach.state_path.exists()


def test_check_answer_and_resume_use_current_step_without_resetting(teacher):
    ask(teacher, "coach_start", topic="python_basics", pace_minutes=0)
    first = ask(teacher, "coach_status").text
    ask(teacher, "coach_answer", answer="A")
    assert ask(teacher, "coach_status").text == first
    assert "step 2/" in ask(teacher, "coach_next").text
    assert "paused" in ask(teacher, "coach_pause").text.lower()
    teacher._learning_coach = LearningCoach(state_path=teacher._learning_coach.state_path)
    assert "step 2/" in ask(teacher, "coach_resume").text
    assert "step 2/" in ask(teacher, "teach", topic="python_basics").text


@pytest.mark.parametrize("query", ["hardware protection", "how to defend against a USB killer"])
def test_direct_knowledge_topic_uses_short_teaching(teacher, query):
    result = KnowledgeDispatchMixin.cyber_knowledge_tool(
        teacher, {"query": query})
    assert result.ok and "step 1/" in result.text


def test_unknown_topic_and_construction_request_do_not_start_session(teacher):
    for topic in ("working ransomware encryptor", "make tasers for electronics"):
        assert not ask(teacher, "coach_start", topic=topic).ok
    assert not teacher._learning_coach.state_path.exists()


def test_short_lesson_does_not_mark_a_curriculum_module_complete(teacher):
    ask(teacher, "coach_start", topic="python_basics")
    for _ in range(5):
        ask(teacher, "coach_next")
        ask(teacher, "coach_resume")
    assert teacher._cyber_curriculum.progress("default")["completed"] == []


def test_pace_settings_before_start_do_not_skip_the_first_step(teacher):
    ask(teacher, "coach_settings", pace_minutes=0, steps_per_session=1)
    assert not ask(teacher, "coach_next").ok
    assert "step 1/" in ask(teacher, "coach_start").text
    paused = ask(teacher, "coach_next")
    assert "paused at step 1/" in paused.text.lower()
    assert "step 2/" not in paused.text


def test_learning_controls_are_exposed_when_offline():
    from orion_core.tool_resolver import ResolverState, resolve

    declaration = next(d for d in TOOL_DECLARATIONS if d["name"] == "cyber_curriculum")
    params = declaration["parameters"]["properties"]
    assert {"format", "pace_minutes", "steps_per_session", "restart"} <= params.keys()
    for query in ("teach me in small steps because I lose focus",
                  "make this lesson simpler", "pause my lesson", "resume my lesson"):
        tools = resolve(query, ResolverState(mode="B", online=False),
                        TOOL_DECLARATIONS, max_tools=10)
        assert "cyber_curriculum" in {d["name"] for d in tools}
