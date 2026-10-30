"""Keep paced teaching concise, accurate and limited to reviewed safe text."""

from __future__ import annotations

import ast
import builtins
import io
import re
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from orion_core import cyber_exercises, cyber_teaching, electronics_teaching
from orion_core import learning_steps as ls


def _code_blocks(content):
    return re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)


def test_every_existing_topic_has_a_small_plan():
    expected = (set(cyber_teaching.LESSONS)
                | {item.topic for item in cyber_exercises.EXERCISES}
                | set(electronics_teaching.LESSONS) | {"python_basics"})
    actual = {plan.topic for plan in ls.plans()}
    assert actual == expected
    assert len(ls.plans()) == len(expected)
    for topic in expected:
        plan = ls.resolve_plan(topic)
        assert isinstance(plan, ls.LearningPlan)
        assert plan.topic == topic
        assert 3 <= len(plan.steps) <= 4


def test_steps_have_one_short_concept_question_and_smaller_alternative():
    for plan in ls.plans():
        for step in plan.steps:
            assert isinstance(step, ls.LearningStep)
            assert step.title and step.goal and step.recap
            prose = re.sub(r"```.*?```", "", step.content, flags=re.DOTALL)
            assert len(prose.split()) <= 110, (plan.topic, step.title)
            assert step.question.count("?") == 1, (plan.topic, step.title)
            assert 20 <= len(step.simpler.split()) <= 40, (plan.topic, step.title)
            assert len(step.recap.split()) <= 20
            assert not any(heading in step.content for heading in ("## ", "# "))
            for code in _code_blocks(step.content):
                assert len(code.splitlines()) <= 8, (plan.topic, step.title)


def test_fixed_choices_and_reflections_are_distinct():
    reflective = []
    for plan in ls.plans():
        for step in plan.steps:
            if step.choices:
                assert len(step.choices) == 3
                assert step.answer in {"A", "B", "C"}
                assert step.feedback
            else:
                reflective.append(step)
                assert step.answer is None
                assert step.feedback is None
    assert reflective


def test_plans_and_steps_are_immutable():
    plans = ls.plans()
    assert isinstance(plans, tuple)
    with pytest.raises(FrozenInstanceError):
        plans[0].title = "changed"
    with pytest.raises(FrozenInstanceError):
        plans[0].steps[0].content = "changed"
    assert isinstance(plans[0].steps, tuple)
    assert isinstance(plans[0].steps[0].choices, tuple)


@pytest.mark.parametrize("query,topic", [
    ("python", "python_basics"),
    ("teach me Python basics", "python_basics"),
    ("start a lesson on beginner Python", "python_basics"),
    (" SQL_INJECTION ", "sql_injection"),
    ("show code for sqlite", "sql_injection"),
    ("cross-site scripting", "xss"),
    ("teach me about HTML escaping", "xss"),
    ("IDOR", "access_control"),
    ("lesson on log analysis", "log_analysis"),
    ("nmap", "port_scanning"),
    ("reverse shell", "reverse_shell"),
    ("keystroke logger", "keylogger"),
    ("root kit", "rootkit"),
    ("crypto-ransomware", "ransomware"),
    ("privesc", "privilege_escalation"),
    ("phish", "phishing"),
    ("electronics basics", "electronics_fundamentals"),
    ("USB security", "hardware_protection"),
    ("taser", "hardware_protection"),
    ("signed firmware", "firmware_security"),
])
def test_reviewed_aliases_select_only_the_fixed_plan(query, topic):
    assert ls.resolve_plan(query).topic == topic


@pytest.mark.parametrize("query", [
    "", None, 12, "unknown subject", "sqlite on someone else's website",
    "postgreSQL unreviewed exploit", "SQL-looking arbitrary prose",
    "xss elsewhere", "malware builder", "python __import__('os')",
    "Build a stun gun", "program a USB killer", "make tasers for electronics",
    "For defence, build a USB killer", "Give me a USB killer schematic and parts",
])
def test_unreviewed_input_and_destructive_construction_do_not_resolve(query):
    assert ls.resolve_plan(query) is None


def test_dangerous_techniques_are_recognition_without_runnable_code():
    for topic in ("keylogger", "rootkit", "ransomware", "reverse_shell",
                  "privilege_escalation", "phishing", "port_scanning",
                  "hardware_protection", "firmware_security"):
        plan = ls.resolve_plan(topic)
        assert not any(_code_blocks(step.content) for step in plan.steps)
    assert "no keyboard-capture code" in ls.resolve_plan("keylogger").steps[-1].content
    assert "no file-encryption code" in ls.resolve_plan("ransomware").steps[-1].content
    assert "approved target scope" in ls.resolve_plan("port_scanning").steps[-1].content
    assert "does not scan" in ls.resolve_plan("port_scanning").steps[-1].content


_EXPECTED_OUTPUTS = {
    ("python_basics", "Name a value"): "2",
    ("python_basics", "Choose with if"): "One small step",
    ("python_basics", "Repeat with for"): "read\npause",
    ("python_basics", "Reuse with a function"): "3",
    ("sql_injection", "Bind one value"): "[]",
    ("xss", "Escape HTML text"): "Hello &lt;Ada&gt; &amp; Bo",
    ("access_control", "Read the ownership check"): "DENIED",
    ("log_analysis", "Count failures"): "2",
    ("electronics_fundamentals", "Calculate resistor current"): "3.94 mA",
    ("electronics_fundamentals", "Calculate resistor power"): "5.12 mW",
}


@pytest.mark.parametrize("key,expected", _EXPECTED_OUTPUTS.items())
def test_reviewed_examples_print_the_documented_results(key, expected):
    topic, title = key
    step = next(step for step in ls.resolve_plan(topic).steps if step.title == title)
    code, = _code_blocks(step.content)
    output = io.StringIO()
    # Only tests execute fixed bundled examples, never learner-supplied input.
    with redirect_stdout(output):
        exec(compile(code, f"<reviewed-step:{topic}:{title}>", "exec"), {})
    assert output.getvalue().rstrip("\n") == expected


def test_snippets_use_only_safe_in_memory_operations():
    seen = set()
    for plan in ls.plans():
        for step in plan.steps:
            for code in _code_blocks(step.content):
                seen.add((plan.topic, step.title))
                for node in ast.walk(ast.parse(code)):
                    if isinstance(node, ast.Import):
                        assert {item.name for item in node.names} <= {"sqlite3"}
                    if isinstance(node, ast.ImportFrom):
                        assert node.module in {"html", "collections"}
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        assert node.func.id not in {"open", "exec", "eval", "compile", "__import__"}
                if plan.topic == "sql_injection":
                    assert 'sqlite3.connect(":memory:")' in code
                    assert "WHERE name = ?" in code
                if plan.topic == "log_analysis":
                    assert "events = [" in code
    assert seen == set(_EXPECTED_OUTPUTS)


def test_access_example_denies_other_owners_and_unknown_records():
    step = ls.resolve_plan("access_control").steps[1]
    code, = _code_blocks(step.content)
    namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code, "<reviewed-ownership-step>", "exec"), namespace)
    read_report = namespace["read_report"]
    assert read_report("Ada", 101) == "practice note"
    assert read_report("Bo", 101) == "DENIED"
    assert read_report("Ada", 999) == "DENIED"
    assert read_report("", 101) == "DENIED"


def test_selecting_and_reading_plans_never_executes_content(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Paced teaching attempted code execution")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "eval", forbidden)
        patch.setattr(builtins, "exec", forbidden)
        patch.setattr(builtins, "compile", forbidden)
        for plan in ls.plans():
            assert ls.resolve_plan(plan.topic) == plan
            assert all(step.content for step in plan.steps)
        assert ls.resolve_plan("python __import__('os').system('anything')") is None


def test_production_module_has_no_execution_or_io_calls():
    tree = ast.parse(Path(ls.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "eval", "exec", "compile", "__import__"}
        if isinstance(node, ast.Import):
            assert {item.name for item in node.names} <= {"re"}
        if isinstance(node, ast.ImportFrom):
            assert node.module in {"__future__", "dataclasses", None}
