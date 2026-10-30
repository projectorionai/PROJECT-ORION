"""Verify reviewed examples offline and keep the teaching API text-only."""

from __future__ import annotations

import ast
import builtins
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import cyber_exercises


def run_bundled_example(item):
    """Only tests execute trusted bundled literals, never caller-supplied code."""
    output = io.StringIO()
    namespace = {}
    with redirect_stdout(output):
        exec(compile(item.code, f"<bundled-example:{item.topic}>", "exec"), namespace)
    return output.getvalue().rstrip("\n"), namespace


@pytest.mark.parametrize("item", cyber_exercises.EXERCISES, ids=lambda item: item.topic)
def test_reviewed_snippets_have_correct_expected_outputs(item):
    actual, _ = run_bundled_example(item)
    assert actual == item.expected_output


def test_examples_only_import_the_needed_standard_library_modules():
    allowed = {"sqlite3", "html", "collections"}
    for item in cyber_exercises.EXERCISES:
        tree = ast.parse(item.code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert {alias.name for alias in node.names} <= allowed
            elif isinstance(node, ast.ImportFrom):
                assert node.module in allowed
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "compile", "__import__"}
    sql = next(item for item in cyber_exercises.EXERCISES if item.topic == "sql_injection")
    assert 'sqlite3.connect(":memory:")' in sql.code


def test_object_access_control_denies_other_owners_and_missing_records():
    item = next(item for item in cyber_exercises.EXERCISES if item.topic == "access_control")
    _, namespace = run_bundled_example(item)
    read_report = namespace["read_report"]
    assert read_report("learner_b", 102) == "Learner B's practice report"
    assert read_report("learner_b", 101) == "DENIED"
    assert read_report("learner_a", 999) == "DENIED"
    assert read_report("", 101) == "DENIED"


@pytest.mark.parametrize("query,expected", [
    ("sql_injection", "sql_injection"),
    (" SQL ", "sql_injection"),
    ("sqlite", "sql_injection"),
    ("secure sql", "sql_injection"),
    ("show code for sql", "sql_injection"),
    ("html escaping", "xss"),
    ("XSS", "xss"),
    ("teach me escaping", "xss"),
    ("access_control", "access_control"),
    ("IDOR", "access_control"),
    ("object-level access control", "access_control"),
    ("log_analysis", "log_analysis"),
    ("detection", "log_analysis"),
    ("exercise on log analysis", "log_analysis"),
])
def test_aliases_resolve_to_reviewed_topics(query, expected):
    assert cyber_exercises.resolve_topic(query) == expected


@pytest.mark.parametrize("query", ["", "malware", "sql exploit on example.com", "xss elsewhere", None, 12])
def test_unreviewed_or_ambiguous_queries_are_not_guessed(query):
    assert cyber_exercises.resolve_topic(query) is None
    assert "Choose one" in cyber_exercises.code_lesson(query)


def test_catalogue_is_independent_of_caller_mutation():
    first = cyber_exercises.catalogue()
    assert len(first) == 4
    assert all(set(row) == {"topic", "title"} for row in first)
    first[0]["title"] = "changed by caller"
    assert cyber_exercises.catalogue()[0]["title"] != first[0]["title"]


@pytest.mark.parametrize("item", cyber_exercises.EXERCISES, ids=lambda item: item.topic)
def test_lessons_and_self_checks_present_code_choices_and_limits(item):
    lesson = cyber_exercises.code_lesson(item.topic)
    practice = cyber_exercises.exercise(item.topic)
    for text in (lesson, practice):
        assert item.title in text
        assert item.code.rstrip() in text
        assert item.question in text
        assert all(f"{letter}. " in text for letter in "ABC")
        assert "never runs" in text
        assert "No network connections" in text
        assert "no files are modified" in text
    assert item.expected_output in lesson
    for _, url in item.sources:
        assert url in lesson


@pytest.mark.parametrize("item", cyber_exercises.EXERCISES, ids=lambda item: item.topic)
def test_fixed_choice_answers_return_explained_feedback(item):
    assert cyber_exercises.check_answer(item.topic, item.answer).startswith("Correct.")
    assert cyber_exercises.check_answer(item.topic, f" {item.answer.lower()} ").startswith("Correct.")
    index = str("ABC".index(item.answer) + 1)
    assert cyber_exercises.check_answer(item.topic, index).startswith("Correct.")
    wrong = next(choice for choice in "ABC" if choice != item.answer)
    feedback = cyber_exercises.check_answer(item.topic, wrong)
    assert feedback.startswith("Not quite.")
    assert item.feedback in feedback
    assert item.expected_output in feedback


@pytest.mark.parametrize("answer", ["", "D", "four", "A because it works", "print('hello')", None, 2])
def test_unsupported_answers_help_user_submit_a_fixed_choice(answer):
    feedback = cyber_exercises.check_answer("sql", answer)
    assert "Please send only A, B or C" in feedback
    assert "does not parse or execute submitted code" in feedback


def test_presentation_and_user_answers_do_not_execute_code(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The text-only teaching API attempted code execution")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "eval", forbidden)
        patch.setattr(builtins, "exec", forbidden)
        patch.setattr(builtins, "compile", forbidden)
        for item in cyber_exercises.EXERCISES:
            assert cyber_exercises.code_lesson(item.topic)
            assert cyber_exercises.exercise(item.topic)
            assert cyber_exercises.check_answer(item.topic, item.answer)
            assert "Please send only" in cyber_exercises.check_answer(item.topic, "__import__('os').system('anything')")


def test_production_module_has_no_code_execution_or_io_calls():
    tree = ast.parse(Path(cyber_exercises.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"exec", "eval", "compile", "open", "__import__"}
        if isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= {"re"}
        elif isinstance(node, ast.ImportFrom):
            assert node.module in {"__future__", "dataclasses", "textwrap"}


def test_detection_teaches_review_instead_of_proof():
    lesson = cyber_exercises.code_lesson("detection")
    assert "A single indicator is not proof of an attack" in lesson
    assert "teaching threshold" in lesson
    assert "time window" in lesson
    assert "not a verdict" in cyber_exercises.check_answer("detection", "C")


def test_learning_path_orders_foundations_before_authorised_lab_practice():
    path = cyber_exercises.learning_path()
    stages = [
        "1. Programming foundations", "2. Networks and operating systems",
        "3. Scope and permission", "4. Secure coding", "5. Authorised lab practice",
        "6. Security reporting", "7. Defence",
    ]
    positions = [path.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "written authorisation" in path
    assert "synthetic data" in path
    assert "No network connections" in path
