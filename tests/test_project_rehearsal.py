"""Project forecasts must preserve dependencies and label their uncertainty."""

from __future__ import annotations

import copy
import math

import pytest

from orion_core.project_rehearsal import analyse_rehearsal, run_rehearsal


def task(task_id, duration, *, depends_on=()):
    return {"id": task_id, "optimistic": duration, "likely": duration,
            "pessimistic": duration, "depends_on": list(depends_on)}


def test_parallel_dag_has_hand_calculated_path_and_slack():
    # A=2 -> B=3 or C=5 -> D=1. C is the bottleneck; B has 2 days slack.
    result = analyse_rehearsal({"tasks": [task("D", 1, depends_on=("B", "C")),
        task("B", 3, depends_on=("A",)), task("A", 2), task("C", 5, depends_on=("A",))],
        "trials": 10, "deadline": 8})
    baseline = result["baseline"]
    assert baseline["expected_duration"] == 8
    assert baseline["critical_path"] == ["A", "C", "D"]
    assert baseline["p50"] == baseline["p80"] == baseline["p95"] == 8
    assert baseline["deadline_probability"] == 1
    rows = {row["id"]: row for row in baseline["tasks"]}
    assert rows["B"]["slack"] == 2
    assert rows["B"]["earliest_start"] == 2
    assert rows["B"]["latest_start"] == 4
    assert rows["B"]["criticality_pct"] == 0
    assert rows["C"]["criticality_pct"] == 100


def test_seeded_sampling_is_repeatable_and_does_not_mutate_inputs():
    args = {"tasks": [{"id": "build", "optimistic": 1, "likely": 2, "pessimistic": 9}],
            "seed": 51, "trials": 500, "deadline": 4}
    untouched = copy.deepcopy(args)
    first = analyse_rehearsal(args)
    assert first == analyse_rehearsal(args)
    assert args == untouched
    baseline = first["baseline"]
    assert 1 <= baseline["p50"] <= baseline["p80"] <= baseline["p95"] <= 9
    assert 0 < baseline["deadline_probability"] < 1
    assert baseline["expected_duration"] == 3
    assert baseline["p50"] != analyse_rehearsal({**args, "seed": 52})["baseline"]["p50"]


def test_what_if_can_move_the_critical_path_and_improve_deadline():
    result = analyse_rehearsal({"tasks": [task("slow", 8), task("fast", 5),
        task("finish", 2, depends_on=("slow", "fast"))],
        "trials": 20, "deadline": 7, "what_if": [{"id": "slow", "multiplier": 0.5}]})
    assert result["baseline"]["critical_path"] == ["slow", "finish"]
    assert result["what_if"]["critical_path"] == ["fast", "finish"]
    assert result["comparison"]["p80_change"] == -3
    assert result["baseline"]["deadline_probability"] == 0
    assert result["what_if"]["deadline_probability"] == 1
    assert result["comparison"]["deadline_probability_change"] == 1


def test_paired_what_if_samples_scale_exactly_for_a_single_task():
    result = analyse_rehearsal({"tasks": [{"id": "work", "optimistic": 1,
        "likely": 3, "pessimistic": 7}], "trials": 100, "what_if": {"work": 0.5}})
    assert result["what_if"]["p80"] == result["baseline"]["p80"] / 2
    assert result["comparison"]["deadline_probability_change"] is None


def test_tied_paths_both_count_as_bottlenecks_and_zero_milestones_are_valid():
    result = analyse_rehearsal({"tasks": [task("a", 2), task("b", 2),
        task("milestone", 0, depends_on=("a", "b"))], "trials": 1})
    assert all(row["criticality_pct"] == 100 for row in result["baseline"]["tasks"])
    assert result["baseline"]["critical_path"][-1] == "milestone"


def test_triangular_forecast_matches_known_distribution():
    result = analyse_rehearsal({"tasks": [{"id": "work", "optimistic": 0,
        "likely": 1, "pessimistic": 2}], "deadline": 1, "trials": 5_000, "seed": 7})
    baseline = result["baseline"]
    assert baseline["deadline_probability"] == pytest.approx(0.5, abs=0.04)
    assert baseline["p80"] == pytest.approx(2 - math.sqrt(0.4), abs=0.04)
    assert baseline["p95"] == pytest.approx(2 - math.sqrt(0.1), abs=0.04)


@pytest.mark.parametrize("tasks", [
    [], [task("same", 1), task("same", 2)],
    [task("a", 1, depends_on=("missing",))],
    [task("a", 1, depends_on=("b",)), task("b", 1, depends_on=("a",))],
    [task("a", 1, depends_on=("a",))],
    [task("a", 1), task("b", 1, depends_on=("a", "a"))],
    [task(str(index), 1) for index in range(61)],
    [None], [{**task("a", 1), "depends_on": "a"}],
])
def test_rejects_invalid_dependency_graphs(tasks):
    with pytest.raises(ValueError):
        analyse_rehearsal({"tasks": tasks})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "3", None, 10**500])
def test_rejects_invalid_duration_values(value):
    with pytest.raises(ValueError):
        analyse_rehearsal({"tasks": [{**task("a", 2), "optimistic": value}]})


@pytest.mark.parametrize("overrides", [
    {"trials": 0}, {"trials": 5001}, {"trials": 1.5}, {"trials": True},
    {"seed": "0"}, {"seed": True}, {"deadline": -1}, {"deadline": math.inf},
    {"unit": ""}, {"what_if": "a"}, {"what_if": {"missing": 0.5}},
    {"what_if": {"a": 0}}, {"what_if": {"a": math.nan}},
    {"what_if": {"a": 1001}},
    {"what_if": [{"id": "a", "multiplier": 1}, {"id": "a", "multiplier": 2}]},
    {"what_if": [None]},
])
def test_rejects_invalid_controls(overrides):
    with pytest.raises(ValueError):
        analyse_rehearsal({"tasks": [task("a", 2)], **overrides})


def test_duration_order_is_checked_and_tool_returns_useful_failure():
    result = run_rehearsal({"tasks": [{"id": "a", "optimistic": 3,
                            "likely": 2, "pessimistic": 4}]})
    assert not result.ok
    assert "optimistic <= likely <= pessimistic" in result.text


def test_tool_exposes_report_and_honest_assumptions():
    result = run_rehearsal({"tasks": [task("a", 1)], "trials": 2, "deadline": 1})
    assert result.ok
    assert result.response_payload()["receipt"]["baseline"]["p80"] == 1
    assert "parallelism is unlimited" in result.text
    assert "not guarantees" in result.text
    assert "triangular" in result.text


def test_no_tasks_are_executed_by_the_engine():
    result = run_rehearsal({"tasks": [task("__import__('os').system('do-not-run')", 1)], "trials": 1})
    assert result.ok
    assert result.receipt["baseline"]["critical_path"] == ["__import__('os').system('do-not-run')"]
