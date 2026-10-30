"""Experiments retain their original design and report only the measured change."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from orion_core.experiment_notebook import ExperimentNotebook


@pytest.fixture
def notebook(tmp_path):
    return ExperimentNotebook(tmp_path / "experiments.db")


def create(notebook, **overrides):
    result = notebook.handle({"action": "create", "title": "Short walks before revision",
                              "hypothesis": "A short walk increases completed practice questions",
                              "metric": "completed questions", "unit": "questions",
                              "direction": "higher", "target_improvement": 3, **overrides})
    assert result.ok, result.text
    assert result.receipt["saved"]
    return result.receipt["experiment"]["id"]


def record(notebook, experiment_id, phase, value, **extra):
    return notebook.handle({"action": "record", "id": experiment_id,
                            "phase": phase, "value": value, **extra})


def test_preregistered_design_and_observations_survive_reload(notebook):
    experiment_id = create(notebook)
    saved = record(notebook, experiment_id, "baseline", 10, note="Before lunch")
    assert saved.ok and saved.receipt["saved"]
    reloaded = ExperimentNotebook(notebook.path)
    shown = reloaded.handle({"action": "show", "id": experiment_id})
    assert shown.ok
    assert shown.receipt["experiment"]["hypothesis"].startswith("A short walk")
    assert shown.receipt["experiment"]["target_improvement"] == 3
    assert shown.receipt["observations"][0]["note"] == "Before lunch"
    assert shown.receipt["analysis"]["baseline"] == {"n": 1, "mean": 10}


def test_analysis_matches_hand_calculation_and_does_not_claim_proof(notebook):
    experiment_id = create(notebook)
    for phase, values in (("baseline", (8, 12)), ("trial", (12, 16))):
        for value in values:
            assert record(notebook, experiment_id, phase, value).ok
    result = notebook.handle({"action": "analyse", "id": experiment_id})
    analysis = result.receipt["analysis"]
    assert analysis["baseline"] == {"n": 2, "mean": 10}
    assert analysis["trial"] == {"n": 2, "mean": 14}
    assert analysis["absolute_change"] == 4
    assert analysis["relative_change_percent"] == 40
    assert analysis["improvement"] == 4
    assert analysis["sufficient_data"] is True and analysis["target_met"] is True
    assert analysis["interpretation"] == "descriptive_only"
    assert "do not establish causality or statistical significance" in result.text


def test_lower_is_better_reverses_improvement_but_not_arithmetic(notebook):
    experiment_id = create(notebook, direction="lower", metric="interruptions")
    for phase, values in (("baseline", (8, 12)), ("trial", (4, 6))):
        for value in values:
            record(notebook, experiment_id, phase, value)
    analysis = notebook.handle({"action": "analyse", "id": experiment_id}).receipt["analysis"]
    assert analysis["absolute_change"] == -5
    assert analysis["relative_change_percent"] == -50
    assert analysis["improvement"] == 5
    assert analysis["target_met"] is True


@pytest.mark.parametrize("counts", [(0, 0), (1, 0), (0, 2), (1, 1), (2, 1)])
def test_insufficient_data_never_declares_target_met(notebook, counts):
    experiment_id = create(notebook)
    for phase, count in zip(("baseline", "trial"), counts):
        for _ in range(count):
            record(notebook, experiment_id, phase, 100 if phase == "trial" else 1)
    result = notebook.handle({"action": "analyse", "id": experiment_id})
    assert result.ok and "Insufficient evidence" in result.text
    assert result.receipt["analysis"]["sufficient_data"] is False
    assert result.receipt["analysis"]["target_met"] is None


@pytest.mark.parametrize("baseline", [0, -2])
def test_percentage_omitted_for_zero_or_negative_baseline(notebook, baseline):
    experiment_id = create(notebook)
    for phase, value in (("baseline", baseline), ("trial", 2)):
        for _ in range(2):
            record(notebook, experiment_id, phase, value)
    analysis = notebook.handle({"action": "analyse", "id": experiment_id}).receipt["analysis"]
    assert analysis["relative_change_percent"] is None
    assert analysis["absolute_change"] == 2 - baseline


@pytest.mark.parametrize("value", [True, None, {}, [], "nonsense", "nan", float("inf"), float("-inf"), 1e101])
def test_bad_observations_do_not_mutate_store(notebook, value):
    experiment_id = create(notebook)
    result = record(notebook, experiment_id, "baseline", value)
    assert not result.ok
    assert notebook.handle({"action": "show", "id": experiment_id}).receipt["analysis"]["baseline"]["n"] == 0


@pytest.mark.parametrize("change", [{"title": ""}, {"hypothesis": []}, {"direction": "perhaps"},
                                     {"target_improvement": -1}, {"target_improvement": True},
                                     {"metric": "x" * 121}])
def test_bad_design_does_not_create_experiment(notebook, change):
    result = notebook.handle({"action": "create", "title": "test", "hypothesis": "test",
                              "metric": "score", "direction": "higher", **change})
    assert not result.ok
    assert notebook.handle({"action": "list"}).receipt["experiments"] == []


def test_design_is_immutable_through_api_and_storage(notebook):
    experiment_id = create(notebook)
    assert not notebook.handle({"action": "update", "id": experiment_id, "direction": "lower"}).ok
    assert not record(notebook, experiment_id, "trial", 2, target_improvement=1).ok
    with sqlite3.connect(notebook.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE experiments SET direction='lower' WHERE id=?", (experiment_id,))
    result = notebook.handle({"action": "show", "id": experiment_id})
    assert result.receipt["experiment"]["direction"] == "higher"
    assert result.receipt["analysis"]["trial"]["n"] == 0


def test_archiving_is_scoped_idempotent_and_preserves_readback(notebook):
    first, second = create(notebook), create(notebook, title="Other experiment")
    record(notebook, first, "baseline", 7)
    result = notebook.handle({"action": "archive", "id": first})
    assert result.ok and result.receipt["changed"] is True
    assert notebook.handle({"action": "archive", "id": first}).receipt["changed"] is False
    assert not record(notebook, first, "trial", 8).ok
    assert notebook.handle({"action": "show", "id": first}).receipt["analysis"]["baseline"]["n"] == 1
    assert [e["id"] for e in notebook.handle({}).receipt["experiments"]] == [second]
    assert len(notebook.handle({"action": "list", "include_archived": True}).receipt["experiments"]) == 2


@pytest.mark.parametrize("action", ["show", "analyse", "record", "archive"])
def test_unknown_id_does_not_fall_back_to_another_experiment(notebook, action):
    experiment_id = create(notebook)
    result = notebook.handle({"action": action, "id": "' OR 1=1 --", "phase": "trial", "value": 2})
    assert not result.ok and "No experiment" in result.text
    assert notebook.handle({"action": "show", "id": experiment_id}).receipt["analysis"]["trial"]["n"] == 0


def test_bounded_history_does_not_truncate_analysis(notebook):
    experiment_id = create(notebook)
    for value in range(30):
        assert record(notebook, experiment_id, "baseline", value).ok
    result = notebook.handle({"action": "show", "id": experiment_id})
    assert len(result.receipt["observations"]) == 25
    assert result.receipt["analysis"]["baseline"] == {"n": 30, "mean": 14.5}
    assert not notebook.handle({"action": "list", "limit": 1000}).ok
    assert not notebook.handle({"action": "list", "include_archived": "false"}).ok


def test_concurrent_records_are_not_lost(notebook):
    experiment_id = create(notebook)
    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(lambda value: record(notebook, experiment_id, "baseline", value), range(12)))
    assert all(result.ok for result in outcomes)
    analysis = notebook.handle({"action": "analyse", "id": experiment_id}).receipt["analysis"]
    assert analysis["baseline"] == {"n": 12, "mean": 5.5}


def test_schema_declares_string_id_for_plan_validation_and_coercion(notebook):
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.plan_executor import ToolCatalogue
    from orion_core.tool_args import coerce_args

    declaration = next(d for d in TOOL_DECLARATIONS if d["name"] == "experiment_notebook")
    assert declaration["parameters"]["properties"]["id"]["type"] == "STRING"
    assert "default 20" in declaration["parameters"]["properties"]["limit"]["description"]

    experiment_id = create(notebook)
    catalogue = ToolCatalogue(TOOL_DECLARATIONS)
    assert catalogue.check_args("experiment_notebook", {"action": "show", "id": experiment_id}) is None

    numeric_hex_id = "123456789012345678901234567890e2"
    coerced, error = coerce_args({"action": "show", "id": numeric_hex_id}, declaration)
    assert error is None and coerced["id"] == numeric_hex_id

