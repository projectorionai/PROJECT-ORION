"""Independent benchmark oracles and conservative Forge improvement decisions."""

from __future__ import annotations

import ast
import copy
import json

import pytest

from orion_core.forge_evaluation import (
    Benchmark,
    BenchmarkCase,
    EvaluationPolicy,
    ExecutionEvidence,
    baseline_source,
    benchmark_catalogue,
    benchmark_probe_source,
    compare_reports,
    evaluate_probe,
    get_benchmark,
    probe_payload,
    probe_source,
    source_sha256,
)
from orion_core.security import SecuritySanitiser


@pytest.fixture
def benchmark():
    return get_benchmark("numeric_summary_v1")


@pytest.fixture
def execution():
    return ExecutionEvidence("container", "sha256:verified-image|linux-python312|512MiB-1CPU", 0,
                             verification_passed=True, typecheck="passed")


def receipt(benchmark, *, latency=10, cpu=5, rss=20_000, mutate=None, source="a" * 64):
    """Synthetic observations test the scoring policy; they are never claimed as real timings."""
    rows = []
    for repetition in range(benchmark.repetitions):
        for case in benchmark.cases:
            row = {"id": case.case_id, "repetition": repetition,
                   "output": json.loads(case.expected_json), "error": case.expected_error,
                   "elapsed_ms": latency, "cpu_ms": cpu, "peak_rss_kib": rss}
            if mutate:
                mutate(row)
            rows.append(row)
    return {"format": "orion-forge-benchmark-v1", "benchmark": benchmark.fingerprint,
            "source_sha256": source, "rows": rows}


def report(benchmark, execution, **kwargs):
    return evaluate_probe(json.dumps(receipt(benchmark, **kwargs)), benchmark, "a" * 64, execution)


def test_registered_benchmark_has_fixed_contract_and_holdout_oracles(benchmark):
    assert benchmark.name == "numeric_summary_v1"
    assert len(benchmark.cases) == 10 and benchmark.repetitions == 5
    assert len(benchmark.fingerprint) == 64
    assert any(case.case_id == "cancellation" for case in benchmark.cases)
    assert sum(case.expected_error is not None for case in benchmark.cases) == 4
    catalogue = benchmark_catalogue()
    assert catalogue[0]["fingerprint"] == benchmark.fingerprint
    assert "expected" not in catalogue[0]
    with pytest.raises(ValueError, match="registered"):
        get_benchmark("model_supplied_benchmark")


def test_case_input_and_expected_output_are_immutable_snapshots():
    original = {"items": [1, 2]}
    expected = {"answer": [3]}
    case = BenchmarkCase.create("snapshot", original, expected)
    original["items"].append(99)
    expected["answer"].clear()
    assert json.loads(case.arguments_json) == {"items": [1, 2]}
    assert json.loads(case.expected_json) == {"answer": [3]}


@pytest.mark.parametrize("arguments", [None, [], "wrong", {1: "bad key"},
                                        {"value": float("nan")}, {"value": float("inf")},
                                        {"value": 10**1000}, {"value": object()}])
def test_invalid_argument_data_is_rejected(arguments):
    with pytest.raises(ValueError):
        BenchmarkCase.create("invalid", arguments, None)


def test_depth_breadth_cycle_and_byte_budgets_are_enforced():
    nested = []
    for _ in range(20):
        nested = [nested]
    cycle = []
    cycle.append(cycle)
    for data in ({"x": nested}, {"x": cycle}, {"x": [0] * 8193}, {"x": "a" * 70_000}):
        with pytest.raises(ValueError):
            BenchmarkCase.create("too_large", data, None)
    repeated = [0] * 8_000
    with pytest.raises(ValueError):
        BenchmarkCase.create("shared_breadth", {"x": [repeated] * 8_000}, None)


@pytest.mark.parametrize("case", [
    lambda: BenchmarkCase("bad-id", "{}"),
    lambda: BenchmarkCase("valid", '{"x":1,"x":2}'),
    lambda: BenchmarkCase("valid", "[]"),
    lambda: BenchmarkCase("valid", "{}", expected_error="SystemExit"),
    lambda: BenchmarkCase("valid", "{}", absolute_tolerance=True),
    lambda: BenchmarkCase("valid", "{}", absolute_tolerance=0.5),
])
def test_invalid_case_definition_fails_closed(case):
    with pytest.raises(ValueError):
        case()


@pytest.mark.parametrize("changes", [
    {"name": "../../escape"}, {"version": True}, {"contract": ""}, {"cases": []},
    {"repetitions": 0}, {"repetitions": True}, {"repetitions": 21},
])
def test_invalid_benchmark_definition_fails_closed(benchmark, changes):
    params = {"name": benchmark.name, "version": benchmark.version,
              "contract": benchmark.contract, "cases": benchmark.cases,
              "repetitions": benchmark.repetitions}
    params.update(changes)
    with pytest.raises(ValueError):
        Benchmark(**params)


def test_duplicate_cases_and_changed_oracles_change_identity(benchmark):
    with pytest.raises(ValueError, match="unique"):
        Benchmark("duplicate", 1, "Contract", (benchmark.cases[0], benchmark.cases[0]))
    altered = Benchmark("numeric_summary_v1", 1, benchmark.contract,
                        (BenchmarkCase.create("empty", {"values": []}, {"incorrect": 1}),
                         *benchmark.cases[1:]))
    assert altered.fingerprint != benchmark.fingerprint


def test_payload_is_source_bound_and_does_not_contain_expected_answers(benchmark):
    source = baseline_source(benchmark, "trial_summary")
    payload = json.loads(probe_payload(benchmark, "trial_summary", source))
    assert payload["source_sha256"] == source_sha256(source)
    assert payload["benchmark"] == benchmark.fingerprint
    assert all(set(case) == {"id", "arguments"} for case in payload["cases"])
    assert "expected" not in payload
    assert source_sha256(source.replace("\n", "\r\n")) == source_sha256(source)


@pytest.mark.parametrize("name", ["../../escape", "a-b", "123", "", None])
def test_tool_name_cannot_escape_container_stage(benchmark, name):
    with pytest.raises(ValueError):
        baseline_source(benchmark, name)
    with pytest.raises(ValueError):
        probe_payload(benchmark, name, "pass")


def test_reference_source_is_admitted_by_actual_trial_policy(benchmark):
    source = baseline_source(benchmark, "trial_summary")
    assert SecuritySanitiser.guard_forged_source(source, trial_name="trial_summary") == source
    ast.parse(source)
    trusted_test = benchmark_probe_source(benchmark, "trial_summary", source)
    tree = ast.parse(trusted_test)
    assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith("test_") for node in ast.walk(tree))
    assert 'payload_path = Path(sys.argv[1])' not in trusted_test
    assert source_sha256(source) in trusted_test
    assert "resource.RUSAGE_SELF" in trusted_test
    assert "InvalidOutput" in trusted_test
    assert "Linux verifier container" in probe_source()


def test_complete_correct_observations_have_measured_samples_and_scope(benchmark, execution):
    outcome = report(benchmark, execution)
    assert outcome["verified"] and outcome["complete"] and outcome["correct"]
    assert outcome["samples"] == outcome["passed_samples"] == 50
    assert outcome["passed_cases"] == 10
    assert outcome["metrics"] == {"median_latency_ms": 10, "p95_latency_ms": 10,
                                  "median_cpu_ms": 5, "peak_rss_kib": 20_000,
                                  "latency_mad_ms": 0}
    assert outcome["confidence"]["scope"] == "fixed_cases_only"
    assert outcome["confidence"]["unseen_task_claim"] is False


@pytest.mark.parametrize("kwargs", [
    {"exit_code": 1}, {"timed_out": True}, {"memory_exceeded": True},
    {"verification_passed": False}, {"typecheck": "disabled"},
])
def test_runner_failure_cannot_be_overridden_by_candidate_receipt(benchmark, execution, kwargs):
    params = dict(execution.__dict__)
    params.update(kwargs)
    outcome = report(benchmark, ExecutionEvidence(**params))
    assert not outcome["verified"] and not outcome["complete"] and not outcome["correct"]
    assert outcome["samples"] == 0 and outcome["metrics"] is None


@pytest.mark.parametrize("args", [
    ("process", "env", 0), ("container", "", 0), ("container", "a" * 257, 0),
    ("container", "env\nmalformed", 0), ("container", "env", True),
])
def test_execution_identity_requires_runner_owned_container_evidence(args):
    with pytest.raises(ValueError):
        ExecutionEvidence(*args)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown", "source", "fingerprint", "extra_field"])
def test_incomplete_or_mismatched_observations_never_receive_a_score(benchmark, execution, kind):
    data = receipt(benchmark)
    if kind == "missing":
        data["rows"].pop()
    elif kind == "duplicate":
        data["rows"][-1] = copy.deepcopy(data["rows"][0])
    elif kind == "unknown":
        data["rows"][0]["id"] = "fake_case"
    elif kind == "source":
        data["source_sha256"] = "b" * 64
    elif kind == "fingerprint":
        data["benchmark"] = "b" * 64
    else:
        data["rows"][0]["claimed_passed"] = True
    outcome = evaluate_probe(json.dumps(data), benchmark, "a" * 64, execution)
    assert not outcome["complete"] and not outcome["correct"]
    assert outcome["samples"] == 0 and outcome["metrics"] is None


@pytest.mark.parametrize("metric,value", [
    ("elapsed_ms", -1), ("elapsed_ms", float("nan")), ("cpu_ms", True),
    ("peak_rss_kib", 0), ("cpu_ms", "5"), ("elapsed_ms", float("inf")),
])
def test_invalid_metrics_are_not_replaced_with_fake_measurements(benchmark, execution, metric, value):
    data = receipt(benchmark)
    data["rows"][0][metric] = value
    outcome = evaluate_probe(json.dumps(data), benchmark, "a" * 64, execution)
    assert not outcome["complete"] and outcome["metrics"] is None


@pytest.mark.parametrize("kind", ["wrong_answer", "boolean_number", "extra_key", "missing_error", "wrong_error"])
def test_host_oracle_overrules_candidate_test_claims(benchmark, execution, kind):
    data = receipt(benchmark)
    row = next(item for item in data["rows"] if item["id"] == ("boolean" if "error" in kind else "positive"))
    if kind == "wrong_answer":
        row["output"]["sum"] = 999
    elif kind == "boolean_number":
        row["output"]["sum"] = True
    elif kind == "extra_key":
        row["output"]["claimed_correct"] = True
    elif kind == "missing_error":
        row["error"] = None
    else:
        row["error"] = "InvalidOutput"
    outcome = evaluate_probe(json.dumps(data), benchmark, "a" * 64, execution)
    assert outcome["complete"] and not outcome["correct"]
    assert outcome["passed_samples"] == 49
    assert outcome["passed_cases"] == 9


@pytest.mark.parametrize("limits", [{"latency": 1_001}, {"rss": 600_000}])
def test_correct_output_does_not_override_time_or_memory_budgets(benchmark, execution, limits):
    outcome = report(benchmark, execution, **limits)
    assert outcome["complete"] and not outcome["correct"]
    assert outcome["passed_samples"] == 0


def test_equivalent_candidate_is_not_claimed_as_an_improvement(benchmark, execution):
    baseline = report(benchmark, execution)
    decision = compare_reports(baseline, copy.deepcopy(baseline))
    assert decision["status"] == "retain_baseline"
    assert not decision["eligible_for_review"] and not decision["activated"]


def test_measured_speed_gain_with_no_regression_recommends_review_only(benchmark, execution):
    baseline = report(benchmark, execution)
    candidate = report(benchmark, execution, latency=8)
    decision = compare_reports(baseline, candidate)
    assert decision["status"] == "eligible_for_review"
    assert decision["eligible_for_review"] and not decision["activated"]
    assert decision["latency_gain"] == pytest.approx(0.2)


def test_correctness_gain_is_useful_without_a_claimed_speed_gain(benchmark, execution):
    def break_baseline(row):
        if row["id"] == "cancellation":
            row["output"]["sum"] = 0
    baseline = report(benchmark, execution, mutate=break_baseline)
    candidate = report(benchmark, execution)
    decision = compare_reports(baseline, candidate)
    assert decision["eligible_for_review"] and decision["latency_gain"] == 0
    assert "Correctness improved" in decision["reasons"][0]


@pytest.mark.parametrize("changes", [{"cpu": 7}, {"rss": 30_000}, {"latency": 12}])
def test_any_known_case_regression_blocks_review(benchmark, execution, changes):
    baseline = report(benchmark, execution)
    candidate = report(benchmark, execution, **changes)
    decision = compare_reports(baseline, candidate)
    assert decision["status"] == "reject" and not decision["eligible_for_review"]
    assert "regression" in decision["reasons"][0]


def test_one_slow_case_cannot_hide_behind_a_faster_aggregate(benchmark, execution):
    def slow_one(row):
        row["elapsed_ms"] = 20 if row["id"] == "single" else 5
    baseline = report(benchmark, execution)
    candidate = report(benchmark, execution, mutate=slow_one)
    assert candidate["metrics"]["median_latency_ms"] == 5
    decision = compare_reports(baseline, candidate)
    assert decision["status"] == "reject"
    assert "single" in decision["reasons"][0]


def test_tiny_noisy_benchmark_does_not_fabricate_a_performance_improvement(benchmark, execution):
    baseline = report(benchmark, execution, latency=0.02)
    candidate = report(benchmark, execution, latency=0.01)
    assert compare_reports(baseline, candidate)["status"] == "retain_baseline"


def test_timing_dispersion_blocks_promotion_on_an_uncertain_gain(benchmark, execution):
    def varying(row):
        row["elapsed_ms"] = (6, 8, 10, 12, 14)[row["repetition"]]
    baseline = report(benchmark, execution, mutate=varying)
    candidate = report(benchmark, execution, latency=8)
    assert compare_reports(baseline, candidate)["status"] == "retain_baseline"


@pytest.mark.parametrize("change", ["environment", "fingerprint", "repetitions", "incomplete"])
def test_incomparable_or_insufficient_measurements_block_improvement(benchmark, execution, change):
    baseline = report(benchmark, execution)
    candidate = report(benchmark, execution, latency=8)
    if change == "environment":
        candidate["environment_id"] = "different-image-or-cpu-limit"
    elif change == "fingerprint":
        candidate["benchmark_fingerprint"] = "b" * 64
    elif change == "repetitions":
        candidate["repetitions"] = 1
    else:
        candidate["complete"] = False
    assert compare_reports(baseline, candidate)["status"] == "reject"


def test_fast_but_incorrect_candidate_is_rejected(benchmark, execution):
    def wrong(row):
        if row["id"] == "positive":
            row["output"]["sum"] = 0
    baseline = report(benchmark, execution)
    candidate = report(benchmark, execution, latency=1, mutate=wrong)
    assert compare_reports(baseline, candidate)["status"] == "reject"


@pytest.mark.parametrize("params", [{"min_repetitions": 1}, {"min_repetitions": True},
                                    {"min_latency_gain": 2}, {"max_case_ms": 0},
                                    {"memory_noise_kib": -1}, {"cpu_noise_ms": float("nan")}])
def test_policy_configuration_has_explicit_finite_bounds(params):
    with pytest.raises(ValueError):
        EvaluationPolicy(**params)


def test_invalid_stdout_is_not_treated_as_a_passing_benchmark(benchmark, execution):
    for stdout in ("", "claimed all tests passed", '{"rows":[],"rows":[]}', "x" * 1_048_577):
        outcome = evaluate_probe(stdout, benchmark, "a" * 64, execution)
        assert not outcome["complete"] and not outcome["correct"]


def test_negative_and_large_numeric_values_use_absolute_not_relative_tolerance(execution):
    fixed = Benchmark("exact_large", 1, "Return the supplied exact integer",
                      (BenchmarkCase.create("large", {}, {"value": 1e16}),))
    data = receipt(fixed)
    data["rows"][0]["output"]["value"] = 1e16 + 2
    outcome = evaluate_probe(json.dumps(data), fixed, "a" * 64, execution)
    assert outcome["complete"] and not outcome["correct"]


def test_latency_benchmark_is_the_existing_original_plugin_contract():
    from orion_core.plugins_analysis import latency_report

    benchmark = get_benchmark("latency_report_v1")
    assert len(benchmark.cases) == 17
    assert "nearest-rank" in benchmark.contract
    for case in benchmark.cases:
        result = latency_report(**json.loads(case.arguments_json))
        if case.expected_error:
            assert result.ok is False, case.case_id
        else:
            assert result.ok is True, (case.case_id, result.text)
            assert json.loads(result.text) == json.loads(case.expected_json), case.case_id
    source = baseline_source(benchmark, "trial_latency")
    assert SecuritySanitiser.guard_forged_source(source, trial_name="trial_latency") == source
    ast.parse(benchmark_probe_source(benchmark, "trial_latency", source))


def test_latency_benchmark_keeps_supplied_timings_distinct_from_real_probe_timings(execution):
    benchmark = get_benchmark("latency_report_v1")
    outcome = report(benchmark, execution, latency=1, cpu=0.5)
    assert outcome["complete"] and outcome["correct"]
    assert outcome["samples"] == 85
    assert outcome["metrics"]["median_latency_ms"] == 1
    partial = next(case for case in benchmark.cases if case.case_id == "partial_coverage")
    supplied = json.loads(partial.expected_json)
    assert supplied["supplied_phase_total"]["mean_ms"] == 600
    assert supplied["measured_by_plugin"] is False
