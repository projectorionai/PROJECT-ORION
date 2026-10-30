"""Original plugins calculate genuine results without dispatching actions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core.dynamic_loader import ReflectiveModuleLoader
from orion_core.plugin_manifest import load_manifest
from orion_core.plugins_analysis import batch_planner, latency_report, schema


ROOT = Path(__file__).resolve().parents[1]


def report(result):
    assert result.ok, result.text
    return json.loads(result.text)


def test_latency_percentiles_and_bottleneck_match_supplied_observations():
    samples = [{"capture": 100, "model": value, "speech": 300} for value in [500, 700, 900, 1100]]
    result = report(latency_report(samples))
    assert result["largest_mean_stage"] == "model"
    assert result["stages"]["model"] == {"observations": 4, "mean_ms": 800, "p50_ms": 700, "p95_ms": 1100, "maximum_ms": 1100}
    assert result["supplied_phase_total"]["mean_ms"] == 1200
    assert result["measured_by_plugin"] is False
    assert result["incomplete_stage_coverage"] is False


def test_incomplete_stage_coverage_is_disclosed():
    result = report(latency_report([{"capture": 100}, {"capture": 200, "model": 900}]))
    assert result["stages"]["model"]["observations"] == 1
    assert result["incomplete_stage_coverage"] is True


def test_latency_accepts_actual_gemini_live_free_form_transport():
    from google.genai import types
    from orion_core.gemini_schema import sanitise_declaration
    converted = sanitise_declaration(schema("orion_latency_report"))
    assert converted["parameters"]["properties"]["samples"]["items"]["type"] == "STRING"
    types.FunctionDeclaration(**converted)
    result = report(latency_report(['{"capture":100,"model":300}']))
    assert result["stages"]["model"]["mean_ms"] == 300


def test_batch_schema_accepts_live_sdk_without_unsupported_keywords():
    from google.genai import types
    from orion_core.gemini_schema import sanitise_declaration
    types.FunctionDeclaration(**sanitise_declaration(schema("orion_batch_planner")))


@pytest.mark.parametrize("samples", ['not json', 'x' * 524_289, ['x' * 4097]],
                         ids=["invalid-json", "oversized-json", "oversized-item"])
def test_invalid_or_oversized_json_transport_is_refused(samples):
    assert not latency_report(samples).ok


@pytest.mark.parametrize("samples", [None, [], [True], [{}], [{"model": -1}], [{"model": True}],
    [{"model": float("nan")}], [{"model": float("inf")}], [{"model": 86_400_001}],
    [{"secret / URL": 1}], [{"model": 1}] * 257, [{f"stage{i}": 1 for i in range(17)}]])
def test_bad_timing_samples_fail_without_invented_measurements(samples):
    assert not latency_report(samples).ok


def test_timing_stage_union_is_bounded():
    assert not latency_report([{f"stage{i}": 1} for i in range(17)]).ok


def test_dependency_tasks_run_in_later_proposed_batches():
    tasks = [{"id": "readA", "kind": "read", "estimate_ms": 100},
             {"id": "readB", "kind": "read", "estimate_ms": 200},
             {"id": "analyse", "kind": "compute", "depends_on": ["readA", "readB"], "estimate_ms": 300},
             {"id": "email", "kind": "external", "depends_on": ["analyse"], "estimate_ms": 50}]
    result = report(batch_planner(tasks))
    assert [batch["tasks"] for batch in result["batches"]] == [["readA", "readB"], ["analyse"], ["email"]]
    assert result["planned_duration_ms"] == 550
    assert result["dependency_critical_path_ms"] == 550
    assert result["requires_action_review"] == ["email"]
    assert result["executed"] is False


def test_unknown_and_mutating_tasks_are_serial_barriers():
    tasks = [{"id": "a", "kind": "read"}, {"id": "b", "kind": "compute"},
             {"id": "write", "kind": "write"}, {"id": "unknown"},
             {"id": "delete", "kind": "destructive"}, {"id": "last", "kind": "read"}]
    result = report(batch_planner(tasks, 8))
    assert [batch["tasks"] for batch in result["batches"]] == [["a", "b"], ["write"], ["unknown"], ["delete"], ["last"]]
    assert [batch["serial_barrier"] for batch in result["batches"]] == [False, True, True, True, False]


def test_concurrency_limit_is_respected():
    result = report(batch_planner([{"id": f"task{i}", "kind": "read"} for i in range(7)], 2))
    assert [len(batch["tasks"]) for batch in result["batches"]] == [2, 2, 2, 1]


@pytest.mark.parametrize("tasks", [None, [], [True], [{}], [{"id": "a", "kind": []}],
    [{"id": "a"}, {"id": "a"}], [{"id": "a", "depends_on": ["a"]}],
    [{"id": "a", "depends_on": ["missing"]}],
    [{"id": "a", "depends_on": ["b", "b"]}, {"id": "b"}],
    [{"id": "a", "depends_on": ["b"]}, {"id": "b", "depends_on": ["a"]}],
    [{"id": "a", "estimate_ms": True}], [{"id": "a", "estimate_ms": float("inf")}],
    [{"id": f"task{i}"} for i in range(65)]])
def test_bad_task_graphs_are_refused(tasks):
    assert not batch_planner(tasks).ok


@pytest.mark.parametrize("concurrency", [0, 9, True, 1.5, "3", None])
def test_concurrency_must_be_an_integer_within_bound(concurrency):
    assert not batch_planner([{"id": "demo"}], concurrency).ok


def test_dense_dependency_graph_cannot_exceed_edge_bound():
    assert not batch_planner([{"id": f"task{i}", "depends_on": [f"task{j}" for j in range(i)]} for i in range(34)]).ok


@pytest.mark.parametrize("name", ["orion_latency_report", "orion_batch_planner"])
async def test_original_plugins_load_through_actual_runtime_contract(name):
    folder = ROOT / "config" / "custom_tools"
    manifest = load_manifest(folder / (name + ".plugin.json"))
    assert manifest.tier == "allow" and not manifest.requires and not manifest.permissions
    assert manifest.parameters == schema(name)["parameters"]
    loader = ReflectiveModuleLoader(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    async def forbidden_install(*_):
        pytest.fail("original analysis plugins must not install any dependency")
    loader._install = forbidden_install
    result = await loader.load_and_register(folder / (name + "_tool.py"))
    assert result.succeeded and result.tool_schema["name"] == name
    args = {"samples": [{"model": 123}]} if name == "orion_latency_report" else {"tasks": [{"id": "observe", "kind": "read"}]}
    assert result.handler_fn(**args).ok


def test_customer_mode_does_not_block_reviewed_original_analysis_plugins(monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "1")
    assert report(latency_report([{"model": 5}]))["measured_by_plugin"] is False
    assert report(batch_planner([{"id": "analyse", "kind": "compute"}]))["executed"] is False


@pytest.fixture(scope="module")
def dataset_plugin():
    path = ROOT / "config" / "custom_tools" / "orion_dataset_compare_tool.py"
    spec = importlib.util.spec_from_file_location("reviewed_dataset_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_comparison_has_exact_counts_and_field_changes(dataset_plugin):
    before = [{"id": "same", "value": 1}, {"id": "gone", "old": 2},
              {"id": "edit", "value": 3, "removed": None}]
    after = [{"id": "edit", "value": 4, "added": None}, {"id": "new", "value": 5},
             {"id": "same", "value": 1}]
    result = report(dataset_plugin.run(before, after))
    assert result["counts"] == {"added": 1, "removed": 1, "changed": 1, "unchanged": 1}
    assert result["rows_before"] == result["rows_after"] == 3
    assert result["examples"] == {"added": ["new"], "removed": ["gone"], "changed": [
        {"id": "edit", "added_fields": ["added"], "removed_fields": ["removed"], "changed_fields": ["value"]}]}
    assert result["fields"]["added"] == ["added"]
    assert result["fields"]["removed"] == ["old", "removed"]
    assert result["source"] == "supplied_snapshots"


def test_dataset_order_and_numeric_representations_are_ignored_but_booleans_are_distinct(dataset_plugin):
    before = [{"id": 1, "value": 1}, {"id": "1", "value": False}, {"id": 2, "value": None}]
    after = [{"id": 2}, {"id": "1", "value": 0}, {"id": 1, "value": 1.0}]
    result = report(dataset_plugin.run(before, after))
    assert result["counts"] == {"added": 0, "removed": 0, "changed": 2, "unchanged": 1}
    assert result["examples"]["changed"] == [
        {"id": 2, "added_fields": [], "removed_fields": ["value"], "changed_fields": []},
        {"id": "1", "added_fields": [], "removed_fields": [], "changed_fields": ["value"]}]


def test_dataset_empty_snapshots_and_custom_key(dataset_plugin):
    assert report(dataset_plugin.run([], []))["counts"] == {
        "added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    result = report(dataset_plugin.run([], [{"reference": "row"}], key="reference"))
    assert result["counts"]["added"] == 1 and result["examples"]["added"] == ["row"]


def test_dataset_does_not_mutate_inputs_or_expose_cell_values(dataset_plugin):
    before = [{"id": "known", "payload": "private-old-cell-value"}]
    after = [{"id": "known", "payload": "private-new-cell-value"}]
    original = json.loads(json.dumps([before, after]))
    outcome = dataset_plugin.run(before, after)
    assert outcome.ok and [before, after] == original
    assert "private-old-cell-value" not in outcome.text
    assert "private-new-cell-value" not in outcome.text
    assert outcome.evidence == [json.loads(outcome.text)]


def test_dataset_examples_are_bounded_without_losing_exact_counts(dataset_plugin):
    before = [{"id": f"removed-{i}"} for i in range(25)] + [
        {"id": f"changed-{i}", "value": 0} for i in range(25)]
    after = [{"id": f"added-{i}"} for i in range(25)] + [
        {"id": f"changed-{i}", "value": 1} for i in range(25)]
    result = report(dataset_plugin.run(before, after))
    assert result["counts"] == {"added": 25, "removed": 25, "changed": 25, "unchanged": 0}
    assert {name: len(rows) for name, rows in result["examples"].items()} == {
        "added": 20, "removed": 20, "changed": 20}
    assert result["examples_truncated"] == {"added": True, "removed": True, "changed": True}


def test_dataset_accepts_json_arrays_and_live_json_row_transport(dataset_plugin):
    before = json.dumps([{"id": "same", "score": 1}])
    after = [json.dumps({"id": "same", "score": 2})]
    assert report(dataset_plugin.run(before, after))["counts"]["changed"] == 1
    assert report(dataset_plugin.run(json.dumps(after), after))["counts"]["unchanged"] == 1


@pytest.mark.parametrize("before", [None, True, {}, 42, "no JSON", "{}", "null", "42", [True], [[]], [{}],
    [{"value": "missing key"}], [{"id": ""}], [{"id": " "}], [{"id": None}], [{"id": True}],
    [{"id": 1.0}], [{"id": "x" * 257}], [{"id": 9_007_199_254_740_992}],
    [{"id": "a"}, {"id": "a"}], [{"id": "a", "nested": {}}], [{"id": "a", "nested": []}],
    [{"id": "a", "value": float("nan")}], [{"id": "a", "value": float("inf")}],
    [{"id": "a", "value": -float("inf")}], [{"id": "a", "value": 9_007_199_254_740_992}],
    [{"id": "a", "": 1}], [{"id": "a", " ": 1}], [{"id": "a", "x" * 81: 1}],
    [{"id": "a", "value": "x" * 4097}], [{"id": "a", "value": object()}],
    [{"id": "a", "value": "\ud800"}], [{"id": "a", 1: "bad field name"}],
    [{"id": i} for i in range(513)],
    '[{"id":"a","value":NaN}]', '[{"id":"a","value":Infinity}]',
    '[{"id":"a","score":1,"score":2}]', '[{"id":"a","id":"b"}]',
    ["not JSON"], 'x' * 262_145], ids=[
        "none", "boolean", "object", "integer", "invalid-json", "json-object", "json-null", "json-number",
        "boolean-row", "array-row", "empty-row", "missing-id", "empty-id", "blank-id", "null-id",
        "boolean-id", "float-id", "long-id", "large-id", "duplicate-ids", "nested-object", "nested-array",
        "nan-cell", "infinite-cell", "negative-infinite-cell", "large-integer-cell", "empty-field", "blank-field",
        "long-field", "long-cell", "non-json-cell", "invalid-unicode-cell", "non-string-field", "too-many-rows",
        "json-nan", "json-infinity", "duplicate-json-field", "duplicate-json-id-field", "invalid-json-row",
        "oversized-json",
    ])
def test_dataset_invalid_or_ambiguous_snapshots_fail(dataset_plugin, before):
    assert not dataset_plugin.run(before, []).ok
    assert not dataset_plugin.run([], before).ok


@pytest.mark.parametrize("key", [None, True, 1, "", " ", "x" * 81, []],
                         ids=["none", "boolean", "integer", "empty", "blank", "long", "array"])
def test_dataset_bad_key_field_names_fail(dataset_plugin, key):
    assert not dataset_plugin.run([], [], key=key).ok


def test_dataset_total_distinct_fields_are_bounded_across_both_snapshots(dataset_plugin):
    before = [{"id": "a", **{f"old{i}": i for i in range(16)}}]
    after = [{"id": "a", **{f"new{i}": i for i in range(16)}}]
    assert not dataset_plugin.run(before, after).ok


def test_dataset_maximum_rows_and_fields_are_accepted(dataset_plugin):
    rows = [{"id": i} for i in range(512)]
    assert report(dataset_plugin.run(rows, list(reversed(rows))))["counts"]["unchanged"] == 512
    full = [{"id": "x" * 256, **{f"field{i}": i for i in range(31)}}]
    assert report(dataset_plugin.run(full, full))["counts"]["unchanged"] == 1
    boundary = [{"id": "a", "n" * 80: "x" * 4096, "exact": -9_007_199_254_740_991}]
    assert report(dataset_plugin.run(boundary, boundary))["counts"]["unchanged"] == 1


def test_dataset_data_and_utf8_transport_budgets_are_combined(dataset_plugin):
    large = [{"id": i, "value": "x" * 4096} for i in range(33)]
    assert not dataset_plugin.run(large, large).ok
    assert not dataset_plugin.run(json.dumps(large), json.dumps(large)).ok
    assert not dataset_plugin.run(" " * 140_000 + "[]", large).ok
    unicode_rows = [{"id": i, "value": "\U0001f680" * 4096} for i in range(17)]
    assert not dataset_plugin.run(unicode_rows, []).ok
    assert not dataset_plugin.run(json.dumps(unicode_rows, ensure_ascii=False), []).ok


def test_dataset_invalid_json_errors_do_not_quote_input(dataset_plugin):
    result = dataset_plugin.run('[{"id": "private-cell-sentinel", broken}]', [])
    assert not result.ok and "line 1, column" in result.text
    assert "private-cell-sentinel" not in result.text


def test_dataset_live_schema_preserves_usable_row_transport(dataset_plugin):
    from google.genai import types
    from orion_core.gemini_schema import sanitise_declaration
    declaration = sanitise_declaration(dataset_plugin.get_tool_schema())
    types.FunctionDeclaration(**declaration)
    assert declaration["parameters"]["properties"]["before"]["items"]["type"] == "STRING"
    assert report(dataset_plugin.run(['{"id":"a"}'], ['{"id":"a"}']))["counts"]["unchanged"] == 1


async def test_dataset_manifest_and_loader_contract_require_no_dependencies(dataset_plugin):
    from orion_core.forge_contract import contract_problems
    from orion_core.plugin_registry import audit_permissions, validate_module
    folder = ROOT / "config" / "custom_tools"
    path = folder / "orion_dataset_compare_tool.py"
    manifest = load_manifest(folder / "orion_dataset_compare.plugin.json")
    assert manifest.tier == "allow" and not manifest.requires and not manifest.permissions
    assert manifest.parameters == dataset_plugin.get_tool_schema()["parameters"]
    assert manifest.description == dataset_plugin.get_tool_schema()["description"]
    assert not contract_problems(dataset_plugin.get_tool_schema(), dataset_plugin.run)
    assert validate_module(path) == []
    assert not audit_permissions(path, manifest.permissions)["undeclared"]
    loader = ReflectiveModuleLoader(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    async def forbidden_install(*_):
        pytest.fail("dataset comparison must not install dependencies")
    loader._install = forbidden_install
    loaded = await loader.load_and_register(path)
    assert loaded.succeeded and loaded.tool_schema["name"] == "orion_dataset_compare"
    assert report(loaded.handler_fn([], [{"id": "added"}]))["counts"]["added"] == 1


def test_dataset_reviewed_plugin_works_in_customer_mode(dataset_plugin, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "1")
    assert report(dataset_plugin.run([], [{"id": "row"}]))["counts"]["added"] == 1
