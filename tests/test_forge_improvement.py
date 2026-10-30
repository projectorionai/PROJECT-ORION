"""Regression gates for independently measured recursive tool improvement."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core.forge import ForgeOrchestrationManager, ImprovementHeartbeat
from orion_core.forge_evaluation import baseline_source, get_benchmark, source_sha256
from orion_core.forge_improvement import (
    ForgeImprovementEngine,
    _contract,
    resolve_verifier_image,
)
from orion_core.sandbox import SandboxVerificationHarness, VerificationOutcome


def numeric_schema(name):
    return {"name": name, "description": "Summarise supplied numeric values", "parameters": {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}}, "required": ["values"]}}


class Signal:
    def __init__(self):
        self.events = []

    def emit(self, *args):
        self.events.append(args)


def receipt(benchmark, source, *, correct=True, elapsed=1.0):
    rows = []
    for repetition in range(benchmark.repetitions):
        for case in benchmark.cases:
            rows.append({"id": case.case_id, "repetition": repetition,
                         "output": json.loads(case.expected_json) if correct else None,
                         "error": case.expected_error, "elapsed_ms": elapsed,
                         "cpu_ms": elapsed, "peak_rss_kib": 20_000})
    return json.dumps({"format": "orion-forge-benchmark-v1", "benchmark": benchmark.fingerprint,
                       "source_sha256": source_sha256(source), "rows": rows})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")
    bus = SimpleNamespace(log=Signal(), dashboard_event=Signal(), thought=Signal())
    manager = ForgeOrchestrationManager(bus)
    manager.sandbox.staging_dir = tmp_path / "staging"
    manager.sandbox.staging_dir.mkdir()
    engine = ForgeImprovementEngine(manager, tmp_path / "evidence", tmp_path / "tools",
                                    image_resolver=lambda: ("sha256:" + "a" * 64, "fixed-local-image|limits"))
    benchmark = get_benchmark("numeric_summary_v1")
    source = baseline_source(benchmark, "trial_summary")
    test = "from trial_summary_tool import run\nassert run([1, 2])['sum'] == 3\n"
    return manager, engine, benchmark, source, test


async def test_recursive_feedback_uses_independent_evidence_and_retains_inert_candidate(setup):
    manager, engine, benchmark, source, test = setup
    prompts, calls = [], []
    first = source + "\n# failed measured candidate\n"
    second = source + "\n# improved measured candidate\n"

    def generate(name, prompt):
        prompts.append(prompt)
        return (first if len(prompts) == 1 else second, test, [])

    async def verify(code, tests, name, **kwargs):
        calls.append(kwargs)
        assert kwargs["require_container"] is True
        assert kwargs["container_image"] == "sha256:" + "a" * 64
        measured = "Trusted Forge measurement probe" in tests
        return VerificationOutcome(True, name, schema=numeric_schema(name), stage="test", isolation="container", typecheck="passed",
                                   stdout=receipt(benchmark, code, correct=code != first,
                                                  elapsed=1 if code == source else 0.5) if measured else "")

    manager.sandbox.verify_tool = verify
    manager.loader.load_and_register = lambda *_: pytest.fail("measurement must never import or activate a candidate")
    result = await engine.run("trial_summary", benchmark.name, generations=2, generator=generate)
    assert result.ok
    record = result.evidence[0]
    assert record["status"] == "eligible_for_review" and record["best_generation"] == 2
    assert record["generations"][0]["status"] == "reject"
    assert "candidate failed" in prompts[1].lower()
    assert len(calls) == 5  # baseline, generated acceptance and independent cases for each generation
    candidate = json.loads(Path(record["review_candidate"]).read_text(encoding="utf-8"))
    assert candidate["source"] == second and not candidate["activated"]
    assert not engine.source_directory.exists()
    assert manager.bus.thought.events[-1][0]["origin"] == "observation"
    recreated = ForgeImprovementEngine(manager, engine.directory, engine.source_directory)
    restored = await recreated.status(record["run_id"])
    assert restored.ok and restored.evidence[0]["best_generation"] == 2


@pytest.mark.parametrize("arguments", [
    {"generations": 0}, {"generations": 4}, {"generations": True}, {"generations": "3"},
    {"budget_seconds": 0}, {"budget_seconds": 601}, {"budget_seconds": True},
    {"plan": "x" * 4097}, {"plan": None},
])
async def test_invalid_budgets_fail_before_generation(setup, arguments):
    manager, engine, benchmark, _, _ = setup
    manager.code_generator = lambda *_: pytest.fail("bad input must not spend model tokens")
    result = await engine.run("trial_summary", benchmark.name, **arguments)
    assert not result.ok and not engine.ledger.exists()


async def test_unknown_benchmark_and_path_are_rejected(setup):
    manager, engine, _, _, _ = setup
    manager.code_generator = lambda *_: pytest.fail("unknown contracts cannot be scored")
    assert not (await engine.run("../../core", "numeric_summary_v1")).ok
    assert not (await engine.run("trial_summary", "invented_score")).ok


async def test_changed_source_invalidates_adoption_basis(setup):
    manager, engine, benchmark, source, test = setup
    engine.source_directory.mkdir()
    path = engine.source_directory / "trial_summary_tool.py"
    path.write_text(source, encoding="utf-8")

    async def verify(code, tests, name, **kwargs):
        if code != source:
            path.write_text(source + "\n# manual edit", encoding="utf-8")
        return VerificationOutcome(True, name, schema=numeric_schema(name), stage="test", isolation="container", typecheck="passed",
                                   stdout=receipt(benchmark, code, elapsed=1 if code == source else 0.5))

    manager.sandbox.verify_tool = verify
    result = await engine.run("trial_summary", benchmark.name, generations=1,
                              generator=lambda *_: (source + "\n# candidate", test, []))
    assert not result.ok and result.evidence[0]["status"] == "baseline_changed"
    assert path.read_text(encoding="utf-8").endswith("manual edit")


async def test_unchanged_candidate_stops_without_repeated_measurement(setup):
    manager, engine, benchmark, source, test = setup
    calls = []

    async def verify(code, tests, name, **kwargs):
        calls.append(code)
        return VerificationOutcome(True, name, schema=numeric_schema(name), isolation="container", typecheck="passed",
                                   stdout=receipt(benchmark, code))

    manager.sandbox.verify_tool = verify
    result = await engine.run("trial_summary", benchmark.name, generator=lambda *_: (source, test, []))
    assert result.ok and result.evidence[0]["status"] == "retain_baseline"
    assert len(calls) == 1 and len(result.evidence[0]["generations"]) == 1


@pytest.mark.parametrize("source_suffix,requirements", [("\nimport os\n", []), ("", ["requests"])])
async def test_generated_io_and_dependencies_never_run(setup, source_suffix, requirements):
    manager, engine, benchmark, source, test = setup
    calls = []

    async def verify(code, tests, name, **kwargs):
        calls.append(code)
        return VerificationOutcome(True, name, schema=numeric_schema(name), isolation="container", typecheck="passed",
                                   stdout=receipt(benchmark, code))

    manager.sandbox.verify_tool = verify
    result = await engine.run("trial_summary", benchmark.name, generations=1,
                              generator=lambda *_: (source + source_suffix, test, requirements))
    assert result.ok and result.evidence[0]["status"] == "retain_baseline"
    assert len(calls) == 1 and result.evidence[0]["generations"][0]["status"] == "admission_failed"


async def test_failed_baseline_never_requests_generation(setup):
    manager, engine, benchmark, _, _ = setup

    async def verify(*args, **kwargs):
        return VerificationOutcome(False, "trial_summary", isolation="container", typecheck="failed")

    manager.sandbox.verify_tool = verify
    result = await engine.run("trial_summary", benchmark.name,
                              generator=lambda *_: pytest.fail("unmeasured baseline must stop"))
    assert not result.ok and not result.evidence[0]["generations"]


async def test_cancellation_retains_evidence_and_releases_tool_lock(setup):
    manager, engine, benchmark, _, _ = setup
    started = asyncio.Event()

    async def verify(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    manager.sandbox.verify_tool = verify
    task = asyncio.create_task(engine.run("trial_summary", benchmark.name, generator=lambda *_: None))
    await started.wait()
    blocked = await engine.run("trial_summary", benchmark.name, generator=lambda *_: None)
    assert not blocked.ok and "already" in blocked.text
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not manager.__dict__["_name_locks"]
    assert engine.records[-1]["status"] == "cancelled"


async def test_force_container_does_not_mutate_global_policy(tmp_path, monkeypatch):
    bus = SimpleNamespace(log=Signal())
    harness = SandboxVerificationHarness(bus)
    harness.staging_dir = tmp_path
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    seen = []

    async def staged(self, *args):
        seen.append((self._isolation_mode, self._typecheck_policy, self._container_image))
        return VerificationOutcome(True, "demo")

    monkeypatch.setattr(SandboxVerificationHarness, "_verify_staged", staged)
    await harness.verify_tool("x", "y", "demo", require_container=True, container_image="sha256:abc")
    await harness.verify_tool("x", "y", "demo")
    assert seen[0] == ("container", "required", "sha256:abc")
    assert seen[1][0:2] == ("process", "off")
    with pytest.raises(ValueError):
        await harness.verify_tool("x", "y", "demo", require_container="yes")


async def test_container_schema_comes_from_probe_not_candidate_stdout(tmp_path, monkeypatch):
    harness = SandboxVerificationHarness(SimpleNamespace(log=Signal()))
    harness.staging_dir = tmp_path
    source = baseline_source(get_benchmark("numeric_summary_v1"), "demo")
    actual = numeric_schema("demo")
    printed = {"format": "a measurement receipt, not a tool schema"}

    def run(self, script, *args):
        if script == "_orion_conformance.py":
            output = json.dumps(actual) + "\nconformance OK"
        elif script == "_orion_typecheck.py":
            output = "typing passed"
        else:
            output = json.dumps(printed)
        return output, "", 0, False, False

    monkeypatch.setattr(SandboxVerificationHarness, "_run_script", run)
    result = await harness.verify_tool(source, "assert True", "demo", require_container=True)
    assert result.passed and result.schema == actual and result.stdout == json.dumps(printed)


def test_image_identity_is_runner_owned_and_immutable(monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CONTAINER_IMAGE", "reviewed:local")
    commands = []

    def inspect(command, **kwargs):
        commands.append(command)
        assert "shell" not in kwargs and kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout="sha256:" + "a" * 64 + "|linux|amd64\n")

    monkeypatch.setattr("orion_core.forge_improvement.subprocess.run", inspect)
    image, environment = resolve_verifier_image()
    assert image == "sha256:" + "a" * 64 and "nonetwork" in environment
    assert commands[0][-1] == "reviewed:local"


@pytest.mark.parametrize("output", ["not-an-image|linux|amd64", "sha256:" + "a" * 64 + "|windows|amd64"])
def test_invalid_image_or_runtime_fails_closed(monkeypatch, output):
    monkeypatch.setattr("orion_core.forge_improvement.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output))
    with pytest.raises(RuntimeError):
        resolve_verifier_image()


def test_tag_lookup_fallback_resolves_only_exact_local_image(monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CONTAINER_IMAGE", "reviewed:local")
    identity = "sha256:" + "a" * 64
    commands = []

    def docker(command, **kwargs):
        commands.append(command)
        if command[2] == "ls":
            return SimpleNamespace(returncode=0, stdout=f"{identity}|reviewed:local\n")
        if command[-1] == identity:
            return SimpleNamespace(returncode=0, stdout=f"{identity}|linux|amd64\n")
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("orion_core.forge_improvement.subprocess.run", docker)
    image, _ = resolve_verifier_image()
    assert image == identity and len(commands) == 3
    assert commands[-1][-1] == identity


async def test_cancelled_save_drains_worker_before_newer_snapshot(setup, monkeypatch):
    _, engine, _, _, _ = setup
    started, release = threading.Event(), threading.Event()
    from orion_core.atomic_io import atomic_write_text
    writes = []

    def slow_write(path, text):
        if not writes:
            started.set()
            assert release.wait(5)
        writes.append(json.loads(text)["runs"][-1]["status"])
        atomic_write_text(path, text)

    monkeypatch.setattr("orion_core.forge_improvement.atomic_write_text", slow_write)
    pending = asyncio.create_task(engine._save({"run_id": "run", "status": "running"}))
    assert await asyncio.to_thread(started.wait, 5)
    pending.cancel()
    newer = asyncio.create_task(engine._save({"run_id": "run", "status": "cancelled"}))
    await asyncio.sleep(0)
    assert not newer.done() and not engine.records
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await newer
    assert writes == ["running", "cancelled"]
    assert json.loads(engine.ledger.read_text(encoding="utf-8"))["runs"][-1]["status"] == "cancelled"


async def test_save_failure_does_not_publish_unwritten_records(setup, monkeypatch):
    _, engine, _, _, _ = setup

    def fail(*_):
        raise OSError("disk unavailable")

    monkeypatch.setattr("orion_core.forge_improvement.atomic_write_text", fail)
    with pytest.raises(OSError):
        await engine._save({"run_id": "run", "status": "running"})
    assert not engine.records


async def test_reviewed_latency_wrapper_uses_explicit_kernel_reference(setup):
    manager, engine, _, _, _ = setup
    name = "orion_latency_report"
    benchmark = get_benchmark("latency_report_v1")
    engine.source_directory.mkdir()
    wrapper = Path(__file__).resolve().parents[1] / "config" / "custom_tools" / f"{name}_tool.py"
    original = wrapper.read_text(encoding="utf-8")
    path = engine.source_directory / wrapper.name
    path.write_text(original, encoding="utf-8")

    async def verify(source, tests, tool_name, **kwargs):
        assert "from orion_core" not in source
        return VerificationOutcome(True, name, schema=numeric_schema(name), isolation="container", typecheck="passed",
                                   stdout=receipt(benchmark, source))

    manager.sandbox.verify_tool = verify
    reference = baseline_source(benchmark, name)
    test = f"from {name}_tool import run\nassert run([{{'model': 1}}])['samples'] == 1\n"
    result = await engine.run(name, benchmark.name, generator=lambda *_: (reference, test, []))
    record = result.evidence[0]
    assert result.ok and record["baseline_origin"] == "reviewed_standalone_latency_kernel"
    assert record["wrapper_overhead_measured"] is False
    assert record["installed_source_sha256"] == source_sha256(original)
    assert path.read_text(encoding="utf-8") == original


async def test_edited_wrapper_does_not_bypass_admission(setup):
    manager, engine, _, _, _ = setup
    engine.source_directory.mkdir()
    path = engine.source_directory / "orion_latency_report_tool.py"
    path.write_text("import orion_core\nraise RuntimeError('must never import')", encoding="utf-8")
    manager.sandbox.verify_tool = lambda *_: pytest.fail("unknown wrapper must never execute")
    result = await engine.run("orion_latency_report", "latency_report_v1", generator=lambda *_: None)
    assert not result.ok and not result.evidence[0]["generations"]


async def test_fast_candidate_with_a_different_callable_contract_is_rejected(setup):
    manager, engine, benchmark, source, test = setup

    async def verify(code, tests, name, **kwargs):
        schema = numeric_schema(name)
        if code != source:
            schema["parameters"]["properties"]["optional_extra"] = {"type": "string"}
        return VerificationOutcome(True, name, schema=schema, isolation="container", typecheck="passed",
                                   stdout=receipt(benchmark, code, elapsed=1 if code == source else 0.5))

    manager.sandbox.verify_tool = verify
    result = await engine.run("trial_summary", benchmark.name, generations=1,
                              generator=lambda *_: (source + "\n# fast fixture", test, []))
    record = result.evidence[0]
    assert result.ok and record["status"] == "retain_baseline"
    assert "contract" in record["generations"][0]["decision"]["reasons"][0]
    assert not record["review_candidate"]


def test_description_parameter_cannot_hide_a_contract_change():
    original = numeric_schema("summary")
    changed = numeric_schema("summary")
    changed["parameters"]["properties"]["description"] = {"type": "string"}
    assert _contract(original, "summary") != _contract(changed, "summary")
    changed = numeric_schema("summary")
    changed["parameters"]["properties"]["values"]["description"] = "Clearer supplied-data wording"
    assert _contract(original, "summary") == _contract(changed, "summary")


async def test_parallel_experiments_do_not_stack_resource_budgets(setup):
    manager, engine, benchmark, _, _ = setup
    started = asyncio.Event()

    async def verify(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    manager.sandbox.verify_tool = verify
    running = asyncio.create_task(engine.run("trial_summary", benchmark.name, generator=lambda *_: None))
    await started.wait()
    blocked = await engine.run("trial_other", benchmark.name, generator=lambda *_: None)
    assert not blocked.ok and "experiment" in blocked.text
    assert "trial_other" not in manager.__dict__["_name_locks"]
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not engine._experiment_lock.locked()


async def test_heartbeat_measured_action_is_bounded_and_review_only(setup, monkeypatch, tmp_path):
    manager, _, _, _, _ = setup
    calls = []
    monkeypatch.setattr("orion_core.forge.SELF_IMPROVE_DIR", tmp_path)
    monkeypatch.setattr("orion_core.forge.IMPROVE_JOURNAL", tmp_path / "journal.jsonl")

    async def improve(name, benchmark, **kwargs):
        calls.append((name, benchmark, kwargs))
        from orion_core.data import ToolResult
        return ToolResult("Retained review evidence.")

    async def response(*args):
        return {"analysis": "Recorded calculation issue needs a benchmark.", "actions": [
            {"type": "improve_tool", "tool_name": "orion_latency_report", "benchmark": "latency_report_v1"},
            {"type": "improve_tool", "tool_name": "pc_control", "benchmark": "latency_report_v1"},
        ]}

    manager.improve_tool = improve
    monkeypatch.setattr("orion_core.forge.strict_json_turn", response)
    heartbeat = ImprovementHeartbeat(manager.bus, SimpleNamespace(text_available=lambda: True), manager)
    first = await heartbeat.tick()
    second = await heartbeat.tick()
    assert len(calls) == 1 and calls[0][2] == {"generations": 2, "budget_seconds": 240, "tool_plan": ""}
    assert first["forged"][0]["measured"] and not first["forged"][0]["activated"]
    assert first["forged"][1]["skipped"] == "unsupported measured contract"
    assert second["forged"][0]["skipped"] == "measured attempt already made or abandoned"
    assert not any("Recorded calculation" in str(item) for item in manager.bus.log.events)
    assert manager.bus.thought.events[-1][0]["origin"] == "generated_reflection"
