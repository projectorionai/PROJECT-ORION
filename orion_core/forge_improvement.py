"""Bounded, measured Forge generations with retained review evidence.

The generator cannot replace the benchmark oracle. Every source snapshot runs
only in the existing restricted Linux verifier, against an immutable image ID.
The best measured candidate informs the next generation; publication is a
separate existing Forge operation. Nothing here imports a candidate on the host,
installs dependencies, changes permissions or rewrites ORION's core.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_text
from .data import ToolResult
from .forge_artefacts import ForgeArtefacts
from .forge_diagnosis import diagnose
from .forge_evaluation import (
    ExecutionEvidence,
    baseline_source,
    benchmark_catalogue,
    benchmark_probe_source,
    compare_reports,
    evaluate_probe,
    get_benchmark,
    source_sha256,
)
from .forge_isolation import save_review_candidate
from .forge_trial_policy import generation_guidance, guard_trial_source
from .sandbox import SANDBOX_MEMORY_LIMIT_MB
from .security import SecuritySanitiser, SecurityViolation
from .utils import utc_stamp

MAX_GENERATIONS = 3
MAX_RECORDS = 40
MAX_SOURCE_BYTES = 60_000
_NAME = re.compile(r"[a-z][a-z0-9_]{0,59}\Z")
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,250}\Z")
_IMAGE_RECEIPT = re.compile(r"(sha256:[a-f0-9]{64})\|linux\|([a-z0-9_]+)\Z")


def resolve_verifier_image() -> tuple[str, str]:
    """Resolve a locally built image once, without a shell, pull or deployment."""
    image = os.getenv("ORION_FORGE_CONTAINER_IMAGE", "orion-forge-verifier:mark33")
    if not _IMAGE.fullmatch(image):
        raise ValueError("invalid Forge verifier image")
    options = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace",
               "timeout": 10, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    inspect = ["docker", "image", "inspect", "--format", "{{.Id}}|{{.Os}}|{{.Architecture}}"]
    result = subprocess.run([*inspect, image], check=False, **options)
    match = _IMAGE_RECEIPT.fullmatch(result.stdout.strip())
    if result.returncode != 0 or match is None:
        # Some Docker Desktop sessions list a local tag while tag inspection
        # fails. Resolve only its exact local catalogue entry, then inspect
        # the immutable ID; never pull or choose a similarly named image.
        catalogue = subprocess.run(
            ["docker", "image", "ls", "--no-trunc", "--format", "{{.ID}}|{{.Repository}}:{{.Tag}}"], check=False, **options)
        ids = set()
        if catalogue.returncode == 0 and len(catalogue.stdout) <= 1_048_576:
            for line in catalogue.stdout.splitlines():
                identity, separator, tag = line.partition("|")
                if separator and tag == image and re.fullmatch(r"sha256:[a-f0-9]{64}", identity):
                    ids.add(identity)
        if len(ids) == 1:
            result = subprocess.run([*inspect, ids.pop()], check=False, **options)
            match = _IMAGE_RECEIPT.fullmatch(result.stdout.strip())
    if result.returncode != 0 or match is None:
        raise RuntimeError("Build the reviewed Linux Forge verifier image and start Docker before improvement.")
    immutable_id, architecture = match.groups()
    identity = (f"{immutable_id}|linux|{architecture}|{SANDBOX_MEMORY_LIMIT_MB}MiB|"
                "cpu1|pids64|readonly|nonetwork|probe-v1")
    return immutable_id, identity


def _read_source(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("baseline source must be a bounded regular file")
    return path.read_text(encoding="utf-8")


def _artefacts(value: Any) -> ForgeArtefacts:
    if isinstance(value, ForgeArtefacts):
        result = value
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        result = ForgeArtefacts(value[0], value[1], value[2] or [])
    else:
        raise ValueError("the generator must return a module, explicit tests and requirements")
    for source in (result.tool_code, result.test_code):
        if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ValueError("improvement needs bounded source and explicit acceptance tests")
    if result.requirements:
        raise ValueError("measured improvements cannot install generated dependencies")
    return result


def _admit(source: str, name: str, *, test: bool = False) -> None:
    SecuritySanitiser.guard_forged_source(source, f"improvement.{name}")
    guard_trial_source(source, name, test=test, context=f"improvement.{name}")


def _contract(schema: Any, name: str) -> dict[str, Any]:
    """Keep callable parameters stable; descriptive wording may improve."""
    if not isinstance(schema, dict) or schema.get("name") != name or not isinstance(schema.get("parameters"), dict):
        raise ValueError("the measured schema must identify the requested tool and its parameters")

    def normalise(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "description":
                    continue
                if key == "properties" and isinstance(item, dict):
                    result[key] = {parameter: normalise(definition) for parameter, definition in item.items()}
                elif key in {"enum", "const", "default"}:
                    result[key] = item
                else:
                    result[key] = str(item).lower() if key == "type" else normalise(item)
            return result
        if isinstance(value, list):
            return [normalise(item) for item in value]
        return value

    return {"name": name, "parameters": normalise(schema["parameters"])}


class ForgeImprovementEngine:
    """A recursive evidence loop; all generations share a total time budget."""

    def __init__(self, manager: Any, directory: Path, source_directory: Path, *,
                 image_resolver: Callable[[], tuple[str, str]] = resolve_verifier_image):
        self.manager = manager
        self.directory = directory
        self.source_directory = source_directory
        self.image_resolver = image_resolver
        self.records: list[dict[str, Any]] = []
        self._loaded = False
        self._store_lock = asyncio.Lock()
        self._experiment_lock = asyncio.Lock()

    @property
    def ledger(self) -> Path:
        return self.directory / "improvements.json"

    async def _load(self) -> None:
        async with self._store_lock:
            if self._loaded:
                return
            if self.directory.is_symlink() or self.ledger.is_symlink():
                raise ValueError("Forge improvement storage must not be a symlink")
            if self.ledger.exists():
                if self.ledger.stat().st_size > 8_388_608:
                    raise ValueError("Forge improvement ledger exceeds its size budget")
                data = await asyncio.to_thread(lambda: json.loads(self.ledger.read_text(encoding="utf-8")))
                if (not isinstance(data, dict) or data.get("format") != "orion-forge-improvements-v1"
                        or not isinstance(data.get("runs"), list)
                        or any(not isinstance(item, dict) for item in data["runs"])):
                    raise ValueError("Forge improvement ledger is invalid; preserve it for review")
                self.records = data["runs"][-MAX_RECORDS:]
            self._loaded = True

    async def _save(self, record: dict[str, Any]) -> None:
        async with self._store_lock:
            records = [item for item in self.records if item.get("run_id") != record["run_id"]]
            # JSON round-trip takes a stable snapshot for the writer and status reader.
            records.append(json.loads(json.dumps(record, allow_nan=False)))
            records = records[-MAX_RECORDS:]
            text = json.dumps({"format": "orion-forge-improvements-v1", "runs": records},
                              ensure_ascii=False, indent=2, allow_nan=False)
            if len(text.encode("utf-8")) > 8_388_608:
                raise ValueError("Forge improvement evidence exceeds its storage budget")
            if self.directory.is_symlink() or self.ledger.is_symlink():
                raise ValueError("Forge improvement storage must not be a symlink")
            self.directory.mkdir(parents=True, exist_ok=True)
            writer = asyncio.create_task(asyncio.to_thread(atomic_write_text, self.ledger, text))
            cancelled = False
            while not writer.done():
                try:
                    await asyncio.shield(writer)
                except asyncio.CancelledError:
                    cancelled = True
            writer.result()
            self.records = records
            if cancelled:
                raise asyncio.CancelledError
        signal = getattr(self.manager.bus, "dashboard_event", None)
        if signal is not None:
            signal.emit("forge_improvement", self._summary(record))
        # Surface completed measurements as activity, not invented inner
        # reasoning or per-generation terminal chatter.
        if record.get("finished_at"):
            thought = getattr(self.manager.bus, "thought", None)
            if thought is not None:
                thought.emit({
                    "id": f"forge:{record['run_id']}:{record['status']}",
                    "kind": "decision", "origin": "observation",
                    "observed_at": record["finished_at"],
                    "text": f"Forge {record['tool_name']}: {record['status'].replace('_', ' ')}.",
                    "evidence": [f"Benchmark: {record['benchmark']}",
                                 f"Generations attempted: {len(record['generations'])}",
                                 "Candidate activation: no"],
                })

    @staticmethod
    def _summary(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record.get(key) for key in (
            "run_id", "tool_name", "benchmark", "started_at", "finished_at", "status",
            "generation_limit", "best_generation", "review_candidate", "activated", "error")}

    async def status(self, run_id: str = "") -> ToolResult:
        try:
            await self._load()
        except (OSError, ValueError) as exc:
            return ToolResult(f"Forge improvement status is unavailable: {exc}", ok=False)
        if run_id:
            record = next((item for item in self.records if item.get("run_id") == run_id), None)
            if record is None:
                return ToolResult("Unknown Forge improvement run.", ok=False)
            return ToolResult(json.dumps(record, ensure_ascii=False), evidence=[record])
        summaries = [self._summary(item) for item in reversed(self.records)]
        return ToolResult(json.dumps(summaries, ensure_ascii=False), evidence=summaries)

    @staticmethod
    def catalogue() -> ToolResult:
        items = benchmark_catalogue()
        return ToolResult(json.dumps(items, ensure_ascii=False), evidence=items)

    async def _measure(self, name: str, source: str, benchmark: Any,
                       image: str, environment: str) -> tuple[dict, Any]:
        _admit(source, name)
        # The trusted measurement source is intentionally separate from the
        # generated test admission gate: it needs time/resource/importlib.
        probe = benchmark_probe_source(benchmark, name, source)
        outcome = await self.manager.sandbox.verify_tool(
            source, probe, name, require_container=True, container_image=image)
        evidence = ExecutionEvidence(
            isolation=outcome.isolation, environment_id=environment,
            exit_code=outcome.exit_code, timed_out=outcome.timed_out,
            memory_exceeded=outcome.memory_exceeded,
            verification_passed=outcome.passed, typecheck=outcome.typecheck)
        report = evaluate_probe(outcome.stdout, benchmark, source_sha256(source), evidence)
        if outcome.passed:
            try:
                report["contract"] = _contract(outcome.schema, name)
            except ValueError as exc:
                report.update(verified=False, complete=False, correct=False, failures=[str(exc)])
        if not outcome.passed:
            report["verification_errors"] = [str(item)[:400] for item in outcome.error_log[:8]]
        return report, outcome

    async def _learn(self, name: str, feedback: list[str]) -> None:
        if self.manager.lessons is not None and feedback:
            await asyncio.to_thread(
                self.manager.lessons.record_failure, name, diagnose(feedback, name), feedback)

    async def run(self, name: str, benchmark_name: str, *, generations: int = MAX_GENERATIONS,
                  budget_seconds: int = 300, plan: str = "", generator: Callable | None = None) -> ToolResult:
        try:
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise ValueError("use a simple lower-case tool identifier")
            benchmark = get_benchmark(benchmark_name)
            if type(generations) is not int or not 1 <= generations <= MAX_GENERATIONS:
                raise ValueError("improvement permits one to three generations")
            if type(budget_seconds) is not int or not 30 <= budget_seconds <= 600:
                raise ValueError("the total improvement budget must be 30 to 600 seconds")
            if not isinstance(plan, str) or len(plan) > 4_096:
                raise ValueError("the improvement instruction exceeds its size budget")
            gen = generator or self.manager.code_generator
            if gen is None:
                raise ValueError("configure a code-generation provider before improvement")
            await self._load()
        except (OSError, ValueError) as exc:
            return ToolResult(f"Forge improvement cannot start: {exc}", ok=False)
        locks = self.manager.__dict__.setdefault("_name_locks", {})
        lock = locks.setdefault(name, asyncio.Lock())
        if lock.locked():
            return ToolResult(f"'{name}' already has a Forge operation running.", ok=False)
        if self._experiment_lock.locked():
            if locks.get(name) is lock:
                locks.pop(name, None)
            return ToolResult("A measured Forge experiment is already running; wait for its result.", ok=False)
        record: dict[str, Any] = {
            "run_id": uuid.uuid4().hex, "tool_name": name, "benchmark": benchmark.name,
            "benchmark_fingerprint": benchmark.fingerprint, "started_at": utc_stamp(),
            "status": "running", "generation_limit": generations, "budget_seconds": budget_seconds,
            "generations": [], "best_generation": 0, "review_candidate": "", "activated": False,
            "dependencies_installed": False,
        }
        async with lock, self._experiment_lock:
            try:
                try:
                    await self._save(record)
                    async with asyncio.timeout(budget_seconds):
                        await self._run_locked(record, benchmark, plan, gen)
                except asyncio.CancelledError:
                    record.update(status="cancelled", finished_at=utc_stamp())
                    await self._save(record)
                    raise
                except TimeoutError:
                    record.update(status="budget_exhausted", error="The total improvement time budget expired.")
                except Exception as exc:  # noqa: BLE001 - retain session without leaking provider data
                    # Provider exceptions may contain credentials. Expose
                    # their type only; bounded local validation is useful.
                    message = str(exc)[:400] if isinstance(exc, ValueError) else type(exc).__name__
                    record.update(status="failed", error=message)
                record["finished_at"] = utc_stamp()
                await self._save(record)
            finally:
                if locks.get(name) is lock:
                    locks.pop(name, None)
        status = record["status"]
        if status == "eligible_for_review":
            text = (f"Forge measured an improvement for '{name}' on {benchmark.name} and retained it for review. "
                    f"Run {record['run_id']}; {len(record['generations'])} generation(s). "
                    "The current tool remains active; the candidate has not been activated.")
        elif status == "retain_baseline":
            text = (f"Forge retained '{name}' after {len(record['generations'])} measured generation(s): "
                    "no candidate met the correctness, speed and resource requirements.")
        else:
            text = f"Forge improvement stopped ({status}): {record.get('error', 'inspect the recorded evidence')}."
        return ToolResult(text, ok=status in {"eligible_for_review", "retain_baseline"}, evidence=[record])

    async def _run_locked(self, record: dict, benchmark: Any, plan: str, generator: Callable) -> None:
        name = record["tool_name"]
        image, environment = await asyncio.to_thread(self.image_resolver)
        record["environment_id"] = environment
        source_path = self.source_directory / f"{name}_tool.py"
        if self.source_directory.is_symlink() or source_path.is_symlink():
            raise ValueError("the baseline tool path must not be a symlink")
        existing = source_path.exists()
        installed_source = await asyncio.to_thread(_read_source, source_path) if existing else None
        source = installed_source if existing else baseline_source(benchmark, name)
        record["baseline_origin"] = "installed_source_snapshot" if existing else "reference_implementation"
        if (name == "orion_latency_report" and benchmark.name == "latency_report_v1" and installed_source
                and ast.dump(ast.parse(installed_source)) == ast.dump(ast.parse(
                    'from orion_core.plugins_analysis import latency_report, schema\n'
                    'def get_tool_schema() -> dict:\n    return schema("orion_latency_report")\n'
                    'def run(samples):\n    return latency_report(samples)\n'))):
            # The shipped adapter imports the host package and returns a
            # ToolResult. Benchmark only its reviewed standalone calculation
            # contract, never weaken trial admission or claim wrapper timing.
            source = baseline_source(benchmark, name)
            record["baseline_origin"] = "reviewed_standalone_latency_kernel"
            record["wrapper_overhead_measured"] = False
        elif (name == "orion_latency_report" and benchmark.name == "latency_report_v1" and installed_source):
            # Permit the shipped module docstring but no additional executable
            # statements. An arbitrary edited adapter remains unsupported.
            tree = ast.parse(installed_source)
            if (tree.body and isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)
                    and isinstance(tree.body[0].value.value, str)):
                tree.body.pop(0)
            expected = ast.parse(
                'from orion_core.plugins_analysis import latency_report, schema\n'
                'def get_tool_schema() -> dict:\n    return schema("orion_latency_report")\n'
                'def run(samples):\n    return latency_report(samples)\n')
            if ast.dump(tree) == ast.dump(expected):
                source = baseline_source(benchmark, name)
                record["baseline_origin"] = "reviewed_standalone_latency_kernel"
                record["wrapper_overhead_measured"] = False
        record["installed_source_sha256"] = source_sha256(installed_source) if installed_source else None
        record["baseline_source_sha256"] = source_sha256(source)
        baseline, _ = await self._measure(name, source, benchmark, image, environment)
        record["baseline"] = baseline
        if not baseline["verified"] or not baseline["complete"]:
            raise RuntimeError("The baseline could not be independently measured; no generations were requested.")
        best_source, best_report = source, baseline
        feedback = list(baseline["failures"])
        await self._save(record)
        for number in range(1, record["generation_limit"] + 1):
            prompt = (
                f"Measured improvement generation {number}/{record['generation_limit']}. "
                "Keep the exact run/schema contract. Produce explicit meaningful offline tests. "
                "Improve correctness first, then latency without increasing peak memory or CPU. "
                "Use no external dependencies. A fixed independent benchmark decides whether the result improves. "
                "Do not hardcode benchmark answers or claim to change a hosted model's weights.\n"
                f"Contract: {benchmark.contract}\nUser instruction: {plan}\n"
                f"Previous measured feedback: {json.dumps(feedback, ensure_ascii=False)[:4000]}\n"
                f"Current measured metrics: {json.dumps(best_report.get('metrics'))}\n"
                f"Current source:\n{best_source}\n{generation_guidance(name)}"
            )
            entry: dict[str, Any] = {"number": number, "status": "generating"}
            record["generations"].append(entry)
            await self._save(record)
            try:
                artefacts = _artefacts(await self.manager._call(generator, name, prompt))
                _admit(artefacts.tool_code, name)
                _admit(artefacts.test_code, name, test=True)
                entry["source_sha256"] = source_sha256(artefacts.tool_code)
                # A repeated source cannot improve against its own best report.
                if entry["source_sha256"] == source_sha256(best_source):
                    entry.update(status="unchanged", reasons=["The generator returned the current source unchanged."])
                    feedback = entry["reasons"]
                    await self._save(record)
                    break
                outcome = await self.manager.sandbox.verify_tool(
                    artefacts.tool_code, artefacts.test_code, name,
                    require_container=True, container_image=image)
                entry["acceptance"] = {key: outcome.to_dict()[key] for key in (
                    "passed", "stage", "isolation", "typecheck", "timed_out", "memory_exceeded", "exit_code")}
                if not outcome.passed or outcome.isolation != "container" or outcome.typecheck != "passed":
                    feedback = [str(item)[:400] for item in outcome.error_log[:8]] or ["Required container acceptance failed."]
                    entry.update(status="acceptance_failed", reasons=feedback)
                    await self._learn(name, feedback)
                    await self._save(record)
                    continue
                report, measured = await self._measure(name, artefacts.tool_code, benchmark, image, environment)
                decision = compare_reports(best_report, report)
                if best_report.get("contract") != report.get("contract"):
                    decision.update(status="reject", eligible_for_review=False,
                                    reasons=["The candidate changed the measured callable/schema contract."])
                entry.update(status=decision["status"], evaluation=report, decision=decision)
                feedback = [*report["failures"], *decision["reasons"]]
                if decision["eligible_for_review"]:
                    # Bind the review receipt to these exact source/test bytes.
                    candidate = await asyncio.to_thread(
                        save_review_candidate, self.directory / "candidates", name,
                        artefacts.tool_code, artefacts.test_code, measured.to_dict())
                    best_source, best_report = artefacts.tool_code, report
                    record.update(best_generation=number, review_candidate=str(candidate))
                else:
                    await self._learn(name, feedback)
            except (ValueError, SyntaxError, SecurityViolation) as exc:
                feedback = [str(exc)[:400]]
                entry.update(status="admission_failed", reasons=feedback)
                await self._learn(name, feedback)
            await self._save(record)
        # A concurrent manual edit invalidates the basis for adoption, even
        # though this review-only operation never overwrites the installed file.
        changed = source_path.exists() != existing
        if existing and not changed:
            changed = source_sha256(await asyncio.to_thread(_read_source, source_path)) != record["installed_source_sha256"]
        record["status"] = "eligible_for_review" if record["best_generation"] else "retain_baseline"
        if changed:
            record.update(status="baseline_changed", error="The installed source changed during measurement; repeat the comparison.")
