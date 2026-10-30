"""Run fixed reference benchmarks in the reviewed local Linux verifier.

No model request, host candidate import, activation or dependency installation
is performed. These measurements test the evaluator/runtime, not AI quality.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.forge import ForgeOrchestrationManager
from orion_core.forge_evaluation import (
    ExecutionEvidence,
    baseline_source,
    benchmark_catalogue,
    benchmark_probe_source,
    evaluate_probe,
    get_benchmark,
    source_sha256,
)
from orion_core.forge_improvement import ForgeImprovementEngine, resolve_verifier_image
from orion_core.sandbox import SandboxVerificationHarness


async def verify(output: Path | None, exercise_loop: bool = False) -> int:
    image, environment = await asyncio.to_thread(resolve_verifier_image)
    harness = SandboxVerificationHarness(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    reports = []
    for item in benchmark_catalogue():
        benchmark = get_benchmark(item["name"])
        name = "trial_" + benchmark.name
        source = baseline_source(benchmark, name)
        outcome = await harness.verify_tool(
            source, benchmark_probe_source(benchmark, name, source), name,
            require_container=True, container_image=image)
        execution = ExecutionEvidence(
            outcome.isolation, environment, outcome.exit_code, outcome.timed_out,
            outcome.memory_exceeded, outcome.passed, outcome.typecheck)
        report = evaluate_probe(outcome.stdout, benchmark, source_sha256(source), execution)
        report["verification_errors"] = outcome.error_log[:8]
        reports.append(report)
        print(json.dumps({"benchmark": benchmark.name, "correct": report["correct"],
                          "observations": report["samples"], "metrics": report["metrics"],
                          "errors": report["verification_errors"]}, ensure_ascii=False))
    data = {"format": "orion-forge-reference-acceptance-v1", "model_requests": 0,
            "activated": False, "reports": reports}
    loop_passed = True
    if exercise_loop:
        # Deliberately faulty fixtures validate feedback/recursion. They are
        # not an existing user defect or evidence of a hosted model learning.
        with tempfile.TemporaryDirectory(prefix="orion-benchmark-fixture-") as directory:
            root = Path(directory)
            name = "trial_loop_fixture"
            benchmark = get_benchmark("numeric_summary_v1")
            good = baseline_source(benchmark, name)
            faulty = good.replace("total = fsum(values)", "total = sum(values)")
            (root / "tools").mkdir()
            (root / "tools" / f"{name}_tool.py").write_text(faulty, encoding="utf-8")
            test = f"from {name}_tool import run\nassert run([1, 2])['sum'] == 3\n"
            prompts = []

            def generate(_name, prompt):
                prompts.append(prompt)
                candidate = faulty + "\n# first fixture generation\n" if len(prompts) == 1 else good
                return candidate, test, []

            manager = ForgeOrchestrationManager(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
            engine = ForgeImprovementEngine(manager, root / "evidence", root / "tools",
                                            image_resolver=lambda: (image, environment))
            result = await engine.run(name, benchmark.name, generations=2, generator=generate)
            record = result.evidence[0]
            loop_passed = (result.ok and record["status"] == "eligible_for_review"
                           and record["best_generation"] == 2 and len(prompts) == 2
                           and record["baseline"]["passed_cases"] < len(benchmark.cases)
                           and record["generations"][-1]["evaluation"]["correct"])
            record.pop("review_candidate", None)  # temporary fixture artefact does not persist
            record.update(fixture_only=True, live_model_requests=0)
            data["recursive_fixture"] = record
            print(json.dumps({"recursive_fixture_passed": loop_passed,
                              "status": record["status"], "best_generation": record["best_generation"]}))
    if output:
        from orion_core.atomic_io import atomic_write_text
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output, json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if loop_passed and all(report["correct"] for report in reports) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Existing reviewed Linux verifier image; never pulled")
    parser.add_argument("--output", type=Path, help="Local reference-measurement report")
    parser.add_argument("--exercise-loop", action="store_true", help="Exercise two fixed faulty/corrected generations; no model calls")
    arguments = parser.parse_args()
    if arguments.image:
        os.environ["ORION_FORGE_CONTAINER_IMAGE"] = arguments.image
    try:
        return asyncio.run(verify(arguments.output, arguments.exercise_loop))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Forge reference verification could not complete ({type(exc).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
