"""Original bounded analysis plugins: calculations only, without side effects.

The latency report describes supplied observations. The task planner describes
a proposed order; it never dispatches actions or claims their completion.
"""
from __future__ import annotations

import heapq
import json
import math
import re
from typing import Any

from .data import ToolResult

MAX_SAMPLES = 256
MAX_STAGES = 16
MAX_TASKS = 64
MAX_EDGES = 512
MAX_MILLISECONDS = 86_400_000
_IDENTIFIER = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{0,59}\Z")


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("durations must be numbers, without boolean values")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= MAX_MILLISECONDS:
        raise ValueError("durations must be finite, from 0 to 86,400,000 milliseconds")
    return number


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("names must start with a letter and contain at most 60 letters, digits, underscores or hyphens")
    return value


def _receipt(report: dict[str, Any]) -> ToolResult:
    return ToolResult(json.dumps(report, ensure_ascii=False, indent=2), evidence=[report])


def latency_report(samples: Any) -> ToolResult:
    """Summarise exclusive stage durations using nearest-rank percentiles."""
    try:
        # Gemini represents free-form object items as JSON strings. Decode only
        # that bounded transport shape, rather than rejecting a valid Live call.
        if isinstance(samples, str):
            if len(samples) > 524_288:
                raise ValueError("timing JSON exceeds the 512 KiB limit")
            samples = json.loads(samples)
        if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
            raise ValueError("supply 1 to 256 timing samples")
        stages: dict[str, list[float]] = {}
        totals: list[float] = []
        for sample in samples:
            if isinstance(sample, str):
                if len(sample) > 4096:
                    raise ValueError("a timing sample exceeds the 4 KiB JSON limit")
                sample = json.loads(sample)
            if not isinstance(sample, dict) or not 1 <= len(sample) <= MAX_STAGES:
                raise ValueError("each sample must contain 1 to 16 exclusive timed stages")
            total = 0.0
            for key, value in sample.items():
                name = _identifier(key)
                duration = _number(value)
                stages.setdefault(name, []).append(duration)
                if len(stages) > MAX_STAGES:
                    raise ValueError("at most 16 distinct timed stages are supported")
                total += duration
            totals.append(total)

        def summary(values: list[float]) -> dict[str, Any]:
            ordered = sorted(values)
            return {"observations": len(values), "mean_ms": round(sum(values) / len(values), 3),
                    "p50_ms": round(ordered[math.ceil(len(values) * 0.5) - 1], 3),
                    "p95_ms": round(ordered[math.ceil(len(values) * 0.95) - 1], 3),
                    "maximum_ms": round(ordered[-1], 3)}

        breakdown = {name: summary(values) for name, values in sorted(stages.items())}
        largest = max(breakdown, key=lambda name: breakdown[name]["mean_ms"])
        advice = [f"Inspect {largest} first: it has the largest mean supplied duration."]
        if len(samples) < 20:
            advice.append("Collect at least 20 comparable requests before treating tail percentiles as stable.")
        incomplete = any(len(values) != len(samples) for values in stages.values())
        if incomplete:
            advice.append("Stage coverage differs between samples; compare per-stage observation counts.")
        return _receipt({"plugin": "orion_latency_report", "source": "supplied_observations",
            "measured_by_plugin": False, "percentile_method": "nearest_rank",
            "samples": len(samples), "stages": breakdown, "largest_mean_stage": largest,
            "supplied_phase_total": summary(totals), "incomplete_stage_coverage": incomplete,
            "recommendations": advice,
            "limitations": "Supply exclusive durations, not cumulative timestamps or overlapping stages. "
                           "Totals cover the supplied phases only; no model, microphone or connection was tested."})
    except (ValueError, OverflowError, RecursionError) as exc:
        return ToolResult(f"Timing analysis failed: {exc}.", ok=False)


def batch_planner(tasks: Any, max_concurrency: Any = 3) -> ToolResult:
    """Plan a bounded dependency graph, with serial barriers for side effects."""
    try:
        if (isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int)
                or not 1 <= max_concurrency <= 8):
            raise ValueError("max_concurrency must be an integer from 1 to 8")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
            raise ValueError("supply 1 to 64 tasks")
        rows: dict[str, dict[str, Any]] = {}
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise ValueError("each task must be an object")
            name = _identifier(task.get("id"))
            if name in rows:
                raise ValueError(f"task id {name!r} is duplicated")
            kind = task.get("kind", "unknown")
            if not isinstance(kind, str) or kind not in {"read", "compute", "write", "external", "destructive", "unknown"}:
                raise ValueError(f"task {name!r} has an unsupported kind")
            dependencies = task.get("depends_on", [])
            if not isinstance(dependencies, list) or len(dependencies) > MAX_TASKS:
                raise ValueError("depends_on must be a list with at most 64 task ids")
            checked = [_identifier(value) for value in dependencies]
            if name in checked or len(set(checked)) != len(checked):
                raise ValueError(f"task {name!r} has a self-reference or duplicate dependency")
            rows[name] = {"index": index, "kind": kind, "depends_on": checked,
                          "estimate_ms": _number(task.get("estimate_ms", 0))}
        if sum(len(row["depends_on"]) for row in rows.values()) > MAX_EDGES:
            raise ValueError("at most 512 dependency edges are supported")
        followers: dict[str, list[str]] = {name: [] for name in rows}
        remaining = {name: len(row["depends_on"]) for name, row in rows.items()}
        for name, row in rows.items():
            for dependency in row["depends_on"]:
                if dependency not in rows:
                    raise ValueError(f"task {name!r} references unknown dependency {dependency!r}")
                followers[dependency].append(name)
        ready = [(rows[name]["index"], name) for name, count in remaining.items() if count == 0]
        heapq.heapify(ready)
        groups = []
        finished: dict[str, float] = {}
        planned_duration = 0.0
        while ready:
            _, first = heapq.heappop(ready)
            selected = [first]
            parallel = rows[first]["kind"] in {"read", "compute"}
            # Keep source order amongst simultaneously ready tasks. A write or
            # external action is a serial barrier, even without dependencies.
            while (parallel and ready and len(selected) < max_concurrency
                   and rows[ready[0][1]]["kind"] in {"read", "compute"}):
                selected.append(heapq.heappop(ready)[1])
            duration = max(rows[name]["estimate_ms"] for name in selected)
            planned_duration += duration
            groups.append({"tasks": selected, "parallel": len(selected) > 1,
                           "serial_barrier": not parallel, "estimate_ms": round(duration, 3)})
            for name in selected:
                row = rows[name]
                finished[name] = max((finished[dep] for dep in row["depends_on"]), default=0) + row["estimate_ms"]
            # Newly ready dependants enter the next batch, never alongside a
            # prerequisite which is still part of the current batch.
            for name in selected:
                for follower in followers[name]:
                    remaining[follower] -= 1
                    if remaining[follower] == 0:
                        heapq.heappush(ready, (rows[follower]["index"], follower))
        if len(finished) != len(rows):
            raise ValueError("task dependencies contain a cycle")
        review = [name for name, row in rows.items() if row["kind"] not in {"read", "compute"}]
        return _receipt({"plugin": "orion_batch_planner", "executed": False,
            "max_concurrency": max_concurrency, "batches": groups,
            "planned_duration_ms": round(planned_duration, 3),
            "dependency_critical_path_ms": round(max(finished.values()), 3),
            "requires_action_review": review,
            "limitations": "This is a proposed schedule using supplied estimates. Declare all data dependencies "
                           "and shared-resource conflicts; read/compute labels must be accurate. No action, "
                           "model request or permission change was executed."})
    except (ValueError, OverflowError) as exc:
        return ToolResult(f"Batch planning failed: {exc}.", ok=False)


def schema(name: str) -> dict[str, Any]:
    if name == "orion_latency_report":
        parameters = {"type": "object", "properties": {"samples": {
            "type": "array", "minItems": 1, "maxItems": MAX_SAMPLES,
            "description": "Observed exclusive stage durations in milliseconds, for example [{capture: 100, model: 800, speech: 450}]. No cumulative timestamps.",
            "items": {"type": "object", "minProperties": 1, "maxProperties": MAX_STAGES,
                      "additionalProperties": {"type": "number", "minimum": 0, "maximum": MAX_MILLISECONDS}}}},
            "required": ["samples"]}
        description = "Analyse supplied timing samples, showing stage percentiles and the largest mean stage. Does not measure live hardware or request a model."
    elif name == "orion_batch_planner":
        parameters = {"type": "object", "properties": {
            "tasks": {"type": "array", "minItems": 1, "maxItems": MAX_TASKS,
                      "items": {"type": "object", "properties": {
                          "id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,59}$"},
                          "kind": {"type": "string", "enum": ["read", "compute", "write", "external", "destructive", "unknown"]},
                          "depends_on": {"type": "array", "maxItems": MAX_TASKS, "items": {"type": "string"}},
                          "estimate_ms": {"type": "number", "minimum": 0, "maximum": MAX_MILLISECONDS}},
                          "required": ["id"]}},
            "max_concurrency": {"type": "integer", "minimum": 1, "maximum": 8}}, "required": ["tasks"]}
        description = "Plan supplied task dependencies, grouping independent read/compute tasks and serialising side effects. Produces a proposal only; executes nothing."
    else:
        raise ValueError("unknown original analysis plugin")
    return {"name": name, "description": description, "parameters": parameters}
