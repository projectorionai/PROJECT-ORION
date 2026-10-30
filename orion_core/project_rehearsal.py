"""Rehearse a project before committing to its deadline.

Local, bounded schedule modelling: PERT estimates expose the dependency path
and slack; seeded triangular sampling estimates schedule uncertainty. Nothing
is executed or persisted, and no model, GUI, or numerical package is needed.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from .data import ToolResult

MAX_TASKS = 60
MAX_TRIALS = 5_000
MAX_DURATION = 1_000_000.0


@dataclass(frozen=True)
class RehearsalTask:
    id: str
    name: str
    optimistic: float
    likely: float
    pessimistic: float
    depends_on: tuple[str, ...]


def _number(value: Any, field: str, *, maximum: float = MAX_DURATION) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a finite number.") from None
    if not math.isfinite(result) or not 0 <= result <= maximum:
        raise ValueError(f"{field} must be between 0 and {maximum:g} and finite.")
    return result


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer between {minimum} and {maximum}.")
    return value


def _text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{field} must be nonempty text of at most {maximum} characters.")
    return value.strip()


def _parse_tasks(raw_tasks: Any) -> list[RehearsalTask]:
    if not isinstance(raw_tasks, list) or not 1 <= len(raw_tasks) <= MAX_TASKS:
        raise ValueError(f"tasks must contain between 1 and {MAX_TASKS} task objects.")
    tasks: dict[str, RehearsalTask] = {}
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            raise ValueError("Each task must be an object.")
        task_id = _text(raw.get("id"), "task id", 80)
        if task_id in tasks:
            raise ValueError(f"Duplicate task id: {task_id}.")
        name = _text(raw.get("name", task_id), f"{task_id}.name", 200)
        values = [_number(raw.get(key), f"{task_id}.{key}")
                  for key in ("optimistic", "likely", "pessimistic")]
        if not values[0] <= values[1] <= values[2]:
            raise ValueError(f"{task_id}: optimistic <= likely <= pessimistic is required.")
        raw_deps = raw.get("depends_on", [])
        if not isinstance(raw_deps, list) or len(raw_deps) > MAX_TASKS:
            raise ValueError(f"{task_id}.depends_on must be a list of task ids.")
        deps = tuple(_text(dep, f"{task_id}.depends_on", 80) for dep in raw_deps)
        if len(set(deps)) != len(deps):
            raise ValueError(f"{task_id} contains duplicate dependencies.")
        tasks[task_id] = RehearsalTask(task_id, name, *values, deps)

    successors: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    degree = {task.id: len(task.depends_on) for task in tasks.values()}
    for task in tasks.values():
        for dep in task.depends_on:
            if dep not in tasks:
                raise ValueError(f"{task.id} depends on unknown task {dep}.")
            successors[dep].append(task.id)
    ready = deque(task_id for task_id in tasks if degree[task_id] == 0)
    ordered: list[RehearsalTask] = []
    while ready:
        task_id = ready.popleft()
        ordered.append(tasks[task_id])
        for successor in successors[task_id]:
            degree[successor] -= 1
            if degree[successor] == 0:
                ready.append(successor)
    if len(ordered) != len(tasks):
        raise ValueError("Task dependencies contain a cycle; use an acyclic project plan.")
    return ordered


def _schedule(tasks: list[RehearsalTask], durations: dict[str, float],
              successors: dict[str, list[str]]) -> tuple[float, dict, dict, dict]:
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for task in tasks:
        starts[task.id] = max((finishes[dep] for dep in task.depends_on), default=0.0)
        finishes[task.id] = starts[task.id] + durations[task.id]
    elapsed = max(finishes.values())
    latest_starts: dict[str, float] = {}
    for task in reversed(tasks):
        latest_finish = min((latest_starts[item] for item in successors[task.id]), default=elapsed)
        latest_starts[task.id] = latest_finish - durations[task.id]
    return elapsed, starts, finishes, latest_starts


def _quantile(values: list[float], probability: float) -> float:
    index = (len(values) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def _analyse(tasks: list[RehearsalTask], trials: int, seed: int,
             deadline: float | None, factors: dict[str, float]) -> dict[str, Any]:
    successors: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        for dep in task.depends_on:
            successors[dep].append(task.id)
    expected = {task.id: ((task.optimistic + 4 * task.likely + task.pessimistic) / 6)
                * factors.get(task.id, 1.0) for task in tasks}
    elapsed, starts, finishes, latest = _schedule(tasks, expected, successors)
    by_id = {task.id: task for task in tasks}
    end = max((task.id for task in tasks if not successors[task.id]), key=finishes.get)
    critical_path = [end]
    while by_id[end].depends_on:
        end = max(by_id[end].depends_on, key=finishes.get)
        critical_path.append(end)
    critical_path.reverse()

    rng = random.Random(seed)
    completion: list[float] = []
    counts = {task.id: 0 for task in tasks}
    for _ in range(trials):
        # Always draw once per task, so the same seed pairs baseline and what-if
        # trials. Only the requested duration factors differ between scenarios.
        sampled = {task.id: rng.triangular(task.optimistic, task.pessimistic, task.likely)
                   * factors.get(task.id, 1.0) for task in tasks}
        duration, sampled_starts, _, sampled_latest = _schedule(tasks, sampled, successors)
        completion.append(duration)
        for task in tasks:
            if math.isclose(sampled_starts[task.id], sampled_latest[task.id],
                            rel_tol=1e-10, abs_tol=1e-9):
                counts[task.id] += 1
    completion.sort()
    rows = []
    for task in tasks:
        slack = max(0.0, latest[task.id] - starts[task.id])
        rows.append({
            "id": task.id, "name": task.name,
            "expected_duration": expected[task.id],
            "earliest_start": starts[task.id], "earliest_finish": finishes[task.id],
            "latest_start": latest[task.id], "latest_finish": latest[task.id] + expected[task.id],
            "slack": slack, "critical": math.isclose(slack, 0.0, abs_tol=1e-9),
            "criticality_pct": counts[task.id] * 100 / trials,
        })
    return {
        "expected_duration": elapsed,
        "critical_path": critical_path,
        "mean": math.fsum(completion) / trials,
        "p50": _quantile(completion, 0.5),
        "p80": _quantile(completion, 0.8),
        "p95": _quantile(completion, 0.95),
        "deadline_probability": (sum(value <= deadline for value in completion) / trials
                                 if deadline is not None else None),
        "tasks": rows,
    }


def analyse_rehearsal(args: dict[str, Any]) -> dict[str, Any]:
    """Return a reproducible, JSON-safe report; invalid input raises ValueError.

    ``tasks`` is a list of objects with id, optimistic/likely/pessimistic and
    optional name/depends_on. All durations and optional deadline use ``unit``
    (default days). ``what_if`` contains {id, multiplier} objects; a mapping of
    task ids to positive duration multipliers is also accepted by Python callers.
    Deadline is elapsed time from a common start, not a calendar date.
    """
    if not isinstance(args, dict):
        raise ValueError("Project rehearsal arguments must be an object.")
    tasks = _parse_tasks(args.get("tasks"))
    unit = _text(args.get("unit", "days"), "unit", 40)
    trials = _integer(args.get("trials", 2_000), "trials", 1, MAX_TRIALS)
    seed = _integer(args.get("seed", 0), "seed", -(2**63), 2**63 - 1)
    deadline = args.get("deadline")
    if deadline is not None:
        deadline = _number(deadline, "deadline", maximum=MAX_DURATION * MAX_TASKS)
    raw_factors = args.get("what_if", [])
    if isinstance(raw_factors, list):
        if len(raw_factors) > MAX_TASKS:
            raise ValueError(f"what_if must contain at most {MAX_TASKS} changes.")
        entries: dict[str, Any] = {}
        for change in raw_factors:
            if not isinstance(change, dict):
                raise ValueError("Each what_if change must contain id and multiplier.")
            task_id = _text(change.get("id"), "what_if id", 80)
            if task_id in entries:
                raise ValueError(f"Duplicate what_if task id: {task_id}.")
            entries[task_id] = change.get("multiplier")
        raw_factors = entries
    if not isinstance(raw_factors, dict) or len(raw_factors) > MAX_TASKS:
        raise ValueError("what_if must contain task ids and positive duration multipliers.")
    factors: dict[str, float] = {}
    by_id = {task.id: task for task in tasks}
    for task_id, value in raw_factors.items():
        if task_id not in by_id:
            raise ValueError(f"what_if refers to unknown task {task_id}.")
        factor = _number(value, f"what_if.{task_id}", maximum=1_000)
        if factor <= 0:
            raise ValueError(f"what_if.{task_id} must be greater than zero.")
        if by_id[task_id].pessimistic * factor > MAX_DURATION:
            raise ValueError(f"what_if.{task_id} makes the duration exceed {MAX_DURATION:g}.")
        factors[task_id] = factor

    baseline = _analyse(tasks, trials, seed, deadline, {})
    report: dict[str, Any] = {
        "unit": unit, "trials": trials, "seed": seed, "deadline": deadline,
        "baseline": baseline,
        "assumptions": [
            "All tasks start as soon as their dependencies finish; parallelism is unlimited.",
            "Task durations are independent; shared delays and resource contention are not modelled.",
            "PERT path and slack use (optimistic + 4*likely + pessimistic)/6; "
            "simulation uses triangular distributions, whose means can differ.",
            "Durations and the elapsed deadline use the declared unit; no calendar is applied.",
            "Bottleneck frequency is the percentage of trials on any critical path; ties count for every tied task.",
            "These are modelled estimates from the supplied assumptions, not guarantees.",
        ],
    }
    if factors:
        alternative = _analyse(tasks, trials, seed, deadline, factors)
        report["what_if"] = alternative
        report["what_if_multipliers"] = factors
        report["comparison"] = {
            "expected_duration_change": alternative["expected_duration"] - baseline["expected_duration"],
            "p80_change": alternative["p80"] - baseline["p80"],
            "deadline_probability_change": (
                alternative["deadline_probability"] - baseline["deadline_probability"]
                if deadline is not None else None),
        }
    return report


def run_rehearsal(args: dict[str, Any]) -> ToolResult:
    """Tool-facing adapter with a readable result and model-visible receipt."""
    try:
        report = analyse_rehearsal(args)
    except ValueError as error:
        return ToolResult(f"Project rehearsal: {error}", ok=False)
    unit = report["unit"]
    baseline = report["baseline"]
    lines = [f"Project rehearsal: {len(baseline['tasks'])} tasks, {report['trials']:,} trials "
             f"(seed {report['seed']}).",
             f"PERT path estimate: {baseline['expected_duration']:.2f} {unit}; "
             f"critical path: {' -> '.join(baseline['critical_path'])}.",
             f"Modelled completion: P50 {baseline['p50']:.2f}, P80 {baseline['p80']:.2f}, "
             f"P95 {baseline['p95']:.2f} {unit}."]
    if report["deadline"] is not None:
        lines.append(f"Chance of finishing within {report['deadline']:g} {unit}: "
                     f"{baseline['deadline_probability']:.1%} under these assumptions.")
    ranked = sorted(baseline["tasks"], key=lambda row: -row["criticality_pct"])
    lines.append("Bottlenecks (share of simulated critical paths): " + "; ".join(
        f"{row['id']} {row['criticality_pct']:.1f}%" for row in ranked[:5]) + ".")
    lines.append("PERT slack: " + "; ".join(
        f"{row['id']} {row['slack']:.2f} {unit}" for row in baseline["tasks"]) + ".")
    if "what_if" in report:
        alternative = report["what_if"]
        comparison = report["comparison"]
        lines.append(f"What-if P80: {alternative['p80']:.2f} {unit} "
                     f"({comparison['p80_change']:+.2f} versus baseline).")
        if report["deadline"] is not None:
            lines.append(f"What-if deadline chance: {alternative['deadline_probability']:.1%} "
                         f"({comparison['deadline_probability_change'] * 100:+.1f} percentage points).")
    lines.extend(report["assumptions"])
    return ToolResult("\n".join(lines), receipt=report)
