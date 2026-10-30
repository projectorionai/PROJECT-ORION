"""Fixed, independently scored evidence for Forge improvement experiments.

The host never imports or executes a candidate here. ``probe_source`` is trusted
measurement code for the existing container verifier; its output is scored on
the host against cases the candidate cannot replace. The caller must supply
execution evidence from its own process runner, never from generated stdout.

Passing these cases establishes behaviour on this benchmark only. A review
recommendation does not activate code, change permissions or train a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any

MAX_CASES = 64
MAX_REPETITIONS = 20
MAX_JSON_BYTES = 65_536
MAX_RECEIPT_BYTES = 1_048_576
MAX_JSON_DEPTH = 16
MAX_JSON_ITEMS = 8_192
RECEIPT_FORMAT = "orion-forge-benchmark-v1"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,59}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_EXPECTED_ERRORS = frozenset({"ValueError", "TypeError", "KeyError", "IndexError"})


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004 - one validation exception contract
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'non-negative'} number")
    return result


def _json_tree(value: Any, *, max_items: int = MAX_JSON_ITEMS) -> None:
    """Bound depth and breadth before serialisation; reject NaN and custom types."""
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > max_items or depth > MAX_JSON_DEPTH:
            raise ValueError("benchmark JSON exceeds its structural budget")
        if item is None or type(item) in {bool, str}:
            continue
        if type(item) in {int, float}:
            try:
                finite = math.isfinite(item)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("benchmark JSON numbers must be finite")
        elif type(item) is list:
            if len(item) > MAX_JSON_ITEMS or count + len(stack) + len(item) > max_items:
                raise ValueError("benchmark JSON list exceeds its budget")
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            if (len(item) > MAX_JSON_ITEMS or count + len(stack) + len(item) > max_items
                    or any(type(key) is not str for key in item)):
                raise ValueError("benchmark JSON objects need bounded string keys")
            stack.extend((child, depth + 1) for child in item.values())
        else:
            raise ValueError("benchmark data must contain JSON values only")


def _canonical(value: Any, limit: int = MAX_JSON_BYTES) -> str:
    _json_tree(value, max_items=65_536 if limit > MAX_JSON_BYTES else MAX_JSON_ITEMS)
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > limit:
        raise ValueError("benchmark JSON exceeds its byte budget")
    return text


def _parse_json(text: str, limit: int = MAX_JSON_BYTES) -> Any:
    if not isinstance(text, str) or len(text.encode("utf-8")) > limit:
        raise ValueError("benchmark JSON exceeds its byte budget")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate benchmark JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid benchmark JSON") from exc
    _json_tree(value, max_items=65_536 if limit > MAX_JSON_BYTES else MAX_JSON_ITEMS)
    return value


@dataclass(frozen=True)
class BenchmarkCase:
    """An immutable argument/oracle pair, expressed as bounded JSON."""

    case_id: str
    arguments_json: str
    expected_json: str = "null"
    expected_error: str | None = None
    absolute_tolerance: float = 1e-9

    def __post_init__(self):
        if not isinstance(self.case_id, str) or not _IDENTIFIER.fullmatch(self.case_id):
            raise ValueError("benchmark case needs a simple identifier")
        if not isinstance(_parse_json(self.arguments_json), dict):
            raise ValueError("benchmark arguments must be an object")  # noqa: TRY004 - invalid benchmark configuration
        _parse_json(self.expected_json)
        if self.expected_error is not None and self.expected_error not in _EXPECTED_ERRORS:
            raise ValueError("unsupported expected benchmark error")
        tolerance = _number(self.absolute_tolerance, "absolute tolerance")
        if tolerance > 0.01:
            raise ValueError("benchmark tolerance must be at most 0.01")

    @classmethod
    def create(cls, case_id: str, arguments: dict, expected: Any = None, *,
               expected_error: str | None = None, absolute_tolerance: float = 1e-9):
        return cls(case_id, _canonical(arguments), _canonical(expected),
                   expected_error, absolute_tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.case_id, "arguments": _parse_json(self.arguments_json),
                "expected": _parse_json(self.expected_json), "expected_error": self.expected_error,
                "absolute_tolerance": self.absolute_tolerance}


@dataclass(frozen=True)
class Benchmark:
    name: str
    version: int
    contract: str
    cases: tuple[BenchmarkCase, ...]
    repetitions: int = 5

    def __post_init__(self):
        if not isinstance(self.name, str) or not _IDENTIFIER.fullmatch(self.name):
            raise ValueError("benchmark needs a simple identifier")
        if type(self.version) is not int or not 1 <= self.version <= 1_000_000:
            raise ValueError("benchmark version must be a positive integer")
        if not isinstance(self.contract, str) or not 1 <= len(self.contract) <= 8_192:
            raise ValueError("benchmark contract is missing or too large")
        if (type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_CASES
                or any(type(case) is not BenchmarkCase for case in self.cases)):
            raise ValueError("benchmark needs a bounded tuple of trusted cases")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("benchmark case identifiers must be unique")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= MAX_REPETITIONS:
            raise ValueError("benchmark repetition count exceeds its budget")

    @property
    def fingerprint(self) -> str:
        text = _canonical({"name": self.name, "version": self.version, "contract": self.contract,
                           "cases": [case.to_dict() for case in self.cases],
                           "repetitions": self.repetitions}, MAX_RECEIPT_BYTES)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


_NUMERIC_CONTRACT = (
    "Implement run(values) returning a JSON object or JSON string with exactly count, sum, "
    "min, max and mean. Accept a bounded list of finite Python integers or floats, excluding "
    "booleans. For an empty list return count=0, sum=0 and min=max=mean=null. Use stable "
    "summation for mixed magnitudes. Raise ValueError for non-lists, booleans, non-numeric "
    "items, or non-finite numbers. The tool must be a pure calculation with no I/O."
)
_NUMERIC_CASES = (
    BenchmarkCase.create("empty", {"values": []},
                         {"count": 0, "sum": 0, "min": None, "max": None, "mean": None}),
    BenchmarkCase.create("positive", {"values": [1, 2, 3]},
                         {"count": 3, "sum": 6, "min": 1, "max": 3, "mean": 2}),
    BenchmarkCase.create("negative", {"values": [-4, -2, 0]},
                         {"count": 3, "sum": -6, "min": -4, "max": 0, "mean": -2}),
    BenchmarkCase.create("fractional", {"values": [0.1, 0.2, 0.3]},
                         {"count": 3, "sum": 0.6, "min": 0.1, "max": 0.3, "mean": 0.2}),
    BenchmarkCase.create("cancellation", {"values": [1e16, 1, -1e16]},
                         {"count": 3, "sum": 1, "min": -1e16, "max": 1e16, "mean": 1 / 3}),
    BenchmarkCase.create("single", {"values": [7]},
                         {"count": 1, "sum": 7, "min": 7, "max": 7, "mean": 7}),
    BenchmarkCase.create("boolean", {"values": [True]}, expected_error="ValueError"),
    BenchmarkCase.create("text_item", {"values": [1, "2"]}, expected_error="ValueError"),
    BenchmarkCase.create("wrong_container", {"values": "1,2,3"}, expected_error="ValueError"),
    BenchmarkCase.create("missing_values", {"values": None}, expected_error="ValueError"),
)
_LATENCY_LIMITATIONS = (
    "Supply exclusive durations, not cumulative timestamps or overlapping stages. "
    "Totals cover the supplied phases only; no model, microphone or connection was tested."
)
_LATENCY_CONTRACT = (
    "Implement the pure calculation contract of ORION's original orion_latency_report plugin. "
    "run(samples) accepts 1 to 256 timing-sample objects, JSON-encoded objects in a list, or "
    "a JSON-encoded list. Each sample contains 1 to 16 exclusive stage durations in "
    "milliseconds; at most 16 distinct stages across samples. Stage identifiers start with "
    "a letter and contain at most 60 ASCII letters, digits, underscores or hyphens. "
    "Durations are finite numbers from 0 to 86400000, excluding booleans. Top-level JSON is "
    "limited to 524288 characters and a JSON sample to 4096 characters. Return the existing "
    "plugin's exact successful JSON payload (or its JSON string): plugin, source, "
    "measured_by_plugin, percentile_method, samples, stages, largest_mean_stage, "
    "supplied_phase_total, incomplete_stage_coverage, recommendations and limitations. "
    "Summaries contain observations, mean_ms, p50_ms, p95_ms and maximum_ms, rounded to "
    "three decimal places using nearest-rank percentiles. Sort stage names and choose the "
    "first sorted stage on a tied largest mean. Preserve the original advice and disclosure "
    "wording; do not invent live timings or mark supplied observations as measured. "
    "In this isolated trial, invalid input raises ValueError; the live plugin normally "
    "wraps that rejection in ToolResult(ok=False). No I/O or host imports."
)


def _summary_expected(observations, mean, p50, p95, maximum):
    return {"observations": observations, "mean_ms": mean, "p50_ms": p50,
            "p95_ms": p95, "maximum_ms": maximum}


def _latency_expected(stages, total, samples, largest, incomplete=False):
    advice = [f"Inspect {largest} first: it has the largest mean supplied duration."]
    if samples < 20:
        advice.append("Collect at least 20 comparable requests before treating tail percentiles as stable.")
    if incomplete:
        advice.append("Stage coverage differs between samples; compare per-stage observation counts.")
    return {"plugin": "orion_latency_report", "source": "supplied_observations",
            "measured_by_plugin": False, "percentile_method": "nearest_rank", "samples": samples,
            "stages": stages, "largest_mean_stage": largest, "supplied_phase_total": total,
            "incomplete_stage_coverage": incomplete, "recommendations": advice,
            "limitations": _LATENCY_LIMITATIONS}


_CAPTURE_100 = _summary_expected(1, 100, 100, 100, 100)
_MODEL_300 = _summary_expected(1, 300, 300, 300, 300)
_LATENCY_CASES = (
    BenchmarkCase.create("complete_pipeline", {"samples": [
        {"capture": 100, "model": value, "speech": 300} for value in [500, 700, 900, 1100]
    ]}, _latency_expected({"capture": _summary_expected(4, 100, 100, 100, 100),
                          "model": _summary_expected(4, 800, 700, 1100, 1100),
                          "speech": _summary_expected(4, 300, 300, 300, 300)},
                         _summary_expected(4, 1200, 1100, 1500, 1500), 4, "model")),
    BenchmarkCase.create("partial_coverage", {"samples": [{"capture": 100}, {"capture": 200, "model": 900}]},
                         _latency_expected({"capture": _summary_expected(2, 150, 100, 200, 200),
                                            "model": _summary_expected(1, 900, 900, 900, 900)},
                                           _summary_expected(2, 600, 100, 1100, 1100), 2, "model", True)),
    BenchmarkCase.create("tail_rank", {"samples": [{"model": value} for value in range(1, 21)]},
                         _latency_expected({"model": _summary_expected(20, 10.5, 10, 19, 20)},
                                           _summary_expected(20, 10.5, 10, 19, 20), 20, "model")),
    BenchmarkCase.create("fractional_rounding", {"samples": [{"model": 0.125}, {"model": 0.375}]},
                         _latency_expected({"model": _summary_expected(2, 0.25, 0.125, 0.375, 0.375)},
                                           _summary_expected(2, 0.25, 0.125, 0.375, 0.375), 2, "model")),
    BenchmarkCase.create("zero_tie", {"samples": [{"model": 0, "capture": 0}]},
                         _latency_expected({"capture": _summary_expected(1, 0, 0, 0, 0),
                                            "model": _summary_expected(1, 0, 0, 0, 0)},
                                           _summary_expected(1, 0, 0, 0, 0), 1, "capture")),
    BenchmarkCase.create("json_array", {"samples": '[{"capture":100,"model":300}]'},
                         _latency_expected({"capture": _CAPTURE_100, "model": _MODEL_300},
                                           _summary_expected(1, 400, 400, 400, 400), 1, "model")),
    BenchmarkCase.create("json_items", {"samples": ['{"capture":100,"model":300}']},
                         _latency_expected({"capture": _CAPTURE_100, "model": _MODEL_300},
                                           _summary_expected(1, 400, 400, 400, 400), 1, "model")),
    BenchmarkCase.create("maximum_sample_count", {"samples": [{"model": value} for value in range(256)]},
                         _latency_expected({"model": _summary_expected(256, 127.5, 127, 243, 255)},
                                           _summary_expected(256, 127.5, 127, 243, 255), 256, "model")),
    BenchmarkCase.create("negative_duration", {"samples": [{"model": -1}]}, expected_error="ValueError"),
    BenchmarkCase.create("boolean_duration", {"samples": [{"model": True}]}, expected_error="ValueError"),
    BenchmarkCase.create("duration_limit", {"samples": [{"model": 86_400_001}]}, expected_error="ValueError"),
    BenchmarkCase.create("empty_samples", {"samples": []}, expected_error="ValueError"),
    BenchmarkCase.create("empty_stage_map", {"samples": [{}]}, expected_error="ValueError"),
    BenchmarkCase.create("sample_count_limit", {"samples": [{"model": 1}] * 257}, expected_error="ValueError"),
    BenchmarkCase.create("stage_union_limit", {"samples": [{f"stage{i}": 1} for i in range(17)]},
                         expected_error="ValueError"),
    BenchmarkCase.create("bad_identifier", {"samples": [{"secret / URL": 1}]}, expected_error="ValueError"),
    BenchmarkCase.create("wrong_samples_type", {"samples": None}, expected_error="ValueError"),
)
_CATALOGUE = {
    "numeric_summary_v1": Benchmark("numeric_summary_v1", 1, _NUMERIC_CONTRACT, _NUMERIC_CASES),
    "latency_report_v1": Benchmark("latency_report_v1", 1, _LATENCY_CONTRACT, _LATENCY_CASES),
}

_LATENCY_BASELINE = '''import json
import math
import re

def get_tool_schema():
    return {"name": __TOOL_NAME__, "description": "Analyse supplied exclusive stage timings",
            "parameters": {"type": "object", "properties": {"samples": {"type": "array"}},
                           "required": ["samples"]}}

def summary(values):
    ordered = sorted(values)
    return {"observations": len(values), "mean_ms": round(sum(values) / len(values), 3),
            "p50_ms": round(ordered[math.ceil(len(values) * 0.5) - 1], 3),
            "p95_ms": round(ordered[math.ceil(len(values) * 0.95) - 1], 3),
            "maximum_ms": round(ordered[-1], 3)}

def run(samples):
    if isinstance(samples, str):
        if len(samples) > 524288:
            raise ValueError("timing JSON exceeds the 512 KiB limit")
        try:
            samples = json.loads(samples)
        except ValueError as exc:
            raise ValueError("Invalid timing JSON") from exc
    if not isinstance(samples, list) or not 1 <= len(samples) <= 256:
        raise ValueError("supply 1 to 256 timing samples")
    stages: dict[str, list[float]] = {}
    totals = []
    for sample in samples:
        if isinstance(sample, str):
            if len(sample) > 4096:
                raise ValueError("a timing sample exceeds the 4 KiB JSON limit")
            try:
                sample = json.loads(sample)
            except ValueError as exc:
                raise ValueError("Invalid timing sample JSON") from exc
        if not isinstance(sample, dict) or not 1 <= len(sample) <= 16:
            raise ValueError("each sample must contain 1 to 16 exclusive timed stages")
        total = 0.0
        for name, value in sample.items():
            if not isinstance(name, str) or not re.fullmatch("[a-zA-Z][a-zA-Z0-9_-]{0,59}", name):
                raise ValueError("invalid stage identifier")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("durations must be numbers, without boolean values")
            try:
                duration = float(value)
            except OverflowError as exc:
                raise ValueError("duration must be finite") from exc
            if not math.isfinite(duration) or not 0 <= duration <= 86400000:
                raise ValueError("duration must be finite and within bounds")
            stages.setdefault(name, []).append(duration)
            if len(stages) > 16:
                raise ValueError("at most 16 distinct timed stages are supported")
            total += duration
        totals.append(total)
    breakdown = {name: summary(values) for name, values in sorted(stages.items())}
    names = sorted(breakdown)
    largest = names[0]
    for name in names[1:]:
        if breakdown[name]["mean_ms"] > breakdown[largest]["mean_ms"]:
            largest = name
    advice = [f"Inspect {largest} first: it has the largest mean supplied duration."]
    if len(samples) < 20:
        advice.append("Collect at least 20 comparable requests before treating tail percentiles as stable.")
    incomplete = any(len(values) != len(samples) for values in stages.values())
    if incomplete:
        advice.append("Stage coverage differs between samples; compare per-stage observation counts.")
    return {"plugin": "orion_latency_report", "source": "supplied_observations",
            "measured_by_plugin": False, "percentile_method": "nearest_rank",
            "samples": len(samples), "stages": breakdown, "largest_mean_stage": largest,
            "supplied_phase_total": summary(totals), "incomplete_stage_coverage": incomplete,
            "recommendations": advice, "limitations": __LIMITATIONS__}
'''


def get_benchmark(name: str) -> Benchmark:
    """Select trusted cases by identifier; generated code cannot supply an oracle."""
    if not isinstance(name, str) or name not in _CATALOGUE:
        raise ValueError("choose a registered Forge benchmark")
    return _CATALOGUE[name]


def benchmark_catalogue() -> list[dict[str, Any]]:
    return [{"name": item.name, "version": item.version, "contract": item.contract,
             "cases": len(item.cases), "repetitions": item.repetitions,
             "fingerprint": item.fingerprint} for item in _CATALOGUE.values()]


def source_sha256(source: str) -> str:
    """Identify the Python snapshot independently of Windows newline conversion."""
    if not isinstance(source, str) or not source or len(source.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise ValueError("benchmark source is missing or too large")
    normalised = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def baseline_source(benchmark: Benchmark, tool_name: str) -> str:
    """Return the fixed reference implementation, never a model-generated baseline."""
    if not isinstance(tool_name, str) or not _IDENTIFIER.fullmatch(tool_name):
        raise ValueError("invalid benchmark tool name")
    if benchmark == get_benchmark("latency_report_v1"):
        return _LATENCY_BASELINE.replace("__TOOL_NAME__", repr(tool_name)).replace("__LIMITATIONS__", repr(_LATENCY_LIMITATIONS))
    if benchmark != get_benchmark("numeric_summary_v1"):
        raise ValueError("this benchmark has no registered reference implementation")
    return f'''from math import fsum, isfinite

def get_tool_schema():
    return {{"name": {tool_name!r}, "description": "Summarise supplied finite numbers",
            "parameters": {{"type": "object", "properties": {{
                "values": {{"type": "array", "items": {{"type": "number"}}}}}},
                "required": ["values"]}}}}

def run(values):
    if not isinstance(values, list) or len(values) > 8192:
        raise ValueError("Supply a bounded list of finite numbers")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Supply finite numbers, excluding booleans")
        try:
            finite = isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Supply finite numbers")
    try:
        total = fsum(values)
    except OverflowError as exc:
        raise ValueError("The total must be finite") from exc
    return {{"count": len(values), "sum": total,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": total / len(values) if values else None}}
'''


@dataclass(frozen=True)
class ExecutionEvidence:
    """Runner-owned evidence; never construct this from candidate stdout."""

    isolation: str
    environment_id: str
    exit_code: int
    timed_out: bool = False
    memory_exceeded: bool = False
    verification_passed: bool = False
    typecheck: str = "not_run"

    def __post_init__(self):
        if self.isolation != "container":
            raise ValueError("Forge benchmark execution requires container isolation")
        if (not isinstance(self.environment_id, str) or not self.environment_id
                or len(self.environment_id) > 256 or any(ord(c) < 32 for c in self.environment_id)):
            raise ValueError("benchmark execution needs a bounded environment identity")
        if type(self.exit_code) is not int:
            raise ValueError("benchmark exit code must be an integer")
        if any(type(item) is not bool for item in
               (self.timed_out, self.memory_exceeded, self.verification_passed)):
            raise ValueError("benchmark execution flags must be booleans")


@dataclass(frozen=True)
class EvaluationPolicy:
    min_repetitions: int = 5
    min_latency_gain: float = 0.10
    min_latency_gain_ms: float = 0.05
    latency_regression: float = 0.10
    latency_noise_ms: float = 0.05
    cpu_regression: float = 0.10
    cpu_noise_ms: float = 0.05
    memory_regression: float = 0.10
    memory_noise_kib: float = 1_024
    max_case_ms: float = 1_000
    max_peak_rss_kib: float = 512 * 1_024

    def __post_init__(self):
        if type(self.min_repetitions) is not int or not 2 <= self.min_repetitions <= MAX_REPETITIONS:
            raise ValueError("evaluation requires at least two bounded repetitions")
        for name in ("min_latency_gain", "latency_regression", "cpu_regression", "memory_regression"):
            if not 0 <= _number(getattr(self, name), name) <= 1:
                raise ValueError("evaluation ratios must be between zero and one")
        for name in ("min_latency_gain_ms", "latency_noise_ms", "cpu_noise_ms", "memory_noise_kib"):
            _number(getattr(self, name), name)
        _number(self.max_case_ms, "case time budget", positive=True)
        _number(self.max_peak_rss_kib, "peak memory budget", positive=True)


def probe_payload(benchmark: Benchmark, tool_name: str, source: str) -> str:
    """Arguments sent into the verifier; expected answers remain on the host."""
    if not isinstance(tool_name, str) or not _IDENTIFIER.fullmatch(tool_name):
        raise ValueError("invalid benchmark tool name")
    return _canonical({"format": RECEIPT_FORMAT, "benchmark": benchmark.fingerprint,
                       "tool_name": tool_name,
                       "source_sha256": source_sha256(source),
                       "repetitions": benchmark.repetitions,
                       "cases": [{"id": case.case_id, "arguments": _parse_json(case.arguments_json)}
                                 for case in benchmark.cases]}, MAX_RECEIPT_BYTES)


def _matches(actual: Any, expected: Any, tolerance: float) -> bool:
    if type(expected) is bool or type(actual) is bool:
        return type(actual) is type(expected) and actual == expected
    if type(expected) in {int, float}:
        return type(actual) in {int, float} and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)
    if type(expected) is dict:
        return (type(actual) is dict and actual.keys() == expected.keys()
                and all(_matches(actual[key], value, tolerance) for key, value in expected.items()))
    if type(expected) is list:
        return (type(actual) is list and len(actual) == len(expected)
                and all(_matches(a, b, tolerance) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def evaluate_probe(stdout: str, benchmark: Benchmark, source_sha256: str,
                   execution: ExecutionEvidence, policy: EvaluationPolicy | None = None) -> dict[str, Any]:
    """Score bounded observations using the fixed host oracle, failing closed."""
    policy = policy or EvaluationPolicy()
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise ValueError("benchmark source identity must be a SHA-256 digest")
    result: dict[str, Any] = {
        "format": "orion-forge-evaluation-v1", "benchmark": benchmark.name,
        "benchmark_fingerprint": benchmark.fingerprint, "source_sha256": source_sha256,
        "environment_id": execution.environment_id, "isolation": execution.isolation,
        "verified": False, "complete": False, "correct": False,
        "cases": len(benchmark.cases), "repetitions": benchmark.repetitions,
        "samples": 0, "passed_samples": 0, "passed_cases": 0, "failures": [],
        "metrics": None, "case_metrics": {},
        "confidence": {"scope": "fixed_cases_only", "unseen_task_claim": False,
                       "unique_cases": len(benchmark.cases), "repetitions": benchmark.repetitions},
    }
    if (execution.exit_code != 0 or execution.timed_out or execution.memory_exceeded
            or not execution.verification_passed or execution.typecheck != "passed"):
        result["failures"] = ["The runner did not confirm successful, typed container verification."]
        return result
    result["verified"] = True
    try:
        data = _parse_json(stdout.strip(), MAX_RECEIPT_BYTES)
        if not isinstance(data, dict) or set(data) != {"format", "benchmark", "source_sha256", "rows"}:
            raise ValueError("benchmark receipt has an invalid shape")
        if (data["format"] != RECEIPT_FORMAT or data["benchmark"] != benchmark.fingerprint
                or data["source_sha256"] != source_sha256):
            raise ValueError("benchmark receipt does not match its fixed cases and source")
        rows = data["rows"]
        if not isinstance(rows, list) or len(rows) != len(benchmark.cases) * benchmark.repetitions:
            raise ValueError("benchmark receipt is incomplete")
        case_lookup = {case.case_id: case for case in benchmark.cases}
        seen = set()
        by_case: dict[str, list] = {case.case_id: [] for case in benchmark.cases}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                    "id", "repetition", "output", "error", "elapsed_ms", "cpu_ms", "peak_rss_kib"}:
                raise ValueError("benchmark observation has an invalid shape")
            case_id, repetition = row["id"], row["repetition"]
            if (not isinstance(case_id, str) or case_id not in case_lookup
                    or type(repetition) is not int or not 0 <= repetition < benchmark.repetitions
                    or (case_id, repetition) in seen):
                raise ValueError("benchmark observations need unique, known case/repetition pairs")
            seen.add((case_id, repetition))
            for field in ("elapsed_ms", "cpu_ms", "peak_rss_kib"):
                _number(row[field], field, positive=field == "peak_rss_kib")
            if row["error"] is not None and (not isinstance(row["error"], str) or len(row["error"]) > 60):
                raise ValueError("benchmark error identity is invalid")
            _canonical(row["output"])
            case = case_lookup[case_id]
            passed = (row["error"] == case.expected_error and
                      (row["output"] is None if case.expected_error else
                       _matches(row["output"], _parse_json(case.expected_json), case.absolute_tolerance)))
            if row["elapsed_ms"] > policy.max_case_ms or row["peak_rss_kib"] > policy.max_peak_rss_kib:
                passed = False
            by_case[case_id].append({**row, "passed": passed})
        result["complete"] = True
        result["samples"] = len(rows)
        result["passed_samples"] = sum(row["passed"] for batch in by_case.values() for row in batch)
        result["passed_cases"] = sum(all(row["passed"] for row in batch) for batch in by_case.values())
        result["correct"] = result["passed_samples"] == result["samples"]
        result["failures"] = [case_id for case_id, batch in by_case.items() if not all(row["passed"] for row in batch)]
        result["case_metrics"] = {case_id: _metrics(batch) for case_id, batch in by_case.items()}
        result["metrics"] = _metrics(rows)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        result["failures"] = [str(exc)[:160]]
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    latency = sorted(float(row["elapsed_ms"]) for row in rows)
    return {"median_latency_ms": statistics.median(latency),
            "p95_latency_ms": latency[math.ceil(0.95 * len(latency)) - 1],
            "median_cpu_ms": statistics.median(float(row["cpu_ms"]) for row in rows),
            "peak_rss_kib": max(float(row["peak_rss_kib"]) for row in rows),
            "latency_mad_ms": statistics.median(abs(value - statistics.median(latency)) for value in latency)}


def compare_reports(baseline: dict, candidate: dict,
                    policy: EvaluationPolicy | None = None) -> dict[str, Any]:
    """Recommend review only for measured improvement without a known regression.

    Reports are host-created objects, not arbitrary user-supplied score claims.
    Timing confidence is deliberately bounded to measured repetitions; this
    comparison does not claim statistical generalisation to new tasks.
    """
    policy = policy or EvaluationPolicy()
    decision: dict[str, Any] = {"status": "reject", "eligible_for_review": False,
                               "activated": False, "reasons": [], "latency_gain": None}
    reasons = decision["reasons"]
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        reasons.append("Both evaluations are required.")
        return decision
    for report in (baseline, candidate):
        if not report.get("verified") or not report.get("complete") or report.get("metrics") is None:
            reasons.append("Both evaluations need complete independently verified measurements.")
            return decision
        if type(report.get("repetitions")) is not int or report["repetitions"] < policy.min_repetitions:
            reasons.append("Too few repeated measurements for an improvement decision.")
            return decision
    if any(baseline.get(key) != candidate.get(key) for key in
           ("benchmark_fingerprint", "environment_id", "cases", "repetitions", "samples")):
        reasons.append("Baseline and candidate need the same fixed cases and execution environment.")
        return decision
    if not candidate.get("correct"):
        reasons.append("The candidate failed a correctness or resource-budget case.")
        return decision
    for case_id, before in baseline["case_metrics"].items():
        after = candidate["case_metrics"].get(case_id)
        if after is None:
            reasons.append("The candidate is missing a baseline case.")
            return decision
        for metric, ratio, noise in (
                ("p95_latency_ms", policy.latency_regression, policy.latency_noise_ms),
                ("median_cpu_ms", policy.cpu_regression, policy.cpu_noise_ms),
                ("peak_rss_kib", policy.memory_regression, policy.memory_noise_kib)):
            if after[metric] > before[metric] * (1 + ratio) + noise:
                reasons.append(f"Known regression in {case_id}: {metric}.")
    if reasons:
        return decision
    before, after = baseline["metrics"], candidate["metrics"]
    old_ms, new_ms = before["median_latency_ms"], after["median_latency_ms"]
    gain = (old_ms - new_ms) / old_ms if old_ms > 0 else None
    decision["latency_gain"] = gain
    correctness_gain = candidate["passed_cases"] > baseline["passed_cases"]
    measurable_speed_gain = (gain is not None and gain >= policy.min_latency_gain
                             and old_ms - new_ms >= policy.min_latency_gain_ms
                             and old_ms - new_ms > 2 * max(before["latency_mad_ms"], after["latency_mad_ms"]))
    if correctness_gain or measurable_speed_gain:
        decision.update(status="eligible_for_review", eligible_for_review=True)
        reasons.append("Correctness improved on the fixed cases." if correctness_gain else
                       "Repeated latency improved beyond the configured gain and noise thresholds.")
    else:
        decision["status"] = "retain_baseline"
        reasons.append("No measured improvement beyond the configured thresholds; retain the baseline.")
    return decision


# This script runs only in the trusted Linux container. Timeout and memory
# enforcement belong to the external runner, which must kill/reap the container.
_PROBE_SOURCE = r'''"""Trusted Forge measurement probe; run inside the verifier container only."""
import contextlib
import hashlib
import importlib.util
import io
import json
import resource
import sys
import time
from pathlib import Path

MAX_OUTPUT = 65536

class BoundedSink(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > MAX_OUTPUT:
            raise ValueError("candidate console output exceeded its budget")
        return super().write(value)

def main():
    if sys.platform != "linux":
        raise RuntimeError("benchmark probe requires the Linux verifier container")
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    name = payload["tool_name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
        raise ValueError("invalid staged tool name")
    path = Path(name + "_tool.py")
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != payload["source_sha256"]:
        raise ValueError("staged source does not match its benchmark identity")
    spec = importlib.util.spec_from_file_location("_measured_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sink = BoundedSink()
    rows = []
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        spec.loader.exec_module(module)
        for repetition in range(payload["repetitions"]):
            # Alternate case order so one case does not always benefit from a cold process.
            cases = payload["cases"] if repetition % 2 == 0 else list(reversed(payload["cases"]))
            for case in cases:
                arguments = json.loads(json.dumps(case["arguments"], allow_nan=False))
                wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
                output, error = None, None
                try:
                    output = module.run(**arguments)
                except Exception as exc:
                    output, error = None, type(exc).__name__
                else:
                    try:
                        if isinstance(output, str):
                            if len(output.encode("utf-8")) > MAX_OUTPUT:
                                raise ValueError("candidate response exceeded its budget")
                            output = json.loads(output)
                        encoded = json.dumps(output, allow_nan=False, ensure_ascii=False)
                        if len(encoded.encode("utf-8")) > MAX_OUTPUT:
                            raise ValueError("candidate response exceeded its budget")
                        output = json.loads(encoded)
                    except Exception:
                        output, error = None, "InvalidOutput"
                elapsed = (time.perf_counter_ns() - wall_start) / 1e6
                cpu = (time.process_time_ns() - cpu_start) / 1e6
                rows.append({"id": case["id"], "repetition": repetition, "output": output,
                             "error": error, "elapsed_ms": elapsed, "cpu_ms": cpu,
                             "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    receipt = {"format": "orion-forge-benchmark-v1", "benchmark": payload["benchmark"],
               "source_sha256": payload["source_sha256"], "rows": rows}
    text = json.dumps(receipt, allow_nan=False, ensure_ascii=False)
    if len(text.encode("utf-8")) > 1048576:
        raise ValueError("benchmark receipt exceeded its budget")
    print(text)

if __name__ == "__main__":
    main()
'''


def probe_source() -> str:
    """Source for the separate trusted measurement script, never a host runner."""
    return _PROBE_SOURCE


def benchmark_probe_source(benchmark: Benchmark, tool_name: str, source: str) -> str:
    """A self-contained trusted TEST artefact for ``sandbox.verify_tool``.

    This script is supplied by ORION, separately from generated acceptance
    tests. Invoke it only with forced container isolation and required typing.
    It can never be admitted as a generated trial test or run on the host.
    """
    payload = probe_payload(benchmark, tool_name, source)
    return _PROBE_SOURCE.replace(
        '    payload_path = Path(sys.argv[1])\n    payload = json.loads(payload_path.read_text(encoding="utf-8"))',
        f"    payload = json.loads({payload!r})",
    )


__all__ = ["Benchmark", "BenchmarkCase", "EvaluationPolicy", "ExecutionEvidence",
           "baseline_source", "benchmark_catalogue", "benchmark_probe_source", "compare_reports",
           "evaluate_probe", "get_benchmark", "probe_payload", "probe_source", "source_sha256"]
