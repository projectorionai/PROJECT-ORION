"""Compare bounded supplied record snapshots without file or network access.

Rows are matched by a nonempty string or an exact integer key. Row order has no
meaning; string and integer keys remain distinct. JSON numbers compare by value
(1 equals 1.0), booleans remain distinct from numbers, and a missing field differs
from a present null. Results expose field names and keys, never other cell values.
"""
from __future__ import annotations

import json
import math
from typing import Any

from orion_core.data import ToolResult

MAX_ROWS = 512
MAX_FIELDS = 32
MAX_FIELD_CHARS = 80
MAX_CELL_CHARS = 4096
MAX_KEY_CHARS = 256
MAX_BYTES = 262_144
MAX_INTEGER = 9_007_199_254_740_991
MAX_EXAMPLES = 20


def get_tool_schema() -> dict:
    records = {
        "type": "array", "maxItems": MAX_ROWS,
        "description": "Supplied flat JSON records; may also be a JSON-encoded array. No file paths. Up to 512 rows, 32 distinct fields across both snapshots and 256 KiB combined JSON.",
        "items": {"type": "object", "maxProperties": MAX_FIELDS,
                  "additionalProperties": {"type": ["string", "number", "boolean", "null"]}},
    }
    return {
        "name": "orion_dataset_compare",
        "description": "Compare two supplied flat record snapshots by a unique key. Return exact added, removed, changed and unchanged row counts, field changes and up to 20 examples per category containing IDs and field names only. Read-only calculation; reads no files and calls no services.",
        "parameters": {
            "type": "object", "properties": {
                "before": {**records}, "after": {**records},
                "key": {"type": "string", "minLength": 1, "maxLength": MAX_FIELD_CHARS,
                        "description": "Unique nonempty string or exact integer ID field in every row; defaults to id. String and integer IDs are distinct."},
            }, "required": ["before", "after"],
        },
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("JSON objects must not contain duplicate field names")
        result[name] = value
    return result


def _invalid_constant(_value: str) -> None:
    raise ValueError("JSON numbers must be finite")


def _decode(value: str, budget: list[int], *, count_transport: bool = True) -> Any:
    if len(value) > MAX_BYTES:
        raise ValueError("combined JSON exceeds the 256 KiB budget")
    size = len(value.encode("utf-8"))
    if size > MAX_BYTES:
        raise ValueError("combined JSON exceeds the 256 KiB budget")
    if count_transport:
        budget[0] += size
        if budget[0] > MAX_BYTES:
            raise ValueError("combined JSON exceeds the 256 KiB budget")
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _spend(budget: list[int], size: int, *, transport: bool) -> None:
    budget[1] += size
    if transport:
        budget[0] += size
    if max(budget) > MAX_BYTES:
        raise ValueError("combined JSON exceeds the 256 KiB budget")


def _index(value: Any, label: str, key: str, fields: set[str], budget: list[int]) -> tuple[dict, set[str]]:
    decoded_array = type(value) is str
    if decoded_array:
        value = _decode(value, budget)
    if type(value) is not list or len(value) > MAX_ROWS:
        raise ValueError(f"{label} must contain 0 to 512 JSON records")
    _spend(budget, 2 + max(0, len(value) - 1), transport=not decoded_array)
    indexed = {}
    local_fields: set[str] = set()
    for position, row in enumerate(value, 1):
        row_transport_counted = decoded_array
        if type(row) is str:
            # Live's free-form object transport represents each row as JSON.
            if len(row) > MAX_BYTES:
                raise ValueError("combined JSON exceeds the 256 KiB budget")
            if not row_transport_counted:
                budget[0] += _size(row)
                if budget[0] > MAX_BYTES:
                    raise ValueError("combined JSON exceeds the 256 KiB budget")
            row = _decode(row, budget, count_transport=False)
            row_transport_counted = True
        if type(row) is not dict or not 1 <= len(row) <= MAX_FIELDS:
            raise ValueError(f"{label} row {position} must be an object with 1 to 32 fields")
        _spend(budget, 2 + max(0, len(row) - 1), transport=not row_transport_counted)
        for name, cell in row.items():
            if type(name) is not str or not name.strip() or len(name) > MAX_FIELD_CHARS:
                raise ValueError("field names must be nonblank strings with at most 80 characters")
            cell_type = type(cell)
            if cell_type is str:
                if len(cell) > MAX_CELL_CHARS:
                    raise ValueError("text cells must contain at most 4096 characters")
            elif cell_type is int:
                if abs(cell) > MAX_INTEGER:
                    raise ValueError("integers must stay within the exact JSON transport range (+/-9007199254740991)")
            elif cell_type is float:
                if not math.isfinite(cell):
                    raise ValueError("numbers must be finite")
            elif cell_type not in (bool, type(None)):
                raise ValueError("cells must be flat JSON scalars, without nested objects or arrays")
            fields.add(name)
            local_fields.add(name)
            if len(fields) > MAX_FIELDS:
                raise ValueError("at most 32 distinct fields are supported across both snapshots")
            _spend(budget, _size(name) + 1 + _size(cell), transport=not row_transport_counted)
        identity = row.get(key)
        if (type(identity) is str and identity.strip() and len(identity) <= MAX_KEY_CHARS):
            pass
        elif type(identity) is int and abs(identity) <= MAX_INTEGER:
            pass
        else:
            raise ValueError(f"{label} row {position} needs the key field as a nonblank string (up to 256 characters) or exact integer")
        typed_identity = (type(identity), identity)
        if typed_identity in indexed:
            raise ValueError(f"{label} contains duplicate row IDs")
        indexed[typed_identity] = row
    return indexed, local_fields


def _equal(before: Any, after: Any) -> bool:
    if type(before) in (int, float) and type(after) in (int, float):
        return before == after
    return type(before) is type(after) and before == after


def run(before: Any, after: Any, key: str = "id") -> ToolResult:
    """Return an exact bounded snapshot diff; never mutate or execute inputs."""
    try:
        if type(key) is not str or not key.strip() or len(key) > MAX_FIELD_CHARS:
            raise ValueError("key must name a nonblank field with at most 80 characters")
        fields: set[str] = set()
        budget = [0, 0]  # Incoming JSON text bytes and normalised scalar JSON bytes.
        previous, previous_fields = _index(before, "before", key, fields, budget)
        current, current_fields = _index(after, "after", key, fields, budget)
        counts = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
        examples: dict[str, list] = {"added": [], "removed": [], "changed": []}
        for identity, row in current.items():
            if identity not in previous:
                counts["added"] += 1
                if len(examples["added"]) < MAX_EXAMPLES:
                    examples["added"].append(row[key])
                continue
            older = previous[identity]
            added = sorted(set(row) - set(older))
            removed = sorted(set(older) - set(row))
            changed = sorted(name for name in set(older) & set(row) if not _equal(older[name], row[name]))
            if added or removed or changed:
                counts["changed"] += 1
                if len(examples["changed"]) < MAX_EXAMPLES:
                    examples["changed"].append({"id": row[key], "added_fields": added,
                                                "removed_fields": removed, "changed_fields": changed})
            else:
                counts["unchanged"] += 1
        for identity, row in previous.items():
            if identity not in current:
                counts["removed"] += 1
                if len(examples["removed"]) < MAX_EXAMPLES:
                    examples["removed"].append(row[key])
        result = {
            "plugin": "orion_dataset_compare", "source": "supplied_snapshots", "key": key,
            "rows_before": len(previous), "rows_after": len(current), "counts": counts,
            "fields": {"before": sorted(previous_fields), "after": sorted(current_fields),
                       "added": sorted(current_fields - previous_fields),
                       "removed": sorted(previous_fields - current_fields)},
            "examples": examples, "example_limit_per_category": MAX_EXAMPLES,
            "examples_truncated": {name: counts[name] > len(rows) for name, rows in examples.items()},
            "comparison": {"row_order": "ignored", "numbers": "compare by value; 1 equals 1.0",
                           "booleans": "distinct from numbers", "missing_fields": "distinct from null",
                           "ids": "exact string or integer; string and integer IDs are distinct"},
            "limitations": "Only the supplied snapshots were compared. Counts are exact; examples are bounded in input order. IDs and field names are included; other cell values are omitted. No files were read, actions executed or services called.",
        }
        return ToolResult(json.dumps(result, ensure_ascii=False, indent=2), evidence=[result])
    except (ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        # JSON decoder errors may quote raw input; return the location only.
        if isinstance(exc, json.JSONDecodeError):
            reason = f"invalid JSON at line {exc.lineno}, column {exc.colno}"
        elif isinstance(exc, UnicodeError):
            reason = "all text must be valid UTF-8"
        else:
            reason = str(exc)
        return ToolResult(f"Dataset comparison failed: {reason}.", ok=False)
