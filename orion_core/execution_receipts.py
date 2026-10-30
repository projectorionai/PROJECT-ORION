"""Bounded host observations of dispatch attempts, never external effect proof.

The ledger intentionally has no place for arguments, outputs, exception text,
credentials or operation receipts. A 'returned' attempt may have prepared a
draft, refused an action or started a job: consult its separate operation result.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
import re
from threading import RLock
import time
from typing import Any
import uuid


_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:\-]{1,128}\Z")
_TERMINAL = frozenset({"returned", "failed", "cancelled", "rejected"})
_INITIALIZATION_LOCK = RLock()


@dataclass
class _Observation:
    call_id: str
    tool: str
    started_at: float
    started_clock: float
    state: str = "running"
    executor_ok: bool | None = None
    finished_at: float | None = None
    finished_clock: float | None = None


class ExecutionObservationStore:
    """At most max_records attempts; even an evicted attempt can settle itself."""

    def __init__(self, max_records: int = 200) -> None:
        if type(max_records) is not int or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self._records: deque[_Observation] = deque(maxlen=max_records)
        self._lock = RLock()

    def begin(self, tool: str) -> _Observation:
        safe_tool = tool if isinstance(tool, str) and _IDENTIFIER.fullmatch(tool) else "[invalid tool identifier]"
        record = _Observation(uuid.uuid4().hex, safe_tool, time.time(), time.perf_counter())
        with self._lock:
            self._records.append(record)
        return record

    def finish(self, record: _Observation, state: str, *, executor_ok: bool | None = None) -> dict[str, Any]:
        if state not in _TERMINAL:
            raise ValueError("Invalid terminal invocation state")
        if executor_ok is not None and type(executor_ok) is not bool:
            raise ValueError("executor_ok must be a boolean or None")
        with self._lock:
            if record.state == "running":
                record.state = state
                record.executor_ok = executor_ok
                record.finished_at = time.time()
                record.finished_clock = time.perf_counter()
            return self._snapshot(record)

    @staticmethod
    def _snapshot(record: _Observation) -> dict[str, Any]:
        until = record.finished_clock if record.finished_clock is not None else time.perf_counter()
        return {"call_id": record.call_id, "tool": record.tool, "state": record.state,
                "executor_ok": record.executor_ok, "started_at": record.started_at,
                "finished_at": record.finished_at,
                "elapsed_ms": round(max(0.0, until - record.started_clock) * 1000.0, 3)}

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if type(limit) is not int:
            raise ValueError("limit must be an integer")
        if limit <= 0:
            return []
        with self._lock:
            return [self._snapshot(record) for record in islice(reversed(self._records), limit)]


def store_for(host: Any) -> ExecutionObservationStore:
    """Initialize once, including callers concurrently using a light fake host."""
    with _INITIALIZATION_LOCK:
        store = getattr(host, "_execution_observations", None)
        if not isinstance(store, ExecutionObservationStore):
            store = ExecutionObservationStore()
            try:
                setattr(host, "_execution_observations", store)
            except (AttributeError, TypeError):
                # Metadata must not make an otherwise supported slotted host
                # unable to dispatch. Its returned call still has observation.
                pass
        return store


__all__ = ["ExecutionObservationStore", "store_for"]
