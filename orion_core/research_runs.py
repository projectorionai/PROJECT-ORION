"""Bounded durable research receipts and observed threaded publication.

Records describe execution evidence, not model promises. Interrupted work is
never replayed automatically. A corrupt/unsupported ledger is preserved and
reported rather than replaced with an empty success-shaped store.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from itertools import islice
import secrets
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import uuid
import weakref

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR
from .session_log import redact
from .research_ownership import acquire_lease

DEFAULT_PATH = CONFIG_DIR / "research_runs.json"
MAX_RECORDS = 128
MAX_BYTES = 1_048_576
ACTIVE = frozenset({"accepted", "running", "cancel_requested"})
STATES = ACTIVE | {"completed", "failed", "cancelled", "interrupted"}
PUBLICATION_SETTLE_S = 1.5
_DEFAULT: ResearchRunStore | None = None
_DEFAULT_LOCK = threading.Lock()


def verified_paths(paths: Any) -> list[str]:
    found = []
    for raw in paths if isinstance(paths, (list, tuple)) else ():
        try:
            path = Path(raw).resolve()
            value = str(path)
            if len(value) <= 2048 and path.exists() and value not in found:
                found.append(value)
        except (OSError, TypeError, ValueError, RuntimeError):
            continue
        if len(found) >= 12:
            break
    return found


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    value.pop("_publication_key", None)
    return value


class ResearchRunStore:
    def __init__(self, path: Path | str, *, limit: int = MAX_RECORDS) -> None:
        self.path = Path(path).resolve()
        self.limit = max(1, min(MAX_RECORDS, int(limit)))
        self._lock = threading.RLock()
        self._rows: list[dict[str, Any]] = []
        self._disk_signature = None
        self.storage_error = ""
        self.ownership_error = ""
        self._lease = None
        self._lease_finalizer = None
        self._closed = False
        self._publication_journals: dict[str, Any] = {}
        self._claim_writer()
        self._load(recover=self._lease is not None)

    def _claim_writer(self) -> None:
        try:
            self._lease = acquire_lease(self.path)
            self.ownership_error = ("Research ledger is owned by another research writer"
                                    if self._lease is None else "")
            if self._lease is not None:
                self._lease_finalizer = weakref.finalize(self, self._lease.close)
        except OSError as exc:
            self.ownership_error = "Research ledger writer lock is unavailable: " + redact(str(exc))[:200]

    def _load(self, *, recover: bool) -> None:
        self.storage_error = ""
        try:
            raw, self._disk_signature = self._read_disk()
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("runs"), list):
                raise ValueError("Unsupported research ledger format")
            seen = set()
            for row in data["runs"]:
                if (not isinstance(row, dict) or not isinstance(row.get("run_id"), str)
                        or not row["run_id"] or row["run_id"] in seen
                        or row.get("research_state") not in STATES
                        or not isinstance(row.get("topic"), str)):
                    raise ValueError("Invalid research run record")
                seen.add(row["run_id"])
            self._rows = data["runs"][-self.limit:]
            changed = False
            recovery_budget = [128 * 1024 * 1024]
            for row in reversed(self._rows):
                if recover and row["research_state"] in ACTIVE:
                    row.update(research_state="interrupted", updated_at=time.time(),
                               error_category="process_interrupted",
                               result="Research was interrupted by a process restart; it has not been resumed.")
                    changed = True
                    self._recover_publications(row, budget=recovery_budget)
            if changed:
                self._save(self._rows)
        except FileNotFoundError:
            self._rows = []
            self._disk_signature = None
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            self.storage_error = redact(str(exc))[:240]
            self._rows = []

    def _read_disk(self) -> tuple[bytes, tuple[int, int, str]]:
        from .research_publication import _ordinary
        before = _ordinary(self.path)
        if before.st_size > MAX_BYTES:
            raise ValueError("Research ledger exceeds its size limit")
        version = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        with self.path.open("rb") as handle:
            if version(os.fstat(handle.fileno())) != version(before):
                raise OSError("Research ledger changed while opening")
            raw = handle.read(MAX_BYTES + 1)
            if version(os.fstat(handle.fileno())) != version(before):
                raise OSError("Research ledger changed while reading")
        if len(raw) > MAX_BYTES:
            raise ValueError("Research ledger exceeds its size limit")
        if len(raw) != before.st_size or version(_ordinary(self.path)) != version(before):
            raise OSError("Research ledger changed while reading")
        return raw, (before.st_dev, before.st_ino, hashlib.sha256(raw).hexdigest())

    def _check_disk(self) -> None:
        try:
            try:
                _raw, current = self._read_disk()
            except FileNotFoundError:
                current = None
            if current != self._disk_signature:
                raise OSError("Research ledger changed or was removed outside its writer")
        except (OSError, ValueError) as exc:
            self.storage_error = redact(str(exc))[:240]
            raise OSError(f"Research ledger is unavailable: {self.storage_error}") from exc

    def _recover_publications(self, row: dict[str, Any], *, budget: list[int] | None = None) -> None:
        from .research_publication import manifest_name, reconcile
        name = row.get("publication_manifest")
        if name is None:  # Compatible receipts predating publication journals.
            return
        try:
            if name != manifest_name(row["run_id"]):
                raise ValueError("Research publication manifest identity mismatch")
            key = row.get("_publication_key")
            if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("Research publication integrity evidence is unavailable")
            recovery = reconcile(self.path.parent / "research_publications" / name, row["run_id"],
                                 key=key, budget=budget)
            recovered = recovery.pop("paths")
            row["file_paths"] = verified_paths([*recovered, *(row.get("file_paths") or [])])
            row["selected_path"] = row["file_paths"][0] if row["file_paths"] else None
            row["publication_recovery"] = recovery
            if recovered:
                row["result"] += (" Verified saved/partial artifacts were retained; "
                                  "this does not establish completed research.")
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            row["publication_recovery"] = {"uncertain": True, "error": redact(str(exc))[:240]}

    def publication_journal(self, run_id: str):
        from .research_publication import PublicationJournal, manifest_name
        with self._lock:
            self._ensure_writer()
            row = next((item for item in self._rows if item["run_id"] == run_id), None)
            if row is None or row.get("publication_manifest") != manifest_name(run_id):
                raise ValueError("Research publication is not bound to an admitted run")
            journal = self._publication_journals.get(run_id)
            if journal is None:
                journal = PublicationJournal(self.path.parent / "research_publications" / manifest_name(run_id),
                                             run_id, row["_publication_key"], self._ensure_writer, self._lock)
                self._publication_journals[run_id] = journal
            return journal

    def _ensure_writer(self) -> None:
        if self._closed:
            raise OSError("Research ledger writer is closed")
        if self._lease is None:
            self._claim_writer()
            if self._lease is None:
                raise OSError(self.ownership_error)
            # Promotion requires fresh disk state; never merge stale reader rows.
            self._load(recover=True)
        if self.storage_error:
            raise OSError(f"Research ledger is unavailable: {self.storage_error}")

    def close(self) -> None:
        """Release only after the owner's work has settled; future writes fail.

        Mutation threads hold this same lock. The default store is kept alive
        through its tasks and is not closed early by cancellation or a reader.
        """
        with self._lock:
            self._closed = True
            if self._lease_finalizer is not None:
                self._lease_finalizer()
            self._lease = None

    def _save(self, rows: list[dict[str, Any]]) -> None:
        if self._lease is None or self._closed:
            raise OSError("Research ledger has no active writer lease")
        if self.storage_error:
            raise OSError(f"Research ledger is unavailable: {self.storage_error}")
        content = json.dumps({"version": 1, "runs": rows}, ensure_ascii=False)
        if len(content.encode("utf-8")) > MAX_BYTES:
            raise OSError("Research ledger exceeds its size limit")
        self._check_disk()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, content, encoding="utf-8")
        try:
            raw, signature = self._read_disk()
            if raw != content.encode("utf-8"):
                raise OSError("Research ledger changed while saving")
            self._disk_signature = signature
        except (OSError, ValueError) as exc:
            self.storage_error = redact(str(exc))[:240]
            raise OSError(f"Research ledger is unavailable: {self.storage_error}") from exc

    def start(self, topic: str, action: str, *, run_id: str | None = None,
              request_id: Any = None, owner: str = "local_session",
              inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        safe_inputs = {key: value for key, value in (inputs or {}).items()
                       if key in {"minutes", "questions", "depth", "style", "wait", "per_question", "rounds"}
                       and isinstance(value, (str, int, float, bool)) and len(str(value)) <= 120}
        now = time.time()
        from .research_publication import manifest_name
        accepted_run_id = run_id or uuid.uuid4().hex
        row = {"run_id": accepted_run_id, "publication_manifest": manifest_name(accepted_run_id),
               "_publication_key": secrets.token_hex(32),
               "topic": redact(str(topic))[:600],
               "action": str(action)[:40], "accepted_intent": redact(str(topic))[:600],
               "owner": str(owner)[:100], "request_id": request_id if type(request_id) is int else None,
               "inputs": safe_inputs, "created_at": now, "updated_at": now,
               "research_state": "running", "error_category": "", "file_paths": [],
               "selected_path": None, "result": "Research is running."}
        with self._lock:
            self._ensure_writer()
            if any(item["run_id"] == row["run_id"] for item in self._rows):
                raise ValueError("Research run ID already exists")
            rows = [*self._rows, row]
            while len(rows) > self.limit:
                index = next((i for i, item in enumerate(rows) if item["research_state"] not in ACTIVE), None)
                if index is None:
                    raise OSError("Research ledger is full of active investigations")
                del rows[index]
            self._save(rows)
            self._rows = rows
            retained_ids = {item["run_id"] for item in rows}
            self._publication_journals = {key: value for key, value in self._publication_journals.items()
                                          if key in retained_ids}
        return _public_row(row)

    def update(self, run_id: str, receipt: dict[str, Any], result: str = "") -> dict[str, Any]:
        state = receipt.get("research_state")
        if state not in STATES:
            raise ValueError("Invalid research run state")
        with self._lock:
            self._ensure_writer()
            rows = copy.deepcopy(self._rows)
            row = next((item for item in rows if item["run_id"] == run_id), None)
            if row is None:
                raise ValueError("Unknown research run ID")
            paths = verified_paths(receipt.get("file_paths"))
            row.update(research_state=state, file_paths=paths,
                       selected_path=paths[0] if paths else None,
                       updated_at=time.time(), result=redact(str(result))[:1800],
                       error_category=str(receipt.get("error_category") or
                           ("" if state == "completed" else state))[:80])
            for key in ("source_count", "section_count", "failed_section_count"):
                if type(receipt.get(key)) is int:
                    row[key] = max(0, receipt[key])
            self._save(rows)
            self._rows = rows
            return _public_row(row)

    def list(self, *, run_id: str = "", topic: str = "", limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            probe = None
            try:
                if self._lease is None:
                    # A reader never reconciles disk. A briefly held free lease
                    # can still prove that stored 'running' work lost its owner.
                    try:
                        probe = acquire_lease(self.path)
                    except OSError as exc:
                        self.ownership_error = "Research owner could not be checked: " + redact(str(exc))[:200]
                    self._load(recover=False)
                if self.storage_error:
                    raise OSError(f"Research ledger is unavailable: {self.storage_error}")
                selected = (row for row in reversed(self._rows)
                            if (not run_id or row["run_id"] == run_id)
                            and (not topic or row["topic"].casefold() == topic.casefold()))
                rows = [_public_row(row) for row in islice(selected, max(0, min(128, limit)))]
                if probe is not None:
                    recovery_budget = [128 * 1024 * 1024]
                    for row in rows:
                        if row["research_state"] in ACTIVE:
                            row.update(research_state="interrupted", error_category="writer_not_present",
                                       result="Research has no active writer; recorded work is interrupted and has not been resumed.")
                            # Integrity keys remain in private persistence; they
                            # are supplied transiently and never enter receipts.
                            private = next(item for item in self._rows if item["run_id"] == row["run_id"])
                            row["_publication_key"] = private.get("_publication_key")
                            self._recover_publications(row, budget=recovery_budget)
                            row.pop("_publication_key", None)
                return rows
            finally:
                if probe is not None:
                    probe.close()


def default_store() -> ResearchRunStore:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = ResearchRunStore(DEFAULT_PATH)
        return _DEFAULT


async def publish(writer: Callable[..., Any], *args: Any,
                  artifact_paths: Callable[[Any], list[Any]], **kwargs: Any) -> Any:
    """Cancellation requests cannot stop a file writer already in a thread.

    Retain its task for the run observer. Wait briefly for settlement, then
    propagate cancellation; the run stays cancel_requested until publication
    finishes and its actual artifact can be recorded.
    """
    from . import background
    current = asyncio.current_task()
    binding = getattr(current, "_orion_research_publication_run", None)

    def write():
        from .research_publication import scope
        journal = binding[0].publication_journal(binding[1]) if binding else None
        with scope(journal):
            try:
                return writer(*args, **kwargs)
            finally:
                if journal is not None:
                    task._orion_saved_artifacts = list(journal.observed_paths)

    task = background.spawn(asyncio.to_thread(write), name="orion-research-publication")
    if task is None:
        raise RuntimeError("Research publication could not start")
    if current is not None:
        publications = getattr(current, "_orion_publications", None)
        if publications is None:
            publications = current._orion_publications = []
        publications.append((task, artifact_paths))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.wait({task}, timeout=PUBLICATION_SETTLE_S)
        raise


def publication_paths(task: asyncio.Task) -> tuple[list[str], list[asyncio.Task]]:
    paths, pending = [], []
    for work, extract in getattr(task, "_orion_publications", ()):
        paths.extend(getattr(work, "_orion_saved_artifacts", ()))
        if not work.done():
            pending.append(work)
        elif not work.cancelled():
            try:
                paths.extend(extract(work.result()))
            except Exception:
                continue
    return verified_paths(paths), pending


def bind_publication_run(task: asyncio.Task, store: ResearchRunStore, run_id: str) -> None:
    """Only durable admission supplies this identity; never infer a latest run."""
    task._orion_research_publication_run = (store, run_id)


# ── research in progress, for the BRAIN and NEURAL pages ─────────────────────
# "Co-ordinate with the brain and the neural network of ORION, only if he's
# ACTUALLY researching" (owner, 30 Oct 2026). Every engine (timed run, dossier,
# reading investigation, paper) reports its real start, its current question
# and exactly one finish through ActivitySignal, as dashboard_event
# ("research", {"run", "topic", "status", ...}). Nothing else writes here, so
# a page can never show research that is not running, and a finish - done,
# draft, failed or cancelled - always clears it.

#: Runs in progress now, by run id; a page built late seeds itself from this.
_LIVE_RUNS: dict[str, dict[str, Any]] = {}
_LIVE_LOCK = threading.Lock()
#: A run cannot outlive the longest timed window (180 minutes); an entry older
#: than this can only be one whose finish was lost, and is not shown.
LIVE_RUN_MAX_S = 4 * 3600.0


def live_activity() -> list[dict[str, Any]]:
    """Research running right now, oldest first (copies)."""
    now = time.monotonic()
    with _LIVE_LOCK:
        for run in [run for run, row in _LIVE_RUNS.items()
                    if now - row.get("_started", now) > LIVE_RUN_MAX_S]:
            _LIVE_RUNS.pop(run, None)
        rows = sorted(_LIVE_RUNS.values(), key=lambda row: row.get("_started", 0.0))
        return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


class ActivitySignal:
    """One research run's start, progress and single finish on the bus."""

    def __init__(self, bus: Any, topic: str, engine: str) -> None:
        self.bus = bus
        self.topic = str(topic or "")[:200]
        self.engine = engine
        self.run = uuid.uuid4().hex[:12]
        self.finished_status = ""

    def _emit(self, status: str, **extra: Any) -> None:
        payload = {"run": self.run, "topic": self.topic, "engine": self.engine,
                   "status": status, "at": time.time(), **extra}
        with _LIVE_LOCK:
            if status == "running":
                row = _LIVE_RUNS.setdefault(self.run, {"_started": time.monotonic()})
                row.update({k: v for k, v in payload.items() if v not in (None, "")})
            else:
                _LIVE_RUNS.pop(self.run, None)
        try:
            self.bus.dashboard_event.emit("research", payload)
        except Exception:
            pass  # no bus (tests, a headless worker): the registry still holds the truth

    def start(self, **extra: Any) -> None:
        if not self.finished_status:
            self._emit("running", **extra)

    def progress(self, question: str = "", **extra: Any) -> None:
        if not self.finished_status:
            self._emit("running", question=str(question or "")[:200], **extra)

    def finish(self, status: str, **extra: Any) -> None:
        """Idempotent: the first finish wins, later ones (a finally) do nothing."""
        if self.finished_status:
            return
        self.finished_status = str(status or "failed")
        self._emit(self.finished_status, **extra)
