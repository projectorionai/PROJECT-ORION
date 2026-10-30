"""
The job store — one durable owner for every job record.

Before this, four things each kept their own idea of "a job": JobManager held
background tool calls in a dict (ids from a counter, gone on restart),
WorkflowEngine held runs in another dict (ids from ``int(time.time())``, so two
starts of one workflow in the same second collided), PlanExecutor kept nothing
at all, and the background/task helpers only held asyncio handles. A restart
lost every record, and nothing could say whether a step that was running when
ORION stopped had already happened.

This store is the single record for all of them — ``kind`` says which —
in SQLite (WAL, via ``db.connect``) in the config directory:

* **Identity** is a uuid4 (``job-`` + 12 hex digits of one), never a clock.
* **State** is one of QUEUED, RUNNING, WAITING_FOR_APPROVAL, PAUSED,
  COMPLETED, FAILED, CANCELLED, INTERRUPTED, and only legal transitions are
  applied. The three terminal states are final.
* **Progress** is real (``fraction`` 0..1) or explicitly indeterminate
  (``fraction`` None) — never a made-up percentage. Every change bumps a
  store-wide ``seq``, so a view that reconnects asks for "everything after
  seq N" and drops stale events.
* **Steps** record what each step is and its SIDE-EFFECT class from the tool
  concurrency table: ``read_only`` steps are idempotent and may be resumed;
  ``side_effect`` and ``exclusive`` steps are never replayed blindly.
* **Restart**: opening the store marks every job that was not finished as
  INTERRUPTED and every step that was running as interrupted. Nothing is
  resumed automatically.
* **Approvals** are bound to one job, one step and a digest of the exact tool
  and arguments, expire, and are single-use. ``consume_approval`` re-checks
  all of that immediately before the step runs, so an approval can never be
  carried to a different call, a later retry, or a changed argument.
* **Durability** is chosen per write, not per database. ``db.connect`` gives
  every store WAL with ``synchronous=NORMAL``: safe against ORION crashing, but
  a power cut or OS crash can roll back the last commits. That is fine for a
  progress fraction or a step note, which the next update rewrites. It is not
  fine for a state change, a step starting or finishing, or an approval being
  given or used: losing one of those can make a side-effecting step look as
  if it never started (and so eligible to run again), or bring a spent
  approval back. Those commits run with ``synchronous=FULL`` (the WAL is
  flushed to disk before the commit returns). MEASURED on this machine
  (SQLite 3.50.4, local NTFS SSD, 200 jobs): a durable commit costs about
  2.2 ms at the median against 0.1 ms relaxed, so progress updates stay
  relaxed. A stronger fsync does not make an external action exactly-once;
  a step already marked running is reported as interrupted, never replayed.

Tests and tools that never install the durable store get an in-memory one —
the same code, nothing written to disk.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable


class State(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL = frozenset({State.COMPLETED, State.FAILED, State.CANCELLED})
#: States a job can be in while something in this process owns it.
LIVE = frozenset({State.QUEUED, State.RUNNING, State.WAITING_FOR_APPROVAL, State.PAUSED})

_TRANSITIONS: dict[State, frozenset[State]] = {
    State.QUEUED: frozenset({State.RUNNING, State.PAUSED, State.CANCELLED, State.FAILED,
                             State.INTERRUPTED}),
    State.RUNNING: frozenset({State.COMPLETED, State.FAILED, State.CANCELLED, State.PAUSED,
                              State.WAITING_FOR_APPROVAL, State.INTERRUPTED, State.QUEUED}),
    State.PAUSED: frozenset({State.QUEUED, State.RUNNING, State.CANCELLED, State.INTERRUPTED,
                             State.FAILED}),
    State.WAITING_FOR_APPROVAL: frozenset({State.RUNNING, State.QUEUED, State.CANCELLED,
                                           State.FAILED, State.INTERRUPTED}),
    State.INTERRUPTED: frozenset({State.QUEUED, State.RUNNING, State.WAITING_FOR_APPROVAL,
                                  State.CANCELLED, State.FAILED}),
}

#: Step side-effect classes (from concurrency.classify).
READ_ONLY = "read_only"
SIDE_EFFECT = "side_effect"
EXCLUSIVE = "exclusive"

#: Finished jobs kept; older ones are pruned so the store stays small.
RETAINED_FINISHED = 500
#: Arguments are kept for resumption and reconciliation, within reason.
MAX_ARGS_CHARS = 20000
APPROVAL_TTL_S = 600.0


def new_job_id() -> str:
    """A job id: uuid4-derived, short enough to say, never time-based."""
    return f"job-{uuid.uuid4().hex[:12]}"


def side_effect_class(tool: str, args: dict[str, Any] | None = None) -> str:
    """How a step may be retried or resumed, from the tool concurrency table.

    PARALLEL tools are read-only, so re-running one is harmless (idempotent).
    SERIAL is the default for anything unclassified — including forged tools —
    and changes something; EXCLUSIVE changes ORION itself.
    """
    from .concurrency import ToolClass, classify
    kind = classify(tool, args)
    if kind is ToolClass.PARALLEL:
        return READ_ONLY
    return EXCLUSIVE if kind is ToolClass.EXCLUSIVE else SIDE_EFFECT


def args_digest(tool: str, args: dict[str, Any] | None) -> str:
    """The binding an approval is tied to: this tool with exactly these args."""
    canonical = json.dumps({"tool": str(tool), "args": args or {}}, sort_keys=True,
                           default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dump(value: Any) -> tuple[str, bool]:
    """JSON for a column, and whether it was kept whole."""
    try:
        text = json.dumps(value if value is not None else {}, default=str)
    except Exception:
        return "{}", False
    if len(text) > MAX_ARGS_CHARS:
        return json.dumps({"_truncated": True}), False
    return text, True


def _load(text: Any) -> Any:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    cls TEXT NOT NULL DEFAULT 'user',
    origin TEXT NOT NULL DEFAULT '',
    args_json TEXT NOT NULL DEFAULT '{}',
    args_complete INTEGER NOT NULL DEFAULT 1,
    fraction REAL,
    note TEXT NOT NULL DEFAULT '',
    step INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    result TEXT NOT NULL DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 0,
    checkpoint_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    started_at REAL,
    updated_at REAL NOT NULL,
    finished_at REAL,
    seq INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_seq ON jobs(seq);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE TABLE IF NOT EXISTS job_steps (
    job_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    step_id TEXT NOT NULL DEFAULT '',
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '{}',
    side_effect TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '',
    started_at REAL,
    finished_at REAL,
    PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    tool TEXT NOT NULL,
    digest TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    requested_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    decided_at REAL,
    decided_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS approvals_job ON approvals(job_id, step_idx);
"""

_JOB_FIELDS = ("label", "cls", "fraction", "note", "step", "total", "result", "ok",
               "checkpoint", "origin")


class JobStore:
    """Durable job records. Thread-safe; every method is a short SQLite call."""

    def __init__(self, path: Path | str | None = None, *, recover: bool = True,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path is not None else None
        self.clock = clock
        self._lock = threading.RLock()
        if self.path is None:
            self._db = sqlite3.connect(":memory:", check_same_thread=False)
            self._db.row_factory = sqlite3.Row
        else:
            from .db import connect
            self._db = connect(self.path)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        row = self._db.execute("SELECT COALESCE(MAX(seq), 0) FROM jobs").fetchone()
        self._seq = int(row[0] or 0)
        self._listeners: list[Callable[[dict[str, Any]], Any]] = []
        #: Commits made with synchronous=FULL (state, step and approval writes).
        self.durable_commits = 0
        self.recovered: list[dict[str, Any]] = self.recover() if recover else []

    # ── durability ──────────────────────────────────────────────────────────

    @contextmanager
    def _durable(self):
        """Run one write with ``synchronous=FULL``, then drop back to NORMAL.

        Called with the lock held and no transaction open (every method
        commits before returning), which is when SQLite applies the setting
        to the next commit. An in-memory store has nothing to flush."""
        if self.path is None:
            yield
            return
        try:
            self._db.execute("PRAGMA synchronous=FULL")
        except sqlite3.Error:
            pass
        try:
            yield
            self.durable_commits += 1
        finally:
            try:
                self._db.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error:
                pass

    # ── records ─────────────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def create(self, *, kind: str, name: str, label: str = "", cls: str = "user",
               state: State | str = State.QUEUED, args: Any = None, origin: str = "",
               steps: Iterable[dict[str, Any]] = (), total: int | None = None,
               job_id: str | None = None) -> dict[str, Any]:
        """A new job and its steps; returns the record."""
        state = State(state)
        steps = list(steps)
        args_text, complete = _dump(args)
        now = self.clock()
        # The parent, every step and retention pruning form one transaction.
        # A bad step or failed write must not leave a partial job to be
        # committed by the next unrelated operation.
        with self._lock, self._durable(), self._db:
            for _attempt in range(5):
                candidate = job_id or new_job_id()
                try:
                    self._db.execute(
                        "INSERT INTO jobs (id, kind, name, label, state, cls, origin, args_json,"
                        " args_complete, total, created_at, updated_at, started_at, seq)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (candidate, str(kind), str(name)[:200], str(label)[:200], state.value,
                         str(cls), str(origin)[:40], args_text, int(complete),
                         int(total if total is not None else len(steps)), now, now,
                         now if state is State.RUNNING else None, self._next_seq()))
                    break
                except sqlite3.IntegrityError:
                    if job_id:
                        raise
                    continue
            else:                                   # pragma: no cover - 48 random bits x5
                raise RuntimeError("could not allocate a unique job id")
            for index, step in enumerate(steps):
                step_args, _ = _dump(step.get("args") or {})
                tool = str(step.get("tool") or "")
                self._db.execute(
                    "INSERT INTO job_steps (job_id, idx, step_id, tool, args_json, side_effect)"
                    " VALUES (?,?,?,?,?,?)",
                    (candidate, int(step.get("index", index)), str(step.get("step_id") or ""),
                     tool, step_args,
                     str(step.get("side_effect") or side_effect_class(tool, step.get("args")))))
            self._prune_locked()
            record = self._get_locked(candidate)
        self._emit(record)
        return record

    def transition(self, job_id: str, state: State | str, **fields: Any) -> dict[str, Any] | None:
        """Move a job to *state* (if legal) and update *fields*. Returns the
        new record, or None when the transition is not allowed."""
        target = State(state)
        with self._lock:
            row = self._db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            current = State(row["state"])
            if current is not target and target not in _TRANSITIONS.get(current, frozenset()):
                return None
            now = self.clock()
            sets = {"state": target.value, "updated_at": now}
            if target is State.RUNNING:
                sets["started_at_if_null"] = now
            if target in TERMINAL:
                sets["finished_at"] = now
            with self._durable():
                record = self._apply_locked(job_id, sets, fields)
        self._emit(record)
        return record

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        """Update progress/note/result fields without changing state."""
        with self._lock:
            row = self._db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            record = self._apply_locked(job_id, {"updated_at": self.clock()}, fields)
        self._emit(record)
        return record

    def _apply_locked(self, job_id: str, sets: dict[str, Any],
                      fields: dict[str, Any]) -> dict[str, Any]:
        columns: list[str] = []
        values: list[Any] = []
        started_if_null = sets.pop("started_at_if_null", None)
        for key, value in fields.items():
            if key not in _JOB_FIELDS:
                raise KeyError(f"not a job field: {key}")
            if key == "checkpoint":
                text, _ = _dump(value)
                columns.append("checkpoint_json=?")
                values.append(text)
            elif key == "fraction":
                columns.append("fraction=?")
                values.append(None if value is None else max(0.0, min(1.0, float(value))))
            elif key == "ok":
                columns.append("ok=?")
                values.append(int(bool(value)))
            elif key in {"step", "total"}:
                columns.append(f"{key}=?")
                values.append(int(value))
            else:
                columns.append(f"{key}=?")
                values.append(str(value)[:4000] if key == "result" else str(value)[:200])
        for key, value in sets.items():
            columns.append(f"{key}=?")
            values.append(value)
        if started_if_null is not None:
            columns.append("started_at=COALESCE(started_at, ?)")
            values.append(started_if_null)
        columns.append("seq=?")
        values.append(self._next_seq())
        self._db.execute(f"UPDATE jobs SET {', '.join(columns)} WHERE id=?", (*values, job_id))
        self._db.commit()
        return self._get_locked(job_id)

    def set_step(self, job_id: str, idx: int, *, state: str | None = None,
                 attempts: int | None = None, detail: str | None = None) -> None:
        now = self.clock()
        columns: list[str] = []
        values: list[Any] = []
        if state is not None:
            columns.append("state=?")
            values.append(str(state))
            if state == "running":
                columns.append("started_at=COALESCE(started_at, ?)")
                values.append(now)
            elif state in {"completed", "failed", "skipped", "cancelled"}:
                columns.append("finished_at=?")
                values.append(now)
        if attempts is not None:
            columns.append("attempts=?")
            values.append(int(attempts))
        if detail is not None:
            columns.append("detail=?")
            values.append(str(detail)[:300])
        if not columns:
            return
        # A step starting or finishing is the record that stops a side
        # effect being replayed after a power cut; attempts/detail are not.
        with self._lock, (self._durable() if state is not None else nullcontext()):
            self._db.execute(f"UPDATE job_steps SET {', '.join(columns)} "
                             "WHERE job_id=? AND idx=?", (*values, job_id, int(idx)))
            self._db.commit()

    # ── reading ─────────────────────────────────────────────────────────────

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        record = {key: row[key] for key in row.keys()}
        record["args"] = _load(record.pop("args_json"))
        record["checkpoint"] = _load(record.pop("checkpoint_json"))
        record["args_complete"] = bool(record["args_complete"])
        record["ok"] = bool(record["ok"])
        record["indeterminate"] = record["fraction"] is None
        return record

    def _get_locked(self, job_id: str) -> dict[str, Any]:
        return self._row(self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._get_locked(job_id)

    def steps(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM job_steps WHERE job_id=? ORDER BY idx",
                                    (job_id,)).fetchall()
        out = []
        for row in rows:
            step = {key: row[key] for key in row.keys()}
            step["args"] = _load(step.pop("args_json"))
            out.append(step)
        return out

    def list(self, *, states: Iterable[str] | None = None, kinds: Iterable[str] | None = None,
             since_seq: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        clauses, values = ["seq > ?"], [int(since_seq)]
        if states:
            wanted = [State(s).value for s in states]
            clauses.append(f"state IN ({','.join('?' * len(wanted))})")
            values.extend(wanted)
        if kinds:
            wanted_kinds = [str(k) for k in kinds]
            clauses.append(f"kind IN ({','.join('?' * len(wanted_kinds))})")
            values.extend(wanted_kinds)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY seq DESC LIMIT ?",
                (*values, max(1, int(limit)))).fetchall()
        return [self._row(row) for row in rows]

    def snapshot(self, *, limit: int = 50, since_seq: int = 0) -> dict[str, Any]:
        """For a task view: the store's seq, the newest jobs, and counts."""
        jobs = self.list(limit=limit, since_seq=since_seq)
        with self._lock:
            counts = {row[0]: row[1] for row in self._db.execute(
                "SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall()}
            seq = self._seq
        return {"seq": seq, "jobs": jobs, "counts": counts}

    # ── restart ─────────────────────────────────────────────────────────────

    def recover(self) -> list[dict[str, Any]]:
        """Mark everything left unfinished by the last run as INTERRUPTED."""
        live = [s.value for s in LIVE]
        now = self.clock()
        with self._lock, self._durable():
            rows = self._db.execute(
                f"SELECT id, state FROM jobs WHERE state IN ({','.join('?' * len(live))})",
                live).fetchall()
            for row in rows:
                self._db.execute(
                    "UPDATE jobs SET state=?, note=?, updated_at=?, seq=? WHERE id=?",
                    (State.INTERRUPTED.value,
                     f"ORION stopped while this was {row['state'].replace('_', ' ')}",
                     now, self._next_seq(), row["id"]))
                self._db.execute(
                    "UPDATE job_steps SET state='interrupted' WHERE job_id=? AND state='running'",
                    (row["id"],))
            self._db.commit()
            records = [self._get_locked(row["id"]) for row in rows]
        return records

    def mark_interrupted(self, job_id: str, note: str) -> dict[str, Any] | None:
        """Record a live job (and its running step) as INTERRUPTED — what an
        orderly shutdown does, so the job can be resumed next time."""
        with self._lock, self._durable():
            row = self._db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or State(row["state"]) not in LIVE:
                return None
            self._db.execute(
                "UPDATE jobs SET state=?, note=?, updated_at=?, seq=? WHERE id=?",
                (State.INTERRUPTED.value, str(note)[:200], self.clock(), self._next_seq(),
                 job_id))
            self._db.execute(
                "UPDATE job_steps SET state='interrupted' WHERE job_id=? AND state='running'",
                (job_id,))
            self._db.commit()
            record = self._get_locked(job_id)
        self._emit(record)
        return record

    def interrupted_step(self, job_id: str) -> dict[str, Any] | None:
        """The step that was in flight when the job stopped, if any."""
        for step in self.steps(job_id):
            if step["state"] == "interrupted":
                return step
        return None

    # ── approvals ───────────────────────────────────────────────────────────

    def request_approval(self, job_id: str, step_idx: int, tool: str,
                         args: dict[str, Any] | None, *, reason: str = "",
                         ttl_s: float = APPROVAL_TTL_S) -> dict[str, Any]:
        """Bind an approval to this job, step, tool and exact args, and put the
        job in WAITING_FOR_APPROVAL."""
        now = self.clock()
        approval_id = uuid.uuid4().hex
        with self._lock, self._durable():
            # A new request replaces any older pending one for the same step.
            self._db.execute(
                "UPDATE approvals SET state='superseded' WHERE job_id=? AND step_idx=? "
                "AND state='pending'", (job_id, int(step_idx)))
            self._db.execute(
                "INSERT INTO approvals (id, job_id, step_idx, tool, digest, reason, state,"
                " requested_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (approval_id, job_id, int(step_idx), str(tool), args_digest(tool, args),
                 str(reason)[:300], "pending", now, now + max(1.0, float(ttl_s))))
            self._db.commit()
        self.transition(job_id, State.WAITING_FOR_APPROVAL,
                        note=f"waiting for approval: {reason}"[:200] if reason
                        else "waiting for approval")
        return self.approval(approval_id) or {}

    def approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM approvals WHERE id=?",
                                   (approval_id,)).fetchone()
        return None if row is None else {key: row[key] for key in row.keys()}

    def latest_approval(self, job_id: str, step_idx: int) -> dict[str, Any] | None:
        """The most recent approval request for one step, in any state."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM approvals WHERE job_id=? AND step_idx=? "
                "ORDER BY requested_at DESC LIMIT 1", (job_id, int(step_idx))).fetchone()
        return None if row is None else {key: row[key] for key in row.keys()}

    def pending_approvals(self, job_id: str | None = None) -> list[dict[str, Any]]:
        now = self.clock()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM approvals WHERE state='pending' AND expires_at > ?"
                + (" AND job_id=?" if job_id else "") + " ORDER BY requested_at",
                (now, job_id) if job_id else (now,)).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]

    def decide_approval(self, approval_id: str, approve: bool,
                        by: str = "user") -> dict[str, Any] | None:
        """Record a decision on a PENDING, unexpired approval. None otherwise."""
        now = self.clock()
        with self._lock, self._durable():
            row = self._db.execute("SELECT * FROM approvals WHERE id=?",
                                   (approval_id,)).fetchone()
            if row is None or row["state"] != "pending":
                return None
            if now >= row["expires_at"]:
                self._db.execute("UPDATE approvals SET state='expired' WHERE id=?",
                                 (approval_id,))
                self._db.commit()
                return None
            self._db.execute(
                "UPDATE approvals SET state=?, decided_at=?, decided_by=? WHERE id=?",
                ("approved" if approve else "denied", now, str(by)[:40], approval_id))
            self._db.commit()
        return self.approval(approval_id)

    def consume_approval(self, job_id: str, step_idx: int, tool: str,
                         args: dict[str, Any] | None) -> tuple[bool, str]:
        """Re-validate and use up the approval for this exact step, just
        before it runs. (True, "") only when an approved, unexpired, unused
        approval is bound to this job, step, tool AND these arguments."""
        now = self.clock()
        digest = args_digest(tool, args)
        with self._lock, self._durable():
            row = self._db.execute(
                "SELECT * FROM approvals WHERE job_id=? AND step_idx=? AND state='approved' "
                "ORDER BY decided_at DESC LIMIT 1", (job_id, int(step_idx))).fetchone()
            if row is None:
                return False, "no approval has been given for this step"
            if now >= row["expires_at"]:
                self._db.execute("UPDATE approvals SET state='expired' WHERE id=?", (row["id"],))
                self._db.commit()
                return False, "the approval expired before the step ran"
            if row["tool"] != str(tool) or row["digest"] != digest:
                self._db.execute("UPDATE approvals SET state='invalidated' WHERE id=?",
                                 (row["id"],))
                self._db.commit()
                return False, "the step changed after it was approved"
            self._db.execute("UPDATE approvals SET state='consumed' WHERE id=?", (row["id"],))
            self._db.commit()
        return True, ""

    def peek_approval(self, job_id: str, step_idx: int, tool: str,
                      args: dict[str, Any] | None) -> tuple[bool, str]:
        """The same test as consume_approval, without using anything up or
        writing: for a resume deciding whether it may start. The approval is
        still consumed (and re-checked) at the step's own dispatch."""
        now = self.clock()
        digest = args_digest(tool, args)
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM approvals WHERE job_id=? AND step_idx=? AND state='approved' "
                "ORDER BY decided_at DESC LIMIT 1", (job_id, int(step_idx))).fetchone()
        if row is None:
            return False, "no approval has been given for this step"
        if now >= row["expires_at"]:
            return False, "the approval expired before the step ran"
        if row["tool"] != str(tool) or row["digest"] != digest:
            return False, "the step changed after it was approved"
        return True, ""

    # ── events and housekeeping ─────────────────────────────────────────────

    def subscribe(self, listener: Callable[[dict[str, Any]], Any]) -> None:
        """Called with the new record after every change (any thread)."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[dict[str, Any]], Any]) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _emit(self, record: dict[str, Any] | None) -> None:
        if record is None:
            return
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:
                pass

    def _prune_locked(self) -> None:
        terminal = [s.value for s in TERMINAL]
        old = self._db.execute(
            f"SELECT id FROM jobs WHERE state IN ({','.join('?' * len(terminal))}) "
            "ORDER BY seq DESC LIMIT -1 OFFSET ?", (*terminal, RETAINED_FINISHED)).fetchall()
        if not old:
            return
        ids = [row[0] for row in old]
        marks = ",".join("?" * len(ids))
        self._db.execute(f"DELETE FROM jobs WHERE id IN ({marks})", ids)
        self._db.execute(f"DELETE FROM job_steps WHERE job_id IN ({marks})", ids)
        self._db.execute(f"DELETE FROM approvals WHERE job_id IN ({marks})", ids)

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass


# ── the process-wide store ────────────────────────────────────────────────────

_STORE: JobStore | None = None
_STORE_LOCK = threading.Lock()


def default_store() -> JobStore:
    """The installed store, or an in-memory one until app.py installs the
    durable file (so tests and tools never write into the config folder)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = JobStore(None)
        return _STORE


def install_store(store: JobStore) -> list[dict[str, Any]]:
    """Make *store* the process-wide job store (app.py, at startup). Returns
    the jobs the previous run left unfinished, now INTERRUPTED."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store
    return list(store.recovered)


def default_path() -> Path:
    from .constants import CONFIG_DIR
    return CONFIG_DIR / "jobs.db"


__all__ = ["APPROVAL_TTL_S", "EXCLUSIVE", "JobStore", "LIVE", "READ_ONLY", "SIDE_EFFECT",
           "State", "TERMINAL", "args_digest", "default_path", "default_store",
           "install_store", "new_job_id", "side_effect_class"]
