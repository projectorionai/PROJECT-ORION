"""
Planner feedback with provenance — the only data the optional local planner
may ever be trained on.

Three things are easy to confuse and must stay apart:

* a PREFERENCE ("use UK spelling") takes effect at once through memory — no
  gradient step is needed and none is taken here;
* TASK STATE lives in the job store and is not training data;
* a TRAINING LABEL is a statement that one plan, for one recorded prompt, was
  right — or better than another. Only this store holds those.

So this is its own SQLite file (WAL, config dir), separate from memory, the
episodic log and the job store. It records nothing unless the user has
consented, and every row says where its label came from:

    user_explicit   typed by the user at the desktop (the CLI below)
    user_relayed    passed on by ORION from something the user said; usable
                    only after a review approves it
    executor        the plan executor's own receipts
    planner         the proposal itself (a prompt, never a label)

A tool returning ok is NOT proof that the goal was met — opening Notepad is
not editing the document. An outcome becomes a positive label only when its
postcondition was independently verified, or when the user accepted it.
Everything else is kept as evidence and never exported as a label.

Events are immutable: reviews and consent live in their own tables, and each
event carries a digest of its content. A task's label is the user's LATEST
verdict, so a later rejection withdraws an earlier acceptance or correction
(``SELECTION_POLICY``). Exporting needs consent in force. Withdrawing consent
can purge every event; exports already written are separate snapshots, which
the store lists (``exports()``) for the user to remove — it never deletes a
dataset or a model on their behalf.

Command line (the user's own, explicit channel):

    python -m orion_core.neural_feedback status
    python -m orion_core.neural_feedback consent on|off [--purge]
    python -m orion_core.neural_feedback correct PLAN_ID CORRECTED.json
    python -m orion_core.neural_feedback review EVENT_ID approve|reject
    python -m orion_core.neural_feedback export NEW_DIRECTORY [--explicit-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterable

from .constants import CONFIG_DIR

#: Where the feedback lives. A module constant so tests (and conftest) can
#: point it at a temporary directory.
FEEDBACK_PATH = CONFIG_DIR / "neural" / "feedback.db"
#: ``0`` withdraws consent for this process whatever the stored choice says.
#: Nothing can switch consent ON from the environment: it is a stored,
#: deliberate decision.
ENV_KILL = "ORION_NEURAL_FEEDBACK"
POLICY_VERSION = "planner-policy-1"
EXPORT_FORMAT = "orion-planner-feedback-v1"

SOURCES = frozenset({"user_explicit", "user_relayed", "executor", "planner"})
KINDS = frozenset({"plan", "correction", "outcome", "acceptance"})
MAX_PROMPT_CHARS = 48_000
MAX_PLAN_CHARS = 16_000
MAX_DETAIL_CHARS = 160

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    prompt_json TEXT,
    plan_json TEXT,
    receipts_json TEXT,
    tool_returned_ok INTEGER,
    postcondition_verified INTEGER,
    goal_accepted INTEGER,
    base_model TEXT NOT NULL DEFAULT '',
    adapter_version TEXT NOT NULL DEFAULT '',
    planner_version TEXT NOT NULL DEFAULT '',
    policy_version TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_task ON events(task_id, created_at);
CREATE TABLE IF NOT EXISTS reviews (
    event_id TEXT NOT NULL,
    reviewed_at REAL NOT NULL,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('approve', 'reject'))
);
CREATE TABLE IF NOT EXISTS consent (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL,
    changed_at REAL NOT NULL,
    changed_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exports (
    directory TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    tasks INTEGER NOT NULL,
    selection_policy TEXT NOT NULL
);
"""

#: How a task's label is chosen; written into every export manifest so a
#: dataset says which rules made it.
SELECTION_POLICY = (
    "latest-verdict-v2: a task's user verdicts (corrections, acceptances, rejections) are "
    "ordered by time and the latest decides — a correction gives SFT of the corrected plan "
    "and a DPO pair over the original; an acceptance gives SFT of the original; a rejection "
    "gives nothing and withdraws earlier positives. Positive verdicts must be explicit, or "
    "relayed and approved on review; rejections always count. With no user verdict, only "
    "a verified postcondition (not a tool returning ok) labels the original plan.")


#: Every event column but id and digest, with the value an absent field takes.
#: The digest covers all of them, so a NULL written later cannot pass as
#: the original.
_EVENT_DEFAULTS: dict[str, Any] = {
    "created_at": 0.0, "kind": "", "task_id": "", "session_id": "", "source": "",
    "prompt_json": None, "plan_json": None, "receipts_json": None,
    "tool_returned_ok": None, "postcondition_verified": None, "goal_accepted": None,
    "base_model": "", "adapter_version": "", "planner_version": "", "policy_version": "",
    "note": "",
}


def _event_digest(event: dict[str, Any]) -> str:
    return _sha256(_canonical({k: event.get(k, default)
                               for k, default in _EVENT_DEFAULTS.items()}))


# ── plan text: one canonical form for inference targets and training ─────────

_PLAN_KEYS = ("summary", "assumptions", "steps", "success_criteria")
_STEP_KEYS = ("id", "tool", "args", "depends_on", "success")


def plan_text(plan: dict[str, Any]) -> str:
    """The canonical JSON of a plan: the schema's key order, compact.

    The planner is asked for exactly this shape, and a training target must
    be the text the model would be asked to produce — not a re-serialisation
    whose key order or spacing differs from what inference expects.
    """
    ordered: dict[str, Any] = {}
    for key in _PLAN_KEYS:
        if key == "steps":
            ordered["steps"] = [
                {k: step.get(k) for k in _STEP_KEYS if k in step}
                for step in (plan.get("steps") or []) if isinstance(step, dict)]
        elif key in plan:
            ordered[key] = plan[key]
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)


def _sha256(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def _contains_secret(text: str) -> bool:
    """Credential-shaped content is never kept, whatever the consent says."""
    try:
        from .spillage_guard import SpillageGuard
        return bool(SpillageGuard().scan(str(text or "")).findings)
    except Exception:
        # If the guard cannot run, refusing is the safe direction.
        return True


# ── the store ────────────────────────────────────────────────────────────────

class FeedbackStore:
    """Consent-gated, provenance-carrying feedback. See the module docstring."""

    def __init__(self, path: Path | str | None = None, *,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path is not None else FEEDBACK_PATH
        self.clock = clock
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self.last_refusal = ""

    # ── connection ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            from .db import connect
            self._db = connect(self.path)
            self._db.executescript(_SCHEMA)
            self._db.commit()
        return self._db

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    # ── consent ─────────────────────────────────────────────────────────────

    def consent(self) -> dict[str, Any]:
        """The stored decision and whether it is in force for this process."""
        with self._lock:
            row = self._conn().execute(
                "SELECT enabled, changed_at, changed_by FROM consent WHERE id = 1").fetchone()
        stored = bool(row["enabled"]) if row else False
        killed = os.getenv(ENV_KILL, "").strip().lower() in {"0", "false", "no", "off"}
        return {"stored": stored, "in_force": stored and not killed,
                "withdrawn_by_environment": killed,
                "changed_at": row["changed_at"] if row else None,
                "changed_by": row["changed_by"] if row else ""}

    def consent_enabled(self) -> bool:
        return bool(self.consent()["in_force"])

    def set_consent(self, enabled: bool, *, changed_by: str, purge: bool = False) -> int:
        """Record the user's decision. Withdrawing with *purge* deletes every
        event and review, and returns how many events went."""
        who = str(changed_by or "").strip()[:60]
        if not who:
            raise ValueError("record who changed consent")
        removed = 0
        with self._lock:
            db = self._conn()
            with db:
                db.execute("INSERT INTO consent (id, enabled, changed_at, changed_by) "
                           "VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                           "enabled = excluded.enabled, changed_at = excluded.changed_at, "
                           "changed_by = excluded.changed_by",
                           (1 if enabled else 0, self.clock(), who))
                if purge and not enabled:
                    removed = db.execute("DELETE FROM events").rowcount
                    db.execute("DELETE FROM reviews")
        return removed

    def exports(self) -> list[dict[str, Any]]:
        """Every export written from this store. Withdrawing consent does not
        reach these copies, so they are listed for the user to decide about;
        nothing is deleted on their behalf."""
        with self._lock:
            rows = self._conn().execute(
                "SELECT directory, data_hash, created_at, tasks, selection_policy "
                "FROM exports ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    # ── recording ───────────────────────────────────────────────────────────

    def _refuse(self, reason: str) -> None:
        self.last_refusal = reason
        return None

    def _insert(self, row: dict[str, Any]) -> str | None:
        if not self.consent_enabled():
            return self._refuse("feedback consent is off")
        if row["source"] not in SOURCES or row["kind"] not in KINDS:
            return self._refuse("unknown source or kind")
        for field_name in ("prompt_json", "plan_json", "receipts_json", "note"):
            value = row.get(field_name)
            if value and _contains_secret(value):
                return self._refuse(f"{field_name.split('_')[0]} looked like it contained a "
                                    "secret; nothing was recorded")
        row = {**_EVENT_DEFAULTS, **row, "created_at": self.clock()}
        row["id"] = f"fb-{uuid.uuid4().hex[:16]}"
        row["digest"] = _event_digest(row)
        columns = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            db = self._conn()
            with db:
                db.execute(f"INSERT INTO events ({columns}) VALUES ({marks})",
                           tuple(row.values()))
        self.last_refusal = ""
        return row["id"]

    def _plan_event(self, task_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn().execute(
                "SELECT * FROM events WHERE task_id = ? AND kind = 'plan' "
                "ORDER BY created_at LIMIT 1", (str(task_id),)).fetchone()

    def record_plan(self, *, task_id: str, prompt: dict[str, str], plan: dict[str, Any],
                    session_id: str = "", base_model: str = "", adapter_version: str = "",
                    planner_version: str = "", policy_version: str = POLICY_VERSION
                    ) -> str | None:
        """The exact prompt the planner saw and the plan it proposed.

        The prompt is kept whole (system and user text) because a training
        example must reproduce inference, not a summary of it.
        """
        system = str((prompt or {}).get("system") or "")
        user = str((prompt or {}).get("user") or "")
        if not user.strip():
            return self._refuse("a plan needs the prompt it answered")
        prompt_json = _canonical({"system": system, "user": user})
        text = plan_text(plan)
        if len(prompt_json) > MAX_PROMPT_CHARS or len(text) > MAX_PLAN_CHARS:
            return self._refuse("prompt or plan too large to keep")
        return self._insert({
            "kind": "plan", "task_id": str(task_id), "session_id": str(session_id or ""),
            "source": "planner", "prompt_json": prompt_json, "plan_json": text,
            "base_model": str(base_model or ""), "adapter_version": str(adapter_version or ""),
            "planner_version": str(planner_version or ""),
            "policy_version": str(policy_version or "")})

    def record_correction(self, *, task_id: str, corrected_plan: dict[str, Any],
                          source: str, note: str = "") -> str | None:
        """What the plan SHOULD have been, for a prompt already recorded."""
        if source not in {"user_explicit", "user_relayed"}:
            return self._refuse("a correction comes from the user")
        plan = self._plan_event(task_id)
        if plan is None:
            return self._refuse("no recorded plan to correct (was consent on at the time?)")
        checked, problem = _check_plan_contract(corrected_plan)
        if checked is None:
            return self._refuse(f"the corrected plan is not a valid plan: {problem}")
        text = plan_text(checked)
        if len(text) > MAX_PLAN_CHARS:
            return self._refuse("corrected plan too large to keep")
        return self._insert({
            "kind": "correction", "task_id": str(task_id), "session_id": plan["session_id"],
            "source": source, "plan_json": text, "note": str(note or "")[:400],
            "base_model": plan["base_model"], "adapter_version": plan["adapter_version"],
            "planner_version": plan["planner_version"],
            "policy_version": plan["policy_version"]})

    def record_outcome(self, *, task_id: str, receipts: Iterable[dict[str, Any]],
                       tool_returned_ok: bool | None,
                       postcondition_verified: bool | None) -> str | None:
        """The executor's receipts. ``tool_returned_ok`` and
        ``postcondition_verified`` are kept apart on purpose: only the second
        (or the user's acceptance) can make a plan a positive label."""
        plan = self._plan_event(task_id)
        if plan is None:
            return self._refuse("no recorded plan for this outcome")
        kept = []
        for receipt in list(receipts or [])[:32]:
            kept.append({
                "step_id": str(receipt.get("step_id") or "")[:40],
                "tool": str(receipt.get("tool") or "")[:64],
                "tool_returned_ok": _tri(receipt.get("tool_returned_ok")),
                "postcondition_verified": _tri(receipt.get("postcondition_verified")),
                "evidence": str(receipt.get("evidence") or "unknown")[:24],
                "skipped": bool(receipt.get("skipped")),
                "detail": str(receipt.get("detail") or "")[:MAX_DETAIL_CHARS]})
        return self._insert({
            "kind": "outcome", "task_id": str(task_id), "session_id": plan["session_id"],
            "source": "executor", "receipts_json": _canonical(kept),
            "tool_returned_ok": _tri_int(tool_returned_ok),
            "postcondition_verified": _tri_int(postcondition_verified),
            "base_model": plan["base_model"], "adapter_version": plan["adapter_version"],
            "planner_version": plan["planner_version"],
            "policy_version": plan["policy_version"]})

    def record_acceptance(self, *, task_id: str, accepted: bool, source: str,
                          note: str = "") -> str | None:
        """The user's verdict on whether the goal was actually met."""
        if source not in {"user_explicit", "user_relayed"}:
            return self._refuse("acceptance comes from the user")
        plan = self._plan_event(task_id)
        if plan is None:
            return self._refuse("no recorded plan to accept or reject")
        return self._insert({
            "kind": "acceptance", "task_id": str(task_id), "session_id": plan["session_id"],
            "source": source, "goal_accepted": 1 if accepted else 0,
            "note": str(note or "")[:400],
            "base_model": plan["base_model"], "adapter_version": plan["adapter_version"],
            "planner_version": plan["planner_version"],
            "policy_version": plan["policy_version"]})

    def review(self, event_id: str, verdict: str, *, reviewer: str) -> None:
        """Curation: approve a relayed label for training, or exclude any event."""
        verdict = str(verdict or "").strip().lower()
        if verdict not in {"approve", "reject"}:
            raise ValueError("verdict must be approve or reject")
        who = str(reviewer or "").strip()[:60]
        if not who:
            raise ValueError("record who reviewed it")
        with self._lock:
            db = self._conn()
            if db.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone() is None:
                raise KeyError(event_id)
            with db:
                db.execute("INSERT INTO reviews (event_id, reviewed_at, reviewer, verdict) "
                           "VALUES (?, ?, ?, ?)", (event_id, self.clock(), who, verdict))

    # ── reading ─────────────────────────────────────────────────────────────

    def events(self, task_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            db = self._conn()
            if task_id is None:
                rows = db.execute("SELECT * FROM events ORDER BY created_at LIMIT ?",
                                  (int(limit),)).fetchall()
            else:
                rows = db.execute("SELECT * FROM events WHERE task_id = ? ORDER BY created_at",
                                  (str(task_id),)).fetchall()
        return [dict(row) for row in rows]

    def verify(self, event: dict[str, Any]) -> bool:
        """Whether an event's content still matches the digest it was written with."""
        return _event_digest(event) == event.get("digest")

    def _verdicts(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT event_id, verdict FROM reviews ORDER BY reviewed_at").fetchall()
        latest: dict[str, str] = {}
        for row in rows:
            latest[row["event_id"]] = row["verdict"]
        return latest

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT kind, source, COUNT(*) AS n FROM events GROUP BY kind, source").fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            counts[f"{row['kind']}/{row['source']}"] = int(row["n"])
        return {"path": str(self.path), "consent": self.consent(), "events": counts}

    # ── export ──────────────────────────────────────────────────────────────

    def examples(self, *, explicit_only: bool = False) -> list[dict[str, Any]]:
        """Training examples, one or two per labelled task, with provenance.

        The user's verdicts on a task — corrections, acceptances and
        rejections — are ordered by time and the LATEST decides
        (``SELECTION_POLICY``): a correction gives an SFT example of the
        corrected plan and a DPO pair preferring it over the original; an
        acceptance gives an SFT example of the original; a rejection gives
        nothing and withdraws every earlier positive. Only with no user
        verdict at all does a verified postcondition label the original plan.
        An outcome that merely returned ok labels nothing.

        Positive verdicts must be eligible (explicit, or relayed and approved
        on review). A rejection always counts, even relayed and unreviewed:
        withdrawing a label is the safe direction. A review that rejects an
        event removes it, whatever it said.
        """
        verdicts = self._verdicts()
        tasks: dict[str, list[dict[str, Any]]] = {}
        for event in self.events(limit=10_000_000):
            if not self.verify(event):
                continue                      # altered after writing — not evidence
            if verdicts.get(event["id"]) == "reject":
                continue
            tasks.setdefault(event["task_id"], []).append(event)

        def eligible(event: dict[str, Any]) -> bool:
            if event["source"] == "user_explicit":
                return True
            if explicit_only:
                return False
            return (event["source"] == "user_relayed"
                    and verdicts.get(event["id"]) == "approve")

        def counts_as_verdict(event: dict[str, Any]) -> bool:
            if event["kind"] == "acceptance" and event["goal_accepted"] == 0:
                return event["source"] in {"user_explicit", "user_relayed"}
            return event["kind"] in {"correction", "acceptance"} and eligible(event)

        out: list[dict[str, Any]] = []
        for task_id, events in tasks.items():
            plans = [e for e in events if e["kind"] == "plan"]
            if not plans:
                continue
            plan = plans[0]
            try:
                prompt = json.loads(plan["prompt_json"] or "{}")
            except Exception:
                continue
            messages = [{"role": "system", "content": str(prompt.get("system") or "")},
                        {"role": "user", "content": str(prompt.get("user") or "")}]
            if not messages[0]["content"]:
                messages = messages[1:]
            ordered = sorted(events, key=lambda e: (e["created_at"], e["id"]))
            user_verdicts = [e for e in ordered if counts_as_verdict(e)]
            outcomes = [e for e in ordered if e["kind"] == "outcome"]
            base = {"task_id": task_id, "created_at": plan["created_at"],
                    "base_model": plan["base_model"], "adapter_version": plan["adapter_version"],
                    "policy_version": plan["policy_version"]}
            if user_verdicts:
                latest = user_verdicts[-1]
                if latest["kind"] == "correction":
                    out.append({**base, "type": "sft", "label_source": latest["source"],
                                "label_event": latest["id"],
                                "messages": messages + [{"role": "assistant",
                                                         "content": latest["plan_json"]}]})
                    if latest["plan_json"] != plan["plan_json"]:
                        out.append({**base, "type": "dpo", "label_source": latest["source"],
                                    "label_event": latest["id"], "prompt": messages,
                                    "chosen": [{"role": "assistant",
                                                "content": latest["plan_json"]}],
                                    "rejected": [{"role": "assistant",
                                                  "content": plan["plan_json"]}]})
                elif latest["goal_accepted"] == 1:
                    out.append({**base, "type": "sft", "label_source": latest["source"],
                                "label_event": latest["id"],
                                "messages": messages + [{"role": "assistant",
                                                         "content": plan["plan_json"]}]})
                # else: the latest word is a rejection — no example.
            elif outcomes and not explicit_only and outcomes[-1]["postcondition_verified"] == 1:
                latest = outcomes[-1]
                out.append({**base, "type": "sft", "label_source": latest["source"],
                            "label_event": latest["id"],
                            "messages": messages + [{"role": "assistant",
                                                     "content": plan["plan_json"]}]})
        return out

    def export(self, directory: Path | str, *, holdout_fraction: float = 0.15,
               replay_fraction: float = 0.3, explicit_only: bool = False) -> dict[str, Any]:
        """Write an immutable training snapshot into a NEW directory.

        Splits are by whole task, chronologically: the newest tasks are held
        out (never trained on — the evaluation's forgetting check), the
        oldest of the rest form the replay set mixed into every later run,
        and the remainder is this run's training data. A prompt that appears
        in the held-out split is removed from the others, so a held-out score
        is never a memory test.
        """
        if not 0.0 <= holdout_fraction < 1.0 or not 0.0 <= replay_fraction < 1.0:
            raise ValueError("fractions must be in [0, 1)")
        consent = self.consent()
        if not consent["in_force"]:
            # Consent covers using the data, not only collecting it.
            raise PermissionError("planner feedback consent is off; nothing was exported")
        target = Path(directory)
        examples = self.examples(explicit_only=explicit_only)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for example in examples:
            key = _sha256(_canonical({k: example.get(k) for k in
                                      ("type", "messages", "prompt", "chosen", "rejected")}))
            if key not in seen:
                seen.add(key)
                unique.append(example)
        order: dict[str, float] = {}
        for example in unique:
            order[example["task_id"]] = min(order.get(example["task_id"], math.inf),
                                            example["created_at"])
        task_ids = sorted(order, key=lambda t: (order[t], t))
        n = len(task_ids)
        held = math.ceil(n * holdout_fraction) if n >= 2 and holdout_fraction > 0 else 0
        heldout_ids = set(task_ids[n - held:]) if held else set()
        rest = task_ids[:n - held] if held else task_ids
        replay_n = math.floor(len(rest) * replay_fraction)
        replay_ids = set(rest[:replay_n])
        split_of = {t: ("heldout" if t in heldout_ids else
                        "replay" if t in replay_ids else "train") for t in task_ids}

        def prompt_key(example: dict[str, Any]) -> str:
            messages = example.get("prompt") or example.get("messages", [])[:-1]
            return _sha256(_canonical([m for m in messages if m.get("role") == "user"]))

        heldout_prompts = {prompt_key(e) for e in unique if split_of[e["task_id"]] == "heldout"}
        leaked = 0
        files: dict[str, list[dict[str, Any]]] = {
            f"{kind}_{split}.jsonl": [] for kind in ("sft", "dpo")
            for split in ("train", "replay", "heldout")}
        for example in unique:
            split = split_of[example["task_id"]]
            if split != "heldout" and prompt_key(example) in heldout_prompts:
                leaked += 1
                continue
            record = {k: v for k, v in example.items() if k not in {"type", "created_at"}}
            files[f"{example['type']}_{split}.jsonl"].append(record)

        target.mkdir(parents=True, exist_ok=False)    # a snapshot is never rewritten
        hashes: dict[str, str] = {}
        counts: dict[str, int] = {}
        for name, rows in files.items():
            text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                           for row in rows)
            (target / name).write_text(text, encoding="utf-8")
            hashes[name] = _sha256(text)
            counts[name] = len(rows)
        data_hash = _sha256(_canonical(hashes))
        manifest = {
            "format": EXPORT_FORMAT, "created_at": self.clock(),
            "policy_version": POLICY_VERSION, "explicit_only": bool(explicit_only),
            "holdout_fraction": holdout_fraction, "replay_fraction": replay_fraction,
            "counts": counts, "files": hashes, "data_hash": data_hash,
            "removed_for_prompt_leakage": leaked,
            "tasks": {split: sorted(t for t, s in split_of.items() if s == split)
                      for split in ("train", "replay", "heldout")},
            "selection_policy": SELECTION_POLICY,
            "label_sources": {source: sum(1 for e in unique if e["label_source"] == source)
                              for source in sorted({e["label_source"] for e in unique})},
            "consent": {"changed_at": consent["changed_at"],
                        "changed_by": consent["changed_by"]},
        }
        from .atomic_io import atomic_write_text
        atomic_write_text(target / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        with self._lock:
            db = self._conn()
            with db:
                db.execute("INSERT INTO exports (directory, data_hash, created_at, tasks, "
                           "selection_policy) VALUES (?, ?, ?, ?, ?)",
                           (str(target.resolve()), data_hash, manifest["created_at"], n,
                            SELECTION_POLICY.split(":", 1)[0]))
        return manifest


def _check_plan_contract(plan: Any) -> tuple[dict[str, Any] | None, str]:
    """(canonical plan, "") when *plan* is one the planner itself could have
    accepted — the plan shape, plannable tools with valid arguments, no
    approval-shaped arguments, a valid dependency graph — else (None, why).
    A correction becomes a training target, so it meets the same contract as
    a model's reply; a hand-written plan the executor would refuse must not
    teach the planner to write it."""
    if not isinstance(plan, dict):
        return None, "it is not a JSON object"
    from .dispatch_schema import TOOL_DECLARATIONS
    from .neural_planner import PlanRejected, check_plan, plannable
    from .plan_executor import MAX_PLAN_STEPS
    named = {str(step.get("tool")) for step in plan.get("steps") or []
             if isinstance(step, dict)}
    offered = [d for d in TOOL_DECLARATIONS
               if d["name"] in named and plannable(d["name"], "effects")]
    try:
        checked, _steps = check_plan(plan, offered=offered, scope="effects",
                                     max_steps=MAX_PLAN_STEPS)
    except PlanRejected as rejected:
        return None, rejected.reasons[0]
    return checked, ""


def _tri(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _tri_int(value: Any) -> int | None:
    return None if value is None else (1 if value else 0)


def verify_export(directory: Path | str) -> dict[str, Any]:
    """The manifest of an export, after checking every file against it.
    Raises ValueError if anything was changed after it was written."""
    folder = Path(directory)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != EXPORT_FORMAT:
        raise ValueError("not a planner feedback export")
    for name, digest in (manifest.get("files") or {}).items():
        if Path(name).name != name:
            raise ValueError(f"unexpected file name in manifest: {name}")
        if _sha256((folder / name).read_text(encoding="utf-8")) != digest:
            raise ValueError(f"{name} does not match the export manifest")
    if _sha256(_canonical(manifest.get("files") or {})) != manifest.get("data_hash"):
        raise ValueError("export data hash does not match its files")
    return manifest


# ── command line: the user's explicit channel ────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m orion_core.neural_feedback",
                                     description="Planner feedback: consent, review, export.")
    parser.add_argument("--db", type=Path, default=None, help="feedback database (default: config)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    consent = sub.add_parser("consent")
    consent.add_argument("state", choices=("on", "off"))
    consent.add_argument("--purge", action="store_true",
                         help="with 'off', delete every recorded event")
    correct = sub.add_parser("correct")
    correct.add_argument("plan_id")
    correct.add_argument("plan_file", type=Path)
    correct.add_argument("--note", default="")
    review = sub.add_parser("review")
    review.add_argument("event_id")
    review.add_argument("verdict", choices=("approve", "reject"))
    export = sub.add_parser("export")
    export.add_argument("directory", type=Path)
    export.add_argument("--explicit-only", action="store_true")
    export.add_argument("--holdout", type=float, default=0.15)
    export.add_argument("--replay", type=float, default=0.3)
    args = parser.parse_args(argv)
    store = FeedbackStore(args.db)
    try:
        if args.command == "status":
            print(json.dumps(store.stats(), indent=2, default=str))
        elif args.command == "consent":
            removed = store.set_consent(args.state == "on", changed_by="user:cli",
                                        purge=args.purge)
            print(f"Planner feedback consent is now {args.state}."
                  + (f" {removed} event(s) deleted." if removed else ""))
            written = store.exports() if args.state == "off" else []
            if written:
                print("These exports were written earlier and are NOT deleted by this; "
                      "remove them (and any candidate trained from them) if you wish:")
                for export in written:
                    print(f"  {export['directory']}  ({export['tasks']} task(s), "
                          f"data {export['data_hash'][:12]})")
        elif args.command == "correct":
            corrected = json.loads(args.plan_file.read_text(encoding="utf-8"))
            event = store.record_correction(task_id=args.plan_id, corrected_plan=corrected,
                                            source="user_explicit", note=args.note)
            if event is None:
                print(f"Not recorded: {store.last_refusal}", file=sys.stderr)
                return 2
            print(f"Recorded correction {event}.")
        elif args.command == "review":
            store.review(args.event_id, args.verdict, reviewer="user:cli")
            print(f"{args.event_id}: {args.verdict}d.")
        elif args.command == "export":
            manifest = store.export(args.directory, holdout_fraction=args.holdout,
                                    replay_fraction=args.replay,
                                    explicit_only=args.explicit_only)
            print(json.dumps({"directory": str(args.directory), "counts": manifest["counts"],
                              "data_hash": manifest["data_hash"]}, indent=2))
    finally:
        store.close()
    return 0


__all__ = ["ENV_KILL", "EXPORT_FORMAT", "FEEDBACK_PATH", "FeedbackStore",
           "POLICY_VERSION", "SELECTION_POLICY", "plan_text", "verify_export"]


if __name__ == "__main__":
    raise SystemExit(main())
