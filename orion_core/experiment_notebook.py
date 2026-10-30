"""Local, preregistered personal experiments with honest descriptive comparisons.

Designs and observations are append-only. Analysis compares user-entered
baseline and trial measurements; it does not establish causality or statistical
significance. Call ``handle`` off the event loop: it performs SQLite I/O.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import ToolResult
from .db import connect

MAX_EXPERIMENTS = 500
MAX_OBSERVATIONS = 2000
MAX_VALUE = 1e100
_DESIGN_FIELDS = frozenset({"title", "hypothesis", "metric", "unit", "direction", "target_improvement"})
_ACTIONS = frozenset({"create", "list", "show", "record", "analyse", "archive"})
_CAUTION = ("Descriptive comparison only: these observations do not establish causality "
            "or statistical significance. Other changes, measurement noise and order effects may explain the difference.")


def default_path() -> Path:
    # Resolve at construction, not import, so tests and relocated configs use
    # the live setting and importing this module has no application side effects.
    from .constants import CONFIG_DIR
    return CONFIG_DIR / "experiments.db"


def _text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text.")
    clean = " ".join(value.replace("\x00", " ").split())
    if required and not clean:
        raise ValueError(f"{name} is required.")
    if len(clean) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return clean


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number.") from None
    if not math.isfinite(number) or abs(number) > MAX_VALUE:
        raise ValueError(f"{name} must be finite and between -1e100 and 1e100.")
    return number


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _fmt(value: float | None) -> str:
    return "not available" if value is None else f"{value:.6g}"


class ExperimentNotebook:
    """No persistent connection: each operation is isolated and thread-safe.

    ``create`` takes title, hypothesis, metric, direction (higher/lower), and
    optional unit/target_improvement (an absolute change in that metric).
    ``record`` takes id, phase (baseline/trial), value, and optional note.
    ``show``, ``analyse`` and ``archive`` take id; ``list`` takes optional
    limit (1..50) and include_archived. IDs are returned in tool receipts.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()

    def _connection(self) -> sqlite3.Connection:
        db = connect(self.path)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    hypothesis TEXT NOT NULL, metric TEXT NOT NULL,
                    unit TEXT NOT NULL, direction TEXT NOT NULL CHECK(direction IN ('higher','lower')),
                    target_improvement REAL, created_at TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0,1))
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    phase TEXT NOT NULL CHECK(phase IN ('baseline','trial')),
                    value REAL NOT NULL, note TEXT NOT NULL, recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS observations_experiment
                    ON observations(experiment_id, id);
                CREATE TRIGGER IF NOT EXISTS immutable_experiment_design
                    BEFORE UPDATE OF title,hypothesis,metric,unit,direction,target_improvement,created_at
                    ON experiments BEGIN
                        SELECT RAISE(ABORT, 'Experiment designs are immutable');
                    END;
            """)
        except BaseException:
            db.close()
            raise
        return db

    def handle(self, args: dict[str, Any]) -> ToolResult:
        """Return bounded text and a typed receipt; invalid inputs never write."""
        try:
            if not isinstance(args, dict):
                raise ValueError("Experiment arguments must be an object.")
            action = _text(args.get("action", "list"), "action", 24).lower()
            if action not in _ACTIONS:
                raise ValueError("Use create, list, show, record, analyse or archive. Designs cannot be edited.")
            if action != "create" and _DESIGN_FIELDS.intersection(args):
                raise ValueError("Experiment designs are immutable. Create a new experiment to change the design.")
            if action == "create":
                return self._create(args)
            if action == "list":
                return self._list(args)
            experiment_id = _text(args.get("id"), "id", 64, required=True)
            if action == "record":
                phase = _text(args.get("phase"), "phase", 16, required=True).lower()
                if phase not in {"baseline", "trial"}:
                    raise ValueError("phase must be baseline or trial.")
                value = _number(args.get("value"), "value")
                note = _text(args.get("note", ""), "note", 1000)
                return self._record(experiment_id, phase, value, note)
            if action == "archive":
                return self._archive(experiment_id)
            return self._read(experiment_id, action)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False, receipt={"tool": "experiment_notebook", "status": "invalid"})
        except (sqlite3.Error, OSError):
            return ToolResult("The experiment notebook could not complete this operation. No success was confirmed; "
                              "check the notebook before retrying a write.", ok=False,
                              receipt={"tool": "experiment_notebook", "status": "storage_error"})

    @staticmethod
    def _result(action: str, text: str, **receipt: Any) -> ToolResult:
        return ToolResult(text, receipt={"tool": "experiment_notebook", "action": action, **receipt})

    @staticmethod
    def _find(db: sqlite3.Connection, experiment_id: str) -> dict[str, Any]:
        row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            raise ValueError(f"No experiment with id '{experiment_id}'. Use list to find its exact id.")
        experiment = dict(row)
        experiment["archived"] = bool(experiment["archived"])
        return experiment

    def _create(self, args: dict[str, Any]) -> ToolResult:
        design = {
            "id": uuid.uuid4().hex,
            "title": _text(args.get("title"), "title", 160, required=True),
            "hypothesis": _text(args.get("hypothesis"), "hypothesis", 1500, required=True),
            "metric": _text(args.get("metric"), "metric", 120, required=True),
            "unit": _text(args.get("unit", ""), "unit", 40),
            "direction": _text(args.get("direction"), "direction", 16, required=True).lower(),
            "target_improvement": None,
            "created_at": _stamp(),
            "archived": False,
        }
        if design["direction"] not in {"higher", "lower"}:
            raise ValueError("direction must be higher or lower.")
        if args.get("target_improvement") is not None:
            design["target_improvement"] = _number(args["target_improvement"], "target_improvement")
            if design["target_improvement"] <= 0:
                raise ValueError("target_improvement must be positive, in the metric's own units.")
        with closing(self._connection()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            if count >= MAX_EXPERIMENTS:
                raise ValueError(f"The notebook limit of {MAX_EXPERIMENTS} experiments has been reached.")
            db.execute("INSERT INTO experiments VALUES (:id,:title,:hypothesis,:metric,:unit,:direction,"
                       ":target_improvement,:created_at,:archived)", design)
        return self._result("create", f"Registered experiment '{design['title']}' ({design['id']}). "
                            f"Measure {design['metric']}; {design['direction']} is better. "
                            "The design is fixed. Record baseline and trial observations separately.",
                            saved=True, experiment=design)

    def _list(self, args: dict[str, Any]) -> ToolResult:
        limit = args.get("limit", 20)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer from 1 to 50.")
        include_archived = args.get("include_archived", False)
        if not isinstance(include_archived, bool):
            raise ValueError("include_archived must be a boolean.")
        with closing(self._connection()) as db:
            rows = db.execute("SELECT id,title,metric,unit,direction,archived FROM experiments "
                              "WHERE (? OR archived=0) ORDER BY created_at DESC,id LIMIT ?",
                              (include_archived, limit)).fetchall()
            experiments = [{**dict(row), "archived": bool(row["archived"])} for row in rows]
        lines = [f"Experiments ({len(experiments)} shown):"]
        lines += [f"- {e['id']}: {e['title']} [{e['metric']}; "
                  f"{'archived' if e['archived'] else 'active'}]" for e in experiments]
        if not experiments:
            lines.append("None saved in this view. Create an experiment with a hypothesis and measurable outcome.")
        return self._result("list", "\n".join(lines), experiments=experiments)

    def _record(self, experiment_id: str, phase: str, value: float, note: str) -> ToolResult:
        with closing(self._connection()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            experiment = self._find(db, experiment_id)
            if experiment["archived"]:
                raise ValueError("This experiment is archived. Its observations remain readable; create a new experiment to continue.")
            count = db.execute("SELECT COUNT(*) FROM observations WHERE experiment_id=?", (experiment_id,)).fetchone()[0]
            if count >= MAX_OBSERVATIONS:
                raise ValueError(f"This experiment has reached its {MAX_OBSERVATIONS}-observation limit.")
            recorded_at = _stamp()
            cursor = db.execute("INSERT INTO observations(experiment_id,phase,value,note,recorded_at) VALUES(?,?,?,?,?)",
                                (experiment_id, phase, value, note, recorded_at))
            observation = {"id": cursor.lastrowid, "experiment_id": experiment_id, "phase": phase,
                           "value": value, "note": note, "recorded_at": recorded_at}
        return self._result("record", f"Saved {phase} observation for '{experiment['title']}': "
                            f"{_fmt(value)} {experiment['unit']}. Observation {observation['id']}.",
                            saved=True, experiment_id=experiment_id, observation=observation)

    def _archive(self, experiment_id: str) -> ToolResult:
        with closing(self._connection()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            experiment = self._find(db, experiment_id)
            changed = not experiment["archived"]
            db.execute("UPDATE experiments SET archived=1 WHERE id=?", (experiment_id,))
        return self._result("archive", f"Experiment '{experiment['title']}' "
                            f"{'archived' if changed else 'was already archived'}. Its design and observations are preserved.",
                            saved=True, experiment_id=experiment_id, changed=changed, archived=True)

    def _read(self, experiment_id: str, action: str) -> ToolResult:
        with closing(self._connection()) as db, db:
            # Snapshot both design/status and observations from the same point.
            db.execute("BEGIN")
            experiment = self._find(db, experiment_id)
            observations = [dict(row) for row in db.execute(
                "SELECT id,phase,value,note,recorded_at FROM observations WHERE experiment_id=? ORDER BY id",
                (experiment_id,)).fetchall()]
        summary = self._summary(experiment, observations)
        lines = [f"Experiment: {experiment['title']} ({experiment_id})" + (" [archived]" if experiment["archived"] else ""),
                 f"Hypothesis: {experiment['hypothesis']}",
                 f"Metric: {experiment['metric']} ({experiment['unit'] or 'unitless'}); {experiment['direction']} is better.",
                 f"Baseline: n={summary['baseline']['n']}, mean={_fmt(summary['baseline']['mean'])}.",
                 f"Trial: n={summary['trial']['n']}, mean={_fmt(summary['trial']['mean'])}."]
        if summary["absolute_change"] is not None:
            lines.append(f"Absolute change (trial - baseline): {_fmt(summary['absolute_change'])} {experiment['unit']}.")
            if summary["relative_change_percent"] is not None:
                lines.append(f"Relative change: {_fmt(summary['relative_change_percent'])}%.")
            else:
                lines.append("Relative change unavailable: baseline mean must be positive for a meaningful percentage.")
        if not summary["sufficient_data"]:
            lines.append("Insufficient evidence: record at least 2 observations in each phase before comparing the target.")
        if experiment["target_improvement"] is not None:
            lines.append(f"Preregistered target: improve by {_fmt(experiment['target_improvement'])} {experiment['unit']}.")
            if summary["target_met"] is not None:
                lines.append("The observed mean difference " + ("meets" if summary["target_met"] else "does not meet")
                             + " that target; this is a descriptive result only.")
        if action == "show" and observations:
            lines.append(f"Latest {min(25, len(observations))} observations (all {len(observations)} used in analysis):")
            for row in observations[-25:]:
                lines.append(f"- #{row['id']} {row['phase']}: {_fmt(row['value'])}; {row['recorded_at']}"
                             + (f"; {row['note']}" if row["note"] else ""))
        lines.append(_CAUTION)
        return self._result(action, "\n".join(lines), experiment=experiment, analysis=summary,
                            **({"observations": observations[-25:]} if action == "show" else {}))

    @staticmethod
    def _summary(experiment: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for phase in ("baseline", "trial"):
            values = [row["value"] for row in observations if row["phase"] == phase]
            summary[phase] = {"n": len(values), "mean": math.fsum(values) / len(values) if values else None}
        before, after = summary["baseline"]["mean"], summary["trial"]["mean"]
        change = after - before if before is not None and after is not None else None
        improvement = (change if experiment["direction"] == "higher" else -change) if change is not None else None
        sufficient = min(summary["baseline"]["n"], summary["trial"]["n"]) >= 2
        relative = change / before * 100 if change is not None and before > 0 else None
        # Very small nonzero baselines can produce an unrepresentable ratio.
        if relative is not None and not math.isfinite(relative):
            relative = None
        target = experiment["target_improvement"]
        summary.update(absolute_change=change, relative_change_percent=relative,
                       improvement=improvement, sufficient_data=sufficient,
                       target_met=(improvement >= target if sufficient and target is not None else None),
                       interpretation="descriptive_only", caution=_CAUTION)
        return summary


__all__ = ["ExperimentNotebook"]
