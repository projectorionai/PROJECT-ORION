"""Concrete, verified operations behind older local plugin entry points.

Stored goals, diagnostic records and draft specifications are local data.
Recording an input does not imply a repair, a reminder or a generated tool.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import closing
import uuid
from datetime import date, datetime, timezone

from .constants import CONFIG_DIR
from .data import ToolResult
from .db import connect

STORE_PATH = CONFIG_DIR / "tool_outcomes.db"


def _text(value: object, label: str, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be non-empty text of at most {limit} characters.")
    return value.strip()


def _save(kind: str, key: str, payload: dict) -> dict:
    """Commit and read back before reporting persistence as successful."""
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    if len(encoded.encode("utf-8")) > 128 * 1024:
        raise ValueError("Record is too large (maximum 128 KiB).")
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(STORE_PATH)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, key TEXT, payload TEXT, updated_at TEXT, PRIMARY KEY(kind,key))")
        db.execute("INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(kind,key) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                   (kind, key, encoded, datetime.now(timezone.utc).isoformat()))
    with closing(connect(STORE_PATH)) as db, db:
        row = db.execute("SELECT payload FROM records WHERE kind=? AND key=?", (kind, key)).fetchone()
    if row is None or row[0] != encoded:
        raise OSError("The saved record could not be verified.")
    return {"kind": kind, "record_id": key, "persisted": True,
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}


def save_goal(goal, deadline, progress=0, status="In Progress") -> ToolResult:
    try:
        goal = _text(goal, "Goal", 500)
        deadline = date.fromisoformat(_text(deadline, "Deadline", 10)).isoformat()
        if type(progress) is not int or not 0 <= progress <= 100:
            raise ValueError("Progress must be an integer from 0 to 100.")
        if status not in {"In Progress", "Completed", "Failed"}:
            raise ValueError("Status must be In Progress, Completed or Failed.")
        if status == "Completed" and progress != 100:
            raise ValueError("A completed goal must have 100% progress.")
        evidence = _save("fitness_goal", goal.casefold(), dict(goal=goal, deadline=deadline, progress=progress, status=status))
        return ToolResult(f"Saved fitness goal: {goal}. Due {deadline}; {progress}%; {status}. No reminder was scheduled.", evidence=[evidence])
    except Exception as exc:
        return ToolResult(f"Fitness goal was not saved: {exc}", ok=False)


def save_progress(user_id, workout_data, reminders, performance_metrics) -> ToolResult:
    try:
        user_id = _text(user_id, "User ID", 120)
        if not isinstance(workout_data, list) or not all(isinstance(x, dict) for x in workout_data):
            raise ValueError("Workout data must be a list of records.")
        if not isinstance(reminders, list) or not all(isinstance(x, str) for x in reminders):
            raise ValueError("Reminders must be a list of strings.")
        if not isinstance(performance_metrics, dict):
            raise ValueError("Performance metrics must be an object.")
        evidence = _save("fitness_progress", user_id, dict(workouts=workout_data, requested_reminders=reminders, metrics=performance_metrics))
        evidence["reminders_scheduled"] = 0
        return ToolResult(f"Saved {len(workout_data)} supplied workout records and metrics for {user_id}. "
                          f"{len(reminders)} reminder requests were recorded; no reminders were scheduled.", evidence=[evidence])
    except Exception as exc:
        return ToolResult(f"Fitness progress was not saved: {exc}", ok=False)


def record_error(error_message, error_type, context="") -> ToolResult:
    try:
        message = _text(error_message, "Error message")
        kind = _text(error_type, "Error type", 120)
        if not isinstance(context, str) or len(context) > 8000:
            raise ValueError("Context must be text of at most 8000 characters.")
        evidence = _save("diagnostic", uuid.uuid4().hex, dict(error_type=kind, message=message, context=context))
        evidence["repaired"] = False
        return ToolResult(f"Recorded {kind} for diagnosis (record {evidence['record_id']}). No repair was applied; use self_repair to investigate.", evidence=[evidence])
    except Exception as exc:
        return ToolResult(f"Diagnostic was not recorded: {exc}", ok=False)


def save_tool_spec(tool_name, capability_plan) -> ToolResult:
    try:
        name = _text(tool_name, "Tool name", 64)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Use a lowercase tool name containing letters, digits and underscores.")
        plan = _text(capability_plan, "Capability plan", 16000)
        evidence = _save("tool_specification", name, dict(name=name, plan=plan, status="draft"))
        evidence.update(created_tool=False, activated=False)
        return ToolResult(f"Saved draft specification for {name}. No executable tool was created or activated; use forge to implement and validate it.", evidence=[evidence])
    except Exception as exc:
        return ToolResult(f"Tool specification was not saved: {exc}", ok=False)


def analyse_text(user_input, emotional_threshold=0.5, max_results=1) -> ToolResult:
    try:
        text = _text(user_input, "Input", 16000)
        if isinstance(emotional_threshold, bool) or not isinstance(emotional_threshold, (int, float)) or not math.isfinite(emotional_threshold) or not 0 <= emotional_threshold <= 1:
            raise ValueError("Threshold must be a finite number from 0 to 1.")
        if type(max_results) is not int or not 1 <= max_results <= 20:
            raise ValueError("Maximum results must be an integer from 1 to 20.")
        from .emotion import SentimentAnalyser
        reading = SentimentAnalyser.analyse(text, origin="user")
        accepted = reading.get("confidence", 0) >= emotional_threshold
        evidence = {"method": "lexical_text_heuristic", "reading": reading if accepted else None,
                    "verified_emotion": False}
        result = json.dumps(reading, ensure_ascii=False) if accepted else "No text cue met the requested threshold."
        return ToolResult("Text sentiment estimate: " + result + " This is a lexical hint, not a verified emotional state.", evidence=[evidence])
    except Exception as exc:
        return ToolResult(f"Text sentiment could not be estimated: {exc}", ok=False)
