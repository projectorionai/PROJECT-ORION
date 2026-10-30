"""User-written work checkpoints with an explicit next action.

A capsule preserves the user's chosen context, blockers and references. It
never captures conversations/screens, opens references, or restores apps.
Saving without an id always creates a new checkpoint, even for the same title.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator
import uuid

from .data import ToolResult


_TEXT_LIMITS = {"title": 160, "intent": 2000, "context": 12000, "next_action": 2000}
_LIST_LIMITS = {"blockers": (16, 1000), "assumptions": (16, 1000),
                "references": (24, 2048)}
_FIELDS = tuple(_TEXT_LIMITS) + tuple(_LIST_LIMITS)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS capsules (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    intent TEXT NOT NULL,
    context TEXT NOT NULL,
    next_action TEXT NOT NULL,
    blockers TEXT NOT NULL,
    assumptions TEXT NOT NULL,
    reference_items TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0, 1))
)
"""


def _text(value: Any, field: str, maximum: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    value = value.strip()
    if "\x00" in value:
        raise ValueError(f"{field} cannot contain a null character.")
    if len(value) > maximum or (required and not value):
        minimum = "1 to " if required else "at most "
        raise ValueError(f"{field} must contain {minimum}{maximum} characters.")
    return value


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"cc-[0-9a-f]{32}", value):
        raise ValueError("Use the exact checkpoint id returned by save or list.")
    return value


def _fields(args: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, maximum in _TEXT_LIMITS.items():
        if field in args:
            result[field] = _text(args[field], field, maximum,
                                  required=field in {"title", "next_action"})
    for field, (count, maximum) in _LIST_LIMITS.items():
        if field not in args:
            continue
        value = args[field]
        if not isinstance(value, list) or len(value) > count:
            raise ValueError(f"{field} must be a list of at most {count} text items.")
        result[field] = [_text(item, field, maximum, required=True) for item in value]
    return result


def _record(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["references"] = result.pop("reference_items")
    for field in _LIST_LIMITS:
        result[field] = json.loads(result[field])
    result["archived"] = bool(result["archived"])
    return result


def _render(record: dict[str, Any]) -> str:
    state = " (archived)" if record["archived"] else ""
    lines = [f"{record['title']}{state}", f"Checkpoint: {record['id']}",
             f"Next action: {record['next_action']}"]
    for field in ("intent", "context"):
        if record[field]:
            lines.append(f"{field.capitalize()}: {record[field]}")
    for field in _LIST_LIMITS:
        if record[field]:
            lines.append(f"{field.capitalize()}:\n" +
                         "\n".join(f"- {item}" for item in record[field]))
    return "\n".join(lines)


def default_path() -> Path:
    """Resolve the active config directory only when a store is constructed."""
    from .constants import CONFIG_DIR
    return CONFIG_DIR / "context_capsules.db"


class ContextCapsuleStore:
    """Small local SQLite store; every operation owns and closes its connection."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            # Resolve at construction, not import: tests and portable installs
            # can select their own configuration directory before first use.
            path = default_path()
        self.path = Path(path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        from .db import connect
        db = connect(self.path)
        try:
            db.execute(_SCHEMA)
            db.commit()
            yield db
        finally:
            db.close()

    def handle(self, args: dict[str, Any]) -> ToolResult:
        """Execute save/list/show/archive with bounded, explicitly supplied data."""
        if not isinstance(args, dict):
            return ToolResult("Capsule arguments must be an object.", ok=False)
        action = args.get("action", "list")
        if not isinstance(action, str) or action not in {"save", "list", "show", "archive"}:
            return ToolResult("Capsule action must be save, list, show, or archive.", ok=False)
        try:
            if action == "save":
                return self._save(args)
            if action == "list":
                return self._list(args)
            return self._by_id(action, _identifier(args.get("id")))
        except (ValueError, TypeError) as exc:
            return ToolResult(f"Capsule request rejected: {exc}", ok=False,
                              receipt={"operation": action, "state": "rejected"})
        except (OSError, sqlite3.Error):
            return ToolResult("Capsule storage could not complete this request. "
                              "No save or archive is confirmed.", ok=False,
                              receipt={"operation": action, "state": "failed"})

    def _save(self, args: dict[str, Any]) -> ToolResult:
        changes = _fields(args)
        identifier = _identifier(args["id"]) if "id" in args else None
        if identifier is None:
            for field in ("title", "next_action"):
                if field not in changes:
                    raise ValueError(f"{field} is required for a new checkpoint.")
        elif not changes:
            raise ValueError("Supply at least one checkpoint field to update.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db, db:
            # Lock before reading for a patch, so simultaneous updates cannot
            # replace another caller's untouched fields with a stale snapshot.
            db.execute("BEGIN IMMEDIATE")
            if identifier is not None:
                row = db.execute("SELECT * FROM capsules WHERE id = ?", (identifier,)).fetchone()
                if row is None:
                    return self._missing(identifier)
                record = _record(row)
            else:
                identifier = "cc-" + uuid.uuid4().hex
                record = {field: "" for field in _TEXT_LIMITS}
                record.update({field: [] for field in _LIST_LIMITS})
                record.update(id=identifier, created_at=now, archived=False)
            record.update(changes, updated_at=now)
            values = tuple(json.dumps(record[field], ensure_ascii=False)
                           if field in _LIST_LIMITS else record[field] for field in _FIELDS)
            if "id" in args:
                db.execute("""UPDATE capsules SET title=?, intent=?, context=?, next_action=?,
                           blockers=?, assumptions=?, reference_items=?, updated_at=? WHERE id=?""",
                           (*values, now, identifier))
            else:
                db.execute("""INSERT INTO capsules
                           (title, intent, context, next_action, blockers, assumptions,
                            reference_items, updated_at, created_at, id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                           (*values, now, now, identifier))
        return ToolResult("Saved work checkpoint.\n" + _render(record),
                          receipt={"operation": "save", "state": "saved", "id": identifier,
                                   "capsule": record})

    def _list(self, args: dict[str, Any]) -> ToolResult:
        query = _text(args.get("query", ""), "query", 200)
        limit = args.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be a whole number from 1 to 50.")
        include_archived = args.get("include_archived", False)
        if type(include_archived) is not bool:
            raise ValueError("include_archived must be true or false.")
        clauses = [] if include_archived else ["archived = 0"]
        params: list[Any] = []
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            columns = ("title", "intent", "context", "next_action", "blockers",
                       "assumptions", "reference_items")
            clauses.append("(" + " OR ".join(f"{name} LIKE ? ESCAPE '\\'" for name in columns) + ")")
            params.extend([f"%{escaped}%"] * len(columns))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connection() as db:
            count = db.execute("SELECT COUNT(*) FROM capsules" + where, params).fetchone()[0]
            rows = db.execute("SELECT id, title, next_action, archived, created_at, updated_at "
                              "FROM capsules" + where + " ORDER BY updated_at DESC, id LIMIT ?",
                              (*params, limit)).fetchall()
        records = [dict(row, archived=bool(row["archived"])) for row in rows]
        if not records:
            text = "No matching work checkpoints."
        else:
            lines = [f"Work checkpoints ({len(records)} of {count}):"]
            for record in records:
                archived = " [archived]" if record["archived"] else ""
                lines.append(f"{record['id']} — {record['title']}{archived}\n"
                             f"Next action: {record['next_action']}")
            text = "\n".join(lines)
        return ToolResult(text, receipt={"operation": "list", "state": "listed",
                                        "capsules": records, "total": count, "limit": limit})

    def _by_id(self, action: str, identifier: str) -> ToolResult:
        with self._connection() as db, db:
            if action == "archive":
                db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM capsules WHERE id = ?", (identifier,)).fetchone()
            if row is None:
                return self._missing(identifier)
            record = _record(row)
            if action == "archive" and not record["archived"]:
                record.update(archived=True, updated_at=datetime.now(timezone.utc).isoformat())
                db.execute("UPDATE capsules SET archived=1, updated_at=? WHERE id=?",
                           (record["updated_at"], identifier))
        prefix = "Archived work checkpoint; it remains available by id.\n" if action == "archive" else ""
        return ToolResult(prefix + _render(record),
                          receipt={"operation": action, "state": "archived" if action == "archive" else "found",
                                   "id": identifier, "capsule": record})

    @staticmethod
    def _missing(identifier: str) -> ToolResult:
        return ToolResult(f"No work checkpoint has id {identifier}.", ok=False,
                          receipt={"state": "not_found", "id": identifier})
