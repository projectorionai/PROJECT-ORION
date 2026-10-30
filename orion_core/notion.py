"""
Notion integration — task management, calendar control, project tracking and
scheduling through the official Notion REST API.

Configuration (config/api_keys.json → "integrations" → "notion", with
environment-variable overrides):

    {
        "integrations": {
            "notion": {
                "token": "secret_…",                ← or ORION_NOTION_TOKEN
                "tasks_database_id": "…",           ← or ORION_NOTION_TASKS_DB
                "calendar_database_id": "…",        ← or ORION_NOTION_CALENDAR_DB
                "projects_database_id": "…"         ← or ORION_NOTION_PROJECTS_DB
            }
        }
    }

The service is schema-tolerant: it discovers each database's title property,
its completion field and its due-date property at call time, so it
works against ordinary user-built Notion databases without configuration
beyond the IDs.  Without a token every method degrades to a clear explanation.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .data import ToolResult
from .utils import first_line

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
_COMPLETED_LABELS = {"done", "complete", "completed", "finished", "resolved", "closed"}
_NON_COMPLETION_LABELS = {"archived", "cancelled", "canceled"}
_TERMINAL_LABELS = _COMPLETED_LABELS | _NON_COMPLETION_LABELS


class _UnverifiedNotionUpdate(RuntimeError):
    """PATCH was accepted, but the response/readback did not verify its values."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


class NotionService:
    """Async REST bridge to the user's Notion workspace."""

    def __init__(self, bus: OrionBus, config: dict[str, Any] | None = None) -> None:
        self.bus = bus
        config = dict(config or {})
        self.token = (
            os.getenv("ORION_NOTION_TOKEN", "").strip()
            or str(config.get("token") or "").strip()
        )
        self.tasks_db = (
            os.getenv("ORION_NOTION_TASKS_DB", "").strip()
            or str(config.get("tasks_database_id") or "").strip()
        )
        self.calendar_db = (
            os.getenv("ORION_NOTION_CALENDAR_DB", "").strip()
            or str(config.get("calendar_database_id") or "").strip()
        )
        self.projects_db = (
            os.getenv("ORION_NOTION_PROJECTS_DB", "").strip()
            or str(config.get("projects_database_id") or "").strip()
        )
        self._schema_cache: dict[str, dict[str, Any]] = {}
        self.cognition: Any | None = None

    # ── availability ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _unavailable(self) -> ToolResult:
        return ToolResult(
            "Notion integration is not configured. Add an integration token to "
            "config/api_keys.json under integrations.notion.token (or set "
            "ORION_NOTION_TOKEN) plus the relevant database IDs.",
            ok=False,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        timeout = ClientTimeout(total=15.0, connect=5.0)
        async with ClientSession(timeout=timeout) as session:
            async with session.request(
                method, f"{NOTION_API}{path}", headers=self._headers(), json=payload
            ) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    message = str(data.get("message") or data)[:300]
                    hint = {401: " Check the integration token.",
                            403: " Enable the integration's required read/update content capabilities.",
                            404: " Check the database/page ID and share it with this integration."}.get(response.status, "")
                    raise RuntimeError(f"Notion API {response.status}: {message}{hint}")
                if not isinstance(data, dict):
                    raise RuntimeError("Notion returned an invalid response")
                return data

    # ── schema discovery (title / status / date properties per database) ─────

    @staticmethod
    def _label(value: Any) -> str:
        return " ".join(re.findall(r"\w+", str(value or "").casefold()))

    def _invalidate_schema(self, database_id: str) -> None:
        # Refresh at each public operation; a renamed property must not leave
        # this long-running service using yesterday's schema indefinitely.
        self._schema_cache.pop(database_id, None)

    async def _database_schema(self, database_id: str) -> dict[str, Any]:
        """Discover task fields and the board's actual completion options."""
        cached = self._schema_cache.get(database_id)
        if cached is not None:
            return cached
        data = await self._request("GET", f"/databases/{database_id}")
        schema: dict[str, Any] = {"title": "", "status": "", "status_kind": "", "date": "",
                                  "status_options": [], "done_options": [], "terminal_options": []}
        properties = data.get("properties") or {}
        candidates = []
        dates = []
        for name, prop in properties.items():
            kind = str(prop.get("type") or "")
            if kind == "title" and not schema["title"]:
                schema["title"] = name
            elif kind in {"status", "select", "checkbox"}:
                options = (prop.get(kind) or {}).get("options", []) if kind != "checkbox" else []
                label = self._label(name)
                task_field = label in {"status", "task status", "progress", "done", "complete", "completed"}
                if kind == "status" or task_field or (kind == "select" and any(
                        self._label(o.get("name")) in _TERMINAL_LABELS for o in options)):
                    if kind == "status":
                        rank = 0 if task_field else 1
                    elif kind == "checkbox" and label in {"done", "complete", "completed"}:
                        # A board can retain a stale Progress select beside
                        # its authoritative Complete checkbox. Prefer that
                        # explicit completion field to the generic select.
                        rank = 2
                    else:
                        rank = 3 if task_field else 4
                    candidates.append((rank, name, kind))
            elif kind == "date":
                preferred = ["due", "due date", "deadline", "date", "start", "start date"]
                label = self._label(name)
                dates.append((preferred.index(label) if label in preferred else len(preferred), name))
        if dates:
            schema["date"] = min(dates, key=lambda item: item[0])[1]
        if candidates:
            _, name, kind = min(candidates, key=lambda item: item[0])
            schema["status"], schema["status_kind"] = name, kind
            holder = properties[name].get(kind) or {}
            if kind != "checkbox":
                options = holder.get("options") or []
                schema["status_options"] = [str(o["name"]) for o in options if o.get("name")]
                complete_ids = {option_id for group in holder.get("groups") or []
                                if self._label(group.get("name")) in _COMPLETED_LABELS
                                for option_id in group.get("option_ids") or []}
                terminal_ids = {option_id for group in holder.get("groups") or []
                                if self._label(group.get("name")) in _TERMINAL_LABELS
                                for option_id in group.get("option_ids") or []}
                schema["done_options"] = [str(o["name"]) for o in options if o.get("name")
                    and self._label(o["name"]) not in _NON_COMPLETION_LABELS and (
                        o.get("id") in complete_ids or self._label(o["name"]) in _COMPLETED_LABELS)]
                schema["terminal_options"] = [str(o["name"]) for o in options if o.get("name") and (
                    o.get("id") in terminal_ids or self._label(o["name"]) in _TERMINAL_LABELS)]
        self._schema_cache[database_id] = schema
        return schema

    # ── page parsing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _plain_text(rich: Any) -> str:
        if not isinstance(rich, list):
            return ""
        return "".join(str(part.get("plain_text") or (part.get("text") or {}).get("content") or "")
                       for part in rich if isinstance(part, dict)).strip()

    def _parse_page(self, page: dict[str, Any], schema: dict[str, Any]) -> dict[str, str]:
        props = page.get("properties") or {}
        title = ""
        if schema["title"] and schema["title"] in props:
            title = self._plain_text(props[schema["title"]].get("title"))
        status = ""
        if schema["status"] and schema["status"] in props:
            value = props[schema["status"]].get(schema["status_kind"])
            status = ("Done" if value is True else "Not started") if schema["status_kind"] == "checkbox" else str((value or {}).get("name") or "")
        date = end = ""
        if schema["date"] and schema["date"] in props:
            holder = props[schema["date"]].get("date") or {}
            date = str(holder.get("start") or "")
            # A range ("31 Aug - 14 Sep") is due when it ENDS (owner_tasks).
            end = str(holder.get("end") or "")
        return {
            "id": str(page.get("id") or ""),
            "title": title or "(untitled)",
            "status": status,
            "date": date[:16].replace("T", " "),
            "date_raw": date,
            "date_end": end,
            "url": str(page.get("url") or ""),
        }

    def _task_is_done(self, task: dict[str, Any], schema: dict[str, Any]) -> bool:
        status = str(task.get("status") or "")
        if self._label(status) in _NON_COMPLETION_LABELS:
            return False
        return self._label(status) in _COMPLETED_LABELS or status.casefold() in {
            str(name).casefold() for name in schema.get("done_options", [])}

    def _task_is_terminal(self, task: dict[str, Any], schema: dict[str, Any]) -> bool:
        status = str(task.get("status") or "")
        return self._task_is_done(task, schema) or self._label(status) in _TERMINAL_LABELS or status.casefold() in {
            str(name).casefold() for name in schema.get("terminal_options", [])}

    def _dismissed_task_predicate(self) -> Any:
        predicate = getattr(getattr(self, "cognition", None), "is_dismissed", None)
        return predicate if callable(predicate) else None

    def _visible_tasks(self, rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
        dismissed = self._dismissed_task_predicate()
        if dismissed is not None:
            rows = [row for row in rows if not dismissed(row["title"])]
        return rows if limit is None else rows[:max(1, int(limit))]

    def _calendar_uses_tasks(self) -> bool:
        if not self.calendar_db:
            return True

        def identity(value: str) -> str:
            compact = value.replace("-", "")
            return compact.casefold() if re.fullmatch(r"[0-9a-fA-F]{32}", compact) else value

        return bool(self.tasks_db) and identity(self.calendar_db) == identity(self.tasks_db)

    async def _query_database(
        self, database_id: str, filter_payload: dict[str, Any] | None = None,
        sorts: list[dict[str, Any]] | None = None, limit: int | None = 20,
        open_only: bool = False,
    ) -> list[dict[str, str]]:
        schema = await self._database_schema(database_id)
        if not schema["title"]:
            raise RuntimeError("The Notion database exposes no title property; its rows could not be read")
        target = max(1, int(limit)) if limit is not None else None
        payload: dict[str, Any] = {"page_size": 100 if open_only else min(100, target or 100)}
        if filter_payload:
            payload["filter"] = filter_payload
        if sorts:
            payload["sorts"] = sorts
        rows = []
        seen_cursors = set()
        for _ in range(100):
            data = await self._request("POST", f"/databases/{database_id}/query", dict(payload))
            if not isinstance(data.get("results"), list):
                raise RuntimeError("Notion returned an invalid query response; the board could not be read")
            for page in data.get("results") or []:
                if page.get("archived") or page.get("in_trash"):
                    continue
                row = self._parse_page(page, schema)
                if not open_only or not self._task_is_terminal(row, schema):
                    rows.append(row)
            if target is not None and len(rows) >= target:
                return rows[:target]
            if not data.get("has_more"):
                return rows
            cursor = data.get("next_cursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                raise RuntimeError("Notion returned an invalid pagination cursor")
            seen_cursors.add(cursor)
            payload["start_cursor"] = cursor
        raise RuntimeError("Notion task query exceeded the pagination limit; the board is not fully read")

    # ── tasks ─────────────────────────────────────────────────────────────────

    async def open_tasks(self, limit: int | None = 12, include_done: bool = False) -> list[dict[str, str]]:
        """The board's tasks as rows (title/status/date/id). Raises when the
        board cannot be read, so callers can tell "clear" from "unknown"."""
        if not self.available:
            raise RuntimeError("Notion is not configured")
        if not self.tasks_db:
            raise RuntimeError("No Notion tasks database is configured")
        self._invalidate_schema(self.tasks_db)
        schema = await self._database_schema(self.tasks_db)
        sorts = (
            [{"property": schema["date"], "direction": "ascending"}]
            if schema["date"] else None
        )
        local_completion = not include_done and self._dismissed_task_predicate() is not None
        past_events = (not include_done and not schema["status"] and bool(schema["date"])
                       and self._calendar_uses_tasks())
        rows = await self._query_database(self.tasks_db, sorts=sorts,
            limit=None if (local_completion or past_events) else limit, open_only=not include_done)
        if past_events:
            # 25 Oct 2026: the owner's task board IS his calendar. With no
            # completion field nothing on it can ever be "done", so every past
            # entry - a finished trip, last month's move - was read out as an
            # overdue task that no tool could clear. A dated row in a calendar
            # without a done field is an event; once it has ended it is history.
            from .owner_tasks import is_overdue
            rows = [r for r in rows if not r.get("date_raw")
                    or not is_overdue(r["date_raw"], end=r.get("date_end"))]
            if not local_completion:
                return rows[:max(1, int(limit))] if limit is not None else rows
        return self._visible_tasks(rows, limit) if local_completion else rows

    @staticmethod
    def format_tasks(tasks: list[dict[str, str]], *, include_done: bool = False) -> str:
        if not tasks:
            return "The task board is clear."
        lines = [f"{'Tasks' if include_done else 'Open tasks'} — {len(tasks)}:"]
        for task in tasks:
            bits = [task["title"]]
            if task.get("status"):
                bits.append(f"[{task['status']}]")
            if task.get("date"):
                bits.append(f"due {task['date']}")
            lines.append("- " + "  ".join(bits))
        return "\n".join(lines)

    async def list_tasks(self, limit: int = 12, include_done: bool = False) -> ToolResult:
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        try:
            tasks = await self.open_tasks(limit=limit, include_done=include_done)
            self.bus.dashboard_event.emit("tasks", tasks)
            return ToolResult(self.format_tasks(tasks, include_done=include_done))
        except Exception as exc:
            return ToolResult(f"Notion task query failed: {first_line(exc)}", ok=False)

    async def create_task(self, title: str, due: str = "", notes: str = "") -> ToolResult:
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        title = str(title or "").strip()
        if not title:
            return ToolResult("A task title is required.", ok=False)
        try:
            self._invalidate_schema(self.tasks_db)
            schema = await self._database_schema(self.tasks_db)
            if not schema["title"]:
                return ToolResult("The tasks database exposes no title property.", ok=False)
            if due.strip() and not schema["date"]:
                return ToolResult("The tasks database has no due-date property; no task was created.", ok=False)
            properties: dict[str, Any] = {
                schema["title"]: {"title": [{"text": {"content": title[:200]}}]}
            }
            if due.strip() and schema["date"]:
                properties[schema["date"]] = {"date": {"start": due.strip()[:25]}}
            payload: dict[str, Any] = {
                "parent": {"database_id": self.tasks_db},
                "properties": properties,
            }
            if notes.strip():
                payload["children"] = [{
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": notes.strip()[:1800]}}]},
                }]
            await self._request("POST", "/pages", payload)
            self.bus.dashboard_event.emit("tasks_dirty", True)
            suffix = f", due {due.strip()}" if due.strip() else ""
            return ToolResult(f"Task created in Notion: '{title}'{suffix}.")
        except Exception as exc:
            return ToolResult(f"Notion task creation failed: {first_line(exc)}", ok=False)

    #: How a task with no title is asked for: the board shows "(untitled)".
    _UNTITLED_ASKS = frozenset({"(untitled)", "untitled", "untitled task", "untitled tasks",
                                "the untitled task", "the untitled tasks", "no title", "blank task",
                                "blank tasks", "empty task", "empty tasks"})

    @classmethod
    def _asks_for_untitled(cls, query: str) -> bool:
        return " ".join(str(query or "").casefold().split()).strip(" .") in cls._UNTITLED_ASKS

    async def _untitled_tasks(self, schema: dict[str, Any]) -> list[dict[str, str]]:
        """Every task whose title is empty. 28 Oct 2026: "clear the tasks in
        Notion" found two untitled rows, and searching the board for the text
        "(untitled)" - what the board shows - matched neither."""
        if not schema["title"]:
            raise RuntimeError("The tasks database exposes no title property")
        rows = await self._query_database(self.tasks_db, {
            "property": schema["title"], "title": {"is_empty": True},
        }, limit=None)
        return [row for row in rows if row["title"] == "(untitled)"]

    async def _find_task(self, query: str, schema: dict[str, Any]) -> dict[str, str]:
        if not schema["title"]:
            raise RuntimeError("The tasks database exposes no title property")
        tasks = await self._query_database(self.tasks_db, {
            "property": schema["title"], "title": {"contains": query},
        }, limit=None)
        matches = [task for task in tasks if query.casefold() in task["title"].casefold()]
        exact = [task for task in matches if task["title"].casefold() == query.casefold()]
        matches = exact or matches
        if not matches:
            raise RuntimeError(f"No Notion task matches '{query}'")
        if len(matches) != 1:
            raise RuntimeError(f"Multiple Notion tasks match '{query}'; use a unique full title")
        return matches[0]

    @staticmethod
    def _same_date(actual: str, expected: str) -> bool:
        if actual == expected:
            return True
        try:
            return datetime.fromisoformat(actual.replace("Z", "+00:00")) == datetime.fromisoformat(
                expected.replace("Z", "+00:00"))
        except ValueError:
            return False

    async def _verified_update(
        self, task: dict[str, str], schema: dict[str, Any], properties: dict[str, Any],
    ) -> dict[str, str]:
        page = await self._request("PATCH", f"/pages/{task['id']}", {"properties": properties})
        try:
            return await self._confirm_update(task, schema, properties, page)
        except Exception as exc:
            receipt: dict[str, Any] = {"page_id": task["id"], "original_task": task["title"],
                "requested_task": task["title"], "board_write_accepted": True,
                "verification_pending": True}
            for name, value in properties.items():
                kind, expected = next(iter(value.items()))
                if kind == "title":
                    receipt["requested_task"] = self._plain_text(expected)
                elif kind == "date":
                    receipt["requested_date"] = expected["start"] if expected else ""
                elif name == schema["status"]:
                    receipt["requested_status"] = ("Done" if expected else "Not started") if kind == "checkbox" else expected["name"]
            raise _UnverifiedNotionUpdate(
                "Notion accepted the task update, but its final values could not be verified: "
                + first_line(exc) + ". Refresh this page before retrying.", receipt) from exc

    async def _confirm_update(
        self, task: dict[str, str], schema: dict[str, Any], properties: dict[str, Any], page: dict[str, Any],
    ) -> dict[str, str]:
        if not all(name in (page.get("properties") or {}) for name in properties):
            page = await self._request("GET", f"/pages/{task['id']}")
        if str(page.get("id") or "") != task["id"] or page.get("archived") or page.get("in_trash"):
            raise RuntimeError("Notion did not confirm the updated task page")
        actual = page.get("properties") or {}
        for name, value in properties.items():
            if name not in actual:
                raise RuntimeError(f"Notion did not return the updated '{name}' property; refresh the board before retrying")
            kind, expected = next(iter(value.items()))
            received = (actual.get(name) or {}).get(kind)
            if kind == "title":
                confirmed = self._plain_text(received) == self._plain_text(expected)
            elif kind == "date":
                confirmed = received is None if expected is None else isinstance(received, dict) and self._same_date(
                    str(received.get("start") or ""), expected["start"])
            elif kind == "checkbox":
                confirmed = received is expected
            else:
                confirmed = isinstance(received, dict) and received.get("name") == expected["name"]
            if not confirmed:
                raise RuntimeError(f"Notion did not confirm the requested value for '{name}'; refresh the board before retrying")
        # A sparse write response can omit unchanged title/date fields. Retain
        # their last observed values instead of recording an untitled task.
        merged = dict(task)
        parsed = self._parse_page(page, schema)
        for field in ("title", "status", "date"):
            if schema.get(field) in actual:
                merged[field] = parsed[field]
                if field == "date":
                    merged["date_raw"] = parsed["date_raw"]
        merged["url"] = parsed["url"] or task.get("url", "")
        return merged

    def _task_receipt(self, original: dict[str, str], updated: dict[str, str], schema: dict[str, Any]) -> dict[str, Any]:
        return {"original_task": original["title"], "task": updated["title"],
                "page_id": updated["id"], "status": updated.get("status", ""),
                "date": updated.get("date_raw", updated.get("date", "")), "url": updated.get("url", ""),
                "completed": self._task_is_done(updated, schema),
                "terminal": self._task_is_terminal(updated, schema)}

    async def complete_task(self, query: str) -> ToolResult:
        """Complete one unambiguous task using its actual completion field."""
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        query = str(query or "").strip()
        if not query:
            return ToolResult("Which task shall I complete? A title fragment is required.", ok=False)
        try:
            self._invalidate_schema(self.tasks_db)
            schema = await self._database_schema(self.tasks_db)
            if not schema["status"]:
                return ToolResult(
                    "The tasks database has no task status or completion checkbox, so tasks "
                    "cannot be marked complete programmatically.",
                    ok=False,
                )
            if self._asks_for_untitled(query):
                return await self._complete_untitled(schema)
            match = await self._find_task(query, schema)
            if self._task_is_done(match, schema):
                return ToolResult(f"Already complete in Notion: '{match['title']}'.",
                                  receipt=self._task_receipt(match, match, schema))
            if self._task_is_terminal(match, schema):
                return ToolResult(f"Notion task '{match['title']}' is '{match['status']}', "
                                  "which is not completed; no task was changed.", ok=False,
                                  receipt=self._task_receipt(match, match, schema))
            if schema["status_kind"] == "checkbox":
                value: Any = True
                done_name = "Done"
            else:
                options = schema.get("done_options") or []
                done_name = next((name for name in options if self._label(name) in {"done", "complete", "completed"}), "")
                if not done_name and len(options) == 1:
                    done_name = options[0]
                if not done_name:
                    return ToolResult("The tasks database has no unambiguous completion option; no task was changed.", ok=False)
                value = {"name": done_name}
            updated = await self._verified_update(match, schema, {
                schema["status"]: {schema["status_kind"]: value}})
            self.bus.dashboard_event.emit("tasks_dirty", True)
            # The exact board title, so callers can remember THIS task as
            # done even when they were given only a fragment of its name.
            return ToolResult(f"Marked complete: '{match['title']}' → {done_name}.",
                              receipt=self._task_receipt(match, updated, schema))
        except _UnverifiedNotionUpdate as exc:
            return ToolResult(str(exc), ok=False, receipt=exc.receipt)
        except Exception as exc:
            return ToolResult(f"Notion task completion failed: {first_line(exc)}", ok=False)

    async def _complete_untitled(self, schema: dict[str, Any]) -> ToolResult:
        """Complete every open untitled task: they cannot be told apart, and
        asking for "the untitled one" means all of them."""
        tasks = [task for task in await self._untitled_tasks(schema)
                 if not self._task_is_done(task, schema) and not self._task_is_terminal(task, schema)]
        if not tasks:
            return ToolResult("No open untitled task is on the Notion board.", ok=False)
        if schema["status_kind"] == "checkbox":
            value: Any = True
        else:
            options = schema.get("done_options") or []
            done_name = next((name for name in options
                              if self._label(name) in {"done", "complete", "completed"}), "")
            if not done_name and len(options) == 1:
                done_name = options[0]
            if not done_name:
                return ToolResult("The tasks database has no unambiguous completion option; "
                                  "no task was changed.", ok=False)
            value = {"name": done_name}
        receipt: dict[str, Any] = {}
        for task in tasks:
            updated = await self._verified_update(task, schema, {
                schema["status"]: {schema["status_kind"]: value}})
            receipt = self._task_receipt(task, updated, schema)
        self.bus.dashboard_event.emit("tasks_dirty", True)
        count = len(tasks)
        return ToolResult(f"Marked {count} untitled Notion task{'s' if count != 1 else ''} complete.",
                          receipt=receipt)

    async def update_task(self, query: str, *, title: str = "", due: str | None = None,
                          status: str = "") -> ToolResult:
        """Rename, reschedule, or change the status of one existing task."""
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        query = str(query or "").strip()
        title, status = str(title or "").strip(), str(status or "").strip()
        if not query or (not title and due is None and not status):
            return ToolResult("A task title to find and at least one title, due date, or status change are required.", ok=False)
        try:
            self._invalidate_schema(self.tasks_db)
            schema = await self._database_schema(self.tasks_db)
            if not schema["title"]:
                return ToolResult("The tasks database exposes no title property; no task was changed.", ok=False)
            properties: dict[str, Any] = {}
            if title:
                if len(title) > 2000:
                    return ToolResult("The new task title exceeds Notion's 2,000-character limit.", ok=False)
                properties[schema["title"]] = {"title": [{"text": {"content": title}}]}
            if due is not None:
                if not schema["date"]:
                    return ToolResult("The tasks database has no due-date property; no task was changed.", ok=False)
                clean_due = str(due).strip()
                properties[schema["date"]] = {"date": {"start": clean_due} if clean_due else None}
            if status:
                kind = schema["status_kind"]
                if not schema["status"]:
                    return ToolResult("The tasks database has no editable task status; no task was changed.", ok=False)
                if kind == "checkbox":
                    label = self._label(status)
                    if label not in _COMPLETED_LABELS | {"open", "not started", "to do", "todo", "false", "true"}:
                        return ToolResult("The completion checkbox accepts done or open; no task was changed.", ok=False)
                    value: Any = label in _COMPLETED_LABELS | {"true"}
                else:
                    option = next((name for name in schema.get("status_options", []) if name.casefold() == status.casefold()), "")
                    if not option:
                        return ToolResult(f"'{status}' is not an existing task status; no task was changed.", ok=False)
                    value = {"name": option}
                properties[schema["status"]] = {kind: value}
            match = await self._find_task(query, schema)
            updated = await self._verified_update(match, schema, properties)
            self.bus.dashboard_event.emit("tasks_dirty", True)
            return ToolResult(f"Updated Notion task: '{updated['title']}'.",
                              receipt=self._task_receipt(match, updated, schema))
        except _UnverifiedNotionUpdate as exc:
            return ToolResult(str(exc), ok=False, receipt=exc.receipt)
        except Exception as exc:
            return ToolResult(f"Notion task update failed: {first_line(exc)}", ok=False)

    async def archive_task_exact(self, title: str) -> ToolResult:
        """Archive one exact task, refusing ambiguous or partial matches."""
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        clean = str(title or "").strip()
        if not clean:
            return ToolResult("An exact task title is required.", ok=False)
        try:
            self._invalidate_schema(self.tasks_db)
            schema = await self._database_schema(self.tasks_db)
            if not schema["title"]:
                return ToolResult("The tasks database exposes no title property.", ok=False)
            if self._asks_for_untitled(clean):
                untitled = await self._untitled_tasks(schema)
                if not untitled:
                    return ToolResult("No untitled task is on the Notion board.", ok=False)
                for task in untitled:
                    page = await self._request("PATCH", f"/pages/{task['id']}", {"archived": True})
                    if page.get("archived") is not True and page.get("in_trash") is not True:
                        return ToolResult("Notion did not confirm that the untitled tasks were "
                                          "archived.", ok=False)
                self.bus.dashboard_event.emit("tasks_dirty", True)
                count = len(untitled)
                return ToolResult(f"Archived {count} untitled Notion task{'s' if count != 1 else ''}.")
            rows = await self._query_database(self.tasks_db, {
                "property": schema["title"], "title": {"equals": clean},
            }, limit=50)
            matches = [r for r in rows if r["title"].casefold() == clean.casefold()]
            if not matches:
                return ToolResult(f"No active Notion task has the exact title '{clean}'.", ok=False)
            if len(matches) != 1:
                return ToolResult(
                    f"Multiple Notion tasks have the title '{clean}'; no page was changed.",
                    ok=False,
                )
            page = await self._request("PATCH", f"/pages/{matches[0]['id']}", {"archived": True})
            if page.get("archived") is not True and page.get("in_trash") is not True:
                return ToolResult("Notion did not confirm that the task was archived.", ok=False)
            self.bus.dashboard_event.emit("tasks_dirty", True)
            return ToolResult(f"Archived Notion task: '{clean}'.")
        except Exception as exc:
            return ToolResult(f"Notion task archival failed: {first_line(exc)}", ok=False)

    async def archive_overdue_tasks_exact(self, title: str) -> ToolResult:
        """Retire all past-due copies of an explicitly supplied exact title."""
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        clean = str(title or "").strip()
        if not clean:
            return ToolResult("An exact task title is required.", ok=False)
        archived = 0
        try:
            self._invalidate_schema(self.tasks_db)
            schema = await self._database_schema(self.tasks_db)
            if not schema["title"] or not schema["date"]:
                return ToolResult("Title and due-date properties are required for retirement.", ok=False)
            cutoff = datetime.now().date().isoformat()
            payload: dict[str, Any] = {"page_size": 100, "filter": {"and": [
                {"property": schema["title"], "title": {"equals": clean}},
                {"property": schema["date"], "date": {"before": cutoff}},
            ]}}
            pages = []
            # Collect first: changing the database whilst paging can skip rows.
            for _ in range(100):
                data = await self._request("POST", f"/databases/{self.tasks_db}/query", payload)
                pages.extend(page for page in data.get("results", [])
                             if self._parse_page(page, schema)["title"].casefold() == clean.casefold())
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
                if not cursor or cursor == payload.get("start_cursor"):
                    raise RuntimeError("Notion returned an invalid pagination cursor")
                payload["start_cursor"] = cursor
            else:
                raise RuntimeError("Task query exceeded the retirement page limit")
            for page in pages:
                response = await self._request("PATCH", f"/pages/{page['id']}", {"archived": True})
                if response.get("archived") is not True and response.get("in_trash") is not True:
                    raise RuntimeError("Notion did not confirm archival")
                archived += 1
            self.bus.dashboard_event.emit("tasks_dirty", True)
            return ToolResult(f"Archived {archived} overdue Notion task(s) titled '{clean}'.")
        except Exception as exc:
            return ToolResult(f"Retirement stopped after {archived} task(s): {first_line(exc)}", ok=False)

    # ── calendar / scheduling ─────────────────────────────────────────────────

    async def upcoming_events(self, days: int = 7, limit: int = 10) -> ToolResult:
        """Events in the calendar database (falls back to dated tasks)."""
        if not self.available:
            return self._unavailable()
        database_id = self.calendar_db or self.tasks_db
        if not database_id:
            return ToolResult("No Notion calendar or tasks database is configured.", ok=False)
        days = max(1, min(60, int(days or 7)))
        try:
            self._invalidate_schema(database_id)
            schema = await self._database_schema(database_id)
            if not schema["date"]:
                return ToolResult("The configured database has no date property.", ok=False)
            today = datetime.now().date()
            filter_payload = {
                "and": [
                    {"property": schema["date"],
                     "date": {"on_or_after": today.isoformat()}},
                    {"property": schema["date"],
                     "date": {"on_or_before": (today + timedelta(days=days)).isoformat()}},
                ]
            }
            events = await self._query_database(
                database_id, filter_payload=filter_payload,
                sorts=[{"property": schema["date"], "direction": "ascending"}],
                limit=None if self._calendar_uses_tasks() and self._dismissed_task_predicate() is not None else limit,
                **({"open_only": True} if self._calendar_uses_tasks() else {}),
            )
            if self._calendar_uses_tasks():
                events = self._visible_tasks(events, limit)
            self.bus.dashboard_event.emit("events", events)
            if not events:
                return ToolResult(f"The calendar is clear for the next {days} day(s).")
            lines = [f"Schedule — next {days} day(s), {len(events)} item(s):"]
            for event in events:
                status = f" [{event['status']}]" if event["status"] else ""
                lines.append(f"- {event['date'] or 'undated'}: {event['title']}{status}")
            return ToolResult("\n".join(lines))
        except Exception as exc:
            return ToolResult(f"Notion calendar query failed: {first_line(exc)}", ok=False)

    async def events_on(self, day: Any, limit: int = 10) -> list[dict[str, Any]] | None:
        """The calendar entries dated *day* (the owner's local date) as
        structured rows, at most *limit* + 1 so a caller can tell a capped
        list from a complete one; None when the calendar cannot be read
        (unavailable, unconfigured or failed), which is not the same as []
        - a clear day.

        For "today" questions. upcoming_events(days=1) spans today AND
        tomorrow (both bounds inclusive) and returns rendered text, and the
        greeting counted that text as today's entries - a tomorrow-only
        appointment became "1 calendar entry today" (audit 6 Oct 2026, A05).
        """
        if not self.available:
            return None
        database_id = self.calendar_db or self.tasks_db
        if not database_id:
            return None
        iso = day.isoformat()
        try:
            self._invalidate_schema(database_id)
            schema = await self._database_schema(database_id)
            if not schema["date"]:
                return None
            events = await self._query_database(
                database_id,
                filter_payload={"and": [
                    {"property": schema["date"], "date": {"on_or_after": iso}},
                    {"property": schema["date"], "date": {"on_or_before": iso}},
                ]},
                sorts=[{"property": schema["date"], "direction": "ascending"}],
                limit=None if self._calendar_uses_tasks() and self._dismissed_task_predicate() is not None else max(1, int(limit)) + 1,
                **({"open_only": True} if self._calendar_uses_tasks() else {}),
            )
            if self._calendar_uses_tasks():
                events = self._visible_tasks(events, max(1, int(limit)) + 1)
        except Exception:
            return None
        # Timed entries carry their own date ("2026-10-06T09:00:00+01:00"):
        # kept only when that date is the day asked about.
        return [e for e in events or [] if str(e.get("date") or "")[:10] == iso]

    async def create_event(self, title: str, start: str, end: str = "") -> ToolResult:
        """Schedule an entry in the calendar database (or dated task fallback)."""
        if not self.available:
            return self._unavailable()
        database_id = self.calendar_db or self.tasks_db
        if not database_id:
            return ToolResult("No Notion calendar or tasks database is configured.", ok=False)
        title = str(title or "").strip()
        start = str(start or "").strip()
        if not title or not start:
            return ToolResult("An event title and start date/time are required.", ok=False)
        try:
            self._invalidate_schema(database_id)
            schema = await self._database_schema(database_id)
            if not schema["title"] or not schema["date"]:
                return ToolResult(
                    "The configured database lacks a title or date property.", ok=False
                )
            date_payload: dict[str, Any] = {"start": start[:25]}
            if end.strip():
                date_payload["end"] = end.strip()[:25]
            await self._request("POST", "/pages", {
                "parent": {"database_id": database_id},
                "properties": {
                    schema["title"]: {"title": [{"text": {"content": title[:200]}}]},
                    schema["date"]: {"date": date_payload},
                },
            })
            self.bus.dashboard_event.emit("tasks_dirty", True)
            return ToolResult(f"Scheduled: '{title}' at {start}.")
        except Exception as exc:
            return ToolResult(f"Notion scheduling failed: {first_line(exc)}", ok=False)

    # ── projects ──────────────────────────────────────────────────────────────

    async def project_overview(self, limit: int = 10) -> ToolResult:
        if not self.available:
            return self._unavailable()
        if not self.projects_db:
            return ToolResult("No Notion projects database is configured.", ok=False)
        try:
            self._invalidate_schema(self.projects_db)
            projects = await self._query_database(self.projects_db, limit=limit)
            if not projects:
                return ToolResult("The projects database is empty.")
            lines = [f"Project tracker — {len(projects)} project(s):"]
            for project in projects:
                status = f" [{project['status']}]" if project["status"] else ""
                dated = f" (target {project['date']})" if project["date"] else ""
                lines.append(f"- {project['title']}{status}{dated}")
            return ToolResult("\n".join(lines))
        except Exception as exc:
            return ToolResult(f"Notion project query failed: {first_line(exc)}", ok=False)
