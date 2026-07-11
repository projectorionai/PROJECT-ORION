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
its status/select property and its first date property at call time, so it
works against ordinary user-built Notion databases without configuration
beyond the IDs.  Without a token every method degrades to a clear explanation.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from .bus import OrionBus
from .data import ToolResult
from .utils import first_line

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


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
        self._schema_cache: dict[str, dict[str, str]] = {}

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
                    raise RuntimeError(f"Notion API {response.status}: {message}")
                return data

    # ── schema discovery (title / status / date properties per database) ─────

    async def _database_schema(self, database_id: str) -> dict[str, str]:
        """Return {'title': name, 'status': name, 'status_kind': type, 'date': name}."""
        cached = self._schema_cache.get(database_id)
        if cached is not None:
            return cached
        data = await self._request("GET", f"/databases/{database_id}")
        schema: dict[str, str] = {"title": "", "status": "", "status_kind": "", "date": ""}
        for name, prop in (data.get("properties") or {}).items():
            kind = str(prop.get("type") or "")
            if kind == "title" and not schema["title"]:
                schema["title"] = name
            elif kind in {"status", "select"} and not schema["status"]:
                schema["status"] = name
                schema["status_kind"] = kind
            elif kind == "date" and not schema["date"]:
                schema["date"] = name
        self._schema_cache[database_id] = schema
        return schema

    # ── page parsing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _plain_text(rich: Any) -> str:
        if not isinstance(rich, list):
            return ""
        return "".join(str(part.get("plain_text") or "") for part in rich).strip()

    def _parse_page(self, page: dict[str, Any], schema: dict[str, str]) -> dict[str, str]:
        props = page.get("properties") or {}
        title = ""
        if schema["title"] and schema["title"] in props:
            title = self._plain_text(props[schema["title"]].get("title"))
        status = ""
        if schema["status"] and schema["status"] in props:
            holder = props[schema["status"]].get(schema["status_kind"]) or {}
            status = str(holder.get("name") or "")
        date = ""
        if schema["date"] and schema["date"] in props:
            holder = props[schema["date"]].get("date") or {}
            date = str(holder.get("start") or "")
        return {
            "id": str(page.get("id") or ""),
            "title": title or "(untitled)",
            "status": status,
            "date": date[:16].replace("T", " "),
            "url": str(page.get("url") or ""),
        }

    async def _query_database(
        self, database_id: str, filter_payload: dict[str, Any] | None = None,
        sorts: list[dict[str, Any]] | None = None, limit: int = 20,
    ) -> list[dict[str, str]]:
        schema = await self._database_schema(database_id)
        payload: dict[str, Any] = {"page_size": max(1, min(50, limit))}
        if filter_payload:
            payload["filter"] = filter_payload
        if sorts:
            payload["sorts"] = sorts
        data = await self._request("POST", f"/databases/{database_id}/query", payload)
        return [self._parse_page(page, schema) for page in (data.get("results") or [])]

    # ── tasks ─────────────────────────────────────────────────────────────────

    async def list_tasks(self, limit: int = 12, include_done: bool = False) -> ToolResult:
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        try:
            schema = await self._database_schema(self.tasks_db)
            sorts = (
                [{"property": schema["date"], "direction": "ascending"}]
                if schema["date"] else None
            )
            tasks = await self._query_database(self.tasks_db, sorts=sorts, limit=limit * 2)
            if not include_done:
                tasks = [
                    t for t in tasks
                    if t["status"].lower() not in {"done", "complete", "completed", "archived"}
                ]
            tasks = tasks[:limit]
            self.bus.dashboard_event.emit("tasks", tasks)
            if not tasks:
                return ToolResult("The task board is clear, sir.")
            lines = [f"Open tasks — {len(tasks)}:"]
            for task in tasks:
                bits = [task["title"]]
                if task["status"]:
                    bits.append(f"[{task['status']}]")
                if task["date"]:
                    bits.append(f"due {task['date']}")
                lines.append("- " + "  ".join(bits))
            return ToolResult("\n".join(lines))
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
            schema = await self._database_schema(self.tasks_db)
            if not schema["title"]:
                return ToolResult("The tasks database exposes no title property.", ok=False)
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

    async def complete_task(self, query: str) -> ToolResult:
        """Best-effort completion: find an open task by title fragment, mark done."""
        if not self.available:
            return self._unavailable()
        if not self.tasks_db:
            return ToolResult("No Notion tasks database is configured.", ok=False)
        query = str(query or "").strip().lower()
        if not query:
            return ToolResult("Which task shall I complete? A title fragment is required.", ok=False)
        try:
            schema = await self._database_schema(self.tasks_db)
            if not schema["status"]:
                return ToolResult(
                    "The tasks database has no status/select property, so tasks "
                    "cannot be marked complete programmatically.",
                    ok=False,
                )
            tasks = await self._query_database(self.tasks_db, limit=50)
            match = next(
                (t for t in tasks
                 if query in t["title"].lower()
                 and t["status"].lower() not in {"done", "complete", "completed"}),
                None,
            )
            if match is None:
                return ToolResult(f"No open task matches '{query}'.", ok=False)
            # Discover the database's own "done"-like option name.
            data = await self._request("GET", f"/databases/{self.tasks_db}")
            prop = (data.get("properties") or {}).get(schema["status"]) or {}
            holder = prop.get(schema["status_kind"]) or {}
            options = holder.get("options") or []
            if schema["status_kind"] == "status":
                for group in holder.get("groups") or []:
                    if str(group.get("name") or "").lower() in {"complete", "done"}:
                        ids = set(group.get("option_ids") or [])
                        options = [o for o in options if o.get("id") in ids] or options
                        break
            done_name = next(
                (str(o.get("name")) for o in options
                 if str(o.get("name") or "").lower() in {"done", "complete", "completed"}),
                str(options[-1].get("name")) if options else "Done",
            )
            await self._request("PATCH", f"/pages/{match['id']}", {
                "properties": {
                    schema["status"]: {schema["status_kind"]: {"name": done_name}}
                }
            })
            self.bus.dashboard_event.emit("tasks_dirty", True)
            return ToolResult(f"Marked complete: '{match['title']}' → {done_name}.")
        except Exception as exc:
            return ToolResult(f"Notion task completion failed: {first_line(exc)}", ok=False)

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
                limit=limit,
            )
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
