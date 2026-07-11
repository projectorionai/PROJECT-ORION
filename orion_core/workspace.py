"""
WorkspaceManager (Phase 5) — persistent workspace awareness so ORION can
resume work exactly where it left off.

A *snapshot* captures the live state of the machine that matters for resuming:

    • open application windows (title, position, size, monitor, minimised)
    • the active/foreground window
    • browser tabs (best-effort, from window titles)
    • open documents (Office/editor windows, resolved to paths where possible)
    • the active project (from MemoryAgent)
    • development sessions (repositories detected among open editor windows)

Snapshots persist through the MemoryAgent's WORKSPACE tier (JSON in SQLite),
so ``restore_workspace_state`` can re-open applications and re-focus windows,
and ``resume_context`` (memory) can brief ORION on where things stood.

``track_changes`` diffs the current live workspace against the last snapshot
and returns what opened/closed/moved — the signal the ProactiveIntelligence
monitor uses to notice "you closed the repo you were working in".
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .utils import first_line, utc_stamp


@dataclass
class WindowState:
    title: str
    left: int
    top: int
    width: int
    height: int
    minimised: bool = False
    monitor: int = 0
    process: str = ""

    @property
    def kind(self) -> str:
        t = self.title.lower()
        if any(b in t for b in ("chrome", "edge", "firefox", "mozilla", "opera")):
            return "browser"
        if any(e in t for e in ("visual studio code", "- code", "pycharm", "intellij", "sublime")):
            return "editor"
        if any(o in t for o in (".docx", ".xlsx", ".pptx", "word", "excel", "powerpoint")):
            return "document"
        return "app"


@dataclass
class WorkspaceSnapshot:
    at: str
    active_window: str
    windows: list[WindowState] = field(default_factory=list)
    browser_tabs: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    dev_sessions: list[str] = field(default_factory=list)
    active_project: str = ""
    open_applications: list[str] = field(default_factory=list)
    monitor_layout: list[dict[str, Any]] = field(default_factory=list)
    notion_sessions: list[str] = field(default_factory=list)
    outlook_sessions: list[str] = field(default_factory=list)
    research_environments: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WorkspaceSnapshot":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("workspace snapshot is not an object")
        windows = [WindowState(**w) for w in data.get("windows", [])]
        data["windows"] = windows
        return cls(**data)

    def summary(self) -> str:
        return (
            f"{len(self.windows)} window(s), {len(self.browser_tabs)} browser tab(s), "
            f"{len(self.documents)} document(s), {len(self.dev_sessions)} dev session(s)"
            f", {len(self.monitor_layout)} monitor(s)"
            + (f"; project '{self.active_project}'" if self.active_project else "")
        )


class WorkspaceManager:
    """Snapshot, persist, restore and track the desktop workspace."""

    def __init__(self, bus: OrionBus, memory: MemoryAgent,
                 desktop: Any | None = None, telemetry: Any | None = None,
                 display: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.desktop = desktop
        self.telemetry = telemetry
        self.display = display
        self._last: Optional[WorkspaceSnapshot] = None

    # ── snapshot ──────────────────────────────────────────────────────────────

    async def snapshot_workspace(self) -> WorkspaceSnapshot:
        """Capture the live workspace (window enumeration runs off-thread)."""
        snap = await asyncio.to_thread(self._capture_sync)
        self._last = snap
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("workspace.windows", float(len(snap.windows)))
        return snap

    def _capture_sync(self) -> WorkspaceSnapshot:
        windows = self._enumerate_windows()
        active = self._active_title()
        browser_tabs: list[str] = []
        documents: list[str] = []
        dev_sessions: list[str] = []
        open_applications: list[str] = []
        notion_sessions: list[str] = []
        outlook_sessions: list[str] = []
        research_environments: list[str] = []
        for w in windows:
            app_label = self._app_for_window(w) or w.kind
            if app_label:
                open_applications.append(app_label)
            if w.kind == "browser":
                tab = re.sub(r"\s*[-â€”]\s*(Google Chrome|Microsoft.Edge|Mozilla Firefox|Opera).*$",
                             "", w.title).strip()
                if tab:
                    browser_tabs.append(tab)
                if "notion" in w.title.lower():
                    notion_sessions.append(w.title)
                if any(token in w.title.lower() for token in ("research", "scholar", "arxiv", "paper", "pubmed")):
                    research_environments.append(w.title)
            elif w.kind == "document":
                documents.append(w.title)
            elif w.kind == "editor":
                repo = self._repo_from_editor_title(w.title)
                if repo:
                    dev_sessions.append(repo)
            if "outlook" in w.title.lower():
                outlook_sessions.append(w.title)
        monitor_layout: list[dict[str, Any]] = []
        if self.display is not None:
            try:
                monitor_layout = list((self.display.describe() or {}).get("monitors", []))
            except Exception:
                monitor_layout = []
        return WorkspaceSnapshot(
            at=utc_stamp(),
            active_window=active,
            windows=windows,
            browser_tabs=sorted(set(browser_tabs)),
            documents=sorted(set(documents)),
            dev_sessions=sorted(set(dev_sessions)),
            active_project=self.memory.active_project,
            open_applications=sorted(set(open_applications)),
            monitor_layout=monitor_layout,
            notion_sessions=sorted(set(notion_sessions)),
            outlook_sessions=sorted(set(outlook_sessions)),
            research_environments=sorted(set(research_environments)),
        )

    def _enumerate_windows(self) -> list[WindowState]:
        try:
            import pygetwindow as gw  # type: ignore
        except Exception:
            return []
        out: list[WindowState] = []
        try:
            for w in gw.getAllWindows():
                title = (w.title or "").strip()
                if not title:
                    continue
                try:
                    out.append(WindowState(
                        title=title,
                        left=int(w.left), top=int(w.top),
                        width=int(w.width), height=int(w.height),
                        minimised=bool(getattr(w, "isMinimized", False)),
                    ))
                except Exception:
                    continue
        except Exception:
            return []
        return out[:60]

    def _active_title(self) -> str:
        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                length = user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                return buf.value
            except Exception:
                return ""
        return ""

    def _repo_from_editor_title(self, title: str) -> str:
        # VS Code: "file.py - repo-name - Visual Studio Code"
        parts = [p.strip() for p in re.split(r"\s[-—]\s", title) if p.strip()]
        if len(parts) >= 2:
            return parts[-2] if "code" in parts[-1].lower() else parts[-1]
        return ""

    # ── persistence ───────────────────────────────────────────────────────────

    async def save_workspace_state(self, name: str = "") -> ToolResult:
        snap = await self.snapshot_workspace()
        key = self._slug(name) or "latest"
        # Persist through the WORKSPACE memory tier (off-thread SQLite write).
        await asyncio.to_thread(
            self.memory.remember, MemoryTier.WORKSPACE, key, snap.to_json()
        )
        # Also keep a human-readable pointer under a stable key for resume.
        await asyncio.to_thread(
            self.memory.remember, MemoryTier.WORKSPACE, "latest_summary",
            f"{snap.at} — {snap.summary()}"
        )
        self.bus.log.emit(f"WORKSPACE: saved '{key}' — {snap.summary()}")
        return ToolResult(f"Workspace saved as '{key}': {snap.summary()}.")

    async def restore_workspace_state(self, name: str = "") -> ToolResult:
        key = self._slug(name) or "latest"
        rows = await asyncio.to_thread(
            self.memory.recall, MemoryTier.WORKSPACE, key, "", 50
        )
        record = next((r for r in rows if r.get("key_ref") == key), None)
        if record is None:
            return ToolResult(f"No saved workspace named '{key}'.", ok=False)
        try:
            snap = WorkspaceSnapshot.from_json(record["value"])
        except Exception as exc:
            return ToolResult(f"Saved workspace '{key}' is unreadable: {first_line(exc)}", ok=False)
        if self.desktop is None:
            return ToolResult(
                f"Workspace '{key}' loaded ({snap.summary()}), but no DesktopAgent is "
                "attached to relaunch applications.",
                ok=False,
            )
        reopened: list[str] = []
        # Re-open applications behind the recorded windows (deduped by executable).
        launched: set[str] = set()
        for w in snap.windows:
            target = self._app_for_window(w)
            if target and target not in launched:
                launched.add(target)
                try:
                    self.desktop.open_app(target)
                    reopened.append(target)
                except Exception:
                    continue
        if self.memory.active_project != snap.active_project and snap.active_project:
            self.memory.set_active_project(snap.active_project)
        self.bus.log.emit(f"WORKSPACE: restore '{key}' relaunched {len(reopened)} app(s).")
        return ToolResult(
            f"Restoring workspace '{key}' ({snap.summary()}). "
            f"Relaunched: {', '.join(reopened) or 'nothing to relaunch'}."
        )

    def _app_for_window(self, w: WindowState) -> str:
        kind = w.kind
        if kind == "browser":
            for name in ("chrome", "edge", "firefox"):
                if name in w.title.lower():
                    return name
            return "edge"
        if kind == "editor":
            if "code" in w.title.lower():
                return "code"
            return ""
        if kind == "document":
            for ext, app in ((".docx", "word"), (".xlsx", "excel"), (".pptx", "powerpoint")):
                if ext in w.title.lower():
                    return app
        return ""

    # ── change tracking ───────────────────────────────────────────────────────

    async def track_changes(self) -> ToolResult:
        """Diff the live workspace against the last snapshot."""
        previous = self._last
        current = await self.snapshot_workspace()
        if previous is None:
            return ToolResult("Baseline workspace snapshot captured; no prior state to compare.")
        prev_titles = {w.title for w in previous.windows}
        curr_titles = {w.title for w in current.windows}
        opened = sorted(curr_titles - prev_titles)
        closed = sorted(prev_titles - curr_titles)
        lines = ["Workspace changes since last snapshot:"]
        if opened:
            lines.append("Opened: " + "; ".join(t[:60] for t in opened[:10]))
        if closed:
            lines.append("Closed: " + "; ".join(t[:60] for t in closed[:10]))
        if current.active_window != previous.active_window:
            lines.append(f"Focus moved to: {current.active_window[:70]}")
        if len(lines) == 1:
            lines.append("No material change.")
        return ToolResult("\n".join(lines))

    def last_snapshot(self) -> Optional[WorkspaceSnapshot]:
        return self._last

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(name: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")[:48]


class DesktopMemoryManager(WorkspaceManager):
    """Named workspace restoration commands for coding, marketing and research."""

    NAMED_WORKSPACES = {
        "coding": "coding",
        "marketing": "marketing",
        "research": "research",
    }

    async def save_named_workspace(self, name: str) -> ToolResult:
        key = self.NAMED_WORKSPACES.get(self._slug(name), self._slug(name) or "latest")
        return await self.save_workspace_state(key)

    async def restore_named_workspace(self, name: str) -> ToolResult:
        key = self.NAMED_WORKSPACES.get(self._slug(name), self._slug(name) or "latest")
        return await self.restore_workspace_state(key)

    async def restore_coding_workspace(self) -> ToolResult:
        return await self.restore_named_workspace("coding")

    async def restore_marketing_workspace(self) -> ToolResult:
        return await self.restore_named_workspace("marketing")

    async def restore_research_workspace(self) -> ToolResult:
        return await self.restore_named_workspace("research")
