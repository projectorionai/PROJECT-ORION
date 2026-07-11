"""
ProactiveIntelligence (Phase 8) — ORION notices things before being asked.

A single low-priority background loop periodically surveys the user's world and
surfaces *actionable* observations:

    • priority email that has gone unanswered;
    • Notion tasks whose due dates are close;
    • today's calendar events / meetings;
    • the active project's git repository (uncommitted work, failing state).

Findings are de-duplicated with a cooldown so ORION never nags: the same
observation is not repeated within ``REPEAT_AFTER_S``.  New findings are pushed
to the Command Centre (``dashboard_event 'proactive'``), logged, and — for
high-salience items — raised as a HUD banner.  Nothing is ever spoken
unprompted, so it cannot interrupt a conversation; ``check_now`` returns the
same findings on demand for the voice channel.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .utils import first_line


@dataclass
class Suggestion:
    key: str            # stable identity for de-duplication
    text: str
    salience: int       # 1 (fyi) .. 3 (urgent)
    at: float


class ProactiveIntelligence:
    DEFAULT_INTERVAL_S = 900.0     # survey every 15 minutes
    FIRST_RUN_DELAY_S = 45.0       # let startup settle first
    REPEAT_AFTER_S = 6 * 3600.0    # don't repeat a finding within 6 hours

    def __init__(
        self,
        bus: OrionBus,
        outlook: Any,
        notion: Any,
        memory: Any,
        telemetry: Any | None = None,
        workspace: Any | None = None,
        interval_s: float | None = None,
    ) -> None:
        self.bus = bus
        self.outlook = outlook
        self.notion = notion
        self.memory = memory
        self.telemetry = telemetry
        self.workspace = workspace
        self.interval_s = float(interval_s or self.DEFAULT_INTERVAL_S)
        self._seen: dict[str, float] = {}
        self._latest: list[Suggestion] = []
        self._stop = asyncio.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Background survey loop; cancel-safe."""
        if self.telemetry is not None:
            self.telemetry.health.register("proactive")
        try:
            await asyncio.sleep(self.FIRST_RUN_DELAY_S)
            while not self._stop.is_set():
                try:
                    await self._survey(announce=True)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.bus.log.emit(f"PROACTIVE: survey recovered - {first_line(exc)}")
                if self.telemetry is not None:
                    self.telemetry.health.beat("proactive", "OK", f"{len(self._latest)} finding(s)")
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()

    # ── on-demand ─────────────────────────────────────────────────────────────

    async def check_now(self) -> ToolResult:
        suggestions = await self._survey(announce=False)
        if not suggestions:
            return ToolResult("Nothing pressing, sir. Your desk is clear.")
        lines = ["Proactive briefing — items for your attention:"]
        for s in sorted(suggestions, key=lambda x: -x.salience):
            mark = "‼" if s.salience >= 3 else "•"
            lines.append(f"{mark} {s.text}")
        return ToolResult("\n".join(lines))

    def latest(self) -> list[dict[str, Any]]:
        return [{"text": s.text, "salience": s.salience} for s in self._latest]

    # ── survey ────────────────────────────────────────────────────────────────

    async def _survey(self, announce: bool) -> list[Suggestion]:
        checks = await asyncio.gather(
            self._check_email(),
            self._check_tasks_and_calendar(),
            self._check_repository(),
            return_exceptions=True,
        )
        suggestions: list[Suggestion] = []
        for result in checks:
            if isinstance(result, list):
                suggestions.extend(result)
        self._latest = suggestions
        # De-duplicate against the cooldown window before announcing.
        fresh: list[Suggestion] = []
        now = time.monotonic()
        for s in suggestions:
            last = self._seen.get(s.key, 0.0)
            if now - last >= self.REPEAT_AFTER_S:
                fresh.append(s)
                self._seen[s.key] = now
        if suggestions:
            self.bus.dashboard_event.emit("proactive", self.latest())
        if announce and fresh:
            for s in fresh:
                self.bus.log.emit(f"PROACTIVE: {s.text}")
            top = max(fresh, key=lambda x: x.salience)
            if top.salience >= 2:
                self.bus.banner.emit(top.text[:120], min(3, top.salience))
            if self.telemetry is not None:
                self.telemetry.metrics.incr("proactive.announcements", len(fresh))
        return suggestions

    async def _check_email(self) -> list[Suggestion]:
        if not getattr(self.outlook, "available", False):
            return []
        try:
            result = await self.outlook.priority_emails(limit=5)
        except Exception:
            return []
        if not result.ok or "No priority email" in result.text:
            return []
        # Count listed priority messages.
        count = result.text.count("\n- ")
        count = max(count, 1)
        return [Suggestion(
            key="email:priority", salience=2,
            text=f"You have {count} priority email(s) awaiting a reply.",
            at=time.monotonic(),
        )]

    async def _check_tasks_and_calendar(self) -> list[Suggestion]:
        if not getattr(self.notion, "available", False):
            return []
        out: list[Suggestion] = []
        try:
            tasks = await self.notion.list_tasks(limit=20)
            for title, due in self._parse_dated_rows(tasks.text):
                hours = self._hours_until(due)
                if hours is None:
                    continue
                if hours < 0:
                    out.append(Suggestion(f"task_overdue:{title}", 3,
                                          f"Task overdue: '{title}' was due {abs(hours):.0f}h ago.",
                                          time.monotonic()))
                elif hours <= 48:
                    sal = 3 if hours <= 12 else 2
                    out.append(Suggestion(f"task_due:{title}", sal,
                                          f"'{title}' is due in {hours:.0f} hours.",
                                          time.monotonic()))
        except Exception:
            pass
        try:
            events = await self.notion.upcoming_events(days=1, limit=10)
            for title, when in self._parse_dated_rows(events.text):
                hours = self._hours_until(when)
                if hours is not None and 0 <= hours <= 24:
                    out.append(Suggestion(f"event:{title}", 2,
                                          f"Today's schedule: '{title}' at {when}.",
                                          time.monotonic()))
        except Exception:
            pass
        return out

    async def _check_repository(self) -> list[Suggestion]:
        root = self._active_repo_root()
        if root is None:
            return []
        try:
            status = await asyncio.to_thread(self._git_status, root)
        except Exception:
            return []
        if status is None:
            return []
        changed, branch = status
        if changed > 0:
            return [Suggestion(
                key=f"git:{root.name}", salience=1,
                text=f"Repository '{root.name}' ({branch}) has {changed} uncommitted change(s).",
                at=time.monotonic(),
            )]
        return []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _active_repo_root(self) -> Optional[Path]:
        candidates = [BASE_DIR]
        try:
            project = getattr(self.memory, "active_project", "")
            if project:
                candidates.insert(0, BASE_DIR)
        except Exception:
            pass
        for root in candidates:
            if (root / ".git").is_dir():
                return root
        return None

    def _git_status(self, root: Path) -> Optional[tuple[int, str]]:
        try:
            porcelain = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=15,
            )
            if porcelain.returncode != 0:
                return None
            changed = len([ln for ln in porcelain.stdout.splitlines() if ln.strip()])
            branch_proc = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=15,
            )
            branch = (branch_proc.stdout or "?").strip() or "?"
            return (changed, branch)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _parse_dated_rows(self, text: str) -> list[tuple[str, str]]:
        """Extract (title, date) pairs from a Notion tool's text output."""
        rows: list[tuple[str, str]] = []
        for line in text.splitlines():
            line = line.strip("- ").strip()
            if not line:
                continue
            due_match = re.search(r"due\s+([0-9]{4}-[0-9]{2}-[0-9]{2}[ T0-9:]*)", line)
            if due_match:
                title = line.split("[")[0].split("  ")[0].strip()
                rows.append((title[:60], due_match.group(1).strip()))
                continue
            date_match = re.match(r"([0-9]{4}-[0-9]{2}-[0-9]{2}[ T0-9:]*)\s*:?\s*(.+)", line)
            if date_match:
                rows.append((date_match.group(2).split("[")[0].strip()[:60],
                             date_match.group(1).strip()))
        return rows

    def _hours_until(self, when: str) -> Optional[float]:
        when = when.strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(when[:len(datetime.now().strftime(fmt))], fmt)
                delta = dt - datetime.now()
                return delta.total_seconds() / 3600.0
            except ValueError:
                continue
        return None
