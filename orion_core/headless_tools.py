"""
HeadlessToolHost — the dispatcher a headless node actually has.

A ``python orion.py --headless`` node built its RemoteGateway with no
dispatcher at all. Two things followed, both found by booting a node and
calling it (audit 2026-10-03):

* ``POST /v1/agent/tasks`` answered **every** request — even the read-only
  tools on the remote whitelist, which need nothing but the node's own memory —
  with ``503 "no dispatcher wired on this node"``.
* With no dispatcher the gateway could not build its gated LocalBrain, so a
  node without a reachable language model answered every chat turn with
  ``502 "no language model or local brain is reachable"`` — while sitting on
  the 88 knowledge entries it had seeded a second earlier.

This host runs the read-only tools a node can honestly serve, through the same
handler code the desktop uses (so the two cannot drift), and answers anything
that needs a screen, apps, mail or files with a plain "that needs the desktop"
rather than "unknown tool". It is deliberately small: the remote whitelist and
hard-deny lists in ``remote.py`` still decide what a phone may ask for.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .data import ToolResult
from .dispatch_knowledge import KnowledgeDispatchMixin
from .dispatch_productivity import ProductivityDispatchMixin
from .utils import first_line


#: The tools this host can run. All read-only.
HEADLESS_TOOLS = frozenset({
    "query_intelligence",
    "recall_conversation",
    "resource_status",
    "token_usage",
    "patch_notes",
    "diagnostics",
})


class HeadlessToolHost(KnowledgeDispatchMixin, ProductivityDispatchMixin):
    """A minimal, read-only dispatcher for a headless node."""

    def __init__(self, bus: Any, memory: Any, *, router: Any | None = None,
                 telemetry: Any | None = None, learning: Any | None = None,
                 changelog: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        # LocalBrain looks for ``dispatcher.learning`` to record a spoken
        # "no, that's wrong" as a correction; the node has a LearningService.
        self.learning = learning
        self.token_ledger = getattr(router, "token_ledger", None)
        self.changelog = changelog
        self._started = time.time()

    def handler_table(self) -> dict[str, Any]:
        return {
            "query_intelligence": self.query_intelligence,
            "recall_conversation": self.recall_conversation,
            "resource_status": self.resource_status_tool,
            "token_usage": self.token_usage_tool,
            "patch_notes": self.patch_notes_tool,
            "diagnostics": self._node_health,
        }

    def can_run(self, name: str) -> bool:
        """Whether this node can run *name* at all (the remote gate asks
        before parking a confirmation the phone could never usefully give)."""
        return str(name or "").strip() in self.handler_table()

    async def dispatch(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        name = str(name or "").strip()
        handler = self.handler_table().get(name)
        if handler is None:
            return ToolResult(
                f"'{name}' needs the desktop — this is the headless node, which has "
                "no screen, apps, mail or files of yours to act on.", ok=False)
        try:
            result = handler(dict(args or {}))
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as exc:
            return ToolResult(f"{name} failed: {first_line(exc, 200)}", ok=False)

    def _node_health(self, args: dict[str, Any]) -> ToolResult:
        """What a node can say about itself without the desktop's full sweep.

        The desktop diagnostic compiles and imports every module and checks
        desktop packages (pyautogui, pywinauto…), all of which a server lacks
        by design — it would report faults that are not faults here.
        """
        lines = [f"Headless node, up {int(time.time() - self._started)} s."]
        health = getattr(self.telemetry, "health", None)
        snapshot = getattr(health, "snapshot", None)
        if callable(snapshot):
            try:
                rows = [row for row in snapshot() if isinstance(row, dict)]
                for row in sorted(rows, key=lambda r: str(r.get("name"))):
                    detail = f" — {row['detail']}" if row.get("detail") else ""
                    lines.append(f"- {row.get('name')}: {row.get('status')}{detail}")
            except Exception:
                pass
        lines.append("Full diagnostics (compile, imports, devices) run on the desktop.")
        return ToolResult("\n".join(lines))
