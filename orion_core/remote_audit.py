"""
RemoteAuditLog — a durable record of what paired devices asked ORION to do.

Cloud Roadmap C4 asked for "a durable audit log (who/when/what tool/what
result) surfaced in the Command Centre, not just a log line". Until the
2026-10-03 audit there was only ``RemoteAgentQueue.tasks``: fifty entries in
memory, gone at restart, and covering only ``/v1/agent/tasks`` — not the tools
a phone's chat turn ran through the capability gate, not what was parked for
approval or approved, and not pairing or revocation.

Each line of ``diagnostics/remote_audit.jsonl`` is one event:

    {"at": "...Z", "event": "tool", "device": "6a87e47b", "tool": "open_app",
     "status": "parked", "detail": "", "args": ["app_name"]}

What it deliberately does NOT hold: argument VALUES, chat text or tool output.
The log answers "what did my phone make ORION do, and did it work?"; it must not
become a second copy of every message and every recalled memory. Argument names
are kept because they distinguish ``send`` from ``read`` without the content.

The file is capped and rotated once (``.1``), so an unattended node cannot grow
it without bound. Writes are best effort: an audit that could break the request
it records would be worse than one that occasionally misses a line.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

from .utils import utc_stamp

#: Rotate at this size; one previous file is kept.
MAX_BYTES = 1_000_000

#: Event names, so readers and tests agree on vocabulary.
EVENTS = frozenset({"paired", "pair_failed", "revoked", "token_refused",
                    "agent_task", "tool", "confirm"})


def _default_path() -> Path:
    from .constants import CONFIG_DIR
    return CONFIG_DIR / "diagnostics" / "remote_audit.jsonl"


class RemoteAuditLog:
    """Append-only, size-capped JSON-lines log of remote activity."""

    def __init__(self, path: Path | str | None = None,
                 max_bytes: int = MAX_BYTES) -> None:
        self.path = Path(path) if path else _default_path()
        self.max_bytes = max(4096, int(max_bytes))
        self._lock = threading.Lock()

    # ── writing ───────────────────────────────────────────────────────────────

    def record(self, event: str, *, device: str = "", tool: str = "",
               status: str = "", detail: str = "",
               args: Iterable[str] | dict[str, Any] | None = None) -> None:
        """Append one event. Never raises."""
        entry = {
            "at": utc_stamp(),
            "event": str(event or "")[:32],
            "device": str(device or "")[:8],
            "tool": str(tool or "")[:64],
            "status": str(status or "")[:24],
            "detail": str(detail or "").splitlines()[0][:160] if detail else "",
            "args": sorted(str(k)[:32] for k in (args or ()))[:16],
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(len(line.encode("utf-8")))
                with open(self.path, "a", encoding="utf-8") as stream:
                    stream.write(line)
        except Exception:
            pass

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size + incoming <= self.max_bytes:
            return
        previous = self.path.with_name(self.path.name + ".1")
        try:
            os.replace(self.path, previous)
        except OSError:
            pass

    # ── reading ───────────────────────────────────────────────────────────────

    def recent(self, limit: int = 50, device: str = "") -> list[dict[str, Any]]:
        """The newest *limit* events (newest first), optionally for one device."""
        limit = max(1, int(limit))
        rows: list[dict[str, Any]] = []
        for path in (self.path, self.path.with_name(self.path.name + ".1")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for raw in reversed(lines):
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                if device and not str(row.get("device", "")).startswith(device[:8]):
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    return rows
        return rows

    def render(self, limit: int = 15) -> str:
        """A spoken/printable summary of recent remote activity."""
        rows = self.recent(limit)
        if not rows:
            return "No remote activity recorded yet."
        lines = ["Recent remote activity (newest first):"]
        for row in rows:
            what = row.get("tool") or row.get("event")
            status = row.get("status") or row.get("event")
            device = f" from {row['device']}…" if row.get("device") else ""
            detail = f" — {row['detail']}" if row.get("detail") else ""
            lines.append(f"- {row.get('at', '')}  {what}: {status}{device}{detail}")
        return "\n".join(lines)
