"""
Shared plain-data structures.

Kept dependency-free (no Qt, no asyncio) so every layer — audio threads,
agents, the dispatcher and the GUI — can exchange results without coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Outcome of a dispatcher tool call (optionally with media and chaining)."""

    text: str
    ok: bool = True
    media: dict[str, Any] | None = None
    chain: list[tuple[str, dict[str, Any]]] | None = None

    def response_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "result": self.text}
