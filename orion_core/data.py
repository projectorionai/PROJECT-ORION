"""
Shared plain-data structures.

Kept dependency-free (no Qt, no asyncio) so every layer — audio threads,
agents, the dispatcher and the GUI — can exchange results without coupling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Outcome of a dispatcher tool call (optionally with media and chaining)."""

    text: str
    ok: bool = True
    media: dict[str, Any] | None = None
    chain: list[tuple[str, dict[str, Any]]] | None = None
    # Structured evidence trail for callers that gathered it (e.g. a
    # specialist agent's investigate() loop) — deliberately separate from
    # `media`, which has an established narrower contract (raw bytes +
    # mime_type, forwarded to Gemini Live's _send_media()); putting evidence
    # there would risk it being mistaken for a media payload on the voice
    # channel. response_payload() does not serialise this — it never reaches
    # the model, only human-facing surfaces that choose to render it
    # (Mark XX design-spec §6: "surface the toolbelt's evidence trail
    # instead of discarding it").
    evidence: list[dict[str, Any]] | None = None
    # Model-visible typed tool output. Keep it apart from Gemini's raw-media
    # contract and the human evidence trail; text alone loses MCP resources.
    receipt: dict[str, Any] | None = None
    # Host-observed invocation metadata, separate from operation/effect evidence.
    # A returned executor does not establish delivery, saving or completion.
    execution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.receipt is None:
            export = getattr(self.text, "as_tool_receipt", None)
            if callable(export):
                self.receipt = export()
                if self.receipt.get("isError"):
                    self.ok = False
        self.receipt = copy.deepcopy(self.receipt)
        self.execution = copy.deepcopy(self.execution)

    def response_payload(self) -> dict[str, Any]:
        payload = {"ok": self.ok, "result": self.text}
        if self.receipt is not None:
            payload["receipt"] = copy.deepcopy(self.receipt)
        if self.execution is not None:
            payload["execution"] = copy.deepcopy(self.execution)
        return payload
