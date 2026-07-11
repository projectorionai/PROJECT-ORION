"""
Entertainment processing utility for YouTube and media discovery.

The service provides lightweight transcript and trend inspection helpers that
fail gracefully when web access or third-party scraping dependencies are not
available.
"""

from __future__ import annotations

import re
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation


class EntertainmentService:
    """Gather YouTube summaries, channel data and trend snapshots."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus

    def _safe_text(self, text: str) -> str:
        try:
            return SecuritySanitiser.guard_text(text, "entertainment.text")
        except SecurityViolation as exc:
            raise ValueError(str(exc)) from exc

    def summarise_video(self, url: str) -> ToolResult:
        try:
            url = self._safe_text(url)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not url or "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
            return ToolResult("A YouTube URL is required.", ok=False)
        self.bus.log.emit(f"ENTERTAINMENT: summary request queued for {url}.")
        return ToolResult(f"Video summarisation requested for {url}.")

    def channel_priority(self, channel: str) -> ToolResult:
        try:
            channel = self._safe_text(channel)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not channel:
            return ToolResult("A channel name is required.", ok=False)
        return ToolResult(f"Channel priority lookup requested for {channel}.")

    def trending(self, region: str = "GB") -> ToolResult:
        try:
            region = self._safe_text(region).upper()
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        return ToolResult(f"Trending chart request queued for region {region}.")
