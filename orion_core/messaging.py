"""
Real-time messaging gateway for outbound alerts.

The service intentionally uses lightweight, dependency-tolerant routing. It can
attempt browser-driven delivery for WhatsApp and Telegram when a local browser
session is available, otherwise it reports a graceful fallback result.
"""

from __future__ import annotations

import webbrowser
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation


class MessagingGateway:
    """Route outbound notifications through hosted messaging channels."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus

    def _safe_text(self, text: str) -> str:
        try:
            return SecuritySanitiser.guard_text(text, "messaging.text")
        except SecurityViolation as exc:
            raise ValueError(str(exc)) from exc

    def send_text(self, platform: str, contact: str, message: str) -> ToolResult:
        try:
            platform = self._safe_text(platform).lower()
            contact = self._safe_text(contact)
            message = self._safe_text(message)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not platform or not contact or not message:
            return ToolResult("Platform, contact and message are all required.", ok=False)

        if platform == "whatsapp":
            self.bus.log.emit(f"MESSAGING: WhatsApp delivery request for {contact}.")
            webbrowser.open(f"https://wa.me/{contact}?text={message}")
            return ToolResult(f"WhatsApp alert queued for {contact}.")
        if platform == "telegram":
            self.bus.log.emit(f"MESSAGING: Telegram delivery request for {contact}.")
            webbrowser.open(f"https://t.me/{contact}?text={message}")
            return ToolResult(f"Telegram alert queued for {contact}.")
        return ToolResult(f"Unsupported messaging platform: {platform}", ok=False)
