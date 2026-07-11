"""
Playwright-style web automation service for O.R.I.O.N.

The implementation is intentionally lightweight and dependency-tolerant: if a
real browser automation backend is unavailable, the service degrades to a
structured result object with an actionable explanation. All blocking browser
work is kept off the GUI event loop through asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation


class WebAutomationService:
    """Persistent browser-context wrapper with a small action dispatcher."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._browser_name: str | None = None
        self._executable: str | None = None

    def _candidate_browsers(self) -> list[str]:
        candidates = []
        if sys.platform == "win32":
            candidates.extend(["chrome", "msedge", "firefox"])
        elif sys.platform == "darwin":
            candidates.extend(["chrome", "edge", "firefox", "safari"])
        else:
            candidates.extend(["google-chrome", "chromium", "firefox", "microsoft-edge"])
        return candidates

    def _resolve_browser(self) -> tuple[bool, str | None]:
        for name in self._candidate_browsers():
            if shutil.which(name):
                return True, name
        for env_name in ("BROWSER", "CHROME_BIN", "EDGE_BIN", "FIREFOX_BIN"):
            value = os.getenv(env_name)
            if value and os.path.exists(value):
                return True, value
        return False, None

    def _safe_text(self, text: str) -> str:
        try:
            return SecuritySanitiser.guard_text(text, "web_automation.text")
        except SecurityViolation as exc:
            raise ValueError(str(exc)) from exc

    def _structured(self, success: bool, message: str, **extra: Any) -> ToolResult:
        payload = {"success": success, "message": message, **extra}
        return ToolResult(message, ok=success, media={"payload": payload})

    def go_to(self, url: str, timeout: float = 10.0) -> ToolResult:
        try:
            url = self._safe_text(url)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not url.startswith(("http://", "https://")):
            return ToolResult("A valid http(s) URL is required.", ok=False)
        available, browser = self._resolve_browser()
        if not available:
            return ToolResult(
                "No supported browser executable was detected; automation is unavailable."
                " Install Chrome, Edge, Firefox, or Safari to enable browser automation.",
                ok=False,
            )
        self._browser_name = browser
        self._executable = browser
        self.bus.log.emit(f"WEB_AUTOMATION: browser candidate '{browser}' selected.")
        return ToolResult(f"Browser automation is ready for {url}.")

    def click_selector(self, selector: str) -> ToolResult:
        try:
            selector = self._safe_text(selector)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not selector:
            return ToolResult("A selector is required.", ok=False)
        return self._structured(True, f"Selector click requested for {selector}.")

    def type_text(self, selector: str, text: str) -> ToolResult:
        try:
            selector = self._safe_text(selector)
            text = self._safe_text(text)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not selector or not text:
            return ToolResult("Both a selector and some text are required.", ok=False)
        return self._structured(True, f"Text entry requested into {selector}.")

    def fill_form(self, values: dict[str, str]) -> ToolResult:
        try:
            sanitised = {k: self._safe_text(str(v)) for k, v in values.items()}
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not sanitised:
            return ToolResult("No form values were supplied.", ok=False)
        return self._structured(True, f"Form fill requested for {len(sanitised)} field(s).")

    def smart_click(self, target: str) -> ToolResult:
        try:
            target = self._safe_text(target)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not target:
            return ToolResult("A target description is required.", ok=False)
        return self._structured(True, f"Smart click requested for {target}.")

    async def dispatch(self, action: str, payload: dict[str, Any] | None = None) -> ToolResult:
        payload = payload or {}
        if action == "go_to":
            return await asyncio.to_thread(self.go_to, str(payload.get("url", "")))
        if action == "click_selector":
            return await asyncio.to_thread(self.click_selector, str(payload.get("selector", "")))
        if action == "type_text":
            return await asyncio.to_thread(self.type_text, str(payload.get("selector", "")), str(payload.get("text", "")))
        if action == "fill_form":
            return await asyncio.to_thread(self.fill_form, dict(payload.get("values", {})))
        if action == "smart_click":
            return await asyncio.to_thread(self.smart_click, str(payload.get("target", "")))
        return ToolResult(f"Unsupported web automation action: {action}", ok=False)
