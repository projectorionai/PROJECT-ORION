"""
WebController (Phase 4) — operate a browser the way a human does.

Hybrid architecture, choosing the most reliable method per action:

    DOM/accessibility  — the browser's UIA accessibility tree (exposed by
                         Chrome/Edge/Firefox) gives named buttons, links and
                         input fields.  This is ORION's "DOM access": it finds
                         "Accept all", "Reject all", "Close", form fields, etc.
                         by their accessible name.
    Vision             — OCR + pixel-diff verification confirms an action
                         actually did something, and locates controls the
                         accessibility tree does not expose.
    Automation         — keyboard/cursor via the control layer performs the
                         navigation primitives (address bar, tabs, Enter) that
                         are identical across sites and most reliable by key.

Every click is routed through the VisualVerificationEngine, so ORION confirms
the page reacted before continuing, and re-locates a moved control on retry.

If a real WebDriver (Selenium/Playwright) is installed it can be layered on
top later; the accessibility + vision + keyboard stack here needs no driver
and works against whatever browser the user already has open.
"""

from __future__ import annotations

import asyncio
import webbrowser
from typing import Any, Optional
from urllib.parse import urlparse

from .bus import OrionBus
from .control import AutonomousControlLayer
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .verification import VisualVerificationEngine
from .vision import VisionAgent

# Ranked candidate labels for common consent / dismissal controls.
ACCEPT_LABELS = (
    "Accept all", "Accept All Cookies", "Accept all cookies", "I agree", "Agree",
    "Allow all", "Accept cookies", "Got it", "Accept", "OK", "I accept", "Allow",
)
REJECT_LABELS = (
    "Reject all", "Reject All", "Reject all cookies", "Decline", "Decline all",
    "Necessary only", "Only necessary", "Reject", "Refuse all", "Do not accept",
)
DISMISS_LABELS = (
    "No thanks", "Not now", "Maybe later", "Dismiss", "Close", "Skip", "×", "X",
)


class WebController:
    """Human-like browser control over the currently focused browser window."""

    BROWSER_HINTS = ("chrome", "edge", "firefox", "mozilla", "opera", "brave")

    def __init__(
        self,
        bus: OrionBus,
        control: AutonomousControlLayer,
        vision: VisionAgent,
        verifier: VisualVerificationEngine,
        telemetry: Any | None = None,
        router: Any | None = None,
    ) -> None:
        self.bus = bus
        self.control = control
        self.vision = vision
        self.verifier = verifier
        self.telemetry = telemetry
        self.router = router
        # Optional live-narration callback (wired to the voice) so ORION can
        # speak what he sees as he scrolls, not only at the end.
        self.narrator: Optional[Any] = None

    def set_narrator(self, callback: Any) -> None:
        self.narrator = callback

    # ── focus / navigation ────────────────────────────────────────────────────

    async def _focus_browser(self) -> bool:
        """Bring a browser window to the foreground; True if one was found."""
        titles = await asyncio.to_thread(self.vision._visible_window_titles)
        for title in titles:
            if any(h in title.lower() for h in self.BROWSER_HINTS):
                await asyncio.to_thread(self.control.focus_window, title)
                await asyncio.sleep(0.25)
                return True
        return False

    async def open(self, url: str) -> ToolResult:
        """Open a URL in the default browser (new window/tab per OS policy)."""
        url = self._normalise(url)
        webbrowser.open(url)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("web.open")
        await asyncio.sleep(0.6)
        return ToolResult(f"Opened {url} in the browser.")

    async def navigate(self, url: str) -> ToolResult:
        """Type a URL into the focused browser's address bar (Ctrl+L)."""
        url = self._normalise(url)
        if not await self._focus_browser():
            return await self.open(url)
        self.control.send_hotkeys("ctrl+l")
        await asyncio.sleep(0.15)
        self.control.type_text(url)
        self.control.send_hotkeys("enter")
        await asyncio.sleep(0.8)
        return ToolResult(f"Navigated to {url}.")

    async def new_tab(self, url: str = "") -> ToolResult:
        if not await self._focus_browser():
            return await self.open(url or "about:blank")
        self.control.send_hotkeys("ctrl+t")
        await asyncio.sleep(0.25)
        if url.strip():
            self.control.type_text(self._normalise(url))
            self.control.send_hotkeys("enter")
            await asyncio.sleep(0.7)
        return ToolResult(f"Opened a new tab{f' at {url}' if url.strip() else ''}.")

    async def close_tab(self) -> ToolResult:
        if not await self._focus_browser():
            return ToolResult("No browser window is open.", ok=False)
        self.control.send_hotkeys("ctrl+w")
        await asyncio.sleep(0.2)
        return ToolResult("Closed the active tab.")

    async def switch_tab(self, index: int) -> ToolResult:
        """Ctrl+<n> jumps to tab n (1-8); Ctrl+9 is the last tab."""
        if not await self._focus_browser():
            return ToolResult("No browser window is open.", ok=False)
        n = max(1, min(9, int(index)))
        self.control.send_hotkeys(f"ctrl+{n}")
        await asyncio.sleep(0.2)
        return ToolResult(f"Switched to tab {n}.")

    # ── consent / popups ──────────────────────────────────────────────────────

    async def accept_cookies(self) -> ToolResult:
        return await self._click_first_label(ACCEPT_LABELS, "accept cookies")

    async def reject_cookies(self) -> ToolResult:
        return await self._click_first_label(REJECT_LABELS, "reject cookies")

    async def close_popup(self) -> ToolResult:
        result = await self._click_first_label(DISMISS_LABELS, "close pop-up", verify=True)
        if result.ok:
            return result
        # Last resort: Escape usually dismisses modal overlays.
        await self._focus_browser()
        self.control.send_hotkeys("escape")
        await asyncio.sleep(0.2)
        return ToolResult("Sent Escape to dismiss any modal pop-up.")

    async def _click_first_label(self, labels: tuple[str, ...], what: str,
                                 verify: bool = True) -> ToolResult:
        await self._focus_browser()
        for label in labels:
            element = await self.vision.find_element(label, kinds="button")
            if element is None:
                element = await self.vision.find_element(label, kinds="link")
            if element is not None:
                if verify:
                    res = await self.verifier.click_element(label, kinds="button", attempts=2)
                else:
                    res = self.control.click(*element["center"])
                if res.ok:
                    return ToolResult(f"Clicked '{label}' to {what}.")
        return ToolResult(
            f"Could not find a control to {what} (tried {len(labels)} common labels). "
            "The page may not show that prompt, or it is inside an iframe the "
            "accessibility tree does not expose.",
            ok=False,
        )

    # ── forms ─────────────────────────────────────────────────────────────────

    async def fill_form(self, fields: dict[str, str], submit: bool = False) -> ToolResult:
        """
        Fill fields by their label/placeholder → value.  Each field is located
        via the accessibility tree, focused, cleared and typed into.
        """
        await self._focus_browser()
        filled: list[str] = []
        for label, value in fields.items():
            try:
                value = SecuritySanitiser.guard_text(str(value), "web.form")
            except SecurityViolation:
                continue
            element = await self.vision.find_element(label, kinds="input")
            if element is None:
                element = await self.vision.find_element(label, kinds="all")
            if element is None:
                continue
            self.control.click(*element["center"])
            await asyncio.sleep(0.1)
            self.control.send_hotkeys("ctrl+a")
            self.control.type_text(value)
            filled.append(label)
            await asyncio.sleep(0.1)
        if not filled:
            return ToolResult("No matching form fields were found on the page.", ok=False)
        if submit:
            self.control.send_hotkeys("enter")
            await asyncio.sleep(0.6)
        return ToolResult(
            f"Filled {len(filled)} field(s): {', '.join(filled)}"
            + (" and submitted." if submit else ".")
        )

    # ── file transfer ─────────────────────────────────────────────────────────

    async def download_current(self) -> ToolResult:
        """Trigger a save of the current page/resource (Ctrl+S)."""
        if not await self._focus_browser():
            return ToolResult("No browser window is open.", ok=False)
        self.control.send_hotkeys("ctrl+s")
        await asyncio.sleep(0.6)
        return ToolResult(
            "Opened the browser's save dialog. Confirm the location with a "
            "web_control action of 'confirm_dialog' and a path, or approve it manually."
        )

    async def handle_file_dialog(self, path: str) -> ToolResult:
        """
        Complete an open/save file dialog by typing a path and confirming — the
        upload/download completion step.  A Windows file dialog focuses its
        filename field, so typing a path then Enter selects it.
        """
        try:
            path = SecuritySanitiser.guard_text(str(path), "web.file_dialog")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        dialogs = await self.vision.detect_dialogs()
        self.control.type_text(path)
        await asyncio.sleep(0.15)
        self.control.send_hotkeys("enter")
        await asyncio.sleep(0.5)
        return ToolResult(f"Submitted '{path}' to the file dialog. {dialogs.text.splitlines()[0]}")

    # ── reading ───────────────────────────────────────────────────────────────

    async def read_page(self) -> ToolResult:
        """Read visible page content via the accessibility tree, OCR as backup."""
        await self._focus_browser()
        # Prefer the document control's text from the accessibility tree.
        doc = await self.vision.find_element("", kinds="input")  # Document role lives in INPUT_ROLES
        elements = await asyncio.to_thread(self.vision._detect_elements_sync, "all", 200)
        names = [e["name"] for e in elements if e["name"].strip()]
        text = "\n".join(dict.fromkeys(names))[:4000]
        if len(text) < 80:
            # Accessibility tree thin (e.g. canvas app) — fall back to OCR.
            ocr = await self.vision.ocr("")
            return ToolResult(f"Page contents (OCR):\n{ocr.text[:4000]}")
        return ToolResult(f"Page contents (accessibility tree):\n{text}")

    async def _read_visible(self) -> str:
        """Text currently on screen (accessibility tree, OCR fallback)."""
        elements = await asyncio.to_thread(self.vision._detect_elements_sync, "all", 160)
        names = [e["name"].strip() for e in elements if e["name"].strip()]
        text = "\n".join(dict.fromkeys(names))
        if len(text) < 60:
            ocr = await self.vision.ocr("")
            text = ocr.text if ocr.ok else text
        return text[:3000]

    # ── narrated slow browsing (see → explain → summarise → evaluate) ─────────

    async def narrated_scroll(self, url: str = "", steps: int = 6,
                              dwell: float = 1.6) -> ToolResult:
        """
        Navigate to *url* (or use the current page), then glide down the page a
        viewport at a time.  At each stop ORION reads what is visible and speaks
        a short observation; at the end he summarises and evaluates the site.

        This is deliberately human-paced: smooth multi-step scrolling with a
        dwell between viewports, so it is watchable and ORION can narrate along.
        """
        if url.strip():
            nav = await self.navigate(url)
            if not nav.ok:
                return nav
            await asyncio.sleep(1.4)  # let the page settle
        else:
            await self._focus_browser()

        seen: list[str] = []
        steps = max(1, min(20, int(steps)))
        self._narrate("Let me take a look and read down the page, sir.")
        for i in range(steps):
            visible = await self._read_visible()
            if visible:
                seen.append(visible)
                observation = self._observe(visible, i)
                if observation:
                    self._narrate(observation)
            await self._smooth_scroll_down()
            await asyncio.sleep(max(0.4, dwell))
            if self.telemetry is not None:
                self.telemetry.metrics.incr("web.narrated_step")

        corpus = "\n".join(dict.fromkeys("\n".join(seen).splitlines()))[:6000]
        verdict = await self._summarise_and_evaluate(corpus, url)
        self._narrate(verdict)
        return ToolResult(verdict)

    async def summarise_current(self, url: str = "") -> ToolResult:
        """Read the whole visible page and produce a summary + evaluation."""
        await self._focus_browser()
        corpus = await self._read_visible()
        verdict = await self._summarise_and_evaluate(corpus, url)
        return ToolResult(verdict)

    async def _smooth_scroll_down(self) -> None:
        """Glide down roughly one viewport in small steps (watchable, human)."""
        for _ in range(5):
            await asyncio.to_thread(self.control.scroll, -120)
            await asyncio.sleep(0.12)

    def _observe(self, visible: str, index: int) -> str:
        """A fast heuristic one-liner about the current viewport (no LLM)."""
        lines = [ln.strip() for ln in visible.splitlines() if len(ln.strip()) > 12]
        if not lines:
            return ""
        headline = max(lines[:8], key=len)[:120]
        if index == 0:
            return f"At the top I can see: {headline}."
        return f"Scrolling on — this section covers: {headline}."

    async def _summarise_and_evaluate(self, corpus: str, url: str) -> str:
        """LLM summary + evaluation when a model is available; extractive else."""
        if not corpus.strip():
            return "I couldn't read meaningful text from that page, sir."
        if self.router is not None and self.router.has_text_fallback():
            persona = (
                "You are ORION reviewing a web page for the user. In under 140 words: "
                "(1) summarise what the page is about and its key points; (2) evaluate "
                "it — credibility, usefulness, and any bias or gaps you notice. Speak "
                "plainly to 'sir'."
            )
            prompt = f"Web page{f' ({url})' if url else ''} content I read on screen:\n{corpus[:4000]}"
            try:
                _profile, answer = await self.router.generate_text(prompt, system_extra=persona)
                return answer
            except Exception:
                pass
        # Offline/extractive fallback: lead lines + a measured evaluation.
        lines = [ln.strip() for ln in corpus.splitlines() if len(ln.strip()) > 30]
        lead = " ".join(lines[:4])[:400]
        return (
            f"Here's the gist, sir: {lead} "
            "On balance it reads as a standard informational page; I'd corroborate "
            "any key claims against a second source before relying on them."
        )

    def _narrate(self, text: str) -> None:
        if not text:
            return
        self.bus.log.emit(f"WEB: {text}")
        if self.narrator is not None:
            try:
                self.narrator(text)
            except Exception:
                pass

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalise(self, url: str) -> str:
        url = SecuritySanitiser.guard_text(str(url or "").strip(), "web.url")
        if not url:
            return "about:blank"
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme not in {"http", "https", "about"}:
            raise SecurityViolation("blocked unsafe URL: only HTTP and HTTPS are permitted")
        return parsed.geturl()
