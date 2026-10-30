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
import re
from typing import Any, Optional
from urllib.parse import urlparse

from .bus import OrionBus
from .control import AutonomousControlLayer
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .verification import VisualVerificationEngine
from .vision import VisionAgent
from .browser import open_url
from .utils import fold_control_label

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

    BROWSER_HINTS = ("chrome", "chromium", "edge", "firefox", "mozilla", "opera", "brave", "vivaldi")
    OBSERVATION_TIMEOUT_S = 8.0

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
        from .execution_policy import require_foreground
        if require_foreground("shared_browser_focus") is not None:
            return False
        try:
            titles = await asyncio.wait_for(
                asyncio.to_thread(self.vision._visible_window_titles), self.OBSERVATION_TIMEOUT_S)
        except Exception:
            return False
        for title in titles:
            if any(h in title.lower() for h in self.BROWSER_HINTS):
                result = await asyncio.to_thread(self.control.focus_window, title)
                if result.ok:
                    await asyncio.sleep(0.25)
                    return True
        return False

    async def _input(self, method: str, *args: Any, **kwargs: Any) -> ToolResult:
        # Human-paced typing and focus can take seconds. They must not block
        # the Live channel's microphone / receive tasks on this event loop.
        from .execution_policy import require_foreground
        blocked = require_foreground("shared_browser_input")
        if blocked is not None:
            return blocked
        return await asyncio.to_thread(getattr(self.control, method), *args, **kwargs)

    async def _elements(self, kinds: str = "all", limit: int = 400) -> list[dict[str, Any]]:
        # Only observations are timed out: a slow UIA worker may finish later,
        # but can never issue a click or type after this request has returned.
        return await asyncio.wait_for(
            asyncio.to_thread(self.vision._detect_elements_sync, kinds, limit),
            self.OBSERVATION_TIMEOUT_S)

    async def open(self, url: str) -> ToolResult:
        """Open a URL in the default browser (new window/tab per OS policy)."""
        from .execution_policy import require_foreground
        blocked = require_foreground("shared_browser_open")
        if blocked is not None:
            return blocked
        url = self._normalise(url)
        if not await asyncio.to_thread(open_url, url):
            return ToolResult(f"The browser could not open {url}.", ok=False)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("web.open")
        await asyncio.sleep(0.6)
        return ToolResult(f"Opened {url} in the browser.")

    async def navigate(self, url: str) -> ToolResult:
        """Type a URL into the focused browser's address bar (Ctrl+L)."""
        url = self._normalise(url)
        if not await self._focus_browser():
            return await self.open(url)
        result = await self._input("send_hotkeys", "ctrl+l")
        if not result.ok:
            return result
        await asyncio.sleep(0.15)
        result = await self._input("type_text", url)
        if not result.ok:
            return result
        result = await self._input("send_hotkeys", "enter")
        if not result.ok:
            return result
        await asyncio.sleep(0.8)
        return ToolResult(f"Navigated to {url}.")

    async def new_tab(self, url: str = "") -> ToolResult:
        target = self._normalise(url) if url.strip() else ""
        if not await self._focus_browser():
            return await self.open(target or "about:blank")
        result = await self._input("send_hotkeys", "ctrl+t")
        if not result.ok:
            return result
        await asyncio.sleep(0.25)
        if target:
            result = await self._input("type_text", target)
            if not result.ok:
                return result
            result = await self._input("send_hotkeys", "enter")
            if not result.ok:
                return result
            await asyncio.sleep(0.7)
        return ToolResult(f"Opened a new tab{f' at {url}' if url.strip() else ''}.")

    async def close_tab(self) -> ToolResult:
        if not await self._focus_browser():
            return ToolResult("No browser window is open.", ok=False)
        result = await self._input("send_hotkeys", "ctrl+w")
        if not result.ok:
            return result
        await asyncio.sleep(0.2)
        return ToolResult("Closed the active tab.")

    async def switch_tab(self, index: int) -> ToolResult:
        """Ctrl+<n> jumps to tab n (1-8); Ctrl+9 is the last tab."""
        n = int(index)
        if not 1 <= n <= 9:
            return ToolResult("Tab index must be between 1 and 9 (9 selects the last tab).", ok=False)
        if not await self._focus_browser():
            return ToolResult("No browser window is open.", ok=False)
        result = await self._input("send_hotkeys", f"ctrl+{n}")
        if not result.ok:
            return result
        await asyncio.sleep(0.2)
        return ToolResult(f"Switched to tab {n}.")

    # ── consent / popups ──────────────────────────────────────────────────────

    async def accept_cookies(self) -> ToolResult:
        return await self._click_first_label(ACCEPT_LABELS, "accept cookies")

    async def reject_cookies(self) -> ToolResult:
        return await self._click_first_label(REJECT_LABELS, "reject cookies")

    async def close_popup(self) -> ToolResult:
        result = await self._click_first_label(DISMISS_LABELS, "close pop-up", verify=True)
        if result.ok or not any(e.get("status") == "missing_browser_control"
                                for e in result.evidence or []):
            return result
        # Last resort: Escape usually dismisses modal overlays.
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused; no input was sent.", ok=False)
        result = await self._input("send_hotkeys", "escape")
        if not result.ok:
            return result
        await asyncio.sleep(0.2)
        return ToolResult("Sent Escape to dismiss any modal pop-up.")

    async def _click_first_label(self, labels: tuple[str, ...], what: str,
                                 verify: bool = True) -> ToolResult:
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused; no input was sent.", ok=False)
        try:
            elements = await self._elements()
        except TimeoutError:
            return ToolResult("The browser accessibility scan timed out; no click was issued.", ok=False)
        except Exception as exc:
            return ToolResult(f"The browser controls could not be read: {exc}", ok=False)
        # Inspect once, then rank the labels in memory. Re-enumerating the
        # whole Chromium tree for every candidate used to take over 45 s.
        for label in labels:
            matches = [e for e in elements if e.get("enabled", True)
                       and e.get("role") in {"Button", "Hyperlink", "Link"}
                       and fold_control_label(e.get("name", "")) == fold_control_label(label)]
            if matches:
                if len(matches) > 1:
                    return ToolResult(f"Several controls are labelled '{label}'; inspect the page to choose one.",
                                      ok=False)
                element = matches[0]
                if verify:
                    res = await self.verifier._resolved_click(label, {
                        "status": "found", "element": element, "candidates": matches,
                    }, attempts=2)
                else:
                    res = await self._input("click", *element["center"])
                # An issued but unverified click may already have dismissed
                # the banner. Preserve its receipt; never click another label.
                return res
        return ToolResult(
            f"Could not find a control to {what} (tried {len(labels)} common labels). "
            "The page may not show that prompt, or it is inside an iframe the "
            "accessibility tree does not expose.",
            ok=False,
            evidence=[{"status": "missing_browser_control"}],
        )

    async def click(self, text: str) -> ToolResult:
        text = SecuritySanitiser.guard_text(text.strip(), "web.click")
        if not text:
            return ToolResult("A visible control name is required for click.", ok=False)
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused; no click was issued.", ok=False)
        # The current view is inspected once. A page-wide scroll search is
        # available through desktop_control, but is too costly for this tool.
        try:
            resolved = await asyncio.wait_for(self.vision.resolve_element(text, "all"),
                                             self.OBSERVATION_TIMEOUT_S)
        except TimeoutError:
            return ToolResult("Browser control lookup timed out; no click was issued.", ok=False)
        if resolved["status"] != "missing":
            return await self.verifier._resolved_click(text, resolved, attempts=2)
        try:
            element = await asyncio.wait_for(self.vision.find_element_via_ocr(text),
                                            self.OBSERVATION_TIMEOUT_S)
        except TimeoutError:
            return ToolResult("Browser text lookup timed out; no click was issued.", ok=False)
        if element is None:
            return ToolResult(f"No visible browser control matches '{text}'.", ok=False)
        async def located():
            return element
        return await self.verifier._click_and_verify(text, located, attempts=2, strategy="ocr")

    async def scroll(self, amount: int = -5) -> ToolResult:
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused; no scrolling was sent.", ok=False)
        return await self._input("scroll", max(-30, min(30, int(amount))))

    async def submit(self) -> ToolResult:
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused; no input was sent.", ok=False)
        return await self._input("send_hotkeys", "enter")

    # ── forms ─────────────────────────────────────────────────────────────────

    async def fill_form(self, fields: dict[str, str], submit: bool = False) -> ToolResult:
        """
        Fill fields by their label/placeholder → value.  Each field is located
        via the accessibility tree, focused, cleared and typed into.
        """
        if not fields:
            return ToolResult("No form fields were supplied.", ok=False)
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused; no form input was sent.", ok=False)
        filled: list[str] = []
        missing: list[str] = []
        for label, value in fields.items():
            try:
                value = SecuritySanitiser.guard_text(str(value), "web.form")
            except SecurityViolation as exc:
                return ToolResult(f"Form input blocked for '{label}': {exc}", ok=False)
            try:
                element = await asyncio.wait_for(self.vision.find_element(str(label), kinds="input"),
                                                 self.OBSERVATION_TIMEOUT_S)
            except TimeoutError:
                return ToolResult(f"Field lookup timed out for '{label}'; the form was not submitted.", ok=False)
            if element is None:
                missing.append(str(label))
                continue
            kwargs = {}
            if hasattr(self.vision, "validate_element_sync"):
                kwargs["target_check"] = lambda element=element: self.vision.validate_element_sync(element)
            result = await self._input("click", *element["center"], **kwargs)
            if not result.ok:
                return result
            await asyncio.sleep(0.1)
            result = await self._input("send_hotkeys", "ctrl+a")
            if not result.ok:
                return result
            result = await self._input("type_text", value)
            if not result.ok:
                return result
            filled.append(str(label))
            await asyncio.sleep(0.1)
        if not filled:
            return ToolResult("No matching form fields were found on the page.", ok=False)
        if missing:
            return ToolResult(f"Filled {', '.join(filled)}. Missing fields: {', '.join(missing)}; "
                              "the form was not submitted.", ok=False)
        if submit:
            result = await self._input("send_hotkeys", "enter")
            if not result.ok:
                return result
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
        result = await self._input("send_hotkeys", "ctrl+s")
        if not result.ok:
            return result
        await asyncio.sleep(0.6)
        return ToolResult(
            "Opened the browser's save dialog. Confirm the location with a "
            "web_control action of 'file_dialog' and a path, or approve it manually."
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
        if not path.strip():
            return ToolResult("A file path is required.", ok=False)
        # A dialog somewhere on the desktop is insufficient: the foreground
        # window must be an Open/Save file dialog before we type a filename.
        title_reader = getattr(self.control, "_foreground_title", None)
        title = await asyncio.to_thread(title_reader) if callable(title_reader) else ""
        if not re.search(r"(?i)\b(?:save(?: as)?|open|upload|choose file|select file)\b", title):
            return ToolResult("No Open/Save file dialog is focused. Open the browser's file dialog first.", ok=False)
        result = await self._input("send_hotkeys", "alt+n")
        if not result.ok:
            return result
        result = await self._input("send_hotkeys", "ctrl+a")
        if not result.ok:
            return result
        result = await self._input("type_text", path)
        if not result.ok:
            return result
        await asyncio.sleep(0.15)
        result = await self._input("send_hotkeys", "enter")
        if not result.ok:
            return result
        await asyncio.sleep(0.5)
        return ToolResult(f"Submitted '{path}' to the focused file dialog.")

    # ── reading ───────────────────────────────────────────────────────────────

    async def read_page(self) -> ToolResult:
        """Read visible page content via the accessibility tree, OCR as backup."""
        if not await self._focus_browser():
            return ToolResult("No browser window could be focused to read.", ok=False)
        return await self._page_contents()

    async def _page_contents(self) -> ToolResult:
        try:
            elements = await self._elements()
        except Exception:
            elements = []
        names = [str(e.get("name", "")).strip() for e in elements if e.get("name")]
        text = "\n".join(dict.fromkeys(names))[:4000]
        if len(text) < 80:
            # Accessibility tree thin (e.g. canvas app) — fall back to OCR.
            try:
                ocr = await asyncio.wait_for(self.vision.ocr(""), self.OBSERVATION_TIMEOUT_S)
            except TimeoutError:
                return ToolResult("Browser text recognition timed out.", ok=False)
            if not ocr.ok:
                return ocr
            if not ocr.text.strip():
                return ToolResult("The browser page contains no readable text.", ok=False)
            return ToolResult(f"Page contents (OCR):\n{ocr.text[:4000]}", media=ocr.media)
        return ToolResult(f"Page contents (accessibility tree):\n{text}")

    async def _read_visible(self) -> str:
        """Text currently on screen (accessibility tree, OCR fallback)."""
        result = await self._page_contents()
        return result.text[:3000] if result.ok else ""

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
            if not await self._focus_browser():
                return ToolResult("No browser window could be focused to browse.", ok=False)

        seen: list[str] = []
        steps = max(1, min(20, int(steps)))
        self._narrate("Let me take a look and read down the page.")
        for i in range(steps):
            visible = await self._read_visible()
            if visible:
                seen.append(visible)
                observation = self._observe(visible, i)
                if observation:
                    self._narrate(observation)
            scrolled = await self._smooth_scroll_down()
            if not scrolled.ok:
                return scrolled
            await asyncio.sleep(max(0.4, min(5.0, dwell)))
            if self.telemetry is not None:
                self.telemetry.metrics.incr("web.narrated_step")

        corpus = "\n".join(dict.fromkeys("\n".join(seen).splitlines()))[:6000]
        if not corpus.strip():
            return ToolResult("I couldn't read meaningful text from that page.", ok=False)
        verdict = await self._summarise_and_evaluate(corpus, url)
        self._narrate(verdict)
        return ToolResult(verdict)

    async def summarise_current(self, url: str = "") -> ToolResult:
        """Read the whole visible page and produce a summary + evaluation."""
        if url.strip():
            result = await self.navigate(url)
            if not result.ok:
                return result
        elif not await self._focus_browser():
            return ToolResult("No browser window could be focused to summarise.", ok=False)
        corpus = await self._read_visible()
        if not corpus.strip():
            return ToolResult("I couldn't read meaningful text from that page.", ok=False)
        verdict = await self._summarise_and_evaluate(corpus, url)
        return ToolResult(verdict)

    async def _smooth_scroll_down(self) -> ToolResult:
        """Glide down roughly one viewport in small steps (watchable, human)."""
        for _ in range(5):
            # The control layer takes wheel notches, not Windows' 120-unit
            # WHEEL_DELTA. Passing -120 jumped dozens of pages at each stop.
            result = await self._input("scroll", -1)
            if not result.ok:
                return result
            await asyncio.sleep(0.12)
        return ToolResult("Scrolled down one viewport.")

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
            return "I couldn't read meaningful text from that page."
        if self.router is not None and self.router.has_text_fallback():
            persona = (
                "You are ORION reviewing a web page for the user. In under 140 words: "
                "(1) summarise what the page is about and its key points; (2) evaluate "
                "it — credibility, usefulness, and any bias or gaps you notice. Speak "
                "plainly."
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
            f"Here's the gist: {lead} "
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
        # Local file paths & file:// URLs — ORION can browse the filesystem in
        # the browser (folder listings and local files), not just the web.  The
        # browser only ever DISPLAYS the path (a read), consistent with the
        # existing unrestricted read access of process_file / find_files.
        file_url = self._as_file_url(url)
        if file_url is not None:
            return file_url
        if url.lower() == "about:blank":
            return "about:blank"
        if re.match(r"(?i)^(?:javascript|data|vbscript|about|ftp):", url):
            raise SecurityViolation("blocked unsafe URL scheme")
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise SecurityViolation(
                "blocked unsafe URL: only HTTP, HTTPS and local file paths are permitted"
            )
        return parsed.geturl()

    @staticmethod
    def _as_file_url(text: str) -> str | None:
        """Convert a local path (``C:\\…``, ``C:/…``, ``~/…``, ``\\\\server\\share``)
        or an explicit ``file://`` URL into a browsable file URL — or None when
        the input is not a local path (so web URLs fall through unchanged)."""
        import os
        import re
        from pathlib import Path
        raw = text.strip().strip('"').strip("'")
        if raw.lower().startswith("file://"):
            return raw
        looks_local = (
            re.match(r"^[A-Za-z]:[\\/]", raw) is not None   # C:\ or C:/
            or raw.startswith("~")                          # home directory
            or raw.startswith("\\\\")                       # UNC \\server\share
        )
        if not looks_local:
            return None
        try:
            return Path(os.path.expanduser(raw)).as_uri()
        except Exception:
            # Best-effort manual conversion if as_uri rejects the spec.
            cleaned = os.path.expanduser(raw).replace("\\", "/")
            return "file:///" + cleaned.lstrip("/")
