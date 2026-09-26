"""
Social automation — ORION drives the user's REAL, logged-in TikTok and
Instagram accounts through a visible browser, to actually post/reply on
their behalf.

Kept entirely separate from BrowserCopilot (browser_copilot.py, general
narrated CDP browsing) — this is a higher-risk, opt-in surface: it logs
into real social accounts and can send things (an Instagram reply) to real
people. Two design choices exist specifically because of that:

  * A DEDICATED, non-default browser profile (config/social_profile), never
    the user's everyday Chrome profile — Chrome (since ~M136/March 2025)
    refuses CDP/remote-debugging attach against the default user-data-dir
    for security hardening, and BrowserCopilot already sidesteps this the
    same way (browser_copilot.py's own dedicated profile dir). The user
    logs into TikTok/Instagram ONCE inside this ORION-owned profile;
    cookies persist across runs.

  * Sending an Instagram reply follows the SAME approval-gated draft/send
    pattern as OutlookService (outlook.py) — draft first, transmit only
    with confirm=True after explicit user approval. TikTok upload does NOT
    need that gate for sending, but it also does not auto-click "Post":
    ORION fills in the video, caption and hashtags and leaves the final
    publish click to the user, since that action is public and permanent.

Threading: Playwright's Python driver needs subprocess support
(WindowsProactorEventLoopPolicy on Windows) that ORION's qasync GUI loop
does not provide. All Playwright calls run on a dedicated thread with their
own freshly created event loop; callers bridge in with
asyncio.run_coroutine_threadsafe + asyncio.wrap_future — the same
"never touch the GUI loop" isolation OutlookService uses for COM
(outlook.py), for an analogous but different underlying reason.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation


class SocialAutomationService:
    """Playwright-driven automation against the user's real TikTok/Instagram."""

    def __init__(self, bus: OrionBus, profile_dir: str | Path | None = None) -> None:
        self.bus = bus
        self.profile_dir = Path(profile_dir) if profile_dir else (CONFIG_DIR / "social_profile")
        self._pending_replies: dict[str, dict[str, str]] = {}
        self._reply_counter = 0

        self._pw_thread: Thread | None = None
        self._pw_loop: asyncio.AbstractEventLoop | None = None
        self._ready = Event()

        self._playwright: Any = None
        self._context: Any = None
        # Serialises context creation AND page interactions across
        # overlapping calls — two tool calls racing _ensure_context would
        # otherwise both see no context yet and try to launch twice against
        # the same profile dir.
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        try:
            import playwright  # noqa: F401
            return True
        except Exception:
            return False

    def _unavailable(self) -> ToolResult:
        return ToolResult(
            "Social automation is unavailable — install it with "
            "'pip install playwright' then run 'playwright install chrome' once.",
            ok=False,
        )

    def _safe(self, text: str, label: str) -> str:
        return SecuritySanitiser.guard_text(str(text or ""), label)

    def _narrate(self, step: str, detail: str = "") -> None:
        line = f"SOCIAL: {step}" + (f" — {detail}" if detail else "")
        try:
            self.bus.log.emit(line)
        except Exception:
            pass
        try:
            self.bus.browser_step.emit({"step": step, "detail": detail, "origin": "social"})
        except Exception:
            pass

    # ── Playwright thread/loop isolation ────────────────────────────────────

    def _ensure_loop_thread(self) -> None:
        if self._pw_thread is not None:
            return

        def _run() -> None:
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._pw_loop = loop
            self._ready.set()
            loop.run_forever()

        self._pw_thread = Thread(target=_run, name="orion-social-playwright", daemon=True)
        self._pw_thread.start()
        self._ready.wait(timeout=5.0)

    async def _run_pw(self, coro: Any) -> ToolResult:
        """Submit an already-built coroutine to the Playwright thread's own
        event loop and await its result from the caller's (qasync) loop."""
        self._ensure_loop_thread()
        if self._pw_loop is None:
            return ToolResult("The social automation thread failed to start.", ok=False)
        fut = asyncio.run_coroutine_threadsafe(coro, self._pw_loop)
        return await asyncio.wrap_future(fut)

    async def _ensure_context(self) -> ToolResult | None:
        """Launch (or reuse) a persistent browser context on the DEDICATED
        social profile. Must be called with self._lock held. Returns None
        on success, or an actionable ToolResult on failure."""
        if self._context is not None:
            return None
        try:
            from playwright.async_api import async_playwright
        except Exception:
            return self._unavailable()
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._narrate("Opening the social automation browser", str(self.profile_dir))
        try:
            self._playwright = await async_playwright().start()
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                channel="chrome",
                headless=False,
                no_viewport=True,
            )
        except Exception as exc:
            self._playwright = None
            self._context = None
            return ToolResult(
                f"Couldn't launch the social automation browser: {exc}. Make sure "
                "Google Chrome is installed and 'playwright install chrome' has "
                "been run once.", ok=False)
        return None

    async def _get_page(self) -> Any:
        if not self._context.pages:
            return await self._context.new_page()
        return self._context.pages[-1]

    # ── TikTok ───────────────────────────────────────────────────────────────

    async def tiktok_upload(self, video_path: str, caption: str = "", hashtags: str = "") -> ToolResult:
        """User-initiated, user-watched upload to TikTok Studio: fills in the
        video, caption and hashtags. Does NOT click Post — publishing is
        public and permanent, so that final click is left to the user in the
        visible window ORION opens. No confirm gate is needed for the
        fill-in itself, matching web_automation's other immediate actions."""
        try:
            video_path = self._safe(video_path, "social.video_path").strip()
            caption = self._safe(caption, "social.caption")
            hashtags = self._safe(hashtags, "social.hashtags")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not video_path:
            return ToolResult("A video file path is required.", ok=False)
        path = Path(video_path).expanduser()
        if not path.exists():
            return ToolResult(f"No file found at '{video_path}'.", ok=False)

        async def _inner() -> ToolResult:
            async with self._lock:
                failure = await self._ensure_context()
                if failure is not None:
                    return failure
                page = await self._get_page()
                self._narrate("Opening TikTok Studio")
                await page.goto(
                    "https://www.tiktok.com/tiktokstudio/upload",
                    wait_until="domcontentloaded",
                )
                try:
                    file_input = page.locator("input[type=file]").first
                    await file_input.wait_for(state="attached", timeout=20_000)
                    self._narrate("Selecting video", path.name)
                    await file_input.set_input_files(str(path))
                except Exception as exc:
                    return ToolResult(
                        "Couldn't find TikTok's upload field — the page layout may "
                        f"have changed, or you're not logged in yet ({exc}). Log "
                        "into TikTok once in the browser window ORION just opened, "
                        "then try again.", ok=False)
                full_caption = (caption + " " + hashtags).strip()
                # Said "with the caption and hashtags filled in" whether the
                # caption box was found or not — and even with no caption to
                # write — which invites posting a video with no caption.
                caption_note = "no caption was given, so the caption box is empty"
                if full_caption:
                    try:
                        self._narrate("Writing caption", full_caption[:60])
                        desc = page.locator('[contenteditable="true"]').first
                        await desc.wait_for(state="visible", timeout=30_000)
                        await desc.click()
                        await desc.fill(full_caption)
                        caption_note = "the caption and hashtags are filled in"
                    except Exception:
                        # The video is still loaded; the user can add it.
                        caption_note = ("I COULDN'T fill in the caption — TikTok's "
                                        "caption box wasn't found — so add it "
                                        "before posting")
                return ToolResult(
                    f"The video is loaded into TikTok Studio; {caption_note}. "
                    "Review it in the browser window — I've left the Post button "
                    "for you to press yourself."
                )

        return await self._run_pw(_inner())

    # ── Instagram ────────────────────────────────────────────────────────────

    async def instagram_check_dms(self, limit: int = 10) -> ToolResult:
        """Read-only: list recent Instagram conversations. No confirm gate."""

        async def _inner() -> ToolResult:
            async with self._lock:
                failure = await self._ensure_context()
                if failure is not None:
                    return failure
                page = await self._get_page()
                self._narrate("Opening Instagram DMs")
                await page.goto(
                    "https://www.instagram.com/direct/inbox/",
                    wait_until="domcontentloaded",
                )
                try:
                    await page.wait_for_selector(
                        'a[href^="/direct/t/"]', timeout=20_000
                    )
                except Exception:
                    return ToolResult(
                        "Couldn't reach the Instagram inbox — you may not be logged "
                        "in yet. Log into Instagram once in the browser window "
                        "ORION just opened, then try again.", ok=False)
                threads = await page.locator('a[href^="/direct/t/"]').all()
                lines: list[str] = []
                for thread in threads[: max(1, int(limit))]:
                    try:
                        text = (await thread.inner_text()).strip().replace("\n", " · ")
                        if text:
                            lines.append(f"- {text[:160]}")
                    except Exception:
                        continue
                if not lines:
                    return ToolResult("No conversations found in the Instagram inbox.")
                return ToolResult("Recent Instagram conversations:\n" + "\n".join(lines))

        return await self._run_pw(_inner())

    async def instagram_draft_reply(self, contact: str, message: str) -> ToolResult:
        """Compose a reply WITHOUT sending it — mirrors outlook.py's
        create_draft exactly. Nothing reaches Instagram until
        instagram_reply is called with confirm=True."""
        try:
            contact = self._safe(contact, "social.contact").strip()
            message = self._safe(message, "social.message").strip()
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not contact or not message:
            return ToolResult("A contact and a message are required to draft a reply.", ok=False)
        self._reply_counter += 1
        ref = f"reply-{self._reply_counter}"
        self._pending_replies[ref] = {"contact": contact, "message": message}
        self.bus.banner.emit(f"INSTAGRAM REPLY READY: {ref} → {contact}", 2)
        return ToolResult(
            f"Reply drafted as {ref} to {contact}: \"{message}\". Awaiting "
            f"approval — confirm to send (social_media with "
            f"action='instagram_reply', reply_ref='{ref}', confirm=true)."
        )

    def pending_replies(self) -> list[dict[str, str]]:
        """Session replies awaiting approval."""
        return [{"ref": ref, **rec} for ref, rec in self._pending_replies.items()]

    async def instagram_reply(self, reply_ref: str, confirm: bool = False) -> ToolResult:
        """Transmit a previously drafted reply. Refuses without confirm=True
        — mirrors outlook.py's send_draft exactly; a live DM reply carries
        the same "sent to the wrong person" risk as an email plus platform
        ban/detection risk on top."""
        reply_ref = str(reply_ref or "").strip().lower()
        record = self._pending_replies.get(reply_ref)
        if record is None:
            known = ", ".join(sorted(self._pending_replies)) or "none"
            return ToolResult(f"No pending reply matches '{reply_ref}'. Pending: {known}.", ok=False)
        if not confirm:
            return ToolResult(
                f"Approval required before sending. {reply_ref} is addressed to "
                f"{record['contact']}: \"{record['message']}\". Repeat the request "
                "with explicit confirmation to send.", ok=False)

        async def _inner() -> ToolResult:
            async with self._lock:
                failure = await self._ensure_context()
                if failure is not None:
                    return failure
                page = await self._get_page()
                contact = record["contact"]
                self._narrate("Opening conversation", contact)
                await page.goto(
                    "https://www.instagram.com/direct/inbox/",
                    wait_until="domcontentloaded",
                )
                try:
                    thread = page.get_by_text(contact, exact=False).first
                    await thread.click(timeout=15_000)
                    box = page.get_by_role("textbox", name="Message")
                    await box.wait_for(state="visible", timeout=15_000)
                    self._narrate("Typing reply", record["message"][:60])
                    await box.click()
                    await box.fill(record["message"])
                    await box.press("Enter")
                except Exception as exc:
                    return ToolResult(
                        f"Couldn't send the reply to {contact} — the conversation "
                        f"may not be visible, or Instagram's layout changed ({exc}). "
                        f"The draft is still held as {reply_ref}.", ok=False)
                self._pending_replies.pop(reply_ref, None)
                self._narrate("Reply sent", contact)
                self.bus.banner.emit(f"INSTAGRAM REPLY SENT → {contact}", 3)
                return ToolResult(f"Sent. {reply_ref} is on its way to {contact}.")

        return await self._run_pw(_inner())

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> ToolResult:
        if self._pw_loop is None:
            return ToolResult("No social automation browser was open.")

        async def _inner() -> None:
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception:
                    pass
                self._context = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

        fut = asyncio.run_coroutine_threadsafe(_inner(), self._pw_loop)
        try:
            await asyncio.wrap_future(fut)
        except Exception:
            pass
        self._narrate("Closed the social automation browser")
        return ToolResult("Closed the social automation browser.")
