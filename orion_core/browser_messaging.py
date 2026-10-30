"""WhatsApp delivery in an ORION-owned headless Edge context.

No personal browser attachment, token extraction, clipboard, OS focus or input.
Login/pairing is a separate explicit setup operation. Selector drift stops work.
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as WorkerTimeout
from contextvars import copy_context
import importlib.util
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any

from .constants import CONFIG_DIR
from .data import ToolResult


class OwnedBrowserMessenger:
    PLATFORMS = frozenset({"whatsapp"})
    HOME_URL = "https://web.whatsapp.com/"
    READY_SELECTOR = "#pane-side"
    PAGE_TIMEOUT_MS = 15000
    WORKER_TIMEOUT_S = 75.0

    def __init__(self, bus: Any, profile_root: Path | None = None) -> None:
        self.bus = bus
        self.platform = next(iter(self.PLATFORMS))
        self.profile_root = Path(profile_root) if profile_root else CONFIG_DIR / "browser_messaging"
        self._worker = None
        self._queue = queue.Queue(maxsize=2)
        self._state_lock = threading.RLock()
        self._closing = False
        self._close_future = None
        self._pending = threading.Lock()
        self._idle_callbacks = []
        self._playwright = self._context = self._page = None
        self._headless = True
        self._submitted = False
        self._abort_before_submit = threading.Event()

    def _submit(self, operation, *, closing: bool = False):
        with self._state_lock:
            if self._closing and not closing:
                raise RuntimeError("Owned browser is shutting down")
            if self._worker is None:
                def consume():
                    while True:
                        callback, future, final = self._queue.get()
                        try:
                            future.set_result(callback())
                        except BaseException as exc:
                            future.set_exception(exc)
                        if final:
                            return
                self._worker = threading.Thread(target=consume, name="orion-message-browser", daemon=True)
                self._worker.start()
            future = Future()
            self._queue.put_nowait((operation, future, closing))
            return future

    def _result(self, status: str, text: str, *, ok: bool = False, attempted: bool = False, sent=False) -> ToolResult:
        return ToolResult(text, ok=ok, receipt={"kind": "messaging", "platform": self.platform,
            "backend": "owned_browser", "status": status, "attempted": attempted, "sent": sent,
            "sender_kind": "connected_personal_account",
            "delivery_verified": False, "desktop_input_used": False})

    def release_when_idle(self, callback) -> bool:
        """Transfer an outer delivery lease until the physical owner returns."""
        with self._state_lock:
            if not self._pending.locked():
                return False
            self._idle_callbacks.append(callback)
            return True

    def _finish_operation(self) -> None:
        with self._state_lock:
            self._pending.release()
            callbacks, self._idle_callbacks = self._idle_callbacks, []
        for callback in callbacks:
            callback()

    def _call(self, operation, *args) -> ToolResult:
        if not self._pending.acquire(blocking=False):
            return self._result("pending", "The owned messaging browser is still completing another operation.")
        with self._state_lock:
            self._submitted = False
            self._abort_before_submit.clear()
        caller_context = copy_context()
        def run():
            try:
                return caller_context.run(operation, *args)
            except Exception as exc:
                # Error types are sufficient diagnostics; browser errors can
                # contain recipient/body text and must not reach the log.
                detail = str(exc).casefold()
                unavailable = isinstance(exc, (ImportError, ModuleNotFoundError)) or any(
                    token in detail for token in ("executable doesn't exist", "distribution 'msedge' is not found", "browser was not found"))
                busy = any(token in detail for token in ("singletonlock", "profile in use", "user data directory is already in use"))
                state = "uncertain" if self._submitted else "unavailable" if unavailable else "profile_busy" if busy else "failed_before_submit"
                return self._result(state,
                    "Owned browser messaging stopped (" + type(exc).__name__ + "). "
                    + ("Check the conversation before sending again." if self._submitted else "No message was submitted."),
                    attempted=self._submitted, sent=None if self._submitted else False)
            finally:
                self._finish_operation()
        try:
            future = self._submit(run)
        except Exception:
            self._finish_operation()
            return self._result("unavailable", "The owned messaging browser worker is unavailable.")
        try:
            return future.result(timeout=self.WORKER_TIMEOUT_S)
        except WorkerTimeout:
            # The worker owns the lock until it physically returns. A timed-out
            # caller cannot queue another send behind an uncertain submission.
            with self._state_lock:
                self._abort_before_submit.set()
                submitted = self._submitted
            return self._result("uncertain" if submitted else "timed_out_before_submit",
                "The owned browser operation has not returned. Further submission was stopped; wait for its result before trying again.",
                attempted=submitted, sent=None if submitted else False)

    def _begin_submit(self) -> bool:
        from .execution_policy import current_scope
        with self._state_lock:
            scope = current_scope()
            if self._closing or self._abort_before_submit.is_set() or (scope is not None and not scope.active):
                return False
            self._submitted = True
            return True

    def _close_context(self) -> None:
        if self._context is not None:
            try:
                self._context.close()
            except Exception as exc:
                # A human may close the separate setup window themselves.
                # Recover only the driver's explicit already-closed outcome.
                if type(exc).__name__ != "TargetClosedError":
                    raise
        self._context = self._page = None

    def _ensure_page(self, *, headless: bool = True, recover_closed: bool = True):
        if self._context is not None and self._headless != headless:
            self._close_context()
        if self._context is None:
            from playwright.sync_api import sync_playwright
            self.profile_root.mkdir(parents=True, exist_ok=True)
            if self._playwright is None:
                self._playwright = sync_playwright().start()
            try:
                self._context = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.profile_root / self.platform), channel="msedge", headless=headless,
                    permissions=[], args=["--no-first-run", "--no-default-browser-check", "--mute-audio",
                        "--disable-audio-output", "--disable-audio-input", "--disable-notifications",
                        "--autoplay-policy=user-gesture-required"], timeout=self.PAGE_TIMEOUT_MS)
            except Exception:
                self._playwright.stop()
                self._playwright = None
                raise
            self._headless = headless
        if self._page is None or self._page.is_closed():
            # This context belongs exclusively to messaging. Do not select the
            # last tab of another browser service or attach through its port.
            try:
                self._page = self._context.new_page()
                self._page.set_default_timeout(self.PAGE_TIMEOUT_MS)
                self._page.goto(self.HOME_URL, wait_until="domcontentloaded", timeout=self.PAGE_TIMEOUT_MS)
            except Exception as exc:
                if not recover_closed or type(exc).__name__ != "TargetClosedError":
                    raise
                self._close_context()
                # One bounded preflight recovery, before any recipient/draft or
                # submission is touched. Never used after click/DOM Enter.
                return self._ensure_page(headless=headless, recover_closed=False)
        return self._page

    def _ready(self) -> tuple[Any, ToolResult | None]:
        if self._context is not None and not self._headless:
            try:
                if self._page is not None and not self._page.is_closed() and self._page.locator(self.READY_SELECTOR).count() != 1:
                    return self._page, self._result("needs_login", "Connect the account in the owned setup window, then check connection status. No message was submitted.")
            except Exception as exc:
                if type(exc).__name__ != "TargetClosedError":
                    raise
                self._close_context()
        page = self._ensure_page()
        try:
            page.locator(self.READY_SELECTOR).wait_for(state="visible", timeout=self.PAGE_TIMEOUT_MS)
        except Exception:
            return page, self._result("needs_login", "Connect " + self.platform.title() + " once in ORION's separate messaging browser profile. Background work will not open a login window.")
        return page, None

    def connect(self, platform: str = "whatsapp") -> ToolResult:
        """Open ONLY the separate setup profile after explicit human request."""
        from .execution_policy import foreground_authorized, require_foreground
        blocked = require_foreground("open messaging account setup")
        if blocked is not None:
            return blocked
        if not foreground_authorized():
            return self._result("setup_authorization_required", "Open account setup from ORION's explicit connection control before pairing.")
        if platform not in self.PLATFORMS:
            return self._result("unavailable", "No owned-browser adapter is available for that platform.")
        def setup():
            self._ensure_page(headless=False)
            return self._result("setup_opened", "The separate ORION " + self.platform.title() + " setup window is open. Connect it yourself, then request connection status. No message was submitted.", ok=True)
        return self._call(setup)

    def status(self, platform: str = "whatsapp") -> ToolResult:
        """Metadata only: no browser, Playwright driver or owner thread starts."""
        if platform not in self.PLATFORMS:
            return self._result("unavailable", "No owned-browser adapter is available for that platform.")
        from .browser import edge_path
        playwright = importlib.util.find_spec("playwright") is not None
        edge = bool(edge_path())
        profile = (self.profile_root / self.platform).is_dir()
        result = self._result("connection_unchecked" if playwright and edge and profile else "needs_setup",
            "Owned browser dependencies: Playwright " + ("available" if playwright else "missing")
            + ", Edge " + ("available" if edge else "missing") + "; separate profile "
            + ("exists. Request check_connection to inspect login readiness." if profile else "not paired. Open explicit account setup to connect."),
            ok=playwright and edge and profile)
        result.receipt.update({"playwright_available": playwright, "edge_available": edge,
                               "owned_profile_exists": profile, "connection_checked": False})
        return result

    def readiness(self, platform: str = "whatsapp") -> ToolResult:
        if platform not in self.PLATFORMS:
            return self._result("unavailable", "No owned-browser adapter is available for that platform.")
        def inspect():
            _page, blocked = self._ready()
            return blocked or self._result("ready", "ORION's separate " + self.platform.title() + " browser profile is connected; no desktop input is required.", ok=True)
        return self._call(inspect)

    @staticmethod
    def _exact(locator, text: str):
        choices = []
        for index in range(locator.count()):
            node = locator.nth(index)
            if (node.get_attribute("title") or node.inner_text()).strip().casefold() == text.strip().casefold():
                choices.append(node)
        return choices

    @staticmethod
    def _header_matches(page, contact: str) -> bool:
        header = page.locator("#main header")
        if header.count() != 1:
            return False
        if contact.casefold().strip() in {"me", "myself", "self", "self chat", "message yourself"}:
            text = header.inner_text()
            return bool(re.search(r"\(\s*you\s*\)", text, re.I)) and "message yourself" in text.casefold()
        exact = OwnedBrowserMessenger._exact(header.locator("[title]"), contact)
        return len(exact) == 1

    @staticmethod
    def _outgoing_ids(page, body: str) -> set[str]:
        observed = set()
        rows = page.locator("#main .message-out")
        for index in range(rows.count()):
            row = rows.nth(index)
            chunks = row.locator(".selectable-text.copyable-text")
            if not any(chunks.nth(part).inner_text() == body for part in range(chunks.count())):
                continue
            identity = row.get_attribute("data-id")
            if not identity:
                bound = row.locator("xpath=ancestor-or-self::*[@data-id][1]")
                identity = bound.get_attribute("data-id") if bound.count() == 1 else None
            if identity:
                observed.add(identity)
        return observed

    def _select_self(self, page) -> bool:
        launch = page.get_by_role("button", name=re.compile(r"^new (?:chat|message)$", re.I))
        if launch.count() != 1:
            return False
        launch.click()
        rows = page.locator('#side [role="row"], #side [role="listitem"]')
        choices = []
        for index in range(rows.count()):
            row = rows.nth(index)
            first = row.inner_text().splitlines()[0].strip() if row.inner_text().splitlines() else ""
            if re.search(r"\(\s*you\s*\)\s*$", first, re.I) or first.casefold() == "message yourself":
                choices.append(row)
        if len(choices) != 1:
            return False
        choices[0].click()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._header_matches(page, "self"):
                return True
            page.wait_for_timeout(100)
        return False

    def send(self, platform: str, contact: str, body: str) -> ToolResult:
        if platform not in self.PLATFORMS:
            return self._result("unavailable", "No owned-browser adapter is available for that platform.")
        return self._call(self._send, contact, body)

    def _send(self, contact: str, body: str) -> ToolResult:
        if not contact.strip() or not body.strip():
            return self._result("invalid_request", "A recipient and a nonempty exact message are required.")
        page, blocked = self._ready()
        if blocked:
            return blocked
        self_chat = contact.casefold().strip() in {"me", "myself", "self", "self chat", "message yourself"}
        if self_chat and not self._header_matches(page, contact):
            if not self._select_self(page):
                return self._result("recipient_unverified", "The owned browser could not verify one explicit self entry and matching account header. No name was guessed.")
        # A currently open header is not a unique identity: two saved contacts
        # may share it. Resolve the named recipient even when that chat is open.
        if not self_chat:
            search = page.locator('#side [contenteditable="true"][role="textbox"]')
            if search.count() != 1:
                return self._result("unsupported_layout", "The owned WhatsApp browser does not expose one contact search control.")
            search.fill(contact)
            # Resolve a unique exact titled recipient only inside chat rows.
            rows = page.locator('#pane-side [role="row"]')
            deadline = time.monotonic() + 5.0
            matches = []
            while time.monotonic() < deadline:
                matches = []
                for index in range(rows.count()):
                    row = rows.nth(index)
                    if len(self._exact(row.locator("[title]"), contact)) == 1:
                        matches.append(row)
                if matches:
                    break
                page.wait_for_timeout(100)
            if len(matches) != 1:
                return self._result("recipient_ambiguous", "The owned browser could not resolve one exact recipient. No message was submitted.")
            if not self._header_matches(page, contact):
                matches[0].click()
            deadline = time.monotonic() + 5.0
            while not self._header_matches(page, contact) and time.monotonic() < deadline:
                page.wait_for_timeout(100)
            if not self._header_matches(page, contact):
                return self._result("recipient_unverified", "The selected browser conversation does not match the requested recipient.")
        composer = page.locator('#main footer [contenteditable="true"][role="textbox"]')
        if composer.count() != 1:
            return self._result("unsupported_layout", "The owned WhatsApp browser has no unique readable message composer.")
        existing = composer.inner_text()
        if existing.strip("\r\n"):
            return self._result("draft_conflict", "An existing draft in ORION's messaging profile was preserved.")
        before = self._outgoing_ids(page, body)
        composer.fill(body)
        if composer.inner_text() != body or not self._header_matches(page, contact):
            return self._result("draft_unverified", "The exact browser recipient/message could not be verified before sending.")
        send = page.locator("#main footer").get_by_role("button", name=re.compile(r"^send(?: message)?$", re.I))
        if send.count() != 1 or not send.is_enabled():
            return self._result("unsupported_layout", "The message is drafted in ORION's owned browser, but one Send button was not found.")
        if not self._begin_submit():
            return self._result("cancelled_before_submit", "The messaging task ended before submission. No message was sent.")
        send.click()  # One boundary. No alternate click, Enter or route retry.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if (self._header_matches(page, contact) and not composer.inner_text().strip("\r\n")
                    and bool(self._outgoing_ids(page, body) - before)):
                return self._result("submitted", "The owned WhatsApp browser shows the new outgoing message. Delivery/read status is not confirmed.",
                                    ok=True, attempted=True, sent=True)
            page.wait_for_timeout(100)
        return self._result("uncertain", "Send was invoked once but a fresh outgoing message was not verified. Check the conversation before sending again.",
                            attempted=True, sent=None)

    def close(self, *, wait_s: float = 5.0) -> ToolResult:
        def stop():
            self._close_context()
            if self._playwright is not None:
                self._playwright.stop()
            self._context = self._playwright = self._page = None
            return self._result("closed", "The owned messaging browser is closed.", ok=True)
        with self._state_lock:
            if self._worker is None:
                self._closing = True
                return self._result("closed", "The owned messaging browser was not started.", ok=True)
            self._closing = True
            self._abort_before_submit.set()
            future = self._close_future
            if future is None:
                future = self._submit(stop, closing=True)
                self._close_future = future
        try:
            return future.result(timeout=max(.01, min(5.0, wait_s)))
        except WorkerTimeout:
            return self._result("closing", "The browser owner will close its profile when the in-flight operation finishes. No further sends will start.")
        except Exception:
            return self._result("close_failed", "The owned browser did not confirm a clean profile close.")


class OwnedDiscordMessenger(OwnedBrowserMessenger):
    """Personal Discord DMs through its separately connected web profile."""
    PLATFORMS = frozenset({"discord"})
    HOME_URL = "https://discord.com/channels/@me"
    READY_SELECTOR = '[data-list-id="private-channels"]'
    COMPOSER = '[role="textbox"][contenteditable="true"][data-slate-editor="true"]'

    @staticmethod
    def _account_identity(page) -> str | None:
        # Read the visibly signed-in account's own User area. No cookies,
        # storage, app internals or account tokens are inspected.
        identity = page.locator('section[aria-label="User area"] [class*="username"]')
        if identity.count() != 1:
            return None
        value = identity.inner_text().strip()
        return value if value else None

    @staticmethod
    def _account_id(page) -> str | None:
        """Read one stable account ID from the visible account area only."""
        nodes = page.locator('section[aria-label="User area"][data-user-id], section[aria-label="User area"] [data-user-id]')
        identities = set()
        for index in range(nodes.count()):
            value = nodes.nth(index).get_attribute("data-user-id") or ""
            if not re.fullmatch(r"[1-9][0-9]{16,19}", value):
                return None
            identities.add(value)
        return next(iter(identities)) if len(identities) == 1 else None

    @staticmethod
    def _header_matches(page, contact: str) -> bool:
        composer = page.locator(OwnedDiscordMessenger.COMPOSER)
        if composer.count() != 1:
            return False
        return (composer.get_attribute("aria-label") or "").strip().casefold() == "message @" + contact.strip().casefold()

    @staticmethod
    def _outgoing_ids(page, body: str) -> set[str]:
        # We are in an exact verified DM. Match fresh message IDs, exact body,
        # and a self-sender marker; an incoming identical message is insufficient.
        self_id = OwnedDiscordMessenger._account_id(page)
        if not self_id:
            return set()
        messages = page.locator('[id^="chat-messages-"]')
        result = set()
        for index in range(messages.count()):
            row = messages.nth(index)
            content = row.locator('[id^="message-content-"]')
            author = row.locator('[id^="message-username-"]')
            if author.count() == 0:
                # Grouped consecutive messages can reference a preceding author
                # through aria-labelledby. Follow that explicit relation only;
                # never infer sender from an arbitrary prior message.
                references = (row.get_attribute("aria-labelledby") or "").split()
                authors = [ref for ref in references if re.fullmatch(r"message-username-[A-Za-z0-9_-]+", ref)]
                if len(authors) == 1:
                    author = page.locator('[id="' + authors[0] + '"]')
            # Visible names can collide. Bind the row's explicit author ID or
            # its explicitly linked username node to the stable own account ID.
            # Never search message content/mentions or infer a preceding sender.
            author_ids = set()
            values = [row.get_attribute("data-author-id")]
            if author.count() == 1:
                values.extend((author.get_attribute("data-user-id"), author.get_attribute("data-author-id")))
            invalid = False
            for value in values:
                if not value:
                    continue
                if not re.fullmatch(r"[1-9][0-9]{16,19}", value):
                    invalid = True
                    break
                author_ids.add(value)
            if (not invalid and author_ids == {self_id} and content.count() == 1 and content.inner_text() == body):
                identity = row.get_attribute("id")
                if identity:
                    result.add(identity)
        return result

    def _send(self, contact: str, body: str) -> ToolResult:
        if not contact.strip() or not body.strip():
            return self._result("invalid_request", "A recipient and a nonempty exact message are required.")
        if contact.casefold().strip() in {"self", "myself", "me", "self chat"}:
            return self._result("unsupported_recipient", "Discord does not provide a self DM. No message was submitted.")
        page, blocked = self._ready()
        if blocked:
            return blocked
        if self._account_identity(page) is None:
            return self._result("unsupported_layout", "Discord's visible signed-in account identity could not be read. No draft or message was submitted.")
        # Resolve unique identity independently of the currently open composer.
        links = page.locator('[data-list-id="private-channels"] a[href^="/channels/@me/"]')
        choices = self._exact(links, contact)
        if len(choices) != 1:
            return self._result("recipient_ambiguous", "The owned Discord browser needs one exact saved DM recipient. No name was guessed.")
        if not self._header_matches(page, contact):
            choices[0].click()
            deadline = time.monotonic() + 5.0
            while not self._header_matches(page, contact) and time.monotonic() < deadline:
                page.wait_for_timeout(100)
            if not self._header_matches(page, contact):
                return self._result("recipient_unverified", "Discord's exact recipient composer could not be verified.")
        composer = page.locator(self.COMPOSER)
        if composer.inner_text().strip("\r\n"):
            return self._result("draft_conflict", "An existing draft in ORION's Discord profile was preserved.")
        if page.locator('[id^="chat-messages-"] [id^="message-username-"]').count() == 0:
            return self._result("unsupported_layout", "Discord message authors are not exposed for outgoing verification. No draft or message was submitted.")
        before = self._outgoing_ids(page, body)
        composer.fill(body)
        if composer.inner_text() != body or not self._header_matches(page, contact):
            return self._result("draft_unverified", "The exact Discord recipient and body could not be verified before sending.")
        if not self._begin_submit():
            return self._result("cancelled_before_submit", "The messaging task ended before submission. No message was sent.")
        composer.press("Enter")  # DOM event in this headless page, never OS input.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if (self._header_matches(page, contact) and not composer.inner_text().strip("\r\n")
                    and bool(self._outgoing_ids(page, body) - before)):
                return self._result("submitted", "The owned Discord browser shows a fresh outgoing message. Delivery/read status is not confirmed.",
                                    ok=True, attempted=True, sent=True)
            page.wait_for_timeout(100)
        return self._result("uncertain", "Discord submission was invoked once but its new outgoing message was not verified. Check before sending again.",
                            attempted=True, sent=None)


class OwnedBrowserMessagingRouter:
    """Lazily own a distinct profile, thread and context for each platform."""
    PLATFORMS = frozenset({"whatsapp", "discord"})

    def __init__(self, bus: Any) -> None:
        self.bus = bus
        self._services = {}
        self._lock = threading.Lock()

    def _get(self, platform):
        with self._lock:
            if platform not in self.PLATFORMS:
                return None
            existing = self._services.get(platform)
            if (existing is not None and existing._closing
                    and (existing._worker is None or not existing._worker.is_alive())):
                self._services.pop(platform, None)
            if platform not in self._services:
                cls = OwnedDiscordMessenger if platform == "discord" else OwnedBrowserMessenger
                self._services[platform] = cls(self.bus)
            return self._services[platform]

    def send(self, platform, contact, body):
        service = self._get(platform)
        return service.send(platform, contact, body) if service else ToolResult("Owned browser platform unavailable.", ok=False)

    def status(self, platform):
        service = self._get(platform)
        return service.status(platform) if service else ToolResult("Owned browser platform unavailable.", ok=False)

    def readiness(self, platform):
        service = self._get(platform)
        return service.readiness(platform) if service else ToolResult("Owned browser platform unavailable.", ok=False)

    def connect(self, platform):
        service = self._get(platform)
        return service.connect(platform) if service else ToolResult("Owned browser platform unavailable.", ok=False)

    def release_when_idle(self, callback) -> bool:
        with self._lock:
            services = list(self._services.values())
        # A seed keeps completion from firing while callbacks are registered.
        gate = threading.Lock()
        remaining, registered = [1], [False]
        def completed():
            with gate:
                remaining[0] -= 1
                done = remaining[0] == 0 and registered[0]
            if done:
                callback()
        for service in services:
            with gate:
                remaining[0] += 1
            if service.release_when_idle(completed):
                with gate:
                    registered[0] = True
            else:
                completed()
        completed()
        return registered[0]

    def close(self):
        with self._lock:
            services = list(self._services.values())
        outcomes = [service.close(wait_s=2.0) for service in services]
        return ToolResult("Owned messaging browser close: " + "; ".join(result.text for result in outcomes),
                          ok=all(result.ok for result in outcomes))
