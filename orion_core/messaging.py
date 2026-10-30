"""
Real-time messaging gateway for outbound alerts.

The service intentionally uses lightweight, dependency-tolerant routing. It can
attempt browser-driven delivery for WhatsApp and Telegram when a local browser
session is available, otherwise it reports a graceful fallback result.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import os
import threading
from contextlib import nullcontext
from typing import Any
from urllib.parse import quote

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .browser import open_url
from .execution_policy import background_scope, foreground_authorized, require_foreground

# Truthy tokens for the dry-run environment switches.
_TRUE = {"1", "true", "yes", "on"}
BOT_TIMEOUT_S = 30.0


def _redacted(text: object, *secrets: str) -> str:
    value = str(text)
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        value = value.replace(secret, "[redacted]").replace(quote(secret, safe=""), "[redacted]")
    return value


def _one_delivery_at_a_time(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        # This lock belongs to the physical synchronous worker. Cancelling its
        # asyncio.to_thread await must not allow another send while HTTP/UIA
        # from the first submission is still running.
        if not self._delivery_lock.acquire(blocking=False):
            return ToolResult("Another outbound message is still in progress. Wait for its result and check any uncertain submission before trying again.",
                              ok=False, receipt={"kind": "messaging", "status": "pending", "sent": False})
        try:
            scope = nullcontext() if foreground_authorized() else background_scope("messaging")
            with scope:
                return method(self, *args, **kwargs)
        finally:
            browser = self._browser_sender
            retain = getattr(browser, "release_when_idle", None)
            # A browser wrapper may have timed out while its owner still runs.
            # Keep chat/email ownership until that physical operation finishes,
            # including when a subsequent caller would choose a different API.
            if not callable(retain) or not retain(self._delivery_lock.release):
                self._delivery_lock.release()
    return guarded


class MessagingGateway:
    """Route outbound notifications through hosted messaging channels."""

    _ENDPOINTS = {
        "whatsapp": ("WhatsApp", "https://wa.me/{contact}?text={message}"),
        "telegram": ("Telegram", "https://t.me/{contact}?text={message}"),
    }

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._desktop_sender = None
        self._browser_sender = None
        self._plugin_sender = None
        self._plugin_secret_ready = {}
        self._delivery_lock = threading.Lock()

    def bind_plugin_sender(self, sender: Any, secret_ready: dict[str, bool] | None = None) -> None:
        """The dispatcher's registered, enabled and vault-scoped invocation seam."""
        self._plugin_sender = sender
        # Concurrent admission may refresh another platform while the current
        # physical sender holds the delivery lock. Preserve each named route's
        # readiness rather than erasing a vault-only account in that race.
        if secret_ready is None:
            self._plugin_secret_ready = None
        elif self._plugin_secret_ready is None:
            self._plugin_secret_ready = dict(secret_ready)
        else:
            self._plugin_secret_ready.update(secret_ready)

    def bind_browser(self) -> None:
        if self._browser_sender is None:
            from .browser_messaging import OwnedBrowserMessagingRouter
            self._browser_sender = OwnedBrowserMessagingRouter(self.bus)

    def browser_status(self, platform: str = "whatsapp") -> ToolResult:
        self.bind_browser()
        return self._browser_sender.status(platform)

    def browser_check_connection(self, platform: str = "whatsapp") -> ToolResult:
        self.bind_browser()
        return self._browser_sender.readiness(platform)

    def browser_connect(self, platform: str = "whatsapp") -> ToolResult:
        self.bind_browser()
        return self._browser_sender.connect(platform)

    def close(self) -> ToolResult:
        if self._browser_sender is None:
            return ToolResult("No owned messaging browser is open.")
        return self._browser_sender.close()

    def bind_desktop(self, control: Any) -> None:
        """Attach the existing guarded desktop controller without opening apps."""
        if control is None or getattr(self._desktop_sender, "control", None) is control:
            return
        from .desktop_messaging import DesktopMessenger

        self._desktop_sender = DesktopMessenger(control, self.bus)

    @staticmethod
    def delivery_suppressed() -> bool:
        """True when outbound messages must be SIMULATED, never opened in a real
        browser window.  Guarantees a full-suite / self-test run never pops
        WhatsApp or Telegram tabs.  Triggers on:
          • ORION_MESSAGING_DRYRUN — the explicit switch for suite runners;
          • PYTEST_CURRENT_TEST    — set automatically while any pytest runs;
          • ORION_SELFTEST / ORION_TEST_MODE — ORION's own self-test flags.
        Genuine user-requested sends (none of these set) still deliver for real.
        """
        if os.getenv("ORION_MESSAGING_DRYRUN", "").strip().lower() in _TRUE:
            return True
        return bool(
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("ORION_SELFTEST", "").strip().lower() in _TRUE
            or os.getenv("ORION_TEST_MODE", "").strip().lower() in _TRUE
        )

    def _safe_text(self, text: str) -> str:
        try:
            return SecuritySanitiser.guard_text(text, "messaging.text")
        except SecurityViolation as exc:
            raise ValueError(str(exc)) from exc

    @_one_delivery_at_a_time
    def send_text(self, platform: str, contact: str, message: str) -> ToolResult:
        try:
            platform = self._safe_text(platform).lower().strip()
            contact = self._safe_text(contact).strip()
            message = self._safe_text(message)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not platform or not contact or not message.strip():
            return ToolResult("Platform, contact and message are all required.", ok=False)

        endpoint = self._ENDPOINTS.get(platform)
        label = endpoint[0] if endpoint else platform.title()

        # Rejected before anything else, including the dry run. Simulating a
        # send to a platform ORION cannot reach reports success for something
        # that could never have happened.
        desktop_platforms = getattr(self._desktop_sender, "PLATFORMS", ())
        if endpoint is None and platform not in self._BOT_PLUGINS and platform not in desktop_platforms:
            return ToolResult(
                f"Unsupported messaging platform: {platform}. ORION can reach "
                + ", ".join(sorted(set(self._ENDPOINTS) | set(self._BOT_PLUGINS) | set(desktop_platforms)))
                + ".", ok=False)

        # Checked before either delivery path. A simulated send that quietly
        # becomes a real one is the worst possible direction for this to fail.
        if self.delivery_suppressed():
            self.bus.log.emit(
                f"MESSAGING: {label} delivery simulated (dry-run) for {contact}.")
            return ToolResult(f"{label} alert simulated (dry-run) for {contact}.")

        # Send it properly if we can. Opening a compose window is a fallback,
        # not delivery: it leaves the message sitting in a browser tab waiting
        # for someone to press send. Tried BEFORE the endpoint lookup, because
        # some platforms — Discord — can be delivered to and have no compose
        # link at all, so requiring one first ruled them out entirely.
        delivered = self._send_via_bot(platform, contact, message)
        if delivered is not None:
            state = "sent to" if delivered.ok else "delivery failed for"
            self.bus.log.emit(f"MESSAGING: {label} {state} {contact}.")
            return delivered

        browser_blocked = None
        if self._browser_sender is not None and platform in self._browser_sender.PLATFORMS:
            browser = self._browser_sender.send(platform, contact, message)
            receipt = browser.receipt or {}
            # Only an explicit pre-submit readiness result permits trying the
            # pattern-only native route. All submitted/uncertain failures stop.
            if receipt.get("attempted") is not False or receipt.get("status") not in {
                    "needs_login", "unavailable", "profile_busy"}:
                return browser
            if self._desktop_sender is None:
                return browser
            browser_blocked = browser

        # A signed-in desktop app can reach saved names without bot credentials
        # or an address-book phone number. The adapter only sends after resolving
        # a unique exact recipient and observing the composer contents. Once it
        # starts, its outcome is terminal: a second path could duplicate a send.
        if self._desktop_sender is not None and platform in desktop_platforms:
            desktop = self._desktop_sender.send(platform, contact, message)
            if browser_blocked is not None and (desktop.receipt or {}).get("status") == "requires_foreground":
                return browser_blocked
            return desktop

        if endpoint is None:
            # A bot-only platform with no token configured. There is no
            # compose link to fall back to, so say what is missing rather
            # than opening something unrelated.
            return ToolResult(
                f"{label} needs a bot token before ORION can send anything — "
                f"add one under \"{platform}\" in config/messaging.json.",
                ok=False)
        if platform == "whatsapp":
            # wa.me needs a phone number: "wa.me/mum" was a dead link. A name
            # is looked up in ORION's contacts; an unknown one opens
            # WhatsApp's own picker with the message already written.
            number = self._whatsapp_number(contact)
            if not number:
                self.bus.log.emit(f"MESSAGING: opening WhatsApp picker for {contact}.")
                return self._open_draft(f"https://wa.me/?text={quote(message, safe='')}",
                    f"I don't have a number for {contact}, so WhatsApp is open with "
                    "your message written - choose them and press send.", platform)
            contact_label, contact = contact, number
        else:
            contact_label = contact
        url = endpoint[1].format(contact=quote(contact, safe=""),
                                 message=quote(message))
        self.bus.log.emit(f"MESSAGING: {label} delivery request for {contact_label}.")
        return self._open_draft(url, f"{label} is open with your message to {contact_label} - "
                                "press send to deliver it.", platform)

    @staticmethod
    def _open_draft(url: str, text: str, platform: str) -> ToolResult:
        blocked = require_foreground("open messaging compose window in the default browser")
        if blocked is not None:
            return blocked
        try:
            opened = open_url(url)
        except Exception:
            opened = False
        if opened is False:
            return ToolResult(f"Could not open the {platform.title()} compose window. No message was sent.",
                              ok=False, receipt={"kind": "messaging", "status": "failed", "sent": False})
        return ToolResult(text, receipt={"kind": "messaging", "platform": platform,
                                         "status": "draft", "sent": False})

    @staticmethod
    def _whatsapp_number(contact: str) -> str:
        """Digits for wa.me (international, no '+'), or '' if unknown."""
        import re
        from .telephony import CONTACTS_PATH, ContactBook, normalise_number
        digits = re.sub(r"\D", "", contact)
        if len(digits) >= 8:
            return normalise_number(contact).lstrip("+")
        try:
            from .constants import CONFIG_DIR
            return ContactBook(CONFIG_DIR / CONTACTS_PATH).resolve(contact).lstrip("+")
        except Exception:
            return ""

    #: Platforms that can be delivered for real, and the plugin that does it.
    #: Desktop delivery is a separate route for signed-in app accounts.
    _BOT_PLUGINS = {
        "telegram": ("telegram_message_tool", "telegram"),
        "discord": ("discord_message_tool", "discord"),
    }

    def _send_via_bot(self, platform: str, contact: str,
                      message: str) -> "ToolResult | None":
        """Deliver through a configured bot, or None to fall back to the link.

        None means "not configured", not "failed". A failure is reported as a
        failure — falling back to a browser tab after a genuine delivery error
        would hide the error behind something that looks like success.
        """
        plugin = self._BOT_PLUGINS.get(platform)
        if plugin is None:
            return None
        module_name, section = plugin
        name = module_name.removesuffix("_tool")
        configured = bool(self._bot_token(section) or (platform == "discord" and self._discord_webhook()))
        if not configured and self._plugin_secret_ready is not None and not self._plugin_secret_ready.get(name, False):
            return None
        if self._plugin_sender is None:
            return ToolResult("Registered messaging plugin invocation is unavailable.", ok=False,
                              receipt={"kind": "messaging", "status": "unavailable", "sent": False, "attempted": False})
        try:
            reply = self._plugin_sender(name, {"action": "send", "to": contact, "message": message})
        except Exception as exc:
            return ToolResult(
                f"{platform.title()} delivery stopped ({type(exc).__name__}); completion is unverified.", ok=False,
                receipt={"kind": "messaging", "status": "uncertain", "sent": None, "attempted": True})
        from .plugin_results import plugin_result
        if isinstance(reply, str) and not reply.strip():
            return ToolResult("The bot returned no delivery result; delivery was not verified.", ok=False)
        result = plugin_result(reply)
        receipt = result.receipt or {}
        if (receipt.get("status") == "not_configured" and receipt.get("sent") is False
                and receipt.get("attempted") is False):
            return None
        token = self._bot_token(section)
        webhook = self._discord_webhook() if platform == "discord" else ""
        lowered = result.text.lower()
        failed = any(word in lowered for word in
                     ("failed", "could not", "not configured", "refused",
                      "no such", "unknown", "don't know", "will not open"))
        # Legacy string plugins also return questions and configuration guidance.
        # A normally returned string alone cannot establish a successful send.
        import re
        sent = bool(re.match(r"^(?:successfully\s+)?(?:sent|delivered)\b|^message\s+(?:was\s+)?(?:sent|delivered)\b", lowered.strip()))
        failed = failed or bool(re.search(r"\b(?:not|never|wasn't|weren't)\s+(?:sent|delivered)\b", lowered))
        if receipt and ("status" in receipt or "sent" in receipt):
            sent = (receipt.get("sent") is True or receipt.get("status") in {"submitted", "sent", "delivered"})
            failed = failed or receipt.get("sent") is False or receipt.get("status") in {"uncertain", "pending", "failed", "unavailable"}
        sender_kind = "configured_bot_or_webhook" if platform == "discord" else "configured_bot"
        sender_label = "Configured Discord bot/webhook API" if platform == "discord" else "Configured Telegram bot API"
        api_receipt = {**receipt, "kind": "messaging", "platform": platform,
                       "backend": "registered_plugin", "sender_kind": receipt.get("sender_kind") or sender_kind}
        accepted = result.ok and sent and not failed
        blocked_before_submission = (not result.ok and receipt.get("attempted") is not True
            and receipt.get("status") in {"requires_confirmation", "awaiting_confirmation", "requires_foreground", "disabled", "unavailable",
                                          "forbidden", "rejected", "blocked", "needs_isolation"})
        if blocked_before_submission or (not result.ok and receipt.get("attempted") is False):
            # Permission/admission refuses this invocation before transport. It
            # is terminal: another account/desktop path must not bypass it.
            api_receipt.update(attempted=False, sent=False)
        api_receipt.setdefault("status", "submitted" if accepted else "uncertain")
        api_receipt.setdefault("sent", True if accepted else None)
        api_receipt.setdefault("attempted", True)
        return ToolResult(sender_label + ": " + _redacted(result.text, token, webhook),
                          ok=accepted, receipt=api_receipt)

    @staticmethod
    def _discord_webhook() -> str:
        from .constants import CONFIG_DIR

        configured = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if configured:
            return configured
        try:
            from .key_seal import load_json        # sealed to the Windows account
            data = load_json(CONFIG_DIR / "messaging.json")
            return str((data.get("discord") or {}).get("webhook_url") or "").strip()
        except (OSError, ValueError, TypeError, AttributeError):
            return ""

    # ── email ────────────────────────────────────────────────────────────────

    #: Where a Gmail compose window opens, pre-filled. This is a FALLBACK: it
    #: puts the message in front of the user with everything typed, and they
    #: press send. It is not delivery and is never reported as delivery.
    _GMAIL_COMPOSE = ("https://mail.google.com/mail/?view=cm&fs=1"
                      "&to={to}&su={subject}&body={body}")

    @_one_delivery_at_a_time
    def send_email(self, to: str, subject: str, body: str) -> ToolResult:
        """Email *to*, for real when ORION has credentials, else in a browser.

        ORION's only mail path was a COM bridge to the local Outlook client,
        so when Outlook misbehaves he cannot email anybody at all. Two routes
        replace that, in order of how much they actually accomplish:

        * SMTP, when an account is configured. This genuinely sends.
        * A pre-filled Gmail compose window otherwise. Nothing is configured,
          nothing can fail to authenticate, and the message is there with the
          recipient, subject and body already typed — the user presses send.

        The distinction is kept in the wording of the result, because "I have
        emailed them" and "I have written it and it is waiting for you" are
        very different claims to make to somebody who then walks away.
        """
        try:
            to = self._safe_text(to)
            subject = self._safe_text(subject)
            body = self._safe_text(body)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not to.strip():
            return ToolResult("An email needs a recipient.", ok=False)
        if not subject.strip() and not body.strip():
            return ToolResult("An email needs a subject or a body.", ok=False)

        if self.delivery_suppressed():
            self.bus.log.emit(f"MESSAGING: email delivery simulated for {to}.")
            return ToolResult(f"Email to {to} simulated (dry-run).")

        sent = self._send_via_smtp(to, subject, body)
        if sent is not None:
            return sent

        url = self._GMAIL_COMPOSE.format(to=quote(to, safe=""),
                                         subject=quote(subject),
                                         body=quote(body))
        self.bus.log.emit(f"MESSAGING: opening a Gmail draft for {to}.")
        return self._open_draft(url,
            f"I could not send that myself — no mail account is configured — "
            f"so I have opened a Gmail draft to {to} with the subject and "
            f"message already written. It is waiting for you to press send. "
            f"To let me send directly, add an app password under \"email\" in "
            f"config/messaging.json.", "email")

    def _send_via_smtp(self, to: str, subject: str, body: str) -> "ToolResult | None":
        """Genuinely send. None means "not configured", never "failed".

        A failure is reported as a failure: falling back to a browser draft
        after a real delivery error would hide the error behind something that
        looks like progress, and the user would believe the mail had gone.
        """
        account = self._email_account()
        if not account:
            return None
        address = account.get("address", "")
        password = account.get("password", "")
        host = account.get("host") or "smtp.gmail.com"
        try:
            port = int(account.get("port") or 587)
            import smtplib
            from email.message import EmailMessage

            message = EmailMessage()
            message["From"] = address
            message["To"] = to
            message["Subject"] = subject or "(no subject)"
            message.set_content(body or "")
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.starttls()
                server.login(address, password)
                refused = server.send_message(message)
                if refused:
                    self.bus.log.emit(f"MESSAGING: email to {to} was only partially accepted; some recipients were refused.")
                    return ToolResult("The mail server refused some recipients. Other recipients may have been accepted; do not retry automatically.", ok=False)
        except Exception as exc:
            detail = _redacted(exc, password)
            self.bus.log.emit(f"MESSAGING: email to {to} failed - {detail}")
            return ToolResult(
                f"I could not send that email: {detail}. The account is "
                f"configured, so this is a delivery problem rather than a "
                f"missing setting.", ok=False)
        self.bus.log.emit(f"MESSAGING: email sent to {to}.")
        return ToolResult(f"Email sent to {to}.")

    @staticmethod
    def _email_account() -> dict[str, str]:
        """The configured sending account, or {} if there is none.

        An app password, not the account password: Google refuses a plain
        password over SMTP, and an app password can be revoked on its own
        without touching the account.
        """
        import os

        from .constants import CONFIG_DIR

        address = os.getenv("ORION_EMAIL_ADDRESS", "").strip()
        password = os.getenv("ORION_EMAIL_APP_PASSWORD", "").strip()
        if address and password:
            return {"address": address, "password": password}
        try:
            from .key_seal import load_json        # sealed to the Windows account
            data = load_json(CONFIG_DIR / "messaging.json")
            section = data.get("email") or {}
            address = str(section.get("address") or "").strip()
            password = str(section.get("app_password") or "").strip()
            if address and password:
                return {
                    "address": address,
                    "password": password,
                    "host": str(section.get("host") or "").strip(),
                    "port": section.get("port") or 0,
                }
        except Exception:
            pass
        return {}

    @staticmethod
    def _bot_token(section: str) -> str:
        """The configured bot token for a platform, or "" if there is none."""
        import os

        from .constants import CONFIG_DIR

        env = os.getenv(f"{section.upper()}_BOT_TOKEN", "").strip()
        if env:
            return env
        try:
            from .key_seal import load_json        # sealed to the Windows account
            data = load_json(CONFIG_DIR / "messaging.json")
            return str((data.get(section) or {}).get("bot_token") or "").strip()
        except Exception:
            return ""
