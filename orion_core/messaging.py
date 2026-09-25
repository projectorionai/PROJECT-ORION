"""
Real-time messaging gateway for outbound alerts.

The service intentionally uses lightweight, dependency-tolerant routing. It can
attempt browser-driven delivery for WhatsApp and Telegram when a local browser
session is available, otherwise it reports a graceful fallback result.
"""

from __future__ import annotations

import os
import webbrowser
from typing import Any
from urllib.parse import quote

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .browser import open_url

# Truthy tokens for the dry-run environment switches.
_TRUE = {"1", "true", "yes", "on"}


class MessagingGateway:
    """Route outbound notifications through hosted messaging channels."""

    _ENDPOINTS = {
        "whatsapp": ("WhatsApp", "https://wa.me/{contact}?text={message}"),
        "telegram": ("Telegram", "https://t.me/{contact}?text={message}"),
    }

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus

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

    def send_text(self, platform: str, contact: str, message: str) -> ToolResult:
        try:
            platform = self._safe_text(platform).lower()
            contact = self._safe_text(contact)
            message = self._safe_text(message)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if not platform or not contact or not message:
            return ToolResult("Platform, contact and message are all required.", ok=False)

        endpoint = self._ENDPOINTS.get(platform)
        label = endpoint[0] if endpoint else platform.title()

        # Rejected before anything else, including the dry run. Simulating a
        # send to a platform ORION cannot reach reports success for something
        # that could never have happened.
        if endpoint is None and platform not in self._BOT_PLUGINS:
            return ToolResult(
                f"Unsupported messaging platform: {platform}. ORION can reach "
                + ", ".join(sorted(set(self._ENDPOINTS) | set(self._BOT_PLUGINS)))
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
            self.bus.log.emit(f"MESSAGING: {label} sent to {contact}.")
            return delivered

        if endpoint is None:
            # A bot-only platform with no token configured. There is no
            # compose link to fall back to, so say what is missing rather
            # than opening something unrelated.
            return ToolResult(
                f"{label} needs a bot token before ORION can send anything — "
                f"add one under \"{platform}\" in config/messaging.json.",
                ok=False)
        url = endpoint[1].format(contact=quote(contact, safe=""),
                                 message=quote(message))
        self.bus.log.emit(f"MESSAGING: {label} delivery request for {contact}.")
        open_url(url)
        return ToolResult(f"{label} alert queued for {contact}.")

    #: Platforms that can be delivered for real, and the plugin that does it.
    #: WhatsApp is absent on purpose: it has no free bot API for messaging
    #: arbitrary contacts, so the compose link genuinely is the best available.
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
        if not self._bot_token(section):
            return None
        try:
            import importlib
            import sys
            from .constants import CONFIG_DIR

            tools = str(CONFIG_DIR / "custom_tools")
            if tools not in sys.path:
                sys.path.append(tools)
            module = importlib.import_module(module_name)
            reply = str(module.run(action="send", to=contact, message=message))
        except Exception as exc:
            return ToolResult(
                f"{platform.title()} delivery failed ({exc}).", ok=False)
        lowered = reply.lower()
        failed = any(word in lowered for word in
                     ("failed", "could not", "not configured", "refused",
                      "no such", "unknown"))
        return ToolResult(reply, ok=not failed)

    # ── email ────────────────────────────────────────────────────────────────

    #: Where a Gmail compose window opens, pre-filled. This is a FALLBACK: it
    #: puts the message in front of the user with everything typed, and they
    #: press send. It is not delivery and is never reported as delivery.
    _GMAIL_COMPOSE = ("https://mail.google.com/mail/?view=cm&fs=1"
                      "&to={to}&su={subject}&body={body}")

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
        self.bus.log.emit(f"MESSAGING: opened a Gmail draft for {to}.")
        open_url(url)
        return ToolResult(
            f"I could not send that myself — no mail account is configured — "
            f"so I have opened a Gmail draft to {to} with the subject and "
            f"message already written. It is waiting for you to press send. "
            f"To let me send directly, add an app password under \"email\" in "
            f"config/messaging.json.")

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
        port = int(account.get("port") or 587)
        try:
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
                server.send_message(message)
        except Exception as exc:
            self.bus.log.emit(f"MESSAGING: email to {to} failed - {exc}")
            return ToolResult(
                f"I could not send that email: {exc}. The account is "
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
        import json
        import os

        from .constants import CONFIG_DIR

        address = os.getenv("ORION_EMAIL_ADDRESS", "").strip()
        password = os.getenv("ORION_EMAIL_APP_PASSWORD", "").strip()
        if address and password:
            return {"address": address, "password": password}
        try:
            data = json.loads(
                (CONFIG_DIR / "messaging.json").read_text(encoding="utf-8"))
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
        import json
        import os

        from .constants import CONFIG_DIR

        env = os.getenv(f"{section.upper()}_BOT_TOKEN", "").strip()
        if env:
            return env
        try:
            data = json.loads(
                (CONFIG_DIR / "messaging.json").read_text(encoding="utf-8"))
            return str((data.get(section) or {}).get("bot_token") or "").strip()
        except Exception:
            return ""
