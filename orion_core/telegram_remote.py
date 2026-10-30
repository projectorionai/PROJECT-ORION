"""
Talk to ORION from Telegram - the jv_telegram_remote plugin, rebuilt into
ORION (Mark XXXIX).

The plugin ("explicit Telegram polling for permitted private chats only")
was removed on 26 Oct 2026: its module had been quarantined as broken
("import did not complete within 8s - module-level code is blocking") and
nothing built in replaced it - ``messaging`` only SENDS. This is that
capability, done properly:

  * It polls the Bot API (getUpdates, long-poll) - no webhook, no open port.
  * OFF unless configured: a bot token AND an allow-list of chat ids
    (config/messaging.json ``"telegram": {"remote": {"enabled": true,
    "chats": [123456789]}}``, or TELEGRAM_REMOTE_CHATS="123,456"). Messages
    from anyone else, from groups or channels, edited messages and anything
    older than MAX_AGE_S (sent while ORION was off) are ignored.
  * A message is answered as a remote turn through the phone gateway's own
    path (RemoteGateway.answer_text): the same capability gate - desktop-only
    tools refused, anything irreversible parked for approval - the same
    history and the same memory.
  * Approvals with buttons (Mark XXXX.3): an action that needs approving is
    sent back with Approve and Decline buttons. A tap counts only from the
    allowed private chat that asked, carries the confirmation's single-use
    token, and expires with it (180 s); phone control off still stops it.
    Before this, an approval asked for over Telegram was filed under a
    device no phone app or desk ever listed, so it could never be given.
  * Bounded: one message per MIN_INTERVAL_S per chat, MAX_TEXT characters in,
    TELEGRAM_MAX_REPLY out; the bot token never reaches a log line.

Blocking HTTP runs off the event loop; the loop itself only awaits.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Awaitable, Callable

from .message_channels import TELEGRAM_API, MAX_RESPONSE, _redact, telegram_config

POLL_TIMEOUT_S = 25
#: Messages sent while ORION was off are not acted on when he comes back.
MAX_AGE_S = 600.0
MAX_TEXT = 2000
TELEGRAM_MAX_REPLY = 3900
MIN_INTERVAL_S = 2.0
#: After a failed poll, wait this long before the next (doubling to the cap).
BACKOFF_S = (5.0, 300.0)

_APPROVAL_RE = re.compile(r"I've sent it to your phone to approve[^.]*\.?", re.I)
APPROVAL_NOTE = ("It needs your approval, and I don't take approvals over Telegram - "
                 "approve it in the ORION app on your phone or ask me at the desk.")
BUTTON_NOTE = "It needs your approval - use the buttons below."
#: callback_data: "<ok|no>:<confirmation id>:<token>" (Telegram allows 64 bytes).
_CALLBACK_RE = re.compile(r"^(ok|no):([A-Za-z0-9_\-]{4,24}):([A-Za-z0-9_\-]{8,40})$")

Answer = Callable[[str, str], Awaitable[str]]
#: chat id -> the confirmations waiting for that chat ({id, token, summary}).
Pending = Callable[[str], list[dict[str, Any]]]
#: (chat id, confirmation id, token, approve) -> what happened, in a sentence.
Decide = Callable[[str, str, str, bool], Awaitable[str]]


def remote_config(config: dict[str, Any] | None = None) -> tuple[str, frozenset[str]]:
    """(bot token, allowed chat ids) - both empty when the remote is off."""
    section = config if config is not None else telegram_config()
    token = str(section.get("bot_token") or "").strip()
    remote = section.get("remote") if isinstance(section.get("remote"), dict) else {}
    chats = {str(c).strip() for c in (remote.get("chats") or []) if str(c).strip()}
    env = os.getenv("TELEGRAM_REMOTE_CHATS", "").strip()
    if env:
        chats |= {c.strip() for c in env.split(",") if c.strip()}
    enabled = bool(remote.get("enabled")) or bool(env)
    chats = {c for c in chats if c.lstrip("-").isdigit()}
    if not (token and enabled and chats):
        return "", frozenset()
    return token, frozenset(chats)


def _http_json(url: str, payload: dict[str, Any] | None = None,
               timeout: float = POLL_TIMEOUT_S + 10) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310 (fixed https host)
        body = response.read(MAX_RESPONSE * 4).decode("utf-8", "replace")
    parsed = json.loads(body or "{}")
    return parsed if isinstance(parsed, dict) else {}


class TelegramRemote:
    """The poller. ``run()`` until *stop* is set; ``handle()`` one update."""

    def __init__(self, answer: Answer, *, log: Callable[[str], None] | None = None,
                 token: str = "", chats: frozenset[str] = frozenset(),
                 http: Callable[..., dict[str, Any]] = _http_json,
                 clock: Callable[[], float] = time.time,
                 pending: Pending | None = None, decide: Decide | None = None) -> None:
        self._answer = answer
        self._pending = pending
        self._decide = decide
        self._offered: set[str] = set()
        self._log = log or (lambda _line: None)
        self.token = token
        self.chats = chats
        self._http = http
        self._clock = clock
        self._offset = 0
        self._last_at: dict[str, float] = {}
        self._ignored: set[str] = set()

    @classmethod
    def from_config(cls, answer: Answer, *, log: Callable[[str], None] | None = None,
                    pending: Pending | None = None, decide: Decide | None = None
                    ) -> "TelegramRemote | None":
        token, chats = remote_config()
        if not token:
            return None
        return cls(answer, log=log, token=token, chats=chats, pending=pending, decide=decide)

    @property
    def buttons(self) -> bool:
        """Can approvals be given here?"""
        return self._pending is not None and self._decide is not None

    def _say(self, line: str) -> None:
        self._log(_redact(line, self.token))

    def _url(self, method: str, **query: Any) -> str:
        url = TELEGRAM_API.format(token=self.token, method=method)
        return url + ("?" + urllib.parse.urlencode(query) if query else "")

    # ── one poll ─────────────────────────────────────────────────────────────

    def fetch(self) -> list[dict[str, Any]]:
        """Blocking getUpdates (long poll). Raises on a transport failure."""
        reply = self._http(self._url("getUpdates", offset=self._offset, timeout=POLL_TIMEOUT_S,
                                     allowed_updates=json.dumps(["message", "callback_query"])))
        if not reply.get("ok"):
            raise RuntimeError(str(reply.get("description") or "getUpdates refused")[:160])
        updates = [u for u in reply.get("result") or [] if isinstance(u, dict)]
        for update in updates:
            try:
                self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
            except (TypeError, ValueError):
                continue
        return updates

    def _accept(self, update: dict[str, Any]) -> tuple[str, str] | None:
        """(chat id, text) for a message ORION should answer, else None."""
        message = update.get("message")
        if not isinstance(message, dict):
            return None                                 # edits, channel posts, callbacks
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = str(chat.get("id") or "")
        if chat.get("type") != "private" or chat_id not in self.chats:
            if chat_id and chat_id not in self._ignored:
                self._ignored.add(chat_id)
                self._say(f"TELEGRAM: ignored a message from a chat that is not allowed ({chat_id}).")
            return None
        sender = message.get("from") if isinstance(message.get("from"), dict) else {}
        if sender.get("is_bot"):
            return None
        try:
            sent = float(message.get("date") or 0)
        except (TypeError, ValueError):
            return None
        if self._clock() - sent > MAX_AGE_S:
            return None                                 # sent while he was off
        text = str(message.get("text") or "").strip()
        if not text:
            return None
        return chat_id, text[:MAX_TEXT]

    async def handle(self, update: dict[str, Any]) -> str | None:
        """Answer one update; returns the reply sent (None if ignored)."""
        if isinstance(update.get("callback_query"), dict):
            return await self.handle_tap(update["callback_query"])
        accepted = self._accept(update)
        if accepted is None:
            return None
        chat_id, text = accepted
        now = self._clock()
        if now - self._last_at.get(chat_id, 0.0) < MIN_INTERVAL_S:
            reply = "One at a time, please - I'm still on the last one."
        else:
            self._last_at[chat_id] = now
            if text.lower() in {"/start", "/help"}:
                reply = ("ORION here. Write to me as you would at the desk - I'll answer, "
                         "and do what's safe from afar. Anything that needs approving waits "
                         "for you at the desk or in the phone app.")
            else:
                try:
                    reply = await self._answer(text, chat_id)
                except Exception as exc:
                    reply = f"I couldn't answer that just now ({type(exc).__name__})."
            note = BUTTON_NOTE if self.buttons else APPROVAL_NOTE
            reply = _APPROVAL_RE.sub(note, str(reply or "")).strip() or "Done."
        reply = reply[:TELEGRAM_MAX_REPLY]
        await self._send(chat_id, reply)
        self._say("TELEGRAM: answered a message from your allowed chat.")
        await self.offer_approvals(chat_id)
        return reply

    async def _send(self, chat_id: str, text: str, markup: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text[:TELEGRAM_MAX_REPLY]}
        if markup:
            payload["reply_markup"] = markup
        try:
            await asyncio.to_thread(self._http, self._url("sendMessage"), payload, 20.0)
        except Exception as exc:
            self._say(f"TELEGRAM: could not send the reply - {type(exc).__name__}.")

    # ── approvals with buttons ───────────────────────────────────────────────

    async def offer_approvals(self, chat_id: str) -> int:
        """Send Approve / Decline buttons for each new confirmation this chat
        is waiting on. Returns how many were offered."""
        if not self.buttons:
            return 0
        try:
            waiting = self._pending(chat_id) or []
        except Exception:
            return 0
        offered = 0
        for conf in waiting:
            conf_id, token = str(conf.get("id") or ""), str(conf.get("token") or "")
            if not conf_id or not token or conf_id in self._offered:
                continue
            self._offered.add(conf_id)
            summary = str(conf.get("summary") or conf.get("tool") or "do this")
            text = f"Approve this? I would {summary}."
            detail = str(conf.get("parameterDiff") or "").strip()
            if detail:
                text += "\n\n" + detail[:1500]
            markup = {"inline_keyboard": [[
                {"text": "Approve", "callback_data": f"ok:{conf_id}:{token}"},
                {"text": "Decline", "callback_data": f"no:{conf_id}:{token}"},
            ]]}
            await self._send(chat_id, text, markup)
            offered += 1
        if offered:
            self._say(f"TELEGRAM: sent {offered} approval request(s) with buttons.")
        return offered

    def _accept_tap(self, tap: dict[str, Any]) -> tuple[str, str, str, bool, Any] | None:
        """(chat id, confirmation id, token, approve, message id) for a tap
        ORION should act on, else None."""
        sender = tap.get("from") if isinstance(tap.get("from"), dict) else {}
        message = tap.get("message") if isinstance(tap.get("message"), dict) else {}
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = str(chat.get("id") or "")
        if (chat.get("type") != "private" or chat_id not in self.chats
                or str(sender.get("id") or "") != chat_id or sender.get("is_bot")):
            return None
        match = _CALLBACK_RE.match(str(tap.get("data") or ""))
        if match is None:
            return None
        return chat_id, match.group(2), match.group(3), match.group(1) == "ok", message.get("message_id")

    async def handle_tap(self, tap: dict[str, Any]) -> str | None:
        """An Approve or Decline tap; returns what was reported (None if ignored)."""
        tap_id = str(tap.get("id") or "")
        accepted = self._accept_tap(tap) if self.buttons else None
        try:
            await asyncio.to_thread(self._http, self._url("answerCallbackQuery"),
                                    {"callback_query_id": tap_id}, 10.0)
        except Exception:
            pass
        if accepted is None:
            if tap_id:
                self._say("TELEGRAM: ignored a button tap that was not from your allowed chat.")
            return None
        chat_id, conf_id, token, approve, message_id = accepted
        try:
            outcome = await self._decide(chat_id, conf_id, token, approve)
        except Exception as exc:
            outcome = f"I couldn't do that just now ({type(exc).__name__})."
        outcome = str(outcome or ("Done." if approve else "Declined - nothing ran."))
        if message_id is not None:
            try:
                await asyncio.to_thread(self._http, self._url("editMessageReplyMarkup"),
                                        {"chat_id": chat_id, "message_id": message_id,
                                         "reply_markup": {"inline_keyboard": []}}, 10.0)
            except Exception:
                pass
        await self._send(chat_id, outcome)
        self._say(f"TELEGRAM: {'approved' if approve else 'declined'} a request from your allowed chat.")
        return outcome

    async def run(self, stop: asyncio.Event) -> None:
        """Poll until *stop* is set. Never raises; backs off on failures."""
        self._say(f"TELEGRAM: remote on for {len(self.chats)} allowed chat(s).")
        delay = BACKOFF_S[0]
        while not stop.is_set():
            try:
                updates = await asyncio.to_thread(self.fetch)
                delay = BACKOFF_S[0]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._say(f"TELEGRAM: poll failed - {str(exc)[:120]}; retrying in {delay:g} s.")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(BACKOFF_S[1], delay * 2)
                continue
            for update in updates:
                if stop.is_set():
                    break
                await self.handle(update)


__all__ = ["APPROVAL_NOTE", "BUTTON_NOTE", "MAX_AGE_S", "TelegramRemote", "remote_config"]
