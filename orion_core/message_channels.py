"""
ORION's own message channels - one gateway over the services' bot APIs
(Mark XXXVII.5).

Telegram, Discord, push notifications and IFTTT used to be four forged plugins
(telegram_message, discord_message, push_notify, ifttt_webhook), each with its
own configuration story, its own approval prompt and its own failure wording.
They are built in now, behind the one ``messaging`` tool, and every send waits
for the owner's on-screen approval (SystemActionGuard SEND_MESSAGE, the token
bound to the exact recipient and words). Nothing here automates an app or a
browser: each channel is the service's own HTTP API.

Configuration is where the plugins kept it, so nothing has to be entered again:

    config/messaging.json
        "telegram": {"bot_token", "default_chat_id", "contacts": {name: chat_id}}
        "discord":  {"webhook_url", "bot_token", "default_channel_id",
                     "contacts": {name: {"user_id" | "channel_id": ...}}}
        "ntfy":     {"topic", "server", "token"}
        "gotify":   {"url", "token"}
        "pushover": {"token", "user"}
    config/ifttt.json   {"key", "default_event"}

Environment variables win (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
DISCORD_BOT_TOKEN, DISCORD_WEBHOOK_URL, NTFY_TOPIC, NTFY_SERVER, NTFY_TOKEN,
GOTIFY_URL, GOTIFY_TOKEN, PUSHOVER_TOKEN, PUSHOVER_USER, IFTTT_KEY), then a
secret stored for the old plugin in ORION's plugin vault, then the file.

Blocking network calls: run them off the event loop.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .data import ToolResult

TIMEOUT_S = 15.0
#: Responses are read no further than this: a service never floods ORION.
MAX_RESPONSE = 64 * 1024

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
DISCORD_API = "https://discord.com/api/v10"
NTFY_DEFAULT = "https://ntfy.sh"
PUSHOVER_API = "https://api.pushover.net/1/messages.json"
IFTTT_API = "https://maker.ifttt.com/trigger/{event}/with/key/{key}"

#: The channels, and what people call them.
CHANNELS = ("telegram", "discord", "push", "ifttt")
ALIASES = {
    "telegram": "telegram", "discord": "discord",
    "push": "push", "notification": "push", "notifications": "push",
    "phone": "push", "my phone": "push", "phone notification": "push",
    "ntfy": "push", "gotify": "push", "pushover": "push",
    "ifttt": "ifttt", "webhook": "ifttt", "applet": "ifttt",
}
#: Push services, best authenticated first (ntfy's public topics are only as
#: private as their names).
PUSH_SERVICES = ("pushover", "gotify", "ntfy")
PRIORITY = {
    "min": {"ntfy": 1, "gotify": 0, "pushover": -2},
    "low": {"ntfy": 2, "gotify": 2, "pushover": -1},
    "normal": {"ntfy": 3, "gotify": 5, "pushover": 0},
    "high": {"ntfy": 4, "gotify": 7, "pushover": 1},
    "urgent": {"ntfy": 5, "gotify": 8, "pushover": 1},
}

#: The setup line each channel gives when it is not configured.
SETUP = {
    "telegram": ("Telegram: create a bot with @BotFather, then put its token under "
                 "\"telegram\": {\"bot_token\": ...} in config/messaging.json (or set "
                 "TELEGRAM_BOT_TOKEN). Each person must message the bot once first."),
    "discord": ("Discord: paste a channel webhook URL under \"discord\": {\"webhook_url\": ...} "
                "in config/messaging.json (Server Settings > Integrations > Webhooks), or a "
                "bot token for direct messages."),
    "push": ("Push: the quickest is ntfy - install the app, subscribe to a long, unguessable "
             "topic name and put it under \"ntfy\": {\"topic\": ...} in config/messaging.json. "
             "Pushover (token + user) and Gotify (url + token) are authenticated alternatives."),
    "ifttt": ("IFTTT: put your Webhooks key under {\"key\": ...} in config/ifttt.json (from "
              "ifttt.com/maker_webhooks > Documentation), then name the applet's event."),
}


def channel(name: Any) -> str:
    """The channel *name* refers to ('Telegram', 'my phone', 'ntfy'), or ''."""
    text = " ".join(str(name or "").lower().replace("_", " ").split())
    return ALIASES.get(text, "")


# ── configuration ────────────────────────────────────────────────────────────

def _file(name: str) -> dict[str, Any]:
    """A channel config file, its bot tokens and webhooks sealed to this
    Windows account on first read (key_seal.py) and opened for use."""
    from .constants import CONFIG_DIR
    from . import key_seal

    path = CONFIG_DIR / name
    try:
        key_seal.seal_file_in_place(path)
    except Exception:
        pass
    return key_seal.load_json(path)


def _vault(plugin: str, env: str) -> str:
    """A secret stored for the plugin this channel replaces, if any."""
    try:
        from .jv_plugins.runtime import secret
        return str(secret(plugin, env) or "").strip()
    except Exception:
        return ""


def _section(name: str, keys: tuple[tuple[str, str], ...], plugin: str = "",
             secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    data = _file("ifttt.json") if name == "ifttt" else _file("messaging.json").get(name) or {}
    section = dict(data) if isinstance(data, dict) else {}
    for key, env in keys:
        value = os.getenv(env, "").strip()
        if not value and plugin and key in secrets:
            value = _vault(plugin, env)
        if value:
            section[key] = value
    return section


def telegram_config() -> dict[str, Any]:
    return _section("telegram", (("bot_token", "TELEGRAM_BOT_TOKEN"),
                                 ("default_chat_id", "TELEGRAM_CHAT_ID")),
                    "telegram_message", ("bot_token",))


def discord_config() -> dict[str, Any]:
    return _section("discord", (("bot_token", "DISCORD_BOT_TOKEN"),
                                ("webhook_url", "DISCORD_WEBHOOK_URL")),
                    "discord_message", ("bot_token", "webhook_url"))


def push_config() -> dict[str, dict[str, Any]]:
    return {
        "ntfy": _section("ntfy", (("topic", "NTFY_TOPIC"), ("server", "NTFY_SERVER"),
                                  ("token", "NTFY_TOKEN")), "push_notify", ("token",)),
        "gotify": _section("gotify", (("url", "GOTIFY_URL"), ("token", "GOTIFY_TOKEN")),
                           "push_notify", ("token",)),
        "pushover": _section("pushover", (("token", "PUSHOVER_TOKEN"), ("user", "PUSHOVER_USER")),
                             "push_notify", ("token", "user")),
    }


def ifttt_config() -> dict[str, Any]:
    return _section("ifttt", (("key", "IFTTT_KEY"),), "ifttt_webhook", ("key",))


def _push_service(config: dict[str, dict[str, Any]]) -> str:
    for service in PUSH_SERVICES:
        section = config.get(service) or {}
        if service == "ntfy" and section.get("topic"):
            return service
        if service == "gotify" and section.get("url") and section.get("token"):
            return service
        if service == "pushover" and section.get("token") and section.get("user"):
            return service
    return ""


def configured(name: Any) -> bool:
    """Whether channel *name* has what it needs to send."""
    kind = channel(name)
    if kind == "telegram":
        return bool(telegram_config().get("bot_token"))
    if kind == "discord":
        config = discord_config()
        return bool(config.get("webhook_url") or config.get("bot_token"))
    if kind == "push":
        return bool(_push_service(push_config()))
    if kind == "ifttt":
        return bool(ifttt_config().get("key"))
    return False


def status() -> str:
    """One line per channel: ready (and how), or how to set it up."""
    lines = []
    for kind in CHANNELS:
        if not configured(kind):
            lines.append("- " + SETUP[kind])
            continue
        if kind == "push":
            lines.append(f"- Push: ready ({_push_service(push_config())}).")
        elif kind == "discord":
            config = discord_config()
            how = " and ".join(part for part, on in (("webhook", config.get("webhook_url")),
                                                     ("bot", config.get("bot_token"))) if on)
            lines.append(f"- Discord: ready ({how}).")
        elif kind == "telegram":
            contacts = sorted((telegram_config().get("contacts") or {}))
            lines.append("- Telegram: ready" + (f" (contacts: {', '.join(contacts)})." if contacts else "."))
        else:
            lines.append("- IFTTT: ready.")
    return "Message channels:\n" + "\n".join(lines)


# ── delivery ─────────────────────────────────────────────────────────────────

@dataclass
class _Reply:
    ok: bool
    status: int
    body: str


def _post(url: str, data: bytes, headers: dict[str, str]) -> _Reply:
    """POST *data*; never raises. Bodies are read only up to MAX_RESPONSE."""
    request = urllib.request.Request(url, data=data, method="POST")
    for key, value in {"User-Agent": "ORION/1.0", **headers}.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return _Reply(True, int(getattr(response, "status", 200) or 200),
                          response.read(MAX_RESPONSE).decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(MAX_RESPONSE).decode("utf-8", "replace")
        except Exception:
            body = ""
        return _Reply(False, int(exc.code), body)
    except urllib.error.URLError as exc:
        return _Reply(False, 0, f"could not reach the service ({exc.reason})")
    except Exception as exc:                       # pragma: no cover - defensive
        return _Reply(False, 0, f"{type(exc).__name__}")


def _json_post(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> _Reply:
    return _post(url, json.dumps(payload).encode("utf-8"),
                 {"Content-Type": "application/json", **(headers or {})})


def _redact(text: str, *secrets: Any) -> str:
    value = str(text)
    for secret in sorted({str(s) for s in secrets if s}, key=len, reverse=True):
        value = value.replace(secret, "[redacted]").replace(urllib.parse.quote(secret, safe=""), "[redacted]")
    return value


def _result(kind: str, ok: bool, text: str, *, attempted: bool = True,
            status: str = "", secrets: tuple[Any, ...] = ()) -> ToolResult:
    receipt = {"kind": "messaging", "platform": kind, "backend": "builtin_api",
               "attempted": attempted,
               "sent": True if ok else (False if not attempted else None),
               "status": status or ("submitted" if ok else ("not_configured" if not attempted
                                                            else "failed"))}
    return ToolResult(_redact(text, *secrets), ok=ok, receipt=receipt)


def _lookup(contacts: Any, name: str) -> Any:
    if not isinstance(contacts, dict) or not name:
        return None
    if name in contacts:
        return contacts[name]
    lowered = {str(k).lower(): v for k, v in contacts.items()}
    return lowered.get(name.lower())


def send_telegram(to: str, message: str) -> ToolResult:
    config = telegram_config()
    token = str(config.get("bot_token") or "").strip()
    if not token:
        return _result("telegram", False, SETUP["telegram"], attempted=False)
    target = str(to or "").strip()
    entry = _lookup(config.get("contacts"), target)
    if isinstance(entry, dict):
        entry = entry.get("chat_id")
    chat = str(entry or "") or (target if target.lstrip("-").isdigit() else "")
    if not target:
        chat = str(config.get("default_chat_id") or "")
    if not chat:
        known = ", ".join(sorted(config.get("contacts") or {})) or "none saved"
        return _result("telegram", False, f"I don't know '{target}' on Telegram (saved: {known}). "
                       "Add them under telegram.contacts in config/messaging.json.", attempted=False,
                       status="unknown_recipient")
    reply = _json_post(TELEGRAM_API.format(token=token, method="sendMessage"),
                       {"chat_id": chat, "text": message})
    try:
        payload = json.loads(reply.body or "{}")
    except ValueError:
        payload = {}
    if reply.ok and payload.get("ok"):
        return _result("telegram", True, f"Sent on Telegram to {target or 'your default chat'}.")
    detail = str(payload.get("description") or reply.body or "")[:200]
    if reply.status == 401:
        detail = "Telegram rejected the bot token (401)."
    elif "chat not found" in detail.lower():
        detail = "Telegram says that chat does not exist - they must message the bot once first."
    return _result("telegram", False, f"Telegram did not send it: {detail}", secrets=(token,))


def send_discord(to: str, message: str) -> ToolResult:
    config = discord_config()
    webhook = str(config.get("webhook_url") or "").strip()
    token = str(config.get("bot_token") or "").strip()
    target = str(to or "").strip()
    if not (webhook or token):
        return _result("discord", False, SETUP["discord"], attempted=False)
    secrets = (webhook, token)
    entry = _lookup(config.get("contacts"), target)
    # The webhook posts to the server's channel: only when that is what was
    # asked for - never in place of a message to a named person.
    use_webhook = webhook and (not target or target.lower() in {
        "server", "channel", "webhook", "the server", "the channel", "everyone"})
    if use_webhook:
        reply = _json_post(webhook, {"content": message[:2000]})
        if reply.ok:
            return _result("discord", True, "Posted to your Discord channel.")
        return _result("discord", False, f"Discord did not post it ({reply.status}).", secrets=secrets)
    if not token:
        return _result("discord", False, "Direct messages on Discord need a bot token in "
                       "config/messaging.json; only the channel webhook is set.", attempted=False,
                       status="not_configured")
    auth = {"Authorization": f"Bot {token}"}
    channel_id = ""
    if isinstance(entry, dict):
        channel_id = str(entry.get("channel_id") or "")
        user_id = str(entry.get("user_id") or "")
    else:
        user_id = str(entry or (target if target.isdigit() else ""))
    if not target:
        channel_id = str(config.get("default_channel_id") or "")
    if not channel_id and user_id:
        opened = _json_post(f"{DISCORD_API}/users/@me/channels", {"recipient_id": user_id}, auth)
        try:
            channel_id = str(json.loads(opened.body or "{}").get("id") or "") if opened.ok else ""
        except ValueError:
            channel_id = ""
    if not channel_id:
        return _result("discord", False, f"I don't know '{target}' on Discord. Add them under "
                       "discord.contacts in config/messaging.json.", attempted=False,
                       status="unknown_recipient")
    reply = _json_post(f"{DISCORD_API}/channels/{channel_id}/messages", {"content": message[:2000]}, auth)
    if reply.ok:
        return _result("discord", True, f"Sent on Discord to {target or 'your default channel'}.")
    why = "the bot token was rejected (401)" if reply.status == 401 else f"status {reply.status}"
    return _result("discord", False, f"Discord did not send it: {why}.", secrets=secrets)


def send_push(message: str, *, title: str = "", priority: str = "") -> ToolResult:
    config = push_config()
    service = _push_service(config)
    if not service:
        return _result("push", False, SETUP["push"], attempted=False)
    section = config[service]
    level = priority if priority in PRIORITY else ""
    secrets = tuple(section.get(k) for k in ("token", "user", "topic"))
    if service == "ntfy":
        server = str(section.get("server") or NTFY_DEFAULT).rstrip("/")
        headers = {"Content-Type": "text/plain; charset=utf-8"}
        body = message
        if title:
            try:
                title.encode("latin-1")
                headers["Title"] = title
            except UnicodeEncodeError:
                body = f"{title}\n\n{message}"
        if level:
            headers["Priority"] = str(PRIORITY[level]["ntfy"])
        if section.get("token"):
            headers["Authorization"] = f"Bearer {section['token']}"
        reply = _post(f"{server}/{urllib.parse.quote(str(section['topic']))}",
                      body.encode("utf-8"), headers)
    elif service == "gotify":
        payload: dict[str, Any] = {"message": message, "title": title or "ORION"}
        if level:
            payload["priority"] = PRIORITY[level]["gotify"]
        reply = _json_post(f"{str(section['url']).rstrip('/')}/message?token="
                           f"{urllib.parse.quote(str(section['token']))}", payload)
    else:
        fields = {"token": section["token"], "user": section["user"], "message": message}
        if title:
            fields["title"] = title
        if level:
            fields["priority"] = PRIORITY[level]["pushover"]
        reply = _post(PUSHOVER_API, urllib.parse.urlencode(fields).encode("utf-8"),
                      {"Content-Type": "application/x-www-form-urlencoded"})
    if reply.ok:
        return _result("push", True, f"{service.title()} accepted the notification "
                       "(the phone's delivery is not confirmed).")
    return _result("push", False, f"{service.title()} refused the notification "
                   f"({reply.status or reply.body}).", secrets=secrets)


def send_ifttt(event: str, message: str) -> ToolResult:
    config = ifttt_config()
    key = str(config.get("key") or "").strip()
    if not key:
        return _result("ifttt", False, SETUP["ifttt"], attempted=False)
    name = "".join(ch for ch in str(event or config.get("default_event") or "")
                   if ch.isalnum() or ch in "_-")
    if not name:
        return _result("ifttt", False, "Which IFTTT event should I trigger? Name the applet's "
                       "event (or set default_event in config/ifttt.json).", attempted=False,
                       status="unknown_recipient")
    reply = _json_post(IFTTT_API.format(event=name, key=urllib.parse.quote(key)),
                       {"value1": message[:1000]})
    if reply.ok:
        return _result("ifttt", True, f"Triggered the IFTTT event '{name}'.")
    why = "the key was rejected" if reply.status in (401, 403) else f"status {reply.status}"
    return _result("ifttt", False, f"IFTTT did not run '{name}': {why}.", secrets=(key,))


def send(name: Any, to: str, message: str, *, title: str = "", priority: str = "") -> ToolResult:
    """Deliver *message* on channel *name*. Blocking; never raises."""
    kind = channel(name)
    text = str(message or "").strip()
    if not kind:
        return ToolResult(f"There is no message channel called {name!r}.", ok=False)
    if not text:
        return _result(kind, False, "There is nothing to send.", attempted=False, status="empty")
    try:
        if kind == "telegram":
            return send_telegram(to, text)
        if kind == "discord":
            return send_discord(to, text)
        if kind == "push":
            return send_push(text, title=title, priority=str(priority or "").lower())
        return send_ifttt(to, text)
    except Exception as exc:                       # pragma: no cover - defensive
        return _result(kind, False, f"{kind.title()} delivery stopped ({type(exc).__name__}); "
                       "completion is unverified.", status="uncertain")


def describe_send(name: Any, to: str, message: str) -> str:
    """What the approval prompt says: the channel, who, and the exact words."""
    kind = channel(name)
    where = {"telegram": f"Telegram to {to or 'your default chat'}",
             "discord": f"Discord to {to or 'your channel'}",
             "push": "a push notification to your phone",
             "ifttt": f"the IFTTT event '{to}'"}.get(kind, kind)
    return f"Send {where}: \"{str(message or '')[:300]}\""


__all__ = ["ALIASES", "CHANNELS", "SETUP", "channel", "configured", "describe_send", "send",
           "send_discord", "send_ifttt", "send_push", "send_telegram", "status"]
