"""Send a message on Discord — to a channel, a webhook, or a person's DM."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

API = "https://discord.com/api/v10"
TIMEOUT = 15

#: Where credentials and contacts live. NEVER in this file: a plugin that ships
#: with a token in its source leaks it to anyone the plugin is shared with.
CONFIG_NAME = "messaging.json"


def _config() -> dict:
    """Read config/messaging.json, then let environment variables win.

    Shape:
        {"discord": {"bot_token": "...", "webhook_url": "...",
                     "default_channel_id": "...",
                     "contacts": {"dave": {"user_id": "..."},
                                  "team": {"channel_id": "..."}}}}
    """
    data: dict = {}
    try:
        here = Path(__file__).resolve().parent          # config/custom_tools
        path = here.parent / CONFIG_NAME               # config/messaging.json
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded.get("discord") or {}
    except Exception:
        data = {}
    for key, env in (("bot_token", "DISCORD_BOT_TOKEN"),
                     ("webhook_url", "DISCORD_WEBHOOK_URL"),
                     ("default_channel_id", "DISCORD_CHANNEL_ID")):
        value = os.getenv(env, "").strip()
        if value:
            data[key] = value
    return data if isinstance(data, dict) else {}


def _post(url: str, payload: dict, token: str = "") -> tuple[bool, str]:
    """POST JSON. Returns (ok, detail). Never raises."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", "ORION (+https://localhost)")
    if token:
        request.add_header("Authorization", f"Bot {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return True, str(response.status)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        if exc.code == 401:
            return False, "Discord rejected the credentials (401) — check the bot token."
        if exc.code == 403:
            return False, ("Discord refused (403) — the bot is not in that server, "
                           "or cannot message that person.")
        if exc.code == 404:
            return False, "Discord could not find that channel or user (404)."
        if exc.code == 429:
            return False, "Discord is rate-limiting (429) — try again shortly."
        return False, f"Discord returned {exc.code}. {detail}".strip()
    except urllib.error.URLError as exc:
        return False, f"Could not reach Discord: {exc.reason}"
    except Exception as exc:                              # pragma: no cover
        return False, f"Unexpected error talking to Discord: {exc}"


def _open_dm(user_id: str, token: str) -> tuple[str, str]:
    """Open (or reuse) a DM channel with a user. Returns (channel_id, error)."""
    body = json.dumps({"recipient_id": str(user_id)}).encode("utf-8")
    request = urllib.request.Request(f"{API}/users/@me/channels", data=body,
                                     method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bot {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            channel = json.loads(response.read().decode("utf-8"))
            return str(channel.get("id") or ""), ""
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            return "", ("Discord will not open a DM with that user — they may not "
                        "share a server with the bot, or have DMs closed.")
        return "", f"Could not open a DM (HTTP {exc.code})."
    except Exception as exc:
        return "", f"Could not open a DM: {exc}"


def _resolve(target: str, config: dict) -> tuple[str, str, str]:
    """Resolve a target to (kind, id, error). kind is 'channel' or 'user'."""
    contacts = config.get("contacts") or {}
    key = str(target or "").strip()
    if not key:
        channel = str(config.get("default_channel_id") or "")
        if channel:
            return "channel", channel, ""
        return "", "", ("Who should I message? Give a contact name, a channel id, "
                        "or set a default_channel_id.")
    entry = contacts.get(key) or contacts.get(key.lower())
    if isinstance(entry, dict):
        if entry.get("user_id"):
            return "user", str(entry["user_id"]), ""
        if entry.get("channel_id"):
            return "channel", str(entry["channel_id"]), ""
    if isinstance(entry, str) and entry.isdigit():
        return "channel", entry, ""
    if key.isdigit():
        return "channel", key, ""
    known = ", ".join(sorted(contacts)) or "(none saved)"
    return "", "", (f"I don't know '{key}' on Discord. Saved contacts: {known}. "
                    "Add one under discord.contacts in config/messaging.json, or "
                    "give me the channel id.")


def get_tool_schema() -> dict:
    """Declare this plugin as a callable ORION tool."""
    return {
        "name": "discord_message",
        "description": (
            "Send a message on DISCORD — to a saved contact, a channel id, or a "
            "webhook. Use when asked to 'message/text/DM someone on Discord' or "
            "'post this to the Discord channel'. 'send' delivers the message; "
            "'contacts' lists who I can reach; 'status' reports whether Discord "
            "is configured. Sending is a real outward action — confirm the "
            "recipient and wording first."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "STRING",
                           "description": "send (default), contacts, or status."},
                "to": {"type": "STRING",
                       "description": "Saved contact name, or a numeric channel id. "
                                      "Omit to use the default channel."},
                "message": {"type": "STRING", "description": "What to say."},
            },
            "required": [],
        },
    }


def run(**kwargs) -> str:
    """Execute the plugin. Returns a short string for ORION to speak or show."""
    action = str(kwargs.get("action") or "send").strip().lower()
    config = _config()

    if action in {"status", "check"}:
        bits = []
        bits.append("bot token: " + ("set" if config.get("bot_token") else "missing"))
        bits.append("webhook: " + ("set" if config.get("webhook_url") else "missing"))
        bits.append("default channel: "
                    + ("set" if config.get("default_channel_id") else "missing"))
        bits.append(f"{len(config.get('contacts') or {})} saved contact(s)")
        if not config:
            return ("Discord is not configured. Add a bot token or a webhook URL "
                    "under \"discord\" in config/messaging.json (or set "
                    "DISCORD_BOT_TOKEN / DISCORD_WEBHOOK_URL).")
        return "Discord — " + "; ".join(bits) + "."

    if action in {"contacts", "list", "who"}:
        contacts = config.get("contacts") or {}
        if not contacts:
            return ("No Discord contacts saved yet. Add them under "
                    "discord.contacts in config/messaging.json.")
        lines = []
        for name, entry in sorted(contacts.items()):
            kind = ("DM" if isinstance(entry, dict) and entry.get("user_id")
                    else "channel")
            lines.append(f"  {name} ({kind})")
        return "Discord contacts:\n" + "\n".join(lines)

    message = str(kwargs.get("message") or kwargs.get("text") or "").strip()
    if not message:
        return "What would you like me to say?"
    if len(message) > 2000:
        message = message[:1997] + "…"          # Discord's hard limit

    target = str(kwargs.get("to") or kwargs.get("contact") or "").strip()
    token = str(config.get("bot_token") or "").strip()
    webhook = str(config.get("webhook_url") or "").strip()

    # A webhook posts to one fixed channel and needs no bot — use it when there
    # is no token, or when the user did not name anybody in particular.
    if webhook and (not token or not target):
        ok, detail = _post(webhook, {"content": message})
        return ("Sent on Discord." if ok
                else f"I could not send that on Discord. {detail}")

    if not token:
        return ("Discord is not configured. Add a bot token (to message people "
                "and channels) or a webhook URL (to post to one channel) under "
                "\"discord\" in config/messaging.json.")

    kind, identifier, error = _resolve(target, config)
    if error:
        return error
    if kind == "user":
        identifier, dm_error = _open_dm(identifier, token)
        if dm_error:
            return dm_error
    ok, detail = _post(f"{API}/channels/{identifier}/messages",
                       {"content": message}, token=token)
    if not ok:
        return f"I could not send that on Discord. {detail}"
    where = f" to {target}" if target else ""
    return f"Sent on Discord{where}."
