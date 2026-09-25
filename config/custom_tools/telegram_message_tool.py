"""Send a Telegram message directly through the Bot API (no browser needed)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 15
CONFIG_NAME = "messaging.json"


def _config() -> dict:
    """Read config/messaging.json ["telegram"], environment wins.

    Shape:
        {"telegram": {"bot_token": "...", "default_chat_id": "...",
                      "contacts": {"mum": "123456789"}}}
    """
    data: dict = {}
    try:
        path = Path(__file__).resolve().parent.parent / CONFIG_NAME
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded.get("telegram") or {}
    except Exception:
        data = {}
    for key, env in (("bot_token", "TELEGRAM_BOT_TOKEN"),
                     ("default_chat_id", "TELEGRAM_CHAT_ID")):
        value = os.getenv(env, "").strip()
        if value:
            data[key] = value
    return data if isinstance(data, dict) else {}


def _send(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    """POST to the Bot API. Returns (ok, detail). Never raises."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
            if payload.get("ok"):
                return True, ""
            return False, str(payload.get("description") or "Telegram refused it.")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            detail = str(body.get("description") or "")
        except Exception:
            pass
        if exc.code == 401:
            return False, "Telegram rejected the bot token (401)."
        if exc.code == 400 and "chat not found" in detail.lower():
            return False, ("Telegram says that chat does not exist — the person "
                           "must message the bot once before it can message them.")
        return False, detail or f"Telegram returned {exc.code}."
    except urllib.error.URLError as exc:
        return False, f"Could not reach Telegram: {exc.reason}"
    except Exception as exc:                              # pragma: no cover
        return False, f"Unexpected error talking to Telegram: {exc}"


def _resolve(target: str, config: dict) -> tuple[str, str]:
    """Resolve a target to (chat_id, error)."""
    contacts = config.get("contacts") or {}
    key = str(target or "").strip()
    if not key:
        default = str(config.get("default_chat_id") or "")
        if default:
            return default, ""
        return "", ("Who should I message on Telegram? Give a saved contact or a "
                    "chat id, or set a default_chat_id.")
    entry = contacts.get(key) or contacts.get(key.lower())
    if isinstance(entry, dict):
        entry = entry.get("chat_id")
    if entry:
        return str(entry), ""
    if key.lstrip("-").isdigit():
        return key, ""
    known = ", ".join(sorted(contacts)) or "(none saved)"
    return "", (f"I don't know '{key}' on Telegram. Saved contacts: {known}. "
                "Add one under telegram.contacts in config/messaging.json.")


def get_tool_schema() -> dict:
    """Declare this plugin as a callable ORION tool."""
    return {
        "name": "telegram_message",
        "description": (
            "Send a TELEGRAM message directly through the Bot API — it is "
            "delivered, not just opened in a browser for you to press send. Use "
            "when asked to 'text/message someone on Telegram'. 'send' delivers; "
            "'contacts' lists who I can reach; 'status' reports configuration. "
            "Sending is a real outward action — confirm recipient and wording."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "STRING",
                           "description": "send (default), contacts, or status."},
                "to": {"type": "STRING",
                       "description": "Saved contact name or numeric chat id."},
                "message": {"type": "STRING", "description": "What to say."},
            },
            "required": [],
        },
    }


def run(**kwargs) -> str:
    """Execute the plugin. Returns a short string for ORION to speak or show."""
    action = str(kwargs.get("action") or "send").strip().lower()
    config = _config()
    token = str(config.get("bot_token") or "").strip()

    if action in {"status", "check"}:
        if not token:
            return ("Telegram is not configured. Create a bot with @BotFather and "
                    "put its token under \"telegram\" in config/messaging.json "
                    "(or set TELEGRAM_BOT_TOKEN).")
        return ("Telegram — bot token set; default chat: "
                + ("set" if config.get("default_chat_id") else "missing")
                + f"; {len(config.get('contacts') or {})} saved contact(s).")

    if action in {"contacts", "list", "who"}:
        contacts = config.get("contacts") or {}
        if not contacts:
            return ("No Telegram contacts saved yet. Add them under "
                    "telegram.contacts in config/messaging.json.")
        return "Telegram contacts:\n" + "\n".join(
            f"  {name}" for name in sorted(contacts))

    message = str(kwargs.get("message") or kwargs.get("text") or "").strip()
    if not message:
        return "What would you like me to say?"
    if len(message) > 4096:
        message = message[:4093] + "…"          # Telegram's hard limit

    if not token:
        return ("Telegram is not configured. Create a bot with @BotFather and put "
                "its token under \"telegram\" in config/messaging.json.")

    chat_id, error = _resolve(str(kwargs.get("to") or kwargs.get("contact") or ""),
                              config)
    if error:
        return error
    ok, detail = _send(token, chat_id, message)
    if not ok:
        return f"I could not send that on Telegram. {detail}"
    target = str(kwargs.get("to") or "").strip()
    return f"Sent on Telegram{(' to ' + target) if target else ''}."
