"""Whitelist-only Telegram inbox with locally reviewed command execution.

Discovery never starts polling. Incoming tool requests remain inert until a
local user approves their exact identifier and payload through ORION's gate.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from ..data import ToolResult
from . import runtime

_poller: asyncio.Task | None = None
_last_error = ""
_inbox_lock = asyncio.Lock()


def _config() -> dict:
    settings = runtime.load("telegram_remote")
    settings["bot_token"] = runtime.secret("jv_telegram_remote", "TELEGRAM_BOT_TOKEN")
    allowed = settings.get("allowed_chat_ids", [])
    if not isinstance(allowed, list):
        raise TypeError("Telegram allowed_chat_ids must be an array")
    settings["allowed_chat_ids"] = {str(x) for x in allowed if str(x).lstrip("-").isdigit()}
    return settings


async def _api(token: str, method: str, payload: dict) -> dict:
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35, connect=5)) as session, \
            session.post(f"https://api.telegram.org/bot{token}/{method}", json=payload,
                         allow_redirects=False) as response:
        raw = await response.content.read(1_000_001)
        if len(raw) > 1_000_000:
            raise RuntimeError("Telegram response exceeded its size limit")
        if response.status != 200:
            # Telegram embeds the bot token in the URL; never report a
            # raw client exception or response URL to a model or log.
            raise RuntimeError(f"Telegram returned HTTP {response.status}")
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError("Telegram refused the request")
        return data


def _request_from_update(update: dict, allowed: set[str]) -> dict | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat", {})
    # Group membership alone cannot authenticate the person who sent it.
    if chat.get("type") != "private" or str(chat.get("id")) not in allowed:
        return None
    text = str(message.get("text") or "").strip()
    if not text.startswith("/tool ") or len(text) > 5000:
        return None
    try:
        payload = json.loads(text[6:])
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name") or "")
    arguments = payload.get("arguments", {})
    if not name or not isinstance(arguments, dict) or name.startswith("jv_"):
        return None
    return {"id": uuid.uuid4().hex[:12], "chat_id": str(chat["id"]), "name": name,
            "arguments": arguments, "received_at": time.time(), "status": "pending"}


async def _poll(config: dict) -> None:
    global _last_error
    state = await asyncio.to_thread(runtime.load, "telegram_inbox")
    offset = int(state.get("offset", 0))
    delay = 2
    while True:
        try:
            data = await _api(config["bot_token"], "getUpdates", {
                "offset": offset, "timeout": 20, "limit": 100, "allowed_updates": ["message"]})
            updates = data.get("result", [])
            if not isinstance(updates, list):
                raise TypeError("Invalid Telegram update list")
            async with _inbox_lock:
                inbox = await asyncio.to_thread(runtime.load, "telegram_inbox")
                requests = inbox.get("requests", [])
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    offset = max(offset, int(update.get("update_id", -1)) + 1)
                    request = _request_from_update(update, config["allowed_chat_ids"])
                    if request is not None:
                        requests.append(request)
                        runtime.emit("jv_remote_request", request)
                        if runtime.context().bus is not None:
                            runtime.context().bus.log.emit(f"TELEGRAM: request {request['id']} awaits local review ({request['name']}).")
                await asyncio.to_thread(runtime.save, "telegram_inbox", {"offset": offset, "requests": requests[-100:]})
            _last_error = ""
            delay = 2
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- external service faults become an honest status
            _last_error = type(exc).__name__
            await asyncio.sleep(delay)
            delay = min(60, delay * 2)


async def telegram_remote(args: dict) -> ToolResult:
    async with _inbox_lock:
        return await _telegram_remote_locked(args)


async def _telegram_remote_locked(args: dict) -> ToolResult:
    global _poller
    action = str(args.get("action") or "status").lower()
    config = await asyncio.to_thread(_config)
    token = config.get("bot_token")
    allowed = config["allowed_chat_ids"]
    if action == "status":
        running = _poller is not None and not _poller.done()
        return ToolResult(f"Telegram: token {'configured' if token else 'missing'}; {len(allowed)} permitted private chats; "
                          f"polling {'on' if running else 'off'}" + (f"; last connection error: {_last_error}." if _last_error else "."))
    if action == "stop":
        if _poller is not None:
            _poller.cancel()
            await asyncio.gather(_poller, return_exceptions=True)
            _poller = None
        return ToolResult("Telegram polling stopped.")
    if action in {"pending", "approve", "reject"}:
        state = await asyncio.to_thread(runtime.load, "telegram_inbox")
        requests = state.get("requests", [])
        if action == "pending":
            pending = [x for x in requests if x.get("status") == "pending"]
            return ToolResult(json.dumps(pending, ensure_ascii=False) if pending else "No Telegram requests await review.")
        request = next((x for x in requests if x.get("id") == args.get("id") and x.get("status") == "pending"), None)
        if request is None:
            return ToolResult("That request is unknown or has already been reviewed.", ok=False)
        if time.time() - request["received_at"] > 600:
            request["status"] = "expired"
            await asyncio.to_thread(runtime.save, "telegram_inbox", state)
            return ToolResult("The request expired. Ask the remote user to issue a fresh request.", ok=False)
        if action == "reject":
            request["status"] = "rejected"
            await asyncio.to_thread(runtime.save, "telegram_inbox", state)
            return ToolResult("Telegram request rejected.")
        # Consume before invoking: retrying this approval can never send twice.
        request["status"] = "running"
        await asyncio.to_thread(runtime.save, "telegram_inbox", state)
        try:
            outcome = await runtime.native(request["name"], request["arguments"])
        except BaseException:
            request["status"] = "interrupted"
            await asyncio.to_thread(runtime.save, "telegram_inbox", state)
            raise
        request.update(status="done" if outcome.ok else "failed", result=outcome.text[:10000])
        await asyncio.to_thread(runtime.save, "telegram_inbox", state)
        return outcome
    if not token or not allowed:
        return ToolResult("Set TELEGRAM_BOT_TOKEN in this plugin's vault or environment, and permitted private "
                          "chat ids in config/jv_plugins/telegram_remote.json under allowed_chat_ids.", ok=False)
    if action == "start":
        if _poller is not None and not _poller.done():
            return ToolResult("Telegram polling is already running.")
        await _api(token, "getMe", {})
        _poller = runtime.spawn(_poll(config), "telegram")
        return ToolResult("Telegram polling started for the permitted private chats. Tool requests require local review.")
    if action == "send":
        chat_id = str(args.get("chat_id") or "")
        message = str(args.get("message") or "")
        if chat_id not in allowed or not message.strip() or len(message) > 4096:
            return ToolResult("Choose a permitted chat and a message of one to 4,096 characters.", ok=False)
        result = await _api(token, "sendMessage", {"chat_id": chat_id, "text": message})
        identifier = result.get("result", {}).get("message_id")
        return ToolResult(f"Telegram bot API accepted the message (message {identifier}).",
                          receipt={"kind": "messaging", "platform": "telegram", "backend": "bot_api",
                                   "sender_kind": "configured_bot", "status": "submitted", "sent": True,
                                   "attempted": True, "delivery_verified": False})
    return ToolResult("Telegram actions: status, start, stop, send, pending, approve or reject.", ok=False)
