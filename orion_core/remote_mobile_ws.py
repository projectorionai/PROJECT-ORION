"""Pairing-authenticated, short-lived mobile viewport WebSocket uplink.

Register with ``register_mobile_ws_routes(app, gateway)`` in
``RemoteGateway._build_app``. Binary messages are raw JPEG frames; text messages
are a deliberately small JSON-RPC 2.0 subset. There is no remote action method
here: device actions must first be approved in the phone's local UI.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .phone_contracts import METHODS as NATIVE_METHODS, valid_command, contact_name


MAX_FRAME_BYTES = 1_000_000
MAX_TEXT_CHARS = 131_072
FRAME_TTL_S = 10.0
MIN_FRAME_INTERVAL_S = 0.40  # More than 2.5 fps is discarded.
#: How often an open uplink re-checks that its device is still authorised.
#: Checking only when a message arrived left a silent socket open after the
#: device was revoked or its token expired.
AUTH_RECHECK_S = 30.0
COMMAND_METHODS = frozenset({
    "viewport.capture_dom", "viewport.click", "viewport.scroll", "viewport.set_text",
    "device.open_app", "device.make_call", "device.send_sms",
}) | NATIVE_METHODS
NODE_ID = re.compile(r"0(?:/[0-9]{1,2}){0,12}\Z")
PACKAGE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z")
PHONE_NUMBER = re.compile(r"\+?[0-9() -]{3,40}\Z")
NORMALIZED_NUMBER = re.compile(r"\+?[0-9]{3,15}\Z")


def _utf16_length(value: str) -> int:
    """Android String.length counts UTF-16 units, including emoji pairs."""
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return 2**31


def _valid_phone_target(value: Any) -> bool:
    """Match Android's numeric or unambiguous contact-name input path."""
    if not isinstance(value, str):
        return False
    if PHONE_NUMBER.fullmatch(value):
        normalized = value.translate(str.maketrans("", "", " ()-"))
        return NORMALIZED_NUMBER.fullmatch(normalized) is not None
    name = value.strip()
    return (2 <= _utf16_length(name) <= 80 and any(ch.isalpha() for ch in name)
            and all(ch.isalpha() or ch in " .'-" for ch in name))


class DeviceCommandRejected(RuntimeError):
    """The phone denied a proposal or failed to perform it."""


def _valid_command_params(method: str, params: dict[str, Any]) -> bool:
    """A small fixed schema keeps agent output from becoming an Android intent."""
    if not isinstance(params, dict):
        return False
    if method in NATIVE_METHODS:
        return valid_command(method, params)
    if method == "viewport.capture_dom":
        return not params
    if method in {"viewport.click", "viewport.scroll", "viewport.set_text"}:
        expected = {"snapshotId", "nodeId"}
        if method == "viewport.scroll":
            expected.add("direction")
        if method == "viewport.set_text":
            expected.add("text")
        return (
            set(params) == expected
            and isinstance(params.get("snapshotId"), str)
            and 1 <= len(params["snapshotId"]) <= 80
            and isinstance(params.get("nodeId"), str)
            and NODE_ID.fullmatch(params["nodeId"]) is not None
            and (method != "viewport.scroll" or
                 (isinstance(params.get("direction"), str)
                  and params["direction"] in {"up", "down", "left", "right"}))
            and (method != "viewport.set_text" or
                 (isinstance(params.get("text"), str) and "\x00" not in params["text"]
                  and _utf16_length(params["text"]) <= 4000))
        )
    if method == "device.open_app":
        if set(params) == {"name"}:
            return contact_name(params["name"])
        return (set(params) == {"package"}
                and isinstance(params.get("package"), str)
                and len(params["package"]) <= 150
                and PACKAGE_NAME.fullmatch(params["package"]) is not None)
    if method in {"device.make_call", "device.send_sms"}:
        expected = {"number"}
        if method == "device.send_sms":
            expected.add("body")
        return (
            set(params) == expected
            and _valid_phone_target(params.get("number"))
            and (method != "device.send_sms" or
                 (isinstance(params.get("body"), str)
                  and 1 <= _utf16_length(params["body"]) <= 1600
                  and not params["body"].isspace()))
        )
    return False


@dataclass(frozen=True)
class ViewportFrame:
    jpeg: bytes
    received_at: float


@dataclass(frozen=True)
class ViewportDom:
    snapshot: dict[str, Any]
    received_at: float


class MobileViewportBridge:
    """Keeps one fresh, bounded frame and DOM snapshot per connected device."""

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self._sessions: dict[str, Any] = {}
        self._session_tokens: dict[str, str] = {}
        self._frames: dict[str, ViewportFrame] = {}
        self._doms: dict[str, ViewportDom] = {}
        self._status: dict[str, tuple[float, dict[str, bool]]] = {}
        self._pending: dict[tuple[str, str], asyncio.Future[dict[str, Any]]] = {}

    async def request_device_action(
        self, device_id: str, method: str, params: dict[str, Any],
        *, timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        """Propose one action and await a locally authorized device result.

        This is an in-process API for a gated agent tool. It deliberately has no
        public HTTP equivalent. Every proposal carries ``requires_local_approval``
        and the phone must answer after its own UI authorization step.
        """
        if method not in COMMAND_METHODS:
            raise ValueError("Unsupported device command")
        if not isinstance(params, dict) or not _valid_command_params(method, params):
            raise ValueError("Invalid device command parameters")
        if len(json.dumps(params)) > MAX_TEXT_CHARS:
            raise ValueError("Device command parameters too large")
        ws = self._sessions.get(device_id)
        token = self._session_tokens.get(device_id, "")
        if ws is None or ws.closed or self.gateway.auth.verify_access(token) != device_id:
            raise ConnectionError("Paired phone is not connected")
        if any(pending_device == device_id for pending_device, _ in self._pending):
            raise RuntimeError("A phone action is already awaiting local approval")
        if method in {"viewport.click", "viewport.scroll", "viewport.set_text"}:
            dom = self.latest_dom(device_id)
            if dom is None or dom.snapshot.get("snapshotId") != params["snapshotId"]:
                raise DeviceCommandRejected("A fresh DOM snapshot is required")

        request_id = uuid.uuid4().hex
        key = (device_id, request_id)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        try:
            await ws.send_json({
                "jsonrpc": "2.0", "id": request_id, "method": method,
                "params": params, "requires_local_approval": True,
            })
            reply = await asyncio.wait_for(future, timeout=max(1.0, min(timeout_s, 120.0)))
        finally:
            self._pending.pop(key, None)

        # A device can be revoked while its local approval dialog is open.
        # The socket watcher closes it eventually, but a reply may arrive first.
        if self._sessions.get(device_id) is not ws or self.gateway.auth.verify_access(token) != device_id:
            raise ConnectionError("Paired phone is no longer authorised")

        if "error" in reply:
            error = reply.get("error")
            detail = error.get("message") if isinstance(error, dict) else "Action denied"
            raise DeviceCommandRejected(str(detail)[:200])
        result = reply.get("result")
        if not isinstance(result, dict) or result.get("authorized") is not True:
            raise DeviceCommandRejected("Phone did not authorize the action")
        if result.get("ok") is not True:
            raise DeviceCommandRejected(str(result.get("detail") or "Phone could not complete the action")[:300])
        if method == "viewport.capture_dom":
            snapshot = result.get("snapshot")
            if not isinstance(snapshot, dict):
                raise DeviceCommandRejected("Phone did not return a DOM snapshot")
            self._doms[device_id] = ViewportDom(snapshot, time.monotonic())
        return result

    def _fail_pending(self, device_id: str) -> None:
        for (pending_device, _), future in tuple(self._pending.items()):
            if pending_device == device_id and not future.done():
                future.set_exception(ConnectionError("Phone disconnected"))

    def open_sockets(self) -> list[Any]:
        """Every uplink socket still open, for the gateway to close at shutdown.
        Its handler then leaves its receive loop and cleans up as on any close."""
        return [ws for ws in self._sessions.values() if not getattr(ws, "closed", False)]

    def device_status(self, device_id: str) -> dict[str, bool]:
        """Permission telemetry informs answers; it never grants authorisation."""
        ws = self._sessions.get(device_id)
        token = self._session_tokens.get(device_id, "")
        connected = bool(ws is not None and not ws.closed
                         and self.gateway.auth.verify_access(token) == device_id)
        result = {"connected": connected}
        report = self._status.get(device_id)
        if connected and report is not None and time.monotonic() - report[0] <= 45:
            result.update(report[1])
        return result

    def latest_frame(self, device_id: str) -> ViewportFrame | None:
        """Return a fresh frame only while this paired device is connected."""
        if not self.device_status(device_id)["connected"]:
            return None
        frame = self._frames.get(device_id)
        if frame is None or time.monotonic() - frame.received_at > FRAME_TTL_S:
            return None
        return frame

    def latest_dom(self, device_id: str) -> ViewportDom | None:
        if not self.device_status(device_id)["connected"]:
            return None
        dom = self._doms.get(device_id)
        if dom is None or time.monotonic() - dom.received_at > FRAME_TTL_S:
            return None
        return dom

    async def handle(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        header = str(request.headers.get("Authorization") or "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        device_id = self.gateway.auth.verify_access(token) if token else None
        if not device_id:
            return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

        ws = web.WebSocketResponse(
            heartbeat=15, max_msg_size=MAX_FRAME_BYTES + 1024, autoclose=True,
        )
        await ws.prepare(request)
        previous = self._sessions.get(device_id)
        if previous is not None:
            self._fail_pending(device_id)
        # A reconnect must not inherit a frame or DOM captured before this
        # connection, even for the same paired device identity.
        self._frames.pop(device_id, None)
        self._doms.pop(device_id, None)
        self._status.pop(device_id, None)
        self._sessions[device_id] = ws
        self._session_tokens[device_id] = token
        if previous is not None and previous is not ws:
            await previous.close(code=1000, message=b"Replaced by a new uplink")

        async def _watch_authorisation() -> None:
            while not ws.closed:
                await asyncio.sleep(AUTH_RECHECK_S)
                if self.gateway.auth.verify_access(token) != device_id:
                    await ws.close(code=1008, message=b"Authorization expired")
                    return

        watcher = asyncio.get_running_loop().create_task(_watch_authorisation())
        last_frame = 0.0
        try:
            async for message in ws:
                if self._sessions.get(device_id) is not ws:
                    await ws.close(code=1000, message=b"Uplink replaced")
                    break
                # An access token lasts 15 minutes and a device may be revoked.
                # Recheck during the stream so an old socket cannot bypass either.
                if self.gateway.auth.verify_access(token) != device_id:
                    await ws.close(code=1008, message=b"Authorization expired")
                    break
                if message.type == WSMsgType.BINARY:
                    data = message.data
                    now = time.monotonic()
                    if not isinstance(data, bytes) or not (
                        4 <= len(data) <= MAX_FRAME_BYTES
                        and data.startswith(b"\xff\xd8")
                        and data.endswith(b"\xff\xd9")
                    ):
                        await ws.close(code=1003, message=b"Invalid JPEG frame")
                        break
                    if now - last_frame >= MIN_FRAME_INTERVAL_S:
                        self._frames[device_id] = ViewportFrame(data, now)
                        last_frame = now
                elif message.type == WSMsgType.TEXT:
                    if len(message.data) > MAX_TEXT_CHARS:
                        await ws.close(code=1009, message=b"JSON message too large")
                        break
                    await self._handle_text(ws, device_id, message.data)
                elif message.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            watcher.cancel()
            if self._sessions.get(device_id) is ws:
                self._sessions.pop(device_id, None)
                self._session_tokens.pop(device_id, None)
                self._frames.pop(device_id, None)
                self._doms.pop(device_id, None)
                self._status.pop(device_id, None)
                self._fail_pending(device_id)
        return ws

    async def _handle_text(self, ws: Any, device_id: str, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            await self._error(ws, None, -32700, "Invalid JSON")
            return
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            await self._error(ws, None, -32600, "Invalid JSON-RPC request")
            return
        request_id = payload.get("id")
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            request_id = None
        if isinstance(request_id, str) and len(request_id) > 80:
            request_id = None
        if "method" not in payload:
            if isinstance(request_id, str):
                future = self._pending.get((device_id, request_id))
                if future is not None and not future.done():
                    future.set_result(payload)
            return
        method = payload.get("method")
        if method == "viewport.ping":
            await ws.send_json({
                "jsonrpc": "2.0", "id": request_id,
                "result": {"server_time_ms": int(time.time() * 1000)},
            })
        elif method == "device.status":
            params = payload.get("params")
            if not isinstance(params, dict):
                await self._error(ws, request_id, -32602, "Invalid device status")
                return
            allowed = {"accessibility", "screen_capture", "background_audio", "microphone",
                       "calls", "sms", "native_speech", "sentinel", "contacts", "email_automation"}
            self._status[device_id] = (time.monotonic(), {
                name: value for name, value in params.items()
                if name in allowed and isinstance(value, bool)})
            if request_id is not None:
                await ws.send_json({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}})
        elif method == "viewport.stop":
            self._frames.pop(device_id, None)
            self._doms.pop(device_id, None)
            if request_id is not None:
                await ws.send_json({"jsonrpc": "2.0", "id": request_id,
                                    "result": {"cleared": True}})
        else:
            # This socket is a data uplink. No method can invoke a gesture,
            # launch an app, place a call, or send a message.
            await self._error(ws, request_id, -32601, "Method not allowed")

    @staticmethod
    async def _error(ws: Any, request_id: str | int | None,
                     code: int, message: str) -> None:
        await ws.send_json({"jsonrpc": "2.0", "id": request_id,
                            "error": {"code": code, "message": message}})


def register_mobile_ws_routes(app: Any, gateway: Any) -> MobileViewportBridge:
    """Register the authenticated endpoint and expose fresh data to the brain.

    After registration, ``gateway.mobile_viewport.latest_frame(device_id)`` and
    ``latest_dom(device_id)`` return only recent data from a live socket. The
    caller must associate its conversation with the same paired device id.
    """
    bridge = MobileViewportBridge(gateway)
    app.router.add_get("/v1/device/uplink", bridge.handle)
    gateway.mobile_viewport = bridge
    return bridge
