"""Paired PC shutdown with a cancellable countdown and no forced app closure.

Windows implies ``/f`` when its own ``/t`` is positive. ORION therefore owns
the countdown and calls ``shutdown.exe /s /t 0`` only when it expires. A host
restart cancels the countdown; durable request IDs prevent an uncertain phone
connection from replaying an earlier shutdown. This module starts no work on
import and never accepts a shell command, executable or delay from the phone.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Callable
import uuid

from .atomic_io import atomic_write_text


DELAY_S = 60
CONFIRM_TTL_S = 90
MAX_RECORDS = 128
MAX_BODY_BYTES = 2048
RECORD_TTL_S = 86400
_ACTIVE = {"scheduled", "executing"}
_STATES = _ACTIVE | {"initiated", "blocked", "uncertain", "cancelled"}


class PowerError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 36:
        raise PowerError(400, "A valid request ID is required.")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise PowerError(400, "A valid request ID is required.") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise PowerError(400, "A valid request ID is required.")
    return value


async def _windows_shutdown() -> int:
    """Run the fixed Windows command without a shell or forced termination."""
    executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "shutdown.exe"
    process = await asyncio.wait_for(asyncio.create_subprocess_exec(
        str(executable), "/s", "/t", "0", "/d", "p:0:0",
        "/c", "ORION: shutdown approved from the paired phone.",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        creationflags=0x08000000,  # CREATE_NO_WINDOW: no interactive helper window.
    ), timeout=3.0)
    try:
        return await asyncio.wait_for(process.wait(), timeout=8.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # The command may already have been admitted by Windows. Never replay
        # it automatically merely because its acknowledgement was interrupted.
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        raise


class RemotePowerController:
    """One PC countdown, device-owned confirmations and a bounded request ledger."""

    def __init__(self, config_dir: Path, *, runner: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.time,
                 windows: bool | None = None, delay_s: float = DELAY_S,
                 log: Callable[[str], Any] | None = None) -> None:
        self._path = Path(config_dir) / "remote_power_requests.json"
        self._runner = runner or _windows_shutdown
        self._clock = clock
        self._windows = sys.platform == "win32" if windows is None else windows
        self._delay_s = delay_s  # Test seam; never supplied by a remote request.
        self._log = log or (lambda _: None)
        self._lock = asyncio.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._confirmations: dict[str, tuple[str, float]] = {}
        self._loaded = False
        self._closed = False
        self._timer: asyncio.Task[Any] | None = None

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        if self._path.stat().st_size > 131_072:
            raise ValueError("The PC power ledger is too large.")
        source = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(source, dict) or len(source) > MAX_RECORDS:
            raise ValueError("The PC power ledger is invalid.")
        for key, row in source.items():
            _request_id(key)
            if (not isinstance(row, dict) or row.get("state") not in _STATES
                    or not isinstance(row.get("device_id"), str)
                    or not isinstance(row.get("action"), str)
                    or row["action"] not in {"shutdown", "cancel"}
                    or type(row.get("created_at")) not in {float, int}
                    or not math.isfinite(row["created_at"])
                    or not isinstance(row.get("message"), str)
                    or len(row["message"]) > 512
                    or (row["action"] == "shutdown" and
                        (type(row.get("shutdown_at")) not in {float, int}
                         or not math.isfinite(row["shutdown_at"])))):
                raise ValueError("The PC power ledger is invalid.")
        return source

    async def _save(self) -> None:
        encoded = json.dumps(self._records, ensure_ascii=False, separators=(",", ":"))
        try:
            await asyncio.to_thread(atomic_write_text, self._path, encoded)
        except OSError as exc:
            raise PowerError(503, "The PC could not save this power request safely.") from exc

    async def _load(self) -> None:
        if self._loaded:
            return
        try:
            self._records = await asyncio.to_thread(self._read)
        except (OSError, ValueError, PowerError) as exc:
            raise PowerError(503, "The PC power ledger needs attention before shutdown is available.") from exc
        for row in self._records.values():
            if row["state"] == "scheduled":
                row.update(state="cancelled", message="ORION restarted before the countdown finished.")
            elif row["state"] == "executing":
                row.update(state="uncertain", message="The earlier shutdown outcome could not be confirmed.")
        await self._save()
        self._loaded = True

    def _result(self, key: str) -> dict[str, Any]:
        row = self._records[key]
        deadline = row.get("shutdown_at")
        return {"ok": True, "request_id": key, "state": row["state"],
                "message": row.get("message", ""), "shutdown_at": deadline,
                "remaining_s": max(0, math.ceil(deadline - self._clock()))
                if row["state"] == "scheduled" and isinstance(deadline, (float, int)) else 0}

    def _previous(self, key: str, device: str, action: str) -> dict[str, Any] | None:
        row = self._records.get(key)
        if row is None:
            return None
        if row["device_id"] != device or row["action"] != action:
            raise PowerError(409, "That request ID belongs to a different power action.")
        return self._result(key)

    def _admit_record(self) -> None:
        cutoff = self._clock() - RECORD_TTL_S
        self._records = {key: row for key, row in self._records.items()
                         if row["state"] in _ACTIVE or row["created_at"] > cutoff}
        if len(self._records) >= MAX_RECORDS:
            # All IDs younger than one day remain. An older network replay
            # cannot use its consumed or expired confirmation to run again.
            raise PowerError(429, "The daily PC power request limit has been reached. Try again tomorrow.")

    async def status(self, device: str) -> dict[str, Any]:
        async with self._lock:
            await self._load()
            pending = next(((key, row) for key, row in reversed(list(self._records.items()))
                            if row["action"] == "shutdown" and row["device_id"] == device), None)
            return {"ok": True, "available": self._windows and not self._closed,
                    "delay_s": DELAY_S, "pending": self._result(pending[0]) if pending else None}

    async def prepare(self, device: str) -> dict[str, Any]:
        async with self._lock:
            await self._load()
            if not self._windows or self._closed:
                raise PowerError(503, "PC shutdown is available only on a running Windows ORION host.")
            if any(row["state"] in _ACTIVE for row in self._records.values()):
                raise PowerError(409, "A PC shutdown is already in progress.")
            self._admit_record()
            now = self._clock()
            self._confirmations = {owner: value for owner, value in self._confirmations.items()
                                   if value[1] > now}
            if len(self._confirmations) >= 64 and device not in self._confirmations:
                raise PowerError(429, "Too many PC confirmation requests. Try again shortly.")
            token = secrets.token_urlsafe(32)
            self._confirmations[device] = (hashlib.sha256(token.encode()).hexdigest(), now + CONFIRM_TTL_S)
            return {"ok": True, "confirmation": token, "expires_in": CONFIRM_TTL_S,
                    "delay_s": DELAY_S}

    async def shutdown(self, device: str, key: Any, confirmation: Any,
                       authorised: Callable[[], bool]) -> dict[str, Any]:
        key = _request_id(key)
        async with self._lock:
            await self._load()
            previous = self._previous(key, device, "shutdown")
            if previous:
                return previous
            if not self._windows or self._closed:
                raise PowerError(503, "PC shutdown is unavailable.")
            if any(row["state"] in _ACTIVE for row in self._records.values()):
                raise PowerError(409, "A PC shutdown is already in progress.")
            challenge = self._confirmations.get(device)
            valid = (isinstance(confirmation, str) and len(confirmation) <= 128 and challenge
                     and challenge[1] > self._clock()
                     and hmac.compare_digest(challenge[0], hashlib.sha256(confirmation.encode()).hexdigest()))
            if not valid:
                raise PowerError(403, "Confirm this shutdown again; the confirmation is invalid or has expired.")
            self._admit_record()
            if not authorised():
                raise PowerError(401, "This phone is no longer paired with the PC.")
            self._confirmations.pop(device, None)
            self._records[key] = {"device_id": device, "action": "shutdown", "state": "scheduled",
                                  "created_at": self._clock(),
                                  "shutdown_at": self._clock() + self._delay_s,
                                  "message": "PC shutdown scheduled. You can cancel the countdown."}
            try:
                await self._save()
            except BaseException:
                self._records.pop(key, None)
                raise
            # The OS command belongs to this independent countdown, never the
            # phone request handler. Disconnecting cannot cause a hidden retry.
            self._timer = asyncio.create_task(self._countdown(key, authorised), name="orion-pc-shutdown")
            self._log("REMOTE: PC shutdown confirmed; 60-second countdown started.")
            return self._result(key)

    async def _countdown(self, key: str, authorised: Callable[[], bool]) -> None:
        admitted = False
        try:
            await asyncio.sleep(self._delay_s)
            async with self._lock:
                row = self._records[key]
                if row["state"] != "scheduled" or self._closed:
                    return
                if not authorised():
                    row.update(state="cancelled", message="The phone pairing expired or was revoked before shutdown.")
                    await self._save()
                    return
                row.update(state="executing", message="Windows is checking whether open apps can close.")
                await self._save()
                if not authorised():
                    row.update(state="cancelled", message="The phone pairing expired or was revoked before shutdown.")
                    await self._save()
                    return
            try:
                admitted = True
                code = await self._runner()
                state = "initiated" if code == 0 else "blocked"
                message = ("Windows accepted the shutdown. Open apps may still prevent it."
                           if code == 0 else "Windows did not accept shutdown. Save open work or check the PC account permissions.")
            except Exception:
                state, message = "uncertain", "The PC shutdown outcome could not be confirmed. It has not been retried."
            async with self._lock:
                self._records[key].update(state=state, message=message)
                await self._save()
                self._log(f"REMOTE: PC shutdown result: {state}.")
        except asyncio.CancelledError:
            async with self._lock:
                row = self._records[key]
                if row["state"] == "executing":
                    row.update(state="uncertain" if admitted else "cancelled",
                               message="The shutdown outcome could not be confirmed. It has not been retried."
                               if admitted else "ORION closed before requesting Windows shutdown.")
                    try:
                        await self._save()
                    except PowerError:
                        pass
            raise
        except (PowerError, OSError) as exc:
            async with self._lock:
                self._records[key].update(
                    state="uncertain" if admitted else "blocked",
                    message="The shutdown outcome could not be saved. It has not been retried."
                    if admitted else "The PC could not save its countdown status. Windows shutdown was not requested.")
                try:
                    await self._save()
                except PowerError:
                    pass
            self._log(f"REMOTE: PC power status could not be saved ({type(exc).__name__}).")

    async def cancel(self, device: str, key: Any, shutdown_id: Any) -> dict[str, Any]:
        key, shutdown_id = _request_id(key), _request_id(shutdown_id)
        async with self._lock:
            await self._load()
            previous = self._previous(key, device, "cancel")
            if previous:
                if self._records[key].get("shutdown_id") != shutdown_id:
                    raise PowerError(409, "That request ID belongs to a different shutdown countdown.")
                return previous
            target = self._records.get(shutdown_id)
            if not target or target["action"] != "shutdown" or target["device_id"] != device:
                raise PowerError(404, "No shutdown countdown belongs to this phone.")
            if target["state"] != "scheduled":
                raise PowerError(409, "The shutdown countdown has already ended.")
            self._admit_record()
            old = dict(target)
            target.update(state="cancelled", message="PC shutdown cancelled.")
            self._records[key] = {"device_id": device, "action": "cancel", "state": "cancelled",
                                  "created_at": self._clock(),
                                  "shutdown_id": shutdown_id, "message": "PC shutdown cancelled."}
            try:
                await self._save()
            except BaseException:
                target.clear(); target.update(old)
                self._records.pop(key, None)
                raise
            if self._timer:
                self._timer.cancel()
            self._log("REMOTE: PC shutdown countdown cancelled from its paired phone.")
            return self._result(key)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._loaded:
                return
            scheduled = [row for row in self._records.values() if row["state"] == "scheduled"]
            for row in scheduled:
                row.update(state="cancelled", message="ORION closed before the countdown finished.")
            if scheduled:
                try:
                    await self._save()
                except PowerError:
                    self._log("REMOTE: PC countdown cancelled; its ledger could not be updated.")
            if self._timer and not self._timer.done():
                self._timer.cancel()
        if self._timer:
            # Cancelling the CLI cannot undo a Windows request already admitted.
            # Its task records uncertainty. Process reaping has its own bound.
            try:
                await asyncio.wait_for(asyncio.shield(self._timer), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass


def register_pc_power_routes(app: Any, gateway: Any) -> None:
    """Attach explicit native PC controls to the existing paired HTTPS gateway."""
    from aiohttp import web

    controller = RemotePowerController(gateway.auth.config_dir, log=gateway.bus.log.emit)
    gateway.power_controller = controller
    headers = {"Cache-Control": "no-store"}

    async def body(request: Any, expected: set[str]) -> dict[str, Any]:
        content = bytearray()
        async for chunk in request.content.iter_chunked(MAX_BODY_BYTES + 1):
            content.extend(chunk)
            if len(content) > MAX_BODY_BYTES:
                raise PowerError(413, "The PC power request is too large.")
        try:
            value = json.loads(content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PowerError(400, "A JSON PC power request is required.") from exc
        if not isinstance(value, dict) or set(value) != expected:
            raise PowerError(400, "The PC power request has invalid fields.")
        return value

    async def handle(request: Any, action: str) -> Any:
        # No query-string or JSON token fallback: power requests require the
        # native app's Bearer header, so browser links cannot trigger shutdown.
        device = gateway._authenticated_device(request, {})
        if not device:
            return web.json_response({"ok": False, "error": "Pair this phone with the PC first."}, status=401, headers=headers)
        if not gateway._rate_ok("pc-power:" + device):
            return web.json_response({"ok": False, "error": "Too many PC power requests. Try again shortly."}, status=429, headers=headers)
        try:
            if action == "status":
                result = await controller.status(device)
            elif action == "prepare":
                await body(request, set())
                result = await controller.prepare(device)
            elif action == "shutdown":
                value = await body(request, {"request_id", "confirmation"})
                result = await controller.shutdown(device, value["request_id"], value["confirmation"],
                                                    lambda: gateway._authenticated_device(request, {}) == device)
            else:
                value = await body(request, {"request_id", "shutdown_id"})
                result = await controller.cancel(device, value["request_id"], value["shutdown_id"])
            return web.json_response(result, headers=headers)
        except PowerError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=exc.status, headers=headers)

    async def status(request: Any) -> Any:
        return await handle(request, "status")

    async def prepare(request: Any) -> Any:
        return await handle(request, "prepare")

    async def shutdown(request: Any) -> Any:
        return await handle(request, "shutdown")

    async def cancel(request: Any) -> Any:
        return await handle(request, "cancel")

    async def cleanup(_: Any) -> None:
        await controller.close()

    app.router.add_get("/api/pc/power", status)
    app.router.add_post("/api/pc/power/prepare", prepare)
    app.router.add_post("/api/pc/power/shutdown", shutdown)
    app.router.add_post("/api/pc/power/cancel", cancel)
    app.on_cleanup.append(cleanup)
