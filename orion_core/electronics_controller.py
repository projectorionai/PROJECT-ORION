"""Own camera-workbench requests without coupling the vision service to Qt UI."""

from __future__ import annotations

import asyncio
from typing import Any

from . import background


class ElectronicsInspectionController:
    """Run one bounded scan at a time and publish its correlated result."""

    def __init__(self, bus: Any, vision: Any, router: Any = None) -> None:
        self.bus, self.vision, self.router = bus, vision, router
        self._task: asyncio.Task | None = None
        self._closed = False
        bus.electronics_scan_requested.connect(self.request)

    def _error(self, request_id: str, message: str) -> None:
        self.bus.electronics_scan_error.emit({
            "request_id": request_id, "status": "error", "message": message,
            "summary": message,
        })

    def request(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        request_id = str(payload.get("request_id") or "")
        if self._closed:
            self._error(request_id, "The workbench is shutting down.")
            return
        if self._task is not None and not self._task.done():
            self._error(request_id, "An inspection is already running. Wait for its result.")
            return
        frame = payload.get("frame")
        if frame is None or not request_id:
            self._error(request_id, "Start the camera and capture a fresh frame first.")
            return
        # The widget hands over a defensive copy. Keep that exact frame all the
        # way to the model; never re-capture a moving board behind the user.
        grid = payload.get("grid") or (0, 0)
        try:
            grid = (int(grid[0]), int(grid[1]))
        except Exception:
            grid = (0, 0)
        self._task = background.spawn(self._inspect(
            request_id, frame, str(payload.get("focus") or "")[:2000],
            str(payload.get("mode") or "electronics"), grid),
            name="camera inspection")
        if self._task is None:
            self._error(request_id, "The inspection service is not ready yet.")

    async def _inspect(self, request_id: str, frame: Any, focus: str,
                       mode: str = "electronics", grid: tuple[int, int] = (0, 0)) -> None:
        self.bus.electronics_scan_started.emit({"request_id": request_id})
        try:
            try:
                call = self.vision.inspect_electronics(
                    prompt=focus, frame=frame, router=self.router, mode=mode, grid=grid)
            except TypeError:       # a vision service without the new options
                call = self.vision.inspect_electronics(
                    prompt=focus, frame=frame, router=self.router)
            result = await asyncio.wait_for(call, timeout=60.0)
            if not result.ok:
                self._error(request_id, result.text)
                return
            report = next((entry for entry in (result.evidence or [])
                           if isinstance(entry, dict)), None)
            if report is None:
                self._error(request_id, "The inspection returned no usable report. Try again.")
                return
            payload = dict(report, request_id=request_id)
            if result.media:
                payload["jpeg"] = result.media.get("data")
            self.bus.electronics_scan_result.emit(payload)
        except asyncio.TimeoutError:
            self._error(request_id, "Inspection timed out. Check the connection and try again.")
        except asyncio.CancelledError:
            self._error(request_id, "Inspection cancelled.")
            raise
        except Exception:
            self._error(request_id, "Inspection could not finish. Check the vision provider and try again.")

    async def shutdown(self) -> None:
        self._closed = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
