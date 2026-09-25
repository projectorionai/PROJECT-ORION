"""
Globe maximum-zoom controller (Section 1 — ``max_zoom_in_globe``).

The on-screen intelligence globe is driven by two possible backends: the
desktop CesiumJS renderer inside QWebEngine, and a browser/web-controlled
globe reached through an automation layer.  Rather than duplicate the zoom
mechanism, this module drives whichever backend is active through a small
protocol and repeatedly issues the existing zoom-in step until the underlying
system reports — or observably behaves as though it has reached — its maximum
zoom.

Design goals (all bounded, none blocking):

* Never loop unbounded: hard caps on attempts, wall-clock time and a
  consecutive "no meaningful change" run all terminate the loop.
* Never block the event loop: the delay between attempts is an ``await``.
* Detect the limit from observable state (camera altitude / scale / zoom
  level) with a configurable relative tolerance, from an explicit maximum the
  backend exposes, or from the backend reporting further zoom is unavailable.
* Degrade sanely when state cannot be read — a bounded number of blind
  attempts, clearly reported.
* Be cancellable between every attempt.
* Return a fully structured, typed result.

The controller is deliberately free of Qt, asyncio-loop assumptions and
provider code so it is unit-testable with fakes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


class ZoomTermination(str, Enum):
    """Why the zoom loop stopped — one of these is always the outcome."""

    MAX_ZOOM_REACHED = "max_zoom_reached"
    STATE_STABLE = "state_stable"                # repeated no-change condition
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    MAX_ATTEMPTS = "max_attempts"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    STATE_UNOBSERVABLE = "state_unobservable"
    CONTROL_FAILURE = "unexpected_control_failure"


@dataclass(frozen=True)
class GlobeZoomConfig:
    """Configurable safeguards.  Every bound is finite by construction."""

    max_attempts: int = 40
    max_seconds: float = 20.0
    delay_between_s: float = 0.05
    unchanged_threshold: int = 3          # consecutive no-change steps → stop
    tolerance: float = 0.01               # relative change treated as "no change"
    blind_attempts: int = 6               # bounded steps when state is unreadable

    def normalised(self) -> "GlobeZoomConfig":
        """Clamp to sane, finite ranges so a bad config can never loop forever."""
        return GlobeZoomConfig(
            max_attempts=max(1, min(500, int(self.max_attempts))),
            max_seconds=max(0.5, min(120.0, float(self.max_seconds))),
            delay_between_s=max(0.0, min(2.0, float(self.delay_between_s))),
            unchanged_threshold=max(1, min(20, int(self.unchanged_threshold))),
            tolerance=max(0.0, min(0.5, float(self.tolerance))),
            blind_attempts=max(1, min(50, int(self.blind_attempts))),
        )


@dataclass
class GlobeZoomResult:
    """Structured outcome of a ``max_zoom_in_globe`` run."""

    success: bool
    termination: ZoomTermination
    start_state: float | None
    final_state: float | None
    attempts: int
    duration_s: float
    backend: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "termination": self.termination.value,
            "start_state": self.start_state,
            "final_state": self.final_state,
            "attempts": self.attempts,
            "duration_s": round(self.duration_s, 3),
            "backend": self.backend,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        reason = {
            ZoomTermination.MAX_ZOOM_REACHED: "maximum zoom reached",
            ZoomTermination.STATE_STABLE: "zoom stabilised (no further change)",
            ZoomTermination.CANCELLED: "cancelled",
            ZoomTermination.TIMEOUT: "timed out",
            ZoomTermination.MAX_ATTEMPTS: "attempt limit reached",
            ZoomTermination.BACKEND_UNAVAILABLE: "globe backend unavailable",
            ZoomTermination.STATE_UNOBSERVABLE: "zoomed blind (state not observable)",
            ZoomTermination.CONTROL_FAILURE: "control failure",
        }[self.termination]
        return (
            f"Globe max-zoom ({self.backend}): {reason} after {self.attempts} "
            f"attempt(s) in {self.duration_s:.2f}s."
        )


@runtime_checkable
class GlobeZoomBackend(Protocol):
    """A globe implementation the controller can drive.

    ``read_state`` and ``zoom_in_step`` may be sync or async; the controller
    awaits either transparently.  ``state`` is any scalar that decreases (or
    changes) monotonically towards maximum zoom — camera altitude for Cesium,
    a zoom level or scale for a web map.
    """

    name: str

    def available(self) -> bool | Awaitable[bool]:
        """Is the backend present and ready to accept zoom commands?"""

    def read_state(self) -> float | None | Awaitable[float | None]:
        """Current observable zoom scalar, or ``None`` if it cannot be read."""

    def zoom_in_step(self) -> float | None | Awaitable[float | None]:
        """Issue one zoom-in step; return the new observable state or ``None``."""

    def at_maximum(self, state: float | None) -> bool | None | Awaitable[bool | None]:
        """Backend-specific boundary check.  ``True``/``False`` when known,
        ``None`` when the backend cannot say (the controller then relies on
        state-stability detection)."""


CancelToken = Callable[[], bool] | Any


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _is_cancelled(cancel: CancelToken | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        try:
            return bool(cancel())
        except Exception:
            return False
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        try:
            return bool(is_set())
        except Exception:
            return False
    return bool(cancel)


def _meaningful_change(previous: float | None, current: float | None, tolerance: float) -> bool:
    """True when ``current`` differs from ``previous`` by more than the relative
    tolerance.  Absolute fallback covers values near zero."""
    if previous is None or current is None:
        return previous != current
    delta = abs(current - previous)
    scale = max(abs(previous), abs(current), 1e-9)
    if delta / scale > tolerance:
        return True
    return delta > tolerance and scale < 1.0


class CesiumGlobeBackend:
    """Desktop backend: the CesiumJS globe inside QWebEngine.

    Kept Qt-free — it only duck-types the ``GlobeView``'s async JS bridge
    (``_eval_js``) and the browser-side helpers ``orionZoomState`` /
    ``orionZoomInStep`` / ``orionZoomMinAltitude`` defined on the page.  Camera
    altitude (metres) is the observable zoom state; the street-level floor
    (~180 m) is the documented maximum-zoom boundary.
    """

    name = "cesium-desktop"

    def __init__(self, view: Any) -> None:
        self._view = view
        self._min_alt: float | None = None

    async def available(self) -> bool:
        return getattr(self._view, "view", None) is not None and getattr(self._view, "_built", False)

    @staticmethod
    def _num(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    async def read_state(self) -> float | None:
        val = await self._view._eval_js("window.orionZoomState?window.orionZoomState():null")
        return self._num(val)

    async def zoom_in_step(self) -> float | None:
        val = await self._view._eval_js("window.orionZoomInStep?window.orionZoomInStep():null")
        return self._num(val)

    async def at_maximum(self, state: float | None) -> bool | None:
        if state is None:
            return None
        if self._min_alt is None:
            raw = await self._view._eval_js(
                "window.orionZoomMinAltitude?window.orionZoomMinAltitude():180")
            self._min_alt = self._num(raw) or 180.0
        return state <= self._min_alt * 1.0001


class WebGlobeBackend:
    """Browser/web-controlled globe backend.

    Drives a globe rendered in an external browser through an automation
    surface exposing ``eval_js``/``run_js`` (Playwright, the MCP browser tools,
    etc.).  The page is expected to expose the same ``orionZoomState`` /
    ``orionZoomInStep`` helpers, or a compatible map zoom-level accessor.  When
    the page cannot report a boundary the controller falls back to
    state-stability detection.
    """

    name = "web-browser"

    def __init__(self, runner: Any, *,
                 state_js: str = "window.orionZoomState?window.orionZoomState():null",
                 step_js: str = "window.orionZoomInStep?window.orionZoomInStep():null",
                 max_js: str | None = "window.orionZoomMinAltitude?window.orionZoomMinAltitude():null",
                 max_is_floor: bool = True) -> None:
        # ``runner`` must expose an async ``eval_js(code)`` (or ``run_js``).
        self._runner = runner
        self._state_js = state_js
        self._step_js = step_js
        self._max_js = max_js
        self._max_is_floor = max_is_floor
        self._boundary: float | None = None

    async def _eval(self, code: str) -> Any:
        fn = getattr(self._runner, "eval_js", None) or getattr(self._runner, "run_js", None)
        if fn is None:
            return None
        return await _maybe_await(fn(code))

    @staticmethod
    def _num(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    async def available(self) -> bool:
        return (getattr(self._runner, "eval_js", None) is not None
                or getattr(self._runner, "run_js", None) is not None)

    async def read_state(self) -> float | None:
        return self._num(await self._eval(self._state_js))

    async def zoom_in_step(self) -> float | None:
        return self._num(await self._eval(self._step_js))

    async def at_maximum(self, state: float | None) -> bool | None:
        if state is None or not self._max_js:
            return None
        if self._boundary is None:
            self._boundary = self._num(await self._eval(self._max_js))
        if self._boundary is None:
            return None
        if self._max_is_floor:
            return state <= self._boundary * 1.0001
        return state >= self._boundary * 0.9999


class GlobeZoomController:
    """Drives a :class:`GlobeZoomBackend` to maximum zoom under strict bounds."""

    def __init__(self, backend: GlobeZoomBackend, config: GlobeZoomConfig | None = None,
                 sleep: Callable[[float], Awaitable[None]] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.backend = backend
        self.config = (config or GlobeZoomConfig()).normalised()
        # Injectable so tests run instantly and the real driver stays non-blocking.
        import asyncio
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic

    async def run(self, cancel: CancelToken | None = None) -> GlobeZoomResult:
        cfg = self.config
        started = self._clock()
        warnings: list[str] = []
        name = getattr(self.backend, "name", "globe")

        def elapsed() -> float:
            return self._clock() - started

        def result(success: bool, term: ZoomTermination, start: float | None,
                   final: float | None, attempts: int) -> GlobeZoomResult:
            return GlobeZoomResult(success, term, start, final, attempts,
                                   elapsed(), name, list(warnings))

        # ── availability ──────────────────────────────────────────────────
        try:
            if not await _maybe_await(self.backend.available()):
                return result(False, ZoomTermination.BACKEND_UNAVAILABLE, None, None, 0)
        except Exception as exc:
            warnings.append(f"availability check failed: {exc}")
            return result(False, ZoomTermination.CONTROL_FAILURE, None, None, 0)

        # ── initial state ─────────────────────────────────────────────────
        try:
            start_state = await _maybe_await(self.backend.read_state())
        except Exception as exc:
            warnings.append(f"initial state read failed: {exc}")
            start_state = None

        # Already at maximum before we touch anything → idempotent fast path.
        try:
            if start_state is not None and await _maybe_await(self.backend.at_maximum(start_state)):
                return result(True, ZoomTermination.MAX_ZOOM_REACHED,
                              start_state, start_state, 0)
        except Exception as exc:
            warnings.append(f"boundary check failed: {exc}")

        state_observable = start_state is not None
        previous = start_state
        current = start_state
        unchanged_run = 0
        attempts = 0
        blind_used = 0

        while True:
            # Cancellation is checked between every attempt.
            if _is_cancelled(cancel):
                return result(False, ZoomTermination.CANCELLED, start_state, current, attempts)
            if attempts >= cfg.max_attempts:
                return result(True, ZoomTermination.MAX_ATTEMPTS, start_state, current, attempts)
            if elapsed() >= cfg.max_seconds:
                return result(True, ZoomTermination.TIMEOUT, start_state, current, attempts)
            if not state_observable and blind_used >= cfg.blind_attempts:
                warnings.append("zoom state was never observable; stopped after bounded blind attempts")
                return result(True, ZoomTermination.STATE_UNOBSERVABLE, start_state, current, attempts)

            attempts += 1
            try:
                new_state = await _maybe_await(self.backend.zoom_in_step())
            except Exception as exc:
                warnings.append(f"zoom step {attempts} failed: {exc}")
                return result(False, ZoomTermination.CONTROL_FAILURE, start_state, current, attempts)

            if new_state is None and current is not None:
                # Step gave us nothing though we had state — re-read once.
                try:
                    new_state = await _maybe_await(self.backend.read_state())
                except Exception:
                    new_state = None

            if new_state is None:
                blind_used += 1
            else:
                state_observable = True
                current = new_state
                # Backend may declare it has hit the wall.
                try:
                    if await _maybe_await(self.backend.at_maximum(current)):
                        return result(True, ZoomTermination.MAX_ZOOM_REACHED,
                                      start_state, current, attempts)
                except Exception as exc:
                    warnings.append(f"boundary check failed at attempt {attempts}: {exc}")
                if _meaningful_change(previous, current, cfg.tolerance):
                    unchanged_run = 0
                else:
                    unchanged_run += 1
                    if unchanged_run >= cfg.unchanged_threshold:
                        return result(True, ZoomTermination.STATE_STABLE,
                                      start_state, current, attempts)
                previous = current

            if cfg.delay_between_s > 0:
                await self._sleep(cfg.delay_between_s)
