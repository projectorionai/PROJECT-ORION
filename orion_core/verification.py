"""
VisualVerificationEngine (Phase 3) — no autonomous action is trusted until
vision confirms it.

This engine sits above the AutonomousControlLayer and the VisionAgent (it is
the only place the two meet) and turns every mutating action into a verified
transaction:

    1. capture the affected region (or the UIA element's rect) *before*;
    2. perform the action;
    3. capture *after*;
    4. compare — pixel-change ratio (cv2/numpy) and/or the appearance of an
       expected UI element / disappearance of a dialog;
    5. on failure, recalculate coordinates from the live UIA tree and retry,
       up to a bounded number of attempts;
    6. explain the outcome.

It exposes two levels:
    • ``verify_action`` — wrap any control callable with a before/after diff.
    • ``click_element`` / ``click_text`` — locate a control by name via UIA,
      click its centre, and verify the screen actually changed; recompute the
      element position on retry (self-correcting coordinates).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .bus import OrionBus
from .control import AutonomousControlLayer
from .data import ToolResult
from .display import DisplayTopologyManager
from .navigation_trace import NavigationTrace
from .vision import VisionAgent


@dataclass
class VerificationResult:
    ok: bool
    change_ratio: float
    attempts: int
    detail: str

    def to_tool_result(self) -> ToolResult:
        return ToolResult(self.detail, ok=self.ok)


class VisualVerificationEngine:
    """Act → capture → verify → retry, with self-correcting coordinates."""

    def __init__(
        self,
        bus: OrionBus,
        control: AutonomousControlLayer,
        vision: VisionAgent,
        display: DisplayTopologyManager,
        telemetry: Any | None = None,
        trace: NavigationTrace | None = None,
    ) -> None:
        self.bus = bus
        self.control = control
        self.vision = vision
        self.display = display
        self.telemetry = telemetry
        # Mark XXI, Track D7: a rolling record of every targeting attempt —
        # built so D1-D6's fixes could be measured against real data,
        # and so a live failure is diagnosable from Diagnostics rather than
        # guessed at. Defaults to a private instance so this engine never
        # needs a None-check at every call site; app.py attaches the one
        # shared instance other subsystems (Diagnostics) also read.
        self.trace = trace if trace is not None else NavigationTrace(bus)

    # ── low-level diff ────────────────────────────────────────────────────────

    def _change_ratio(self, before: Any, after: Any) -> float:
        """Fraction of pixels that changed meaningfully between two captures."""
        try:
            import cv2
            import numpy as np
            if before is None or after is None:
                return 0.0
            if before.shape != after.shape:
                h = min(before.shape[0], after.shape[0])
                w = min(before.shape[1], after.shape[1])
                before, after = before[:h, :w], after[:h, :w]
            gb = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
            ga = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(gb, ga)
            changed = int(np.count_nonzero(diff > 25))
            total = diff.size or 1
            return changed / total
        except Exception:
            return 0.0

    async def _capture(self, region: Optional[tuple[int, int, int, int]]) -> Any:
        return await asyncio.to_thread(self.vision.grabber.capture_array, region)

    # ── generic verified action ───────────────────────────────────────────────

    async def verify_action(
        self,
        action: Callable[[], ToolResult],
        region: Optional[tuple[int, int, int, int]] = None,
        min_change: float = 0.002,
        settle_s: float = 0.35,
        attempts: int = 2,
        repeatable: bool = True,
    ) -> VerificationResult:
        """
        Run *action* (a control-layer call) and confirm the screen changed.

        ``region`` bounds the comparison and must be known BEFORE acting (the
        caller derives it from the target); None compares the whole screen.
        It used to fall back to ``control.last_action`` read before the action
        ran — the PREVIOUS action's region — and, when that was empty, to
        capture "before" only AFTER acting, comparing the result with itself.

        ``repeatable=False`` for anything that must never happen twice:
        typing, clicks, drags. An unchanged screen is then re-measured after a
        longer settle, never answered by doing it again — a retry used to type
        the whole text a second time, and re-click a toggle straight back off.
        """
        last_detail = ""
        best_ratio = 0.0
        for attempt in range(1, max(1, attempts) + 1):
            target_region = region
            before = await self._capture(target_region)
            result = await asyncio.to_thread(action) if not asyncio.iscoroutinefunction(action) else await action()
            if not result.ok:
                return VerificationResult(False, 0.0, attempt, result.text)
            last_detail = result.text
            waits = [settle_s] if repeatable else [settle_s, max(0.6, settle_s * 2)]
            for wait in waits:
                await asyncio.sleep(wait)
                after = await self._capture(target_region)
                ratio = self._change_ratio(before, after)
                best_ratio = max(best_ratio, ratio)
                if self.telemetry is not None:
                    self.telemetry.metrics.observe("verify.change_ratio_pct", ratio * 100.0)
                if ratio >= min_change:
                    return VerificationResult(
                        True, ratio, attempt,
                        f"{last_detail}  [verified: {ratio*100:.2f}% of the region changed]",
                    )
            if not repeatable:
                break
            self.bus.log.emit(
                f"VERIFY: attempt {attempt} saw only {ratio*100:.2f}% change; retrying."
            )
        if not repeatable:
            return VerificationResult(
                False, best_ratio, 1,
                f"{last_detail}  [unverified: the screen barely changed "
                f"({best_ratio*100:.2f}%). Not repeated — look before trying again, "
                f"so nothing is typed or clicked twice]")
        return VerificationResult(
            False, best_ratio, attempts,
            f"{last_detail}  [unverified: screen barely changed "
            f"({best_ratio*100:.2f}%); the action may not have registered]",
        )

    # ── element-targeted, self-correcting click ───────────────────────────────

    async def _click_and_verify(
        self, query: str, locate: Callable[[], Awaitable[dict[str, Any] | None]],
        attempts: int, strategy: str,
    ) -> ToolResult:
        """
        Shared click+verify+retry loop: re-locate via *locate* each attempt
        (self-correcting coordinates), click, verify the screen actually
        changed. Used by both the UIA path (click_element) and the OCR
        fallback (click_via_ocr, Track D2) so they share one shape — and
        every attempt is recorded to self.trace (Track D7).
        """
        for attempt in range(1, max(1, attempts) + 1):
            element = await locate()
            if element is None:
                if attempt == 1:
                    detail = await self._not_found_message(query, strategy)
                    self.trace.record(query=query, strategy=strategy, found=False,
                                      detail=detail)
                    return ToolResult(detail, ok=False)
                await asyncio.sleep(0.3)
                continue
            cx, cy = element["center"]
            region = self._padded_region(element["rect"])
            before = await self._capture(region)
            click = await asyncio.to_thread(self.control.click, cx, cy)
            if not click.ok:
                self.trace.record(query=query, strategy=strategy, found=True,
                                  coordinates=(cx, cy), verified=False, detail=click.text)
                return click
            await asyncio.sleep(0.4)
            after = await self._capture(region)
            ratio = self._change_ratio(before, after)
            if self.telemetry is not None:
                self.telemetry.metrics.observe("verify.click_change_pct", ratio * 100.0)
            if ratio >= 0.003:
                self.trace.record(query=query, strategy=strategy, found=True,
                                  coordinates=(cx, cy), verified=True, change_ratio=ratio)
                return ToolResult(
                    f"Clicked '{element['name']}' [{element['role']}] at ({cx},{cy}); "
                    f"verified ({ratio*100:.2f}% change)."
                )
            self.bus.log.emit(
                f"VERIFY: click on '{element['name']}' showed {ratio*100:.2f}% change; "
                f"recalculating (attempt {attempt})."
            )
            await asyncio.sleep(0.3)
        self.trace.record(query=query, strategy=strategy, found=True, verified=False)
        return ToolResult(
            f"Clicked '{query}' but could not visually confirm a response after "
            f"{attempts} attempts. It may be a no-op control, or the view did not change.",
            ok=False,
        )

    async def click_element(self, query: str, kinds: str = "all",
                            attempts: int = 3) -> ToolResult:
        """
        Find a control by visible name, click its centre, and verify.

        On each retry the element is re-located from the live UIA tree, so a
        moved/re-laid-out control is clicked at its *new* position rather than
        a stale coordinate — the self-correcting-coordinates requirement.
        """
        return await self._click_and_verify(
            query, lambda: self.vision.find_element(query, kinds=kinds), attempts, "uia")

    async def click_via_ocr(self, query: str, attempts: int = 2) -> ToolResult:
        """
        OCR-based click fallback (Mark XXI, Track D2) — used when the
        accessibility tree comes back empty for *query* across every role.
        Electron/canvas UIs, games and some installers expose an empty or
        unreliable UIA tree; this used to be a dead end with no second
        strategy at all. Mirrors click_element's self-correcting retry
        shape, re-locating via OCR each attempt instead.
        """
        return await self._click_and_verify(
            query, lambda: self.vision.find_element_via_ocr(query), attempts, "ocr")

    async def _not_found_message(self, query: str, strategy: str) -> str:
        """An informative 'not found' message (Track D5) — what WAS on
        screen instead of a bare dead end, so a retry can be intelligent
        rather than repeating the identical failed query."""
        parts = [f"Could not find '{query}' on screen."]
        try:
            candidates = await self.vision.nearby_candidates(query)
        except Exception:
            candidates = []
        if candidates:
            parts.append(f"The nearest matches on screen were: {', '.join(candidates[:3])}.")
        window = ""
        try:
            window = self.control._foreground_title()
        except Exception:
            pass
        if window:
            parts.append(f"The active window is {window}.")
        return " ".join(parts)

    async def click_text(self, text: str, attempts: int = 3) -> ToolResult:
        """
        Click a button/link/menu item by its visible text, scrolling to find
        it in both directions if necessary (Mark XXI, Track D3).

        Web targets are frequently below OR above the current viewport, or
        not yet realised in the accessibility tree until they approach it,
        which is the usual reason a first attempt "can't find" a section
        title. So if the element isn't located at the current scroll
        position, ORION scrolls down in steps, then back past the origin
        and up — the previous version only ever scrolled down, so anything
        above the fold was structurally unreachable. On total failure the
        view is scrolled back to exactly where the search started, rather
        than left wherever the last, unsuccessful probe happened to land.
        """
        located = await self._locate_and_click(text, attempts)
        if located.ok:
            return located

        step = 4
        net_scrolled = 0

        for _ in range(5):
            try:
                self.control.scroll(-step)          # negative = scroll down
            except Exception:
                break
            net_scrolled -= step
            await asyncio.sleep(0.35)
            probe = await self._locate_and_click(text, attempts)
            if probe.ok:
                return probe

        if net_scrolled:
            try:
                self.control.scroll(-net_scrolled)  # back to the starting point
                net_scrolled = 0
            except Exception:
                pass
            await asyncio.sleep(0.35)

        for _ in range(5):
            try:
                self.control.scroll(step)           # positive = scroll up
            except Exception:
                break
            net_scrolled += step
            await asyncio.sleep(0.35)
            probe = await self._locate_and_click(text, attempts)
            if probe.ok:
                return probe

        if net_scrolled:
            try:
                self.control.scroll(-net_scrolled)  # restore the original position
            except Exception:
                pass
        return located  # the original, informative failure

    async def _locate_and_click(self, text: str, attempts: int) -> ToolResult:
        """Try the most interactive roles first for a precise hit; no
        scrolling. Falls back to OCR (Track D2) only when UIA finds
        nothing under any role."""
        for kinds in ("button", "link", "menu"):
            element = await self.vision.find_element(text, kinds=kinds)
            if element is not None:
                return await self.click_element(text, kinds=kinds, attempts=attempts)
        element = await self.vision.find_element(text, kinds="all")
        if element is not None:
            return await self.click_element(text, kinds="all", attempts=attempts)
        return await self.click_via_ocr(text, attempts=max(1, attempts - 1))

    def _padded_region(self, rect: tuple[int, int, int, int], pad: int = 40) -> tuple[int, int, int, int]:
        x, y, w, h = rect
        x0, y0 = self.display.clamp_to_desktop(x - pad, y - pad)
        x1, y1 = self.display.clamp_to_desktop(x + w + pad, y + h + pad)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # ── diagnostics ───────────────────────────────────────────────────────────

    async def snapshot_change(self, region: Optional[tuple[int, int, int, int]],
                              settle_s: float = 0.4) -> float:
        """Measure how much a region changes over a short window (idle probe)."""
        before = await self._capture(region)
        await asyncio.sleep(settle_s)
        after = await self._capture(region)
        return self._change_ratio(before, after)
