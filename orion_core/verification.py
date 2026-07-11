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
    ) -> None:
        self.bus = bus
        self.control = control
        self.vision = vision
        self.display = display
        self.telemetry = telemetry

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
    ) -> VerificationResult:
        """
        Run *action* (a control-layer call) and confirm the screen changed.

        ``region`` bounds the comparison; when None the action's own recorded
        region (control.last_action) is used, falling back to the full screen.
        """
        last_detail = ""
        best_ratio = 0.0
        for attempt in range(1, max(1, attempts) + 1):
            target_region = region or self.control.last_action.get("region")
            before = await self._capture(target_region)
            result = await asyncio.to_thread(action) if not asyncio.iscoroutinefunction(action) else await action()
            if not result.ok:
                return VerificationResult(False, 0.0, attempt, result.text)
            # If the action did not record a region, adopt whatever it set now.
            if target_region is None:
                target_region = self.control.last_action.get("region")
                before = await self._capture(target_region)
            await asyncio.sleep(settle_s)
            after = await self._capture(target_region)
            ratio = self._change_ratio(before, after)
            best_ratio = max(best_ratio, ratio)
            last_detail = result.text
            if self.telemetry is not None:
                self.telemetry.metrics.observe("verify.change_ratio_pct", ratio * 100.0)
            if ratio >= min_change:
                return VerificationResult(
                    True, ratio, attempt,
                    f"{last_detail}  [verified: {ratio*100:.2f}% of the region changed]",
                )
            self.bus.log.emit(
                f"VERIFY: attempt {attempt} saw only {ratio*100:.2f}% change; retrying."
            )
        return VerificationResult(
            False, best_ratio, attempts,
            f"{last_detail}  [unverified: screen barely changed "
            f"({best_ratio*100:.2f}%); the action may not have registered]",
        )

    # ── element-targeted, self-correcting click ───────────────────────────────

    async def click_element(self, query: str, kinds: str = "all",
                            attempts: int = 3) -> ToolResult:
        """
        Find a control by visible name, click its centre, and verify.

        On each retry the element is re-located from the live UIA tree, so a
        moved/re-laid-out control is clicked at its *new* position rather than
        a stale coordinate — the self-correcting-coordinates requirement.
        """
        for attempt in range(1, max(1, attempts) + 1):
            element = await self.vision.find_element(query, kinds=kinds)
            if element is None:
                if attempt == 1:
                    return ToolResult(
                        f"Could not find a UI element matching '{query}'. "
                        "Use vision_verify describe to see what is on screen.",
                        ok=False,
                    )
                await asyncio.sleep(0.3)
                continue
            cx, cy = element["center"]
            region = self._padded_region(element["rect"])
            before = await self._capture(region)
            click = await asyncio.to_thread(self.control.click, cx, cy)
            if not click.ok:
                return click
            await asyncio.sleep(0.4)
            after = await self._capture(region)
            ratio = self._change_ratio(before, after)
            if self.telemetry is not None:
                self.telemetry.metrics.observe("verify.click_change_pct", ratio * 100.0)
            if ratio >= 0.003:
                return ToolResult(
                    f"Clicked '{element['name']}' [{element['role']}] at ({cx},{cy}); "
                    f"verified ({ratio*100:.2f}% change)."
                )
            self.bus.log.emit(
                f"VERIFY: click on '{element['name']}' showed {ratio*100:.2f}% change; "
                f"recalculating (attempt {attempt})."
            )
            await asyncio.sleep(0.3)
        return ToolResult(
            f"Clicked '{query}' but could not visually confirm a response after "
            f"{attempts} attempts. It may be a no-op control, or the view did not change.",
            ok=False,
        )

    async def click_text(self, text: str, attempts: int = 3) -> ToolResult:
        """
        Click a button/link/menu item by its visible text, scrolling to find it.

        Web targets are frequently below the fold or not yet realised in the
        accessibility tree until they approach the viewport, which is the usual
        reason a first attempt "can't find" a section title. So if the element
        isn't located at the current scroll position, ORION scrolls the page in
        steps and retries before giving up — self-correcting navigation.
        """
        located = await self._locate_and_click(text, attempts)
        if located.ok:
            return located
        for _ in range(5):
            try:
                self.control.scroll(-4)          # negative = scroll down
            except Exception:
                break
            await asyncio.sleep(0.35)
            probe = await self._locate_and_click(text, attempts)
            if probe.ok:
                return probe
        return located  # the original, informative failure

    async def _locate_and_click(self, text: str, attempts: int) -> ToolResult:
        """Try the most interactive roles first for a precise hit; no scrolling."""
        for kinds in ("button", "link", "menu"):
            element = await self.vision.find_element(text, kinds=kinds)
            if element is not None:
                return await self.click_element(text, kinds=kinds, attempts=attempts)
        element = await self.vision.find_element(text, kinds="all")
        if element is not None:
            return await self.click_element(text, kinds="all", attempts=attempts)
        return ToolResult(f"Could not find '{text}' on screen, sir.", ok=False)

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
