"""
Spending frames only where they show.

The measured problem
--------------------
ORION's two hand-painted widgets each cost about 7 ms of QPainter work per
frame, and both ran at a fixed 30 Hz:

    CentralHud    7.04 ms/frame -> 211 ms/sec  (21% of a core)
    HologramFace  7.66 ms/frame -> 230 ms/sec  (23% of a core)
                                   ---------
                                   441 ms/sec  (44% of one core)

Under qasync the Qt thread IS the asyncio event loop, so that 44% is not
competing with ORION's brain from a safe distance — it is *inside* it. Every
7 ms paint is 7 ms during which the audio callback cannot be serviced, the
Live socket cannot be read, and a queued coroutine cannot run. That is what
"his face and orb and voice stutter alongside the GUI" is: a repaint budget
spent against a real-time deadline.

The observation that fixes it
-----------------------------
Almost all of those frames are identical. Every animated channel in both
widgets EASES toward a target — ``value += (target - value) * k`` — so once a
channel has arrived, it keeps being recomputed to the same number for as long
as nothing changes. When ORION is idle and silent, 30 frames a second are
redrawing a still picture.

So the frame rate follows the animation rather than the clock. While anything
is genuinely moving the widget runs at its full rate and looks exactly as it
did; once everything has settled it drops to IDLE_HZ, and any change — a new
amplitude, a state change, a fresh target — restores full speed on the very
next frame.

Deliberately NOT frame-skipping or interpolation: those change what the user
sees. This changes only how often an unchanged picture is redrawn, which is
invisible by construction.
"""

from __future__ import annotations

from typing import Any

#: Frames per second while something is actually moving.
ACTIVE_HZ = 30.0

#: Frames per second once every channel has settled. Not zero: a settled
#: widget still needs to notice the next change promptly, and 8 Hz is a 125 ms
#: worst-case reaction, well below what anyone perceives as lag on an idle
#: face. It also keeps slow ambient drift (scanlines, breathing) alive rather
#: than freezing the picture, which would read as a hang rather than as calm.
IDLE_HZ = 8.0

#: How many consecutive still frames before dropping down. A short run, so a
#: pause between two words does not cause a visible cadence change, but long
#: enough that easing tails are never cut off mid-motion.
SETTLE_FRAMES = 12

#: Below this, a channel counts as arrived. Chosen against the widgets' own
#: easing constants (0.08–0.45): a channel moving by less than this per frame
#: is contributing under half a pixel to the rendered result.
STILL_EPSILON = 0.002


class AnimationBudget:
    """Tracks whether a widget is really animating, and picks its frame rate.

    Used by mixing into a QWidget that owns a QTimer and eases a handful of
    float channels. The widget reports its channel values each tick; this
    decides how fast the timer should run.
    """

    def __init__(self, timer: Any, active_hz: float = ACTIVE_HZ,
                 idle_hz: float = IDLE_HZ) -> None:
        self._timer = timer
        self._active_ms = max(1, int(round(1000.0 / max(1.0, active_hz))))
        self._idle_ms = max(1, int(round(1000.0 / max(1.0, idle_hz))))
        self._previous: tuple[float, ...] = ()
        self._still_for = 0
        self._idling = False

    @property
    def idling(self) -> bool:
        return self._idling

    @property
    def still_frames(self) -> int:
        return self._still_for

    def note(self, channels: tuple[float, ...]) -> bool:
        """Record this frame's channel values. Returns whether to repaint.

        A still frame while already idling does not need a repaint at all —
        the picture is identical to the one on screen. That is where most of
        the saving comes from: not a cheaper paint, but no paint.
        """
        moved = self._moved(channels)
        self._previous = channels
        if moved:
            self._still_for = 0
            if self._idling:
                self._idling = False
                self._apply(self._active_ms)
            return True
        self._still_for += 1
        if not self._idling and self._still_for >= SETTLE_FRAMES:
            self._idling = True
            self._apply(self._idle_ms)
        # While idling, still repaint occasionally so ambient drift and any
        # painting that does not read from these channels stays alive.
        return not self._idling

    def _moved(self, channels: tuple[float, ...]) -> bool:
        previous = self._previous
        if len(previous) != len(channels):
            return True
        for was, now in zip(previous, channels):
            if abs(now - was) > STILL_EPSILON:
                return True
        return False

    def wake(self) -> None:
        """Force full rate immediately — for a state change the channels have
        not caught up to yet."""
        self._still_for = 0
        if self._idling:
            self._idling = False
            self._apply(self._active_ms)

    def _apply(self, interval_ms: int) -> None:
        try:
            if self._timer.interval() != interval_ms:
                self._timer.setInterval(interval_ms)
        except Exception:
            pass          # a stub timer in tests, or a deleted Qt object


__all__ = ["ACTIVE_HZ", "IDLE_HZ", "SETTLE_FRAMES", "STILL_EPSILON",
           "AnimationBudget"]
