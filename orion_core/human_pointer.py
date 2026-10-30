"""
Human-like pointer motion: where the cursor goes, and when.

The owner asked for ORION to "move the cursor in a human way and click, or do
it fast". The control layer used to glide in a straight line with pyautogui's
easeInOutQuad (0.15-0.55 s) and click with a press and release in the same
instant. A person does neither: the hand travels on a slight arc, accelerates
and brakes smoothly, overshoots a far target a little and corrects, and holds
the button for about a tenth of a second.

Like human_typing, the PLANNING here is a pure, seedable function — no mouse,
no screen, no clock — so it is unit-tested directly; control.py only replays a
plan (move, sleep, move...) on its worker thread, never on the GUI thread.

The model:
    * duration — Fitts's law, MT = a + b*log2(D/W + 1), clamped to 0.25-0.9 s;
    * path     — a cubic Bezier whose control points sit at increasing
                 fractions along the straight line, bowed to one side. Their
                 along-track spacing keeps progress towards the target
                 monotonic for the main movement;
    * timing   — a minimum-jerk profile (10t^3 - 15t^4 + 6t^5): ease-in and
                 ease-out with no sudden start or stop, peak speed a little
                 before halfway;
    * tremor   — sub-pixel perpendicular jitter, zero at both ends;
    * overshoot— on long moves the main movement lands a few pixels past the
                 target and a short correction brings it back;
    * clicks   — a 60-140 ms press, and a double-click gap that stays well
                 inside the system's double-click time.

``fast`` mode is the old behaviour's spirit at its extreme: one jump to the
target and an immediate click.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

HUMAN = "human"
FAST = "fast"

_FAST_WORDS = {"fast", "instant", "instantly", "quick", "quickly", "direct", "jump",
               "machine", "rapid", "immediate", "immediately"}
_HUMAN_WORDS = {"human", "humanlike", "human-like", "human_like", "natural", "naturally",
                "smooth", "normal", "realistic", "slow", "slowly"}

#: Fitts's law constants (seconds, seconds per bit) for a mouse, from the
#: middle of the published range; the clamp below is what the owner asked for.
FITTS_A = 0.20
FITTS_B = 0.11
MIN_MOVE_S = 0.25
MAX_MOVE_S = 0.90
#: One sample every ~10 ms, near a real mouse's 125 Hz report rate.
SAMPLE_S = 0.010
#: Moves shorter than this are a nudge: one smooth hop, no arc, no overshoot.
NUDGE_PX = 3.0
#: Moves at least this long may overshoot and correct.
OVERSHOOT_MIN_PX = 320.0
OVERSHOOT_CHANCE = 0.7

#: How long a button is held, and the release-to-press gap of a double click.
PRESS_MIN_S = 0.060
PRESS_MAX_S = 0.140
GAP_MIN_S = 0.070
GAP_MAX_S = 0.150
#: A short look at the target before pressing.
SETTLE_MIN_S = 0.020
SETTLE_MAX_S = 0.070


# ── the speed of the current request ─────────────────────────────────────────

_SPEED: ContextVar[Optional[str]] = ContextVar("orion_input_speed", default=None)


def normalise_speed(value: Any) -> Optional[str]:
    """'human' or 'fast' from whatever the caller said, or None if unclear."""
    if isinstance(value, bool):
        return FAST if value else None
    text = str(value or "").strip().lower()
    if text in _FAST_WORDS:
        return FAST
    if text in _HUMAN_WORDS:
        return HUMAN
    return None


@contextmanager
def speed_scope(value: Any) -> Iterator[Optional[str]]:
    """Make *value* the input speed for everything this request does.

    A context variable, so it follows the request into asyncio.to_thread
    workers (the verifier's click, the scroll search) without threading a
    parameter through every call, and never leaks into another request."""
    speed = normalise_speed(value)
    token = _SPEED.set(speed)
    try:
        yield speed
    finally:
        _SPEED.reset(token)


def current_speed() -> Optional[str]:
    return _SPEED.get()


# ── pointer paths ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PointerStep:
    """Move to (x, y), then wait *delay* seconds before the next step."""
    x: int
    y: int
    delay: float


@dataclass(frozen=True)
class PointerPlan:
    steps: tuple[PointerStep, ...]
    mode: str
    overshoot: bool = False

    @property
    def duration(self) -> float:
        return sum(step.delay for step in self.steps)

    @property
    def end(self) -> Optional[tuple[int, int]]:
        return (self.steps[-1].x, self.steps[-1].y) if self.steps else None


def fitts_duration(distance: float, target_width: float = 24.0,
                   rng: Optional[random.Random] = None) -> float:
    """Seconds for a movement of *distance* px to a target *target_width* px
    wide: Fitts's law with +-10% individual variation, kept in 0.25-0.9 s.
    Zero for no movement."""
    if distance < 0.5:
        return 0.0
    index = math.log2(distance / max(4.0, float(target_width)) + 1.0)
    seconds = FITTS_A + FITTS_B * index
    if rng is not None:
        seconds *= rng.uniform(0.9, 1.1)
    return min(MAX_MOVE_S, max(MIN_MOVE_S, seconds))


def minimum_jerk(t: float) -> float:
    """Position fraction at time fraction *t*: smooth start and stop."""
    t = min(1.0, max(0.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _bezier(p0: float, c1: float, c2: float, p3: float, t: float) -> float:
    u = 1.0 - t
    return u * u * u * p0 + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t * t * t * p3


def _segment(start: tuple[float, float], end: tuple[float, float], seconds: float,
             rng: random.Random, *, bow: float, tremor: float, skew: float
             ) -> list[tuple[float, float, float]]:
    """(x, y, progress) samples after *start* up to exactly *end*."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    distance = math.hypot(dx, dy)
    if distance < 1e-9:
        return [(ex, ey, 1.0)]
    ux, uy = dx / distance, dy / distance
    nx, ny = -uy, ux
    # Control points at increasing fractions of the way (0 < c1 < c2 < 1):
    # the curve's projection onto the line is then strictly increasing.
    c1, c2 = rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)
    b1, b2 = bow * rng.uniform(0.6, 1.2), bow * rng.uniform(0.4, 1.0)
    cx1, cy1 = sx + dx * c1 + nx * b1, sy + dy * c1 + ny * b1
    cx2, cy2 = sx + dx * c2 + nx * b2, sy + dy * c2 + ny * b2
    count = max(2, int(round(seconds / SAMPLE_S)))
    samples = []
    for i in range(1, count + 1):
        frac = i / count
        # skew < 1 puts peak speed a little before halfway, as in real reaches.
        t = minimum_jerk(frac ** skew)
        x = _bezier(sx, cx1, cx2, ex, t)
        y = _bezier(sy, cy1, cy2, ey, t)
        if tremor > 0 and i < count:
            shake = rng.gauss(0.0, tremor) * math.sin(math.pi * frac)
            x, y = x + nx * shake, y + ny * shake
        samples.append((x, y, t))
    samples[-1] = (float(ex), float(ey), 1.0)
    return samples


def plan_pointer_path(start: tuple[int, int], end: tuple[int, int], *,
                      mode: str = HUMAN, rng: Optional[random.Random] = None,
                      seed: Optional[int] = None, target_width: float = 24.0,
                      overshoot: bool = True,
                      inside: Optional[Callable[[int, int], bool]] = None,
                      duration_scale: float = 1.0) -> PointerPlan:
    """The steps that take the pointer from *start* to exactly *end*.

    ``fast``: a single jump. ``human``: a curved, eased, slightly trembling
    path timed by Fitts's law, with a small overshoot and correction on long
    moves when *overshoot* allows. *inside(x, y)* says whether a point is on
    a screen; points that would leave the desktop fall back onto the straight
    line, and an overshoot that would leave it is not made. Deterministic for
    a given *seed* or *rng* state."""
    ex, ey = int(end[0]), int(end[1])
    if normalise_speed(mode) == FAST:
        return PointerPlan((PointerStep(ex, ey, 0.0),), FAST)
    rng = rng if rng is not None else random.Random(seed)
    sx, sy = int(start[0]), int(start[1])
    distance = math.hypot(ex - sx, ey - sy)
    if distance < 0.5:
        return PointerPlan((PointerStep(ex, ey, 0.0),), HUMAN)
    seconds = fitts_duration(distance, target_width, rng) * max(0.5, float(duration_scale))
    seconds = min(MAX_MOVE_S * max(1.0, float(duration_scale)), seconds)
    if distance < NUDGE_PX:
        count = 3
        raw = [(sx + (ex - sx) * minimum_jerk(i / count), sy + (ey - sy) * minimum_jerk(i / count))
               for i in range(1, count + 1)]
        steps = tuple(PointerStep(int(round(x)), int(round(y)), MIN_MOVE_S / count / 4)
                      for x, y in raw)
        return PointerPlan(steps[:-1] + (PointerStep(ex, ey, steps[-1].delay),), HUMAN)

    side = rng.choice((-1.0, 1.0))
    bow = side * min(120.0, distance * rng.uniform(0.04, 0.16))
    tremor = 0.45
    skew = rng.uniform(0.85, 1.0)
    ux, uy = (ex - sx) / distance, (ey - sy) / distance
    nx, ny = -uy, ux

    landing: Optional[tuple[float, float]] = None
    if overshoot and distance >= OVERSHOOT_MIN_PX and rng.random() < OVERSHOOT_CHANCE:
        past = min(28.0, max(6.0, distance * rng.uniform(0.025, 0.06)))
        drift = rng.uniform(-0.3, 0.3) * past
        ox, oy = ex + ux * past + nx * drift, ey + uy * past + ny * drift
        if inside is None or inside(int(round(ox)), int(round(oy))):
            landing = (ox, oy)

    # (from, to, seconds, bow, tremor, skew) for each sub-movement.
    phases: list[tuple[tuple[float, float], tuple[float, float], float, float, float, float]]
    if landing is None:
        phases = [((sx, sy), (ex, ey), seconds, bow, tremor, skew)]
    else:
        main_s = seconds * 0.82
        phases = [((sx, sy), landing, main_s, bow, tremor, skew),
                  (landing, (ex, ey), max(0.07, seconds - main_s), 0.0, 0.0, 1.0)]

    steps: list[PointerStep] = []
    last = (sx, sy)
    for (fx, fy), (tx, ty), phase_s, phase_bow, phase_tremor, phase_skew in phases:
        samples = _segment((fx, fy), (tx, ty), phase_s, rng,
                           bow=phase_bow, tremor=phase_tremor, skew=phase_skew)
        dt = phase_s / len(samples)
        span = math.hypot(tx - fx, ty - fy) or 1.0
        vx, vy = (tx - fx) / span, (ty - fy) / span
        best = 0.0                   # furthest progress made in this phase
        for x, y, t in samples:
            px, py = int(round(x)), int(round(y))
            if inside is not None and not inside(px, py):
                # Off every screen (a gap between monitors, past an edge):
                # take the straight line's point at the same progress.
                px, py = int(round(fx + (tx - fx) * t)), int(round(fy + (ty - fy) * t))
            progress = (px - fx) * vx + (py - fy) * vy
            final = (px, py) == (int(round(tx)), int(round(ty))) and t >= 1.0
            if (px, py) == last or (not final and not best <= progress <= span):
                # A repeat, a sub-pixel step back or past this phase's end:
                # wait here instead, so the timing is kept and progress never
                # runs backwards.
                if steps:
                    tail = steps[-1]
                    steps[-1] = PointerStep(tail.x, tail.y, tail.delay + dt)
                else:
                    steps.append(PointerStep(last[0], last[1], dt))
                continue
            best = progress
            steps.append(PointerStep(px, py, dt))
            last = (px, py)
    if (steps[-1].x, steps[-1].y) != (ex, ey):
        steps.append(PointerStep(ex, ey, 0.0))
    return PointerPlan(tuple(steps), HUMAN, overshoot=landing is not None)


# ── clicks ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClickTiming:
    settle: float                 # pause on arrival before the first press
    holds: tuple[float, ...]      # how long each press is held down
    gaps: tuple[float, ...]       # release-to-press gaps between presses


def plan_click(clicks: int = 1, *, mode: str = HUMAN,
               rng: Optional[random.Random] = None, seed: Optional[int] = None,
               double_click_s: float = 0.5) -> ClickTiming:
    """Press, hold and gap timings for *clicks* presses.

    Human: each press held 60-140 ms (a truncated normal around 95 ms) and,
    for double/triple clicks, release-to-press gaps of 70-150 ms, shortened if
    needed so a hold plus a gap stays inside 60% of the system double-click
    time (*double_click_s*). Fast: no settle, no hold, no gap."""
    clicks = max(1, int(clicks))
    if normalise_speed(mode) == FAST:
        return ClickTiming(0.0, (0.0,) * clicks, (0.0,) * (clicks - 1))
    rng = rng if rng is not None else random.Random(seed)

    def bounded(mean: float, sd: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, rng.gauss(mean, sd)))

    holds = tuple(bounded(0.095, 0.02, PRESS_MIN_S, PRESS_MAX_S) for _ in range(clicks))
    window = max(0.1, float(double_click_s)) * 0.6
    gaps = []
    for i in range(clicks - 1):
        gap = bounded(0.105, 0.022, GAP_MIN_S, GAP_MAX_S)
        gaps.append(max(0.03, min(gap, window - holds[i])))
    return ClickTiming(rng.uniform(SETTLE_MIN_S, SETTLE_MAX_S), holds, tuple(gaps))
