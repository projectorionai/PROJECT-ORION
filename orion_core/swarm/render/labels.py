"""
labels.py — the holographic callout ring (Mark XXIII).

Aerospace targeting rather than UI widgets: subsystem names sit around the
outer edge of the viewport, each tethered to its live node by a leader line,
never overlapping, sliding smoothly as the cognition rings turn beneath them.

THE ALGORITHM. Collision-free 2D label placement is genuinely hard in the
general case. This makes it easy by removing a dimension: every label is
constrained to a PERIMETER, so its position is a single scalar — arc length
around that perimeter. Overlap resolution then becomes 1-D constraint
relaxation (sort by preferred position, sweep out, sweep back), which is
cheap, stable, and cannot produce the jitter or oscillation a 2-D force
solver does when two labels shove each other in opposite directions.

Anchors come from `orbital_position` at the same instant the frame is drawn,
so a callout tracks its subsystem exactly as the rings rotate rather than
drifting behind it.

Pure numpy — no Qt, no GL, no text rasterisation. This module decides WHERE
every callout and leader line goes; drawing them is the renderer's job. That
split is what lets the whole placement system be unit-tested headlessly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Fraction of the viewport's half-width/height the callout ring sits at. Just
# inside the edge, so labels read as an instrument bezel around the scene
# rather than floating loose in it.
RING_INSET_X = 0.86
RING_INSET_Y = 0.80

# Minimum arc-length gap between adjacent callouts, in pixels. Enforced by
# the relaxation sweep below.
MIN_LABEL_GAP = 26.0

# Minimum VERTICAL separation, in pixels, between two labels on the same side.
# Text runs horizontally, so this — not arc length — is what actually decides
# whether two callouts print through each other. Sized to the rendered line
# height of the callout type (a 48px atlas at CALLOUT_SCALE 0.30).
MIN_ROW_GAP = 20.0

# A node closer to the screen centre than this (as a fraction of the ring
# radius) has no meaningful outward direction, so its callout would flap
# between sides frame to frame. Those are dropped rather than jitter.
_CENTRE_DEADZONE = 0.06


@dataclass(frozen=True, slots=True)
class Callout:
    """One placed label: where the text goes, and what it points at."""

    node_id: str
    text: str
    # Screen-space, pixels, origin top-left.
    anchor: tuple[float, float]     # the node itself
    position: tuple[float, float]   # where the label sits on the ring
    elbow: tuple[float, float]      # leader-line bend, just inside the ring
    side: str                       # "left" | "right" — text alignment
    alpha: float                    # fade from depth and edge-on angle
    perimeter_t: float              # its resolved slot, for stable ordering

    @property
    def leader(self) -> tuple[tuple[float, float], ...]:
        """The polyline from subsystem to label: anchor -> elbow -> label.

        A single bend rather than a straight line is what makes the routing
        read as an instrument callout instead of a spider web, and the
        horizontal final segment gives the text something to sit against."""
        return (self.anchor, self.elbow, self.position)


def project_to_screen(
    points: np.ndarray, view_projection: np.ndarray, width: float, height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels. Returns (screen_xy, depth_w).

    depth_w is the clip-space w — positive means in front of the camera.
    Returned rather than discarded because callouts for nodes BEHIND the
    viewer must be suppressed, and w is the only thing that distinguishes
    them from ones in front after the perspective divide.
    """
    if points.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ np.asarray(view_projection, dtype=np.float64).T
    w = clip[:, 3].copy()
    safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / safe[:, None]
    screen = np.column_stack([
        (ndc[:, 0] * 0.5 + 0.5) * width,
        # Y flips: NDC runs up from the bottom, screen runs down from the top.
        (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height,
    ])
    return screen, w


def _perimeter_point(t: float, cx: float, cy: float,
                     rx: float, ry: float) -> tuple[float, float]:
    """Point at arc parameter *t* (radians) on the callout ellipse."""
    return (cx + math.cos(t) * rx, cy + math.sin(t) * ry)


def _relax(slots: list[float], min_gap_rad: float) -> list[float]:
    """Push overlapping slots apart along the perimeter.

    Two sweeps — forward then backward — over a sorted list. This is the
    whole reason placement is constrained to a 1-D parameter: with labels
    ordered around the ring, resolving overlap is a monotone pass rather
    than a force simulation that can oscillate forever between two labels
    pushing each other in opposite directions.
    """
    if not slots:
        return slots
    out = list(slots)
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < min_gap_rad:
            out[i] = out[i - 1] + min_gap_rad
    # Backward pass keeps the block centred on its original span instead of
    # letting everything creep in one direction.
    for i in range(len(out) - 2, -1, -1):
        if out[i + 1] - out[i] < min_gap_rad:
            out[i] = out[i + 1] - min_gap_rad
    return out


def layout_callouts(
    entries: list[tuple[str, str, tuple[float, float, float]]],
    view_projection: np.ndarray,
    width: float,
    height: float,
    min_gap: float = MIN_LABEL_GAP,
    max_labels: int = 24,
    min_row_gap: float = MIN_ROW_GAP,
) -> list[Callout]:
    """Place callouts for *entries* — (node_id, text, world_position).

    Returns only the labels that should actually be drawn: nodes behind the
    camera and those too close to the screen centre to have a stable outward
    direction are dropped rather than allowed to flicker.
    """
    if not entries or width <= 0 or height <= 0:
        return []

    points = np.array([e[2] for e in entries], dtype=np.float64)
    screen, depth = project_to_screen(points, view_projection, width, height)

    cx, cy = width * 0.5, height * 0.5
    rx, ry = cx * RING_INSET_X, cy * RING_INSET_Y

    candidates: list[tuple[float, int, float]] = []   # (angle, index, alpha)
    for i, (_node_id, _text, _pos) in enumerate(entries):
        if depth[i] <= 0.0:
            continue                                   # behind the camera
        dx = screen[i, 0] - cx
        dy = screen[i, 1] - cy
        # Normalise into the ellipse's own space so "distance from centre" is
        # measured consistently on a non-square viewport.
        nx, ny = dx / max(rx, 1e-6), dy / max(ry, 1e-6)
        radial = math.hypot(nx, ny)
        if radial < _CENTRE_DEADZONE:
            continue                                   # no stable direction
        angle = math.atan2(ny, nx)
        # Fade the ones nearly edge-on to the ring and the very distant ones,
        # so the bezel stays legible instead of crowded with faint clutter.
        alpha = float(min(1.0, 0.35 + 0.65 * min(1.0, radial)))
        candidates.append((angle, i, alpha))

    if not candidates:
        return []

    # Keep the outermost when there are more subsystems than slots: those are
    # the ones with the clearest leader lines and the least crowded anchors.
    candidates.sort(key=lambda c: -c[2])
    candidates = candidates[:max_labels]
    candidates.sort(key=lambda c: c[0])

    min_gap_rad = float(min_gap) / max(1.0, (rx + ry) * 0.5)
    resolved = _relax([c[0] for c in candidates], min_gap_rad)

    callouts: list[Callout] = []
    # Rows already occupied on each side, so labels cannot print through each
    # other. The relaxation above guarantees separation ALONG the perimeter,
    # which is exactly what is needed down the steep left and right flanks —
    # but across the top and bottom the perimeter runs horizontally, so two
    # labels a full gap apart there sit on the same line and overlap. Real
    # frames showed "BUSINESS" printed through "SYSTEM" whenever the camera
    # drifted to put several clusters overhead.
    rows: dict[str, list[float]] = {"left": [], "right": []}
    for (angle, index, alpha), slot in zip(candidates, resolved):
        node_id, text, _world = entries[index]
        position = _perimeter_point(slot, cx, cy, rx, ry)
        side = "right" if math.cos(slot) >= 0.0 else "left"
        if any(abs(position[1] - taken) < min_row_gap for taken in rows[side]):
            # Dropped rather than nudged: nudging it clear would collide with
            # the next one along, and this module's whole discipline is that a
            # callout it cannot place cleanly is a callout it does not draw.
            continue
        rows[side].append(position[1])
        # Elbow sits slightly inside the ring on the same bearing, giving the
        # leader a single clean bend into a horizontal run to the text.
        elbow = _perimeter_point(slot, cx, cy, rx * 0.88, ry * 0.88)
        callouts.append(Callout(
            node_id=node_id,
            text=text,
            anchor=(float(screen[index, 0]), float(screen[index, 1])),
            position=position,
            elbow=elbow,
            side=side,
            alpha=alpha,
            perimeter_t=float(slot),
        ))
    return callouts


def callout_at(callouts: list[Callout], x: float, y: float,
               radius: float = 90.0) -> Callout | None:
    """The callout whose TEXT is nearest (x, y), for hover and click.

    Matched against the label position rather than the node, because the
    label is what the pointer is actually over — the subsystem itself may be
    hundreds of pixels away at the other end of the leader line."""
    best: Callout | None = None
    best_distance = float(radius)
    for callout in callouts:
        distance = math.hypot(callout.position[0] - x, callout.position[1] - y)
        if distance < best_distance:
            best_distance = distance
            best = callout
    return best
