"""
pick.py — resolving a click to a node (Mark XXII, Phase 1).

Ray/sphere intersection on the CPU, deliberately in preference to the usual
colour-ID picking pass. Colour-ID means a second offscreen render target, a
second draw of the whole scene, a pipeline stall on glReadPixels, and logic
that can only be exercised with a live GL context — which this suite has
already been burned by. Ray casting against the instance buffer is a linear
scan over a contiguous float32 array: at the 1,700-node target that is
microseconds, it needs no GPU round trip, and it is fully unit-testable.

The ray is unprojected from the camera matrices, so picking automatically
stays correct under orbit, pan and zoom without any separate bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .instances import _OFF_POS, _OFF_RADIUS


def ray_from_screen(
    x: float, y: float, width: float, height: float,
    view_projection: np.ndarray, eye: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a pixel into a world-space ray (origin, unit direction).

    *view_projection* is the row-major combined matrix; *eye* is the camera
    position. Qt gives y downward from the top, OpenGL NDC runs upward from
    the bottom, so y is flipped here — getting this wrong produces picking
    that is mirrored vertically and is maddening to diagnose from the UI."""
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    ndc_x = (2.0 * float(x)) / width - 1.0
    ndc_y = 1.0 - (2.0 * float(y)) / height

    try:
        inverse = np.linalg.inv(view_projection)
    except np.linalg.LinAlgError:      # pragma: no cover — degenerate matrix
        return np.asarray(eye, dtype=np.float64), np.array([0.0, 0.0, -1.0])

    near = inverse @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    far = inverse @ np.array([ndc_x, ndc_y, 1.0, 1.0])
    if abs(near[3]) < 1e-12 or abs(far[3]) < 1e-12:   # pragma: no cover
        return np.asarray(eye, dtype=np.float64), np.array([0.0, 0.0, -1.0])
    near = near[:3] / near[3]
    far = far[:3] / far[3]

    direction = far - near
    length = float(np.linalg.norm(direction))
    if length < 1e-12:                                # pragma: no cover
        return np.asarray(eye, dtype=np.float64), np.array([0.0, 0.0, -1.0])
    return near, direction / length


def pick_slot(
    instances: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    draw_count: int,
    padding: float = 0.35,
) -> int | None:
    """The slot of the nearest sphere the ray hits, or None.

    Vectorised over the whole buffer rather than looped: one pass of numpy
    arithmetic across every instance beats a Python loop by a wide margin and
    keeps picking imperceptible even as the graph grows.

    *padding* enlarges each sphere's hit radius slightly. Small nodes (an MCP
    tool at radius 0.45) are otherwise fiddly to hit at a distance, and a
    click that lands one pixel outside the silhouette reads to the user as
    the app ignoring them.
    """
    if draw_count <= 0 or instances.size == 0:
        return None
    data = instances[:draw_count]
    centres = data[:, _OFF_POS:_OFF_POS + 3].astype(np.float64)
    radii = data[:, _OFF_RADIUS].astype(np.float64) + float(padding)

    # Retired slots have radius 0 and must never be pickable — otherwise a
    # click could select a node that no longer exists.
    live = data[:, _OFF_RADIUS] > 0.0
    if not live.any():
        return None

    to_centre = centres - np.asarray(origin, dtype=np.float64)
    along = to_centre @ np.asarray(direction, dtype=np.float64)
    # Perpendicular distance² from each centre to the ray line.
    perpendicular_sq = np.einsum("ij,ij->i", to_centre, to_centre) - along * along

    hit = live & (along > 0.0) & (perpendicular_sq <= radii * radii)
    if not hit.any():
        return None

    # Nearest along the ray, measured to the sphere's near surface so a large
    # distant node cannot win over a small one directly in front of it.
    depth = along - np.sqrt(np.maximum(0.0, radii * radii - perpendicular_sq))
    depth = np.where(hit, depth, np.inf)
    return int(np.argmin(depth))
