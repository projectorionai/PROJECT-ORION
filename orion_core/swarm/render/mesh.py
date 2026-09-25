"""
mesh.py — the one sphere every node is drawn from (Mark XXII, Phase 1).

Generated once at startup and uploaded once. Every node in the graph is the
SAME mesh, drawn by a single instanced call with per-instance position,
colour, radius and state — which is what makes 1,700 nodes one draw call
instead of 1,700, and what makes "do not recreate geometry every frame"
structurally true rather than merely intended.

An icosphere rather than a UV sphere: a UV sphere concentrates vertices at
the poles, wasting them where they are least visible and leaving the
silhouette coarse at the equator. An icosphere is near-uniform, so it looks
better at a lower triangle count — and at two subdivisions (320 triangles) it
is smooth enough for a shaded ball at these screen sizes while staying cheap
enough to instance thousands of times.

Pure numpy — no GL. Uploaded by gl_view, but generated and tested here.
"""

from __future__ import annotations

import numpy as np

# Two subdivisions: 20 -> 80 -> 320 triangles. Three would be 1,280, which
# buys nothing perceptible at the size a node occupies on screen.
DEFAULT_SUBDIVISIONS = 2


def _icosahedron() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """The 12 vertices and 20 faces of a unit icosahedron."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = np.array([
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ], dtype=np.float64)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return vertices, faces


def build_sphere(subdivisions: int = DEFAULT_SUBDIVISIONS) -> np.ndarray:
    """A unit sphere as a flat float32 array of triangle vertices.

    Returned as (n, 3) positions ready for a VBO. Because it is a unit
    sphere centred on the origin, each vertex is also its own normal — the
    shader needs no separate normal attribute, halving the mesh's memory and
    removing an attribute binding.
    """
    subdivisions = max(0, int(subdivisions))
    vertices, faces = _icosahedron()
    points = [v / np.linalg.norm(v) for v in vertices]
    # Cache keyed by the ordered pair, so an edge shared by two faces yields
    # ONE midpoint. Without it the mesh would split along every edge and the
    # vertex count would explode with each subdivision.
    midpoints: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        cached = midpoints.get(key)
        if cached is not None:
            return cached
        mid = points[a] + points[b]
        mid = mid / np.linalg.norm(mid)     # push back onto the unit sphere
        points.append(mid)
        index = len(points) - 1
        midpoints[key] = index
        return index

    for _ in range(subdivisions):
        subdivided: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            subdivided += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = subdivided

    flat = np.empty((len(faces) * 3, 3), dtype=np.float32)
    for i, (a, b, c) in enumerate(faces):
        flat[i * 3 + 0] = points[a]
        flat[i * 3 + 1] = points[b]
        flat[i * 3 + 2] = points[c]
    return flat


def triangle_count(subdivisions: int = DEFAULT_SUBDIVISIONS) -> int:
    return 20 * (4 ** max(0, int(subdivisions)))
