"""
brain_geometry.py — ORION's mind at the centre of the swarm (Mark XXIII).

His FACE lives in the Core Window now, drawn by face3d. That freed the centre
of the swarm to be the thing the graph is actually about: not a portrait of
him, but his mind, with every subsystem radiating out of it.

SAME VERTEX FORMAT AS THE FACE. This emits (n, FACE_FLOATS) rows — position,
colour, size, feature tag, phase, normal — so it goes through the existing
face shader, buffer and draw call completely untouched. No second pipeline, no
new uniforms, no new attribute table. Swapping the model at the centre of the
scene is one buffer upload.

Every point is tagged FEATURE_SKIN or FEATURE_HALO on purpose. The face shader
animates eyes, mouth and brows off tags 1-4; a brain with those tags would
blink and mouth its way through speech, which is exactly the kind of detail
that looks like a bug rather than a feature.

WHAT MAKES IT READ AS A BRAIN. Not the silhouette — a brain-shaped blob is
just a blob. It is the GYRI: the interlocking ridges and the sulci between
them. Those are generated as a displacement of the surface, so the normals
follow them and the renderer's key light finds them, the same trick that made
the face's nose visible. Then three things fix the read as anatomy rather than
texture: the longitudinal fissure splitting the hemispheres, the cerebellum
tucked low and behind with its own much finer foliation, and the brain stem
descending from underneath.

Pure numpy — no Qt, no GL. The whole structure is testable headlessly.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from .face_geometry import (
    FACE_FLOATS,
    FEATURE_HALO,
    FEATURE_SKIN,
    _MAX_CHANNEL,
    _resolve_colourway,
)

# Sized to match the head it replaces, so the layout's CORE_KEEPOUT_RADIUS
# (15.0) still clears it and nothing has to move.
BRAIN_RADIUS = 10.5

# Proportions. A brain is longer front-to-back than it is wide, and wider than
# it is tall — the reverse of the head, which is what stops the two silhouettes
# reading as the same object.
BRAIN_LENGTH = 1.16      # z, front-back
BRAIN_WIDTH = 0.86       # x, lateral
BRAIN_HEIGHT = 0.80      # y, vertical

# How deep the sulci cut. The single most important number in this file: too
# shallow and the surface is a smooth egg, too deep and it breaks into
# unconnected worms.
FOLD_DEPTH = 0.082
# How deep the midline fissure cuts between the hemispheres.
FISSURE_DEPTH = 0.105

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _rng(seed: str) -> np.random.Generator:
    """crc32, never Python's randomised str hash, so his mind is identical on
    every launch."""
    return np.random.default_rng(zlib.crc32(seed.encode("utf-8")) & 0xFFFFFFFF)


def _fibonacci_sphere(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3))
    i = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * i / max(1.0, count - 1)
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = _GOLDEN_ANGLE * i
    return np.column_stack([np.cos(theta) * ring, y, np.sin(theta) * ring])


def _gyri(p: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """The cortical fold field.

    Three sine waves, each PHASE-MODULATED by a sine of a different axis. A
    plain sum of sines gives a regular egg-box; modulating the phase bends the
    ridges around each other, which is what produces the meandering,
    interlocking pattern a cortex actually has. Deterministic and continuous,
    so normals can be differentiated straight out of it.
    """
    x, y, z = p[:, 0] * scale, p[:, 1] * scale, p[:, 2] * scale
    a = np.sin(9.1 * x + 1.9 * np.sin(5.7 * z + 0.7))
    b = np.sin(8.3 * z + 1.7 * np.sin(6.1 * y + 1.3))
    c = np.sin(7.4 * y + 1.5 * np.sin(6.8 * x + 2.1))
    return (a + b + 0.85 * c) / 2.85


def _cerebrum(unit: np.ndarray) -> np.ndarray:
    """The main mass: two folded hemispheres with a fissure between them."""
    if unit.size == 0:
        return unit
    out = unit.astype(np.float64).copy()
    out[:, 0] *= BRAIN_WIDTH
    out[:, 1] *= BRAIN_HEIGHT
    out[:, 2] *= BRAIN_LENGTH

    x, y, z = out[:, 0], out[:, 1], out[:, 2]

    # The frontal lobe is rounder and the occipital narrower, so the shape has
    # a front and a back rather than being an ellipsoid from every angle.
    taper = 1.0 - 0.16 * np.clip(-z / BRAIN_LENGTH, 0.0, 1.0) ** 2
    out[:, 0] *= taper
    out[:, 1] *= taper

    # Underside flattened: a brain rests on its base, it is not a floating egg.
    below = np.clip(-y / BRAIN_HEIGHT, 0.0, 1.0)
    out[:, 1] *= 1.0 - 0.22 * below ** 2

    radial = out / np.maximum(
        np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    # ── gyri ────────────────────────────────────────────────────────────────
    # Suppressed underneath, where a real cortex is smoother and where the
    # cerebellum and stem attach.
    exposure = np.clip((y / BRAIN_HEIGHT + 0.55) / 1.2, 0.0, 1.0) ** 0.7
    folds = _gyri(np.column_stack([x, y, z])) * FOLD_DEPTH * (0.35 + 0.65 * exposure)
    out += radial * folds[:, None]

    # ── the longitudinal fissure ────────────────────────────────────────────
    # The deep midline groove. Without it the two hemispheres are one lump,
    # and nothing else in the shape says "brain" as immediately.
    depth = np.exp(-(x / 0.055) ** 2)
    top = np.clip(y / BRAIN_HEIGHT + 0.15, 0.0, 1.0)
    out -= radial * (depth * top * FISSURE_DEPTH)[:, None]

    # ── the lateral (Sylvian) fissure ───────────────────────────────────────
    # The other groove anyone would recognise: a diagonal cleft low on each
    # side, separating the temporal lobe.
    sylvian = np.exp(-((np.abs(y / BRAIN_HEIGHT + 0.18) - 0.06 * z) / 0.075) ** 2)
    sylvian *= np.clip(np.abs(x) / BRAIN_WIDTH, 0.0, 1.0) ** 1.5
    out -= radial * (sylvian * 0.045)[:, None]

    return out * BRAIN_RADIUS


def _cerebellum(unit: np.ndarray) -> np.ndarray:
    """The little brain, low and behind, with much finer horizontal foliation.

    Deliberately a different texture from the cortex rather than a smaller
    copy of it: the cerebellum's folia are far finer and run in parallel
    bands, and that contrast is what makes it read as a separate organ
    tucked under the cerebrum instead of a lump on it.
    """
    if unit.size == 0:
        return unit
    out = unit.astype(np.float64).copy()
    out[:, 0] *= 0.62
    out[:, 1] *= 0.34
    out[:, 2] *= 0.42

    radial = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    # Foliation: tight parallel bands, almost entirely a function of height.
    folia = np.sin(38.0 * out[:, 1] + 1.2 * np.sin(7.0 * out[:, 0])) * 0.016
    out += radial * folia[:, None]

    # A shallow midline notch, mirroring the cerebrum's fissure at its scale.
    out -= radial * (np.exp(-(out[:, 0] / 0.045) ** 2) * 0.020)[:, None]

    centre = np.array([0.0, -0.46, -0.60])
    return (out + centre) * BRAIN_RADIUS


def _normals_of(sculpt, unit: np.ndarray, epsilon: float = 3.5e-3) -> np.ndarray:
    """Surface normals by finite differences on the sphere.

    Differentiating the sculpt rather than assuming a radial normal is the
    whole reason the folds are visible: a radial normal describes the
    ellipsoid the gyri were carved out of, so ridge and sulcus would light
    identically and the cortex would render as a smooth egg.
    """
    if unit.size == 0:
        return np.zeros((0, 3))
    d = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    reference = np.where(np.abs(d[:, 1:2]) < 0.9,
                         np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    u = np.cross(d, reference)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(d, u)

    def at(offset: np.ndarray) -> np.ndarray:
        shifted = d + offset * epsilon
        shifted /= np.maximum(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-12)
        return sculpt(shifted)

    base = sculpt(d)
    normal = np.cross(at(u) - base, at(v) - base)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(length > 1e-9, normal / np.maximum(length, 1e-12), d)
    flipped = np.einsum("ij,ij->i", normal, d) < 0.0
    normal[flipped] *= -1.0
    return normal


def _stem(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The brain stem: a tapering tube descending from underneath.

    Built analytically rather than as another deformed sphere — it is a
    generalised cylinder, and its normals are exactly the radial direction
    perpendicular to its axis, so there is nothing to differentiate.
    """
    if count <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    t = rng.random(count)
    theta = rng.uniform(0.0, math.tau, count)
    # Axis: from under the cerebrum, curving down and slightly back.
    top = np.array([0.0, -0.38, -0.18])
    bottom = np.array([0.0, -1.02, -0.34])
    axis_point = top[None, :] + (bottom - top)[None, :] * t[:, None]
    # Narrows as it descends — the medulla is thinner than the midbrain.
    radius = 0.145 * (1.0 - 0.42 * t)
    offset = np.column_stack([np.cos(theta) * radius,
                              np.zeros(count),
                              np.sin(theta) * radius])
    points = (axis_point + offset) * BRAIN_RADIUS
    normals = np.column_stack([np.cos(theta), np.zeros(count), np.sin(theta)])
    return points, normals


# Point sizes, matching the face's grain so the two models read as the same
# substance at the same distance.
_SIZE_CORTEX = (0.58, 0.76)
_SIZE_CEREBELLUM = (0.48, 0.64)
_SIZE_STEM = (0.56, 0.74)
_SIZE_HALO = (0.30, 0.70)


def build_brain(cortex_points: int = 34000, cerebellum_points: int = 6000,
                stem_points: int = 1800, halo_points: int = 3000,
                jitter: float = 0.010,
                colourway: str | None = None) -> np.ndarray:
    """ORION's mind as a (n, FACE_FLOATS) float32 cloud.

    Same format as build_face, so the renderer swaps between them with one
    upload and no shader change at all.
    """
    rng = _rng("orion:brain")
    colours = _resolve_colourway(colourway)

    cortex_unit = _fibonacci_sphere(max(0, int(cortex_points)))
    cortex = _cerebrum(cortex_unit)
    cortex_n = _normals_of(_cerebrum, cortex_unit)

    cere_unit = _fibonacci_sphere(max(0, int(cerebellum_points)))
    cerebellum = _cerebellum(cere_unit)
    cerebellum_n = _normals_of(_cerebellum, cere_unit)

    stem, stem_n = _stem(rng, max(0, int(stem_points)))

    # The halo is the cerebrum's own shell, swollen — the visible edge of his
    # presence, and what the network's particles appear to mingle with.
    halo_unit = _fibonacci_sphere(max(0, int(halo_points)))
    halo = _cerebrum(halo_unit)
    halo_n = _normals_of(_cerebrum, halo_unit)
    if halo.size:
        swell = 1.10 + rng.random((len(halo), 1)) * 0.32
        halo = halo * swell

    blocks = [cortex, cerebellum, stem, halo]
    normals = [cortex_n, cerebellum_n, stem_n, halo_n]
    tags = [np.full(len(cortex), FEATURE_SKIN),
            np.full(len(cerebellum), FEATURE_SKIN),
            np.full(len(stem), FEATURE_SKIN),
            np.full(len(halo), FEATURE_HALO)]
    sizes = [_SIZE_CORTEX, _SIZE_CEREBELLUM, _SIZE_STEM, _SIZE_HALO]

    keep = [i for i, b in enumerate(blocks) if len(b)]
    if not keep:
        return np.zeros((0, FACE_FLOATS), dtype=np.float32)

    positions = np.concatenate([blocks[i] for i in keep])
    all_normals = np.concatenate([normals[i] for i in keep])
    all_tags = np.concatenate([tags[i] for i in keep])

    if jitter:
        # Far smaller than the fold depth it sits on. Enough to break the
        # Fibonacci regularity so the surface reads as a swarm holding a
        # shape; any larger and it fills the sulci back in.
        positions = positions + rng.normal(
            scale=jitter * BRAIN_RADIUS, size=positions.shape)

    count = len(positions)
    data = np.zeros((count, FACE_FLOATS), dtype=np.float32)
    data[:, 0:3] = positions
    data[:, 7] = all_tags
    data[:, 8] = rng.uniform(0.0, math.tau, count)
    data[:, 9:12] = all_normals

    cursor = 0
    for i in keep:
        n = len(blocks[i])
        lo, hi = sizes[i]
        data[cursor:cursor + n, 6] = rng.uniform(lo, hi, n)
        cursor += n

    for tag in (FEATURE_SKIN, FEATURE_HALO):
        mask = all_tags == tag
        if not mask.any():
            continue
        tint = 1.0 + rng.uniform(-0.06, 0.06, (int(mask.sum()), 1))
        colour = np.array(colours[tag], dtype=np.float32)
        data[mask, 3:6] = np.clip(colour * tint, 0.0, _MAX_CHANNEL)
    return np.ascontiguousarray(data, dtype=np.float32)


# ── inspection, so the anatomy can be asserted numerically ───────────────────

def region_counts(brain: np.ndarray) -> dict[str, int]:
    """Rough point census by region — catches a deformation that silently
    wiped out the cerebellum or the stem."""
    if brain.size == 0:
        return {"cortex": 0, "cerebellum": 0, "stem": 0, "halo": 0}
    p = brain[:, 0:3] / BRAIN_RADIUS
    halo = brain[:, 7] == FEATURE_HALO
    body = ~halo
    stem = body & (p[:, 1] < -0.42) & (np.abs(p[:, 0]) < 0.22) & (p[:, 2] > -0.45)
    cerebellum = body & ~stem & (p[:, 1] < -0.28) & (p[:, 2] < -0.30)
    return {
        "cortex": int((body & ~stem & ~cerebellum).sum()),
        "cerebellum": int(cerebellum.sum()),
        "stem": int(stem.sum()),
        "halo": int(halo.sum()),
    }


def surface_radius(x: float, y: float, z: float) -> float:
    """How far the cerebrum's surface reaches along a direction, in brain
    units. Exposed so the fissure and the folds can be measured rather than
    eyeballed — a fissure that silently flattens is invisible to any test that
    only counts points."""
    d = np.array([[x, y, z]], dtype=np.float64)
    d /= np.linalg.norm(d)
    return float(np.linalg.norm(_cerebrum(d)[0]) / BRAIN_RADIUS)


def fissure_relief(samples: int = 20000) -> float:
    """How much lower the midline sits than the crowns either side of it.

    Averaged over a band rather than sampled at two points: the gyri are deep
    enough that a single sample beside the midline can land in a sulcus and
    report no fissure at all. What matters is that the midline is
    systematically lower, which is what the eye reads as a cleft.
    """
    unit = _fibonacci_sphere(samples)
    top = unit[unit[:, 1] > 0.35]
    if not len(top):
        return 0.0
    radii = np.linalg.norm(_cerebrum(top), axis=1) / BRAIN_RADIUS
    x = np.abs(top[:, 0])
    midline = radii[x < 0.06]
    flank = radii[(x > 0.18) & (x < 0.45)]
    if not len(midline) or not len(flank):
        return 0.0
    return float(flank.mean() - midline.mean())


def fold_relief(samples: int = 4000) -> float:
    """Peak-to-trough of the fold field over the exposed cortex."""
    unit = _fibonacci_sphere(samples)
    upper = unit[unit[:, 1] > 0.2]
    radii = np.linalg.norm(_cerebrum(upper), axis=1) / BRAIN_RADIUS
    return float(np.percentile(radii, 97) - np.percentile(radii, 3))


def bounds(brain: np.ndarray) -> tuple[tuple[float, float, float],
                                       tuple[float, float, float]]:
    if brain.size == 0:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    p = brain[:, 0:3]
    return (tuple(float(v) for v in p.min(axis=0)),
            tuple(float(v) for v in p.max(axis=0)))
