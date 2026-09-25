"""
face_geometry.py — ORION's face as native scene geometry (Mark XXIII).

The last thing standing outside the 3D scene was ORION himself: the avatar
was a QWebEngineView layered over the GL surface, which meant particles could
never pass in front of him, nothing could light him, and connections appeared
to radiate from an invisible point rather than from ORION.

That layering is not merely undesirable, it is impossible. Measured on real
hardware: once a QOpenGLWidget has existed in a window, the WebEngine
compositor's framebuffer comes up with zero-size attachments and every WebGL
draw fails with GL_INVALID_FRAMEBUFFER_OPERATION. Hiding the GL widget does
not help; neither does deleting it. So there is no arrangement in which the
Three.js avatar and the native renderer coexist, and ORION's real face has to
be built here.

WHY A POINT CLOUD. The face ORION has always had is a quantum voxel swarm —
thousands of points holding the shape of a head. That is exactly what
particles.py already instances by the thousand, so the avatar needs no mesh
renderer, no glTF loader and no second pipeline: it is another point cloud in
the same buffer format, drawn by the same instanced pass, inheriting depth
testing, occlusion and glow for free.

WHY NORMALS. The first version of this file deformed a sphere and shaded it
by camera distance alone, on the theory that "points nearer the viewer read
brighter" would supply form. It does not. Distance fade is a single global
gradient across the whole head, identical for the brow and the cheek beside
it, and — fatally — the back of the skull draws straight through the front.
The result was a ball of blue fog with two bright specks, not a face. Every
point therefore now carries a SURFACE NORMAL, computed from the sculpt
itself, so the head can be lit, silhouetted and back-culled like the solid
object it is meant to be.

WHY SCULPTED FEATURES. A tapered ellipsoid is a head-shaped ball. What makes
a cloud of points legible as a face is the nose above all — then the brow
ridge, the eye sockets it shades, the cheekbones and the chin. Those are
displacements of the surface itself, so the normals follow them and the
lighting reveals them without a single extra point.

Pure numpy — no Qt, no GL — so the whole facial structure is testable
headlessly, and animation (blink, speech, expression) is applied in the
shader from per-point FEATURE tags rather than by rewriting positions on the
CPU, keeping the zero-upload guarantee the rest of the renderer depends on.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

# Per-point floats, matching the layout the face shader reads:
#   0..2   rest position (model space, head centred on origin)
#   3..5   colour rgb
#   6      point size
#   7      feature tag (see FEATURE_*) — selects shader-side animation
#   8      per-point phase, so shimmer and drift never move in lockstep
#   9..11  surface normal — what makes the head a lit solid rather than fog
FACE_FLOATS = 12

FEATURE_SKIN = 0.0
FEATURE_EYE_L = 1.0
FEATURE_EYE_R = 2.0
FEATURE_MOUTH = 3.0
FEATURE_BROW = 4.0
FEATURE_HALO = 5.0      # the loose shell that breathes around the head

# Head proportions in world units. Taller than wide and narrower at the chin.
# Sized against the layout's CORE_KEEPOUT_RADIUS (15.0): large enough that
# ORION reads as the origin of the network rather than one more node in it,
# with the halo still clearing the keep-out so nothing intersects him.
HEAD_RADIUS = 10.5
HEAD_WIDTH = 0.80
HEAD_DEPTH = 0.88
CHIN_TAPER = 0.58

# ORION faces +Z. Anything framing him (the compact overlay orb, a focus
# camera) must approach from that side or it photographs his ear.
FACING = (0.0, 0.0, 1.0)

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _rng(seed: str) -> np.random.Generator:
    """Deterministic per-feature generator — crc32, never Python's randomised
    str hash, so ORION's face is identical on every launch."""
    return np.random.default_rng(zlib.crc32(seed.encode("utf-8")) & 0xFFFFFFFF)


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Near-uniform points on a unit sphere.

    Fibonacci rather than a lat/long grid: a grid bunches points at the poles,
    which on a head shows up as a dense cap on the crown and a bald patch at
    the temples."""
    if count <= 0:
        return np.zeros((0, 3))
    i = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * i / max(1.0, count - 1)
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = _GOLDEN_ANGLE * i
    return np.column_stack([np.cos(theta) * ring, y, np.sin(theta) * ring])


def _front_weight(z: np.ndarray) -> np.ndarray:
    """How much of the face-side sculpting applies here.

    Every facial feature fades out toward the ears; without this the nose
    ridge wraps round the back of the skull as a fin."""
    return np.clip((z - 0.05) / 0.55, 0.0, 1.0) ** 1.5


def _lobe(dx: np.ndarray, dy: np.ndarray, wx: float, wy: float) -> np.ndarray:
    """A smooth elliptical bump, 1 at the centre and 0 beyond its extent."""
    d = (dx / wx) ** 2 + (dy / wy) ** 2
    return np.exp(-d * 2.2)


# Feature placement, in head-relative units (x right, y up, z forward).
_EYE_Y = 0.14
_EYE_X = 0.31
_EYE_Z = 0.72
_EYE_R = 0.125
_MOUTH_Y = -0.40
_MOUTH_Z = 0.74
_MOUTH_W = 0.28
_MOUTH_H = 0.06
_BROW_Y = 0.30
_NOSE_TIP_Y = -0.06
_CHIN_Y = -0.68


def _sculpt(unit: np.ndarray) -> np.ndarray:
    """Deform a unit sphere into an actual head.

    Everything here is a displacement of the SURFACE, so normals derived from
    it pick the features up automatically and the lighting reveals them. That
    is what a plain tapered ellipsoid cannot do at any point density: it has
    no structure for light to find.
    """
    if unit.size == 0:
        return unit
    out = unit.astype(np.float64).copy()
    out[:, 0] *= HEAD_WIDTH
    out[:, 2] *= HEAD_DEPTH

    x, y, z = out[:, 0], out[:, 1], out[:, 2]

    # Jaw: narrows below the equator. Squared so the cheeks stay full and only
    # the lower face draws in.
    below = np.clip(-y, 0.0, 1.0)
    narrowing = 1.0 - CHIN_TAPER * below ** 2
    out[:, 0] *= narrowing
    out[:, 2] *= narrowing

    # Cranium: slightly flattened at the back, and taller at the crown, so the
    # silhouette reads as a skull rather than a ball from any angle.
    out[z < 0, 2] *= 0.84
    out[:, 1] *= 1.0 + 0.10 * np.clip(y, 0.0, 1.0) ** 2

    x, y, z = out[:, 0], out[:, 1], out[:, 2]
    face = _front_weight(z / max(HEAD_DEPTH, 1e-6))

    # ── the nose ────────────────────────────────────────────────────────────
    # The single most important feature. Without it a point cloud reads as a
    # mask; with it, as a head. Built as a ridge running from between the
    # brows down to the tip, widening into the nostrils.
    ridge = np.exp(-(x / 0.085) ** 2)
    along = np.clip((0.30 - y) / 0.42, 0.0, 1.0)
    ridge_profile = np.sin(np.clip(along, 0.0, 1.0) * math.pi * 0.92) ** 0.7
    tip = _lobe(x, y - _NOSE_TIP_Y, 0.11, 0.075)
    nose = (ridge * ridge_profile * 0.115 + tip * 0.085) * face
    out[:, 2] += nose

    # Nostril wings flare either side of the tip and sit slightly back.
    wing = _lobe(np.abs(x) - 0.105, y - (_NOSE_TIP_Y - 0.015), 0.055, 0.05)
    out[:, 2] += wing * 0.030 * face
    out[:, 0] += np.sign(x) * wing * 0.022 * face

    # ── brow ridge and eye sockets ──────────────────────────────────────────
    brow = _lobe(np.abs(x) - _EYE_X, y - _BROW_Y, 0.20, 0.075)
    out[:, 2] += brow * 0.052 * face
    out[:, 1] += brow * 0.012 * face

    # The socket is a recess. It is what gives the eyes somewhere to sit
    # instead of appearing painted on the front of a sphere.
    socket = _lobe(np.abs(x) - _EYE_X, y - _EYE_Y, 0.17, 0.10)
    out[:, 2] -= socket * 0.060 * face

    # ── cheekbones ──────────────────────────────────────────────────────────
    cheek = _lobe(np.abs(x) - 0.40, y - (_EYE_Y - 0.18), 0.20, 0.14)
    out[:, 2] += cheek * 0.040 * face
    out[:, 0] += np.sign(x) * cheek * 0.030 * face

    # A hollow beneath them, which is most of what makes a face look modelled
    # rather than inflated.
    hollow = _lobe(np.abs(x) - 0.36, y - (_MOUTH_Y + 0.04), 0.15, 0.10)
    out[:, 2] -= hollow * 0.030 * face

    # ── mouth ───────────────────────────────────────────────────────────────
    lips = _lobe(x, y - _MOUTH_Y, 0.30, 0.075)
    out[:, 2] += lips * 0.030 * face
    # The seam: a thin recess along the mouth line, so upper and lower lip are
    # distinguishable instead of one cushion.
    seam = np.exp(-((y - _MOUTH_Y) / 0.016) ** 2) * np.exp(-(x / 0.26) ** 2)
    out[:, 2] -= seam * 0.026 * face

    # ── chin and jawline ────────────────────────────────────────────────────
    chin = _lobe(x, y - _CHIN_Y, 0.20, 0.13)
    out[:, 2] += chin * 0.055 * face
    out[:, 1] -= chin * 0.020 * face
    # A crease above the chin separates it from the lower lip.
    crease = _lobe(x, y - (_MOUTH_Y - 0.11), 0.18, 0.035)
    out[:, 2] -= crease * 0.022 * face

    # ── temples ─────────────────────────────────────────────────────────────
    temple = _lobe(np.abs(x) - 0.70, y - 0.34, 0.18, 0.16)
    out[:, 0] -= np.sign(x) * temple * 0.035

    return out * HEAD_RADIUS


# Kept under its original name: several tests and the feature-patch builder
# call it, and it is still exactly "put a direction onto the head surface".
_head_shape = _sculpt


def _normals(unit: np.ndarray, epsilon: float = 4e-3) -> np.ndarray:
    """Surface normals for points generated from directions *unit*.

    Derived from the sculpt by finite differences along two tangents of the
    sphere rather than assumed radial. That distinction is the whole point: a
    radial normal describes the ellipsoid the features were carved out of, so
    the nose and the cheek beside it would light identically and none of the
    modelling would be visible.
    """
    if unit.size == 0:
        return np.zeros((0, 3))
    d = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    # A tangent basis that never degenerates: pick the reference axis the
    # direction is least aligned with.
    reference = np.where(np.abs(d[:, 1:2]) < 0.9,
                         np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    u = np.cross(d, reference)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(d, u)

    def at(offset: np.ndarray) -> np.ndarray:
        shifted = d + offset * epsilon
        shifted /= np.maximum(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-12)
        return _sculpt(shifted)

    base = _sculpt(d)
    normal = np.cross(at(u) - base, at(v) - base)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    # Where the sculpt is locally flat to within float precision, fall back to
    # the radial direction rather than emitting a zero normal that would light
    # as pure black.
    normal = np.where(length > 1e-9, normal / np.maximum(length, 1e-12), d)
    # The cross product's sign depends on the tangent basis; force outward.
    flipped = np.einsum("ij,ij->i", normal, d) < 0.0
    normal[flipped] *= -1.0
    return normal


# ── colourways ───────────────────────────────────────────────────────────────
#
# ORION had a crimson face before the native rebuild and the blue one arrived
# with it. Both are real options rather than one being a leftover, so the
# palette is data: ORION_FACE_COLOUR=crimson|blue picks it, and a new
# colourway is a dict entry, not a hunt through the module for tuples.
#
# No channel is 1.00 in any of these. A value that reaches exactly 1.0 came
# back as ~0 through this renderer's readback path, which turned the brightest
# lit points on the old skin (blue 1.00) into yellow-green speckles. The
# shader also rolls highlights off below 1.0, but there is no reason for the
# source data to sit on the edge of a known cliff.
_COLOURWAYS: dict[str, dict[float, tuple[float, float, float]]] = {
    "blue": {
        FEATURE_SKIN:   (0.60, 0.74, 0.99),
        FEATURE_EYE_L:  (0.96, 0.99, 0.99),
        FEATURE_EYE_R:  (0.96, 0.99, 0.99),
        FEATURE_MOUTH:  (0.52, 0.60, 0.86),
        FEATURE_BROW:   (0.42, 0.52, 0.84),
        FEATURE_HALO:   (0.40, 0.52, 0.92),
    },
    # Crimson. Deliberately not "blue with the channels swapped": a red at the
    # blue's luminance reads muddy, because the eye is far less sensitive to
    # red. The skin is lifted and warmed toward the highlights, and the halo
    # stays deep so his silhouette still reads against a black field.
    "crimson": {
        FEATURE_SKIN:   (0.99, 0.30, 0.30),
        FEATURE_EYE_L:  (0.99, 0.90, 0.88),
        FEATURE_EYE_R:  (0.99, 0.90, 0.88),
        FEATURE_MOUTH:  (0.82, 0.22, 0.26),
        FEATURE_BROW:   (0.72, 0.16, 0.20),
        FEATURE_HALO:   (0.78, 0.14, 0.22),
    },
}

DEFAULT_COLOURWAY = "crimson"

# Nothing this module emits may reach 1.0. See the colourway note above.
_MAX_CHANNEL = 0.995


def colourway_names() -> tuple[str, ...]:
    return tuple(sorted(_COLOURWAYS))


def _resolve_colourway(name: str | None) -> dict[float, tuple[float, float, float]]:
    """The requested palette, or the default. Never raises: an unknown name in
    an environment variable must not stop ORION having a face."""
    import os
    requested = str(name or os.getenv("ORION_FACE_COLOUR", "")
                    or DEFAULT_COLOURWAY).strip().lower()
    return _COLOURWAYS.get(requested, _COLOURWAYS[DEFAULT_COLOURWAY])



# Much smaller than the first version across the board. Point size is
# perspective-correct, so at portrait distance — the compact orb's framing —
# the original sizes rendered as 17-pixel balls and the head looked like a
# ball pit rather than a face. Sized to just touch at the skin's actual
# spacing (~0.19 world units at 30k points on a radius-10.5 head), which is
# what lets the shading read as a surface.
_FEATURE_SIZES = {
    # Skin points must OVERLAP, not merely touch: the fragment disc is soft at
    # the rim, so a point covers less than its own size and a just-touching
    # grid still shows background between the dots.
    FEATURE_SKIN: (0.74, 0.94),
    FEATURE_EYE_L: (0.62, 0.92),
    FEATURE_EYE_R: (0.62, 0.92),
    FEATURE_MOUTH: (0.66, 0.90),
    FEATURE_BROW: (0.66, 0.88),
    FEATURE_HALO: (0.30, 0.70),
}


def _classify(points: np.ndarray) -> np.ndarray:
    """Tag each skin point with the facial feature it belongs to.

    Tagging rather than generating features separately keeps the surface
    continuous — eyes and mouth are regions OF the face that animate, not
    objects floating in front of it."""
    if points.size == 0:
        return np.zeros(0)
    scaled = points / HEAD_RADIUS
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    tags = np.full(len(points), FEATURE_SKIN)

    front = z > 0.30
    for sign, tag in ((-1.0, FEATURE_EYE_L), (1.0, FEATURE_EYE_R)):
        near_eye = (np.hypot(x - sign * _EYE_X, y - _EYE_Y) < _EYE_R) & front
        tags[near_eye] = tag

    mouth = (np.abs(x) < _MOUTH_W) & (np.abs(y - _MOUTH_Y) < _MOUTH_H) & front
    tags[mouth] = FEATURE_MOUTH

    brow = ((np.abs(np.abs(x) - _EYE_X) < 0.17)
            & (np.abs(y - _BROW_Y) < 0.035) & front)
    tags[brow] = FEATURE_BROW
    return tags


def _cone_directions(rng: np.random.Generator, axis: np.ndarray,
                     angular_radius: float, count: int) -> np.ndarray:
    """*count* unit directions within *angular_radius* of *axis*.

    sqrt on the radius gives uniform density over the cap; sampling the angle
    linearly instead would crowd points at the centre and leave the rim
    sparse, which on an eye looks like a bright pupil and no iris."""
    if count <= 0:
        return np.zeros((0, 3))
    axis = axis / np.linalg.norm(axis)
    seed = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = rng.uniform(0.0, math.tau, count)
    r = np.sqrt(rng.random(count)) * angular_radius
    offset = (np.cos(theta) * r)[:, None] * u + (np.sin(theta) * r)[:, None] * v
    directions = axis[None, :] + offset
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _feature_patch(rng: np.random.Generator, centre: tuple[float, float, float],
                   angular_radius: float, count: int, tag: float,
                   jitter: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A dense patch of points sitting ON the head surface at *centre*.

    Features are generated in their OWN right rather than only tagged out of
    the general skin cloud. Tagging alone gave each eye ~25 points out of
    5,200 — the eye regions are a tiny fraction of a head's surface area, so
    at any sane skin density the face had no visible eyes at all.
    """
    axis = np.array(centre, dtype=np.float64)
    directions = _cone_directions(rng, axis, angular_radius, count)
    points = _sculpt(directions)
    normals = _normals(directions)
    if points.size:
        # Lift very slightly off the surface so features sit proud of the skin
        # instead of z-fighting with it.
        points *= 1.010
        points = points + rng.normal(scale=jitter * HEAD_RADIUS * 0.35,
                                     size=points.shape)
    return points, np.full(len(points), tag), normals


def build_face(skin_points: int = 30000, halo_points: int = 3400,
               jitter: float = 0.011, colourway: str | None = None) -> np.ndarray:
    """ORION's face as a (n, FACE_FLOATS) float32 cloud.

    *jitter* breaks the Fibonacci regularity so the surface reads as a quantum
    swarm holding a shape rather than a printed dot grid. It is far smaller
    than the first version's 0.045: at that amplitude the scatter was ±0.47
    world units, deep enough to swallow the nose (0.12) entirely, so the head
    could only ever be a fog ball no matter how it was lit.
    """
    rng = _rng("orion:face")
    unit = _fibonacci_sphere(max(0, int(skin_points)))
    skin = _sculpt(unit)
    skin_normals = _normals(unit)
    if skin.size:
        skin = skin + rng.normal(scale=jitter * HEAD_RADIUS, size=skin.shape)
    tags = _classify(skin)

    # A loose shell around the head: the visible edge of his presence, and
    # what particles from the wider network appear to mingle with.
    halo_unit = _fibonacci_sphere(max(0, int(halo_points)))
    halo = _sculpt(halo_unit)
    halo_normals = _normals(halo_unit)
    if halo.size:
        swell = 1.10 + rng.random((len(halo), 1)) * 0.34
        halo = halo * swell + rng.normal(scale=jitter * HEAD_RADIUS * 3.0,
                                         size=halo.shape)
    halo_tags = np.full(len(halo), FEATURE_HALO)

    # Dense feature patches, in addition to the tagged skin beneath them.
    # Counts are proportional to skin density so the face keeps its balance at
    # any resolution. The 0.2 floor keeps the eyes and mouth visible when
    # adaptive quality thins the face — losing the features entirely would
    # leave a faceless blob. But a caller asking for NO skin wants no face at
    # all, so the floor must not resurrect features from an empty request.
    scale = max(0.2, skin_points / 30000.0) if skin_points > 0 else 0.0
    patches = [
        _feature_patch(rng, (-_EYE_X, _EYE_Y, _EYE_Z), 0.15,
                       int(1600 * scale), FEATURE_EYE_L, jitter),
        _feature_patch(rng, (_EYE_X, _EYE_Y, _EYE_Z), 0.15,
                       int(1600 * scale), FEATURE_EYE_R, jitter),
        _feature_patch(rng, (0.0, _MOUTH_Y, _MOUTH_Z), 0.24,
                       int(1900 * scale), FEATURE_MOUTH, jitter),
        _feature_patch(rng, (-_EYE_X, _BROW_Y, _EYE_Z), 0.14,
                       int(760 * scale), FEATURE_BROW, jitter),
        _feature_patch(rng, (_EYE_X, _BROW_Y, _EYE_Z), 0.14,
                       int(760 * scale), FEATURE_BROW, jitter),
    ]

    blocks = [skin] + [p for p, _t, _n in patches]
    tag_blocks = [tags] + [t for _p, t, _n in patches]
    normal_blocks = [skin_normals] + [n for _p, _t, n in patches]
    if len(halo):
        blocks.append(halo)
        tag_blocks.append(halo_tags)
        normal_blocks.append(halo_normals)

    positions = np.concatenate(blocks)
    all_tags = np.concatenate(tag_blocks)
    normals = np.concatenate(normal_blocks)

    count = len(positions)
    data = np.zeros((count, FACE_FLOATS), dtype=np.float32)
    if not count:
        return data
    data[:, 0:3] = positions
    data[:, 7] = all_tags
    data[:, 8] = rng.uniform(0.0, math.tau, count)
    data[:, 9:12] = normals

    colours = _resolve_colourway(colourway)
    for tag, colour in colours.items():
        mask = all_tags == tag
        if not mask.any():
            continue
        tint = 1.0 + rng.uniform(-0.06, 0.06, (int(mask.sum()), 1))
        # Ceiling BELOW 1.0, not at it. The palettes deliberately stop at 0.99
        # to stay off the clipping cliff, but a +6% tint lands above 1.0 and a
        # plain clip pins it exactly ON the cliff — which is the value that
        # came back as ~0 and speckled his temple yellow-green.
        data[mask, 3:6] = np.clip(np.array(colour, dtype=np.float32) * tint,
                                  0.0, _MAX_CHANNEL)
        lo, hi = _FEATURE_SIZES[tag]
        data[mask, 6] = rng.uniform(lo, hi, int(mask.sum()))

    _shade_eyes(data, colours)
    return np.ascontiguousarray(data, dtype=np.float32)


# How the eye is coloured from its centre outward: a bright iris core, a
# cooler ring, then a dark lash edge that sits the eye INTO its socket.
def _eye_grade(colours: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(iris, sclera, lash) for this colourway.

    Derived from the palette rather than fixed, or a crimson ORION would keep
    two blue eyes. The iris is the palette's own eye colour, the sclera a
    dimmed skin, and the lash line a near-black of the same hue so the eye
    seats into its socket instead of being outlined in grey."""
    iris = np.array(colours[FEATURE_EYE_L], dtype=np.float64)
    sclera = np.array(colours[FEATURE_SKIN], dtype=np.float64) * 0.62
    lash = np.array(colours[FEATURE_HALO], dtype=np.float64) * 0.22
    return iris, sclera, lash


def _shade_eyes(data: np.ndarray, colours: dict) -> None:
    """Give each eye radial structure, in place.

    A uniformly white patch is a headlamp, not an eye — on the first real
    render the two eyes and the mouth read as three glowing blobs and drowned
    every bit of modelling around them. Grading from a bright iris out to a
    dark rim is what makes them sit in the sockets the sculpt already carved.
    """
    iris_colour, sclera_colour, lash_colour = _eye_grade(colours)
    for tag, sign in ((FEATURE_EYE_L, -1.0), (FEATURE_EYE_R, 1.0)):
        mask = data[:, 7] == tag
        if not mask.any():
            continue
        points = data[mask, 0:3].astype(np.float64)
        # _sculpt takes UNIT directions. Feeding it the raw feature coordinate
        # (length 0.80) put the eye's centre well inside the head, so every
        # point measured as far from it and the whole eye graded to the dark
        # lash colour — two black sockets, no iris at all.
        axis = np.array([[sign * _EYE_X, _EYE_Y, _EYE_Z]], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        centre = _sculpt(axis) * 1.010
        distance = np.linalg.norm(points - centre, axis=1) / HEAD_RADIUS
        # 0 at the pupil, 1 at the lash line.
        t = np.clip(distance / 0.155, 0.0, 1.0)
        iris = np.clip(1.0 - t / 0.42, 0.0, 1.0)[:, None]
        lash = np.clip((t - 0.66) / 0.34, 0.0, 1.0)[:, None]
        colour = sclera_colour * (1.0 - iris) + iris_colour * iris
        colour = colour * (1.0 - lash) + lash_colour * lash
        data[mask, 3:6] = np.clip(colour, 0.0, _MAX_CHANNEL)
        # The pupil is the largest point in the eye, the lash line the finest.
        data[mask, 6] *= (0.75 + 0.55 * iris[:, 0])


def feature_counts(face: np.ndarray) -> dict[str, int]:
    """How many points landed in each feature — used by tests and diagnostics
    to catch a deformation that silently wiped out the eyes or mouth."""
    names = {FEATURE_SKIN: "skin", FEATURE_EYE_L: "eye_left",
             FEATURE_EYE_R: "eye_right", FEATURE_MOUTH: "mouth",
             FEATURE_BROW: "brow", FEATURE_HALO: "halo"}
    if face.size == 0:
        return {name: 0 for name in names.values()}
    tags = face[:, 7]
    return {name: int((tags == tag).sum()) for tag, name in names.items()}


def bounds(face: np.ndarray) -> tuple[tuple[float, float, float],
                                      tuple[float, float, float]]:
    """Axis-aligned extent, for camera framing and the keep-out radius."""
    if face.size == 0:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    points = face[:, 0:3]
    return (tuple(float(v) for v in points.min(axis=0)),
            tuple(float(v) for v in points.max(axis=0)))


def profile_depth(y: float, x: float = 0.0) -> float:
    """How far the sculpted surface reaches forward at (x, y), head units.

    Exposed so the facial structure can be asserted numerically — a nose that
    silently flattens is invisible in every test that only counts points.
    """
    direction = np.array([[x, y, math.sqrt(max(1e-6, 1.0 - x * x - y * y))]])
    direction /= np.linalg.norm(direction)
    return float(_sculpt(direction)[0, 2] / HEAD_RADIUS)
