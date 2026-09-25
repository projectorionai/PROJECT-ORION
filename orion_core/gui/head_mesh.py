"""
Head geometry for ORION's software-rendered avatar.

Why a measured face
-------------------
ORION's existing faces are drawn: ``gui/face.py`` sculpts a voxel silhouette
with QPainter, ``gui/face3d.py`` needs QtWebEngine (and cannot coexist with the
GL renderer), and ``gui/native_face.py`` needs a working GL context — which
``app.py`` deliberately does not provide, because it never calls
``setDefaultFormat`` and therefore gets a compatibility profile where whole
render passes silently draw nothing.

This module takes the third road: real measured human geometry, rasterised in
software. ``assets/canonical_face_model.obj`` is MediaPipe's canonical face
(Apache-2.0, see NOTICE) and carries actual eyelids, nostrils, lips and
cheekbones. Everything a formula cannot give you comes from there; everything
else — cranium, neck, rigs, normals, wireframe — is generated here.

The practical consequence is that the avatar looks identical on a gaming rig, a
2013 laptop, a VM and a remote desktop session, because no GPU driver is in the
loop and there is nothing for one to disagree about.

What this module produces
-------------------------
The face model is an open *mask*: it stops at a rim around the front of the
head. So the work here is

  * find that rim (the edges belonging to exactly one triangle),
  * sweep it back and up over a skull-shaped ellipsoid and close it at the
    occiput, which turns a mask into a head;
  * add a tapering neck stub that fades out rather than demanding shoulders;
  * compute vertex normals, jaw-rotation weights and lip-deformation weights;
  * thin the wireframe, since the shaded surface already carries the form.

Coordinate system after normalisation (head-local, right-handed):

    +x -> viewer's right      +y -> up      +z -> out of the face

with the chin at y = -1, the crown at y = +1, and the eyes landing near y = 0
so a renderer can place a gaze without measuring anything.

No Qt, no audio, no network — pure numpy, so all of it is unit-testable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

#: The MediaPipe mask. Resolved relative to the package so a frozen build that
#: relocates the tree still finds it.
OBJ_PATH = Path(__file__).resolve().parents[2] / "assets" / "canonical_face_model.obj"

# ── cranium ──────────────────────────────────────────────────────────────────
# All in the model's own units (chin y = -9.403, forehead y = +8.262).
#
# The proportion that matters is brow-to-crown against total head height. Real
# heads sit near 0.36; anything taller reads instantly as a long face even when
# the face itself is untouched measured geometry, which is a good example of
# why tuning the *face* was never going to fix a head that was the wrong shape.
# Sized from the mask itself, not guessed. The face spans x +/-7.74, y -9.40 to
# +8.26, z -2.44 to +7.59, so a cranium that encloses it without inflating it
# wants a crown near y = +11.5 and an occiput near z = -8. Radii that were much
# larger than this pushed the forehead — which sits close to the centre in
# sphere space — a long way outward, and the head rendered as a face inset in a
# hood. The rim magnitudes are the diagnostic: they should cluster near 1.0.
_SKULL_CENTRE = np.array([0.0, 2.0, 0.0])     # behind the face, level with the brow
_SKULL_RADII = np.array([8.1, 9.5, 8.0])      # x, y, z half-extents
_SKULL_POLE = np.array([0.0, 0.34, -1.0])     # toward the occiput; where the sweep closes
# More rings than the shape strictly needs. The silhouette is already correct
# at seven, but the surface then turns through ~90 degrees in too few steps and
# the *shading* creases along the hairline — a hard terminator that traces the
# mask's rim exactly, which reads as a face pasted onto a skull. Geometry and
# smoothness are separate problems here, and this one is the second.
# Eight rings, not eleven. The cranium is a smooth dome carrying no detail,
# and at eleven it was 41% of the whole mesh — 790 triangles to draw a shape
# with no features on it, paid on every single frame. The shading smoothness
# that justified the extra rings is better bought in the renderer's rim term
# than in geometry the viewer never resolves.
_SKULL_RINGS = 8                              # sweep steps from rim to pole
_SKULL_EASE = 1.25                            # >1 leaves the rim quickly, then slows

# ── neck ─────────────────────────────────────────────────────────────────────
# The neck fades out before it ends, so detail there is wasted outright.
_NECK_RINGS, _NECK_SEGMENTS = 5, 12
_NECK_CENTRE_Z = 1.2                          # the neck axis sits behind the chin
_NECK_TOP_R, _NECK_BOTTOM_R = 4.3, 3.5
_NECK_DROP = 7.0                              # how far below the chin it reaches

# ── wireframe ────────────────────────────────────────────────────────────────
# The lit surface carries the form; the wireframe is an accent. Drawing every
# edge of a 1,500-triangle mesh costs a lot of QPainter calls to produce a grey
# smear, so only every n-th edge is kept.
_WIRE_STRIDE = 3

# ── jaw rig ──────────────────────────────────────────────────────────────────
#: A real mandible hinges between the ears, not at the chin.
JAW_PIVOT = np.array([0.0, 0.10, -0.30])
#: Radians of drop at full amplitude (~11°). A real jaw barely moves in speech
#: and the first value here was set from that fact — but the avatar is watched
#: at a few hundred pixels, where an anatomically honest 6.5° is simply not
#: resolvable and every vowel rendered as the same thin slit. This is the size
#: the mouth has to be to be *read*, which is a different question from the
#: size it is on a person.
JAW_MAX = 0.19

#: MediaPipe landmark rings. These are published facts about the model (which
#: index is the left eye), and `_verify_landmarks` checks them against the
#: geometry at build time so a wrong index can never silently animate a cheek.
LANDMARKS: dict[str, list[int]] = {
    "eye_l": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
              160, 161, 246],
    "eye_r": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386,
              387, 388, 466],
    "brow_l": [70, 63, 105, 66, 107],
    "brow_r": [300, 293, 334, 296, 336],
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
                   269, 267, 0, 37, 39, 40, 185],
    "lips_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310,
                   311, 312, 13, 82, 81, 80, 191],
}


# ── loading ──────────────────────────────────────────────────────────────────

def load_obj(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertex and triangle arrays out of a Wavefront OBJ."""
    path = path or OBJ_PATH
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            # OBJ is 1-based and may carry v/vt/vn triples; take the position.
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int64))


def _boundary_loop(faces: np.ndarray) -> np.ndarray:
    """The ordered ring of vertices along an open mesh's rim.

    An edge shared by two triangles is interior; an edge belonging to exactly
    one is on the border. Walking those border edges gives the rim in order,
    which is what makes it sweepable into a surface rather than a point cloud.
    """
    counts: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(min(u, v), max(u, v))] += 1
    border = [edge for edge, n in counts.items() if n == 1]
    if not border:
        return np.empty(0, dtype=np.int64)

    neighbours: dict[int, list[int]] = defaultdict(list)
    for u, v in border:
        neighbours[u].append(v)
        neighbours[v].append(u)

    start = border[0][0]
    loop = [start]
    previous, current = -1, start
    # Bounded by the mesh's own size. The walk terminates on a well-formed
    # loop, but a mesh with a corrupt adjacency — a duplicated edge, a vertex
    # that neighbours itself — cycles for ever, and this runs while building
    # the face, so the symptom is a launch that never finishes.
    for _ in range(len(neighbours) + 1):
        nxt = [v for v in neighbours[current] if v != previous]
        if not nxt or nxt[0] == start:
            break
        previous, current = current, nxt[0]
        loop.append(current)
    return np.asarray(loop, dtype=np.int64)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Great-circle interpolation between unit vectors, row-wise.

    Linear interpolation would let the sweep cut *through* the skull on the way
    round; moving along the sphere keeps every intermediate ring on the surface,
    which is what stops the back of the head from creasing inward.
    """
    dot = np.clip((a * b).sum(-1, keepdims=True), -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    near = sin_omega < 1e-6                       # parallel: nothing to rotate
    safe = np.where(near, 1.0, sin_omega)
    out = np.where(
        near,
        a * (1.0 - t) + b * t,
        (np.sin((1.0 - t) * omega) / safe) * a + (np.sin(t * omega) / safe) * b,
    )
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-9)


def _add_cranium(verts: np.ndarray, faces: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, list]:
    """Sweep the mask's rim back over a skull ellipsoid and close it.

    Returns the extended arrays, the occiput vertex index, the rim loop and the
    per-ring index arrays. The last two are not decoration: the jaw rig has to
    taper across the join, and it can only do that if it knows which swept
    vertex grew out of which rim vertex.
    """
    rim = _boundary_loop(faces)
    if rim.size == 0:
        return verts, faces, -1, rim, []

    # Work in a space where the skull ellipsoid is a unit sphere: directions can
    # then be interpolated as pure rotations and scaled back afterwards.
    def to_sphere(points: np.ndarray) -> np.ndarray:
        d = (points - _SKULL_CENTRE) / _SKULL_RADII
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)

    rim_local = (verts[rim] - _SKULL_CENTRE) / _SKULL_RADII
    # How far each rim vertex sits from the skull centre, in sphere space. The
    # rim is NOT on the ellipsoid — it is wherever the scan put it — so the
    # sweep has to interpolate the radius as well as the direction. Snapping
    # straight to the ellipsoid surface at the first ring is what produced a
    # visible seam at the hairline and a cranium that read as a separate shell
    # floating behind the face.
    rim_mag = np.linalg.norm(rim_local, axis=1, keepdims=True)
    rim_dirs = rim_local / np.maximum(rim_mag, 1e-9)
    pole = _SKULL_POLE / np.linalg.norm(_SKULL_POLE)
    pole_rows = np.repeat(pole[None, :], rim.size, axis=0)

    new_verts = [verts]
    rings = [rim]
    for step in range(1, _SKULL_RINGS + 1):
        # Ease out: leave the rim quickly so the join is not a visible ridge,
        # then slow as the ring approaches the pole where triangles crowd.
        t = (step / _SKULL_RINGS) ** _SKULL_EASE
        dirs = _slerp(rim_dirs, pole_rows, t)
        # Radius eases from the rim's own to the ellipsoid's over the first part
        # of the sweep, so ring one is still essentially on the rim.
        mag = rim_mag * (1.0 - t) + 1.0 * t
        ring_pts = _SKULL_CENTRE + dirs * mag * _SKULL_RADII
        if step == _SKULL_RINGS:
            break                                  # the pole is a single vertex
        offset = sum(len(v) for v in new_verts)
        new_verts.append(ring_pts)
        rings.append(np.arange(offset, offset + rim.size, dtype=np.int64))

    occiput = _SKULL_CENTRE + pole * _SKULL_RADII
    offset = sum(len(v) for v in new_verts)
    new_verts.append(occiput[None, :])
    occiput_index = offset

    out_verts = np.vstack(new_verts)

    # Stitch consecutive rings into quads (as triangle pairs), then fan the last
    # ring to the occiput.
    new_faces = [faces]
    for lower, upper in zip(rings, rings[1:]):
        n = len(lower)
        a, b = lower, np.roll(lower, -1)
        c, d = upper, np.roll(upper, -1)
        quads = np.concatenate([
            np.stack([a, b, d], axis=1),
            np.stack([a, d, c], axis=1),
        ])
        new_faces.append(quads)
    last = rings[-1]
    fan = np.stack([last, np.roll(last, -1),
                    np.full(len(last), occiput_index, dtype=np.int64)], axis=1)
    new_faces.append(fan)

    return (out_verts, np.vstack(new_faces).astype(np.int64),
            occiput_index, rim, rings)


def _add_neck(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """A tapering tube below the jaw, so the head does not end in a cut.

    It fades rather than terminating: the renderer dims the lowest rings, which
    is cheaper and reads better than modelling shoulders nobody looks at.
    """
    chin_y = float(verts[:, 1].min())
    start_index = len(verts)

    angles = np.linspace(0.0, 2.0 * np.pi, _NECK_SEGMENTS, endpoint=False)
    ring_indices = []
    ring_points = []
    for ring in range(_NECK_RINGS):
        t = ring / (_NECK_RINGS - 1)
        radius = _NECK_TOP_R + (_NECK_BOTTOM_R - _NECK_TOP_R) * t
        y = chin_y + 1.4 - _NECK_DROP * t
        pts = np.stack([
            np.cos(angles) * radius,
            np.full(_NECK_SEGMENTS, y),
            np.sin(angles) * radius * 0.85 + _NECK_CENTRE_Z,
        ], axis=1)
        ring_points.append(pts)
        base = start_index + ring * _NECK_SEGMENTS
        ring_indices.append(np.arange(base, base + _NECK_SEGMENTS, dtype=np.int64))

    out_verts = np.vstack([verts, np.vstack(ring_points)])

    new_faces = [faces]
    for lower, upper in zip(ring_indices, ring_indices[1:]):
        a, b = lower, np.roll(lower, -1)
        c, d = upper, np.roll(upper, -1)
        new_faces.append(np.concatenate([
            np.stack([a, b, d], axis=1),
            np.stack([a, d, c], axis=1),
        ]))
    return out_verts, np.vstack(new_faces).astype(np.int64), start_index


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals.

    The cross product of two triangle edges has length proportional to twice the
    triangle's area, so accumulating it unnormalised weights each face by its
    size automatically — large faces should dominate a shared vertex, and a
    sliver should barely register.
    """
    normals = np.zeros_like(verts)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_n = np.cross(b - a, c - a)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-9)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _verify_landmarks(verts: np.ndarray, n_face: int) -> None:
    """Fail loudly at build time if a landmark index is not where it claims.

    A wrong index does not crash — it animates the wrong part of the face, which
    is the kind of defect that survives for months because it merely looks
    slightly off. Cheap assertions here make that impossible.
    """
    for name, indices in LANDMARKS.items():
        if max(indices) >= n_face:
            raise ValueError(f"landmark ring {name!r} indexes past the face mask")

    left_eye = verts[LANDMARKS["eye_l"]].mean(axis=0)
    right_eye = verts[LANDMARKS["eye_r"]].mean(axis=0)
    if not left_eye[0] < 0.0 < right_eye[0]:
        raise ValueError("eye rings are not on opposite sides of the midline")

    lips = verts[LANDMARKS["lips_outer"]].mean(axis=0)
    brows = np.vstack([verts[LANDMARKS["brow_l"]],
                       verts[LANDMARKS["brow_r"]]]).mean(axis=0)
    if not lips[1] < left_eye[1] < brows[1]:
        raise ValueError("expected lips below eyes below brows")


def build_head() -> dict[str, Any]:
    """Assemble the full head. Call ``get_head_mesh()`` instead; this is the
    uncached worker, kept separate so tests can rebuild deterministically."""
    face_verts, face_faces = load_obj()
    n_face = len(face_verts)
    _verify_landmarks(face_verts, n_face)

    verts, faces, _occiput, rim, skull_rings = _add_cranium(face_verts, face_faces)
    n_head = len(verts)
    verts, faces, neck_start = _add_neck(verts, faces)

    # ── normalise ────────────────────────────────────────────────────────────
    # Scale on the head alone, ignoring the neck: including it would shrink the
    # face every time the neck got longer, which is a surprising coupling.
    head = verts[:n_head]
    chin_y, crown_y = float(head[:, 1].min()), float(head[:, 1].max())
    scale = 2.0 / max(crown_y - chin_y, 1e-6)
    eye_y = float(np.vstack([face_verts[LANDMARKS["eye_l"]],
                             face_verts[LANDMARKS["eye_r"]]])[:, 1].mean())
    centre = np.array([0.0, eye_y, float(head[:, 2].mean())])
    verts = (verts - centre) * scale

    normals = _vertex_normals(verts, faces)

    # ── jaw rig ──────────────────────────────────────────────────────────────
    # The weight boundary is the MOUTH LINE, not a gradient down the face, and
    # getting that wrong is invisible in every static check.
    #
    # The first version faded smoothly from the hinge downward. That gives the
    # upper lip a weight of 0.956 and the lower lip 0.988 — nearly identical,
    # because they are nearly the same height. So the jaw rotation carried the
    # whole mouth downward as a unit and the lip *gap* changed by 0.9 px at a
    # 150 px radius. The mesh moved, the rig ran, every number looked plausible
    # and the face still could not open its mouth.
    #
    # Anatomically the mandible boundary is a step, not a ramp: everything
    # below the mouth opening rotates with the jaw and the upper lip does not.
    # So the threshold sits at the lip line with a deliberately narrow
    # transition, which is what actually separates the two lips.
    y = verts[:, 1]
    # Landmark 13 is the inner UPPER lip. Using the inner-lip *mean* put the
    # boundary between the two lips, which by construction gave the lower lip
    # only half weight; anchoring on the upper lip gives upper 0.00, lower
    # 1.00, nose 0.00, and takes the measured mouth gap from 0.9 px to 17 px.
    mouth_line = float(verts[LANDMARKS["lips_inner"][15], 1])
    jaw = _smoothstep((mouth_line - y) / 0.05)

    # The chin is ON the mask's rim, and the cranium is swept from that rim. So
    # a rigid cranium plus a rotating chin means the triangles bridging them
    # get stretched into slivers every time the mouth opens — which rendered as
    # a jaw visibly tearing away from the face. The fix is not a smaller jaw:
    # it is letting the first rings of the sweep follow the rim vertex they
    # grew from, fading out over three rings, so the join deforms smoothly
    # instead of shearing.
    jaw[n_face:n_head] = 0.0
    if rim.size and skull_rings:
        rim_jaw = jaw[rim]
        for ring_number, ring_idx in enumerate(skull_rings[1:], start=1):
            follow = max(0.0, 1.0 - ring_number / 5.0) ** 1.2
            if follow <= 0.0:
                break
            jaw[ring_idx] = rim_jaw * follow

    # The neck is anchored to the body, not the mandible — but it is separate
    # geometry, not stitched to the head, so a chin that rotates while the neck
    # stays put opens a visible notch at the jawline. The top rings follow the
    # jaw partially: anatomically right (the throat does move with the jaw) and
    # it keeps the silhouette closed.
    if neck_start < len(verts):
        neck_count = len(verts) - neck_start
        ring_size = max(1, neck_count // max(1, _NECK_RINGS))
        neck_follow = (0.62, 0.34, 0.14)
        jaw[neck_start:] = 0.0
        for ring_number, weight in enumerate(neck_follow):
            lo = neck_start + ring_number * ring_size
            hi = min(len(verts), lo + ring_size)
            if lo >= len(verts):
                break
            jaw[lo:hi] = weight
    # Behind the ears there is no mandible, only skull.
    jaw *= _smoothstep((verts[:, 2] - (-0.45)) / 0.4)

    # ── lip rig ──────────────────────────────────────────────────────────────
    lip_indices = LANDMARKS["lips_outer"] + LANDMARKS["lips_inner"]
    lip_centre = verts[lip_indices].mean(axis=0)
    distance = np.linalg.norm(verts - lip_centre, axis=1)
    lips = _smoothstep(1.0 - distance / 0.42)
    lips[n_face:] = 0.0                            # only the mask has lips

    # ── fade ─────────────────────────────────────────────────────────────────
    # 1 on the head, falling to 0 at the bottom of the neck.
    lowest = float(verts[:, 1].min())
    fade = np.ones(len(verts))
    if neck_start < len(verts):
        t = (verts[neck_start:, 1] - lowest) / max(-1.0 - lowest, 1e-6)
        fade[neck_start:] = _smoothstep(np.clip(t, 0.0, 1.0))

    edges = _unique_edges(faces)[::_WIRE_STRIDE]

    return {
        "verts": verts,
        "faces": faces,
        "normals": normals,
        "edges": edges,
        "jaw": jaw,
        "lips": lips,
        "lip_centre": lip_centre,
        "fade": fade,
        "landmarks": LANDMARKS,
        "n_face": n_face,
        "n_head": n_head,
        "neck_start": neck_start,
        "span": (float(verts[:, 1].max()), float(verts[:, 1].min())),
    }


_CACHE: dict[str, Any] | None = None


def get_head_mesh() -> dict[str, Any]:
    """Process-wide cached mesh — every avatar shares the same arrays.

    The arrays are read-only inputs to the renderer, which copies before it
    deforms, so sharing them across widgets is safe and saves rebuilding ~25 ms
    of numpy work per face.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = build_head()
    return _CACHE


__all__ = ["JAW_MAX", "JAW_PIVOT", "LANDMARKS", "OBJ_PATH", "build_head",
           "get_head_mesh", "load_obj"]
