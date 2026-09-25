"""
particles.py — the neural field (Mark XXIII).

The graph had ~85 discrete spheres, which reads as a node editor rather than
as the inside of a mind. This builds the dense layered field around it:

    Layer 1  subsystem nodes        the existing instanced spheres
    Layer 2  cluster halos          medium points around each cluster
    Layer 3  micro neural particles fine cloud through the whole field
    Layer 4  synapse sparks         fast, bright, short-lived-looking
    Layer 5  cosmic dust            distant, near-static background

Two decisions carry the whole design.

MOTION LIVES IN THE SHADER. Every particle is emitted ONCE with orbit
parameters — centre, radius, axis, speed, phase — and the vertex shader
derives its position from those plus a time uniform. Nothing is recomputed or
re-uploaded per frame. That is what makes several thousand moving particles
cost approximately nothing: no CPU work, no bus traffic, one draw call for the
entire field. Animating on the CPU would have meant rewriting a ~200 KB buffer
sixty times a second, which is precisely the "CPU spikes / no lag" failure the
brief rules out.

DENSITY IS DERIVED FROM REAL STATE. Particles are seeded FROM the real graph —
each subsystem, agent, MCP server and memory tier emits its own cloud, with
counts and colours taken from what that node actually is. A busier ORION is
visibly denser. Nothing here is decorative filler that would keep moving if
the system behind it went away, which is the difference between a living
readout and a screensaver.

Deterministic throughout (crc32 of node ids, never Python's randomised hash),
so the field is identical across restarts and reproducible in tests.

numpy only — no Qt, no GL.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from ..model import NodeKind, SwarmNode

# Per-particle floats, in the order the vertex shader reads them:
#   0..2   orbit centre xyz
#   3      orbit radius
#   4      orbit speed (radians/sec; sign sets direction)
#   5      orbit phase (radians)
#   6..8   orbit axis xyz (unit)
#   9..11  colour rgb
#   12     point size (px at unit distance)
#   13     twinkle rate (Hz; 0 = steady)
PARTICLE_FLOATS = 14

# Layer identifiers, ordered outward-in by visual weight.
LAYER_HALO = 1
LAYER_MICRO = 2
LAYER_SYNAPSE = 3
LAYER_COSMIC = 4

# How many particles each real node contributes per layer. Kept modest per
# node because the count multiplies by the whole graph — at ~85 real nodes
# these yield roughly 3,000 particles, and the field scales automatically as
# ORION grows rather than needing a hand-tuned total.
_HALO_PER_NODE = 16
_MICRO_PER_NODE = 44
_SYNAPSE_PER_NODE = 10
_COSMIC_COUNT = 1500

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _rng_for(seed_text: str) -> np.random.Generator:
    """A generator seeded deterministically from *seed_text*.

    crc32 rather than hash(): str hashing is randomised per process, so the
    field would be different on every launch and untestable."""
    return np.random.default_rng(zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF)


def _unit_axes(rng: np.random.Generator, count: int) -> np.ndarray:
    """*count* random unit vectors, used as per-particle orbit axes.

    Random axes rather than a shared one: a single axis makes every particle
    orbit in parallel planes, which reads as a mechanical carousel. Varied
    axes give the shell motion in every direction at once, which is what
    makes it look organic."""
    axes = rng.normal(size=(count, 3))
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return axes / norms


@dataclass(frozen=True, slots=True)
class ParticleField:
    """An emitted field, ready to upload. `layers` maps a layer id to the
    (start, count) span within `data`, so a caller can draw layers separately
    (different point sizes, blend modes) without slicing copies."""

    data: np.ndarray                      # (n, PARTICLE_FLOATS) float32
    layers: dict[int, tuple[int, int]]

    @property
    def count(self) -> int:
        return int(self.data.shape[0])

    def layer_view(self, layer: int) -> np.ndarray:
        start, count = self.layers.get(layer, (0, 0))
        return self.data[start:start + count]

    def stats(self) -> dict[str, int]:
        out = {"total": self.count}
        for layer, (_start, count) in sorted(self.layers.items()):
            out[f"layer_{layer}"] = count
        return out


def _emit(
    rng: np.random.Generator, count: int, centre: tuple[float, float, float],
    radius_lo: float, radius_hi: float, speed_lo: float, speed_hi: float,
    colour: tuple[float, float, float], size_lo: float, size_hi: float,
    twinkle_hi: float, colour_jitter: float = 0.10,
) -> np.ndarray:
    """One homogeneous burst of particles as a float32 block."""
    if count <= 0:
        return np.zeros((0, PARTICLE_FLOATS), dtype=np.float32)
    block = np.zeros((count, PARTICLE_FLOATS), dtype=np.float32)
    block[:, 0:3] = centre
    # cube-root so particles distribute evenly through the SHELL VOLUME
    # rather than bunching at the inner radius.
    t = rng.random(count) ** (1.0 / 3.0)
    block[:, 3] = radius_lo + (radius_hi - radius_lo) * t
    speeds = rng.uniform(speed_lo, speed_hi, count)
    # Half orbit each way; counter-rotation is what stops the field reading
    # as one rigid spinning object.
    speeds[rng.random(count) < 0.5] *= -1.0
    block[:, 4] = speeds
    block[:, 5] = rng.uniform(0.0, 2.0 * math.pi, count)
    block[:, 6:9] = _unit_axes(rng, count)
    tint = 1.0 + rng.uniform(-colour_jitter, colour_jitter, (count, 1))
    block[:, 9:12] = np.clip(np.array(colour, dtype=np.float32) * tint, 0.0, 1.0)
    block[:, 12] = rng.uniform(size_lo, size_hi, count)
    block[:, 13] = rng.uniform(0.0, twinkle_hi, count)
    return block


def build_field(
    nodes: list[SwarmNode],
    positions: dict[str, tuple[float, float, float]],
    colours: dict[str, tuple[float, float, float]] | None = None,
    density: float = 1.0,
    cosmic_radius: float = 150.0,
) -> ParticleField:
    """Emit the full layered field around a laid-out graph.

    *density* scales every layer at once — the single knob adaptive quality
    turns when a slower GPU cannot hold frame rate, so degrading is a matter
    of thinning the field rather than losing whole layers.
    """
    colours = colours or {}
    density = max(0.0, float(density))
    halo: list[np.ndarray] = []
    micro: list[np.ndarray] = []
    synapse: list[np.ndarray] = []

    for node in nodes:
        centre = positions.get(node.id)
        if centre is None or node.kind is NodeKind.CORE:
            # Core is where ORION's own face sits; leaving it clear is what
            # keeps him the origin rather than one more glowing ball.
            continue
        rng = _rng_for(node.id)
        base = colours.get(node.id, (0.55, 0.62, 0.95))
        cluster_node = node.is_cluster

        # Halos are tight and slow: they read as the node's own substance.
        halo.append(_emit(
            rng, int(_HALO_PER_NODE * (2.2 if cluster_node else 1.0) * density),
            centre, 1.6, 4.2 if cluster_node else 2.6, 0.05, 0.22,
            base, 2.0, 4.2, 0.35))

        # Micro particles spread wide and drift slowly — the connective haze
        # that turns discrete nodes into one continuous field.
        micro.append(_emit(
            rng, int(_MICRO_PER_NODE * (2.6 if cluster_node else 1.0) * density),
            centre, 2.5, 11.0 if cluster_node else 6.5, 0.03, 0.14,
            base, 1.0, 2.2, 0.8, colour_jitter=0.18))

        # Synapses are fast, small and bright: the visible flicker of thought.
        synapse.append(_emit(
            rng, int(_SYNAPSE_PER_NODE * density), centre,
            1.0, 5.0, 0.45, 1.5, base, 1.2, 2.4, 2.6, colour_jitter=0.05))

    # Deep background, centred on ORION so the whole scene reads as one body
    # rather than a graph floating in an unrelated starfield.
    cosmic = _emit(
        _rng_for("orion:cosmic"), int(_COSMIC_COUNT * density), (0.0, 0.0, 0.0),
        cosmic_radius * 0.42, cosmic_radius, 0.004, 0.02,
        (0.46, 0.48, 0.80), 0.9, 2.0, 0.5, colour_jitter=0.30)

    blocks: list[tuple[int, np.ndarray]] = []
    for layer, parts in ((LAYER_HALO, halo), (LAYER_MICRO, micro),
                         (LAYER_SYNAPSE, synapse)):
        joined = (np.concatenate(parts) if parts
                  else np.zeros((0, PARTICLE_FLOATS), dtype=np.float32))
        blocks.append((layer, joined))
    blocks.append((LAYER_COSMIC, cosmic))

    layers: dict[int, tuple[int, int]] = {}
    cursor = 0
    ordered: list[np.ndarray] = []
    for layer, block in blocks:
        layers[layer] = (cursor, int(block.shape[0]))
        cursor += int(block.shape[0])
        ordered.append(block)

    data = (np.ascontiguousarray(np.concatenate(ordered), dtype=np.float32)
            if ordered else np.zeros((0, PARTICLE_FLOATS), dtype=np.float32))
    return ParticleField(data=data, layers=layers)


def estimate_count(node_count: int, density: float = 1.0) -> int:
    """Field size for *node_count* real nodes — used by adaptive quality to
    choose a density before paying to emit anything."""
    per_node = _HALO_PER_NODE + _MICRO_PER_NODE + _SYNAPSE_PER_NODE
    return int((node_count * per_node + _COSMIC_COUNT) * max(0.0, density))
