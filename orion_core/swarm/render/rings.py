"""
rings.py — orbital cognition rings (Mark XXIII).

Clusters previously sat at fixed points on a sphere. That separated them
cleanly but said nothing: every zone was the same kind of thing at the same
distance, so the arrangement carried no meaning beyond "not overlapping".

Here the graph is organised into nested ORBITS around ORION, each ring a
layer of cognition, so distance from him means something:

    Ring 0  Executive Core         his own reasoning, closest, slowest
    Ring 1  Active Agents          the specialists currently doing work
    Ring 2  Memory                 what he knows, further out
    Ring 3  Connections            MCP servers — the edge of his reach
    Ring 4  Background             monitoring, security, infrastructure

Each ring has its own radius, tilt axis, angular speed and wobble, so they
drift out of phase with one another and the field never reads as one rigid
object rotating.

THE CRITICAL DESIGN CONSTRAINT: rings rotate in the SHADER. This module emits
orbit PARAMETERS — ring radius, axis, angular speed, phase — and the vertex
shader derives a node's position from those plus a time uniform, exactly as
particles.py already does. Rotating rings by rewriting node positions on the
CPU would mean re-uploading the instance buffer every frame, which is the
per-frame churn the whole renderer was built to avoid. Anything that needs a
node's CURRENT position on the CPU (picking, edge endpoints, labels) calls
`orbital_position` with the same time value and gets the identical answer.

Pure maths — numpy only, no Qt, no GL.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

from ..clusters import (
    AUTOMATION,
    BUSINESS,
    COMMUNICATION,
    CREATIVE,
    DEVELOPMENT,
    INTELLIGENCE,
    MEMORY,
    MONITORING,
    OPERATIONS,
    RESEARCH,
    SECURITY,
    SYSTEM,
)

Vec3 = tuple[float, float, float]

RING_EXECUTIVE = 0
RING_AGENTS = 1
RING_MEMORY = 2
RING_CONNECTIONS = 3
RING_BACKGROUND = 4


@dataclass(frozen=True, slots=True)
class Ring:
    """One orbital cognition layer."""

    index: int
    name: str
    radius: float
    # Unit normal of the orbit plane. Distinct per ring so the orbits visibly
    # cross rather than sitting as concentric circles on one plane.
    axis: Vec3
    speed: float          # radians/sec; sign sets direction
    wobble: float         # amplitude of the out-of-plane sway, world units
    wobble_rate: float    # radians/sec of that sway


def _norm(v: Vec3) -> Vec3:
    length = math.sqrt(sum(c * c for c in v))
    if length < 1e-9:
        return (0.0, 1.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


# Speeds are deliberately small and mutually non-harmonic: no two rings share
# a period, so the arrangement never snaps back into visible alignment.
# Alternating signs make adjacent rings counter-rotate, which is what makes
# the nesting legible instead of looking like one spinning body.
RINGS: tuple[Ring, ...] = (
    Ring(RING_EXECUTIVE, "Executive Core", 16.0,
         _norm((0.0, 1.0, 0.08)), 0.042, 0.5, 0.11),
    Ring(RING_AGENTS, "Active Agents", 30.0,
         _norm((0.16, 1.0, -0.10)), -0.029, 1.0, 0.08),
    Ring(RING_MEMORY, "Memory", 44.0,
         _norm((-0.22, 1.0, 0.14)), 0.019, 1.6, 0.06),
    Ring(RING_CONNECTIONS, "Connections", 58.0,
         _norm((0.10, 1.0, 0.28)), -0.013, 2.1, 0.05),
    Ring(RING_BACKGROUND, "Background", 74.0,
         _norm((-0.12, 1.0, -0.22)), 0.008, 2.8, 0.035),
)

_BY_INDEX = {ring.index: ring for ring in RINGS}

# Which cognition layer each zone belongs to. Grouped by what the zone IS to
# ORION rather than alphabetically: the specialists that actively do work
# share a ring, the things he knows share another, the outside world sits at
# the edge of his reach.
_CLUSTER_RING: dict[str, int] = {
    INTELLIGENCE:  RING_EXECUTIVE,
    RESEARCH:      RING_AGENTS,
    DEVELOPMENT:   RING_AGENTS,
    BUSINESS:      RING_AGENTS,
    CREATIVE:      RING_AGENTS,
    MEMORY:        RING_MEMORY,
    COMMUNICATION: RING_CONNECTIONS,
    AUTOMATION:    RING_BACKGROUND,
    OPERATIONS:    RING_BACKGROUND,
    MONITORING:    RING_BACKGROUND,
    SECURITY:      RING_BACKGROUND,
    SYSTEM:        RING_BACKGROUND,
}


def ring_for_cluster(cluster: str) -> Ring:
    """The cognition ring a zone orbits on. Unknown zones fall to Background
    rather than being dropped — a new zone must still appear somewhere."""
    return _BY_INDEX[_CLUSTER_RING.get(str(cluster or "").upper(), RING_BACKGROUND)]


def _stable_angle(key: str) -> float:
    """A fixed starting angle for *key*, identical across restarts.

    crc32, never Python's randomised str hash — otherwise every launch would
    deal the clusters into different slots around their ring."""
    return ((zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF) * math.tau


@dataclass(frozen=True, slots=True)
class Orbit:
    """A node's orbit, as parameters rather than a position.

    This is what gets written into the instance buffer and what the vertex
    shader turns into a position each frame."""

    centre: Vec3          # what it orbits (origin for clusters, parent otherwise)
    radius: float
    axis: Vec3
    speed: float
    phase: float
    wobble: float
    wobble_rate: float

    def as_tuple(self) -> tuple[float, ...]:
        return (*self.centre, self.radius, *self.axis, self.speed, self.phase,
                self.wobble, self.wobble_rate)


def orbit_for_cluster(cluster: str) -> Orbit:
    """The orbit a cluster anchor rides."""
    ring = ring_for_cluster(cluster)
    return Orbit(centre=(0.0, 0.0, 0.0), radius=ring.radius, axis=ring.axis,
                 speed=ring.speed, phase=_stable_angle(cluster),
                 wobble=ring.wobble, wobble_rate=ring.wobble_rate)


def _rotate_about(v: Vec3, axis: Vec3, angle: float) -> Vec3:
    """Rodrigues rotation — the exact operation the vertex shader performs, so
    CPU and GPU agree on where a node is at a given time."""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = axis
    vx, vy, vz = v
    dot = ax * vx + ay * vy + az * vz
    cross = (ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx)
    return (v[0] * c + cross[0] * s + ax * dot * (1.0 - c),
            v[1] * c + cross[1] * s + ay * dot * (1.0 - c),
            v[2] * c + cross[2] * s + az * dot * (1.0 - c))


def _orbit_basis(axis: Vec3) -> Vec3:
    """A unit vector in the orbit plane, used as the zero-angle direction."""
    seed = (0.0, 1.0, 0.0) if abs(axis[1]) < 0.95 else (1.0, 0.0, 0.0)
    cross = (axis[1] * seed[2] - axis[2] * seed[1],
             axis[2] * seed[0] - axis[0] * seed[2],
             axis[0] * seed[1] - axis[1] * seed[0])
    return _norm(cross)


def orbital_position(orbit: Orbit, elapsed: float) -> Vec3:
    """Where *orbit* is at time *elapsed*.

    Must stay in exact agreement with the vertex shader's version: picking,
    edge endpoints and label anchors all use this, and any divergence would
    show up as clicks landing next to nodes rather than on them.
    """
    basis = _orbit_basis(orbit.axis)
    radial = tuple(c * orbit.radius for c in basis)
    angle = elapsed * orbit.speed + orbit.phase
    x, y, z = _rotate_about(radial, orbit.axis, angle)
    if orbit.wobble:
        sway = math.sin(elapsed * orbit.wobble_rate + orbit.phase) * orbit.wobble
        x += orbit.axis[0] * sway
        y += orbit.axis[1] * sway
        z += orbit.axis[2] * sway
    return (orbit.centre[0] + x, orbit.centre[1] + y, orbit.centre[2] + z)


def ring_of(index: int) -> Ring:
    return _BY_INDEX.get(int(index), RINGS[RING_BACKGROUND])


def describe() -> list[dict[str, object]]:
    """The ring table, for diagnostics and the label layer."""
    return [{"index": r.index, "name": r.name, "radius": r.radius,
             "speed": r.speed, "period_s": round(abs(math.tau / r.speed), 1)}
            for r in RINGS]
