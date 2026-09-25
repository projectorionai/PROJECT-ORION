"""
layout.py — deterministic cluster-orbit positions (Mark XXII, Phase 1).

The brief asks for clusters that "naturally organise themselves", with physics
that feels "intentional, never chaotic, never overlapping, never unreadable".
That rules out a naive N-body force simulation, which is the usual reflex here
but produces exactly the failure modes named: it settles differently on every
run, it jitters forever without aggressive damping, and it happily parks two
nodes on top of each other.

So the layout is analytic, not iterative. Three nested spatial layers:

    core            origin
    cluster         fixed spherical bearing around core, by CLUSTER_ORDER index
                    per zone, so INTELLIGENCE is always in the same direction
    member          Fibonacci-sphere shell around its cluster centre
    child           small shell around its parent (MCP tools under a server)

Two properties matter more than beauty here:

DETERMINISM. Positions derive from crc32 of the node id, never from Python's
built-in hash() — str hashing is randomised per process by PYTHONHASHSEED, so
using it would silently reshuffle the entire graph on every restart and make
the layout untestable.

STABILITY. A node's angle depends only on its OWN id, never on how many
siblings it has or on its index in a list. Adding the 51st agent therefore
leaves the other fifty exactly where the user last saw them — the property an
index-based layout cannot offer, and the reason nodes stop "jumping between
refreshes".
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Iterable

from ..clusters import CLUSTER_ORDER, cluster_index, normalise_cluster
from ..model import NodeKind, SwarmNode

Vec3 = tuple[float, float, float]

# Orbit radii, in world units.
#
# Cluster centres sit on a SPHERE around core, not a ring. A ring was the
# first attempt and it read as a mess: twelve centres on a near-flat circle
# project onto each other from most viewing angles, so distinct zones
# overlapped on screen and the graph looked like scattered confetti rather
# than twelve groups. Twelve points spread over a sphere separate in all three
# axes, so orbiting actually reveals structure.
CORE_RADIUS = 0.0
CLUSTER_SHELL_RADIUS = 34.0
CHILD_SHELL_RADIUS = 2.1

# The centre is kept CLEAR. ORION's face is overlaid there — the network is
# meant to read as a brain with him at the middle of it, everything radiating
# out from him — so nothing but the (hidden) core node is allowed inside this
# radius. Members are pushed outward if their shell would intrude.
CORE_KEEPOUT_RADIUS = 15.0

# Member shells scale with population. A fixed radius gave a 3-node cluster
# the same footprint as a 22-node one (COMMUNICATION, once MCP tools are
# counted), leaving one sparse and the other crammed. Scaling by the square
# root keeps node DENSITY roughly constant, since surface area grows with r².
MEMBER_SHELL_MIN = 4.5
MEMBER_SHELL_MAX = 9.0
# 12 points on a sphere sit ~1.05*R apart, so at R=34 the nearest centres are
# ~35 units apart and even two MAX shells (18) leave a clear gap between them.
MEMBER_SHELL_PER_NODE = 1.55

# Kept as an alias: external callers and tests referred to the ring radius
# before the arrangement became spherical.
CLUSTER_RING_RADIUS = CLUSTER_SHELL_RADIUS

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))   # ≈2.39996 rad


def stable_hash(text: str) -> int:
    """A hash that is identical across processes and Python restarts.

    zlib.crc32 rather than hash(): the built-in is randomised per process
    unless PYTHONHASHSEED is pinned, which would make node placement differ
    on every launch and make these tests meaningless."""
    return zlib.crc32(str(text).encode("utf-8")) & 0xFFFFFFFF


def _fibonacci_point(index: int, total: int, radius: float) -> Vec3:
    """One point on a Fibonacci sphere — the standard construction for
    distributing N points over a sphere with near-uniform spacing and no
    clustering at the poles (which a naive lat/long grid suffers from)."""
    if total <= 1:
        return (0.0, 0.0, radius)
    # y walks linearly from +1 to -1; the ring radius follows from it.
    y = 1.0 - (2.0 * index) / max(1, total - 1)
    ring = math.sqrt(max(0.0, 1.0 - y * y))
    theta = _GOLDEN_ANGLE * index
    return (math.cos(theta) * ring * radius, y * radius,
            math.sin(theta) * ring * radius)


def _hashed_shell_point(node_id: str, radius: float) -> Vec3:
    """A point on a sphere derived purely from the node's own id.

    This is what makes the layout stable under insertion: the position never
    depends on sibling count or ordering, so new nodes slot into gaps instead
    of displacing everyone."""
    h = stable_hash(node_id)
    # Two independent fields out of one hash: low bits for the polar angle,
    # high bits for the azimuth.
    y = 1.0 - 2.0 * ((h & 0xFFFF) / 0xFFFF)
    ring = math.sqrt(max(0.0, 1.0 - y * y))
    theta = _GOLDEN_ANGLE * ((h >> 16) & 0xFFFF)
    return (math.cos(theta) * ring * radius, y * radius,
            math.sin(theta) * ring * radius)


def cluster_centre_at(cluster: str, elapsed: float) -> Vec3:
    """Where a zone sits at time *elapsed*, riding its cognition ring.

    The ring architecture replaces the fixed sphere: distance from ORION now
    means something (executive core closest, background infrastructure
    furthest) and each ring turns at its own rate. Positions are still
    DERIVED from parameters rather than stored, so this stays in agreement
    with the shader that draws them."""
    from .rings import orbit_for_cluster, orbital_position

    return orbital_position(orbit_for_cluster(cluster), elapsed)


def cluster_centre(cluster: str) -> Vec3:
    """Where a zone sits: a fixed point on a sphere around core.

    Placement is by CLUSTER_ORDER index via the same Fibonacci construction
    used for members, so a given zone is always in the same direction — the
    spatial memory that lets a user learn "security is over there" and have it
    stay true across restarts — while separating in all three axes instead of
    collapsing onto a single plane."""
    idx = cluster_index(normalise_cluster(cluster))
    return _fibonacci_point(idx, len(CLUSTER_ORDER), CLUSTER_SHELL_RADIUS)


def member_shell_radius(member_count: int) -> float:
    """Shell radius for a cluster holding *member_count* nodes.

    sqrt scaling keeps density constant: doubling the members grows the
    surface area, not the crowding.

    The count is first rounded up to a power-of-two BUCKET, which protects the
    stability guarantee this layout exists to provide. Scaling on the exact
    count meant adding a single node resized the shell and therefore moved
    every one of its siblings — precisely the "nodes jump between refreshes"
    behaviour the analytic layout was built to eliminate. Bucketing means a
    cluster re-spreads only when it roughly doubles, which is rare and is a
    real change worth showing.
    """
    if member_count <= 1:
        return MEMBER_SHELL_MIN
    bucket = 2 ** math.ceil(math.log2(member_count))
    radius = MEMBER_SHELL_PER_NODE * math.sqrt(bucket)
    return max(MEMBER_SHELL_MIN, min(MEMBER_SHELL_MAX, radius))


def _push_clear_of_core(position: Vec3) -> Vec3:
    """Move *position* outside the centre keep-out, along its own direction.

    Radially rather than by nudging, so a node keeps the bearing that ties it
    to its cluster — it moves further out, never sideways into a neighbour."""
    distance = math.sqrt(sum(v * v for v in position))
    if distance >= CORE_KEEPOUT_RADIUS:
        return position
    if distance < 1e-6:
        return (0.0, 0.0, CORE_KEEPOUT_RADIUS)
    scale = CORE_KEEPOUT_RADIUS / distance
    return (position[0] * scale, position[1] * scale, position[2] * scale)


@dataclass(frozen=True, slots=True)
class LayoutResult:
    """Positions keyed by node id, plus the radius each node should draw at."""

    positions: dict[str, Vec3]
    radii: dict[str, float]

    def position_of(self, node_id: str) -> Vec3:
        return self.positions.get(node_id, (0.0, 0.0, 0.0))


# Draw radius per kind. Core dominates; clusters are secondary anchors;
# everything else is a peer. Encoded here rather than in the shader so the
# hierarchy is reviewable in Python and testable without a GPU.
_KIND_RADIUS = {
    NodeKind.CORE: 2.6,
    NodeKind.SUBSYSTEM: 0.85,
    NodeKind.AGENT: 1.05,
    NodeKind.MCP_SERVER: 1.15,
    NodeKind.MCP_TOOL: 0.45,
    NodeKind.WORKFLOW: 0.75,
    NodeKind.MEMORY_TIER: 0.9,
    NodeKind.MODEL: 1.1,
    NodeKind.TOOL: 0.5,
    # Deck pages are destinations the user navigates to, so they read a little
    # larger than the machinery around them.
    NodeKind.PAGE: 1.25,
}
CLUSTER_NODE_RADIUS = 1.7


class LayoutSolver:
    """Turns a set of nodes into world positions.

    `elapsed` selects the moment on the cognition rings to solve for. It
    defaults to 0.0 so a caller that does not care about ring rotation (tests,
    the label layer's initial pass) gets the stable reference arrangement,
    while the renderer passes real time to follow the orbits.

    Stateless by design — `solve` is a pure function of its input. There is no
    incremental mode because the analytic layout makes one unnecessary: a
    node's position depends only on its own id and its parent's, so recomputing
    everything costs the same as recomputing one, and correctness is trivial.
    """

    def __init__(self, cluster_prefix: str = "cluster:") -> None:
        self.cluster_prefix = cluster_prefix

    def _is_cluster_node(self, node: SwarmNode) -> bool:
        return node.id.startswith(self.cluster_prefix)

    def solve(self, nodes: Iterable[SwarmNode],
              elapsed: float = 0.0) -> LayoutResult:
        """Place a snapshot in stable, learnable positions.

        ``elapsed`` remains accepted so older callers retain their contract,
        but topology does not orbit with wall-clock time.  Activity is shown
        through pulses and edge traffic rather than moving the user's map.
        """
        nodes = list(nodes)
        positions: dict[str, Vec3] = {}
        radii: dict[str, float] = {}

        # Pass 1 — core and the twelve cluster anchors. These must exist before
        # members can be placed relative to them.
        for node in nodes:
            if node.kind is NodeKind.CORE:
                positions[node.id] = (0.0, 0.0, CORE_RADIUS)
                radii[node.id] = _KIND_RADIUS[NodeKind.CORE]
            elif self._is_cluster_node(node):
                positions[node.id] = cluster_centre(node.cluster)
                radii[node.id] = CLUSTER_NODE_RADIUS

        # Pass 2 — direct members of a cluster, on a shell around its centre.
        # Ordered by id purely so the Fibonacci fallback is deterministic;
        # the hashed placement below does not depend on it.
        remaining = [n for n in nodes
                     if n.id not in positions and n.kind is not NodeKind.CORE]
        by_cluster: dict[str, list[SwarmNode]] = {}
        children: list[SwarmNode] = []
        for node in sorted(remaining, key=lambda n: n.id):
            # A node whose parent is another real node (an MCP tool under its
            # server) orbits that parent, not the cluster.
            if node.parent and not node.parent.startswith(self.cluster_prefix):
                children.append(node)
            else:
                by_cluster.setdefault(normalise_cluster(node.cluster), []).append(node)

        for cluster, members in by_cluster.items():
            centre = cluster_centre(cluster)
            shell = member_shell_radius(len(members))
            for node in members:
                offset = _hashed_shell_point(node.id, shell)
                point = (centre[0] + offset[0], centre[1] + offset[1],
                         centre[2] + offset[2])
                # Memory tiers ride OUT from ORION by recency rather than
                # sitting on their cluster shell like everything else: their
                # distance is the information. Scaling the whole position
                # (not just the offset) keeps them on the same bearing as
                # their cluster, so the constellation still reads as memory.
                if node.kind is NodeKind.MEMORY_TIER:
                    from ..memory_field import band_for

                    band = band_for(node.label)
                    point = (point[0] * band, point[1] * band, point[2] * band)
                positions[node.id] = _push_clear_of_core(point)
                radii[node.id] = _KIND_RADIUS.get(node.kind, 0.7)

        # Pass 3 — children of a placed node. Resolved after their parents so
        # an MCP tool always finds its server. A child whose parent is missing
        # falls back to its cluster shell rather than being dropped or piling
        # up at the origin.
        for node in children:
            parent_pos = positions.get(node.parent or "")
            if parent_pos is None:
                centre = cluster_centre_at(node.cluster, elapsed)
                offset = _hashed_shell_point(node.id, MEMBER_SHELL_MAX)
            else:
                centre = parent_pos
                offset = _hashed_shell_point(node.id, CHILD_SHELL_RADIUS)
            positions[node.id] = (centre[0] + offset[0], centre[1] + offset[1],
                                  centre[2] + offset[2])
            radii[node.id] = _KIND_RADIUS.get(node.kind, 0.5)

        return LayoutResult(positions=positions, radii=radii)
