"""
orion_core.swarm — ORION's living nervous system (Mark XXII).

The Command Deck's SWARM page used to be a flat golden-angle ring of six node
kinds hanging off a single core node, rebuilt wholesale every two seconds and
pushed to a Chromium process as one JSON blob. This package replaces the model
underneath it — the renderer is swapped separately, in a later phase, so the
existing view keeps working throughout.

Layering is strictly downward; nothing here imports Qt, so the entire model is
testable without a GUI and reusable by a native renderer, the WebEngine view,
the remote/phone surface, or a headless diagnostic:

    model.py           typed nodes/edges/snapshots — no behaviour, no I/O
    clusters.py        the twelve real ORION zones a node can belong to
    telemetry_bind.py  reads the metrics ORION already collects onto nodes
    sources.py         composes a live snapshot from the real subsystems
    diff.py            what actually changed since the last snapshot

The diff layer is the point. Rendering cost should scale with what CHANGED,
not with how much exists — a node whose p95 latency ticked by 0.4 ms needs an
inspector refresh, not a GPU buffer upload, and a graph where nothing changed
needs neither.
"""

from __future__ import annotations

from .clusters import CLUSTER_ORDER, ClusterId, cluster_for_module, cluster_of_agent
from .diff import SwarmDelta, diff_snapshots
from .model import (
    Activity,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmEdge,
    SwarmNode,
    SwarmSnapshot,
)

__all__ = [
    "Activity",
    "CLUSTER_ORDER",
    "ClusterId",
    "Health",
    "NodeKind",
    "NodeTelemetry",
    "SwarmDelta",
    "SwarmEdge",
    "SwarmNode",
    "SwarmSnapshot",
    "cluster_for_module",
    "cluster_of_agent",
    "diff_snapshots",
]
