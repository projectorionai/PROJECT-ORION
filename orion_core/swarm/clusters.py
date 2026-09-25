"""
clusters.py — the twelve real ORION zones a swarm node can belong to.

These are NOT invented categories. They are exactly the twelve zones the
Command Deck already organises itself into (UnifiedDashboard.ZONE_ORDER), so
a cluster in the swarm and a zone in the deck mean the same thing, and a node
orbiting the "AUTOMATION" cluster is the same automation the AUTOMATION tab
opens. Inventing a parallel taxonomy here would have produced a prettier graph
that told the user nothing about their own system.

The module map below is grounded in the 36 subsystems app.py actually
registers into SystemRegistries.modules — not a guess at what ORION might
contain. An unrecognised module falls back to SYSTEM rather than being dropped
or crashing: a newly added subsystem must still appear in the graph, in a
defensible place, the moment it is registered and before anyone updates this
file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

# Mirrors UnifiedDashboard.ZONE_ORDER exactly. Kept as plain strings (not an
# Enum) because these values cross into JSON, the GL renderer's colour table
# and the deck's own zone names, and a str is the common currency of all three.
ClusterId = str

INTELLIGENCE: Final[ClusterId] = "INTELLIGENCE"
RESEARCH: Final[ClusterId] = "RESEARCH"
DEVELOPMENT: Final[ClusterId] = "DEVELOPMENT"
BUSINESS: Final[ClusterId] = "BUSINESS"
CREATIVE: Final[ClusterId] = "CREATIVE"
AUTOMATION: Final[ClusterId] = "AUTOMATION"
OPERATIONS: Final[ClusterId] = "OPERATIONS"
COMMUNICATION: Final[ClusterId] = "COMMUNICATION"
MEMORY: Final[ClusterId] = "MEMORY"
MONITORING: Final[ClusterId] = "MONITORING"
SECURITY: Final[ClusterId] = "SECURITY"
SYSTEM: Final[ClusterId] = "SYSTEM"

CLUSTER_ORDER: Final[tuple[ClusterId, ...]] = (
    INTELLIGENCE, RESEARCH, DEVELOPMENT, BUSINESS, CREATIVE, AUTOMATION,
    OPERATIONS, COMMUNICATION, MEMORY, MONITORING, SECURITY, SYSTEM,
)

# The cluster every unclassifiable node lands in. SYSTEM rather than a
# separate "OTHER" bucket, because an unmapped ORION subsystem genuinely IS
# system-level infrastructure until someone classifies it otherwise.
FALLBACK_CLUSTER: Final[ClusterId] = SYSTEM

# ── modules → cluster ────────────────────────────────────────────────────────
# Every key here is a real name from app.py's registries.modules.register loop.
_MODULE_CLUSTERS: Final[dict[str, ClusterId]] = {
    # thinking
    "reasoning": INTELLIGENCE,
    "strategy": INTELLIGENCE,
    "executive_core": INTELLIGENCE,
    "cognition": INTELLIGENCE,
    "missions": INTELLIGENCE,
    "graph": INTELLIGENCE,
    "perception": INTELLIGENCE,
    # knowledge work
    "research_director": RESEARCH,
    "evidence": RESEARCH,
    "briefing_engine": RESEARCH,
    # building
    "forge": DEVELOPMENT,
    "debugger_service": DEVELOPMENT,
    "skills": DEVELOPMENT,
    # commercial
    "creator_intel": BUSINESS,
    # embodiment / expression
    "avatar": CREATIVE,
    "face_tracker": CREATIVE,
    # doing things repeatably
    "automation": AUTOMATION,
    "workflow_engine": AUTOMATION,
    "pattern_detector": AUTOMATION,
    "control": AUTOMATION,
    # acting on the world
    "desktop": OPERATIONS,
    "companion": OPERATIONS,
    "proactive": OPERATIONS,
    # talking to people and services
    "outlook": COMMUNICATION,
    "notion": COMMUNICATION,
    "router": COMMUNICATION,
    # remembering
    "memory": MEMORY,
    # watching itself
    "telemetry": MONITORING,
    "navigation_trace": MONITORING,
    "registries": MONITORING,
    # defending
    "security_recon": SECURITY,
    "speaker_id": SECURITY,
    # the substrate
    "bus": SYSTEM,
    "dispatcher": SYSTEM,
    "vision": SYSTEM,
    "identity": SYSTEM,
    "agents": SYSTEM,
}

# ── specialist agents → cluster ──────────────────────────────────────────────
# The six built-ins, placed in the zone their workspace page already lives in
# (see app.py's _AGENT_WORKSPACES), so an agent's cluster and its deck page
# never disagree.
_AGENT_CLUSTERS: Final[dict[str, ClusterId]] = {
    "marketing": BUSINESS,
    "coding": DEVELOPMENT,
    "design": CREATIVE,
    "fashion": CREATIVE,
    "entertainment": CREATIVE,
    "research": RESEARCH,
}

# Keyword fallback for DECLARATIVE agents (config/agents/*.json), which can be
# added without touching any Python. Ordered most- to least-specific; first
# match wins. This is why a new manifest agent still lands somewhere sensible
# rather than defaulting to SYSTEM alongside the event bus.
_AGENT_KEYWORD_CLUSTERS: Final[tuple[tuple[tuple[str, ...], ClusterId], ...]] = (
    (("secur", "threat", "pentest", "forensic"), SECURITY),
    (("legal", "finance", "account", "sales", "market", "business"), BUSINESS),
    (("code", "coding", "engineer", "developer", "debug", "devops"), DEVELOPMENT),
    (("research", "analys", "evidence", "scholar"), RESEARCH),
    (("design", "art", "fashion", "music", "video", "creative", "style"), CREATIVE),
    (("memory", "recall", "archive"), MEMORY),
    (("monitor", "observ", "diagnos", "health"), MONITORING),
    (("automat", "workflow", "schedul", "pipeline"), AUTOMATION),
    (("email", "mail", "message", "chat", "comms"), COMMUNICATION),
    (("plan", "reason", "strateg", "decision"), INTELLIGENCE),
)


@lru_cache(maxsize=512)
def cluster_for_module(name: str) -> ClusterId:
    """The zone a registered subsystem belongs to.

    Never raises and never returns an empty cluster — an unknown module is
    SYSTEM, so newly registered subsystems appear immediately."""
    return _MODULE_CLUSTERS.get(str(name or "").strip().lower(), FALLBACK_CLUSTER)


@lru_cache(maxsize=2048)
def cluster_of_agent(name: str) -> ClusterId:
    """The zone a specialist agent belongs to.

    Built-ins are mapped explicitly; declarative agents are classified by
    keyword so a manifest-only agent still clusters meaningfully.

    Memoised because it is called once per agent per refresh and the keyword
    fallback is a full scan of the table above. Profiling the 1,000-agent
    target showed that scan costing 570,000 substring tests and ~45% of the
    entire compose — for a pure function over a small, stable set of names,
    which is exactly what a cache is for. Both maxsizes sit far above any
    plausible number of distinct names, so neither ever thrashes."""
    key = str(name or "").strip().lower()
    if not key:
        return FALLBACK_CLUSTER
    explicit = _AGENT_CLUSTERS.get(key)
    if explicit is not None:
        return explicit
    for needles, cluster in _AGENT_KEYWORD_CLUSTERS:
        if any(needle in key for needle in needles):
            return cluster
    return FALLBACK_CLUSTER


def is_known_cluster(cluster: str) -> bool:
    return str(cluster or "") in CLUSTER_ORDER


def normalise_cluster(cluster: str) -> ClusterId:
    """Coerce arbitrary input to a real cluster id, case-insensitively."""
    candidate = str(cluster or "").strip().upper()
    return candidate if candidate in CLUSTER_ORDER else FALLBACK_CLUSTER


def cluster_index(cluster: str) -> int:
    """Stable ordinal for a cluster — used by the layout solver to place
    clusters deterministically around core, so the arrangement a user learns
    is the arrangement they get back after a restart."""
    try:
        return CLUSTER_ORDER.index(normalise_cluster(cluster))
    except ValueError:  # pragma: no cover — normalise_cluster guarantees membership
        return len(CLUSTER_ORDER)
