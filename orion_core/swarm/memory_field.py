"""
memory_field.py — memory as constellations by recency (Mark XXIII).

Memory was a flat handful of nodes on the MEMORY cluster, which said nothing
about how ORION actually remembers. Here distance from him means RECENCY:

    working      what he is holding right now      closest, brightest
    short_term   the current conversation          near orbit
    project      the work in hand                  near orbit
    episodic     things that happened              mid distance
    semantic     what he knows to be true          outer
    procedural   how he does things                outer
    long_term    consolidated knowledge            distant constellations
    archived     cold storage                      deep space, barely lit

So the shape of the field IS the shape of his recall: what is close is what is
live, and searching illuminates constellations outward from him.

The tier list is ORION's real memory architecture (memory.py's seven tiers),
not an invented taxonomy — an unknown tier still gets placed rather than
dropped, so a new one appears the moment it exists.

Pure Python: no Qt, no GL, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

# Recency bands, innermost first. The value is a distance multiplier applied
# to the MEMORY ring's radius, so the constellation stretches inward and
# outward from that ring rather than replacing it.
_TIER_BANDS: dict[str, float] = {
    "working":     0.42,
    "short_term":  0.66,
    "project":     0.78,
    "episodic":    1.00,
    "semantic":    1.26,
    "procedural":  1.42,
    "long_term":   1.72,
    "archived":    2.15,
}

# Anything unrecognised sits with episodic: mid-field, clearly present, but
# not claiming to be either live or cold.
_DEFAULT_BAND = 1.00

# How brightly a tier reads at rest. Recall is what lights the far tiers, so
# their resting state is deliberately dim — otherwise "searching illuminates
# the constellation" has nothing to illuminate against.
_TIER_REST_GLOW: dict[str, float] = {
    "working":     1.00,
    "short_term":  0.82,
    "project":     0.78,
    "episodic":    0.58,
    "semantic":    0.46,
    "procedural":  0.42,
    "long_term":   0.30,
    "archived":    0.16,
}
_DEFAULT_GLOW = 0.5


@dataclass(frozen=True, slots=True)
class MemoryTier:
    """One tier's place in the constellation."""

    name: str
    rows: int
    band: float          # distance multiplier from ORION
    rest_glow: float     # emissive at rest, before any recall
    recency_rank: int    # 0 = most immediate

    @property
    def node_id(self) -> str:
        return f"memory:{self.name}"

    @property
    def is_live(self) -> bool:
        """Working and short-term memory — what he is actively holding."""
        return self.band <= 0.7

    @property
    def is_cold(self) -> bool:
        """Far enough out to read as archive rather than active recall."""
        return self.band >= 1.7


def band_for(tier: str) -> float:
    return _TIER_BANDS.get(str(tier or "").strip().lower(), _DEFAULT_BAND)


def rest_glow_for(tier: str) -> float:
    return _TIER_REST_GLOW.get(str(tier or "").strip().lower(), _DEFAULT_GLOW)


def _rank(tier: str) -> int:
    """Position in the recency ordering; unknown tiers sort mid-field."""
    ordered = sorted(_TIER_BANDS, key=lambda t: _TIER_BANDS[t])
    name = str(tier or "").strip().lower()
    return ordered.index(name) if name in ordered else len(ordered) // 2


def describe_tiers(snapshot: dict[str, int] | None) -> list[MemoryTier]:
    """Turn MemoryAgent.tiers_snapshot() into placed constellation tiers.

    Ordered innermost-first so callers can rely on the sequence being the
    recency ordering rather than dictionary order."""
    if not snapshot:
        return []
    tiers = [
        MemoryTier(
            name=str(name),
            rows=int(rows or 0),
            band=band_for(name),
            rest_glow=rest_glow_for(name),
            recency_rank=_rank(name),
        )
        for name, rows in snapshot.items()
    ]
    return sorted(tiers, key=lambda t: (t.band, t.name))


def recall_targets(tiers: list[MemoryTier], query: str = "") -> list[str]:
    """Which memory nodes a recall should illuminate, innermost first.

    A search propagates OUTWARD from what is closest to hand — that ordering
    is what makes recall look like ORION reaching back through what he knows
    rather than every tier flashing at once. Empty tiers are skipped: lighting
    a constellation that holds nothing would be a lie about where the answer
    came from.
    """
    return [t.node_id for t in tiers if t.rows > 0]


def detail_for(tier: MemoryTier) -> str:
    """The inspector line for a tier — states what it IS, not just a count."""
    if tier.is_live:
        held = "held now" if tier.band <= 0.5 else "current context"
        return f"{tier.rows:,} row(s) · {held}"
    if tier.is_cold:
        return f"{tier.rows:,} row(s) · cold storage"
    return f"{tier.rows:,} row(s) · recallable"
