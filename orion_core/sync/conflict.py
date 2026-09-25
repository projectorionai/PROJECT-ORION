"""
Tier-aware conflict resolution (Cloud Roadmap C2, step 3).

The rule, verbatim from the roadmap:

    Last-writer-wins per key, EXCEPT the ``long_term`` and ``knowledge`` tiers,
    which merge additively (both values kept, newest pointed-to). Episodes never
    conflict (append-only by design).

"Last writer" is decided by the Lamport timestamp, with the node id as a
deterministic tiebreaker so every node resolves an identical winner from the
same set of journal entries — the property that makes the whole scheme
convergent.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Tiers whose values accumulate rather than overwrite. Everything else is
# last-writer-wins. (Compared case-insensitively.)
ADDITIVE_TIERS = frozenset({"long_term", "knowledge"})


def is_additive(tier: str) -> bool:
    return str(tier or "").strip().lower() in ADDITIVE_TIERS


def _rank(entry) -> tuple[int, str]:
    """Deterministic ordering key: newer Lamport wins; node id breaks ties."""
    return (int(entry.lamport_ts), str(entry.node_id))


def resolve_key(entries: Sequence) -> list:
    """Resolve every journal entry for a *single* (tier, category, key) into the
    surviving payload(s), newest last.

    * additive tier → all distinct payloads (deduped by value_hash), ordered so
      the newest is last (the "pointed-to" current value);
    * otherwise      → a single-element list holding the last writer's payload.
    """
    entries = list(entries)
    if not entries:
        return []
    if is_additive(entries[0].tier):
        newest_by_hash: dict[str, object] = {}
        for entry in sorted(entries, key=_rank):
            newest_by_hash[entry.value_hash] = entry   # keep the latest per hash
        merged = sorted(newest_by_hash.values(), key=_rank)
        return [entry.payload for entry in merged]
    winner = max(entries, key=_rank)
    return [winner.payload]


def group_by_key(entries: Iterable) -> dict[tuple[str, str, str], list]:
    """Bucket journal entries by their (tier, category, key) identity."""
    buckets: dict[tuple[str, str, str], list] = {}
    for entry in entries:
        buckets.setdefault((entry.tier, entry.category, entry.key), []).append(entry)
    return buckets
