"""
O.R.I.O.N. synchronisation (Cloud Roadmap C2).

Desktop⇄cloud memory continuity built on an append-only change journal.

    journal.py     — SyncJournal: the append-only (node_id, lamport_ts, tier,
                     category, key, value_hash, payload) store, with a Lamport
                     clock and idempotent apply.
    conflict.py    — the tier-aware merge rule: last-writer-wins per key, except
                     the long_term/knowledge tiers, which merge additively.
    replicator.py  — reconcile journal entries into materialised state and push
                     it back into the memory matrix.

The HTTP segment exchange (`/v1/sync/journal`) and the KnowledgeGraphEngine seam
build on this deterministic core and are wired separately.
"""

from __future__ import annotations

from .conflict import ADDITIVE_TIERS, is_additive, resolve_key
from .journal import JournalEntry, SyncJournal
from .replicator import materialise_into, reconcile

__all__ = [
    "ADDITIVE_TIERS",
    "is_additive",
    "resolve_key",
    "JournalEntry",
    "SyncJournal",
    "materialise_into",
    "reconcile",
]
