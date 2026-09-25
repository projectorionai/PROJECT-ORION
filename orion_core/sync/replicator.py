"""
Reconciliation (Cloud Roadmap C2, step 4).

Turns a set of journal entries into materialised state by applying the
tier-aware conflict rule per key, and writes the winning values back into the
memory matrix. The push/pull segment exchange over ``/v1/sync/journal`` sits on
top of this deterministic core (and is wired with the RemoteGateway separately);
everything here is pure enough to unit-test without a network.
"""

from __future__ import annotations

from typing import Iterable

from .conflict import group_by_key, resolve_key


def reconcile(entries: Iterable) -> dict[tuple[str, str, str], list]:
    """Resolve every (tier, category, key) to its surviving payload(s),
    newest last. Deterministic: the same entry set always yields the same map."""
    return {
        identity: resolve_key(bucket)
        for identity, bucket in group_by_key(entries).items()
    }


def materialise_into(journal, matrix) -> int:
    """Apply a journal's reconciled state into the memory matrix.

    Writes suppress journalling (``journal=False``) so materialisation never
    loops back into new journal entries. For additive tiers the newest payload
    becomes the matrix's current value; the full accumulated history remains in
    the journal itself. Returns the number of keys written.
    """
    written = 0
    for (_tier, category, key), payloads in reconcile(journal.all_entries()).items():
        if not payloads:
            continue
        current = payloads[-1]   # resolve_key orders newest-last
        try:
            matrix.save(category, key, current, journal=False)
            written += 1
        except Exception:
            continue
    return written
