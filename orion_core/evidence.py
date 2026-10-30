"""
EvidenceEngine (Phase 3) — every claim ORION learns carries provenance.

The self-expanding knowledge system's audit trail: a durable SQLite store of
claims, each with its source, when it was learned, and a confidence score.
On top of the store sit the two analytical passes the Phase 3 charter asks
for:

    corroboration  — claims that agree (high token overlap, same polarity)
                     raise each other's confidence;
    contradiction  — claims that collide (high overlap, opposite polarity,
                     or clashing figures) are surfaced rather than silently
                     coexisting.

This complements the KnowledgeGraphEngine (entities and events) — the graph
answers *what relates to what*; the evidence store answers *why we believe
it and how much*. Thread-safe; all writes serialised behind one lock.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .constants import CONFIG_DIR
from .utils import utc_stamp
from .db import apply_pragmas

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in into is it its of on or "
    "s t that the their there these this to was were which will with".split()
)
_NEGATIONS = frozenset("not no never cannot without isn don doesn won neither nor".split())
_HEDGES = ("may ", "might ", "could ", "possibly", "suggests", "unclear",
           "reportedly", "rumoured", "allegedly", "perhaps")
_STRONG = ("study", "data", "measured", "according to", "survey", "trial",
           "confirmed", "official", "%")

_OVERLAP_THRESHOLD = 0.55


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS and w not in _NEGATIONS}


def _polarity(text: str) -> bool:
    """True when the claim is negated ('X is not Y', 'no evidence that…')."""
    words = set(re.findall(r"[a-z]+", str(text or "").lower()))
    return bool(words & _NEGATIONS)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", str(text or "")))


def _source_identity(source: str, url: str) -> set[tuple[str, str]]:
    """Known provenance keys; a renamed note cannot hide the same publisher."""
    keys: set[tuple[str, str]] = set()
    label = " ".join(str(source or "").casefold().split())
    if label and label != "unknown":
        keys.add(("source", label))
    try:
        parsed = urlsplit(str(url or ""))
        if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname:
            host = parsed.hostname.casefold().rstrip(".").removeprefix("www.")
            keys.add(("publisher", host))
    except ValueError:
        pass  # Malformed provenance is not evidence of independence.
    return keys


def _contributor_count(own: set[tuple[str, str]],
                       peers: list[set[tuple[str, str]]]) -> int:
    """Count source groups, joining aliases transitively and excluding our own."""
    groups = [set(own)]
    for keys in peers:
        joined = set(keys)
        matching = [group for group in groups if group & joined]
        for group in matching:
            joined.update(group)
            groups.remove(group)
        groups.append(joined)
    return len(groups) - 1


def score_confidence(claim: str, base: float = 0.6) -> float:
    """Deterministic prior from the wording: hedges lower it, evidence raises it."""
    lowered = str(claim or "").lower()
    score = base
    score -= 0.15 * sum(1 for h in _HEDGES if h in lowered)
    score += 0.10 * sum(1 for s in _STRONG if s in lowered)
    return round(max(0.05, min(0.95, score)), 2)


class EvidenceEngine:
    """Durable claim store with provenance, corroboration and contradiction."""

    DB_PATH = CONFIG_DIR / "evidence_graph.db"

    def __init__(self, bus: Any = None, path: Path | None = None,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.path = Path(path) if path is not None else self.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS claims (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic      TEXT NOT NULL,
                    claim      TEXT NOT NULL,
                    source     TEXT NOT NULL,
                    url        TEXT DEFAULT '',
                    confidence REAL NOT NULL,
                    learned_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_claims_topic ON claims(topic);
                """
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── recording ─────────────────────────────────────────────────────────────

    def record_claim(self, topic: str, claim: str, source: str,
                     url: str = "", confidence: float | None = None) -> int:
        """Store a claim with provenance; corroborating claims lift confidence."""
        topic = str(topic or "").strip()[:160]
        claim = str(claim or "").strip()[:600]
        source = str(source or "unknown").strip()[:200]
        if not topic or len(claim) < 8:
            return 0
        conf = score_confidence(claim) if confidence is None else float(confidence)
        conf = max(0.05, min(0.95, conf))
        with self._lock:
            # Exact-duplicate guard: same topic + claim text refreshes the row.
            row = self._db.execute(
                "SELECT id, confidence FROM claims WHERE topic=? AND claim=?",
                (topic, claim)).fetchone()
            if row is not None:
                self._db.execute(
                    "UPDATE claims SET confidence=?, learned_at=? WHERE id=?",
                    (max(conf, float(row["confidence"])), utc_stamp(), row["id"]))
                self._db.commit()
                return int(row["id"])
            cursor = self._db.execute(
                "INSERT INTO claims(topic, claim, source, url, confidence, learned_at) "
                "VALUES (?,?,?,?,?,?)",
                (topic, claim, source, url[:400], conf, utc_stamp()))
            claim_id = int(cursor.lastrowid or 0)
            self._corroborate_locked(topic, claim_id, claim)
            self._db.commit()
        if self.telemetry is not None:
            self.telemetry.metrics.incr("evidence.claims")
        return claim_id

    def _corroborate_locked(self, topic: str, claim_id: int, claim: str) -> None:
        """Each agreeing known source contributes once, including across reopen."""
        mine = _tokens(claim)
        if not mine:
            return
        own = self._db.execute("SELECT source, url FROM claims WHERE id=?", (claim_id,)).fetchone()
        identities = _source_identity(own["source"], own["url"])
        if not identities:
            return
        polarity, numbers = _polarity(claim), _numbers(claim)
        rows = list(self._db.execute(
                "SELECT id, claim, confidence, source, url FROM claims WHERE topic=? AND id!=?",
                (topic, claim_id)))
        # Retained claims are the durable credit history. This avoids a new
        # schema or rewriting old confidences whose original prior is unknown.
        records = [(int(row["id"]), _source_identity(row["source"], row["url"]),
                    _tokens(row["claim"]), _polarity(row["claim"]), _numbers(row["claim"]))
                   for row in rows]
        incoming = (claim_id, identities, mine, polarity, numbers)

        def agrees(a, b):
            return bool(a[2] and b[2]) and a[3] == b[3] \
                and len(a[2] & b[2]) / min(len(a[2]), len(b[2])) >= _OVERLAP_THRESHOLD \
                and not (a[4] and b[4] and a[4] != b[4])

        matching = [row for row in records if row[1] and agrees(incoming, row)]
        contributors = _contributor_count(identities, [row[1] for row in matching])
        if contributors:
            self._db.execute("UPDATE claims SET confidence=MIN(0.95, confidence+?) WHERE id=?",
                             (0.1 * contributors, claim_id))
        for row in matching:
            peers = [other[1] for other in records
                     if other[0] != row[0] and other[1] and agrees(row, other)]
            before = _contributor_count(row[1], peers)
            after = _contributor_count(row[1], [*peers, identities])
            if after > before:
                self._db.execute(
                    "UPDATE claims SET confidence=MIN(0.95, confidence+0.1) WHERE id=?",
                    (row[0],))

    # ── reading ───────────────────────────────────────────────────────────────

    def claims_for(self, topic: str, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM claims WHERE topic=? ORDER BY confidence DESC, id DESC LIMIT ?",
                (str(topic or "").strip()[:160], max(1, limit))).fetchall()
        return [dict(r) for r in rows]

    def provenance(self, claim_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM claims WHERE id=?",
                                   (int(claim_id),)).fetchone()
        return dict(row) if row is not None else None

    def detect_contradictions(self, topic: str) -> list[dict[str, Any]]:
        """Pairs of overlapping claims with opposite polarity or clashing figures."""
        claims = self.claims_for(topic, limit=80)
        conflicts: list[dict[str, Any]] = []
        for i, a in enumerate(claims):
            ta = _tokens(a["claim"])
            if not ta:
                continue
            for b in claims[i + 1:]:
                tb = _tokens(b["claim"])
                if not tb:
                    continue
                overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
                if overlap < _OVERLAP_THRESHOLD:
                    continue
                na, nb = _numbers(a["claim"]), _numbers(b["claim"])
                polarity_clash = _polarity(a["claim"]) != _polarity(b["claim"])
                number_clash = bool(na and nb and na != nb)
                if polarity_clash or number_clash:
                    conflicts.append({
                        "claim_a": a["claim"], "source_a": a["source"],
                        "claim_b": b["claim"], "source_b": b["source"],
                        "kind": "polarity" if polarity_clash else "figures",
                    })
        return conflicts

    # ── verification (Track C) ────────────────────────────────────────────────

    def search_claims(self, text: str, limit: int = 8, scan: int = 400) -> list[dict[str, Any]]:
        """Stored claims whose wording overlaps *text*, across every topic.

        ``claims_for`` needs the topic string to match exactly, which is fine
        for harvesting but useless for checking a sentence that arrived from
        somewhere else entirely.  This scans the most recent *scan* claims and
        ranks them by token overlap, so a claim can be checked against anything
        ORION has ever recorded rather than only against its own topic bucket.
        """
        mine = _tokens(text)
        if not mine:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM claims ORDER BY id DESC LIMIT ?", (max(1, int(scan)),)
            ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            theirs = _tokens(row["claim"])
            if not theirs:
                continue
            overlap = len(mine & theirs) / max(1, min(len(mine), len(theirs)))
            if overlap >= _OVERLAP_THRESHOLD:
                item = dict(row)
                item["overlap"] = round(overlap, 2)
                scored.append((overlap, item))
        scored.sort(key=lambda pair: -pair[0])
        return [item for _score, item in scored[: max(1, int(limit))]]

    def assess_claim(self, claim: str, topic: str = "") -> dict[str, Any]:
        """Check one sentence against the store: supported, contradicted or unsupported.

        Contradiction outranks support deliberately.  A claim the store both
        agrees and disagrees with is exactly the claim the user needs flagged;
        reporting it as supported because two sources nodded would bury the one
        that did not.  ``unsupported`` means silence, not falsity — an empty
        store makes everything unsupported, which is honest.
        """
        claim = str(claim or "").strip()
        if len(claim) < 8:
            return {"claim": claim, "verdict": "unsupported", "confidence": 0.0,
                    "support": [], "conflict": []}
        candidates = self.search_claims(claim, limit=8)
        if topic.strip():
            known = {c["id"] for c in candidates}
            for row in self.claims_for(topic, limit=40):
                if row["id"] not in known:
                    theirs = _tokens(row["claim"])
                    mine = _tokens(claim)
                    if theirs and mine and \
                            len(mine & theirs) / max(1, min(len(mine), len(theirs))) >= _OVERLAP_THRESHOLD:
                        candidates.append(row)
        polarity = _polarity(claim)
        numbers = _numbers(claim)
        support: list[dict[str, Any]] = []
        conflict: list[dict[str, Any]] = []
        for row in candidates:
            other_numbers = _numbers(row["claim"])
            clash = (_polarity(row["claim"]) != polarity) or bool(
                numbers and other_numbers and numbers != other_numbers)
            entry = {"claim": row["claim"], "source": row["source"],
                     "url": row.get("url", ""), "confidence": float(row["confidence"])}
            (conflict if clash else support).append(entry)
        if conflict:
            verdict, confidence = "contradicted", max(c["confidence"] for c in conflict)
        elif support:
            verdict, confidence = "supported", max(s["confidence"] for s in support)
        else:
            verdict, confidence = "unsupported", 0.0
        return {"claim": claim, "verdict": verdict, "confidence": round(confidence, 2),
                "support": support[:3], "conflict": conflict[:3]}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            topics = self._db.execute(
                "SELECT COUNT(DISTINCT topic) FROM claims").fetchone()[0]
            avg = self._db.execute(
                "SELECT AVG(confidence) FROM claims").fetchone()[0] or 0.0
        return {"claims": int(total), "topics": int(topics),
                "avg_confidence": round(float(avg), 2)}


__all__ = ["EvidenceEngine", "score_confidence"]
