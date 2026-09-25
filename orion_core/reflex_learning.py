"""
Learned reflexes — ORION discovering his own fast-paths from what he actually does.

A reflex (see ``reflex.py``) replaces a 1-3 second model round-trip with a ~3
microsecond local dispatch. It is by far the biggest perceived-latency win in
the system, and until now every reflex was hand-written by a developer guessing
which phrases matter. That guess ages badly: the phrases that matter are the
ones *this user* actually says.

So this module mines them. Every turn where the model chose a tool is observed;
a phrase that has gone to the same tool and action, successfully, on N separate
occasions becomes a candidate reflex. The next time it is said, ORION answers
instantly instead of asking the model something it has answered identically
three times already.

**Why this is safe, in order of importance.**

1. *Exact normalised phrase only.* No stemming, no fuzzy matching, no
   generalisation to "similar" phrases. The only utterance that can fire a
   learned reflex is one whose normalised form is character-identical to one
   already seen going to that tool. Generalisation is where a learned classifier
   would start guessing, and a wrong guess here means ORION acts without
   thinking — the one failure mode worth engineering the whole design around.

2. *Read-only tools only.* Candidacy is gated on two existing, maintained
   classifications ANDed together: ``concurrency.classify`` must say PARALLEL
   (its definition of read-only, action-aware, with SERIAL as the fail-safe
   default for anything unclassified) and ``remote_capability.classify`` must
   not say FORBID. No new denylist is invented here to drift out of date. This
   is a stricter bar than a hand-written reflex gets, on purpose — a rule ORION
   derives about himself with nobody watching should clear a higher one.

3. *Unanimity, not majority.* A phrase that ever went to a second tool is
   disqualified permanently, not out-voted. Ambiguity means the model should
   decide.

4. *Failures poison the phrase.* A tool call that returned not-ok does not count
   towards promotion — otherwise ORION would learn to fail faster.

5. *Nothing is promoted silently.* ``candidates()`` proposes; ``promote()``
   commits. The caller decides whether that needs the user.

The observation store uses ORION's SQLite policy (WAL, relaxed fsync) because it
is written on the turn path — see ``orion_core/db.py``.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import apply_pragmas

#: How many separate successful, unanimous observations before a phrase is a
#: candidate. Three is the smallest number that is not a coincidence: twice can
#: be one session repeating itself, and waiting for five means the win arrives
#: long after the user has noticed the lag.
MIN_OBSERVATIONS = 3

#: A phrase longer than this is a sentence, not a command — the same ceiling
#: reflex.py applies, kept in step deliberately.
MAX_PHRASE_LEN = 90

DEFAULT_PATH = Path("config") / "reflex_learning.db"

_PUNCT = re.compile(r"[^\w\s']+")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Nothing more.

    Deliberately not stemming or lemmatising: "delete the file" and "deleting
    the files" should NOT collapse to one key. Aggressive normalisation is how a
    matcher starts firing on things it was never taught.
    """
    lowered = (text or "").strip().lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", lowered)).strip()


@dataclass(frozen=True)
class LearnedReflex:
    """A phrase that has earned a local fast-path."""
    phrase: str
    tool: str
    action: str
    hits: int
    promoted: bool = False

    def why(self) -> str:
        return f"learned:{self.tool}" + (f".{self.action}" if self.action else "")

    def args(self) -> dict[str, Any]:
        return {"action": self.action} if self.action else {}

    def line(self) -> str:
        mark = "✓" if self.promoted else "·"
        target = self.tool + (f".{self.action}" if self.action else "")
        return f"  {mark} {self.phrase!r} -> {target}  ({self.hits} times)"


class ReflexLearner:
    """Observes turns, proposes reflexes, and matches promoted ones."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
                phrase     TEXT NOT NULL,
                tool       TEXT NOT NULL,
                action     TEXT NOT NULL DEFAULT '',
                hits       INTEGER NOT NULL DEFAULT 0,
                failures   INTEGER NOT NULL DEFAULT 0,
                last_at    REAL NOT NULL DEFAULT 0,
                promoted   INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (phrase, tool, action)
            );
            CREATE INDEX IF NOT EXISTS idx_obs_phrase ON observations(phrase);
            CREATE INDEX IF NOT EXISTS idx_obs_promoted ON observations(promoted);
            """
        )
        self._db.commit()
        self._cache: dict[str, LearnedReflex] | None = None

    # ── observing ────────────────────────────────────────────────────────────

    def observe(self, utterance: str, tool: str, args: dict[str, Any] | None = None,
                ok: bool = True) -> str | None:
        """Record that ``utterance`` led to ``tool``. Returns the stored phrase.

        Returns None (and stores nothing) when the turn is not a candidate: an
        empty or over-long utterance, or a tool that is not read-only.
        """
        phrase = normalise(utterance)
        if not phrase or len(phrase) > MAX_PHRASE_LEN or not tool:
            return None
        if not self._is_read_only(tool, args):
            return None
        action = str((args or {}).get("action") or "").strip().lower()
        self._db.execute(
            """
            INSERT INTO observations (phrase, tool, action, hits, failures, last_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(phrase, tool, action) DO UPDATE SET
                hits     = hits + excluded.hits,
                failures = failures + excluded.failures,
                last_at  = excluded.last_at
            """,
            (phrase, tool, action, 1 if ok else 0, 0 if ok else 1, time.time()),
        )
        self._db.commit()
        self._cache = None
        return phrase

    @staticmethod
    def _is_read_only(tool: str, args: dict[str, Any] | None) -> bool:
        """Only genuinely read-only, non-forbidden calls may become a reflex.

        Two existing, maintained classifications are ANDed rather than a new
        denylist being invented here to drift out of date:

        * ``concurrency.classify`` must return PARALLEL — its definition of
          "read-only or otherwise independent". It is action-aware (a call whose
          arguments select a write branch is demoted) and SERIAL is its
          fail-safe default, so an unclassified tool is excluded automatically.
        * ``remote_capability.classify`` must not return FORBID, which covers
          code execution, self-modification and security toggles.

        This is deliberately stricter than the bar for a hand-written reflex.
        ``study.stats`` is a perfectly good reflex — and it IS one, in
        reflex.py, reviewed by a human. A rule ORION derives about himself with
        nobody looking gets the tighter gate.
        """
        payload = dict(args or {})
        try:
            from .concurrency import ToolClass, classify as classify_concurrency
            if classify_concurrency(tool, payload) is not ToolClass.PARALLEL:
                return False
        except Exception:
            return False              # cannot classify it -> do not learn it
        try:
            from .remote_capability import Tier, classify as classify_tier
            return classify_tier(tool, payload) is not Tier.FORBID
        except Exception:
            return False

    # ── proposing ────────────────────────────────────────────────────────────

    def candidates(self, min_observations: int = MIN_OBSERVATIONS) -> list[LearnedReflex]:
        """Phrases that have earned a fast-path but are not promoted yet."""
        return [c for c in self._eligible(min_observations) if not c.promoted]

    def _eligible(self, min_observations: int = MIN_OBSERVATIONS) -> list[LearnedReflex]:
        rows = self._db.execute(
            "SELECT phrase, tool, action, hits, failures, promoted FROM observations"
        ).fetchall()
        by_phrase: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_phrase.setdefault(row["phrase"], []).append(row)

        out: list[LearnedReflex] = []
        for phrase, group in by_phrase.items():
            # Unanimity: a phrase seen going anywhere else is disqualified,
            # regardless of how lopsided the counts are.
            if len({(r["tool"], r["action"]) for r in group}) != 1:
                continue
            row = group[0]
            if row["failures"]:
                continue              # a phrase that ever failed is not settled
            if row["hits"] < min_observations:
                continue
            out.append(LearnedReflex(phrase=phrase, tool=row["tool"],
                                     action=row["action"], hits=row["hits"],
                                     promoted=bool(row["promoted"])))
        return sorted(out, key=lambda c: -c.hits)

    # ── promoting ────────────────────────────────────────────────────────────

    def promote(self, phrase: str) -> bool:
        """Turn an eligible candidate into a live reflex. False if not eligible."""
        key = normalise(phrase)
        if not any(c.phrase == key for c in self._eligible()):
            return False
        self._db.execute("UPDATE observations SET promoted = 1 WHERE phrase = ?", (key,))
        self._db.commit()
        self._cache = None
        return True

    def demote(self, phrase: str) -> bool:
        """Withdraw a learned reflex — the escape hatch when one is unwanted."""
        key = normalise(phrase)
        cursor = self._db.execute(
            "UPDATE observations SET promoted = 0 WHERE phrase = ? AND promoted = 1",
            (key,))
        self._db.commit()
        self._cache = None
        return cursor.rowcount > 0

    def forget(self, phrase: str) -> bool:
        """Erase every observation of a phrase, so it starts over."""
        key = normalise(phrase)
        cursor = self._db.execute("DELETE FROM observations WHERE phrase = ?", (key,))
        self._db.commit()
        self._cache = None
        return cursor.rowcount > 0

    def learned(self) -> list[LearnedReflex]:
        """Every promoted reflex, newest-strongest first."""
        return [c for c in self._eligible() if c.promoted]

    # ── matching (the point of it all) ───────────────────────────────────────

    def _table(self) -> dict[str, LearnedReflex]:
        if self._cache is None:
            self._cache = {c.phrase: c for c in self.learned()}
        return self._cache

    def match(self, text: str) -> LearnedReflex | None:
        """The promoted reflex for this exact normalised phrase, or None.

        A dict lookup on a normalised string — the cost is the normalisation,
        not the search, so this stays flat as the learned table grows.
        """
        if not text:
            return None
        phrase = normalise(text)
        if not phrase or len(phrase) > MAX_PHRASE_LEN:
            return None
        return self._table().get(phrase)

    # ── reporting ────────────────────────────────────────────────────────────

    def report(self) -> str:
        live = self.learned()
        pending = self.candidates()
        if not live and not pending:
            return ("No learned reflexes yet — ORION needs to see the same "
                    f"request reach the same tool {MIN_OBSERVATIONS} times first.")
        parts: list[str] = []
        if live:
            parts.append(f"{len(live)} learned reflex(es) answering instantly:")
            parts.extend(c.line() for c in live)
        if pending:
            parts.append(f"{len(pending)} candidate(s) ready to promote:")
            parts.extend(c.line() for c in pending)
        return "\n".join(parts)

    def stats(self) -> dict[str, int]:
        row = self._db.execute(
            "SELECT COUNT(*) AS phrases, COALESCE(SUM(hits), 0) AS hits "
            "FROM observations").fetchone()
        return {"phrases": row["phrases"], "observations": row["hits"],
                "learned": len(self.learned()), "candidates": len(self.candidates())}

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_PATH", "MAX_PHRASE_LEN", "MIN_OBSERVATIONS", "LearnedReflex",
    "ReflexLearner", "normalise",
]
