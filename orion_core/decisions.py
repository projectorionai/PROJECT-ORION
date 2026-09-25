"""
Decision journal & calibration (Mark XXVI) — was that judgement any good?

ORION remembers what was *said* (rewind), what was *learned* (study) and what was
*done* (focus, missions). What nothing captured is the thing that actually
compounds: **decisions, and whether they turned out to be right.**

This is the superforecasting discipline, made routine:

  1. When a decision is made, record the CHOICE, the REASONING, an explicit
     PREDICTION, and a CONFIDENCE (0-100).
  2. ORION resurfaces it on the review date.
  3. You record what actually happened.
  4. ORION scores your **calibration** — when you say 90%, how often are you
     right? — and your **Brier score**, the standard accuracy measure for
     probabilistic forecasts.

Why it is worth the trouble: the reasoning is captured *before* the outcome is
known, which is the only defence against hindsight bias ("I knew that would
happen"). Over time it answers a question nobody usually gets an answer to —
*am I actually good at this, or just confident?*

Everything is local (``config/decisions.db``), deterministic, and offline: the
scoring is arithmetic, not a model call.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR
from .db import apply_pragmas

#: How a resolved decision turned out.
OUTCOMES = ("right", "wrong", "mixed", "unknowable")

#: Outcomes that count as "the prediction happened" when scoring. 'mixed' counts
#: as a half — a partly-right call is neither a hit nor a clean miss.
_SCORE = {"right": 1.0, "wrong": 0.0, "mixed": 0.5}

#: Default review horizon. Long enough that most decisions have actually played
#: out, short enough to still remember the reasoning.
DEFAULT_REVIEW_DAYS = 30

#: Confidence bins for the calibration curve.
BINS = ((50, 60), (60, 70), (70, 80), (80, 90), (90, 101))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Decision:
    question: str                      # what was being decided
    choice: str                        # what was chosen
    confidence: int = 70               # 0-100, how sure at the time
    prediction: str = ""               # what is expected to happen
    rationale: str = ""                # WHY — captured before the outcome
    tags: list[str] = field(default_factory=list)
    made_at: str = ""
    review_at: str = ""
    outcome: str = ""                  # "" until resolved; then OUTCOMES
    actual: str = ""                   # what really happened
    lesson: str = ""
    resolved_at: str = ""
    id: int | None = None

    @property
    def resolved(self) -> bool:
        return bool(self.outcome)

    def is_due(self, now: datetime | None = None) -> bool:
        if self.resolved or not self.review_at:
            return False
        try:
            return _parse(self.review_at) <= (now or _now())
        except ValueError:
            return False

    def age_days(self, now: datetime | None = None) -> float:
        if not self.made_at:
            return 0.0
        try:
            return ((now or _now()) - _parse(self.made_at)).total_seconds() / 86400.0
        except ValueError:
            return 0.0

    def summary(self) -> str:
        head = f"#{self.id} {self.question}"
        bits = [f"chose: {self.choice}", f"{self.confidence}% sure"]
        if self.resolved:
            bits.append(f"→ {self.outcome.upper()}")
        elif self.review_at:
            bits.append(f"review {self.review_at[:10]}")
        return head + "\n     " + "  ·  ".join(bits)


# ── scoring (pure arithmetic) ────────────────────────────────────────────────

def brier_score(pairs: list[tuple[float, float]]) -> float | None:
    """Mean squared error of probabilistic forecasts.

    ``pairs`` is [(confidence 0-1, actual 0/0.5/1), …]. 0.0 is perfect, 0.25 is
    what you get by always saying 50%, 1.0 is confidently wrong every time.
    None when there is nothing scored yet.
    """
    if not pairs:
        return None
    total = sum((confidence - actual) ** 2 for confidence, actual in pairs)
    return round(total / len(pairs), 4)


def calibration_bins(pairs: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Group forecasts into confidence bands and report the real hit rate.

    This is the honest mirror: if the 90% band comes back at 60%, the confidence
    is inflated — a specific, fixable habit rather than a vague feeling.
    """
    out: list[dict[str, Any]] = []
    for low, high in BINS:
        band = [actual for confidence, actual in pairs
                if low <= confidence * 100 < high]
        if not band:
            continue
        actual_rate = sum(band) / len(band)
        stated = (low + min(high, 100)) / 2 / 100.0
        out.append({
            "band": f"{low}-{min(high, 100)}%",
            "n": len(band),
            "stated": round(stated * 100),
            "actual": round(actual_rate * 100),
            "gap": round((actual_rate - stated) * 100),
        })
    return out


def describe_calibration(bins: list[dict[str, Any]]) -> str:
    """One honest sentence about the shape of the error."""
    scored = [b for b in bins if b["n"] >= 2]
    if not scored:
        return "Not enough resolved decisions yet to judge calibration."
    gaps = [b["gap"] for b in scored]
    average = sum(gaps) / len(gaps)
    if average <= -12:
        return ("Consistently OVERCONFIDENT — outcomes land well below the stated "
                "confidence. Try shading estimates down, or widening what counts "
                "as 'wrong'.")
    if average >= 12:
        return ("Consistently UNDERCONFIDENT — things work out more often than "
                "predicted. The judgement is better than it feels.")
    return "Well calibrated — stated confidence broadly matches what happens."


# ── storage ──────────────────────────────────────────────────────────────────

class DecisionStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "decisions.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL, choice TEXT NOT NULL,
                confidence INTEGER DEFAULT 70, prediction TEXT DEFAULT '',
                rationale TEXT DEFAULT '', tags TEXT DEFAULT '',
                made_at TEXT DEFAULT '', review_at TEXT DEFAULT '',
                outcome TEXT DEFAULT '', actual TEXT DEFAULT '',
                lesson TEXT DEFAULT '', resolved_at TEXT DEFAULT '')""")
        self._db.commit()

    def _row(self, row: sqlite3.Row) -> Decision:
        return Decision(
            id=row["id"], question=row["question"], choice=row["choice"],
            confidence=row["confidence"], prediction=row["prediction"] or "",
            rationale=row["rationale"] or "",
            tags=[t for t in (row["tags"] or "").split(",") if t],
            made_at=row["made_at"] or "", review_at=row["review_at"] or "",
            outcome=row["outcome"] or "", actual=row["actual"] or "",
            lesson=row["lesson"] or "", resolved_at=row["resolved_at"] or "")

    def add(self, decision: Decision) -> Decision:
        cur = self._db.execute(
            "INSERT INTO decisions (question, choice, confidence, prediction, "
            "rationale, tags, made_at, review_at) VALUES (?,?,?,?,?,?,?,?)",
            (decision.question, decision.choice, decision.confidence,
             decision.prediction, decision.rationale, ",".join(decision.tags),
             decision.made_at, decision.review_at))
        self._db.commit()
        decision.id = int(cur.lastrowid)
        return decision

    def update(self, decision: Decision) -> None:
        if decision.id is None:
            return
        self._db.execute(
            "UPDATE decisions SET outcome=?, actual=?, lesson=?, resolved_at=?, "
            "review_at=? WHERE id=?",
            (decision.outcome, decision.actual, decision.lesson,
             decision.resolved_at, decision.review_at, decision.id))
        self._db.commit()

    def get(self, decision_id: int) -> Decision | None:
        row = self._db.execute("SELECT * FROM decisions WHERE id=?",
                               (decision_id,)).fetchone()
        return self._row(row) if row else None

    def all(self) -> list[Decision]:
        return [self._row(r) for r in self._db.execute(
            "SELECT * FROM decisions ORDER BY id").fetchall()]

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


# ── the journal ──────────────────────────────────────────────────────────────

class DecisionJournal:
    """Record decisions, resurface them, and score the judgement behind them."""

    def __init__(self, store: DecisionStore | None = None) -> None:
        self.store = store or DecisionStore()

    def record(self, question: str, choice: str, *, confidence: int = 70,
               prediction: str = "", rationale: str = "",
               review_days: int = DEFAULT_REVIEW_DAYS, tags: list[str] | None = None,
               now: datetime | None = None) -> Decision:
        question = str(question or "").strip()
        choice = str(choice or "").strip()
        if not question or not choice:
            raise ValueError("a decision needs both the question and the choice")
        now = now or _now()
        confidence = max(0, min(100, int(confidence)))
        return self.store.add(Decision(
            question=question, choice=choice, confidence=confidence,
            prediction=str(prediction or "").strip(),
            rationale=str(rationale or "").strip(),
            tags=[t.strip() for t in (tags or []) if str(t).strip()],
            made_at=_iso(now),
            review_at=_iso(now + timedelta(days=max(1, int(review_days))))))

    def due(self, now: datetime | None = None) -> list[Decision]:
        now = now or _now()
        return sorted((d for d in self.store.all() if d.is_due(now)),
                      key=lambda d: d.review_at)

    def open_decisions(self) -> list[Decision]:
        return [d for d in self.store.all() if not d.resolved]

    def resolve(self, decision_id: int, outcome: str, *, actual: str = "",
                lesson: str = "", now: datetime | None = None) -> Decision | None:
        decision = self.store.get(int(decision_id))
        if decision is None:
            return None
        outcome = str(outcome or "").strip().lower()
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        decision.outcome = outcome
        decision.actual = str(actual or "").strip()
        decision.lesson = str(lesson or "").strip()
        decision.resolved_at = _iso(now or _now())
        self.store.update(decision)
        return decision

    def defer(self, decision_id: int, days: int = 30,
              now: datetime | None = None) -> Decision | None:
        """Not resolvable yet — push the review out rather than lose it."""
        decision = self.store.get(int(decision_id))
        if decision is None or decision.resolved:
            return None
        decision.review_at = _iso((now or _now()) + timedelta(days=max(1, int(days))))
        self.store.update(decision)
        return decision

    # -- scoring --
    def _scored_pairs(self) -> list[tuple[float, float]]:
        pairs: list[tuple[float, float]] = []
        for decision in self.store.all():
            if decision.outcome in _SCORE:
                pairs.append((decision.confidence / 100.0, _SCORE[decision.outcome]))
        return pairs

    def calibration(self) -> dict[str, Any]:
        pairs = self._scored_pairs()
        bins = calibration_bins(pairs)
        return {
            "scored": len(pairs),
            "brier": brier_score(pairs),
            "bins": bins,
            "verdict": describe_calibration(bins),
            "hit_rate": (round(100 * sum(a for _c, a in pairs) / len(pairs))
                         if pairs else None),
        }

    def stats(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now()
        everything = self.store.all()
        resolved = [d for d in everything if d.resolved]
        return {
            "total": len(everything),
            "open": len(everything) - len(resolved),
            "resolved": len(resolved),
            "due": len(self.due(now)),
            "right": sum(1 for d in resolved if d.outcome == "right"),
            "wrong": sum(1 for d in resolved if d.outcome == "wrong"),
            "mixed": sum(1 for d in resolved if d.outcome == "mixed"),
            "average_confidence": (round(sum(d.confidence for d in everything)
                                         / len(everything)) if everything else None),
        }

    def lessons(self, limit: int = 10) -> list[str]:
        """What was actually learned — the point of the exercise."""
        out = []
        for decision in reversed(self.store.all()):
            if decision.lesson:
                out.append(f"{decision.question}: {decision.lesson}")
            if len(out) >= limit:
                break
        return out


__all__ = [
    "OUTCOMES", "BINS", "DEFAULT_REVIEW_DAYS", "Decision", "DecisionStore",
    "DecisionJournal", "brier_score", "calibration_bins", "describe_calibration",
]
