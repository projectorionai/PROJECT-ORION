"""
Study & Recall (Mark XXIV — Mastery) — active recall over anything ORION knows.

The gap this closes: ORION could already ingest notes, PDFs, research and whole
conversations, but *knowing* something once is not *remembering* it. This gives
ORION a general-purpose spaced-repetition memory for the USER's own learning —
not vocabulary (the language tutor already owns that), but any subject the user
studies: neural engineering, psychology, a certification, the internals of ORION
itself.

Three parts:

  * ``StudyCard`` — a question/answer card carrying SM-2 scheduling state, using
    the *exact* algorithm the language tutor is built on (``sm2_step``), so the
    two spaced-repetition systems can never drift apart.
  * ``StudyStore`` — a small SQLite database (``config/study.db``): the cards, and
    a review log that makes a real retention figure possible.
  * ``StudyEngine`` — add cards, pull what is DUE, grade a review (advancing the
    schedule), run a review session, report mastery, and turn a block of source
    material into cards automatically: model-written when a generator is wired in,
    extractive when it is not, so it degrades to something real rather than to
    nothing.

Everything here is deterministic and offline except the optional model-written
card generation, which always has an extractive fallback. No network, no fakes.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

from .constants import CONFIG_DIR
from .language_tutor import sm2_step
from .db import apply_pragmas

#: An interval at or beyond this many days means the card has moved from active
#: learning into long-term retention. 21 days is the conventional Anki threshold
#: for a "mature" card and it matches the point where reviews become rare.
MASTERED_DAYS = 21.0

_STOP = frozenset(
    "the a an of to in on at for and or but is are was were be been being this "
    "that these those it its as by with from into about over under it's".split())


def human_interval(days: float) -> str:
    """A spoken description of when a card comes back — 'later today', '6 days',
    '3 months' — so ORION can say it naturally after a review."""
    if days < 1:
        return "later today"
    if days < 2:
        return "1 day"
    if days < 60:
        return f"{int(round(days))} days"
    if days < 365:
        return f"{round(days / 30)} months"
    return f"{round(days / 365, 1)} years"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── the card ─────────────────────────────────────────────────────────────────

@dataclass
class StudyCard:
    front: str                          # the prompt / question
    back: str                           # the answer
    deck: str = "General"
    source: str = ""                    # where it came from (a note, a PDF, a URL)
    tags: list[str] = field(default_factory=list)
    # SM-2 scheduling state — identical field set to the tutor's VocabCard.
    ease: float = 2.5
    interval_days: float = 0.0
    repetitions: int = 0
    lapses: int = 0
    due_at: str = ""                    # ISO; empty == due now (never reviewed)
    reviews: int = 0                    # total times reviewed
    created_at: str = field(default_factory=lambda: _iso(_now()))
    last_review: str = ""
    id: int | None = None

    def is_due(self, now: datetime | None = None) -> bool:
        if not self.due_at:
            return True                 # never reviewed → due now
        now = now or _now()
        try:
            return _parse_iso(self.due_at) <= now
        except ValueError:
            return True

    @property
    def is_new(self) -> bool:
        return self.reviews == 0

    @property
    def is_mastered(self) -> bool:
        return self.interval_days >= MASTERED_DAYS

    def review(self, quality: int, now: datetime | None = None) -> "StudyCard":
        """Advance one SM-2 step and stamp the review. Mutates and returns self."""
        now = now or _now()
        self.ease, self.interval_days, self.repetitions, self.lapses = sm2_step(
            ease=self.ease, interval_days=self.interval_days,
            repetitions=self.repetitions, lapses=self.lapses, quality=quality)
        self.due_at = _iso(now + timedelta(days=self.interval_days))
        self.reviews += 1
        self.last_review = _iso(now)
        return self


# ── persistence ──────────────────────────────────────────────────────────────

class StudyStore:
    """The cards and the review log, in one small SQLite file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "study.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the store is touched from the asyncio tool
        # path and may later be read by a GUI deck page or the standby loop; the
        # writes are small and serialised by SQLite's own locking.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deck TEXT NOT NULL DEFAULT 'General',
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                source TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                ease REAL DEFAULT 2.5,
                interval_days REAL DEFAULT 0,
                repetitions INTEGER DEFAULT 0,
                lapses INTEGER DEFAULT 0,
                due_at TEXT DEFAULT '',
                reviews INTEGER DEFAULT 0,
                created_at TEXT DEFAULT '',
                last_review TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER NOT NULL,
                at TEXT NOT NULL,
                quality INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck);
            CREATE INDEX IF NOT EXISTS idx_cards_due ON cards(due_at);
            """
        )
        self._db.commit()

    # -- writes --
    def add(self, card: StudyCard) -> StudyCard:
        cur = self._db.execute(
            "INSERT INTO cards (deck, front, back, source, tags, ease, "
            "interval_days, repetitions, lapses, due_at, reviews, created_at, "
            "last_review) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (card.deck, card.front, card.back, card.source, json.dumps(card.tags),
             card.ease, card.interval_days, card.repetitions, card.lapses,
             card.due_at, card.reviews, card.created_at, card.last_review))
        self._db.commit()
        card.id = int(cur.lastrowid)
        return card

    def update(self, card: StudyCard) -> None:
        if card.id is None:
            return
        self._db.execute(
            "UPDATE cards SET deck=?, front=?, back=?, source=?, tags=?, ease=?, "
            "interval_days=?, repetitions=?, lapses=?, due_at=?, reviews=?, "
            "last_review=? WHERE id=?",
            (card.deck, card.front, card.back, card.source, json.dumps(card.tags),
             card.ease, card.interval_days, card.repetitions, card.lapses,
             card.due_at, card.reviews, card.last_review, card.id))
        self._db.commit()

    def delete(self, card_id: int) -> bool:
        cur = self._db.execute("DELETE FROM cards WHERE id=?", (card_id,))
        self._db.commit()
        return cur.rowcount > 0

    def log_review(self, card_id: int, quality: int, at: datetime | None = None) -> None:
        self._db.execute("INSERT INTO reviews (card_id, at, quality) VALUES (?,?,?)",
                         (card_id, _iso(at or _now()), int(quality)))
        self._db.commit()

    # -- reads --
    def _row(self, row: sqlite3.Row) -> StudyCard:
        try:
            tags = json.loads(row["tags"] or "[]")
        except (ValueError, TypeError):
            tags = []
        return StudyCard(
            id=row["id"], deck=row["deck"], front=row["front"], back=row["back"],
            source=row["source"] or "", tags=list(tags), ease=row["ease"],
            interval_days=row["interval_days"], repetitions=row["repetitions"],
            lapses=row["lapses"], due_at=row["due_at"] or "", reviews=row["reviews"],
            created_at=row["created_at"] or "", last_review=row["last_review"] or "")

    def get(self, card_id: int) -> StudyCard | None:
        row = self._db.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
        return self._row(row) if row else None

    def all(self, deck: str | None = None) -> list[StudyCard]:
        if deck:
            rows = self._db.execute("SELECT * FROM cards WHERE deck=? ORDER BY id",
                                    (deck,)).fetchall()
        else:
            rows = self._db.execute("SELECT * FROM cards ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    # -- indexed queries -------------------------------------------------
    #
    # These exist because the Python-side equivalents were O(n) over the whole
    # deck: with 3,000 cards, due()/next_due()/stats() each cost ~15 ms, and
    # next_due() runs on EVERY review. Pushing the filter into SQL (against the
    # existing due_at/deck indexes) makes them flat.

    def due_cards(self, deck: str | None = None, limit: int = 20,
                  now: datetime | None = None) -> list[StudyCard]:
        """Cards ready for review, never-seen first then most overdue."""
        stamp = _iso(now or _now())
        clause = "(due_at = '' OR due_at <= ?)"
        params: list[Any] = [stamp]
        if deck:
            clause += " AND deck = ?"
            params.append(deck)
        params.append(max(1, int(limit)))
        rows = self._db.execute(
            f"SELECT * FROM cards WHERE {clause} "
            "ORDER BY (reviews > 0) ASC, due_at ASC, id ASC LIMIT ?",
            params).fetchall()
        return [self._row(r) for r in rows]

    def due_count(self, deck: str | None = None,
                  now: datetime | None = None) -> int:
        stamp = _iso(now or _now())
        clause = "(due_at = '' OR due_at <= ?)"
        params: list[Any] = [stamp]
        if deck:
            clause += " AND deck = ?"
            params.append(deck)
        row = self._db.execute(
            f"SELECT COUNT(*) AS n FROM cards WHERE {clause}", params).fetchone()
        return int(row["n"] if row else 0)

    def counts(self, deck: str | None = None,
               now: datetime | None = None) -> dict[str, int]:
        """total / new / mastered / lapses in ONE aggregate query."""
        where, params = ("WHERE deck = ?", [deck]) if deck else ("", [])
        row = self._db.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN reviews = 0 THEN 1 ELSE 0 END) AS new_cards, "
            f"SUM(CASE WHEN interval_days >= {MASTERED_DAYS} THEN 1 ELSE 0 END) AS mastered, "
            "SUM(lapses) AS lapses "
            f"FROM cards {where}", params).fetchone()
        if row is None:
            return {"total": 0, "new": 0, "mastered": 0, "lapses": 0}
        return {"total": int(row["total"] or 0), "new": int(row["new_cards"] or 0),
                "mastered": int(row["mastered"] or 0), "lapses": int(row["lapses"] or 0)}

    def deck_summary(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Per-deck totals in one grouped query (was one full scan per deck)."""
        stamp = _iso(now or _now())
        rows = self._db.execute(
            "SELECT deck, COUNT(*) AS total, "
            "SUM(CASE WHEN due_at = '' OR due_at <= ? THEN 1 ELSE 0 END) AS due, "
            f"SUM(CASE WHEN interval_days >= {MASTERED_DAYS} THEN 1 ELSE 0 END) AS mastered "
            "FROM cards GROUP BY deck ORDER BY deck", (stamp,)).fetchall()
        return [{"deck": r["deck"], "total": int(r["total"] or 0),
                 "due": int(r["due"] or 0), "mastered": int(r["mastered"] or 0)}
                for r in rows]

    def decks(self) -> list[str]:
        rows = self._db.execute(
            "SELECT DISTINCT deck FROM cards ORDER BY deck").fetchall()
        return [r["deck"] for r in rows]

    def retention(self, deck: str | None = None, days: int = 30,
                  now: datetime | None = None) -> float | None:
        """Share of reviews graded a pass (>=3) over the window, as a percentage.
        None when there is nothing to measure yet.

        ``now`` anchors the window, like every other query on this store. It
        used to read the wall clock unconditionally while ``stats()`` passed an
        explicit ``now`` to each of its siblings, so a caller supplying a clock
        got counts from that date and retention from today.
        """
        since = _iso((now or _now()) - timedelta(days=days))
        if deck:
            rows = self._db.execute(
                "SELECT r.quality FROM reviews r JOIN cards c ON c.id=r.card_id "
                "WHERE r.at>=? AND c.deck=?", (since, deck)).fetchall()
        else:
            rows = self._db.execute(
                "SELECT quality FROM reviews WHERE at>=?", (since,)).fetchall()
        if not rows:
            return None
        passed = sum(1 for r in rows if r["quality"] >= 3)
        return round(100.0 * passed / len(rows), 1)

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


# ── card generation from source material ─────────────────────────────────────

_Q_PREFIX = re.compile(r"^\s*Q\s*[:.\-]\s*(.+)$", re.IGNORECASE)
#: The answer marker: an optional leading '|', then 'A:' / 'A-' / 'A.'.
_A_SPLIT = re.compile(r"\s*\|?\s*A\s*[:.\-]\s*", re.IGNORECASE)
_DEFN = re.compile(r"^\s*(?P<term>[A-Z][\w \-/]{2,60}?)\s*(?:[:\-–—]|\bis\b|\bare\b|"
                   r"\bmeans\b|\brefers to\b)\s+(?P<defn>.+?)\.?\s*$")


def parse_generated_cards(text: str) -> list[tuple[str, str]]:
    """Pull ``Q: … A: …`` (or ``Q: … | …``) pairs out of a model's reply. Lenient
    about the separators a model actually produces — a bare '|', an 'A:' marker,
    or both together."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _Q_PREFIX.match(line)
        if not m:
            continue
        body = m.group(1)
        halves = _A_SPLIT.split(body, maxsplit=1)
        if len(halves) == 2:
            q, a = halves
        elif "|" in body:
            q, a = body.split("|", 1)
        else:
            continue
        q = q.strip().rstrip("|-–—: ").strip()
        a = a.strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def extractive_cards(text: str, limit: int = 12) -> list[tuple[str, str]]:
    """A model-free fallback that still yields *real* cards from the material:

      * a "Term: definition" / "X is Y" line becomes  What is <Term>? → <definition>
      * otherwise a substantial sentence becomes a cloze: its rarest content word
        is blanked and becomes the answer.

    Deterministic, so the same notes always give the same cards."""
    cards: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Definitions first — they make the cleanest questions.
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) < 8:
            continue
        m = _DEFN.match(line)
        if m:
            term = m.group("term").strip().rstrip(":-–—").strip()
            defn = m.group("defn").strip()
            key = term.lower()
            if term and defn and len(defn) >= 3 and key not in seen:
                seen.add(key)
                cards.append((f"What is {term}?", defn))
                if len(cards) >= limit:
                    return cards

    # Then cloze deletions from remaining sentences.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.strip()
        if len(s) < 30 or len(s.split()) < 6:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z\-']{3,}", s)
        candidates = [w for w in words if w.lower() not in _STOP]
        if not candidates:
            continue
        # rarest = longest here (a cheap, dependency-free proxy for salience)
        answer = max(candidates, key=len)
        if answer.lower() in seen:
            continue
        seen.add(answer.lower())
        blanked = re.sub(rf"\b{re.escape(answer)}\b", "_____", s, count=1)
        if "_____" in blanked:
            cards.append((blanked, answer))
        if len(cards) >= limit:
            break
    return cards


# ── the engine ───────────────────────────────────────────────────────────────

Generator = Callable[[str], Awaitable[str]]


class StudyEngine:
    """Add cards, surface what's due, grade recall, and report mastery."""

    def __init__(self, store: StudyStore | None = None,
                 generate: Generator | None = None) -> None:
        self.store = store or StudyStore()
        self._generate = generate
        # The card currently being asked in an interactive review, so a follow-up
        # "grade" call knows what it is grading without the model re-stating it.
        self._current: StudyCard | None = None

    # -- authoring --
    def add(self, front: str, back: str, deck: str = "General", source: str = "",
            tags: Sequence[str] | None = None) -> StudyCard:
        front, back = front.strip(), back.strip()
        if not front or not back:
            raise ValueError("a card needs both a front and a back")
        return self.store.add(StudyCard(
            front=front, back=back, deck=(deck or "General").strip() or "General",
            source=source, tags=list(tags or [])))

    def add_many(self, pairs: Iterable[tuple[str, str]], deck: str = "General",
                 source: str = "") -> list[StudyCard]:
        made: list[StudyCard] = []
        for front, back in pairs:
            try:
                made.append(self.add(front, back, deck=deck, source=source))
            except ValueError:
                continue
        return made

    async def generate_cards(self, text: str, deck: str = "General",
                             count: int = 8, source: str = "") -> list[StudyCard]:
        """Turn a block of study material into cards. Uses the model when one is
        wired in (better questions); always falls back to extractive cards so the
        result is real even offline. Never invents facts beyond the material."""
        text = (text or "").strip()
        if not text:
            return []
        pairs: list[tuple[str, str]] = []
        if self._generate is not None:
            prompt = (
                f"From the study material below, write up to {count} concise "
                "flashcards for active recall. Each on its own line, exactly as "
                "'Q: <question> | A: <answer>'. Only use facts present in the "
                "material; do not add outside information.\n\nMATERIAL:\n" + text[:6000])
            try:
                reply = await self._generate(prompt)
                pairs = parse_generated_cards(reply or "")
            except Exception:
                pairs = []
        if not pairs:
            pairs = extractive_cards(text, limit=count)
        return self.add_many(pairs[:count], deck=deck, source=source)

    # -- review --
    def due(self, deck: str | None = None, limit: int = 20,
            now: datetime | None = None) -> list[StudyCard]:
        """Cards ready for review — resolved in SQL against the due_at index."""
        return self.store.due_cards(deck, limit=limit, now=now or _now())

    def next_due(self, deck: str | None = None,
                 now: datetime | None = None) -> StudyCard | None:
        cards = self.due(deck, limit=1, now=now)
        self._current = cards[0] if cards else None
        return self._current

    def grade(self, quality: int, card_id: int | None = None,
              now: datetime | None = None) -> StudyCard | None:
        """Grade a review (0–5), advance the schedule, persist and log it.

        Grades the given card, or the one most recently surfaced by next_due when
        no id is passed (the natural voice flow: ORION asks, the user answers,
        ORION grades). Returns the updated card, or None if it cannot be found."""
        now = now or _now()
        card = self.store.get(card_id) if card_id is not None else self._current
        if card is None or card.id is None:
            return None
        card.review(int(quality), now=now)
        self.store.update(card)
        self.store.log_review(card.id, int(quality), at=now)
        if self._current is not None and self._current.id == card.id:
            self._current = None
        return card

    # -- reporting --
    def stats(self, deck: str | None = None,
              now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        counts = self.store.counts(deck, now)
        due = self.store.due_count(deck, now)
        new = counts["new"]
        mastered = counts["mastered"]
        learning = counts["total"] - new - mastered
        lapses = counts["lapses"]
        return {
            "deck": deck or "all",
            "total": counts["total"],
            "due": due,
            "new": new,
            "learning": max(0, learning),
            "mastered": mastered,
            "lapses": lapses,
            "retention": self.store.retention(deck, now=now),   # % or None
        }

    def decks(self) -> list[dict[str, object]]:
        """One grouped query rather than a full scan per deck."""
        return self.store.deck_summary(_now())


__all__ = [
    "MASTERED_DAYS", "StudyCard", "StudyStore", "StudyEngine",
    "parse_generated_cards", "extractive_cards", "human_interval",
]
