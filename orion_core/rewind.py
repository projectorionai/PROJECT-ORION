"""
Rewind (CAP-04) — the scrubbable day.

    "What did we decide about the exe yesterday?"

ORION already writes a verbatim transcript of every turn to
``conversations/<date>_<id>.jsonl`` — timestamp, speaker, text, one line each.
The data for a memory has been on disk all along; what was missing was a spine
and a surface. This is both:

* a **timeline** — every turn, in order, filterable to a single day;
* a **search** — jump to the moment something was said, with its timestamp;
* a **decisions** view — the turns where something was actually settled ("let's
  ...", "we'll ...", "the plan is ...") rather than merely discussed, which is
  what "what did we decide about X" is really asking for;
* a **recap** — what a given day amounted to, extractively or, when a model is
  available, in a sentence or two.

It reads the transcripts and nothing else — no new store, no duplicate of data
that already exists — so it is always in step with what was really said, and it
is pure and deterministic: given a folder of transcripts it returns the same
timeline every time, which is what the tests pin down.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

# Common words that carry no signal for relevance ranking.
_RECALL_STOP = frozenset(
    "the a an and or of to in on for with is are was were be do did what when "
    "how why who did we you i about that this it me my our your they them".split())

# Phrases that mark a turn as a DECISION rather than discussion.
_DECISION_RE = re.compile(
    r"\b(let's|lets|we'll|we will|i'll|i will|we should|let me|"
    r"the plan is|going to|decided|decision|we're going to|we are going to|"
    r"i've decided|agreed|final answer|conclusion)\b", re.IGNORECASE)


def _parse_at(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # tolerate a trailing 'Z' or space-separated form
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class Turn:
    at: datetime | None
    role: str
    content: str
    source: str

    @property
    def is_user(self) -> bool:
        return str(self.role or "").lower().startswith("user")

    @property
    def speaker(self) -> str:
        return "You" if self.is_user else "ORION"

    @property
    def clock(self) -> str:
        return self.at.strftime("%Y-%m-%d %H:%M") if self.at else "—"

    def line(self) -> str:
        return f"[{self.clock}] {self.speaker}: {self.content}"


class RewindTimeline:
    """A read-only view over ORION's verbatim transcripts."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._turns: list[Turn] | None = None

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self) -> list[Turn]:
        if self._turns is not None:
            return self._turns
        turns: list[Turn] = []
        if self.directory.is_dir():
            for path in sorted(self.directory.glob("*.jsonl")):
                try:
                    raw = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = str(rec.get("content") or "").strip()
                    if not content:
                        continue
                    turns.append(Turn(
                        at=_parse_at(rec.get("at")),
                        role=str(rec.get("role") or "user"),
                        content=content, source=path.name))
        turns.sort(key=lambda t: (t.at or datetime.min.replace(tzinfo=timezone.utc)))
        self._turns = turns
        return turns

    def reload(self) -> None:
        self._turns = None

    def refresh(self) -> int:
        """Re-read the transcripts now; returns how many turns were loaded.
        Blocking file I/O — async callers run it in a worker thread."""
        self.reload()
        return len(self._load())

    # ── views ─────────────────────────────────────────────────────────────────

    def days(self) -> list[date]:
        seen = {t.at.date() for t in self._load() if t.at}
        return sorted(seen)

    @staticmethod
    def _resolve_day(day: Any) -> date | None:
        if day is None or day == "":
            return None
        if isinstance(day, date) and not isinstance(day, datetime):
            return day
        token = str(day).strip().lower()
        today = datetime.now(timezone.utc).date()
        if token in {"today", "now"}:
            return today
        if token in {"yesterday"}:
            from datetime import timedelta
            return today - timedelta(days=1)
        try:
            return date.fromisoformat(token[:10])
        except ValueError:
            return None

    def timeline(self, day: Any = None, limit: int = 200) -> list[Turn]:
        turns = self._load()
        target = self._resolve_day(day)
        if target is not None:
            turns = [t for t in turns if t.at and t.at.date() == target]
        return turns[-max(1, limit):]

    def search(self, query: str, limit: int = 20) -> list[Turn]:
        terms = [w for w in re.findall(r"[a-z0-9]+", str(query or "").lower()) if len(w) > 1]
        if not terms:
            return []
        hits = []
        for t in self._load():
            low = t.content.lower()
            if all(term in low for term in terms):
                hits.append(t)
        hits.reverse()          # newest first
        return hits[: max(1, limit)]

    def relevant(self, query: str, limit: int = 12) -> list[Turn]:
        """Long-term recall (Mark XXIII): rank turns by RELEVANCE to the query,
        not strict keyword AND-matching.

        ``search`` needs every word present, so "what did we conclude about the
        exe icon" finds nothing unless one turn used all those words. This scores
        each turn by how many query terms it carries, weighted so a rare word
        (``icon``) counts for more than a common one (``about``), and returns the
        best matches even when no single turn is a perfect hit — which is what
        "what did we decide about X three weeks ago" actually needs."""
        terms = [w for w in re.findall(r"[a-z0-9]+", str(query or "").lower())
                 if len(w) > 1 and w not in _RECALL_STOP]
        if not terms:
            return []
        turns = self._load()
        # Document frequency for a light idf weight — rarer terms matter more.
        df: dict[str, int] = {}
        toksets = []
        for t in turns:
            toks = set(re.findall(r"[a-z0-9]+", t.content.lower()))
            toksets.append(toks)
            for term in set(terms):
                if term in toks:
                    df[term] = df.get(term, 0) + 1
        n = max(1, len(turns))
        scored: list[tuple[float, int, Turn]] = []
        for i, t in enumerate(turns):
            toks = toksets[i]
            score = 0.0
            for term in terms:
                if term in toks:
                    score += math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
            if score > 0:
                scored.append((score, i, t))
        # Best score first; recency breaks ties (higher index == later).
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [t for _s, _i, t in scored[: max(1, limit)]]

    def around(self, when: Any, window: int = 6) -> list[Turn]:
        target = _parse_at(when)
        turns = self._load()
        if target is None or not turns:
            return turns[-window:]
        # nearest turn by absolute time distance, then a window around it
        nearest = min(range(len(turns)),
                      key=lambda i: abs(((turns[i].at or target) - target).total_seconds()))
        lo = max(0, nearest - window // 2)
        return turns[lo: lo + window]

    def decisions(self, query: str = "", limit: int = 20) -> list[Turn]:
        terms = [w for w in re.findall(r"[a-z0-9]+", str(query or "").lower()) if len(w) > 1]
        hits = []
        for t in self._load():
            if not _DECISION_RE.search(t.content):
                continue
            if terms and not all(term in t.content.lower() for term in terms):
                continue
            hits.append(t)
        hits.reverse()
        return hits[: max(1, limit)]

    async def recap(self, day: Any = None,
                    generate: Callable[[str], Awaitable[str]] | None = None) -> str:
        turns = self.timeline(day, limit=1000)
        if not turns:
            target = self._resolve_day(day)
            when = target.isoformat() if target else "that period"
            return f"No conversation recorded for {when}."
        if generate is not None:
            body = "\n".join(t.line() for t in turns)[:8000]
            prompt = ("Recap this conversation in a few sentences — the topics "
                      "covered and anything decided:\n\n" + body + "\n\nRECAP:")
            try:
                out = str(await generate(prompt) or "").strip()
                if out:
                    return out
            except Exception:
                pass
        # extractive recap
        user_turns = [t for t in turns if t.is_user]
        decisions = self.decisions()  # across all; filter to these turns' day below
        target = self._resolve_day(day)
        if target is not None:
            decisions = [d for d in decisions if d.at and d.at.date() == target]
        span = f"{turns[0].clock} → {turns[-1].clock}"
        lines = [f"{len(turns)} turns ({span}). "
                 f"You spoke {len(user_turns)} time(s)."]
        if decisions:
            lines.append("")
            lines.append("Decisions:")
            for d in decisions[:8]:
                lines.append(f"  • [{d.clock}] {d.content[:160]}")
        return "\n".join(lines)


__all__ = ["RewindTimeline", "Turn"]
