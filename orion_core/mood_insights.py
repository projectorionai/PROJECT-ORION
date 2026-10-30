"""
How the owner has been, over days - the emotional_insight_generator plugin,
rebuilt into ORION (Mark XXXIX).

The plugin (archived 26 Oct 2026 as "the built-in emotion engine reads and
shows feelings") promised "analyzes user emotions over time and provides
actionable insights". The built-in engine only ever read the moment: what was
just said, how it sounded. Nothing looked back across a week.

This does, and stores nothing new to do it: the owner's own turns are already
in the episodic memory, and ``emotion.SentimentAnalyser`` - the offline,
deterministic scorer the face already uses on his words - labels each one
again on demand. From those labels it reports what a friend would notice:
the feeling that came up most, when in the day the hard ones cluster, the
best and the hardest day, and one practical suggestion drawn from the
pattern. Every figure is a count of real turns; a quiet week says so rather
than inventing a mood.

Blocking (SQLite read + regex scoring of a few thousand short lines):
the emotion tool runs it off the event loop.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

#: Sentiment labels that are a feeling worth counting (the rest - neutral,
#: confident, analytical - are the ordinary business of a conversation).
#: "urgent" is NOT one: it fires on the subject ("overdue", "failure",
#: "offline") - measured on the owner's real week, 13 of 20 "hard" turns were
#: him describing a bug to ORION, not how he felt.
FEELINGS: dict[str, str] = {
    "celebratory": "upbeat", "excitement": "excited", "amusement": "amused",
    "gratitude": "grateful", "affection": "warm",
    "empathetic_concern": "stressed or worried", "sadness": "low",
    "frustration": "frustrated", "tiredness": "tired", "insult": "cross",
}
HARD = frozenset({"empathetic_concern", "sadness", "frustration", "tiredness", "insult"})
GOOD = frozenset({"celebratory", "excitement", "amusement", "gratitude", "affection"})
#: Below this confidence a label is a guess, and is not counted.
MIN_CONFIDENCE = 0.3
#: Fewer feeling-bearing turns than this is not a pattern.
MIN_FEELINGS = 4
MAX_DAYS = 90

_PARTS = (("night", range(0, 5)), ("morning", range(5, 12)),
          ("afternoon", range(12, 17)), ("evening", range(17, 24)))


def _part_of_day(hour: int) -> str:
    for name, hours in _PARTS:
        if hour in hours:
            return name
    return "evening"


def _parse(stamp: Any) -> datetime | None:
    text = str(stamp or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone()                     # local time: "evenings" are his


@dataclass
class MoodReport:
    days: int
    turns: int = 0
    feelings: Counter = field(default_factory=Counter)
    hard_by_part: Counter = field(default_factory=Counter)
    by_day: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    @property
    def counted(self) -> int:
        return sum(self.feelings.values())

    def text(self) -> str:
        span = "today" if self.days <= 1 else f"the last {self.days} days"
        if self.turns == 0:
            return f"I have nothing you've said to me in {span}, so no read on your mood."
        if self.counted < MIN_FEELINGS:
            return (f"Over {span} you've said {self.turns} things to me and almost none "
                    "carried a clear feeling - an even keel, or simply all business.")
        top = self.feelings.most_common(3)
        lead = ", ".join(f"{FEELINGS[k]} ({n})" for k, n in top)
        hard = sum(n for k, n in self.feelings.items() if k in HARD)
        good = sum(n for k, n in self.feelings.items() if k in GOOD)
        lines = [f"Over {span}, from {self.turns} things you said: mostly {lead}."]
        lines.append(f"Good moments {good}, harder ones {hard}.")
        if hard >= 3 and self.hard_by_part:
            part, n = self.hard_by_part.most_common(1)[0]
            share = round(100 * n / max(1, hard))
            if share >= 40:
                lines.append(f"The harder ones cluster in the {part} ({share}% of them).")
        scored = {day: sum(c[k] for k in GOOD) - sum(c[k] for k in HARD)
                  for day, c in self.by_day.items() if sum(c.values())}
        if len(scored) >= 2:
            best = max(scored, key=scored.get)
            worst = min(scored, key=scored.get)
            if best != worst:
                lines.append(f"Best day: {best}. Hardest: {worst}.")
        lines.append(self._suggestion(hard, good))
        return " ".join(lines)

    def _suggestion(self, hard: int, good: int) -> str:
        if hard == 0:
            return "Nothing to fix there - carry on."
        tired = self.feelings.get("tiredness", 0)
        stressed = self.feelings.get("empathetic_concern", 0)
        frustrated = self.feelings.get("frustration", 0) + self.feelings.get("insult", 0)
        late = self.hard_by_part.get("night", 0)
        worst_part = self.hard_by_part.most_common(1)[0][0] if self.hard_by_part else ""
        if late >= 2 and worst_part == "night":
            return ("Suggestion: the small hours are where it goes wrong - an earlier "
                    "finish would do more than any app.")
        if tired >= max(stressed, frustrated) and tired >= 2:
            return "Suggestion: you sound tired more than anything - protect your sleep this week."
        if stressed >= frustrated and stressed >= 2:
            return ("Suggestion: put the deadlines in one list and let me pace them - "
                    "pressure shrinks when it's written down.")
        if frustrated >= 2:
            return ("Suggestion: the frustration tends to come with things not working - "
                    "tell me which, and I'll take the fiddly part.")
        return "Suggestion: more of whatever made the good days good."


def analyse(turns: Iterable[tuple[str, Any]], *, days: int = 7,
            classify: Any = None, now: datetime | None = None) -> MoodReport:
    """A report from (text, created_at) user turns. *classify* (text ->
    (label, confidence)) defaults to emotion.SentimentAnalyser.classify."""
    if classify is None:
        from .emotion import SentimentAnalyser
        classify = SentimentAnalyser.classify
    days = max(1, min(MAX_DAYS, int(days or 7)))
    current = (now or datetime.now(timezone.utc)).astimezone()
    since = current - timedelta(days=days)
    report = MoodReport(days=days)
    for text, stamp in turns:
        moment = _parse(stamp)
        if moment is None or moment < since or moment > current + timedelta(minutes=5):
            continue
        words = str(text or "").strip()
        if not words:
            continue
        report.turns += 1
        try:
            label, confidence = classify(words)
        except Exception:
            continue
        if label not in FEELINGS or float(confidence) < MIN_CONFIDENCE:
            continue
        report.feelings[label] += 1
        report.by_day[moment.strftime("%A %d %B").replace(" 0", " ")][label] += 1
        if label in HARD:
            report.hard_by_part[_part_of_day(moment.hour)] += 1
    return report


def user_turns(memory: Any, days: int) -> list[tuple[str, str]]:
    """The owner's own turns from the last *days* days, from the episodic store."""
    matrix = getattr(memory, "matrix", memory)
    since = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days))))
    stamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    reader = getattr(matrix, "user_turns_since", None)
    if callable(reader):
        return list(reader(stamp))
    return []


def insights(memory: Any, days: int = 7) -> str:
    """The spoken answer to "how have I been this week?"."""
    if memory is None:
        return "My memory isn't available just now, so I can't look back at your week."
    return analyse(user_turns(memory, days), days=days).text()


__all__ = ["FEELINGS", "MoodReport", "analyse", "insights", "user_turns"]
