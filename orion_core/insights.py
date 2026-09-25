"""
Cross-domain insights (Mark XXVI) — the patterns only ORION can see.

ORION quietly accumulates sixteen databases: focus blocks, study reviews,
wellbeing check-ins, decisions, spending, token use. Each system reports on
itself; **nothing ever looked across them.** That is a waste, because the
interesting questions are all cross-domain:

    "Do I actually focus better on nights I sleep seven hours?"
    "Does my recall drop in weeks I skip deep work?"
    "Is my spending correlated with the weeks I stop focusing?"

This engine builds day-indexed series from whatever stores exist, correlates
every meaningful pair, and reports the strong ones in plain language.

Three rules keep it honest:

  * **Correlation is not causation, and the wording never pretends otherwise.**
    Findings are phrased as "on days when X was higher, Y tended to be…".
  * **Thin evidence is refused, not padded.** A pair needs ``MIN_DAYS`` overlapping
    days and a coefficient past ``MIN_R`` before it is reported at all.
  * **Every store is optional.** A missing database drops one series, never the
    report — which is what makes this safe to run on a fresh install.

Pure arithmetic (it reuses the same Pearson helper the wellbeing engine uses),
fully offline, and no model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from .wellbeing import pearson

#: Overlapping days a pair needs before it is worth reporting.
MIN_DAYS = 5

#: |r| below this is noise for data this small.
MIN_R = 0.45

#: Series worth correlating, and how to say them out loud.
LABELS: dict[str, str] = {
    "focus_minutes": "minutes of deep focus",
    "focus_completed": "focus blocks completed",
    "interruptions": "distractions logged",
    "study_reviews": "flashcards reviewed",
    "study_accuracy": "recall accuracy",
    "energy": "your energy",
    "sleep_hours": "hours slept",
    "mood": "your mood",
    "stress": "your stress",
    "spend": "money spent",
}

#: Pairs where a link is plausible and actionable. Correlating everything with
#: everything invites nonsense ("spending correlates with mood on Tuesdays"), so
#: the search space is deliberately curated.
PAIRS: tuple[tuple[str, str], ...] = (
    ("sleep_hours", "focus_minutes"),
    ("sleep_hours", "study_accuracy"),
    ("sleep_hours", "energy"),
    ("energy", "focus_minutes"),
    ("energy", "focus_completed"),
    ("energy", "study_accuracy"),
    ("stress", "focus_minutes"),
    ("stress", "interruptions"),
    ("stress", "energy"),
    ("mood", "focus_minutes"),
    ("focus_minutes", "study_reviews"),
    ("focus_minutes", "spend"),
    ("interruptions", "focus_completed"),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_day(text: str) -> date | None:
    try:
        stamp = datetime.fromisoformat(str(text))
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).date()
    except (TypeError, ValueError):
        return None


@dataclass
class Finding:
    """One observed relationship, with the honesty attached."""
    left: str
    right: str
    coefficient: float
    days: int

    @property
    def strength(self) -> str:
        magnitude = abs(self.coefficient)
        if magnitude >= 0.8:
            return "strong"
        if magnitude >= 0.6:
            return "clear"
        return "modest"

    def sentence(self) -> str:
        left = LABELS.get(self.left, self.left)
        right = LABELS.get(self.right, self.right)
        direction = "higher" if self.coefficient > 0 else "lower"
        # "more your energy" reads badly; a possessive label takes "higher" instead.
        opening = (f"On days when {left} was higher" if left.startswith("your ")
                   else f"On days with more {left}")
        return (f"{opening}, {right} tended to be {direction} "
                f"({self.strength}, r={self.coefficient:+.2f}, {self.days} days).")


# ── series builders (each guarded; a missing store drops one series) ─────────

def _focus_series(store: Any, since: datetime) -> dict[str, dict[date, float]]:
    minutes: dict[date, float] = {}
    completed: dict[date, float] = {}
    interruptions: dict[date, float] = {}
    for session in store.all():
        day = _parse_day(session.started_at)
        if day is None or not session.ended_at:
            continue
        minutes[day] = minutes.get(day, 0.0) + session.elapsed_minutes()
        completed[day] = completed.get(day, 0.0) + (1.0 if session.completed else 0.0)
        interruptions[day] = interruptions.get(day, 0.0) + float(session.interruptions)
    return {"focus_minutes": minutes, "focus_completed": completed,
            "interruptions": interruptions}


def _wellbeing_series(store: Any, since: datetime) -> dict[str, dict[date, float]]:
    out: dict[str, dict[date, list[float]]] = {
        "energy": {}, "sleep_hours": {}, "mood": {}, "stress": {}}
    for checkin in store.since(since):
        day = _parse_day(checkin.at)
        if day is None:
            continue
        for name, value in (("energy", checkin.energy),
                            ("sleep_hours", checkin.sleep_hours),
                            ("mood", checkin.mood_valence),
                            ("stress", checkin.stress)):
            if value is not None:
                out[name].setdefault(day, []).append(float(value))
    return {name: {day: sum(v) / len(v) for day, v in series.items()}
            for name, series in out.items()}


def _study_series(store: Any, since: datetime) -> dict[str, dict[date, float]]:
    """Reviews per day and the day's accuracy, read from the review log."""
    reviews: dict[date, float] = {}
    passes: dict[date, list[float]] = {}
    try:
        rows = store._db.execute(
            "SELECT at, quality FROM reviews WHERE at>=?",
            (since.isoformat(),)).fetchall()
    except Exception:
        return {}
    for row in rows:
        day = _parse_day(row["at"])
        if day is None:
            continue
        reviews[day] = reviews.get(day, 0.0) + 1.0
        passes.setdefault(day, []).append(1.0 if row["quality"] >= 3 else 0.0)
    accuracy = {day: 100.0 * sum(v) / len(v) for day, v in passes.items()}
    return {"study_reviews": reviews, "study_accuracy": accuracy}


def _finance_series(store: Any, since: datetime) -> dict[str, dict[date, float]]:
    spend: dict[date, float] = {}
    for txn in store.transactions(since=since):
        if txn.direction != "out":
            continue
        day = _parse_day(txn.at)
        if day is None:
            continue
        spend[day] = spend.get(day, 0.0) + float(txn.amount)
    return {"spend": spend}


def collect_series(*, focus: Any = None, wellbeing: Any = None, study: Any = None,
                   finance: Any = None, days: int = 60,
                   now: datetime | None = None) -> dict[str, dict[date, float]]:
    """Build every available day-indexed series. Each source is independently
    guarded, so one broken store costs one series rather than the report."""
    since = (now or _now()) - timedelta(days=max(7, int(days)))
    series: dict[str, dict[date, float]] = {}
    for source, builder in ((focus, _focus_series), (wellbeing, _wellbeing_series),
                            (study, _study_series), (finance, _finance_series)):
        if source is None:
            continue
        try:
            store = getattr(source, "store", source)
            series.update(builder(store, since))
        except Exception:
            continue
    return {name: data for name, data in series.items() if data}


def find_patterns(series: dict[str, dict[date, float]], *,
                  min_days: int = MIN_DAYS, min_r: float = MIN_R) -> list[Finding]:
    """Correlate the curated pairs and keep only what clears the evidence bar."""
    findings: list[Finding] = []
    for left, right in PAIRS:
        a, b = series.get(left), series.get(right)
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        if len(common) < min_days:
            continue
        coefficient = pearson([a[d] for d in common], [b[d] for d in common])
        if coefficient is None or abs(coefficient) < min_r:
            continue
        findings.append(Finding(left, right, coefficient, len(common)))
    findings.sort(key=lambda f: -abs(f.coefficient))
    return findings


def render(findings: list[Finding], series: dict[str, dict[date, float]]) -> str:
    """The report — including an honest answer when there is nothing to say."""
    tracked = ", ".join(sorted(LABELS.get(k, k) for k in series)) or "nothing yet"
    if not findings:
        return ("No pattern is strong enough to report yet.\n"
                f"I am tracking: {tracked}.\n"
                "Keep logging focus blocks, study reviews and check-ins — patterns "
                "need a couple of weeks of overlapping days before they mean anything.")
    lines = [f"{len(findings)} pattern(s) worth noticing:"]
    lines += [f"  • {finding.sentence()}" for finding in findings[:6]]
    lines.append("")
    lines.append("These are correlations, not proof of cause — but they are drawn "
                 "from your own days, not a general study.")
    return "\n".join(lines)


__all__ = [
    "MIN_DAYS", "MIN_R", "LABELS", "PAIRS", "Finding", "collect_series",
    "find_patterns", "render",
]


# ── daily rhythm: when does the work actually land? ──────────────────────────
#
# The day-level correlations above answer "what goes with what". This answers a
# different and more actionable question: "WHEN am I actually any good?" It reads
# the hour each focus block STARTED and how it ended, so the answer comes from
# behaviour rather than from what anyone believes about themselves.

#: A block of hours needs at least this many sessions before it is worth naming.
MIN_SESSIONS = 3

#: Named parts of the day (start hour inclusive, end hour exclusive).
DAYPARTS: tuple[tuple[str, int, int], ...] = (
    ("early morning", 5, 9),
    ("morning", 9, 12),
    ("afternoon", 12, 17),
    ("evening", 17, 21),
    ("night", 21, 29),          # 29 wraps to 05:00 — handled by _part_of_day
)


def _part_of_day(hour: int) -> str:
    for name, start, end in DAYPARTS:
        if start <= hour < end or (end > 24 and (hour >= start or hour < end - 24)):
            return name
    return "night"


@dataclass
class DaypartProfile:
    """What actually happens when work is attempted at this time of day."""
    part: str
    sessions: int
    completed: int
    minutes: float
    interruptions: int

    @property
    def completion_rate(self) -> int:
        return round(100 * self.completed / self.sessions) if self.sessions else 0

    @property
    def average_minutes(self) -> int:
        return round(self.minutes / self.sessions) if self.sessions else 0

    def sentence(self) -> str:
        return (f"{self.part}: {self.completion_rate}% of {self.sessions} block(s) "
                f"finished, {self.average_minutes} min average"
                + (f", {self.interruptions} distraction(s)" if self.interruptions else ""))


def daily_rhythm(focus: Any, *, days: int = 90,
                 now: datetime | None = None) -> list[DaypartProfile]:
    """Group focus attempts by part of day. Empty when there is no focus history."""
    since = (now or _now()) - timedelta(days=max(7, int(days)))
    buckets: dict[str, dict[str, float]] = {}
    try:
        store = getattr(focus, "store", focus)
        sessions = store.all()
    except Exception:
        return []
    for session in sessions:
        if not session.started_at:
            continue
        try:
            started = datetime.fromisoformat(session.started_at)
            started = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if started < since:
            continue
        bucket = buckets.setdefault(_part_of_day(started.hour),
                                    {"sessions": 0, "completed": 0,
                                     "minutes": 0.0, "interruptions": 0})
        bucket["sessions"] += 1
        bucket["completed"] += 1 if session.completed else 0
        bucket["interruptions"] += float(session.interruptions)
        if session.ended_at:
            bucket["minutes"] += session.elapsed_minutes()
    profiles = [
        DaypartProfile(part, int(v["sessions"]), int(v["completed"]),
                       v["minutes"], int(v["interruptions"]))
        for part, v in buckets.items()
    ]
    order = [name for name, _s, _e in DAYPARTS]
    profiles.sort(key=lambda p: order.index(p.part) if p.part in order else 99)
    return profiles


def render_rhythm(profiles: list[DaypartProfile]) -> str:
    """The rhythm report — and an honest refusal when the evidence is thin."""
    usable = [p for p in profiles if p.sessions >= MIN_SESSIONS]
    if not profiles:
        return ("No focus history yet — start a few blocks and I can tell you when "
                "you actually work best.")
    lines = ["How your focus lands across the day:"]
    lines += [f"  • {p.sentence()}" for p in profiles]
    if not usable:
        lines.append("")
        lines.append(f"Too few blocks in any one part of the day to call a pattern "
                     f"yet (I want {MIN_SESSIONS}+ in a slot).")
        return "\n".join(lines)
    best = max(usable, key=lambda p: (p.completion_rate, p.average_minutes))
    worst = min(usable, key=lambda p: (p.completion_rate, p.average_minutes))
    lines.append("")
    if best.part == worst.part or best.completion_rate - worst.completion_rate < 15:
        lines.append("Your focus holds up fairly evenly whenever you start.")
    else:
        lines.append(f"You finish what you start in the {best.part} "
                     f"({best.completion_rate}%) far more than in the {worst.part} "
                     f"({worst.completion_rate}%) — worth putting the hard work "
                     f"in the {best.part}.")
    return "\n".join(lines)
