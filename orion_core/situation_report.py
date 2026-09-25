"""
Situation report (Mark XXIII, "Presence & Autonomy") — "catch me up."

The pieces to be a proactive assistant already exist separately: standing
questions know what's NEW on the topics you watch, Rewind knows what was DECIDED,
the sentinel knows what's WRONG. What was missing is the thing a real chief of
staff does — walk in and, in thirty seconds, tell you where everything stands,
worst first.

This is that synthesiser. It takes the raw signals — the deltas, the decisions,
the alerts, the open items — ranks them by how much they deserve your attention
right now, and renders one tight, sectioned brief. It is deliberately PURE and
injectable: it holds no clients and makes no calls, so the ranking and the
wording are tested exactly, and the tool feeds it live data from the standing
questions, the Rewind timeline and the sentinel.

The same report serves two masters: spoken proactively when something genuinely
changed (the "Presence" half of the mark), and produced on demand when you ask
"catch me up" or "where do things stand".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable


class Priority(IntEnum):
    """How much a line deserves attention right now. Higher speaks first."""
    ROUTINE = 0
    NOTABLE = 1
    IMPORTANT = 2
    URGENT = 3


class Category(str):
    ALERT = "alert"          # something is wrong / needs a decision
    WATCH = "watch"          # a delta on a standing question
    DECISION = "decision"    # something recently settled
    TASK = "task"            # an open/overdue item


# The order categories are presented in, and each one's floor priority.
_CATEGORY_ORDER = (Category.ALERT, Category.WATCH, Category.DECISION, Category.TASK)
_CATEGORY_LABEL = {
    Category.ALERT: "Needs your attention",
    Category.WATCH: "New on what you're watching",
    Category.DECISION: "Recently decided",
    Category.TASK: "Open items",
}


@dataclass
class SituationItem:
    category: str
    title: str
    priority: Priority = Priority.NOTABLE
    detail: str = ""
    source: str = ""

    def line(self) -> str:
        bullet = self.title.strip()
        if self.detail.strip():
            bullet += f" — {self.detail.strip()}"
        return bullet


@dataclass
class SituationReport:
    items: list[SituationItem] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.items

    @property
    def top_priority(self) -> Priority:
        return max((i.priority for i in self.items), default=Priority.ROUTINE)

    def worth_speaking(self, floor: Priority = Priority.IMPORTANT) -> bool:
        """Whether anything here is important enough to SAY unprompted, rather
        than just log — the gate for proactive delivery."""
        return self.top_priority >= floor

    def render(self, greeting: str = "") -> str:
        if self.is_empty:
            return (greeting + " " if greeting else "") + \
                "Nothing new needs your attention — all quiet."
        lines: list[str] = []
        if greeting:
            lines.append(greeting.strip())
        for category in _CATEGORY_ORDER:
            group = sorted((i for i in self.items if i.category == category),
                           key=lambda x: (-int(x.priority),))
            if not group:
                continue
            lines.append("")
            lines.append(f"{_CATEGORY_LABEL.get(category, category.title())}:")
            for item in group:
                mark = "⚠ " if item.priority >= Priority.URGENT else "• "
                lines.append(f"  {mark}{item.line()}")
        return "\n".join(lines).strip()

    def headline(self) -> str:
        """The single most important thing — for a one-line spoken lead."""
        if self.is_empty:
            return "All quiet — nothing new."
        top = max(self.items, key=lambda i: int(i.priority))
        return top.line()


def build_report(
    *,
    alerts: Iterable[Any] = (),
    watch_deltas: Iterable[Any] = (),
    decisions: Iterable[Any] = (),
    tasks: Iterable[Any] = (),
    max_per_section: int = 5,
) -> SituationReport:
    """Fuse the raw signals into one ranked report.

    Each input is an iterable of light dicts/strings so callers can feed it from
    anywhere without a shared type:
      * alerts       : {"text", "critical": bool} or a plain string
      * watch_deltas : {"question", "points": [str]} (from StandingQuestions)
      * decisions    : {"text"|"content", "when"} or a plain string
      * tasks        : {"text", "overdue": bool} or a plain string
    """
    report = SituationReport()

    def _text(entry: Any, *keys: str) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for k in keys:
                if entry.get(k):
                    return str(entry[k])
        return str(entry or "")

    for entry in list(alerts)[:max_per_section]:
        critical = isinstance(entry, dict) and bool(entry.get("critical"))
        report.items.append(SituationItem(
            Category.ALERT, _text(entry, "text", "message"),
            Priority.URGENT if critical else Priority.IMPORTANT, source="sentinel"))

    for delta in list(watch_deltas)[:max_per_section]:
        if isinstance(delta, dict):
            question = str(delta.get("question") or "")
            points = list(delta.get("points") or [])
        else:
            question, points = str(delta or ""), []
        if not points and not question:
            continue
        detail = ("; ".join(str(p) for p in points[:3])) if points else ""
        # More new points == more worth surfacing.
        pri = Priority.IMPORTANT if len(points) >= 3 else Priority.NOTABLE
        report.items.append(SituationItem(
            Category.WATCH, question or "watched topic", pri, detail, "standing_questions"))

    for entry in list(decisions)[:max_per_section]:
        when = entry.get("when", "") if isinstance(entry, dict) else ""
        report.items.append(SituationItem(
            Category.DECISION, _text(entry, "text", "content"),
            Priority.NOTABLE, str(when), "rewind"))

    for entry in list(tasks)[:max_per_section]:
        overdue = isinstance(entry, dict) and bool(entry.get("overdue"))
        report.items.append(SituationItem(
            Category.TASK, _text(entry, "text", "title"),
            Priority.IMPORTANT if overdue else Priority.ROUTINE,
            "overdue" if overdue else "", "tasks"))

    return report


__all__ = ["Priority", "Category", "SituationItem", "SituationReport",
           "build_report"]
