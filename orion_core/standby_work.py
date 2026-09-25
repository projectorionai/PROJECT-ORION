"""
What ORION is still allowed to do while he is not listening to you.

  "When I tell ORION that I will need him to stop listening to me, he must go
   in standby mode - during this time he cannot listen to me, talk to me and is
   limited to research tasks, or any task that has been given to him (this will
   show in logs) ... He is still free to think at this time and autonomously
   improve himself."

Standby already closed the microphone and stopped the voice. What it did not
have was an *answer* — anywhere in the code — to "is this piece of work allowed
to run right now?". Background research, the forge and the improvement loop
simply kept going because nothing had been told to stop them, which happened to
be the desired behaviour but was an accident rather than a decision. And
nothing recorded what continued, so "this will show in logs" had nothing behind
it.

So permission is stated once, here, and the work that runs is written down.

The distinction that matters
----------------------------
Standby is about ATTENTION, not about activity. It means "stop directing things
at me" — not "stop working". Everything that reaches outward (speaking,
listening, volunteering) is closed; everything that runs inward (thinking,
researching, finishing what he was already asked to do, improving himself)
continues, because pausing it would waste exactly the quiet stretch that is
most useful for it.

The one exception is danger, which is not really an exception at all: a warning
about the machine or about a person is not ORION directing his attention at
you, it is him doing his job. It is handled by
``GenAILiveWorker.alert()`` and deliberately does not consult this module,
because a safety path that can be gated is not a safety path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Work(str, Enum):
    """Classes of activity, split by which direction they point.

    Inward work continues in standby; outward work does not.
    """

    # ── inward: continues ────────────────────────────────────────────────────
    RESEARCH = "research"               # explicitly named in the request
    ASSIGNED_TASK = "assigned_task"     # something already asked for
    THINKING = "thinking"               # the ambient thought stream
    SELF_IMPROVEMENT = "self_improvement"   # forge, self-repair, dependencies
    INGESTION = "ingestion"             # filing what he has learned
    MAINTENANCE = "maintenance"         # housekeeping, backups, indexing

    # ── outward: stops ───────────────────────────────────────────────────────
    LISTENING = "listening"
    PROACTIVE_SPEECH = "proactive_speech"   # reminders, suggestions, notices
    CONVERSATION = "conversation"
    BRIEFING = "briefing"

    # ── neither: always allowed ──────────────────────────────────────────────
    SAFETY = "safety"


#: Work that continues while ORION is in standby.
PERMITTED_IN_STANDBY: frozenset[Work] = frozenset({
    Work.RESEARCH, Work.ASSIGNED_TASK, Work.THINKING,
    Work.SELF_IMPROVEMENT, Work.INGESTION, Work.MAINTENANCE, Work.SAFETY,
})


def may_run(kind: Work | str, standby: bool) -> bool:
    """Whether *kind* of work may proceed. The single source of truth."""
    if not standby:
        return True
    if isinstance(kind, Work):
        work = kind
    else:
        try:
            # NOT Work(str(kind)): Work subclasses (str, Enum), so str() on a
            # member yields "Work.RESEARCH" rather than "research" — every
            # lookup raised ValueError and silently fell through to "stop",
            # which made standby look like it halted everything.
            work = Work(str(kind))
        except ValueError:
            # An unrecognised kind is treated as OUTWARD and stopped. Standby
            # is a promise to leave the user alone; honouring it for things
            # nobody classified is the safe direction to be wrong in.
            return False
    return work in PERMITTED_IN_STANDBY


@dataclass
class Entry:
    at: float
    kind: str
    detail: str

    def describe(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


@dataclass
class StandbyLedger:
    """What ORION did while you were not listening — so it shows in the logs.

    Kept as a ledger rather than only log lines because the useful moment for
    it is on WAKE: "while you were away I finished two research tasks and
    installed a missing dependency" is a sentence the user can act on, and it
    cannot be assembled from scrolled-past output.
    """

    #: Enough to summarise a long session without growing without bound.
    LIMIT: int = 200

    entries: list[Entry] = field(default_factory=list)
    blocked: dict[str, int] = field(default_factory=dict)
    since: float = 0.0

    def start(self) -> None:
        self.entries.clear()
        self.blocked.clear()
        self.since = time.time()

    def record(self, kind: Work | str, detail: str = "") -> None:
        self.entries.append(Entry(time.time(), str(getattr(kind, "value", kind)),
                                  str(detail)[:200]))
        if len(self.entries) > self.LIMIT:
            del self.entries[:len(self.entries) - self.LIMIT]

    def note_blocked(self, kind: Work | str) -> None:
        key = str(getattr(kind, "value", kind))
        self.blocked[key] = self.blocked.get(key, 0) + 1

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for entry in self.entries:
            tally[entry.kind] = tally.get(entry.kind, 0) + 1
        return tally

    def summary(self) -> str:
        """One sentence for ORION to say on waking, or "" if he did nothing.

        Silence when there is nothing to report: "while you were away I did
        nothing" is a sentence no one needs to hear.
        """
        tally = self.counts()
        if not tally:
            return ""
        # Singular and plural given explicitly. A naive +"s" produced
        # "2 researchs" — and a sentence ORION says out loud on every wake is
        # the last place to have a grammar bug.
        names = {
            "research": ("research task", "research tasks"),
            "assigned_task": ("task", "tasks"),
            "self_improvement": ("improvement", "improvements"),
            "ingestion": ("filing job", "filing jobs"),
            "thinking": ("train of thought", "trains of thought"),
            "maintenance": ("maintenance job", "maintenance jobs"),
            "safety": ("safety check", "safety checks"),
        }
        parts = []
        for kind, count in sorted(tally.items(), key=lambda item: -item[1]):
            singular, plural = names.get(
                kind, (kind.replace("_", " "), kind.replace("_", " ") + "s"))
            parts.append(f"{count} {singular if count == 1 else plural}")
        return "While you were focused I got through " + ", ".join(parts) + "."

    def report(self, limit: int = 12) -> str:
        """The detail, for the log and for 'what did you do?'."""
        if not self.entries:
            return "Nothing ran while you were in standby."
        lines = [f"  - {entry.describe()}" for entry in self.entries[-limit:]]
        head = f"Standby activity ({len(self.entries)} item(s)):"
        tail = ""
        if self.blocked:
            held = ", ".join(f"{k} x{v}" for k, v in sorted(self.blocked.items()))
            tail = f"\n  held back until you're free: {held}"
        return head + "\n" + "\n".join(lines) + tail


#: The live ledger. One per process; standby is a global state of ORION.
LEDGER = StandbyLedger()


def gate(kind: Work | str, standby: bool, detail: str = "") -> bool:
    """Ask permission AND record the answer. The call sites use this.

    Returning the decision rather than raising keeps every caller's control
    flow ordinary — a background loop that must skip a beat should skip it, not
    handle an exception.
    """
    allowed = may_run(kind, standby)
    if standby:
        if allowed:
            LEDGER.record(kind, detail)
        else:
            LEDGER.note_blocked(kind)
    return allowed


__all__ = ["LEDGER", "PERMITTED_IN_STANDBY", "Entry", "StandbyLedger", "Work",
           "gate", "may_run"]
