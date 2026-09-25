"""
ProactivityEngine (Mark XXVI, Phase 2) — turn the learning loop from reactive to
proactive, without becoming a pest.

ORION already surfaces flashcards-due and an over-running focus block *when asked*
(``catch_up``). This closes the loop: a background survey that volunteers those
two things aloud at the right moment, and stays quiet otherwise.

It does NOT invent a new speech path or a new permission model — both already
exist and are load-bearing:

  * every unprompted word goes through the shared ``proactive_policy.POLICY`` gate
    (``should_speak``), which already silences AMBIENT chatter, holds ACTIONABLE
    remarks while the user is in FOCUS/STANDBY, and arms auto-quiet after remarks
    are ignored; and
  * approved speech is emitted on ``bus.speak_request``, which ``live_worker``'s
    ``announce()`` speaks through whatever channel is live — Gemini Live when
    connected, the offline local TTS otherwise. So proactivity is offline-capable
    for free.

The decision is a pure function (``gather`` + ``plan``) over a snapshot and a
cooldown ledger, so it is fully unit-testable with no timers, no audio and no Qt.
Ships behind ``ORION_PROACTIVE_ENGINE`` (default off): the loop is not started
unless the flag is set, so wiring it in changes nothing until it is switched on.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .proactive_policy import POLICY, Decision, Urgency
from .utils import first_line

#: Only nudge about a review backlog once it is worth interrupting for.
STUDY_DUE_THRESHOLD = 15

#: A literal double quote, built rather than escaped, so the
#: nudge text below stays readable.
Q = chr(34)
#: Per-kind cooldowns — the debounce that stops a true condition being repeated
#: every survey. A break-due block is re-offered every few minutes; a review
#: backlog only every few hours.
FOCUS_BREAK_COOLDOWN_S = 300.0
STUDY_DUE_COOLDOWN_S = 3 * 3600.0
#: Offering a shortcut is a courtesy, not a reminder. Once a day at
#: most, and only ever as AMBIENT, so it can never interrupt anything.
REFLEX_OFFER_COOLDOWN_S = 24 * 3600.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ProactiveSnapshot:
    """The learning-loop state a survey reads, as plain data."""
    focus_break_due: bool = False
    focus_label: str = ""
    focus_overrun_min: float = 0.0
    study_due: int = 0
    #: A phrase the user keeps saying that always reaches the same read-only
    #: tool. Offering to answer it instantly is the visible half of reflex
    #: learning: without it ORION would accumulate candidates forever and never
    #: actually get faster. See orion_core/reflex_learning.
    reflex_phrase: str = ""
    reflex_hits: int = 0


@dataclass(frozen=True)
class Nudge:
    kind: str
    urgency: Urgency
    text: str
    cooldown_s: float


@dataclass
class Plan:
    """What a tick decided: the (coalesced) thing to say, and the accounting."""
    utterance: str = ""
    spoken_kinds: list[str] = field(default_factory=list)
    held_kinds: list[str] = field(default_factory=list)


# ── pure decision core ────────────────────────────────────────────────────────

def gather(snapshot: ProactiveSnapshot) -> list[Nudge]:
    """The candidate nudges a snapshot warrants, before any gate or cooldown."""
    nudges: list[Nudge] = []
    if snapshot.focus_break_due:
        label = f' "{snapshot.focus_label}"' if snapshot.focus_label else ""
        nudges.append(Nudge(
            "focus_break_due", Urgency.ACTIONABLE,
            f"Your focus block{label} is up — a good moment to stop and take your break.",
            FOCUS_BREAK_COOLDOWN_S))
    if snapshot.reflex_phrase:
        quoted = Q + snapshot.reflex_phrase + Q
        nudges.append(Nudge(
            "reflex_offer", Urgency.AMBIENT,
            f"You have asked me {quoted} {snapshot.reflex_hits} times now "
            "and it always goes to the same place. Say the word and I will "
            "answer that one instantly from now on.",
            REFLEX_OFFER_COOLDOWN_S))
    if snapshot.study_due >= STUDY_DUE_THRESHOLD:
        nudges.append(Nudge(
            "study_due", Urgency.ACTIONABLE,
            f"You have {snapshot.study_due} flashcards due for review whenever you're ready.",
            STUDY_DUE_COOLDOWN_S))
    return nudges


def _coalesce(texts: list[str]) -> str:
    """One spoken remark, not a volley. Multiple approved nudges become a single
    utterance so ORION never machine-guns the user."""
    return " ".join(t.strip() for t in texts if t.strip())


def plan(
    nudges: list[Nudge],
    *,
    clock: float,
    last_spoken: dict[str, float],
    should_speak: Callable[[str, Urgency], Decision],
) -> Plan:
    """Apply per-kind cooldown, then the shared speech gate, then coalesce.

    Pure: no I/O. ``clock`` is a monotonic-seconds value; ``last_spoken`` maps a
    kind to the clock value it was last spoken at (not mutated here — the caller
    records only what actually went out)."""
    speak_texts: list[str] = []
    spoken: list[str] = []
    held: list[str] = []
    for n in nudges:
        if clock - last_spoken.get(n.kind, float("-inf")) < n.cooldown_s:
            continue                                  # cooling down — silent skip
        decision = should_speak(n.kind, n.urgency)
        if decision.speak:
            speak_texts.append(n.text)
            spoken.append(n.kind)
        else:
            held.append(n.kind)                       # gate held it (FOCUS/quiet)
    return Plan(_coalesce(speak_texts), spoken, held)


# ── the engine (thin IO shell around the pure core) ──────────────────────────

class ProactivityEngine:
    """Survey the learning loop and volunteer what's worth interrupting for."""

    DEFAULT_INTERVAL_S = 60.0

    def __init__(
        self,
        bus: Any,
        *,
        focus: Any = None,
        study: Any = None,
        policy: Any = None,
        interval_s: float | None = None,
    ) -> None:
        self.bus = bus
        self.focus = focus
        self.study = study
        # Optional: set by the worker once its learner exists. Kept as a plain
        # attribute rather than a constructor argument so every existing caller
        # keeps working unchanged.
        self.reflex_learner: Any = None
        self.policy = policy or POLICY
        self.interval_s = float(interval_s or self.DEFAULT_INTERVAL_S)
        self._last_spoken: dict[str, float] = {}
        self._speaking = False
        if bus is not None:
            try:
                bus.speaking.connect(self._on_speaking)
            except Exception:
                pass

    def _on_speaking(self, active: bool) -> None:
        # Etiquette: never begin a proactive remark over ORION's own voice.
        self._speaking = bool(active)

    def snapshot(self, wall_now: datetime | None = None) -> ProactiveSnapshot:
        wall_now = wall_now or _now()
        snap = ProactiveSnapshot()
        try:
            session = self.focus.active(wall_now) if self.focus is not None else None
            if session is not None and session.is_break_due(wall_now):
                snap.focus_break_due = True
                snap.focus_label = session.label
                snap.focus_overrun_min = session.elapsed_minutes(wall_now) - session.planned_minutes
        except Exception:
            pass
        try:
            if self.study is not None:
                snap.study_due = len(self.study.due(limit=10_000, now=wall_now))
        except Exception:
            pass
        try:
            learner = self.reflex_learner
            if learner is not None:
                candidates = learner.candidates()
                if candidates:
                    snap.reflex_phrase = candidates[0].phrase
                    snap.reflex_hits = candidates[0].hits
        except Exception:
            pass
        return snap

    def tick(self, *, clock: float | None = None,
             wall_now: datetime | None = None) -> Plan:
        clock = time.monotonic() if clock is None else clock
        result = plan(
            gather(self.snapshot(wall_now)),
            clock=clock, last_spoken=self._last_spoken,
            should_speak=self.policy.should_speak)
        if result.utterance and not self._speaking:
            self._emit(result.utterance)
            for kind in result.spoken_kinds:
                self._last_spoken[kind] = clock
        # else: held by the gate, cooling down, or ORION is mid-utterance — the
        # condition will still be true next survey, so nothing is lost.
        return result

    def _emit(self, text: str) -> None:
        if self.bus is None:
            return
        try:
            self.bus.speak_request.emit(text)      # → announce() → live or offline TTS
        except Exception:
            pass

    async def run(self) -> None:
        while True:
            try:
                self.tick()
            except Exception as exc:
                if self.bus is not None:
                    try:
                        self.bus.log.emit(f"PROACTIVITY: survey recovered - {first_line(exc)}")
                    except Exception:
                        pass
            await asyncio.sleep(self.interval_s)


def engine_enabled() -> bool:
    """Whether the proactive engine should run. Off by default — volunteering
    speech is a behavioural change the user opts into."""
    return os.getenv("ORION_PROACTIVE_ENGINE", "").strip().lower() in {"1", "true", "on", "yes"}


__all__ = [
    "STUDY_DUE_THRESHOLD", "FOCUS_BREAK_COOLDOWN_S", "STUDY_DUE_COOLDOWN_S",
    "REFLEX_OFFER_COOLDOWN_S",
    "ProactiveSnapshot", "Nudge", "Plan", "gather", "plan", "ProactivityEngine",
    "engine_enabled",
]
