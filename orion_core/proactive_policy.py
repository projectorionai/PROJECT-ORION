"""
ProactivePolicy — the one gate every unprompted word passes through.

The complaint
-------------
"ORION speaks regarding processes which start up when I've said in the past I
don't want it announcing things which AREN'T dangerous or affect me in any way."

He was right to complain. ``SentinelAgent._check_new_processes()`` spoke aloud
whenever any unfamiliar process over 300 MB appeared — which is how a local
``llama-server.exe`` (ORION's *own* model server) got narrated. A new process
starting is not dangerous and does not affect the user. It is trivia.

The deeper problem was that there was no policy at all: nine different modules
each decided for themselves whether something was worth saying out loud, and
every one of them reached straight for ``bus.speak_request``. Adding a filter to
the sentinel alone would have left the other eight free to do the same thing
tomorrow.

The rule
--------
Unprompted speech has to earn itself. Every announcement declares an urgency and
the gate decides:

    CRITICAL    something is happening TO the user, now, and they must act.
                Battery about to die, drive full, a security event, ORION's own
                voice about to fail. Spoken even in do-not-disturb.

    ACTIONABLE  the user asked for it, or it changes what they can do next.
                Reminders, a protocol they started, a device they switched to.
                Spoken — unless they have asked for quiet.

    AMBIENT     true, mildly interesting, affects nothing. A process started.
                CPU was briefly busy. A feed came back. NEVER spoken, by
                anybody, ever. It goes to the log and the banner, where it is
                available without interrupting anyone.

Anything that cannot say why it is CRITICAL is not CRITICAL.

Do-not-disturb vs standby
-------------------------
These are different states and conflating them was the other half of the
problem:

    STANDBY   dormant. Microphone closed, nothing heard, nothing said.
              You have to wake him.

    FOCUS     present but busy. He still HEARS you and answers instantly when
              you speak to him — he simply stops volunteering. This is the
              state the user actually wanted when they said "he keeps pestering
              me when I'm there and I am busy".

FOCUS can be entered by asking, and it also arms itself: repeatedly ignoring
ORION's unprompted remarks is a clear signal, so after several in a row go
unanswered he stops offering them until the user next speaks to him.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class Urgency(IntEnum):
    """How much interrupting a person this is worth."""

    AMBIENT = 0       # never spoken
    ACTIONABLE = 1    # spoken unless the user asked for quiet
    CRITICAL = 2      # spoken regardless


class Attention(IntEnum):
    """What ORION is allowed to volunteer right now."""

    OPEN = 0        # normal: ambient still silent, everything else spoken
    FOCUS = 1       # present but busy: only CRITICAL
    STANDBY = 2     # dormant: only CRITICAL, and the UI is collapsed


#: Announcement kinds classified once, centrally, instead of at each call site.
#: A kind that is not listed defaults to ACTIONABLE — a new announcement has to
#: opt IN to being ambient, so nothing becomes silently unspeakable by accident,
#: but it also cannot become chatter without somebody choosing that.
URGENCY_BY_KIND: dict[str, Urgency] = {
    # ── never spoken ──────────────────────────────────────────────────────────
    "process_started": Urgency.AMBIENT,      # the actual complaint
    "process_ended": Urgency.AMBIENT,
    "cpu_high": Urgency.AMBIENT,
    "ram_high": Urgency.AMBIENT,
    "feed_recovered": Urgency.AMBIENT,
    "provider_switched": Urgency.AMBIENT,
    "tool_forged": Urgency.AMBIENT,
    "self_improvement": Urgency.AMBIENT,
    "index_built": Urgency.AMBIENT,
    "presence_check": Urgency.AMBIENT,       # "are you still there?" — pestering
    "focus_block": Urgency.AMBIENT,
    "network_mode": Urgency.AMBIENT,
    # ── worth saying, unless asked for quiet ─────────────────────────────────
    "reminder": Urgency.ACTIONABLE,
    "protocol": Urgency.ACTIONABLE,
    "briefing": Urgency.ACTIONABLE,
    "device_changed": Urgency.ACTIONABLE,
    "welcome_back": Urgency.ACTIONABLE,
    "task_due": Urgency.ACTIONABLE,
    # Mark XXVI — the learning loop. A block STARTING is ambient trivia
    # ("focus_block" above); a block having RUN OVER, or a review backlog, is
    # actionable — it changes what the user should do next.
    "focus_break_due": Urgency.ACTIONABLE,
    "study_due": Urgency.ACTIONABLE,
    # ── interrupts anything ───────────────────────────────────────────────────
    "battery_critical": Urgency.CRITICAL,
    "disk_full": Urgency.CRITICAL,
    "security_alert": Urgency.CRITICAL,
    "breach": Urgency.CRITICAL,
    "emergency": Urgency.CRITICAL,
    "audio_failure": Urgency.CRITICAL,
    "crash": Urgency.CRITICAL,
}


@dataclass
class Decision:
    speak: bool
    urgency: Urgency
    reason: str


@dataclass
class ProactivePolicy:
    """Decides whether an unprompted announcement is spoken, logged, or dropped."""

    #: Consecutive unanswered volunteered remarks before ORION takes the hint.
    IGNORED_BEFORE_FOCUS: int = 3
    #: FOCUS entered automatically expires after this, so he is not mute for ever.
    AUTO_FOCUS_SECONDS: float = 45 * 60.0

    bus: Any = None
    attention: Attention = Attention.OPEN
    _auto_focus_since: float = 0.0
    _unanswered: int = 0
    _spoken_at: float = 0.0
    _log: Callable[[str], None] | None = None
    suppressed: dict[str, int] = field(default_factory=dict)

    # ── state ─────────────────────────────────────────────────────────────────

    def set_attention(self, state: Attention, reason: str = "") -> None:
        if state is self.attention:
            return
        previous = self.attention
        self.attention = state
        self._auto_focus_since = 0.0
        if state is not Attention.OPEN:
            self._unanswered = 0
        self._emit(f"ATTENTION: {previous.name} -> {state.name}"
                   + (f" ({reason})" if reason else ""))

    def enter_focus(self, reason: str = "user asked") -> None:
        self.set_attention(Attention.FOCUS, reason)

    def leave_focus(self, reason: str = "user spoke") -> None:
        if self.attention is Attention.FOCUS:
            self.set_attention(Attention.OPEN, reason)

    def user_spoke(self) -> None:
        """Any direct address clears the auto-focus and the ignore counter.

        Someone talking to ORION is the strongest possible evidence that they
        are available to be talked to.
        """
        self._unanswered = 0
        if self.attention is Attention.FOCUS and self._auto_focus_since:
            self.set_attention(Attention.OPEN, "you spoke to me")

    def note_ignored(self) -> None:
        """A volunteered remark went unanswered.

        Three in a row is the user telling ORION something without saying it.
        """
        if self.attention is not Attention.OPEN:
            return
        self._unanswered += 1
        if self._unanswered >= self.IGNORED_BEFORE_FOCUS:
            self.set_attention(
                Attention.FOCUS,
                f"{self._unanswered} unanswered remarks — assuming you are busy")
            # AFTER set_attention, which clears the marker: this is what
            # distinguishes a focus ORION assumed from one the user asked for.
            # An assumed one is released the moment they speak; a requested one
            # is not, because they said they were busy and meant it.
            self._auto_focus_since = time.monotonic()

    def _expire_auto_focus(self) -> None:
        if (self.attention is Attention.FOCUS and self._auto_focus_since
                and time.monotonic() - self._auto_focus_since > self.AUTO_FOCUS_SECONDS):
            self.set_attention(Attention.OPEN, "auto-quiet expired")

    # ── the decision ──────────────────────────────────────────────────────────

    def classify(self, kind: str) -> Urgency:
        return URGENCY_BY_KIND.get(str(kind or "").strip().lower(),
                                   Urgency.ACTIONABLE)

    def should_speak(self, kind: str, urgency: Urgency | None = None) -> Decision:
        """Whether an announcement of *kind* may be spoken aloud right now."""
        self._expire_auto_focus()
        level = urgency if urgency is not None else self.classify(kind)

        if level <= Urgency.AMBIENT:
            self._note_suppressed(kind)
            return Decision(False, level,
                            "ambient — logged, never spoken: it is not dangerous "
                            "and does not affect you")

        if level >= Urgency.CRITICAL:
            return Decision(True, level, "critical — spoken regardless of state")

        if self.attention is Attention.STANDBY:
            self._note_suppressed(kind)
            return Decision(False, level, "in standby — dormant")

        if self.attention is Attention.FOCUS:
            self._note_suppressed(kind)
            return Decision(False, level, "you are busy — held until you speak")

        return Decision(True, level, "actionable and you are available")

    def _note_suppressed(self, kind: str) -> None:
        key = str(kind or "unknown")
        self.suppressed[key] = self.suppressed.get(key, 0) + 1

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
            return
        bus = self.bus
        if bus is not None:
            try:
                bus.log.emit(message)
            except Exception:
                pass

    # ── reporting ─────────────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        return {
            "attention": self.attention.name,
            "auto": bool(self._auto_focus_since),
            "unanswered": self._unanswered,
            "suppressed": dict(self.suppressed),
        }

    def report(self) -> str:
        lines = [f"Attention: {self.attention.name}"
                 + (" (entered automatically)" if self._auto_focus_since else "")]
        if self.suppressed:
            lines.append("Held back rather than spoken:")
            for kind, count in sorted(self.suppressed.items(),
                                      key=lambda kv: -kv[1]):
                level = self.classify(kind).name
                lines.append(f"  {count:>4} x {kind} ({level})")
        else:
            lines.append("Nothing has been held back this session.")
        return "\n".join(lines)


#: The shared policy. Import this; do not build a second one, or the two will
#: disagree about whether the user wants to be left alone.
POLICY = ProactivePolicy()


def should_speak(kind: str, urgency: Urgency | None = None) -> Decision:
    return POLICY.should_speak(kind, urgency)


__all__ = ["Attention", "Decision", "POLICY", "ProactivePolicy", "URGENCY_BY_KIND",
           "Urgency", "should_speak"]
