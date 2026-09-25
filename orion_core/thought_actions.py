"""
Acting on a thought without spending a token.

ORION thinks continuously — the thought stream narrates what he notices. Until
now that is *all* it did: he would observe that a write-ahead log had grown to
eight megabytes, say so, and leave it there. Noticing without acting is the
least useful half of attention.

The constraint is the interesting part of the design. Acting must cost NO
tokens, which rules out the obvious approach of handing the thought text back
to a model and asking "what should I do about this?". So nothing here reads a
thought's *words*.

Instead each action is keyed to a **fact ORION can measure himself** — a file
size, a flag, a count. That is a stronger design than parsing prose would have
been, for two reasons beyond the token budget:

  * a measurement cannot be hallucinated, and a thought can. Acting on
    generated text means one bad sentence becomes one unwanted action;
  * it is decidable. "Is this WAL over 4 MB?" has an answer; "does this
    sentence mean I should tidy up?" does not.

What may be automated here
--------------------------
Only work that is **safe, local, reversible and ORION's own housekeeping.**
Nothing that touches the user's files, spends money, sends anything, or is
hard to undo. A candidate action that cannot be described in one sentence
ending "...and if it was wrong, nothing is lost" does not belong here.

Everything is off unless a condition fires, every action is rate-limited so a
persistent condition cannot become a loop, and every outcome is announced on
the bus so the user can see what he did on his own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Fold a write-ahead log back into its database once it grows past this. The
#: number is a durability threshold, not a performance one — see
#: maintenance.checkpoint_all for why a large WAL is how ORION loses memory.
WAL_BYTES_BEFORE_CHECKPOINT = 4 * 1024 * 1024

#: The shortest gap between two runs of the SAME action. A condition that
#: stays true must not turn into a loop that runs every tick.
DEFAULT_COOLDOWN_SECONDS = 900.0


@dataclass(frozen=True)
class ThoughtAction:
    """Something ORION can do for himself, for free.

    ``condition`` returns a reason to act, or "" to stay put. Keeping the
    reason as its return value means the log says WHY he acted, not merely
    that he did.
    """

    name: str
    condition: Callable[[], str]
    act: Callable[[], str]
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS


@dataclass
class ActionOutcome:
    name: str
    reason: str
    detail: str
    ok: bool = True


@dataclass
class ThoughtActor:
    """Runs the token-free actions whose conditions are currently true."""

    bus: Any = None
    config_dir: Path = field(default_factory=lambda: Path("config"))
    housekeeper: Any = None
    _last_run: dict[str, float] = field(default_factory=dict)
    _actions: list[ThoughtAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._actions:
            self._actions = self._default_actions()

    # ── the catalogue ────────────────────────────────────────────────────────

    def _default_actions(self) -> list[ThoughtAction]:
        return [
            ThoughtAction(
                name="checkpoint-wal",
                condition=self._wal_is_large,
                act=self._checkpoint_wal,
            ),
        ]

    def _wal_files(self) -> list[Path]:
        try:
            return sorted(p for p in self.config_dir.glob("*.db-wal") if p.is_file())
        except Exception:
            return []

    def _wal_bytes(self) -> int:
        total = 0
        for path in self._wal_files():
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _wal_is_large(self) -> str:
        total = self._wal_bytes()
        if total < WAL_BYTES_BEFORE_CHECKPOINT:
            return ""
        return (f"{total / 1024 / 1024:.1f} MB of my memory is sitting in "
                f"write-ahead logs rather than in the databases themselves")

    def _checkpoint_wal(self) -> str:
        before = self._wal_bytes()
        housekeeper = self.housekeeper
        if housekeeper is None:
            from .bus import OrionBus
            from .maintenance import DatabaseHousekeeper

            housekeeper = DatabaseHousekeeper(self.bus or OrionBus(),
                                              config_dir=self.config_dir)
        count = housekeeper.checkpoint_all()
        after = self._wal_bytes()
        return (f"folded {max(0, before - after) / 1024 / 1024:.1f} MB back into "
                f"{count} database(s)")

    # ── running them ─────────────────────────────────────────────────────────

    def _ready(self, action: ThoughtAction, now: float) -> bool:
        last = self._last_run.get(action.name)
        return last is None or (now - last) >= action.cooldown_seconds

    def run_once(self, now: float | None = None) -> list[ActionOutcome]:
        """Act on every condition that is true and off cooldown.

        Never raises: this runs inside the thought loop, and a failed bit of
        self-maintenance must not stop ORION thinking.
        """
        moment = time.monotonic() if now is None else float(now)
        outcomes: list[ActionOutcome] = []
        for action in self._actions:
            if not self._ready(action, moment):
                continue
            try:
                reason = action.condition()
            except Exception:
                continue
            if not reason:
                continue
            self._last_run[action.name] = moment
            try:
                detail = action.act()
                outcome = ActionOutcome(action.name, reason, detail, ok=True)
            except Exception as exc:
                outcome = ActionOutcome(action.name, reason, str(exc)[:160], ok=False)
            outcomes.append(outcome)
            self._announce(outcome)
        return outcomes

    def _announce(self, outcome: ActionOutcome) -> None:
        """Say what he did, unprompted work being exactly the kind the user
        should be able to see afterwards."""
        if self.bus is None:
            return
        if outcome.ok:
            message = f"THOUGHT: {outcome.reason} — so I {outcome.detail}."
        else:
            message = (f"THOUGHT: {outcome.reason}, but I could not act on it "
                       f"({outcome.detail}).")
        try:
            self.bus.log.emit(message)
        except Exception:
            pass


__all__ = [
    "ActionOutcome", "ThoughtAction", "ThoughtActor",
    "DEFAULT_COOLDOWN_SECONDS", "WAL_BYTES_BEFORE_CHECKPOINT",
]
