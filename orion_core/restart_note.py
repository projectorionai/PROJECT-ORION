"""
"I've restarted" - a restart remembered across the process that ends.

A restart is a clean shutdown and then a fresh process (app.main respawns it),
and the fresh process knew nothing of the old one: it greeted like a first
launch, with the time of day and an offer of the morning briefing. The owner
(26 Oct 2026): when ORION restarts he should remember, and tell him he has
restarted, instead of the normal greeting.

The ending process writes a note (CONFIG_DIR/restart_note.json) when the
restart is asked for; the new one takes it - reads and deletes it - once, at
start, and says it is back (and how long the restart took) in place of the
greeting and the briefing offer. A note is honoured only while it is fresh
(MAX_AGE_S): a restart that never came back - the successor refused to start,
the PC went down - cannot make a launch hours later claim to be a restart.

Mark XXXIX (owner, 27 Oct 2026: "when ORION restarts he must remember that
he's just restarted and not come out with the same greetings"). The note only
covered "restart yourself"; the owner's other restarts - shut down, update,
start again; close the window and reopen it - still opened with "Good
evening! Welcome back..." and the weather. So every CLEAN exit now leaves a
second note (CONFIG_DIR/last_exit.json: when, why, which Mark), and a start
reads it as one of three returns:

  * ``restart``  - asked for (the restart note): "I'm back", no greeting;
  * ``relaunch`` - a start within QUICK_RELAUNCH_S of a clean shut-down: a
    short "back so soon" in place of the greeting and the briefing offer;
  * ``update``   - the Mark changed since the last exit: the usual greeting,
    which also says - in a few words - which Mark he is running now.

And he remembers what he actually SAID when he came back (``note_said``,
CONFIG_DIR/greeting_memory.json): the composed example greeting had been kept,
not his own words, so nothing told him he had opened with "Good morning...
you'll want a coat" five starts running. Each opening now carries his last few
real ones, to be said differently.
"""

from __future__ import annotations

import json
import re
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .atomic_io import atomic_write_text
from .constants import APP_MARK, CONFIG_DIR

NOTE_PATH = CONFIG_DIR / "restart_note.json"
#: Written by every clean exit, taken by the next start.
EXIT_PATH = CONFIG_DIR / "last_exit.json"
#: The Mark of the last start (kept, never taken): an update is known even
#: after an exit that left no note - a crash, a forced close, or the first
#: start after a build that predates the exit note.
MARK_PATH = CONFIG_DIR / "last_mark.json"
#: Proof an older ORION has run on this PC (his identity file).
EVIDENCE_PATH = CONFIG_DIR / "identity.json"
#: What he said when he came back, per kind of return.
SAID_PATH = CONFIG_DIR / "greeting_memory.json"
#: A note older than this is from a restart that did not come straight back.
MAX_AGE_S = 600.0
#: A restart that took longer than this is not worth timing out loud.
SAY_SECONDS_UNDER = 120.0
#: A start this soon after a clean shut-down is a return, not a new day.
QUICK_RELAUNCH_S = 20 * 60.0
#: An update is mentioned on the first start of the new Mark within this long.
UPDATE_MENTION_S = 7 * 24 * 3600.0
#: Openings remembered per kind (and shown to the model).
SAID_KEEP = 6

KINDS = ("restart", "relaunch", "update")


@dataclass(frozen=True)
class Restart:
    """How this process came to be running: the return the previous one left."""

    at: float                       # wall-clock time it was asked for / it exited
    kind: str = "restart"           # restart | relaunch | update
    previous_mark: str = ""         # the Mark that exited (when it is known)
    current_mark: str = APP_MARK

    def took_s(self, now: float | None = None) -> float:
        return max(0.0, (time.time() if now is None else float(now)) - self.at)

    def took_phrase(self, now: float | None = None) -> str:
        """'9 seconds', '1 second', or '' when too long (or too odd) to say."""
        seconds = round(self.took_s(now))
        if not 0 < seconds < SAY_SECONDS_UNDER:
            return ""
        return "1 second" if seconds == 1 else f"{seconds} seconds"

    def gap_phrase(self, now: float | None = None) -> str:
        """'4 minutes', 'a minute', 'under a minute' - how long he was off."""
        seconds = self.took_s(now)
        if seconds < 60:
            return "under a minute"
        minutes = round(seconds / 60.0)
        if minutes < 60:
            return "a minute" if minutes == 1 else f"{minutes} minutes"
        hours = round(seconds / 3600.0)
        return "an hour" if hours == 1 else f"{hours} hours"

    @property
    def updated(self) -> bool:
        if self.kind == "update":
            return True
        return bool(self.previous_mark) and not same_mark(self.previous_mark, self.current_mark)


def _write(target: Path, data: dict[str, Any]) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps(data), encoding="utf-8")
        return True
    except Exception:
        return False


def _take_json(target: Path) -> dict[str, Any] | None:
    """Read *target* and delete it whatever it says (honoured at most once)."""
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        data = None
    try:
        target.unlink()
    except Exception:
        pass                        # absent, or locked: the age check still holds
    return data if isinstance(data, dict) else None


def _at(data: dict[str, Any] | None) -> float | None:
    if not data:
        return None
    try:
        return float(data.get("at"))
    except (TypeError, ValueError):
        return None


def remember(*, now: float | None = None, path: Path | None = None,
             mark: str = APP_MARK) -> bool:
    """The ending process: note that a restart was asked for. False (and
    nothing else) when it cannot be written - the restart still happens, it
    is just greeted like a launch."""
    note = {"schema": "orion.restart_note.v1", "mark": str(mark or ""),
            "at": time.time() if now is None else float(now)}
    return _write(Path(path or NOTE_PATH), note)


def remember_exit(reason: str = "shutdown", *, now: float | None = None,
                  path: Path | None = None, mark: str = APP_MARK) -> bool:
    """Every clean exit: when, why, and which Mark - so the next start knows a
    quick return or an update from a new day."""
    note = {"schema": "orion.last_exit.v1", "reason": str(reason or "shutdown"),
            "mark": str(mark or ""), "at": time.time() if now is None else float(now)}
    return _write(Path(path or EXIT_PATH), note)


def take(*, now: float | None = None, path: Path | None = None) -> Restart | None:
    """The new process: the restart that started it, or None for a launch.
    The note is deleted whatever it says, so it is honoured at most once."""
    data = _take_json(Path(path or NOTE_PATH))
    at = _at(data)
    if at is None:
        return None
    current = time.time() if now is None else float(now)
    if not -60.0 <= current - at <= MAX_AGE_S:      # stale, or a clock jump
        return None
    return Restart(at, "restart", str((data or {}).get("mark") or ""))


def _last_mark(path: Path) -> str | None:
    """The Mark recorded at the last start, '' if unreadable, None if absent."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return ""
    return str(data.get("mark") or "") if isinstance(data, dict) else ""


def take_return(*, now: float | None = None, path: Path | None = None,
                exit_path: Path | None = None, current_mark: str = APP_MARK,
                mark_path: Path | None = None, evidence_path: Path | None = None
                ) -> Restart | None:
    """How this start came about: a restart, a quick relaunch, an update - or
    None for an ordinary launch. Both notes are taken (deleted) every start,
    and this start's Mark is recorded for the next one."""
    current = time.time() if now is None else float(now)
    restart = take(now=current, path=path)
    last_exit = _take_json(Path(exit_path or EXIT_PATH))
    marks = Path(mark_path or MARK_PATH)
    started_as = _last_mark(marks)
    _write(marks, {"schema": "orion.last_mark.v1", "mark": current_mark, "at": current})
    exit_at = _at(last_exit)
    exit_mark = str((last_exit or {}).get("mark") or "")
    previous = exit_mark or (started_as or "")
    if restart is not None:
        return Restart(restart.at, "restart", restart.previous_mark or previous, current_mark)
    if exit_at is not None:
        gap = current - exit_at
        if gap < -60.0:                              # a clock jump
            return None
        if previous and not same_mark(previous, current_mark) and gap <= UPDATE_MENTION_S:
            return Restart(exit_at, "update", previous, current_mark)
        if gap <= QUICK_RELAUNCH_S:
            return Restart(exit_at, "relaunch", previous, current_mark)
        return None
    # No exit note: the last run crashed or was forced closed, or it predates
    # exit notes. A different Mark than last time is still an update.
    if started_as and not same_mark(started_as, current_mark):
        return Restart(current, "update", started_as, current_mark)
    if started_as is None:
        try:
            older = Path(evidence_path or EVIDENCE_PATH).is_file()
        except OSError:
            older = False
        if older:                                    # first start of a build that keeps marks
            return Restart(current, "update", "", current_mark)
    return None


# ── what he said when he came back ──────────────────────────────────────────

_SAID_LOCK = threading.Lock()


def _said(path: Path | None = None) -> dict[str, list[str]]:
    try:
        data = json.loads(Path(path or SAID_PATH).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(x)[:400] for x in v if isinstance(x, str)]
            for k, v in data.items() if isinstance(v, list)}


def note_said(kind: str, text: str, *, path: Path | None = None) -> bool:
    """Remember an opening he actually said ('greeting', 'restart', ...).
    Blocking file I/O: call it off the event loop."""
    line = " ".join(str(text or "").split())[:400]
    if not line or not kind:
        return False
    with _SAID_LOCK:
        data = _said(path)
        lines = [x for x in data.get(kind, []) if x != line]
        data[kind] = [*lines, line][-SAID_KEEP:]
        return _write(Path(path or SAID_PATH), data)


def recent_said(kind: str, *, path: Path | None = None, limit: int = 4) -> list[str]:
    """His last few real openings of this kind, oldest first."""
    return _said(path).get(kind, [])[-max(0, int(limit)):]


# ── what he says ─────────────────────────────────────────────────────────────

_LINES_TIMED = (
    "I'm back - that restart took {took}.",
    "Restart complete. I'm back with you; it took {took}.",
    "Back again - restarted in {took}.",
    "Back in {took}. Did you miss me?",
    "And we're back. {took}, door to door.",
)
_LINES = (
    "I'm back - restart complete.",
    "Restart complete. I'm back with you.",
    "Back again - I've restarted and I'm ready.",
    "Back, and none the worse for it.",
    "Restarted and present. Where were we?",
)
_RELAUNCH_LINES = (
    "Back so soon? I'm here.",
    "And I'm back. That was a short break.",
    "Hello again - that didn't take long.",
    "Back with you. I hadn't gone far.",
)


def spoken_line(restart: Restart, *, now: float | None = None,
                choose=random.choice, recent: Sequence[str] = ()) -> str:
    """What he says when there is no live model to say it in his own words -
    never one of the lines he used on his last returns while another is left."""
    took = restart.took_phrase(now)
    if getattr(restart, "kind", "restart") == "relaunch":
        pool = list(_RELAUNCH_LINES)
    else:
        pool = [line.format(took=took) for line in _LINES_TIMED] if took else list(_LINES)
    said = {" ".join(str(x).split()).lower() for x in recent}
    fresh = [line for line in pool if " ".join(line.split()).lower() not in said]
    return choose(fresh or pool)


_ONES = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TENS = ("", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def spoken_mark(mark: str) -> str:
    """'Mark XXXX' as a voice should say it: 'Mark forty'. An offline voice read
    the numeral letter by letter ('Mark ex ell')."""
    match = re.fullmatch(r"\s*Mark\s+([IVXLC]+)\s*", str(mark or ""))
    if not match:
        return str(mark or "")
    total, previous = 0, 0
    for letter in reversed(match.group(1)):
        value = _ROMAN[letter]
        total += -value if value < previous else value
        previous = max(previous, value)
    if not 0 < total < 100:
        return str(mark)
    if total < 10:
        words = _ONES[total]
    elif total < 20:
        words = _TEENS[total - 10]
    else:
        words = _TENS[total // 10] + (f"-{_ONES[total % 10]}" if total % 10 else "")
    return f"Mark {words}"


def same_mark(first: str, second: str) -> bool:
    """One mark written two ways ('Mark XL' and 'Mark XXXX' are both forty)
    is not an update (28 Oct 2026: the owner's numeral replaced XL)."""
    return spoken_mark(first) == spoken_mark(second)


def _avoid(recent: Sequence[str]) -> str:
    lines = [str(x) for x in recent if str(x).strip()]
    if not lines:
        return ""
    return (" Your last openings on a return like this were: "
            + json.dumps(lines, ensure_ascii=False)
            + " - say it differently: new words, a new shape.")


def instruction(restart: Restart, *, now: float | None = None,
                recent: Sequence[str] = ()) -> str:
    """The live model's turn: say it is back, nothing else."""
    kind = getattr(restart, "kind", "restart")
    if kind == "relaunch":
        gap = restart.gap_phrase(now)
        return ("You were closed " + gap + " ago and have just been started again - a "
                "quick return, not a new day. In one short, natural sentence, say you're "
                "back (a dry word about the short break is welcome). Do NOT give the usual "
                "time-of-day greeting, do NOT mention the weather, do NOT offer or mention "
                "the briefing." + _avoid(recent) + " Then stand by.")
    took = restart.took_phrase(now)
    timing = (f" The restart took {took}; you may say so, using exactly that figure."
              if took else "")
    updated = (f" You are now running {restart.current_mark}"
               + (f" (it was {restart.previous_mark})" if restart.previous_mark else "")
               + "; say so in a few words."
               if getattr(restart, "updated", False) else "")
    return ("You have just restarted and are back. In one short, natural sentence, "
            "tell the user that you've restarted and you're back." + timing + updated +
            " A dry touch is welcome. Do NOT give the usual time-of-day greeting, do NOT "
            "offer or mention the briefing, and do not describe what changed unless asked."
            + _avoid(recent) + " Then stand by.")


def update_line(restart: Restart) -> str:
    """The extra line for the launch greeting after an update."""
    if not getattr(restart, "updated", False):
        return ""
    was = f" (you were {restart.previous_mark})" if restart.previous_mark else ""
    said = spoken_mark(restart.current_mark)
    aloud = f" (say it '{said}', never the letters)" if said != restart.current_mark else ""
    return (f" You have just been updated: this is your first start as "
            f"{restart.current_mark}{aloud}{was}. Mention it in a few words - a dry touch is "
            "welcome - and describe what changed only if asked.")


def update_spoken(restart: Restart) -> str:
    """The offline greeting's extra sentence after an update."""
    if not getattr(restart, "updated", False):
        return ""
    return f"I'm running {spoken_mark(restart.current_mark)} now."


__all__ = ["EXIT_PATH", "KINDS", "MARK_PATH", "MAX_AGE_S", "NOTE_PATH", "QUICK_RELAUNCH_S", "Restart",
           "SAID_PATH", "instruction", "note_said", "recent_said", "remember",
           "remember_exit", "spoken_line", "take", "take_return", "update_line",
           "update_spoken"]
