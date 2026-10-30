"""
Rewards and mistakes - ORION learning from how each turn actually went.

Until now ORION learned from every tool call that *returned* ok: the intent
network (intent_brain) took a lesson, the learned-reflex ledger
(reflex_learning) counted a hit. Whether the owner was pleased, or said "no, I
meant Spotify", changed nothing. This module closes that loop.

**A turn is judged by what the owner says next.**

* Praise right after it ("perfect", "thanks", "well done", "that's right") is a
  REWARD: counted, logged, and taught to the intent network a second time, so
  a praised route weighs double.
* Moving on to something else (or saying nothing for WINDOW_S) is a quieter
  reward: the turn worked and nobody objected.
* A correction ("no, I meant ...", "that's wrong", "not that") is a PENALTY:
  counted, and the phrase is poisoned in the reflex ledger, so it can never
  become (or stay) an instant shortcut. When the retry that follows works, the
  corrected route is rewarded and, if the retry used a DIFFERENT tool, the
  network gets both lessons: these words did not mean the old tool (a negative
  lesson, intent_brain.unlearn_now) and did mean the new one. A retry with the
  same tool was a wrong detail, not a wrong route, so routing is left alone.
* A failed tool result is a penalty too. It is not a routing lesson: a forecast
  that fails to load says nothing about which tool "what's the weather" means.
* "no", "stop", "never mind" decide nothing - no reward, no penalty.

**Effortless repeats.** A read-only, action-only route rewarded
REWARDS_TO_PROMOTE times, with no penalty, that reflex_learning itself calls
eligible (read-only, unanimous, never failed) is OFFERED as an instant shortcut,
once: a note asks the model to offer it and, only on the owner's yes, to call
diagnostics promote=... (the explicit path, which re-checks eligibility).
Promotion stays a person's act - a learned reflex bypasses the model entirely
(tests/test_reflex_offer.py holds that line); the rewards decide when to ask.
A multi-step turn that was rewarded has its tool sequence remembered, and the
next time the same words come, the model is given it as a short context hint
("last time this worked: ..."). Nothing is replayed: actions with side effects
always go through the model and its guards.

Counts persist in CONFIG_DIR/reward_learning.db (SQLite, WAL). Everything here
is bookkeeping on the turn path and never raises into a turn.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .constants import CONFIG_DIR
from .db import apply_pragmas
from .reflex_learning import MAX_PHRASE_LEN, normalise

DEFAULT_PATH = CONFIG_DIR / "reward_learning.db"

#: Praise or a correction this long after a turn is about it; silence this long
#: means it worked.
WINDOW_S = 120.0
#: A spoken turn's transcript is stored after its own tool call: words that
#: arrive this soon and match are that same turn, not a new request.
SAME_TURN_S = 60.0
#: Rewards (praise or uncorrected success) before a read-only route becomes an
#: instant reflex. reflex_learning still needs its own three unanimous hits.
REWARDS_TO_PROMOTE = 3
MAX_STEPS = 8
MAX_EVENTS = 5000
#: Words this similar to the last request, said within the window, are the
#: owner asking again - the last answer is not rewarded.
REPHRASE_OVERLAP = 0.6

REWARD_KINDS = ("praise", "implicit", "corrected_route", "retry")
PENALTY_KINDS = ("corrected", "failed")
_COLUMNS = {"praise": ("rewards", "praise"), "implicit": ("rewards",),
            "corrected_route": ("rewards",), "retry": ("rewards",),
            "corrected": ("penalties", "corrections"), "failed": ("penalties", "failures")}
_NOT_STEPS = frozenset({"", "find_tool", "use_tool"})

# ── judging what the owner said ──────────────────────────────────────────────

_PRAISE_WORDS = (
    r"perfect|thanks|thank you|thankyou|cheers|well done|good job|great job|nice one|"
    r"nice work|good work|great work|nice|great|brilliant|excellent|awesome|amazing|"
    r"lovely|fantastic|superb|spot on|exactly|exactly right|that's right|thats right|"
    r"that is right|that's it|thats it|that's correct|that is correct|correct|"
    r"that works|that worked|it works|it worked|you got it|much better|good stuff|"
    r"great stuff|legend|appreciate it|love it|very good|nailed it|you nailed it|"
    r"that's perfect|that's great|that's brilliant|that's better|good boy|ta")
_PRAISE = re.compile(
    r"(?:(?:oh|ok|okay|yes|yeah|yep|ah|wow|orion|hey|right)\s+)*(?:" + _PRAISE_WORDS + r")"
    r"(?:\s+(?:orion|mate|sir|buddy|man|bro|very much|so much|a lot|again|" + _PRAISE_WORDS + r"))*")
#: Decides nothing: a bare no is usually an answer to "anything else?".
_NEUTRAL = re.compile(
    r"(?:(?:no|nope|nah)(?:\s+(?:thanks|thank you|need|i'm good|im good|i'm fine|that's all|"
    r"thats all|that's it|thats it|that's fine|thats fine|it's fine|its fine|all good|"
    r"not now|not right now|later|maybe later|orion|mate|sir))*"
    r"|not now|not right now|maybe later|stop|stop it|cancel|cancel that|never mind|"
    r"nevermind|forget it|leave it|hold on|wait|quiet|shut up)(?:\s+(?:orion|please|now))*")
_WRONG_ANYWHERE = re.compile(
    r"\b(?:that's|thats|that is|it's|its) (?:wrong|not right|not it|not what i)\b"
    r"|\bnot what i (?:asked|meant|wanted|said)\b|\byou got (?:it|that) wrong\b"
    r"|\bwrong (?:one|app|tool|thing|answer|place|window|song|file|room|person)\b")
_CLAUSE = re.compile(r"[,.;:!?]|\s(?:and|but)\s")
_FALLBACK_CORRECTION = re.compile(
    r"(?i)^(?:no+[,.!]?\s|wrong\b|not that\b|i meant\b|try again\b|actually,? i\b)")


def _starts_as_correction(text: str) -> bool:
    try:
        # One definition of "no, I meant ..." for training and for live use.
        from .training.sources import is_correction
        return is_correction(text)
    except Exception:
        return bool(_FALLBACK_CORRECTION.match(text))


def judge(text: str) -> str:
    """'praise', 'correction', 'neutral' or 'other' for the owner's words."""
    raw = " ".join(str(text or "").split())
    phrase = normalise(raw)
    if not phrase:
        return "other"
    if _NEUTRAL.fullmatch(phrase):
        return "neutral"
    if _starts_as_correction(raw) or _WRONG_ANYWHERE.search(phrase):
        return "correction"
    first = normalise(_CLAUSE.split(raw, 1)[0])
    if first and _PRAISE.fullmatch(first):
        return "praise"
    return "other"


def _same_words(a: str, b: str) -> bool:
    """One utterance, possibly a partial transcript of the other (a prefix of
    two words or more: "thanks" is not the start of every later sentence)."""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return len(short.split()) >= 2 and long_.startswith(short + " ")


def _rephrased(a: str, b: str) -> bool:
    wa, wb = set(a.split()), set(b.split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= REPHRASE_OVERLAP


_FILLER = frozenset("a an the to my me please can could would will you i it that this for of "
                    "on in at and is are what what's whats do does now just orion no not "
                    "meant mean said asked wanted actually sorry wrong one".split())


def _shares_content(a: str, b: str) -> bool:
    """A retry names something the corrected request named."""
    return bool((set(a.split()) - _FILLER) & (set(b.split()) - _FILLER))


def _plain_call(args: dict[str, Any]) -> bool:
    """True when a learned reflex (which sends only the action) would make
    exactly this call. "weather in Paris" must not become a reflex that
    forgets Paris."""
    return all(key == "action" or value in (None, "", [], {}) for key, value in args.items())


def _action(args: dict[str, Any] | None) -> str:
    return str((args or {}).get("action") or "").strip().lower()   # as reflex_learning keys it


def hint_text(phrase: str, steps: list[str]) -> str:
    return (f"[Learned from experience: the last time you were asked {phrase!r}, this worked: "
            f"{' -> '.join(steps)}. Use it as a guide if it still fits, check each step, and "
            "do not repeat any action with side effects that this request does not ask for. "
            "Context only; do not reply to this note.]")


def offer_text(phrase: str, route: str, rewards: int) -> str:
    return (f"[Learned: {phrase!r} has gone to {route} {rewards} times and the owner was happy "
            "each time. At a natural pause, offer once, in one short sentence, to answer it "
            "instantly from now on. Only if they say yes, call diagnostics with "
            f"action='reflexes' and promote={phrase!r}. Do not bring it up again.]")


# ── the persistent counts ─────────────────────────────────────────────────────

class RewardStore:
    """Rewards and penalties per (phrase, tool, action), an event log for the
    trend, and the tool sequences of rewarded multi-step requests."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS routes (
                phrase TEXT NOT NULL, tool TEXT NOT NULL, action TEXT NOT NULL DEFAULT '',
                rewards INTEGER NOT NULL DEFAULT 0, praise INTEGER NOT NULL DEFAULT 0,
                penalties INTEGER NOT NULL DEFAULT 0, corrections INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0, last_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (phrase, tool, action));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, kind TEXT NOT NULL,
                phrase TEXT NOT NULL, tool TEXT NOT NULL DEFAULT '', action TEXT NOT NULL DEFAULT '');
            CREATE INDEX IF NOT EXISTS idx_reward_events_at ON events(at);
            CREATE TABLE IF NOT EXISTS sequences (
                phrase TEXT PRIMARY KEY, steps TEXT NOT NULL,
                times INTEGER NOT NULL DEFAULT 1, last_at REAL NOT NULL DEFAULT 0);
            """)
        self._db.commit()
        self._lock = threading.Lock()
        self._writes = 0

    def record(self, kind: str, phrase: str, tool: str = "", action: str = "",
               at: float | None = None) -> dict[str, int]:
        """Count one reward/penalty (or a 'promoted' event); the route's new counts."""
        at = time.time() if at is None else float(at)
        with self._lock:
            columns = _COLUMNS.get(kind, ())
            if columns and tool:
                self._db.execute("INSERT OR IGNORE INTO routes (phrase, tool, action) VALUES (?, ?, ?)",
                                 (phrase, tool, action))
                sets = ", ".join(f"{c} = {c} + 1" for c in columns)   # fixed names, never input
                self._db.execute(f"UPDATE routes SET {sets}, last_at = ? "
                                 "WHERE phrase = ? AND tool = ? AND action = ?",
                                 (at, phrase, tool, action))
            self._db.execute("INSERT INTO events (at, kind, phrase, tool, action) VALUES (?, ?, ?, ?, ?)",
                             (at, kind, phrase, tool, action))
            self._writes += 1
            if self._writes % 200 == 0:
                self._db.execute("DELETE FROM events WHERE id <= "
                                 "(SELECT MAX(id) FROM events) - ?", (MAX_EVENTS,))
            self._db.commit()
        return self.counts(phrase, tool, action)

    def counts(self, phrase: str, tool: str, action: str = "") -> dict[str, int]:
        row = self._db.execute(
            "SELECT rewards, praise, penalties, corrections, failures FROM routes "
            "WHERE phrase = ? AND tool = ? AND action = ?", (phrase, tool, action)).fetchone()
        keys = ("rewards", "praise", "penalties", "corrections", "failures")
        return {k: int(row[k]) for k in keys} if row else dict.fromkeys(keys, 0)

    def offered(self, phrase: str) -> bool:
        """An instant shortcut for *phrase* has been offered already."""
        return self._db.execute("SELECT 1 FROM events WHERE kind = 'offered' AND phrase = ? "
                                "LIMIT 1", (phrase,)).fetchone() is not None

    def remember_sequence(self, phrase: str, steps: list[str], at: float | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO sequences (phrase, steps, times, last_at) VALUES (?, ?, 1, ?) "
                "ON CONFLICT(phrase) DO UPDATE SET times = CASE WHEN steps = excluded.steps "
                "THEN times + 1 ELSE 1 END, steps = excluded.steps, last_at = excluded.last_at",
                (phrase, json.dumps(list(steps)[:MAX_STEPS]), time.time() if at is None else at))
            self._db.commit()

    def sequence(self, phrase: str) -> list[str]:
        row = self._db.execute("SELECT steps FROM sequences WHERE phrase = ?", (phrase,)).fetchone()
        try:
            return [str(s) for s in json.loads(row["steps"])] if row else []
        except (TypeError, ValueError):
            return []

    def forget_sequence(self, phrase: str) -> bool:
        with self._lock:
            cursor = self._db.execute("DELETE FROM sequences WHERE phrase = ?", (phrase,))
            self._db.commit()
        return cursor.rowcount > 0

    def _window(self, start: float, end: float) -> dict[str, int]:
        marks = ",".join("?" * len(REWARD_KINDS))
        bad = ",".join("?" * len(PENALTY_KINDS))
        row = self._db.execute(
            f"SELECT SUM(kind IN ({marks})) AS good, SUM(kind IN ({bad})) AS bad "
            "FROM events WHERE at >= ? AND at < ?",
            (*REWARD_KINDS, *PENALTY_KINDS, start, end)).fetchone()
        return {"rewards": int(row["good"] or 0), "penalties": int(row["bad"] or 0)}

    def stats(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        row = self._db.execute(
            "SELECT COUNT(*) AS routes, SUM(rewards) AS rewards, SUM(praise) AS praise, "
            "SUM(penalties) AS penalties, SUM(corrections) AS corrections, "
            "SUM(failures) AS failures FROM routes").fetchone()
        out: dict[str, Any] = {k: int(row[k] or 0) for k in
                               ("routes", "rewards", "praise", "penalties", "corrections", "failures")}
        out["offered"] = int(self._db.execute(
            "SELECT COUNT(DISTINCT phrase) FROM events WHERE kind = 'offered'").fetchone()[0])
        out["sequences"] = int(self._db.execute("SELECT COUNT(*) FROM sequences").fetchone()[0])
        week = 7 * 86400.0
        out["week"] = self._window(now - week, now + 1)
        out["prior_week"] = self._window(now - 2 * week, now - week)
        return out

    def top(self, limit: int = 3) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT phrase, tool, action, rewards FROM routes WHERE penalties = 0 AND rewards > 0 "
            "ORDER BY rewards DESC, last_at DESC LIMIT ?", (limit,)).fetchall()

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


# ── judging turns ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    utterance: str
    phrase: str
    tool: str = ""
    action: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    at: float = 0.0
    request_id: int = 0
    heard: bool = False        # its words have been seen (typed: before; spoken: after)
    credited: bool = False     # already rewarded (a corrected retry)
    via: str = "tool"          # tool | reflex | turn (several tools, no single route)

    @property
    def route(self) -> str:
        return self.tool + (f".{self.action}" if self.action else "")


class RewardLearner:
    """Judges each turn by what the owner says next and teaches from it.

    *teach(utterance, tool, positive)* gives the intent network a lesson,
    *reflexes()* returns the ReflexLearner (or None), *steps_for(request_id)*
    returns a Live turn's tool calls as (name, action, ok), *hint(text)* puts a
    note in the model's context, *log(text)* writes the session log."""

    def __init__(self, store: RewardStore | None = None, *,
                 teach: Callable[[str, str, bool], Any] | None = None,
                 reflexes: Callable[[], Any] | None = None,
                 steps_for: Callable[[int], list[tuple[str, str, Any]]] | None = None,
                 hint: Callable[[str], Any] | None = None,
                 log: Callable[[str], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store if store is not None else RewardStore()
        self._teach, self._reflexes, self._steps_for = teach, reflexes, steps_for
        self._hint, self._log, self._clock = hint, log, clock
        self._lock = threading.RLock()
        self.pending: Turn | None = None
        self.corrected: Turn | None = None
        self._last_phrase = ""
        self._last_at = -1e18
        self._last_verdict = "other"
        self._hinted = ""

    # ── the two inputs ────────────────────────────────────────────────────────

    def note_utterance(self, text: str, *, request_id: int = 0,
                       reflex: tuple[str, dict[str, Any]] | None = None,
                       voice: bool = False) -> str:
        """The owner said *text*. Judges the previous turn by it. *reflex* is
        the learned reflex (tool, args) that will answer it, if one will;
        *request_id* a spoken turn's id, so its tool sequence can be read."""
        phrase = normalise(text)
        if not phrase or len(phrase) > 4 * MAX_PHRASE_LEN:
            return "other"
        with self._lock:
            now = self._clock()
            pending = self.pending
            if (pending is not None and not pending.heard and now - pending.at <= SAME_TURN_S
                    and _same_words(phrase, pending.phrase)):
                pending.heard = True       # a spoken turn's transcript, after its tool call
                return "same"
            if now - self._last_at <= SAME_TURN_S and _same_words(phrase, self._last_phrase):
                return "same"              # the same words stored twice
            verdict = self._judge_and_settle(text, phrase, now)
            if verdict in ("other", "neutral"):
                if reflex is not None:
                    tool, args = reflex
                    self.pending = Turn(text, phrase, tool, _action(args), dict(args or {}),
                                        True, now, heard=True, via="reflex")
                elif request_id:
                    self.pending = Turn(text, phrase, at=now, request_id=int(request_id),
                                        heard=True, via="turn")
        if not voice:
            self._offer_hint(phrase)
        return verdict

    def note_result(self, utterance: str, tool: str, args: dict[str, Any] | None = None,
                    ok: bool = True, *, request_id: int = 0, live: bool | None = None) -> None:
        """A turn's (single) tool call ended. *live* False means the words are
        not the turn in progress: a long job ending after the talk moved on."""
        phrase = normalise(utterance)
        if not phrase or not tool:
            return
        fresh = False
        with self._lock:
            now = self._clock()
            current = now - self._last_at <= WINDOW_S and _same_words(phrase, self._last_phrase)
            if not current:
                if live is False:
                    if not ok:             # it finally failed: still a penalty
                        self._failed(Turn(utterance, phrase, tool, _action(args), dict(args or {}),
                                          False, now))
                    return
                fresh = True               # spoken: the tool call came before the transcript
                verdict = self._judge_and_settle(utterance, phrase, now)
            else:
                verdict = self._last_verdict
            turn = Turn(utterance, phrase, tool, _action(args), dict(args or {}), bool(ok),
                        now, int(request_id or 0), heard=current)
            fixed, self.corrected = self.corrected, None
            if (fixed is not None and now - fixed.at <= WINDOW_S
                    and (verdict == "correction" or _shares_content(phrase, fixed.phrase))):
                self._retry(fixed, turn)
            else:
                if not turn.ok:
                    self._failed(turn)
                self.pending = turn
        if fresh:
            self._offer_hint(phrase)

    # ── judging ───────────────────────────────────────────────────────────────

    def _judge_and_settle(self, text: str, phrase: str, now: float) -> str:
        verdict = judge(text)
        self._last_phrase, self._last_at, self._last_verdict = phrase, now, verdict
        if self.corrected is not None and now - self.corrected.at > WINDOW_S:
            self.corrected = None
        turn, self.pending = self.pending, None
        if turn is None:
            return verdict
        if now - turn.at > WINDOW_S:
            self._settle(turn, "implicit")        # nobody objected in time
        elif verdict == "praise":
            self._settle(turn, "praise")
        elif verdict == "correction":
            self._penalise(turn, now)
        elif verdict == "other" and not (phrase == turn.phrase or _rephrased(phrase, turn.phrase)):
            self._settle(turn, "implicit")        # moved on: it worked
        return verdict

    def _settle(self, turn: Turn, kind: str) -> None:
        if not turn.ok:
            return
        calls = self._calls(turn)
        if len(calls) >= 2:
            # Several steps: the method is what worked, not its first tool, so
            # it is remembered as a method and never made a one-tool reflex.
            if all(ok is True for *_, ok in calls) and len(calls) <= MAX_STEPS:
                steps = [n + (f".{a}" if a else "") for n, a, _ in calls]
                counts = self.store.record(kind, turn.phrase, "multi-step")
                self.store.remember_sequence(turn.phrase, steps)
                self._say(f"LEARN: rewarded the {len(steps)}-step method for {turn.phrase!r} "
                          f"({counts['rewards']} time{'s' if counts['rewards'] != 1 else ''}); "
                          f"remembered: {' -> '.join(steps)}.")
            return
        if turn.tool and not (turn.credited and kind != "praise"):
            counts = self.store.record(kind, turn.phrase, turn.tool, turn.action)
            if kind == "praise":
                self._lesson(turn.utterance, turn.tool, True)   # praised: taught twice
            self._say(f"LEARN: rewarded {turn.route} for {turn.phrase!r} "
                      f"({counts['rewards']} time{'s' if counts['rewards'] != 1 else ''}"
                      f"{', praised' if kind == 'praise' else ''}).")
            self._maybe_offer(turn, counts)

    def _penalise(self, turn: Turn, now: float) -> None:
        self.corrected = replace(turn, at=now)
        self.store.forget_sequence(turn.phrase)
        if not turn.tool or not turn.ok:
            return                     # nothing routed, or already counted as a failure
        counts = self.store.record("corrected", turn.phrase, turn.tool, turn.action)
        withdrawn = self._poison(turn)
        self._say(f"LEARN: penalised {turn.route} for {turn.phrase!r} - corrected "
                  f"({counts['penalties']} mistake{'s' if counts['penalties'] != 1 else ''})"
                  + ("; its instant shortcut is withdrawn." if withdrawn else "."))

    def _failed(self, turn: Turn) -> None:
        counts = self.store.record("failed", turn.phrase, turn.tool, turn.action)
        self._say(f"LEARN: penalised {turn.route} for {turn.phrase!r} - it failed "
                  f"({counts['failures']} failure{'s' if counts['failures'] != 1 else ''}).")

    def _retry(self, fixed: Turn, turn: Turn) -> None:
        """The first result after a correction: the corrected route."""
        if not turn.ok:
            self._failed(turn)
            self.corrected = fixed     # still waiting for one that works
            self.pending = replace(turn, heard=True)
            return
        if fixed.tool:
            if turn.tool != fixed.tool:
                kind = "corrected_route"
                self._lesson(fixed.utterance, fixed.tool, False)
                self._lesson(fixed.utterance, turn.tool, True)
                self._say(f"LEARN: corrected - {fixed.phrase!r} means {turn.route}, "
                          f"not {fixed.route}. I'll remember that.")
            else:
                kind = "retry"         # the right tool; a detail was wrong
                self._say(f"LEARN: {fixed.phrase!r} -> {turn.route} worked on the second try.")
            self.store.record(kind, fixed.phrase, turn.tool, turn.action)
        self.pending = replace(turn, utterance=fixed.utterance, phrase=fixed.phrase,
                               heard=True, credited=bool(fixed.tool))

    # ── effects ───────────────────────────────────────────────────────────────

    def _lesson(self, utterance: str, tool: str, positive: bool) -> None:
        if self._teach is None:
            return
        try:
            self._teach(utterance, tool, positive)
        except Exception:
            pass

    def _reflex_ledger(self) -> Any:
        try:
            return self._reflexes() if self._reflexes is not None else None
        except Exception:
            return None

    def _poison(self, turn: Turn) -> bool:
        """Count the mistake in the reflex ledger: a phrase that ever failed is
        never eligible, so a learned shortcut for it stops answering at once."""
        ledger = self._reflex_ledger()
        if ledger is None:
            return False
        try:
            was_shortcut = ledger.match(turn.utterance) is not None
            ledger.observe(turn.utterance, turn.tool, turn.args, ok=False)
            return was_shortcut and ledger.match(turn.utterance) is None
        except Exception:
            return False

    def _maybe_offer(self, turn: Turn, counts: dict[str, int]) -> None:
        """A route that has earned an instant shortcut is OFFERED, once.

        Promotion stays a person's act (tests/test_reflex_offer.py): a learned
        reflex bypasses the model. The rewards decide when to ask; the owner's
        yes goes through diagnostics promote=..., which re-checks eligibility."""
        if (counts["rewards"] < REWARDS_TO_PROMOTE or counts["penalties"]
                or not _plain_call(turn.args) or self.store.offered(turn.phrase)):
            return
        ledger = self._reflex_ledger()
        if ledger is None:
            return
        try:
            eligible = any(c.phrase == turn.phrase and c.tool == turn.tool
                           and c.action == turn.action for c in ledger.candidates())
        except Exception:
            return
        if not eligible:
            return
        self.store.record("offered", turn.phrase, turn.tool, turn.action)
        self._say(f"LEARN: {turn.phrase!r} has earned an instant shortcut ({turn.route}, rewarded "
                  f"{counts['rewards']} times) - offering it.")
        if self._hint is not None:
            try:
                self._hint(offer_text(turn.phrase, turn.route, counts["rewards"]))
            except Exception:
                pass

    def _calls(self, turn: Turn) -> list[tuple[str, str, Any]]:
        """The turn's real tool calls (Live receipts), or [] when unknown."""
        if not turn.request_id or self._steps_for is None:
            return []
        try:
            calls = [(str(n or ""), str(a or ""), ok) for n, a, ok in self._steps_for(turn.request_id)]
        except Exception:
            return []
        return [c for c in calls if c[0] not in _NOT_STEPS]

    def _offer_hint(self, phrase: str) -> None:
        if self._hint is None or phrase == self._hinted:
            return
        try:
            steps = self.store.sequence(phrase)
        except Exception:
            return
        if len(steps) < 2:
            return
        self._hinted = phrase
        try:
            self._hint(hint_text(phrase, steps))
        except Exception:
            return
        self._say(f"LEARN: reminded myself how {phrase!r} was done last time ({len(steps)} steps).")

    def _say(self, line: str) -> None:
        if self._log is not None:
            try:
                self._log(line)
            except Exception:
                pass


# ── the live worker's two hooks ──────────────────────────────────────────────

def _teach_network(utterance: str, tool: str, positive: bool) -> None:
    from . import intent_brain
    intent_brain.observe_async(utterance, tool, ok=True, negative=not positive)


def _turn_steps(worker: Any, request_id: int) -> list[tuple[str, str, Any]]:
    guard = worker._research_commitments()       # a Live turn's tool receipts
    return [(r.name, r.action, r.ok) for r in guard.receipts(request_id)]


def _send_hint(worker: Any, text: str) -> None:
    ready = getattr(worker, "_transport_ready", None)
    if not callable(ready) or not ready():
        return                                   # a stale note later helps nobody
    notes = getattr(worker, "_pending_context_notes", None)
    if notes is None:
        from collections import deque
        notes = worker._pending_context_notes = deque(maxlen=24)
    notes.append(text)
    from . import background
    background.spawn(worker._flush_context_notes(), name="orion-learned-hint")


def learner_for(worker: Any) -> RewardLearner | None:
    """The worker's reward learner, built on first use. Never raises."""
    learner = getattr(worker, "_reward_learner", None)
    if learner is None and not getattr(worker, "_reward_learner_failed", False):
        bus = getattr(worker, "bus", None)
        log = getattr(getattr(bus, "log", None), "emit", None)
        try:
            learner = RewardLearner(
                RewardStore(), teach=_teach_network,
                reflexes=getattr(worker, "reflex_learner", None),
                steps_for=lambda rid: _turn_steps(worker, rid),
                hint=lambda text: _send_hint(worker, text), log=log)
            worker._reward_learner = learner
        except Exception as exc:
            worker._reward_learner_failed = True
            if callable(log):
                log(f"LEARN: reward learning unavailable - {str(exc)[:120]}")
    return learner


def _spoken(worker: Any, text: str) -> bool:
    """True while *text* is the Live turn's own words (spoken, or typed to Live)."""
    return normalise(str(getattr(worker, "_turn_said", "") or "")) == normalise(text)


def note_utterance(worker: Any, text: str) -> None:
    """Hook: the owner's words were stored (live_worker._persist_episode)."""
    try:
        said_by_me = getattr(worker, "_said_by_me", None)
        if callable(said_by_me) and said_by_me(text):
            return                               # his own voice from the speakers
        learner = learner_for(worker)
        if learner is None:
            return
        voice = _spoken(worker, text)
        reflex = None
        if not voice:
            from .reflex import reflex_enabled
            matcher = getattr(worker, "_learned_reflex", None)
            found = matcher(text) if callable(matcher) and reflex_enabled() else None
            if found is not None:
                reflex = (found.tool, found.args())
        learner.note_utterance(text, reflex=reflex, voice=voice,
                               request_id=int(getattr(worker, "_turn_request_id", 0) or 0)
                               if voice else 0)
    except Exception:
        pass


def note_result(worker: Any, utterance: str, tool: str, args: dict[str, Any] | None,
                ok: bool) -> None:
    """Hook: a learnable turn's tool call ended (live_worker.observe_for_reflex)."""
    try:
        learner = learner_for(worker)
        if learner is None:
            return
        live = _spoken(worker, utterance)
        rid = int(getattr(worker, "_turn_request_id", 0) or 0) if live else 0
        learner.note_result(utterance, tool, args, ok, request_id=rid, live=live)
    except Exception:
        pass


# ── what he has learned, for the benchmark and "what have you learned" ───────

def learning_stats(path: str | Path | None = None) -> dict[str, Any]:
    """Rewards, penalties, reflexes, remembered methods and routing on real
    turns. Reads stores only; never builds the intent network."""
    out: dict[str, Any] = {}
    try:
        store = RewardStore(path)
        try:
            out.update(store.stats())
        finally:
            store.close()
    except Exception:
        pass
    try:
        from .reflex_learning import ReflexLearner
        ledger = ReflexLearner()
        try:
            facts = ledger.stats()
            out["reflexes"] = {"learned": facts["learned"], "candidates": facts["candidates"]}
        finally:
            ledger.close()
    except Exception:
        pass
    try:
        from . import intent_brain
        net = intent_brain.loaded()
        if net is not None:
            out["network"] = {"observed": int(net.observed), "actions": len(net.labels)}
    except Exception:
        pass
    try:
        from .resolver_shadow import ShadowEvaluator
        shadow = ShadowEvaluator()
        try:
            trend = shadow.recall_trend()
        finally:
            shadow.close()
        if trend.get("turns"):
            out["real_routing"] = trend
    except Exception:
        pass
    return out


def _rate(window: dict[str, int] | None) -> float | None:
    if not window:
        return None
    judged = window.get("rewards", 0) + window.get("penalties", 0)
    return 100.0 * window["rewards"] / judged if judged else None


def learning_lines(stats: dict[str, Any], previous: dict[str, Any] | None = None) -> list[str]:
    """Plain lines for a report; *previous* is the last benchmark's stats."""
    previous = previous or {}
    lines: list[str] = []
    if "rewards" in stats:
        since = ""
        if "rewards" in previous:
            since = (f" (+{stats['rewards'] - previous.get('rewards', 0)} rewards, "
                     f"+{stats['penalties'] - previous.get('penalties', 0)} mistakes since last run)")
        lines.append(f"  Learning: {stats['rewards']} rewards ({stats.get('praise', 0)} praised), "
                     f"{stats['penalties']} mistakes ({stats.get('corrections', 0)} corrected, "
                     f"{stats.get('failures', 0)} failed){since}")
        now, before = _rate(stats.get("week")), _rate(stats.get("prior_week"))
        if now is not None:
            lines.append(f"  Turns judged right this week: {now:.0f}%"
                         + (f" (the week before: {before:.0f}%)" if before is not None else ""))
    reflexes = stats.get("reflexes") or {}
    if reflexes or "offered" in stats:
        lines.append(f"  Instant reflexes learned: {reflexes.get('learned', 0)}, "
                     f"{reflexes.get('candidates', 0)} candidate(s) "
                     f"({stats.get('offered', 0)} offered after rewards); "
                     f"{stats.get('sequences', 0)} multi-step method(s) remembered")
    network = stats.get("network") or {}
    if network:
        lines.append(f"  Intent network: learned from {network['observed']} of your requests, "
                     f"{network['actions']} actions known")
    real = stats.get("real_routing") or {}
    if real.get("turns"):
        earlier = real.get("earlier")
        lines.append(f"  Real-turn routing, right tool shortlisted: {real['recent']:.0f}% of the last "
                     f"{real['recent_turns']}" + (f" (before that: {earlier:.0f}%)"
                                                   if earlier is not None else ""))
    return lines


def summary(path: str | Path | None = None) -> str:
    """A short spoken-friendly account of rewards and mistakes."""
    lines = learning_lines(learning_stats(path))
    try:
        store = RewardStore(path)
        try:
            best = store.top(3)
        finally:
            store.close()
    except Exception:
        best = []
    if best:
        lines.append("  Most rewarded: " + "; ".join(
            f"{r['phrase']!r} -> {r['tool']}{'.' + r['action'] if r['action'] else ''} "
            f"({r['rewards']})" for r in best))
    return "What I've learned from rewards and mistakes:\n" + "\n".join(lines) if lines else ""


__all__ = ["DEFAULT_PATH", "REWARDS_TO_PROMOTE", "RewardLearner", "RewardStore", "Turn",
           "WINDOW_S", "hint_text", "judge", "offer_text", "learner_for", "learning_lines", "learning_stats",
           "note_result", "note_utterance", "summary"]
