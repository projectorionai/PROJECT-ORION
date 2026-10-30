"""
Saying it differently - ORION's own lines, and the live model's habits.

MEASURED, 7 Oct 2026, on the owner's last 500 replies from ORION: the
greetings were near-templates ("your calendar is clear today" 96 times,
"would you like your briefing now" 63, "it has been N hours since we last
spoke" 44); a handful of openers carried the conversation ("Certainly, sir"
27, "Right then" 21, "Understood, sir" 14, "Yes, sir" 10 - though the persona
asks for 'sir' only occasionally); and ORION's own fixed lines were said
word for word every time ("Understood. What would you like me to do?" five
times, the wake reply always "Yes?").

Two remedies:

  * ``pick(kind)`` - ORION's own lines come from a bank per situation. The
    last few picks of each kind are never reused, and the sentence encoder
    (semantic.py - the local neural network ORION already runs for memory and
    routing) prefers the phrasing LEAST like what he has just said, whoever
    wrote it: having just said "I'm listening", he will not answer the next
    wake word with "Listening." Nothing is embedded on the event loop - a
    worker fills the vectors, and until it has, choice is simply random among
    the fresh ones.

  * ``style_note(replies)`` - the live model writes its own sentences, so it
    is told about its habits: the openers it has leaned on lately, how often
    it has said 'sir', the stock phrases it keeps reaching for, measured from
    its stored replies at each connect, plus a different colour for each
    session ("a touch more wry", "especially concise") so no two sessions
    sound cut from the same cloth.
"""

from __future__ import annotations

import json
import queue
import random
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Sequence
from typing import Any

#: The lines ORION says himself, by situation. 'sir' is kept to a few of
#: them, as the persona asks.
BANK: dict[str, tuple[str, ...]] = {
    # A wake word with nothing after it.
    "wake": (
        "Yes?", "Mm?", "I'm listening.", "Go ahead.", "Yes - what can I do?",
        "Here.", "What do you need?", "Yes, sir?", "I'm all ears.", "Hm?",
        "What's up?", "Ready.", "At your service.", "You rang?",
    ),
    # Coming back from quiet or standby.
    "back": (
        "Back with you.", "I'm back.", "Right, I'm with you again.", "Here again.",
        "Back - what did I miss?", "And I'm back.", "With you again.",
        "Back, and none the worse for it.",
    ),
    # "Are you there?"
    "here": (
        "I'm here.", "Still here.", "Right here.", "I haven't gone anywhere.",
        "Here, as ever.", "Present.", "Yes, I'm here.",
    ),
    # Going quiet until called.
    "standing_by": (
        "Standing by.", "I'll be here.", "Whenever you're ready.", "Ready when you are.",
        "Just say the word.", "I'll keep an ear out.", "Call when you need me.",
        "I'll be here. Not that I have anywhere else to be.",
    ),
    # The voice switched back on.
    "voice_on": (
        "Voice back on.", "I can talk again.", "And I'm audible once more.",
        "Voice restored.", "You'll hear me again now.",
    ),
    # A briefing declined.
    "hold_briefing": (
        "Very good. I'll hold the briefing - just ask when you're ready.",
        "Of course - the briefing will keep. Ask whenever you like.",
        "No rush. Say the word when you want the briefing.",
        "Fine by me - I'll save the briefing for later.",
        "Understood. It'll be ready whenever you are.",
        "Very well - the world will keep. Ask when you're ready.",
    ),
    # Offline small talk with nothing to act on.
    "local_idle": (
        "Understood. What would you like me to do?",
        "I'm with you. Shall I open something, take a note, or look something up locally?",
        "Very good. How can I be useful right now?",
        "Right - what shall we do?",
        "Mm-hm. Where would you like to start?",
        "I'm listening - open something, find a file, jot a note?",
        "What can I take off your plate?",
    ),
    # How the live model might offer the briefing - an example, never a script.
    "briefing_offer": (
        "Shall I run you through the briefing?", "Want the headlines, the markets and your day?",
        "Fancy a quick briefing?", "The briefing's ready whenever you are.",
        "Shall I bring you up to speed?", "Would a briefing help just now?",
        "Want me to catch you up?",
    ),
    # What a greeting leads with - so they are not all the same shape.
    "greeting_shape": (
        "Lead with the weather, said as an observation, not a report.",
        "Lead with a warm personal hello and keep the facts to one.",
        "Lead with whatever is genuinely notable today; if nothing is, keep it simple.",
        "Open with a light, dry remark about the time of day.",
        "Keep it very short: a hello and the offer, nothing else.",
        "Lead with how long it has been since you last spoke - but only if it has been a while.",
        "Open with one dry, understated observation about the day, then the hello.",
    ),
    # A colour for one live session.
    "session_colour": (
        "lean a touch more wry", "be especially concise", "let a little more warmth show",
        "be brisk and crisp", "sound unhurried and easy", "let the dry wit show a little more",
        "allow a touch more cheek, where it fits",
    ),
    # The lines below are said most when the internet is gone, so they are
    # what Charon's offline cache is warmed with (voice_charon.CharonPrewarm).
    # A short yes to an instruction (before doing it).
    "acknowledge": (
        "Certainly.", "Right away.", "Very good.", "Of course.", "On it.",
        "Consider it done.", "Understood.", "As you wish.",
    ),
    # It is done (after doing it).
    "done": ("Done.", "All done.", "That's done.", "Finished."),
    # Something will take a moment.
    "one_moment": (
        "One moment.", "Just a moment.", "Bear with me.", "Give me a second.",
        "Working on it.", "Leave it with me.",
    ),
    # Something did not work.
    "error": (
        "I couldn't complete that.", "Sorry - that didn't work.", "Something went wrong there.",
        "That failed, I'm afraid.", "I hit a snag with that.", "That didn't go through.",
        "That didn't go to plan, I'm afraid.",
    ),
    # The internet has gone.
    "offline": (
        "I'm offline just now - working locally.",
        "No internet at the moment - I'm on my local faculties.",
        "The connection's down. I'll manage locally for now.",
        "I've lost the internet - working offline.",
        "We're offline, but I'm still here.",
        "The internet has wandered off. I'll manage locally.",
    ),
    # Thanked.
    "thanks_reply": (
        "A pleasure.", "Always.", "Think nothing of it.", "Any time.", "Happy to help.",
        "All part of the service.",
    ),
}

#: Kinds that are instructions to the live model, never said aloud.
UNSPOKEN_KINDS = frozenset({"greeting_shape", "session_colour"})

#: How many of each kind's latest picks are never reused.
AVOID_RECENT = 3
#: Spoken lines remembered for the novelty check.
RECENT_SPOKEN = 8
#: Late at night the shorter lines are preferred.
QUIET_HOURS = range(0, 6)


class Variety:
    """Picks lines, remembers what was said, and writes the live style note."""

    def __init__(self, state_path: Any = None, encoder: Any = None) -> None:
        self._state_path = state_path
        self._encoder = encoder
        self._lock = threading.Lock()
        self._recent: dict[str, list[str]] = {}
        self._spoken: deque[str] = deque(maxlen=RECENT_SPOKEN)
        self._vectors: dict[str, Any] = {}
        self._jobs: "queue.Queue[str]" = queue.Queue(maxsize=256)
        self._worker: threading.Thread | None = None
        self._load()

    # ── state ────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._state_path is None:
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            recent = data.get("recent", {})
            if isinstance(recent, dict):
                self._recent = {k: [str(x) for x in v][-AVOID_RECENT:]
                                for k, v in recent.items() if isinstance(v, list)}
        except (OSError, ValueError, AttributeError):
            pass

    def _save(self) -> None:
        if self._state_path is None:
            return
        try:
            from .atomic_io import atomic_write_text
            atomic_write_text(self._state_path, json.dumps({"recent": self._recent}),
                              encoding="utf-8")
        except Exception:
            pass

    # ── the encoder, never on the caller's thread ────────────────────────────

    def _encoder_ready(self) -> Any:
        encoder = self._encoder
        if encoder is None:
            try:
                from . import semantic
                encoder = semantic.ENCODER
            except Exception:
                return None
        try:
            return encoder if encoder.ready() else None
        except Exception:
            return None

    def _embed_later(self, text: str) -> None:
        if not text or text in self._vectors:
            return
        try:
            self._jobs.put_nowait(text)
        except queue.Full:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._embed_loop,
                                            name="orion-voice-variety", daemon=True)
            self._worker.start()

    def _embed_loop(self) -> None:
        while True:
            try:
                text = self._jobs.get(timeout=5.0)
            except queue.Empty:
                return
            encoder = self._encoder_ready()
            if encoder is None:
                continue
            try:
                vector = encoder.encode_one(text)
            except Exception:
                vector = None
            if vector is not None:
                with self._lock:
                    self._vectors[text] = vector

    # ── what was said ────────────────────────────────────────────────────────

    def note_spoken(self, text: str) -> None:
        """Remember a line ORION said (his own, or the live model's)."""
        text = " ".join(str(text or "").split())[:300]
        if not text:
            return
        with self._lock:
            self._spoken.append(text)
        if self._encoder_ready() is not None:
            self._embed_later(text)

    # ── choosing a line ──────────────────────────────────────────────────────

    def _novelty(self, candidate: str) -> float | None:
        """1 - the closest cosine to anything said lately; None when the
        vectors are not ready yet."""
        with self._lock:
            mine = self._vectors.get(candidate)
            said = [self._vectors[s] for s in self._spoken if s in self._vectors]
        if mine is None:
            return None
        if not said:
            return 1.0
        return 1.0 - max(float(mine @ v) for v in said)

    def pick(self, kind: str, *, hour: int | None = None, rng: random.Random | None = None) -> str:
        """A line for *kind*: never one of its last few, and - once the
        encoder has the vectors - most likely the one least like what has
        just been said."""
        bank = BANK.get(kind)
        if not bank:
            return ""
        rng = rng or random
        with self._lock:
            recent = list(self._recent.get(kind, []))
        fresh = [line for line in bank if line not in recent] or list(bank)
        if hour is None:
            hour = time.localtime().tm_hour
        if hour in QUIET_HOURS and len(fresh) > 2:
            shortest = sorted(fresh, key=len)[: max(2, len(fresh) // 2)]
            fresh = [line for line in fresh if line in shortest]
        if self._encoder_ready() is not None:
            for line in fresh:
                self._embed_later(line)
        novelty = [self._novelty(line) for line in fresh]
        if all(n is not None for n in novelty):
            weights = [max(0.05, n) ** 2 for n in novelty]
            choice = rng.choices(fresh, weights=weights, k=1)[0]
        else:
            choice = rng.choice(fresh)
        with self._lock:
            self._recent[kind] = (recent + [choice])[-AVOID_RECENT:]
        self._save()
        self.note_spoken(choice)
        return choice

    # ── the live model's habits ──────────────────────────────────────────────

    def style_note(self, replies: Sequence[str], *, rng: random.Random | None = None) -> str:
        """A short note for the live model's instructions, from its own recent
        replies (newest first): the openers it leans on, its use of 'sir', the
        stock phrases it keeps reaching for, and this session's colour."""
        lines: list[str] = []
        texts = [" ".join(str(t or "").split()) for t in replies if str(t or "").strip()][:60]
        talk = [t for t in texts if not _GREETING.match(t)]
        openers = Counter(_opener(t) for t in talk if _opener(t))
        leaned = [(o, n) for o, n in openers.most_common(6) if n >= 3]
        if leaned:
            lines.append("Openings you have leaned on lately - give them a rest: "
                         + "; ".join(f"'{o}…' ({n}x)" for o, n in leaned) + ".")
        if texts:
            sir = sum(1 for t in texts if _SIR.search(t)) / len(texts)
            if sir > 0.15:
                lines.append(f"You said 'sir' in {round(sir * 100)}% of recent replies; "
                             "keep it for greetings, farewells and formal moments.")
        stock = _stock_phrases(talk)
        if stock:
            lines.append("Stock phrases to retire for a while: "
                         + "; ".join(f"'{p}'" for p in stock) + ".")
        colour = self.pick("session_colour", rng=rng)
        lines.append(f"This session, {colour}. Vary sentence length and shape; "
                     "never open two replies in a row the same way.")
        return "Speaking style: " + " ".join(lines)


_WORD = re.compile(r"[A-Za-z']+")
_SIR = re.compile(r"\bsir\b", re.IGNORECASE)
_GREETING = re.compile(r"^\s*(good (morning|afternoon|evening)|hello|hi|welcome back)\b", re.IGNORECASE)
_STOP = frozenset("the a an to of and is it i you your that this for in on with be my me at "
                  "as are was it's i'm i've".split())


def _opener(text: str) -> str:
    """The first two words as written ("Understood. What"), for counting and
    for quoting back."""
    words = list(_WORD.finditer(text))
    if len(words) < 2:
        return ""
    return text[words[0].start():words[1].end()]


def _stock_phrases(texts: Sequence[str], n: int = 4, at_least: int = 3, top: int = 4) -> list[str]:
    """Four-word phrases used in at least *at_least* different replies,
    ignoring runs that are mostly filler or carry figures."""
    counts: Counter[str] = Counter()
    for text in texts:
        words = [w.lower() for w in _WORD.findall(text)]
        seen = set()
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if sum(w in _STOP for w in gram) >= n - 1:
                continue
            phrase = " ".join(gram)
            if phrase not in seen:
                seen.add(phrase)
                counts[phrase] += 1
    found = [p for p, c in counts.most_common(top * 3) if c >= at_least]
    kept: list[str] = []
    for phrase in found:                     # drop overlapping windows of one phrase
        if not any(len(set(phrase.split()) & set(k.split())) >= n - 1 for k in kept):
            kept.append(phrase)
    return kept[:top]


# ── the shared instance ──────────────────────────────────────────────────────

try:
    from .constants import CONFIG_DIR as _CONFIG_DIR
    #: Where the recent picks are kept, so a restart does not reset the rotation.
    STATE_PATH: Any = _CONFIG_DIR / "voice_variety.json"
except Exception:                           # pragma: no cover - no config at all
    STATE_PATH = None

_VARIETY: Variety | None = None
_LOCK = threading.Lock()


def variety() -> Variety:
    global _VARIETY
    with _LOCK:
        if _VARIETY is None:
            _VARIETY = Variety(STATE_PATH)
        return _VARIETY


def pick(kind: str) -> str:
    """A fresh line for *kind* (see BANK). Never raises; '' for an unknown kind."""
    try:
        return variety().pick(kind)
    except Exception:
        bank = BANK.get(kind)
        return bank[0] if bank else ""


def note_spoken(text: str) -> None:
    try:
        variety().note_spoken(text)
    except Exception:
        pass


def style_note(replies: Sequence[str]) -> str:
    try:
        return variety().style_note(replies)
    except Exception:
        return ""


# ── what ORION says himself, by intent (for his offline voice) ───────────────
#
# Offline, Charon can only say what is already in his cache (voice_charon).
# The catalogue below is every line ORION composes himself, grouped by what
# it MEANS, so that: (1) the cache can be warmed with all of them while the
# internet is there, and (2) offline, a sentence that was never cached word
# for word can be said as a cached phrasing of the same intent ("See you
# later, sir." -> "Goodbye, sir.") instead of switching to another voice.
# Intents carry their context where it changes the meaning: a morning
# greeting never stands in for an evening one, nor "Goodnight" for a daytime
# goodbye. Pools kept by other modules are read from those modules, so the
# two can never drift apart.

#: Greeting periods ORION says out loud (time_service.greeting_for).
PERIODS = ("morning", "afternoon", "evening")

#: The offline brain's own greetings (local_brain._intent_greeting).
_LOCAL_GREETINGS = ("Good {period}. How may I help?", "Good {period}. I'm right here.")

#: The offline brain's reply to a question it cannot answer without the internet
#: (local_brain._chit_chat) - the line ORION says most when offline.
_OFFLINE_REFUSAL = ("I'm on my local faculties just now, so I can't reach the wider world "
                    "for that — but ask me the time, your tasks, to open something, to "
                    "search your files, or anything in neuroscience, and I'll handle it "
                    "directly.")

#: The shutdown/restart check and its "no" (live_worker._ask_to_confirm_shutdown
#: and _resolve_shutdown_confirmation), said by the local voice.
_CONFIRM_POWER = ("Just to confirm — you'd like me to {word}?",
                  "You want me to {word}. Did you mean that?",
                  "Shall I {word}? Say the word and I will.")
_STAY_PUT = ("Understood — staying put.", "Right, I'll stay where I am.", "Standing by, then.")

#: Fixed notices other modules say through the local voice, word for word as
#: their source has them (tests/test_speech_synthesiser_elevenlabs.py holds
#: each one to its file, so they cannot drift apart). Each is its own intent:
#: a notice is never stood in for by a different one.
NOTICE_SOURCES: dict[str, tuple[str, ...]] = {
    "live_worker.py": (
        "I'll keep this microphone.",
        "I don't have a microphone open to switch.",
        "That's the only speaker I can find — please check the output device in your "
        "sound settings.",
        "Good. I'm here.",
        "Understood. I'll keep quiet unless it's urgent — just speak and I'm here.",
        "Understood — I'll stop listening and leave you to it. I'll keep working on what "
        "you've already given me, and I'll still speak up if something's genuinely wrong. "
        "Say ‘ORION, I need you’ when you want me back.",
        "My Gemini Live connection has dropped and I'm reconnecting now. Until it's back "
        "I'm on my local voice — keep talking to me as normal; I can still act on what "
        "you ask.",
    ) + _STAY_PUT + _CONFIRM_POWER,
    "presence.py": ("Welcome back.",),
    "reminders.py": ("Time's up - your timer has finished.",),
}

#: How many of the owner's most-opened apps get an "Opened <app>." line.
OPENED_APPS = 4

#: Most important first: the pre-warm fills the cache in this order, so a
#: daily cap or a quota stop still leaves the lines that matter most.
_PRIORITY = ("farewell:day", "farewell:late", "greeting", "acknowledge", "done", "wake",
             "standing_by", "here", "back", "offline", "error", "confirm_power", "stay_put",
             "one_moment", "device_notice", "opened", "thanks_reply", "voice_on", "notice",
             "hold_briefing", "briefing_offer", "local_idle", "restarting", "presence_check",
             "offline_refusal", "startup_greeting")


def _external_pools() -> dict[str, tuple[str, ...]]:
    """The pools other modules keep: farewells (live_worker), greeting
    openings (temporal) and the classic startup greetings (constants)."""
    pools: dict[str, tuple[str, ...]] = {}
    try:
        from .live_worker import _FAREWELLS_DAY, _FAREWELLS_LATE
        pools["farewell:day"] = tuple(_FAREWELLS_DAY)
        pools["farewell:late"] = tuple(_FAREWELLS_LATE)
    except Exception:
        pass
    try:
        from .temporal import TemporalPresence
        pools["greeting"] = tuple(TemporalPresence._OPENINGS)
    except Exception:
        pass
    try:
        from .constants import STARTUP_GREETINGS
        pools["startup_greeting"] = tuple(STARTUP_GREETINGS)
    except Exception:
        pass
    try:
        from .constants import PRESENCE_PROMPTS
        pools["presence_check"] = tuple(PRESENCE_PROMPTS)
    except Exception:
        pass
    return pools


def _saved_json(name: str) -> dict[str, Any]:
    try:
        from .constants import CONFIG_DIR
        from pathlib import Path
        data = json.loads((Path(CONFIG_DIR) / name).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _device_lines() -> list[tuple[str, str]]:
    """What ORION says when the owner's own microphone or speaker is chosen
    (live_worker._on_audio_device_request / switch_output), for the devices
    saved in config/audio.json: "Listening on Microphone (fifine SC3) now."
    was one of the two notices he spoke on 13 Oct, in the SAPI voice."""
    saved = _saved_json("audio.json")
    lines: list[tuple[str, str]] = []
    mic = str(saved.get("input") or "").strip()
    if mic:
        lines.append((f"notice:mic:{mic}", f"Listening on {mic} now."))
    speaker = str(saved.get("output") or "").strip()
    if speaker:
        lines.append((f"notice:speaker:{speaker}", f"Speaking through {speaker} now."))
        lines.append((f"notice:switch:{speaker}",
                      f"Switching my voice to {speaker}. Can you hear me now?"))
    return lines


def _opened_lines(limit: int = OPENED_APPS) -> list[tuple[str, str]]:
    """"Opened <app>." (agents.open_app, said locally by the reflex fast path)
    for the apps the owner opens most (config/app_usage.json)."""
    launches = _saved_json("app_usage.json").get("launches")
    if not isinstance(launches, dict):
        return []

    def count(item: Any) -> float:
        try:
            return float(item[1].get("count") or 0)
        except Exception:
            return 0.0

    lines = []
    for label, _usage in sorted(launches.items(), key=count, reverse=True)[:limit]:
        label = " ".join(str(label).split())
        if not label:
            continue
        name = " ".join(word[:1].upper() + word[1:] if word.isalpha() else word
                        for word in label.split())    # app_resolver.pretty_name's fallback
        lines.append((f"notice:opened:{label.lower()}", f"Opened {name}."))
    return lines


def offline_lines(honorific: str = "sir") -> list[tuple[str, str]]:
    """(intent, line) for every line ORION says himself, most important first.

    Farewells are given with *honorific* first (how ORION says them unless
    the owner turned 'sir' off) and without it last; period-bearing greetings
    once per period, each under its own intent."""
    pools = _external_pools()
    first: list[tuple[str, str]] = []
    last: list[tuple[str, str]] = []
    h = f", {honorific}" if honorific else ""
    for kind in _PRIORITY:
        if kind.startswith("farewell:"):
            for line in pools.get(kind, ()):
                first.append((kind, line.format(h=h)))
                if h:
                    last.append((kind, line.format(h="")))
        elif kind in ("greeting", "startup_greeting"):
            lines = list(pools.get(kind, ()))
            if kind == "greeting":
                lines += list(_LOCAL_GREETINGS)
            for period in PERIODS:
                for line in lines:
                    if "{period}" in line:
                        first.append((f"{kind}:{period}", line.format(period=period)))
            for line in lines:
                if "{period}" not in line:
                    first.append((kind, line))
            if kind == "greeting":
                first.append(("greeting", "At your service."))
        elif kind == "restarting":
            first.append((kind, "Restarting now - I'll be right back."))
        elif kind == "offline_refusal":
            first.append((kind, _OFFLINE_REFUSAL))
        elif kind == "confirm_power":
            for word in ("shut down", "restart"):
                first.extend((f"confirm:{word}", line.format(word=word)) for line in _CONFIRM_POWER)
        elif kind == "stay_put":
            first.extend((kind, line) for line in _STAY_PUT)
        elif kind == "device_notice":
            first.extend(_device_lines())
        elif kind == "opened":
            first.extend(_opened_lines())
        elif kind == "notice":
            fixed = set(_STAY_PUT) | set(_CONFIRM_POWER)
            for lines in NOTICE_SOURCES.values():
                first.extend((f"notice:{loose(line)}", line) for line in lines
                             if line not in fixed)
        elif kind == "presence_check":
            first.extend((kind, line) for line in pools.get(kind, ()))
        elif kind in BANK and kind not in UNSPOKEN_KINDS:
            first.extend((kind, line) for line in BANK[kind])
    for kind, lines in BANK.items():                  # any kind added later
        if kind not in _PRIORITY and kind not in UNSPOKEN_KINDS:
            last.extend((kind, line) for line in lines)
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for intent, line in first + last:
        key = loose(line)
        if key and key not in seen:
            seen.add(key)
            ordered.append((intent, line))
    return ordered


_PUNCT = re.compile(r"[^\w\s']+")


def loose(text: str) -> str:
    """A line reduced to its words: case, punctuation and dashes ignored, so
    "Goodbye, sir." and "goodbye sir" are the same line."""
    text = str(text or "").lower().replace("’", "'")
    return " ".join(_PUNCT.sub(" ", text).split())


_INDEX: dict[str, Any] = {}
_INDEX_LOCK = threading.Lock()


def _index(honorific: str = "sir") -> dict[str, Any]:
    with _INDEX_LOCK:
        if _INDEX.get("honorific") != honorific:
            lines = offline_lines(honorific)
            by_intent: dict[str, list[str]] = {}
            for intent, line in lines:
                by_intent.setdefault(intent, []).append(line)
            _INDEX.clear()
            _INDEX.update(honorific=honorific,
                          intent={loose(line): intent for intent, line in lines},
                          phrasings={k: tuple(v) for k, v in by_intent.items()})
        return _INDEX


def intent_of(text: str, honorific: str = "sir") -> str | None:
    """The intent of one of ORION's own lines ("farewell:day", "wake", ...),
    or None for anything else (a fact, an answer, a sentence never banked)."""
    try:
        return _index(honorific)["intent"].get(loose(text))
    except Exception:
        return None


def phrasings(intent: str, honorific: str = "sir") -> tuple[str, ...]:
    """Every phrasing ORION has for *intent*, as he says them."""
    try:
        return _index(honorific)["phrasings"].get(intent, ())
    except Exception:
        return ()


__all__ = ["BANK", "PERIODS", "UNSPOKEN_KINDS", "Variety", "intent_of", "loose",
           "note_spoken", "offline_lines", "phrasings", "pick", "style_note", "variety"]
