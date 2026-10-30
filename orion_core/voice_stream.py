"""
Streaming speech for the local brain (local brain, stage 2).

Stage 1 spoke the local agent's answer whole: the model generated every token,
the reply went to the voice, and only then was the first piece of audio made.
The first audible word therefore came after the entire generation. Here the
final answer streams: the model's text arrives token by token, this module
cuts it into speakable pieces the moment each one is complete, and the voice
starts on the first piece while the rest is still being generated.

The pieces
----------
SentenceSplitter   model text in, speakable chunks out, incrementally: whole
                   sentences, or a clause once ~12 words have gone by without a
                   full stop. A very short opening ("Sure.", "Right.") is held
                   for the sentence after it, so a check on the chunk sees
                   "Sure. I've opened Notepad." whole, and so the voice never
                   opens on a one-word fragment.
clean_chunk        the agent's reply cleaning (markup, control tokens), for one
                   chunk.
looks_like_call    a reply that is a function call written as JSON text
                   (llama3.1 does this): never spoken, the agent parses it.
endpoint_hangover  how much silence ends an utterance on the local recogniser.
                   1.5 s was the only setting; the local and hybrid brains use
                   ~0.65 s, and the long hangover again whenever the words so
                   far plainly continue ("open the", "and then").
SpokenReply        the speaking side of one streamed turn, for the live worker:
                   drops the words of a cancelled turn, holds them while the
                   owner is talking, times the first chunk, and plays the
                   "still working" earcon when the first words are slow.

Everything here is standard library only and runs without audio, a model or
Qt, so the tests drive it directly.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable

# ── chunking ─────────────────────────────────────────────────────────────────

#: A sentence end: terminal punctuation (and any closing quote or bracket)
#: followed by whitespace. At the very end of the text so far nobody knows
#: yet - "3." may become "3.5" - so the end waits for the next character.
_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
#: Where a long sentence may be cut without breaking a phrase.
_CLAUSE = re.compile(r"[,;:](?=\s)|\s[-–—](?=\s)")
#: A full stop after these is not the end of a sentence.
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs", "etc",
                            "e.g", "i.e", "approx", "fig", "no", "vol", "mt", "ft"})

#: A clause is cut off once this many words have gone by without a sentence end.
CLAUSE_WORDS = 12
#: ...and only at a clause boundary at least this far in.
MIN_CLAUSE_WORDS = 5
#: With no clause boundary at all, a run of words is cut here.
MAX_RUN_WORDS = 24
#: A chunk shorter than this is held for the next sentence ("Sure." "Okay.").
MIN_CHUNK_WORDS = 3


def _words(text: str) -> int:
    return len(str(text or "").split())


def clean_chunk(text: str) -> str:
    """One chunk as it should be spoken: no markdown, no model control tokens,
    single spaces. The same rules as the agent's whole-reply cleaning."""
    out = str(text or "")
    try:
        from .text_tools import strip_markup
        out = strip_markup(out)
    except Exception:
        pass
    out = re.sub(r"<\|[^|>]{1,40}\|>", "", out)
    out = re.sub(r"[*_`#]{1,3}", "", out)
    return " ".join(out.split())


def looks_like_call(text: str) -> bool:
    """True when the reply so far is (the start of) a function call written as
    text rather than a sentence: JSON, a fenced block, a python tag."""
    head = str(text or "").lstrip()[:16]
    return bool(head) and (head[0] in "{[<" or head.startswith("```"))


class SentenceSplitter:
    """Incremental sentence/clause chunking of streamed model text.

    ``feed(delta)`` returns the chunks completed by *delta* (often none);
    ``flush()`` returns what is left once the stream has ended. Chunks are
    cleaned (clean_chunk) and never empty. Order is preserved exactly: the
    concatenation of every chunk is the cleaned text, sentence by sentence.
    """

    def __init__(self, *, clause_words: int = CLAUSE_WORDS,
                 min_words: int = MIN_CHUNK_WORDS, max_run_words: int = MAX_RUN_WORDS) -> None:
        self.clause_words = max(4, int(clause_words))
        self.min_words = max(1, int(min_words))
        self.max_run_words = max(self.clause_words + 1, int(max_run_words))
        self._buffer = ""
        self._held = ""          # a short sentence waiting for the next one
        self.raw = ""            # everything fed, verbatim
        self.emitted = 0         # chunks handed out

    def _sentence_end(self) -> int | None:
        """Index just past the first complete sentence in the buffer, or None."""
        newline = self._buffer.find("\n")
        for match in _END.finditer(self._buffer):
            if 0 <= newline < match.start():
                break
            candidate = self._buffer[:match.end()]
            if match.group(0).startswith(".") and len(match.group(0).rstrip("\"'”’)]")) == 1:
                word = candidate[:match.start()].rsplit(None, 1)
                last = word[-1].lower() if word else ""
                if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()) \
                        or re.fullmatch(r"(?:[a-z]\.)+[a-z]", last):
                    continue
            return match.end()
        if newline >= 0:
            return newline + 1
        return None

    def _clause_cut(self) -> int | None:
        """Where to cut a long run with no sentence end yet, or None."""
        pending = (self._held + " " + self._buffer).strip()
        if _words(pending) < self.clause_words:
            return None
        cuts = [m.end() for m in _CLAUSE.finditer(self._buffer)
                if _words(self._held + " " + self._buffer[:m.end()]) >= MIN_CLAUSE_WORDS]
        if cuts:
            return cuts[-1]
        if _words(pending) >= self.max_run_words:
            # Every word before the last space is complete; the last may not be.
            space = self._buffer.rstrip().rfind(" ")
            return space if space > 0 else None
        return None

    def _take(self, end: int) -> list[str]:
        piece, self._buffer = self._buffer[:end], self._buffer[end:]
        text = clean_chunk(f"{self._held} {piece}")
        if not text:
            self._held = ""
            return []
        if _words(text) < self.min_words:
            self._held = text
            return []
        self._held = ""
        self.emitted += 1
        return [text]

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []
        self.raw += delta
        self._buffer += delta
        out: list[str] = []
        while True:
            end = self._sentence_end()
            if end is None:
                end = self._clause_cut()
            if end is None:
                return out
            out.extend(self._take(end))

    def flush(self) -> list[str]:
        """The rest, once the stream has ended (a held short sentence included)."""
        text = clean_chunk(f"{self._held} {self._buffer}")
        self._held, self._buffer = "", ""
        if not text:
            return []
        self.emitted += 1
        return [text]


def split_text(text: str) -> list[str]:
    """A whole reply as the chunks a stream of it would have produced."""
    splitter = SentenceSplitter()
    return splitter.feed(str(text or "") + " ") + splitter.flush()


# ── endpointing ──────────────────────────────────────────────────────────────

#: Silence that ends an utterance on the local recogniser in the local and
#: hybrid brains (ORION_LOCAL_ENDPOINT_MS). The general hangover is 1.5 s.
LOCAL_ENDPOINT_S = 0.65
#: Words that leave a sentence open: whoever said them has not finished.
CONTINUATION_WORDS = frozenset(
    "and or but so then because if when while that which who the a an my your his her our "
    "their its this these those to of for with in on at from by into about as is are was "
    "were be um uh erm er hmm like also plus".split())


def local_endpoint_s() -> float:
    """The configured short endpoint (ORION_LOCAL_ENDPOINT_MS), bounded."""
    try:
        value = float(os.getenv("ORION_LOCAL_ENDPOINT_MS", "") or LOCAL_ENDPOINT_S * 1000.0)
    except ValueError:
        value = LOCAL_ENDPOINT_S * 1000.0
    return max(0.3, min(3.0, value / 1000.0))


def endpoint_hangover(last_words: str, short_s: float, long_s: float) -> float:
    """Seconds of silence that end the utterance whose latest words (the local
    recogniser's partial result) are *last_words*: *short_s*, or *long_s* while
    those words plainly continue. With no words yet, the short one."""
    words = re.findall(r"[a-z']+", str(last_words or "").lower())
    if words and words[-1].strip("'") in CONTINUATION_WORDS:
        return max(short_s, long_s)
    return short_s


# ── the speaking side of one streamed turn ───────────────────────────────────

#: First words later than this after the turn began: the earcon (or filler).
EARCON_AFTER_S = 2.0
#: The owner started talking over a turn that had said nothing yet: its words
#: wait at most this long for his utterance to be decided (a transcript), and
#: this long after he stops.
HOLD_MAX_S = 8.0
HOLD_AFTER_SPEECH_S = 2.5


def filler_mode() -> str:
    """ORION_LOCAL_FILLER: "earcon" (default), "words" ("One moment.") or "off"."""
    mode = os.getenv("ORION_LOCAL_FILLER", "earcon").strip().lower()
    return mode if mode in {"earcon", "words", "off"} else "earcon"


def earcon_after_s() -> float:
    try:
        value = float(os.getenv("ORION_LOCAL_EARCON_S", "") or EARCON_AFTER_S)
    except ValueError:
        value = EARCON_AFTER_S
    return max(0.0, value)


class SpokenReply:
    """One streamed local reply on its way to the voice.

    ``say(chunk)`` is what the agent calls for each speakable chunk. It is
    dropped when the turn is no longer current (cancelled, superseded, ORION
    stopping), buffered while held (the owner is talking), and otherwise
    handed to *speak* at once. Everything runs on the event loop; nothing here
    blocks.
    """

    def __init__(self, *, speak: Callable[[str], Any], live: Callable[[], bool] | None = None,
                 open_stream: Callable[[], Any] | None = None,
                 close_stream: Callable[[Any], Any] | None = None,
                 earcon: Callable[[], Any] | None = None,
                 earcon_after: float | None = None, filler: str | None = None,
                 loop: Any = None, clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], Any] | None = None, request: str = "") -> None:
        self._speak = speak
        self._live = live or (lambda: True)
        self._open_stream = open_stream
        self._close_stream = close_stream
        self._earcon = earcon
        self._earcon_after = earcon_after_s() if earcon_after is None else max(0.0, earcon_after)
        self._filler = filler_mode() if filler is None else filler
        self._loop = loop
        self._clock = clock
        self._log = log
        self.request = str(request or "")
        self.started_at = clock()
        self.first_at: float | None = None
        self.spoken: list[str] = []
        self.dropped = 0
        self.finished = False        # the agent will say nothing more
        self.closed = False          # finished and nothing held: the voice may end
        self.cancelled = False
        self.signalled = ""          # "earcon" | "words" once the late signal played
        self._held: list[str] = []
        self._holding = False
        self._token: Any = None
        self._timers: dict[str, Any] = {}
        self.task: Any = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> "SpokenReply":
        if self._open_stream is not None:
            try:
                self._token = self._open_stream()
            except Exception:
                self._token = None
        if self._earcon_after > 0 and self._filler != "off":
            self._later("late", self._earcon_after, self._late_signal)
        return self

    def close(self) -> None:
        """The agent has finished. Words held while the owner talks stay held
        until release() (or the hold timer) decides; then the voice may end."""
        self.finished = True
        if self.closed or self._holding:
            return
        self.closed = True
        self._stop_timers()
        token, self._token = self._token, None
        if self._close_stream is not None and token is not None:
            try:
                self._close_stream(token)
            except Exception:
                pass

    def cancel(self) -> None:
        """The turn was stopped: nothing more is said, nothing held is said."""
        self.cancelled = True
        self.dropped += len(self._held)
        self._held.clear()
        self._holding = False
        self.close()

    # ── speech ───────────────────────────────────────────────────────────────

    @property
    def has_spoken(self) -> bool:
        return bool(self.spoken)

    @property
    def earcon_after(self) -> float:
        return self._earcon_after

    @property
    def holding(self) -> bool:
        return self._holding

    @property
    def text(self) -> str:
        return " ".join(self.spoken)

    def first_chunk_ms(self) -> float | None:
        if self.first_at is None:
            return None
        return (self.first_at - self.started_at) * 1000.0

    def say(self, chunk: str) -> bool:
        """Speak one chunk now (True), hold it, or drop it (False)."""
        text = " ".join(str(chunk or "").split())
        if not text:
            return False
        if self.finished or self.closed or not self._current():
            self.dropped += 1
            return False
        if self._holding:
            self._held.append(text)
            return True
        return self._emit(text)

    def _current(self) -> bool:
        try:
            return bool(self._live())
        except Exception:
            return False

    def _emit(self, text: str) -> bool:
        if self.first_at is None:
            self.first_at = self._clock()
            self._cancel_timer("late")
        self.spoken.append(text)
        try:
            self._speak(text)
        except Exception as exc:
            self._note(f"VOICE: a streamed sentence could not be spoken - {str(exc)[:120]}")
        return True

    # ── holding while the owner talks ────────────────────────────────────────

    def hold(self, max_s: float = HOLD_MAX_S) -> bool:
        """Keep further words back (the owner is talking). True if newly held.
        Released by release(), or by itself after *max_s*."""
        if self.closed:
            return False
        newly = not self._holding
        self._holding = True
        self._cancel_timer("late")
        self._later("hold", max_s, self.release)
        return newly

    def extend_hold(self, seconds: float = HOLD_AFTER_SPEECH_S) -> None:
        """The owner stopped talking: decide within *seconds* (a transcript)."""
        if self._holding and not self.closed:
            self._later("hold", seconds, self.release)

    def release(self) -> int:
        """Say what was held, in order. Returns how many chunks were released."""
        self._cancel_timer("hold")
        if not self._holding:
            return 0
        self._holding = False
        held, self._held = self._held, []
        released = 0
        if self.closed or self.cancelled or not self._current():
            self.dropped += len(held)
        else:
            for text in held:
                self._emit(text)
            released = len(held)
        if self.finished:
            self.close()
        return released

    # ── the late signal ──────────────────────────────────────────────────────

    def _late_signal(self) -> None:
        self._timers.pop("late", None)
        if (self.closed or self.first_at is not None or self._holding
                or not self._current()):
            return
        if self._filler == "words":
            self.signalled = "words"
            try:
                self._speak("One moment.")
            except Exception:
                pass
            return
        if self._earcon is not None:
            try:
                played = self._earcon()
            except Exception:
                played = False
            if played is not False:
                self.signalled = "earcon"

    # ── timers (loop.call_later; none without a loop) ────────────────────────

    def _later(self, name: str, delay: float, callback: Callable[[], Any]) -> None:
        self._cancel_timer(name)
        loop = self._loop
        if loop is None:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        try:
            self._timers[name] = loop.call_later(max(0.0, delay), callback)
        except Exception:
            pass

    def _cancel_timer(self, name: str) -> None:
        handle = self._timers.pop(name, None)
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    def _stop_timers(self) -> None:
        for name in list(self._timers):
            self._cancel_timer(name)

    def _note(self, text: str) -> None:
        if self._log is not None:
            try:
                self._log(text)
            except Exception:
                pass


def merge_requests(earlier: str, later: str, limit: int = 400) -> str:
    """One request from an utterance cut short and the words that followed it:
    "what's the weather tomorrow" + "in London" -> "what's the weather tomorrow
    in London". Bounded to the last *limit* characters."""
    first = " ".join(str(earlier or "").split())
    second = " ".join(str(later or "").split())
    return " ".join(part for part in (first, second) if part)[-limit:]


__all__ = [
    "CLAUSE_WORDS", "CONTINUATION_WORDS", "EARCON_AFTER_S", "HOLD_AFTER_SPEECH_S", "HOLD_MAX_S",
    "LOCAL_ENDPOINT_S", "MIN_CHUNK_WORDS", "SentenceSplitter", "SpokenReply", "clean_chunk",
    "earcon_after_s", "endpoint_hangover", "filler_mode", "local_endpoint_s", "looks_like_call",
    "merge_requests", "split_text",
]
