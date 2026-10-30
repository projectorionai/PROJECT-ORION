"""
Typing slips, put right before ORION works out what a request means.

A typed request with a slip in it ("colse spotify", "pasue the musik") lost
the very words the tool ranker matches on: measured 6 Oct 2026 on the
development half of the everyday routing set, one or two realistic slips per
request (swapped, dropped, doubled or neighbouring keys) took top-1 routing
from 77.7% to 59.2% with the sentence encoder warm. The intent network's
letter-trigrams soften a slip; the word ranker and the sentence encoder do
not.

`correct` changes only a word that is unknown - in no tool's wording and not
an English word the sentence encoder knows - and only to a word ORION's own
tool wording uses, one edit away (a letter dropped, doubled, swapped with its
neighbour or replaced). Ties go to the word the tools use most. Real words
are never touched, so a name or a word ORION has never seen passes through.

The corrected text is for SCORING only; what was actually said is what the
model, the transcript and memory see.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from typing import Any

#: Shorter words are left alone: one edit from a 3-letter word reaches dozens
#: of others, so the guess would be noise.
MIN_LENGTH = 4

_WORD = re.compile(r"[a-z]+")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CHUNK = re.compile(r"\S+")
#: A chunk with any of these inside it is a link, path, address, file name or
#: identifier - spelt exactly as meant, even when no dictionary knows it.
_LITERAL = re.compile(r"[0-9/\\@_=:#<>{}\[\]|~*+]|\.\w")
_EDGE_PUNCTUATION = ".,!?;:\"'()"
#: Endings that make a known word into another real one.
_ENDINGS = ("s", "es", "ed", "ing")


_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
#: A plain English word is offered as a correction only from the commoner
#: half of the dictionary (by the tokenizer's order): the rarer half is
#: mostly names and specialist words ("reggie"), poor guesses for a slip.
COMMON_RANK = int(__import__("os").environ.get("ORION_SPELL_COMMON", "14684"))


def _edits(word: str) -> set[str]:
    """Every string one edit from *word*: a letter dropped, added, replaced,
    or swapped with its neighbour (~450 for a 7-letter word). Generated per
    unknown word and looked up, so no index is held: one over the ~22k
    dictionary words measured 22 MB and 0.4 s to build."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    edits = {a + b[1:] for a, b in splits if b}
    edits |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    edits |= {a + c + b[1:] for a, b in splits if b for c in _ALPHABET}
    edits |= {a + c + b for a, b in splits for c in _ALPHABET}
    edits.discard(word)
    return edits


class Speller:
    """Corrects unknown words towards ORION's own vocabulary.

    *asked* and *described* count each word in the requests people make of
    the tools and in the tools' descriptions (or, for memory, the words of
    what is remembered); *english* ranks the dictionary's words, commonest
    first. Between candidates one edit away the most-asked word wins, then
    the most-described, then the commoner English word. With
    *english_candidates* a plain English word may be the answer too, after
    every word of ORION's own."""

    def __init__(self, asked: dict[str, int], described: dict[str, int] | None = None,
                 english: dict[str, int] | Iterable[str] = (),
                 english_candidates: bool = False) -> None:
        described = described or {}
        self.english = english if isinstance(english, dict) else dict.fromkeys(english, 0)
        self.english_candidates = english_candidates
        self._rarest = len(self.english) + 1
        self.priority = {w: (asked.get(w, 0), described.get(w, 0), -self.english.get(w, self._rarest))
                         for w in (*asked, *described) if len(w) >= MIN_LENGTH - 1}

    def _known(self, low: str) -> bool:
        """A known word, or a known word with an ending ("backups", "alerted")."""
        def known(word: str) -> bool:
            return word in self.english or word in self.priority
        if known(low):
            return True
        return any(low.endswith(end) and known(low[:-len(end)])
                   for end in _ENDINGS if len(low) - len(end) >= MIN_LENGTH - 1)

    def _rank(self, word: str) -> tuple[int, int, int]:
        return self.priority.get(word) or (0, 0, -self.english.get(word, self._rarest))

    def suggest(self, word: str) -> str | None:
        """The vocabulary word *word* was probably meant to be, or None."""
        low = word.lower()
        if len(low) < MIN_LENGTH or not low.isalpha() or self._known(low):
            return None
        found = [w for w in _edits(low) if w in self.priority
                 or (self.english_candidates and len(w) >= MIN_LENGTH - 1
                     and self.english.get(w, COMMON_RANK) < COMMON_RANK)]
        if not found:
            return None
        return max(sorted(found), key=self._rank)

    def correct(self, text: str) -> str:
        """*text* with each unknown word replaced by its likely intent.
        Links, file names, addresses and code ("main.py", "parse_config",
        "https://...") are left exactly as typed."""
        if not text:
            return text

        def fix(match: re.Match[str]) -> str:
            token = match.group(0)
            if "'" in token:
                return token
            better = self.suggest(token)
            return better if better else token

        def chunk(match: re.Match[str]) -> str:
            piece = match.group(0)
            if _LITERAL.search(piece.strip(_EDGE_PUNCTUATION)):
                return piece
            return _TOKEN.sub(fix, piece)
        return _CHUNK.sub(chunk, text)


# ── the dictionary ───────────────────────────────────────────────────────────

#: How often to look again for the dictionary while it is missing.
RECHECK_S = 60.0
_ENGLISH: dict[str, int] | None = None
_ENGLISH_CHECKED = [0.0]
_ENGLISH_LOCK = threading.Lock()


def english_words() -> dict[str, int]:
    """The dictionary (see _english_words), read once; {} while the model is
    not installed, looked for again every RECHECK_S."""
    global _ENGLISH
    with _ENGLISH_LOCK:
        if _ENGLISH is None or (not _ENGLISH
                                and time.monotonic() - _ENGLISH_CHECKED[0] > RECHECK_S):
            _ENGLISH_CHECKED[0] = time.monotonic()
            _ENGLISH = _english_words()
        return _ENGLISH


def _english_words() -> dict[str, int]:
    """Whole English words the sentence encoder's tokenizer knows (~20k), by
    their place in its vocabulary - which runs roughly commonest first
    ("the" 1996, "music" 2189, "boil" 26077) - when its model is installed.
    Never raises."""
    try:
        from .semantic import MODEL, SemanticEncoder
        folder = SemanticEncoder(MODEL).model_dir()
        if folder is None:
            return {}
        with open(folder / "tokenizer.json", encoding="utf-8") as handle:
            vocab: dict[str, Any] = json.load(handle)["model"]["vocab"]
        return {w: int(i) for w, i in vocab.items() if w.isalpha() and w.islower()}
    except Exception:
        return {}


# ── spellers over words that change ──────────────────────────────────────────

class CachedSpeller:
    """A Speller over words that change - the tool schema, the memory store -
    rebuilt when *key()* changes. Never built on the main thread: there the
    text is used as typed while a build runs in the background.

    *words()* returns (asked, described) counts, as Speller takes them."""

    def __init__(self, name: str, words: Callable[[], tuple[dict[str, int], dict[str, int]]],
                 key: Callable[[], Hashable], english_candidates: bool = False) -> None:
        self.name = name
        self.english_candidates = english_candidates
        self._words = words
        self._key = key
        self._speller: Speller | None = None
        self._built_for: tuple[Any, float] | None = None
        self._lock = threading.Lock()
        self._building = threading.Lock()

    def _current(self, key: Hashable) -> bool:
        """Built for this key, and - if it found no dictionary - recently."""
        built = self._built_for
        return (built is not None and built[0] == key
                and (self._speller is not None or time.monotonic() - built[1] <= RECHECK_S))

    def get(self) -> Speller | None:
        """Built here if needed (tens of ms: a worker thread's job); None
        without a dictionary."""
        key = self._key()
        with self._lock:
            if not self._current(key):
                english = english_words()
                # Without a dictionary every real word ORION has not seen
                # ("oven", "paris", "bike") looks like a slip - so no guessing.
                self._speller = (Speller(*self._words(), english, self.english_candidates)
                                 if english else None)
                self._built_for = (key, time.monotonic())
            return self._speller

    def _build_in_background(self) -> None:
        if not self._building.acquire(blocking=False):
            return

        def build() -> None:
            try:
                self.get()
            except Exception:
                pass
            finally:
                self._building.release()
        threading.Thread(target=build, name=f"orion-speller-{self.name}", daemon=True).start()

    def correct(self, text: str) -> str:
        """*text* with typing slips put right; unchanged without a
        dictionary, or on the main thread before the first build. Never raises."""
        try:
            if threading.current_thread() is threading.main_thread():
                if not self._current(self._key()):
                    self._build_in_background()
                    return text
            active = self.get()
            return active.correct(text) if active is not None else text
        except Exception:
            return text


# ── the speller ORION routes with ────────────────────────────────────────────

def _tool_words(declarations: Iterable[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    """(words people ask the tools with, words the tools are described in).
    Asked: tool names, curated vocabulary, everyday requests, kinds of app.
    Kept apart because asking outranks describing - "wether" is "weather" to
    an assistant with a weather tool, though its descriptions say "whether"
    more often."""
    asked: list[str] = []
    prose: list[str] = []
    for decl in declarations:
        asked.append(str(decl.get("name", "")).replace("_", " "))
        prose.append(str(decl.get("description", "")))
    try:
        from .tool_vocabulary import VOCABULARY
        asked.extend(VOCABULARY.values())
    except Exception:
        pass
    try:
        from .intent_situations import phrases
        for items in phrases().values():
            asked.extend(items)
    except Exception:
        pass
    try:
        from .app_resolver import CATEGORIES
        for category in CATEGORIES:
            asked.extend(category.phrases)
    except Exception:
        pass
    count = lambda texts: Counter(w for text in texts for w in _WORD.findall(text.lower()))
    return count(asked), count(prose)


def _schema() -> list[dict[str, Any]]:
    from .dispatch_schema import TOOL_DECLARATIONS
    return list(TOOL_DECLARATIONS)


TOOLS = CachedSpeller(
    "tools", lambda: _tool_words(_schema()),
    lambda: hash(tuple(sorted(str(d.get("name", "")) for d in _schema()))))


def speller() -> Speller | None:
    """The speller for the whole current tool schema, built here if needed
    (~90 ms: call from a worker thread); None without a dictionary."""
    return TOOLS.get()


def correct(text: str) -> str:
    """*text* with typing slips put right, for routing. Never raises."""
    return TOOLS.correct(text)
