"""
Which language the user actually speaks — learned once, then never asked again.

Why this exists
---------------
A live model replies in whatever language it was just spoken to, so a
conversation already adapts on its own. What does NOT adapt is everything ORION
says FIRST: the morning briefing, a proactive check-in, an alert, the greeting
after a restart. Those have no user turn to take a cue from, so they come out in
whatever language the prompt happens to be written in — English — regardless of
who is listening.

So the language is observed while the user talks and remembered, and the prompt
carries it into the next session. Nothing is announced and nothing is asked: a
setting the user has to find is a setting most people never find, and being
asked "what language do you speak?" by something that has just heard you speak
is worse than useless.

How the detection works
-----------------------
Dependency-free, because adding a language-detection package to identify a
handful of European languages would be a poor trade. Two stages:

  * **Script.** Cyrillic, Greek, CJK, Arabic, Hebrew, Devanagari and Thai are
    decided by character range and that is the end of it — they are unambiguous
    and no word list is needed.
  * **Function words.** Latin-script languages share an alphabet, so they are
    told apart by the small closed class of words that carry no meaning and
    appear in nearly every sentence. "the/and/is" against "der/und/ist" against
    "le/et/est". Content words are useless here: they are the words people
    borrow across languages.

It reports UNKNOWN rather than guessing. A wrong language confidently applied
is far worse than none — it would have ORION greeting an English speaker in
Portuguese every morning.

Stability
---------
One sentence never changes the answer. People quote, swear, paste and use
loanwords, and a single "merci" must not move a whole assistant into French.
The stored language changes only after several consecutive observations agree,
which makes a genuine switch take a few turns and makes a stray phrase cost
nothing.

Pure text and JSON — no Qt, no network, no model.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, deque
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

LANGUAGE_PATH = CONFIG_DIR / "language.json"

UNKNOWN = ""

#: Scripts that identify a language (or a small family) on sight. Ordered so
#: the first match wins; ranges are checked against the raw characters.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("ru", ((0x0400, 0x04FF), (0x0500, 0x052F))),      # Cyrillic
    ("el", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),      # Greek
    ("he", ((0x0590, 0x05FF),)),                        # Hebrew
    ("ar", ((0x0600, 0x06FF), (0x0750, 0x077F))),      # Arabic
    ("hi", ((0x0900, 0x097F),)),                        # Devanagari
    ("th", ((0x0E00, 0x0E7F),)),                        # Thai
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),      # Hangul
    ("ja", ((0x3040, 0x309F), (0x30A0, 0x30FF))),      # Kana
    ("zh", ((0x4E00, 0x9FFF),)),                        # Han
)

#: Function words: the closed class that appears in almost every sentence and
#: that nobody borrows. Content words are deliberately absent — "computer" and
#: "internet" are the same in a dozen languages and would identify none of them.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and is are was you i it to of in that this with for have not but".split()),
    "de": frozenset("der die das und ist sind ich du nicht mit ein eine zu von auf dass aber".split()),
    "fr": frozenset("le la les et est sont je tu ne pas un une de du dans que avec pour mais".split()),
    "es": frozenset("el la los las y es son yo no un una de en que con para pero muy".split()),
    "it": frozenset("il lo la i gli le e sono io non un una di in che con per ma".split()),
    "pt": frozenset("o a os as e sao eu nao um uma de em que com para mas muito".split()),
    "nl": frozenset("de het een en is zijn ik niet van in dat met voor maar heb".split()),
    "pl": frozenset("i w na nie jest sie z do to ale tak ja co za bardzo".split()),
    "tr": frozenset("bir ve bu ile de da icin ama cok ben sen var yok mi".split()),
    "sv": frozenset("och att det som en ar jag inte med for pa den har".split()),
    "da": frozenset("og at det som en er jeg ikke med for pa den har".split()),
    "no": frozenset("og at det som en er jeg ikke med for pa den har".split()),
    "fi": frozenset("ja on ei se han mina sina mutta kun niin ole".split()),
    "cs": frozenset("a je se na to ne v z do ale co jsem jsou".split()),
    "ro": frozenset("si este sunt nu de la in cu pentru dar eu tu".split()),
    "id": frozenset("dan yang di ke dari itu ini tidak saya kamu untuk dengan".split()),
    "vi": frozenset("va la cua khong toi ban co duoc cho nay".split()),
}

#: Human names, for a prompt line that reads like a sentence.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "ro": "Romanian", "id": "Indonesian",
    "vi": "Vietnamese", "ru": "Russian", "el": "Greek", "he": "Hebrew",
    "ar": "Arabic", "hi": "Hindi", "th": "Thai", "ko": "Korean",
    "ja": "Japanese", "zh": "Chinese",
}

#: Shortest utterance worth judging. "yes", "ok" and "thanks" identify nothing.
MIN_WORDS = 4

#: Consecutive agreeing observations before the stored language changes. Three
#: is enough that a quoted sentence or a loanword phrase cannot move it, and
#: short enough that a real switch is picked up within a few turns.
SWITCH_AFTER = 3


def _strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word)
                   if not unicodedata.combining(c))


def detect_script(text: str) -> str:
    """A language decided by writing system alone, or "" if Latin/unknown."""
    counts: Counter = Counter()
    for char in text or "":
        code = ord(char)
        for code_name, ranges in _SCRIPT_RANGES:
            if any(low <= code <= high for low, high in ranges):
                counts[code_name] += 1
                break
    if not counts:
        return UNKNOWN
    best, hits = counts.most_common(1)[0]
    letters = sum(1 for c in text if c.isalpha())
    # A couple of stray characters is a quotation, not a language.
    return best if letters and hits / letters > 0.4 else UNKNOWN


def detect_language(text: str) -> str:
    """Best guess at the language of *text*, or "" when it cannot tell.

    Deliberately conservative. Returning UNKNOWN costs nothing — the caller
    simply keeps what it already had — whereas a confident wrong answer has
    ORION greeting an English speaker in Portuguese every morning.
    """
    text = (text or "").strip()
    if not text:
        return UNKNOWN

    by_script = detect_script(text)
    if by_script:
        return by_script

    words = [_strip_accents(w).lower() for w in
             "".join(c if c.isalpha() or c.isspace() else " " for c in text).split()]
    words = [w for w in words if w]
    if len(words) < MIN_WORDS:
        return UNKNOWN

    scores: Counter = Counter()
    for code, markers in _FUNCTION_WORDS.items():
        scores[code] = sum(1 for w in words if w in markers)
    if not scores:
        return UNKNOWN
    ranked = scores.most_common(2)
    best, best_score = ranked[0]
    if best_score == 0:
        return UNKNOWN
    # A near-tie between two languages is not evidence for either. English and
    # Dutch, or Danish and Norwegian, overlap enough to need a clear margin.
    if len(ranked) > 1 and ranked[1][1] >= best_score:
        return UNKNOWN
    return best


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code or "")


class LanguageMemory:
    """Remembers the language the user speaks, across sessions.

    ``observe`` is called with each user transcript; ``code`` is what the
    prompt should carry. Nothing here ever speaks to the user.
    """

    def __init__(self, path: Any = None, *, switch_after: int = SWITCH_AFTER) -> None:
        self._path = path or LANGUAGE_PATH
        self._switch_after = max(1, int(switch_after))
        self._recent: deque[str] = deque(maxlen=self._switch_after)
        self._code = UNKNOWN
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8-sig"))
            code = str(raw.get("language") or "")
            if code in LANGUAGE_NAMES:
                self._code = code
        except (OSError, ValueError, UnicodeDecodeError, AttributeError):
            self._code = UNKNOWN          # absent or unreadable: learn it again

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._path,
                json.dumps({"language": self._code,
                            "name": language_name(self._code)}, indent=2),
                encoding="utf-8")
        except OSError:
            pass          # unpersisted is still usable for this session

    # ── observation ──────────────────────────────────────────────────────────

    @property
    def code(self) -> str:
        return self._code

    @property
    def name(self) -> str:
        return language_name(self._code)

    @property
    def known(self) -> bool:
        return bool(self._code)

    def observe(self, text: str) -> bool:
        """Note one user utterance. Returns True if the stored language changed.

        A single sentence never decides. People quote, swear, paste and borrow,
        and one "merci" must not move an entire assistant into French.
        """
        guess = detect_language(text)
        if not guess:
            return False
        self._recent.append(guess)
        if guess == self._code:
            return False
        if len(self._recent) < self._switch_after:
            return False
        if any(seen != guess for seen in self._recent):
            return False
        self._code = guess
        self._recent.clear()
        self._save()
        return True

    def forget(self) -> None:
        self._code = UNKNOWN
        self._recent.clear()
        self._save()

    # ── what the prompt carries ──────────────────────────────────────────────

    def prompt_line(self) -> str:
        """One sentence for the system prompt, or "" when nothing is known."""
        if not self._code:
            return ""
        return (
            f"LANGUAGE: the user speaks {self.name}. Everything YOU open — a "
            f"briefing, a check-in, an alert, a greeting — must be in "
            f"{self.name}, because those have no question of theirs to take a "
            f"cue from. If they write to you in another language, simply "
            f"answer in that one; do not comment on the change."
        )


__all__ = ["LANGUAGE_NAMES", "LANGUAGE_PATH", "MIN_WORDS", "SWITCH_AFTER",
           "UNKNOWN", "LanguageMemory", "detect_language", "detect_script",
           "language_name"]
