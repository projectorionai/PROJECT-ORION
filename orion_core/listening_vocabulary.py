"""
The words ORION should expect to hear - so he hears them.

The owner: "the transcription of the audio is not as accurate sometimes"
(7 Oct 2026). Both recognisers were guessing blind. The live channel asked
Gemini for a transcript with an EMPTY configuration - language detected
afresh on every turn, no idea that "ORION", "Kieran" or "Kidderminster" are
likely - and offline Whisper got no prompt at all. Names are exactly where a
general recogniser fails, and exactly what a personal assistant hears most.

``vocabulary()`` gathers the phrases worth biasing towards:

  * ORION's own name and the words people use with him;
  * proper names in what he remembers - people, places, things (capitalised
    mid-sentence in stored facts, so "Hannah", "Kwik Fit", "Snow Hill");
  * the applications this person actually opens.

The live channel sends them as ``adaptation_phrases`` with a British English
language hint (the Gemini API supports both - see
google.genai._live_converters._AudioTranscriptionConfig_to_mldev); Whisper
gets them as its ``initial_prompt``. Nothing leaves the machine that the live
channel was not already being sent: these words are in the system
instruction's memory context already.
"""

from __future__ import annotations

import re
import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

#: The language ORION is spoken to in.
LANGUAGE_HINTS = ("en-GB",)

# Retains all twelve clean British SAPI utterances measured on 8 October
# (mean confidence 0.911–1.0). Confidence cannot repair a confidently wrong
# name: complete desktop utterances therefore use Whisper instead.
VOSK_FINAL_CONFIDENCE = 0.85


def vosk_final_text(payload: Any) -> str:
    """Accept only a final with measured word confidence, never a partial."""
    try:
        words = payload.get("result") or []
        scores = [float(word["conf"]) for word in words]
        if not scores or any(not math.isfinite(n) or not 0 <= n <= 1 for n in scores):
            return ""
        if sum(scores) / len(scores) < VOSK_FINAL_CONFIDENCE:
            return ""
        return str(payload.get("text") or "").strip()
    except (AttributeError, KeyError, TypeError, ValueError):
        return ""

#: Always expected.
CORE_PHRASES = ("ORION", "Orion", "Command Deck", "standby", "stand by")

#: How many phrases are sent at most. Adaptation biases the recogniser; a few
#: dozen well-chosen terms help, hundreds of rare ones start to hurt.
MAX_PHRASES = 60

#: Words capitalised for grammar rather than because they are names.
_NOT_NAMES = frozenset("""
The A An I It Its This That These Those He She They We You Your His Her Their Our My
User User's Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February
March April May June July August September October November December And But Or If
When Where What Who Why How Mr Mrs Ms Dr Sir Yes No OK Okay There Here Also Every
Mondays Tuesdays Wednesdays Thursdays Fridays Saturdays Sundays
""".split())

#: A word straight after one of these starts a line, a bullet or a heading,
#: so its capital is layout, not a name.
_LINE_STARTS = "-*:#>"

_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z][A-Za-z'-]{1,24}(?:\s+[A-Z][a-z][A-Za-z'-]{1,24}){0,2})\b")


#: Facts longer than this are documents (ingested notes, chapters), whose
#: capitalised words are headings rather than the names of people and places.
MAX_FACT_CHARS = 240


def _common_words() -> frozenset[str]:
    """Ordinary English words (the commoner half of the dictionary), which
    are capitalised in a fact for grammar or as a heading, not as a name."""
    try:
        from .spelling import COMMON_RANK, english_words
        return frozenset(w for w, rank in english_words().items() if rank < COMMON_RANK)
    except Exception:
        return frozenset()


def names_in(texts: Iterable[str], limit: int = 40,
             common: frozenset[str] | None = None) -> list[str]:
    """Proper names in *texts*, most frequent first: capitalised words that are
    not at the start of a sentence, not capitalised merely by grammar, not an
    ordinary English word, and not a filename or code identifier."""
    common = _common_words() if common is None else common
    counts: Counter[str] = Counter()
    for text in texts:
        text = str(text or "")
        if len(text) > MAX_FACT_CHARS:
            continue
        for match in _NAME.finditer(text):
            before = text[:match.start()].rstrip(" \t")
            if (not before or before[-1] in _LINE_STARTS or before[-1] in "\n•"
                    or re.search(r"(^|\n)\s*\d+[.)]$", before)):
                continue
            words = [re.sub(r"'s$", "", w) for w in match.group(1).strip().split()]
            while words and (words[0] in _NOT_NAMES or words[0].lower() in common):
                words.pop(0)
            words = [w for w in words if not re.search(r"[-0-9]|[a-z][A-Z]", w)]
            if not words or all(w in _NOT_NAMES or w.lower() in common for w in words):
                continue
            counts[" ".join(words)] += 1
    return [name for name, _n in counts.most_common(limit)]


def vocabulary(memory: Any = None, apps: Any = None, limit: int = MAX_PHRASES) -> list[str]:
    """The phrases to bias recognition towards, best first. Never raises."""
    phrases: list[str] = list(CORE_PHRASES)
    try:
        facts = _fact_texts(memory)
        phrases.extend(names_in(facts))
    except Exception:
        pass
    try:
        if apps is None:
            from .app_resolver import USAGE as apps
        phrases.extend(str(name) for name in apps.most_used(15))
    except Exception:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        key = phrase.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(phrase.strip())
    return out[:limit]


def _fact_texts(memory: Any) -> list[str]:
    """The values of stored facts (not graph mirrors or request patterns)."""
    if memory is None:
        return []
    matrix = getattr(memory, "matrix", memory)
    conn = getattr(matrix, "conn", None)
    lock = getattr(matrix, "_lock", None)
    if conn is None:
        return []
    sql = ("SELECT value FROM intelligence WHERE NOT (substr(key_ref, 1, 12) = 'graph_event_' "
           "OR substr(key_ref, 1, 7) = 'intent_') ORDER BY updated_at DESC LIMIT 400")
    if lock is not None:
        with lock:
            rows = conn.execute(sql).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    return [str(row[0] or "") for row in rows]


def whisper_prompt(phrases: Iterable[str], max_chars: int = 400) -> str:
    """The same vocabulary as a Whisper initial prompt: a short, natural list
    (Whisper's prompt window is 224 tokens; this stays well inside it)."""
    out = ""
    for phrase in phrases:
        piece = phrase if not out else f", {phrase}"
        if len(out) + len(piece) > max_chars:
            break
        out += piece
    return (out + ".") if out else ""


def transcription_config(phrases: Iterable[str]) -> dict[str, Any]:
    """The live channel's input-transcription settings."""
    return {"language_hints": {"language_codes": list(LANGUAGE_HINTS)},
            "adaptation_phrases": list(phrases)}


__all__ = ["CORE_PHRASES", "LANGUAGE_HINTS", "MAX_PHRASES", "names_in",
           "transcription_config", "vocabulary", "whisper_prompt"]
