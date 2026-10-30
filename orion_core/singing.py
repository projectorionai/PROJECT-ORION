"""
ORION sings.

Asked to sing on 16 Oct 2026 he answered (session.log 14:15:31): "I am not quite
equipped to deliver a musical performance, sir. My vocal capabilities are, shall
we say, strictly conversational." Two separate gaps:

  * The Live channel. Gemini's native-audio model can carry a tune when it is
    asked and allowed to, but nothing in his persona said he could, and its
    one voice rule ("natural conversational pacing ... never rush or
    accelerate") pointed the other way. ``SINGING_DIRECTIVE`` is the persona
    line that says he can sing, and how; identity.py adds it to the persona.
  * Everywhere else. Gemini TTS only speaks, and the local voices only speak.
    The ``sing`` tool makes a short song properly sung, in his own Charon
    voice, repeatably, and can save it to a file.

How ``sing`` turns speech into singing
--------------------------------------
1. Words: the lyrics the model wrote, a public-domain song from memory (Happy
   Birthday, Twinkle Twinkle, Row Row Row Your Boat), an original composed by
   the provider router, or - offline - one of ORION's own songs below.
2. Melody: a public-domain song keeps its own tune; anything else gets one
   composed here (a chord progression for the mood, one note per syllable,
   stepwise and chord-aware, a long note to end each line, a cadence home).
3. Voice: Charon (Gemini TTS) reads the words one at a time ("Happy. Birthday.
   To. You."), so each word arrives clearly articulated with a pause around
   it. Two lyric lines per request, the requests in parallel.
4. Each word, then each syllable, is found in that audio (silences, energy
   valleys and voicing breaks, chosen by dynamic programming against the
   expected lengths), and re-sung with TD-PSOLA: the vowel's pitch periods are
   re-laid at the note's frequency (formants kept, so it is still his voice),
   the vowel is held for the note's length while consonants keep their natural
   length, with a short glide between notes and vibrato on long ones.
5. A soft synthesised backing (piano, pad and bass on the song's chords) and a
   little reverb, mixed well under the voice, and played through ORION's own
   voice output so the face lip-syncs to it.

numpy only - the exe's build environment has numpy and scipy but not librosa -
and imported lazily: identity.py imports this module for the directive, so
nothing heavy may load at import time.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import threading
import time
import wave
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# ── the persona directive (identity.py imports this) ─────────────────────────

SINGING_DIRECTIVE = (
    "SINGING: you can sing, and you sing well - but ONLY through the sing tool. "
    "Your speaking voice cannot hold a tune: words you speak come out spoken, "
    "however you write them, so never speak or write the lyrics in your reply. "
    "Whenever the user asks you to sing - a song, a lullaby, a jingle, happy "
    "birthday, 'sing me something' - never refuse and never recite: call the sing "
    "tool (in a voice session: use_tool with name 'sing') at once, passing four to "
    "eight short rhyming lines you write yourself as 'lyrics' and the mood as "
    "'style' (lullaby, upbeat, ballad, sad, jazzy, shanty, waltz or gentle), or a "
    "public-domain song by name as 'song' (Happy Birthday, Twinkle Twinkle Little "
    "Star, Row Row Row Your Boat - those you may sing in full). The tool tunes your "
    "own voice to a melody over a soft piano, legato and with a natural vibrato. At "
    "most a few words before it ('Very well.'), then say nothing while the song "
    "plays; afterwards you speak normally again. Stay in character as Orion. "
    "COVERS: when the user asks for a copyrighted song (a pop hit, say), do not "
    "refuse: sing a SHORT SNIPPET of it as a cover through the tool - its title as "
    "'song', at most two short lines as 'lyrics' and 'cover' true - then say it was "
    "a little snippet and offer an original song in the same style. Never sing, "
    "recite or write out more of a copyrighted song than that snippet."
)

#: A request to sing, not talk about singing ("I can't sing", "do you like
#: singing"). Spoken requests arrive without punctuation and often with his
#: name misheard in front ("Ryan, sing ...").
_SING_REQUEST = re.compile(
    r"(?:^|[,.!?]\s*|\b(?:orion|ryan|please|now|then|just)\s+)"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+|go on\s+(?:and\s+)?)?"
    r"sing\b(?!\s+along)"
    r"|\b(?:give|play)\s+(?:me|us)\s+a\s+(?:little\s+)?(?:song|tune|lullaby|jingle)\b"
    r"|\b(?:i\s+want|i'?d\s+like)\s+(?:you\s+)?to\s+sing\b",
    re.I)
_NOT_A_REQUEST = re.compile(
    r"\b(?:i|we|they|he|she)\s+(?:can'?t|cannot|don'?t|do not|never)\s+sing\b"
    r"|\b(?:do|did)\s+you\s+(?:like|enjoy|ever)\b|\bsinging\b(?!\s+(?:me|us))", re.I)
_SING_ABOUT = re.compile(r"\babout\s+(?P<topic>[\w' -]{2,60}?)\s*[?.!]?$", re.I)
_BIRTHDAY_TO = re.compile(r"\bhappy\s+birthday(?:\s+to\s+(?P<name>[a-z][\w'-]{1,30}))?", re.I)
#: Lyrics a model wrote into its reply: "*Twinkle, twinkle, little star,*".
_LYRIC = re.compile(r"\*([^*\n]{3,120})\*|♪\s*([^♪\n]{3,120})")


def is_sing_request(text: str) -> bool:
    """True when the words ask ORION to sing now."""
    words = " ".join(str(text or "").split())
    if not words or len(words) > 160 or _NOT_A_REQUEST.search(words):
        return False
    return bool(_SING_REQUEST.search(words))


def sing_args(text: str) -> dict[str, Any]:
    """Arguments for the sing tool from the words that asked: a known song by
    name, Happy Birthday to someone, a mood, or what the song is about."""
    words = " ".join(str(text or "").split())
    low = words.lower()
    args: dict[str, Any] = {}
    birthday = _BIRTHDAY_TO.search(words)
    if birthday:
        args["song"] = "happy birthday"
        if birthday.group("name") and birthday.group("name").lower() not in ("me", "you", "us"):
            args["name"] = birthday.group("name").title()
        return args
    flat = re.sub(r"[^a-z ]", " ", low)
    for title in PUBLIC_DOMAIN:
        if title in " ".join(flat.split()) or title.split()[0] * 2 in flat.replace(" ", ""):
            args["song"] = title
            return args
    for mood in MOODS:
        if re.search(rf"\b{mood}\b", low):
            args["style"] = mood
            break
    about = _SING_ABOUT.search(words)
    if about:
        args["topic"] = about.group("topic").strip()
    return args


def reads_lyrics(reply: str) -> bool:
    """True when a spoken reply is reciting lyrics instead of singing them."""
    return len(_LYRIC.findall(str(reply or ""))) >= 2


def lyric_lines(reply: str) -> list[str]:
    """The lyric lines a model wrote into its reply, in order."""
    lines = []
    for starred, noted in _LYRIC.findall(str(reply or "")):
        line = " ".join((starred or noted).strip(" ,").split())
        if line:
            lines.append(line)
    return lines[:MAX_LINES_FROM_REPLY]


MAX_LINES_FROM_REPLY = 8

# ── constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000                 # ORION's voice output (RECEIVE_SAMPLE_RATE)
CHUNK_BYTES = 9600                  # 0.2 s, as voice_charon streams
#: Lyric lines per voice request. One request per song would be fewest, but a
#: 20-second song read word by word is ~25 s of speech: one long generation
#: instead of a few short ones in parallel.
LINES_PER_CALL = 2
MAX_LINES = 12
MAX_WORDS_PER_LINE = 14
#: The whole render (voice requests + DSP) must stay inside Live's 45 s tool
#: answer window (live_worker.LIVE_TOOL_ANSWER_S).
RENDER_TIMEOUT_S = 40.0
#: A cover of a copyrighted song is a snippet, never the song: at most this
#: many lines are sung (owner, 26 Oct 2026: "only snippets of the song").
COVER_MAX_LINES = 2
TTS_TIMEOUT_S = 25.0
COMPOSE_DEADLINE_S = 9.0

HOP = 120                           # 5 ms analysis hop at 24 kHz
F0_MIN, F0_MAX = 60.0, 420.0
GLIDE_S = 0.075                     # portamento into each new note
VIBRATO_HZ = 5.4
VIBRATO_CENTS = 38.0
#: The silence left between two words of one line.
LEGATO_WORD_GAP_S = 0.022

# ── a human singer, not a tuner (Mark XXXIX) ─────────────────────────────────
# Owner, 27 Oct 2026: "his singing must not sound robotic". Every sung note was
# exactly its frequency from the first period to the last, every vibrato the
# same 5.4 Hz, every syllable the same loudness and every onset on the grid -
# the auto-tune sound - and a held vowel was a few pitch periods repeated in
# order, which buzzes. A singer does none of that. All of it is deterministic
# (seeded by the song), so a song renders the same twice.
SCOOP_CENTS = 45.0                  # a note started fresh is reached from below...
SCOOP_S = 0.09                      # ...over this long
OVERSHOOT_CENTS = 14.0              # a leap up of 3+ semitones overshoots and settles
DRIFT_CENTS = 8.0                   # the slow wander of a real voice around a note
VIBRATO_RATE_SPREAD = 0.45          # each long note's vibrato rate, +- Hz
VIBRATO_DEPTH_SPREAD = 0.25         # ...and its depth, +- this fraction
VIBRATO_ONSET_S = 0.17              # vibrato blooms after the note has settled
VIBRATO_RAMP_S = 0.35
VIBRATO_MIN_NOTE_S = 0.32
FALL_CENTS = 55.0                   # a phrase's last note lets go downward
FALL_S = 0.14
TIMING_JITTER_S = 0.011             # onsets breathe around the beat (s.d.)
GRAIN_JITTER = 1.3                  # held vowels: source periods chosen +- this many
BREATH_LEVEL = 0.07                 # an in-breath before a new line, vs the voice
#: Backing RMS relative to the voice's: about -11 dB, so the words lead.
BACKING_LEVEL = 0.28
REVERB_WET = 0.16

# ── the songs ORION knows ────────────────────────────────────────────────────
# Lyrics use "|" between the syllables of a word; each syllable has a list of
# (semitones from the tonic, beats) notes - one note, or several for a melisma.

C, G, F, G7, C7 = (0, 4, 7), (7, 11, 14), (5, 9, 12), (7, 11, 14, 17), (0, 4, 7, 10)

#: Name slot of Happy Birthday's third line ("dear Na-me").
_HB_NAME = [(-1, 1.0), (-3, 1.0)]

PUBLIC_DOMAIN: dict[str, dict[str, Any]] = {
    # "Happy Birthday to You" - public domain (Marya Music v. Warner/Chappell, 2015).
    "happy birthday": {
        "title": "Happy Birthday", "bpm": 100, "meter": 3, "pickup": 1.0,
        "mood": "waltz", "pattern": "oompah",
        "lines": [
            ("Hap|py birth|day to you", [(-5, .75), (-5, .25), (-3, 1), (-5, 1), (0, 1), (-1, 2)]),
            ("Hap|py birth|day to you", [(-5, .75), (-5, .25), (-3, 1), (-5, 1), (2, 1), (0, 2)]),
            ("Hap|py birth|day dear {name}", [(-5, .75), (-5, .25), (7, 1), (4, 1), (0, 1)]),
            ("Hap|py birth|day to you", [(5, .75), (5, .25), (4, 1), (0, 1), (2, 1), (0, 3)]),
        ],
        "chords": [(-3, 3, C), (0, 3, C), (3, 3, G), (6, 3, G), (9, 3, C), (12, 3, C7),
                   (15, 3, F), (18, 2, C), (20, 1, G7), (21, 3, C)],
    },
    # Jane Taylor's 1806 poem to the 18th-century French tune - public domain.
    "twinkle twinkle little star": {
        "title": "Twinkle, Twinkle, Little Star", "bpm": 108, "meter": 4, "pickup": 0.0,
        "mood": "lullaby", "pattern": "arpeggio",
        "lines": [
            ("Twin|kle twin|kle lit|tle star", [(0, 1), (0, 1), (7, 1), (7, 1), (9, 1), (9, 1), (7, 2)]),
            ("How I won|der what you are", [(5, 1), (5, 1), (4, 1), (4, 1), (2, 1), (2, 1), (0, 2)]),
            ("Up a|bove the world so high", [(7, 1), (7, 1), (5, 1), (5, 1), (4, 1), (4, 1), (2, 2)]),
            ("Like a dia|mond in the sky", [(7, 1), (7, 1), (5, 1), (5, 1), (4, 1), (4, 1), (2, 2)]),
            ("Twin|kle twin|kle lit|tle star", [(0, 1), (0, 1), (7, 1), (7, 1), (9, 1), (9, 1), (7, 2)]),
            ("How I won|der what you are", [(5, 1), (5, 1), (4, 1), (4, 1), (2, 1), (2, 1), (0, 3)]),
        ],
        "chords": ([(-4, 4, C)]
                   + [(b, 4, C) for b in (0, 32)] + [(b, 2, F) for b in (4, 36)]
                   + [(b, 2, C) for b in (6, 38)]
                   + [(b, 2, F) for b in (8, 40)] + [(b, 2, C) for b in (10, 42)]
                   + [(b, 2, G) for b in (12, 44)] + [(b, 2, C) for b in (14, 46)]
                   + [(b, 2, C) for b in (16, 20, 24, 28)] + [(b, 2, F) for b in (18, 26)]
                   + [(b, 2, G) for b in (22, 30)]),
    },
    # Traditional, first published 1852 - public domain.
    "row row row your boat": {
        "title": "Row, Row, Row Your Boat", "bpm": 138, "meter": 3, "pickup": 0.0,
        "mood": "upbeat", "pattern": "oompah",
        "lines": [
            ("Row row row your boat", [(0, 1.5), (0, 1.5), (0, 1), (2, .5), (4, 1.5)]),
            ("Gent|ly down the stream", [(4, 1), (2, .5), (4, 1), (5, .5), (7, 3)]),
            ("Mer|ri|ly mer|ri|ly mer|ri|ly mer|ri|ly",
             [(12, .5)] * 3 + [(7, .5)] * 3 + [(4, .5)] * 3 + [(0, .5)] * 3),
            ("Life is but a dream", [(7, 1), (5, .5), (4, 1), (2, .5), (0, 3)]),
        ],
        "chords": [(-3, 3, C)] + [(b, 3, C) for b in range(0, 18, 3)] + [(18, 3, G7), (21, 3, C)],
    },
}

_PD_ALIASES = {
    "happy birthday": ("happy birthday", "birthday song", "birthday"),
    "twinkle twinkle little star": ("twinkle", "little star"),
    "row row row your boat": ("row row", "row your boat"),
}

#: ORION's own songs, for when there is no one to compose a new one (offline).
#: Original lyrics written for ORION; keywords pick one for the topic asked.
ORIGINALS: list[dict[str, Any]] = [
    {"title": "The Quiet Hours", "mood": "lullaby",
     "keywords": ("sleep", "night", "lullaby", "bed", "dream", "tired", "calm", "quiet"),
     "lines": ["The lamps are low, the day is done,",
               "The stars come out there one by one,",
               "So close your eyes and drift away,",
               "I'll keep the watch till break of day."]},
    {"title": "Starlight and Code", "mood": "upbeat",
     "keywords": ("you", "orion", "yourself", "ai", "robot", "computer", "code", "star"),
     "lines": ["I'm made of starlight, wire and code,",
               "I keep your secrets, share your load,",
               "So call my name and I'll be there,",
               "Orion singing in the air."]},
    {"title": "A Proper Cup of Tea", "mood": "shanty",
     "keywords": ("tea", "coffee", "kettle", "british", "england", "drink"),
     "lines": ["The kettle sings, the morning's grey,",
               "A proper cup will save the day,",
               "With milk and one and not too strong,",
               "It puts right all that's going wrong."]},
    {"title": "Good Morning", "mood": "upbeat",
     "keywords": ("morning", "wake", "sun", "day", "breakfast", "happy", "cheer"),
     "lines": ["Good morning, sir, the sun is high,",
               "There's blue across the English sky,",
               "Your calendar is clear and bright,",
               "Let's make this morning go just right."]},
    {"title": "Silver Rain", "mood": "ballad",
     "keywords": ("rain", "sad", "storm", "weather", "grey", "love", "heart"),
     "lines": ["The rain is tapping on the pane,",
               "The streets are silver in the lane,",
               "But every storm will blow away,",
               "And leave a softer, brighter day."]},
]

#: (tempo, beats a bar, mode, chord roots as scale degrees per bar, backing pattern)
MOODS: dict[str, dict[str, Any]] = {
    "lullaby": {"bpm": 76, "meter": 3, "mode": "major", "prog": [0, 3, 0, 4], "pattern": "arpeggio"},
    "gentle":  {"bpm": 92, "meter": 4, "mode": "major", "prog": [0, 5, 3, 4], "pattern": "arpeggio"},
    "upbeat":  {"bpm": 116, "meter": 4, "mode": "major", "prog": [0, 4, 5, 3], "pattern": "pulse"},
    "ballad":  {"bpm": 72, "meter": 4, "mode": "major", "prog": [5, 3, 0, 4], "pattern": "sustain"},
    "sad":     {"bpm": 68, "meter": 4, "mode": "minor", "prog": [0, 5, 2, 6], "pattern": "sustain"},
    "jazzy":   {"bpm": 100, "meter": 4, "mode": "major", "prog": [1, 4, 0, 5], "pattern": "pulse"},
    "shanty":  {"bpm": 108, "meter": 4, "mode": "major", "prog": [0, 3, 0, 4], "pattern": "oompah"},
    "waltz":   {"bpm": 96, "meter": 3, "mode": "major", "prog": [0, 3, 4, 0], "pattern": "oompah"},
}
_MOOD_ALIASES = {
    "lullaby": ("lullaby", "sleepy", "soft", "calm", "bedtime", "quiet", "soothing", "gentle lullaby"),
    "upbeat": ("upbeat", "happy", "cheerful", "bright", "pop", "fun", "celebration", "energetic",
               "joyful", "lively", "silly"),
    "ballad": ("ballad", "love", "romantic", "tender", "warm", "heartfelt", "emotional"),
    "sad": ("sad", "melancholy", "blue", "blues", "sombre", "somber", "minor", "moody"),
    "jazzy": ("jazz", "jazzy", "swing", "smooth", "lounge", "crooner"),
    "shanty": ("shanty", "sea", "pirate", "rousing", "folk", "drinking", "pub", "jaunty"),
    "waltz": ("waltz", "birthday"),
    "gentle": ("gentle", "nice", "sweet", "simple"),
}

_SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
_NOTE_NAMES = ("C", "C sharp", "D", "E flat", "E", "F", "F sharp", "G", "A flat", "A", "B flat", "B")


class SingingError(RuntimeError):
    """A song could not be rendered (no voice, or the voice failed)."""


def _np():
    """numpy, imported on first use (see the module docstring)."""
    import numpy
    return numpy


# ── words and syllables ──────────────────────────────────────────────────────

_SYLLABLE_EXCEPTIONS = {
    "orion": 3, "every": 2, "everything": 3, "everyone": 3, "quiet": 2, "diamond": 2,
    "fire": 1, "hour": 1, "hours": 1, "our": 1, "being": 2, "poem": 2, "lion": 2,
    "idea": 3, "area": 3, "really": 2, "people": 2, "evening": 2, "heaven": 2,
    "beautiful": 3, "maybe": 2, "somewhere": 2, "something": 2, "someone": 2,
    "forever": 3, "anyone": 3, "goodbye": 2, "tomorrow": 3, "radio": 3, "piano": 3,
    "science": 2, "create": 2, "tired": 1, "flower": 2, "flowers": 2, "power": 2,
    "shower": 2, "towers": 2, "family": 3, "different": 2, "favourite": 2,
    "chocolate": 2, "business": 2, "ai": 2, "eyes": 1, "sometimes": 2, "lovely": 2,
}


def count_syllables(word: str) -> int:
    """How many syllables *word* is sung on (a heuristic, with exceptions)."""
    w = re.sub(r"[^a-z]", "", str(word).lower())
    if not w:
        return 0
    if w in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[w]
    if w.endswith("s") and w[:-1] in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[w[:-1]]
    if len(w) <= 3:
        return 1
    count = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("le") and w[-3] in "aeiouy" and count > 1:
        count -= 1                                   # smile, whole
    elif w.endswith("e") and not w.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1                                   # make, home
    elif w.endswith("es") and w[-3] not in "aeiouysxzhcg" and count > 1:
        count -= 1                                   # makes, tunes
    elif w.endswith("ed") and w[-3] not in "aeiouytd" and count > 1:
        count -= 1                                   # played, loved
    return max(1, count)


def clean_line(text: str) -> str:
    """A lyric line as the voice should read it: words, apostrophes, light punctuation."""
    text = str(text or "").replace("’", "'").replace("‘", "'")
    text = re.sub(r"[\"“”()\[\]{}*_#~<>]", " ", text)
    text = re.sub(r"[^A-Za-z0-9'\-.,!?;: ]", " ", text)
    try:
        from .utils import normalise_for_speech
        text = normalise_for_speech(text)
    except Exception:
        pass
    return " ".join(text.split())


def split_words(line: str) -> list[str]:
    """Sung words of a line: hyphenated words are separate words; no punctuation."""
    out = []
    for raw in clean_line(line).replace("-", " ").split():
        core = re.sub(r"[^A-Za-z0-9']", "", raw).strip("'")
        if core and count_syllables(core):
            out.append(core)
    return out


def parse_lyrics(text: Any) -> list[str]:
    """Lyric lines from what the model passed: newlines, " / ", or a list."""
    if isinstance(text, (list, tuple)):
        lines = [str(item) for item in text]
    else:
        raw = str(text or "")
        lines = re.split(r"\r?\n|\s/\s|\|", raw)
        if len(lines) == 1 and len(raw.split()) > MAX_WORDS_PER_LINE:
            lines = re.split(r"(?<=[.!?;,])\s+", raw)
    kept = []
    for line in lines:
        words = split_words(line)
        if not words:
            continue
        while len(words) > MAX_WORDS_PER_LINE:      # a run-on line becomes two
            kept.append(" ".join(words[:MAX_WORDS_PER_LINE // 2]))
            words = words[MAX_WORDS_PER_LINE // 2:]
        kept.append(" ".join(words))
    return kept[:MAX_LINES]


# ── the song ─────────────────────────────────────────────────────────────────

@dataclass
class SungWord:
    text: str                                         # what the voice reads
    notes: list[list[tuple[float, float]]]           # per syllable: [(semitones, beats)]


@dataclass
class Song:
    title: str
    lines: list[list[SungWord]]
    bpm: float
    meter: int
    pickup: float
    #: (start beat, beats, chord tones in semitones from the tonic); beat 0 is
    #: the first bar line, the intro bar is [-meter, 0).
    chords: list[tuple[float, float, tuple[float, ...]]]
    mood: str = "gentle"
    pattern: str = "arpeggio"
    source: str = "original"                          # original | public domain | your lyrics

    @property
    def text_lines(self) -> list[str]:
        return [" ".join(word.text for word in line) for line in self.lines]

    def syllable_timeline(self) -> list[tuple[int, int, int, list[tuple[float, float, float]]]]:
        """(line, word, syllable, [(start beat, beats, semitones)]) in sung order."""
        beat = -self.pickup
        out = []
        for li, line in enumerate(self.lines):
            for wi, word in enumerate(line):
                for si, notes in enumerate(word.notes):
                    placed = []
                    for pitch, beats in notes:
                        placed.append((beat, float(beats), float(pitch)))
                        beat += float(beats)
                    out.append((li, wi, si, placed))
        return out

    @property
    def end_beat(self) -> float:
        timeline = self.syllable_timeline()
        if not timeline:
            return 0.0
        start, beats, _pitch = timeline[-1][3][-1]
        return start + beats


def _deg_to_semi(degree: int, mode: str) -> int:
    scale = _SCALES.get(mode, _SCALES["major"])
    return scale[degree % 7] + 12 * (degree // 7)


def _line_rhythm(word_syllables: Sequence[int], beats: float, final: bool) -> list[float]:
    """Durations (beats) for every syllable of a line filling exactly *beats*."""
    n = sum(word_syllables)
    if n <= 0:
        return []
    last = 3.0 if final and n <= 6 else 2.0 if n <= 9 else 1.5 if n <= 12 else 1.0
    last = min(last, beats - 0.5 * (n - 1))
    room = beats - last
    m = n - 1
    if m == 0:
        return [beats]
    if m <= room:
        durations = [1.0] * m
        spare = room - m
        # Spare beats lengthen the word-initial (stressed) syllables, latest first,
        # so the line leans into its ending as sung lines do.
        starts = []
        position = 0
        for count in word_syllables:
            starts.append(position)
            position += count
        targets = [i for i in reversed(starts) if i < m] or list(range(m))
        k = 0
        while spare >= 0.5 and targets:
            add = 1.0 if spare >= 1.0 else 0.5
            durations[targets[k % len(targets)]] += add
            spare -= add
            k += 1
        durations.append(last + spare)
        return durations
    eighths = int(round(2 * (m - room)))
    quarters = m - eighths
    if quarters < 0:
        eighth = room / m
        return [eighth] * m + [last]
    # Eighths in pairs from the start, so the quarters stay on the beat.
    durations = [0.5] * eighths + [1.0] * quarters
    durations.append(last)
    return durations


def compose_melody(lines: Sequence[str], mood: str = "gentle", *, title: str = "",
                   seed: int | None = None, source: str = "original") -> Song:
    """An original melody for *lines*: chord progression per *mood*, one note per
    syllable, stepwise and chord-aware, each line arching and landing on a chord
    tone, the last on the tonic. Deterministic for the same words."""
    preset = MOODS.get(mood) or MOODS["gentle"]
    meter, mode = int(preset["meter"]), str(preset["mode"])
    words_by_line = [split_words(line) for line in lines]
    words_by_line = [words for words in words_by_line if words]
    if seed is None:
        seed = zlib.crc32(" / ".join(lines).encode("utf-8"))
    rng = random.Random(seed)
    prog = list(preset["prog"])
    bars_per_line = []
    for words in words_by_line:
        n = sum(count_syllables(w) for w in words)
        limit = 2 * meter + (3 if meter == 4 else 2)
        bars_per_line.append(2 if n <= limit else 4)
    total_bars = sum(bars_per_line)
    roots = [prog[bar % len(prog)] for bar in range(total_bars)]
    # Cadences: home at the very end (V-I), a half cadence on V mid-song.
    if total_bars >= 2:
        roots[-1], roots[-2] = 0, 4
    if len(words_by_line) >= 4:
        mid = sum(bars_per_line[:2]) - 1
        if 0 <= mid < total_bars - 2:
            roots[mid] = 4
    chord_degrees = [{(r + k) % 7 for k in (0, 2, 4)} for r in roots]

    song_lines: list[list[SungWord]] = []
    first_line_melody: tuple[tuple[int, ...], list[int]] | None = None
    bar_cursor = 0
    previous = 2
    for li, words in enumerate(words_by_line):
        syllables = [count_syllables(w) for w in words]
        bars = bars_per_line[li]
        beats_in_line = float(bars * meter)
        final = li == len(words_by_line) - 1
        rhythm = _line_rhythm(syllables, beats_in_line, final)
        starts = []
        position = 0.0
        for duration in rhythm:
            starts.append(position)
            position += duration
        signature = (tuple(syllables), tuple(rhythm))
        degrees: list[int]
        if li == 2 and first_line_melody is not None and first_line_melody[0] == signature:
            degrees = list(first_line_melody[1])      # a repeated line is what makes a song
            if final:
                degrees[-1] = 0
        else:
            degrees = []
            last_bar = bar_cursor + bars - 1
            end_options = sorted(
                (d for d in range(-2, 6) if d % 7 in chord_degrees[last_bar]),
                key=lambda d: abs(d - 2))
            end = 0 if final else (end_options[0] if end_options else 2)
            height = 2.5 + rng.random() * 1.5
            begin = previous if abs(previous - 1) <= 3 else 1
            n = len(rhythm)
            for i, (start, duration) in enumerate(zip(starts, rhythm)):
                if i == n - 1:
                    degrees.append(end)
                    break
                fraction = i / max(1, n - 1)
                target = begin + (end - begin) * fraction + height * math.sin(math.pi * fraction)
                bar = bar_cursor + int(start // meter)
                chord = chord_degrees[min(bar, total_bars - 1)]
                strong = abs(start % meter) < 1e-6 or (meter == 4 and abs(start % meter - 2) < 1e-6)
                best = None
                for d in range(-2, 6):
                    cost = abs(d - target) + 0.55 * abs(d - previous)
                    if abs(d - previous) > 4:
                        cost += 2.0
                    if d % 7 not in chord:
                        cost += 1.3 if strong else 0.25
                    cost += rng.random() * 0.5
                    if best is None or cost < best[0]:
                        best = (cost, d)
                degrees.append(best[1])
                previous = best[1]
        previous = degrees[-1]
        if li == 0:
            first_line_melody = (signature, degrees)
        notes_iter = iter(zip(degrees, rhythm))
        sung = []
        for word, count in zip(words, syllables):
            notes = []
            for _ in range(count):
                degree, duration = next(notes_iter)
                notes.append([(float(_deg_to_semi(degree, mode)), float(duration))])
            sung.append(SungWord(word, notes))
        song_lines.append(sung)
        bar_cursor += bars

    chords = [(-float(meter), float(meter), tuple(float(_deg_to_semi(k, mode)) for k in (0, 2, 4)))]
    for bar, root in enumerate(roots):
        tones = tuple(float(_deg_to_semi(root + k, mode)) for k in (0, 2, 4))
        chords.append((float(bar * meter), float(meter), tones))
    return Song(title=title or "An Original Song", lines=song_lines, bpm=float(preset["bpm"]),
                meter=meter, pickup=0.0, chords=chords, mood=mood if mood in MOODS else "gentle",
                pattern=str(preset["pattern"]), source=source)


def _syllabified(text: str) -> list[tuple[str, int]]:
    """("Hap|py", ...) -> [("Happy", 2), ...]."""
    out = []
    for raw in text.split():
        parts = [p for p in raw.split("|") if p]
        out.append(("".join(parts), len(parts)))
    return out


def public_domain_song(key: str, name: str = "") -> Song:
    """A public-domain song with its own tune. *name* fills Happy Birthday's slot."""
    spec = PUBLIC_DOMAIN[key]
    lines: list[list[SungWord]] = []
    for text, notes in spec["lines"]:
        tail: list[SungWord] = []
        if "{name}" in text:
            text = text.replace("{name}", "").strip()
            names = split_words(name)[:3] or ["friend"]
            counts = [count_syllables(w) for w in names]
            total = sum(counts)
            if total == len(_HB_NAME):
                flat = [[note] for note in _HB_NAME]
            elif total == 1:
                flat = [list(_HB_NAME)]                     # one syllable, sung over both notes
            else:
                share = 2.0 / total
                flat = [[(_HB_NAME[0][0] if i < total / 2 else _HB_NAME[1][0], share)]
                        for i in range(total)]
            cursor = 0
            for word, count in zip(names, counts):
                tail.append(SungWord(word.capitalize(), flat[cursor:cursor + count]))
                cursor += count
        sung = []
        cursor = 0
        for word, count in _syllabified(text):
            sung.append(SungWord(word, [[notes[cursor + i]] for i in range(count)]))
            cursor += count
        lines.append(sung + tail)
    chords = [(float(s), float(b), tuple(float(t) for t in tones)) for s, b, tones in spec["chords"]]
    return Song(title=spec["title"], lines=lines, bpm=float(spec["bpm"]), meter=int(spec["meter"]),
                pickup=float(spec["pickup"]), chords=chords, mood=str(spec["mood"]),
                pattern=str(spec["pattern"]), source="public domain")


def find_public_domain(request: str) -> str | None:
    words = " ".join(re.sub(r"[^a-z ]", " ", str(request or "").lower()).split())
    if not words:
        return None
    for key, aliases in _PD_ALIASES.items():
        if key in words or any(alias in words for alias in aliases):
            return key
    return None


def mood_for(text: str) -> str:
    words = " ".join(re.sub(r"[^a-z ]", " ", str(text or "").lower()).split())
    for mood, aliases in _MOOD_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", words) for alias in aliases):
            return mood
    return "gentle"


def original_from_bank(topic: str = "", mood: str = "") -> dict[str, Any]:
    """One of ORION's own songs, best matching *topic* (then *mood*)."""
    words = set(re.sub(r"[^a-z ]", " ", str(topic or "").lower()).split())
    scored = []
    for order, song in enumerate(ORIGINALS):
        score = len(words & set(song["keywords"])) * 2 + (1 if mood and song["mood"] == mood else 0)
        scored.append((-score, order, song))
    return min(scored)[2]


# ── composing new lyrics (online) ────────────────────────────────────────────

COMPOSE_INSTRUCTION = (
    "You are ORION's songwriter. You write short, ORIGINAL, singable song lyrics in "
    "British English and reply with JSON only.")


def compose_prompt(topic: str, mood: str, style_of: str = "", lines: int = 4) -> str:
    about = f" about {topic}" if topic else ""
    style = (f" in the style and mood of '{style_of}', but with entirely new words - never "
             "quote or paraphrase that song's lyrics" if style_of else "")
    return (
        f"Write the lyrics of a short original song{about}{style}. Mood: {mood}. "
        f"Exactly {lines} lines, 6 to 9 syllables each, simple words that are easy to sing, "
        "with a clear rhyme (AABB or ABAB). Never reproduce lyrics from an existing "
        "copyrighted song. Spell numbers as words. No stage directions, no chorus labels.\n"
        'Reply with JSON only: {"title": "...", "lines": ["...", "..."]}')


def parse_composition(text: str) -> tuple[str, list[str]] | None:
    """(title, lines) from the songwriter's reply, or None if it is unusable."""
    match = re.search(r"\{.*\}", str(text or ""), re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        return None
    lines = parse_lyrics([str(line) for line in data["lines"]])
    if not 2 <= len(lines) <= MAX_LINES:
        return None
    title = " ".join(str(data.get("title") or "").split())[:60] or "An Original Song"
    return re.sub(r"[^\w '\-,!?]", "", title), lines


# ── the voice (one seam to voice_charon, so the voice lane can swap it) ──────

def voices_on() -> bool:
    """Whether this is a real run. voice_charon.enable() is the switch orion.py
    flips; the tests never flip it, so no test speaks, runs SAPI, calls Gemini or
    writes the owner's Music folder."""
    try:
        from . import voice_charon
        return bool(getattr(voice_charon, "_ON", False))
    except Exception:
        return False


def charon_render(text: str, style: str = "") -> bytes | None:
    """*text* in Charon's voice as 24 kHz mono 16-bit PCM, or None when he is
    switched off. voice_charon.render_pcm: the speaking path's own CharonVoice
    (one client, one quota state, one set of stand-downs; an unstyled line
    already in his cache costs no request). *style* is Gemini TTS's
    natural-language direction ("Say warmly: ..."). Raises CharonUnavailable
    when the service cannot say it (perform tries the next voice)."""
    from . import voice_charon
    if not voice_charon.enabled():
        return None
    return voice_charon.render_pcm(text, style=style) or None


def charon_saving() -> bool:
    """Whether Charon's requests are scarce today (free tier), so the singer
    should not spend them."""
    try:
        from . import voice_charon
        return bool(voice_charon.enabled() and voice_charon.shared_voice().on_free_tier())
    except Exception:
        return False


def kokoro_render(text: str, style: str = "") -> bytes | None:
    """ORION's own offline neural voice (voice_kokoro, bundled with the app):
    the one that speaks for him whenever Charon cannot. It used to be missing
    here, so with Charon's daily quota spent (26 Oct 2026, 13:35) he sang in
    the Windows SAPI voice - robotic speech re-pitched, which is what sounded
    distorted."""
    del style
    from . import voice_kokoro
    if not voice_kokoro.enabled():
        return None
    voice = voice_kokoro.shared_voice()
    return (voice.stream_pcm(text) or None) if voice.available() else None


def piper_render(text: str, style: str = "") -> bytes | None:
    """The local neural voice (voice_piper), when the owner has installed one."""
    del style
    from . import voice_piper
    if not voice_piper.enabled():
        return None
    voice = voice_piper.PiperLocalVoice()
    return voice.stream_pcm(text) if voice.available() else None


def sapi_render(text: str, style: str = "") -> bytes | None:
    """Windows' own male voice to a WAV file in a child process (never in this
    process: pyttsx3.init() returns the speaking thread's engine, and two
    run loops on one engine fail), resampled to 24 kHz PCM."""
    del style
    import subprocess
    import tempfile
    from .voice_charon import to_pcm
    handle, path = tempfile.mkstemp(suffix=".wav", prefix="orion_sing_")
    os.close(handle)
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Male) } catch {}; "
        "$s.Rate = -1; "
        "$s.SetOutputToWaveFile('" + path.replace("'", "''") + "'); "
        "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()")
    try:
        from .quiet_subprocess import quiet_kwargs
        kwargs = quiet_kwargs()
    except Exception:
        kwargs = {}
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       input=text.encode("utf-8", errors="replace"), capture_output=True,
                       timeout=TTS_TIMEOUT_S, check=False, **kwargs)
        data = Path(path).read_bytes()
        return (to_pcm(data, "audio/wav") or None) if len(data) > 44 else None
    finally:
        try:
            os.unlink(path)              # our own temporary file
        except OSError:
            pass


@dataclass(frozen=True)
class Voice:
    name: str
    render: Callable[[str, str], bytes | None]
    parallel: bool = True


def voice_chain() -> list[Voice]:
    """Who can sing, best first: Charon (his online voice), then Kokoro (his
    offline voice), then a Piper voice if one is installed, then SAPI last."""
    if not voices_on():
        return []
    chain = [Voice("Charon", charon_render),
             Voice("offline voice", kokoro_render, False),
             Voice("local neural voice", piper_render, False)]
    if charon_saving():
        # A song is several Charon requests; on the free tier (ten a day) they
        # are his speech for the afternoon. His offline voice sings first.
        chain[0], chain[1] = chain[1], chain[0]
    if os.name == "nt":
        chain.append(Voice("Windows voice", sapi_render))
    return chain


def word_text(words: Sequence[str]) -> str:
    """The words read one at a time, so each arrives whole with a pause around it."""
    return " ".join(w[:1].upper() + w[1:] + "." for w in words)


# ── analysis: level, pitch, pitch marks ──────────────────────────────────────

def to_float(pcm: bytes):
    np = _np()
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def analyse(x, sr: int = SAMPLE_RATE) -> dict[str, Any]:
    """Per 5 ms frame: level in dB and f0 in Hz (0 = unvoiced), by YIN's
    cumulative-mean-normalised difference, vectorised with FFTs in blocks."""
    np = _np()
    maxlag = int(sr / F0_MIN)
    minlag = int(sr / F0_MAX)
    width = maxlag
    frame = width + maxlag + 2
    padded = np.concatenate([np.zeros(frame // 2, np.float32), x.astype(np.float32),
                             np.zeros(frame, np.float32)])
    n = max(1, int(math.ceil(len(x) / HOP)))
    view = np.lib.stride_tricks.sliding_window_view(padded, frame)[::HOP][:n]
    nfft = 1 << int(math.ceil(math.log2(frame + width)))
    centre = frame // 2
    level = np.empty(n, np.float32)
    f0 = np.zeros(n, np.float32)
    aperiodic = np.ones(n, np.float32)
    lags = np.arange(maxlag + 1)
    for begin in range(0, n, 512):
        block = np.asarray(view[begin:begin + 512], dtype=np.float64)
        rms_part = block[:, centre - 240:centre + 240]
        level[begin:begin + len(block)] = 10 * np.log10(np.mean(rms_part ** 2, axis=1) + 1e-12)
        head = block[:, :width]
        spectrum = np.conj(np.fft.rfft(head, nfft)) * np.fft.rfft(block, nfft)
        corr = np.fft.irfft(spectrum, nfft)[:, :maxlag + 1]
        energy0 = np.sum(head ** 2, axis=1, keepdims=True)
        squares = np.concatenate([np.zeros((len(block), 1)), np.cumsum(block ** 2, axis=1)], axis=1)
        energy_tau = squares[:, lags + width] - squares[:, lags]
        diff = np.maximum(energy0 + energy_tau - 2 * corr, 0.0)
        cumulative = np.cumsum(diff[:, 1:], axis=1)
        cmndf = np.ones_like(diff)
        cmndf[:, 1:] = diff[:, 1:] * lags[1:] / np.maximum(cumulative, 1e-12)
        region = cmndf[:, minlag:]
        below = region < 0.15
        has = below.any(axis=1)
        first = np.where(has, below.argmax(axis=1), region.argmin(axis=1))
        for row in range(len(block)):
            tau = int(first[row]) + minlag
            if has[row]:                                # walk down to the local minimum
                while tau + 1 <= maxlag and cmndf[row, tau + 1] < cmndf[row, tau]:
                    tau += 1
            value = float(cmndf[row, tau])
            if 1 <= tau < maxlag:                       # parabolic refinement
                a, b, c = cmndf[row, tau - 1], cmndf[row, tau], cmndf[row, tau + 1]
                denominator = a - 2 * b + c
                shift = 0.5 * (a - c) / denominator if abs(denominator) > 1e-12 else 0.0
                tau_f = tau + max(-1.0, min(1.0, shift))
            else:
                tau_f = float(tau)
            aperiodic[begin + row] = value
            f0[begin + row] = sr / tau_f if tau_f > 0 else 0.0
    loud = float(np.percentile(level, 97)) if n else -120.0
    floor = float(np.percentile(level, 10)) if n else -120.0
    threshold = max(loud - 40.0, floor + 8.0, -75.0)
    voiced = (aperiodic < 0.3) & (level > threshold + 4.0)
    f0 = np.where(voiced, f0, 0.0).astype(np.float32)
    f0 = _clean_f0(f0)
    return {"level": level, "f0": f0, "threshold": threshold, "sr": sr}


def _clean_f0(f0):
    """Median-smooth each voiced run and fold octave slips back to the run."""
    np = _np()
    out = f0.copy()
    voiced = out > 0
    n = len(out)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        run = out[i:j].astype(np.float64)
        if j - i < 4:                                  # a 15 ms blip is not a vowel
            out[i:j] = 0.0
        else:
            median = float(np.median(run))
            ratio = run / median
            run = np.where(ratio > 1.6, run / 2, np.where(ratio < 0.62, run * 2, run))
            padded = np.pad(run, 2, mode="edge")
            smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
            out[i:j] = smooth
        i = j
    return out


def pitch_marks(x, f0, sr: int = SAMPLE_RATE):
    """Analysis epochs: one per pitch period through every voiced run, each on
    the waveform's local peak (low-passed), so successive grains are aligned."""
    np = _np()
    kernel = max(3, int(sr / 1500))
    lp = np.convolve(x, np.ones(kernel, np.float32) / kernel, mode="same")
    marks: list[int] = []
    periods: list[float] = []
    voiced = f0 > 0
    n = len(f0)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        start, end = i * HOP, min(len(x) - 2, j * HOP)
        period = sr / float(f0[i])
        hi = min(end, int(start + period))
        if hi - start >= 4:
            m = start + int(np.argmax(lp[start:hi]))
            while m < end:
                frame = min(n - 1, m // HOP)
                if f0[frame] > 0:
                    period = sr / float(f0[frame])
                marks.append(m)
                periods.append(period)
                lo = int(m + 0.75 * period)
                hi = int(m + 1.25 * period) + 1
                if hi >= len(x) or lo >= end:
                    break
                nxt = lo + int(np.argmax(lp[lo:hi]))
                m = nxt if nxt > m else int(m + period)
        i = j
    return np.asarray(marks, dtype=np.int64), np.asarray(periods, dtype=np.float64)


# ── finding words and syllables in the speech ────────────────────────────────

def choose_cuts(candidates: Sequence[tuple[float, float, float]], start: float, end: float,
                weights: Sequence[float], *, min_len: float,
                duration_weight: float = 1.0) -> list[tuple[float, float, float]]:
    """Pick len(weights)-1 boundaries from *candidates* (left end, right start,
    strength) so the pieces between them are as long as *weights* expect and the
    cuts fall on strong boundaries - dynamic programming, numpy-vectorised.
    Falls back to proportional cuts when there are too few candidates."""
    np = _np()
    k = len(weights) - 1
    if k <= 0:
        return []
    share = np.asarray(weights, float)
    share = share / share.sum()

    def proportional() -> list[tuple[float, float, float]]:
        cuts = start + np.cumsum(share)[:-1] * (end - start)
        return [(float(c), float(c), 0.0) for c in cuts]

    usable = [c for c in candidates if c[0] - start >= min_len and end - c[1] >= min_len]
    if len(usable) < k:
        return proportional()
    left = np.array([start] + [c[0] for c in usable] + [end], float)
    right = np.array([start] + [c[1] for c in usable] + [end], float)
    strength = np.array([0.0] + [c[2] for c in usable] + [0.0])
    n = len(left)
    expected = share * (end - start)
    duration = left[None, :] - right[:, None]
    valid = duration >= min_len
    big = 1e9

    def cost(j: int):
        with np.errstate(divide="ignore", invalid="ignore"):
            c = duration_weight * np.log(np.maximum(duration, 1e-9) / expected[j]) ** 2
        return np.where(valid, c, big)

    best = cost(0)[0, :] - strength
    best[0] = best[-1] = big
    back = []
    for j in range(1, k):
        total = best[:, None] + cost(j)
        arg = np.argmin(total, axis=0)
        best = total[arg, np.arange(n)] - strength
        best[0] = best[-1] = big
        back.append(arg)
    final = best + cost(k)[:, n - 1]
    index = int(np.argmin(final))
    if final[index] >= big / 2:
        return proportional()
    chosen = [index]
    for arg in reversed(back):
        index = int(arg[index])
        chosen.append(index)
    chosen.reverse()
    return [usable[c - 1] for c in chosen]


def _runs(mask):
    """(start, end) of each True run in a boolean array."""
    np = _np()
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def _valleys(level, lo: int, hi: int, *, min_depth: float = 3.0) -> list[tuple[int, float]]:
    """Local minima of the smoothed level in frames [lo, hi): (frame, depth dB)."""
    np = _np()
    if hi - lo < 5:
        return []
    segment = np.convolve(level[lo:hi], np.ones(5) / 5, mode="same")
    out = []
    for i in range(2, len(segment) - 2):
        if segment[i] <= segment[i - 1] and segment[i] < segment[i + 1]:
            depth = min(segment[max(0, i - 24):i].max(), segment[i + 1:i + 25].max()) - segment[i]
            if depth >= min_depth:
                out.append((lo + i, float(depth)))
    return out


def segment_syllables(analysis: dict[str, Any], words: Sequence[str]
                      ) -> list[list[tuple[int, int, int | None, int | None]]]:
    """For each word, its syllables as (start, end, voiced start, voiced end) in
    samples of the analysed speech."""
    level, f0, threshold = analysis["level"], analysis["f0"], analysis["threshold"]
    loud = level > threshold
    frames = len(level)
    above = [i for i in range(frames) if loud[i]]
    if not above or not words:
        return [[] for _ in words]
    first, last = above[0], above[-1] + 1
    # Word boundaries: silences (strong, the longer the stronger) and dips.
    candidates = []
    for s, e in _runs(~loud[first:last]):
        s, e = int(s) + first, int(e) + first
        if e - s >= 4:
            candidates.append((float(s), float(e), 2.0 + 20.0 * (e - s) * HOP / SAMPLE_RATE))
    for frame, depth in _valleys(level, first, last, min_depth=4.0):
        if loud[frame]:
            candidates.append((float(frame), float(frame), depth / 8.0))
    candidates.sort()
    syllable_counts = [max(1, count_syllables(w)) for w in words]
    weights = [0.2 + 0.6 * c + 0.08 * len(w) for w, c in zip(words, syllable_counts)]
    cuts = choose_cuts(candidates, float(first), float(last), weights, min_len=8, duration_weight=1.5)
    bounds = [float(first)] + [x for cut in cuts for x in (cut[0], cut[1])] + [float(last)]
    voiced = f0 > 0
    out = []
    for wi, count in enumerate(syllable_counts):
        ws, we = int(bounds[2 * wi]), int(bounds[2 * wi + 1])
        ws, we = min(ws, we), max(ws, we)
        # Trim the word to where it is actually audible.
        while ws < we - 1 and not loud[ws]:
            ws += 1
        while we > ws + 1 and not loud[we - 1]:
            we -= 1
        inner = []
        for frame, depth in _valleys(level, ws, we, min_depth=2.0):
            score = depth / 6.0 + (0.8 if not voiced[frame] else 0.0)
            inner.append((float(frame), float(frame), score))
        for s, e in _runs(~voiced[ws:we]):
            s, e = int(s) + ws, int(e) + ws
            if s > ws and e < we:                      # a voicing break inside the word
                inner.append(((s + e) / 2.0, (s + e) / 2.0, 0.8 + 0.1 * (e - s)))
        inner.sort()
        cuts = choose_cuts(inner, float(ws), float(we), [1.0] * count, min_len=6)
        edges = [float(ws)] + [cut[0] for cut in cuts] + [float(we)]
        syllables = []
        for si in range(count):
            a, b = int(round(edges[si])), int(round(edges[si + 1]))
            b = max(b, a + 1)
            run = [i for i in range(a, b) if voiced[i]]
            if len(run) >= 5:
                syllables.append((a * HOP, b * HOP, run[0] * HOP, (run[-1] + 1) * HOP))
            else:
                syllables.append((a * HOP, b * HOP, None, None))
        out.append(syllables)
    return out


# ── re-singing: TD-PSOLA ─────────────────────────────────────────────────────

_WINDOWS: dict[int, Any] = {}


def _hann(half: int):
    np = _np()
    window = _WINDOWS.get(half)
    if window is None:
        window = np.hanning(2 * half + 3)[1:-1].astype(np.float32)
        _WINDOWS[half] = window
    return window


@dataclass(frozen=True)
class NoteStyle:
    """How one syllable is sung (Mark XXXIX): the human part of the pitch."""

    scoop: bool = True                  # approached from below (not slurred in)
    fall: bool = False                  # a phrase's last syllable lets go
    vib_rate: float = VIBRATO_HZ
    vib_cents: float = VIBRATO_CENTS
    vib_phase: float = 0.0
    #: (Hz, phase, cents) sines summed into a smooth, slow pitch wander.
    drift: tuple[tuple[float, float, float], ...] = ()
    seed: int = 0


def note_style(rng: random.Random, *, scoop: bool, fall: bool) -> NoteStyle:
    """A fresh, seeded NoteStyle: every long note's vibrato is its own."""
    drift = tuple((rng.uniform(0.35, 0.8) * k, rng.uniform(0, 2 * math.pi),
                   DRIFT_CENTS / k ** 0.8) for k in (1.0, 2.3, 4.1))
    return NoteStyle(scoop=scoop, fall=fall,
                     vib_rate=VIBRATO_HZ + rng.uniform(-VIBRATO_RATE_SPREAD, VIBRATO_RATE_SPREAD),
                     vib_cents=VIBRATO_CENTS * (1 + rng.uniform(-VIBRATO_DEPTH_SPREAD,
                                                                VIBRATO_DEPTH_SPREAD)),
                     vib_phase=rng.uniform(0, 2 * math.pi), drift=drift,
                     seed=rng.randrange(1 << 30))


def _smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3 - 2 * s)


def target_hz(t: float, notes: Sequence[tuple[float, float, float]], glide_from: float | None,
              sr: int = SAMPLE_RATE, style: NoteStyle | None = None) -> float:
    """The sung frequency at output sample *t*: the note's, gliding in from the
    previous note, with vibrato growing on long notes. With a *style* it is
    sung as a person would (NoteStyle): a fresh note scooped up to, a leap
    overshot a hair, a slow wander around the pitch, each long note's own
    vibrato, and the phrase's last note falling away at its release."""
    index = 0
    for index, (start, end, _hz) in enumerate(notes):
        if t < end:
            break
    start, end, hz = notes[index]
    previous = notes[index - 1][2] if index > 0 else glide_from
    elapsed = max(0.0, (t - start) / sr)
    log_f = math.log(hz)
    ln2 = math.log(2)
    slurred = bool(previous) and abs(math.log(previous / hz)) > 1e-4
    if previous and elapsed < GLIDE_S:
        s = _smoothstep(elapsed / GLIDE_S)
        log_f = math.log(previous) * (1 - s) + log_f * s
    length = (end - start) / sr
    if style is None:
        if length >= 0.4 and elapsed > 0.15:
            depth = VIBRATO_CENTS / 1200.0 * min(1.0, (elapsed - 0.15) / 0.3)
            log_f += ln2 * depth * math.sin(2 * math.pi * VIBRATO_HZ * (elapsed - 0.15))
        return math.exp(log_f)
    if slurred and previous < hz * 2 ** (-2.5 / 12) and GLIDE_S <= elapsed < GLIDE_S + 0.25:
        settle = math.exp(-(elapsed - GLIDE_S) / 0.06)
        log_f += ln2 * OVERSHOOT_CENTS / 1200.0 * settle
    elif style.scoop and not slurred and index == 0 and elapsed < SCOOP_S:
        log_f -= ln2 * SCOOP_CENTS / 1200.0 * (1.0 - _smoothstep(elapsed / SCOOP_S))
    seconds = t / sr
    wander = sum(cents * math.sin(2 * math.pi * hz_d * seconds + phase)
                 for hz_d, phase, cents in style.drift)
    log_f += ln2 * wander / 1200.0
    if length >= VIBRATO_MIN_NOTE_S and elapsed > VIBRATO_ONSET_S:
        bloom = min(1.0, (elapsed - VIBRATO_ONSET_S) / VIBRATO_RAMP_S)
        depth = style.vib_cents / 1200.0 * bloom
        log_f += ln2 * depth * math.sin(2 * math.pi * style.vib_rate * (elapsed - VIBRATO_ONSET_S)
                                        + style.vib_phase)
    if style.fall and index == len(notes) - 1:
        remaining = (end - t) / sr
        if remaining < FALL_S:
            log_f -= ln2 * FALL_CENTS / 1200.0 * _smoothstep(1.0 - remaining / FALL_S)
    return math.exp(log_f)


def sing_syllable(x, marks, periods, syllable: tuple[int, int, int | None, int | None],
                  out_start: int, out_end: int, notes: Sequence[tuple[float, float, float]],
                  glide_from: float | None, sr: int = SAMPLE_RATE,
                  style: NoteStyle | None = None):
    """One syllable re-sung on *notes* between output samples *out_start* and
    *out_end*. Returns (buffer, buffer's first output sample).

    Consonants keep their own length (squeezed only if they would eat most of a
    short note); the vowel is held - its first and last 50 ms at natural speed,
    the steady middle stretched - and its pitch periods are re-laid at the note's
    frequency (TD-PSOLA), which moves the pitch and leaves the formants: it
    still sounds like him.

    With a *style* the stretched middle no longer replays the same few periods
    in order (the buzz of a held vowel): each output period takes a source
    period a little either side of where the stretch maps it (GRAIN_JITTER),
    so the vowel keeps the small irregularity a living voice has, and its
    level moves gently with the vibrato."""
    jitter = random.Random(style.seed) if style is not None else None
    np = _np()
    a, b, v0, v1 = syllable
    pad = int(0.04 * sr)
    base = out_start - pad
    buffer = np.zeros(max(1, out_end - out_start) + 2 * pad, np.float32)
    length = len(x)

    def add(centre: float, grain) -> None:
        s = int(round(centre)) - base - len(grain) // 2
        e = s + len(grain)
        if e <= 0 or s >= len(buffer):
            return
        if s < 0:
            grain, s = grain[-s:], 0
        if e > len(buffer):
            grain = grain[:len(buffer) - s]
        buffer[s:s + len(grain)] += grain

    def take(centre: int, half: int, window):
        s, e = centre - half, centre + half + 1
        if s >= 0 and e <= length:
            return x[s:e] * window
        piece = np.zeros(2 * half + 1, np.float32)
        lo, hi = max(0, s), min(length, e)
        if hi > lo:
            piece[lo - s:hi - s] = x[lo:hi]
        return piece * window

    out_len = out_end - out_start
    usable = None
    if v0 is not None and v1 is not None:
        selected = (marks >= v0) & (marks < v1)
        if selected.sum() >= 2:
            usable = (marks[selected], periods[selected])
    if usable is not None:
        onset, coda = v0 - a, b - v1
        squeeze = 1.0
        if onset + coda > 0.45 * out_len:
            squeeze = 0.45 * out_len / (onset + coda)
        ov0, ov1 = out_start + onset * squeeze, out_end - coda * squeeze
        vin, vout = v1 - v0, ov1 - ov0
        edge = min(0.2 * vin, 0.05 * sr)
        if vout > vin and vin > 2 * edge > 0:
            out_points = [out_start, ov0, ov0 + edge, ov1 - edge, ov1, out_end]
            in_points = [a, v0, v0 + edge, v1 - edge, v1, b]
        else:
            out_points, in_points = [out_start, ov0, ov1, out_end], [a, v0, v1, b]
        unvoiced = [(out_start, ov0), (ov1, out_end)]
    else:
        ov0 = ov1 = None
        out_points, in_points = [out_start, out_end], [a, b]
        unvoiced = [(out_start, out_end)]

    # Unvoiced parts: 10 ms Hann grains every 5 ms (sums to one), time-mapped.
    half_u = HOP
    window_u = _hann(half_u)
    for zone_start, zone_end in unvoiced:
        t = float(zone_start)
        while t < zone_end:
            u = int(np.interp(t, out_points, in_points))
            add(t, take(u, half_u, window_u))
            t += half_u
    if usable is not None:
        vm, vp = usable
        t = float(ov0)
        held_from, held_to = (out_points[2], out_points[3]) if len(out_points) == 6 else (ov1, ov1)
        while t < ov1:
            hz = target_hz(t, notes, glide_from, sr, style)
            p_out = sr / hz
            u = float(np.interp(t, out_points, in_points))
            k = int(np.searchsorted(vm, u))
            if k >= len(vm) or (k > 0 and u - vm[k - 1] < vm[k] - u):
                k -= 1
            if jitter is not None and held_from <= t < held_to:
                k += int(round(jitter.uniform(-GRAIN_JITTER, GRAIN_JITTER)))
            k = max(0, min(len(vm) - 1, k))
            p_in = float(vp[k])
            half = max(8, int(round(p_in)))
            gain = max(0.6, min(1.8, p_out / p_in))
            if style is not None:
                # A breath of loudness with the vibrato (about +-0.5 dB).
                gain *= 1.0 + 0.055 * math.sin(2 * math.pi * style.vib_rate * t / sr
                                               + style.vib_phase)
            add(t, take(int(vm[k]), half, _hann(half)) * gain)
            t += p_out
    return buffer, base


# ── the backing ──────────────────────────────────────────────────────────────

def midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _piano(hz: float, seconds: float, sr: int, cache: dict):
    np = _np()
    key = (round(hz, 2), round(seconds, 3))
    tone = cache.get(key)
    if tone is None:
        n = int((seconds + 0.5) * sr)
        t = np.arange(n, dtype=np.float32) / sr
        tone = np.zeros(n, np.float32)
        for k in range(1, 7):
            partial = hz * k * (1 + 0.0004 * k * k)
            if partial > sr / 2.2:
                break
            tone += (1.0 / k ** 1.4) * np.sin(2 * np.pi * partial * t) * np.exp(-t * (1.1 + 0.9 * k))
        tone *= np.minimum(1.0, t / 0.004)
        tail = t > seconds
        tone[tail] *= np.exp(-(t[tail] - seconds) / 0.09)
        cache[key] = tone
    return tone


def _pad(hz: float, seconds: float, sr: int):
    np = _np()
    n = int((seconds + 0.4) * sr)
    t = np.arange(n, dtype=np.float32) / sr
    tone = (np.sin(2 * np.pi * hz * t) + np.sin(2 * np.pi * hz * 1.003 * t)
            + 0.3 * np.sin(2 * np.pi * 2 * hz * t))
    envelope = np.minimum(1.0, t / 0.25) * np.where(t > seconds, np.exp(-(t - seconds) / 0.25), 1.0)
    return (tone * envelope * 0.35).astype(np.float32)


def _bass(hz: float, seconds: float, sr: int):
    np = _np()
    n = int((seconds + 0.2) * sr)
    t = np.arange(n, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * hz * t) + 0.35 * np.sin(2 * np.pi * 2 * hz * t)
    envelope = np.minimum(1.0, t / 0.006) * np.exp(-t / 0.7)
    envelope = np.where(t > seconds, envelope * np.exp(-(t - seconds) / 0.05), envelope)
    return (tone * envelope).astype(np.float32)


def render_backing(song: Song, tonic: float, seconds_at: Callable[[float], float],
                   total: int, sr: int = SAMPLE_RATE):
    """Piano (and pad, and bass) on the song's chords, in a register above the
    voice so it does not mask the words."""
    np = _np()
    out = np.zeros(total, np.float32)
    cache: dict = {}
    beat_s = 60.0 / song.bpm

    def place(start_s: float, tone, gain: float) -> None:
        s = int(start_s * sr)
        if s >= total:
            return
        e = min(total, s + len(tone))
        out[s:e] += tone[:e - s] * gain

    chords = sorted(song.chords)
    if chords:                                         # let the last chord ring a bar
        start, beats, tones = chords[-1]
        chords[-1] = (start, beats + song.meter, tones)
    for start, beats, tones in chords:
        voiced = []
        for tone in tones:
            midi = tonic + 12 + tone
            while midi > tonic + 24:
                midi -= 12
            voiced.append(midi)
        voiced.sort()
        root = tonic + (tones[0] % 12) - 12
        if root < 36:
            root += 12
        begin = seconds_at(start)
        length = beats * beat_s
        place(begin, _bass(midi_hz(root), min(length, 2.0), sr), 0.55)
        if song.pattern == "arpeggio":
            order = voiced + [voiced[0] + 12] + voiced[1:][::-1]
            steps = int(round(beats * 2))
            for i in range(steps):
                place(begin + i * beat_s / 2, _piano(midi_hz(order[i % len(order)]), beat_s * 1.4, sr, cache), 0.22)
        elif song.pattern == "pulse":
            for i in range(int(round(beats))):
                for midi in voiced:
                    place(begin + i * beat_s, _piano(midi_hz(midi), beat_s * 0.8, sr, cache), 0.12)
        elif song.pattern == "oompah":
            for i in range(1, int(round(beats))):
                for midi in voiced:
                    place(begin + i * beat_s, _piano(midi_hz(midi), beat_s * 0.6, sr, cache), 0.12)
            for midi in voiced:
                place(begin, _pad(midi_hz(midi), length, sr), 0.05)
        else:                                          # sustain
            for midi in voiced:
                place(begin, _pad(midi_hz(midi), length, sr), 0.10)
                place(begin, _piano(midi_hz(midi), min(length, 2.5), sr, cache), 0.10)
    return out


def _reverb(signal, sr: int = SAMPLE_RATE, seconds: float = 1.1, seed: int = 7):
    """A small room: convolution with exponentially decaying noise (FFT)."""
    np = _np()
    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / seconds)
    ir = np.convolve(ir, np.ones(6) / 6, mode="same")   # darker tail
    ir[:int(0.02 * sr)] = 0.0                           # 20 ms pre-delay
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    size = 1 << int(math.ceil(math.log2(len(signal) + n)))
    wet = np.fft.irfft(np.fft.rfft(signal, size) * np.fft.rfft(ir, size), size)[:len(signal)]
    return wet.astype(np.float32)


# ── the performance ──────────────────────────────────────────────────────────

@dataclass
class Performance:
    song: Song
    pcm: bytes
    seconds: float
    voice: str
    tonic: float
    timings: dict[str, float] = field(default_factory=dict)
    #: (start s, end s, Hz) of every note sung, for checking the tuning.
    notes: list[tuple[float, float, float]] = field(default_factory=list)
    vocal: Any = None                                  # the dry voice, when asked for

    @property
    def key(self) -> str:
        mode = (MOODS.get(self.song.mood) or MOODS["gentle"])["mode"]
        if self.song.source == "public domain":
            mode = "major"
        return f"{_NOTE_NAMES[int(round(self.tonic)) % 12]} {mode}"

    def wav_bytes(self) -> bytes:
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(self.pcm)
        return stream.getvalue()


def _render_words(voice: Voice, texts: Sequence[str], deadline: float) -> list[bytes | None]:
    """Every chunk in *voice*, in parallel where the voice allows it."""
    if not voice.parallel or len(texts) == 1:
        out: list[bytes | None] = []
        for text in texts:
            if time.monotonic() > deadline:
                out.append(None)
                continue
            try:
                out.append(voice.render(text, ""))
            except Exception:
                out.append(None)
        return out
    from concurrent.futures import ThreadPoolExecutor, wait
    pool = ThreadPoolExecutor(max_workers=min(4, len(texts)), thread_name_prefix="orion-sing")
    try:
        futures = [pool.submit(voice.render, text, "") for text in texts]
        wait(futures, timeout=max(0.1, deadline - time.monotonic()))
        results: list[bytes | None] = []
        for future in futures:
            try:
                results.append(future.result(timeout=0) if future.done() else None)
            except Exception:
                results.append(None)
        return results
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def add_breaths(vocal, starts: Sequence[int], rng: random.Random, sr: int = SAMPLE_RATE) -> int:
    """A soft in-breath in the quiet before each new line (in place). Only
    where there IS quiet: a line sung straight on from the last gets none.
    Returns how many were added."""
    np = _np()
    loud = np.abs(vocal) > 1e-4
    if not np.any(loud):
        return 0
    level = float(np.sqrt(np.mean(vocal[loud] ** 2))) * BREATH_LEVEL
    noise_rng = np.random.default_rng(rng.randrange(1 << 30))
    added = 0
    for start in starts:
        end = int(start) - int(0.03 * sr)
        begin = end - int(0.24 * sr)
        if begin <= 0 or end >= len(vocal):
            continue
        window = np.abs(vocal[begin:end])
        quiet = window < 0.02 * (level / max(BREATH_LEVEL, 1e-6))
        if quiet.mean() < 0.85:
            continue                    # no room: the line runs straight on
        n = end - begin
        noise = noise_rng.standard_normal(n + 8).astype(np.float32)
        hiss = np.diff(noise)[:n + 4]                       # tilt up, away from rumble
        hiss = np.convolve(hiss, np.ones(5, np.float32) / 5.0, mode="same")[:n]
        shape = np.sin(np.pi * np.linspace(0.0, 1.0, n, dtype=np.float32) ** 0.8) ** 2
        hiss *= shape * (level / (float(np.sqrt(np.mean(hiss ** 2))) + 1e-9))
        vocal[begin:end] += hiss
        added += 1
    return added


def choose_tonic(song: Song, voice_hz: float) -> float:
    """The key: the melody's average a whole tone above his speaking pitch, so the
    shifts stay small (PSOLA is cleanest within about half an octave)."""
    np = _np()
    pitches, weights = [], []
    for _li, _wi, _si, notes in song.syllable_timeline():
        for _start, beats, pitch in notes:
            pitches.append(pitch)
            weights.append(beats)
    centre = float(np.average(pitches, weights=weights)) if pitches else 0.0
    voice_midi = 69 + 12 * math.log2(max(50.0, voice_hz) / 440.0)
    return float(round(voice_midi + 2.0 - centre))


def perform(song: Song, voices: Sequence[Voice] | None = None, *, keep_vocal: bool = False,
            timeout_s: float = RENDER_TIMEOUT_S) -> Performance:
    """Render *song*: words in the first voice that manages all of them, each
    syllable re-sung on its notes, the backing under it, mixed to 24 kHz PCM.
    Blocking and CPU-bound: call it off the event loop."""
    np = _np()
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    chain = list(voices) if voices is not None else voice_chain()
    if not chain:
        raise SingingError("no voice is available to sing with")
    chunks = [song.lines[i:i + LINES_PER_CALL] for i in range(0, len(song.lines), LINES_PER_CALL)]
    texts = [word_text([w.text for line in chunk for w in line]) for chunk in chunks]
    rendered: list[bytes | None] = []
    used = None
    for voice in chain:
        if time.monotonic() > deadline:
            break
        rendered = _render_words(voice, texts, deadline)
        if rendered and all(rendered):
            used = voice
            break
    if used is None:
        raise SingingError("the voice could not render the words")
    voice_done = time.perf_counter()

    sr = SAMPLE_RATE
    speech = [to_float(pcm) for pcm in rendered]         # type: ignore[arg-type]
    analyses = [analyse(x, sr) for x in speech]
    voiced = np.concatenate([a["f0"][a["f0"] > 0] for a in analyses] or [np.zeros(0)])
    voice_hz = float(np.median(voiced)) if voiced.size else 110.0
    tonic = choose_tonic(song, voice_hz)
    beat_s = 60.0 / song.bpm
    lead = 0.15

    def seconds_at(beat: float) -> float:
        return lead + (beat + song.meter) * beat_s

    total = int((seconds_at(song.end_beat + song.meter) + 1.2) * sr)
    timeline = song.syllable_timeline()
    by_position = {(li, wi, si): notes for li, wi, si, notes in timeline}

    pieces = []                                          # (buffer, base, rms, dynamic)
    sung_notes: list[tuple[float, float, float]] = []
    previous_end = 0
    previous_hz: float | None = None
    previous_stop = -10.0
    line_offset = 0
    # Seeded by the song, so the same song is sung the same way twice.
    human = random.Random(zlib.crc32(("|".join(song.text_lines) + song.mood).encode("utf-8")))
    line_starts: dict[int, int] = {}                     # line -> first output sample
    for chunk, x, analysis in zip(chunks, speech, analyses):
        words = [w for line in chunk for w in line]
        positions = [(line_offset + li, wi) for li, line in enumerate(chunk) for wi in range(len(line))]
        segmented = segment_syllables(analysis, [w.text for w in words])
        marks, periods = pitch_marks(x, analysis["f0"], sr)
        for word, (li, wi), syllables in zip(words, positions, segmented):
            line_len = len(song.lines[li])
            for si, syllable in enumerate(syllables):
                notes = by_position.get((li, wi, si))
                if not notes:
                    continue
                note_start = seconds_at(notes[0][0])
                note_end = seconds_at(notes[-1][0] + notes[-1][1])
                last_in_word = si == len(syllables) - 1
                last_in_line = last_in_word and wi == line_len - 1
                # Legato: words within a line join up as a singer joins them
                # (the 50 ms between every word was a reading, not a line).
                gap = 0.14 if last_in_line else LEGATO_WORD_GAP_S if last_in_word else 0.006
                a, b, v0, v1 = syllable
                onset = (v0 - a) if v0 is not None else 0
                lead_in = min(onset, int(0.06 * sr))
                # Onsets breathe around the beat: never metronome-exact.
                late = max(-2.2, min(2.2, human.gauss(0.0, 1.0))) * TIMING_JITTER_S
                out_start = max(previous_end, int((note_start + late) * sr) - lead_in)
                out_end = max(out_start + int(0.05 * sr), int((note_end - gap) * sr))
                note_samples = [(seconds_at(s) * sr, seconds_at(s + d) * sr, midi_hz(tonic + p))
                                for s, d, p in notes]
                glide = previous_hz if note_start - previous_stop < 0.25 else None
                style = note_style(human, scoop=glide is None or si == 0,
                                   fall=last_in_line)
                buffer, base = sing_syllable(x, marks, periods, syllable, out_start, out_end,
                                             note_samples, glide, sr, style)
                core = buffer[int(0.04 * sr):-int(0.04 * sr)] if len(buffer) > int(0.1 * sr) else buffer
                # Phrase dynamics: a line rises to its middle and eases off,
                # a downbeat leans in, a long note swells and settles.
                where = (wi + 0.5) / max(1, line_len)
                accent = 1.06 if abs((notes[0][0] % song.meter)) < 1e-6 else 1.0
                arc = 0.9 + 0.16 * math.sin(math.pi * where)
                pieces.append((buffer, base, float(np.sqrt(np.mean(core ** 2)) + 1e-9),
                               arc * accent * (0.94 if last_in_line else 1.0)))
                sung_notes.extend((s / sr, e / sr, hz) for s, e, hz in note_samples)
                line_starts.setdefault(li, out_start)
                previous_end = out_end
                previous_hz = note_samples[-1][2]
                previous_stop = note_end - gap
        line_offset += len(chunk)

    vocal = np.zeros(total, np.float32)
    if pieces:
        target = float(np.median([rms for _b, _base, rms, _d in pieces]))
        for buffer, base, rms, dynamic in pieces:
            gain = max(0.5, min(2.5, (target / rms) ** 0.8)) * dynamic
            length = len(buffer)
            if length > int(0.5 * sr):
                # A held note swells a little, then settles (messa di voce).
                u = np.linspace(0.0, 1.0, length, dtype=np.float32)
                buffer = buffer * (0.93 + 0.12 * np.sin(np.pi * np.minimum(1.0, u * 1.25)))
            s = max(0, base)
            e = min(total, base + length)
            if e > s:
                vocal[s:e] += buffer[s - base:e - base] * gain
        add_breaths(vocal, sorted(v for k, v in line_starts.items() if k > 0), human, sr)
    backing = render_backing(song, tonic, seconds_at, total, sr)
    vocal_rms = float(np.sqrt(np.mean(vocal[np.abs(vocal) > 1e-4] ** 2))) if np.any(np.abs(vocal) > 1e-4) else 0.1
    backing_rms = float(np.sqrt(np.mean(backing[np.abs(backing) > 1e-5] ** 2))) if np.any(np.abs(backing) > 1e-5) else 1.0
    backing *= BACKING_LEVEL * vocal_rms / max(backing_rms, 1e-9)
    dry = vocal + backing
    mixed = master(dry + REVERB_WET * _reverb(vocal + 1.5 * backing, sr))
    pcm = (mixed * 32767.0).astype("<i2").tobytes()
    finished = time.perf_counter()
    timings = {"voice_s": round(voice_done - started, 3), "dsp_s": round(finished - voice_done, 3),
               "total_s": round(finished - started, 3), "voice_hz": round(voice_hz, 1)}
    return Performance(song=song, pcm=pcm, seconds=round(len(mixed) / sr, 2), voice=used.name,
                       tonic=tonic, timings=timings, notes=sung_notes,
                       vocal=vocal if keep_vocal else None)


#: Where the master's soft limiter starts (about -2 dBFS); below it the song
#: passes untouched.
MASTER_KNEE = 0.79


def master(mixed):
    """Level the mix and catch its few loudest peaks - nothing else.

    It used to normalise the peak to the tanh curve's knee and push the WHOLE
    song through tanh, which saturated every loud vowel (about 8% compression
    with odd harmonics from half scale up): a constant, audible distortion.
    Now the level comes from the 99.9th percentile, everything below
    MASTER_KNEE is linear, and only the rare samples above it are rounded off."""
    np = _np()
    magnitude = np.abs(mixed)
    level = float(np.percentile(magnitude, 99.9)) if magnitude.size else 0.0
    if level <= 1e-9:
        return np.zeros_like(mixed, dtype=np.float32)
    y = (mixed * (MASTER_KNEE / level)).astype(np.float32)
    over = np.abs(y) > MASTER_KNEE
    if np.any(over):
        room = 1.0 - MASTER_KNEE
        excess = (np.abs(y[over]) - MASTER_KNEE) / room
        y[over] = np.sign(y[over]) * (MASTER_KNEE + room * np.tanh(excess))
    return np.clip(y, -0.999, 0.999)


# ── playing and keeping ──────────────────────────────────────────────────────

def play(speech: Any, pcm: bytes) -> bool:
    """Through ORION's own voice output (SpeechQueueManager's native-audio path),
    which drives the face's lip-sync and the speaking state the microphone gate
    follows, and which "Orion, stop" clears. False when there is no output or he
    is muted."""
    enqueue = getattr(speech, "enqueue_native_audio", None)
    if not callable(enqueue) or getattr(speech, "muted", False):
        return False
    for start in range(0, len(pcm), CHUNK_BYTES):
        enqueue(pcm[start:start + CHUNK_BYTES])
    finish = getattr(speech, "finish_native_audio", None)
    if callable(finish):
        finish()
    return True


def songs_folder() -> Path:
    """Music\\ORION - the owner's real Music folder (as camera_tools keeps
    Pictures\\ORION and Videos\\ORION), created on first use."""
    base: Path | None = None
    try:
        from .app_resolver import known_folder
        base = known_folder("music folder")
    except Exception:
        base = None
    folder = (base or Path.home() / "Music") / "ORION"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save(performance: Performance, folder: Path | None = None) -> Path:
    folder = folder or songs_folder()
    stem = re.sub(r"[^\w\- ]", "", performance.song.title).strip()[:50] or "Song"
    path = folder / f"{time.strftime('%Y-%m-%d %H%M')} {stem}.wav"
    from .atomic_io import atomic_write_bytes
    atomic_write_bytes(path, performance.wav_bytes())
    return path


# ── the tool ─────────────────────────────────────────────────────────────────

_LAST: dict[str, Any] = {"performance": None, "path": None}
_BUSY = threading.Lock()

_ACTIONS = {
    "sing": {"sing", "perform", "song", "compose", "write", "serenade", "sing_song", "make", "create"},
    "record": {"record", "sing_and_save", "save_song", "make_recording"},
    "save": {"save", "keep", "save_last", "export"},
    "replay": {"replay", "again", "repeat", "encore", "play", "play_again"},
    "songs": {"songs", "list", "repertoire", "known", "saved"},
    "status": {"status", "describe", "can_you_sing", "help"},
}


def _action(args: dict[str, Any]) -> str:
    raw = str(args.get("action") or "sing").strip().lower().replace(" ", "_").replace("-", "_")
    for action, names in _ACTIONS.items():
        if raw in names:
            return action
    return "sing"


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _honorific(host: Any) -> str:
    identity = getattr(host, "identity", None)
    try:
        return str(identity.preferences.get("honorific") or "sir")
    except Exception:
        return "sir"


async def plan_song(host: Any, args: dict[str, Any]) -> tuple[Song, str]:
    """The song to sing and a note on where its words came from."""
    import asyncio
    topic = " ".join(str(args.get("topic") or args.get("about") or args.get("for") or "").split())[:160]
    request = " ".join(str(args.get("song") or args.get("title") or "").split())[:120]
    style = " ".join(str(args.get("style") or args.get("mood") or "").split())[:60]
    mood = mood_for(f"{style} {topic}") if (style or topic) else "gentle"
    pd = find_public_domain(request) or (find_public_domain(topic) if not args.get("lyrics") else None)
    if pd is not None and not args.get("lyrics"):
        name = str(args.get("name") or args.get("person") or "").strip() or _honorific(host)
        return public_domain_song(pd, name), "a public-domain song"
    lyrics = parse_lyrics(args.get("lyrics")) if args.get("lyrics") else []
    title = " ".join(str(args.get("title") or "").split())[:60]
    covered = " ".join(str(args.get("song") or "").split())[:120]
    if lyrics and (_flag(args.get("cover")) or covered):
        # Words given for a named song that is not in the public domain: a
        # cover, so a snippet only, to his own tune.
        snippet = lyrics[:COVER_MAX_LINES]
        name = covered or title or "a cover"
        return (compose_melody(snippet, mood if style else "upbeat",
                               title=f"{name} (snippet)", source="a cover snippet"),
                f"a short snippet of {name}, as a cover")
    if lyrics:
        return compose_melody(lyrics, mood, title=title or "Your Song", source="your lyrics"), "your words"
    ai = getattr(host, "ai", None)
    router = getattr(ai, "router", None) or getattr(host, "router", None)
    if router is not None and hasattr(router, "generate_text"):
        try:
            lines = max(2, min(8, int(args.get("lines") or 4)))
        except (TypeError, ValueError):
            lines = 4
        try:
            _profile, text = await asyncio.wait_for(router.generate_text(
                compose_prompt(topic, mood, request, lines), instruction=COMPOSE_INSTRUCTION,
                task="sing.compose", max_tokens=400, deadline_s=COMPOSE_DEADLINE_S),
                COMPOSE_DEADLINE_S + 1.0)
            parsed = parse_composition(text)
        except Exception:
            parsed = None
        if parsed is not None:
            return compose_melody(parsed[1], mood, title=title or parsed[0]), "newly written"
    bank = original_from_bank(topic or request, mood if style else "")
    return (compose_melody(bank["lines"], mood if style else bank["mood"], title=bank["title"]),
            "one of my own (no songwriter was reachable to write a new one)")


def describe(host: Any = None) -> str:
    chain = [voice.name for voice in voice_chain()]
    voice = (f"My singing voice now: {chain[0]}" + (f", then {', '.join(chain[1:])}" if len(chain) > 1 else "")
             if chain else "My singing voice is not switched on in this run")
    known = ", ".join(spec["title"] for spec in PUBLIC_DOMAIN.values())
    own = ", ".join(song["title"] for song in ORIGINALS)
    return (f"I can sing: natively in a voice conversation, and with this tool as a produced "
            f"performance - words, a melody, my voice tuned to the notes and a soft piano "
            f"under it. {voice}. Public-domain songs I know: {known}. My own: {own}; online I "
            f"write new ones. Saved songs go to Music\\ORION.")


async def handle(host: Any, args: dict[str, Any] | None) -> Any:
    """The sing tool: sing (default), record (sing and save), save, replay, songs, status."""
    import asyncio
    from .data import ToolResult
    args = dict(args or {})
    action = _action(args)
    if action == "status":
        return ToolResult(describe(host))
    if action == "songs":
        saved = []
        if voices_on():
            try:
                saved = sorted(p.name for p in songs_folder().glob("*.wav"))[-10:]
            except OSError:
                saved = []
        return ToolResult(describe(host) + (" Saved: " + "; ".join(saved) + "." if saved else ""))
    speech = getattr(getattr(host, "live_worker", None), "speech", None)
    last: Performance | None = _LAST["performance"]
    if action == "replay":
        if last is None:
            return ToolResult("I haven't sung anything yet this session.", ok=False)
        played = await asyncio.to_thread(play, speech, last.pcm)
        return ToolResult(f"Playing '{last.song.title}' again ({last.seconds:.0f} s). Say nothing over it."
                          if played else "There is no voice output to play it through.", ok=played)
    if action == "save":
        if last is None:
            return ToolResult("There's no song to save yet - ask me to sing one first.", ok=False)
        path = await asyncio.to_thread(save, last)
        _LAST["path"] = path
        return ToolResult(f"Saved '{last.song.title}' to {path}.")
    if not voices_on():
        return ToolResult("My singing voice isn't switched on in this run, so I can't perform "
                          "the song - sing it yourself in your own voice if you can.", ok=False)
    if not _BUSY.acquire(blocking=False):
        return ToolResult("I'm already working on a song - one at a time.", ok=False)
    try:
        started = time.monotonic()
        song, origin = await plan_song(host, args)
        if not song.lines:
            return ToolResult("Those lyrics had no words I could sing.", ok=False)
        # Writing the words (up to COMPOSE_DEADLINE_S) comes out of the same
        # budget: the whole tool answers inside Live's tool window.
        budget = max(15.0, RENDER_TIMEOUT_S - (time.monotonic() - started))
        try:
            performance = await asyncio.wait_for(
                asyncio.to_thread(perform, song, timeout_s=budget), budget + 3.0)
        except asyncio.TimeoutError:
            return ToolResult("The song took too long to render, so I stopped. Sing it "
                              "natively instead:\n" + "\n".join(song.text_lines), ok=False)
        except SingingError as exc:
            return ToolResult(f"I couldn't render it ({exc}). The words, to sing yourself:\n"
                              + "\n".join(song.text_lines), ok=False)
        _LAST["performance"] = performance
        _LAST["path"] = None
        want_play = not (str(args.get("play", "")).strip().lower() in {"0", "false", "no", "off"}
                         or args.get("play") is False)
        played = await asyncio.to_thread(play, speech, performance.pcm) if want_play else False
        saved = ""
        if action == "record" or _flag(args.get("save")):
            path = await asyncio.to_thread(save, performance)
            _LAST["path"] = path
            saved = f" Saved to {path}."
        lines = " / ".join(song.text_lines)
        if played:
            lead = (f"Now singing '{song.title}' ({origin}), {performance.seconds:.0f} s, in "
                    f"{performance.key} - it is playing in your voice right now. Do not sing it "
                    "again or talk over it; afterwards, one short line at most.")
        else:
            lead = (f"Rendered '{song.title}' ({origin}), {performance.seconds:.0f} s"
                    + (", but there is no voice output to play it through" if want_play else "") + ".")
        return ToolResult(f"{lead}{saved} Lyrics: {lines}. (Voice: {performance.voice}; ready in "
                          f"{performance.timings.get('total_s', 0):.1f} s.)")
    finally:
        _BUSY.release()


__all__ = [
    "MOODS", "ORIGINALS", "PUBLIC_DOMAIN", "SINGING_DIRECTIVE", "Performance", "Song",
    "SingingError", "SungWord", "Voice", "analyse", "choose_cuts", "compose_melody",
    "count_syllables", "handle", "parse_lyrics", "perform", "pitch_marks", "play",
    "public_domain_song", "render_backing", "save", "segment_syllables", "sing_syllable",
    "songs_folder", "voice_chain",
]
