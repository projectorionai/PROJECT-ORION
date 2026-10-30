"""
Human-like typing engine.

The autonomous control layer already has a *reliable* bulk-write path
(``control.edit_text`` — native UIA → clipboard → keyboard, verified) whose job
is to get large text into a control without garbling.  This module is the
opposite: a way to make ORION type the way a *person* does — one key at a time,
at a configurable words-per-minute, with realistic timing jitter and the
occasional typo that gets noticed and corrected.

The design keeps the *planning* (which keystrokes, in what order, with what
delays) as a **pure, deterministic, seedable function** — so it is fully
unit-testable without a keyboard, a screen, or a real clock.  The control layer
just executes the plan (press a key, sleep, press backspace, …).

Nothing here bypasses the SecuritySanitiser — the caller guards the text before
planning, exactly as the existing typing paths do.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# QWERTY neighbours — a realistic typo lands on an adjacent key, not a random one.
_NEIGHBOURS: dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

MIN_WPM = 20.0
MAX_WPM = 600.0          # 20 ms a key: still a key rate Notepad keeps up with

#: Bounds on any one gap between keys. The floor keeps the key rate one that
#: Win11 Notepad does not drop or repeat characters at (see control.edit_text);
#: the ceiling caps a "thinking" pause.
MIN_KEY_S = 0.015
MAX_KEY_S = 1.2

#: The commonest English letter pairs: typed as one rolling motion, so the
#: second key follows the first faster than average.
_COMMON_BIGRAMS = frozenset(
    "th he in er an re on at en nd ti es or te of ed is it al ar st to nt ng "
    "se ha as ou io le ve co me de hi ri ro ic ne ea ra ce li ch ll be ma si "
    "om ur wh ow ly".split())
_SENTENCE_END = frozenset(".!?")
_CLAUSE_END = frozenset(",;:")
#: Characters typed with Shift on a UK/US layout (besides capitals).
_SHIFTED = frozenset('~!@#$%^&*()_+{}|:"<>?£')


@dataclass(frozen=True)
class TypingProfile:
    """How human the typing should feel."""
    wpm: float = 90.0            # words per minute (1 word ≈ 5 chars)
    jitter: float = 0.35         # spread of per-key timing (0 = metronome)
    typo_rate: float = 0.0       # probability a letter is mistyped then fixed
    seed: int | None = None      # set for reproducible plans (tests / demos)
    pause_rate: float = 0.04     # chance of a brief pause at a word boundary
    fast: bool = False           # machine speed: no delays, no typos

    def base_delay(self) -> float:
        """Seconds per character for the configured WPM (clamped to sane bounds)."""
        wpm = min(MAX_WPM, max(MIN_WPM, float(self.wpm)))
        return 60.0 / (wpm * 5.0)


@dataclass(frozen=True)
class Keystroke:
    """One planned action. kind ∈ {'char','typo','backspace'}."""
    kind: str
    char: str        # the character to send ('' for backspace)
    delay: float     # seconds to wait AFTER sending this key


def key_rhythm(ch: str, following: str) -> float:
    """How much longer (>1) or shorter (<1) than average the gap after
    typing *ch* is, when *following* comes next.

    A common bigram rolls off the fingers; a doubled letter is one finger
    tapping twice; a space is followed by the start of a new word, a comma by
    a breath and a full stop by the next thought; reaching for Shift or the
    number row costs a little."""
    factor = 1.0
    pair = (ch + following).lower()
    if pair in _COMMON_BIGRAMS:
        factor *= 0.72
    elif ch.isalpha() and ch.lower() == following.lower():
        factor *= 0.85
    if ch in _SENTENCE_END:
        factor *= 2.2
    elif ch in _CLAUSE_END:
        factor *= 1.7
    elif ch == "\n":
        factor *= 2.5
    elif ch == " ":
        factor *= 1.15
    if following.isupper() or following in _SHIFTED:
        factor *= 1.2
    elif following.isdigit() and not ch.isdigit():
        factor *= 1.15
    return factor


def keystroke_delays(text: str, profile: TypingProfile,
                     rng: random.Random | None = None) -> list[float]:
    """Seconds to wait after each character of *text*.

    Each gap is the WPM's average gap shaped by key_rhythm, with log-normal
    spread (``jitter``; real inter-key times are right-skewed), plus a rare
    0.25-0.7 s pause at a word boundary (``pause_rate``). The whole run is
    then rescaled so it still takes what the WPM says — the rhythm moves time
    between keys rather than adding it — and every gap is kept within
    MIN_KEY_S..MAX_KEY_S. ``fast``: all zeros."""
    if profile.fast:
        return [0.0] * len(text)
    rng = rng if rng is not None else random.Random(profile.seed)
    base = profile.base_delay()
    sigma = max(0.0, float(profile.jitter)) * 0.7
    keys: list[float] = []
    pauses: list[float] = []
    for i, ch in enumerate(text):
        following = text[i + 1] if i + 1 < len(text) else ""
        gap = key_rhythm(ch, following)
        if sigma > 0:
            gap *= math.exp(rng.gauss(-sigma * sigma / 2.0, sigma))   # mean 1
        keys.append(gap)
        boundary = ch in " \n" or ch in _SENTENCE_END or ch in _CLAUSE_END
        pause = 0.0
        if following and boundary and profile.pause_rate > 0 and rng.random() < profile.pause_rate:
            pause = rng.uniform(0.25, 0.7)
        pauses.append(pause)
    if not keys:
        return []
    budget = base * len(keys)
    # Pauses may take at most a quarter of the time the WPM allows.
    paused = sum(pauses)
    if paused > budget * 0.25:
        pauses = [p * budget * 0.25 / paused for p in pauses]
        paused = budget * 0.25
    scale = (budget - paused) / sum(keys)
    return [min(MAX_KEY_S, max(MIN_KEY_S, k * scale + p)) for k, p in zip(keys, pauses)]


def plan_keystrokes(text: str, profile: TypingProfile) -> list[Keystroke]:
    """
    Turn ``text`` into a realistic keystroke sequence.  Deterministic when
    ``profile.seed`` is set.  Newlines/whitespace are typed verbatim; a mistyped
    letter is emitted as a wrong key, then a backspace, then the correct key —
    so the control layer replays a believable self-correction.  Gaps between
    keys come from keystroke_delays; ``profile.fast`` plans no delays and no
    typos.
    """
    rng = random.Random(profile.seed)
    delays = keystroke_delays(text, profile, rng)
    base = profile.base_delay()
    plan: list[Keystroke] = []
    for ch, delay in zip(text, delays):
        lower = ch.lower()
        can_typo = (not profile.fast and profile.typo_rate > 0
                    and lower in _NEIGHBOURS
                    and rng.random() < profile.typo_rate)
        if can_typo:
            wrong = rng.choice(_NEIGHBOURS[lower])
            if ch.isupper():
                wrong = wrong.upper()
            plan.append(Keystroke("typo", wrong, delay))
            # Noticing the mistake takes longer than typing it.
            plan.append(Keystroke("backspace", "",
                                  min(MAX_KEY_S, max(MIN_KEY_S, base * rng.uniform(1.5, 3.0)))))
        plan.append(Keystroke("char", ch, delay))
    return plan


def plan_duration(plan: list[Keystroke]) -> float:
    """Total wall-clock time the plan will take (sum of delays)."""
    return sum(k.delay for k in plan)


#: A comfortable, clearly-human pace for short text.
NATURAL_WPM = 140.0
#: Longest a visible typing run should take before it speeds up to fit.
NATURAL_TARGET_S = 20.0


def natural_wpm(n_chars: int, *, base: float = NATURAL_WPM,
                target_s: float = NATURAL_TARGET_S) -> float:
    """A pace you can watch: ~140 wpm for a sentence, faster for a paragraph so
    it lands in about twenty seconds, never above MAX_WPM."""
    words = max(1.0, float(n_chars) / 5.0)
    needed = words / (target_s / 60.0)
    return float(min(MAX_WPM, max(base, needed)))
