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


@dataclass(frozen=True)
class TypingProfile:
    """How human the typing should feel."""
    wpm: float = 90.0            # words per minute (1 word ≈ 5 chars)
    jitter: float = 0.35         # ± fraction of the base delay, per keystroke
    typo_rate: float = 0.0       # probability a letter is mistyped then fixed
    seed: int | None = None      # set for reproducible plans (tests / demos)

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


def _delay(rng: random.Random, base: float, jitter: float) -> float:
    if jitter <= 0:
        return base
    lo, hi = base * (1.0 - jitter), base * (1.0 + jitter)
    return max(0.0, rng.uniform(lo, hi))


def plan_keystrokes(text: str, profile: TypingProfile) -> list[Keystroke]:
    """
    Turn ``text`` into a realistic keystroke sequence.  Deterministic when
    ``profile.seed`` is set.  Newlines/whitespace are typed verbatim; a mistyped
    letter is emitted as a wrong key, then a backspace, then the correct key —
    so the control layer replays a believable self-correction.
    """
    rng = random.Random(profile.seed)
    base = profile.base_delay()
    plan: list[Keystroke] = []
    for ch in text:
        lower = ch.lower()
        can_typo = (profile.typo_rate > 0
                    and lower in _NEIGHBOURS
                    and rng.random() < profile.typo_rate)
        if can_typo:
            wrong = rng.choice(_NEIGHBOURS[lower])
            if ch.isupper():
                wrong = wrong.upper()
            plan.append(Keystroke("typo", wrong, _delay(rng, base, profile.jitter)))
            # A brief "notice the mistake" pause before the backspace.
            plan.append(Keystroke("backspace", "", _delay(rng, base * 1.5, profile.jitter)))
        plan.append(Keystroke("char", ch, _delay(rng, base, profile.jitter)))
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
