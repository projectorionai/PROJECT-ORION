"""
Viseme pipeline (Mark XXVI, Phase 3) — mouth shapes from speech, not just an
amplitude-driven jaw.

The current lip-sync opens the jaw by loudness, so every sound looks the same. A
viseme is the visible mouth POSTURE of a speech sound — lips together for /m/,
rounded for /o/, spread for /i/ — and mapping speech to visemes is what makes a
face look like it is saying words rather than merely making noise.

Fidelity depends on whether the phonemes are known ahead of the audio, so there
are two sources, both local:

  * ``SAPI_VISEME_MAP`` — Windows SAPI5 emits real ``SPEI_VISEME`` events during
    synthesis; mapping its 22 ids onto our set is the accurate, zero-inference
    path for the offline local voice.
  * ``text_to_visemes`` — a dependency-free grapheme heuristic for any TTS whose
    text is known but which emits no viseme events. Approximate, but far better
    than amplitude alone, and deterministic.

Everything here is pure data and arithmetic — no Qt, no audio device, no network
— so the whole pipeline is unit-testable. The consumer (a face) reads ``VISEMES``
for the target mouth posture and eases toward it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Mouth:
    """A normalised mouth posture. ``width`` 0.5 is neutral; >0.5 spreads, <0.5
    narrows. ``round`` 0 is flat, 1 is fully pursed."""
    open: float      # 0 closed … 1 wide
    width: float     # 0 narrow … 0.5 neutral … 1 spread
    round: float     # 0 flat … 1 pursed


#: The reduced viseme set (Oculus-style), each with its target mouth posture.
VISEMES: dict[str, Mouth] = {
    "sil": Mouth(0.05, 0.50, 0.20),   # rest
    "PP":  Mouth(0.00, 0.45, 0.30),   # p b m — lips closed
    "FF":  Mouth(0.15, 0.55, 0.15),   # f v — lower lip to teeth
    "TH":  Mouth(0.22, 0.50, 0.20),   # th dh
    "DD":  Mouth(0.28, 0.50, 0.20),   # t d n l
    "kk":  Mouth(0.32, 0.50, 0.25),   # k g ng
    "CH":  Mouth(0.22, 0.42, 0.60),   # ch j sh — pursed
    "SS":  Mouth(0.15, 0.62, 0.10),   # s z — spread, narrow gap
    "RR":  Mouth(0.30, 0.45, 0.50),   # r
    "aa":  Mouth(1.00, 0.55, 0.15),   # father — wide open
    "E":   Mouth(0.50, 0.72, 0.10),   # bet — spread
    "ih":  Mouth(0.35, 0.78, 0.05),   # bit — very spread
    "oh":  Mouth(0.60, 0.40, 0.80),   # boat — rounded
    "ou":  Mouth(0.40, 0.28, 1.00),   # boot — pursed
}
NEUTRAL = VISEMES["sil"]


# ── ARPAbet phoneme → viseme (for a real phoniser, later) ─────────────────────

PHONEME_TO_VISEME: dict[str, str] = {
    "AA": "aa", "AO": "oh", "AE": "E", "AH": "aa", "AW": "aa", "AY": "aa",
    "EH": "E", "ER": "RR", "EY": "E", "IH": "ih", "IY": "ih", "OW": "oh",
    "OY": "oh", "UH": "ou", "UW": "ou", "W": "ou", "Y": "ih", "R": "RR",
    "L": "DD", "M": "PP", "B": "PP", "P": "PP", "F": "FF", "V": "FF",
    "TH": "TH", "DH": "TH", "T": "DD", "D": "DD", "N": "DD", "K": "kk",
    "G": "kk", "NG": "kk", "CH": "CH", "JH": "CH", "SH": "CH", "ZH": "CH",
    "S": "SS", "Z": "SS", "HH": "aa",
}


#: Windows SAPI5 SP_VISEME id (0–21) → our viseme id.
SAPI_VISEME_MAP: dict[int, str] = {
    0: "sil", 1: "aa", 2: "aa", 3: "oh", 4: "E", 5: "RR", 6: "ih", 7: "ou",
    8: "oh", 9: "aa", 10: "oh", 11: "aa", 12: "aa", 13: "RR", 14: "DD",
    15: "SS", 16: "CH", 17: "TH", 18: "FF", 19: "DD", 20: "kk", 21: "PP",
}


def sapi_viseme_to_id(sapi_id: int) -> str:
    """Map a Windows SAPI viseme id onto our set; unknown ids rest the mouth."""
    return SAPI_VISEME_MAP.get(int(sapi_id), "sil")


def viseme_to_spectral(viseme_id: str, weight: float = 1.0) -> tuple[float, float, float, float]:
    """Translate a viseme posture into ``(amplitude, low, mid, high)`` for the
    3-D face's existing spectral mouth bridge (Mark X.14) — so phoneme shapes can
    drive the WebGL mouth WITHOUT any renderer change.

    openness → jaw amplitude; roundness → low-band energy (rounded lips);
    spread (width above neutral) → high-band energy (spread lips); the remainder
    is mid. Returns values in 0..1."""
    shape = VISEMES.get(str(viseme_id), NEUTRAL)
    w = max(0.0, min(1.0, float(weight)))
    amp = max(0.0, min(1.0, shape.open * w))
    spread = max(0.0, (shape.width - 0.5) * 2.0)
    low = max(0.0, min(1.0, shape.round * w))
    high = max(0.0, min(1.0, spread * w))
    mid = max(0.0, min(1.0, (1.0 - abs(shape.width - 0.5) * 2.0) * w * 0.6))
    return amp, low, mid, high


# ── grapheme → viseme (dependency-free g2p heuristic) ─────────────────────────

#: Two-letter graphemes resolved before single letters.
_DIGRAPHS = {"th": "TH", "sh": "CH", "ch": "CH", "ph": "FF", "wh": "ou",
             "ng": "kk", "qu": "kk", "ck": "kk", "oo": "ou", "ee": "ih",
             "ea": "ih", "ou": "ou", "ow": "oh", "ai": "E", "ay": "E"}
_LETTER = {
    "a": "aa", "e": "E", "i": "ih", "o": "oh", "u": "ou", "y": "ih",
    "b": "PP", "p": "PP", "m": "PP", "f": "FF", "v": "FF", "w": "ou",
    "r": "RR", "l": "DD", "t": "DD", "d": "DD", "n": "DD", "s": "SS",
    "z": "SS", "c": "SS", "k": "kk", "g": "kk", "q": "kk", "j": "CH",
    "x": "kk", "h": "aa",
}
_VOWEL_VISEMES = {"aa", "E", "ih", "oh", "ou"}
_WORD = re.compile(r"[a-z]+")


# ── language-free reduction ──────────────────────────────────────────────────
# Articulation is a property of the SOUND, not of a language, so the tables
# above are keyed on the 26 bare Latin letters and every script reaches them by
# reduction rather than by a table of its own:
#
#   * diacritics are stripped by Unicode decomposition (é→e, ü→u, ş→s, ğ→g,
#     ế→e, ñ→n, å→a …), which covers every Latin-script language at once;
#   * the handful of letters with no decomposition get an explicit entry;
#   * Cyrillic and Greek transliterate into the same 26 letters.
#
# The previous implementation matched ``[a-z]+`` and therefore produced an
# EMPTY timeline for every non-English script — ORION's mouth simply did not
# move when he spoke Turkish, Russian or Greek, and nothing reported it.
# Adding a language now costs nothing, because there is nothing to add.

#: Letters with no Latin base under NFD decomposition.
_UNDECOMPOSED = {
    "ı": "i", "ø": "o", "đ": "d", "ħ": "h", "ŀ": "l", "ŧ": "t", "ß": "s",
    "æ": "a", "œ": "o", "þ": "t", "ð": "d", "ŋ": "n", "ł": "l",
}

_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "s", "ч": "s", "ш": "s", "щ": "s", "ъ": "",
    "ы": "i", "ь": "", "э": "e", "ю": "u", "я": "a",
    "і": "i", "ї": "i", "є": "e", "ґ": "g", "ў": "u",
}

_GREEK = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i",
    "θ": "t", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "s",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "i",
    "φ": "f", "χ": "h", "ψ": "s", "ω": "o",
}

#: Below this share of reducible letters the text is in a script whose spelling
#: does not reveal pronunciation (CJK, Arabic, Devanagari, Hebrew, Thai). Miming
#: shapes onto it would be worse than not trying, so the caller falls back to
#: the audio-only mouth — which is physics, and therefore already language
#: independent. Less detail, never wrong.
_MIN_COVERAGE = 0.55

_PAUSE = frozenset(".,;:!?…")


def to_latin(ch: str) -> str:
    """Reduce any character to a bare Latin letter, or "" if it has none."""
    c = (ch or "").lower()
    if "a" <= c <= "z":
        return c
    if c in _UNDECOMPOSED:
        return _UNDECOMPOSED[c]
    if c in _CYRILLIC:
        return _CYRILLIC[c]
    if c in _GREEK:
        return _GREEK[c]
    base = "".join(k for k in unicodedata.normalize("NFD", c)
                   if not unicodedata.combining(k))
    if len(base) == 1 and "a" <= base <= "z":
        return base
    if base and base != c:                  # ligatures: ﬁ → fi, take the first
        return to_latin(base[0])
    return ""


def script_coverage(text: str) -> float:
    """Fraction of the letters in *text* reducible to a Latin sound."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if to_latin(c)) / len(letters)


def text_to_visemes(text: str) -> list[tuple[str, float]]:
    """A deterministic (viseme_id, duration_seconds) timeline for *text*.

    A grapheme heuristic rather than a true phoniser, but repeatable, instant
    and language-independent: short consonants, longer vowels, a beat between
    words. Returns [] for scripts whose written form does not reveal
    pronunciation, so the caller falls back to the audio-only mouth instead of
    miming nonsense at the viewer.
    """
    source = (text or "").lower()
    if not source:
        return []
    if script_coverage(source) < _MIN_COVERAGE:
        return []

    timeline: list[tuple[str, float]] = []
    last = ""
    i, n = 0, len(source)
    pending_word = False

    while i < n:
        ch = source[i]
        if ch in _PAUSE or ch.isspace() or not to_latin(ch):
            if pending_word:
                timeline.append(("sil", 0.08))   # a beat between words
                last = "sil"
                pending_word = False
            i += 1
            continue

        # Digraphs are an orthographic quirk; match them on the REDUCED letters
        # so accented and uppercase spellings resolve the same way.
        first = to_latin(ch)
        second = to_latin(source[i + 1]) if i + 1 < n else ""
        pair = first + second
        if len(pair) == 2 and pair in _DIGRAPHS:
            vis = _DIGRAPHS[pair]
            i += 2
        else:
            i += 1
            vis = _LETTER.get(first, "sil")
        pending_word = True
        if vis == last:                          # collapse a run of one posture
            continue
        last = vis
        timeline.append((vis, 0.12 if vis in _VOWEL_VISEMES else 0.06))

    if pending_word:
        timeline.append(("sil", 0.08))
    return timeline


# ── audio → mouth shape (formant bands) ──────────────────────────────────────

#: A 1024-sample window resolves formants at 24 kHz (~43 ms); a 20 ms hop gives
#: 50 mouth shapes a second. ORION's visualiser ran at 20 Hz on three broad
#: bands, which is smooth enough for a waveform and far too coarse for a mouth:
#: at 50 ms a plosive can begin and end inside a single frame.
_ANALYSIS_WINDOW = 1024
_ANALYSIS_HOP_S = 0.020


def pcm_shapes(pcm, sample_rate: int = 24000,
               hop_seconds: float = _ANALYSIS_HOP_S) -> list[tuple[float, float, float]]:
    """Slice PCM into (level, openness, width) frames, one per hop.

    Two numbers come out of the spectrum, and neither needs a transcript or a
    language:

      * **openness** tracks the first formant. F1 climbs as the jaw drops, so
        /a/ reads open and /i/ or /u/ read closed.
      * **width** tracks the second. F2 is high for spread vowels (/i/, /e/)
        and low for rounded ones (/u/, /o/).

    This is physics, so it behaves identically for English and Turkish. What it
    cannot do is tell you the lips are CLOSED — /m/, /b/ and /p/ look nearly
    identical to a filter bank and completely different on a face. That is what
    the transcript is for; see :class:`VisemeStream`.

    Returns [] on anything unexpected, so a caller falls back to loudness
    rather than having to handle an error mid-sentence.
    """
    try:
        import numpy as _np

        x = _np.asarray(pcm, dtype=_np.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        if x.size < 64:
            return []
        if float(_np.abs(x).max()) > 1.5:        # int16 handed in raw
            x = x / 32768.0
        hop = max(1, int(round(sample_rate * hop_seconds)))
        window = _np.hanning(_ANALYSIS_WINDOW).astype(_np.float32)
        freqs = _np.fft.rfftfreq(_ANALYSIS_WINDOW, 1.0 / sample_rate)
        f1_low = (freqs >= 150) & (freqs < 450)      # F1 of close vowels
        f1_high = (freqs >= 450) & (freqs < 1100)    # F1 of open vowels
        f2_back = (freqs >= 600) & (freqs < 1300)    # F2 of rounded vowels
        f2_front = (freqs >= 1700) & (freqs < 3200)  # F2 of spread vowels
        hiss_band = (freqs >= 3800) & (freqs < 8000)  # fricatives

        out: list[tuple[float, float, float]] = []
        # Step across the WHOLE block. Advancing only while a full window fits
        # stops short of the end, so a 200 ms batch yields 160 ms of mouth and
        # each batch drifts further behind the voice than the last.
        for start in range(0, x.size, hop):
            piece = x[start:start + hop]
            level = float(_np.sqrt(float(_np.mean(piece ** 2)))) if piece.size else 0.0
            level = min(1.0, level * 3.2)
            if level <= 0.004:
                out.append((0.0, 0.0, 0.0))
                continue
            seg = x[start:start + _ANALYSIS_WINDOW]
            if seg.size < _ANALYSIS_WINDOW:
                seg = _np.concatenate(
                    [seg, _np.zeros(_ANALYSIS_WINDOW - seg.size, dtype=_np.float32)])
            mag = _np.abs(_np.fft.rfft((seg - seg.mean()) * window))
            lo = float(mag[f1_low].sum())
            hi = float(mag[f1_high].sum())
            back = float(mag[f2_back].sum())
            front = float(mag[f2_front].sum())
            sss = float(mag[hiss_band].sum())

            openness = hi / (lo + hi + 1e-6)
            width = (front - back) / (front + back + 1e-6)
            # A wide-open jaw physically cannot purse. /a/ has a low enough F2
            # to read as "rounded" on the bands alone, and letting openness damp
            # the width term is what stops an open vowel from pursing.
            width *= (1.0 - openness) ** 0.8
            # Fricatives are formed with a nearly closed mouth.
            share = sss / (lo + hi + back + front + sss + 1e-6)
            openness *= 1.0 - 0.65 * min(1.0, share * 2.5)
            out.append((level,
                        float(min(1.0, max(0.0, openness))),
                        float(min(1.0, max(-1.0, width)))))
        return out
    except Exception:
        return []


class VisemeStream:
    """Fuses the transcript's mouth shapes onto the audio's timing.

    The transcript knows WHICH shape — including the closures no spectrum can
    see — and the audio knows WHEN and HOW HARD. Feeding text is optional: with
    nothing queued the stream passes the audio-only shape straight through,
    which is what ORION did before and is never wrong, only less detailed.
    """

    #: Seconds a posture occupies at a normal speaking rate. The clock moves
    #: between these as the backlog grows or drains, so a fast turn catches up
    #: instead of drifting further behind the voice with every phrase.
    _MIN_STEP = 0.045
    _MAX_STEP = 0.105
    _MAX_BACKLOG = 600

    def __init__(self) -> None:
        self._queue: deque = deque()
        self._current = ("sil", 1.0)
        self._carry = 0.0

    def reset(self) -> None:
        self._queue.clear()
        self._current = ("sil", 1.0)
        self._carry = 0.0

    @property
    def pending(self) -> int:
        return len(self._queue)

    def feed_text(self, text: str) -> None:
        """Queue the mouth shapes for a line ORION is about to say."""
        for viseme_id, duration in text_to_visemes(text):
            self._queue.append((viseme_id, max(0.25, duration / 0.09)))
        # A stalled turn must not pile up an unbounded backlog.
        while len(self._queue) > self._MAX_BACKLOG:
            self._queue.popleft()

    def _step_seconds(self) -> float:
        backlog = min(1.0, len(self._queue) / 45.0)
        return self._MAX_STEP - (self._MAX_STEP - self._MIN_STEP) * backlog

    def frames(self, audio, hop: float) -> list[tuple[float, float, float]]:
        """Blend audio frames [(level, openness, width)] with the text queue."""
        out: list[tuple[float, float, float]] = []
        for level, audio_open, audio_wide in audio:
            if level <= 0.0:
                # Silence: let the queue WAIT rather than burning through it
                # during a pause, or the mouth ends up ahead of the voice.
                out.append((0.0, 0.0, 0.0))
                continue

            self._carry += hop / max(1e-3, self._step_seconds() * self._current[1])
            while self._carry >= 1.0 and self._queue:
                self._current = self._queue.popleft()
                self._carry -= 1.0
            if self._carry >= 1.0:
                self._carry = 1.0               # queue empty — hold the shape

            posture = VISEMES.get(self._current[0], NEUTRAL)
            if self._queue or self._current[0] != "sil":
                target_open = posture.open
                target_wide = (posture.width - 0.5) * 2.0
                # A closure is the one thing loudness must never override.
                closure = 1.0 if self._current[0] == "PP" else 0.0
                # Text leads the shape; the audio keeps it honest, so a bad
                # alignment still tracks the real voice rather than drifting
                # off into a scripted mime.
                blended_open = (0.72 * target_open + 0.28 * audio_open) * (1.0 - closure)
                blended_wide = 0.78 * target_wide + 0.22 * audio_wide
            else:
                blended_open, blended_wide = audio_open, audio_wide
            out.append((level,
                        max(0.0, min(1.0, blended_open)),
                        max(-1.0, min(1.0, blended_wide))))
        return out


# ── scheduler (walk a timeline against elapsed time) ─────────────────────────

class VisemeScheduler:
    """Advances a (viseme, duration) timeline against a clock, yielding the
    current viseme so a face can ease toward it. Co-articulation smoothing is the
    face's job (it eases); this only decides *which* posture is current."""

    def __init__(self, timeline: list[tuple[str, float]]) -> None:
        self._timeline = list(timeline)
        # Cumulative end-time of each segment.
        self._ends: list[float] = []
        t = 0.0
        for _vis, dur in self._timeline:
            t += max(0.0, dur)
            self._ends.append(t)
        self.total = t

    def at(self, elapsed: float) -> str:
        """The viseme id active at *elapsed* seconds, or 'sil' past the end."""
        if not self._timeline or elapsed < 0:
            return "sil"
        for (vis, _dur), end in zip(self._timeline, self._ends):
            if elapsed < end:
                return vis
        return "sil"

    def done(self, elapsed: float) -> bool:
        return elapsed >= self.total


__all__ = [
    "Mouth", "VISEMES", "NEUTRAL", "PHONEME_TO_VISEME", "SAPI_VISEME_MAP",
    "sapi_viseme_to_id", "viseme_to_spectral", "text_to_visemes", "VisemeScheduler",
    "to_latin", "script_coverage", "pcm_shapes", "VisemeStream",
]
