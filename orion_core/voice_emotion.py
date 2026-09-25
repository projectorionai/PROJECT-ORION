"""
Hearing HOW something was said, not just what.

"I want ORION to be able to distinguish between my different emotions such as
anger, frustration, excitement, happiness, scared and etc."

The approach, and why
---------------------
Emotion is read from PROSODY — the acoustics of the utterance — with the words
used only as corroboration. That ordering is deliberate and it is the whole
point of the feature. A text classifier reading "it's fine" cannot tell calm
from clenched, and those are the two cases where being read correctly actually
matters. Loudness, pitch height, pitch variability, speech rate and voice
quality are what carry that, and they are all measurable directly.

Five acoustic features, each chosen because it separates states the others
confuse:

  * **energy** (RMS, in dB) — the loudest single cue. Anger and excitement are
    loud; fear and sadness are quiet.
  * **pitch height** (median F0, normalised against the speaker's own baseline)
    — arousal raises it. Normalised, because a 180 Hz utterance means opposite
    things from different people, which is exactly why an absolute threshold
    would misread anyone it was not tuned on.
  * **pitch variability** (F0 standard deviation) — this is what separates the
    two loud, high states: excitement swoops, anger stays flat and hard.
    Without it, "I'm THRILLED" and "I'm FURIOUS" look identical.
  * **speech rate** (voiced-segment transitions per second) — frustration and
    excitement are fast, sadness is slow.
  * **spectral centroid** — brightness/harshness. Fear and anger push energy
    into the higher partials; contentment does not.

Scored rather than classified: each state gets a score from how well the
features match its profile, and the winner is reported with a confidence and,
importantly, with WHAT DROVE IT ("louder and flatter than usual"). A number
alone would be unfalsifiable — ORION should be able to say why he thinks you
sound annoyed, so you can tell him he is wrong.

What this is not
----------------
It is not a trained model and does not claim one's accuracy. It reports a
confidence, and everything above LOW is meant to be treated as a hint about
tone rather than a fact about a person's inner state. Acting confidently on a
weak read is worse than not reading at all — being told you sound angry when
you are concentrating is precisely the kind of thing that makes a system
tiring to live with.

Baselines are per speaker and learned over time (see :class:`EmotionBaseline`),
because "loud for you" is the only version of loud that means anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

STATE_FILE = CONFIG_DIR / "voice_emotion.json"

#: Sample rate the feature thresholds below assume.
NOMINAL_SR = 16000

#: Below this many seconds there is not enough signal to say anything.
MIN_SECONDS = 0.35


@dataclass
class Features:
    """The measured acoustics of one utterance."""

    seconds: float = 0.0
    energy_db: float = -60.0
    pitch_hz: float = 0.0
    pitch_spread: float = 0.0
    rate: float = 0.0
    centroid_hz: float = 0.0
    voiced_fraction: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "seconds": round(self.seconds, 2),
            "energy_db": round(self.energy_db, 1),
            "pitch_hz": round(self.pitch_hz, 1),
            "pitch_spread": round(self.pitch_spread, 1),
            "rate": round(self.rate, 2),
            "centroid_hz": round(self.centroid_hz, 1),
            "voiced_fraction": round(self.voiced_fraction, 2),
        }


@dataclass
class Reading:
    """What ORION thinks he heard, and why."""

    emotion: str = "neutral"
    confidence: float = 0.0
    because: str = ""
    features: Features = field(default_factory=Features)
    runner_up: str = ""
    speaker: str = ""

    @property
    def certain_enough_to_mention(self) -> bool:
        """Below this, saying anything is worse than saying nothing.

        Guessing at someone's emotional state out loud and being wrong is
        actively unpleasant, so the bar to VOICE a reading is higher than the
        bar to record one.
        """
        return self.confidence >= 0.55 and self.emotion != "neutral"

    def describe(self) -> str:
        if self.emotion == "neutral":
            return "You sound level."
        strength = ("clearly" if self.confidence >= 0.75
                    else "fairly" if self.confidence >= 0.6 else "possibly")
        return f"You sound {strength} {self.emotion} - {self.because}."


# ── the profiles ─────────────────────────────────────────────────────────────
#
# Each is expressed in units of "deviation from this speaker's own baseline",
# so they carry across voices. Values are (target, weight): the target is where
# that feature sits for the emotion in z-like units, the weight is how much
# that feature actually distinguishes it.
#
# The distinctions that these encode, and that matter most in practice:
#   anger vs excitement  — both loud and high; anger is FLAT (low spread),
#                          excitement is variable.
#   frustration vs anger — frustration is nearer baseline loudness but fast and
#                          tight; anger is loud.
#   scared vs excited    — both high and fast; fear is QUIET and bright.
# Each entry is (target, weight, mode). ``mode`` is what stops an extreme
# utterance from matching every milder profile as well as its own:
#
#   "min"  — at least this far in this direction; overshoot is free. Right for
#            the features that DEFINE an intense state (anger is loud, and
#            furious is more so).
#   "at"   — about this far, in either direction. Right for features that
#            BOUND a state. Frustration is near-normal loudness: someone
#            shouting has stopped being frustrated and started being angry,
#            and without a bound the shout scores perfectly on both. Same for
#            calm, which means near normal in both directions — a whisper
#            overshot every one of its targets and read as calm.
_PROFILES: dict[str, dict[str, tuple[float, float, str]]] = {
    "angry": {
        "energy": (1.3, 1.4, "min"), "pitch": (0.8, 0.9, "min"),
        "spread": (-0.4, 1.2, "at"), "rate": (0.4, 0.6, "min"),
        "centroid": (0.7, 0.8, "min"),
    },
    "frustrated": {
        # Bounded loudness and pitch: this is the state's whole distinction
        # from anger.
        "energy": (0.4, 1.2, "at"), "pitch": (0.4, 0.8, "at"),
        "spread": (-0.4, 1.0, "at"), "rate": (1.0, 1.1, "min"),
        "centroid": (0.4, 0.5, "at"),
    },
    "excited": {
        # Loudness heavily weighted: it is the ONLY thing separating this from
        # happiness, which is also high and lilting but not loud. Weighted
        # lightly, a pleasant quiet remark read as excitement.
        "energy": (1.1, 1.9, "min"), "pitch": (1.1, 1.2, "min"),
        "spread": (1.2, 1.5, "min"), "rate": (1.0, 1.0, "min"),
        "centroid": (0.5, 0.5, "min"),
    },
    "happy": {
        "energy": (0.4, 0.9, "at"), "pitch": (1.0, 0.8, "at"),
        "spread": (0.9, 1.2, "min"), "rate": (0.3, 0.6, "at"),
        "centroid": (0.3, 0.5, "at"),
    },
    "scared": {
        "energy": (-0.4, 1.0, "at"), "pitch": (1.3, 1.3, "min"),
        "spread": (0.6, 0.7, "at"), "rate": (0.9, 0.8, "min"),
        "centroid": (1.0, 1.0, "min"),
    },
    "sad": {
        # Pitch heavily weighted and required to drop a long way: this is what
        # separates sadness from tiredness, which is equally quiet and slow but
        # does not take the pitch down with it.
        "energy": (-1.0, 1.2, "min"), "pitch": (-1.1, 1.5, "min"),
        "spread": (-0.6, 0.9, "min"), "rate": (-0.9, 1.2, "min"),
        "centroid": (-0.5, 0.7, "min"),
    },
    "tired": {
        # Distinguished from sadness by being flat and slow WITHOUT the pitch
        # dropping as far — bounded pitch is what separates them.
        "energy": (-0.8, 1.0, "min"), "pitch": (-0.4, 0.9, "at"),
        "spread": (-0.7, 1.0, "min"), "rate": (-0.8, 1.0, "min"),
        "centroid": (-0.7, 0.8, "min"),
    },
    "calm": {
        # Everything bounded: calm IS "near normal", not "beyond normal".
        "energy": (0.0, 0.9, "at"), "pitch": (0.0, 0.9, "at"),
        "spread": (0.0, 0.7, "at"), "rate": (0.0, 0.8, "at"),
        "centroid": (0.0, 0.7, "at"),
    },
}

#: Plain-language reasons, so a reading is always explainable.
_CUES = {
    "energy": ("louder than usual", "quieter than usual"),
    "pitch": ("higher-pitched than usual", "lower-pitched than usual"),
    "spread": ("with a lot more variation", "flatter than usual"),
    "rate": ("and speaking faster", "and speaking slower"),
    "centroid": ("with a harder edge", "with a softer tone"),
}


# ── feature extraction ───────────────────────────────────────────────────────

def extract(audio: Any, sample_rate: int = NOMINAL_SR) -> Features:
    """Measure one utterance. Never raises on odd input — returns empty
    Features, which scores as neutral."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).flatten()
    if samples.size == 0 or sample_rate <= 0:
        return Features()
    seconds = samples.size / float(sample_rate)
    # NOT peak-normalised. An earlier version divided by the peak on the
    # reasoning that microphone gain is not emotion — which deleted loudness
    # entirely: five deliberately different test utterances (shouted, whispered,
    # level) all measured within 1 dB of each other, and energy is the single
    # strongest cue there is. Gain is handled where it belongs, by the
    # per-speaker baseline, which tracks a changed mic level within a few dozen
    # utterances instead of throwing the feature away every time.
    # DC offset does have to go — a biased signal inflates RMS.
    samples = samples - float(np.mean(samples))
    features = Features(seconds=seconds)

    frame = max(256, int(sample_rate * 0.025))
    hop = max(128, int(sample_rate * 0.010))
    if samples.size < frame:
        return features

    count = 1 + (samples.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(count, frame),
        strides=(samples.strides[0] * hop, samples.strides[0])).copy()
    windowed = frames * np.hanning(frame).astype(np.float32)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    # Voiced frames only, judged against this utterance's own loudest frames —
    # an absolute gate would call a quiet recording entirely silent.
    gate = max(float(np.max(rms)) * 0.18, 1e-4)
    voiced = rms > gate
    features.voiced_fraction = float(np.mean(voiced))
    if not np.any(voiced):
        return features

    features.energy_db = float(20.0 * np.log10(float(np.mean(rms[voiced])) + 1e-12))

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    centroid = (spectrum * freqs).sum(axis=1) / spectrum.sum(axis=1)
    features.centroid_hz = float(np.mean(centroid[voiced]))

    pitches = _pitch_track(frames[voiced], sample_rate)
    if pitches.size:
        features.pitch_hz = float(np.median(pitches))
        features.pitch_spread = float(np.std(pitches))

    # Speech rate as voiced-onset density: how often speech starts after a gap.
    transitions = int(np.count_nonzero(np.diff(voiced.astype(np.int8)) == 1))
    features.rate = transitions / max(seconds, 0.2)
    return features


def _pitch_track(frames: Any, sample_rate: int) -> Any:
    """Per-frame F0 by autocorrelation, in the human speech range.

    Autocorrelation rather than an FFT peak because the fundamental is often
    weaker than its harmonics in speech — especially over a phone or a cheap
    microphone — and picking the loudest bin lands on a harmonic and reports
    double the true pitch.
    """
    import numpy as np

    if frames.size == 0:
        return np.array([])
    lo = int(sample_rate / 400.0)          # 400 Hz ceiling
    hi = int(sample_rate / 60.0)           # 60 Hz floor
    if hi <= lo or frames.shape[1] <= hi:
        return np.array([])
    out = []
    for row in frames:
        row = row - row.mean()
        norm = float(np.dot(row, row))
        if norm < 1e-9:
            continue
        correlation = np.correlate(row, row, mode="full")[len(row) - 1:]
        window = correlation[lo:hi]
        if window.size == 0:
            continue
        best = int(np.argmax(window))
        # Reject frames with no real periodicity — unvoiced sounds (s, f, sh)
        # produce a noisy autocorrelation whose peak means nothing.
        if window[best] / norm < 0.3:
            continue
        out.append(sample_rate / float(best + lo))
    return np.array(out, dtype=np.float64)


# ── per-speaker baselines ────────────────────────────────────────────────────

class EmotionBaseline:
    """What NORMAL sounds like, per speaker, learned over time.

    Everything above is expressed relative to this. Without it the system is
    tuned to one voice and misreads every other one — which would defeat the
    "recognise my mum's voice" half of the same feature, since she would be
    measured against the user's baseline and read as permanently quiet or
    permanently high-pitched.

    Updated with an exponential moving average so it tracks a cold or a noisy
    room within a few dozen utterances, and only from readings that came out
    near-neutral: learning "normal" from shouting would gradually redefine
    shouting as normal.
    """

    ALPHA = 0.06
    KEYS = ("energy_db", "pitch_hz", "pitch_spread", "rate", "centroid_hz")
    #: Sensible starting spreads, so the first few utterances are not wild.
    DEFAULT_SCALE = {
        "energy_db": 6.0, "pitch_hz": 35.0, "pitch_spread": 20.0,
        "rate": 1.2, "centroid_hz": 500.0,
    }

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else STATE_FILE
        self._data: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            self._data = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def observations(self, speaker: str) -> int:
        return int(self._data.get(speaker or "you", {}).get("_n", 0))

    def normalise(self, features: Features, speaker: str = "you") -> dict[str, float]:
        """Features as deviations from this speaker's normal."""
        entry = self._data.get(speaker or "you", {})
        out: dict[str, float] = {}
        for key, short in (("energy_db", "energy"), ("pitch_hz", "pitch"),
                           ("pitch_spread", "spread"), ("rate", "rate"),
                           ("centroid_hz", "centroid")):
            value = getattr(features, key)
            mean = entry.get(key)
            if mean is None or value == 0:
                out[short] = 0.0
                continue
            scale = max(entry.get(f"{key}_scale", 0.0), self.DEFAULT_SCALE[key] * 0.4)
            out[short] = max(-3.0, min(3.0, (value - mean) / scale))
        return out

    def observe(self, features: Features, speaker: str = "you") -> None:
        """Fold a near-neutral utterance into this speaker's normal."""
        if features.seconds < MIN_SECONDS or features.voiced_fraction < 0.15:
            return
        key = speaker or "you"
        entry = self._data.setdefault(key, {"_n": 0})
        n = int(entry.get("_n", 0))
        alpha = 1.0 if n == 0 else self.ALPHA
        for field_name in self.KEYS:
            value = float(getattr(features, field_name))
            if value == 0:
                continue
            mean = entry.get(field_name, value)
            # Spread is measured against the mean AS IT WAS, not as it becomes.
            # Measuring against the updated mean subtracts out most of the very
            # deviation being measured, so the estimate shrinks every time it is
            # updated: after 50 varied utterances every scale had collapsed to
            # its floor, which made ordinary speech saturate the ±3 clamp and
            # read as extreme.
            deviation = abs(value - mean)
            entry[field_name] = (1 - alpha) * mean + alpha * value
            scale = entry.get(f"{field_name}_scale", self.DEFAULT_SCALE[field_name])
            entry[f"{field_name}_scale"] = max(
                (1 - alpha) * scale + alpha * max(deviation, 1e-3),
                self.DEFAULT_SCALE[field_name] * 0.4)
        entry["_n"] = n + 1
        if entry["_n"] % 10 == 0:
            self.save()

    def forget(self, speaker: str) -> bool:
        if speaker in self._data:
            del self._data[speaker]
            self.save()
            return True
        return False


# ── scoring ──────────────────────────────────────────────────────────────────

def _shortfall(observed: float, target: float, mode: str) -> float:
    """Distance from a profile's target, one-sided for "min", symmetric for "at".

    The one-sided case is the substance of the scoring rather than a detail of
    it. Emotion varies in intensity: a profile says anger is louder than
    normal, and someone genuinely furious is far louder than normal. Measuring
    symmetrically treats that as a mismatch and scores real anger BELOW a mild
    reading of it — which is what the first version did, and why five
    deliberately-constructed utterances with textbook-correct feature vectors
    all came back wrong.

    But one-sidedness alone then lets any extreme utterance match every milder
    profile too, since it overshoots all of their targets for free. Anger read
    as frustration and a whisper read as calm for exactly that reason. "at" is
    the correction: the features that BOUND a state are measured both ways.

    Direction is enforced throughout, which is what keeps the states apart:
    anger's profile wants pitch flatter than normal, so a swooping utterance is
    penalised by how much it swoops and excitement wins it; fear wants quiet,
    so a shout can never read as fear however high-pitched.
    """
    if mode == "at":
        return abs(observed - target)
    if target > 0:
        return max(0.0, target - observed)
    if target < 0:
        return max(0.0, observed - target)
    return abs(observed)


def _reason(deviations: dict[str, float]) -> str:
    """The two features that moved most, in plain words."""
    ranked = sorted(deviations.items(), key=lambda kv: -abs(kv[1]))
    parts = []
    for name, value in ranked[:2]:
        if abs(value) < 0.45 or name not in _CUES:
            continue
        parts.append(_CUES[name][0 if value > 0 else 1])
    return " ".join(parts) if parts else "from the overall tone"


class EmotionReader:
    """Reads emotion from utterances, learning each speaker's normal as it goes."""

    def __init__(self, baseline: EmotionBaseline | None = None) -> None:
        self.baseline = baseline or EmotionBaseline()

    def read(self, audio: Any, sample_rate: int = NOMINAL_SR,
             speaker: str = "you", learn: bool = True) -> Reading:
        """Read one utterance."""
        features = extract(audio, sample_rate)
        return self.read_features(features, speaker=speaker, learn=learn)

    def read_features(self, features: Features, speaker: str = "you",
                      learn: bool = True) -> Reading:
        reading = Reading(features=features, speaker=speaker or "you")
        if features.seconds < MIN_SECONDS or features.voiced_fraction < 0.12:
            reading.because = "too little speech to judge"
            return reading

        observed = self.baseline.observations(speaker)
        deviations = self.baseline.normalise(features, speaker)

        scores: dict[str, float] = {}
        for emotion, profile in _PROFILES.items():
            total = weight_total = 0.0
            for key, (target, weight, mode) in profile.items():
                gap = _shortfall(deviations.get(key, 0.0), target, mode)
                total += weight * math.exp(-0.5 * (gap / 1.0) ** 2)
                weight_total += weight
            scores[emotion] = total / weight_total if weight_total else 0.0

        # Nothing moved far from normal → level, whatever the profiles prefer.
        movement = max(abs(v) for v in deviations.values()) if deviations else 0.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        second, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)

        # One standard deviation. Below that, an utterance is inside the range
        # this speaker's ordinary speech already covers, and calling it an
        # emotion is reading tea leaves — a plain baseline utterance was coming
        # back "frustrated, 0.54" at a lower bar.
        if movement < 1.0 or best == "calm":
            reading.emotion = "neutral"
            reading.confidence = 0.0
            reading.because = "nothing much moved from your normal"
        else:
            reading.emotion = best
            # Confidence is the MARGIN over the runner-up, not the raw score:
            # a state that scores 0.9 while its neighbour scores 0.88 has not
            # been distinguished from it, and reporting 0.9 there would be a
            # lie about how much was actually established.
            margin = best_score - second_score
            reading.confidence = max(0.0, min(1.0, best_score * (0.55 + 1.8 * margin)))
            reading.runner_up = second
            reading.because = _reason(deviations)

        # A speaker ORION has barely heard has no meaningful "normal" yet, so
        # every reading is against a half-guessed baseline. Say so by holding
        # confidence down rather than by staying silent — the readings still
        # accumulate, they just do not get spoken.
        if observed < 12:
            reading.confidence *= 0.45 + 0.045 * observed
            if reading.because and observed < 6:
                reading.because += " (still learning how you normally sound)"

        if learn and reading.emotion in ("neutral", "calm"):
            self.baseline.observe(features, speaker)
        return reading

    def describe(self) -> str:
        speakers = [s for s in self.baseline._data if not s.startswith("_")]
        if not speakers:
            return ("I haven't built a baseline for anyone's voice yet, so I'm "
                    "not reading tone.")
        parts = [f"{name} ({self.baseline.observations(name)} samples)"
                 for name in speakers]
        return ("I read tone from loudness, pitch, pitch variation, pace and "
                "brightness, against how each person normally sounds. "
                "Baselines so far: " + ", ".join(parts) + ".")


__all__ = [
    "MIN_SECONDS", "NOMINAL_SR", "STATE_FILE", "EmotionBaseline",
    "EmotionReader", "Features", "Reading", "extract",
]
