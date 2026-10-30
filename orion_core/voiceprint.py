"""
Knowing whose voice it is.

The question this answers is "is that SampleUser, or someone else in the room" —
and the answer is not frequencies. Pitch tells you roughly how large someone's
vocal folds are, which is why the gender classifier can use it, but it does
not distinguish two people of similar build and it changes when you have a
cold, when you shout, or when you are tired. What does distinguish people is
the shape of the whole vocal tract across a whole utterance, and that is what
a trained speaker encoder measures: it maps a few seconds of speech to a point
in a 256-dimensional space where the same person's utterances land close
together and different people's land apart, regardless of what was said.

ORION already had such an encoder. He could not use it.
-----------------------------------------------------
``voice_speaker_id`` wraps resemblyzer's pretrained encoder, which is the
right model — but resemblyzer runs on torch, torch is 527 MB, and the packaged
build therefore excludes it. So the capability existed on a developer machine
and was absent from the application actually installed. Worse, nothing acted
on the result: presence was recorded and used to colour the tone of a reply,
but no part of ORION ever decided anything differently because of who spoke.

This module removes both problems. The encoder is the same published network
with the same trained weights, exported once to ONNX (5.8 MB) and run under
onnxruntime; given identical input it agrees with the original to 1.7e-06.
The mel front end it expects is reimplemented in numpy and matches librosa's
to a relative error of 1.5e-07.

What is NOT identical is the preprocessing. resemblyzer trims silence with
webrtcvad; this uses a smoothed energy threshold, so the two feed slightly
different frames to the same network and their embeddings of one clip agree
to about 0.6-0.7 rather than 1.0. That does not matter, because enrolment and
recognition both go through THIS pipeline and are compared only with each
other — but it does mean a profile enrolled by one cannot be read by the
other, and the claim to check is separation, not agreement. Measured on
voices with different vocal tracts: same-voice pairs 0.94-1.00, different-
voice pairs 0.40-0.64. The threshold sits in a gap of 0.29.

Failing open, on purpose
------------------------
``is_owner`` returns True when it is not confident. A speaker check that
wrongly rejects its owner makes ORION deaf — he sits there while you talk at
him, and the failure looks like a broken microphone, which is a miserable
thing to debug. Wrongly accepting a stranger costs one unwanted reply. Those
are not comparable, so the doubt always resolves towards listening.

One person, several points (profile version 2)
----------------------------------------------
A single averaged voiceprint sits between how you sound quietly at night and
how you sound calling across a room, and is a little far from both. A guided
enrolment (``voice_enrolment``) therefore stores the mean AND up to three
sub-centroids found by spherical k-means over its twenty-odd samples; a new
utterance scores against the nearest of them. It also stores how well the
samples agree with each other (leave-one-out), and from that a personal
threshold that may only ever be LOWER than the global one, and never below
``MIN_PERSONAL_THRESHOLD``: a loose speaker is not locked out, and a tight
one is not made stricter than the margin the rest of the system assumes.

Every decision comes out of :func:`decide`, which says owner, known (another
enrolled person), unknown, uncertain or unreadable. None of these verdicts can
authorise anything: they decide whether ORION answers and how he addresses
the speaker. The only consequence a verdict can have for an action is the
existing guard in ``speaker_gate``, which can REFUSE a spoken go-ahead.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .constants import CONFIG_DIR, resource_path

#: What the encoder was trained on. None of these are free parameters.
SAMPLE_RATE = 16000
N_FFT = 400                 # 25 ms
HOP = 160                   # 10 ms
N_MELS = 40
PARTIAL_FRAMES = 160        # 1.6 s per partial utterance
TARGET_DBFS = -30.0

#: Cosine similarity above which two utterances are called the same person.
#:
#: Chosen from how this encoder's embeddings are distributed rather than as a
#: round number: same-speaker pairs sit around 0.8 and different-speaker pairs
#: around 0.5, so the gap is wide and the midpoint is forgiving in the
#: direction that matters. ORION_VOICE_THRESHOLD overrides it.
DEFAULT_THRESHOLD = 0.70

#: How far below a threshold a score must fall to count as confidently
#: somebody else. The band in between is "uncertain", which listens.
REJECT_MARGIN = 0.08

#: The floor for a personal threshold. A profile may relax its own bar for a
#: speaker whose samples agree loosely, but never below this.
MIN_PERSONAL_THRESHOLD = 0.62

#: A non-owner's profile must beat the owner's by this much before an
#: utterance counts as that person rather than as a doubtful owner — so an
#: enrolled relative with a similar voice can never silence the owner.
KNOWN_MARGIN = 0.05

#: Sub-centroids kept per profile, and how many samples each one needs.
MAX_CENTROIDS = 3
SAMPLES_PER_CENTROID = 6

PROFILE_VERSION = 2
BACKEND = "voiceprint-onnx"

#: Below this, an utterance is too short to characterise a speaker. Shorter
#: clips do produce an embedding, but a noisy one, and a noisy embedding
#: rejected at threshold is exactly the failure this module must not have.
MIN_SECONDS = 1.2

#: The enrolled voices themselves — the SAME file voice_speaker_id writes, so
#: enrolling once is enough. Two stores would have been the obvious way to
#: write this and the wrong one: you would enrol your voice, ORION would
#: confirm it, and the gate would still have nothing to compare against.
PROFILES_PATH = CONFIG_DIR / "voice_profiles.json"

#: Which of them is you, and whether to ignore everyone else. Kept apart from
#: the embeddings because it is a preference, not data, and because that file
#: has a shape and a set of tests that predate this module.
SETTINGS_PATH = CONFIG_DIR / "voiceprints.json"

#: Kept as an alias: it named the settings file before the two were split.
PROFILE_PATH = SETTINGS_PATH

MODEL_PATH_PARTS = ("assets", "voiceprint", "encoder.onnx")

_LOCK = threading.Lock()
_SESSION: Any = None
_FILTERS: Any = None
_WINDOW: Any = None

#: The last few embeddings, keyed on the audio itself. The offline path reads
#: every utterance twice — once for presence and tone, once for the gate — and
#: each read was a full encoder pass over identical samples.
_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_CACHE_SIZE = 8
_CACHE_LOCK = threading.Lock()


# ── the front end: audio to mel spectrogram, in plain numpy ──────────────────

def _hz_to_mel(hz: Any) -> Any:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above it."""
    import numpy as np

    # atleast_1d: a 0-d array cannot be assigned into by a boolean mask, and
    # the endpoints of the filterbank are passed as scalars.
    hz = np.atleast_1d(np.asarray(hz, dtype=float))
    mel = hz / (200.0 / 3)
    high = hz >= 1000.0
    mel[high] = 15.0 + np.log(hz[high] / 1000.0) / (np.log(6.4) / 27.0)
    return mel


def _mel_to_hz(mel: Any) -> Any:
    import numpy as np

    mel = np.atleast_1d(np.asarray(mel, dtype=float))
    hz = (200.0 / 3) * mel
    high = mel >= 15.0
    hz[high] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel[high] - 15.0))
    return hz


def _mel_filters() -> Any:
    """The filterbank, built once.

    Slaney normalisation gives each filter equal AREA rather than equal peak.
    Without it the low filters dominate and every embedding shifts, which
    would not raise an error anywhere — it would just quietly make everyone
    look alike.
    """
    global _FILTERS
    if _FILTERS is not None:
        return _FILTERS
    import numpy as np

    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, 1 + N_FFT // 2)
    points = np.linspace(float(_hz_to_mel(0.0)[0]),
                         float(_hz_to_mel(SAMPLE_RATE / 2.0)[0]), N_MELS + 2)
    freqs = _mel_to_hz(points)

    weights = np.zeros((N_MELS, 1 + N_FFT // 2))
    widths = np.diff(freqs)
    ramps = freqs.reshape(-1, 1) - fft_freqs.reshape(1, -1)
    for i in range(N_MELS):
        weights[i] = np.maximum(
            0, np.minimum(-ramps[i] / widths[i], ramps[i + 2] / widths[i + 1]))
    weights *= (2.0 / (freqs[2:N_MELS + 2] - freqs[:N_MELS]))[:, np.newaxis]
    _FILTERS = weights.astype("float32")
    return _FILTERS


def mel_spectrogram(wav: Any) -> Any:
    """The transform the encoder expects, to a relative error of 1.5e-07.

    Reimplemented rather than imported: librosa brings numba and soundfile,
    which is a great deal of machinery to bundle for forty filter
    coefficients and an FFT.
    """
    global _WINDOW
    import numpy as np

    if _WINDOW is None:
        _WINDOW = np.hanning(N_FFT + 1)[:-1].astype("float32")

    wav = np.asarray(wav, dtype=np.float32).flatten()
    padded = np.pad(wav, N_FFT // 2, mode="constant")
    frames = 1 + (len(padded) - N_FFT) // HOP
    if frames < 1:
        return np.zeros((0, N_MELS), dtype=np.float32)
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    windowed = padded[index] * _WINDOW
    power = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1)) ** 2
    return (power @ _mel_filters().T).astype(np.float32)


def _normalise_volume(wav: Any) -> Any:
    """Bring the clip to a fixed loudness.

    The encoder is not loudness-invariant, and the same person at arm's length
    and across the room is otherwise two different points in the space.
    """
    import numpy as np

    rms = float(np.sqrt(np.mean(np.square(wav * 32767.0))))
    if rms <= 0:
        return wav
    change = TARGET_DBFS - 20.0 * float(np.log10(rms / 32767.0))
    return wav * (10.0 ** (change / 20.0))


def _trim_silence(wav: Any) -> Any:
    """Drop the quiet parts.

    Silence is the same for everybody, so leaving it in drags every embedding
    towards a common centre and narrows the gap the threshold has to find.
    Energy-based and smoothed over 0.2 s, which is enough for this purpose and
    avoids a dependency that would have to be bundled.
    """
    import numpy as np

    if wav.size < HOP * 4:
        return wav
    frames = wav.size // HOP
    energy = np.square(wav[:frames * HOP].reshape(frames, HOP)).mean(axis=1)
    if not energy.size:
        return wav
    width = max(1, int(0.2 * SAMPLE_RATE / HOP))
    smoothed = np.convolve(energy, np.ones(width) / width, mode="same")
    # Relative to the clip's own loud parts, so it works at any recording gain.
    keep = smoothed > (float(smoothed.max()) * 0.02)
    if not keep.any():
        return wav
    voiced = np.repeat(keep, HOP)
    trimmed = wav[:voiced.size][voiced]
    return trimmed if trimmed.size >= HOP * 4 else wav


def _resample(wav: Any, source_rate: int) -> Any:
    """To 16 kHz. Linear, because the encoder's own band stops at 8 kHz and
    the difference between interpolators is far below what it can see."""
    import numpy as np

    if source_rate == SAMPLE_RATE or wav.size == 0:
        return wav
    duration = wav.size / float(source_rate)
    target = int(round(duration * SAMPLE_RATE))
    if target < 2:
        return wav
    return np.interp(np.linspace(0.0, duration, target, endpoint=False),
                     np.linspace(0.0, duration, wav.size, endpoint=False),
                     wav).astype(np.float32)


# ── the encoder ──────────────────────────────────────────────────────────────

def model_path() -> Path:
    return resource_path(*MODEL_PATH_PARTS)


def available() -> bool:
    """Whether a voiceprint can be computed at all on this machine."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return model_path().is_file()


def _session() -> Any:
    """The ONNX session, built once and shared.

    Single-threaded deliberately: this runs beside real-time audio, and an
    inference that grabs every core to save eight milliseconds is how the
    voice pipeline starts stuttering.
    """
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            from . import accelerator

            # On the CPU by policy (accelerator.POLICY): the GPU saves under
            # 3 ms a run and its session build holds Python's lock ~300 ms.
            _SESSION = accelerator.create_session(
                "voiceprint", model_path(), threads=1, inter_threads=1)
    return _SESSION


def embed(wav: Any, sample_rate: int = SAMPLE_RATE) -> Any:
    """A 256-dimensional voiceprint, L2-normalised. None if it cannot be made.

    The utterance is split into overlapping 1.6-second partials and their
    embeddings averaged, which is what the encoder was trained to expect and
    what makes the result depend on the speaker rather than on the sentence.
    """
    import numpy as np

    if not available():
        return None
    wav = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).flatten())
    if wav.size == 0:
        return None
    key = (int(sample_rate), wav.size,
           hashlib.blake2b(wav.tobytes(), digest_size=16).digest())
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached.copy()
    vector = _embed_uncached(wav, int(sample_rate))
    if vector is not None:
        with _CACHE_LOCK:
            _CACHE[key] = vector
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        vector = vector.copy()
    return vector


def clear_cache() -> None:
    """Forget the cached embeddings (after "forget my voice")."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _embed_uncached(wav: Any, sample_rate: int) -> Any:
    import numpy as np

    if np.abs(wav).max() > 1.5:             # int16 handed in as floats
        wav = wav / 32768.0
    wav = _resample(wav, int(sample_rate))

    # Silence is not a speaker. Without this the encoder is handed a flat
    # signal, returns some arbitrary point in the space, and a quiet room
    # scores as "a different voice" — which would make ORION stop listening
    # precisely when nobody is talking, and stay that way.
    if float(np.sqrt(np.mean(np.square(wav)))) < 1e-4:
        return None

    wav = _trim_silence(_normalise_volume(wav))
    if wav.size < MIN_SECONDS * SAMPLE_RATE * 0.5:
        return None

    mels = mel_spectrogram(wav)
    if mels.shape[0] < PARTIAL_FRAMES:
        # Too short for a whole partial: pad by repeating rather than with
        # silence, which would pull the embedding towards everyone else's.
        repeats = int(np.ceil(PARTIAL_FRAMES / max(1, mels.shape[0])))
        mels = np.tile(mels, (repeats, 1))[:PARTIAL_FRAMES]

    step = PARTIAL_FRAMES // 2
    starts = list(range(0, mels.shape[0] - PARTIAL_FRAMES + 1, step)) or [0]
    batch = np.stack([mels[s:s + PARTIAL_FRAMES] for s in starts]).astype(
        np.float32)
    try:
        embeds = _session().run(["embeds"], {"mels": batch})[0]
    except Exception:
        return None
    mean = embeds.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm) if norm > 0 else None


def similarity(a: Any, b: Any) -> float:
    """Cosine similarity of two voiceprints, in [-1, 1]."""
    import numpy as np

    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size != b.size or not a.size:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


# ── profiles: several points per person ──────────────────────────────────────

def _unit(vector: Any) -> Any:
    import numpy as np

    if vector is None:
        return None
    v = np.asarray(vector, dtype=np.float32).flatten()
    if not np.isfinite(v).all():
        return None
    norm = float(np.linalg.norm(v)) if v.size else 0.0
    return (v / norm) if norm > 0 else None


def _centroids(stack: Any, k: int) -> Any:
    """Spherical k-means over unit vectors. Deterministic: seeded from the
    sample the mean describes WORST, then each next-farthest — so the
    conditions an average blurs (very quiet, very loud) get their own point."""
    import numpy as np

    mean = _unit(stack.mean(axis=0))
    chosen = [int(np.argmin(stack @ mean))]
    while len(chosen) < k:
        nearest = np.max(stack @ stack[chosen].T, axis=1)
        chosen.append(int(np.argmin(nearest)))
    centres = stack[chosen].copy()
    for _ in range(12):
        labels = np.argmax(stack @ centres.T, axis=1)
        updated = np.stack([
            _unit(stack[labels == j].mean(axis=0)) if np.any(labels == j)
            else centres[j] for j in range(k)])
        if np.allclose(updated, centres, atol=1e-6):
            break
        centres = updated
    return centres


def _profile_points(stack: Any) -> list:
    """The mean, plus sub-centroids once there are enough samples for them."""
    mean = _unit(stack.mean(axis=0))
    k = min(MAX_CENTROIDS, stack.shape[0] // SAMPLES_PER_CENTROID)
    points = [mean]
    if k >= 2:
        points.extend(list(_centroids(stack, k)))
    return points


def _score_points(vector: Any, points: list) -> float:
    import numpy as np

    v = _unit(vector)
    if v is None:
        return 0.0
    usable = [p for p in points if p is not None and p.size == v.size]
    return max((float(np.dot(v, p)) for p in usable), default=0.0)


def profile_vectors(record: Any) -> list:
    """The unit vectors a stored profile scores against: its mean, and its
    sub-centroids when it has them (version 1 profiles have only the mean)."""
    if not isinstance(record, dict):
        return []
    points = [_unit(record.get("embedding"))]
    for centre in record.get("centroids") or []:
        points.append(_unit(centre))
    return [p for p in points if p is not None]


def score_profile(vector: Any, record: Any) -> float:
    """Cosine similarity of *vector* to the nearest point of a profile."""
    return _score_points(vector, profile_vectors(record))


def _personal_threshold(self_scores: list) -> float:
    """A bar the speaker's own samples clear: the 5th percentile of how each
    scored against a profile built WITHOUT it, less a little room for the
    conditions an enrolment session never sees. Bounded on both sides — see
    profile_threshold — so it can only relax the global bar, never tighten
    it."""
    import numpy as np

    if len(self_scores) < 6:
        return DEFAULT_THRESHOLD
    p05 = float(np.percentile(np.asarray(self_scores, dtype=float), 5))
    return max(MIN_PERSONAL_THRESHOLD, min(DEFAULT_THRESHOLD, p05 - 0.03))


def profile_threshold(record: Any, default: float = DEFAULT_THRESHOLD) -> float:
    """The accept threshold for one profile: its personal one, never above
    *default* and never below MIN_PERSONAL_THRESHOLD."""
    try:
        personal = float(record.get("threshold"))
    except Exception:
        return default
    return max(MIN_PERSONAL_THRESHOLD, min(default, personal))


def build_profile(embeddings: Any, styles: Any = None) -> dict | None:
    """A stored profile from several embeddings of ONE person.

    Holds embeddings only — never audio. ``embedding`` stays the plain mean,
    so anything that reads a version 1 profile still reads this one.
    """
    import numpy as np

    samples = [] if embeddings is None else embeddings
    vectors = [v for v in (_unit(e) for e in samples) if v is not None]
    if not vectors:
        return None
    stack = np.stack(vectors)
    points = _profile_points(stack)
    record: dict[str, Any] = {
        "embedding": [float(x) for x in points[0]],
        "clips": len(vectors),
        "sample_count": len(vectors),
        "backend": BACKEND,
        "version": PROFILE_VERSION,
    }
    if len(points) > 1:
        record["centroids"] = [[float(x) for x in p] for p in points[1:]]
    if len(vectors) >= 3:
        # Leave-one-out: how each sample scores against the profile the
        # others would have built. Scoring a sample against a profile it is
        # part of flatters every speaker equally and says nothing.
        self_scores = []
        for i in range(len(vectors)):
            rest = np.delete(stack, i, axis=0)
            self_scores.append(_score_points(stack[i], _profile_points(rest)))
        scores = np.asarray(self_scores, dtype=float)
        record["self_similarity"] = float(scores.min())
        record["spread"] = {"mean": round(float(scores.mean()), 4),
                            "std": round(float(scores.std()), 4),
                            "p05": round(float(np.percentile(scores, 5)), 4)}
        record["threshold"] = round(_personal_threshold(self_scores), 4)
    else:
        record["self_similarity"] = float(min(
            similarity(v, points[0]) for v in vectors))
    if styles:
        counts: dict[str, int] = {}
        for style in styles:
            counts[str(style)] = counts.get(str(style), 0) + 1
        record["styles"] = counts
    return record


def _comparable(record: Any) -> bool:
    """A profile from a different encoder scores meaninglessly against this
    one's embeddings, so it is skipped rather than compared."""
    if not isinstance(record, dict) or not record.get("embedding"):
        return False
    backend = str(record.get("backend") or BACKEND)
    return backend == BACKEND


def resolve_owner(profiles: dict, nominated: str = "") -> str:
    """Whose voice is the owner's: the nominated one if enrolled, else the
    single enrolled voice, else nobody (see Voiceprints.owner)."""
    if nominated and nominated in (profiles or {}):
        return nominated
    names = list(profiles or {})
    return names[0] if len(names) == 1 else ""


# ── enrolment ────────────────────────────────────────────────────────────────

#: What a verdict can say about who spoke.
KINDS = ("owner", "known", "unknown", "uncertain", "unreadable", "nobody")


@dataclass
class Verdict:
    """What the check concluded, and how sure it is.

    ``score`` is the similarity to the OWNER's voice (to the best match when
    nobody is the owner); ``match`` is the similarity to ``speaker``. ``kind``
    is one of KINDS; verdicts made before it existed leave it empty, and
    :attr:`category` infers it.
    """

    is_owner: bool
    score: float = 0.0
    speaker: str = ""
    reason: str = ""
    confident: bool = False
    kind: str = ""
    match: float = 0.0

    def __bool__(self) -> bool:
        return self.is_owner

    @property
    def category(self) -> str:
        if self.kind:
            return self.kind
        if not self.is_owner:
            return "unknown" if self.confident else "uncertain"
        return "owner" if self.confident else "uncertain"


def _env_threshold() -> float | None:
    raw = os.getenv("ORION_VOICE_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return None


def decide(vector: Any, profiles: dict, owner: str = "",
           threshold: float | None = None) -> Verdict:
    """Whose voice *vector* is, among *profiles*, and whether it is the owner.

    owner      the owner's profile clears its threshold (and beats any other)
    known      another enrolled person clears theirs AND beats the owner's
               score by KNOWN_MARGIN
    unknown    nobody clears, and the owner's score is below their threshold
               by more than REJECT_MARGIN
    uncertain  anything in between — which listens, like every doubt here
    unreadable no voiceprint could be made
    nobody     nothing comparable is enrolled

    ``is_owner`` is False only for a confident known or unknown voice while an
    owner exists: that is the one case a gate may act on.
    """
    if vector is None:
        return Verdict(True, reason="not enough speech to tell", kind="unreadable")
    explicit = threshold if threshold is not None else _env_threshold()
    entries = []
    for name, record in (profiles or {}).items():
        if not _comparable(record):
            continue
        limit = explicit if explicit is not None else profile_threshold(record)
        entries.append((score_profile(vector, record), str(name), limit))
    if not entries:
        return Verdict(True, reason="nobody is enrolled", kind="nobody")

    scores = {name: (score, limit) for score, name, limit in entries}
    owner = owner if owner in scores else ""
    passing = sorted((e for e in entries if e[0] >= e[2]), reverse=True)

    if not owner:
        # Nobody is the owner, so there is nothing for a gate to protect:
        # identify if possible, and always listen.
        if passing:
            score, name, _ = passing[0]
            return Verdict(True, score, name, "recognised", confident=True,
                           kind="known", match=score)
        best = max(entries)[0]
        return Verdict(True, best, "", "not a voice ORION knows",
                       confident=False, kind="unknown")

    owner_score, owner_limit = scores[owner]
    if passing:
        score, name, _ = passing[0]
        if name == owner:
            return Verdict(True, score, owner, "recognised", confident=True,
                           kind="owner", match=score)
        if score - owner_score >= KNOWN_MARGIN:
            return Verdict(False, owner_score, name, f"recognised as {name}",
                           confident=True, kind="known", match=score)
        # Close to the owner AND to someone else. Treated as the owner's
        # doubtful turn, never as a reason to ignore them.
        return Verdict(True, owner_score, owner,
                       f"close to both {owner} and {name}, so assumed to be you",
                       confident=False, kind="uncertain", match=owner_score)
    if owner_score < owner_limit - REJECT_MARGIN:
        return Verdict(False, owner_score, "", "a different voice",
                       confident=True, kind="unknown")
    return Verdict(True, owner_score, owner, "uncertain, so assumed to be you",
                   confident=False, kind="uncertain", match=owner_score)


@dataclass
class Voiceprints:
    """Enrolled voices, and which of them is the person ORION works for.

    Two files, deliberately. The embeddings live where voice_speaker_id
    already writes them, so enrolling through either route is enough for
    both; the preferences live beside them. Storage is JSON because that is
    genuinely what this is — a handful of 256-float vectors read once at
    start-up. The heavier stores exist for data that grows.

    Each file is parsed once per change, keyed on its modification time and
    size: the audio loop asks whether a gate is on for every chunk of speech,
    and re-reading both files each time was dozens of reads a second on the
    capture thread. Another part of ORION enrolling a voice still shows at
    once, because writing a file changes its stamp.
    """

    path: Path = field(default_factory=lambda: SETTINGS_PATH)
    profiles_path: Path = field(default_factory=lambda: PROFILES_PATH)

    #: path -> ((mtime_ns, size), parsed contents). Shared by every instance.
    _parsed: ClassVar[dict[Path, tuple[tuple[int, int], dict]]] = {}

    # -- persistence ---------------------------------------------------------
    @classmethod
    def _read(cls, path: Path) -> dict:
        """The file's JSON object, as a fresh top-level copy (callers add
        and delete entries; records inside are replaced, never edited)."""
        try:
            info = path.stat()
        except OSError:
            cls._parsed.pop(path, None)
            return {}
        stamp = (info.st_mtime_ns, info.st_size)
        cached = cls._parsed.get(path)
        if cached is None or cached[0] != stamp:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
            cached = (stamp, data if isinstance(data, dict) else {})
            cls._parsed[path] = cached
        return dict(cached[1])

    def _load(self) -> dict:
        settings = self._read(self.path)
        settings.setdefault("owner", "")
        settings.setdefault("only_owner", False)
        settings.setdefault("guard_actions", False)
        settings["profiles"] = self._read(self.profiles_path)
        return settings

    def _save(self, data: dict) -> None:
        """Write the preferences. The embeddings are not ours to rewrite."""
        payload = {k: v for k, v in data.items() if k != "profiles"}
        _write_json(self.path, payload)
        self._parsed.pop(self.path, None)

    def _save_profiles(self, profiles: dict) -> None:
        _write_json(self.profiles_path, profiles)
        self._parsed.pop(self.profiles_path, None)

    # -- reading -------------------------------------------------------------
    @property
    def owner(self) -> str:
        """Whose voice ORION treats as the user's.

        With nobody nominated and exactly ONE voice enrolled, that voice is
        the owner. Enrolling through the older ``voice_speaker_id`` tool does
        not nominate anyone, so without this you could enrol your voice, be
        told it worked, ask ORION to answer only you, and be told to enrol a
        voice first — which you just had.

        With two or more enrolled it stays empty on purpose: guessing which
        of several people the machine belongs to is not a guess worth making
        silently, and the ``owner`` action exists to say.
        """
        data = self._load()
        nominated = str(data.get("owner") or "")
        if nominated:
            return nominated
        names = list(data["profiles"])
        return names[0] if len(names) == 1 else ""

    def names(self) -> list[str]:
        return sorted(self._load()["profiles"])

    def enrolled(self, name: str = "") -> bool:
        profiles = self._load()["profiles"]
        return bool(profiles.get(name or self.owner))

    @property
    def only_owner(self) -> bool:
        """Whether ORION should ignore voices that are not his owner's.

        Off unless asked for. On, a wrong rejection makes him appear deaf,
        and that is a bad thing to discover by accident.
        """
        return bool(self._load().get("only_owner"))

    def set_only_owner(self, enabled: bool) -> tuple[bool, str]:
        data = self._load()
        if enabled and not self.enrolled(self.owner):
            return False, ("no voice is enrolled yet, so ORION would have "
                           "nothing to compare against and would ignore "
                           "everyone. Enrol a voice first.")
        data["only_owner"] = bool(enabled)
        self._save(data)
        return True, ("ORION will now answer only your voice."
                      if enabled else
                      "ORION will answer anyone again.")

    @property
    def guard_actions(self) -> bool:
        """Whether a SPOKEN go-ahead for a sensitive action (send, call,
        delete, pay) must come in the owner's voice. See speaker_gate."""
        return bool(self._load().get("guard_actions"))

    def set_guard_actions(self, enabled: bool) -> tuple[bool, str]:
        data = self._load()
        if enabled and not self.enrolled(self.owner):
            return False, ("no voice is enrolled yet, so there is nothing to "
                           "check a go-ahead against. Enrol a voice first.")
        data["guard_actions"] = bool(enabled)
        self._save(data)
        return True, ("Sensitive actions confirmed by voice now need your voice."
                      if enabled else
                      "Anyone's spoken go-ahead is accepted again.")

    @property
    def judging(self) -> bool:
        """Whether each utterance's speaker should be judged.

        True while a gate is on, and ALSO whenever the owner's voice is
        enrolled: ORION then needs to know on every turn whether it is the
        owner talking, to know whether to use their name. This is the switch
        the live capture path reads (audio.py ``_speaker_judged``), so it is
        what makes identification reach a Live session at all.

        Read on the capture thread for every voiced chunk, so it costs one
        cached load and nothing else.
        """
        data = self._load()
        if data.get("only_owner") or data.get("guard_actions"):
            return True
        profiles = data["profiles"]
        owner = resolve_owner(profiles, str(data.get("owner") or ""))
        return bool(owner and profiles.get(owner))

    def set_owner(self, name: str) -> tuple[bool, str]:
        data = self._load()
        if name not in data["profiles"]:
            return False, f"no voiceprint is stored for {name}."
        data["owner"] = name
        self._save(data)
        return True, f"ORION will treat {name}'s voice as yours."

    def threshold(self) -> float:
        raw = os.getenv("ORION_VOICE_THRESHOLD", "").strip()
        try:
            return max(0.0, min(1.0, float(raw))) if raw else DEFAULT_THRESHOLD
        except ValueError:
            return DEFAULT_THRESHOLD

    # -- writing -------------------------------------------------------------
    def enrol(self, name: str, clips: list, sample_rate: int = SAMPLE_RATE,
              owner: bool = True) -> tuple[bool, str]:
        """Learn a voice from one or more clips.

        Several short clips beat one long one: the average of embeddings taken
        at different moments is less tied to one sentence, one mood and one
        distance from the microphone.
        """
        name = str(name or "").strip()
        if not name:
            return False, "a name is needed to enrol a voice."
        if not available():
            return False, ("the voice encoder is not available "
                           f"({model_path().name} is missing).")
        embeddings = [e for e in (embed(clip, sample_rate) for clip in clips)
                      if e is not None]
        if not embeddings:
            return False, ("none of those clips held enough speech — about "
                           f"{MIN_SECONDS:.0f} seconds of talking is needed.")
        return self.enrol_embeddings(name, embeddings, owner=owner)

    def enrol_embeddings(self, name: str, embeddings: list, *, owner: bool = True,
                         styles: Any = None, method: str = "clips") -> tuple[bool, str]:
        """Store a profile built from embeddings already computed — the guided
        enrolment computes them one utterance at a time as they arrive, and
        never keeps the audio."""
        name = str(name or "").strip()
        if not name:
            return False, "a name is needed to enrol a voice."
        record = build_profile(embeddings, styles)
        if record is None:
            return False, "the clips produced no usable voiceprint."
        # How tightly the samples agree is stored with the profile, and from
        # it a personal threshold (see build_profile). The backend stamp is
        # what voice_speaker_id checks before comparing: a profile from a
        # different encoder is skipped rather than scored meaninglessly.
        record["enrolled_at"] = time.time()
        record["method"] = method
        data = self._load()
        profiles = self._read(self.profiles_path)    # fresh, never a stale copy
        profiles[name] = record
        self._save_profiles(profiles)
        if owner or not data.get("owner"):
            data["owner"] = name
        self._save(data)
        count = int(record.get("clips") or 0)
        return True, (f"{name}'s voice is enrolled from {count} "
                      f"clip(s)" + (" — ORION will treat this as your voice."
                                    if data['owner'] == name else "."))

    def forget(self, name: str) -> tuple[bool, str]:
        """Delete a voiceprint. Forgetting the OWNER also switches off both
        voice gates: with nothing to compare against they would only report
        themselves as on while doing nothing."""
        data = self._load()
        if name not in data["profiles"]:
            return False, f"no voiceprint is stored for {name}."
        was_owner = (name == self.owner)
        profiles = self._read(self.profiles_path)
        profiles.pop(name, None)
        self._save_profiles(profiles)
        if data.get("owner") == name:
            data["owner"] = ""
        if was_owner:
            data["only_owner"] = False
            data["guard_actions"] = False
        self._save(data)
        return True, f"{name}'s voiceprint is forgotten."

    # -- the actual question -------------------------------------------------
    def verdict_for(self, vector: Any) -> Verdict:
        """Judge an embedding already computed (see :func:`decide`)."""
        data = self._load()
        profiles = data["profiles"]
        owner = resolve_owner(profiles, str(data.get("owner") or ""))
        return decide(vector, profiles, owner, _env_threshold())

    def identify(self, wav: Any, sample_rate: int = SAMPLE_RATE) -> Verdict:
        """Whose voice this is, among those enrolled."""
        if not self._load()["profiles"]:
            return Verdict(True, reason="nobody is enrolled", kind="nobody")
        vector = embed(wav, sample_rate)
        if vector is None:
            return Verdict(True, reason="not enough speech to tell",
                           kind="unreadable")
        return self.verdict_for(vector)

    def is_owner(self, wav: Any, sample_rate: int = SAMPLE_RATE) -> Verdict:
        """Whether this is the person ORION works for.

        Fails OPEN. When no owner is enrolled, when the clip is too short,
        when the encoder is unavailable, or when the score sits near the
        threshold, the answer is yes — because a check that wrongly rejects
        its owner makes ORION deaf, and that failure looks like broken
        hardware rather than a misjudged voice.
        """
        data = self._load()
        # The PROPERTY, not the raw field. Enrolling through the older
        # voice_speaker_id tool nominates nobody, and the property is what
        # treats a single enrolled voice as the owner. Reading the field here
        # meant set_only_owner succeeded — it uses the property — while this,
        # the one method the gate actually calls, always answered "no owner
        # enrolled". The gate would have been on and doing nothing.
        owner = self.owner
        record = data["profiles"].get(owner)
        if not record:
            return Verdict(True, reason="no owner voice is enrolled", kind="nobody")
        if not available():
            return Verdict(True, reason="the voice encoder is unavailable",
                           kind="unreadable")

        vector = embed(wav, sample_rate)
        if vector is None:
            return Verdict(True, reason="not enough speech to tell",
                           kind="unreadable")
        # Only a clear miss counts as somebody else. The margin is the whole
        # safety property: without it, every borderline utterance silences
        # him. decide() keeps it, and adds the other enrolled voices.
        return decide(vector, data["profiles"], owner, _env_threshold())

    def describe(self) -> str:
        data = self._load()
        if not available():
            return ("Voice recognition is unavailable: "
                    f"{model_path().name} or onnxruntime is missing.")
        names = sorted(data["profiles"])
        if not names:
            return ("No voice is enrolled yet. Say \"learn my voice\" (or use "
                    "Preferences) and read about twenty short lines - about "
                    "five minutes - and ORION will know your voice from "
                    "anyone else's.")
        owner = self.owner or "nobody"
        record = data["profiles"].get(self.owner) or {}
        limit = _env_threshold()
        limit = limit if limit is not None else profile_threshold(record)
        return (f"Enrolled: {', '.join(names)}. ORION treats {owner} as your "
                f"voice, at a similarity threshold of {limit:.2f}.")


def _write_json(path: Path, payload: Any) -> None:
    """Write via a temporary file and rename.

    A half-written profile would be read back as no profile at all, and with
    the gate on that means ORION stops answering his owner.
    """
    from .atomic_io import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")


#: The process-wide store.
_STORE: Voiceprints | None = None


def store() -> Voiceprints:
    global _STORE
    if _STORE is None:
        _STORE = Voiceprints()
    return _STORE


def is_owner(wav: Any, sample_rate: int = SAMPLE_RATE) -> Verdict:
    return store().is_owner(wav, sample_rate)


def judge(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> Verdict:
    """Whose voice this 16-bit mono PCM is, while any gate needs to know.

    Unlike ``should_listen`` this answers for the action guard too, and it
    records the verdict for ``speaker_gate``. With every gate off nothing is
    computed and the verdict is a plain yes.
    """
    import numpy as np

    from . import speaker_gate

    prints = store()
    if not prints.judging:
        return Verdict(True, reason="no voice gate is on")
    if _learning():
        # Whoever is talking during a guided enrolment is the voice being
        # learned. Judging them against an older profile could mute the very
        # samples the new one is built from.
        return Verdict(True, reason="learning a voice right now",
                       kind="unreadable")
    try:
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return Verdict(True, reason="the audio could not be read")
    verdict = prints.is_owner(wav, sample_rate)
    speaker_gate.note_verdict(verdict)
    try:
        from . import voice_enrolment

        # Part-way through the utterance (the live path asks at 1.6 s), so
        # it is recorded as provisional; the whole utterance replaces it.
        voice_enrolment.turns().record(verdict, stage="partial",
                                       seconds=wav.size / float(sample_rate or 1))
    except Exception:
        pass
    return verdict


def _learning() -> bool:
    try:
        from . import voice_enrolment

        return voice_enrolment.learning()
    except Exception:
        return False


def should_listen(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> Verdict:
    """Whether ORION should act on this utterance, given 16-bit mono PCM.

    Always True unless the gate is on AND the voice is confidently not his
    owner's. Every other path - gate off, nobody enrolled, encoder missing,
    too little speech, a borderline score - resolves to listening.
    """
    import numpy as np

    prints = store()
    if not prints.only_owner:
        return Verdict(True, reason="ORION answers anyone")
    try:
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return Verdict(True, reason="the audio could not be read")
    return prints.is_owner(wav, sample_rate)


__all__ = [
    "DEFAULT_THRESHOLD", "KINDS", "KNOWN_MARGIN", "MIN_PERSONAL_THRESHOLD",
    "MIN_SECONDS", "N_MELS", "PARTIAL_FRAMES", "PROFILE_PATH", "REJECT_MARGIN",
    "SAMPLE_RATE",
    "Verdict", "Voiceprints",
    "available", "build_profile", "clear_cache", "decide", "embed", "is_owner",
    "judge", "mel_spectrogram", "model_path", "profile_threshold",
    "profile_vectors", "resolve_owner", "score_profile", "should_listen",
    "similarity", "store",
]
