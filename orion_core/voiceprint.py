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
"""

from __future__ import annotations

import json
import os
import threading
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
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            _SESSION = ort.InferenceSession(
                str(model_path()), options,
                providers=["CPUExecutionProvider"])
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
    wav = np.asarray(wav, dtype=np.float32).flatten()
    if wav.size == 0:
        return None
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


# ── enrolment ────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    """What the check concluded, and how sure it is."""

    is_owner: bool
    score: float = 0.0
    speaker: str = ""
    reason: str = ""
    confident: bool = False

    def __bool__(self) -> bool:
        return self.is_owner


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
        """Whether any gate needs each utterance's speaker judged."""
        data = self._load()
        return bool(data.get("only_owner") or data.get("guard_actions"))

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
        import numpy as np

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

        centroid = np.mean(np.stack(embeddings), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0:
            return False, "the clips produced no usable voiceprint."
        centroid = centroid / norm

        # How tightly the clips agree. A speaker whose own clips sit at 0.9
        # can be judged more strictly than one whose sit at 0.75, and storing
        # it means the threshold can adapt to the person rather than to an
        # average of everybody.
        spread = [similarity(e, centroid) for e in embeddings]
        data = self._load()
        data["profiles"][name] = {
            "embedding": [float(v) for v in centroid],
            "clips": len(embeddings),
            "self_similarity": float(min(spread)) if spread else 0.0,
            # The encoder that produced it. voice_speaker_id stamps the same
            # field and skips profiles from a different one, because the two
            # preprocess audio differently and their embeddings are not
            # comparable — a silent, plausible-looking wrong answer.
            "backend": "voiceprint-onnx",
        }
        self._save_profiles(data["profiles"])
        if owner or not data.get("owner"):
            data["owner"] = name
        self._save(data)
        return True, (f"{name}'s voice is enrolled from {len(embeddings)} "
                      f"clip(s)" + (" — ORION will treat this as your voice."
                                    if data['owner'] == name else "."))

    def forget(self, name: str) -> tuple[bool, str]:
        data = self._load()
        if name not in data["profiles"]:
            return False, f"no voiceprint is stored for {name}."
        del data["profiles"][name]
        self._save_profiles(data["profiles"])
        if data.get("owner") == name:
            data["owner"] = ""
        self._save(data)
        return True, f"{name}'s voiceprint is forgotten."

    # -- the actual question -------------------------------------------------
    def identify(self, wav: Any, sample_rate: int = SAMPLE_RATE) -> Verdict:
        """Whose voice this is, among those enrolled."""
        profiles = self._load()["profiles"]
        if not profiles:
            return Verdict(True, reason="nobody is enrolled")
        vector = embed(wav, sample_rate)
        if vector is None:
            return Verdict(True, reason="not enough speech to tell")

        best, score = "", -1.0
        for name, record in profiles.items():
            value = similarity(vector, record.get("embedding"))
            if value > score:
                best, score = name, value
        limit = self.threshold()
        if score >= limit:
            return Verdict(best == (self.owner or best), score, best,
                           "recognised", confident=True)
        return Verdict(False, score, "", "not a voice ORION knows",
                       confident=score < limit - 0.08)

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
            return Verdict(True, reason="no owner voice is enrolled")
        if not available():
            return Verdict(True, reason="the voice encoder is unavailable")

        vector = embed(wav, sample_rate)
        if vector is None:
            return Verdict(True, reason="not enough speech to tell")
        score = similarity(vector, record.get("embedding"))
        limit = self.threshold()
        if score >= limit:
            return Verdict(True, score, owner, "recognised", confident=True)
        # Only a clear miss counts as somebody else. The margin is the whole
        # safety property: without it, every borderline utterance silences him.
        if score < limit - 0.08:
            return Verdict(False, score, "", "a different voice",
                           confident=True)
        return Verdict(True, score, owner, "uncertain, so assumed to be you")

    def describe(self) -> str:
        data = self._load()
        if not available():
            return ("Voice recognition is unavailable: "
                    f"{model_path().name} or onnxruntime is missing.")
        names = sorted(data["profiles"])
        if not names:
            return ("No voice is enrolled yet. Give ORION about ten seconds "
                    "of your speech and he will know your voice from anyone "
                    "else's.")
        owner = self.owner or "nobody"
        return (f"Enrolled: {', '.join(names)}. ORION treats {owner} as your "
                f"voice, at a similarity threshold of {self.threshold():.2f}.")


def _write_json(path: Path, payload: Any) -> None:
    """Write via a temporary file and rename.

    A half-written profile would be read back as no profile at all, and with
    the gate on that means ORION stops answering his owner.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    except Exception:
        pass


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
    try:
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return Verdict(True, reason="the audio could not be read")
    verdict = prints.is_owner(wav, sample_rate)
    speaker_gate.note_verdict(verdict)
    return verdict


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
    "DEFAULT_THRESHOLD", "MIN_SECONDS", "N_MELS", "PARTIAL_FRAMES",
    "PROFILE_PATH", "SAMPLE_RATE",
    "Verdict", "Voiceprints",
    "available", "embed", "is_owner", "judge", "mel_spectrogram", "model_path",
    "should_listen", "similarity", "store",
]
