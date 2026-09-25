"""
ORION's hearing beyond words — what a sound IS, not only what was said.

Everything ORION heard used to go one of two ways: speech to a transcript, or
nothing. Music, a doorbell, a dog, an alarm, typing, traffic — all silence to
him. Even Gemini Live never heard them: the microphone gate forwards audio only
while it detects a VOICE, so a song playing in the room never left the machine.

This is a deliberate listening sense, three instruments deep:

* **What sounds are present** — YAMNet (AudioSet, 521 classes: speech, music,
  instruments, singing, animals, alarms, vehicles, household sounds …) run
  locally over 0.96 s windows, so the answer comes with WHEN each sound was.
* **Music** — tempo from the onset envelope's autocorrelation, key from a
  chroma profile matched against Krumhansl-Schmuckler major/minor profiles,
  brightness (spectral centroid) and loudness.
* **Voice and tone** — the fundamental pitch (autocorrelation), how much of the
  clip is voiced, and the dominant frequencies of anything tonal (a whine,
  a hum, a beep) in plain Hz.

It reads the rolling buffer of what the microphone already heard
(``audio.RECENT_AUDIO``), so "what was that noise?" works AFTER the noise, and
it analyses files (audio or a video's soundtrack) the same way. Pure numpy +
onnxruntime; the classifier is loaded on first use and is small (15 MB).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000

# ── YAMNet front end (exactly the reference features.py parameters) ─────────
_WIN = 400                  # 25 ms
_HOP = 160                  # 10 ms
_NFFT = 512
_MELS = 64
_FMIN, _FMAX = 125.0, 7500.0
_PATCH = 96                 # frames per example (0.96 s)
_PATCH_HOP = 48             # 0.48 s between examples
_LOG_OFFSET = 0.001


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel_matrix() -> np.ndarray:
    """(257, 64) linear-to-mel weights, as tf.signal.linear_to_mel_weight_matrix
    builds them (HTK mel, triangular bands, DC bin zeroed)."""
    bins = _NFFT // 2 + 1
    freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, bins)
    mel_f = _hz_to_mel(freqs)
    edges = np.linspace(_hz_to_mel(_FMIN), _hz_to_mel(_FMAX), _MELS + 2)
    lower, centre, upper = edges[:-2], edges[1:-1], edges[2:]
    up = (mel_f[:, None] - lower[None, :]) / (centre - lower)[None, :]
    down = (upper[None, :] - mel_f[:, None]) / (upper - centre)[None, :]
    weights = np.maximum(0.0, np.minimum(up, down))
    weights[0, :] = 0.0
    return weights.astype(np.float32)


_MEL = _mel_matrix()
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / _WIN)).astype(np.float32)


def magnitude_spectrogram(wave: np.ndarray) -> np.ndarray:
    """(frames, 257) STFT magnitudes, 25 ms Hann windows every 10 ms."""
    if wave.size < _WIN:
        wave = np.pad(wave, (0, _WIN - wave.size))
    frames = 1 + (wave.size - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(wave[idx] * _HANN, n=_NFFT)).astype(np.float32)


def log_mel(wave: np.ndarray) -> np.ndarray:
    return np.log(magnitude_spectrogram(wave) @ _MEL + _LOG_OFFSET)


# ── the classifier ──────────────────────────────────────────────────────────

def _model_dir() -> Path:
    try:
        from .constants import resource_path
        return Path(resource_path("assets", "yamnet"))
    except Exception:
        return Path(__file__).resolve().parent.parent / "assets" / "yamnet"


class _Yamnet:
    _lock = threading.Lock()
    _session: Any = None
    _labels: list[str] = []

    @classmethod
    def load(cls) -> bool:
        with cls._lock:
            if cls._session is not None:
                return True
            folder = _model_dir()
            model = folder / "yamnet.onnx"
            if not model.is_file():
                return False
            try:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                cls._session = ort.InferenceSession(str(model), options,
                                                    providers=["CPUExecutionProvider"])
                cls._labels = (folder / "labels.txt").read_text(
                    encoding="utf-8").splitlines()
            except Exception:
                cls._session = None
                return False
            return True

    @classmethod
    def scores(cls, wave: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """(examples, 521) class scores and each example's start time (s)."""
        mel = log_mel(wave)
        if mel.shape[0] < _PATCH:
            mel = np.pad(mel, ((0, _PATCH - mel.shape[0]), (0, 0)),
                         constant_values=np.log(_LOG_OFFSET))
        starts = list(range(0, mel.shape[0] - _PATCH + 1, _PATCH_HOP)) or [0]
        out = []
        name = cls._session.get_inputs()[0].name
        for start in starts:
            patch = mel[start:start + _PATCH][None, None, :, :].astype(np.float32)
            out.append(cls._session.run(None, {name: patch})[0][0])
        scores = np.stack(out)
        if scores.min() < 0.0 or scores.max() > 1.0:          # logits, not probabilities
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        return scores, [s * _HOP / SAMPLE_RATE for s in starts]


# ── music and tone ──────────────────────────────────────────────────────────

_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def tempo_bpm(wave: np.ndarray) -> tuple[float | None, float]:
    """(BPM, confidence 0..1) from the spectral-flux onset envelope."""
    mel = log_mel(wave)
    if mel.shape[0] < 300:                                    # < 3 s: no tempo
        return None, 0.0
    flux = np.maximum(0.0, np.diff(mel, axis=0)).sum(axis=1)
    flux = flux - flux.mean()
    if not np.any(flux):
        return None, 0.0
    ac = np.correlate(flux, flux, mode="full")[flux.size - 1:]
    ac /= ac[0] or 1.0
    frame_rate = SAMPLE_RATE / _HOP                           # 100 frames/s
    lo, hi = int(frame_rate * 60 / 180), int(frame_rate * 60 / 60)   # 60-180 BPM
    if hi >= ac.size:
        return None, 0.0
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    confidence = float(max(0.0, min(1.0, ac[lag] * 2.0)))
    return round(60.0 * frame_rate / lag, 1), confidence


def fine_spectrum(wave: np.ndarray, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """(freqs, mean magnitude) at ~3.9 Hz resolution. The 512-point frames the
    classifier uses are 31 Hz wide — coarser than a semitone below ~500 Hz, so
    notes (and peaks) cannot be told apart in them."""
    if wave.size < n:
        wave = np.pad(wave, (0, n - wave.size))
    hop = n // 2
    frames = 1 + (wave.size - n) // hop
    window = np.hanning(n).astype(np.float32)
    total = np.zeros(n // 2 + 1, dtype=np.float64)
    for i in range(frames):
        total += np.abs(np.fft.rfft(wave[i * hop:i * hop + n] * window))
    return np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE), total / max(1, frames)


def musical_key(wave: np.ndarray) -> tuple[str | None, float]:
    """('A minor', correlation) from a chroma profile, or (None, 0).

    Built from a fine spectrum between 130 Hz and 2 kHz: below that, bass
    drums and room rumble land on arbitrary pitch classes and outvote the
    harmony (a 60 Hz kick read a clear A minor as B major)."""
    freqs, mag = fine_spectrum(wave)
    usable = (freqs >= 130.0) & (freqs <= 2000.0)
    if not usable.any() or mag[usable].sum() <= 0:
        return None, 0.0
    midi = 69 + 12 * np.log2(freqs[usable] / 440.0)
    pitch_class = np.mod(np.rint(midi), 12).astype(int)
    chroma = np.bincount(pitch_class, weights=mag[usable], minlength=12)
    chroma = chroma / chroma.sum()
    scores: dict[tuple[int, str], float] = {}
    for tonic in range(12):
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            scores[(tonic, mode)] = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
    (tonic, mode), best_r = max(scores.items(), key=lambda kv: kv[1])
    # A major key and its relative minor share every note (C major / A
    # minor); from pitch content alone they are often a near tie, and naming
    # one confidently would be a coin toss dressed as an answer.
    rel = ((tonic - 3) % 12, "minor") if mode == "major" else ((tonic + 3) % 12, "major")
    name = f"{_NOTES[tonic]} {mode}"
    if best_r - scores[rel] < 0.04:
        name += f" (or its relative, {_NOTES[rel[0]]} {rel[1]})"
    return name, best_r


def pitch_hz(wave: np.ndarray) -> tuple[float | None, float]:
    """(median fundamental in Hz, voiced fraction) by frame autocorrelation."""
    frame, hop = 1024, 512
    if wave.size < frame:
        return None, 0.0
    f0s, frames = [], 0
    lo, hi = SAMPLE_RATE // 500, SAMPLE_RATE // 60             # 60-500 Hz
    for start in range(0, wave.size - frame, hop):
        seg = wave[start:start + frame]
        if np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        frames += 1
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            f0s.append(SAMPLE_RATE / lag)
    if not f0s:
        return None, 0.0
    return round(float(np.median(f0s)), 1), len(f0s) / max(1, frames)


def dominant_frequencies(wave: np.ndarray, count: int = 3) -> list[float]:
    """The strongest distinct spectral PEAKS (Hz) — a hum, a whine, a beep.
    Local maxima of the fine spectrum only, so a single tone is one answer
    rather than the tone plus the leakage either side of it."""
    freqs, mag = fine_spectrum(wave)
    mag = mag.copy()
    mag[freqs < 40.0] = 0.0
    if mag.max() <= 0:
        return []
    is_peak = np.zeros(mag.size, dtype=bool)
    is_peak[1:-1] = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    is_peak &= mag >= 0.08 * mag.max()
    peaks: list[float] = []
    for index in np.argsort(np.where(is_peak, mag, 0.0))[::-1]:
        if not is_peak[index]:
            break
        f = float(freqs[index])
        if all(abs(f - p) > max(15.0, 0.03 * f) for p in peaks):
            peaks.append(round(f))
        if len(peaks) >= count:
            break
    return peaks


# ── the report ──────────────────────────────────────────────────────────────

#: AudioSet classes too generic to be worth naming on their own.
_GENERIC = {"Silence", "Inside, small room", "Inside, large room or hall",
            "Outside, urban or manmade", "Outside, rural or natural", "Sound effect",
            "Noise", "Environmental noise", "Static", "Hum", "Mains hum"}


@dataclass
class SoundReport:
    seconds: float
    loudness_dbfs: float
    sounds: list[tuple[str, float]] = field(default_factory=list)     # overall
    timeline: list[tuple[float, str, float]] = field(default_factory=list)
    music: bool = False
    speech: bool = False
    tempo: float | None = None
    tempo_confidence: float = 0.0
    key: str | None = None
    key_confidence: float = 0.0
    pitch: float | None = None
    voiced: float = 0.0
    peaks: list[float] = field(default_factory=list)
    classifier: bool = True

    def summary(self) -> str:
        if self.loudness_dbfs < -65:
            return (f"Near silence over {self.seconds:.0f} s "
                    f"({self.loudness_dbfs:.0f} dBFS) — nothing distinct to hear.")
        lines = []
        if self.sounds:
            named = ", ".join(f"{label} ({score:.0%})" for label, score in self.sounds[:5])
            lines.append(f"Heard over {self.seconds:.0f} s: {named}.")
        elif not self.classifier:
            lines.append("The sound classifier is not installed, so only the "
                         "acoustic measurements below are available.")
        if self.music:
            bits = []
            if self.tempo and self.tempo_confidence >= 0.25:
                bits.append(f"about {self.tempo:.0f} BPM")
            if self.key and self.key_confidence >= 0.55:
                bits.append(f"most likely in {self.key}")
            if bits:
                lines.append("The music is " + " and ".join(bits) + ".")
        if self.pitch and self.voiced >= 0.2:
            register = ("a low voice" if self.pitch < 150 else
                        "a mid-range voice" if self.pitch < 230 else "a high voice")
            what = register if self.speech else "a tonal sound"
            lines.append(f"Fundamental pitch around {self.pitch:.0f} Hz ({what}).")
        if self.peaks and not self.speech and not self.music:
            lines.append("Strongest frequencies: "
                         + ", ".join(f"{p:.0f} Hz" for p in self.peaks) + ".")
        changes = [f"{at:.1f}s {label}" for at, label, _ in self.timeline[:8]]
        if len(changes) > 1:
            lines.append("Timeline: " + "; ".join(changes) + ".")
        lines.append(f"Level {self.loudness_dbfs:.0f} dBFS.")
        return " ".join(lines)


def analyse(wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> SoundReport:
    """Everything above over one mono clip (float -1..1 or int16)."""
    wave = np.asarray(wave)
    if wave.dtype == np.int16:
        wave = wave.astype(np.float32) / 32768.0
    wave = wave.astype(np.float32)
    if sample_rate != SAMPLE_RATE and wave.size:
        positions = np.arange(0, wave.size, sample_rate / SAMPLE_RATE)
        wave = np.interp(positions, np.arange(wave.size), wave).astype(np.float32)
    seconds = wave.size / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0
    report = SoundReport(seconds=seconds, loudness_dbfs=20 * np.log10(rms + 1e-9))
    if report.loudness_dbfs < -65 or seconds < 0.3:
        return report
    labels: list[str] = []
    if _Yamnet.load():
        scores, starts = _Yamnet.scores(wave)
        labels = _Yamnet._labels
        overall = scores.mean(axis=0)
        peak = scores.max(axis=0)
        # A sound heard clearly once counts, even if brief: blend mean and peak.
        blend = 0.5 * overall + 0.5 * peak
        order = np.argsort(blend)[::-1]
        report.sounds = [(labels[i], float(blend[i])) for i in order[:12]
                         if blend[i] >= 0.12 and labels[i] not in _GENERIC][:6]
        last = ""
        for at, row in zip(starts, scores):
            ranked = [i for i in np.argsort(row)[::-1][:3] if labels[i] not in _GENERIC]
            if not ranked or row[ranked[0]] < 0.2:
                continue
            top = labels[ranked[0]]
            if top != last:
                report.timeline.append((round(at, 1), top, float(row[ranked[0]])))
                last = top
        names = {label for label, _ in report.sounds}
        report.speech = bool(names & {"Speech", "Conversation", "Narration, monologue",
                                      "Male speech, man speaking", "Female speech, woman speaking",
                                      "Child speech, kid speaking"})
        report.music = "Music" in names or any(
            w in " ".join(names) for w in ("Musical instrument", "Guitar", "Piano",
                                           "Drum", "Singing", "Pop music", "Rock music"))
    else:
        report.classifier = False
    if report.music or not labels:
        report.tempo, report.tempo_confidence = tempo_bpm(wave)
        report.key, report.key_confidence = musical_key(wave)
        report.music = report.music or report.tempo_confidence >= 0.4
    report.pitch, report.voiced = pitch_hz(wave)
    report.peaks = dominant_frequencies(wave)
    return report


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def load_file(path: str, max_seconds: float = 600.0) -> np.ndarray:
    """Decode an audio file (or a video's soundtrack) to 16 kHz mono float."""
    import av  # PyAV — already a dependency of video analysis

    frames: list[np.ndarray] = []
    total = 0
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("that file has no audio track")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for packet in container.demux(stream):
            for frame in packet.decode():
                for out in resampler.resample(frame):
                    data = out.to_ndarray().reshape(-1)
                    frames.append(data)
                    total += data.size
            if total >= max_seconds * SAMPLE_RATE:
                break
    if not frames:
        raise ValueError("no audio could be decoded")
    return np.concatenate(frames).astype(np.float32) / 32768.0


__all__ = ["SoundReport", "analyse", "dominant_frequencies", "load_file", "log_mel",
           "musical_key", "pcm16_to_float", "pitch_hz", "tempo_bpm"]
