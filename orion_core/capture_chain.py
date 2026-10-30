"""
Cleaning the microphone before anything listens to it (Mark XXXV W2.1 + W2.4).

Everything that hears the owner — Gemini Live, offline Whisper, the Vosk wake
and command listeners, the echo guard — used to be handed the raw microphone.
On a webcam microphone a metre away that is a quiet voice in a reverberant
room, under fan noise and mains hum, and, whenever ORION is talking, under
ORION's own voice coming back out of the speakers.

This module is the stage in between. Per 10 ms frame, on the audio gate thread
(never the Qt/asyncio thread):

  1. HIGH-PASS at 90 Hz (2nd-order Butterworth): desk thumps, mains hum and
     fan rumble carry no speech and eat the recogniser's dynamic range.
  2. ECHO CANCELLATION + NOISE SUPPRESSION with the WebRTC audio-processing
     module (its echo canceller and noise suppressor, via the optional
     ``aec-audio-processing`` wheel). The canceller is fed what ORION ACTUALLY
     PLAYED — the renderer appends every slice it writes to the device, with
     the moment it was written, to ``speaker_reference()`` — so it subtracts
     his voice sample by sample instead of guessing from loudness.
  3. AUTOMATIC GAIN toward a fixed speech level, then a soft peak limiter.

Timing is the hard part, and it is measured, not assumed. Two clocks matter:
when a slice was handed to the speaker, and when the microphone captured a
block. Both are placed on one absolute timeline (monotonic seconds x 16 000):
the renderer's slices are laid end to end from the moment the first one was
written (re-anchored only if the device ran dry), and microphone chunks retain
their counted spacing despite callback jitter, re-anchoring only on a real
discontinuity. The remaining offset — output latency, the trip through the
room, input latency, typically 50-500 ms and different on every machine — is
estimated by cross-correlating the two (GCC-PHAT) while ORION speaks. The
reference is pre-delayed for larger measured offsets. Real-device ERLE, CPU,
recognition accuracy and interruption latency require a separate acceptance
measurement; the existence of this adapter does not establish those results.

Nothing here can take the microphone down. Every entry point catches its own
faults; a chunk that fails to process is passed through unchanged, the fault is
logged once, and three consecutive faults switch the chain off for the session.
The chain adds exactly one frame (10 ms) of latency, constant.

ORION_CAPTURE_DSP=0 restores the untouched capture path exactly (the gate then
never builds a chain and the renderer records no reference). Finer switches for
diagnosis: ORION_CAPTURE_AEC, ORION_CAPTURE_NS, ORION_CAPTURE_AGC (each 0/1),
ORION_CAPTURE_NS_LEVEL (0 low .. 3 very high).
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
#: The WebRTC module's unit: 10 ms at 16 kHz.
FRAME = 160
_FRAME_S = FRAME / SAMPLE_RATE
_FULL_SCALE = 32768.0
_OFF = frozenset({"0", "false", "no", "off"})


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in _OFF


def enabled() -> bool:
    """Whether the capture chain runs at all. ORION_CAPTURE_DSP=0 turns it off
    and restores the capture path exactly as it was before the chain existed."""
    return _flag("ORION_CAPTURE_DSP")


def _db(energy: Any) -> Any:
    """Mean-square energy in int16 units -> dBFS."""
    return 10.0 * np.log10(np.maximum(energy, 1e-3) / (_FULL_SCALE * _FULL_SCALE))


# ──────────────────────────────────────────────────────────────────────────────
# What ORION played (written by the renderer thread)
# ──────────────────────────────────────────────────────────────────────────────

class PlaybackReference:
    """Every slice ORION's renderer handed to the speaker, and when.

    The renderer's write loop is the most timing-sensitive code in ORION (see
    AudioPlaybackThread._write), so recording costs it exactly one deque
    append per ~85 ms slice: no copy, no maths, no lock. Resampling and
    placement happen later on the capture side. Reading is non-destructive and
    sequence-numbered, so more than one capture chain (a microphone being
    replaced while its successor starts) can follow the same speaker.
    """

    def __init__(self, maxlen: int = 512) -> None:
        self._items: deque = deque(maxlen=maxlen)
        self._seq = itertools.count(1)

    def note(self, pcm: Any, at: float | None = None, sample_rate: int = 24_000) -> None:
        """Record a slice about to be written to the speaker. Never raises."""
        try:
            stamp = time.monotonic() if at is None else float(at)
            self._items.append((next(self._seq), stamp, int(sample_rate), pcm))
        except Exception:  # noqa: BLE001, S110 - recording must not break the device writer
            pass

    def since(self, seq: int) -> list:
        """Slices recorded after sequence number *seq*, oldest first."""
        try:
            if self._items[-1][0] <= seq:  # the usual case: nothing new
                return []
        except IndexError:
            return []
        for _ in range(3):
            try:
                items = list(self._items)
                break
            except RuntimeError:           # mutated mid-copy; try again
                continue
        else:
            return []
        i = len(items)
        while i > 0 and items[i - 1][0] > seq:
            i -= 1
        return items[i:]

    def clear(self) -> None:
        self._items.clear()


_REFERENCE = PlaybackReference()


def speaker_reference() -> PlaybackReference:
    """The process-wide record of ORION's own playback."""
    return _REFERENCE


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _Resampler:
    """Stateful rational resampler to 16 kHz (24 kHz in: up 2, down 3).

    Continuous across calls, so slices written back to back resample exactly
    as one signal would. ``reset`` starts afresh after a gap.
    """

    def __init__(self, rate_in: int, signal: Any) -> None:
        rate_in = int(rate_in)
        g = math.gcd(rate_in, SAMPLE_RATE)
        self.up, self.down = SAMPLE_RATE // g, rate_in // g
        self._signal = signal
        self.h = None
        if self.up == 1 and self.down == 1:
            return
        if signal is None:
            raise RuntimeError("resampling needs scipy")
        taps = 16 * max(self.up, self.down) + 1
        self.h = signal.firwin(taps, 0.45 * min(rate_in, SAMPLE_RATE),
                               fs=rate_in * self.up) * self.up
        self._zi = np.zeros(taps - 1)
        self._phase = 0

    def reset(self) -> None:
        if self.h is not None:
            self._zi = np.zeros(self.h.size - 1)
            self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.h is None:
            return x.astype(np.float32, copy=False)
        u = np.zeros(x.size * self.up)
        u[:: self.up] = x
        y, self._zi = self._signal.lfilter(self.h, 1.0, u, zi=self._zi)
        out = y[self._phase:: self.down]
        self._phase = (self._phase - u.size) % self.down
        return out.astype(np.float32)


class _Timeline:
    """The last few seconds of a 16 kHz signal, addressed by absolute sample
    index (monotonic seconds x 16 000). Unwritten positions read as silence."""

    def __init__(self, seconds: float = 8.0) -> None:
        self.size = 1 << math.ceil(math.log2(seconds * SAMPLE_RATE))
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.end: int | None = None

    def _put(self, start: int, x: np.ndarray) -> None:
        i = start % self.size
        first = min(x.size, self.size - i)
        self.buf[i:i + first] = x[:first]
        if first < x.size:
            self.buf[:x.size - first] = x[first:]

    def write(self, start: int, x: np.ndarray) -> None:
        if x.size == 0:
            return
        if x.size > self.size:
            start += x.size - self.size
            x = x[-self.size:]
        if self.end is not None and start > self.end:
            gap = start - self.end
            if gap >= self.size:
                self.buf[:] = 0.0
            else:
                self._put(self.end, np.zeros(gap, dtype=np.float32))
        self._put(start, x)
        end = start + x.size
        if self.end is None or end > self.end:
            self.end = end

    def read(self, start: int, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if self.end is None:
            return out
        lo = max(start, self.end - self.size)
        hi = min(start + n, self.end)
        if hi <= lo:
            return out
        i, count = lo % self.size, hi - lo
        first = min(count, self.size - i)
        out[lo - start:lo - start + first] = self.buf[i:i + first]
        if first < count:
            out[lo - start + first:hi - start] = self.buf[:count - first]
        return out

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.end = None


class _MicClock:
    """Places microphone chunks on the absolute timeline.

    A chunk's PortAudio callback time is when it ARRIVED, which is when it was
    captured plus input latency plus however long the callback waited for the
    GIL. Counting samples gives the spacing exactly. Keep the initial offset:
    a sliding minimum changed reference alignment by a few samples whenever
    an older low-jitter callback left the window (12 October).
    Re-anchor only when lateness exceeds 20 ms for half a second of samples,
    an arrival is more than 20 ms earlier than counted timing, or a gap is
    larger than 250 ms. Persistent lateness uses the minimum over its recent
    callbacks so one late callback cannot define the replacement clock.
    """

    _GAP = int(0.25 * SAMPLE_RATE)
    _PERSIST = int(0.5 * SAMPLE_RATE)
    _LATE = int(0.02 * SAMPLE_RATE)

    def __init__(self, window: int = 64) -> None:
        self.count = 0
        self._offsets: deque = deque(maxlen=window)
        self.offset: int | None = None
        self._late_samples = 0
        self.discontinuous = False

    def place(self, n: int, at: float | None) -> int:
        """Absolute index of the first sample of an *n*-sample chunk whose last
        sample arrived at monotonic time *at*."""
        stamp = time.monotonic() if at is None else float(at)
        raw = round(stamp * SAMPLE_RATE) - (self.count + n)
        self.discontinuous = False
        if self.offset is None:
            self.offset = raw
        elif raw < self.offset - self._LATE or raw > self.offset + self._GAP:
            self.offset = raw
            self.discontinuous = True
            self._offsets.clear()
            self._late_samples = 0
        elif raw > self.offset + self._LATE:
            self._offsets.append(raw)
            self._late_samples += n
            if self._late_samples >= self._PERSIST:
                self.offset = min(self._offsets)
                self.discontinuous = True
                self._offsets.clear()
                self._late_samples = 0
        else:
            self._offsets.clear()
            self._late_samples = 0
        start = self.count + self.offset
        self.count += n
        return start


class _DelayEstimator:
    """How long ORION's voice takes to come back, measured by GCC-PHAT.

    Correlates the last second of microphone against the reference up to
    0.8 s earlier. The phase transform whitens both spectra, so the peak is
    sharp for speech even in a reverberant room, and its height above the
    correlation's spread says how much to trust it. Two agreeing estimates in
    a row are required before a lag is believed.
    """

    WINDOW_S = 1.0
    MAX_LAG_S = 0.8
    MIN_CONFIDENCE = 8.0
    AGREE = int(0.003 * SAMPLE_RATE)

    def __init__(self) -> None:
        self.w = int(self.WINDOW_S * SAMPLE_RATE)
        self.max_lag = int(self.MAX_LAG_S * SAMPLE_RATE)
        self.n = 1 << math.ceil(math.log2(2 * self.w + self.max_lag))
        freqs = np.fft.rfftfreq(self.n, 1.0 / SAMPLE_RATE)
        self._band = ((freqs >= 200.0) & (freqs <= 4000.0)).astype(np.float32)
        self.next_at: int | None = None
        self._candidate: int | None = None
        self.lag: int | None = None
        self.confidence = 0.0
        self.estimates = 0

    def reset(self) -> None:
        self.next_at = None
        self._candidate = None
        self.lag = None
        self.confidence = 0.0

    def due(self, now_abs: int) -> bool:
        return self.next_at is None or now_abs >= self.next_at

    def run(self, mic: _Timeline, ref: _Timeline, now_abs: int) -> int | None:
        """A confirmed lag in samples, or None. Schedules its own next run."""
        settled = self.lag is not None
        self.next_at = now_abs + int((2.0 if settled else 0.5) * SAMPLE_RATE)
        a = ref.read(now_abs - self.w - self.max_lag, self.w + self.max_lag)
        frames = a[: a.size // FRAME * FRAME].reshape(-1, FRAME)
        active = _db(np.mean(frames * frames, axis=1)) > -50.0
        if active.mean() < 0.3:
            return None                    # ORION was not talking enough to measure
        b = mic.read(now_abs - self.w, self.w)
        if float(np.mean(b * b)) < 1.0:
            return None
        spec = np.fft.rfft(a, self.n) * np.conj(np.fft.rfft(b, self.n))
        spec = spec / (np.abs(spec) ** 0.8 + 1e-9) * self._band
        corr = np.fft.irfft(spec, self.n)[: self.max_lag + 1]
        k = int(np.argmax(corr))
        spread = float(np.std(corr))
        self.confidence = float(corr[k]) / spread if spread > 0 else 0.0
        self.estimates += 1
        if self.confidence < self.MIN_CONFIDENCE:
            self._candidate = None
            return None
        lag = self.max_lag - k
        if self._candidate is not None and abs(lag - self._candidate) <= self.AGREE:
            self.lag = (lag + self._candidate) // 2
            self._candidate = lag
            return self.lag
        self._candidate = lag
        return None


# ──────────────────────────────────────────────────────────────────────────────
# The chain
# ──────────────────────────────────────────────────────────────────────────────

class CaptureChain:
    """High-pass -> echo cancellation + noise suppression -> gain, per 10 ms.

    ``process(chunk, at)`` takes 16 kHz mono int16 bytes of any length and
    returns the same number of bytes, cleaned and delayed by exactly one frame.
    Construction is cheap and imports nothing heavy: scipy and the WebRTC
    module load on the first ``process`` call, which is on the audio thread.
    """

    #: Gain targets. Speech level is the running dB mean of voiced frames.
    TARGET_DBFS = -26.0
    MAX_GAIN_DB = 18.0
    MIN_GAIN_DB = -10.0
    GAIN_UP_DB_PER_S = 8.0
    GAIN_DOWN_DB_PER_S = 20.0
    #: Soft limiter: transparent below the knee, never beyond the ceiling.
    KNEE = 0.5 * _FULL_SCALE              # -6 dBFS
    CEILING = 0.89 * _FULL_SCALE          # -1 dBFS
    HPF_HZ = 90.0
    #: The canceller's comfortable span (measured on this module, see above).
    #: Inside it the reference goes in undelayed; outside it the reference is
    #: pre-delayed so the residual sits at TARGET.
    ALIGN_MIN_S, ALIGN_MAX_S, ALIGN_TARGET_S = 0.015, 0.38, 0.06
    #: Echo-removal (dB) below which the canceller is not trusted.
    CONVERGED_DB = 12.0

    def __init__(self, *, log: Callable[[str], None] | None = None,
                 reference: PlaybackReference | None = None,
                 aec: bool | None = None, ns: bool | None = None,
                 agc: bool | None = None, hpf: bool = True,
                 ns_level: int | None = None) -> None:
        self._log_cb = log
        self._reference = reference if reference is not None else speaker_reference()
        self.aec_wanted = _flag("ORION_CAPTURE_AEC") if aec is None else bool(aec)
        self.ns_wanted = _flag("ORION_CAPTURE_NS") if ns is None else bool(ns)
        self.agc_on = _flag("ORION_CAPTURE_AGC") if agc is None else bool(agc)
        self.hpf_wanted = bool(hpf)
        if ns_level is None:
            try:
                ns_level = int(os.getenv("ORION_CAPTURE_NS_LEVEL", "1"))
            except ValueError:
                ns_level = 1
        self.ns_level = max(0, min(3, int(ns_level)))
        self._ready = False
        self._failed = False
        self._errors = 0
        self._logged: set[str] = set()
        self._signal: Any = None
        self._apm: Any = None
        self._ns_apm: Any = None
        self._sos: Any = None
        self._hpf_zi: Any = None
        self._clock = _MicClock()
        self._ref_tl = _Timeline(8.0)
        self._mic_tl = _Timeline(4.0)
        self._estimator = _DelayEstimator()
        self._resamplers: dict[int, _Resampler] = {}
        self._ref_seq = 0
        self._ref_cursor: int | None = None
        self._ref_rate: int | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        # Exactly one frame of silence ahead of the output: the frame that is
        # still filling never holds back what was asked for.
        self._outq = np.zeros(FRAME, dtype=np.float32)
        self.d_pre = 0
        self.delay_s: float | None = None
        # Echo-removal measurements, single-talk dominated (see _update_echo).
        self._erle: deque = deque(maxlen=300)
        self.converged = False
        self.erle_db = 0.0
        self.near_end = False
        self.near_end_at = -1e9
        self._in_floor_db = -60.0
        self._out_floor_db = -70.0
        # Gain state.
        self._gain_db = 0.0
        self._level_db = self.TARGET_DBFS
        self._agc_floor_db = -60.0
        self._cost_s = 0.0
        self._audio_s = 0.0
        self.chunks = 0
        self._last_at: float | None = None
        self._last_echo_measure_at = -1e9

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, chunk: bytes, at: float | None = None) -> bytes:
        """Clean one chunk. Never raises; on any fault returns *chunk* unchanged."""
        if self._failed or not chunk or len(chunk) % 2:
            return chunk
        started = time.perf_counter()
        try:
            if at is not None:
                at = float(at)
                if not math.isfinite(at):
                    raise ValueError("capture timestamp is not finite")
                # Samples skipped by an overflow, queue drain or device change
                # cannot be joined to an old partial frame or filter state.
                duration = len(chunk) / (2.0 * SAMPLE_RATE)
                if self._last_at is not None and abs(at - self._last_at - duration) > 0.05:
                    self.reset_alignment()
            out = self._process(chunk, at)
            self._last_at = at
            self._errors = 0
        except Exception as exc:  # noqa: BLE001 - preserve hearing after any DSP fault
            out = chunk
            self._errors += 1
            self.reset_alignment()
            self._log_once("fault", "AUDIO: capture chain fault, passing the microphone "
                           f"through unchanged - {type(exc).__name__}: "
                           f"{str(exc).splitlines()[0][:100] if str(exc) else ''}")
            if self._errors >= 3:
                self._failed = True
                self._log_once("off", "AUDIO: capture chain switched off for this session "
                               "after repeated faults; the raw microphone is used.")
        self._cost_s += time.perf_counter() - started
        self._audio_s += len(chunk) / (2.0 * SAMPLE_RATE)
        self.chunks += 1
        return out

    @property
    def active(self) -> bool:
        return not self._failed

    @property
    def cancelling(self) -> bool:
        """Echo cancellation is loaded and running."""
        return self._apm is not None and self.aec_wanted and not self._failed

    def echo_free(self, now: float | None = None, within_s: float = 0.6) -> bool:
        """True when the canceller is converged and, within *within_s*, the
        cleaned signal carried a voice that is not ORION's echo."""
        if not (self.cancelling and self.converged):
            return False
        now = time.monotonic() if now is None else now
        age = now - self.near_end_at
        return 0.0 <= age <= within_s

    def cpu_fraction(self) -> float:
        """Processing time per second of audio (0.01 = 1 % of one core)."""
        return self._cost_s / self._audio_s if self._audio_s > 0 else 0.0

    def reset_alignment(self) -> None:
        """Forget discontinuous microphone/filter/FIFO state, keep reference history.

        The next call recreates the native processors on the capture thread.
        Returning raw audio after a fault must not leave old speech queued to
        reappear in a later, successfully processed chunk.
        """
        self._estimator.reset()
        self._clock = _MicClock()
        self._mic_tl.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._outq = np.zeros(FRAME, dtype=np.float32)
        self._last_at = None
        self._ready = False
        self._apm = None
        self._ns_apm = None
        self._sos = None
        self._hpf_zi = None
        self._erle.clear()
        self.converged = False
        self.erle_db = 0.0
        self.delay_s = None
        self.d_pre = 0
        self.near_end = False
        self.near_end_at = -1e9
        self._last_echo_measure_at = -1e9
        self._echo_expected = False
        self._gain_db = 0.0
        self._level_db = self.TARGET_DBFS
        self._agc_floor_db = -60.0
        self._in_floor_db = -60.0
        self._out_floor_db = -70.0

    def stats(self) -> dict[str, Any]:
        return {
            "active": self.active, "cancelling": self.cancelling,
            "noise_suppression": self._ns_apm is not None and not self._failed,
            "agc": self.agc_on, "high_pass": self._sos is not None,
            "converged": self.converged, "erle_db": round(self.erle_db, 1),
            "delay_ms": None if self.delay_s is None else round(self.delay_s * 1000.0, 1),
            "reference_predelay_ms": round(self.d_pre * 1000.0 / SAMPLE_RATE, 1),
            "gain_db": round(self._gain_db, 1), "cpu_fraction": round(self.cpu_fraction(), 5),
            "latency_ms": FRAME * 1000.0 / SAMPLE_RATE,
        }

    # ── setup ─────────────────────────────────────────────────────────────────

    def _log_once(self, key: str, message: str) -> None:
        if key in self._logged:
            return
        self._logged.add(key)
        try:
            if self._log_cb is not None:
                self._log_cb(message)
        except Exception:  # noqa: BLE001, S110 - optional logging cannot break capture
            pass

    def _initialise(self) -> None:
        """Load scipy and the WebRTC module — on the audio thread, once."""
        self._ready = True
        try:
            from scipy import signal  # type: ignore
            self._signal = signal
            if self.hpf_wanted:
                self._sos = signal.butter(2, self.HPF_HZ, btype="highpass",
                                          fs=SAMPLE_RATE, output="sos")
                self._hpf_zi = signal.sosfilt_zi(self._sos) * 0.0
        except Exception as exc:  # noqa: BLE001 - optional native/scipy loading may fail
            self._log_once("scipy", f"AUDIO: capture high-pass unavailable ({exc}).")
        if self.aec_wanted or self.ns_wanted:
            try:
                from aec_audio_processing import AudioProcessor  # type: ignore
                # Measure cancellation BEFORE noise suppression. Otherwise NS
                # reducing fan noise can falsely attest that AEC converged and
                # open the microphone while ORION is still audible.
                if self.aec_wanted:
                    apm = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False)
                    apm.set_stream_format(SAMPLE_RATE, 1, SAMPLE_RATE, 1)
                    apm.set_reverse_stream_format(SAMPLE_RATE, 1)
                    self._apm = apm
                if self.ns_wanted:
                    ns_apm = AudioProcessor(enable_aec=False, enable_ns=True,
                                            ns_level=self.ns_level, enable_agc=False)
                    ns_apm.set_stream_format(SAMPLE_RATE, 1, SAMPLE_RATE, 1)
                    self._ns_apm = ns_apm
            except Exception as exc:  # noqa: BLE001 - fall back when native APM is unavailable
                self._apm = None
                self._ns_apm = None
                self._log_once("apm", "AUDIO: WebRTC echo cancellation / noise suppression "
                               f"unavailable ({type(exc).__name__}: {exc}); the microphone "
                               "gets high-pass and gain only.")
        parts = [name for name, on in (("high-pass", self._sos is not None),
                                       ("echo cancellation", self.cancelling),
                                        ("noise suppression", self._ns_apm is not None),
                                       ("gain", self.agc_on)) if on]
        self._log_once("ready", "AUDIO: capture chain active - " + (", ".join(parts) or "pass-through")
                       + " (10 ms, on the audio thread).")

    # ── the reference ─────────────────────────────────────────────────────────

    def _pull_reference(self) -> None:
        """Place newly played slices on the timeline, end to end."""
        items = self._reference.since(self._ref_seq)
        if not items:
            return
        for seq, at, rate, pcm in items:
            self._ref_seq = seq
            x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            if x.size == 0:
                continue
            resampler = self._resamplers.get(rate)
            if resampler is None:
                resampler = self._resamplers[rate] = _Resampler(rate, self._signal)
            anchor = round(at * SAMPLE_RATE)
            # A slice written while the device still holds earlier audio plays
            # right after it; one written later than that means the device ran
            # dry (or a new reply began), so it starts from its own timestamp.
            if self._ref_cursor is None or anchor > self._ref_cursor or rate != self._ref_rate:
                resampler.reset()
                start = anchor
            else:
                start = self._ref_cursor
            y = resampler.process(x)
            self._ref_tl.write(start, y)
            self._ref_cursor = start + y.size
            self._ref_rate = rate

    def _align(self, lag: int) -> None:
        self.delay_s = lag / SAMPLE_RATE
        residual = (lag - self.d_pre) / SAMPLE_RATE
        if not self.ALIGN_MIN_S <= residual <= self.ALIGN_MAX_S:
            self.d_pre = max(0, lag - int(self.ALIGN_TARGET_S * SAMPLE_RATE))
            self._erle.clear()
            self.converged = False
            self.near_end = False
            self.near_end_at = -1e9
            residual = (lag - self.d_pre) / SAMPLE_RATE
        delay_setter = getattr(self._apm, "set_stream_delay", None)
        if callable(delay_setter):
            delay_setter(max(0, round(residual * 1000.0)))
        self._log_once(f"align{self.d_pre // 1600}",
                       f"AUDIO: echo canceller aligned - ORION's voice returns to the "
                       f"microphone {lag * 1000.0 / SAMPLE_RATE:.0f} ms after it is played.")

    # ── per chunk ─────────────────────────────────────────────────────────────

    def _process(self, chunk: bytes, at: float | None) -> bytes:
        if not self._ready:
            self._initialise()
        x = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
        n = x.size
        start = self._clock.place(n, at)
        if self._clock.discontinuous:
            # A persistent capture loss can be smaller than the per-callback
            # gap check. Partial frames and AEC convergence belong to the old
            # alignment and must not survive the new microphone anchor.
            self.reset_alignment()
            self._initialise()
            start = self._clock.place(n, at)
        stamp = time.monotonic() if at is None else float(at)
        self._pull_reference()
        if self._sos is not None:
            y, self._hpf_zi = self._signal.sosfilt(self._sos, x, zi=self._hpf_zi)
            x = y.astype(np.float32)
        self._mic_tl.write(start, x)
        if self.cancelling and self._estimator.due(start + n) and self._ref_tl.end is not None \
                and self._ref_tl.end > start + n - self._estimator.w - self._estimator.max_lag:
            lag = self._estimator.run(self._mic_tl, self._ref_tl, start + n)
            if lag is not None:
                self._align(lag)

        fifo = np.concatenate((self._pending, x)) if self._pending.size else x
        first = start - self._pending.size
        count = fifo.size // FRAME
        self._pending = fifo[count * FRAME:].copy()
        if count:
            frames = fifo[:count * FRAME]
            cleaned = self._frames(frames, first, count, stamp)
            self._outq = np.concatenate((self._outq, cleaned))
        out, self._outq = self._outq[:n], self._outq[n:]
        return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()

    def _frames(self, frames: np.ndarray, first: int, count: int, stamp: float) -> np.ndarray:
        apm = self._apm
        ref = self._ref_tl.read(first - self.d_pre, count * FRAME)
        if apm is None:
            cleaned = frames
        else:
            mic16 = np.clip(np.rint(frames), -32768, 32767).astype("<i2").tobytes()
            ref16 = np.clip(np.rint(ref), -32768, 32767).astype("<i2").tobytes()
            step = FRAME * 2
            parts = []
            for i in range(0, count * step, step):
                apm.process_reverse_stream(ref16[i:i + step])
                parts.append(apm.process_stream(mic16[i:i + step]))
            cleaned = np.frombuffer(b"".join(parts), dtype="<i2").astype(np.float32)
        self._update_echo(frames, cleaned, first, count, stamp)
        if self._ns_apm is not None:
            data = np.clip(np.rint(cleaned), -32768, 32767).astype("<i2").tobytes()
            step = FRAME * 2
            parts = [self._ns_apm.process_stream(data[i:i + step])
                     for i in range(0, count * step, step)]
            cleaned = np.frombuffer(b"".join(parts), dtype="<i2").astype(np.float32)
            if cleaned.size != frames.size:
                raise ValueError("noise suppressor returned an invalid frame size")
        if self.agc_on:
            cleaned = self._gain(cleaned, count)
        return cleaned

    def _update_echo(self, frames: np.ndarray, cleaned: np.ndarray, first: int,
                     count: int, stamp: float) -> None:
        """Track how much echo the canceller removes, and whether the cleaned
        signal holds a voice that is not ORION's."""
        e_in = np.mean(frames.reshape(count, FRAME) ** 2, axis=1)
        e_out = np.mean(cleaned.reshape(count, FRAME) ** 2, axis=1)
        in_db, out_db = _db(e_in), _db(e_out)
        for value in in_db:
            self._in_floor_db = min(self._in_floor_db + 0.02, float(value))
        for value in out_db:
            self._out_floor_db = min(self._out_floor_db + 0.02, float(value))
        # Was ORION audible in the microphone just now? His voice reaches it
        # `delay_s` after it is played and rings on for a few hundred ms; until
        # that delay is measured, anything played in the last 0.8 s counts.
        if self.delay_s is None:
            lag, span = 0, int(0.8 * SAMPLE_RATE)
        else:
            lag, span = int(self.delay_s * SAMPLE_RATE), int(0.35 * SAMPLE_RATE)
        window = self._ref_tl.read(first - lag - span, span + count * FRAME)
        echo_expected = bool(_db(float(np.mean(window * window))) > -55.0)
        if echo_expected and self.cancelling and self.delay_s is not None:
            loud = in_db > self._in_floor_db + 10.0
            aligned = self._ref_tl.read(first - lag, count * FRAME).reshape(count, FRAME)
            mic = frames.reshape(count, FRAME)
            correlation = np.abs(np.sum(aligned * mic, axis=1)) / np.sqrt(
                np.maximum(1.0, np.sum(aligned ** 2, axis=1) * np.sum(mic ** 2, axis=1)))
            # Only reference-correlated, far-end-dominated input can establish
            # convergence. Double talk, unrelated noise and an unmeasured lag
            # are not evidence of successful echo cancellation.
            single_talk = loud & (correlation >= 0.65)
            if single_talk.any():
                self._erle.extend((in_db - out_db)[single_talk].tolist())
                self._last_echo_measure_at = stamp
            if len(self._erle) >= 40:
                p75 = float(np.percentile(np.fromiter(self._erle, dtype=np.float32), 75))
                if p75 >= self.CONVERGED_DB:
                    self.converged = True
                elif len(self._erle) >= self._erle.maxlen and p75 < self.CONVERGED_DB / 2:
                    self.converged = False
                self.erle_db = p75
            if stamp - self._last_echo_measure_at > 3.0:
                self.converged = False
        chunk_in = float(_db(np.mean(e_in)))
        chunk_out = float(_db(np.mean(e_out)))
        audible = chunk_out > self._out_floor_db + 8.0 and chunk_out > -60.0
        if not audible:
            self.near_end = False
        elif not echo_expected:
            self.near_end = True
        else:
            expected_residual = chunk_in - min(self.erle_db, 30.0)
            self.near_end = self.converged and chunk_out - expected_residual >= 10.0
        if self.near_end:
            self.near_end_at = stamp
        self._echo_expected = echo_expected

    _echo_expected = False

    def _gain(self, cleaned: np.ndarray, count: int) -> np.ndarray:
        """Speech-gated gain toward TARGET_DBFS, frozen while ORION is audible
        (his residual must not teach it), then a soft limiter."""
        energy = np.mean(cleaned.reshape(count, FRAME) ** 2, axis=1)
        levels = _db(energy)
        gains = np.empty(count + 1, dtype=np.float64)
        gains[0] = self._gain_db
        up = self.GAIN_UP_DB_PER_S * _FRAME_S
        down = self.GAIN_DOWN_DB_PER_S * _FRAME_S
        frozen = self._echo_expected
        for i, level in enumerate(levels.tolist()):
            self._agc_floor_db = min(self._agc_floor_db + 0.03, level)
            if not frozen and level > self._agc_floor_db + 12.0 and level > -60.0:
                self._level_db += 0.04 * (level - self._level_db)
                desired = min(self.MAX_GAIN_DB, max(self.MIN_GAIN_DB,
                                                    self.TARGET_DBFS - self._level_db))
                self._gain_db += min(up, max(-down, desired - self._gain_db))
            gains[i + 1] = self._gain_db
        # Ramp across each frame from the previous frame's gain: no zipper noise.
        edges = np.arange(count + 1, dtype=np.float64) * FRAME
        ramp = np.interp(np.arange(count * FRAME, dtype=np.float64), edges, gains)
        y = cleaned * np.power(10.0, ramp / 20.0).astype(np.float32)
        over = np.abs(y) > self.KNEE
        if over.any():
            room = self.CEILING - self.KNEE
            mag = self.KNEE + room * np.tanh((np.abs(y[over]) - self.KNEE) / room)
            y[over] = np.sign(y[over]) * mag
        return y


__all__ = ["FRAME", "SAMPLE_RATE", "CaptureChain", "PlaybackReference", "enabled",
           "speaker_reference"]
