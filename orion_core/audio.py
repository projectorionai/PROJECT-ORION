"""
Audio subsystem — capture, voice-activity gating, recognition, playback and
the Mark VIII speech pipeline.

Voice-system guarantees (objective 1 of the Mark VIII upgrade):

    1. PERMANENT VOICE — both speech paths read from the frozen
       ``constants.VOICE_PROFILE``.  The native Gemini channel is locked to
       one prebuilt male voice; the local fallback selects a professional
       male SAPI voice once at startup and never re-selects.

    2. NO CUT-OFFS — playback uses an unbounded queue (model audio streams
       faster than realtime; a bounded queue would drop chunks and
       time-compress speech), and ``is_active()`` keeps ORION in the
       SPEAKING state through the device-buffer tail so the last syllable
       is never clipped by a state change.

    3. SPEECH QUEUE — ``SpeechQueueManager`` serialises every utterance from
       both channels and is the single authority on "is ORION speaking?".

    4. SPEAK-THEN-LISTEN — the microphone gate is half-duplex by default:
       while the queue manager reports active output, captured audio is
       neither forwarded to the live channel nor fed to local recognition,
       so ORION always finishes speaking before listening resumes (and can
       never transcribe its own speaker output).  The legacy barge-in
       behaviour remains available via ORION_ALLOW_BARGE_IN=1.

Threading model (unchanged from Mark VII):
    PortAudio callbacks only copy bytes; VAD maths, Vosk inference and
    asyncio hand-off happen on the AudioGateThread worker, never on the
    device thread and never on the GUI event loop.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from array import array
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread, current_thread
from typing import Any, Callable

import numpy as np
# sounddevice costs ~162 ms (PortAudio + CFFI) and every use is inside a
# method, so it is deferred to the first audio call rather than paid
# before ORION's window exists. test_startup_performance asserts the
# package is still genuinely installed, so a missing dependency still
# fails loudly rather than at first speech.
from .lazy_import import LazyModule

sd = LazyModule("sounddevice")

from .audio_state import AudioStateMachine, SpeechState
from .bus import OrionBus
from .constants import (
    ALLOW_BARGE_IN,
    BARGE_IN_CONFIDENCE,
    BARGE_IN_MIN_AMPLITUDE,
    BARGE_IN_MIN_VOICED_MS,
    CHANNELS,
    CHUNK_SIZE,
    MIC_QUEUE_LIMIT,
    PLAYBACK_DEVICE_LATENCY_S,
    PLAYBACK_PREBUFFER_CHUNKS,
    PLAYBACK_PREBUFFER_MAX_WAIT,
    PLAYBACK_QUEUE_HIGH_WATER,
    PLAYBACK_TAIL_SECONDS,
    PRESENCE_SILENCE_SECONDS,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
    VAD_SAMPLE_LIMIT,
    VOICE_HANGOVER_SECONDS,
    VOICE_PROFILE,
    WAKE_WORDS,
)
from .utils import normalise_for_speech

from .voice_elevenlabs import ElevenLabsVoice, apply_fade
from .voice_elevenlabs import resolve_voice_id as _elevenlabs_voice_id

#: How many frames PortAudio hands over per CAPTURE callback.
#:
#: This was CHUNK_SIZE (512), which at 16 kHz is a 32 ms deadline on EVERY
#: callback — and the callback is a Python function contending for the GIL with
#: a Qt GUI, a Chromium WebEngine view, forge builds and model calls. Miss the
#: deadline and PortAudio reports "input overflow", which is what was filling
#: the log.
#:
#: Nothing downstream ever required 512-frame blocks — only the stream open
#: referenced CHUNK_SIZE, and the gate thread reads whatever arrives — so the
#: deadline is four times longer and the callback re-frames to CHUNK_SIZE
#: before handing anything on. The gate, the VAD and the live channel see
#: exactly what they saw before.
#:
#: 2048 frames is 128 ms of capture buffering, which is not perceptible on the
#: listening path: barge-in responsiveness is set by the VAD's own sustained
#: streak requirements, not by this.
CAPTURE_BLOCK_FRAMES = CHUNK_SIZE * 4


# ──────────────────────────────────────────────────────────────────────────────
# NATIVE AUDIO PLAYBACK  (Gemini Live PCM renderer)
# ──────────────────────────────────────────────────────────────────────────────

def _boost_thread_priority(label: str = "audio") -> None:
    """Ask Windows to schedule the calling thread ahead of ordinary work.

    ORION's renderer is a plain Python thread competing with the 3-D swarm,
    the face renderer and the telemetry loop.  Audio is the one subsystem
    where being late is immediately audible, so it gets priority.  Purely
    best-effort: a platform that refuses simply keeps the default, and the
    buffer headroom below carries it regardless.
    """
    try:
        import ctypes
        THREAD_PRIORITY_HIGHEST = 2
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST)
    except Exception:
        pass
    try:
        # A shorter switch interval lets a latency-sensitive thread reacquire
        # the GIL sooner after being descheduled.  5 ms (the default) is a long
        # time to wait when you owe the sound card a buffer.
        if sys.getswitchinterval() > 0.002:
            sys.setswitchinterval(0.002)
    except Exception:
        pass


class _VisualiserThread(Thread):
    """Computes the avatar's mouth shape and the amplitude orb OFF the write path.

    The renderer used to do this inline, once per 4 KB slice: a 512-sample
    Python loop, an FFT, and two Qt signal emissions for every 85 ms of audio.
    Under GIL contention that turned a 341 ms write into a 2.3 s one and broke
    the speech up audibly (measured).  Sampling the newest buffer at a fixed
    20 Hz gives the visuals everything they can actually show while touching
    the audio thread exactly once per slice, with a single assignment.
    """

    # 40 Hz. Twenty was chosen for an amplitude orb, where it is more than the
    # eye needs; a MOUTH is a different problem. A plosive can begin and end
    # inside one 50 ms frame, so at 20 Hz the closure that distinguishes /m/
    # from /n/ simply never reaches the face. The extra cost is one 1024-point
    # FFT per frame, which is tens of microseconds and stays off the write path
    # exactly as before.
    INTERVAL_S = 0.025

    def __init__(self, renderer: "AudioPlaybackThread") -> None:
        super().__init__(name="orion-audio-visualiser", daemon=True)
        self.renderer = renderer
        self.stop_event = Event()

    def run(self) -> None:
        emitted_silence = True
        while not self.stop_event.is_set():
            time.sleep(self.INTERVAL_S)
            slot = self.renderer._viz_slot
            self.renderer._viz_slot = None
            if slot is None:
                if not emitted_silence and not self.renderer.is_active():
                    emitted_silence = True
                    self._emit(0.0, {"low": 0.0, "mid": 0.0, "high": 0.0})
                continue
            emitted_silence = False
            try:
                chunk = bytes(slot)
                self._emit(self.renderer._amplitude(chunk),
                           self.renderer._spectral_bands(chunk))
                self._emit_viseme(chunk)
            except Exception:
                # The visuals are cosmetic; they must never take audio down.
                pass

    def _emit(self, amplitude: float, bands: dict) -> None:
        try:
            self.renderer.bus.amplitude.emit(amplitude)
            self.renderer.bus.voice_spectrum.emit(bands)
        except RuntimeError:
            pass       # Qt shutting down

    def _emit_viseme(self, chunk: bytes) -> None:
        """Measure a mouth posture from the audio and put it on the bus.

        Until now nothing in ORION emitted ``bus.viseme`` at all. The signal
        was declared, the Core Window subscribed to it, ``orion_core/viseme.py``
        implemented a whole scheduler for it — and no producer ever existed, so
        every face in the app has been driven by loudness alone while a
        complete lip-sync pipeline sat inert beside it.

        The posture is derived from the formant bands rather than the volume:
        the first formant gives how far the jaw is open, the second gives
        whether the lips are spread or rounded. That is physics, so it needs no
        transcript and no language. Lip CLOSURES cannot be measured this way —
        /m/, /b/ and /p/ look alike to a filter bank — so when a transcript is
        available ``VisemeStream`` supplies those on top; with none, the mouth
        runs on the measured shape, which is less detailed and never wrong.
        """
        try:
            import numpy as _np

            from .viseme import pcm_shapes

            usable = len(chunk) - (len(chunk) % 2)
            if usable < 128:
                return
            pcm = _np.frombuffer(chunk[:usable], dtype="<i2")
            frames = pcm_shapes(pcm, RECEIVE_SAMPLE_RATE)
            if not frames:
                return
            stream = getattr(self.renderer, "_viseme_stream", None)
            if stream is not None:
                frames = stream.frames(frames, 0.020)
            level, openness, width = frames[-1]
            # The echo guard needs to know what we just played in order to
            # subtract it from the microphone later. This thread already has
            # the PCM decoded, so feeding it here costs nothing extra.
            try:
                from .echo_guard import shared as _shared_guard

                _shared_guard().note_output(pcm, RECEIVE_SAMPLE_RATE,
                                            max(f[0] for f in frames))
            except Exception:
                pass
            if level <= 0.0:
                return
            self.renderer.bus.viseme.emit(
                {"open": float(openness), "width": float(width),
                 "closure": 0.0, "weight": float(level)})
        except RuntimeError:
            pass       # Qt shutting down
        except Exception:
            pass       # the mouth is cosmetic; never take audio down for it

    def stop(self) -> None:
        self.stop_event.set()


class AudioPlaybackThread(Thread):
    """
    Native PCM renderer with deterministic buffering (Mark IX).

    The queue is intentionally unbounded — the model streams faster than
    realtime and dropping chunks time-compresses speech.  Determinism comes
    from two thread-owned mechanisms instead of a bounded queue:

      • PREBUFFER — on the first chunk of a fresh utterance the thread waits
        until it has accumulated a small prebuffer (or a short cap elapses)
        before the first device write, so a cold device cannot underrun and
        clip the opening syllable.
      • TAIL-OWNED DRAIN — when the queue empties, the thread itself waits the
        playback tail for more audio before declaring the native source
        stopped.  Because the thread that knows the true device state owns the
        transition, there is no external poll and no empty-queue TOCTOU.

    The thread notifies the AudioStateMachine on start/stop, so the mic gate
    and HUD read the same flag it sets.
    """

    def __init__(
        self,
        bus: OrionBus,
        machine: AudioStateMachine | None = None,
        telemetry: Any | None = None,
    ) -> None:
        super().__init__(name="orion-audio-renderer", daemon=True)
        self.bus       = bus
        self.machine   = machine
        self.telemetry = telemetry
        self.queue: Queue[bytes | None] = Queue(maxsize=0)
        self.stop_event      = Event()
        self.last_audio_time = 0.0
        self._active         = False
        self._stream: Any    = None
        # Live output-device switch (user says "I can't hear you"): the request
        # is applied on THIS thread between writes, so the device stream is never
        # touched cross-thread.  None means "no switch pending".
        self._reopen_event   = Event()
        self._pending_device: int | None = None
        # True voice interruption (Mark X.5): while held, the renderer stops
        # writing but PRESERVES the queue and the unwritten remainder of the
        # current buffer, so resume continues from the exact interruption point
        # with no context loss and no response regeneration.
        self._hold           = Event()
        self._held_tail: bytes = b""
        # ── visualisation hand-off ────────────────────────────────────────────
        # The newest slice handed to the device, for the avatar's mouth and the
        # amplitude orb.  Written by the renderer with a bare assignment and
        # read by _VisualiserThread; deliberately NOT a queue or a lock, both of
        # which would put a contended synchronisation primitive back on the
        # write path.  A dropped or repeated sample is invisible at 20 Hz —
        # a stutter in the actual speech is not.
        self._viz_slot: Any = None
        self._visualiser = _VisualiserThread(self)

    def enqueue(self, chunk: bytes) -> None:
        if not chunk:
            return
        # Never drop model audio: every discarded chunk skips playback forward.
        self.queue.put_nowait(bytes(chunk))
        if self.telemetry is not None:
            depth = self.queue.qsize()
            self.telemetry.metrics.gauge("audio.playback.queue_depth", float(depth))
            if depth > PLAYBACK_QUEUE_HIGH_WATER:
                self.telemetry.metrics.incr("audio.playback.high_water")

    # Coalesce queued chunks into contiguous writes of up to this many bytes.
    # At 24 kHz / 16-bit mono this is ~0.34 s — large enough that PortAudio's
    # ring buffer never underruns between Python-side writes (the stutter fix),
    # small enough that an interruption is still responsive.
    _COALESCE_MAX_BYTES = 16384
    # Device writes are sliced this finely so a hold takes effect between
    # slices: ~85 ms at 24 kHz / 16-bit mono.  The slice being written when the
    # hold lands finishes playing (device-buffer depth), everything after it is
    # preserved verbatim for resume.
    _WRITE_SLICE_BYTES = 4096

    def _open_output_stream(self, device: Any) -> None:
        """Open (or reopen) the PortAudio output stream on *device*.

        blocksize=0 lets PortAudio pick its optimal transfer size; combined
        with coalesced multi-chunk writes this eliminates the per-512-frame
        underrun that caused stuttering.

        The explicit ``latency`` is the second half of the stutter fix.  The
        renderer is a Python thread that WILL occasionally be descheduled for
        tens of milliseconds when the GUI is busy — that is not something the
        audio path can prevent, only survive.  Asking PortAudio for a deep
        device buffer gives it that much slack to ride out a stall without the
        card ever running dry.  It costs latency we can afford: ORION's speech
        is a monologue, not a musical instrument, and a fraction of a second of
        extra buffering is imperceptible next to speech that breaks up.
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        try:
            self._stream = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=0,
                device=device,
                latency=PLAYBACK_DEVICE_LATENCY_S,
            )
        except Exception:
            # Some host APIs reject an explicit latency; the default is still
            # far better than no stream at all.
            self._stream = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=0,
                device=device,
            )
        self._stream.start()

    def _try_open(self, device: Any) -> bool:
        """Open the output stream on *device*, falling back to the system default
        if that specific device fails.  Never raises; returns True when a live
        stream is in place, False when no device could be opened at all (the
        thread then keeps retrying rather than dying)."""
        candidates: tuple[Any, ...] = (device, None) if device is not None else (None,)
        for candidate in candidates:
            try:
                self._open_output_stream(candidate)
                return True
            except Exception as exc:
                self.bus.log.emit(
                    f"AUDIO: output device open failed ({candidate!r}) - {exc}"
                )
        self._stream = None
        return False

    def request_output_device(self, index: int | None) -> None:
        """Ask the renderer to move to output device *index* at the next safe
        point (called from another thread; applied on the renderer thread)."""
        self._pending_device = index
        self._reopen_event.set()

    def _apply_pending_device(self) -> None:
        from .audio_devices import device_name, note_live_device
        self._reopen_event.clear()
        device = self._pending_device
        self._pending_device = None
        try:
            self._open_output_stream(device)
            note_live_device("output", device)
            self.bus.log.emit(
                f"AUDIO: output renderer moved to {device_name('output')}."
            )
            return
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: could not switch output device - {exc}")
        # The chosen device failed — never leave the renderer without a stream,
        # or the next write kills playback.  Fall back to the system default.
        try:
            self._open_output_stream(None)
            note_live_device("output", None)
            self.bus.log.emit("AUDIO: reverted voice output to the system default device.")
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: output device recovery failed - {exc}")

    def run(self) -> None:
        from .audio_devices import device_name, resolve
        # Audio is the one subsystem where being late is immediately audible.
        _boost_thread_priority("renderer")
        if not self._visualiser.is_alive():
            self._visualiser.start()
        try:
            if self._try_open(resolve("output")):
                self.bus.log.emit(
                    f"AUDIO: output renderer initialised — voice → {device_name('output')}."
                )
            else:
                self.bus.log.emit(
                    "AUDIO: no output device could be opened yet — retrying until one appears."
                )
            last_reopen_try = 0.0
            while not self.stop_event.is_set():
                try:
                    # ── live output-device switch requested (I can't hear you) ─
                    if self._reopen_event.is_set():
                        self._apply_pending_device()
                    # ── self-heal: never sit permanently silent.  If the stream
                    #    died (boot-time device failure, a mid-session unplug, or
                    #    a failed manual pick) keep trying to reopen one — this is
                    #    why changing the device used to make "no difference": the
                    #    renderer thread had already exited on the first fault.
                    if self._stream is None:
                        now = time.monotonic()
                        if now - last_reopen_try >= 1.0:
                            last_reopen_try = now
                            if self._try_open(resolve("output")):
                                self.bus.log.emit(
                                    f"AUDIO: output renderer recovered — voice → {device_name('output')}."
                                )
                        if self._stream is None:
                            time.sleep(0.1)
                            continue
                    # ── held (true interruption): stay silent, preserve everything
                    if self._hold.is_set():
                        if self._active:
                            self._mark_inactive()
                        time.sleep(0.04)
                        continue
                    # ── just resumed: flush the preserved remainder first ─────
                    if self._held_tail:
                        tail, self._held_tail = self._held_tail, b""
                        if not self._active:
                            self._active = True
                            if self.machine is not None:
                                self.machine.native_started()
                        self._write(tail)
                        continue
                    try:
                        chunk = self.queue.get(timeout=0.1)
                    except Empty:
                        # Queue drained: if we were speaking, run the tail-owned
                        # drain to decide whether the native source has truly ended.
                        if self._active and not self._drain_tail():
                            self._mark_inactive()
                        continue
                    if chunk is None:
                        break
                    if not self._active:
                        self._active = True
                        if self.machine is not None:
                            self.machine.native_started()
                        chunk = self._prebuffer(chunk)
                    # Merge everything already waiting into one large write.
                    self._write(self._coalesce(chunk))
                except Exception as exc:
                    # A device write/open fault must NOT terminate the renderer.
                    # Drop the dead stream and let the loop reopen one next pass.
                    self.bus.log.emit(f"AUDIO: output renderer recovered from fault - {exc}")
                    try:
                        if self._stream is not None:
                            self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                    self._mark_inactive()
                    time.sleep(0.05)
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: output renderer fault - {exc}")
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._mark_inactive()
            self.bus.amplitude.emit(0.0)

    # ── deterministic buffering helpers ───────────────────────────────────────

    def _coalesce(self, first: bytes) -> bytes:
        """Drain all immediately-available chunks into one contiguous buffer."""
        parts = [first]
        total = len(first)
        while total < self._COALESCE_MAX_BYTES:
            try:
                nxt = self.queue.get_nowait()
            except Empty:
                break
            if nxt is None:
                self.stop_event.set()
                break
            parts.append(nxt)
            total += len(nxt)
        return b"".join(parts) if len(parts) > 1 else first

    def _prebuffer(self, first_chunk: bytes) -> bytes:
        """
        Accumulate a small prebuffer before the first write of an utterance so
        a cold device cannot underrun and clip the opening syllable.  Returns
        the coalesced prebuffer as one buffer (bounded by a short time cap so
        it never adds perceptible latency).
        """
        parts = [first_chunk]
        total = len(first_chunk)
        target = max(1, PLAYBACK_PREBUFFER_CHUNKS) * CHUNK_SIZE * 2  # bytes
        deadline = time.monotonic() + PLAYBACK_PREBUFFER_MAX_WAIT
        while total < target and time.monotonic() < deadline and not self.stop_event.is_set():
            try:
                nxt = self.queue.get(timeout=0.02)
            except Empty:
                continue
            if nxt is None:
                self.stop_event.set()
                break
            parts.append(nxt)
            total += len(nxt)
        return b"".join(parts)

    def _drain_tail(self) -> bool:
        """
        Wait the playback tail for more audio.  Returns True if audio resumed
        (still speaking), False if the tail elapsed silent (utterance ended).
        """
        deadline = time.monotonic() + PLAYBACK_TAIL_SECONDS
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self._hold.is_set():
                return False  # held mid-tail: settle silent, queue preserved
            try:
                chunk = self.queue.get(timeout=0.03)
            except Empty:
                continue
            if chunk is None:
                self.stop_event.set()
                return False
            self._write(self._coalesce(chunk))
            return True
        return False

    def _write(self, buffer: bytes) -> None:
        """Sliced device write; a hold between slices parks the remainder.

        NOTHING but the write itself belongs in this loop.  It used to also
        compute an amplitude (a pure-Python scan of 512 samples), an FFT
        spectral profile, and emit two Qt signals — per 4 KB slice, i.e. six
        extra GIL acquisitions for every 85 ms of audio.

        Measured on this machine: idle, a 341 ms buffer took 322 ms to write
        (correct — the write blocks on the device consuming it).  With three
        Python threads competing for the GIL — which is exactly what the 3-D
        swarm, the face renderer and the telemetry loop are — the SAME buffer
        took 2311 ms, mean, peaking at 2884 ms.  The renderer was being
        descheduled between slices while PortAudio's buffer drained to empty.
        That is the stutter, and it is why it only ever happens in the real
        app and never on a bench.

        The analysis now happens on a separate visualiser thread that samples
        the most recent buffer; the only work here is a single reference
        assignment (one bytecode op) to hand it over.
        """
        if not buffer:
            return
        view = memoryview(buffer)
        offset = 0
        start = time.perf_counter()
        while offset < len(view):
            if self.stop_event.is_set():
                return
            if self._hold.is_set():
                # Preserve everything not yet handed to the device — this is
                # the exact resume point.  (At most one ~85 ms slice is already
                # in the device buffer and finishes playing.)
                self._held_tail = bytes(view[offset:])
                return
            slice_end = min(offset + self._WRITE_SLICE_BYTES, len(view))
            piece = view[offset:slice_end]
            # Hand the visualiser the newest audio without computing anything.
            self._viz_slot = piece
            self._stream.write(piece)
            self.last_audio_time = time.monotonic()
            offset = slice_end
        if self.telemetry is not None:
            # Once per buffer, not once per slice: telemetry is observability,
            # and observability must never be the reason audio breaks up.
            self.telemetry.metrics.observe(
                "audio.playback.write_latency_ms", (time.perf_counter() - start) * 1000.0
            )
            self.telemetry.metrics.gauge(
                "audio.playback.queue_depth", float(self.queue.qsize())
            )

    def _mark_inactive(self) -> None:
        if self._active:
            self._active = False
            self.bus.amplitude.emit(0.0)
            if self.machine is not None:
                self.machine.native_stopped()

    def stop(self) -> None:
        self.stop_event.set()
        self._hold.clear()
        self._visualiser.stop()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass

    def clear(self) -> None:
        """Flush pending playback immediately (destructive interruption only)."""
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self._held_tail = b""
        self._hold.clear()
        self.last_audio_time = 0.0
        self._mark_inactive()
        self.bus.amplitude.emit(0.0)

    # ── true interruption: hold / resume (queue position preserved) ───────────

    def hold(self) -> bool:
        """
        Silence playback within one write slice (~85 ms) while preserving the
        queue AND the unwritten remainder of the current buffer.  Returns True
        if audio was actually playing when the hold landed.
        """
        was_active = self._active
        self._hold.set()
        return was_active

    def resume_playback(self) -> bool:
        """Lift a hold; playback continues from the exact preserved point.
        Returns True when there is held or queued audio to continue with."""
        pending = bool(self._held_tail) or not self.queue.empty()
        self._hold.clear()
        return pending

    def held(self) -> bool:
        return self._hold.is_set()

    def is_active(self) -> bool:
        """True while this renderer is mid-utterance (thread-owned flag)."""
        return self._active

    def speaking_recently(self) -> bool:
        """Legacy alias retained for older call sites."""
        return self._active

    def _amplitude(self, chunk: bytes) -> float:
        if len(chunk) < 2:
            return 0.0
        sample_count = min(len(chunk) // 2, 512)
        if sample_count <= 0:
            return 0.0
        total  = 0
        stride = max(1, (len(chunk) // 2) // sample_count)
        for index in range(0, sample_count * stride * 2, stride * 2):
            if index + 1 >= len(chunk):
                break
            value = int.from_bytes(chunk[index:index + 2], "little", signed=True)
            total += abs(value)
        return min(1.0, (total / sample_count) / 32768.0)

    def _spectral_bands(self, chunk: bytes) -> dict[str, float]:
        """Rough vowel/consonant spectral profile — energy ratios in three
        broad bands — so the avatar's mouth can reflect what's being said,
        not just how loud it is: loudness alone can't distinguish an open
        'oh' from a narrow 'ee' at the same volume. A stylised heuristic
        (band-energy ratios), not real formant tracking — proportionate for
        a stylised avatar, not a broadcast-quality lip-sync product."""
        usable = len(chunk) - (len(chunk) % 2)
        if usable < 128:
            return {"low": 0.0, "mid": 0.0, "high": 0.0}
        try:
            pcm = np.frombuffer(chunk[:usable], dtype="<i2").astype(np.float32)
            spectrum = np.abs(np.fft.rfft(pcm * np.hanning(pcm.size)))
            freqs = np.fft.rfftfreq(pcm.size, d=1.0 / RECEIVE_SAMPLE_RATE)
            total = float(spectrum.sum()) + 1e-9
            band = lambda lo, hi: float(spectrum[(freqs >= lo) & (freqs < hi)].sum()) / total
            return {"low": band(80, 500), "mid": band(500, 2000), "high": band(2000, 6000)}
        except Exception:
            return {"low": 0.0, "mid": 0.0, "high": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL SPEECH SYNTHESIS  (offline fallback voice — profile-locked)
# ──────────────────────────────────────────────────────────────────────────────

class SpeechSynthesiser(Thread):
    """
    Local text-to-speech voice — gives ORION a spoken voice even when the
    native Gemini audio channel is offline.

    Mark VIII: the voice is selected ONCE at engine initialisation from the
    frozen VOICE_PROFILE male-voice search order and is never re-selected,
    so the local voice can never switch mid-session.  Utterances are consumed
    strictly in FIFO order from an internal queue — one at a time, each spoken
    to completion unless explicitly interrupted.
    """

    MAX_UTTERANCE_CHARS = 4000

    _SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]+|\S[^.!?…]*$")

    def __init__(
        self,
        bus: OrionBus,
        machine: AudioStateMachine | None = None,
        telemetry: Any | None = None,
    ) -> None:
        super().__init__(name="orion-local-voice", daemon=True)
        self.bus        = bus
        self.machine    = machine
        self.telemetry  = telemetry
        self.queue: Queue[str | None] = Queue()
        self.stop_event = Event()
        self.available  = True
        self.voice_name = "system default"
        self.state_cb: Callable[[bool], None] | None = None
        self._engine: Any = None
        self._proc: Any   = None
        self._speaking    = Event()
        self._interrupted = Event()
        # True voice interruption (Mark X.5): word-accurate hold/resume.  The
        # engine's word-boundary callback keeps ``_word_location`` current, so
        # a hold can preserve the unspoken remainder of the utterance and
        # resume() re-speaks from that exact word — no regeneration.
        self._held        = Event()
        self._current_text  = ""
        self._word_location = 0
        self._resume_text   = ""
        # Emotional delivery (#14): the current emotion nudges rate/volume per
        # utterance — subtly, and without ever changing the locked voice itself.
        # We only listen to the bus; the emotion engine is the single authority.
        self._emotion = "neutral"
        # ElevenLabs (Mark XX design-spec: "ElevenLabs Integration") — a
        # genuinely swappable voice provider for THIS fallback path only,
        # tried first whenever an API key + voice_id are configured, always
        # falling back to pyttsx3/PowerShell on a per-utterance failure so a
        # transient API outage never leaves ORION silent. Never wired into
        # the live Gemini channel — that is native audio streamed directly
        # from the model, a different system entirely.
        self._elevenlabs = ElevenLabsVoice()
        self._active_backend = ""   # "elevenlabs" | "pyttsx3" | "powershell"
        # The shared native renderer, attached by SpeechQueueManager.  When
        # present, ElevenLabs audio streams into ORION's single verified output
        # stream instead of opening a private one per utterance.
        self._renderer: Any = None
        try:
            bus.emotion_changed.connect(self._on_emotion_changed)
        except Exception:
            pass

    def _on_emotion_changed(self, name: Any = "neutral", params: Any = None) -> None:
        self._emotion = str(name or "neutral")

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def is_busy(self) -> bool:
        """Speaking now, or utterances still waiting in the queue."""
        return self._speaking.is_set() or not self.queue.empty() or bool(self._resume_text)

    def queue_depth(self) -> int:
        return self.queue.qsize()

    def speak(self, text: str) -> None:
        """Queue an utterance. Never interrupts what is currently being said."""
        text = re.sub(r"[*_`#]+", "", str(text or "")).strip()
        if not text or not self.available:
            return
        self.queue.put_nowait(text[: self.MAX_UTTERANCE_CHARS])

    def interrupt(self) -> None:
        """
        Explicit interruption (deliberate pause / emergency / shutdown): flush
        the queue and stop the current utterance immediately.  ``engine.stop()``
        is used to cut synthesis promptly so utterances can be spoken gaplessly
        in a single pass; it is only reached on a rare, user-initiated event,
        not on per-chunk polling.  The subprocess fallback kill is cross-thread
        safe.
        """
        self._interrupted.set()
        self._held.clear()
        self._resume_text = ""
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        if self._active_backend == "elevenlabs":
            # sd.stop() is documented cross-thread-safe and aborts an active
            # blocking sd.play() immediately, same as engine.stop()/proc.kill()
            # do for the other two backends. A harmless no-op if nothing
            # ElevenLabs-driven happens to be playing at this instant.
            try:
                sd.stop()
            except Exception:
                pass
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    # ── true interruption: hold / resume (utterance position preserved) ───────

    def hold(self) -> bool:
        """
        Pause the local voice, preserving the unspoken remainder of the current
        utterance (from the last word boundary) and everything still queued.
        Returns True if an utterance was actually cut short.  The engine.stop()
        here is a deliberate, user-initiated event — the same rare cross-thread
        path as interrupt(), never per-chunk polling.
        """
        was_speaking = self._speaking.is_set()
        self._held.set()
        if was_speaking:
            remainder = (self._current_text or "")[max(0, self._word_location):].strip()
            if remainder:
                self._resume_text = remainder
            self._interrupted.set()
            engine = self._engine
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            proc = self._proc
            if proc is not None:
                # PowerShell fallback has no word boundaries; replay in full.
                self._resume_text = self._current_text
                try:
                    proc.kill()
                except Exception:
                    pass
            if self._active_backend == "elevenlabs":
                # ElevenLabs playback has no word-boundary callback either (a
                # single REST response played back whole) — same shape as the
                # PowerShell fallback: replay the full utterance on resume
                # rather than an exact mid-word position.
                self._resume_text = self._current_text
                try:
                    sd.stop()
                except Exception:
                    pass
        return was_speaking

    def resume_speech(self) -> bool:
        """Lift a hold; the preserved remainder (if any) is spoken first.
        Returns True when there is held or queued speech to continue with."""
        pending = bool(self._resume_text) or not self.queue.empty()
        self._held.clear()
        return pending

    def held(self) -> bool:
        return self._held.is_set()

    def stop(self) -> None:
        self.stop_event.set()
        self.interrupt()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass

    def run(self) -> None:
        self._initialise_engine()
        while not self.stop_event.is_set():
            if self._held.is_set():
                time.sleep(0.05)   # held: stay silent, keep everything queued
                continue
            if self._resume_text:
                # Resume takes priority over queued utterances — continue the
                # interrupted sentence from the preserved word boundary.
                text, self._resume_text = self._resume_text, ""
            else:
                try:
                    text = self.queue.get(timeout=0.2)
                except Empty:
                    continue
                if text is None:
                    break
            self._interrupted.clear()
            self._speaking.set()
            if self.machine is not None:
                self.machine.tts_started()
            if self.state_cb is not None:
                try:
                    self.state_cb(True)
                except Exception:
                    pass
            started = time.perf_counter()
            try:
                self._speak_one(text)
            except Exception as exc:
                self.bus.log.emit(f"VOICE: local speech fault - {str(exc).splitlines()[0][:100]}")
            finally:
                self._speaking.clear()
                if self.telemetry is not None:
                    self.telemetry.metrics.observe(
                        "audio.tts.utterance_ms", (time.perf_counter() - started) * 1000.0
                    )
                if self.machine is not None:
                    self.machine.tts_stopped()
                if self.state_cb is not None:
                    try:
                        self.state_cb(False)
                    except Exception:
                        pass

    def _initialise_engine(self) -> None:
        """
        One-time engine + voice selection.  The chosen voice is logged and
        cached; nothing after this point may change it (voice-lock guarantee).

        The pyttsx3/PowerShell engine below is initialised UNCONDITIONALLY,
        even when ElevenLabs is configured and preferred — it is the fallback
        chain a mid-session API failure lands on, and this file's standing
        rule is that ORION never goes silent because one voice path failed.
        """
        if self._elevenlabs.available():
            self.bus.log.emit(
                f"VOICE: ElevenLabs voice ready (voice_id {_elevenlabs_voice_id()}); "
                "local engine still initialised underneath as the fallback.")
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            engine.setProperty("rate", VOICE_PROFILE.local_rate_wpm)
            engine.setProperty("volume", VOICE_PROFILE.local_volume)
            try:
                voices = engine.getProperty("voices") or []
                preferred = None
                # Walk the profile's search order — first pattern with a match
                # wins, guaranteeing a deterministic professional male voice.
                for pattern in VOICE_PROFILE.local_voice_patterns:
                    preferred = next(
                        (v for v in voices if re.search(
                            pattern, f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}"
                        )),
                        None,
                    )
                    if preferred is not None:
                        break
                if preferred is not None:
                    engine.setProperty("voice", preferred.id)
                    self.voice_name = str(getattr(preferred, "name", "") or preferred.id)
            except Exception:
                pass

            def _on_word(name: Any = None, location: int = 0, length: int = 0) -> None:
                # Word boundaries drive the HUD orb while the local voice
                # speaks — and record the exact utterance position so a hold
                # can resume from the word being spoken when it landed.
                try:
                    self._word_location = int(location)
                except (TypeError, ValueError):
                    pass
                self.bus.amplitude.emit(0.35 + 0.4 * random.random())

            try:
                engine.connect("started-word", _on_word)
            except Exception:
                pass
            self._engine = engine
            self.bus.log.emit(
                f"VOICE: local speech engine ready (pyttsx3, voice locked: {self.voice_name})."
            )
        except Exception:
            self._engine = None
            if sys.platform == "win32":
                self.voice_name = "Windows System.Speech (male hint)"
                self.bus.log.emit(
                    "VOICE: pyttsx3 not detected; using the Windows System.Speech male voice."
                )
            else:
                self.available = False
                self.bus.log.emit(
                    "VOICE: no local speech engine available; offline replies stay text-only."
                )

    def _speak_one(self, text: str) -> None:
        """Dispatch one utterance to whichever backend should speak it:
        ElevenLabs first when configured, falling back to the local engine
        for just this utterance if ElevenLabs itself produced no audio.
        Extracted from run()'s loop body so the dispatch/fallback logic is
        directly callable from tests without needing a real thread/queue.
        """
        spoken = False
        if self._elevenlabs.available():
            self._active_backend = "elevenlabs"
            spoken = self._speak_elevenlabs(text)
        if not spoken:
            if self._engine is not None:
                self._active_backend = "pyttsx3"
                self._speak_pyttsx3(text)
            else:
                self._active_backend = "powershell"
                self._speak_powershell(text)

    def _speak_elevenlabs(self, text: str) -> bool:
        """Try ElevenLabs for this one utterance. Returns True once the
        utterance is SETTLED — spoken, or deliberately skipped because it
        arrived already interrupted — so the caller knows not to also fall
        back to a local engine. Returns False only when ElevenLabs itself
        failed (no audio came back), which is the one case that should fall
        through to pyttsx3/PowerShell for this utterance.

        Streams into the shared renderer (Mark XXIV)
        --------------------------------------------
        This used to fetch the WHOLE utterance and then call
        ``sd.play(..., blocking=True)``, which was wrong in three ways:

          • no audio played until the entire synthesis had downloaded, so a
            long reply began with a long silence and a multi-utterance reply
            played as speech-gap-speech-gap;
          • ``sd.play`` opens its OWN output stream per utterance, so every
            sentence paid a device open/close (tens to hundreds of ms of dead
            air on Windows) — and it uses PortAudio's default device, not the
            one ORION verified, so his voice could come out of a different
            speaker than his native channel;
          • blocking inside this thread meant an interruption could not cut it.

        Now each chunk goes into ``AudioPlaybackThread``'s queue as it arrives
        off the wire: one persistent, verified, deep-buffered stream, shared
        with the native channel, with the prebuffer/coalescing/tail-drain and
        hold/resume behaviour already proven there.  This is the architecture
        the brief asks for — TTS stream -> audio queue -> dedicated worker ->
        device — rather than a second, private audio path.
        """
        if self._interrupted.is_set() or self.stop_event.is_set():
            return True
        self._current_text = text
        self._word_location = 0

        renderer = self._renderer
        if renderer is None:
            # No shared renderer wired (a bare SpeechSynthesiser, as in unit
            # tests): keep the original one-shot behaviour rather than losing
            # the voice entirely.
            return self._speak_elevenlabs_standalone(text)

        def stop_now() -> bool:
            return self._interrupted.is_set() or self.stop_event.is_set()

        streamed = {"bytes": 0}
        # MARK XL's idea, adapted to a stream: cap over-long punctuation pauses
        # so a multi-sentence reply keeps its rhythm instead of stopping dead
        # between clauses. State carries across chunk boundaries, because a
        # pause almost always straddles one (see audio_smoothing.py).
        from .audio_smoothing import SilenceCompressor
        smoother = SilenceCompressor(sample_rate=self._elevenlabs.SAMPLE_RATE)

        def feed(chunk: bytes) -> None:
            trimmed = smoother.feed(chunk)
            if trimmed:
                streamed["bytes"] += len(trimmed)
                renderer.enqueue(trimmed)

        try:
            pcm = self._elevenlabs.stream_pcm(
                text, emotion=self._emotion, on_chunk=feed, should_stop=stop_now)
        except Exception as exc:
            self.bus.log.emit(f"VOICE: ElevenLabs stream fault - {str(exc)[:100]}")
            pcm = None

        tail = smoother.flush()
        if tail:
            streamed["bytes"] += len(tail)
            renderer.enqueue(tail)
        if smoother.trimmed_ms > 250:
            self.bus.log.emit(f"VOICE: {smoother.describe()}")

        if streamed["bytes"] == 0:
            # Nothing ever reached the renderer — a genuine ElevenLabs failure.
            # Fall through to the local engine for this one utterance.
            return pcm is not None

        # Audio is in the renderer's queue.  Wait for it to actually finish so
        # utterances stay strictly FIFO and the state machine's "speaking"
        # window covers the whole thing — without this, the next utterance
        # would be queued on top of this one mid-sentence.
        self._await_renderer(renderer)
        return True

    def _await_renderer(self, renderer: Any, timeout: float = 300.0) -> None:
        """Block until the renderer has drained what we just gave it."""
        deadline = time.monotonic() + timeout
        # Give the renderer a moment to pick the first chunk up, so an empty
        # queue is never mistaken for "already finished".
        settle = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if self._interrupted.is_set() or self.stop_event.is_set():
                return
            if renderer.is_active() or not renderer.queue.empty():
                settle = time.monotonic() + 0.4
            elif time.monotonic() > settle:
                return
            time.sleep(0.02)

    def _speak_elevenlabs_standalone(self, text: str) -> bool:
        """One-shot ElevenLabs playback on its own stream.

        Only used when no shared renderer is attached.  Kept because a
        SpeechSynthesiser constructed on its own must still be able to speak.
        """
        pcm = self._elevenlabs.synthesize_pcm(text, emotion=self._emotion)
        if pcm is None:
            return False
        if self._interrupted.is_set() or self.stop_event.is_set():
            return True
        try:
            samples = np.frombuffer(pcm, dtype=np.int16)
            samples = apply_fade(samples)
            self.bus.amplitude.emit(0.4)
            sd.play(samples, samplerate=self._elevenlabs.SAMPLE_RATE, blocking=True)
        except Exception as exc:
            self.bus.log.emit(f"VOICE: ElevenLabs playback fault - {str(exc)[:100]}")
        self.bus.amplitude.emit(0.0)
        return True

    def attach_renderer(self, renderer: Any) -> None:
        """Share the native renderer so ElevenLabs audio uses ORION's one
        verified, deep-buffered output stream instead of opening its own."""
        self._renderer = renderer

    def _speak_pyttsx3(self, text: str) -> None:
        """
        Speak the whole utterance in a single synthesis pass so there are NO
        inter-sentence gaps — continuous, natural delivery (the user asked for
        speech without pausing).  Prompt interruption for an explicit pause or
        emergency stop is handled by ``interrupt()`` via ``engine.stop()``; that
        is a rare, deliberate, user-initiated event, not the per-chunk polling
        that would fragment the voice.
        """
        if self._interrupted.is_set() or self.stop_event.is_set():
            return
        self._current_text = text
        self._word_location = 0
        # Emotional prosody (#14): flex only delivery — rate and volume — within
        # a tight, non-creepy envelope keyed to the current emotion.  The voice
        # identity (which SAPI voice) is never touched, so the lock still holds.
        try:
            from .voice_emotion import prosody_for
            rate, volume = prosody_for(
                self._emotion, VOICE_PROFILE.local_rate_wpm, VOICE_PROFILE.local_volume)
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
        except Exception:
            pass
        self._engine.say(text)
        self._engine.runAndWait()
        self.bus.amplitude.emit(0.0)

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        parts = [m.group(0).strip() for m in cls._SENTENCE_RE.finditer(text)]
        return [p for p in parts if p]

    def _speak_powershell(self, text: str) -> None:
        # SelectVoiceByHints(Male) keeps the fallback consistent with the
        # locked profile even without pyttsx3 installed.
        self._current_text = text
        self._word_location = 0
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Male) } catch {}; "
            "$s.Rate = 0; "
            "$s.Speak([Console]::In.ReadToEnd())"
        )
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self._proc.stdin.write(text.encode("utf-8", errors="replace"))
            self._proc.stdin.close()
        except Exception:
            pass
        while self._proc.poll() is None:
            if self._interrupted.is_set() or self.stop_event.is_set():
                try:
                    self._proc.kill()
                except Exception:
                    pass
                break
            self.bus.amplitude.emit(0.3 + 0.4 * random.random())
            time.sleep(0.09)
        self.bus.amplitude.emit(0.0)
        self._proc = None


# ──────────────────────────────────────────────────────────────────────────────
# SPEECH QUEUE MANAGER  — single authority on "is ORION speaking?"
# ──────────────────────────────────────────────────────────────────────────────

def _join_bounded(thread: Any, timeout: float) -> bool:
    """Wait for *thread* to finish, for at most *timeout* seconds.

    Returns whether it actually stopped. Never raises and never deadlocks: a
    thread that was never started has no join to make, and a thread calling
    this on itself would wait forever.
    """
    try:
        if thread is None or thread is current_thread():
            return False
        if not thread.is_alive():
            return True
        thread.join(timeout=max(0.0, float(timeout)))
        return not thread.is_alive()
    except Exception:
        return False


class SpeechQueueManager:
    """
    Serialises and supervises everything ORION says.

    Both output channels feed through here:
        • native Gemini PCM  → AudioPlaybackThread (streamed chunks)
        • local fallback TTS → SpeechSynthesiser (queued utterances)

    Mark IX: this class no longer polls.  The AudioStateMachine is the single
    source of truth; the playback and TTS threads push transitions into it the
    instant they start or stop, and it fires ``bus.speaking`` + ``state_cb``
    from that transition.  ``output_active()`` reads the same machine flag the
    microphone gate uses, so gate and HUD can never desynchronise, and the
    speak-then-listen guarantee holds with zero poll latency.
    """

    def __init__(
        self,
        bus: OrionBus,
        playback: AudioPlaybackThread,
        tts: SpeechSynthesiser,
        machine: AudioStateMachine,
        telemetry: Any | None = None,
    ) -> None:
        self.bus       = bus
        self.playback  = playback
        self.tts       = tts
        self.machine   = machine
        self.telemetry = telemetry
        self._state_cb: Callable[[bool], None] | None = None
        #: Voice muted: both channels DROP what they are given rather than
        #: queue it, so unmuting never releases a backlog of stale sentences.
        self.muted = False
        # One output stream for ORION's whole voice.  The ElevenLabs path used
        # to open its own per utterance, which cost a device open/close between
        # every sentence and could route his voice to a different speaker than
        # the one he had just verified.
        self.tts.attach_renderer(self.playback)

    # ── state-callback bridge (worker marshals onto its loop) ─────────────────

    @property
    def state_cb(self) -> Callable[[bool], None] | None:
        return self._state_cb

    @state_cb.setter
    def state_cb(self, cb: Callable[[bool], None] | None) -> None:
        self._state_cb = cb
        # The machine already fires bus.speaking; wrap the caller's callback so
        # it only receives the boolean "speaking" edge, exactly as before.
        self.machine.on_transition = (
            (lambda _state, active: cb(active)) if cb is not None else None
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.playback.start()
        self.tts.start()
        self.bus.log.emit(
            f"VOICE: speech queue manager online — {VOICE_PROFILE.describe()}; "
            f"barge-in {'enabled' if ALLOW_BARGE_IN else 'disabled (speak-then-listen)'}; "
            "event-driven audio state machine active."
        )

    #: How long to wait for each audio thread to actually leave native code.
    #: Bounded deliberately: a wedged device must never hang shutdown, and
    #: app.py's watchdog sits behind this as the last resort.
    #:
    #: The value is not free to choose. It is spent INSIDE the window the
    #: watchdog allows, on top of the farewell ceiling, so two joins of four
    #: seconds put the worst legitimate shutdown at 25.3s against a 25s grace —
    #: the watchdog would have force-exited a shutdown that was working.
    #: tests/test_shutdown_budget.py holds the arithmetic.
    STOP_JOIN_SECONDS = 2.0

    def stop(self) -> None:
        """Silence the voice BEFORE closing the device it speaks through.

        These two ran in start order — playback first, then TTS — which is
        exactly backwards for shutdown. ``playback.stop()`` ends with the
        renderer closing its PortAudio stream, while ``tts.stop()`` had not yet
        asked SAPI5 to stop, so the speech thread could still be inside
        ``runAndWait()`` driving a stream being torn down underneath it.

        That is the precise pair of frames in ORION's STATUS_HEAP_CORRUPTION
        dumps — ``sounddevice.py:close`` as the faulting thread with
        ``pyttsx3/drivers/sapi5.py startLoop`` alongside it — recorded on
        2026-09-18 and again on 2026-09-20.

        Reversing the order is only half of it: both ``stop()`` methods just
        set a flag and return, so without a join teardown races the threads
        regardless. Each is now waited for, with a bound.
        """
        self.tts.stop()
        _join_bounded(self.tts, self.STOP_JOIN_SECONDS)
        self.playback.stop()
        _join_bounded(self.playback, self.STOP_JOIN_SECONDS)

    # ── speech submission ─────────────────────────────────────────────────────

    def set_muted(self, muted: bool) -> bool:
        """Mute or unmute ORION's voice. Muting also cuts what he is saying.

        Only OUTPUT is affected: he keeps listening (so "unmute" by voice
        works) and his replies still reach the screen as text."""
        self.muted = bool(muted)
        if self.muted:
            self.interrupt_all()
        try:
            self.bus.voice_muted.emit(self.muted)
        except Exception:
            pass
        return self.muted

    def enqueue_native_audio(self, chunk: bytes) -> None:
        """Stream a native PCM chunk (Gemini Live) into ordered playback."""
        if self.muted:
            return
        self.playback.enqueue(chunk)

    def speak_text(self, text: str) -> None:
        """Queue a local-voice utterance; spoken after everything already queued.

        The text is normalised for speech first — digit clocks, bare years and
        the all-caps name are rewritten as words the SAPI voice pronounces
        correctly ("22:27" was read as "20:27"; "ORION" as "ORIN")."""
        if self.muted:
            return
        self.tts.speak(normalise_for_speech(text))

    # ── state ─────────────────────────────────────────────────────────────────

    def output_active(self) -> bool:
        """True while either source is active — the machine's atomic flag."""
        return self.machine.is_active()

    def is_busy(self) -> bool:
        """Speaking now, or with anything still queued to speak.

        This method did not exist, and ``app._await_farewell`` polls exactly
        it. Its ``except Exception: return`` then swallowed the AttributeError
        on the first poll, so the careful "let him finish the sentence before
        tearing down" logic returned immediately every single time and ORION
        was cut off mid-goodbye — with the SAPI5 thread still inside the engine
        while teardown carried on around it.

        Broader than ``output_active()`` on purpose: the state machine flag
        goes false between two queued utterances, and a farewell that is still
        waiting its turn has very much not been said yet.
        """
        if self.machine.is_active():
            return True
        for source in (self.tts, self.playback):
            try:
                if source.is_busy():
                    return True
            except AttributeError:
                try:
                    if source.is_active():
                        return True
                except Exception:
                    continue
            except Exception:
                continue
        return False

    def speech_state(self) -> SpeechState:
        return self.machine.state

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        """Block (worker threads only) until all speech has completed."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if not self.output_active():
                return True
            time.sleep(0.05)
        return not self.output_active()

    def telemetry_snapshot(self) -> dict[str, Any]:
        """Live voice telemetry for the Command Centre."""
        return {
            **self.machine.describe(),
            "playback_queue_depth": self.playback.queue.qsize(),
            "tts_queue_depth": self.tts.queue_depth(),
            "local_voice": self.tts.voice_name,
            "held": self.output_held(),
        }

    # ── interruption (explicit only) ──────────────────────────────────────────

    def interrupt_all(self) -> bool:
        """Halt all speech immediately AND discard it. Returns True if anything
        was cut.  This is the destructive path (server barge-in, shutdown);
        for the user's spoken pause use hold_all(), which preserves position."""
        interrupted = False
        if self.playback.is_active():
            self.playback.clear()
            interrupted = True
        if self.tts.is_busy():
            self.tts.interrupt()
            interrupted = True
        # Force the machine through INTERRUPTED→IDLE so listeners settle now.
        self.machine.interrupted()
        return interrupted

    # ── true interruption (Mark X.5): hold / resume, nothing discarded ────────

    def hold_all(self) -> bool:
        """
        Silence both output channels immediately while preserving the playback
        queue position and the unspoken utterance remainder.  Returns True if
        anything was actually speaking.  The producing threads own the state
        transitions, so the microphone gate settles to idle the moment output
        stops — no external polling.
        """
        cut = self.playback.hold()
        cut = self.tts.hold() or cut
        return cut

    def resume_all(self) -> bool:
        """Resume from the exact interruption point on whichever channel was
        held.  Returns True when preserved speech actually continues."""
        resumed = self.playback.resume_playback()
        resumed = self.tts.resume_speech() or resumed
        return resumed

    def output_held(self) -> bool:
        return self.playback.held() or self.tts.held()

    # ── explicit output-device switch ("I can't hear you") ────────────────────

    def switch_output_device(self) -> str | None:
        """Rotate ORION's voice to the next available output device, persist the
        choice, and move the live renderer onto it.  Returns the new device name,
        or None when only one output device exists (nothing to switch to)."""
        from . import audio_devices as ad
        current = ad.resolve_effective("output")
        nxt = ad.next_index("output", current)
        if nxt is None:
            self.bus.log.emit("AUDIO: only one output device present — cannot switch.")
            return None
        index, name = nxt
        # Session-only: ORION follows the SYSTEM DEFAULT on every start (user
        # directive).  A rotation helps the user hear him *now*; to make a
        # device permanent they say "set output device …" (audio_devices tool).
        self.playback.request_output_device(index)   # move the live renderer now
        self.bus.log.emit(
            f"AUDIO: voice output switching to [{index}] {name} for this session "
            "(next start follows the system default).")
        return name

    def set_output_device(self, spec: Any, persist: bool = True) -> str | None:
        """Move ORION's voice to a SPECIFIC output device the user picked (by
        index or name fragment), and persist it.  The manual counterpart to
        switch_output_device's round-robin.  Returns the device name, or None
        when the spec matches no output device."""
        from . import audio_devices as ad
        spec_s = str(spec).strip()
        if spec_s.lower() in {"default", "reset", "auto", ""}:
            ad.set_device("output", "default")
            self.playback.request_output_device(None)     # follow the default now
            self.bus.log.emit("AUDIO: voice output reset to the system default.")
            return "system default"
        index = ad._match(spec_s, "output")
        if index is None:
            self.bus.log.emit(f"AUDIO: no speaker matches '{spec}'.")
            return None
        name = str(ad._devices()[index].get("name", "?"))
        if persist:
            ad.set_device("output", str(index))
        self.playback.request_output_device(index)
        self.bus.log.emit(f"AUDIO: voice output set to [{index}] {name} by the user.")
        return name


# ──────────────────────────────────────────────────────────────────────────────
# VOICE ACTIVITY DETECTION
# ──────────────────────────────────────────────────────────────────────────────

class SileroVADGatekeeper:
    """Local voice-activity gate with optional Silero inference and deterministic fallback."""

    def __init__(self, bus: OrionBus, threshold: float = 0.65,
                 defer: bool = False) -> None:
        """*defer* loads Silero on first use instead of during construction.

        Measured: ``_initialise_silero()`` costs ~1.5 s, and it ran inside the
        live worker's constructor on ORION's startup path — 1.5 s in which Qt
        could not pump a message. The deterministic amplitude fallback below
        works from the very first chunk, so deferring costs nothing but a
        slightly less clever gate for the first second or two of a session.
        """
        self.bus           = bus
        self.threshold     = max(0.05, min(0.95, float(threshold)))
        self._model: Any   = None
        self._torch: Any   = None
        self._fallback_floor = 0.012
        self._last_voice   = 0.0
        self._initialised  = False
        if not defer:
            self.ensure_ready()

    def ensure_ready(self) -> bool:
        """Load Silero if it has not been loaded. Idempotent, never raises."""
        if self._initialised:
            return self._model is not None
        self._initialised = True
        try:
            self._initialise_silero()
        except Exception:
            self._model = None
        return self._model is not None

    def accepts(self, chunk: bytes) -> bool:
        confidence = self.confidence(chunk)
        accepted   = confidence > self.threshold
        if accepted:
            self._last_voice = time.monotonic()
        return accepted

    def confidence(self, chunk: bytes) -> float:
        if not chunk:
            return 0.0
        if self._model is not None and self._torch is not None:
            try:
                pcm = array("h")
                pcm.frombytes(chunk[: min(len(chunk), VAD_SAMPLE_LIMIT * 2)])
                if sys.byteorder != "little":
                    pcm.byteswap()
                try:
                    tensor = (
                        self._torch.frombuffer(pcm, dtype=self._torch.int16)
                        .to(dtype=self._torch.float32) / 32768.0
                    )
                except Exception:
                    tensor = self._torch.tensor(pcm.tolist(), dtype=self._torch.float32) / 32768.0
                with self._torch.no_grad():
                    value = self._model(tensor, SEND_SAMPLE_RATE)
                return max(0.0, min(1.0, float(value.item() if hasattr(value, "item") else value)))
            except Exception:
                self._model = None
                self._torch = None
                self.bus.log.emit("AUDIO: packaged Silero VAD unavailable; local acoustic gate active.")
        return self._local_confidence(chunk)

    def _initialise_silero(self) -> None:
        try:
            import torch  # type: ignore
            from silero_vad import load_silero_vad  # type: ignore
            self._torch = torch
            self._model = load_silero_vad()
            if hasattr(self._model, "eval"):
                self._model.eval()
            self.bus.log.emit("AUDIO: local Silero VAD gatekeeper initialised.")
        except Exception:
            self._model = None
            self._torch = None
            self.bus.log.emit("AUDIO: packaged Silero VAD not detected; local acoustic gate active.")

    def _local_confidence(self, chunk: bytes) -> float:
        pcm    = array("h")
        usable = len(chunk) - (len(chunk) % 2)
        if usable <= 0:
            return 0.0
        pcm.frombytes(chunk[:usable])
        if sys.byteorder != "little":
            pcm.byteswap()
        if not pcm:
            return 0.0
        sample_count = len(pcm)
        stride       = max(1, sample_count // VAD_SAMPLE_LIMIT)
        total_sq     = 0.0
        total_abs    = 0.0
        peak         = 0
        crossings    = 0
        previous     = 0
        used         = 0
        for sample in pcm[::stride]:
            value    = int(sample)
            total_sq += value * value
            absolute  = abs(value)
            total_abs += absolute
            if absolute > peak:
                peak = absolute
            if used and ((value >= 0) != (previous >= 0)):
                crossings += 1
            previous = value
            used     += 1
        if used <= 0:
            return 0.0
        rms        = math.sqrt(total_sq / used) / 32768.0
        mean_abs   = (total_abs / used) / 32768.0
        peak_norm  = peak / 32768.0
        zcr        = crossings / max(1, used - 1)
        speech_band  = 1.0 - min(1.0, abs(zcr - 0.075) / 0.16)
        energy_score = max(0.0, min(1.0, (rms - self._fallback_floor) / 0.055))
        peak_score   = max(0.0, min(1.0, (peak_norm - 0.04) / 0.28))
        compactness  = max(0.0, min(1.0, mean_abs / max(0.0001, rms * 0.82)))
        confidence   = (
            energy_score * 0.52
            + speech_band * 0.24
            + peak_score  * 0.16
            + compactness * 0.08
        )
        if rms < self._fallback_floor:
            confidence *= max(0.0, rms / self._fallback_floor)
        return max(0.0, min(1.0, confidence))


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL SPEECH RECOGNITION  (Vosk — optional)
# ──────────────────────────────────────────────────────────────────────────────

class LocalSpeechRecogniser:
    """
    Offline speech recognition (Vosk) used for wake-word activation and local
    transcript logging.  Optional dependency: if Vosk or its model is absent,
    `available` stays False and the wake-word gate is disabled (mic always live).
    """

    def __init__(self, bus: OrionBus, sample_rate: int = SEND_SAMPLE_RATE) -> None:
        self.bus        = bus
        self.available  = False
        self._recogniser: Any = None
        self._model: Any = None
        self._lock      = RLock()
        self._sample_rate = sample_rate
        # Model loading (and a possible first-run download) can take seconds;
        # it must never block the GUI event loop.
        Thread(target=self._initialise, name="orion-vosk-loader", daemon=True).start()

    def _initialise(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
            SetLogLevel(-1)
            model_path = os.getenv("ORION_VOSK_MODEL", "").strip()
            model = Model(model_path) if model_path else Model(lang="en-us")
            with self._lock:
                self._model      = model
                self._recogniser = KaldiRecognizer(model, float(self._sample_rate))
                self.available   = True
            self.bus.log.emit("SR: local Vosk recogniser initialised; wake-word gate armed.")
        except Exception as exc:
            self.bus.log.emit(
                "SR: local recognition unavailable "
                f"({str(exc).splitlines()[0][:90]}); wake-word gate disabled."
            )

    def create_command_recogniser(self, phrases: list[str]) -> Any:
        """
        Build a lightweight sibling recogniser CONSTRAINED to a phrase grammar,
        sharing the already-loaded model (no extra model memory).  Used by the
        VoiceInterruptManager so interruption commands are matched with high
        precision even against the acoustic bleed of ORION's own voice.
        Returns None until the model has finished loading.
        """
        with self._lock:
            if self._model is None:
                return None
            try:
                from vosk import KaldiRecognizer  # type: ignore
                grammar = json.dumps([*phrases, "[unk]"])
                return KaldiRecognizer(self._model, float(self._sample_rate), grammar)
            except Exception:
                return None

    def create_open_recogniser(self) -> Any:
        """
        Build an unconstrained sibling recogniser sharing the already-loaded
        model (no extra model memory). Used only to decode a sustained
        barge-in candidate so it can be compared against ORION's own recent
        speech (see live_worker._is_own_echo) — unlike
        create_command_recogniser, this has no phrase grammar because a real
        interruption can be any words at all, not a fixed command set.
        Returns None until the model has finished loading.
        """
        with self._lock:
            if self._model is None:
                return None
            try:
                from vosk import KaldiRecognizer  # type: ignore
                return KaldiRecognizer(self._model, float(self._sample_rate))
            except Exception:
                return None

    def feed(self, chunk: bytes) -> str:
        """Feed 16 kHz PCM; returns final text, or partial text if it contains a wake word."""
        if not self.available or self._recogniser is None or not chunk:
            return ""
        try:
            with self._lock:
                if self._recogniser.AcceptWaveform(chunk):
                    result = json.loads(self._recogniser.Result() or "{}")
                    return str(result.get("text") or "").strip()
                partial_payload = json.loads(self._recogniser.PartialResult() or "{}")
                partial = str(partial_payload.get("partial") or "").strip()
                if partial and any(word in partial.lower() for word in WAKE_WORDS):
                    try:
                        self._recogniser.Reset()
                    except Exception:
                        pass
                    return partial
        except Exception:
            return ""
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO GATE  — capture gating, VAD and qasync hand-off (off device thread)
# ──────────────────────────────────────────────────────────────────────────────

class RecentAudio:
    """The last few seconds of what the microphone heard, for deliberate
    listening ("what's that sound?", "what song is this?").

    Fed by the gate thread from chunks it already has — no second stream on
    the device (two handles on one input is how Windows capture fails), and
    retroactive: ORION can analyse a sound that ALREADY happened. Each chunk
    is tagged with whether ORION himself was speaking, so his own voice can be
    left out. 30 s of 16 kHz int16 is under 1 MB.
    """

    def __init__(self, seconds: float = 30.0, sample_rate: int = SEND_SAMPLE_RATE) -> None:
        from collections import deque
        self.sample_rate = int(sample_rate)
        self.limit = int(seconds * sample_rate * 2)          # bytes of int16 mono
        self._chunks: Any = deque()
        self._size = 0
        self._lock = RLock()
        self.last_at = 0.0

    def push(self, chunk: bytes, speaking: bool = False) -> None:
        if not chunk:
            return
        with self._lock:
            self._chunks.append((time.monotonic(), bytes(chunk), bool(speaking)))
            self._size += len(chunk)
            while self._size > self.limit and self._chunks:
                self._size -= len(self._chunks.popleft()[1])
            self.last_at = time.monotonic()

    def snapshot(self, seconds: float, *, include_own_voice: bool = False) -> bytes:
        """The most recent *seconds* of audio as int16 mono PCM bytes."""
        cutoff = time.monotonic() - float(seconds)
        with self._lock:
            parts = [c for at, c, own in self._chunks
                     if at >= cutoff and (include_own_voice or not own)]
        return b"".join(parts)

    def fresh(self, within_s: float = 2.0) -> bool:
        """Whether the microphone pipeline is feeding the buffer right now."""
        return time.monotonic() - self.last_at <= within_s


#: The process-wide buffer (the gate thread writes, sound_sense reads).
RECENT_AUDIO = RecentAudio()


class AudioGateThread(Thread):
    """
    Real-time audio gatekeeper.

    PortAudio callbacks must not perform Torch inference, VAD maths, asyncio
    queue mutation, or expensive Python loops.  The callback copies bytes and
    exits; this worker performs capture gating, VAD, amplitude calculation,
    and qasync hand-off away from the audio device thread.

    Half-duplex rule (Mark VIII): while ``speaking_check()`` reports active
    output, NOTHING is forwarded to the live channel and nothing is fed to
    local recognition — ORION finishes speaking before listening resumes.
    On each speaking→listening transition the raw queue is drained so stale
    chunks captured during ORION's own speech can never masquerade as a user
    turn (which previously caused phantom interruptions and cut-offs).
    Setting ORION_ALLOW_BARGE_IN=1 restores voice interruption for
    high-confidence speech.

    Mark X.5 — true voice interruption: the half-duplex rule previously made
    ORION deaf to "ORION stop" while he was speaking.  The gate now feeds
    every chunk captured DURING speech to a lightweight, grammar-constrained
    interruption listener (``interrupt_feed``); a matched command phrase is
    marshalled to the worker via ``on_interrupt``.  Nothing else changes —
    the chunk is still never forwarded to the live channel or the general
    recogniser, so the speak-then-listen guarantee holds.

    Mark X.9 — hardened open-mic barge-in: the legacy ORION_ALLOW_BARGE_IN
    path used to fire ``on_barge_in`` off a single qualifying VAD chunk,
    which is exactly what made it unsafe (ORION's own voice bleeding into
    the mic is real, VAD-qualifying, grammatical speech too). It now
    requires a SUSTAINED high-confidence, above-floor streak
    (``BARGE_IN_MIN_VOICED_MS``/``BARGE_IN_MIN_AMPLITUDE``) that decodes
    (via ``is_own_echo``) to words that are NOT what ORION is currently
    saying before committing to the destructive interrupt.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        out_q: asyncio.Queue,
        bus: OrionBus,
        can_capture: Callable[[], bool],
        vad: SileroVADGatekeeper,
        raw_limit: int = MIC_QUEUE_LIMIT,
        recogniser: "LocalSpeechRecogniser | None" = None,
        on_transcript: Callable[[str], None] | None = None,
        speaking_check: Callable[[], bool] | None = None,
        on_barge_in: Callable[[], None] | None = None,
        interrupt_feed: Callable[[bytes], str] | None = None,
        on_interrupt: Callable[[str], None] | None = None,
        on_input_silent: Callable[[], None] | None = None,
        on_input_restored: Callable[[], None] | None = None,
        speaker_tracker: Any | None = None,
        is_own_echo: Callable[[str], bool] | None = None,
        command_listen_check: Callable[[], bool] | None = None,
        voice_presence: Any | None = None,
    ) -> None:
        super().__init__(name="orion-audio-gatekeeper", daemon=True)
        self.loop        = loop
        self.out_q       = out_q
        self.bus         = bus
        # Wrapped, not replaced: every existing reason not to capture —
        # paused, muted, no device — still applies. Push-to-talk only ever
        # subtracts, and with it off this is the caller's answer unchanged.
        self._raw_can_capture = can_capture
        self.can_capture = self._capture_allowed
        self.vad         = vad
        self.recogniser     = recogniser
        self.on_transcript  = on_transcript
        self.speaking_check = speaking_check
        self.on_barge_in    = on_barge_in
        self.interrupt_feed = interrupt_feed
        self.on_interrupt   = on_interrupt
        # True when ORION should still LISTEN FOR COMMANDS even though he is not
        # speaking — i.e. while paused. This is the "keep his hearing active,
        # just not his interactions" separation: the dedicated command-grammar
        # listener runs, so "ORION resume" is heard, while the live interaction
        # channel stays gated off. Without it, pausing made him deaf to the very
        # word meant to bring him back.
        self.command_listen_check = command_listen_check
        # Hardened open-mic barge-in (Mark X.9): a real interruption must be
        # corroborated against ORION's own live utterance before it's
        # trusted — see the class docstring.
        self.is_own_echo = is_own_echo
        # Set while ORION is delivering something LONG — the morning greeting
        # and briefing. Barge-in is off by default because his own voice
        # bleeding into the mic clears the same VAD and amplitude bar as a real
        # interruption, and a false trigger mid-answer is worse than waiting.
        # A monologue is the opposite trade: it runs for a minute or more, and
        # the user may have something urgent. Every piece of the hardening
        # below still applies — this only decides whether the path is REACHED,
        # never how easily it fires.
        self._interruptible = Event()
        self._barge_in_streak_started: float | None = None
        self._barge_in_rec: Any = None
        # Speaker gender recognition (#14): voiced user chunks are fed to a
        # tracker that estimates whether the speaker is male or female.  Feeding
        # only appends bytes; the tracker throttles the (sub-ms) evaluation, and
        # feed() itself never raises — the live mic path stays untouched.
        self.speaker_tracker = speaker_tracker
        # Who is speaking, and how they sound (not merely male or female).
        #
        # The offline transcriber is handed a whole utterance and reads this in
        # one place. Live has no such moment — it forwards every chunk so the
        # server's VAD can find end-of-turn — so the utterance is assembled
        # here and handed over when the speech stops. Accumulating costs a
        # list append per chunk; the embedding and the pitch tracking happen on
        # another thread entirely, because this one has a 32 ms deadline.
        self.voice_presence = voice_presence
        self._presence_tasks = set()
        try:
            from .live_presence import LivePresenceBridge

            self._presence = LivePresenceBridge(
                presence=voice_presence, schedule=self._run_off_thread,
                sample_rate=SEND_SAMPLE_RATE,
                log=lambda message: self.loop.call_soon_threadsafe(
                    self.bus.log.emit, message))
        except Exception:
            # Reading who is speaking is a nice-to-have. Starting the
            # microphone is not, and the first must never be able to stop the
            # second — a paused ORION listening for "resume" depends on this
            # loop running whatever else is broken.
            self._presence = None
        # Dead-microphone detection: if the input flatlines (exact digital
        # silence — a muted, disconnected or wrong device, NOT a quiet room,
        # whose mic still has a noise floor) while ORION should be listening,
        # on_input_silent fires so the engine can rotate to another mic.
        self.on_input_silent   = on_input_silent
        self.on_input_restored = on_input_restored
        self._last_signal_at = time.monotonic()
        self._silence_fired  = False
        self.raw_q: Queue[bytes | None] = Queue(maxsize=max(8, int(raw_limit or MIC_QUEUE_LIMIT)))
        self.stop_event   = Event()
        self._voice_until = 0.0
        self._was_speaking = False
        # Echo guard: after ORION stops speaking, ignore captured audio for a
        # short window so the acoustic tail / room echo of his OWN voice is
        # never transcribed and answered again (the "repeats himself" bug).
        self._deaf_until = 0.0

    #: In-flight presence work, held so the tasks are not collected mid-flight.
    #: A class-level default for the same reason _presence has one: a gate
    #: built with __new__ must not raise from the capture loop.
    _presence_tasks: set = frozenset()

    #: Default, so every instance has one however it was constructed. A test
    #: that builds this class with __new__ and sets only the fields the
    #: capture loop touches is a reasonable test of the capture loop; it
    #: should not have to know that reading the speaker was added later.
    _presence = None

    # Deafness window after speech ends (seconds).
    #
    # This is a blunt instrument and always was: for most of a second after
    # every reply ORION cannot hear anything at all, so answering the instant
    # he stops — which is how people actually talk — does not work. It stays as
    # the floor because it cannot fail, and `_past_echo_guard` lifts it early
    # whenever the microphone is carrying something that demonstrably is not
    # ORION's own voice.
    _ECHO_GUARD_SECONDS = 0.9

    #: ORION_ACOUSTIC_ECHO=0 restores the plain timed window.
    _ACOUSTIC_ECHO = os.getenv("ORION_ACOUSTIC_ECHO", "1").strip().lower() not in {
        "0", "false", "no", "off"}

    def _capture_allowed(self) -> bool:
        """The caller's own gate, AND push-to-talk when it is switched on.

        A push-to-talk that nothing consults is dead code, and ORION already
        had one of those: bus.viseme was declared, subscribed to and fully
        implemented for a whole release without a single producer. So the gate
        is reached from here directly rather than waiting for a settings screen
        to exist.
        """
        try:
            if not self._raw_can_capture():
                return False
        except Exception:
            return False
        try:
            from .push_to_talk import gate_allows_capture

            return gate_allows_capture()
        except Exception:
            return True          # a broken gate must never make ORION deaf

    def _past_echo_guard(self, chunk: bytes, now: float) -> bool:
        """True when captured audio may be trusted again.

        The timed window is the fallback. Inside it, the acoustic guard is
        asked whether this block is ORION's own voice returning: it reduces
        both streams to band energies and subtracts as much of what was just
        played as fits. Pure echo cancels to almost nothing; a second voice
        survives, because its formants sit in bands where ORION's were weak.
        That distinction holds even when the two arrive at the same loudness,
        which is precisely the case a loudness test gets wrong.

        Never raises, and any doubt resolves to "keep waiting" — a guard that
        opened the microphone on an error would have ORION answering himself,
        which is the bug this whole mechanism exists to prevent.
        """
        if now >= self._deaf_until:
            return True
        if not self._ACOUSTIC_ECHO:
            return False
        try:
            import numpy as _np

            from .echo_guard import shared as _shared_guard

            usable = len(chunk) - (len(chunk) % 2)
            if usable < 256:
                return False
            pcm = _np.frombuffer(chunk[:usable], dtype="<i2")
            is_echo, residual = _shared_guard().classify(pcm, SEND_SAMPLE_RATE)
            if is_echo or residual <= 0.0:
                return False
            # Demonstrably not ORION: stop being deaf for the rest of the window.
            self._deaf_until = 0.0
            return True
        except Exception:
            return False
    # Dead-mic watchdog: flatline floor (0-1 amplitude) and how long the input
    # may stay below it, while ORION is listening, before raising on_input_silent
    # (which now prompts a presence check upstream — no automatic device hop).
    _SIGNAL_FLOOR = 0.004
    _SILENCE_TIMEOUT = PRESENCE_SILENCE_SECONDS
    _WATCHDOG_ON = os.getenv("ORION_MIC_WATCHDOG", os.getenv("ORION_MIC_AUTOSWITCH", "1")).strip().lower() not in {"0", "false", "no", "off"}

    def note_signal(self) -> None:
        """Externally reset the dead-mic clock (e.g. just after a device swap)."""
        self._last_signal_at = time.monotonic()
        self._silence_fired = False

    def _check_input_silence(self) -> None:
        """Raise on_input_silent once if the input has flatlined while ORION is
        listening — the caller decides what to do (a presence check), instead of
        this thread silently rotating the microphone."""
        if self.on_input_silent is None or not self._WATCHDOG_ON:
            return
        now = time.monotonic()
        speaking = self.speaking_check() if self.speaking_check is not None else False
        # Only count silence when ORION is actually meant to be hearing the
        # user; a pause, a tool run, standby or ORION's own speech must never
        # be mistaken for a dead mic.
        listening = self.can_capture() and not speaking
        if not listening:
            self._last_signal_at = now
            self._silence_fired = False
            return
        if not self._silence_fired and (now - self._last_signal_at) > self._SILENCE_TIMEOUT:
            self._silence_fired = True
            try:
                self.on_input_silent()
            except Exception:
                pass

    def enqueue(self, chunk: bytes) -> None:
        if not chunk:
            return
        try:
            self.raw_q.put_nowait(bytes(chunk))
            return
        except Full:
            pass
        try:
            self.raw_q.get_nowait()
        except Empty:
            pass
        try:
            self.raw_q.put_nowait(bytes(chunk))
        except Full:
            pass

    def drain(self) -> None:
        while True:
            try:
                self.raw_q.get_nowait()
            except Empty:
                break

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.raw_q.put_nowait(None)
        except Full:
            pass

    def run(self) -> None:
        while not self.stop_event.is_set():
            # Dead-mic watchdog runs every tick, even when no audio arrives at
            # all (a disconnected device delivers no callbacks) — that silence
            # is exactly what we want to catch.
            self._check_input_silence()
            try:
                chunk = self.raw_q.get(timeout=0.05)
            except Empty:
                continue
            if chunk is None:
                break
            try:
                try:
                    RECENT_AUDIO.push(chunk, self.speaking_check() if self.speaking_check else False)
                except Exception:
                    pass
                confidence = self.vad.confidence(chunk)
                now    = time.monotonic()
                voiced = confidence > self.vad.threshold
                # Any real input signal (voice OR room noise floor) proves the
                # mic is alive; only exact digital silence counts as a dead mic.
                if self._amplitude(chunk) > self._SIGNAL_FLOOR:
                    self._last_signal_at = now
                    if self._silence_fired:
                        self._silence_fired = False
                        if self.on_input_restored is not None:
                            try:
                                self.on_input_restored()
                            except Exception:
                                pass
                if voiced:
                    self._voice_until = now + VOICE_HANGOVER_SECONDS
                in_speech = voiced or now < self._voice_until
                speaking  = self.speaking_check() if self.speaking_check is not None else False

                # ── half-duplex transitions ──────────────────────────────────
                if speaking and not self._was_speaking:
                    # ORION just started speaking: everything captured before
                    # this instant is either the user's finished turn (already
                    # forwarded) or room echo — drop it.
                    self.drain()
                elif self._was_speaking and not speaking:
                    # ORION just finished: drop audio captured *during* speech
                    # and open an echo-guard window so his own trailing sound is
                    # not transcribed as a fresh user turn.
                    self.drain()
                    self._voice_until = 0.0
                    self._deaf_until = now + self._ECHO_GUARD_SECONDS
                self._was_speaking = speaking

                # Speaker gender recognition (#14): only the user's own voice,
                # only while ORION is listening — never his own output or echo.
                own_voice = (
                    voiced
                    and not speaking
                    and self._past_echo_guard(chunk, now)
                    and self.can_capture()
                )
                if own_voice and self.speaker_tracker is not None:
                    self.speaker_tracker.feed(chunk, SEND_SAMPLE_RATE)

                # The user's utterance, assembled for VoicePresence. Same
                # conditions as the gender tracker above: only real speech,
                # never ORION's own output or its echo.
                presence = self._presence
                if presence is not None and own_voice:
                    presence.utterance.feed(chunk)
                    if (self._speaker_judged()
                            and presence.utterance.ready_to_decide()):
                        # Asked once, part-way through: a stranger is cut off
                        # early rather than answered. Nobody can be identified
                        # before they have spoken, so the opening second does
                        # reach the model — that is a property of the problem,
                        # not of this code.
                        presence.decide(presence.utterance.audio(),
                                        self._on_speaker_verdict)
                elif (presence is not None and not in_speech
                        and presence.utterance.seconds > 0):
                    # Speech ended. Hand the whole thing over at once.
                    finished = presence.utterance.end()
                    if finished:
                        presence.observe(finished)

                # Command-grammar listener (Mark X.5, extended): command phrases
                # like "ORION stop" / "ORION resume" are matched by the
                # grammar-constrained listener and honoured instantly. It runs
                # while ORION is SPEAKING (interrupt him) AND while he is PAUSED
                # (bring him back) — the latter is the fix for "he can't hear me
                # when paused". It is separate from the interaction channel, so
                # hearing a command never means acting on general speech.
                command_listening = speaking or (
                    self.command_listen_check is not None
                    and self.command_listen_check())
                if command_listening and self.interrupt_feed is not None:
                    try:
                        phrase = self.interrupt_feed(chunk)
                    except Exception:
                        phrase = ""
                    if phrase and self.on_interrupt is not None:
                        self.loop.call_soon_threadsafe(self.on_interrupt, phrase)

                if speaking:
                    if not (ALLOW_BARGE_IN or self._interruptible.is_set()):
                        # Speak-then-listen: swallow the chunk entirely.
                        continue
                    # Hardened barge-in (Mark X.9): a single qualifying chunk
                    # is not evidence of a real interruption — ORION's own
                    # voice bleeding into the mic clears the same VAD/
                    # amplitude bar. Require a SUSTAINED streak, then decode
                    # it and reject anything that matches what he's saying.
                    amplitude = self._amplitude(chunk)
                    qualifies = (
                        voiced
                        and confidence >= BARGE_IN_CONFIDENCE
                        and amplitude >= BARGE_IN_MIN_AMPLITUDE
                    )
                    if not qualifies:
                        self._barge_in_streak_started = None
                        continue
                    if self._barge_in_streak_started is None:
                        self._barge_in_streak_started = now
                        if self._barge_in_rec is None and self.recogniser is not None:
                            self._barge_in_rec = self.recogniser.create_open_recogniser()
                    streak_ms = (now - self._barge_in_streak_started) * 1000.0
                    if streak_ms < BARGE_IN_MIN_VOICED_MS:
                        continue
                    text = ""
                    if self._barge_in_rec is not None:
                        try:
                            if self._barge_in_rec.AcceptWaveform(chunk):
                                result = json.loads(self._barge_in_rec.Result() or "{}")
                                text = str(result.get("text") or "").strip()
                        except Exception:
                            text = ""
                    if text and self.is_own_echo is not None and self.is_own_echo(text):
                        # Corroborated as ORION's own bleed-through — not a
                        # real interruption. Reset the streak and stay silent.
                        self._barge_in_streak_started = None
                        continue
                    if not text:
                        # Sustained and loud enough, but nothing decoded yet
                        # this streak — keep accumulating rather than firing
                        # on amplitude/VAD alone.
                        continue
                    self._barge_in_streak_started = None
                    if self._barge_in_rec is not None:
                        try:
                            self._barge_in_rec.Reset()
                        except Exception:
                            pass
                    if self.on_barge_in is not None:
                        self.loop.call_soon_threadsafe(self.on_barge_in)

                # Local recognition runs during the silence hangover so Vosk can
                # finalise utterances, but never during the post-speech echo
                # guard — that window belongs to ORION's own fading voice.
                if (
                    in_speech
                    and self._past_echo_guard(chunk, now)
                    and self.recogniser is not None
                    and self.on_transcript is not None
                ):
                    transcript = self.recogniser.feed(chunk)
                    if transcript:
                        self.loop.call_soon_threadsafe(self.on_transcript, transcript)
                if not self.can_capture():
                    continue
                # Continuous streaming (JARVIS-style): forward ALL audio — voice
                # AND silence — so the server-side VAD hears complete utterances
                # and can detect end-of-turn.  Filtering to voiced-only chunks is
                # what made the live channel deaf.
                if self._presence is not None and self._presence.utterance.muted:
                    # Judged to be somebody else while they were still talking.
                    # Dropping the rest means the turn never completes, so the
                    # model is not asked to answer a voice ORION was told to
                    # ignore.
                    continue
                media = {"data": chunk, "mime_type": "audio/pcm;rate=16000"}
                self.loop.call_soon_threadsafe(self._safe_put, media)
                if voiced:
                    self.loop.call_soon_threadsafe(self.bus.amplitude.emit, self._amplitude(chunk))
            except Exception as exc:
                try:
                    self.loop.call_soon_threadsafe(
                        self.bus.log.emit,
                        f"AUDIO: gatekeeper recovered - {str(exc).splitlines()[0][:120]}",
                    )
                except RuntimeError:
                    return  # event loop already closed during shutdown

    def _run_off_thread(self, work: Callable[[], Any]) -> None:
        """Run *work* somewhere that is not the capture thread.

        Hops to the event loop and then straight out to a worker, because the
        loop is the Qt thread here and a speaker embedding on it would stutter
        the face. Never raises on this thread: an exception in the capture loop
        costs ORION his hearing, and a tone reading is not worth that.
        """
        def guarded() -> None:
            # Swallowed HERE, on the worker, rather than left on the task.
            # Nothing awaits these, so an exception that reached the task
            # would sit there until garbage collection and then surface as
            # "Task exception was never retrieved" — a log full of noise from
            # a feature nobody is waiting on.
            try:
                work()
            except Exception as exc:
                try:
                    self.loop.call_soon_threadsafe(
                        self.bus.log.emit,
                        f"AUDIO: reading the speaker failed - "
                        f"{str(exc).splitlines()[0][:90]}")
                except Exception:
                    pass

        def dispatch() -> None:
            try:
                task = asyncio.ensure_future(asyncio.to_thread(guarded))
                # Held until it finishes: a task with no reference can be
                # collected mid-flight, and the callback also clears the
                # reference so this does not become a slow leak.
                self._presence_tasks.add(task)
                task.add_done_callback(self._presence_tasks.discard)
            except Exception:
                pass

        try:
            self.loop.call_soon_threadsafe(dispatch)
        except Exception:
            pass

    def _owner_gate_on(self) -> bool:
        """Whether ORION has been told to answer only his owner's voice."""
        try:
            from . import voiceprint

            return voiceprint.store().only_owner
        except Exception:
            return False

    def _speaker_judged(self) -> bool:
        """Whether any voice gate needs this utterance's speaker judged:
        answering only the owner, or guarding sensitive actions."""
        try:
            from . import voiceprint

            return voiceprint.store().judging
        except Exception:
            return False

    def _on_speaker_verdict(self, verdict: Any) -> None:
        """Act on who is talking. Called off the capture thread.

        Only a confident miss mutes anything. Every uncertain case — too
        little speech, no enrolment, a borderline score, a missing encoder —
        already resolves to listening inside voiceprint, and this must not
        add a second place where ORION can decide to go deaf.
        """
        try:
            if getattr(verdict, "is_owner", True) or self._presence is None:
                return
            # The action guard only needed the verdict recorded (voiceprint
            # did that); muting belongs to the answer-only-the-owner gate.
            if not self._owner_gate_on():
                return
            self._presence.utterance.muted = True
            self.drain()
            self.loop.call_soon_threadsafe(
                self.bus.log.emit,
                f"AUDIO: not your voice (similarity "
                f"{getattr(verdict, 'score', 0.0):.2f}) - ignoring this turn. "
                f"Say \"answer anyone\" to turn this off.")
        except Exception:
            pass

    def _safe_put(self, media: dict[str, Any]) -> None:
        try:
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.out_q.put_nowait(media)
        except asyncio.QueueFull:
            try:
                self.out_q.get_nowait()
                self.out_q.put_nowait(media)
            except Exception:
                pass

    def _amplitude(self, chunk: bytes) -> float:
        usable = len(chunk) - (len(chunk) % 2)
        if usable <= 0:
            return 0.0
        pcm = array("h")
        pcm.frombytes(chunk[:usable])
        if sys.byteorder != "little":
            pcm.byteswap()
        if not pcm:
            return 0.0
        sample_count = min(len(pcm), 256)
        stride       = max(1, len(pcm) // sample_count)
        total        = 0
        used         = 0
        for sample in pcm[::stride]:
            total += abs(int(sample))
            used  += 1
            if used >= sample_count:
                break
        if used <= 0:
            return 0.0
        return min(1.0, (total / used) / 32768.0)


# ──────────────────────────────────────────────────────────────────────────────
# MICROPHONE ENGINE
    def set_interruptible(self, on: bool) -> None:
        """Allow barge-in for the duration of a long-form utterance."""
        if on:
            self._interruptible.set()
        else:
            self._interruptible.clear()
            self._barge_in_streak_started = None


# ──────────────────────────────────────────────────────────────────────────────

class MicrophoneEngine:
    """Minimal PortAudio callback plus external VAD gatekeeper."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        out_q: asyncio.Queue,
        bus: OrionBus,
        can_capture: Callable[[], bool],
        vad: SileroVADGatekeeper,
        recogniser: "LocalSpeechRecogniser | None" = None,
        on_transcript: Callable[[str], None] | None = None,
        speaking_check: Callable[[], bool] | None = None,
        on_barge_in: Callable[[], None] | None = None,
        interrupt_feed: Callable[[bytes], str] | None = None,
        on_interrupt: Callable[[str], None] | None = None,
        on_input_silent: Callable[[], None] | None = None,
        on_input_restored: Callable[[], None] | None = None,
        speaker_tracker: Any | None = None,
        voice_presence: Any | None = None,
        is_own_echo: Callable[[str], bool] | None = None,
        command_listen_check: Callable[[], bool] | None = None,
    ) -> None:
        self.loop        = loop
        self.out_q       = out_q
        self.bus         = bus
        self.can_capture = can_capture
        self.vad         = vad
        self.recogniser     = recogniser
        self.on_transcript  = on_transcript
        self.speaking_check = speaking_check
        self.on_barge_in    = on_barge_in
        self.interrupt_feed = interrupt_feed
        self.on_interrupt   = on_interrupt
        self.command_listen_check = command_listen_check
        self.speaker_tracker = speaker_tracker
        self.voice_presence = voice_presence
        self.is_own_echo    = is_own_echo
        # Caller hooks for the dead-mic watchdog.  on_input_silent fires ONCE
        # when the input flatlines while listening; the caller (the live worker)
        # runs a presence check rather than this engine hopping devices on its
        # own.  Microphone rotation now happens only via an explicit
        # switch_to_next_input(explicit=True) call ("switch microphone").
        self._on_input_silent_cb   = on_input_silent
        self._on_input_restored_cb = on_input_restored
        self.enabled     = True
        self._stream: Any = None
        self._gatekeeper: AudioGateThread | None = None
        # A device override index chosen by an explicit rotation, and a counter
        # capped at one full cycle so repeated "switch mic" requests do not hop
        # forever when no microphone in the room is working.
        self._forced_input: int | None = None
        self._rotation_attempts = 0

    #: Remembered when set before the gate exists; applied in start().
    _interruptible_default = False

    def set_interruptible(self, on: bool) -> None:
        """Forwarded to the gate; safe before it exists, so callers need no
        knowledge of start-up order."""
        self._interruptible_default = bool(on)
        gate = getattr(self, "_gatekeeper", None)
        if gate is not None:
            gate.set_interruptible(on)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._gatekeeper = AudioGateThread(
            self.loop, self.out_q, self.bus, self._can_gate_capture, self.vad,
            recogniser=self.recogniser,
            on_transcript=self.on_transcript,
            speaking_check=self.speaking_check,
            on_barge_in=self.on_barge_in,
            interrupt_feed=self.interrupt_feed,
            on_interrupt=self.on_interrupt,
            on_input_silent=self._on_input_silent,
            on_input_restored=self._on_input_restored,
            speaker_tracker=self.speaker_tracker,
            voice_presence=self.voice_presence,
            is_own_echo=self.is_own_echo,
            command_listen_check=self.command_listen_check,
        )
        # Carry across anything set before the gate existed, so a caller never
        # has to know the start-up order. Without this the flag was stored and
        # silently never read.
        if getattr(self, "_interruptible_default", False):
            self._gatekeeper.set_interruptible(True)
        self._gatekeeper.start()
        from .audio_devices import device_name, resolve
        in_device = self._forced_input if self._forced_input is not None else resolve("input")
        # latency="high" gives PortAudio a deeper capture buffer so a busy
        # event loop (forge builds, model calls) cannot starve the callback
        # into "input overflow" — depth costs nothing audible on capture.
        # A saved/forced device that no longer opens (mic unplugged since last
        # run) must not leave ORION deaf for the whole session — fall back to
        # the system default rather than propagating out of start().
        try:
            self._stream = self._open_input_stream(in_device)
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: microphone [{in_device}] would not open - {exc}")
            self._forced_input = None
            try:
                self._stream = self._open_input_stream(None)
                self.bus.log.emit("AUDIO: started on the system default microphone instead.")
            except Exception as fallback_exc:
                self.bus.log.emit(
                    f"AUDIO: no microphone available at startup - {fallback_exc}. "
                    "Say 'switch microphone' once one is connected.")
                return
        self.bus.log.emit(
            f"AUDIO: microphone pipeline initialised — listening on {device_name('input')}."
        )

    # ── dead-microphone watchdog + explicit device rotation ──────────────────

    def _on_input_silent(self) -> None:
        """Gate-thread callback: the input has flatlined while listening.  Hand
        it to the caller (a presence check) — no automatic device hop."""
        cb = self._on_input_silent_cb
        if cb is None:
            return
        try:
            self.loop.call_soon_threadsafe(cb)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _on_input_restored(self) -> None:
        """The mic started hearing sound again — re-arm a full rotation cycle and
        let the caller clear any presence check."""
        self._rotation_attempts = 0
        cb = self._on_input_restored_cb
        if cb is None:
            return
        try:
            self.loop.call_soon_threadsafe(cb)
        except RuntimeError:
            pass

    def switch_to_next_input(self, explicit: bool = False) -> None:
        """Rotate capture to the next available input device (round-robin).  Now
        only called on the user's explicit say-so ('switch microphone'); pass
        ``explicit=True`` so the request always cycles and is confirmed aloud."""
        try:
            from .audio_devices import _devices, resolve
            devices = _devices()
            inputs = [i for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
            if len(inputs) < 2:
                self.bus.log.emit(
                    "AUDIO: only one microphone is present — nothing to switch to."
                )
                if explicit:
                    try:
                        self.bus.speak_request.emit(
                            "That's the only microphone I can find — please check "
                            "it's connected and unmuted."
                        )
                    except Exception:
                        pass
                if self._gatekeeper is not None:
                    self._gatekeeper.note_signal()
                return
            # An explicit request always resets the cycle guard so the user can
            # keep cycling through every microphone.
            if explicit:
                self._rotation_attempts = 0
            if self._rotation_attempts >= len(inputs):
                self.bus.log.emit(
                    "AUDIO: tried every microphone — staying on the current one."
                )
                if self._gatekeeper is not None:
                    self._gatekeeper.note_signal()
                return
            current = self._forced_input if self._forced_input is not None else resolve("input")
            if current is None:
                try:
                    default_in = sd.default.device[0]
                    current = default_in if isinstance(default_in, int) and default_in >= 0 else inputs[0]
                except Exception:
                    current = inputs[0]
            nxt = inputs[(inputs.index(current) + 1) % len(inputs)] if current in inputs else inputs[0]
            self._forced_input = nxt
            self._rotation_attempts += 1
            name = str(devices[nxt].get("name", "?"))
            self._restart_stream(nxt)
            # Persist ONLY an explicit user choice ("switch microphone") — the
            # default policy is to follow the system default device, and an
            # automatic hop must never silently pin ORION to one microphone.
            if explicit:
                try:
                    from . import audio_devices as _ad
                    _ad.set_device("input", str(nxt))
                except Exception:
                    pass
            self.bus.log.emit(f"AUDIO: microphone switched to [{nxt}] {name}.")
            if explicit:
                try:
                    self.bus.speak_request.emit(
                        f"Switching to {name}. Try me now."
                    )
                except Exception:
                    pass
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: microphone switch failed - {exc}")

    def switch_to_input(self, spec: Any, persist: bool = True) -> str | None:
        """Pin capture to a SPECIFIC input device chosen by the user (by index or
        name fragment), reopen the stream on it, and persist the choice.  Returns
        the device name, or None if the spec matches no input device.  This is
        the manual counterpart to switch_to_next_input's round-robin — the user
        keeps landing on the wrong mic, so let them pick the right one."""
        try:
            from . import audio_devices as ad
            spec_s = str(spec).strip()
            if spec_s.lower() in {"default", "reset", "auto", ""}:
                ad.set_device("input", "default")
                self._forced_input = None
                from .audio_devices import resolve
                target = resolve("input")
                self._restart_stream(target if target is not None else 0)
                self.bus.log.emit("AUDIO: microphone reset to the system default.")
                return "system default"
            index = ad._match(spec_s, "input")
            if index is None:
                self.bus.log.emit(f"AUDIO: no microphone matches '{spec}'.")
                return None
            self._forced_input = index
            self._rotation_attempts = 0
            name = str(ad._devices()[index].get("name", "?"))
            self._restart_stream(index)
            if persist:
                ad.set_device("input", str(index))
            self.bus.log.emit(f"AUDIO: microphone set to [{index}] {name} by the user.")
            return name
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: manual microphone selection failed - {exc}")
            return None

    def _open_input_stream(self, device_idx: Any) -> Any:
        """Open and start a capture stream on *device_idx*. Raises on failure —
        callers decide the recovery policy."""
        stream = sd.RawInputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CAPTURE_BLOCK_FRAMES,
            callback=self._callback,
            device=device_idx,
            latency="high",
        )
        stream.start()
        return stream

    def _restart_stream(self, device_idx: int) -> None:
        """Reopen just the input stream on a new device; the gate thread stays.

        ORION must never stop listening. If the requested device will not open
        (unplugged, held exclusively by another app, a driver hiccup mid-switch)
        this previously logged the failure and left ``self._stream`` at None —
        the microphone was then permanently dead for the rest of the session,
        with nothing anywhere retrying it. That is the "he just ignores me when
        I'm speaking" fault: capture silently gone with only one log line.
        Now a failed open always falls back to the system default, and a failed
        fallback restores the previous working stream, so capture survives.
        """
        previous = self._stream
        self._stream = None
        try:
            self._stream = self._open_input_stream(device_idx)
            from .audio_devices import note_live_device
            note_live_device("input", device_idx)
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: could not open microphone [{device_idx}] - {exc}")
            try:
                self._stream = self._open_input_stream(None)   # system default
                self._forced_input = None
                self.bus.log.emit(
                    "AUDIO: fell back to the system default microphone — still listening.")
            except Exception as fallback_exc:
                self.bus.log.emit(
                    f"AUDIO: default microphone also failed - {fallback_exc}")
                if previous is not None:
                    # Keep the old stream rather than ending up with no capture
                    # at all; it was working a moment ago.
                    self._stream = previous
                    previous = None
                    self.bus.log.emit("AUDIO: kept the previous microphone stream open.")
                else:
                    self.bus.log.emit(
                        "AUDIO: no microphone could be opened — say 'switch microphone' "
                        "or pick one in the audio settings.")
        if previous is not None:
            try:
                previous.stop()
                previous.close()
            except Exception:
                pass
        if self._gatekeeper is not None:
            self._gatekeeper.note_signal()   # give the new device a fresh window

    def stop(self) -> None:
        stream       = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                self.bus.log.emit(f"AUDIO: microphone close fault - {exc}")
        gatekeeper       = self._gatekeeper
        self._gatekeeper = None
        if gatekeeper is not None:
            gatekeeper.stop()
            gatekeeper.join(timeout=0.75)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.bus.mic_enabled.emit(self.enabled)
        if not self.enabled:
            self._drain()
            if self._gatekeeper is not None:
                self._gatekeeper.drain()

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            # Rate-limit: an overflow burst fires this every block; one line
            # per 30 s (with a suppressed-count) keeps the log readable.
            now = time.monotonic()
            self._status_suppressed = getattr(self, "_status_suppressed", 0)
            if now - getattr(self, "_status_logged_at", 0.0) >= 30.0:
                suffix = (f" ({self._status_suppressed} similar suppressed)"
                          if self._status_suppressed else "")
                self._status_logged_at = now
                self._status_suppressed = 0
                try:
                    self.loop.call_soon_threadsafe(
                        self.bus.log.emit, f"AUDIO: input status - {status}{suffix}")
                except RuntimeError:
                    return  # event loop already closed during shutdown
            else:
                self._status_suppressed += 1
        if not self.enabled:
            return
        chunk = bytes(indata)
        if not chunk:
            return
        gatekeeper = self._gatekeeper
        if gatekeeper is None:
            return
        # Re-frame to CHUNK_SIZE so the deeper capture block is invisible
        # downstream: the gate, the VAD and the live channel all keep seeing
        # exactly the 512-sample frames they saw before. Slicing bytes is a few
        # microseconds; being called four times as often is what was costing
        # the deadline.
        step = CHUNK_SIZE * 2                     # int16 mono
        if len(chunk) <= step:
            gatekeeper.enqueue(chunk)
            return
        for start in range(0, len(chunk), step):
            gatekeeper.enqueue(chunk[start:start + step])

    def _can_gate_capture(self) -> bool:
        return self.enabled and self.can_capture()

    def _drain(self) -> None:
        while True:
            try:
                self.out_q.get_nowait()
            except asyncio.QueueEmpty:
                break
