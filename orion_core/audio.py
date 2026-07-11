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
from threading import Event, RLock, Thread
from typing import Any, Callable

import sounddevice as sd

from .audio_state import AudioStateMachine, SpeechState
from .bus import OrionBus
from .constants import (
    ALLOW_BARGE_IN,
    BARGE_IN_CONFIDENCE,
    CHANNELS,
    CHUNK_SIZE,
    MIC_QUEUE_LIMIT,
    PLAYBACK_PREBUFFER_CHUNKS,
    PLAYBACK_PREBUFFER_MAX_WAIT,
    PLAYBACK_QUEUE_HIGH_WATER,
    PLAYBACK_TAIL_SECONDS,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
    VAD_SAMPLE_LIMIT,
    VOICE_HANGOVER_SECONDS,
    VOICE_PROFILE,
    WAKE_WORDS,
)


# ──────────────────────────────────────────────────────────────────────────────
# NATIVE AUDIO PLAYBACK  (Gemini Live PCM renderer)
# ──────────────────────────────────────────────────────────────────────────────

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
        # True voice interruption (Mark X.5): while held, the renderer stops
        # writing but PRESERVES the queue and the unwritten remainder of the
        # current buffer, so resume continues from the exact interruption point
        # with no context loss and no response regeneration.
        self._hold           = Event()
        self._held_tail: bytes = b""

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

    def run(self) -> None:
        try:
            # blocksize=0 lets PortAudio pick its optimal buffer; combined with
            # coalesced multi-chunk writes this eliminates the per-512-frame
            # underrun that caused stuttering.
            from .audio_devices import device_name, resolve
            out_device = resolve("output")
            self._stream = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=0,
                device=out_device,
            )
            self._stream.start()
            self.bus.log.emit(
                f"AUDIO: output renderer initialised — voice → {device_name('output')}."
            )
            while not self.stop_event.is_set():
                # ── held (true interruption): stay silent, preserve everything ─
                if self._hold.is_set():
                    if self._active:
                        self._mark_inactive()
                    time.sleep(0.04)
                    continue
                # ── just resumed: flush the preserved remainder first ─────────
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
        """Sliced device write; a hold between slices parks the remainder."""
        if not buffer:
            return
        view = memoryview(buffer)
        offset = 0
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
            start = time.perf_counter()
            self.bus.amplitude.emit(self._amplitude(bytes(piece)))
            self._stream.write(piece)
            self.last_audio_time = time.monotonic()
            if self.telemetry is not None:
                self.telemetry.metrics.observe(
                    "audio.playback.write_latency_ms", (time.perf_counter() - start) * 1000.0
                )
                self.telemetry.metrics.gauge(
                    "audio.playback.queue_depth", float(self.queue.qsize())
                )
            offset = slice_end

    def _mark_inactive(self) -> None:
        if self._active:
            self._active = False
            self.bus.amplitude.emit(0.0)
            if self.machine is not None:
                self.machine.native_stopped()

    def stop(self) -> None:
        self.stop_event.set()
        self._hold.clear()
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
                if self._engine is not None:
                    self._speak_pyttsx3(text)
                else:
                    self._speak_powershell(text)
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
        """
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

    def stop(self) -> None:
        self.playback.stop()
        self.tts.stop()

    # ── speech submission ─────────────────────────────────────────────────────

    def enqueue_native_audio(self, chunk: bytes) -> None:
        """Stream a native PCM chunk (Gemini Live) into ordered playback."""
        self.playback.enqueue(chunk)

    def speak_text(self, text: str) -> None:
        """Queue a local-voice utterance; spoken after everything already queued."""
        self.tts.speak(text)

    # ── state ─────────────────────────────────────────────────────────────────

    def output_active(self) -> bool:
        """True while either source is active — the machine's atomic flag."""
        return self.machine.is_active()

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


# ──────────────────────────────────────────────────────────────────────────────
# VOICE ACTIVITY DETECTION
# ──────────────────────────────────────────────────────────────────────────────

class SileroVADGatekeeper:
    """Local voice-activity gate with optional Silero inference and deterministic fallback."""

    def __init__(self, bus: OrionBus, threshold: float = 0.65) -> None:
        self.bus           = bus
        self.threshold     = max(0.05, min(0.95, float(threshold)))
        self._model: Any   = None
        self._torch: Any   = None
        self._fallback_floor = 0.012
        self._last_voice   = 0.0
        self._initialise_silero()

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
    ) -> None:
        super().__init__(name="orion-audio-gatekeeper", daemon=True)
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
        self.raw_q: Queue[bytes | None] = Queue(maxsize=max(8, int(raw_limit or MIC_QUEUE_LIMIT)))
        self.stop_event   = Event()
        self._voice_until = 0.0
        self._was_speaking = False
        # Echo guard: after ORION stops speaking, ignore captured audio for a
        # short window so the acoustic tail / room echo of his OWN voice is
        # never transcribed and answered again (the "repeats himself" bug).
        self._deaf_until = 0.0

    # Deafness window after speech ends (seconds).
    _ECHO_GUARD_SECONDS = 0.9

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
            try:
                chunk = self.raw_q.get(timeout=0.05)
            except Empty:
                continue
            if chunk is None:
                break
            try:
                confidence = self.vad.confidence(chunk)
                now    = time.monotonic()
                voiced = confidence > self.vad.threshold
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

                if speaking:
                    # True interruption listener (Mark X.5): even while ORION
                    # speaks, command phrases like "ORION stop" are matched by
                    # the grammar-constrained listener and honoured instantly.
                    if self.interrupt_feed is not None:
                        try:
                            phrase = self.interrupt_feed(chunk)
                        except Exception:
                            phrase = ""
                        if phrase and self.on_interrupt is not None:
                            self.loop.call_soon_threadsafe(self.on_interrupt, phrase)
                    if not ALLOW_BARGE_IN:
                        # Speak-then-listen: swallow the chunk entirely.
                        continue
                    # Legacy barge-in path: require high-confidence voice.
                    if not voiced or confidence < BARGE_IN_CONFIDENCE:
                        continue
                    if self.on_barge_in is not None:
                        self.loop.call_soon_threadsafe(self.on_barge_in)

                # Local recognition runs during the silence hangover so Vosk can
                # finalise utterances, but never during the post-speech echo
                # guard — that window belongs to ORION's own fading voice.
                if (
                    in_speech
                    and now >= self._deaf_until
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
        self.enabled     = True
        self._stream: Any = None
        self._gatekeeper: AudioGateThread | None = None

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
        )
        self._gatekeeper.start()
        from .audio_devices import device_name, resolve
        in_device = resolve("input")
        self._stream = sd.RawInputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=self._callback,
            device=in_device,
        )
        self._stream.start()
        self.bus.log.emit(
            f"AUDIO: microphone pipeline initialised — listening on {device_name('input')}."
        )

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
            try:
                self.loop.call_soon_threadsafe(self.bus.log.emit, f"AUDIO: input status - {status}")
            except RuntimeError:
                return  # event loop already closed during shutdown
        if not self.enabled:
            return
        chunk = bytes(indata)
        if not chunk:
            return
        gatekeeper = self._gatekeeper
        if gatekeeper is not None:
            gatekeeper.enqueue(chunk)

    def _can_gate_capture(self) -> bool:
        return self.enabled and self.can_capture()

    def _drain(self) -> None:
        while True:
            try:
                self.out_q.get_nowait()
            except asyncio.QueueEmpty:
                break
