"""Explicit, local recordings of a room conversation on the existing mic.

Nothing starts at import or on restart. Every session needs consent=True.
The capture callback only puts bounded blocks on a queue; one worker writes
the WAV, another reads speech with the existing local transcriber and enrolled
voiceprints. Uncertain voice identity stays unknown. This is segment-level
identification, not separation of overlapping voices or automatic enrolment.
Distinct, clear unenrolled voices can receive tentative session-only numbers;
their embeddings are never saved and ambiguous matches stay unknown.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import threading
import time
import uuid
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR
from .data import ToolResult

SAMPLE_RATE = 16000
MAX_SECONDS = 2 * 60 * 60
MAX_SEGMENT_SECONDS = 15.0
MAX_SEGMENTS = 2400
SILENCE_SECONDS = 0.65
MIN_SPEECH_SECONDS = 0.25
ANONYMOUS_MATCH = 0.82
ANONYMOUS_DISTINCT = 0.60
ANONYMOUS_MARGIN = 0.08
MAX_ANONYMOUS_SPEAKERS = 12
_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


def _speaker(verdict: Any) -> tuple[str, float]:
    """Only an affirmative enrolled-person verdict earns a person's name."""
    kind = str(getattr(verdict, "category", "") or getattr(verdict, "kind", ""))
    name = str(getattr(verdict, "speaker", "") or "").strip()[:100]
    if kind not in {"owner", "known"} or not name or not getattr(verdict, "confident", False):
        return "Unknown speaker", 0.0
    try:
        score = float(getattr(verdict, "match", 0.0) or getattr(verdict, "score", 0.0))
    except (TypeError, ValueError):
        return "Unknown speaker", 0.0
    if not math.isfinite(score):
        return "Unknown speaker", 0.0
    return name, max(0.0, min(1.0, score))


@dataclass
class _Session:
    path: Path
    meta: dict[str, Any]
    started: float
    wav: Any
    stream: Any
    queue: Queue
    analysis: Queue
    accepting: bool = True
    stop: threading.Event = field(default_factory=threading.Event)
    written: threading.Event = field(default_factory=threading.Event)
    analysed: threading.Event = field(default_factory=threading.Event)
    feed_lock: Any = field(default_factory=threading.Lock)
    meta_lock: Any = field(default_factory=threading.RLock)
    persist_lock: Any = field(default_factory=threading.Lock)
    writer: Any = None
    reader: Any = None
    dropped_frames: int = 0
    dropped_blocks: int = 0
    anonymous: list[Any] = field(default_factory=list)
    stopped: float | None = None


class ConversationRecorder:
    """One explicitly consented session at a time; no microphone ownership."""

    def __init__(self, root: str | Path | None = None, *, sample_rate: int = SAMPLE_RATE,
                 max_seconds: float = MAX_SECONDS, queue_limit: int = 256,
                 analysis_limit: int = 4, transcribe: Callable | None = None,
                 identify: Callable | None = None, embed: Callable | None = None,
                 clock: Callable = time.monotonic) -> None:
        self.root = Path(root) if root is not None else CONFIG_DIR / "conversation_recordings"
        self.sample_rate = int(sample_rate)
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError("conversation recording uses the existing 16 kHz microphone")
        self.max_seconds = max(0.1, min(float(max_seconds), MAX_SECONDS))
        self.queue_limit = max(1, min(int(queue_limit), 512))
        self.analysis_limit = max(1, min(int(analysis_limit), 8))
        self._transcribe = transcribe
        self._identify = identify
        self._embed = embed
        self._clock = clock
        self._session: _Session | None = None
        self._control = threading.RLock()
        self._capture_check: Callable[[], bool] | None = None
        self._closed = False
        self._privacy_windows: tuple[tuple[float, float], ...] = ()

    def set_capture_check(self, check: Callable[[], bool] | None) -> None:
        """App supplies its explicit mic/pause/standby/shutdown eligibility.

        This must be cheap: it runs on the existing capture callback. It must
        not include the wake-word or owner-only *interaction* gates, since a
        room recording includes speech not addressed to ORION.
        """
        self._capture_check = check

    @property
    def active(self) -> bool:
        session = self._session
        return bool(session and session.accepting and not session.stop.is_set()
                    and self._clock() - session.started < self.max_seconds)

    @property
    def record_only(self) -> bool:
        session = self._session
        return bool(self.active and session and session.meta["record_only"])

    def record_only_at(self, captured_at: float | None = None, *, duration: float = 0.0) -> bool:
        """Sticky privacy policy for delayed capture blocks, including stop.

        Device blocks have ADC timestamps and can reach a queue after the
        recording stops. A room block captured in record-only mode remains
        local even if the current interaction mode has since changed.
        """
        if captured_at is None:
            return self.record_only
        try:
            at = float(captured_at)
        except (TypeError, ValueError):
            return self.record_only
        if not math.isfinite(at):
            return self.record_only
        begin_at = at - max(0.0, float(duration))
        session = self._session
        if session and session.meta["record_only"]:
            end = session.stopped if session.stopped is not None else session.started + self.max_seconds
            if at >= session.started and begin_at <= end:
                return True
        return any(at >= begin and begin_at <= end for begin, end in self._privacy_windows)

    def snapshot(self) -> dict[str, Any]:
        """A cheap, memory-only snapshot for the visible recording control."""
        session = self._session
        if session is None:
            return {"active": False, "record_only": False, "state": "off"}
        with session.meta_lock:
            data = copy.deepcopy(session.meta)
        data.update(active=self.active, record_only=self.record_only,
                    dropped_blocks=session.dropped_blocks,
                    dropped_seconds=session.dropped_frames / self.sample_rate,
                    waiting_for_audio=data["frames"] == 0)
        return data

    def start(self, *, consent: bool = False, record_only: bool = True) -> ToolResult:
        if consent is not True:
            return ToolResult("Recording stays off. Start needs consent=true after everyone "
                              "taking part has agreed to a local audio recording and transcript.", ok=False)
        if not isinstance(record_only, bool):
            return ToolResult("record_only must be true or false.", ok=False)
        with self._control:
            if self._closed:
                return ToolResult("Recording is unavailable because ORION is shutting down.", ok=False)
            previous = self._session
            if previous and (self.active or not previous.analysed.is_set()):
                return ToolResult("A recording is already running or still finishing. "
                                  "Stop it and wait for it to finish first.", ok=False)
            if previous and previous.meta["record_only"] and previous.stopped is not None:
                self._privacy_windows = (*self._privacy_windows[-127:], (previous.started, previous.stopped))
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            session_id = stamp + "-" + uuid.uuid4().hex[:12]
            path = self.root / session_id
            stream = wav = None
            try:
                path.mkdir(parents=True, exist_ok=False)
                stream = (path / "recording.wav").open("xb")
                wav = wave.open(stream, "wb")
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(b"")
                stream.flush()
                os.fsync(stream.fileno())
                meta = {"version": 1, "id": session_id, "state": "recording",
                        "started_at": time.time(), "ended_at": None,
                        "consent": True, "record_only": record_only,
                        "sample_rate": self.sample_rate, "channels": 1,
                        "audio": "recording.wav", "transcript": "transcript.jsonl",
                        "frames": 0, "audio_seconds": 0.0, "dropped_blocks": 0,
                        "dropped_seconds": 0.0, "analysis_skipped": 0,
                        "segments": 0, "transcribed_segments": 0,
                        "anonymous_speakers": 0,
                        "transcription_status": "waiting for speech",
                        "speaker_policy": "Only confident enrolled matches are named. Clear distinct "
                        "unenrolled voices may get tentative session-only speaker numbers; uncertain "
                        "matches stay unknown. Each label estimates a whole speech segment; fast "
                        "turn-taking or overlapping voices may be mixed and are not separated.",
                        "capture_gaps": [], "errors": [], "stop_reason": ""}
                session = _Session(path, meta, self._clock(), wav, stream,
                                   Queue(self.queue_limit), Queue(self.analysis_limit))
                self._checkpoint(session)
                self._session = session
                session.reader = threading.Thread(target=self._analyse, args=(session,),
                                                  name="orion-recording-transcript", daemon=True)
                session.writer = threading.Thread(target=self._write, args=(session,),
                                                  name="orion-conversation-recorder", daemon=True)
                session.reader.start()
                session.writer.start()
            except Exception as exc:
                failed = self._session if self._session and self._session.path == path else None
                if failed:
                    failed.accepting = False
                    failed.stopped = self._clock()
                    failed.stop.set()
                    self._error(failed, "recording startup", exc)
                    with failed.meta_lock:
                        failed.meta["state"] = "failed"
                        failed.meta["ended_at"] = time.time()
                try:
                    if wav is not None:
                        wav.close()
                except Exception:
                    pass
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
                if failed:
                    failed.written.set()
                    if failed.reader is None or failed.reader.ident is None:
                        try:
                            self._checkpoint(failed)
                        except Exception:
                            pass
                        failed.analysed.set()
                return ToolResult(f"Recording could not start ({type(exc).__name__}). "
                                  "Any partial files have been kept.", ok=False)
        mode = "ORION will listen without replying to the room" if record_only else "ORION can still answer"
        return ToolResult(f"Recording session {session_id} is armed on the existing microphone; "
                          f"{mode}. Audio and any available local transcript stay in {path}. "
                          "It is waiting for microphone input. Stop from the recording control "
                          "or type a stop-recording request. It will stop after two hours and "
                          "will never restart by itself.", receipt=self.snapshot())

    def feed(self, pcm: bytes, captured_at: float | None = None) -> bool:
        """Nonblocking bounded enqueue; no I/O, VAD, inference or Qt work."""
        session = self._session
        if session is None or not session.accepting:
            return False
        check = self._capture_check
        try:
            if check is not None and not check():
                return False
            at = self._clock() if captured_at is None else float(captured_at)
            if not math.isfinite(at):
                return False
        except Exception:
            return False
        usable = len(pcm) - len(pcm) % 2
        if not usable:
            return False
        frames = usable // 2
        if not session.feed_lock.acquire(blocking=False):
            session.dropped_blocks += 1
            session.dropped_frames += frames
            return False
        try:
            if not session.accepting:
                return False
            if self._clock() - session.started >= self.max_seconds:
                session.accepting = False
                session.stopped = session.started + self.max_seconds
                session.meta["stop_reason"] = "duration limit"
                session.stop.set()
                return False
            # Never retain pre-consent bytes from a delayed capture block.
            # Work directly in sample offsets. Two timestamp subtractions can
            # turn an exact boundary (4800 frames) into 4800.00000000009 and
            # incorrectly discard the next sample. Clock values themselves
            # have finite precision: allow only their combined ULP uncertainty.
            offset = (session.started - at) * self.sample_rate + frames
            precision = (math.ulp(session.started) + math.ulp(at)) * self.sample_rate
            trim = max(0, int(math.ceil(offset - precision)))
            usable_pcm = pcm[min(frames, trim) * 2:usable]
            if not usable_pcm:
                return False
            if len(usable_pcm) > self.sample_rate * 2:
                session.dropped_blocks += 1
                session.dropped_frames += len(usable_pcm) // 2
                return False
            try:
                session.queue.put_nowait((bytes(usable_pcm), at))
                return True
            except Full:
                session.dropped_blocks += 1
                session.dropped_frames += len(usable_pcm) // 2
                return False
        finally:
            session.feed_lock.release()

    def note_transcript(self, text: str, verdict: Any = None) -> bool:
        """Optional already-local final transcript; never use partial words."""
        session = self._session
        text = str(text or "").strip()[:2000]
        if session is None or not self.active or not text:
            return False
        if not session.feed_lock.acquire(blocking=False):
            return False
        try:
            with session.meta_lock:
                offset = session.meta["audio_seconds"]
            if not session.accepting:
                return False
            name, score = _speaker(verdict)
            item = {"start": offset, "end": offset, "speaker": name,
                    "speaker_confidence": score, "text": text, "source": "local listener"}
            try:
                session.analysis.put_nowait(("transcript", item))
                return True
            except Full:
                with session.meta_lock:
                    session.meta["analysis_skipped"] += 1
                return False
        finally:
            session.feed_lock.release()

    def stop(self, timeout: float = 3.0) -> ToolResult:
        with self._control:
            session = self._session
            if session is None:
                return ToolResult("No conversation recording is running.", receipt=self.snapshot())
            with session.feed_lock:
                session.accepting = False
                if session.stopped is None:
                    session.stopped = self._clock()
                session.stop.set()
            deadline = time.monotonic() + max(0.0, float(timeout))
            for thread in (session.writer, session.reader):
                if (thread is not None and thread.ident is not None
                        and thread is not threading.current_thread()):
                    thread.join(max(0.0, deadline - time.monotonic()))
            data = self.snapshot()
            finishing = not session.analysed.is_set()
            text = ("Capture has stopped; the saved audio and queued local transcript are "
                    "still finishing." if finishing else
                    f"Recording {data['id']} stopped. Saved {data['audio_seconds']:.1f} seconds "
                    f"of microphone audio and {data['transcribed_segments']} transcript segments.")
            if data["dropped_blocks"] or data["analysis_skipped"] or data["errors"]:
                text += (f" Partial evidence: {data['dropped_blocks']} audio blocks dropped, "
                         f"{data['analysis_skipped']} analysis segments skipped, "
                         f"{len(data['errors'])} errors; details are in the manifest.")
            if data["transcription_status"] == "unavailable":
                text += " No local transcription engine was available; the audio is retained."
            return ToolResult(text, receipt=data)

    def close(self, timeout: float = 3.0) -> None:
        with self._control:
            self._closed = True
            self.stop(timeout=timeout)

    def _checkpoint(self, session: _Session) -> None:
        with session.persist_lock:
            with session.meta_lock:
                session.meta["dropped_blocks"] = session.dropped_blocks
                session.meta["dropped_seconds"] = session.dropped_frames / self.sample_rate
                payload = json.dumps(session.meta, ensure_ascii=False, indent=2)
            atomic_write_text(session.path / "manifest.json", payload, encoding="utf-8")

    @staticmethod
    def _error(session: _Session, area: str, exc: BaseException) -> None:
        with session.meta_lock:
            if len(session.meta["errors"]) < 30:
                session.meta["errors"].append(f"{area}: {type(exc).__name__}")

    def _write(self, session: _Session) -> None:
        pieces: list[bytes] = []
        segment_start = segment_capture_start = 0.0
        segment_frames = speech_frames = silent_frames = 0
        last_at: float | None = None
        checkpoint = time.monotonic()

        def finish_segment() -> None:
            nonlocal pieces, segment_frames, speech_frames, silent_frames
            if speech_frames / self.sample_rate >= MIN_SPEECH_SECONDS:
                item = {"pcm": b"".join(pieces), "start": segment_start,
                        "end": segment_start + segment_frames / self.sample_rate,
                        "capture_start": segment_capture_start,
                        "capture_end": max(0.0, (last_at or session.started) - session.started)}
                try:
                    session.analysis.put_nowait(("audio", item))
                except Full:
                    with session.meta_lock:
                        session.meta["analysis_skipped"] += 1
            pieces = []
            segment_frames = speech_frames = silent_frames = 0

        try:
            while not session.stop.is_set() or not session.queue.empty():
                if self._clock() - session.started >= self.max_seconds:
                    with session.feed_lock:
                        session.accepting = False
                        if session.stopped is None:
                            session.stopped = session.started + self.max_seconds
                        session.stop.set()
                    with session.meta_lock:
                        session.meta["stop_reason"] = "duration limit"
                try:
                    block, at = session.queue.get(timeout=0.05)
                except Empty:
                    continue
                with session.meta_lock:
                    start = session.meta["frames"] / self.sample_rate
                    remaining = max(0, int(self.max_seconds * self.sample_rate) - session.meta["frames"])
                if len(block) // 2 > remaining:
                    session.dropped_frames += len(block) // 2 - remaining
                    session.dropped_blocks += 1
                    block = block[:remaining * 2]
                if not block:
                    continue
                count = len(block) // 2
                capture_start = at - count / self.sample_rate
                if last_at is not None and capture_start - last_at > 0.08:
                    finish_segment()
                    with session.meta_lock:
                        gaps = session.meta["capture_gaps"]
                        if len(gaps) < 100:
                            gaps.append({"after_audio_second": start,
                                         "missing_seconds": round(capture_start - last_at, 3)})
                session.wav.writeframes(block)
                with session.meta_lock:
                    session.meta["frames"] += count
                    session.meta["audio_seconds"] = session.meta["frames"] / self.sample_rate
                samples = array("h")
                samples.frombytes(block)
                voiced = sum(abs(value) for value in samples) / max(1, len(samples)) > 260
                if voiced or pieces:
                    if not pieces:
                        segment_start = start
                        segment_capture_start = max(0.0, capture_start - session.started)
                    pieces.append(block)
                    segment_frames += count
                    speech_frames += count if voiced else 0
                    silent_frames = 0 if voiced else silent_frames + count
                last_at = at
                if (silent_frames / self.sample_rate >= SILENCE_SECONDS
                        or segment_frames / self.sample_rate >= MAX_SEGMENT_SECONDS):
                    finish_segment()
                if time.monotonic() - checkpoint >= 1.0:
                    session.stream.flush()
                    os.fsync(session.stream.fileno())
                    self._checkpoint(session)
                    checkpoint = time.monotonic()
            finish_segment()
        except Exception as exc:
            self._error(session, "audio writer", exc)
            with session.feed_lock:
                session.accepting = False
                if session.stopped is None:
                    session.stopped = self._clock()
                session.stop.set()
            while True:
                try:
                    block, _ = session.queue.get_nowait()
                    session.dropped_blocks += 1
                    session.dropped_frames += len(block) // 2
                except Empty:
                    break
        finally:
            try:
                session.wav.close()
                session.stream.flush()
                os.fsync(session.stream.fileno())
            except Exception as exc:
                self._error(session, "audio finalization", exc)
            finally:
                try:
                    session.stream.close()
                except Exception as exc:
                    self._error(session, "audio stream close", exc)
            with session.meta_lock:
                session.meta["state"] = "finalizing"
                session.meta["ended_at"] = time.time()
            session.written.set()

    def _read_audio(self, pcm: bytes) -> tuple[str, str, Any, Any]:
        vector = None
        try:
            if self._identify is not None:
                verdict = self._identify(pcm, self.sample_rate)
                if self._embed is not None:
                    vector = self._embed(pcm, self.sample_rate)
            else:
                import numpy as np
                from . import voiceprint
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                vector = (self._embed(pcm, self.sample_rate) if self._embed is not None
                          else voiceprint.embed(samples, self.sample_rate))
                # Read enrolled profiles without changing the reply/action
                # guard. Reuse this one embedding for anonymous turn labels.
                verdict = voiceprint.store().verdict_for(vector)
                # A long segment may contain rapid turn-taking. When there
                # is enough speech for two independent readings, a changed
                # voice at the ends must not inherit one confident name.
                speech = voiceprint._trim_silence(samples)
                window = int(1.2 * self.sample_rate)
                if vector is not None and self._embed is None and speech.size >= 2 * window:
                    first = voiceprint.embed(speech[:window], self.sample_rate)
                    last = voiceprint.embed(speech[-window:], self.sample_rate)
                    consistent = (first is not None and last is not None
                                  and voiceprint.similarity(first, last) >= 0.70)
                    name, _ = _speaker(verdict)
                    if consistent and name != "Unknown speaker":
                        prints = voiceprint.store()
                        consistent = (_speaker(prints.verdict_for(first))[0] == name
                                      and _speaker(prints.verdict_for(last))[0] == name)
                    if not consistent:
                        verdict, vector = None, None
        except Exception:
            verdict = None              # identity failure must not lose words
        if self._transcribe is not None:
            return str(self._transcribe(pcm, self.sample_rate) or "").strip()[:2000], "local", verdict, vector
        from .speech_offline import shared
        transcriber = shared()
        if not transcriber.available:
            return "", "unavailable", verdict, vector
        # Room speech must not be discarded by the owner's interaction gate,
        # nor may observing it change that gate or voice-enrolment samples.
        text = transcriber.transcribe_pcm(pcm, self.sample_rate, identity_checked=True)
        return str(text or "").strip()[:2000], "local", verdict, vector

    @staticmethod
    def _anonymous_speaker(session: _Session, vector: Any) -> tuple[str, float]:
        """Conservative, tentative within-session labels, never saved prints.

        Keep the first clear vector as each anchor so repeated weak matches
        cannot drift a cluster into somebody else's voice. Borderline or
        competing matches remain unknown instead of inventing a new person.
        """
        from . import voiceprint
        vector = voiceprint._unit(vector)
        if vector is None:
            return "Unknown speaker", 0.0
        scores = sorted(((voiceprint.similarity(vector, anchor), i)
                         for i, anchor in enumerate(session.anonymous)), reverse=True)
        if scores:
            score, index = scores[0]
            competing = len(scores) > 1 and score - scores[1][0] < ANONYMOUS_MARGIN
            if score >= ANONYMOUS_MATCH and not competing:
                return f"Speaker {index + 1}", score
            if score >= ANONYMOUS_DISTINCT:
                return "Unknown speaker", 0.0
        if len(session.anonymous) >= MAX_ANONYMOUS_SPEAKERS:
            return "Unknown speaker", 0.0
        session.anonymous.append(vector)
        return f"Speaker {len(session.anonymous)}", 0.0

    def _analyse(self, session: _Session) -> None:
        try:
            while not session.written.is_set() or not session.analysis.empty():
                try:
                    kind, item = session.analysis.get(timeout=0.05)
                except Empty:
                    continue
                with session.meta_lock:
                    if session.meta["segments"] >= MAX_SEGMENTS:
                        session.meta["analysis_skipped"] += 1
                        continue
                if kind == "audio":
                    pcm = item.pop("pcm")
                    try:
                        text, status, verdict, vector = self._read_audio(pcm)
                    except Exception as exc:
                        self._error(session, "speech analysis", exc)
                        text, status, verdict, vector = "", "failed", None, None
                    name, score = _speaker(verdict)
                    source = "enrolled voiceprint" if name != "Unknown speaker" else "unidentified"
                    category = str(getattr(verdict, "category", "") or getattr(verdict, "kind", ""))
                    if (name == "Unknown speaker" and category in {"unknown", "nobody"}
                            and len(pcm) / (self.sample_rate * 2) >= 1.2):
                        name, score = self._anonymous_speaker(session, vector)
                        if name != "Unknown speaker":
                            source = "tentative session voice cluster"
                    with session.meta_lock:
                        session.meta["anonymous_speakers"] = len(session.anonymous)
                    item.update(text=text, speaker=name, speaker_confidence=score,
                                source="local audio", speaker_source=source,
                                speaker_scope="whole segment estimate; may contain multiple voices",
                                transcription_status=status)
                else:
                    status = "local"
                with (session.path / "transcript.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                with session.meta_lock:
                    session.meta["segments"] += 1
                    session.meta["transcribed_segments"] += int(bool(item["text"]))
                    session.meta["transcription_status"] = status
                self._checkpoint(session)
        except Exception as exc:
            self._error(session, "transcript writer", exc)
        finally:
            # Never publish 'stopped' before the WAV has closed. A transcript
            # failure must not abandon a still-running audio writer.
            session.written.wait()
            session.anonymous.clear()
            with session.meta_lock:
                session.meta["state"] = ("partial" if session.meta["errors"]
                                         or session.dropped_blocks
                                         or session.meta["analysis_skipped"] else "stopped")
            try:
                self._checkpoint(session)
            except Exception as exc:
                self._error(session, "manifest finalization", exc)
            session.analysed.set()

    def _path(self, session_id: str) -> Path:
        if not _ID.fullmatch(session_id):
            raise ValueError("Choose a recording's session ID from list.")
        root = self.root.resolve()
        path = (root / session_id).resolve()
        if path.parent != root:
            raise ValueError("Recording path leaves its local store.")
        return path

    def _manifest(self, path: Path) -> dict[str, Any]:
        manifest = path / "manifest.json"
        if manifest.stat().st_size > 1_000_000:
            raise ValueError("Recording manifest is too large.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("id") != path.name:
            raise ValueError("Recording manifest identity is invalid.")
        current = self._session
        if data.get("state") in {"recording", "finalizing"} and not (
                current and current.path.resolve() == path and not current.analysed.is_set()):
            data["state"] = "interrupted"
        return data

    def list_sessions(self) -> ToolResult:
        rows = []
        if self.root.is_dir():
            for path in sorted(self.root.iterdir(), reverse=True)[:200]:
                try:
                    candidate = self._path(path.name)
                    data = self._manifest(candidate)
                except (OSError, ValueError):
                    continue
                rows.append(data)
        text = "\n".join(f"{row['id']} — {row.get('state', 'unknown')}, "
                         f"{float(row.get('audio_seconds', 0)):.1f}s, "
                         f"{int(row.get('transcribed_segments', 0))} transcript segments"
                         for row in rows[:30])
        return ToolResult(text or "No conversation recordings are saved.",
                          receipt={"recordings": rows[:30]})

    def recall(self, session_id: str = "", query: str = "") -> ToolResult:
        if not session_id:
            listed = self.list_sessions().receipt or {}
            rows = listed.get("recordings", [])
            if not rows:
                return ToolResult("No conversation recordings are saved.")
            session_id = rows[0]["id"]
        try:
            path = self._path(session_id)
            meta = self._manifest(path)
            transcript = path / "transcript.jsonl"
            lines = []
            if transcript.is_file():
                if transcript.stat().st_size > 8_000_000:
                    raise ValueError("Transcript is too large to read safely.")
                for line in transcript.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue              # a crash can leave an incomplete last line
                    if isinstance(item, dict) and item.get("text") and (
                            not query or query.casefold() in str(item["text"]).casefold()):
                        lines.append(item)
            shown = lines[-20:]
            text = "\n".join(f"[{float(item.get('start', 0)):.1f}s] "
                             f"{item.get('speaker', 'Unknown speaker')}: {item['text']}" for item in shown)
            if not text:
                text = ("No matching transcript text. " if query else "No transcript text is available. ")
                text += "The WAV retains the captured audio."
            text = f"Recording {session_id} ({meta['state']}).\n{text}\nAudio: {path / 'recording.wav'}"
            return ToolResult(text, receipt={"recording": meta, "segments": shown})
        except (OSError, ValueError) as exc:
            return ToolResult(f"Recording could not be read: {exc}", ok=False)

    def forget(self, session_id: str) -> ToolResult:
        try:
            path = self._path(session_id)
            self._manifest(path)
            current = self._session
            if current and current.path.resolve() == path and not current.analysed.is_set():
                return ToolResult("Stop the recording and let it finish before forgetting it.", ok=False)
            from . import recycle_bin
            reason = recycle_bin.refusal(path)
            if reason:
                return ToolResult(f"Recording kept: it cannot be safely recycled ({reason}).", ok=False)
            if not recycle_bin.send_to_recycle_bin(path):
                return ToolResult("Recording was not confirmed in the Recycle Bin; inspect its "
                                  "location and the bin before retrying.", ok=False)
            return ToolResult(f"Recording {session_id}, including audio and transcript, "
                              "was moved to the Recycle Bin.")
        except (OSError, ValueError) as exc:
            return ToolResult(f"Recording kept: {exc}", ok=False)

    def status(self) -> ToolResult:
        data = self.snapshot()
        if data["state"] == "off":
            return ToolResult("Conversation recording is off. It never starts automatically.", receipt=data)
        if data["active"]:
            text = (f"Recording {data['id']} is active; {data['audio_seconds']:.1f} seconds saved. "
                    + ("Waiting for microphone input. " if data["waiting_for_audio"] else "")
                    + ("ORION is listening without replying to the room. " if data["record_only"] else ""))
        else:
            text = f"Conversation recording is off; last session {data['id']} is {data['state']}. "
        text += (f"Dropped audio: {data['dropped_blocks']} blocks. "
                 f"Skipped analysis: {data['analysis_skipped']}. Local transcript: "
                 f"{data['transcription_status']}. {data['speaker_policy']}")
        return ToolResult(text, receipt=data)


_RECORDER: ConversationRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def recorder() -> ConversationRecorder:
    global _RECORDER
    if _RECORDER is None:
        with _RECORDER_LOCK:
            if _RECORDER is None:
                _RECORDER = ConversationRecorder()
    return _RECORDER


async def handle(action: str = "status", consent: bool = False, session_id: str = "",
                 query: str = "", record_only: bool = True) -> ToolResult:
    """Dispatcher adapter: filesystem and bounded joins stay off the GUI loop."""
    service = recorder()
    action = str(action or "status").strip().lower()
    if action == "start":
        return await asyncio.to_thread(service.start, consent=consent, record_only=record_only)
    if action == "stop":
        return await asyncio.to_thread(service.stop)
    if action == "status":
        return service.status()
    if action == "list":
        return await asyncio.to_thread(service.list_sessions)
    if action in {"recall", "transcript"}:
        return await asyncio.to_thread(service.recall, session_id, query)
    if action in {"forget", "delete"}:
        return await asyncio.to_thread(service.forget, session_id)
    return ToolResult("Use start, stop, status, list, recall or forget for a conversation recording.", ok=False)


__all__ = ["ConversationRecorder", "handle", "recorder"]
