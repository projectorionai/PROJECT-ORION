"""
Watching a video: its pictures, its voices, and what to do about it.

  "ORION must be able to analyse images, analyse voices and transcribe the
   voices too within those videos - I want ORION to be able to watch videos and
   identify what course of action to take dependent on the prompt given to him."

A video is two streams that have to be read differently and then read together.

  * The PICTURE — sampled as keyframes across the running time and described.
    Where a model with vision is reachable it looks at the frames directly; with
    no key or no network it falls back to OCR and structural analysis, which is
    exactly right for the videos most worth analysing (screen recordings,
    tutorials, slides, anything with text on screen).
  * The VOICE — the audio track pulled out, resampled, and transcribed by the
    same offline Whisper the live channel uses, so it works with no network at
    all.

Neither alone answers "what should I do about this"; the last stage hands the
frame notes AND the transcript to the model together, with the user's prompt, so
the course of action is reasoned from what was seen and what was said at once.

Every stage is independent and degrades rather than fails: no audio track, no
model key, an unreadable frame — each removes one input and is said plainly in
the result, never aborts the whole analysis.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import BASE_DIR
from .utils import first_line

#: Keyframes sampled by default. Enough to follow a video's arc without turning
#: every analysis into a dozen model calls.
DEFAULT_FRAMES = 8

# An independent fallback must not hold every sampled frame behind the SDK's
# retry chain. The routed vision path has already had its opportunity; on this
# bounded attempt OCR remains available. Native work keeps its owner until close.
MODEL_FALLBACK_TIMEOUT_S = 12.0

#: Whisper wants 16 kHz mono.
AUDIO_RATE = 16_000

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
                            ".wmv", ".flv", ".mpg", ".mpeg"})


@dataclass
class FrameNote:
    at: float               # seconds into the video
    text: str
    via: str = "ocr"        # "model" | "ocr"


@dataclass(eq=False)
class _CaptureWork:
    stop: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    future: Any = None


@dataclass
class VideoReport:
    path: str
    seconds: float = 0.0
    duration: float = 0.0
    frames: list[FrameNote] = field(default_factory=list)
    transcript: str = ""
    summary: str = ""
    action: str = ""
    notes: list[str] = field(default_factory=list)   # what degraded, if anything
    captured_frames: int = 0
    audio_state: str = "unavailable"  # transcribed | no_speech | no_audio | unavailable | not_captured
    audio_note: str = ""

    def audio_evidence(self) -> str:
        """Describe the observation, without treating absent evidence as silence."""
        if self.transcript:
            return "with speech transcribed"
        if self.audio_note:
            return self.audio_note
        if self.audio_state == "not_captured":
            return "no audio was captured"
        if self.audio_state == "no_audio":
            return "no decoded audio was available"
        if self.audio_state == "no_speech":
            return "the recognizer detected no discernible speech"
        return "audio transcription unavailable"

    def announcement(self) -> str:
        """What ORION says when the analysis is done — explicit, per the ask."""
        name = Path(self.path).name
        if not self.frames and not self.transcript:
            detail = "; ".join(self.notes) or self.audio_evidence()
            return f"I could not analyse {name}: no readable frames or transcript. {detail}"
        mins = int(self.duration // 60)
        secs = int(self.duration % 60)
        length = f"{mins}m{secs:02d}s" if self.duration else "unknown length"
        return (f"I analysed {name} ({length}) from {len(self.frames)} sampled frames, "
                f"{self.audio_evidence()}. " + (self.action or self.summary or "")).strip()


class VideoAnalyser:
    """Reads a video's frames and voice, then reasons about them together."""

    def __init__(self, bus: Any = None, router: Any = None, vision: Any = None,
                 transcriber: Any = None, api_key: str = "") -> None:
        self.bus = bus
        self.router = router
        self.vision = vision
        self.transcriber = transcriber
        self.api_key = api_key
        self._capture_lock = threading.Lock()
        self._capture_workers: set[_CaptureWork] = set()

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except Exception:
                pass

    # ── the whole pipeline ───────────────────────────────────────────────────

    async def _capture_owned(self, capture: Any, *args: Any) -> Any:
        """Request cooperative capture stop while retaining real settlement.

        Cancelling an executor await cannot interrupt a native read. The
        worker owns its handle until its finally runs, checks stop before the
        next capture, and discards this request's ownership only on settlement.
        No cancellation path waits synchronously for a device to respond.
        """
        work = _CaptureWork()
        with self._capture_lock:
            self._capture_workers.add(work)

        def run() -> Any:
            with self._capture_lock:
                work.started = True
            try:
                return capture(*args, work.stop)
            finally:
                with self._capture_lock:
                    self._capture_workers.discard(work)

        def settled(future: Any) -> None:
            if future.cancelled():
                # Loop shutdown may cancel a queued executor future too.
                work.stop.set()
                with self._capture_lock:
                    if not work.started:
                        self._capture_workers.discard(work)
            else:
                future.exception()  # observe a failure arriving after cancellation

        try:
            work.future = asyncio.get_running_loop().run_in_executor(None, run)
        except BaseException:
            with self._capture_lock:
                self._capture_workers.discard(work)
            raise
        work.future.add_done_callback(settled)
        try:
            return await asyncio.shield(work.future)
        except asyncio.CancelledError:
            work.stop.set()
            raise

    async def analyse(self, path: str, prompt: str = "",
                      max_frames: int = DEFAULT_FRAMES) -> VideoReport:
        import time
        started = time.monotonic()
        report = VideoReport(path=str(path))
        source = Path(path).expanduser()
        if not source.is_file():
            report.notes.append("file not found")
            return report
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            report.notes.append(f"{source.suffix} is not a recognised video format")

        self._log(f"VIDEO: analysing {source.name} …")

        # Frames and audio are independent — extract both off the event loop.
        frames, duration = await asyncio.to_thread(
            self._extract_keyframes, source, max_frames)
        report.duration = duration
        if not frames:
            report.notes.append("no frames could be read")

        # Describe frames and transcribe audio concurrently.
        describe = self._describe_frames(frames, prompt)
        transcribe = self._transcribe(source)
        report.frames, (report.transcript, audio_note) = await asyncio.gather(
            describe, transcribe)
        report.audio_note = audio_note
        if report.transcript:
            report.audio_state = "transcribed"
        elif audio_note in {"the video has no audio track", "the audio track was empty"}:
            report.audio_state = "no_audio"
        elif audio_note == "audio had no discernible speech":
            report.audio_state = "no_speech"
        if audio_note:
            report.notes.append(audio_note)

        await self._synthesise(report, prompt)
        report.seconds = time.monotonic() - started
        self._log(f"VIDEO: finished {source.name} in {report.seconds:.0f}s "
                  f"({len(report.frames)} frames, "
                  f"audio: {report.audio_state}).")
        return report

    # ── watching the SCREEN (a video playing, a tutorial, a stream) ──────────

    async def watch_screen(self, seconds: float = 30.0, prompt: str = "",
                           max_frames: int = DEFAULT_FRAMES,
                           window_only: bool = False) -> VideoReport:
        """Watch whatever is playing ON SCREEN for *seconds*.

        analyse() needs a file, so a video in the browser (YouTube, a course,
        a stream) was invisible to ORION. This samples the screen — or only
        the window in front — every ~1.5 s, keeps the frames where something
        actually changed, describes them the same way as a file's keyframes,
        and takes the words from what the microphone heard over the same
        window (speakers only: headphone audio never reaches the mic, and
        the report says so when nothing was heard)."""
        import time
        started = time.monotonic()
        seconds = max(3.0, min(180.0, float(seconds)))
        report = VideoReport(path="the screen" if not window_only else "the window in front")
        self._log(f"VIDEO: watching {report.path} for {seconds:.0f}s …")
        frames = await self._capture_owned(self._sample_screen, seconds, max_frames, window_only)
        report.duration = seconds
        if not frames:
            report.notes.append("the screen could not be captured")
        report.frames = await self._describe_frames(frames, prompt)
        report.transcript = await asyncio.to_thread(self._heard_during, seconds)
        if report.transcript:
            report.audio_state = "transcribed"
        else:
            report.audio_note = ("no microphone transcript was available "
                                 "(headphone audio does not reach the microphone)")
            report.notes.append(report.audio_note)
        await self._synthesise(report, prompt)
        report.seconds = time.monotonic() - started
        return report

    async def watch_camera(self, seconds: float = 6.0, prompt: str = "",
                           max_frames: int = 6,
                           camera_index: int | None = None) -> VideoReport:
        """Watch a short, explicitly requested webcam action in volatile memory.

        Unlike a single camera still, ordered frames show whether a movement
        happened and how it changed. Capture owns one camera handle, or borrows
        the face tracker's live frames; no recording is written to disk.
        """
        import time

        started = time.monotonic()
        seconds = max(2.0, min(15.0, float(seconds)))
        max_frames = max(2, min(8, int(max_frames)))
        report = VideoReport(path="the camera", duration=seconds, audio_state="not_captured")
        self._log(f"CAMERA: watching for {seconds:.0f} seconds on request.")
        frames, problem = await self._capture_owned(
            self._sample_camera, seconds, max_frames, camera_index)
        if problem:
            report.notes.append(problem)
        await self._describe_camera_sequence(report, frames, prompt)
        report.seconds = time.monotonic() - started
        return report

    async def analyse_recent_camera(self, frames: list[tuple[float, bytes]],
                                    prompt: str = "") -> VideoReport:
        """Explain a prior, volatile camera sequence supplied by local perception."""
        import time

        started = time.monotonic()
        duration = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0.0
        report = VideoReport(path="the recent camera view", duration=duration, audio_state="not_captured")
        if not frames:
            report.notes.append("no recent camera frames are available; start local camera watching first")
            return report
        await self._describe_camera_sequence(report, frames, prompt)
        report.seconds = time.monotonic() - started
        return report

    async def _describe_camera_sequence(self, report: VideoReport,
                                        frames: list[tuple[float, bytes]],
                                        prompt: str) -> None:
        """Share one description path for live and retrospective camera frames."""
        if frames:
            report.captured_frames = len(frames)
            has_vision = callable(getattr(self.router, "generate_vision", None))
            if has_vision:
                from .vision_describe import describe_image

                sheet = await asyncio.to_thread(self._camera_contact_sheet, frames)
                request = (
                    "These numbered webcam frames are in time order. Describe "
                    "what changed, answer the user's question, and give specific "
                    "advice grounded in visible evidence. There is no audio. "
                    f"User's question: {prompt or 'What happened?'}"
                )
                ok, answer = await describe_image(self.router, sheet, request,
                                                  max_tokens=450)
                if ok:
                    report.frames = [FrameNote(at=0.0, text=answer, via="model")]
                    report.summary = answer
                else:
                    report.notes.append(answer)
            if not report.frames:
                # Keep a provider outage to one attempt. Local OCR can still
                # recover labels in a demonstration without spending more RPM.
                for at, jpeg in frames:
                    text = await self._describe_with_ocr(jpeg)
                    if text:
                        report.frames.append(FrameNote(at=at, text=text, via="ocr"))
                report.summary = self._offline_summary(report)
                if report.frames and self.router is not None and not has_vision:
                    await self._synthesise(report, prompt)
            if not report.frames:
                report.notes.append("the camera frames could not be described")

    @staticmethod
    def _camera_contact_sheet(frames: list[tuple[float, bytes]]) -> bytes:
        """One labelled image for one vision call, irrespective of frame count."""
        from io import BytesIO
        from PIL import Image, ImageDraw

        columns = min(3, len(frames))
        rows = (len(frames) + columns - 1) // columns
        width, height = 320, 204
        sheet = Image.new("RGB", (columns * width, rows * height), "#111111")
        draw = ImageDraw.Draw(sheet)
        for number, (at, jpeg) in enumerate(frames, start=1):
            x = ((number - 1) % columns) * width
            y = ((number - 1) // columns) * height
            with Image.open(BytesIO(jpeg)) as image:
                tile = image.convert("RGB")
                tile.thumbnail((width, 180))
                sheet.paste(tile, (x + (width - tile.width) // 2,
                                   y + (180 - tile.height) // 2))
            draw.text((x + 8, y + 184), f"Frame {number} - {at:.1f} s",
                      fill="white")
        output = BytesIO()
        sheet.save(output, format="JPEG", quality=83)
        return output.getvalue()

    def _sample_camera(self, seconds: float, max_frames: int,
                       camera_index: int | None,
                       stop: threading.Event | None = None) -> tuple[list[tuple[float, bytes]], str]:
        """Sample one webcam session without retaining its frames afterwards."""
        import time

        if stop is not None and stop.is_set():
            return [], "camera watch cancelled"
        try:
            import cv2
        except ImportError:
            return [], "OpenCV is not installed"

        from . import camera_devices
        from .utils import open_camera_capture

        if stop is not None and stop.is_set():
            return [], "camera watch cancelled"
        index = camera_devices.resolve() if camera_index is None else camera_index
        source = getattr(self.vision, "_live_frame_source", None)
        owner = getattr(source, "__self__", None)
        if getattr(owner, "camera_index", index) != index:
            source = None
        is_capturing = getattr(owner, "is_capturing", None)
        if callable(is_capturing) and not is_capturing():
            source = None

        capture = None
        if source is None:
            capture, _backend = open_camera_capture(index)
            if capture is None:
                return [], (f"camera {index} could not be opened; check Windows "
                            "camera permission or close another camera app")
        frames: list[tuple[float, bytes]] = []
        failed = 0
        start = time.monotonic()
        try:
            for slot in range(max_frames):
                due = start + (seconds * slot / (max_frames - 1))
                delay = max(0.0, due - time.monotonic())
                if stop is None:
                    time.sleep(delay)
                elif stop.wait(delay):
                    break
                if source is not None:
                    frame = source()
                else:
                    ok, frame = capture.read()
                    if not ok:
                        frame = None
                if stop is not None and stop.is_set():
                    break
                if frame is None:
                    failed += 1
                    continue
                frame = self._shrink(cv2, frame, 1024)
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ok:
                    frames.append((round(time.monotonic() - start, 2), bytes(encoded)))
                else:
                    failed += 1
        except Exception as exc:
            return frames, f"camera stopped delivering frames ({first_line(exc, 100)})"
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
        if not frames:
            return [], "the camera returned no frames"
        return frames, (f"{failed} camera frame(s) were missed" if failed else "")

    @staticmethod
    def _sample_screen(seconds: float, max_frames: int,
                       window_only: bool,
                       stop: threading.Event | None = None) -> list[tuple[float, bytes]]:
        """(t, jpeg) for frames that changed meaningfully, evenly thinned."""
        import time

        if stop is not None and stop.is_set():
            return []
        import cv2
        import mss
        import numpy as np

        region = None
        if window_only:
            try:
                import ctypes
                from ctypes import wintypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                rect = wintypes.RECT()
                if hwnd and ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    region = {"left": rect.left, "top": rect.top,
                              "width": max(64, rect.right - rect.left),
                              "height": max(64, rect.bottom - rect.top)}
            except Exception:
                region = None
        kept: list[tuple[float, bytes]] = []
        last_small = None
        start = time.monotonic()
        if stop is not None and stop.is_set():
            return []
        with mss.mss() as grab:
            target = region or grab.monitors[1]
            while time.monotonic() - start < seconds:
                if stop is not None and stop.is_set():
                    break
                shot = np.asarray(grab.grab(target))[:, :, :3]
                if stop is not None and stop.is_set():
                    break
                small = cv2.resize(cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY), (160, 90))
                changed = (last_small is None or
                           np.mean(np.abs(small.astype(np.int16) - last_small) > 18) > 0.03)
                if changed:
                    last_small = small.astype(np.int16)
                    frame = shot
                    h, w = frame.shape[:2]
                    scale = min(1.0, 1280.0 / max(h, w))
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        kept.append((round(time.monotonic() - start, 1), bytes(buf)))
                    if len(kept) > max_frames * 6:           # bound memory
                        kept = kept[::2]
                if stop is None:
                    time.sleep(1.5)
                elif stop.wait(1.5):
                    break
        if len(kept) > max_frames:
            step = len(kept) / float(max_frames)
            kept = [kept[int(i * step)] for i in range(max_frames)]
        return kept

    def _heard_during(self, seconds: float) -> str:
        """Speech the microphone picked up over the last *seconds*."""
        try:
            from .audio import RECENT_AUDIO
            pcm = RECENT_AUDIO.snapshot(min(seconds, 30.0))
            if len(pcm) < 16000:
                return ""
            transcriber = self.transcriber
            if transcriber is None:
                from . import speech_offline
                transcriber = speech_offline.shared()
            return (transcriber.transcribe_pcm(pcm, 16000) or "").strip()
        except Exception:
            return ""

    # ── frames ───────────────────────────────────────────────────────────────

    def _extract_keyframes(self, source: Path,
                           count: int) -> tuple[list[tuple[float, bytes]], float]:
        """Evenly-spaced JPEG keyframes + the video's duration. Never raises."""
        try:
            import cv2
        except Exception:
            return [], 0.0
        cap = None
        try:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                return [], 0.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            duration = (total / fps) if fps > 0 else 0.0
            out: list[tuple[float, bytes]] = []
            if total <= 0:
                return [], duration
            # Sample inside the clip, avoiding the very first/last frame (often
            # black or a fade).
            step = total / (count + 1)
            for i in range(1, count + 1):
                index = int(step * i)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                # Downscale wide frames — a 4K still is wasted on description.
                frame = self._shrink(cv2, frame, 1024)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    at = (index / fps) if fps > 0 else 0.0
                    out.append((at, bytes(buf)))
            return out, duration
        except Exception:
            return [], 0.0
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    @staticmethod
    def _shrink(cv2: Any, frame: Any, max_side: int) -> Any:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return frame
        scale = max_side / longest
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    async def _describe_frames(self, frames: list[tuple[float, bytes]],
                               prompt: str) -> list[FrameNote]:
        notes: list[FrameNote] = []
        for at, jpeg in frames:
            text, via = await self._describe_one(jpeg, prompt)
            if text:
                notes.append(FrameNote(at=at, text=text, via=via))
        return notes

    async def _describe_one(self, jpeg: bytes, prompt: str) -> tuple[str, str]:
        # The router first: its vision path tries every vision-capable
        # provider with their key rotation and cooldowns. The direct call
        # below is an independent fallback when the router cannot inspect the
        # frame.  Its default must remain a currently supported image-input
        # model, rather than the retired Gemini 2.0 endpoint.
        if self.router is not None and hasattr(self.router, "generate_vision"):
            from .vision_describe import describe_image
            ask = ("Describe what is happening in this video frame in one or two "
                   "sentences" + (f", with an eye to: {prompt}" if prompt else "")
                   + ". Note any visible text, people, actions or UI.")
            ok, text = await describe_image(self.router, jpeg, ask, max_tokens=160)
            if ok:
                return first_line(text, 400), "model"
        # …then the configured key directly…
        if self.api_key:
            described = await self._describe_with_model(jpeg, prompt)
            if described:
                return described, "model"
        # …otherwise OCR + structural, which always works and is ideal for the
        # text-bearing videos most worth analysing.
        ocr = await self._describe_with_ocr(jpeg)
        return ocr, "ocr"

    async def _describe_with_model(self, jpeg: bytes, prompt: str) -> str:
        def _call(stop: threading.Event) -> str:
            if stop.is_set():
                return ""
            try:
                from google import genai
                from google.genai import types
            except Exception:
                return ""
            client = None
            try:
                if stop.is_set():
                    return ""
                client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=max(1, int(MODEL_FALLBACK_TIMEOUT_S * 1000)),
                        retry_options=types.HttpRetryOptions(attempts=1)),
                )
                # Cancellation can arrive during client construction too. It
                # must not admit a new model request after the caller has left.
                if stop.is_set():
                    return ""
                ask = ("Describe what is happening in this video frame in one or "
                       "two sentences" + (f", with an eye to: {prompt}" if prompt else "")
                       + ". Note any visible text, people, actions or UI.")
                resp = client.models.generate_content(
                    model=os.getenv("ORION_VIDEO_MODEL", "gemini-3.5-flash-lite").strip()
                          or "gemini-3.5-flash-lite",
                    contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                              ask])
                return first_line(str(getattr(resp, "text", "") or ""), 400)
            except Exception:
                return ""
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception as exc:
                        self._log("VIDEO: fallback client cleanup failed "
                                  f"({type(exc).__name__}).")
        try:
            # Reuse the owned native-work lifecycle: cancelling a waiter cannot
            # kill SDK I/O. The worker closes its client after actual settlement.
            return await asyncio.wait_for(
                self._capture_owned(_call), timeout=MODEL_FALLBACK_TIMEOUT_S)
        except TimeoutError:
            self._log("VIDEO: direct frame fallback exceeded its deadline; trying OCR.")
            return ""

    async def _describe_with_ocr(self, jpeg: bytes) -> str:
        if self.vision is None:
            return ""
        image_reader = getattr(self.vision, "_ocr_image_text", None)
        if callable(image_reader):
            def read_image() -> str:
                from io import BytesIO
                from PIL import Image

                with Image.open(BytesIO(jpeg)) as image:
                    return first_line(str(image_reader(image.convert("RGB")) or ""), 400)

            try:
                return await asyncio.to_thread(read_image)
            except Exception:
                return ""
        reader = getattr(self.vision, "_ocr_file_text", None)
        if reader is None:
            return ""

        def read_frame() -> str:
            # One worker owns creation, use and removal even if its awaiter is
            # cancelled. id(jpeg) collided when two calls shared the same bytes.
            with tempfile.TemporaryDirectory(prefix="orion_vframe_") as directory:
                tmp = Path(directory) / "frame.jpg"
                tmp.write_bytes(jpeg)
                return first_line(str(reader(tmp) or ""), 400)

        try:
            return await asyncio.to_thread(read_frame)
        except Exception:
            return ""

    # ── audio ────────────────────────────────────────────────────────────────

    async def _transcribe(self, source: Path) -> tuple[str, str]:
        """(transcript, note). Extract the audio track, then transcribe it."""
        if self.transcriber is None:
            return "", "no transcriber available"
        import threading
        cancelled = threading.Event()
        extracted = []

        def cleanup(path: Path | None) -> None:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass  # A running recognizer may still hold the Windows file handle.

        def extract() -> tuple[Path | None, str]:
            result = self._extract_audio(source)
            extracted.append(result[0])
            if cancelled.is_set():
                cleanup(result[0])
            return result

        try:
            wav_path, extract_note = await asyncio.to_thread(extract)
        except asyncio.CancelledError:
            # Thread cancellation cannot stop extraction. Its owner cleans a
            # late result, including the race where the result already exists
            # but the cancelled await never assigns wav_path.
            cancelled.set()
            for path in extracted:
                cleanup(path)
            raise
        if wav_path is None:
            return "", extract_note or "no audio track"

        def transcribe() -> Any:
            try:
                return self.transcriber.transcribe_wav(str(wav_path))
            finally:
                if cancelled.is_set():
                    cleanup(wav_path)

        try:
            ready = getattr(self.transcriber, "ensure_ready", None)
            if ready is not None:
                if await asyncio.to_thread(ready) is False:
                    return "", "transcription unavailable (recognizer is not ready)"
            text = await asyncio.to_thread(transcribe)
            text = str(text or "").strip()
            return text, ("" if text else "audio had no discernible speech")
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except Exception as exc:
            return "", f"transcription failed ({first_line(exc, 80)})"
        finally:
            cleanup(wav_path)

    def _extract_audio(self, source: Path) -> tuple[Path | None, str]:
        """Pull the audio track to a 16 kHz mono wav via PyAV. Never raises."""
        try:
            import av
        except Exception:
            return None, "PyAV not installed — cannot read the audio track"
        try:
            container = av.open(str(source))
        except Exception as exc:
            return None, f"could not open the video ({first_line(exc, 60)})"
        out = None
        try:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return None, "the video has no audio track"
            resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE)
            # Unique per call: named only after the clip, two readings of
            # "clip.mp4" at once (ORION and a test run, say) wrote, and then
            # deleted, the same file under each other.
            out = Path(tempfile.gettempdir()) / (
                f"orion_vaudio_{source.stem[:20]}_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav")
            with wave.open(str(out), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(AUDIO_RATE)
                for frame in container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        wav.writeframes(bytes(resampled.planes[0]))
            if out.stat().st_size <= 44:          # header only → silence
                out.unlink()
                return None, "the audio track was empty"
            return out, ""
        except Exception as exc:
            # Decode failures can arrive after a partial WAV was written. Only
            # remove this call's unique temporary file, retaining the source.
            if out is not None:
                try:
                    out.unlink(missing_ok=True)
                except OSError:
                    pass
            return None, f"audio extraction failed ({first_line(exc, 60)})"
        finally:
            try:
                container.close()
            except Exception:
                pass

    # ── the two, read together ───────────────────────────────────────────────

    async def _synthesise(self, report: VideoReport, prompt: str) -> None:
        if self.router is None or (not report.frames and not report.transcript):
            report.summary = self._offline_summary(report)
            return
        frame_text = "\n".join(f"[{fn.at:.0f}s] {fn.text}" for fn in report.frames)
        transcript = report.transcript[:6000]
        if transcript:
            report.audio_state = "transcribed"
        camera = report.audio_state == "not_captured" or report.path == "the camera"
        ask = (
            ("You are analysing a short webcam sequence from ordered frames. "
             "No audio was captured; do not infer what was said.\n\n" if camera else
             "You are analysing a video from its keyframes and its transcript.\n\n")
            + f"THE USER ASKED: {prompt or 'Summarise what changed and recommend what to do.'}\n\n"
            + f"KEYFRAMES (in order):\n{frame_text or '(none read)'}\n\n"
            + ("" if camera else
               f"AUDIO EVIDENCE: {report.audio_state}: {report.audio_evidence()}\n"
               f"TRANSCRIPT (what was said):\n{transcript or '(no transcript available)'}\n"
               "Missing or failed audio transcription does not establish silence; "
               "do not infer words that were not transcribed.\n\n")
            + "Write two short sections:\n"
            "SUMMARY: what the sequence actually shows.\n"
            "ACTION: the specific course of action the user's request calls for, "
            "grounded in the video. If the video does not support a confident "
            "answer, say so plainly rather than inventing one."
        )
        try:
            _profile, text = await self.router.generate_text(ask)
        except Exception as exc:
            report.notes.append(f"synthesis failed ({first_line(exc, 80)})")
            report.summary = self._offline_summary(report)
            return
        report.summary, report.action = self._split_summary_action(str(text or ""))

    @staticmethod
    def _split_summary_action(raw: str) -> tuple[str, str]:
        # Headings arrive as "SUMMARY:", "SUMMARY", "**Summary**" or "## Summary"
        # — models do not keep to one — so match the word on its own line or
        # before a colon, not only the exact "SUMMARY:".
        import re
        heading = r"(?im)(?:^[\s#*_]*{word}[\s*_]*:?[\s*_]*$|{word}[\s*_]*:)"
        summary, action = raw.strip(), ""
        found = re.search(heading.format(word="ACTION"), raw)
        if found:
            summary, action = raw[:found.start()], raw[found.end():]
        summary = summary.strip()
        lead = re.match(heading.format(word="SUMMARY"), summary)
        if lead:
            summary = summary[lead.end():]
        tidy = lambda text: text.strip().strip("*_#").strip()
        return tidy(summary), tidy(action)

    @staticmethod
    def _offline_summary(report: VideoReport) -> str:
        bits = []
        if report.frames:
            where = "Through the camera" if report.path == "the camera" else "On screen"
            bits.append(where + ": " + "; ".join(fn.text for fn in report.frames[:4]))
        if report.transcript:
            bits.append("Said: " + report.transcript[:400])
        return " ".join(bits) or "The video could not be read for content."


__all__ = ["AUDIO_RATE", "DEFAULT_FRAMES", "VIDEO_SUFFIXES", "FrameNote",
           "VideoAnalyser", "VideoReport"]
