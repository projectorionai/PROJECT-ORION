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

#: Whisper wants 16 kHz mono.
AUDIO_RATE = 16_000

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
                            ".wmv", ".flv", ".mpg", ".mpeg"})


@dataclass
class FrameNote:
    at: float               # seconds into the video
    text: str
    via: str = "ocr"        # "model" | "ocr"


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

    def announcement(self) -> str:
        """What ORION says when the analysis is done — explicit, per the ask."""
        name = Path(self.path).name
        mins = int(self.duration // 60)
        secs = int(self.duration % 60)
        length = f"{mins}m{secs:02d}s" if self.duration else "unknown length"
        heard = "with speech transcribed" if self.transcript else "no speech found"
        return (f"I've watched {name} ({length}): {len(self.frames)} frames "
                f"analysed, {heard}. " + (self.action or self.summary or "")).strip()


class VideoAnalyser:
    """Reads a video's frames and voice, then reasons about them together."""

    def __init__(self, bus: Any = None, router: Any = None, vision: Any = None,
                 transcriber: Any = None, api_key: str = "") -> None:
        self.bus = bus
        self.router = router
        self.vision = vision
        self.transcriber = transcriber
        self.api_key = api_key

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except Exception:
                pass

    # ── the whole pipeline ───────────────────────────────────────────────────

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
        if audio_note:
            report.notes.append(audio_note)

        await self._synthesise(report, prompt)
        report.seconds = time.monotonic() - started
        self._log(f"VIDEO: finished {source.name} in {report.seconds:.0f}s "
                  f"({len(report.frames)} frames, "
                  f"{'transcript' if report.transcript else 'no audio'}).")
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
        frames = await asyncio.to_thread(self._sample_screen, seconds, max_frames, window_only)
        report.duration = seconds
        if not frames:
            report.notes.append("the screen could not be captured")
        report.frames = await self._describe_frames(frames, prompt)
        report.transcript = await asyncio.to_thread(self._heard_during, seconds)
        if not report.transcript:
            report.notes.append("no speech reached the microphone (headphone audio "
                                "cannot be heard)")
        await self._synthesise(report, prompt)
        report.seconds = time.monotonic() - started
        return report

    @staticmethod
    def _sample_screen(seconds: float, max_frames: int,
                       window_only: bool) -> list[tuple[float, bytes]]:
        """(t, jpeg) for frames that changed meaningfully, evenly thinned."""
        import time

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
        with mss.mss() as grab:
            target = region or grab.monitors[1]
            while time.monotonic() - start < seconds:
                shot = np.asarray(grab.grab(target))[:, :, :3]
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
                time.sleep(1.5)
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
        # below pins one model id (gemini-2.0-flash) outside all of that, so
        # a retired model or a cooled key silently dropped every frame to OCR.
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
        def _call() -> str:
            try:
                from google import genai
                from google.genai import types
            except Exception:
                return ""
            try:
                client = genai.Client(api_key=self.api_key)
                ask = ("Describe what is happening in this video frame in one or "
                       "two sentences" + (f", with an eye to: {prompt}" if prompt else "")
                       + ". Note any visible text, people, actions or UI.")
                resp = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                              ask])
                return first_line(str(getattr(resp, "text", "") or ""), 400)
            except Exception:
                return ""
        return await asyncio.to_thread(_call)

    async def _describe_with_ocr(self, jpeg: bytes) -> str:
        if self.vision is None:
            return ""
        tmp = Path(tempfile.gettempdir()) / f"orion_vframe_{id(jpeg)}.jpg"
        try:
            tmp.write_bytes(jpeg)
        except OSError:
            return ""
        try:
            reader = getattr(self.vision, "_ocr_file_text", None)
            if reader is None:
                return ""
            text = await asyncio.to_thread(reader, tmp)
            return first_line(str(text or ""), 400)
        except Exception:
            return ""
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    # ── audio ────────────────────────────────────────────────────────────────

    async def _transcribe(self, source: Path) -> tuple[str, str]:
        """(transcript, note). Extract the audio track, then transcribe it."""
        if self.transcriber is None:
            return "", "no transcriber available"
        wav_path, extract_note = await asyncio.to_thread(self._extract_audio, source)
        if wav_path is None:
            return "", extract_note or "no audio track"
        try:
            ready = getattr(self.transcriber, "ensure_ready", None)
            if ready is not None:
                await asyncio.to_thread(ready)
            text = await asyncio.to_thread(self.transcriber.transcribe_wav, str(wav_path))
            return str(text or "").strip(), ("" if text else "audio had no discernible speech")
        except Exception as exc:
            return "", f"transcription failed ({first_line(exc, 80)})"
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass

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
            return None, f"audio extraction failed ({first_line(exc, 60)})"
        finally:
            try:
                container.close()
            except Exception:
                pass

    # ── the two, read together ───────────────────────────────────────────────

    async def _synthesise(self, report: VideoReport, prompt: str) -> None:
        if self.router is None:
            report.summary = self._offline_summary(report)
            return
        frame_text = "\n".join(f"[{fn.at:.0f}s] {fn.text}" for fn in report.frames)
        transcript = report.transcript[:6000]
        ask = (
            "You are analysing a video from its keyframes and its transcript.\n\n"
            f"THE USER ASKED: {prompt or 'Summarise the video and recommend what to do.'}\n\n"
            f"KEYFRAMES (what was on screen, in order):\n{frame_text or '(none read)'}\n\n"
            f"TRANSCRIPT (what was said):\n{transcript or '(no speech)'}\n\n"
            "Write two short sections:\n"
            "SUMMARY: what the video actually shows and says.\n"
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
            bits.append("On screen: " + "; ".join(fn.text for fn in report.frames[:4]))
        if report.transcript:
            bits.append("Said: " + report.transcript[:400])
        return " ".join(bits) or "The video could not be read for content."


__all__ = ["AUDIO_RATE", "DEFAULT_FRAMES", "VIDEO_SUFFIXES", "FrameNote",
           "VideoAnalyser", "VideoReport"]
