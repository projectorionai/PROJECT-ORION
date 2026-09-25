"""
Dispatch domain — Screen capture, visual analysis and verified on-screen interaction.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line


class VisionDispatchMixin:
    """Vision tools, including the continuous perception loop (Track E)."""

    async def perception_tool(self, args: dict[str, Any]) -> ToolResult:
        """Continuous local watching, with cloud understanding only on change.

        Distinct from ``vision_analyse``, which is one paid look on demand:
        this runs locally and free the whole time it is on, and reaches for the
        cloud only when something has actually happened.
        """
        if self.perception is None:
            return ToolResult(
                "The perception loop is not available; use vision_analyse for a "
                "single look instead.", ok=False)
        action = str(args.get("action") or "status").strip().lower().replace(" ", "_")
        if action in {"start", "watch", "on"}:
            return ToolResult(self.perception.start())
        if action in {"stop", "off"}:
            return ToolResult(await self.perception.stop())
        if action in {"scene", "what", "view"}:
            return ToolResult(self.perception.scene.describe())
        if action in {"events", "history", "recent"}:
            return ToolResult(self.perception.recent(
                limit=int(args.get("limit") or 12)))
        if action in {"status", "report"}:
            return ToolResult(self.perception.report())
        if action in {"analyse", "analyze", "numbers", "read", "measure", "grid"}:
            return await self._perception_measure(show_grid=action == "grid"
                                                  or bool(args.get("grid")))
        if action in {"detect", "objects", "identify"}:
            return await self._perception_detect()
        if action in {"pose", "body", "gestures"}:
            pose, problem = await self.perception.pose_now()
            return ToolResult(problem, ok=False) if pose is None else ToolResult(pose.describe())
        if action in {"install_detector", "install_objects", "install_model"}:
            return await self._perception_install_detector()
        if action in {"rules", "list_rules"}:
            if self.perception.rules is None:
                return ToolResult("Vision rules are not available.", ok=False)
            return ToolResult(self.perception.rules.describe())
        if action in {"add_rule", "rule", "when"}:
            return self._perception_add_rule(args)
        if action in {"remove_rule", "delete_rule"}:
            rules = self.perception.rules
            removed = rules.remove(str(args.get("rule") or args.get("name") or "")) if rules else None
            return ToolResult(f"Removed vision rule {removed.id} ({removed.name})." if removed
                              else "I have no vision rule by that id or name.", ok=removed is not None)
        if action in {"enable_rule", "disable_rule"}:
            rules = self.perception.rules
            rule = rules.set_enabled(str(args.get("rule") or args.get("name") or ""),
                                     action == "enable_rule") if rules else None
            return ToolResult(rule.describe() if rule else "I have no vision rule by that id or name.",
                              ok=rule is not None)
        if action in {"autostart", "watch_on_startup"}:
            rules = self.perception.rules
            if rules is None:
                return ToolResult("Vision rules are not available.", ok=False)
            on = str(args.get("enabled", "true")).strip().lower() not in {"false", "0", "off", "no"}
            rules.set_watch_on_startup(on)
            return ToolResult("I'll start watching whenever I start, so your vision rules are "
                              "always armed." if on else
                              "I won't watch at start-up; rules run while I'm watching.")
        if action in {"overlay", "camera_view", "show"}:
            return self._perception_overlay(str(args.get("mode") or "normal"))
        return ToolResult(
            f"Unsupported perception action: {action}. Use start, stop, status, scene, "
            "events, analyse, grid, detect, pose, install_detector, rules, add_rule, "
            "remove_rule, enable_rule, disable_rule, autostart or overlay.", ok=False)

    async def _perception_measure(self, show_grid: bool) -> ToolResult:
        reading = await self.perception.read_now()
        if reading is None:
            return ToolResult("I can't get a frame from the camera to measure.", ok=False)
        text = reading.summary()
        if show_grid:
            text += ("\n\nThe view as a grid of numbers (brightness 0-9, 16×12):\n"
                     + reading.number_grid())
        return ToolResult(text)

    async def _perception_detect(self) -> ToolResult:
        from .object_detection import describe
        detections, problem = await self.perception.detect_now()
        if detections is None:
            return ToolResult(problem, ok=False)
        return ToolResult(describe(detections))

    async def _perception_install_detector(self) -> ToolResult:
        detector = self.perception.objects
        if detector is None:
            return ToolResult("Object detection is not part of this build.", ok=False)
        if not detector.runtime_available():
            return ToolResult("Object detection needs onnxruntime, which is not installed.",
                              ok=False)
        text = await asyncio.to_thread(
            detector.install_model, lambda note: self.bus.log.emit(f"VISION: {note}"))
        return ToolResult(text, ok=detector.available)

    def _perception_add_rule(self, args: dict[str, Any]) -> ToolResult:
        rules = self.perception.rules
        if rules is None:
            return ToolResult("Vision rules are not available.", ok=False)

        def number(key: str) -> float | None:
            value = args.get(key)
            try:
                return float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        # A rule that names a workflow which doesn't exist would only find out
        # when the camera fired it, with nobody watching. Refuse it now.
        workflow = str(args.get("workflow") or "").strip()
        engine = getattr(getattr(self, "automation", None), "engine", None)
        if workflow and engine is not None and engine._slug(workflow) not in engine.definitions:
            known = ", ".join(sorted(engine.definitions)) or "none yet"
            return ToolResult(f"I have no workflow called '{workflow}'. Your workflows: "
                              f"{known}.", ok=False)
        try:
            rule = rules.add(
                trigger=str(args.get("when") or args.get("trigger") or ""),
                target=str(args.get("target") or ""),
                zone=args.get("zone"),
                count=int(number("count") or 0) or None,
                hold_s=number("hold"),
                cooldown_s=number("cooldown"),
                action=str(args.get("then") or "announce"),
                workflow=str(args.get("workflow") or ""),
                message=str(args.get("message") or ""),
                name=str(args.get("name") or ""),
                min_confidence=number("confidence"),
                share=number("share"),
            )
        except ValueError as exc:
            return ToolResult(f"I can't set that rule up: {exc}.", ok=False)
        notes = [f"Rule added — {rule.describe()}."]
        needs = {"object": "objects", "object_gone": "objects", "count": "objects",
                 "pose": "pose"}.get(rule.trigger)
        if needs == "objects" and not getattr(self.perception.objects, "available", False):
            notes.append("It needs the object detector, which isn't installed yet — ask me "
                         "to install the object detector (a one-off 34 MB download).")
        if needs == "pose" and not getattr(self.perception.pose_tracker, "available", False):
            notes.append("It needs pose tracking, which this build doesn't include.")
        if not self.perception.running:
            notes.append(self.perception.start())
        return ToolResult(" ".join(notes))

    def _perception_overlay(self, mode: str) -> ToolResult:
        from .vision_lab import OVERLAY_MODES
        window = getattr(self, "live_camera_window", None)
        mode = mode.strip().lower()
        mode = {"objects": "detect", "boxes": "detect", "grid": "numbers", "plain": "normal",
                "flow": "motion", "color": "colours", "colors": "colours",
                "colour": "colours", "edge": "edges"}.get(mode, mode)
        if mode not in OVERLAY_MODES:
            return ToolResult(f"'{mode}' isn't a camera view — use one of: "
                              + ", ".join(OVERLAY_MODES), ok=False)
        if window is None:
            return ToolResult("The live camera window is not available.", ok=False)
        # The window only CONSUMES frames; watching is what switches the
        # camera on (and keeps the readings the views draw from flowing).
        note = "" if self.perception.running else " " + self.perception.start()
        window.set_overlay(mode)
        window.start()
        return ToolResult(f"Showing the live camera in '{mode}' view.{note}")

    """Screen capture, visual analysis and verified on-screen interaction."""

    async def sound_sense_tool(self, args: dict[str, Any]) -> ToolResult:
        """ORION's hearing beyond speech (see sound_sense)."""
        from . import sound_sense
        from .audio import RECENT_AUDIO

        action = str(args.get("action") or "recent").lower().strip()
        if action in {"watch", "watch_on", "start_watch", "watch_off", "stop_watch",
                      "watch_status"}:
            return await self._sound_watch_action(action)
        try:
            seconds = max(1.0, min(30.0, float(args.get("seconds") or 8.0)))
        except (TypeError, ValueError):
            seconds = 8.0
        transcribe = args.get("transcribe", True) is not False
        if action in {"file", "analyse_file", "analyze_file", "audio_file", "video"}:
            path = str(args.get("path") or "").strip().strip('"')
            if not path or not os.path.isfile(path):
                return ToolResult("Give me the path of an audio or video file to listen to.",
                                  ok=False)
            try:
                wave = await asyncio.to_thread(sound_sense.load_file, path)
            except Exception as exc:
                return ToolResult(f"I couldn't read the sound in that file: {exc}", ok=False)
            where = f"in {os.path.basename(path)}"
        else:
            if action in {"listen", "record", "now"}:
                await asyncio.sleep(seconds)
            pcm = RECENT_AUDIO.snapshot(seconds)
            if len(pcm) < 16000 and not RECENT_AUDIO.fresh():
                # The live microphone pipeline is not running (muted, no
                # device, test harness): record directly for the window asked.
                pcm = await asyncio.to_thread(_record_pcm16, seconds)
            if len(pcm) < 16000 // 2:
                return ToolResult("I haven't heard anything in that window — the "
                                  "microphone may be muted or not capturing.", ok=False)
            wave = sound_sense.pcm16_to_float(pcm)
            where = f"in the last {seconds:.0f} seconds"
        report = await asyncio.to_thread(sound_sense.analyse, wave)
        text = report.summary().replace(f"Heard over {report.seconds:.0f} s:",
                                        f"Heard {where}:")
        if transcribe and report.speech:
            words = await asyncio.to_thread(_transcribe_float, wave)
            if words:
                text += f' Words heard: "{words[:600]}"'
        return ToolResult(text)

    async def _sound_watch_action(self, action: str) -> ToolResult:
        """Start, stop or report the background sound watch (sound_watch)."""
        from . import sound_watch

        bus = getattr(self, "bus", None)

        def announce(alert: "sound_watch.Alert") -> None:
            # Called on the watch thread: Qt queues each emit to its receivers.
            if bus is None:
                return
            bus.log.emit(f"SOUND: {alert.category} ({alert.label}, {alert.score:.2f})")
            bus.dashboard_event.emit("sound_alert", alert.as_dict())
            bus.speak_request.emit(alert.sentence())

        log = bus.log.emit if bus is not None else None
        watcher = sound_watch.watch(announce, log)
        if action in {"watch_off", "stop_watch"}:
            await asyncio.to_thread(watcher.stop)
            return ToolResult("Stopped listening for alarms and the door.")
        if action == "watch_status":
            return ToolResult(watcher.describe())
        ok, message = await asyncio.to_thread(watcher.start)
        return ToolResult(("Now " if ok and "already" not in message else "") + message
                          if ok else f"I can't watch for sounds: {message}", ok=ok)

    async def vision_analyse(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "describe").lower().strip()
        prompt = str(args.get("prompt") or args.get("question") or "")
        path   = str(args.get("path") or "").strip()
        scan_actions = {"scan", "inspect", "identify", "identify_object", "scan_object",
                        "what_is_this", "count", "count_items", "read_label", "scan_text"}
        if action in scan_actions or action in {"pcb", "electronics", "inspect_pcb", "scan_pcb",
                                                "circuit_board", "inspect_electronics"}:
            import uuid
            from .electronics_inspection import normalise_mode

            if action in scan_actions:
                mode = normalise_mode(args.get("mode") or (
                    "count" if action.startswith("count") else
                    "text" if action in {"read_label", "scan_text"} else "anything"))
            else:
                mode = "electronics"
            raw_grid = args.get("grid")
            grid = (8, 6)
            if isinstance(raw_grid, str) and "x" in raw_grid.lower():
                try:
                    cols, rows = raw_grid.lower().replace("×", "x").split("x", 1)
                    grid = (int(cols), int(rows))
                except ValueError:
                    pass
            elif raw_grid in (0, "0", "none", "off", False):
                grid = (0, 0)
            request_id = uuid.uuid4().hex
            bus = getattr(self, "bus", None)

            def emit(name: str, payload: dict[str, Any]) -> None:
                signal = getattr(bus, name, None)
                if signal is not None:
                    signal.emit(payload)

            emit("electronics_scan_started", {"request_id": request_id})
            try:
                raw_index = args.get("camera_index", args.get("camera"))
                camera_index = max(0, int(raw_index)) if raw_index is not None else None
            except (TypeError, ValueError):
                camera_index = None
            ai = getattr(self, "ai", None)
            router = getattr(ai, "router", None) or getattr(self, "router", None)
            try:
                result = await asyncio.wait_for(
                    self.vision.inspect_electronics(prompt, camera_index, router=router,
                                                    mode=mode, grid=grid), timeout=60.0)
            except asyncio.CancelledError:
                emit("electronics_scan_error", {"request_id": request_id, "status": "error",
                                                "message": "Electronics inspection cancelled."})
                raise
            except asyncio.TimeoutError:
                result = ToolResult("The electronics scan timed out. Try capturing another frame.", ok=False)
            except Exception:
                result = ToolResult("The electronics scan could not complete. Try capturing another frame.", ok=False)
            if not result.ok:
                emit("electronics_scan_error", {"request_id": request_id, "status": "error",
                                              "summary": result.text, "message": result.text})
                return result
            report = dict((result.evidence or [{}])[0])
            report.update(request_id=request_id, jpeg=(result.media or {}).get("data"), ok=True)
            emit("electronics_scan_result", report)
            return result
        if action in {"describe", "screen", "analyse_screen", "analyze_screen", "look"}:
            return await self.vision.analyse_screen(prompt)
        if action in {"read_window", "exact_text", "read_screen_text", "window_text",
                      "what_am_i_typing", "read_app", "read_page"}:
            window = str(args.get("window") or args.get("app") or args.get("target") or "")
            return await self.vision.read_window(window)
        if action in {"camera", "webcam", "background", "surroundings", "look_around", "room"}:
            # None unless the user named one: capture_live_frame then resolves
            # the REMEMBERED camera and can borrow the face tracker's frame.
            # A hard-coded 0 opened whatever device happened to be first (a
            # virtual camera, the wrong webcam) and, when the tracker held the
            # real one at another index, refused to borrow and tried a second
            # handle that Windows rejects — "the camera gets confused".
            raw_index = args.get("camera_index")
            if raw_index is None:
                raw_index = args.get("camera")
            try:
                camera_index = int(raw_index) if raw_index not in (None, "") else None
            except (TypeError, ValueError):
                camera_index = None
            return await self.vision.analyse_camera(prompt, camera_index)
        if action in {"ocr", "read_text", "extract_text"}:
            return await self.vision.ocr(path)
        if action in {"find_errors", "detect_errors", "errors", "diagnose"}:
            return await self.vision.detect_errors()
        if action in {"analyse_image", "analyze_image", "image"}:
            if not path:
                return ToolResult("An image path is required for analyse_image.", ok=False)
            return await self.vision.analyse_image(path, prompt)
        if action in {"watch_screen", "watch_this", "screen_video", "watch_my_screen",
                      "watch_window"}:
            return await self._watch_screen(prompt, args, window_only=(
                action == "watch_window" or bool(args.get("window_only"))))
        if action in {"video", "analyse_video", "analyze_video", "watch",
                      "watch_video"}:
            if not path:
                return ToolResult("A video file path is required to watch a video.",
                                  ok=False)
            return await self._analyse_video(path, prompt, args)
        return ToolResult(
            f"Unsupported vision action: {action}. "
            "Use describe, camera, pcb, electronics, ocr, find_errors, analyse_image, or video.",
            ok=False,
        )

    @staticmethod
    def _video_api_key(router: Any) -> str:
        """A Gemini key for on-frame model vision, if one is configured.

        Best-effort only — with no key the analyser falls back to OCR, so this
        never blocks a video analysis, it only enriches it.
        """
        if router is None:
            return ""
        try:
            settings = getattr(router, "settings", None)
            providers = getattr(settings, "providers", {}) if settings else {}
            for name, profile in providers.items():
                if "gemini" in str(name).lower() or "google" in str(name).lower():
                    key = getattr(profile, "api_key", "") or ""
                    if key and key.lower() not in {"local", "none", "no-key"}:
                        return key
        except Exception:
            pass
        return ""

    async def screen_read_tool(self, args: dict[str, Any]) -> ToolResult:
        """Shoulder (CAP-03) — read the active screen ON REQUEST and say what to
        DO about it for the prompt. Distinct from vision 'describe', which only
        reports what's there: this understands the screen (model vision, OCR
        fallback) and then decides the next action. Captures a single still,
        never stores it, never runs passively."""
        from .screen_read import ScreenReader, capture_available
        action = str(args.get("action") or "read").strip().lower()
        if action in {"status", "available"}:
            return ToolResult(
                "Screen reading is available." if capture_available()
                else "Screen reading needs a capture backend — install 'mss' "
                     "or 'pillow'.", ok=capture_available())
        prompt = str(args.get("prompt") or args.get("question") or args.get("text") or "")

        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        api_key = self._video_api_key(router)

        async def _describe(image: bytes, ask: str) -> tuple[str, str]:
            from .video_analysis import VideoAnalyser
            analyser = VideoAnalyser(bus=getattr(self, "bus", None), router=router,
                                     vision=self.vision, api_key=api_key)
            return await analyser._describe_one(image, ask)

        async def _generate(ask: str) -> str:
            if router is None:
                return ""
            _profile, text = await router.generate_text(ask, task="screen_read")
            return str(text or "")

        bus = getattr(self, "bus", None)
        reader = ScreenReader(
            describe=_describe,
            generate=_generate if router is not None else None,
            log=(bus.log.emit if bus is not None else None))
        reading = await reader.read(prompt)
        return ToolResult(reading.describe(), ok=reading.ok)

    async def _analyse_video(self, path: str, prompt: str,
                             args: dict[str, Any]) -> ToolResult:
        """Watch a video: describe its frames, transcribe its voice, and reason
        about both against the prompt (see video_analysis.py)."""
        from .video_analysis import DEFAULT_FRAMES, VideoAnalyser

        try:
            frames = int(args.get("frames") or args.get("max_frames") or DEFAULT_FRAMES)
        except (TypeError, ValueError):
            frames = DEFAULT_FRAMES
        # The router and offline transcriber both live behind AIModeInfo
        # (self.ai); fall back to any directly-attached handles for a bare
        # dispatcher in tests.
        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        transcriber = (getattr(ai, "offline_stt", None)
                       or getattr(self, "offline_stt", None)
                       or getattr(self, "transcriber", None))
        analyser = VideoAnalyser(
            bus=getattr(self, "bus", None),
            router=router,
            vision=self.vision,
            transcriber=transcriber,
            api_key=self._video_api_key(router),
        )
        report = await analyser.analyse(path, prompt=prompt, max_frames=frames)
        del router, transcriber          # released; nothing below needs them
        body = [report.announcement()]
        if report.summary:
            body.append("\nSummary:\n" + report.summary)
        if report.action:
            body.append("\nWhat I'd do:\n" + report.action)
        if report.transcript:
            body.append("\nTranscript:\n" + report.transcript[:1500])
        if report.notes:
            body.append("\n(" + "; ".join(report.notes) + ")")
        # Announce completion out loud, since watching a long video is the kind
        # of task that finishes while the user's attention is elsewhere.
        try:
            self.bus.speak_request.emit(report.announcement())
        except Exception:
            pass
        return ToolResult("\n".join(body))

    async def _watch_screen(self, prompt: str, args: dict[str, Any], *,
                            window_only: bool) -> ToolResult:
        """Watch what is playing on screen (see VideoAnalyser.watch_screen)."""
        from .video_analysis import DEFAULT_FRAMES, VideoAnalyser
        try:
            seconds = float(args.get("seconds") or 30)
        except (TypeError, ValueError):
            seconds = 30.0
        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        analyser = VideoAnalyser(bus=getattr(self, "bus", None), router=router,
                                 vision=self.vision,
                                 transcriber=getattr(ai, "offline_stt", None),
                                 api_key=self._video_api_key(router))
        report = await analyser.watch_screen(seconds, prompt=prompt,
                                             max_frames=DEFAULT_FRAMES,
                                             window_only=window_only)
        body = [f"I watched {report.path} for {report.duration:.0f} seconds "
                f"({len(report.frames)} distinct moments)."]
        if report.summary:
            body.append("\nSummary:\n" + report.summary)
        if report.action:
            body.append("\nWhat I'd do:\n" + report.action)
        if report.transcript:
            body.append("\nWhat was said:\n" + report.transcript[:1500])
        if report.notes:
            body.append("\n(" + "; ".join(report.notes) + ")")
        return ToolResult("\n".join(body), ok=bool(report.frames))

    async def capture_screen(self, args: dict[str, Any]) -> ToolResult:
        quality      = int(args.get("quality") or 78)
        max_side     = int(args.get("max_side") or 1024)
        # mss grab + PIL encode block for tens of milliseconds; keep the GUI
        # event loop clear by offloading to a worker thread.
        image_bytes  = await asyncio.to_thread(
            self.grabber.capture_jpeg, max_side=max_side, quality=quality
        )
        from .vision import SCREEN_FRAME_NOTE

        return ToolResult(
            f"Captured primary monitor in volatile memory: "
            f"{len(image_bytes)} JPEG bytes.\n" + SCREEN_FRAME_NOTE,
            media={"data": image_bytes, "mime_type": "image/jpeg"},
        )

    async def vision_verify(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "elements").lower().strip()
        if action in {"elements", "ui"}:
            return await self.vision.detect_elements(str(args.get("kinds") or "all"))
        if action in {"dialogs", "popups"}:
            return await self.vision.detect_dialogs()
        return await self.vision_analyse(args)

    def display_info(self, args: dict[str, Any]) -> ToolResult:
        if self.display is None:
            return ToolResult("Display topology manager is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action in {"set_name", "rename", "name"}:
            device = str(args.get("device") or args.get("monitor") or "")
            label = str(args.get("label") or args.get("name") or "")
            return self.display.set_custom_name(device, label)
        if bool(args.get("refresh")):
            self.display.refresh()
        return ToolResult(self.display.summary())



def _record_pcm16(seconds: float) -> bytes:
    """Record from the default input directly (16 kHz mono int16)."""
    try:
        import sounddevice as sd
        frames = int(seconds * 16000)
        data = sd.rec(frames, samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        return data.tobytes()
    except Exception:
        return b""


def _transcribe_float(wave: Any) -> str:
    """Speech in a clip through the shared offline transcriber (or "")."""
    try:
        import numpy as np

        from . import speech_offline
        pcm = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return (speech_offline.shared().transcribe_pcm(pcm, 16000) or "").strip()
    except Exception:
        return ""
