"""
Dispatch domain — Screen capture, visual analysis and verified on-screen interaction.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import io
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


#: The everyday camera, by voice (camera_tools.py).
_CAMERA_TOOL_ACTIONS = {
    "photo": "photo", "take_photo": "photo", "picture": "photo", "take_picture": "photo",
    "snapshot": "photo", "selfie": "photo", "save_photo": "photo",
    "document": "document", "scan_document": "document", "scan_page": "document",
    "scan_paper": "document", "scan_letter": "document",
    "read_code": "codes", "qr": "codes", "qr_code": "codes", "barcode": "codes",
    "scan_code": "codes", "read_qr": "codes", "read_barcode": "codes", "codes": "codes",
    "record": "record", "record_video": "record", "record_clip": "record",
    "video_clip": "record", "film": "record",
}


class VisionDispatchMixin:
    """Vision tools, including the continuous perception loop (Track E)."""

    @staticmethod
    def _screen_monitor(args: dict[str, Any]) -> int | None:
        """Tool display indexes are zero-based, matching display_info.

        mss indexes individual displays from one and reserves zero for the
        whole desktop. Omission means the whole desktop, not display zero.
        """
        value = args.get("monitor")
        if value in (None, ""):
            return None
        if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value).strip()):
            raise ValueError("Use a non-negative monitor index from display_info.")
        return int(value) + 1

    def _capture_monitor(self, monitor: int | None) -> int | None:
        """Resolve the tool's display index on the same worker as its capture."""
        if monitor is None:
            return None
        display = getattr(self, "display", None)
        grabber = getattr(self, "grabber", None) or getattr(getattr(self, "vision", None), "grabber", None)
        resolver = getattr(grabber, "monitor_index_for_bounds", None)
        if display is None or not callable(resolver):
            return monitor
        selected = display.monitor_by_index(monitor - 1)
        if selected is None:
            raise ValueError("That monitor is not connected. Use display_info to list the displays.")
        return resolver((selected.x, selected.y, selected.width, selected.height))

    async def perception_tool(self, args: dict[str, Any]) -> ToolResult:
        """Continuous local watching, with separately consented cloud sharing.

        Distinct from ``vision_analyse``, which is one paid look on demand:
        Automatic cloud calls need a bounded, session-only grant.
        """
        action = str(args.get("action") or "status").strip().lower().replace(" ", "_")
        memory_action = {"recall": "recall", "memory": "recall", "remembered": "recall",
                         "compare": "compare", "forget": "forget", "forget_memory": "forget"}.get(action)
        if memory_action is not None:
            return await self._camera_memory(memory_action, args)
        if self.perception is None:
            return ToolResult(
                "The perception loop is not available; use vision_analyse for a "
                "single look instead.", ok=False)
        if action in {"start", "watch", "on"}:
            if isinstance(args.get("retain_frames"), bool):
                self.perception.set_frame_history(args["retain_frames"])
            message = self.perception.start()
            return ToolResult(message, ok=self.perception.running)
        if action in {"stop", "off"}:
            return ToolResult(await self.perception.stop())
        if action == "cloud_off":
            return ToolResult(self.perception.set_cloud_sharing(False))
        if action in {"history_on", "history_off"}:
            return ToolResult(self.perception.set_frame_history(action == "history_on"))
        if action == "cloud_on":
            if args.get("consent") is not True:
                return ToolResult("Automatic camera sharing needs your explicit consent "
                                  "to send frames to the configured cloud provider.", ok=False)
            try:
                message = self.perception.set_cloud_sharing(
                    True, max_calls=args.get("max_calls", 20))
            except ValueError as exc:
                return ToolResult(str(exc), ok=False)
            return ToolResult(message)
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
            "remove_rule, enable_rule, disable_rule, autostart, cloud_on, cloud_off or overlay.", ok=False)

    async def _camera_memory(self, action: str, args: dict[str, Any]) -> ToolResult:
        """Recall/compare/forget written scenes; none of these opens a camera."""
        from . import scene_analysis
        recorder = getattr(self, "scene_recorder", None)
        if recorder is None:
            recorder = getattr(getattr(self, "vision", None), "scene_recorder", None)
        memory = getattr(recorder, "memory", None) or getattr(self, "memory", None)
        if memory is None:
            return ToolResult("Camera memory is not available here.", ok=False)
        when = str(args.get("when") or args.get("from") or "")
        until = str(args.get("until") or "")
        if action == "forget":
            # Invalidate pending looks before removing records, so a delayed
            # camera analysis cannot re-create what the user just forgot.
            removed, message = (await recorder.forget(when) if recorder is not None else
                                await asyncio.to_thread(scene_analysis.forget, memory, when))
            if scene_analysis.time_window(when)[3]:
                perception = getattr(self, "perception", None)
                if perception is not None:
                    perception.clear_scene_memories()
            return ToolResult(message, ok=scene_analysis.time_window(when)[3],
                              evidence=[{"forgotten_camera_memories": removed}])
        if recorder is not None:
            await recorder.wait_pending()
        if action == "compare":
            text = await asyncio.to_thread(scene_analysis.compare, memory, when=when, until=until)
        else:
            try:
                limit = max(1, min(10, int(args.get("limit") or 4)))
            except (TypeError, ValueError):
                limit = 4
            query = str(args.get("query") or args.get("prompt") or args.get("question") or "")
            text = await asyncio.to_thread(scene_analysis.recall, memory, query,
                                           when=when, until=until, limit=limit)
        return ToolResult(text)

    async def _remember_camera_text(self, description: str, *, source: str,
                                    prompt: str = "", details: Any = ()) -> None:
        recorder = (getattr(self, "scene_recorder", None) or
                    getattr(getattr(self, "vision", None), "scene_recorder", None))
        if recorder is None:
            return
        try:
            await recorder.remember_text(description, source=source, prompt=prompt, details=details)
        except Exception:
            pass  # A memory failure must not discard a completed camera report.

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
        watcher.context_service = getattr(self, "context_service", None)
        if action in {"watch_off", "stop_watch"}:
            await asyncio.to_thread(watcher.stop)
            return ToolResult("Stopped listening for alarms and the door.")
        if action == "watch_status":
            return ToolResult(watcher.describe())
        ok, message = await asyncio.to_thread(watcher.start)
        return ToolResult(("Now " if ok and "already" not in message else "") + message
                          if ok else f"I can't watch for sounds: {message}", ok=ok)

    async def _camera_tool(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Photo, document scan, code reading and clip recording through the
        camera - local, saved to the user's own Pictures / Videos."""
        from . import camera_tools
        vision = getattr(self, "vision", None)
        if vision is None or not hasattr(vision, "grab_frame"):
            return ToolResult("The camera is not available here.", ok=False)
        if tool == "record":
            source = getattr(vision, "_live_frame_source", None)
            owner = getattr(source, "__self__", None)
            capturing = getattr(owner, "is_capturing", None)
            if source is None or not callable(capturing) or not capturing():
                return ToolResult("Turn the camera on first - say 'show me the camera' - "
                                  "and then ask me to record.", ok=False)
            recorder = getattr(self, "_clip_recorder", None)
            if recorder is None:
                recorder = self._clip_recorder = camera_tools.ClipRecorder()
            if recorder.recording:
                recorder.stop()
                return ToolResult("Stopping the recording now.")
            try:
                seconds = float(args.get("seconds") or args.get("duration") or 10)
            except (TypeError, ValueError):
                seconds = 10.0
            seconds = max(1.0, min(seconds, camera_tools.ClipRecorder.MAX_SECONDS))
            bus = self.bus

            def done(path: Any, message: str) -> None:
                where = f" Saved as {path.name} in {path.parent}." if path else ""
                bus.log.emit(f"VISION: {message}{where}")
            recorder.start(source, seconds=seconds, fps=15, on_done=done)
            return ToolResult(f"Recording {seconds:.0f} seconds from the camera to "
                              f"Videos\\ORION. Ask me to stop to end it early.")
        index = args.get("camera_index", args.get("camera"))
        try:
            index = int(index) if index is not None else None
        except (TypeError, ValueError):
            index = None
        if tool == "photo":
            # Self-timer: "take a photo in five seconds". Said aloud so the
            # person knows when to look up; capped so a misheard number can't
            # leave the camera waiting for minutes.
            try:
                delay = max(0.0, min(float(args.get("delay") or 0), 15.0))
            except (TypeError, ValueError):
                delay = 0.0
            if delay:
                bus = getattr(self, "bus", None)
                if bus is not None:
                    count = round(delay)
                    bus.speak_request.emit(f"Photo in {count} second{'s' if count != 1 else ''}. Smile.")
                await asyncio.sleep(delay)
        frame, outcome = await vision.grab_frame(index)
        if frame is None:
            return outcome if not outcome.ok else ToolResult("The camera gave no frame.", ok=False)
        if tool == "photo":
            note = ""
            if camera_tools.mean_brightness(frame) < 60:
                frame = camera_tools.enhance_low_light(frame)
                note = " It was dark, so I brightened it."
            path = await asyncio.to_thread(camera_tools.save_image, frame, "photo")
            if path is None:
                return ToolResult("The photo could not be saved.", ok=False)
            return ToolResult(f"Photo taken and saved to {path.parent} as {path.name}.{note}")
        if tool == "codes":
            codes = await asyncio.to_thread(camera_tools.read_codes, frame)
            if not codes:
                return ToolResult("I can't read a QR code or barcode in view. Hold it flat, "
                                  "closer to the camera and in focus.", ok=False)
            lines = []
            for code in codes[:6]:
                if code.meaning == "wifi":
                    lines.append(code.summary)           # the password is never spoken
                elif code.meaning in {"link", "email", "phone"}:
                    lines.append(f"{code.summary}: {code.text}")
                else:
                    lines.append(f"{code.summary}" if code.summary == code.text
                                 else f"{code.summary}: {code.text}")
            text = ("Read " + ("one code" if len(codes) == 1 else f"{len(codes)} codes")
                    + ":\n" + "\n".join(f"- {line}" for line in lines))
            await self._remember_camera_text(text, source="camera_codes")
            return ToolResult(text)
        # document
        scan, found = await asyncio.to_thread(camera_tools.scan_document, frame)
        if scan is None:
            return ToolResult("The page could not be scanned.", ok=False)
        path = await asyncio.to_thread(camera_tools.save_image, scan, "scan")
        engine = getattr(vision, "ocr_engine", None)
        text = ""
        if engine is not None and hasattr(engine, "image_to_text"):
            try:
                from PIL import Image
                text = await asyncio.to_thread(engine.image_to_text, Image.fromarray(scan))
            except Exception:
                text = ""
        head = ("Scanned the page" if found else "No page edges were visible, so I scanned the whole view")
        head += f" and saved it to {path.parent} as {path.name}." if path else "."
        if not text.strip():
            return ToolResult(head + " I couldn't read any text on it.")
        result_text = f"{head} Its text:\n{text.strip()[:4000]}"
        document_text = text.strip()[:4000]
        await self._remember_camera_text(
            head, source="camera_document",
            details=[document_text[index:index + 1000] for index in range(0, len(document_text), 1000)])
        return ToolResult(result_text)

    async def vision_analyse(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "describe").lower().strip()
        memory_action = {"recall_camera": "recall", "camera_memory": "recall",
                         "scene_memory": "recall", "recall_scene": "recall",
                         "compare_camera": "compare", "compare_scenes": "compare",
                         "forget_camera": "forget", "forget_scenes": "forget"}.get(action.replace(" ", "_"))
        if memory_action is not None:
            return await self._camera_memory(memory_action, args)
        tool = _CAMERA_TOOL_ACTIONS.get(action.replace(" ", "_"))
        if tool is not None:
            return await self._camera_tool(tool, args)
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
            try:
                monitor = self._screen_monitor(args)
                if monitor is None:
                    return await self.vision.analyse_screen(prompt)
                sync = getattr(self.vision, "_analyse_screen_sync", None)
                if callable(sync):
                    return await asyncio.to_thread(lambda: sync(prompt, self._capture_monitor(monitor)))
                return await self.vision.analyse_screen(prompt, monitor=monitor)
            except ValueError as exc:
                return ToolResult(str(exc), ok=False)
        if action in {"read_window", "exact_text", "read_screen_text", "window_text",
                      "what_am_i_typing", "read_app", "read_page"}:
            window = str(args.get("window") or args.get("app") or args.get("target") or "")
            return await self.vision.read_window(window)
        if action in {"watch_camera", "watch_me", "watch_webcam"}:
            return await self._watch_camera(prompt, args)
        if action in {"watch_camera_past", "what_just_happened_camera"}:
            return await self._watch_camera_past(prompt, args)
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
            # A named place helps later recall ("what was on the office desk?").
            place = str(args.get("place") or "").strip()
            recorder = getattr(self, "scene_recorder", None)
            if place and recorder is not None and recorder.enabled:
                return await self.vision.analyse_camera(prompt, camera_index, place=place)
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
            "Use describe, camera, watch_camera, pcb, electronics, ocr, "
            "find_errors, analyse_image, or video.",
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
        from .screen_read import ScreenReader, capture_available, default_capture
        action = str(args.get("action") or "read").strip().lower()
        if action in {"status", "available"}:
            return ToolResult(
                "Screen reading is available." if capture_available()
                else "Screen reading needs a capture backend — install 'mss' "
                     "or 'pillow'.", ok=capture_available())
        prompt = str(args.get("prompt") or args.get("question") or args.get("text") or "")
        try:
            monitor = self._screen_monitor(args)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)

        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        grabber = getattr(self, "grabber", None) or getattr(self.vision, "grabber", None)
        native: dict[str, Any] = {}

        def _capture() -> bytes:
            # Keep the original pixels for local OCR. A two-monitor panorama
            # reduced to 1024 pixels made ordinary desktop text unreadable.
            if grabber is None:
                if monitor is not None:
                    raise ValueError("Monitor selection needs the desktop capture backend.")
                return default_capture()
            kwargs = {"max_side": 0}
            if monitor is not None:
                kwargs["monitor"] = self._capture_monitor(monitor)
            image = grabber.capture_image(**kwargs).convert("RGB")
            native["image"] = image
            outbound = image.copy()
            from .utils import PIL_RESAMPLE
            outbound.thumbnail((2560, 2560), PIL_RESAMPLE)
            buffer = io.BytesIO()
            outbound.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()

        async def _describe(image: bytes, ask: str) -> tuple[str, str]:
            from .vision import SCREEN_FRAME_NOTE
            from .vision_describe import describe_image
            request = (SCREEN_FRAME_NOTE + "\nRead the current desktop screen: " +
                       (ask or "Describe the app, relevant visible text, errors and controls."))
            ok, text = await describe_image(router, image, request, max_tokens=1200)
            if ok:
                return text, "model"
            exact = getattr(self.vision, "_exact_window_text", None)
            if monitor is None and callable(exact):
                try:
                    text = await asyncio.to_thread(exact, max_chars=12000)
                except Exception:
                    text = ""
                if text:
                    return str(text), "uia"
            ocr = getattr(self.vision, "_ocr_image_text", None)
            if callable(ocr):
                def _read() -> str:
                    from PIL import Image
                    frame = native.get("image")
                    if frame is None:
                        with Image.open(io.BytesIO(image)) as decoded:
                            frame = decoded.convert("RGB")
                    return str(ocr(frame) or "")[:12000]
                return await asyncio.to_thread(_read), "ocr"
            return "", "none"

        async def _generate(ask: str) -> str:
            if router is None:
                return ""
            _profile, text = await router.generate_text(ask, task="screen_read")
            return str(text or "")

        bus = getattr(self, "bus", None)
        reader = ScreenReader(
            capture=_capture,
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
        ok = bool(report.frames or report.transcript)
        return ToolResult("\n".join(body), ok=ok,
                          receipt={"analysis_state": "completed" if ok else "failed",
                                   "frame_count": len(report.frames),
                                   "audio_state": report.audio_state,
                                   "audio_note": report.audio_note})

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

    async def _watch_camera(self, prompt: str, args: dict[str, Any]) -> ToolResult:
        """Watch an explicitly requested movement through the selected webcam."""
        from .video_analysis import VideoAnalyser

        try:
            seconds = float(args.get("seconds") or 6.0)
        except (TypeError, ValueError):
            seconds = 6.0
        try:
            frames = int(args.get("frames") or 6)
        except (TypeError, ValueError):
            frames = 6
        raw_index = args.get("camera_index", args.get("camera"))
        try:
            camera_index = int(raw_index) if raw_index not in (None, "") else None
        except (TypeError, ValueError):
            camera_index = None
        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        analyser = VideoAnalyser(bus=getattr(self, "bus", None), router=router,
                                 vision=self.vision,
                                 api_key=self._video_api_key(router))
        report = await analyser.watch_camera(seconds, prompt=prompt,
                                             max_frames=frames,
                                             camera_index=camera_index)
        if not report.frames:
            return ToolResult("I could not read the camera sequence. "
                              + "; ".join(report.notes), ok=False)
        body = [f"I watched the camera for {report.duration:.0f} seconds "
                f"({report.captured_frames or len(report.frames)} moments)."]
        if report.summary:
            body.append("\nWhat changed:\n" + report.summary)
        if report.action:
            body.append("\nWhat I'd do:\n" + report.action)
        if report.notes:
            body.append("\n(" + "; ".join(report.notes) + ")")
        await self._remember_camera_text(
            report.summary or "\n".join(body), source="camera_watch", prompt=prompt,
            details=[f"At {item.at:.1f} seconds: {item.text}" for item in report.frames])
        return ToolResult("\n".join(body))

    async def _watch_camera_past(self, prompt: str, args: dict[str, Any]) -> ToolResult:
        """Explain recent frames retained by an already active local watch."""
        from .video_analysis import VideoAnalyser

        perception = getattr(self, "perception", None)
        if perception is None or not perception.running:
            return ToolResult("There is no recent camera view. Start local camera watching first.",
                              ok=False)
        if not getattr(perception, "retain_camera_history", False):
            return ToolResult("Recent camera frames are off. Enable recent camera frames "
                              "while watching, or ask me to recall my written camera memories.", ok=False)
        try:
            seconds = max(2.0, min(15.0, float(args.get("seconds") or 10.0)))
            count = max(2, min(8, int(args.get("frames") or 8)))
        except (TypeError, ValueError):
            return ToolResult("Camera history needs valid seconds and frames.", ok=False)
        frames = perception.recent_camera_frames(seconds, count)
        if len(frames) < 2:
            return ToolResult("I need at least two recent camera moments to describe a change.",
                              ok=False)
        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        analyser = VideoAnalyser(bus=getattr(self, "bus", None), router=router,
                                 vision=self.vision,
                                 api_key=self._video_api_key(router))
        report = await analyser.analyse_recent_camera(frames, prompt)
        if not report.frames:
            return ToolResult("I could not describe the recent camera view. "
                              + "; ".join(report.notes), ok=False)
        text = report.summary or "\n".join(frame.text for frame in report.frames)
        await self._remember_camera_text(
            text, source="camera_recent", prompt=prompt,
            details=[f"At {item.at:.1f} seconds: {item.text}" for item in report.frames])
        return ToolResult(text)

    async def capture_screen(self, args: dict[str, Any]) -> ToolResult:
        try:
            quality = max(45, min(95, int(args.get("quality") or 90)))
            max_side = max(256, min(4096, int(args.get("max_side") or 2560)))
            monitor = self._screen_monitor(args)
        except (TypeError, ValueError, OverflowError):
            return ToolResult("Use numeric quality, max_side and a monitor index from display_info.", ok=False)
        # mss grab + PIL encode block for tens of milliseconds; keep the GUI
        # event loop clear by offloading to a worker thread.
        def _capture() -> bytes:
            kwargs = {"max_side": max_side, "quality": quality}
            if monitor is not None:
                kwargs["monitor"] = self._capture_monitor(monitor)
            return self.grabber.capture_jpeg(**kwargs)
        try:
            image_bytes = await asyncio.to_thread(_capture)
        except Exception as exc:
            return ToolResult(f"I couldn't capture the screen: {first_line(str(exc), 200)}", ok=False)
        from .vision import SCREEN_FRAME_NOTE

        return ToolResult(
            f"Captured {'all displays' if monitor is None else f'monitor {monitor - 1}'} in volatile memory: "
            f"{len(image_bytes)} JPEG bytes.\n" + SCREEN_FRAME_NOTE,
            media={"data": image_bytes, "mime_type": "image/jpeg"},
        )

    async def vision_verify(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "elements").lower().strip()
        if action in {"find", "locate"}:
            import json
            resolved = await self.vision.resolve_element(str(args.get("query") or args.get("text") or ""),
                str(args.get("kinds") or "all"), str(args.get("automation_id") or ""))
            return ToolResult(json.dumps(resolved, ensure_ascii=False), ok=resolved["status"] == "found")
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
        from .audio_devices import stream_guard
        frames = int(seconds * 16000)
        # Counted while open, so a PortAudio re-initialisation waits for it.
        with stream_guard():
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
