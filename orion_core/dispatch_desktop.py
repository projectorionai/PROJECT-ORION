"""
Dispatch domain — Desktop / OS control: apps, windows, cursor, clipboard, processes, peripherals, system actions.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import subprocess
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line

# What models actually send for the peripherals actions.
_PERIPHERAL_ALIASES = {
    "set_volume": "volume", "louder": "volume_up", "turn_up": "volume_up",
    "increase_volume": "volume_up", "raise_volume": "volume_up", "vol_up": "volume_up",
    "quieter": "volume_down", "turn_down": "volume_down", "decrease_volume": "volume_down",
    "lower_volume": "volume_down", "vol_down": "volume_down",
    "get_volume": "volume_status", "volume_level": "volume_status",
    "mute_toggle": "toggle_mute", "silence": "mute",
    "set_brightness": "brightness", "brighter": "brightness_up",
    "increase_brightness": "brightness_up", "dimmer": "brightness_down", "dim": "brightness_down",
    "decrease_brightness": "brightness_down", "get_brightness": "brightness_status",
}

# interface_control's face and deck actions, as models actually send them.
_EXPRESSION_ACTIONS = frozenset({"emotion", "expression", "show_emotion", "show_expression",
                                 "face_expression", "make_face", "emote"})
_TOUR_ACTIONS = frozenset({"emotion_tour", "expression_tour", "cycle_emotions",
                           "cycle_expressions", "show_emotions", "show_expressions",
                           "expressions", "emotions"})
_DECK_QUERIES = {
    "pages": "pages", "list_pages": "pages", "deck_pages": "pages",
    "read": "read", "read_page": "read", "describe": "read", "describe_page": "read",
    "what_is_on": "read",
    "find": "find", "where": "find", "locate": "find", "search_deck": "find",
    "find_on_deck": "find",
    "press": "press", "click": "press", "tap": "press", "push": "press",
    "press_button": "press", "click_button": "press",
}


#: What he says when the reply pause changes (Mark XXXX.1).
_PAUSE_REPLIES = {
    "brisk": "Brisk it is - I'll answer the moment you stop.",
    "natural": "A natural pause, then - I'll give you a breath before I answer.",
    "patient": "Understood. I'll let you finish your sentences before I say a word.",
}


class DesktopDispatchMixin:
    """Desktop / OS control: apps, windows, cursor, clipboard, processes, peripherals, system actions."""

    def _refresh_live_channel(self, reason: str) -> None:
        """Refresh the live channel once he is quiet (live_worker)."""
        refresh = getattr(getattr(self, "live_worker", None), "request_channel_refresh", None)
        if callable(refresh):
            try:
                refresh(reason)
            except Exception:
                pass

    async def presentation_tool(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Report success only after the GUI has acknowledged a real update."""
        # A card or a theme in his own windows (require_own_window).
        from .execution_policy import require_own_window
        blocked = require_own_window("gui_presentation")
        if blocked is not None:
            return blocked
        from .presentation import prepare
        from uuid import uuid4
        try:
            payload = await prepare(tool, args)
        except Exception as exc:
            return ToolResult(str(exc) if isinstance(exc, ValueError) else
                              "The display media could not be loaded.", ok=False)
        loop = asyncio.get_running_loop()
        result = loop.create_future()
        request_id = uuid4().hex
        def receive(receipt):
            if not isinstance(receipt, dict) or receipt.get("request_id") != request_id:
                return
            def deliver():
                if not result.done():
                    result.set_result(receipt)
            loop.call_soon_threadsafe(deliver)
        signal = getattr(self.bus, "gui_result", None)
        if signal is None:
            return ToolResult("The graphical display is unavailable.", ok=False)
        signal.connect(receive)
        try:
            self.bus.gui_command.emit({"action": "presentation", "request_id": request_id,
                                       "presentation": payload})
            receipt = await asyncio.wait_for(result, 3)
        except asyncio.TimeoutError:
            return ToolResult("The desktop display did not acknowledge the update.", ok=False)
        finally:
            try:
                signal.disconnect(receive)
            except (TypeError, RuntimeError, AttributeError):
                pass
        if not receipt.get("ok"):
            return ToolResult(str(receipt.get("error") or "The display update failed."), ok=False)
        return ToolResult("Displayed the card." if tool == "display" else "Updated the interface.",
                          evidence=[{"presentation": payload}])

    async def open_app(self, args: dict[str, Any]) -> ToolResult:
        # Off the event loop: an unmatched name now falls back to a document
        # search, and a search on the loop thread stalls the face and audio.
        requested = str(args.get("app_name") or args.get("name") or "")
        from .file_search import KINDS, OPENABLE_KINDS
        candidate = requested.strip()
        quoted = len(candidate) >= 2 and candidate[0] in {"'", '"'} and candidate[-1] == candidate[0]
        unquoted = candidate[1:-1] if quoted else candidate
        extension = Path(unquoted).suffix.lower().lstrip(".")
        # A supplied document filename is an explicit file intent. Apps such
        # as Word, File Explorer and PDF Viewer still use the app resolver;
        # URI schemes keep their existing handling there as well.
        document_file = (not re.match(r"^[a-z][a-z0-9+.-]+:", unquoted, re.IGNORECASE)
                         and extension in {ext for kind in OPENABLE_KINDS for ext in KINDS[kind]})
        spoken = re.sub(r"^(?:open|show|bring up)\s+", "", candidate, flags=re.IGNORECASE)
        explicit = re.fullmatch(
            r"(?:(?:my|the|a|an|this|that)\s+)?"
            r"(?:file|document|spreadsheet|presentation|pdf)\s+"
            r"(?:called|named|about|on|for)\s+(.+)", spoken, re.IGNORECASE)
        possessive = re.fullmatch(r"(?:my|our|the|this|that)\s+(.+)", spoken, re.IGNORECASE)
        document_words = r"cv|résumé|resume|essay|report|proposal|invoice|thesis|assignment|document|spreadsheet|presentation|pdf"
        spoken_document = possessive is not None and (
            re.fullmatch(rf"(?:{document_words})(?:\s+\d{{4}})?", possessive.group(1), re.IGNORECASE)
            or re.search(rf"\s+(?:{document_words})$", possessive.group(1), re.IGNORECASE))
        references = getattr(self, "_recent_file_references", None)
        is_reference = (references is not None and references.resolve(candidate)[0]) or bool(
            re.fullmatch(r"(?:this|that|the)\s+file", candidate, re.IGNORECASE))
        if re.search(r"\b(?:folder|directory)\b", requested, re.IGNORECASE) and not document_file and not explicit:
            # A filename can itself contain "folder"; otherwise preserve the
            # established route for an explicitly requested directory.
            return await self.find_files({"query": requested, "type": "folder", "action": "open"})
        if document_file or explicit or spoken_document or is_reference:
            query = candidate
            if not quoted and not is_reference:
                query = (explicit.group(1) if explicit else possessive.group(1)
                         if possessive else spoken)
            return await self.find_files({"query": query, "type": "file", "action": "open"})
        return await asyncio.to_thread(
            self.desktop.open_app, requested,
            str(args.get("remember_as") or ""))

    async def close_app(self, args: dict[str, Any]) -> ToolResult:
        # Off the event loop: a polite close waits for the windows to go.
        return await asyncio.to_thread(
            self.desktop.close_app, str(args.get("app_name") or args.get("name") or ""))

    async def window_control(self, args: dict[str, Any]) -> ToolResult:
        # DesktopAgent uses native OS APIs only; GUI/clipboard operations keep
        # their existing GUI-thread routes. Native enumeration and readback can
        # wait for another application's window, so keep the live loop free.
        return await asyncio.to_thread(self.desktop.window_control,
            str(args.get("action") or "list"), str(args.get("title") or ""),
            str(args.get("screen") or args.get("monitor") or ""),
        )

    async def media_control(self, args: dict[str, Any]) -> ToolResult:
        # Off the event loop: the media-session calls and "open the music app,
        # then play" both wait.
        query = " ".join(str(args.get(key) or "").strip() for key in ("query", "song", "track", "title")
                         if args.get(key)) or str(args.get("name") or "").strip()
        artist = str(args.get("artist") or "").strip()
        if artist and artist.lower() not in query.lower():
            query = f"{query} by {artist}" if query else artist
        service = str(args.get("service") or args.get("app") or args.get("platform") or "").strip()
        return await asyncio.to_thread(
            self.desktop.media_control,
            str(args.get("action") or ("play" if query else "play_pause")),
            int(args.get("steps") or 2), query, service)

    async def desktop_control(self, args: dict[str, Any]) -> ToolResult:
        """
        Cursor/keyboard/window control with optional visual verification.
        Every mutating action runs off the event loop (to_thread) and, when
        'verify' is set (default for clicks), is confirmed by vision.
        """
        if self.control is None:
            return ToolResult("Autonomous control layer is not available.", ok=False)
        # The caption beside the pointer shows this reason for every step of
        # the call (showing his working); cleared however the call ends.
        self.control.intent = " ".join(str(args.get("why") or "").split())[:90]
        from .human_pointer import speed_scope
        # speed: human (curved, eased pointer, natural clicks and keys - the
        # default) or fast. It reaches every move and click this call makes,
        # the verifier's click_text included, through a context variable.
        try:
            with speed_scope(_requested_speed(args)):
                return await self._desktop_control(args)
        finally:
            self.control.intent = ""

    async def _desktop_control(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").lower().strip()
        verify = bool(args.get("verify", True)) and self.verifier is not None
        coordinates = {
            "move_cursor": ("x", "y"),
            "move_window": ("x", "y"),
            "drag": ("x1", "y1", "x2", "y2"),
        }.get(action, ())
        if any(args.get(key) is None for key in coordinates):
            return ToolResult(f"{action} requires coordinates: {', '.join(coordinates)}.", ok=False)

        def _run(fn: Callable[..., ToolResult], *a: Any, **k: Any) -> Any:
            return asyncio.to_thread(fn, *a, **k)

        # Element-targeted click using coordinates from the current view.
        if action in {"locate", "find_target"}:
            vision = getattr(self, "vision", None)
            if vision is None or not hasattr(vision, "resolve_element"):
                return ToolResult("Desktop target inspection is unavailable.", ok=False)
            result = await vision.resolve_element(str(args.get("text") or args.get("query") or ""),
                str(args.get("kinds") or "all"), str(args.get("automation_id") or ""))
            return ToolResult(json.dumps(result, ensure_ascii=False), ok=result["status"] == "found")
        if action in {"click_text", "click_element"}:
            if self.verifier is None:
                return ToolResult("Verification engine unavailable for element clicks.", ok=False)
            query = str(args.get("text") or args.get("query") or "")
            selector = str(args.get("automation_id") or "")
            kinds = str(args.get("kinds") or "all")
            if kinds != "all" or action == "click_element":
                return await self.verifier.click_element(query, kinds=kinds, automation_id=selector)
            if selector:
                return await self.verifier.click_text(query, automation_id=selector)
            return await self.verifier.click_text(query)
        if action == "move_cursor":
            return await _run(self.control.move_cursor, int(args.get("x", 0)), int(args.get("y", 0)),
                              args.get("monitor"))
        if action in {"click", "left_click"}:
            call = lambda: self.control.click(args.get("x"), args.get("y"), monitor=args.get("monitor"))
            return await self._maybe_verify(call, verify, region=_point_region(self.control, args),
                                            repeatable=False)
        if action == "double_click":
            call = lambda: self.control.double_click(args.get("x"), args.get("y"), monitor=args.get("monitor"))
            return await self._maybe_verify(call, verify, region=_point_region(self.control, args),
                                            repeatable=False)
        if action == "right_click":
            call = lambda: self.control.right_click(args.get("x"), args.get("y"), args.get("monitor"))
            return await self._maybe_verify(call, verify, region=_point_region(self.control, args),
                                            repeatable=False)
        if action == "drag":
            call = lambda: self.control.drag_cursor(
                int(args.get("x1", 0)), int(args.get("y1", 0)),
                int(args.get("x2", 0)), int(args.get("y2", 0)), args.get("monitor"))
            return await self._maybe_verify(call, verify, repeatable=False)
        if action == "scroll":
            # No amount used to mean scroll(0): "Scrolled 0" and nothing moved.
            amount = int(args.get("amount") or 0)
            if not amount:
                direction = str(args.get("direction") or "down").lower()
                amount = 5 if direction in {"up", "top", "back"} else -5
            return await _run(self.control.scroll, amount,
                              args.get("x"), args.get("y"), args.get("monitor"))
        if action in {"smooth_scroll", "scroll_smooth", "glide_scroll"}:
            return await _run(self.control.smooth_scroll, int(args.get("amount", -10)),
                              args.get("x"), args.get("y"), args.get("monitor"),
                              float(args.get("duration", 0.8)))
        speed = _requested_speed(args)
        if action in {"type_text", "type", "type_human", "human_type"}:
            # Visible, person-paced typing by default — the user watches ORION
            # type rather than seeing a block of text appear as if pasted.
            # fast=true / speed=fast (or ORION_VISIBLE_TYPING=0) restores
            # machine speed.
            from .human_typing import natural_wpm
            text_arg = str(args.get("text") or "")
            human = (action in {"type_human", "human_type"} or _flag(args.get("human"))
                     or speed == "human"
                     or (speed != "fast" and _visible_typing_default()))
            seed = args.get("seed")
            call = lambda: self.control.type_text(
                text_arg,
                float(args.get("interval", 0.01)),
                human=human,
                wpm=float(args.get("wpm") or natural_wpm(len(text_arg))),
                typos=float(args.get("typos", 0.0)),
                jitter=float(args.get("jitter", 0.35)),
                seed=int(seed) if seed is not None else None)
            return await self._maybe_verify(call, verify, region=_foreground_region(),
                                            repeatable=False)
        if action in {"edit_text", "write_text", "write_to", "set_text"}:
            return await _run(self.control.edit_text, str(args.get("text") or ""),
                              str(args.get("title") or args.get("window") or ""),
                              bool(args.get("replace")),
                              visible=speed == "human"
                              or (speed != "fast" and _visible_typing_default()))
        if action in {"hotkey", "send_hotkeys"}:
            return await _run(self.control.send_hotkeys, args.get("keys") or args.get("hotkey") or "")
        if action in {"open_app", "open_application"}:
            return await _run(self.control.open_application, str(args.get("app_name") or args.get("name") or ""))
        if action in {"close_app", "close_application"}:
            return await _run(self.control.close_application, str(args.get("app_name") or args.get("name") or ""))
        if action in {"focus_window", "focus", "switch"}:
            call = lambda: self.control.focus_window(str(args.get("title") or ""))
            return await self._maybe_verify(call, verify, repeatable=False)
        if action in {"resize_window", "resize"}:
            call = lambda: self.control.resize_window(
                str(args.get("title") or ""),
                int(args.get("width", 800)), int(args.get("height", 600)))
            return await self._maybe_verify(call, verify, repeatable=False)
        if action in {"move_window"}:
            call = lambda: self.control.move_window(
                str(args.get("title") or ""),
                int(args.get("x", 0)), int(args.get("y", 0)), args.get("monitor"))
            return await self._maybe_verify(call, verify, repeatable=False)
        if action in {"minimise_window", "minimize_window"}:
            call = lambda: self.control.minimise_window(str(args.get("title") or ""))
            return await self._maybe_verify(call, verify, repeatable=False)
        if action in {"maximise_window", "maximize_window"}:
            call = lambda: self.control.maximise_window(str(args.get("title") or ""))
            return await self._maybe_verify(call, verify, repeatable=False)
        if action in {"list_windows", "windows"}:
            return await _run(self.control.list_windows)
        return ToolResult(
            f"Unsupported desktop_control action: {action}. Use click, click_text, "
            "move_cursor, double_click, right_click, drag, scroll, type_text, "
            "edit_text (reliable write into Notepad/editors with a 'title'), hotkey, "
            "open_app, close_app, focus_window, resize_window, move_window, "
            "minimise_window, maximise_window, or list_windows.",
            ok=False,
        )

    async def _maybe_verify(self, call: Callable[[], ToolResult], verify: bool, *,
                            region: Any = None, repeatable: bool = True) -> ToolResult:
        if verify and self.verifier is not None:
            if region is None and repeatable:
                return (await self.verifier.verify_action(call)).to_tool_result()
            return (await self.verifier.verify_action(
                call, region=region, repeatable=repeatable)).to_tool_result()
        return await asyncio.to_thread(call)



    def clipboard_operate(self, args: dict[str, Any]) -> ToolResult:
        action    = str(args.get("action") or "read").lower().strip()
        if action in {"copy", "write", "set"}:
            from .execution_policy import require_foreground
            blocked = require_foreground("clipboard_write")
            if blocked is not None:
                return blocked
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return ToolResult("Clipboard unavailable.", ok=False)
        if action == "read":
            text = clipboard.text()
            return ToolResult(text if text else "Clipboard is empty.")
        if action in {"copy", "write", "set"}:
            text = SecuritySanitiser.guard_text(str(args.get("text") or ""), "clipboard.text")
            blocked = require_foreground("clipboard_write")
            if blocked is not None:
                return blocked
            clipboard.setText(text)
            return ToolResult("Clipboard updated.")
        return ToolResult(f"Unsupported clipboard action: {action}", ok=False)

    async def process_governor_tool(self, args: dict[str, Any]) -> ToolResult:
        """process_governor off the UI loop: top/list sample CPU for a moment,
        which on the qasync thread would freeze the interface."""
        action = str((args or {}).get("action") or "list").lower().strip()
        if action in {"top", "heavy", "hogs", "power", "high_power", "list", "inventory"}:
            return await asyncio.to_thread(self.process_governor, args)
        return self.process_governor(args)

    def process_governor(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        # ── the biggest power users (what the user asks ORION to police) ──────
        if action in {"top", "heavy", "hogs", "power", "high_power"}:
            from .resource_monitor import sample_processes
            metric = str(args.get("metric") or "cpu").lower()
            procs = [(p.pid, p.name, p.cpu_percent, p.mem_percent)
                     for p in sample_processes(interval=0.5)]
            key = 3 if metric in {"ram", "memory", "mem"} else 2
            procs.sort(key=lambda p: -p[key])
            lines = [f"Top processes by {'RAM' if key == 3 else 'CPU'} "
                     f"(CPU is a share of the whole machine, as in Task Manager):"]
            for pid, name, cpu, ram in procs[:12]:
                lines.append(f"  PID {pid:>6}  {name[:30]:<30}  CPU {cpu:5.1f}%  RAM {ram:4.1f}%")
            try:
                from . import gpu_stats
                g = gpu_stats.sample()
                if g.get("available"):
                    lines.append(f"GPU {g['name']}: {g['util']:.0f}% util, "
                                 f"{g['mem_percent']:.0f}% VRAM, {g['temp_c']}°C")
            except Exception:
                pass
            lines.append("Say 'terminate PID <n>' or 'close <name>' and confirm to stop one.")
            return ToolResult("\n".join(lines))
        if action in {"list", "inventory"}:
            from .resource_monitor import sample_processes
            sampled = sorted(sample_processes(interval=0.3),
                             key=lambda p: (p.cpu_percent, p.mem_percent), reverse=True)
            rows = [f"{p.pid:>6}  {p.name or 'unknown'}  CPU {p.cpu_percent:.1f}%  "
                    f"RAM {p.mem_percent:.1f}%" for p in sampled]
            return ToolResult("\n".join(rows[:80]))
        if action in {"terminate", "kill", "stop", "restart"}:
            pid   = args.get("pid")
            label = SecuritySanitiser.guard_text(
                str(args.get("name") or args.get("label") or ""), "process.name"
            )
            # Permission gate. This used to read "repeat the request with
            # confirm=true and I shall stop it", checking a parameter the MODEL
            # fills in — which is a convention, not a gate: nothing stopped the
            # model sending confirm=true on the first call, and the message
            # explained how. Killing a process cannot be undone and takes
            # unsaved work with it, so it now arms the same human-issued,
            # single-use, expiring token the power actions use. The target is
            # bound to that token, so the process the user approves on screen
            # is necessarily the process that dies.
            target = f"PID {pid}" if pid is not None else f"'{label}'"
            if pid is None and not label:
                return ToolResult("No process PID or label supplied.", ok=False)
            guard = getattr(self, "system_guard", None)
            if guard is None:
                return ToolResult(
                    "I cannot terminate anything without the confirmation "
                    "guard available.", ok=False)
            from .system_guard import ActionIntent as _Intent
            created_at = None
            if pid is not None:
                if isinstance(pid, bool) or not str(pid).strip().isdigit() or int(pid) <= 0:
                    return ToolResult("Supply a positive integer process PID.", ok=False)
                pid = int(pid)
                try:
                    proc = psutil.Process(pid)
                    created_at = proc.create_time()
                    target = f"PID {pid} ({proc.name()})"
                except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, OverflowError):
                    return ToolResult(f"Cannot inspect PID {pid}; no action was armed.", ok=False)
            from .execution_policy import require_foreground
            blocked = require_foreground("process_confirmation_dialog")
            if blocked is not None:
                return blocked
            decision = guard.request_confirmation(
                _Intent.PROCESS_KILL, origin="process_control",
                payload={"pid": pid, "label": label, "action": action,
                         "target": target, "created_at": created_at})
            if decision.token:
                self.bus.confirm_action.emit({
                    "token": decision.token,
                    "intent": "process_kill",
                    "message": f"Terminate {target}? Unsaved work in it will "
                               f"be lost.",
                })
            return ToolResult(
                f"Terminating {target} needs your approval — confirm the "
                f"on-screen prompt and I shall stop it.", ok=False)
        return ToolResult(
            f"Unsupported process action: {action}. Use top (biggest power "
            "users), list, terminate, or restart.",
            ok=False,
        )

    def system_notify(self, args: dict[str, Any]) -> ToolResult:
        message  = SecuritySanitiser.guard_text(
            str(args.get("message") or args.get("text") or "System event."), "notify.message"
        )
        priority = int(args.get("priority") or 1)
        self.bus.banner.emit(message, priority)
        self.bus.log.emit(f"ALERT: {message}")
        return ToolResult("System notification projected.")

    def shutdown_orion(self, args: dict[str, Any]) -> ToolResult:
        from .execution_policy import require_foreground
        blocked = require_foreground("orion_shutdown")
        if blocked is not None:
            return blocked
        self.bus.log.emit("SYS: shutdown directive accepted.")
        self.bus.request_shutdown.emit()
        return ToolResult("Orion shutdown initiated.")

    def restart_orion(self, args: dict[str, Any]) -> ToolResult:
        """Mark X.6: ORION restarts himself — a clean shutdown followed by the
        launcher respawning the process.  This is also how an applied
        self-repair patch gets loaded without the user touching anything."""
        from .execution_policy import require_foreground
        blocked = require_foreground("orion_restart")
        if blocked is not None:
            return blocked
        self.bus.log.emit("SYS: restart directive accepted — going down to come back up.")
        self.bus.banner.emit("SYSTEM RESTART — back in a moment", 2)
        self.bus.request_restart.emit()
        return ToolResult("Orion restart initiated — I'll be back with you in a moment.")

    # ── installing software, on the owner's click (software_install.py) ──────

    async def software_install_tool(self, args: dict[str, Any]) -> ToolResult:
        """search / install / status. Install only ever ASKS: the owner's click
        on the confirmation dialog is what installs."""
        from . import software_install as si
        from .execution_policy import require_foreground
        action = str(args.get("action") or "install").lower().strip()
        target = str(args.get("package") or args.get("name") or args.get("query")
                     or args.get("target") or "").strip()
        if si.winget() is None:
            return ToolResult("winget, Windows' package manager, isn't on this PC, so I can't "
                              "install software here.", ok=False)
        if action in {"status", "results", "last"}:
            done = si.recent_results()
            if not done:
                return ToolResult("Nothing has been installed through me this session.")
            return ToolResult("; ".join(f"{r['package']}: {'installed' if r['ok'] else 'failed'}"
                                        f" - {r['text']}" for r in done[-5:]))
        if not target:
            return ToolResult("Say which program, e.g. 'install VLC'.", ok=False)
        if action in {"search", "find", "look_up"}:
            found = await asyncio.to_thread(si.search, target)
            if not found:
                return ToolResult(f"Nothing in the winget repository matches '{target}'.", ok=False)
            return ToolResult("Found: " + "; ".join(p.line() for p in found)
                              + ". Say which one to install.")
        blocked = require_foreground("software_install")
        if blocked is not None:
            return blocked
        package, found = await asyncio.to_thread(si.resolve, target)
        if package is None:
            if not found:
                return ToolResult(f"Nothing in the winget repository matches '{target}' - "
                                  "nothing was installed.", ok=False)
            return ToolResult("Nothing was installed yet - more than one package matches: "
                              + "; ".join(p.line() for p in found)
                              + ". Ask the owner which one, then install it by its id.", ok=False)
        guard = getattr(self, "system_guard", None)
        if guard is None:
            return ToolResult("Installing needs your approval on screen, and that isn't "
                              "available right now - nothing was installed.", ok=False)
        from .system_guard import ActionIntent, DecisionKind
        decision = guard.request_confirmation(
            ActionIntent.SOFTWARE_INSTALL, origin="software_install",
            payload={"id": package.id, "name": package.name, "version": package.version})
        if decision.kind is DecisionKind.CONFIRM:
            self.bus.confirm_action.emit({
                "token": decision.token, "intent": "software_install",
                "message": (f"Install {package.name}?\n\nPackage: {package.id}\n"
                            f"Version: {package.version or 'latest'}\n"
                            "From: the winget community repository\n\n"
                            "Installing accepts the package's licence. Windows may also ask "
                            "for administrator permission.")})
        self.bus.log.emit(f"INSTALL: {package.id} waiting for the owner's approval.")
        return ToolResult(f"I've put {package.line()} on your screen to approve. Nothing is "
                          "installed until you click to confirm - do not say it is installing.",
                          ok=False)

    def _run_confirmed_install(self, payload: dict[str, Any]) -> ToolResult:
        """Approved by the owner's click: install in the background and report
        on the BRAIN page (INSTALL: lines) and the banner when it ends."""
        from . import background
        from . import software_install as si
        package_id = str((payload or {}).get("id") or "")
        name = str((payload or {}).get("name") or package_id)
        if not si.ID_RE.match(package_id):
            return ToolResult("The approved install lost its package.", ok=False)
        self.bus.log.emit(f"INSTALL: {package_id} started — {name}")

        async def run() -> None:
            ok, text = await asyncio.to_thread(si.install, package_id)
            si.note_result(name, ok, text)
            self.bus.log.emit(f"INSTALL: {package_id} {'finished' if ok else 'failed'} — {text}")
            self.bus.banner.emit(text, 2)

        try:
            background.spawn(run(), name=f"orion-install-{package_id}")
        except RuntimeError:
            return ToolResult("I couldn't start the install.", ok=False)
        return ToolResult(f"Installing {name} now - I'll say when it's done.")

    def confirm_system_action(self, token: str) -> ToolResult:
        """Deterministic release point for a guarded destructive action.

        Called by the GUI confirmation dialog (human approval) with the bound
        token — NOT by the model.  Verifies the token with the guard, then runs
        the corresponding raw executor with confirm=True."""
        from .system_guard import ActionIntent, DecisionKind
        decision = self.system_guard.confirm(str(token or ""))
        if decision.kind is not DecisionKind.EXECUTE:
            return ToolResult(decision.message, ok=False)
        self.bus.log.emit(f"SYS: releasing confirmed action {decision.intent.value}.")
        if decision.intent is ActionIntent.PROCESS_KILL:
            # Released by human approval; the target came back bound to the
            # token rather than being supplied again here.
            return self._run_confirmed_process_kill(decision.payload)
        if decision.intent is ActionIntent.MCP_CONFIGURE:
            return self._run_confirmed_mcp_configuration(decision.payload)
        if decision.intent is ActionIntent.SOFTWARE_INSTALL:
            return self._run_confirmed_install(decision.payload)
        if decision.intent is ActionIntent.MEMORY_FORGET and callable(
                getattr(self, "run_confirmed_forget", None)):
            return self.run_confirmed_forget(decision.payload)
        if decision.intent is ActionIntent.DATA_ERASE and callable(
                getattr(self, "run_confirmed_erase", None)):
            return self.run_confirmed_erase(decision.payload)
        if decision.intent in (ActionIntent.FILE_DELETE, ActionIntent.FILE_WRITE):
            return self._run_confirmed_file_action(decision.intent, decision.payload)
        if decision.intent is ActionIntent.SEND_MESSAGE and decision.payload.get("plugin"):
            from .dispatcher import _PluginApproval, _plugin_approval_digest, _plugin_blocked
            name = decision.payload.get("plugin")
            arguments = decision.payload.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                return _plugin_blocked("The approved plugin lost its bound payload.", "invalid_arguments")
            try:
                approval = _PluginApproval(self, name, _plugin_approval_digest(name, arguments),
                    source_sha256=str(decision.payload.get("plugin_source_sha256") or ""))
            except (TypeError, ValueError, OverflowError):
                return _plugin_blocked("The approved plugin payload is invalid.", "invalid_arguments")
            # Only this real token-consumption branch can mint the capability.
            return DesktopDispatchMixin._run_confirmed_plugin(self, decision.payload, _approval=approval)
        if decision.intent is ActionIntent.SEND_MESSAGE and decision.payload.get("channel"):
            return self._run_confirmed_channel(decision.payload)
        if decision.intent in (ActionIntent.PLACE_CALL,
                               ActionIntent.SEND_MESSAGE):
            return self._run_confirmed_call(decision.intent, decision.payload)
        if self.peripherals is None:
            return ToolResult("Peripheral control is unavailable.", ok=False)
        runners = {
            ActionIntent.OS_SHUTDOWN:  lambda: self.peripherals.shutdown(confirm=True),
            ActionIntent.OS_RESTART:   lambda: self.peripherals.restart(confirm=True),
            ActionIntent.OS_LOGOUT:    lambda: self.peripherals.logout(confirm=True),
            ActionIntent.OS_SLEEP:     lambda: self.peripherals.sleep(confirm=True),
            ActionIntent.OS_HIBERNATE: lambda: self.peripherals.hibernate(confirm=True),
        }
        runner = runners.get(decision.intent)
        if runner is None:
            return ToolResult(f"No executor for {decision.intent.value}.", ok=False)
        return runner()

    def _run_confirmed_call(self, intent: Any, payload: dict) -> ToolResult:
        """Place the call the user approved on screen.

        The number and the words come from the payload bound to the token,
        never from anything supplied afterwards — so what was approved is
        necessarily what happens.

        Spawned rather than awaited because this is reached from the GUI's
        confirmation handler, which is synchronous. tasks.spawn holds the
        reference: asyncio keeps only a weak one, and a dropped task here
        would mean the user approved a call that silently never happened.
        """
        if payload.get("plugin"):
            from .dispatcher import _plugin_blocked
            return _plugin_blocked("A plugin action needs its own consumed human approval token.", "requires_confirmation")
        if payload.get("server") and payload.get("tool"):
            # An approved MCP spend (mcp call / mcp__server__tool). It used to
            # fall through to the telephony branch, which wants to/message,
            # so every approval ended "The approved action lost its target."
            return self._run_confirmed_mcp(payload)
        gateway = getattr(self, "telephony", None)
        if gateway is None:
            return ToolResult("Telephony is not available.", ok=False)
        to = str(payload.get("to") or "")
        message = str(payload.get("message") or "")
        if not to or not message:
            return ToolResult("The approved action lost its target.", ok=False)

        from .system_guard import ActionIntent as _Intent
        from .tasks import spawn

        texting = intent is _Intent.SEND_MESSAGE
        coro = (gateway.text(to, message) if texting
                else gateway.call(to, message, reason="approved on screen"))
        if spawn(coro, bus=self.bus,
                 label=f"{'texting' if texting else 'calling'} {to}") is None:
            return ToolResult(
                "There is no running event loop to place the call on.",
                ok=False)
        return ToolResult(f"{'Texting' if texting else 'Calling'} {to} now.")

    def _run_confirmed_channel(self, payload: dict) -> ToolResult:
        """Send the one message the owner approved on screen, on one of ORION's
        own channels (message_channels) - recipient and words from the token's
        bound payload only. Off the GUI thread: the approval handler is
        synchronous and an HTTP send can take seconds."""
        from . import message_channels
        from .tasks import spawn
        kind = str(payload.get("channel") or "")
        to = str(payload.get("to") or "")
        message = str(payload.get("message") or "")
        options = {"title": str(payload.get("title") or ""),
                   "priority": str(payload.get("priority") or "")}
        if not message_channels.channel(kind) or not message.strip():
            return ToolResult("The approved message lost its channel or its words.", ok=False)

        async def _deliver() -> None:
            result = await asyncio.to_thread(message_channels.send, kind, to, message, **options)
            try:
                self.bus.log.emit(f"MESSAGING: {result.text}")
                if not result.ok:
                    self.bus.speak_request.emit(result.text[:200])
            except RuntimeError:
                pass                                      # Qt shutting down

        if spawn(_deliver(), bus=self.bus, label=f"sending on {kind}") is None:
            return message_channels.send(kind, to, message, **options)
        return ToolResult(f"Sending on {kind.title()} now.")

    def _run_confirmed_plugin(self, payload: dict, *, _approval: Any = None) -> ToolResult:
        """Queue a token-bound call through the ordinary owned plugin executor."""
        from .dispatcher import (OrionDispatcher, _PluginApproval, _plugin_approval_scope,
                                 _plugin_blocked)
        from .execution_policy import background_scope
        from .tasks import spawn
        name = payload.get("plugin")
        arguments = copy.deepcopy(payload.get("arguments"))
        if (not isinstance(name, str) or not isinstance(arguments, dict)
                or not isinstance(_approval, _PluginApproval)
                or not _approval.matches(self, name, arguments)):
            return _plugin_blocked("The plugin call has no matching consumed human approval.", "requires_confirmation")

        async def _run() -> ToolResult:
            try:
                # A message approval permits this exact network payload only;
                # it cannot inherit or manufacture desktop input permission.
                with background_scope("approved_plugin", run_id=name), _plugin_approval_scope(_approval):
                    outcome = await OrionDispatcher.invoke_plugin(self, name, arguments)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                outcome = _plugin_blocked(f"Approved plugin setup failed: {type(exc).__name__}.", "unavailable")
            finally:
                _approval.active = False
            try:
                self.bus.log.emit(f"PLUGIN: approved {name} -> {outcome.text[:160]}")
            except Exception:
                pass
            return outcome

        task = spawn(_run(), bus=self.bus, label=f"running {name}")
        if task is None:
            _approval.active = False
            return _plugin_blocked("There is no running event loop to run it on.", "unavailable")
        task.add_done_callback(lambda _done: setattr(_approval, "active", False))
        return ToolResult(f"Queued the approved {name} action.", receipt={"kind": "plugin",
            "status": "queued", "attempted": False, "sent": False})

    def _run_confirmed_mcp_configuration(self, payload: dict) -> ToolResult:
        """Apply only the executable configuration bound to a human-issued token."""
        from .tasks import spawn
        from .mcp_host import load_config, _spec_fingerprint
        host = getattr(self, "mcp_host", None)
        if host is None:
            return ToolResult("The MCP host is not available.", ok=False)
        args = payload.get("mcp_configuration") or {}
        action = str(args.get("action", "")).lower().strip()
        server = str(args.get("server") or args.get("name") or "").strip()

        async def apply() -> None:
            if action in {"add", "declare"}:
                raw = args.get("arguments") or args.get("args") or []
                text = await asyncio.to_thread(
                    host.add_server, server, str(args.get("command") or ""),
                    raw if isinstance(raw, list) else [],
                    args.get("env") if isinstance(args.get("env"), dict) else {},
                    str(args.get("description") or ""))
            else:
                spec = (await asyncio.to_thread(load_config)).get("servers", {}).get(server)
                if not isinstance(spec, dict) or _spec_fingerprint(spec) != payload.get("spec_fingerprint"):
                    self.bus.log.emit("MCP: configuration changed after approval; request a new confirmation.")
                    return
                if action in {"enable", "on"}:
                    text = await host.set_enabled(server, True)
                elif action in {"reconnect", "restart"}:
                    text = f"MCP reconnect '{server}': {await host.reconnect(server)}"
                else:
                    return
            self.bus.log.emit(f"MCP: approved configuration: {str(text)[:200]}")

        if spawn(apply(), bus=self.bus, label=f"configuring MCP {server}") is None:
            return ToolResult("There is no running event loop to configure MCP.", ok=False)
        return ToolResult(f"Applying the approved MCP configuration for '{server}'.")

    def _run_confirmed_mcp(self, payload: dict) -> ToolResult:
        """Run exactly the MCP call the user approved (server, tool and
        arguments come from the token's payload, never from afterwards)."""
        host = getattr(self, "mcp_host", None)
        if host is None:
            return ToolResult("The MCP host is not available.", ok=False)
        server, tool = str(payload["server"]), str(payload["tool"])
        arguments = dict(payload.get("arguments") or {})
        from .tasks import spawn

        async def _run() -> None:
            text = await host.call(server, tool, arguments)
            try:
                self.bus.log.emit(f"MCP: approved {server}.{tool} -> {str(text)[:160]}")
            except Exception:
                pass

        if spawn(_run(), bus=self.bus, label=f"running {server}.{tool}") is None:
            return ToolResult("There is no running event loop to run it on.", ok=False)
        return ToolResult(f"Running {server}.{tool} now.")

    def _run_confirmed_process_kill(self, payload: dict) -> ToolResult:
        """Terminate the process the user approved — and only that one.

        Every value here comes from the token's payload, never from a fresh
        tool call, which is the whole point: between arming and release the
        model cannot substitute a different target for the one shown on the
        prompt.
        """
        import psutil

        if self.desktop is None:
            return ToolResult("Desktop control is unavailable.", ok=False)
        pid = payload.get("pid")
        label = str(payload.get("label") or "")
        action = str(payload.get("action") or "terminate")
        if pid is not None:
            try:
                proc = psutil.Process(int(pid))
                # A PID can be recycled while its confirmation dialog is open.
                if payload.get("created_at") is None or proc.create_time() != payload["created_at"]:
                    return ToolResult("The approved process has changed; request the action again.", ok=False)
                name = proc.name()
                terminated: list[str] = []
                self.desktop._terminate_process(proc, terminated)
            except (psutil.NoSuchProcess, ValueError):
                return ToolResult(f"No process with PID {pid} any more.", ok=False)
            except psutil.AccessDenied:
                return ToolResult(f"Permission denied for PID {pid}; termination was not confirmed.", ok=False)
            except SecurityViolation:
                return ToolResult("That process is protected; nothing was terminated.", ok=False)
            if not terminated:
                return ToolResult(f"Could not confirm a termination request for PID {pid}.", ok=False)
            result = f"Termination requested for PID {pid} ({name})."
            if action == "restart":
                result += f" Restart: {self.desktop.open_app(name).text}"
            return ToolResult(result)
        if label:
            closed = self.desktop.close_app(label)
            if action == "restart" and closed.ok:
                return ToolResult(f"{closed.text} "
                                  f"Restart: {self.desktop.open_app(label).text}")
            return closed
        return ToolResult("The approved action named no process.", ok=False)

    def cancel_system_action(self, token: str | None = None) -> ToolResult:
        """Cancel a pending guarded action (GUI 'Cancel' button or spoken cancel)."""
        self.system_guard.cancel(token)
        return ToolResult("Cancelled — nothing was executed.")

    def peripherals_tool(self, args: dict[str, Any]) -> "ToolResult | Awaitable[ToolResult]":
        """Hardware and OS peripheral control.

        Destructive host power actions (shutdown, restart, logout, sleep,
        hibernate) are NEVER executed straight from this call — they are armed
        with the deterministic SystemActionGuard and released only by a
        token-bound human confirmation (see confirm_system_action)."""
        from .system_guard import ActionIntent, DecisionKind
        if self.peripherals is None:
            return ToolResult("Peripheral control is unavailable.", ok=False)
        from .peripherals import parse_level
        action = str(args.get("action") or "").lower().strip().replace(" ", "_").replace("-", "_")
        action = _PERIPHERAL_ALIASES.get(action, action)
        level = parse_level(args.get("level"))
        direction = str(args.get("direction") or "").lower().strip()
        if action in {"volume", "brightness"} and level is None and direction:
            if direction in {"up", "louder", "higher", "brighter", "increase", "raise"}:
                action += "_up"
            elif direction in {"down", "quieter", "lower", "dimmer", "decrease", "reduce"}:
                action += "_down"
        if action == "volume":
            # An explicit 0 used to become 50% (`level or 0.5`), and a call
            # with no level set 50% blind. Without a level, say where it is.
            if level is None:
                return self.peripherals.describe_volume()
            return self.peripherals.set_volume(level)
        if action in {"volume_up", "volume_down"}:
            step = level if level else 0.1
            return self.peripherals.change_volume(step if action == "volume_up" else -step)
        if action == "volume_status":
            return self.peripherals.describe_volume()
        if action == "mute":
            return self.peripherals.set_mute(True)
        if action == "unmute":
            return self.peripherals.set_mute(False)
        if action == "toggle_mute":
            return self.peripherals.set_mute(None)
        if action == "brightness":
            if level is None:
                return self.peripherals.describe_brightness()
            return self.peripherals.set_brightness(level)
        if action in {"brightness_up", "brightness_down"}:
            step = level if level else 0.1
            return self.peripherals.change_brightness(step if action == "brightness_up" else -step)
        if action == "brightness_status":
            return self.peripherals.describe_brightness()
        if action in {"lock", "lock_screen"}:
            return self.peripherals.lock_screen()
        if action in {"wifi", "wi_fi", "ethernet", "bluetooth"}:
            state = str(args.get("state") or direction or "").lower().strip()
            enabled = (True if state in {"on", "enable", "enabled", "up", "connect"}
                       else False if state in {"off", "disable", "disabled", "down", "disconnect"}
                       else None)
            # Off the loop: netsh and the radio script take ~1-2 s, and a sync
            # handler runs on the thread that draws the face and plays audio.
            if action == "bluetooth":
                return asyncio.to_thread(self.peripherals.set_bluetooth, enabled)
            return asyncio.to_thread(self.peripherals.set_network,
                                     "ethernet" if action == "ethernet" else "wifi", enabled)
        if action in {"screen_off", "monitor_off", "display_off", "turn_off_screen"}:
            return self.peripherals.screen_off()

        power_intents = {
            "shutdown": ActionIntent.OS_SHUTDOWN,
            "restart":  ActionIntent.OS_RESTART,
            "reboot":   ActionIntent.OS_RESTART,
            "logout":   ActionIntent.OS_LOGOUT,
            "log_out":  ActionIntent.OS_LOGOUT,
            "sleep":    ActionIntent.OS_SLEEP,
            "suspend":  ActionIntent.OS_SLEEP,
            "hibernate": ActionIntent.OS_HIBERNATE,
        }
        intent = power_intents.get(action)
        if intent is not None:
            from .execution_policy import require_foreground
            blocked = require_foreground("host_power_confirmation")
            if blocked is not None:
                return blocked
            # A previously-issued, still-valid confirmation token releases it.
            token = str(args.get("confirm_token") or "").strip()
            if token:
                return self.confirm_system_action(token)
            # Otherwise arm the action and surface an accessible confirmation
            # prompt.  The token is delivered to the UI over the bus only — never
            # returned to the model, so approval cannot be spoofed by generated
            # text claiming confirm=true.
            decision = self.system_guard.request_confirmation(intent, origin="peripherals_tool")
            self.bus.confirm_action.emit({
                "token": decision.token,
                "intent": intent.value,
                "message": decision.message,
                "machine": decision.machine,
            })
            return ToolResult(
                decision.message + " (Awaiting your confirmation — I have not "
                "executed anything.)", ok=False)
        return ToolResult(f"Unsupported peripherals action: {action}.", ok=False)

    def interface_control(self, args: dict[str, Any]) -> ToolResult:
        """Let ORION navigate his OWN GUI: switch core view (hud/face/log/memory/
        telemetry), open a deck page (widgets/toolkit/studio/command/globe/
        diagnostics), show the live camera feed, toggle the dashboard/overlay/
        fullscreen, refresh the environment node, or fly the globe.  Driven
        over the bus so it runs on the GUI thread (no fragile pixel-clicking of
        his own window).

        action='camera' opens the Camera Lab with the LIVE FEED running. That
        is a different request from vision_analyse action='camera', which grabs
        a single frame and describes it — use this one whenever the user asks
        to SEE the camera, and that one when they ask what ORION can see."""
        action = str(args.get("action") or "").lower().strip()
        if action != "refresh_environment":
            # His OWN windows, driven over the bus: allowed while he answers
            # the owner, deferred in jobs, workflows and plugins. It was
            # require_foreground, which every tool call fails since Mark
            # XXXVI, so "show me that on the Command Deck" and "show me your
            # sad face" were refused every time (26 Oct 2026).
            from .execution_policy import require_own_window
            blocked = require_own_window("gui_navigation")
            if blocked is not None:
                return blocked
        target = str(args.get("target") or args.get("name") or args.get("place") or "").strip()
        if not action:
            return ToolResult(
                "interface_control needs an 'action' (view, page, pages, read, find, "
                "press, emotion, emotion_tour, dashboard, globe, refresh_environment, "
                "overlay, fullscreen).", ok=False)
        # Navigating to a page that does not exist used to report "done": the
        # command went onto the bus, nothing matched at the other end, and the
        # deck simply did not move while ORION said it had. Resolve the name
        # here, against the deck's real page list, before claiming anything.
        resolver = getattr(self, "page_resolver", None)
        if action == "page" and target and callable(resolver):
            try:
                resolved = resolver(target)
            except Exception:
                resolved = None
            if resolved is None:
                known = getattr(self, "page_names", None)
                options = ""
                if callable(known):
                    try:
                        options = " Pages: " + ", ".join(sorted(known())) + "."
                    except Exception:
                        options = ""
                return ToolResult(
                    f"There is no deck page matching {target!r}.{options}",
                    ok=False)
            target = resolved

        if action in {"face_design", "design", "face_style", "look"}:
            # Mark XXXIX: one Screen Face, several designs (appearance.py).
            from .appearance import (FACE_DESIGNS, design_label, designs_sentence, face_design,
                                     normalise_design)

            key = normalise_design(target)
            step = {"next": 1, "another": 1, "different": 1, "last": -1,
                    "back": -1}.get(str(target or "").strip().lower())
            if not key and step:
                # Mark XXXX.1: "next face design" flicks through them in order.
                here = FACE_DESIGNS.index(face_design()) if face_design() in FACE_DESIGNS else 0
                key = FACE_DESIGNS[(here + step) % len(FACE_DESIGNS)]
            if not key:
                current = design_label(face_design())
                return ToolResult(
                    f"I'm wearing the {current} design. My designs: {designs_sentence()}."
                    if not target else
                    f"{target!r} isn't one of my designs. I have {designs_sentence()}.",
                    ok=not target)
            self.bus.gui_command.emit({"action": "face_design", "target": key})
            return ToolResult(f"Done - I'm wearing the {design_label(key)} design now.")

        if action in {"face_form", "appearance", "form"}:
            from .appearance import SCREEN_ONLY_REPLY, form_unavailable, normalise

            form = normalise(target)
            if not form:
                return ToolResult(
                    f"{target!r} is not one of my forms. " + SCREEN_ONLY_REPLY,
                    ok=False)
            if form_unavailable(form):
                # The classic face and the orb are still in the code, but this
                # version does not offer them (appearance.ENABLED_FORMS).
                return ToolResult(SCREEN_ONLY_REPLY, ok=False)
            self.bus.gui_command.emit({"action": "face_form", "target": form})
            return ToolResult({
                "screen": "I am wearing my screen face now — just my eyes on a display.",
                "classic": "I am wearing my classic face now — the rendered 3-D one.",
                "orb": "I am now in my orb form — the orb, rather than a face.",
            }[form])

        if action in {"dock", "show_dock", "hide_dock"}:
            # The docked Screen Face is not on screen at start; this is how he
            # is asked for it (and asked to go again). Handled in the core
            # window, which holds the dock control app.py wires up.
            hide = action == "hide_dock" or target.lower() in {
                "hide", "off", "away", "close", "dismiss", "remove", "down"}
            self.bus.gui_command.emit({"action": "hide_dock" if hide else "show_dock",
                                       "target": ""})
            return ToolResult("Putting my dock away." if hide else
                              "Putting my face on the edge of your screen - my dock.")

        # His face's expressions, held on screen for a moment ("show me your
        # sad face") or one after another ("cycle through your expressions").
        # Owner, 26 Oct 2026: both were asked for and neither was possible.
        if action in _EXPRESSION_ACTIONS or action in _TOUR_ACTIONS:
            engine = getattr(self, "emotion", None)
            if engine is None or not hasattr(engine, "show"):
                return ToolResult("My expressions are not available right now.", ok=False)
            if action in _EXPRESSION_ACTIONS:
                message, ok = engine.show(target)
            else:
                names = [part.strip() for part in re.split(r",|;|/|\band\b", target)
                         if part.strip()] if target else None
                message, ok = engine.tour(names)
            return ToolResult(message, ok=ok)

        # Reading and pressing the Command Deck itself (gui/deck_reader.py):
        # what a page shows, where something is, and pressing a button there.
        if action in _DECK_QUERIES:
            return self._deck_request(_DECK_QUERIES[action], target,
                                      str(args.get("page") or "").strip())

        self.bus.gui_command.emit({"action": action, "target": target})
        return ToolResult(
            f"Interface: {action}{(' → ' + target) if target else ''} — done.")

    async def smart_home_tool(self, args: dict[str, Any]) -> ToolResult:
        """Lights, switches, scenes and routines through Home Assistant and the
        Hue bridge (smart_home.py). Network work runs off the event loop."""
        from . import smart_home
        action = str(args.get("action") or "status").lower().strip().replace(" ", "_")
        target = str(args.get("target") or args.get("name") or args.get("device") or "").strip()
        value = args.get("value", args.get("level", args.get("colour", args.get("color"))))
        if action in {"status", "devices", "list", "what_is_on", "describe"}:
            text, rows = await asyncio.to_thread(smart_home.describe)
            return ToolResult(text, ok=bool(smart_home.backends()),
                              evidence=[{"smart_home_devices": rows}])
        if action in {"routines", "list_routines"}:
            return ToolResult("Routines: " + ", ".join(smart_home.routines()) + ".")
        if action in {"routine", "run_routine", "goodnight", "morning", "movie", "away"}:
            name = target or (action if action not in {"routine", "run_routine"} else "")
            return await asyncio.to_thread(smart_home.run_routine, name)
        if action in {"pair", "pair_hue", "connect_hue"}:
            return await asyncio.to_thread(smart_home.pair_hue, str(args.get("bridge") or target))
        verbs = {"on": "on", "turn_on": "on", "off": "off", "turn_off": "off", "toggle": "toggle",
                 "brightness": "brightness", "dim": "brightness", "set_brightness": "brightness",
                 "colour": "colour", "color": "colour", "set_colour": "colour",
                 "scene": "scene", "activate_scene": "scene", "lock": "lock", "close": "close"}
        do = verbs.get(action)
        if do is None:
            return ToolResult(f"Unsupported smart_home action {action!r}. Use status, on, off, "
                              "toggle, brightness, colour, scene, routine, routines or pair.", ok=False)
        if do in {"lock", "close"}:
            do = "on" if do == "lock" else "off"
        return await asyncio.to_thread(smart_home.apply, do, target or "all lights", value)

    async def _deck_request(self, query: str, target: str, page: str) -> ToolResult:
        """Ask the core window to read, search or press the Command Deck and
        wait for its receipt - so ORION reports what the deck really did."""
        from uuid import uuid4
        signal = getattr(getattr(self, "bus", None), "gui_result", None)
        if signal is None:
            return ToolResult("The Command Deck is unavailable.", ok=False)
        loop = asyncio.get_running_loop()
        answer = loop.create_future()
        request_id = uuid4().hex

        def receive(receipt: Any) -> None:
            if not isinstance(receipt, dict) or receipt.get("request_id") != request_id:
                return

            def deliver() -> None:
                if not answer.done():
                    answer.set_result(receipt)
            loop.call_soon_threadsafe(deliver)

        signal.connect(receive)
        try:
            self.bus.gui_command.emit({"action": "deck_query", "request_id": request_id,
                                       "query": query, "target": target, "page": page})
            receipt = await asyncio.wait_for(answer, 4)
        except asyncio.TimeoutError:
            return ToolResult("The Command Deck did not answer in time.", ok=False)
        finally:
            try:
                signal.disconnect(receive)
            except (TypeError, RuntimeError, AttributeError):
                pass
        if not receipt.get("ok"):
            return ToolResult(str(receipt.get("error") or "That did not work."), ok=False)
        return ToolResult(str(receipt.get("text") or "Done."),
                          evidence=[{"deck": query, "page": receipt.get("page")}])

    def gesture_control(self, args: dict[str, Any]) -> ToolResult:
        """Control gestures and their persisted mappings; frame interpretation
        stays on GestureEngine's capture thread."""
        from .execution_policy import require_foreground
        if str(args.get("action") or "").lower().strip() in {"start", "pointer_start"}:
            blocked = require_foreground("gesture_input_session")
            if blocked is not None:
                return blocked
        if self.gestures is None:
            return ToolResult(
                "Gesture control is unavailable — install mediapipe with "
                "'pip install mediapipe'.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "start":
            return self.gestures.start()
        if action == "pointer_start":
            return self.gestures.start_pointer()
        if action == "pointer_stop":
            return self.gestures.stop_pointer()
        if action == "stop":
            return self.gestures.stop()
        if action == "status":
            return self.gestures.status()
        if action in {"list", "gestures", "mappings", "list_gestures"}:
            return self.gestures.list_gestures()
        if action in {"map", "set_mapping"}:
            tool_args = args.get("args") or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except (TypeError, ValueError):
                    return ToolResult("Mapped tool arguments must be a JSON object.", ok=False)
            if not isinstance(tool_args, dict):
                return ToolResult("Mapped tool arguments must be an object.", ok=False)
            known = self.handler_table()
            return self.gestures.set_mapping(str(args.get("gesture") or ""),
                action=str(args.get("mapped_action") or ""), tool=str(args.get("tool") or ""),
                args=tool_args, tool_exists=lambda name: name in known)
        if action in {"reset", "reset_mappings"}:
            return self.gestures.reset_mappings()
        if action in {"sensitivity", "set_sensitivity"}:
            return self.gestures.set_sensitivity(args.get("value", "normal"))
        if action in {"low_power", "set_low_power"}:
            enabled = args.get("enabled")
            if type(enabled) is not bool:
                return ToolResult("Low-power mode needs enabled=true or enabled=false.", ok=False)
            return self.gestures.set_low_power(enabled)
        if action == "calibrate":
            return self.gestures.calibrate(args.get("seconds", 3.0))
        if action in {"test", "self_test"}:
            execute = args.get("execute", False)
            if type(execute) is not bool:
                return ToolResult("Gesture tests need execute=true or execute=false.", ok=False)
            return self.gestures.test(str(args.get("gesture") or ""), execute=execute)
        return ToolResult(f"Unsupported gesture_control action: {action}.", ok=False)

    # Native actions ORION can hand to the paired phone (call/text/navigate…).
    _PHONE_ACTIONS = {"call", "sms", "text", "email", "navigate", "map",
                      "openurl", "open_url", "share"}

    def phone_action(self, args: dict[str, Any]) -> ToolResult:
        """Hand a native action to the user's paired phone (the ORION app):
        place a call, start a text/e-mail, open navigation, or share.

        The phone opens the relevant app PRE-FILLED — the user taps to confirm —
        so nothing is dialled or sent automatically.  Only useful while the user
        is on their phone; a desktop-only session just records the intent."""
        kind = str(args.get("kind") or args.get("action") or "").lower().strip()
        if kind == "text":
            kind = "sms"
        if kind in {"openurl", "open_url"}:
            kind = "openUrl"
        if kind not in {"call", "sms", "email", "navigate", "map", "openUrl", "share"}:
            return ToolResult(
                "phone_action needs a 'kind': call, sms, email, navigate, map, "
                "openUrl or share.", ok=False)
        payload: dict[str, Any] = {"kind": kind}
        for field in ("number", "body", "to", "subject", "query", "url", "text"):
            value = args.get(field)
            if value:
                payload[field] = str(value)
        # Minimal per-kind requirement so we never emit an empty action.
        required = {"call": "number", "sms": "number", "email": "to",
                    "navigate": "query", "map": "query", "openUrl": "url",
                    "share": "text"}.get(kind)
        if required and required not in payload:
            return ToolResult(f"phone_action '{kind}' needs '{required}'.", ok=False)
        try:
            self.bus.phone_action.emit(payload)
        except Exception as exc:
            return ToolResult(f"Couldn't reach the phone bridge: {exc}", ok=False)
        human = {"call": f"call {payload.get('number','')}",
                 "sms": f"text {payload.get('number','')}",
                 "email": f"email {payload.get('to','')}",
                 "navigate": f"navigate to {payload.get('query','')}",
                 "map": f"map {payload.get('query','')}",
                 "openUrl": f"open {payload.get('url','')}",
                 "share": "share that"}.get(kind, kind)
        return ToolResult(
            f"Sent to your phone — tap to {human}. (If you're not on the "
            "phone right now, it'll be waiting in the app.)")

    def cursor_overlay_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.cursor_overlay is None:
            return ToolResult("The cursor overlay is not available.", ok=False)
        action = str(args.get("action") or "toggle").lower().strip()
        if action in {"show", "on", "enable"}:
            self.cursor_overlay.start()
            return ToolResult("Cursor halo on — you'll see where I move the mouse.")
        if action in {"hide", "off", "disable"}:
            self.cursor_overlay.stop()
            return ToolResult("Cursor halo hidden.")
        state = self.cursor_overlay.toggle()
        return ToolResult(f"Cursor halo {'on' if state else 'off'}.")

    def audio_devices_tool(self, args: dict[str, Any]) -> ToolResult:
        """List, inspect or choose ORION's microphone and speaker.

        set_output/set_input persist the choice AND emit bus.audio_device_request
        — the same live-swap signal the Command Centre's device picker uses
        (live_worker._on_audio_device_request) — so a voice/text command takes
        effect immediately instead of only on the next start.
        """
        from . import audio_devices as ad
        action = str(args.get("action") or "status").lower().strip()
        if action in {"list", "devices"}:
            return ToolResult(ad.list_devices())
        if action in {"status", "current", "describe"}:
            return ToolResult(ad.describe())
        if action in {"set_output", "output", "speaker", "set_speaker"}:
            device = str(args.get("device") or "")
            message = ad.set_device("output", device)
            if not message.startswith("No "):
                self.bus.audio_device_request.emit("output", device)
                message = _switching_now(message)
            return ToolResult(message, ok=not message.startswith("No "))
        if action in {"set_input", "input", "mic", "set_mic", "microphone"}:
            device = str(args.get("device") or "")
            message = ad.set_device("input", device)
            if not message.startswith("No "):
                self.bus.audio_device_request.emit("input", device)
                message = _switching_now(message)
            return ToolResult(message, ok=not message.startswith("No "))
        return ToolResult(
            f"Unsupported audio_devices action: {action}. Use list, status, "
            "set_output, or set_input.",
            ok=False,
        )

    def elevenlabs_voice_tool(self, args: dict[str, Any]) -> ToolResult:
        """Configure ORION's ElevenLabs fallback voice (Mark XX design-spec:
        "ElevenLabs Integration") — status, list the account's available
        voices, or set the permanent voice_id. The API key itself is
        deliberately NOT settable here (same convention as every other
        provider credential in this app: config/api_keys.json directly, not
        a spoken/typed command) — this tool only ever touches voice_id."""
        from . import voice_elevenlabs as ev
        from . import voice_kokoro
        action = str(args.get("action") or "status").lower().strip()
        from . import voice_settings as vs
        if action in {"status", "describe"}:
            return ToolResult(ev.describe() + "\n" + voice_kokoro.describe_choice()
                              + "\n" + vs.describe())
        # How ORION sounds (voice_settings): a whole style - professional -
        # sets the offline and the online voice together; the other two
        # actions change one half each.
        if action in {"voice_style", "style", "voice_choice"}:
            wanted = str(args.get("style") or args.get("name") or args.get("voice") or "").strip()
            if not wanted:
                return ToolResult(vs.describe() + " Styles: "
                                  + ", ".join(s.label.lower() for s in vs.STYLES.values()) + ".")
            message, ok = vs.apply_style(wanted)
            return ToolResult(message, ok=ok)
        if action in {"online_voice", "cloud_voice", "gemini_voice"}:
            name = str(args.get("voice") or args.get("name") or "").strip()
            if not name:
                return ToolResult(f"My online voice is {vs.cloud_voice()}. Choose from: "
                                  + ", ".join(vs.GEMINI_VOICES) + ".")
            chosen = vs.normalise_cloud_voice(name)
            if chosen is None or not vs.set_value("cloud_voice", chosen):
                return ToolResult(f"There is no online voice called {name!r}. Choose from: "
                                  + ", ".join(vs.GEMINI_VOICES) + ".", ok=False)
            self._refresh_live_channel("new online voice")
            return ToolResult(f"My online voice is now {chosen}. It changes as soon as I finish speaking.")
        # Mark XXXX.1: how long he waits before answering, and whether his tone
        # follows the owner's. Both live in the channel's setup, so the
        # channel is refreshed (conversation kept) once he is quiet.
        if action in {"reply_pause", "pause", "patience", "wait", "listening"}:
            wanted = vs.normalise_reply_pause(args.get("pause") or args.get("value")
                                              or args.get("style") or "")
            if wanted is None:
                return ToolResult(f"My reply pause is {vs.reply_pause()} ({vs.reply_pause_ms() / 1000:g} s "
                                  "of silence before I answer). Choose brisk, natural or patient.")
            if not vs.set_value("reply_pause", wanted):
                return ToolResult("I couldn't save that just now.", ok=False)
            self._refresh_live_channel("new reply pause")
            return ToolResult(_PAUSE_REPLIES[wanted])
        if action in {"expressive", "expressive_voice", "tone", "affective"}:
            flag = vs.normalise_flag(args.get("enabled") if args.get("enabled") is not None
                                     else args.get("value", True))
            if flag is None or not vs.set_value("expressive", flag):
                return ToolResult("Say on or off for the expressive voice.", ok=False)
            self._refresh_live_channel("expressive voice " + ("on" if flag else "off"))
            return ToolResult("My tone will follow yours again." if flag
                              else "One even tone from now on, then.")
        if action in {"voice_speed", "offline_speed", "speaking_speed"}:
            speed = vs.normalise_speed(args.get("speed") if args.get("speed") is not None
                                       else args.get("value"))
            if speed is None or not vs.set_value("kokoro_speed", speed):
                low, high = vs.KOKORO_SPEED_LIMITS
                return ToolResult(f"My offline speed is {vs.kokoro_speed():g}. Give me a number "
                                  f"between {low:g} and {high:g}.", ok=False)
            return ToolResult(f"My offline speed is now {speed:g}.")
        # ORION's offline voice (voice_kokoro): which Kokoro male voice says
        # whatever Charon cannot - kept in config/voices/kokoro.json.
        if action in {"offline_voice", "offline", "local_voice", "kokoro"}:
            name = str(args.get("voice") or args.get("name") or "").strip()
            if not name:
                return ToolResult(voice_kokoro.describe_choice())
            message, ok = voice_kokoro.choose_voice(name)
            return ToolResult(message, ok=ok)
        if action in {"offline_voices", "local_voices", "kokoro_voices"}:
            return ToolResult(voice_kokoro.voices_text())
        if action in {"list", "voices", "list_voices"}:
            voices = ev.list_voices()
            if not voices:
                return ToolResult(
                    "No voices available — either no ElevenLabs API key is "
                    "configured (config/api_keys.json), or the lookup failed.",
                    ok=False)
            lines = [f"- {v['name']} ({v['voice_id']})" for v in voices[:30]]
            return ToolResult("ElevenLabs voices:\n" + "\n".join(lines))
        if action in {"set_voice_id", "set_voice", "voice_id"}:
            voice_id = str(args.get("voice_id") or args.get("voice") or "").strip()
            if not voice_id:
                return ToolResult("No voice_id supplied.", ok=False)
            return ToolResult(ev.set_voice(voice_id=voice_id))
        if action in {"preset", "character", "voice_preset", "sound_like"}:
            name = str(args.get("preset") or args.get("name")
                       or args.get("voice") or "").strip()
            if not name:
                return ToolResult(
                    "Which character voice? Presets: "
                    + ", ".join(ev.VOICE_PRESETS) + ".", ok=False)
            return ToolResult(ev.apply_preset(name))
        # A bare 'ultron'/'jarvis'/'narrator' also selects that preset.
        if action in ev.VOICE_PRESETS:
            return ToolResult(ev.apply_preset(action))
        if action in {"presets", "list_presets"}:
            return ToolResult("Character voice presets:\n" + "\n".join(
                f"- {k} — {v['description']}" for k, v in ev.VOICE_PRESETS.items()))
        return ToolResult(
            f"Unsupported elevenlabs_voice action: {action}. Use status, "
            "list, set_voice_id, preset (ultron/jarvis/narrator), presets, "
            "offline_voice, or offline_voices.",
            ok=False,
        )

    async def voice_speaker_id_tool(self, args: dict[str, Any]) -> ToolResult:
        """Guided learning uses the conversation stream; legacy enroll is
        kept for an explicitly requested one-clip recording. Both require
        consent. A match never authorises a sensitive action."""
        service = getattr(self, "voiceprint_service", None)
        if service is None:
            return ToolResult("Voice speaker recognition is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"learn", "learn_my_voice", "guided", "remember"}:
            name = str(args.get("name") or "").strip()
            identity = getattr(self, "identity", None)
            if not name and identity is not None:
                try:
                    name = identity.user_name()
                except (AttributeError, TypeError):
                    pass
            name = name or "Owner"
            return service.start_learning(
                name, consent=args.get("consent") is True,
                origin=str(args.get("origin") or "voice"), user_name=name,
                on_event=getattr(self, "voice_learning_event", None))
        if action in {"progress", "learning_status", "finish", "cancel"}:
            from . import voice_enrolment

            session = voice_enrolment.current()
            if session is None:
                return ToolResult("No voice learning is running.", ok=False)
            if action == "finish":
                ok, message = await asyncio.to_thread(session.finish)
                return ToolResult(message, ok=ok)
            if action == "cancel":
                return ToolResult(session.cancel())
            return ToolResult(session.describe())
        if action in {"forget_me", "forget_my_voice"}:
            from . import voice_enrolment

            ok, message = await asyncio.to_thread(
                voice_enrolment.forget_my_voice, service._prints())
            return ToolResult(message, ok=ok)
        if action in {"enroll", "enroll_clip"}:
            name = str(args.get("name") or "").strip()
            if not name:
                return ToolResult("A name is required to enrol a voice profile.", ok=False)
            if args.get("consent") is not True:
                return ToolResult(
                    "Voice enrolment requires explicit consent — ask the person to "
                    "confirm, then call again with consent=true.", ok=False)
            try:
                seconds = max(2.0, min(10.0, float(args.get("seconds") or 5.0)))
            except (TypeError, ValueError, OverflowError):
                return ToolResult("Recording length must be a number of seconds.", ok=False)
            from . import voice_speaker_id as vid
            try:
                clip = await asyncio.to_thread(vid.record_enrollment_clip, seconds)
            except Exception as exc:
                return ToolResult(f"Could not record from the microphone: {exc}", ok=False)
            return await asyncio.to_thread(service.enroll, name, [clip], 16000, True)
        if action in {"list", "profiles", "who"}:
            names = service.list_profiles()
            return ToolResult(
                "Enrolled voices: " + ", ".join(names) if names else "No voice profiles enrolled yet.")
        if action in {"remove", "forget", "delete"}:
            return service.remove_profile(str(args.get("name") or ""))
        if action in {"status", "describe"}:
            from . import voice_enrolment

            prints = service._prints()
            gate = ("He is answering ONLY your voice."
                    if prints.only_owner else "He answers anyone who speaks.")
            guard = (" A spoken go-ahead for a sensitive action must be in your voice."
                     if prints.guard_actions else "")
            return ToolResult(f"{service.describe()} {gate}{guard} "
                              f"{voice_enrolment.summary(prints)}")
        # ── acting on who spoke, rather than only noting it ───────────────────
        if action in {"owner", "set_owner", "this_is_me"}:
            from . import voiceprint

            ok, message = voiceprint.store().set_owner(
                str(args.get("name") or "").strip())
            return ToolResult(message, ok=ok)
        if action in {"only_me", "only_my_voice", "ignore_others"}:
            from . import voiceprint

            ok, message = voiceprint.store().set_only_owner(True)
            return ToolResult(message, ok=ok)
        if action in {"anyone", "answer_anyone", "listen_to_everyone"}:
            from . import voiceprint

            ok, message = voiceprint.store().set_only_owner(False)
            return ToolResult(message, ok=ok)
        if action in {"guard_actions", "protect_actions", "my_voice_for_actions"}:
            from . import voiceprint

            ok, message = voiceprint.store().set_guard_actions(True)
            return ToolResult(message, ok=ok)
        if action in {"unguard_actions", "anyone_can_confirm"}:
            from . import voiceprint

            ok, message = voiceprint.store().set_guard_actions(False)
            return ToolResult(message, ok=ok)
        return ToolResult(
            f"Unsupported voice_speaker_id action: {action}. Use learn, progress, "
            "finish, cancel, forget_me, enroll, list, remove, status, owner, only_me, anyone, guard_actions or "
            "unguard_actions.",
            ok=False,
        )

    def voice_tone_tool(self, args: dict[str, Any]) -> ToolResult:
        """How the last thing said SOUNDED, and who said it.

        Reports what ORION heard in the prosody rather than the words —
        loudness, pitch, pitch variation, pace and brightness, against how that
        person normally sounds. Answers "do I sound annoyed?", "can you tell
        how I'm feeling?", and "who is talking?".
        """
        presence = getattr(self, "voice_presence", None)
        if presence is None:
            return ToolResult("Voice tone reading is not available.", ok=False)
        action = str(args.get("action") or "last").lower().strip()
        if action in {"last", "now", "read", "tone", "mood", "emotion"}:
            latest = presence.last
            if latest is None:
                return ToolResult(
                    "I haven't heard a full spoken utterance to read yet - this "
                    "works on your voice, so it needs you to say something out loud.")
            detail = latest.emotion
            return ToolResult(
                f"{latest.describe()} "
                f"(confidence {detail.confidence:.0%}"
                + (f", next closest {detail.runner_up}" if detail.runner_up else "")
                + ".)")
        if action in {"status", "describe", "explain"}:
            return ToolResult(presence.describe())
        if action in {"forget", "reset"}:
            who = str(args.get("name") or "you").strip()
            removed = presence.emotion.baseline.forget(who)
            return ToolResult(
                f"Cleared what I'd learned about how {who} normally sounds."
                if removed else f"I had no voice baseline for {who}.")
        return ToolResult(
            f"Unsupported voice_tone action: {action}. Use last, status, or forget.",
            ok=False)

    async def system_startup_tool(self, args: dict[str, Any]) -> ToolResult:
        """Whether ORION starts with Windows, is installed as a desktop app,
        which browser he opens, and phone access on/off.

        Async because turning phone access on/off starts/stops the remote
        gateway, which is a coroutine; the dispatcher awaits awaitable results.
        """
        action = str(args.get("action") or "status").lower().strip()
        from . import autostart, browser, desktop_app

        if action in {"install_app", "install", "make_app", "desktop_app",
                      "add_shortcut", "add_shortcuts", "pin"}:
            result = desktop_app.install_shortcuts()
            if not result.get("supported", True):
                return ToolResult("Installing as a desktop app is a Windows "
                                  "feature only.", ok=False)
            made = [name for name in ("desktop", "start_menu") if result.get(name)]
            if not made:
                return ToolResult("I couldn't create the shortcuts.", ok=False)
            return ToolResult(
                "Done — I'm now a desktop app. There's an ORION icon on your "
                "Desktop and in the Start menu; right-click the Start-menu entry "
                "and choose 'Pin to taskbar' to keep me there. I launch without "
                "a console window.")
        if action in {"app_status", "am_i_an_app"}:
            return ToolResult(desktop_app.describe_status())
        if action in {"phone_on", "enable_phone", "remote_on", "phone_access"}:
            gateway = getattr(self, "gateway", None)
            if gateway is not None:
                return ToolResult("Phone access is already on. Open the Security "
                                  "Centre for the pairing QR code.")
            starter = getattr(self, "start_remote_gateway", None)
            if starter is None:
                return ToolResult(
                    "To turn on phone access, restart me with ORION_REMOTE_ACCESS=1 "
                    "— it's off by default now that I'm a desktop app, so nothing "
                    "runs on localhost unless you ask.", ok=False)
            try:
                await starter()
                return ToolResult("Phone access is on — open the Security Centre "
                                  "for the pairing code, then add me to your home "
                                  "screen.")
            except Exception as exc:
                return ToolResult(f"Could not start phone access: {first_line(exc, 100)}",
                                  ok=False)
        if action in {"phone_off", "disable_phone", "remote_off"}:
            gateway = getattr(self, "gateway", None)
            if gateway is None:
                return ToolResult("Phone access is already off — nothing is "
                                  "listening on localhost or the network.")
            try:
                await gateway.stop()
                self.gateway = None
                return ToolResult("Phone access is off; the local server is stopped.")
            except Exception as exc:
                return ToolResult(f"Could not stop phone access: {first_line(exc, 100)}",
                                  ok=False)
        if action in {"status", "check"}:
            state = autostart.status()
            stale = " (but it points somewhere that no longer exists - say "
            stale += "'fix startup' and I'll repair it)" if autostart.is_stale() else ""
            return ToolResult(
                f"{state.describe()}{stale} I open links in {browser.browser_name()}. "
                f"{desktop_app.describe_status()}")
        if action in {"enable", "start_with_windows", "on", "autostart"}:
            state = autostart.enable()
            if not state.supported:
                return ToolResult(f"I couldn't set that up - {state.detail}", ok=False)
            return ToolResult(
                "Done - I'll start automatically when you sign in to Windows. "
                "You'll find me in Task Manager's Startup tab if you ever want "
                "to switch it off yourself.")
        if action in {"disable", "off", "stop_starting"}:
            state = autostart.disable()
            if state.enabled:
                return ToolResult(f"I couldn't remove that - {state.detail}", ok=False)
            return ToolResult("I won't start with Windows any more.")
        if action in {"repair", "fix"}:
            state = autostart.ensure_current()
            return ToolResult(state.detail or state.describe())
        if action in {"browser", "which_browser"}:
            path = browser.browser_path()
            return ToolResult(
                f"I open links in {browser.browser_name()}"
                + (f" ({path})." if path else " - your system default."))
        return ToolResult(
            f"Unsupported action: {action}. Use status, enable, disable, repair, "
            "browser, or install_app.", ok=False)

    def navigation_trace_tool(self, args: dict[str, Any]) -> ToolResult:
        """Report on ORION's own UI-targeting history this session (Mark
        XXI, Track D7) — which clicks were found (by UIA or OCR), which
        failed and what was on screen instead, whether the screen-change
        verification confirmed each one. 'summary' gives the headline
        numbers; 'report' adds the recent per-attempt log; 'failures' lists
        only the misses. Diagnosable evidence for "is navigation actually
        working", not a guess."""
        trace = getattr(self, "navigation_trace", None)
        if trace is None:
            return ToolResult("Navigation trace is not available.", ok=False)
        action = str(args.get("action") or "summary").lower().strip()
        limit = int(args.get("limit") or 15)
        if action in {"summary", "status"}:
            return ToolResult(trace.summary())
        if action in {"report", "recent", "detail", "details"}:
            return ToolResult(trace.report(limit=limit))
        if action in {"failures", "misses", "failed"}:
            failures = trace.failures(limit=limit)
            if not failures:
                return ToolResult("No failed navigation attempts recorded yet this session.")
            lines = [f"- '{f['query']}' ({f['strategy']})"
                    + (f" — nearest: {', '.join(f['candidates'][:3])}" if f["candidates"] else "")
                    for f in failures]
            return ToolResult("Failed navigation attempts:\n" + "\n".join(lines))
        return ToolResult(
            f"Unsupported navigation_trace action: {action}. Use summary, report, or failures.",
            ok=False,
        )

    async def gaming_tool(self, args: dict[str, Any]) -> ToolResult:
        """Local gaming client discovery and launch."""
        if self.gaming is None:
            return ToolResult("Gaming client service is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "index":
            return await asyncio.to_thread(self.gaming.index_installs)
        if action == "launch":
            app_id = str(args.get("app_id") or "")
            return await asyncio.to_thread(self.gaming.launch_app, app_id)
        if action == "status":
            return await asyncio.to_thread(self.gaming.update_status)
        return ToolResult(f"Unsupported gaming action: {action}.", ok=False)

    def entertainment_tool(self, args: dict[str, Any]) -> ToolResult:
        """Entertainment discovery and media analysis (YouTube, trends)."""
        if self.entertainment is None:
            return ToolResult("Entertainment service is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "summarise":
            url = str(args.get("url") or "")
            return self.entertainment.summarise_video(url)
        if action == "channel":
            channel = str(args.get("channel") or "")
            return self.entertainment.channel_priority(channel)
        if action == "trending":
            region = str(args.get("region") or "GB")
            return self.entertainment.trending(region)
        return ToolResult(f"Unsupported entertainment action: {action}.", ok=False)


def _switching_now(message: str) -> str:
    """The tool also swaps the live stream, so say so. set_device's own line
    ("takes effect the next time ORION starts") is true only for a saved
    preference; on 10 Oct ORION read it out after "change to fifine" and the
    owner believed he was still on the webcam microphone."""
    return message.replace(
        "It takes effect the next time ORION starts (or restart the voice).",
        "Switching to it now.")


def _foreground_region() -> Any:
    """(left, top, width, height) of the window that will receive typing, in
    the same physical pixels the screen grabber uses; None if unknown."""
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None
        rect = wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 40 or height < 40:
            return None
        return (int(rect.left), int(rect.top), int(width), int(height))
    except Exception:
        return None


def _point_region(control: Any, args: dict[str, Any]) -> Any:
    """The area around a click target, known before clicking — so the check
    compares THIS click's neighbourhood, not the last action's."""
    if args.get("monitor") is not None or args.get("x") is None or args.get("y") is None:
        return None
    try:
        return control._region_around(int(args["x"]), int(args["y"]))
    except Exception:
        return None


def _visible_typing_default() -> bool:
    """ORION_VISIBLE_TYPING=0 makes typing machine-speed by default."""
    import os
    return os.getenv("ORION_VISIBLE_TYPING", "1").strip().lower() not in {"0", "false", "no", "off"}


def _flag(value: Any) -> bool:
    """A model's boolean: True, or a string that says so ("false" is not)."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _requested_speed(args: dict[str, Any]) -> str | None:
    """'human', 'fast' or None (the default) for a desktop_control call:
    its speed, or fast=true."""
    from .human_pointer import FAST, normalise_speed
    speed = normalise_speed(args.get("speed"))
    if speed is None and _flag(args.get("fast")):
        return FAST
    return speed
