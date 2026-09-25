"""
Dispatch domain — Desktop / OS control: apps, windows, cursor, clipboard, processes, peripherals, system actions.

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


class DesktopDispatchMixin:
    """Desktop / OS control: apps, windows, cursor, clipboard, processes, peripherals, system actions."""

    async def open_app(self, args: dict[str, Any]) -> ToolResult:
        # Off the event loop: an unmatched name now falls back to a document
        # search, and a search on the loop thread stalls the face and audio.
        return await asyncio.to_thread(
            self.desktop.open_app, str(args.get("app_name") or args.get("name") or ""))

    def close_app(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.close_app(str(args.get("app_name") or args.get("name") or ""))

    def window_control(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.window_control(
            str(args.get("action") or "list"), str(args.get("title") or "")
        )

    def media_control(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.media_control(
            str(args.get("action") or "play_pause"), int(args.get("steps") or 2)
        )

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
        try:
            return await self._desktop_control(args)
        finally:
            self.control.intent = ""

    async def _desktop_control(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").lower().strip()
        verify = bool(args.get("verify", True)) and self.verifier is not None

        def _run(fn: Callable[..., ToolResult], *a: Any, **k: Any) -> Any:
            return asyncio.to_thread(fn, *a, **k)

        # Element-targeted click with self-correcting coordinates.
        if action in {"click_text", "click_element"}:
            if self.verifier is None:
                return ToolResult("Verification engine unavailable for element clicks.", ok=False)
            return await self.verifier.click_text(str(args.get("text") or args.get("query") or ""))
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
            return await _run(self.control.scroll, int(args.get("amount", 0)),
                              args.get("x"), args.get("y"), args.get("monitor"))
        if action in {"smooth_scroll", "scroll_smooth", "glide_scroll"}:
            return await _run(self.control.smooth_scroll, int(args.get("amount", -10)),
                              args.get("x"), args.get("y"), args.get("monitor"),
                              float(args.get("duration", 0.8)))
        if action in {"type_text", "type", "type_human", "human_type"}:
            # Visible, person-paced typing by default — the user watches ORION
            # type rather than seeing a block of text appear as if pasted.
            # fast=true (or ORION_VISIBLE_TYPING=0) restores machine speed.
            from .human_typing import natural_wpm
            text_arg = str(args.get("text") or "")
            human = (action in {"type_human", "human_type"} or bool(args.get("human"))
                     or (not bool(args.get("fast")) and _visible_typing_default()))
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
                              visible=not bool(args.get("fast")) and _visible_typing_default())
        if action in {"hotkey", "send_hotkeys"}:
            return await _run(self.control.send_hotkeys, args.get("keys") or args.get("hotkey") or "")
        if action in {"open_app", "open_application"}:
            return await _run(self.control.open_application, str(args.get("app_name") or args.get("name") or ""))
        if action in {"close_app", "close_application"}:
            return await _run(self.control.close_application, str(args.get("app_name") or args.get("name") or ""))
        if action in {"focus_window", "focus", "switch"}:
            call = lambda: self.control.focus_window(str(args.get("title") or ""))
            return await self._maybe_verify(call, verify)
        if action in {"resize_window", "resize"}:
            call = lambda: self.control.resize_window(
                str(args.get("title") or ""),
                int(args.get("width", 800)), int(args.get("height", 600)))
            return await self._maybe_verify(call, verify)
        if action in {"move_window"}:
            call = lambda: self.control.move_window(
                str(args.get("title") or ""),
                int(args.get("x", 0)), int(args.get("y", 0)), args.get("monitor"))
            return await self._maybe_verify(call, verify)
        if action in {"minimise_window", "minimize_window"}:
            call = lambda: self.control.minimise_window(str(args.get("title") or ""))
            return await self._maybe_verify(call, verify)
        if action in {"maximise_window", "maximize_window"}:
            call = lambda: self.control.maximise_window(str(args.get("title") or ""))
            return await self._maybe_verify(call, verify)
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
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return ToolResult("Clipboard unavailable.", ok=False)
        if action == "read":
            text = clipboard.text()
            return ToolResult(text if text else "Clipboard is empty.")
        if action in {"copy", "write", "set"}:
            text = SecuritySanitiser.guard_text(str(args.get("text") or ""), "clipboard.text")
            clipboard.setText(text)
            return ToolResult("Clipboard updated.")
        return ToolResult(f"Unsupported clipboard action: {action}", ok=False)

    def process_governor(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        # ── the biggest power users (what the user asks ORION to police) ──────
        if action in {"top", "heavy", "hogs", "power", "high_power"}:
            metric = str(args.get("metric") or "cpu").lower()
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    procs.append((int(info["pid"]), str(info.get("name") or "?"),
                                  float(info.get("cpu_percent") or 0.0),
                                  float(info.get("memory_percent") or 0.0)))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            key = 3 if metric in {"ram", "memory", "mem"} else 2
            procs.sort(key=lambda p: -p[key])
            lines = [f"Top processes by {'RAM' if key == 3 else 'CPU'}:"]
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
            rows = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    rows.append(
                        f"{info['pid']:>6}  {info.get('name') or 'unknown'}  "
                        f"CPU {info.get('cpu_percent') or 0:.1f}%  "
                        f"RAM {info.get('memory_percent') or 0:.1f}%"
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
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
        self.bus.log.emit("SYS: shutdown directive accepted.")
        self.bus.request_shutdown.emit()
        return ToolResult("Orion shutdown initiated.")

    def restart_orion(self, args: dict[str, Any]) -> ToolResult:
        """Mark X.6: ORION restarts himself — a clean shutdown followed by the
        launcher respawning the process.  This is also how an applied
        self-repair patch gets loaded without the user touching anything."""
        self.bus.log.emit("SYS: restart directive accepted — going down to come back up.")
        self.bus.banner.emit("SYSTEM RESTART — back in a moment", 2)
        self.bus.request_restart.emit()
        return ToolResult("Orion restart initiated — I'll be back with you in a moment.")

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
            return self._run_confirmed_plugin(payload)
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

    def _run_confirmed_plugin(self, payload: dict) -> ToolResult:
        """Run exactly the plugin call the user approved (name and arguments
        come from the token's payload)."""
        name = str(payload["plugin"])
        handler = (getattr(self, "_tool_handlers", {}) or {}).get(name)
        if handler is None:
            return ToolResult(f"The '{name}' plugin is no longer loaded.", ok=False)
        arguments = dict(payload.get("arguments") or {})
        from .tasks import spawn

        async def _run() -> None:
            try:
                outcome = await asyncio.to_thread(handler, **arguments)
                if asyncio.iscoroutine(outcome):
                    outcome = await outcome
                from .plugin_results import plugin_result
                text = plugin_result(outcome).text
            except Exception as exc:
                text = f"failed: {exc}"
            try:
                self.bus.log.emit(f"PLUGIN: approved {name} -> {str(text)[:160]}")
            except Exception:
                pass

        if spawn(_run(), bus=self.bus, label=f"running {name}") is None:
            return ToolResult("There is no running event loop to run it on.", ok=False)
        return ToolResult(f"Running {name} now.")

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

    def peripherals_tool(self, args: dict[str, Any]) -> ToolResult:
        """Hardware and OS peripheral control.

        Destructive host power actions (shutdown, restart, logout, sleep,
        hibernate) are NEVER executed straight from this call — they are armed
        with the deterministic SystemActionGuard and released only by a
        token-bound human confirmation (see confirm_system_action)."""
        from .system_guard import ActionIntent, DecisionKind
        if self.peripherals is None:
            return ToolResult("Peripheral control is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "volume":
            level = float(args.get("level") or 0.5)
            return self.peripherals.set_volume(level)
        if action == "mute":
            return self.peripherals.toggle_mute()
        if action == "brightness":
            level = float(args.get("level") or 0.5)
            return self.peripherals.set_brightness(level)
        if action in {"lock", "lock_screen"}:
            return self.peripherals.lock_screen()
        if action == "wifi":
            return self.peripherals.toggle_wifi()
        if action == "ethernet":
            return self.peripherals.toggle_ethernet()

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
        target = str(args.get("target") or args.get("name") or args.get("place") or "").strip()
        if not action:
            return ToolResult(
                "interface_control needs an 'action' (view, page, dashboard, "
                "globe, refresh_environment, overlay, fullscreen).", ok=False)
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

        if action in {"face_form", "appearance", "form"}:
            from .appearance import normalise

            form = normalise(target)
            if not form:
                return ToolResult(
                    f"{target!r} is not one of my forms — I can be my face or my orb.",
                    ok=False)
            self.bus.gui_command.emit({"action": "face_form", "target": form})
            return ToolResult(f"I am now in my {form} form"
                              + (" — the orb, rather than a human face."
                                 if form == "orb" else "."))

        self.bus.gui_command.emit({"action": action, "target": target})
        return ToolResult(
            f"Interface: {action}{(' → ' + target) if target else ''} — done.")

    def gesture_control(self, args: dict[str, Any]) -> ToolResult:
        """Start/stop webcam hand-gesture control of this PC's volume,
        brightness and media playback. Gesture->action mapping runs entirely
        on GestureEngine's own capture thread — this tool only flips the
        mode on/off, matching interface_control's bus-driven start/stop
        shape rather than round-tripping every frame through the model."""
        if self.gestures is None:
            return ToolResult(
                "Gesture control is unavailable — install mediapipe with "
                "'pip install mediapipe'.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "start":
            return self.gestures.start()
        if action == "stop":
            return self.gestures.stop()
        if action == "status":
            return self.gestures.status()
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
            return ToolResult(message, ok=not message.startswith("No "))
        if action in {"set_input", "input", "mic", "set_mic", "microphone"}:
            device = str(args.get("device") or "")
            message = ad.set_device("input", device)
            if not message.startswith("No "):
                self.bus.audio_device_request.emit("input", device)
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
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "describe"}:
            return ToolResult(ev.describe())
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
            "list, set_voice_id, preset (ultron/jarvis/narrator), or presets.",
            ok=False,
        )

    async def voice_speaker_id_tool(self, args: dict[str, Any]) -> ToolResult:
        """Real speaker recognition — a SPECIFIC enrolled individual, not
        gender classification (that's the separate speaker_id/voice_gender
        tool). enroll records a short clip from the live microphone and
        requires explicit consent=true (mirrors send_draft's confirm=true
        for any other consequential action); list/remove/status are pure
        metadata over already-enrolled profiles. Live per-utterance
        identification during normal conversation is NOT wired here — see
        voice_speaker_id.record_enrollment_clip's docstring for why."""
        service = getattr(self, "voiceprint_service", None)
        if service is None:
            return ToolResult("Voice speaker recognition is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"enroll", "learn", "remember"}:
            name = str(args.get("name") or "").strip()
            if not name:
                return ToolResult("A name is required to enrol a voice profile.", ok=False)
            if not bool(args.get("consent")):
                return ToolResult(
                    "Voice enrolment requires explicit consent — ask the person to "
                    "confirm, then call again with consent=true.", ok=False)
            seconds = max(2.0, min(10.0, float(args.get("seconds") or 5.0)))
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
            from . import voiceprint

            prints = voiceprint.store()
            gate = ("He is answering ONLY your voice."
                    if prints.only_owner else "He answers anyone who speaks.")
            guard = (" A spoken go-ahead for a sensitive action must be in your voice."
                     if prints.guard_actions else "")
            return ToolResult(f"{service.describe()} {gate}{guard}")
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
            f"Unsupported voice_speaker_id action: {action}. Use enroll, list, "
            "remove, status, owner, only_me, anyone, guard_actions or "
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

    def gaming_tool(self, args: dict[str, Any]) -> ToolResult:
        """Local gaming client discovery and launch."""
        if self.gaming is None:
            return ToolResult("Gaming client service is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "index":
            return self.gaming.index_installs()
        if action == "launch":
            app_id = str(args.get("app_id") or "")
            return self.gaming.launch_app(app_id)
        if action == "status":
            return self.gaming.update_status()
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
