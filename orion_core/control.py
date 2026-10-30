"""
AutonomousControlLayer (Phase 2) — ORION's hands on the host machine.

Independent cursor, keyboard and window control, all coordinate-aware through
the DisplayTopologyManager so actions land correctly on any monitor at any DPI
scale.  Every method returns a structured ``ToolResult`` and records
``last_action`` metadata (a screen region and an expectation) that the
VisualVerificationEngine (Phase 3) reads to confirm the action visually before
ORION proceeds.

Backends, chosen for reliability:
    • cursor / scroll   — pyautogui (PAUSE disabled for low latency;
                          FAILSAFE kept so dragging to a screen corner aborts).
    • keyboard / hotkeys— the ``keyboard`` library (SendInput, full Unicode via
                          KEYEVENTF_UNICODE), falling back to pyautogui.
    • windows           — pygetwindow, with a Win32 SetForegroundWindow
                          fallback for focus.
    • app launch/close  — delegated to the existing DesktopAgent so there is
                          one implementation of the Start-Menu index.

Safety:
    • A global ``enabled`` flag and ORION_AUTONOMY env kill-switch; while
      disabled every mutating action refuses.
    • Typed text passes through the SecuritySanitiser, so ORION cannot be
      talked into typing a destructive shell command into a terminal.
    • All coordinates are clamped to the visible desktop.
"""

from __future__ import annotations

from contextlib import contextmanager

import math
import os
import random
import time
from typing import Any, Callable, Optional

from .bus import OrionBus
from .data import ToolResult
from .display import DisplayTopologyManager, physical_pixels
from .human_pointer import (FAST, HUMAN, PointerPlan, current_speed, normalise_speed,
                            plan_click, plan_pointer_path)
from .security import SecuritySanitiser, SecurityViolation
from .utils import fold_title

WHEEL_DELTA = 120          # one mouse-wheel notch, in Windows' wheel units
_SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
                  "Windows.UI.Core.CoreWindow", "NotifyIconOverflowWindow"}


def scroll_target(x: int, y: int) -> tuple[int, int, str] | None:
    """Where a wheel event reaches the window the user means.

    None when the pointer is already over another app's window (scroll right
    there). When it is over ORION itself, the desktop or the taskbar: the
    centre of the front-most visible window of any other app, and its title.
    Windows only; never raises."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        own = {os.getpid()}
        try:
            import psutil
            own |= {child.pid for child in psutil.Process().children(recursive=True)}
        except Exception:
            pass

        def owner(hwnd: int) -> int:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return int(pid.value)

        def klass(hwnd: int) -> str:
            buffer = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(hwnd, buffer, 128)
            return buffer.value

        under = user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
        root = user32.GetAncestor(under, 2) if under else 0          # GA_ROOT
        if root and owner(root) not in own and klass(root) not in _SHELL_CLASSES:
            return None
        found: list[tuple[int, int, str]] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _lparam):
            if found:
                return False
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0 or owner(hwnd) in own or klass(hwnd) in _SHELL_CLASSES:
                return True
            cloaked = wintypes.DWORD()
            try:
                ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), 4)
            except Exception:
                cloaked.value = 0
            if cloaked.value:
                return True
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if rect.right - rect.left < 120 or rect.bottom - rect.top < 120:
                return True
            title = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title, length + 1)
            found.append(((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2,
                          title.value))
            return False

        user32.EnumWindows(visit, 0)
        return found[0] if found else None
    except Exception:
        return None


class AutonomousControlLayer:
    """Cursor, keyboard and window control with per-action metadata."""

    def __init__(
        self,
        bus: OrionBus,
        display: DisplayTopologyManager,
        telemetry: Any | None = None,
        desktop: Any | None = None,
    ) -> None:
        self.bus = bus
        self.display = display
        self.telemetry = telemetry
        self.desktop = desktop
        self.enabled = os.getenv("ORION_AUTONOMY", "1").strip().lower() not in {"0", "false", "no", "off"}
        # Showing his working: the pointer GLIDES to each target instead of
        # teleporting, lingers on the click, and a caption beside it says what
        # he is doing (and why, when the model gave a reason). Pure
        # presentation — ORION_SHOW_WORKING=0 restores instant moves.
        self.show_working = os.getenv("ORION_SHOW_WORKING", "1").strip().lower() \
            not in {"0", "false", "no", "off"}
        #: The reason for the action in progress, set by the dispatcher from
        #: the tool call's 'why' argument; shown in the caption.
        self.intent = ""
        # The region + description of the most recent action, for the verifier.
        self.last_action: dict[str, Any] = {}
        self._pg: Any = None
        self._kb: Any = None
        # Randomness for human pointer paths and click timing (seed it to
        # replay a motion exactly).
        self._rng = random.Random()
        self._configure_backends()

    # ── backend setup ─────────────────────────────────────────────────────────

    def _configure_backends(self) -> None:
        try:
            import pyautogui  # type: ignore
            pyautogui.PAUSE = 0.0            # we pace deliberately, not globally
            pyautogui.FAILSAFE = True        # corner-abort remains a hard stop
            self._pg = pyautogui
        except Exception as exc:
            self._pg = None
            self.bus.log.emit(f"CONTROL: pyautogui unavailable - {exc}")
        try:
            import keyboard  # type: ignore
            self._kb = keyboard
        except Exception:
            self._kb = None

    # ── guards ────────────────────────────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.bus.log.emit(f"CONTROL: autonomy {'ENABLED' if self.enabled else 'DISABLED'}.")

    #: The policy effect each action is checked as (execution_policy). Mark
    #: XL.2 let the owner's live turn move the pointer and use the keyboard,
    #: but every action here was checked under its own name ("click",
    #: "type_text"...), none of them on that list - so "press Start Camera"
    #: was still refused (28 Oct 2026). Background work keeps both blocked.
    _EFFECT_OF = {"click": "move_cursor", "drag_cursor": "move_cursor", "scroll": "move_cursor",
                  "smooth_scroll": "move_cursor", "type_text": "keyboard_input",
                  "send_hotkeys": "keyboard_input", "edit_text": "keyboard_input",
                  "open_application": "application_open"}

    def _guard(self, action: str, *, pointer: bool = True) -> Optional[ToolResult]:
        from .execution_policy import require_foreground
        deferred = require_foreground(self._EFFECT_OF.get(action, action))
        if deferred is not None:
            return deferred
        if not self.enabled:
            return ToolResult(
                f"Autonomous control is disabled; refused '{action}'. "
                "Enable it in the Command Centre or set ORION_AUTONOMY=1.",
                ok=False,
            )
        if pointer and self._pg is None:
            return ToolResult("pyautogui is not available; cursor/keyboard control offline.", ok=False)
        return None

    def _record(self, action: str, region: Optional[tuple[int, int, int, int]],
                detail: str, **extra: Any) -> None:
        self.last_action = {"action": action, "region": region, "detail": detail,
                            "at": time.monotonic(), **extra}
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"control.{action}")
        # Flare the visible cursor halo so the user sees ORION acting.
        try:
            self.bus.control_activity.emit(action)
        except RuntimeError:
            pass

    # ── showing his working ───────────────────────────────────────────────────

    LINGER_S = 0.25

    def _narrate(self, doing: str, *, x: int | None = None, y: int | None = None,
                 region: Optional[tuple[int, int, int, int]] = None) -> None:
        """Put a caption beside the pointer: the reason if there is one,
        otherwise what the action is. Emitted from the worker thread; the
        overlay receives it on the GUI thread (queued connection)."""
        if not self.show_working:
            return
        text = f"{self.intent} — {doing}" if self.intent else doing
        try:
            self.bus.control_narration.emit({"text": text[:140], "x": x, "y": y,
                                             "region": region})
        except (RuntimeError, AttributeError):
            pass

    def pointer_mode(self, mode: Any = None) -> str:
        """'human' or 'fast' for this action: the caller's word, else the
        request's (human_pointer.speed_scope, set by the dispatcher from the
        tool call's speed), else human while showing his working and fast
        when ORION_SHOW_WORKING=0."""
        return (normalise_speed(mode) or current_speed()
                or (HUMAN if self.show_working else FAST))

    def _inside_desktop(self) -> Optional[Callable[[int, int], bool]]:
        """Whether a point is on a connected screen, for keeping a curved
        path off the gaps between offset monitors."""
        try:
            monitors = list(self.display.topology().monitors)
            if monitors:
                return lambda x, y: any(m.contains(x, y) for m in monitors)
        except Exception:
            pass
        clamp = getattr(self.display, "clamp_to_desktop", None)
        if clamp is None:
            return None
        return lambda x, y: tuple(clamp(x, y)) == (x, y)

    def _travel(self, vx: int, vy: int, mode: Any = None, *,
                overshoot: bool = True) -> bool:
        """Move the pointer to exactly (vx, vy).

        Human (the default): a curved, eased path timed by Fitts's law, with
        a small overshoot and correction on long moves (human_pointer), so the
        eye can follow it across the desktop and between monitors. Fast: one
        jump. Runs on the caller's worker thread, never the GUI thread.
        Returns False when autonomy was switched off or the request lost its
        permission mid-way; the pointer then stays where it got to and the
        caller's own guard reports why."""
        from .execution_policy import require_foreground
        deferred = require_foreground("move_cursor")
        if deferred is not None:
            raise RuntimeError(deferred.text)
        if self.pointer_mode(mode) == FAST:
            self._pg.moveTo(vx, vy)
            return True
        try:
            start = self.display.cursor_position()
        except Exception:
            start = None
        if start is None:
            self._pg.moveTo(vx, vy)
            return True
        plan = plan_pointer_path(start, (vx, vy), rng=self._rng, overshoot=overshoot,
                                 inside=self._inside_desktop())
        return self._follow(plan)

    def _follow(self, plan: PointerPlan, *, drag: bool = False) -> bool:
        """Replay *plan* against the clock (sleeping to each step's due time,
        so slow moveTo calls do not stretch the motion). Each step goes
        through pyautogui, whose fail-safe corner still aborts it."""
        from .execution_policy import require_foreground
        started = time.monotonic()
        due = 0.0
        for step in plan.steps:
            if not self.enabled or require_foreground(
                    "drag_cursor" if drag else "move_cursor") is not None:
                return False
            self._pg.moveTo(step.x, step.y)
            due += step.delay
            wait = started + due - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        return True

    @staticmethod
    def _double_click_s() -> float:
        """The system double-click time, in seconds (0.5 s by default)."""
        try:
            import ctypes
            ms = int(ctypes.windll.user32.GetDoubleClickTime())
            if 100 <= ms <= 5000:
                return ms / 1000.0
        except Exception:
            pass
        return 0.5

    def _press(self, vx: int, vy: int, button: str, clicks: int, mode: Any = None) -> None:
        """Click *clicks* times at (vx, vy).

        Human: a brief look at the target, then each press held 60-140 ms
        and, for a double click, a gap that stays inside the system's
        double-click time (human_pointer.plan_click). A press always gets its
        release, even if the wait between them fails. Fast (or a backend
        without separate press and release): pyautogui's immediate click."""
        if (self.pointer_mode(mode) == FAST or not hasattr(self._pg, "mouseDown")
                or not hasattr(self._pg, "mouseUp")):
            self._pg.click(vx, vy, clicks=clicks, interval=0.05, button=button)
            return
        timing = plan_click(clicks, rng=self._rng, double_click_s=self._double_click_s())
        time.sleep(timing.settle)
        for index, hold in enumerate(timing.holds):
            if index:
                time.sleep(timing.gaps[index - 1])
            self._pg.mouseDown(vx, vy, button=button)
            try:
                time.sleep(hold)
            finally:
                self._pg.mouseUp(vx, vy, button=button)

    def _region_around(self, vx: int, vy: int, pad: int = 90) -> tuple[int, int, int, int]:
        x0, y0 = self.display.clamp_to_desktop(vx - pad, vy - pad)
        x1, y1 = self.display.clamp_to_desktop(vx + pad, vy + pad)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # ── cursor ────────────────────────────────────────────────────────────────

    @contextmanager
    def _hand_the_mouse_back(self, mode: Any = None):
        """Put the pointer back where the user left it.

        ORION drives one physical cursor — the same one the user's hand is on.
        Without this, every click he makes teleports the pointer and leaves it
        there, so working alongside him means having the mouse pulled out from
        under you, and any drag or selection the user was mid-way through is
        destroyed.

        Restoring it does not make him invisible — the pointer still jumps for
        as long as the action takes — but it does mean the user gets it back
        the instant he is done, which is the difference between "I cannot
        touch the mouse" and "the cursor flickered".

        Set ORION_KEEP_CURSOR=1 to leave the pointer where he put it, which is
        occasionally what you want while watching him work.
        """
        keep = os.getenv("ORION_KEEP_CURSOR", "").strip().lower() in {"1", "true", "yes", "on"}
        origin = None
        if not keep:
            try:
                origin = self.display.cursor_position()
            except Exception:
                origin = None
        try:
            yield
        finally:
            if origin is not None:
                try:
                    if self.pointer_mode(mode) == HUMAN:
                        # Long enough to see where the click landed, then a
                        # quick glide home rather than a jump.
                        time.sleep(self.LINGER_S)
                        tween = getattr(self._pg, "easeOutQuad", None)
                        if tween is not None:
                            self._pg.moveTo(origin[0], origin[1], duration=0.12, tween=tween)
                        else:
                            self._pg.moveTo(origin[0], origin[1], duration=0.12)
                    else:
                        self._pg.moveTo(origin[0], origin[1], duration=0.0)
                except Exception:
                    pass

    @physical_pixels
    def move_cursor(self, x: int, y: int, monitor: int | None = None,
                    duration: float = 0.0, mode: Any = None) -> ToolResult:
        """Move the pointer to (x, y): a human path by default (see _travel),
        one jump with mode='fast', or a straight glide of exactly *duration*
        seconds when one is given."""
        guard = self._guard("move_cursor")
        if guard:
            return guard
        try:
            vx, vy = self._resolve(x, y, monitor, exact=True)
        except (TypeError, ValueError, OverflowError) as exc:
            return ToolResult(f"Invalid pointer target: {exc}", ok=False)
        self._narrate("moving here", x=vx, y=vy)
        try:
            if duration > 0:
                self._pg.moveTo(vx, vy, duration=float(duration))
            elif not self._travel(vx, vy, mode):
                return self._guard("move_cursor") or ToolResult(
                    "The pointer move was stopped part-way.", ok=False)
        except Exception as exc:
            # pyautogui's fail-safe (the pointer pushed into a screen corner)
            # lands here: a deliberate stop, reported rather than raised.
            return ToolResult(f"The pointer move stopped: {exc}", ok=False)
        self._record("move_cursor", self._region_around(vx, vy), f"cursor → ({vx},{vy})",
                     x=vx, y=vy, mode=self.pointer_mode(mode))
        return ToolResult(f"Cursor moved to ({vx}, {vy}).")

    @physical_pixels
    def gesture_pointer(self, action: str, x: float = 0.5,
                        y: float = 0.5) -> ToolResult:
        """Drive the primary-screen pointer from explicitly armed hand control.

        A button release is always allowed, even if autonomy is switched off
        halfway through a drag. GestureEngine owns the dwell/cooldown policy.
        """
        if action == "up":
            if self._pg is not None:
                try:
                    self._pg.mouseUp(button="left")
                except Exception:
                    pass
            return ToolResult("Gesture pointer released.")
        guard = self._guard("gesture_pointer")
        if guard:
            return guard
        monitor = self.display.primary()
        if monitor is None:
            return ToolResult("No screen is available for gesture control.", ok=False)
        if action == "ready":
            return ToolResult("Hand pointer is available.")
        # Mirror camera X: moving the right hand right moves the cursor right.
        vx = monitor.x + round((1.0 - max(0.0, min(1.0, x))) * (monitor.width - 1))
        vy = monitor.y + round(max(0.0, min(1.0, y)) * (monitor.height - 1))
        try:
            if action == "move":
                self._pg.moveTo(vx, vy, duration=0)
            elif action == "down":
                self._pg.mouseDown(vx, vy, button="left")
            elif action in {"scroll_up", "scroll_down"}:
                self._pg.scroll(2 if action == "scroll_up" else -2)
            else:
                return ToolResult(f"Unsupported gesture pointer action: {action}.", ok=False)
        except Exception as exc:
            return ToolResult(f"Gesture pointer stopped: {exc}", ok=False)
        if action != "move":
            self._record("gesture_pointer", self._region_around(vx, vy),
                         f"hand {action} at ({vx},{vy})")
        return ToolResult(f"Gesture pointer {action}.")

    @physical_pixels
    def drag_cursor(self, x1: int, y1: int, x2: int, y2: int, monitor: int | None = None,
                    duration: float = 0.25, button: str = "left",
                    mode: Any = None) -> ToolResult:
        guard = self._guard("drag_cursor")
        if guard:
            return guard
        try:
            sx, sy = self._resolve(x1, y1, monitor, exact=True)
            ex, ey = self._resolve(x2, y2, monitor, exact=True)
        except (TypeError, ValueError, OverflowError) as exc:
            return ToolResult(f"Invalid drag target: {exc}", ok=False)
        mode = self.pointer_mode(mode)
        human = mode == HUMAN and hasattr(self._pg, "mouseDown") and hasattr(self._pg, "mouseUp")
        with self._hand_the_mouse_back(mode):
            self._narrate("dragging", x=sx, y=sy)
            try:
                if not self._travel(sx, sy, mode):
                    return self._guard("drag_cursor") or ToolResult(
                        "The drag was stopped before it began; nothing was moved.", ok=False)
                guard = self._guard("drag_cursor")
                if guard:
                    return guard
                if not human:
                    self._pg.dragTo(ex, ey, duration=max(0.05, duration), button=button)
                elif not self._human_drag(sx, sy, ex, ey, button):
                    return ToolResult("The drag was stopped part-way and the button released "
                                      f"at the pointer; it did not reach ({ex},{ey}).", ok=False)
            except Exception as exc:
                return ToolResult(f"Drag failed or its result is uncertain; not repeated: {exc}",
                                  ok=False)
        self._record("drag_cursor", self._region_around(ex, ey), f"drag ({sx},{sy})→({ex},{ey})")
        return ToolResult(f"Dragged from ({sx},{sy}) to ({ex},{ey}).")

    def _human_drag(self, sx: int, sy: int, ex: int, ey: int, button: str) -> bool:
        """Press, carry the item along a human path a little slower than a
        free move (and without overshooting, which would drop it in the
        wrong place first), pause, release. The button is always released."""
        carried = False
        self._pg.mouseDown(sx, sy, button=button)
        try:
            time.sleep(self._rng.uniform(0.08, 0.15))          # get a grip
            plan = plan_pointer_path((sx, sy), (ex, ey), rng=self._rng, overshoot=False,
                                     inside=self._inside_desktop(), duration_scale=1.3)
            carried = self._follow(plan, drag=True)
            if carried:
                time.sleep(self._rng.uniform(0.05, 0.12))      # settle before letting go
        finally:
            if carried:
                self._pg.mouseUp(ex, ey, button=button)
            else:
                self._pg.mouseUp(button=button)
        return carried

    @physical_pixels
    def click(self, x: int | None = None, y: int | None = None, button: str = "left",
              clicks: int = 1, monitor: int | None = None, *,
              target_check: Callable[[], bool] | None = None,
              mode: Any = None) -> ToolResult:
        """Click at (x, y), or where the pointer is. Human by default: a
        curved approach, a moment on the target, then each press held about
        a tenth of a second (double clicks inside the system double-click
        time). mode='fast': a jump and an immediate click."""
        mode = self.pointer_mode(mode)
        guard = self._guard("click")
        if guard:
            return guard
        if (x is None) != (y is None):
            return ToolResult("Supply both x and y coordinates, or neither.", ok=False)
        if button not in {"left", "right", "middle"} or not isinstance(clicks, int) or not 1 <= clicks <= 3:
            return ToolResult("Use left, right or middle with one to three clicks.", ok=False)
        try:
            vx, vy = (self.display.cursor_position() if x is None else
                      self._resolve(x, y, monitor, exact=True))
            if target_check is not None and not target_check():
                return ToolResult("The target changed or is covered; no click was issued. Inspect the current view.", ok=False)
        except (TypeError, ValueError, OverflowError) as exc:
            return ToolResult(f"Invalid click target: {exc}", ok=False)
        except Exception as exc:
            return ToolResult(f"Target validation failed; no click was issued: {exc}", ok=False)
        if x is None or y is None:
            # No coordinates: he is clicking wherever the pointer already is,
            # so there is nothing to restore and nothing to take away.
            self._narrate(self._click_words(button, clicks), x=vx, y=vy)
            try:
                self._press(vx, vy, button, clicks, mode)
            except Exception as exc:
                return ToolResult(f"Click failed or its result is uncertain; not repeated: {exc}", ok=False)
        else:
            with self._hand_the_mouse_back(mode):
                self._narrate(self._click_words(button, clicks), x=vx, y=vy)
                try:
                    self._travel(vx, vy, mode)
                    # A glide/hover can take half a second and expose a menu.
                    # Re-check after travelling, immediately before input.
                    guard = self._guard("click")
                    if guard:
                        return guard
                    if target_check is not None and not target_check():
                        return ToolResult("The target changed or is covered; no click was issued. Inspect the current view.", ok=False)
                    self._press(vx, vy, button, clicks, mode)
                except Exception as exc:
                    return ToolResult(f"Click failed or its result is uncertain; not repeated: {exc}", ok=False)
        self._record("click", self._region_around(vx, vy),
                     f"{button} click x{clicks} @ ({vx},{vy})", x=vx, y=vy, button=button,
                     mode=mode)
        return ToolResult(f"{button.capitalize()} click ({clicks}x) at ({vx}, {vy}).")

    def _narrate_typing(self, text: str, where: str = "") -> None:
        preview = " ".join(str(text).split())
        preview = preview if len(preview) <= 40 else preview[:37] + "..."
        target = f" into {where}" if where else ""
        # Outline the window receiving the keys: typing moves no pointer, so
        # without this the only sign of where ORION is working was a caption
        # next to wherever the USER's mouse happened to be.
        region = None
        try:
            win = self._find_window(where) if where else None
            region = self._window_region(win) if win is not None else _foreground_rect()
        except Exception:
            region = None
        self._narrate(f"typing “{preview}”{target}", region=region)

    @staticmethod
    def _click_words(button: str, clicks: int) -> str:
        if clicks >= 2:
            return "double-clicking"
        return "right-clicking" if button == "right" else "clicking"

    def double_click(self, x: int | None = None, y: int | None = None,
                     monitor: int | None = None, mode: Any = None) -> ToolResult:
        return self.click(x, y, button="left", clicks=2, monitor=monitor, mode=mode)

    def right_click(self, x: int | None = None, y: int | None = None,
                    monitor: int | None = None, mode: Any = None) -> ToolResult:
        return self.click(x, y, button="right", clicks=1, monitor=monitor, mode=mode)

    @physical_pixels
    def scroll(self, amount: int, x: int | None = None, y: int | None = None,
               monitor: int | None = None) -> ToolResult:
        guard = self._guard("scroll")
        if guard:
            return guard
        if (x is None) != (y is None):
            return ToolResult("Supply both x and y coordinates, or neither.", ok=False)
        if not int(amount):
            return ToolResult("No scrolling requested (0).")
        way = "scrolling down" if int(amount) < 0 else "scrolling up"
        where = ""
        if x is not None and y is not None:
            try:
                vx, vy = self._resolve(x, y, monitor, exact=True)
            except (TypeError, ValueError, OverflowError) as exc:
                return ToolResult(f"Invalid scroll target: {exc}", ok=False)
            with self._hand_the_mouse_back():
                self._narrate(way, x=vx, y=vy)
                if not self._travel(vx, vy):
                    return self._stopped("scroll")
                self._wheel(int(amount))
        else:
            vx, vy = self.display.cursor_position()
            target = scroll_target(vx, vy)
            if target is not None:
                # The pointer is over ORION, the desktop or the taskbar: the
                # wheel goes to whatever is under the pointer, so "scroll
                # down" scrolled nothing the user was reading.
                vx, vy, where = target
                with self._hand_the_mouse_back():
                    self._narrate(way, x=vx, y=vy)
                    if not self._travel(vx, vy):
                        return self._stopped("scroll")
                    self._wheel(int(amount))
            else:
                self._narrate(way, x=vx, y=vy)
                self._wheel(int(amount))
        self._record("scroll", self._region_around(vx, vy), f"scroll {amount} @ ({vx},{vy})")
        return ToolResult(f"Scrolled {'down' if int(amount) < 0 else 'up'} {abs(int(amount))} "
                          f"notch{'es' if abs(int(amount)) != 1 else ''}"
                          + (f" in {where}" if where else "") + f" at ({vx}, {vy}).")

    def _stopped(self, action: str) -> ToolResult:
        """Why a pointer journey ended before the action: the guard's own
        refusal when autonomy or permission went mid-way."""
        return self._guard(action) or ToolResult(
            f"'{action}' was stopped before it acted; nothing was pressed or turned.", ok=False)

    def _wheel(self, notches: float) -> None:
        """Turn the wheel by *notches*. On Windows pyautogui passes its number
        straight through as the raw wheel delta, where ONE notch is 120: the
        scroll(-5) the model sends for "scroll down" was 1/24th of a notch,
        which Notepad ignores and a browser moves by two pixels."""
        delta = notches * WHEEL_DELTA if os.name == "nt" else notches
        delta = int(round(delta))
        if delta:
            self._pg.scroll(delta)

    @physical_pixels
    def smooth_scroll(self, amount: int, x: int | None = None, y: int | None = None,
                      monitor: int | None = None, duration: float = 0.8) -> ToolResult:
        """
        Scroll by *amount* wheel-notches smoothly, using many small eased steps
        so the page glides rather than jumps.  Negative amount scrolls down.
        The cursor stays put (and the visible halo tracks it), so the motion
        reads as continuous, cursor-aligned navigation.
        """
        guard = self._guard("smooth_scroll")
        if guard:
            return guard
        if (x is None) != (y is None):
            return ToolResult("Supply both x and y coordinates, or neither.", ok=False)
        amount = int(amount)
        if not amount:
            return ToolResult("No scrolling requested (0).")
        if x is not None and y is not None:
            try:
                vx, vy = self._resolve(x, y, monitor, exact=True)
            except (TypeError, ValueError, OverflowError) as exc:
                return ToolResult(f"Invalid scroll target: {exc}", ok=False)
            if not self._travel(vx, vy):
                return self._stopped("smooth_scroll")
        else:
            vx, vy = self.display.cursor_position()
        self._narrate("scrolling down" if int(amount) < 0 else "scrolling up", x=vx, y=vy)
        steps = min(60, max(abs(amount), 12))
        # Notches, as the docstring always said: the raw value used to go to
        # pyautogui, which on Windows is 1/120th of a notch per unit.
        unit = WHEEL_DELTA if os.name == "nt" else 1
        duration = max(0.2, min(4.0, float(duration)))
        pause = duration / steps
        delivered = 0
        for i in range(steps):
            guard = self._guard("smooth_scroll")
            if guard:
                return guard
            # Ease-in-out: a sine bell makes steps gentle at the ends and
            # quicker in the middle, so the glide feels natural.
            phase = (i + 0.5) / steps
            weight = math.sin(math.pi * phase)
            # Round cumulative distance, not each step: this preserves the
            # requested total for small amounts and amounts above 60 alike.
            target = round(amount * unit * (i + 1) / steps)
            if target != delivered:
                self._pg.scroll(target - delivered)
            delivered = target
            time.sleep(pause * (0.6 + 0.8 * weight))
            self.bus.control_activity.emit("smooth_scroll")
        self._record("smooth_scroll", self._region_around(vx, vy),
                     f"smooth scroll {amount} in {steps} steps @ ({vx},{vy})")
        return ToolResult(f"Scrolled smoothly ({amount}) over {steps} steps at ({vx}, {vy}).")

    # ── keyboard ──────────────────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.01, *,
                  human: bool = False, wpm: float = 90.0,
                  typos: float = 0.0, jitter: float = 0.35,
                  seed: int | None = None) -> ToolResult:
        """
        Type text through the OS as a real key sequence (SendInput/Unicode).

        With ``human=True`` ORION types the way a person does: one key at a time
        at ``wpm`` words-per-minute, with per-key timing ``jitter`` and an
        optional ``typos`` rate (0..1) that produces a believable mistype →
        backspace → correct-key self-correction.  This is the preferred path when
        an application should see genuine keystrokes rather than a clipboard
        paste (chat boxes, editors, terminals, anti-paste fields).
        """
        guard = self._guard("type_text")
        if guard:
            return guard
        try:
            text = SecuritySanitiser.guard_text(str(text or ""), "control.type_text")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not text:
            return ToolResult("No text supplied to type.", ok=False)
        self._narrate_typing(text)
        if human:
            return self._type_human(text, wpm=wpm, typos=typos, jitter=jitter, seed=seed)
        try:
            if self._kb is not None:
                # SendInput with KEYEVENTF_UNICODE — handles any character.
                self._kb.write(text, delay=max(0.0, interval))
            else:
                if not text.isascii():
                    return ToolResult("Unicode typing requires the keyboard backend.", ok=False)
                self._pg.typewrite(text, interval=max(0.0, interval))
        except Exception as exc:
            return ToolResult(f"Typing failed: {exc}. Not repeated; some text may have landed.",
                              ok=False)
        self._record("type_text", None, f"typed {len(text)} chars")
        return ToolResult(f"Typed {len(text)} character(s).")

    def _press_backspace(self) -> None:
        from .execution_policy import require_foreground
        deferred = require_foreground("keyboard_input")
        if deferred is not None:
            raise RuntimeError(deferred.text)
        if self._kb is not None:
            self._kb.send("backspace")
        elif self._pg is not None:
            self._pg.press("backspace")
        else:
            raise RuntimeError("No keyboard backend available")

    def _send_char(self, ch: str) -> None:
        from .execution_policy import require_foreground
        deferred = require_foreground("keyboard_input")
        if deferred is not None:
            raise RuntimeError(deferred.text)
        if self._kb is not None:
            self._kb.write(ch)
        elif self._pg is not None:
            if not ch.isascii():
                raise RuntimeError("Unicode typing requires the keyboard backend")
            self._pg.typewrite(ch)
        else:
            raise RuntimeError("No keyboard backend available")

    def _type_human(self, text: str, *, wpm: float, typos: float,
                    jitter: float, seed: int | None) -> ToolResult:
        from .human_typing import TypingProfile, plan_keystrokes
        profile = TypingProfile(wpm=wpm, jitter=jitter, typo_rate=typos, seed=seed)
        plan = plan_keystrokes(text, profile)
        corrections = 0
        for stroke in plan:
            guard = self._guard("type_text")
            if guard:
                return guard
            try:
                if stroke.kind == "backspace":
                    self._press_backspace()
                else:
                    if stroke.kind == "typo":
                        corrections += 1
                    self._send_char(stroke.char)
            except Exception as exc:
                return ToolResult(f"Typing failed: {exc}. Not repeated; some text may have landed.",
                                  ok=False)
            if stroke.delay > 0:
                time.sleep(stroke.delay)
        detail = f"human-typed {len(text)} chars at ~{int(wpm)} wpm"
        if corrections:
            detail += f", {corrections} self-correction(s)"
        self._record("type_text", None, detail)
        return ToolResult(f"Typed {len(text)} character(s) naturally at ~{int(wpm)} wpm"
                          + (f" with {corrections} self-correction(s)." if corrections else "."))

    #: Longest text typed visibly by edit_text; beyond it the fast verified
    #: paths run (a page of text is not something anyone wants to watch).
    VISIBLE_TYPING_MAX = 1500

    def edit_text(self, text: str, title: str = "", replace: bool = False,
                  interval: float = 0.02, visible: bool = False) -> ToolResult:
        """
        Reliably write text into an editable control (Notepad, editors, forms)
        and VERIFY it landed before claiming success (Phase 8, Section C —
        hardened after the 'tttttttttte ....' incident):

            1. NATIVE UIA (pywinauto)  — set the Edit/Document control's value
               directly.  Immune to focus races and key-rate glitches.
            2. CLIPBOARD INJECTION     — copy → focus → paste.  Unicode-safe.
               The user's clipboard is restored only AFTER verification, never
               mid-paste (the old 80 ms restore raced Win11 Notepad's async
               paste and could inject the PREVIOUS clipboard contents).
            3. KEYBOARD SIMULATION     — SendInput at a sane key rate.  The old
               pywinauto ``type_keys(pause=0.0)`` fallback is gone: firing keys
               with zero pause into Win11's Notepad drops and repeats
               characters, which is exactly how 'The tttttttttte…' happened.

        After each method the control is read back through UIA and the write
        is only reported as a success when the text is actually there.  A
        garbled attempt is undone (Ctrl+Z) before the next method runs.
        """
        guard = self._guard("edit_text")
        if guard:
            return guard
        try:
            text = SecuritySanitiser.guard_text(str(text or ""), "control.edit_text")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not text:
            return ToolResult("No text supplied to edit.", ok=False)
        self._narrate_typing(text, where=title)

        title = str(title or "").strip()
        # Always bring the target to the foreground first — the root cause of
        # the old Notepad failures was typing into an unfocused window.
        focused_title = ""
        if title:
            win = self._find_window(title)
            if win is None and getattr(self, "_window_candidates", ()):
                return self._missing_window_result(title)
            if win is not None:
                try:
                    if getattr(win, "isMinimized", False):
                        win.restore()
                    win.activate()
                except Exception:
                    self._win32_foreground(win)
                focused_title = getattr(win, "title", "") or title
                time.sleep(0.25)   # let the window settle into focus

        self._clipboard_backup: Optional[str] = None
        before = self._read_editable_text(title)     # None ⇒ no read-back path
        attempts: list[tuple[str, Any]] = [
            ("native UIA", lambda: self._edit_via_uia(text, title, replace)),
        ]
        # Keystroke-based methods go to whichever window HAS FOCUS — sending
        # them while the target is not foreground would paste into whatever
        # the user is working in.  Only add them once focus is confirmed.
        foreground_ok = True
        if title:
            foreground_ok = fold_title(title) in fold_title(self._foreground_title())
            if not foreground_ok:
                win = self._find_window(title)
                if win is not None:
                    self._win32_foreground(win)
                    time.sleep(0.3)
                foreground_ok = fold_title(title) in fold_title(self._foreground_title())
        if foreground_ok:
            attempts += [
                ("clipboard paste", lambda: self._edit_via_clipboard(text, replace)),
                ("keyboard", lambda: self._edit_via_keyboard(text, replace, interval)),
            ]
            if visible and len(text) <= self.VISIBLE_TYPING_MAX:
                # Seen, not pasted: the user watches the words arrive the way
                # a person types them. Still read back and verified like every
                # other method, and undone before the fast paths if garbled.
                attempts.insert(0, ("visible typing",
                                    lambda: self._edit_via_human(text, replace)))
        else:
            self.bus.log.emit(
                f"CONTROL: '{title}' is not the foreground window — keystroke "
                "paths withheld so nothing lands in the wrong application.")
        method: Optional[str] = None
        verified = False
        failures: list[str] = []
        try:
            for name, attempt in attempts:
                guard = self._guard("edit_text")
                if guard:
                    return guard
                try:
                    executed = attempt()
                except Exception as exc:
                    return ToolResult(f"Text entry stopped during {name}: {exc}. "
                                      "Not repeated; inspect the current text before continuing.",
                                      ok=False)
                if not executed:
                    continue
                readback = self._read_editable_text(title)
                if readback is None:
                    # No way to inspect the control — accept the write but
                    # say so honestly rather than claiming verification.
                    method, verified = name, False
                    break
                if self._text_landed(text, readback):
                    method, verified = name, True
                    break
                failures.append(name)
                if not self._undo_garbled(before, title):
                    return ToolResult(
                        f"Text entry via {name} did not match, and the previous text "
                        "could not be safely restored. Stopped before trying another method.",
                        ok=False,
                    )
        finally:
            # Restore the user's clipboard only now — the paste (if any) has
            # long since been processed, so this can never race it.
            if self._clipboard_backup is not None:
                try:
                    import pyperclip  # type: ignore
                    pyperclip.copy(self._clipboard_backup)
                except Exception:
                    pass
                self._clipboard_backup = None

        target = f" into '{focused_title}'" if focused_title else ""
        detail = (f"edit {len(text)} chars via {method}"
                  + (" (verified)" if verified else "")
                  + (f" after {len(failures)} garbled attempt(s)" if failures else ""))
        self._record("edit_text", None, detail + (f" into '{focused_title}'" if focused_title else ""))
        if method and verified:
            return ToolResult(
                f"Wrote and verified {len(text)} character(s){target} via the {method} path.")
        if method:
            return ToolResult(
                f"Text entry was attempted{target} via the {method} path, "
                "but the control could not be read back. Check the text before "
                "continuing; completion is unverified.", ok=False)
        return ToolResult(
            "Could not write the text: every method either failed or produced "
            "output that did not match (each attempt was undone). "
            + (f"Tried: {', '.join(failures)}." if failures else ""),
            ok=False,
        )

    # ── edit_text helpers ─────────────────────────────────────────────────────

    def _uia_window(self, desktop: Any, title: str) -> Any:
        window = None
        if title:
            folded = fold_title(title)
            matches = []
            for w in desktop.windows():
                try:
                    if folded in fold_title(w.window_text() or ""):
                        matches.append(w)
                except Exception:
                    continue
            exact = [w for w in matches if fold_title(w.window_text() or "") == folded]
            candidates = exact or matches
            # A named target must never fall back to whichever app is active.
            return candidates[0] if len(candidates) == 1 else None
        if window is None:
            try:
                window = desktop.window(active_only=True)
            except Exception:
                return None
        return window

    @staticmethod
    def _uia_editable(window: Any) -> Any:
        """The real text surface inside a window (Notepad's Document, an Edit box).

        ``descendants(control_type=…)`` — the window objects from
        ``Desktop.windows()`` are UIAWrappers, which have NO ``child_window``
        method; the old code silently threw here on every call, which is why
        the 'native UIA' path never once fired."""
        for ctrl_type in ("Document", "Edit"):
            try:
                found = window.descendants(control_type=ctrl_type)
                if found:
                    return found[0]
            except Exception:
                continue
        return None

    @staticmethod
    @contextmanager
    def _uia_desktop():
        """UIA needs a COM apartment on every worker, not just at import time."""
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()
        try:
            from pywinauto import Desktop  # type: ignore
            yield Desktop(backend="uia")
        finally:
            pythoncom.CoUninitialize()

    def _read_editable_text(self, title: str) -> Optional[str]:
        """Best-effort read of the target control's current text (None = can't)."""
        try:
            with self._uia_desktop() as desktop:
                window = self._uia_window(desktop, title)
                if window is None:
                    return None
                control = self._uia_editable(window)
                if control is None:
                    return None
                for getter in (
                    lambda: control.get_value(),
                    lambda: control.iface_text.DocumentRange.GetText(-1),
                    lambda: control.window_text(),
                ):
                    try:
                        value = getter()
                        if isinstance(value, str):
                            return value
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    @staticmethod
    def _text_landed(expected: str, readback: str) -> bool:
        """True when the written text is actually present in the control."""
        def norm(s: str) -> str:
            return " ".join(str(s).split())
        expected_n, readback_n = norm(expected), norm(readback)
        if not expected_n:
            return True
        return expected_n in readback_n

    def _undo_garbled(self, before: Optional[str], title: str) -> bool:
        """Undo only with known prior text and a confirmed foreground target."""
        if before is None:
            return False
        for _ in range(4):
            current = self._read_editable_text(title)
            if current is None:
                return False
            if current == before:
                return True
            if not title or fold_title(title) not in fold_title(self._foreground_title()):
                return False
            if not self.send_hotkeys("ctrl+z").ok:
                return False
            time.sleep(0.15)
        return self._read_editable_text(title) == before

    def _edit_via_uia(self, text: str, title: str, replace: bool) -> bool:
        """Set/append an Edit or Document control's value through UIA.

        Focus-independent: the value is written through the control's own
        UIA patterns (EditWrapper set_edit_text, then ValuePattern SetValue),
        never through synthesized keystrokes."""
        from .execution_policy import require_foreground
        if require_foreground("edit_text_focus") is not None:
            return False
        try:
            with self._uia_desktop() as desktop:
                window = self._uia_window(desktop, title)
                if window is None:
                    return False
                control = self._uia_editable(window)
                if control is None:
                    return False
                try:
                    control.set_focus()
                except Exception:
                    pass
                existing = None
                if not replace:
                    for getter in (
                        lambda: control.get_value(),
                        lambda: control.iface_text.DocumentRange.GetText(-1),
                    ):
                        try:
                            value = getter()
                            if isinstance(value, str):
                                existing = value
                                break
                        except Exception:
                            continue
                    if existing is None:
                        # SetValue replaces everything; appending needs the
                        # original contents, otherwise use an insertion path.
                        return False
                desired = text if replace else (existing or "") + text
                try:
                    control.set_edit_text(desired)
                    return True
                except Exception:
                    pass
                try:
                    control.iface_value.SetValue(desired)
                    return True
                except Exception:
                    return False
        except Exception:
            return False

    def _edit_via_clipboard(self, text: str, replace: bool) -> bool:
        """Copy → (optional select-all) → paste.  Unicode-safe.

        The previous clipboard contents are parked in ``_clipboard_backup``;
        ``edit_text`` restores them AFTER verification.  Restoring 80 ms after
        Ctrl+V (the old behaviour) raced Win11 Notepad's asynchronous paste —
        the paste could pick up the RESTORED clipboard and write stale junk."""
        from .execution_policy import require_foreground
        if require_foreground("clipboard_write") is not None:
            return False
        try:
            import pyperclip  # type: ignore
        except Exception:
            return False
        try:
            try:
                self._clipboard_backup = pyperclip.paste()
            except Exception:
                self._clipboard_backup = None
            pyperclip.copy(text)
            # Confirm the copy actually took before pasting it anywhere.
            deadline = time.time() + 1.0
            while time.time() < deadline:
                try:
                    if pyperclip.paste() == text:
                        break
                except Exception:
                    pass
                time.sleep(0.05)
            else:
                return False
        except Exception:
            return False
        if replace:
            selected = self.send_hotkeys("ctrl+a")
            if not selected.ok:
                raise RuntimeError(selected.text)
            time.sleep(0.08)
        pasted = self.send_hotkeys("ctrl+v")
        if not pasted.ok:
            raise RuntimeError(pasted.text)
        time.sleep(0.45)   # let the target process the paste fully
        return True

    def _edit_via_human(self, text: str, replace: bool) -> bool:
        """Type at a natural, watchable pace (no deliberate typos: this is the
        path that must also be RELIABLE)."""
        from .human_typing import TypingProfile, natural_wpm, plan_keystrokes
        if self._kb is None and self._pg is None:
            return False
        if replace:
            selected = self.send_hotkeys("ctrl+a")
            if not selected.ok:
                raise RuntimeError(selected.text)
            time.sleep(0.08)
        profile = TypingProfile(wpm=natural_wpm(len(text)), jitter=0.3, typo_rate=0.0)
        for stroke in plan_keystrokes(text, profile):
            guard = self._guard("edit_text")
            if guard:
                raise RuntimeError(guard.text)
            self._send_char(stroke.char)
            if stroke.delay > 0:
                time.sleep(stroke.delay)
        return True

    def _edit_via_keyboard(self, text: str, replace: bool, interval: float) -> bool:
        """SendInput typing at a deliberate key rate — the last resort."""
        rate = max(0.01, float(interval))   # never zero: zero-pause typing is
        if replace:                          # what garbled Notepad in the past
            selected = self.send_hotkeys("ctrl+a")
            if not selected.ok:
                raise RuntimeError(selected.text)
            time.sleep(0.08)
        result = self.type_text(text, interval=rate)
        if not result.ok:
            raise RuntimeError(result.text)
        return True

    def send_hotkeys(self, keys: Any) -> ToolResult:
        """Accept 'ctrl+c', ['ctrl','c'] or ('ctrl','shift','esc')."""
        guard = self._guard("send_hotkeys")
        if guard:
            return guard
        if isinstance(keys, str):
            parts = [k.strip().lower() for k in keys.replace(" ", "").split("+") if k.strip()]
        else:
            parts = [str(k).strip().lower() for k in (keys or []) if str(k).strip()]
        if not parts:
            return ToolResult("No hotkey supplied.", ok=False)
        combo = "+".join(parts)
        self._narrate(f"pressing {combo.upper()}")
        sent = False
        if self._kb is not None:
            try:
                self._kb.send(combo)
                sent = True
            except Exception:
                sent = False
        if not sent:
            try:
                self._pg.hotkey(*parts)
                sent = True
            except Exception as exc:
                return ToolResult(f"Hotkey '{combo}' failed: {exc}", ok=False)
        self._record("send_hotkeys", None, f"hotkey {combo}")
        return ToolResult(f"Sent hotkey: {combo}.")

    # ── application launch / close (delegated) ────────────────────────────────

    # How long to wait for a launched app's window to actually appear before
    # returning — open_app() used to return the instant Popen/startfile fired,
    # so a chained action (type into it, click it) right after open_app was
    # racing the app's own startup time with no signal either way.
    OPEN_APP_WINDOW_TIMEOUT_S = 6.0
    OPEN_APP_POLL_INTERVAL_S = 0.5

    def open_application(self, app_name: str) -> ToolResult:
        guard = self._guard("open_application")
        if guard:
            return guard
        if self.desktop is None:
            return ToolResult("No DesktopAgent attached for application launch.", ok=False)
        before_titles: set[str] = set()
        try:
            before_titles = {fold_title(t) for t in self._windows().getAllTitles() if t and t.strip()}
        except Exception:
            pass   # pygetwindow unavailable — launch still proceeds, just unconfirmed
        self._narrate(f"opening {app_name}")
        result = self.desktop.open_app(app_name)
        self._record("open_application", None, f"open {app_name}", title_hint=app_name)
        # The agent itself now waits for the app's window; "started, no window
        # yet" is a launch that happened, so it gets this layer's own (new
        # window) wait too, before being reported as such.
        started_only = (result.receipt or {}).get("status") == "started_no_window"
        if not result.ok and not started_only:
            return result
        # Confirm the application actually chosen: "a browser" opens a window
        # titled "... Microsoft Edge", which never contains the word "browser".
        opened = app_name
        seen = ""
        for item in getattr(result, "evidence", None) or []:
            if isinstance(item, dict) and item.get("app"):
                opened = str(item["app"])
                seen = str(item.get("window") or "")
                break
        if seen:
            return ToolResult(f"{result.text} Window confirmed: '{seen}'.",
                              evidence=result.evidence)
        confirmed = self._wait_for_new_window(opened, before_titles)
        if confirmed:
            lead = result.text if result.ok else f"Opened {opened}."
            return ToolResult(f"{lead} Window confirmed: '{confirmed}'.")
        # Never "Opened X." without a window: what follows (typing, clicking)
        # would land in whatever has focus instead.
        lead = result.text if started_only else f"Started {opened}, but no window has appeared."
        return ToolResult(
            f"{lead} No new window was seen within {self.OPEN_APP_WINDOW_TIMEOUT_S:g}s — "
            "it may still be launching, be tray-only, or already have a window open.",
            ok=False, receipt={"status": "started_no_window", "started": True,
                               "window": False, "app": opened})

    def _wait_for_new_window(self, app_name: str, before_titles: set[str]) -> Optional[str]:
        """Poll for a window that wasn't present before the launch (or, failing
        that, one whose title contains the app name), so callers know whether
        the app actually opened before firing the next action."""
        # Any distinctive word of the name counts ("microsoft edge" -> "edge"):
        # vendor words alone would accept an unrelated new Microsoft window.
        words = [w for w in fold_title(app_name).split()
                 if len(w) > 2 and w not in {"microsoft", "windows", "google", "the", "app"}]
        tokens = words or fold_title(app_name).split()[:1]
        deadline = time.monotonic() + self.OPEN_APP_WINDOW_TIMEOUT_S
        while True:
            try:
                titles = [t for t in self._windows().getAllTitles() if t and t.strip()]
            except Exception:
                return None
            for title in titles:
                folded = fold_title(title)
                if folded not in before_titles and (not tokens or any(t in folded for t in tokens)):
                    return title
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.OPEN_APP_POLL_INTERVAL_S)

    def close_application(self, app_name: str) -> ToolResult:
        guard = self._guard("close_application", pointer=False)
        if guard:
            return guard
        if self.desktop is None:
            return ToolResult("No DesktopAgent attached for application close.", ok=False)
        result = self.desktop.close_app(app_name)
        self._record("close_application", None, f"close {app_name}")
        return result

    # ── window management ─────────────────────────────────────────────────────

    def _windows(self) -> Any:
        import pygetwindow  # type: ignore
        return pygetwindow

    def _find_window(self, title: str) -> Any:
        self._window_candidates = ()
        gw = self._windows()
        # Fold both sides — real titles can hide invisible Unicode (Edge's
        # "Microsoft​ Edge" contains a zero-width space).
        title = fold_title(title)
        if not title:
            return None
        windows = gw.getAllWindows()
        matches = [w for w in windows if title in fold_title(w.title)]
        if matches:
            exact = [w for w in matches if fold_title(w.title) == title]
            choices = exact or matches
            if len(choices) == 1:
                return choices[0]
            self._window_candidates = tuple(w.title for w in choices)
            return None
        # No title says it: find the window through its APPLICATION —
        # Spotify is titled with the song playing, and "the browser" is in no
        # title at all. Serves focus, resize, minimise and typing alike.
        finder = getattr(self.desktop, "_windows_of_app", None)
        if callable(finder):
            try:
                titled = [(int(w._hWnd), w.title) for w in windows
                          if getattr(w, "_hWnd", None) and (w.title or "").strip()]
                order = [hwnd for hwnd, _ in finder(title, titled)]
                by_handle = {int(w._hWnd): w for w in windows if getattr(w, "_hWnd", None)}
                choices = [by_handle[hwnd] for hwnd in dict.fromkeys(order) if hwnd in by_handle]
                if len(choices) == 1:
                    return choices[0]
                if choices:
                    self._window_candidates = tuple(w.title for w in choices)
            except Exception:
                pass
        return None

    def _missing_window_result(self, title: str) -> ToolResult:
        candidates = getattr(self, "_window_candidates", ())
        if candidates:
            return ToolResult("Several windows match. Which window do you mean?\n"
                              + "\n".join(f"- {name}" for name in candidates[:8]), ok=False)
        return ToolResult(f"No window matches '{title}'.", ok=False)

    def list_windows(self) -> ToolResult:
        try:
            gw = self._windows()
            titles = [t for t in gw.getAllTitles() if t and t.strip()]
        except Exception as exc:
            return ToolResult(f"Window enumeration failed: {exc}", ok=False)
        return ToolResult("\n".join(titles[:60]) or "No titled windows.")

    def focus_window(self, title: str) -> ToolResult:
        guard = self._guard("focus_window", pointer=False)
        if guard:
            return guard
        win = self._find_window(title)
        if win is None:
            if getattr(self, "_window_candidates", ()):
                return self._missing_window_result(title)
            # "Switch to Spotify" (titled with the song) or "the browser" (in
            # no title): find the window through the application instead.
            if self.desktop is not None and hasattr(self.desktop, "window_control"):
                result = self.desktop.window_control("focus", title)
                if result.ok:
                    self._record("focus_window", None, f"focus {title}", title_hint=title)
                    return result
            return ToolResult(f"No window matches '{title}'.", ok=False)
        self._narrate(f"switching to {win.title[:50]}", region=self._window_region(win))
        try:
            if getattr(win, "isMinimized", False):
                win.restore()
            win.activate()
        except Exception:
            self._win32_foreground(win)
        for _ in range(3):
            try:
                active = win.isActive
            except Exception:
                active = fold_title(win.title) == fold_title(self._foreground_title())
            if active:
                break
            time.sleep(0.1)
        else:
            return ToolResult(f"Could not confirm focus on '{win.title}'.", ok=False)
        self._record("focus_window", self._window_region(win), f"focus {win.title}",
                     title_hint=win.title)
        return ToolResult(f"Focused window: {win.title}")

    @physical_pixels
    def resize_window(self, title: str, width: int, height: int) -> ToolResult:
        guard = self._guard("resize_window", pointer=False)
        if guard:
            return guard
        win = self._find_window(title)
        if win is None:
            return self._missing_window_result(title)
        try:
            width, height = max(120, int(width)), max(80, int(height))
            win.resizeTo(width, height)
        except Exception as exc:
            return ToolResult(f"Resize failed: {exc}", ok=False)
        region = self._read_window_until(lambda: self._window_region(win),
                                        lambda r: r is not None and r[2:] == (width, height))
        if region is None or region[2:] != (width, height):
            actual = f"{region[2]}x{region[3]}" if region else "unavailable"
            return ToolResult(f"Could not confirm resize of '{win.title}' to {width}x{height}; "
                              f"actual size: {actual}. The request was not repeated.", ok=False)
        self._record("resize_window", region, f"resize {win.title}")
        return ToolResult(f"Resized '{win.title}' to {width}x{height}.")

    @physical_pixels
    def move_window(self, title: str, x: int, y: int, monitor: int | None = None) -> ToolResult:
        guard = self._guard("move_window", pointer=False)
        if guard:
            return guard
        win = self._find_window(title)
        if win is None:
            return self._missing_window_result(title)
        try:
            vx, vy = self._resolve(x, y, monitor)
        except (TypeError, ValueError, OverflowError) as exc:
            return ToolResult(f"Invalid window target: {exc}", ok=False)
        try:
            win.moveTo(int(vx), int(vy))
        except Exception as exc:
            return ToolResult(f"Move failed: {exc}", ok=False)
        region = self._read_window_until(lambda: self._window_region(win),
                                        lambda r: r is not None and r[:2] == (vx, vy))
        if region is None or region[:2] != (vx, vy):
            actual = f"({region[0]}, {region[1]})" if region else "unavailable"
            return ToolResult(f"Could not confirm move of '{win.title}' to ({vx}, {vy}); "
                              f"actual position: {actual}. The request was not repeated.", ok=False)
        self._record("move_window", region, f"move {win.title}")
        return ToolResult(f"Moved '{win.title}' to ({vx}, {vy}).")

    def minimise_window(self, title: str) -> ToolResult:
        guard = self._guard("minimise_window", pointer=False)
        if guard:
            return guard
        win = self._find_window(title)
        if win is None:
            return self._missing_window_result(title)
        try:
            win.minimize()
        except Exception as exc:
            return ToolResult(f"Minimise failed: {exc}", ok=False)
        if self._read_window_until(lambda: bool(win.isMinimized), lambda state: state is True) is not True:
            return ToolResult(f"Could not confirm that '{win.title}' was minimised. "
                              "The request was not repeated.", ok=False)
        self._record("minimise_window", None, f"minimise {win.title}")
        return ToolResult(f"Minimised '{win.title}'.")

    def maximise_window(self, title: str) -> ToolResult:
        guard = self._guard("maximise_window", pointer=False)
        if guard:
            return guard
        win = self._find_window(title)
        if win is None:
            return self._missing_window_result(title)
        try:
            win.maximize()
        except Exception as exc:
            return ToolResult(f"Maximise failed: {exc}", ok=False)
        if self._read_window_until(lambda: bool(win.isMaximized), lambda state: state is True) is not True:
            return ToolResult(f"Could not confirm that '{win.title}' was maximised. "
                              "The request was not repeated.", ok=False)
        self._record("maximise_window", self._window_region(win), f"maximise {win.title}")
        return ToolResult(f"Maximised '{win.title}'.")

    def switch_application(self, title: str) -> ToolResult:
        """Bring an app to the foreground (focus, restoring if minimised)."""
        return self.focus_window(title)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _read_window_until(read: Callable[[], Any], matches: Callable[[Any], bool]) -> Any:
        """Allow OS state to settle, polling only; never replay a mutation."""
        actual = None
        for attempt in range(3):
            try:
                actual = read()
                if matches(actual):
                    return actual
            except Exception:
                actual = None
            if attempt < 2:
                time.sleep(0.05)
        return actual

    def _resolve(self, x: int, y: int, monitor: int | None, *,
                 exact: bool = False) -> tuple[int, int]:
        if exact and hasattr(self.display, "resolve_point"):
            return self.display.resolve_point(int(x), int(y), monitor)
        if monitor is not None:
            vx, vy = self.display.to_virtual(monitor, int(x), int(y))
        else:
            vx, vy = int(x), int(y)
        return self.display.clamp_to_desktop(vx, vy)

    @physical_pixels
    def _window_region(self, win: Any) -> Optional[tuple[int, int, int, int]]:
        try:
            return (int(win.left), int(win.top), int(win.width), int(win.height))
        except Exception:
            return None

    def _win32_foreground(self, win: Any) -> None:
        from .execution_policy import require_foreground
        if require_foreground("focus_window") is not None:
            return
        try:
            hwnd = getattr(win, "_hWnd", None)
            if hwnd:
                from .agents import force_foreground
                force_foreground(int(hwnd))
        except Exception:
            pass

    def _foreground_title(self) -> str:
        """Title of the window that actually has the foreground right now."""
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if not hwnd:
                return ""
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 2)
            ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value or ""
        except Exception:
            return ""

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pyautogui": self._pg is not None,
            "keyboard": self._kb is not None,
            "last_action": self.last_action.get("detail", ""),
        }


def _foreground_rect() -> Optional[tuple[int, int, int, int]]:
    """The focused window's rectangle in physical pixels, or None."""
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        rect = wintypes.RECT()
        if not hwnd or not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 40 or height < 40:
            return None
        return (int(rect.left), int(rect.top), int(width), int(height))
    except Exception:
        return None
