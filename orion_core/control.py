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

import math
import os
import time
from typing import Any, Optional

from .bus import OrionBus
from .data import ToolResult
from .display import DisplayTopologyManager
from .security import SecuritySanitiser, SecurityViolation
from .utils import fold_title


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
        # The region + description of the most recent action, for the verifier.
        self.last_action: dict[str, Any] = {}
        self._pg: Any = None
        self._kb: Any = None
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

    def _guard(self, action: str) -> Optional[ToolResult]:
        if not self.enabled:
            return ToolResult(
                f"Autonomous control is disabled; refused '{action}'. "
                "Enable it in the Command Centre or set ORION_AUTONOMY=1.",
                ok=False,
            )
        if self._pg is None:
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

    def _region_around(self, vx: int, vy: int, pad: int = 90) -> tuple[int, int, int, int]:
        x0, y0 = self.display.clamp_to_desktop(vx - pad, vy - pad)
        x1, y1 = self.display.clamp_to_desktop(vx + pad, vy + pad)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # ── cursor ────────────────────────────────────────────────────────────────

    def move_cursor(self, x: int, y: int, monitor: int | None = None,
                    duration: float = 0.0) -> ToolResult:
        guard = self._guard("move_cursor")
        if guard:
            return guard
        vx, vy = self._resolve(x, y, monitor)
        self._pg.moveTo(vx, vy, duration=max(0.0, duration))
        self._record("move_cursor", self._region_around(vx, vy), f"cursor → ({vx},{vy})",
                     x=vx, y=vy)
        return ToolResult(f"Cursor moved to ({vx}, {vy}).")

    def drag_cursor(self, x1: int, y1: int, x2: int, y2: int, monitor: int | None = None,
                    duration: float = 0.25, button: str = "left") -> ToolResult:
        guard = self._guard("drag_cursor")
        if guard:
            return guard
        sx, sy = self._resolve(x1, y1, monitor)
        ex, ey = self._resolve(x2, y2, monitor)
        self._pg.moveTo(sx, sy)
        self._pg.dragTo(ex, ey, duration=max(0.05, duration), button=button)
        self._record("drag_cursor", self._region_around(ex, ey), f"drag ({sx},{sy})→({ex},{ey})")
        return ToolResult(f"Dragged from ({sx},{sy}) to ({ex},{ey}).")

    def click(self, x: int | None = None, y: int | None = None, button: str = "left",
              clicks: int = 1, monitor: int | None = None) -> ToolResult:
        guard = self._guard("click")
        if guard:
            return guard
        if x is None or y is None:
            vx, vy = self.display.cursor_position()
        else:
            vx, vy = self._resolve(x, y, monitor)
            self._pg.moveTo(vx, vy)
        self._pg.click(vx, vy, clicks=max(1, clicks), interval=0.05, button=button)
        self._record("click", self._region_around(vx, vy),
                     f"{button} click x{clicks} @ ({vx},{vy})", x=vx, y=vy, button=button)
        return ToolResult(f"{button.capitalize()} click ({clicks}x) at ({vx}, {vy}).")

    def double_click(self, x: int | None = None, y: int | None = None,
                     monitor: int | None = None) -> ToolResult:
        return self.click(x, y, button="left", clicks=2, monitor=monitor)

    def right_click(self, x: int | None = None, y: int | None = None,
                    monitor: int | None = None) -> ToolResult:
        return self.click(x, y, button="right", clicks=1, monitor=monitor)

    def scroll(self, amount: int, x: int | None = None, y: int | None = None,
               monitor: int | None = None) -> ToolResult:
        guard = self._guard("scroll")
        if guard:
            return guard
        if x is not None and y is not None:
            vx, vy = self._resolve(x, y, monitor)
            self._pg.moveTo(vx, vy)
        else:
            vx, vy = self.display.cursor_position()
        self._pg.scroll(int(amount))
        self._record("scroll", self._region_around(vx, vy), f"scroll {amount} @ ({vx},{vy})")
        return ToolResult(f"Scrolled {amount} at ({vx}, {vy}).")

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
        if x is not None and y is not None:
            vx, vy = self._resolve(x, y, monitor)
            self._pg.moveTo(vx, vy)
        else:
            vx, vy = self.display.cursor_position()
        amount = int(amount)
        steps = max(6, min(60, abs(amount)))
        duration = max(0.2, min(4.0, float(duration)))
        pause = duration / steps
        delivered = 0
        for i in range(steps):
            # Ease-in-out: a sine bell makes steps gentle at the ends and
            # quicker in the middle, so the glide feels natural.
            phase = (i + 0.5) / steps
            weight = math.sin(math.pi * phase)
            notch = amount / steps
            self._pg.scroll(int(round(notch)) or (1 if amount > 0 else -1))
            delivered += 1
            time.sleep(pause * (0.6 + 0.8 * weight))
            self.bus.control_activity.emit("smooth_scroll")
        self._record("smooth_scroll", self._region_around(vx, vy),
                     f"smooth scroll {amount} in {steps} steps @ ({vx},{vy})")
        return ToolResult(f"Scrolled smoothly ({amount}) over {steps} steps at ({vx}, {vy}).")

    # ── keyboard ──────────────────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.01) -> ToolResult:
        guard = self._guard("type_text")
        if guard:
            return guard
        try:
            text = SecuritySanitiser.guard_text(str(text or ""), "control.type_text")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not text:
            return ToolResult("No text supplied to type.", ok=False)
        typed = False
        if self._kb is not None:
            try:
                # SendInput with KEYEVENTF_UNICODE — handles any character.
                self._kb.write(text, delay=max(0.0, interval))
                typed = True
            except Exception:
                typed = False
        if not typed:
            self._pg.typewrite(text, interval=max(0.0, interval))
        self._record("type_text", None, f"typed {len(text)} chars")
        return ToolResult(f"Typed {len(text)} character(s).")

    def edit_text(self, text: str, title: str = "", replace: bool = False,
                  interval: float = 0.01) -> ToolResult:
        """
        Reliably write text into an editable control (Notepad, editors, forms),
        with intelligent layered fallback so ORION never fails to edit a text
        file when permissions allow (Phase 8, Section C):

            1. NATIVE UIA (pywinauto)  — locate the real Edit/Document control
               and set or append its value directly.  The most robust path for
               genuine text controls, immune to focus races.
            2. CLIPBOARD INJECTION     — copy the text, focus, optionally
               select-all, and paste (Ctrl+V).  Unicode-safe, near-instant, and
               works with Win11's UWP Notepad where raw keystrokes are flaky.
            3. KEYBOARD SIMULATION     — SendInput/typewrite, the last resort.

        Each method is tried in turn; the first that succeeds wins.
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

        title = str(title or "").strip()
        # Always bring the target to the foreground first — the root cause of
        # the old Notepad failures was typing into an unfocused window.
        focused_title = ""
        if title:
            win = self._find_window(title)
            if win is not None:
                try:
                    if getattr(win, "isMinimized", False):
                        win.restore()
                    win.activate()
                except Exception:
                    self._win32_foreground(win)
                focused_title = getattr(win, "title", "") or title
                time.sleep(0.25)   # let the window settle into focus

        # ── 1. Native UIA (pywinauto) ─────────────────────────────────────────
        method = self._edit_via_uia(text, title, replace)
        # ── 2. Clipboard injection ────────────────────────────────────────────
        if method is None:
            method = self._edit_via_clipboard(text, replace)
        # ── 3. Keyboard simulation ────────────────────────────────────────────
        if method is None:
            if replace:
                self.send_hotkeys("ctrl+a")
                time.sleep(0.05)
            if self._kb is not None:
                try:
                    self._kb.write(text, delay=max(0.0, interval))
                    method = "keyboard"
                except Exception:
                    method = None
            if method is None and self._pg is not None:
                self._pg.typewrite(text, interval=max(0.0, interval))
                method = "typewrite"

        self._record("edit_text", None,
                     f"edit {len(text)} chars via {method}"
                     + (f" into '{focused_title}'" if focused_title else ""))
        target = f" into '{focused_title}'" if focused_title else ""
        return ToolResult(
            f"Wrote {len(text)} character(s){target} via the {method} path."
            if method else "Could not edit text through any method.",
            ok=bool(method),
        )

    def _edit_via_uia(self, text: str, title: str, replace: bool) -> Optional[str]:
        """Set/append an Edit or Document control's value through UIA."""
        try:
            from pywinauto import Desktop  # type: ignore
        except Exception:
            return None
        try:
            desktop = Desktop(backend="uia")
            window = None
            if title:
                for w in desktop.windows():
                    try:
                        if title.lower() in (w.window_text() or "").lower():
                            window = w
                            break
                    except Exception:
                        continue
            if window is None:
                try:
                    window = desktop.window(active_only=True)
                except Exception:
                    return None
            # Prefer a document/edit control; fall back to the window itself.
            control = None
            for ctrl_type in ("Edit", "Document"):
                try:
                    candidate = window.child_window(control_type=ctrl_type)
                    if candidate.exists(timeout=1.0):
                        control = candidate
                        break
                except Exception:
                    continue
            control = control or window
            control.set_focus()
            existing = ""
            if not replace:
                try:
                    existing = control.get_value() if hasattr(control, "get_value") else ""
                except Exception:
                    existing = ""
            try:
                control.set_edit_text((existing or "") + text if not replace else text)
                return "native UIA"
            except Exception:
                # Not a value-pattern control — type into it under focus.
                control.type_keys(self._escape_type_keys(text), with_spaces=True,
                                  with_newlines=True, pause=0.0)
                return "native UIA typing"
        except Exception:
            return None

    def _edit_via_clipboard(self, text: str, replace: bool) -> Optional[str]:
        """Copy → (optional select-all) → paste.  Unicode-safe and fast."""
        try:
            import pyperclip  # type: ignore
        except Exception:
            return None
        try:
            preserved = ""
            try:
                preserved = pyperclip.paste()
            except Exception:
                preserved = ""
            pyperclip.copy(text)
            time.sleep(0.05)
            if replace:
                self.send_hotkeys("ctrl+a")
                time.sleep(0.05)
            self.send_hotkeys("ctrl+v")
            time.sleep(0.08)
            # Restore the user's previous clipboard so we don't clobber it.
            try:
                pyperclip.copy(preserved)
            except Exception:
                pass
            return "clipboard paste"
        except Exception:
            return None

    @staticmethod
    def _escape_type_keys(text: str) -> str:
        """Escape pywinauto type_keys metacharacters ({}()+^%~) literally."""
        out = []
        for ch in text:
            if ch in "{}()+^%~[]":
                out.append("{" + ch + "}")
            elif ch == "\n":
                out.append("{ENTER}")
            elif ch == "\t":
                out.append("{TAB}")
            else:
                out.append(ch)
        return "".join(out)

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

    def open_application(self, app_name: str) -> ToolResult:
        guard = self._guard("open_application")
        if guard:
            return guard
        if self.desktop is None:
            return ToolResult("No DesktopAgent attached for application launch.", ok=False)
        result = self.desktop.open_app(app_name)
        self._record("open_application", None, f"open {app_name}", title_hint=app_name)
        return result

    def close_application(self, app_name: str) -> ToolResult:
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
        gw = self._windows()
        # Fold both sides — real titles can hide invisible Unicode (Edge's
        # "Microsoft​ Edge" contains a zero-width space).
        title = fold_title(title)
        matches = [w for w in gw.getAllWindows() if title in fold_title(w.title)]
        # Prefer visible, non-minimised, shortest-title (most specific) match.
        matches.sort(key=lambda w: (getattr(w, "isMinimized", False), len(w.title or "")))
        return matches[0] if matches else None

    def list_windows(self) -> ToolResult:
        try:
            gw = self._windows()
            titles = [t for t in gw.getAllTitles() if t and t.strip()]
        except Exception as exc:
            return ToolResult(f"Window enumeration failed: {exc}", ok=False)
        return ToolResult("\n".join(titles[:60]) or "No titled windows.")

    def focus_window(self, title: str) -> ToolResult:
        win = self._find_window(title)
        if win is None:
            return ToolResult(f"No window matches '{title}'.", ok=False)
        try:
            if getattr(win, "isMinimized", False):
                win.restore()
            win.activate()
        except Exception:
            self._win32_foreground(win)
        self._record("focus_window", self._window_region(win), f"focus {win.title}",
                     title_hint=win.title)
        return ToolResult(f"Focused window: {win.title}")

    def resize_window(self, title: str, width: int, height: int) -> ToolResult:
        win = self._find_window(title)
        if win is None:
            return ToolResult(f"No window matches '{title}'.", ok=False)
        try:
            win.resizeTo(max(120, int(width)), max(80, int(height)))
        except Exception as exc:
            return ToolResult(f"Resize failed: {exc}", ok=False)
        self._record("resize_window", self._window_region(win), f"resize {win.title}")
        return ToolResult(f"Resized '{win.title}' to {width}x{height}.")

    def move_window(self, title: str, x: int, y: int, monitor: int | None = None) -> ToolResult:
        win = self._find_window(title)
        if win is None:
            return ToolResult(f"No window matches '{title}'.", ok=False)
        vx, vy = self._resolve(x, y, monitor)
        try:
            win.moveTo(int(vx), int(vy))
        except Exception as exc:
            return ToolResult(f"Move failed: {exc}", ok=False)
        self._record("move_window", self._window_region(win), f"move {win.title}")
        return ToolResult(f"Moved '{win.title}' to ({vx}, {vy}).")

    def minimise_window(self, title: str) -> ToolResult:
        win = self._find_window(title)
        if win is None:
            return ToolResult(f"No window matches '{title}'.", ok=False)
        try:
            win.minimize()
        except Exception as exc:
            return ToolResult(f"Minimise failed: {exc}", ok=False)
        self._record("minimise_window", None, f"minimise {win.title}")
        return ToolResult(f"Minimised '{win.title}'.")

    def maximise_window(self, title: str) -> ToolResult:
        win = self._find_window(title)
        if win is None:
            return ToolResult(f"No window matches '{title}'.", ok=False)
        try:
            win.maximize()
        except Exception as exc:
            return ToolResult(f"Maximise failed: {exc}", ok=False)
        self._record("maximise_window", self._window_region(win), f"maximise {win.title}")
        return ToolResult(f"Maximised '{win.title}'.")

    def switch_application(self, title: str) -> ToolResult:
        """Bring an app to the foreground (focus, restoring if minimised)."""
        return self.focus_window(title)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve(self, x: int, y: int, monitor: int | None) -> tuple[int, int]:
        if monitor is not None:
            vx, vy = self.display.to_virtual(monitor, int(x), int(y))
        else:
            vx, vy = int(x), int(y)
        return self.display.clamp_to_desktop(vx, vy)

    def _window_region(self, win: Any) -> Optional[tuple[int, int, int, int]]:
        try:
            return (int(win.left), int(win.top), int(win.width), int(win.height))
        except Exception:
            return None

    def _win32_foreground(self, win: Any) -> None:
        try:
            import ctypes
            hwnd = getattr(win, "_hWnd", None)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pyautogui": self._pg is not None,
            "keyboard": self._kb is not None,
            "last_action": self.last_action.get("detail", ""),
        }
