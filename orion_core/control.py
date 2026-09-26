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

    # ── showing his working ───────────────────────────────────────────────────

    GLIDE_MIN_S = 0.15
    GLIDE_MAX_S = 0.55
    GLIDE_PX_PER_S = 1800.0
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

    def _travel(self, vx: int, vy: int) -> None:
        """Move the pointer to (vx, vy): an eased glide when showing working,
        so the eye can follow it across the desktop and between monitors."""
        if not self.show_working:
            self._pg.moveTo(vx, vy)
            return
        try:
            cx, cy = self.display.cursor_position()
            distance = math.hypot(vx - cx, vy - cy)
        except Exception:
            distance = 600.0
        seconds = min(self.GLIDE_MAX_S, max(self.GLIDE_MIN_S, distance / self.GLIDE_PX_PER_S))
        tween = getattr(self._pg, "easeInOutQuad", None)
        if tween is not None:
            self._pg.moveTo(vx, vy, duration=seconds, tween=tween)
        else:
            self._pg.moveTo(vx, vy, duration=seconds)

    def _region_around(self, vx: int, vy: int, pad: int = 90) -> tuple[int, int, int, int]:
        x0, y0 = self.display.clamp_to_desktop(vx - pad, vy - pad)
        x1, y1 = self.display.clamp_to_desktop(vx + pad, vy + pad)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # ── cursor ────────────────────────────────────────────────────────────────

    @contextmanager
    def _hand_the_mouse_back(self):
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
                    if self.show_working:
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

    def move_cursor(self, x: int, y: int, monitor: int | None = None,
                    duration: float = 0.0) -> ToolResult:
        guard = self._guard("move_cursor")
        if guard:
            return guard
        vx, vy = self._resolve(x, y, monitor)
        self._narrate("moving here", x=vx, y=vy)
        if duration > 0 or not self.show_working:
            self._pg.moveTo(vx, vy, duration=max(0.0, duration))
        else:
            self._travel(vx, vy)
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
        with self._hand_the_mouse_back():
            self._narrate("dragging", x=sx, y=sy)
            self._travel(sx, sy)
            self._pg.dragTo(ex, ey, duration=max(0.05, duration), button=button)
        self._record("drag_cursor", self._region_around(ex, ey), f"drag ({sx},{sy})→({ex},{ey})")
        return ToolResult(f"Dragged from ({sx},{sy}) to ({ex},{ey}).")

    def click(self, x: int | None = None, y: int | None = None, button: str = "left",
              clicks: int = 1, monitor: int | None = None) -> ToolResult:
        guard = self._guard("click")
        if guard:
            return guard
        if x is None or y is None:
            # No coordinates: he is clicking wherever the pointer already is,
            # so there is nothing to restore and nothing to take away.
            vx, vy = self.display.cursor_position()
            self._narrate(self._click_words(button, clicks), x=vx, y=vy)
            self._pg.click(vx, vy, clicks=max(1, clicks), interval=0.05, button=button)
        else:
            vx, vy = self._resolve(x, y, monitor)
            with self._hand_the_mouse_back():
                self._narrate(self._click_words(button, clicks), x=vx, y=vy)
                self._travel(vx, vy)
                self._pg.click(vx, vy, clicks=max(1, clicks), interval=0.05,
                               button=button)
        self._record("click", self._region_around(vx, vy),
                     f"{button} click x{clicks} @ ({vx},{vy})", x=vx, y=vy, button=button)
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
        way = "scrolling down" if int(amount) < 0 else "scrolling up"
        if x is not None and y is not None:
            vx, vy = self._resolve(x, y, monitor)
            with self._hand_the_mouse_back():
                self._narrate(way, x=vx, y=vy)
                self._travel(vx, vy)
                self._pg.scroll(int(amount))
        else:
            vx, vy = self.display.cursor_position()
            self._narrate(way, x=vx, y=vy)
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
            self._travel(vx, vy)
        else:
            vx, vy = self.display.cursor_position()
        self._narrate("scrolling down" if int(amount) < 0 else "scrolling up", x=vx, y=vy)
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

    def _press_backspace(self) -> None:
        if self._kb is not None:
            try:
                self._kb.send("backspace")
                return
            except Exception:
                pass
        if self._pg is not None:
            try:
                self._pg.press("backspace")
            except Exception:
                pass

    def _send_char(self, ch: str) -> None:
        if self._kb is not None:
            try:
                self._kb.write(ch)
                return
            except Exception:
                pass
        if self._pg is not None:
            try:
                self._pg.write(ch)
            except Exception:
                pass

    def _type_human(self, text: str, *, wpm: float, typos: float,
                    jitter: float, seed: int | None) -> ToolResult:
        from .human_typing import TypingProfile, plan_keystrokes
        profile = TypingProfile(wpm=wpm, jitter=jitter, typo_rate=typos, seed=seed)
        plan = plan_keystrokes(text, profile)
        corrections = 0
        for stroke in plan:
            if stroke.kind == "backspace":
                self._press_backspace()
            else:
                if stroke.kind == "typo":
                    corrections += 1
                self._send_char(stroke.char)
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
                try:
                    executed = attempt()
                except Exception:
                    executed = False
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
                self._undo_garbled(before, title)
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
                f"Wrote {len(text)} character(s){target} via the {method} path "
                "(the control could not be read back for verification).")
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
            for w in desktop.windows():
                try:
                    if folded in fold_title(w.window_text() or ""):
                        window = w
                        break
                except Exception:
                    continue
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

    def _read_editable_text(self, title: str) -> Optional[str]:
        """Best-effort read of the target control's current text (None = can't)."""
        try:
            from pywinauto import Desktop  # type: ignore
        except Exception:
            return None
        try:
            desktop = Desktop(backend="uia")
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
        head = expected_n[:120]
        tail = expected_n[-60:] if len(expected_n) > 120 else ""
        return head in readback_n and (not tail or tail in readback_n)

    def _undo_garbled(self, before: Optional[str], title: str) -> None:
        """Roll a failed write attempt back with Ctrl+Z (bounded, best-effort)."""
        for _ in range(4):
            self.send_hotkeys("ctrl+z")
            time.sleep(0.15)
            if before is not None:
                current = self._read_editable_text(title)
                if current is not None and " ".join(current.split()) == " ".join(before.split()):
                    return

    def _edit_via_uia(self, text: str, title: str, replace: bool) -> bool:
        """Set/append an Edit or Document control's value through UIA.

        Focus-independent: the value is written through the control's own
        UIA patterns (EditWrapper set_edit_text, then ValuePattern SetValue),
        never through synthesized keystrokes."""
        try:
            from pywinauto import Desktop  # type: ignore
        except Exception:
            return False
        try:
            desktop = Desktop(backend="uia")
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
            existing = ""
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
            if replace:
                self.send_hotkeys("ctrl+a")
                time.sleep(0.08)
            self.send_hotkeys("ctrl+v")
            time.sleep(0.45)   # let the target process the paste fully
            return True
        except Exception:
            return False

    def _edit_via_human(self, text: str, replace: bool) -> bool:
        """Type at a natural, watchable pace (no deliberate typos: this is the
        path that must also be RELIABLE)."""
        from .human_typing import TypingProfile, natural_wpm, plan_keystrokes
        if self._kb is None and self._pg is None:
            return False
        if replace:
            self.send_hotkeys("ctrl+a")
            time.sleep(0.08)
        profile = TypingProfile(wpm=natural_wpm(len(text)), jitter=0.3, typo_rate=0.0)
        for stroke in plan_keystrokes(text, profile):
            self._send_char(stroke.char)
            if stroke.delay > 0:
                time.sleep(stroke.delay)
        return True

    def _edit_via_keyboard(self, text: str, replace: bool, interval: float) -> bool:
        """SendInput typing at a deliberate key rate — the last resort."""
        rate = max(0.01, float(interval))   # never zero: zero-pause typing is
        if replace:                          # what garbled Notepad in the past
            self.send_hotkeys("ctrl+a")
            time.sleep(0.08)
        if self._kb is not None:
            try:
                self._kb.write(text, delay=rate)
                return True
            except Exception:
                pass
        if self._pg is not None:
            try:
                self._pg.typewrite(text, interval=rate)
                return True
            except Exception:
                pass
        return False

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
        if not result.ok:
            return result
        confirmed = self._wait_for_new_window(app_name, before_titles)
        if confirmed:
            return ToolResult(f"{result.text} Window confirmed: '{confirmed}'.")
        return ToolResult(
            f"{result.text} No new window was seen within "
            f"{self.OPEN_APP_WINDOW_TIMEOUT_S:g}s — it may still be launching, "
            "be tray-only, or already have a window open."
        )

    def _wait_for_new_window(self, app_name: str, before_titles: set[str]) -> Optional[str]:
        """Poll for a window that wasn't present before the launch (or, failing
        that, one whose title contains the app name), so callers know whether
        the app actually opened before firing the next action."""
        token = fold_title(app_name).split(" ")[0] if app_name.strip() else ""
        deadline = time.monotonic() + self.OPEN_APP_WINDOW_TIMEOUT_S
        while True:
            try:
                titles = [t for t in self._windows().getAllTitles() if t and t.strip()]
            except Exception:
                return None
            for title in titles:
                folded = fold_title(title)
                if folded not in before_titles and (not token or token in folded):
                    return title
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.OPEN_APP_POLL_INTERVAL_S)

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
        self._narrate(f"switching to {win.title[:50]}", region=self._window_region(win))
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
