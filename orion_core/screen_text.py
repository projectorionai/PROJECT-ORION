"""
Read what is on screen EXACTLY — the text itself, not a picture of it.

  "ORION's OCR engine must be enhanced to a point to where he can see anything
   on the screen in absolute accuracy - this includes anything that I'm typing
   on in the desktop such as notepad, word, google chrome, Microsoft edge and
   literally any app - even in games"

OCR, however good, is a guess at what pixels say. Almost every application
already knows its own text and will hand it over through Windows UI
Automation — the interface screen readers use. Notepad, Word, Outlook,
Explorer, WhatsApp, Discord, VS Code, Edge and Chrome all answer it. Read that
way, a document comes back character-perfect, including the parts scrolled
out of view, in a fraction of a second.

What it cannot reach, OCR still covers: games and anything drawn straight to
the GPU expose no text at all, and some canvases draw their own. So this is
the FIRST reader, never the only one (see vision.py).

Chromium keeps its page accessibility tree switched off until an assistive
technology asks for it — Edge, Chrome and every Electron app answered with
nothing but their address bar. A standard ``AccessibleObjectFromWindow`` call
on the page's render window is that request; MEASURED on this machine it took
Edge/Chrome/VS Code/Discord from 0 characters to the full page (up to 340k)
in about 0.1 s.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field
from typing import Any

#: Most characters returned from one window. A long document is still read in
#: full up to this; beyond it the reply would drown whatever asked.
MAX_CHARS = 60_000

#: Seconds to spend walking one window's control tree.
WALK_BUDGET_S = 3.0

_CHROMIUM_CLASSES = {"Chrome_WidgetWin_1", "Chrome_WidgetWin_0"}
_ENABLED_HWNDS: set[int] = set()


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    pid: int
    process: str = ""

    @property
    def app(self) -> str:
        name = os.path.splitext(self.process)[0].lower() if self.process else ""
        return {"msedge": "Microsoft Edge", "chrome": "Google Chrome",
                "winword": "Microsoft Word", "notepad": "Notepad",
                "excel": "Microsoft Excel", "powerpnt": "PowerPoint",
                "code": "Visual Studio Code", "explorer": "File Explorer",
                "outlook": "Outlook", "olk": "Outlook", "discord": "Discord",
                "firefox": "Firefox"}.get(name, self.process or self.class_name)


@dataclass
class WindowText:
    window: WindowInfo
    text: str = ""
    focused: str = ""
    selection: str = ""
    method: str = ""
    truncated: bool = False
    extras: list[str] = field(default_factory=list)

    def render(self, limit: int = MAX_CHARS) -> str:
        head = f"{self.window.app} — \"{self.window.title}\""
        parts = [head, f"(read exactly via {self.method})" if self.method else ""]
        if self.selection:
            parts.append(f"SELECTED TEXT:\n{self.selection[:4000]}")
        if self.focused and self.focused not in self.text:
            parts.append(f"FOCUSED FIELD:\n{self.focused[:4000]}")
        if self.text:
            body = self.text[:limit]
            parts.append("TEXT:\n" + body + ("\n…[truncated]" if len(self.text) > limit
                                               or self.truncated else ""))
        elif self.extras:
            parts.append("VISIBLE LABELS:\n" + "\n".join(self.extras[:300]))
        return "\n".join(p for p in parts if p)


def available() -> bool:
    if os.name != "nt":
        return False
    try:
        import uiautomation  # noqa: F401
        return True
    except Exception:
        return False


# ── windows ──────────────────────────────────────────────────────────────────

def _process_name(pid: int) -> str:
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:
        return ""


def list_windows() -> list[WindowInfo]:
    """Visible, titled top-level windows, front-most first."""
    if os.name != "nt":
        return []
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[WindowInfo] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _l: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        klass = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, klass, 256)
        if klass.value in {"Progman", "WorkerW", "Shell_TrayWnd"}:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found.append(WindowInfo(int(hwnd), title.value, klass.value, int(pid.value)))
        return True

    user32.EnumWindows(proc(callback), 0)   # z-order: front-most first
    for info in found:
        info.process = _process_name(info.pid)
    return found


def foreground_window() -> WindowInfo | None:
    if os.name != "nt":
        return None
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    for info in list_windows():
        if info.hwnd == hwnd:
            return info
    windows = list_windows()
    return windows[0] if windows else None


def find_window(name: str) -> WindowInfo | None:
    """The front-most window whose title, app or process matches *name*."""
    wanted = str(name or "").strip().lower()
    if not wanted:
        return foreground_window()
    aliases = {"edge": "msedge", "word": "winword", "powerpoint": "powerpnt",
               "vs code": "code", "vscode": "code", "file explorer": "explorer"}
    wanted_proc = aliases.get(wanted, wanted)
    windows = list_windows()
    for info in windows:
        if wanted_proc == os.path.splitext(info.process)[0].lower():
            return info
    for info in windows:
        if wanted in info.title.lower() or wanted in info.app.lower():
            return info
    return None


# ── Chromium ─────────────────────────────────────────────────────────────────

def enable_chromium_accessibility(hwnd: int) -> bool:
    """Ask a Chromium window (Edge, Chrome, Electron) to build its page tree."""
    if os.name != "nt":
        return False
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    oleacc = ctypes.windll.oleacc
    targets: list[int] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(child: int, _l: int) -> bool:
        klass = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(child, klass, 256)
        if klass.value == "Chrome_RenderWidgetHostHWND":
            targets.append(int(child))
        return True

    user32.EnumChildWindows(hwnd, proc(callback), 0)
    fresh = [t for t in targets if t not in _ENABLED_HWNDS]
    if not fresh:
        return bool(targets)

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                    ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

    # IID_IAccessible {618736E0-3C3D-11CF-810C-00AA00389B71}
    iid = _GUID(0x618736E0, 0x3C3D, 0x11CF,
                (ctypes.c_ubyte * 8)(0x81, 0x0C, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71))
    for target in fresh:
        pointer = ctypes.c_void_p()
        try:
            oleacc.AccessibleObjectFromWindow(target, -4, ctypes.byref(iid),   # OBJID_CLIENT
                                              ctypes.byref(pointer))
            _ENABLED_HWNDS.add(target)
        except Exception:
            continue
    time.sleep(0.35)        # the tree is built asynchronously
    return True


# ── reading ──────────────────────────────────────────────────────────────────

_TEXT_TYPES = {"DocumentControl", "EditControl"}
_LABEL_TYPES = {"TextControl", "ButtonControl", "HyperlinkControl", "ListItemControl",
                "TabItemControl", "MenuItemControl", "HeaderItemControl",
                "DataItemControl", "TreeItemControl", "CheckBoxControl"}


def read_window(window: WindowInfo | None = None, *, max_chars: int = MAX_CHARS,
                budget: float = WALK_BUDGET_S) -> WindowText | None:
    """The exact text of *window* (default: the one in front), or None."""
    if not available():
        return None
    import uiautomation as auto

    window = window or foreground_window()
    if window is None:
        return None
    result = WindowText(window=window, method="Windows UI Automation")
    if window.class_name in _CHROMIUM_CLASSES:
        enable_chromium_accessibility(window.hwnd)
    with auto.UIAutomationInitializerInThread():
        try:
            control = auto.ControlFromHandle(window.hwnd)
        except Exception:
            control = None
        if control is None:
            return None
        deadline = time.monotonic() + budget
        best = ""
        labels: list[str] = []
        seen_labels: set[str] = set()
        try:
            for ctrl, _depth in auto.WalkControl(control, maxDepth=40):
                if time.monotonic() > deadline:
                    result.truncated = True
                    break
                kind = ctrl.ControlTypeName
                if kind in _TEXT_TYPES:
                    text = _pattern_text(ctrl, max_chars)
                    if len(text) > len(best):
                        best = text
                    if kind == "DocumentControl" and len(best) > 2000:
                        break      # the document is the thing; stop walking
                elif kind in _LABEL_TYPES and len(labels) < 400:
                    name = (ctrl.Name or "").strip()
                    if name and name not in seen_labels and len(name) < 300:
                        seen_labels.add(name)
                        labels.append(name)
        except Exception:
            pass
        result.text = best
        result.extras = labels
        try:
            focused = auto.GetFocusedControl()
            if focused is not None and _belongs_to(focused, window.pid):
                result.focused = _pattern_text(focused, 8000) or _value(focused)
                result.selection = _selection(focused)
        except Exception:
            pass
    if not (result.text or result.extras or result.focused):
        return None
    return result


def _belongs_to(ctrl: Any, pid: int) -> bool:
    try:
        return int(ctrl.ProcessId) == int(pid)
    except Exception:
        return False


def _pattern_text(ctrl: Any, max_chars: int) -> str:
    try:
        pattern = ctrl.GetTextPattern()
        if pattern is not None:
            return str(pattern.DocumentRange.GetText(max_chars) or "")
    except Exception:
        pass
    return _value(ctrl)


def _value(ctrl: Any) -> str:
    try:
        pattern = ctrl.GetValuePattern()
        if pattern is not None:
            return str(pattern.Value or "")
    except Exception:
        pass
    return ""


def _selection(ctrl: Any) -> str:
    try:
        pattern = ctrl.GetTextPattern()
        if pattern is None:
            return ""
        ranges = pattern.GetSelection() or []
        return " ".join(str(r.GetText(4000) or "") for r in ranges).strip()
    except Exception:
        return ""


def read_screen(windows: int = 3, max_chars: int = 20_000) -> list[WindowText]:
    """The exact text of the front-most few windows — "what's on my screen"."""
    out: list[WindowText] = []
    for info in list_windows()[: max(1, windows) * 2]:
        if len(out) >= windows:
            break
        text = read_window(info, max_chars=max_chars, budget=2.0)
        if text is not None:
            out.append(text)
    return out


__all__ = ["MAX_CHARS", "WindowInfo", "WindowText", "available",
           "enable_chromium_accessibility", "find_window", "foreground_window",
           "list_windows", "read_screen", "read_window"]
