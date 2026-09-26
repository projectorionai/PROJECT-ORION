"""
Tests for the desktop-control robustness fixes:

    * open_application() used to return the instant Popen/os.startfile fired,
      racing a chained action against the app's own startup time — it now
      polls (bounded) for a new window before returning.
    * desktop_control's visual-verification-with-retry machinery
      (_maybe_verify) used to cover only click/double_click — it now also
      covers type_text, drag, right_click and window management, using the
      SAME existing VisualVerificationEngine rather than a new one.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.control import AutonomousControlLayer
from orion_core.data import ToolResult


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubDisplay:
    def clamp_to_desktop(self, vx, vy):
        return (min(max(int(vx), 0), 5119), min(max(int(vy), 0), 1439))

    def cursor_position(self):
        return (50, 60)

    def to_virtual(self, monitor, x, y):
        return (int(x) + 2560 * int(monitor), int(y))


class _FakeWindow:
    def __init__(self, title: str):
        self.title = title
        self.isMinimized = False

    def restore(self): pass
    def activate(self): pass
    def resizeTo(self, w, h): pass
    def moveTo(self, x, y): pass
    def minimize(self): self.isMinimized = True
    def maximize(self): pass


class _FakeGetWindow(types.ModuleType):
    """Stands in for pygetwindow — getAllTitles()/getAllWindows() return
    whatever the test set."""

    def __init__(self):
        super().__init__("pygetwindow")
        self.titles: list[str] = []

    def getAllTitles(self):
        return list(self.titles)

    def getAllWindows(self):
        return [_FakeWindow(t) for t in self.titles]


class _FakePG(types.ModuleType):
    def __init__(self):
        super().__init__("pyautogui")
        self.PAUSE = None
        self.FAILSAFE = None

    def moveTo(self, x, y, duration=0.0): pass
    def click(self, x, y, clicks=1, interval=0.0, button="left"): pass
    def dragTo(self, x, y, duration=0.0, button="left"): pass
    def scroll(self, amount): pass
    def typewrite(self, text, interval=0.0): pass


class _FakeKB(types.ModuleType):
    def __init__(self):
        super().__init__("keyboard")

    def write(self, text, delay=0.0): pass


class _StubDesktopAgent:
    def __init__(self, result: ToolResult):
        self._result = result
        self.calls: list[str] = []

    def open_app(self, app_name: str) -> ToolResult:
        self.calls.append(app_name)
        return self._result


def _layer(monkeypatch, desktop, fake_gw) -> AutonomousControlLayer:
    monkeypatch.setitem(sys.modules, "pyautogui", _FakePG())
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.delenv("ORION_AUTONOMY", raising=False)
    layer = AutonomousControlLayer(_StubBus(), _StubDisplay(), desktop=desktop)
    layer.OPEN_APP_WINDOW_TIMEOUT_S = 0.3   # keep the test fast
    layer.OPEN_APP_POLL_INTERVAL_S = 0.05
    return layer


# ── open_application: wait-for-window ───────────────────────────────────────

def test_open_application_confirms_a_new_matching_window(monkeypatch):
    fake_gw = _FakeGetWindow()
    fake_gw.titles = ["Existing Window"]
    desktop = _StubDesktopAgent(ToolResult("Opened application: notepad."))
    layer = _layer(monkeypatch, desktop, fake_gw)

    def _open_then_appear(app_name):
        fake_gw.titles = ["Existing Window", "Untitled - Notepad"]
        return desktop._result
    desktop.open_app = _open_then_appear

    result = layer.open_application("notepad")
    assert result.ok
    assert "Window confirmed" in result.text
    assert "Notepad" in result.text


def test_open_application_times_out_without_crashing_when_no_window_appears(monkeypatch):
    fake_gw = _FakeGetWindow()
    fake_gw.titles = ["Existing Window"]
    desktop = _StubDesktopAgent(ToolResult("Opened application: trayapp."))
    layer = _layer(monkeypatch, desktop, fake_gw)

    result = layer.open_application("trayapp")
    assert result.ok   # the launch itself succeeded — window just unconfirmed
    assert "No new window was seen" in result.text


def test_open_application_does_not_poll_when_the_launch_itself_failed(monkeypatch):
    fake_gw = _FakeGetWindow()
    desktop = _StubDesktopAgent(ToolResult("Unable to open application 'bogus'.", ok=False))
    layer = _layer(monkeypatch, desktop, fake_gw)

    poll_calls = []
    original = layer._wait_for_new_window
    def _spy(*a, **k):
        poll_calls.append(1)
        return original(*a, **k)
    layer._wait_for_new_window = _spy

    result = layer.open_application("bogus")
    assert not result.ok
    assert poll_calls == []


def test_open_application_refuses_when_autonomy_disabled(monkeypatch):
    fake_gw = _FakeGetWindow()
    desktop = _StubDesktopAgent(ToolResult("should not be reached"))
    monkeypatch.setitem(sys.modules, "pyautogui", types.ModuleType("pyautogui"))
    monkeypatch.setitem(sys.modules, "keyboard", types.ModuleType("keyboard"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setenv("ORION_AUTONOMY", "0")
    layer = AutonomousControlLayer(_StubBus(), _StubDisplay(), desktop=desktop)
    result = layer.open_application("notepad")
    assert not result.ok
    assert desktop.calls == []


# ── desktop_control: expanded verification coverage ─────────────────────────

class _FakeVerifier:
    def __init__(self):
        self.calls: list[str] = []
        self.kwargs: list[dict] = []

    async def verify_action(self, call, **kwargs):
        self.calls.append("verify_action")
        self.kwargs.append(kwargs)
        result = await asyncio.to_thread(call) if not asyncio.iscoroutinefunction(call) else await call()
        class _Outcome:
            def to_tool_result(_self):
                return result
        return _Outcome()


def _dispatcher(control, verifier):
    import types as t
    d = t.SimpleNamespace()
    d.control = control
    d.verifier = verifier
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    d._maybe_verify = DesktopDispatchMixin._maybe_verify.__get__(d)
    # desktop_control is the wrapper that captions the pointer with the call's
    # reason; the actions themselves live in _desktop_control.
    d._desktop_control = DesktopDispatchMixin._desktop_control.__get__(d)
    d.desktop_control = DesktopDispatchMixin.desktop_control.__get__(d)
    return d


def test_type_text_routes_through_verification_when_verifier_attached(monkeypatch):
    fake_gw = _FakeGetWindow()
    layer = _layer(monkeypatch, _StubDesktopAgent(ToolResult("x")), fake_gw)
    verifier = _FakeVerifier()
    d = _dispatcher(layer, verifier)
    asyncio.run(d.desktop_control({"action": "type_text", "text": "hello"}))
    assert verifier.calls == ["verify_action"]
    # Typing must never be retried by re-typing: an unchanged screen used to
    # make the verifier run the action again and type the text twice.
    assert verifier.kwargs[0].get("repeatable") is False


def test_drag_routes_through_verification_when_verifier_attached(monkeypatch):
    fake_gw = _FakeGetWindow()
    layer = _layer(monkeypatch, _StubDesktopAgent(ToolResult("x")), fake_gw)
    verifier = _FakeVerifier()
    d = _dispatcher(layer, verifier)
    asyncio.run(d.desktop_control({"action": "drag", "x1": 0, "y1": 0, "x2": 10, "y2": 10}))
    assert verifier.calls == ["verify_action"]
    assert verifier.kwargs[0].get("repeatable") is False


def test_minimise_window_routes_through_verification_when_verifier_attached(monkeypatch):
    fake_gw = _FakeGetWindow()
    layer = _layer(monkeypatch, _StubDesktopAgent(ToolResult("x")), fake_gw)
    verifier = _FakeVerifier()
    d = _dispatcher(layer, verifier)
    asyncio.run(d.desktop_control({"action": "minimise_window", "title": "Notepad"}))
    assert verifier.calls == ["verify_action"]


def test_type_text_skips_verification_when_verify_false(monkeypatch):
    fake_gw = _FakeGetWindow()
    layer = _layer(monkeypatch, _StubDesktopAgent(ToolResult("x")), fake_gw)
    verifier = _FakeVerifier()
    d = _dispatcher(layer, verifier)
    asyncio.run(d.desktop_control({"action": "type_text", "text": "hello", "verify": False}))
    assert verifier.calls == []


def test_type_text_works_without_a_verifier_attached(monkeypatch):
    fake_gw = _FakeGetWindow()
    layer = _layer(monkeypatch, _StubDesktopAgent(ToolResult("x")), fake_gw)
    d = _dispatcher(layer, None)
    result = asyncio.run(d.desktop_control({"action": "type_text", "text": "hello"}))
    assert result.ok
