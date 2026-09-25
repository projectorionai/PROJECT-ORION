"""
Headless tests for the AutonomousControlLayer (Improvement Pass, Priority 1.1):
coordinate clamping, the ORION_AUTONOMY kill-switch, and the pyautogui
FAILSAFE/PAUSE configuration. The real input backends are replaced with
recording fakes before construction so no test ever moves the actual cursor
or types into the host.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.control import AutonomousControlLayer


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
    """A 2-monitor virtual desktop spanning x 0..5119, y 0..1439."""

    def clamp_to_desktop(self, vx, vy):
        return (min(max(int(vx), 0), 5119), min(max(int(vy), 0), 1439))

    def cursor_position(self):
        return (50, 60)

    def to_virtual(self, monitor, x, y):
        return (int(x) + 2560 * int(monitor), int(y))


class _FakePG(types.ModuleType):
    """Stands in for pyautogui; records every call, moves nothing."""

    def __init__(self):
        super().__init__("pyautogui")
        self.PAUSE = None
        self.FAILSAFE = None
        self.calls = []

    def moveTo(self, x, y, duration=0.0):
        self.calls.append(("moveTo", x, y))

    def click(self, x, y, clicks=1, interval=0.0, button="left"):
        self.calls.append(("click", x, y, clicks, button))

    def dragTo(self, x, y, duration=0.0, button="left"):
        self.calls.append(("dragTo", x, y, button))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def typewrite(self, text, interval=0.0):
        self.calls.append(("typewrite", text))


class _FakeKB(types.ModuleType):
    def __init__(self):
        super().__init__("keyboard")
        self.written = []

    def write(self, text, delay=0.0):
        self.written.append(text)


@pytest.fixture
def layer(monkeypatch):
    """A control layer whose backends are recording fakes."""
    fake_pg, fake_kb = _FakePG(), _FakeKB()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    monkeypatch.setitem(sys.modules, "keyboard", fake_kb)
    monkeypatch.delenv("ORION_AUTONOMY", raising=False)
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    assert built._pg is fake_pg and built._kb is fake_kb
    return built


# ── backend configuration: FAILSAFE stays armed, PAUSE stays off ─────────────

def test_failsafe_and_pause_configuration(layer):
    assert layer._pg.FAILSAFE is True   # corner-abort must remain a hard stop
    assert layer._pg.PAUSE == 0.0       # pacing is deliberate, not global


def test_missing_pyautogui_degrades_gracefully(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyautogui", None)  # import → TypeError
    monkeypatch.setitem(sys.modules, "keyboard", None)
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    assert built._pg is None
    result = built.move_cursor(10, 10)
    assert not result.ok and "offline" in result.text


# ── ORION_AUTONOMY kill-switch ────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_kill_switch_env_disables_all_actions(monkeypatch, value):
    fake_pg = _FakePG()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    monkeypatch.setenv("ORION_AUTONOMY", value)
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    assert built.enabled is False
    for action in (
        lambda: built.move_cursor(5, 5),
        lambda: built.click(5, 5),
        lambda: built.drag_cursor(0, 0, 5, 5),
        lambda: built.scroll(3),
        lambda: built.type_text("hello"),
    ):
        result = action()
        assert not result.ok and "disabled" in result.text.lower()
    assert fake_pg.calls == []          # the backend was never touched


@pytest.mark.parametrize("value", ["1", "true", "on", ""])
def test_kill_switch_defaults_to_enabled(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "pyautogui", _FakePG())
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    if value:
        monkeypatch.setenv("ORION_AUTONOMY", value)
    else:
        monkeypatch.delenv("ORION_AUTONOMY", raising=False)
    assert AutonomousControlLayer(_StubBus(), _StubDisplay()).enabled is True


def test_set_enabled_runtime_toggle(layer):
    layer.set_enabled(False)
    assert not layer.move_cursor(5, 5).ok
    assert layer._pg.calls == []
    layer.set_enabled(True)
    assert layer.move_cursor(5, 5).ok
    assert ("moveTo", 5, 5) in layer._pg.calls


# ── coordinate clamping ───────────────────────────────────────────────────────

def test_move_clamps_to_visible_desktop(layer):
    layer.move_cursor(999999, -500)
    assert ("moveTo", 5119, 0) in layer._pg.calls


def test_click_clamps_and_records_region(layer):
    result = layer.click(-100, 999999)
    assert result.ok
    assert ("click", 0, 1439, 1, "left") in layer._pg.calls
    x, y, w, h = layer.last_action["region"]
    assert x >= 0 and y >= 0 and w >= 1 and h >= 1
    assert x + w <= 5120 and y + h <= 1440


def test_monitor_relative_coordinates_resolve_to_virtual(layer):
    layer.move_cursor(100, 200, monitor=1)
    assert ("moveTo", 2660, 200) in layer._pg.calls


def test_drag_clamps_both_endpoints(layer):
    layer.drag_cursor(-50, -50, 999999, 999999)
    assert ("moveTo", 0, 0) in layer._pg.calls
    assert ("dragTo", 5119, 1439, "left") in layer._pg.calls


def test_click_without_coordinates_uses_cursor_position(layer):
    layer.click()
    assert ("click", 50, 60, 1, "left") in layer._pg.calls


# ── typed text passes through the security firewall ──────────────────────────

def test_type_text_blocks_destructive_payloads(layer):
    result = layer.type_text("rm -rf /")
    assert not result.ok
    assert layer._kb.written == [] and layer._pg.calls == []


def test_type_text_blocks_core_mutation(layer):
    result = layer.type_text("del orion_core")
    assert not result.ok
    assert layer._kb.written == []


def test_type_text_benign_goes_to_keyboard_backend(layer):
    result = layer.type_text("hello world")
    assert result.ok
    assert layer._kb.written == ["hello world"]


def test_type_text_falls_back_to_pyautogui(layer):
    layer._kb = None
    layer.type_text("fallback text")
    assert ("typewrite", "fallback text") in layer._pg.calls


def test_empty_type_text_refused(layer):
    assert not layer.type_text("").ok


# ── action metadata for the verifier ─────────────────────────────────────────

def test_actions_record_last_action_metadata(layer):
    layer.click(100, 100)
    meta = layer.last_action
    assert meta["action"] == "click"
    assert meta["x"] == 100 and meta["y"] == 100
    assert meta["region"] is not None
