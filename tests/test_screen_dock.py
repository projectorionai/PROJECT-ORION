"""
The docked Screen Face - where it sits, how it grows, when it steps aside.

Real widgets, offscreen. The work area is a fake screen (1920 x 1040, taskbar
below), the busy check is a list the test edits, and the face inside is the
real ScreenFacePanel (headless: no render thread). What is held:

  * it is a Tool window with no taskbar entry, always on top, frameless, and
    it can never take focus - a click must not pull the caret out of the
    document the person is typing into;
  * it sits flush on the right (or left) edge of the work area, its centre at
    70% of the height, and stays fully on screen;
  * it is about 1.5x larger while ORION listens or speaks, with a grace period
    so a turn is one swell and not three;
  * dragging moves it along the edge only, and a press that barely moves is a
    click that activates ORION;
  * it hides while a game or presentation owns the screen and returns after,
    but not for Do Not Disturb;
  * everything ORION's face protocol sends is forwarded to the face inside.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QEvent, QPointF, QRect, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import windows_presence as wp  # noqa: E402
from orion_core import windows_settings  # noqa: E402
from orion_core.gui.screen_dock import GROW_FACTOR, IDLE_SIZE, ScreenDock, dock_rect  # noqa: E402

AREA = QRect(0, 0, 1920, 1040)
SCREEN = SimpleNamespace(availableGeometry=lambda: QRect(AREA))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _own_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(windows_settings, "SETTINGS_PATH", tmp_path / "windows_integration.json")
    for key in windows_settings.DEFAULTS:
        monkeypatch.delenv(f"ORION_WIN_{key.upper()}", raising=False)


@pytest.fixture
def make(qapp):
    made: list[ScreenDock] = []

    def _make(**kwargs):
        kwargs.setdefault("screen_provider", lambda: SCREEN)
        kwargs.setdefault("animate", False)
        kwargs.setdefault("shrink_delay_ms", 0)
        kwargs.setdefault("busy_state", lambda: "available")
        dock = ScreenDock(**kwargs)
        made.append(dock)
        return dock

    yield _make
    for dock in made:
        dock.stop()
        dock.deleteLater()
    qapp.processEvents()


def _mouse(kind, local, global_, button=Qt.MouseButton.LeftButton, buttons=None):
    buttons = button if buttons is None else buttons
    args = (kind, QPointF(*local), QPointF(*global_), button, buttons, Qt.KeyboardModifier.NoModifier)
    try:
        return QMouseEvent(*args)
    except TypeError:                    # the scene-position overload
        return QMouseEvent(kind, QPointF(*local), QPointF(*local), QPointF(*global_),
                           button, buttons, Qt.KeyboardModifier.NoModifier)


# ══════════════════════════════════════════════════════════════════════════════
# WHERE IT SITS
# ══════════════════════════════════════════════════════════════════════════════

def test_the_dock_sits_flush_on_the_right_edge_at_seventy_percent():
    rect = dock_rect(AREA, 150, 120, "right", 0.70)
    assert (rect.width(), rect.height()) == (150, 120)
    assert rect.right() == AREA.right()                       # flush against the edge
    assert rect.center().y() == pytest.approx(0.70 * AREA.height(), abs=1)
    assert rect.top() == 668


def test_the_left_edge_and_an_offset_work_area():
    left = dock_rect(AREA, 150, 120, "left", 0.70)
    assert left.left() == 0 and left.top() == 668
    taskbar_left = QRect(48, 0, 1872, 1040)                   # a taskbar on the left of the screen
    assert dock_rect(taskbar_left, 150, 120, "left", 0.5).left() == 48
    assert dock_rect(taskbar_left, 150, 120, "right", 0.5).right() == 1919


def test_the_dock_is_kept_fully_on_screen():
    assert dock_rect(AREA, 150, 120, "right", 0.0).top() == 0
    assert dock_rect(AREA, 150, 120, "right", 1.0).bottom() == AREA.bottom()
    assert dock_rect(AREA, 150, 120, "right", -3).top() == 0
    assert dock_rect(AREA, 225, 180, "right", 5).bottom() == AREA.bottom()


def test_a_new_dock_is_placed_on_the_edge_at_its_idle_size(make):
    dock = make()
    assert (dock.width(), dock.height()) == (IDLE_SIZE.width(), IDLE_SIZE.height()) == (150, 120)
    assert dock.geometry() == dock_rect(AREA, 150, 120, "right", 0.70)
    assert dock.edge == "right" and dock.grown is False


def test_the_edge_comes_from_the_setting_and_can_change(make):
    dock = make(edge="left")
    assert dock.edge == "left" and dock.x() == 0
    dock.set_edge("right")
    assert dock.x() == AREA.right() + 1 - 150
    assert make(edge="sideways").edge == "right"              # unknown falls back to the default


def test_the_dock_follows_the_work_area_when_it_changes(make):
    area = [QRect(0, 0, 1920, 1040)]
    dock = make(screen_provider=lambda: SimpleNamespace(availableGeometry=lambda: QRect(area[0])))
    assert dock.x() == 1770
    area[0] = QRect(0, 0, 1280, 680)                          # a smaller display
    dock.redock()
    assert dock.geometry() == dock_rect(area[0], 150, 120, "right", 0.70)


def test_on_a_real_offscreen_screen_the_dock_is_inside_the_work_area(make, qapp):
    from PyQt6.QtGui import QGuiApplication

    dock = ScreenDock(animate=False, busy_state=lambda: "available")   # no fake screen
    try:
        area = QGuiApplication.primaryScreen().availableGeometry()
        assert area.contains(dock.geometry()) and dock.geometry().right() == area.right()
    finally:
        dock.stop()
        dock.deleteLater()


# ══════════════════════════════════════════════════════════════════════════════
# IT NEVER STEALS FOCUS
# ══════════════════════════════════════════════════════════════════════════════

def test_it_is_a_frameless_always_on_top_tool_window_that_refuses_focus(make):
    dock = make()
    flags = dock.windowFlags()
    assert dock.windowType() == Qt.WindowType.Tool                 # no taskbar button, no Alt-Tab entry
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    assert dock.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert dock.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert dock.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert dock.panel.focusPolicy() == Qt.FocusPolicy.NoFocus
    # The face inside lets clicks through to the dock, which owns drag and click.
    assert dock.panel.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert dock.accessibleName() and dock.toolTip()


def test_nothing_in_the_dock_activates_or_raises_a_window():
    """Showing, re-docking and un-hiding must never call activateWindow() or
    raise_() - either would pull focus from the application being typed into."""
    source = (ROOT / "orion_core" / "gui" / "screen_dock.py").read_text(encoding="utf-8")
    for call in ("activateWindow", "raise_(", ".setFocus(", "grabKeyboard"):
        assert call not in source, call


# ══════════════════════════════════════════════════════════════════════════════
# GROWING WHILE HE LISTENS AND SPEAKS
# ══════════════════════════════════════════════════════════════════════════════

def test_it_grows_one_and_a_half_times_while_listening_and_settles_after(make):
    dock = make()
    dock.set_state("LISTENING")
    assert dock.grown is True
    assert (dock.width(), dock.height()) == (225, 180)
    assert GROW_FACTOR == 1.5
    assert dock.geometry().right() == AREA.right()             # still flush on the edge
    assert dock.geometry().center().y() == pytest.approx(0.70 * 1040, abs=1)   # grows about its centre line
    dock.set_state("STANDBY")
    assert dock.grown is False and (dock.width(), dock.height()) == (150, 120)
    assert dock.geometry() == dock_rect(AREA, 150, 120, "right", 0.70)


def test_speaking_alone_makes_it_grow(make):
    dock = make()
    dock.set_state("STANDBY")
    dock.set_speaking(True)
    assert dock.grown and (dock.width(), dock.height()) == (225, 180)
    dock.set_speaking(False)
    assert not dock.grown


def test_processing_alone_does_not_grow_it(make):
    dock = make()
    dock.set_state("PROCESSING")
    assert not dock.grown
    dock.set_state("speaking")                                 # case does not matter
    assert dock.grown


def test_a_grace_period_stops_a_turn_flickering_between_sizes(make, qapp):
    dock = make(shrink_delay_ms=80)
    dock.set_state("LISTENING")
    dock.set_state("PROCESSING")                               # listening -> thinking
    assert dock.grown and dock._shrink.isActive()              # still big: shrinking is pending
    dock.set_state("SPEAKING")                                 # thinking -> speaking
    assert dock.grown and not dock._shrink.isActive()          # the pending shrink was cancelled
    dock.set_state("STANDBY")
    assert dock.grown
    QTest.qWait(250)
    assert not dock.grown and dock.width() == 150


def test_the_size_change_animates_when_visible_and_lands_on_the_target(make, qapp):
    dock = make(animate=True)
    dock.show_dock()
    dock.set_state("LISTENING")
    assert dock._animation is not None                         # moving, not jumping
    QTest.qWait(400)
    assert dock.geometry() == dock_rect(AREA, 225, 180, "right", 0.70)


# ══════════════════════════════════════════════════════════════════════════════
# STEPPING ASIDE FOR GAMES AND PRESENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_it_hides_for_a_fullscreen_app_or_presentation_and_returns_after(make):
    state = ["available"]
    dock = make(busy_state=lambda: state[0])
    dock.show_dock()
    assert dock.isVisible() and dock.wanted and not dock.busy_hidden

    state[0] = "fullscreen_app"
    dock._poll_busy()
    assert not dock.isVisible() and dock.busy_hidden and dock.wanted

    state[0] = "presentation"
    dock._poll_busy()
    assert not dock.isVisible()

    state[0] = "available"
    dock._poll_busy()
    assert dock.isVisible() and not dock.busy_hidden
    assert dock.geometry() == dock_rect(AREA, 150, 120, "right", 0.70)


def test_do_not_disturb_and_unknown_do_not_hide_it(make):
    state = ["quiet_hours"]
    dock = make(busy_state=lambda: state[0])
    dock.show_dock()
    assert dock.isVisible()
    for other in ("unknown", "busy", "quiet_hours"):
        state[0] = other
        dock._poll_busy()
        assert dock.isVisible(), other


def test_asked_to_show_during_a_game_it_waits_and_appears_when_the_game_ends(make):
    state = ["fullscreen_app"]
    dock = make(busy_state=lambda: state[0])
    dock.show_dock()
    assert not dock.isVisible() and dock.wanted and dock.busy_hidden
    state[0] = "available"
    dock._poll_busy()
    assert dock.isVisible()


def test_a_hidden_dock_does_not_reappear_on_its_own(make):
    dock = make()
    dock.show_dock()
    dock.hide_dock()
    assert not dock.isVisible() and not dock.wanted
    dock._poll_busy()
    assert not dock.isVisible()


def test_the_busy_check_is_polled_every_five_seconds_only_while_wanted(make):
    dock = make()
    assert not dock._poll.isActive()
    dock.show_dock()
    assert dock._poll.interval() == 5000 and dock._poll.isActive()
    dock.hide_dock()
    assert not dock._poll.isActive()


def test_a_busy_check_that_raises_leaves_the_dock_alone(make):
    def broken():
        raise OSError("shell gone")

    dock = make(busy_state=broken)
    dock.show_dock()
    assert dock.isVisible()
    dock._poll_busy()
    assert dock.isVisible()


def test_the_default_busy_check_is_the_real_one_but_is_not_called_here(make):
    dock = ScreenDock(animate=False, screen_provider=lambda: SCREEN)
    try:
        assert dock._busy_state is wp.user_busy_state
    finally:
        dock.stop()
        dock.deleteLater()


# ══════════════════════════════════════════════════════════════════════════════
# DRAG AND CLICK
# ══════════════════════════════════════════════════════════════════════════════

def test_a_press_that_barely_moves_is_a_click_that_activates_orion(make):
    clicked = []
    dock = make(on_activate=lambda: clicked.append(1))
    dock.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, (20, 20), (1800, 700)))
    dock.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, (22, 22), (1802, 702)))     # a jiggle
    dock.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, (22, 22), (1802, 702),
                                  buttons=Qt.MouseButton.NoButton))
    assert clicked == [1]
    assert dock.geometry() == dock_rect(AREA, 150, 120, "right", 0.70)           # did not move


def test_dragging_moves_it_along_the_edge_only_and_is_not_a_click(make):
    clicked = []
    dock = make(on_activate=lambda: clicked.append(1))
    x = dock.x()
    top = dock.y()
    dock.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, (20, 20), (1800, 700)))
    dock.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, (20, 120), (1700, 800)))     # 100 down, 100 left
    dock.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, (20, 120), (1700, 800),
                                  buttons=Qt.MouseButton.NoButton))
    assert clicked == []
    assert dock.x() == x                                       # stayed docked: no sideways travel
    assert dock.y() == top + 100
    assert dock.y_fraction == pytest.approx((top + 100 + 60) / 1040, abs=0.001)
    # The remembered position survives a resize.
    dock.set_state("LISTENING")
    assert dock.geometry().center().y() == pytest.approx(dock.y_fraction * 1040, abs=1)


def test_dragging_is_kept_on_screen(make):
    dock = make()
    dock.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, (20, 20), (1800, 700)))
    dock.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, (0, 0), (1800, 5000)))
    assert dock.geometry().bottom() == AREA.bottom()
    dock.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, (0, 0), (1800, -5000)))
    assert dock.y() == 0
    dock.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, (0, 0), (1800, -5000),
                                  buttons=Qt.MouseButton.NoButton))


def test_a_right_click_press_is_not_a_click(make):
    clicked = []
    dock = make(on_activate=lambda: clicked.append(1))
    dock.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, (5, 5), (1800, 700),
                                button=Qt.MouseButton.RightButton))
    dock.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, (5, 5), (1800, 700),
                                  button=Qt.MouseButton.RightButton, buttons=Qt.MouseButton.NoButton))
    assert clicked == []


def test_a_raising_activation_callback_cannot_escape_the_event_handler(make):
    def boom():
        raise RuntimeError("lead's bug")

    dock = make(on_activate=boom)
    dock.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, (5, 5), (1800, 700)))
    dock.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, (5, 5), (1800, 700),
                                  buttons=Qt.MouseButton.NoButton))


# ══════════════════════════════════════════════════════════════════════════════
# THE MENU
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def all_forms(monkeypatch):
    """The dormant code path: every face enabled (as a developer would have it)."""
    from orion_core import appearance

    monkeypatch.setattr(appearance, "ENABLED_FORMS", appearance.FACE_FORMS)


def test_the_menu_offers_only_show_and_hide_with_the_screen_face_alone(make):
    """Mark XXXVII: Classic Face and Orb are gone from the right-click menu, and
    with one face there is nothing to choose, so no face entries at all."""
    dock = make(face_form="orb")
    menu = dock.build_menu()
    assert [a.text() for a in menu.actions()] == ["Show ORION", "", "Hide Dock"]
    texts = " ".join(a.text() for a in menu.actions())
    assert "Classic" not in texts and "Orb" not in texts


def test_the_dormant_menu_offers_the_three_faces_when_they_are_enabled(make, all_forms):
    dock = make(face_form="orb")
    menu = dock.build_menu()
    assert [a.text() for a in menu.actions()] == [
        "Show ORION", "", "Screen Face", "Classic Face", "Orb", "", "Hide Dock"]
    checked = {a.text(): a.isChecked() for a in menu.actions() if a.isCheckable()}
    assert checked == {"Screen Face": False, "Classic Face": False, "Orb": True}


def test_the_menu_actions_call_back(make, all_forms):
    shown, forms, hidden = [], [], []
    dock = make(on_show=lambda: shown.append(1), on_face_form=forms.append,
                on_hide=lambda: hidden.append(1))
    dock.show_dock()
    menu = dock.build_menu()
    by_text = {a.text(): a for a in menu.actions() if a.text()}
    by_text["Show ORION"].trigger()
    by_text["Classic Face"].trigger()
    by_text["Orb"].trigger()
    assert shown == [1] and forms == ["classic", "orb"]
    assert dock.build_menu().actions()[4].isChecked()           # the menu now ticks Orb
    by_text["Hide Dock"].trigger()
    assert hidden == [1] and not dock.isVisible() and not dock.wanted


def test_show_orion_falls_back_to_the_activate_callback(make):
    activated = []
    dock = make(on_activate=lambda: activated.append(1))
    {a.text(): a for a in dock.build_menu().actions()}["Show ORION"].trigger()
    assert activated == [1]


def test_the_context_menu_is_popped_up_not_run_in_a_nested_loop():
    """A blocking menu run would nest an event loop under qasync and stall the audio."""
    source = (ROOT / "orion_core" / "gui" / "screen_dock.py").read_text(encoding="utf-8")
    assert ".popup(" in source and ".exec(" not in source


# ══════════════════════════════════════════════════════════════════════════════
# THE FACE PROTOCOL, FORWARDED
# ══════════════════════════════════════════════════════════════════════════════

def test_everything_in_the_face_protocol_reaches_the_face_inside(make):
    dock = make()
    seen = []
    for name in ("set_state", "set_amplitude", "set_speaking", "set_viseme", "glance", "apply_emotion"):
        setattr(dock.panel, name, lambda *args, _n=name, **kw: seen.append((_n, args)))
    dock.set_state("LISTENING")
    dock.set_amplitude(0.4)
    dock.set_speaking(True)
    dock.set_viseme(0.3, 0.2, 0.0)
    dock.glance(0.1, -0.2, 1.0)
    dock.apply_emotion("happy", {"intensity": 0.5})
    assert seen == [("set_state", ("LISTENING",)), ("set_amplitude", (0.4,)), ("set_speaking", (True,)),
                    ("set_viseme", (0.3, 0.2, 0.0)), ("glance", (0.1, -0.2, 1.0)),
                    ("apply_emotion", ("happy", {"intensity": 0.5}))]


def test_the_palette_reaches_the_face_and_the_rim(make):
    dock = make()
    dock.set_palette_colours("#112233", "#44aa66", "#050505")
    assert dock.panel._primary.name() == "#112233" and dock.panel._accent.name() == "#44aa66"
    assert dock._accent.name() == "#44aa66"


def test_the_dock_follows_the_bus(make):
    from orion_core.bus import OrionBus

    dock = make()
    seen = []
    dock.panel.set_amplitude = lambda value: seen.append(("amp", value))
    dock.panel.apply_emotion = lambda name, params=None: seen.append(("emotion", name))
    dock.panel.set_viseme = lambda *args: seen.append(("viseme", args))
    bus = OrionBus()
    dock.connect_bus(bus)
    bus.state.emit("LISTENING")
    assert dock.grown
    bus.amplitude.emit(0.5)
    bus.emotion_changed.emit("happy", {})
    bus.viseme.emit({"viseme": "PP", "weight": 1.0})
    bus.viseme.emit({"open": 0.4, "width": 0.1, "closure": 0.0})
    bus.viseme.emit("not a dict")
    bus.state.emit("STANDBY")
    assert not dock.grown
    assert seen == [("amp", 0.5), ("emotion", "happy"), ("viseme", ("PP", 1.0)),
                    ("viseme", (0.4, 0.1, 0.0))]


# ══════════════════════════════════════════════════════════════════════════════
# STOPPING
# ══════════════════════════════════════════════════════════════════════════════

def test_stop_stops_the_face_pipeline_the_timers_and_hides(make):
    dock = make(shrink_delay_ms=500)
    stopped = []
    dock.panel.stop_pipeline = lambda: stopped.append(1)
    dock.show_dock()
    dock.set_state("LISTENING")
    dock.set_state("STANDBY")
    assert dock._shrink.isActive() and dock._poll.isActive()
    dock.stop()
    assert stopped and not dock.isVisible()                    # (hiding a shown face stops it too)
    assert not dock._poll.isActive() and not dock._shrink.isActive()
    dock.show_dock()                                           # a stopped dock stays stopped
    assert not dock.isVisible()
    dock.stop()                                                # idempotent


def test_closing_the_dock_stops_it(make):
    dock = make()
    stopped = []
    dock.panel.stop_pipeline = lambda: stopped.append(1)
    dock.show_dock()
    dock.close()
    assert stopped and not dock.isVisible()


# ══════════════════════════════════════════════════════════════════════════════
# CROSSING TO THE QT THREAD, AND THE HANDLE'S DOCK
# ══════════════════════════════════════════════════════════════════════════════

def test_post_to_qt_runs_the_callable_on_the_gui_thread_not_the_caller(qapp):
    seen = []
    worker = threading.Thread(
        target=lambda: wp.post_to_qt(lambda: seen.append(threading.get_ident())))
    worker.start()
    worker.join()
    assert seen == []                                          # queued, not run on the worker
    QTest.qWait(100)
    assert seen == [threading.get_ident()]


def test_call_on_qt_runs_at_once_on_the_gui_thread_and_queues_from_elsewhere(qapp):
    seen = []
    wp.call_on_qt(lambda: seen.append("direct"))
    assert seen == ["direct"]
    worker = threading.Thread(target=lambda: wp.call_on_qt(lambda: seen.append("queued")))
    worker.start()
    worker.join()
    assert seen == ["direct"]
    QTest.qWait(100)
    assert seen == ["direct", "queued"]


def test_a_raising_callable_does_not_break_the_dispatcher(qapp):
    seen = []
    wp.post_to_qt(lambda: 1 / 0)
    wp.post_to_qt(lambda: seen.append("still alive"))
    QTest.qWait(100)
    assert seen == ["still alive"]


class _Part:
    def __init__(self, callback, **kwargs):
        self.kwargs = kwargs

    def start(self):
        return True

    def stop(self):
        pass

    def describe(self):
        return "part"


def test_the_handle_makes_a_dock_with_the_palette_and_edge_and_stops_it(qapp):
    windows_settings.set_value("dock_edge", "left")
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda text: None))
    handle = wp.start_windows_presence(
        bus, on_hotkey=lambda: None, on_session_event=lambda e: None, palette=("#112233", "#44aa66"),
        hotkey_factory=_Part, watcher_factory=_Part, dispatch=lambda fn: fn())
    activated = []
    dock = handle.make_dock(on_activate=lambda: activated.append(1), face_form="screen",
                            screen_provider=lambda: SCREEN, animate=False)
    try:
        assert isinstance(dock, ScreenDock) and dock.edge == "left"
        assert dock.panel._primary.name() == "#112233" and dock.panel._accent.name() == "#44aa66"
        assert handle.dock_factory() == handle.make_dock
        handle.set_palette("#aa0000", "#ffffff")
        assert dock.panel._primary.name() == "#aa0000"
        stopped = []
        dock.panel.stop_pipeline = lambda: stopped.append(1)
        handle.stop()
        assert stopped                                         # the handle stopped the dock's pipeline
    finally:
        dock.stop()
        dock.deleteLater()
        qapp.processEvents()
