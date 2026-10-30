"""Desktop actions must target the intended app and must not replay uncertain effects."""
import asyncio
import os
import threading
from types import SimpleNamespace

import pytest

from orion_core.control import WHEEL_DELTA, AutonomousControlLayer
from orion_core.data import ToolResult
from orion_core.plan_executor import PlanExecutor, _Step
from test_control_layer import _FakeKB, _FakePG, _StubBus, _StubDisplay
from test_desktop_robustness import _dispatcher
from test_verification_engine import _engine, _element, _StubControl, _StubVision
from test_vision_element_matching import _el, _vision


@pytest.fixture(autouse=True)
def immediate_waits(monkeypatch):
    async def immediate(_seconds):
        pass
    monkeypatch.setattr(asyncio, "sleep", immediate)


@pytest.fixture
def control(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyautogui", _FakePG())
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace(
        CoInitialize=lambda: None, CoUninitialize=lambda: None))
    monkeypatch.setenv("ORION_AUTONOMY", "1")
    return AutonomousControlLayer(_StubBus(), _StubDisplay())


def test_explicit_missing_window_never_falls_back_to_the_active_editor(control):
    active_requests = []
    desktop = SimpleNamespace(windows=lambda: [],
                              window=lambda **kw: active_requests.append(kw) or object())
    assert control._uia_window(desktop, "Nonexistent Notepad") is None
    assert not active_requests


def test_blank_window_target_cannot_match_an_arbitrary_window(control, monkeypatch):
    monkeypatch.setattr(control, "_windows", lambda: SimpleNamespace(
        getAllWindows=lambda: [SimpleNamespace(title="Working editor", isMinimized=False)]))
    assert control._find_window("  ") is None


def test_native_append_refuses_to_replace_text_it_cannot_read(control, monkeypatch):
    import sys
    writes = []
    def unreadable():
        raise RuntimeError("text pattern unavailable")
    edit = SimpleNamespace(get_value=unreadable, set_focus=lambda: None,
                           set_edit_text=writes.append,
                           iface_value=SimpleNamespace(SetValue=writes.append))
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Desktop=lambda **kw: object()))
    monkeypatch.setattr(control, "_uia_window", lambda *args: object())
    monkeypatch.setattr(control, "_uia_editable", lambda *args: edit)
    assert not control._edit_via_uia("add this", "Editor", False)
    assert writes == []
    assert control._edit_via_uia("replace this", "Editor", True)
    assert writes == ["replace this"]


def test_text_verification_checks_the_middle_of_long_content():
    expected = "A" * 120 + "important middle content" + "Z" * 60
    corrupted = "A" * 120 + "WRONG" + "Z" * 60
    assert not AutonomousControlLayer._text_landed(expected, corrupted)
    assert AutonomousControlLayer._text_landed(expected, expected)


@pytest.mark.parametrize("action,args", [
    ("focus_window", ("Editor",)), ("resize_window", ("Editor", 600, 400)),
    ("move_window", ("Editor", 100, 100)), ("minimise_window", ("Editor",)),
    ("maximise_window", ("Editor",)), ("close_application", ("Editor",)),
])
def test_window_mutations_honor_the_autonomy_stop_switch(control, monkeypatch, action, args):
    calls = []
    win = SimpleNamespace(title="Editor", isMinimized=False,
                          activate=lambda: calls.append("focus"),
                          resizeTo=lambda *a: calls.append("resize"),
                          moveTo=lambda *a: calls.append("move"),
                          minimize=lambda: calls.append("minimise"),
                          maximize=lambda: calls.append("maximise"))
    monkeypatch.setattr(control, "_find_window", lambda title: win)
    control.desktop = SimpleNamespace(close_app=lambda name: calls.append("close") or ToolResult("closed"))
    control.set_enabled(False)
    result = getattr(control, action)(*args)
    assert not result.ok and "disabled" in result.text.lower()
    assert not calls


def test_disabled_exact_match_does_not_hide_the_enabled_target():
    disabled = dict(_el("Save", x=10), enabled=False)
    enabled = _el("Save", x=100)
    assert _vision([disabled, enabled])._find_element_sync("Save", "button") == enabled


def test_a_shared_word_is_not_enough_to_click_a_different_command():
    assert _vision([_el("Delete account")])._find_element_sync("Create account", "button") is None


def test_invisible_controls_are_not_click_candidates():
    hidden = dict(_el("Save", x=10), visible=False)
    assert _vision([hidden])._find_element_sync("Save", "button") is None


def stable_vision():
    vision = _StubVision()
    async def find(query, kinds="all"):
        return _element(query, 80, 90)
    vision.find_element = find
    return vision


def test_unchanged_pixels_never_repeat_an_element_click():
    engine = _engine(stable_vision(), ratios=[0, 0, 0])
    result = asyncio.run(engine.click_element("Submit", attempts=3))
    assert not result.ok
    assert engine.control.clicks == [(80, 90)]
    assert "not repeated" in result.text.lower()


def test_unverified_click_does_not_start_scrolling_and_clicking_elsewhere():
    engine = _engine(stable_vision(), ratios=[0] * 100)
    result = asyncio.run(engine.click_text("Submit", attempts=3))
    assert not result.ok
    assert engine.control.clicks == [(80, 90)]
    assert engine.control.scrolls == []


def test_empty_click_query_never_scrolls_the_desktop():
    engine = _engine()
    assert not asyncio.run(engine.click_text("  ")).ok
    assert not engine.control.scrolls


def test_search_scrolls_run_off_the_event_loop_thread():
    threads = []
    main_thread = threading.get_ident()
    control = _StubControl()
    def scroll(amount):
        threads.append(threading.get_ident())
        return ToolResult("Scrolled")
    control.scroll = scroll
    asyncio.run(_engine(control=control).click_text("Missing item", attempts=1))
    assert threads and main_thread not in threads


@pytest.mark.parametrize("tool,args", [
    ("desktop_control", {"action": "type_text", "text": "hello"}),
    ("desktop_control", {"action": "click", "x": 10, "y": 10}),
    ("web_control", {"action": "click", "text": "Send"}),
])
def test_plan_never_replays_a_visual_side_effect_when_pixels_do_not_change(tool, args):
    calls = []
    async def dispatch(tool, args):
        calls.append((tool, args))
        return ToolResult("Action issued")
    async def capture(region):
        return object()
    verifier = SimpleNamespace(_capture=capture, _change_ratio=lambda *a: 0.)
    executor = PlanExecutor(_StubBus(), dispatch, verifier=verifier)
    result = asyncio.run(executor._run_step(_Step(1, "step", tool, args, "retry"), 3))
    assert len(calls) == 1
    assert not result.ok


@pytest.mark.parametrize("tool,action", [
    ("vision_verify", "elements"), ("desktop_control", "list_windows"),
    ("desktop_control", "windows"), ("web_control", "read"), ("web_control", "read_page"),
])
def test_read_only_screen_inspection_does_not_require_the_screen_to_change(tool, action):
    calls = []
    async def dispatch(tool, args):
        calls.append(tool)
        return ToolResult("Three visible buttons")
    async def capture(region):
        pytest.fail("read-only inspection must not require a pixel change")
    executor = PlanExecutor(_StubBus(), dispatch, verifier=SimpleNamespace(_capture=capture))
    result = asyncio.run(executor._run_step(_Step(1, "step", tool, {"action": action}, "retry"), 3))
    assert result.ok and calls == [tool]


@pytest.mark.parametrize("amount", [0, 1, -1, 7, -7, 100, -100])
def test_smooth_scroll_delivers_exactly_the_requested_distance(control, monkeypatch, amount):
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    result = control.smooth_scroll(amount)
    assert result.ok
    # amount is in wheel NOTCHES; on Windows pyautogui takes raw wheel units
    # (120 per notch), so the delivered total is amount * 120 there.
    unit = WHEEL_DELTA if os.name == "nt" else 1
    assert sum(call[1] for call in control._pg.calls if call[0] == "scroll") == amount * unit
    if amount == 0:
        assert not control._pg.calls


def test_smooth_scroll_stops_during_the_action(control, monkeypatch):
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: control.set_enabled(False))
    result = control.smooth_scroll(20)
    assert not result.ok and "disabled" in result.text.lower()
    assert len([c for c in control._pg.calls if c[0] == "scroll"]) == 1


def test_slow_typing_stops_during_the_action(control, monkeypatch):
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: control.set_enabled(False))
    result = control.type_text("hello", human=True)
    assert not result.ok and "disabled" in result.text.lower()
    assert control._kb.written == ["h"]


@pytest.mark.parametrize("human", [False, True])
def test_keyboard_failure_is_not_reported_as_success_or_replayed(control, monkeypatch, human):
    def broken(*args, **kwargs):
        raise RuntimeError("input backend failed after possible partial input")
    control._kb.write = broken
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    result = control.type_text("hello", human=human)
    assert not result.ok
    assert not control._pg.calls


@pytest.mark.parametrize("action", ["click", "double_click", "right_click", "scroll", "smooth_scroll"])
def test_partial_coordinates_never_act_at_the_current_cursor(control, action):
    result = getattr(control, action)(**({"amount": 1} if "scroll" in action else {}), x=100)
    assert not result.ok
    assert not control._pg.calls


@pytest.mark.parametrize("args", [
    {"action": "move_cursor", "x": 10},
    {"action": "drag", "x1": 20, "y1": 30, "x2": 50},
    {"action": "move_window", "title": "Editor"},
])
def test_dispatch_rejects_missing_required_coordinates(control, args):
    result = asyncio.run(_dispatcher(control, None).desktop_control(args))
    assert not result.ok and "coordinate" in result.text.lower()
    assert not control._pg.calls


def test_failed_scroll_stops_the_search_without_more_input():
    control = _StubControl()
    def refused(amount):
        control.scrolls.append(amount)
        return ToolResult("Autonomy disabled", ok=False)
    control.scroll = refused
    result = asyncio.run(_engine(control=control).click_text("Missing"))
    assert not result.ok and "disabled" in result.text
    assert control.scrolls == [-4]


def test_unknown_previous_text_never_triggers_blind_undo(control, monkeypatch):
    keys = []
    monkeypatch.setattr(control, "send_hotkeys", lambda key: keys.append(key) or ToolResult("sent"))
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    assert not control._undo_garbled(None, "Editor")
    assert not keys


def test_unchanged_text_does_not_undo_the_users_previous_edit(control, monkeypatch):
    keys = []
    monkeypatch.setattr(control, "send_hotkeys", lambda key: keys.append(key) or ToolResult("sent"))
    monkeypatch.setattr(control, "_read_editable_text", lambda title: "original")
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    assert control._undo_garbled("original", "Editor")
    assert not keys


def test_recovery_never_sends_undo_to_a_different_app(control, monkeypatch):
    keys = []
    monkeypatch.setattr(control, "send_hotkeys", lambda key: keys.append(key) or ToolResult("sent"))
    monkeypatch.setattr(control, "_read_editable_text", lambda title: "garbled")
    monkeypatch.setattr(control, "_foreground_title", lambda: "Notes working in VS Code")
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    assert not control._undo_garbled("original", "Notepad")
    assert not keys


def test_unrestored_failed_write_does_not_try_another_method(control, monkeypatch):
    writes = []
    monkeypatch.setattr(control, "_narrate_typing", lambda *a, **kw: None)
    monkeypatch.setattr(control, "_read_editable_text", lambda title: "garbled")
    monkeypatch.setattr(control, "_edit_via_uia", lambda *a: writes.append("native") or True)
    monkeypatch.setattr(control, "_edit_via_clipboard", lambda *a: writes.append("paste") or True)
    monkeypatch.setattr(control, "_edit_via_keyboard", lambda *a: writes.append("keys") or True)
    monkeypatch.setattr(control, "_undo_garbled", lambda *a: False)
    result = control.edit_text("hello")
    assert not result.ok and writes == ["native"]


def test_unreadable_notepad_write_cannot_claim_completion(control, monkeypatch):
    monkeypatch.setattr(control, "_narrate_typing", lambda *a, **kw: None)
    monkeypatch.setattr(control, "_find_window", lambda title: None)
    monkeypatch.setattr(control, "_foreground_title", lambda: "Notepad")
    monkeypatch.setattr(control, "_read_editable_text", lambda title: None)
    monkeypatch.setattr(control, "_edit_via_uia", lambda *a: True)
    result = control.edit_text("Hello", title="Notepad")
    assert not result.ok
    assert "unverified" in result.text


def test_keyboard_edit_failure_does_not_repeat_partial_text(control):
    def broken(*args, **kwargs):
        raise RuntimeError("partial write")
    control._kb.write = broken
    with pytest.raises(RuntimeError):
        control._edit_via_keyboard("hello", False, 0.01)
    assert not control._pg.calls


def test_replacement_does_not_type_when_select_all_failed(control, monkeypatch):
    monkeypatch.setattr(control, "send_hotkeys", lambda *a: ToolResult("disabled", ok=False))
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError):
        control._edit_via_human("hello", True)
    assert not control._kb.written


def test_focus_does_not_report_success_when_activation_failed(control, monkeypatch):
    def failed():
        raise RuntimeError("activation denied")
    win = SimpleNamespace(title="Editor", isMinimized=False, isActive=False, activate=failed)
    monkeypatch.setattr(control, "_find_window", lambda title: win)
    monkeypatch.setattr(control, "_win32_foreground", lambda win: None)
    monkeypatch.setattr(control, "_foreground_title", lambda: "Different app")
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _: None)
    assert not control.focus_window("Editor").ok


@pytest.mark.parametrize("operation", ["read", "write", "error"])
def test_accessibility_calls_initialize_and_release_com_on_the_worker(control, monkeypatch, operation):
    import sys
    events = []
    def event(name):
        events.append((name, threading.get_ident()))
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace(
        CoInitialize=lambda: event("init"), CoUninitialize=lambda: event("release")))
    def desktop(**kwargs):
        assert events and events[-1][0] == "init"
        event("desktop")
        if operation == "error":
            raise RuntimeError("UIA unavailable")
        return object()
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Desktop=desktop))
    edit = SimpleNamespace(get_value=lambda: "existing", set_focus=lambda: None,
                           set_edit_text=lambda value: event("write"))
    monkeypatch.setattr(control, "_uia_window", lambda *a: object())
    monkeypatch.setattr(control, "_uia_editable", lambda *a: edit)
    async def run():
        if operation == "write":
            assert await asyncio.to_thread(control._edit_via_uia, "text", "Editor", True)
        else:
            result = await asyncio.to_thread(control._read_editable_text, "Editor")
            assert result == (None if operation == "error" else "existing")
    asyncio.run(run())
    assert events[0][0] == "init" and events[-1][0] == "release"
    assert len({ident for _, ident in events}) == 1
    assert events[0][1] != threading.get_ident()


@pytest.mark.parametrize("args,expected", [
    ({}, "human"), ({"speed": "fast"}, "fast"), ({"fast": True}, "fast"),
    ({"fast": "false"}, "human"), ({"speed": "human", "fast": True}, "human"),
    ({"speed": "quickly"}, "fast"),
])
def test_the_calls_speed_reaches_the_pointer_on_its_worker_thread(control, monkeypatch, args, expected):
    control.show_working = True
    seen = []
    def move(*_a, **_k):
        seen.append((control.pointer_mode(), threading.get_ident()))
        return ToolResult("moved")
    monkeypatch.setattr(control, "move_cursor", move)
    result = asyncio.run(_dispatcher(control, None).desktop_control(
        {"action": "move_cursor", "x": 5, "y": 5, **args}))
    assert result.ok and [mode for mode, _ in seen] == [expected]
    assert seen[0][1] != threading.get_ident()          # never the event-loop thread
    assert control.pointer_mode() == "human"            # nothing leaks past the call


@pytest.mark.parametrize("speed,human", [("fast", False), ("human", True), (None, True)])
def test_the_calls_speed_sets_how_text_is_typed(control, monkeypatch, speed, human):
    typed = []
    monkeypatch.setattr(control, "type_text",
                        lambda text, interval, **kw: typed.append(kw["human"]) or ToolResult("typed"))
    args = {"action": "type_text", "text": "hello"}
    if speed:
        args["speed"] = speed
    assert asyncio.run(_dispatcher(control, None).desktop_control(args)).ok
    assert typed == [human]
