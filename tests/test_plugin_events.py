"""
Plugin event hooks (Mark XXVI improvement #6) — plugins that REACT.

A plugin declares ``"events": [...]`` in its manifest and exports ``on_event``;
the bridge fans the matching bus signals out to it. The load-bearing property is
isolation: a faulting plugin is counted, logged and eventually MUTED — it can
never break ORION, and it can never subscribe to something it did not declare.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.plugin_events import (  # noqa: E402
    MAX_FAULTS,
    SUBSCRIBABLE,
    PluginEventBridge,
)


class _Sig:
    def __init__(self):
        self._slots = []

    def connect(self, fn):
        self._slots.append(fn)

    def disconnect(self, fn):
        self._slots.remove(fn)

    def emit(self, *args):
        for fn in list(self._slots):
            fn(*args)


class _Bus:
    def __init__(self):
        for name in SUBSCRIBABLE:
            setattr(self, name, _Sig())
        self.log = _Sig()


class _Plugin:
    """A plugin module that records what it was told."""

    def __init__(self):
        self.seen: list[tuple] = []

    def on_event(self, event, payload):
        self.seen.append((event, payload))


class _Faulty:
    def on_event(self, event, payload):
        raise RuntimeError("plugin exploded")


class _Passive:
    """A plugin from before hooks existed — no on_event at all."""


@pytest.fixture()
def bridge():
    return PluginEventBridge(bus=_Bus(), log=lambda _m: None)


# ── subscription ──────────────────────────────────────────────────────────────

def test_declared_events_are_bound_and_delivered(bridge):
    plugin = _Plugin()
    bound = bridge.subscribe("lamp", plugin, ["state", "speaking"])
    assert set(bound) == {"state", "speaking"}
    bridge.bus.state.emit("SPEAKING")
    bridge.bus.speaking.emit(True)
    assert plugin.seen == [("state", "SPEAKING"), ("speaking", True)]


def test_direct_plugin_event_control_call_is_background_and_never_reaches_input(bridge):
    from types import SimpleNamespace
    from orion_core.control import AutonomousControlLayer
    from orion_core.execution_policy import is_background
    os_calls, outcomes = [], []
    control = AutonomousControlLayer.__new__(AutonomousControlLayer)
    control.enabled = True
    control._pg = SimpleNamespace(typewrite=lambda *_args, **_kwargs: os_calls.append("typewrite"))
    control._kb = SimpleNamespace(write=lambda *_args, **_kwargs: os_calls.append("write"))
    control._narrate_typing = lambda *_: None
    control._record = lambda *_: None
    class Plugin:
        def on_event(self, event, payload):
            outcomes.append((is_background(), control.type_text("fixture input")))
    bridge.subscribe("fixture", Plugin(), ["state"])
    bridge.bus.state.emit("LISTENING")
    assert len(outcomes) == 1 and outcomes[0][0] is True
    assert not outcomes[0][1].ok and outcomes[0][1].receipt["status"] == "requires_foreground"
    assert os_calls == []


def test_an_undeclared_event_is_never_delivered(bridge):
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    bridge.bus.speaking.emit(True)          # not declared
    assert plugin.seen == []


def test_an_event_outside_the_whitelist_is_refused(bridge):
    plugin = _Plugin()
    bound = bridge.subscribe("nosy", plugin, ["confirm_action", "request_shutdown"])
    assert bound == [], "a plugin must not subscribe to non-whitelisted signals"


def test_an_unknown_signal_name_is_ignored_not_fatal(bridge):
    plugin = _Plugin()
    assert bridge.subscribe("lamp", plugin, ["not_a_real_signal"]) == []


def test_a_plugin_without_on_event_binds_nothing(bridge):
    assert bridge.subscribe("old", _Passive(), ["state"]) == []


def test_the_whitelist_excludes_command_and_confirmation_signals():
    for forbidden in ("confirm_action", "request_shutdown", "request_restart",
                      "gui_command", "speak_request"):
        assert forbidden not in SUBSCRIBABLE


# ── isolation: a plugin can never break ORION ────────────────────────────────

def test_a_faulting_plugin_does_not_raise_into_the_bus(bridge):
    bridge.subscribe("bad", _Faulty(), ["state"])
    bridge.bus.state.emit("LISTENING")      # must not raise
    assert bridge.stats["bad"].faults == 1


def test_a_repeatedly_faulting_plugin_is_muted(bridge):
    bridge.subscribe("bad", _Faulty(), ["state"])
    for _ in range(MAX_FAULTS + 2):
        bridge.bus.state.emit("LISTENING")
    stats = bridge.stats["bad"]
    assert stats.muted is True
    assert stats.faults == MAX_FAULTS, "a muted plugin must stop being called"


def test_a_muted_plugin_can_be_revived(bridge):
    bridge.subscribe("bad", _Faulty(), ["state"])
    for _ in range(MAX_FAULTS):
        bridge.bus.state.emit("X")
    assert bridge.stats["bad"].muted is True
    assert bridge.unmute("bad") is True
    assert bridge.stats["bad"].muted is False


def test_one_plugin_faulting_does_not_stop_another(bridge):
    good = _Plugin()
    bridge.subscribe("bad", _Faulty(), ["state"])
    bridge.subscribe("good", good, ["state"])
    bridge.bus.state.emit("SPEAKING")
    assert good.seen == [("state", "SPEAKING")]


# ── introspection ─────────────────────────────────────────────────────────────

def test_hooked_and_report_describe_the_wiring(bridge):
    bridge.subscribe("lamp", _Plugin(), ["state"])
    assert bridge.hooked() == {"lamp": ["state"]}
    bridge.bus.state.emit("X")
    report = bridge.report()
    assert "lamp" in report and "delivered" in report


def test_report_with_nothing_hooked(bridge):
    assert "No plugin is subscribed" in bridge.report()


def test_delivery_counts_are_tracked(bridge):
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    bridge.bus.state.emit("A")
    bridge.bus.state.emit("B")
    assert bridge.stats["lamp"].delivered == 2


# ── manifest + wiring ─────────────────────────────────────────────────────────

def test_the_manifest_carries_declared_events():
    from orion_core.plugin_manifest import ManifestError, PluginManifest
    m = PluginManifest.from_dict({"name": "lamp", "module": "lamp_tool.py",
                                  "events": ["state", "speaking"]})
    assert m.events == ("state", "speaking")
    assert PluginManifest.from_dict({"name": "x", "module": "x_tool.py"}).events == ()
    with pytest.raises(ManifestError):
        PluginManifest.from_dict({"name": "x", "module": "x_tool.py",
                                  "events": "state"})


def test_the_loader_binds_hooks_after_activation():
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "plugin_manifest.py").read_text(encoding="utf-8")
    assert "events.subscribe(" in src
    # the module object comes off the loader, not the outcome
    assert "loaded_modules" in src


def test_the_app_creates_the_bridge_and_the_tool_reports_it():
    app_src = (Path(__file__).resolve().parents[1] / "orion_core"
               / "app.py").read_text(encoding="utf-8")
    assert "PluginEventBridge(bus=bus, registry=plugins)" in app_src
    assert "events=plugin_events" in app_src
    disp = (Path(__file__).resolve().parents[1] / "orion_core"
            / "dispatch_productivity.py").read_text(encoding="utf-8")
    assert "bridge.report()" in disp


# ── slow-hook protection (a hook runs on ORION's own thread) ─────────────────

class _Slow:
    def __init__(self, seconds=0.06):
        self.seconds = seconds

    def on_event(self, event, payload):
        import time as _t
        _t.sleep(self.seconds)


def test_hook_timing_is_measured(bridge):
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    bridge.bus.state.emit("X")
    stats = bridge.stats["lamp"]
    assert stats.delivered == 1
    assert stats.max_ms >= 0.0 and stats.average_ms >= 0.0


def test_a_chronically_slow_plugin_is_muted(bridge):
    from orion_core.plugin_events import MAX_SLOW, SLOW_MS
    bridge.subscribe("sluggish", _Slow(), ["state"])
    for _ in range(MAX_SLOW + 2):
        bridge.bus.state.emit("X")
    stats = bridge.stats["sluggish"]
    assert stats.slow_calls == MAX_SLOW, "a muted plugin must stop being called"
    assert stats.muted is True
    assert stats.max_ms > SLOW_MS


def test_a_fast_plugin_is_never_muted(bridge):
    from orion_core.plugin_events import MAX_SLOW
    plugin = _Plugin()
    bridge.subscribe("quick", plugin, ["state"])
    for _ in range(MAX_SLOW * 3):
        bridge.bus.state.emit("X")
    stats = bridge.stats["quick"]
    assert stats.muted is False and stats.slow_calls == 0


def test_the_report_surfaces_timing(bridge):
    bridge.subscribe("lamp", _Plugin(), ["state"])
    bridge.bus.state.emit("X")
    assert "avg" in bridge.report()


# ── hot-reload: hooks must survive, and must not stack (improvement #14) ────

def test_resubscribing_replaces_rather_than_stacks(bridge):
    """A hot-reload re-subscribes the plugin. Without a disconnect the plugin
    would receive every event twice, three times after the next reload…"""
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    bridge.subscribe("lamp", plugin, ["state"])      # the reload
    bridge.bus.state.emit("X")
    assert plugin.seen == [("state", "X")], "the event was delivered more than once"


def test_forget_disconnects_a_plugin(bridge):
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    assert bridge.forget("lamp") is True
    bridge.bus.state.emit("X")
    assert plugin.seen == []
    assert bridge.hooked() == {}


def test_forgetting_an_unknown_plugin_is_harmless(bridge):
    assert bridge.forget("never_existed") is False


def test_a_reload_that_drops_on_event_also_drops_the_hooks(bridge):
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    # the reloaded module no longer exports on_event
    assert bridge.subscribe("lamp", _Passive(), ["state"]) == []
    bridge.bus.state.emit("X")
    assert plugin.seen == []


def test_the_reload_path_passes_the_event_bridge():
    """Without this a hot-reloaded plugin keeps working as a tool but silently
    stops reacting."""
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "dispatch_productivity.py").read_text(encoding="utf-8")
    reload_at = src.index("async def _reload_plugins")
    window = src[reload_at:reload_at + 2500]
    assert "events=getattr(self, \"plugin_events\", None)" in window


def test_no_double_delivery_even_if_disconnect_fails(bridge):
    """Belt and braces: a signal whose disconnect does not take (a stub, an
    exotic signal object) must still not deliver twice after a hot-reload —
    the superseded slot goes inert by generation."""
    class _StubbornSig(_Sig):
        def disconnect(self, fn):          # pretends to work, keeps the slot
            return None

    bridge.bus.state = _StubbornSig()
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    bridge.subscribe("lamp", plugin, ["state"])       # reload
    bridge.bus.state.emit("X")
    assert plugin.seen == [("state", "X")]
    assert len(bridge.bus.state._slots) == 2, "both slots are still connected…"


# ── disabling must stop a plugin reacting IMMEDIATELY (improvement #15) ─────

def test_disable_and_remove_unhook_the_plugin_at_once():
    """'Disabled' must mean disabled. Without this a disabled plugin keeps
    receiving events until the next restart."""
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "dispatch_productivity.py").read_text(encoding="utf-8")
    disable_at = src.index('if action in {"disable", "deactivate", "off"}')
    assert "bridge.forget(name)" in src[disable_at:disable_at + 700]
    remove_at = src.index('if action in {"remove", "uninstall", "delete"}')
    assert "bridge.forget(removed_name)" in src[remove_at:remove_at + 1000]


def test_forget_stops_delivery_immediately(bridge):
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state", "speaking"])
    bridge.bus.state.emit("A")
    bridge.forget("lamp")
    bridge.bus.state.emit("B")
    bridge.bus.speaking.emit(True)
    assert plugin.seen == [("state", "A")]


def test_the_tool_can_revive_a_muted_plugin():
    """A plugin muted for faulting or blocking could otherwise only be revived by
    restarting ORION."""
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "dispatch_productivity.py").read_text(encoding="utf-8")
    at = src.index('if action in {"unmute", "revive", "unsilence"}')
    window = src[at:at + 900]
    assert "bridge.unmute(" in window
    # with no name it must revive every muted plugin
    assert "st.muted" in window


def test_unmute_resets_the_fault_count(bridge):
    bridge.subscribe("bad", _Faulty(), ["state"])
    for _ in range(MAX_FAULTS):
        bridge.bus.state.emit("X")
    assert bridge.stats["bad"].muted is True
    bridge.unmute("bad")
    stats = bridge.stats["bad"]
    assert stats.muted is False and stats.faults == 0, (
        "a revived plugin needs a fresh budget, not one strike from being muted again")


_HOOK_QT_APP = None


@pytest.fixture()
def qt_bridge():
    from PyQt6.QtWidgets import QApplication
    from orion_core.bus import OrionBus

    global _HOOK_QT_APP
    _HOOK_QT_APP = QApplication.instance() or QApplication([])
    bridge = PluginEventBridge(bus=OrionBus(), log=lambda _m: None)
    yield bridge, _HOOK_QT_APP
    for name in list(bridge.hooked()):
        bridge.forget(name)
    _HOOK_QT_APP.processEvents()


@pytest.mark.parametrize("event,args,expected", [
    ("state", ("LISTENING",), "LISTENING"),
    ("speaking", (True,), True),
    ("banner", ("ready", 500), ("ready", 500)),
    ("emotion_changed", ("thinking", {"strength": 0.6}),
     ("thinking", {"strength": 0.6})),
    ("telemetry_sample", ({"cpu": 12},), {"cpu": 12}),
    ("connection_state", ({"online": True},), {"online": True}),
    ("dashboard_event", ("tasks", []), ("tasks", [])),
    ("paused", (False,), False),
    ("safety_alert", ("synthetic",), "synthetic"),
    ("log", ("synthetic",), "synthetic"),
])
def test_qt_hook_runs_after_emit_and_preserves_signal_payload(qt_bridge, event, args, expected):
    bridge, app = qt_bridge
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, [event])
    getattr(bridge.bus, event).emit(*args)
    assert plugin.seen == [], "arbitrary plugin code ran inside the emitter"
    app.processEvents()
    assert plugin.seen == [(event, expected)]


@pytest.mark.parametrize("change", ["forget", "reload", "disable"])
def test_qt_hook_queued_before_retirement_does_not_run_afterward(qt_bridge, change):
    from types import SimpleNamespace

    bridge, app = qt_bridge
    old, new = _Plugin(), _Plugin()
    enabled = [True]
    bridge.registry = SimpleNamespace(is_enabled=lambda _: enabled[0])
    bridge.subscribe("lamp", old, ["state"])
    bridge.bus.state.emit("old")
    if change == "reload":
        bridge.subscribe("lamp", new, ["state"])
        bridge.bus.state.emit("new")
    elif change == "disable":
        enabled[0] = False
    else:
        bridge.forget("lamp")
    app.processEvents()
    assert old.seen == [], "an already queued hook survived retirement"
    assert new.seen == ([("state", "new")] if change == "reload" else [])


def test_qt_hook_emitted_while_disabled_is_not_replayed_on_enable(qt_bridge):
    from types import SimpleNamespace

    bridge, app = qt_bridge
    enabled = [False]
    bridge.registry = SimpleNamespace(is_enabled=lambda _: enabled[0])
    plugin = _Plugin()
    bridge.subscribe("lamp", plugin, ["state"])
    bridge.bus.state.emit("disabled")
    enabled[0] = True
    bridge.bus.state.emit("enabled")
    assert plugin.seen == []
    app.processEvents()
    assert plugin.seen == [("state", "enabled")]


def test_qt_hook_receiver_stays_on_bus_thread_when_subscribed_from_worker(qt_bridge):
    import threading
    from types import SimpleNamespace
    from PyQt6.QtCore import QThread

    bridge, app = qt_bridge
    seen = []
    plugin = SimpleNamespace(on_event=lambda *_: seen.append(QThread.currentThread()))
    worker = threading.Thread(target=lambda: bridge.subscribe("lamp", plugin, ["state"]))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    bridge.bus.state.emit("ready")
    app.processEvents()
    assert seen == [bridge.bus.thread()], "the callback was stranded on the subscription thread"


def test_qt_hook_emitted_from_worker_runs_on_bus_thread(qt_bridge):
    import threading
    from types import SimpleNamespace
    from PyQt6.QtCore import QThread

    bridge, app = qt_bridge
    seen = []
    plugin = SimpleNamespace(on_event=lambda *args: seen.append((args, QThread.currentThread())))
    bridge.subscribe("lamp", plugin, ["state"])
    worker = threading.Thread(target=lambda: bridge.bus.state.emit("ready"))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive() and seen == []
    app.processEvents()
    assert seen == [(("state", "ready"), bridge.bus.thread())]


@pytest.mark.parametrize("plugin", [_Faulty(), _Slow()])
def test_qt_hook_fault_and_slow_budgets_hold_for_already_queued_events(qt_bridge, plugin):
    from orion_core.plugin_events import MAX_SLOW

    bridge, app = qt_bridge
    bridge.subscribe("limited", plugin, ["state"])
    for _ in range(MAX_SLOW + MAX_FAULTS):
        bridge.bus.state.emit("ready")
    app.processEvents()
    stats = bridge.stats["limited"]
    assert stats.muted
    if isinstance(plugin, _Faulty):
        assert stats.faults == MAX_FAULTS and stats.delivered == 0
    else:
        assert stats.slow_calls == MAX_SLOW and stats.delivered == MAX_SLOW


def test_qt_hook_nested_pump_cannot_destroy_an_async_receive_task(tmp_path):
    """A plugin can open a modal Qt pump; it must run outside an asyncio step."""
    import json
    import subprocess

    probe = tmp_path / "plugin_pump.py"
    probe.write_text(r'''
import asyncio, json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.getcwd())
from PyQt6.QtCore import QObject, QEventLoop, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication
import qasync
from orion_core.plugin_events import PluginEventBridge
app = QApplication([])
loop = qasync.QEventLoop(app)
asyncio.set_event_loop(loop)
errors, seen, ticks = [], [], [0]
loop.set_exception_handler(lambda _l, ctx: errors.append(str(ctx.get("exception"))))
class Bus(QObject):
    state = pyqtSignal(str)
class Plugin:
    def on_event(self, *_):
        seen.append(asyncio.current_task() is not None)
        nested = QEventLoop()
        QTimer.singleShot(150, nested.quit)
        nested.exec()
bus = Bus()
bridge = PluginEventBridge(bus=bus, log=lambda _: None)
bridge.subscribe("pump", Plugin(), ["state"])
async def receive():
    while True:
        await asyncio.sleep(0.01)
        ticks[0] += 1
async def main():
    receiver = asyncio.ensure_future(receive())
    await asyncio.sleep(0.05)
    bus.state.emit("ready")
    await asyncio.sleep(0.2)
    before = ticks[0]
    await asyncio.sleep(0.1)
    print(json.dumps({"inside_task": seen, "errors": errors,
                      "later_ticks": ticks[0] - before}), flush=True)
    bridge.forget("pump")
    receiver.cancel()
with loop:
    loop.run_until_complete(main())
''', encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", str(probe)], cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.splitlines()[-1])
    (tmp_path / "plugin_pump_result.json").write_text(json.dumps(result), encoding="utf-8")
    assert result["inside_task"] == [False], result
    assert result["errors"] == [], result
    assert result["later_ticks"] >= 3, result
