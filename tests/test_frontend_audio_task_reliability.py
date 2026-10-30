"""Preferences preserve audio choices; delayed task replies survive page disposal.

All audio, planning and execution transports are synthetic. No devices open and
no user actions run during these regressions.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import sip
from PyQt6.QtWidgets import QApplication

from orion_core import audio_devices, background
from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.gui import preferences
from orion_core.gui.tasks_view import TasksView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def no_threads(monkeypatch):
    queued = []
    monkeypatch.setattr(preferences.threading, "Thread", lambda target, **_kwargs:
                        SimpleNamespace(start=lambda: queued.append(target)))
    return queued


def test_saved_audio_spec_is_raw_and_does_not_probe_hardware(tmp_path, monkeypatch):
    path = tmp_path / "audio.json"
    path.write_text('{"input":"fifine", "output":"USB Headset"}', encoding="utf-8")
    monkeypatch.setattr(audio_devices, "AUDIO_CONFIG_PATH", path)
    monkeypatch.setattr(audio_devices, "_devices", lambda: pytest.fail("settings queried hardware"))
    assert audio_devices.configured_spec("input") == "fifine"
    assert audio_devices.configured_spec("output") == "USB Headset"


@pytest.mark.parametrize("saved", ["", "default", "auto", "reset"])
def test_saved_audio_default_uses_the_picker_empty_value(tmp_path, monkeypatch, saved):
    import json
    path = tmp_path / "audio.json"
    path.write_text(json.dumps({"input": saved}), encoding="utf-8")
    monkeypatch.setattr(audio_devices, "AUDIO_CONFIG_PATH", path)
    assert audio_devices.configured_spec("input") == ""


@pytest.mark.parametrize("saved", ["USB Headset", "headset", "Unplugged mic"])
def test_audio_scan_preserves_the_configured_spec_even_if_device_is_missing(app, monkeypatch, no_threads, saved):
    monkeypatch.setattr(audio_devices, "configured_spec", lambda _kind: saved, raising=False)
    monkeypatch.setattr(audio_devices, "device_name", lambda _kind: "[16] USB Headset")
    monkeypatch.setattr(audio_devices, "usable_devices", lambda _kind: [(16, "USB Headset")])
    row = preferences._DeviceRow("input")
    try:
        row.refresh()
        no_threads.pop()()
        assert row.selected() == saved
        assert row.combo.isEnabled()
    finally:
        sip.delete(row)


def test_legacy_zero_device_index_is_preserved_without_becoming_default(tmp_path, monkeypatch):
    path = tmp_path / "audio.json"
    path.write_text('{"input":0}', encoding="utf-8")
    monkeypatch.setattr(audio_devices, "AUDIO_CONFIG_PATH", path)
    assert audio_devices.configured_spec("input") == "0"


def test_accepting_settings_during_audio_scan_retains_existing_preference(app, monkeypatch, no_threads):
    monkeypatch.setattr(audio_devices, "configured_spec", lambda _kind: "Saved mic", raising=False)
    row = preferences._DeviceRow("input")
    try:
        row.refresh()
        assert not row.combo.isEnabled()
        assert not row.test_button.isEnabled()
        assert row.selected() == "Saved mic"
    finally:
        sip.delete(row)


def test_disappeared_audio_device_is_not_tested_as_the_system_default(app, monkeypatch, no_threads):
    probes = []
    monkeypatch.setattr(audio_devices, "usable_devices", lambda _kind: [(3, "Other mic")])
    monkeypatch.setattr(audio_devices, "transport_works", lambda *args, **_kwargs:
                        probes.append(args) or (True, "synthetic other device worked"))
    row = preferences._DeviceRow("input")
    results = []
    row.tested.connect(lambda ok, detail: results.append((ok, detail)))
    try:
        row._populate([(2, "Chosen mic")], "Chosen mic")
        row._run_test()
        no_threads.pop()()
        assert probes == []
        assert results and not results[0][0]
        assert "unavailable" in results[0][1].lower()
    finally:
        sip.delete(row)


def test_audio_probe_prefers_the_exact_selected_device_to_a_longer_name(app, monkeypatch, no_threads):
    probes = []
    offered = [(3, "Other Chosen mic"), (7, "Chosen mic")]
    monkeypatch.setattr(audio_devices, "usable_devices", lambda _kind: offered)
    monkeypatch.setattr(audio_devices, "transport_works", lambda *args, **_kwargs:
                        probes.append(args) or (True, "synthetic transport"))
    row = preferences._DeviceRow("input")
    try:
        row._populate(offered, "Chosen mic")
        row._run_test()
        no_threads.pop()()
        assert probes == [("input", 7)]
    finally:
        sip.delete(row)


def test_unchanged_preferences_never_reopen_audio_streams(app, monkeypatch):
    from orion_core.gui.core_window import OrionCoreWindow
    saved = {"input": "fifine", "output": "USB Headset"}
    fake_dialog = SimpleNamespace(exec=lambda: 1, DialogCode=SimpleNamespace(Accepted=1),
        values=lambda: {"input_device": saved["input"], "output_device": saved["output"]})
    monkeypatch.setattr(preferences, "PreferencesDialog", lambda *_args, **_kwargs: fake_dialog)
    monkeypatch.setattr(audio_devices, "configured_spec", saved.__getitem__, raising=False)
    monkeypatch.setattr(audio_devices, "device_name", lambda kind: f"[16] {saved[kind]}")
    requests = []
    core = SimpleNamespace(_identity=lambda: None,
        bus=SimpleNamespace(audio_device_request=SimpleNamespace(emit=lambda *args: requests.append(args)),
                            log=SimpleNamespace(emit=lambda _line: None)))
    OrionCoreWindow.open_preferences(core)
    assert requests == []


@pytest.fixture
def scheduled(monkeypatch):
    coroutines = []
    monkeypatch.setattr(background, "spawn", lambda coro, **_kwargs: coroutines.append(coro))
    yield coroutines
    for coro in coroutines:
        coro.close()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
async def test_delayed_planning_never_touches_a_disposed_tasks_page(app, monkeypatch, scheduled, outcome):
    from orion_core import plan_executor, remote_tasks
    entered, release = asyncio.Event(), asyncio.Event()
    async def plan(*_args, **_kwargs):
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("synthetic planner failure")
        return [{"tool": "synthetic", "args": {}}]
    monkeypatch.setattr(remote_tasks, "notepad_plan", lambda _goal: None)
    monkeypatch.setattr(remote_tasks, "plan_request", plan)
    monkeypatch.setattr(plan_executor, "validate_plan", lambda steps, _catalogue: (steps, []))
    view = TasksView(OrionBus(), dispatcher=SimpleNamespace(
        plan_executor=SimpleNamespace(catalogue=lambda: {})), router=object())
    view.task_entry.setText("synthetic task")
    view._prepare_task()
    pending = asyncio.create_task(scheduled.pop())
    await entered.wait()
    sip.delete(view)
    if outcome == "cancelled":
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        release.set()
        await pending
    assert not view._planning


@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_reviewed_task_completion_outlives_its_page_and_still_logs_result(app, scheduled, outcome):
    entered, release = asyncio.Event(), asyncio.Event()
    async def execute(*_args, **_kwargs):
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("synthetic executor failure")
        return ToolResult("synthetic task completed")
    bus, logs = OrionBus(), []
    bus.log.connect(logs.append)
    view = TasksView(bus, dispatcher=SimpleNamespace(plan_executor=SimpleNamespace(execute=execute)))
    view.task_entry.setText("synthetic task")
    view._planned_goal = "synthetic task"
    view._planned_steps = [{"tool": "synthetic", "args": {}}]
    view._start_task()
    pending = asyncio.create_task(scheduled.pop())
    await entered.wait()
    sip.delete(view)
    release.set()
    await pending
    assert any(("completed" if outcome == "success" else "executor failure") in line for line in logs)


async def test_resuming_task_is_owned_and_can_finish_after_page_disposal(app, monkeypatch, scheduled):
    entered, release = asyncio.Event(), asyncio.Event()
    async def resume(job_id):
        assert job_id == "synthetic-job"
        entered.set()
        await release.wait()
        return ToolResult("synthetic task resumed")
    bus, logs = OrionBus(), []
    bus.log.connect(logs.append)
    view = TasksView(bus, jobs=SimpleNamespace(resume=resume))
    monkeypatch.setattr(view, "_selected_job_id", lambda: "synthetic-job")
    view._act("resume")
    assert len(scheduled) == 1, "resume task bypassed the background task owner"
    pending = asyncio.create_task(scheduled.pop())
    await entered.wait()
    sip.delete(view)
    release.set()
    await pending
    assert any("resumed" in line for line in logs)
