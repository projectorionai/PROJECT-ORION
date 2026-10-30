"""Durable reminder state, truthful receipts, and responsive callers."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import threading
from types import SimpleNamespace

import pytest

import orion_core.reminders as reminders
from orion_core.time_service import TIME


class Signal:
    def __init__(self):
        self.sent = []
        self.callback = None

    def emit(self, *args):
        self.sent.append(args)
        if self.callback:
            self.callback()


@pytest.fixture
def state(tmp_path, monkeypatch):
    clock = [datetime(2026, 10, 6, 12, tzinfo=timezone.utc)]
    elapsed = [10000.0]
    monkeypatch.setattr(TIME, "_clock", lambda: clock[0])
    monkeypatch.setattr(reminders, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    bus = SimpleNamespace(**{name: Signal() for name in
                              ("banner", "log", "speak_request", "dashboard_event", "phone_action")})
    return SimpleNamespace(clock=clock, elapsed=elapsed, bus=bus, path=tmp_path / "reminders.json")


def service(state):
    return reminders.ReminderService(state.bus, path=state.path)


def test_restart_rebases_portable_deadline_onto_new_monotonic_clock(state):
    first = service(state)
    assert first.add("check the oven", minutes=60, channel="SMS", recipient="me").ok
    state.clock[0] += timedelta(minutes=30)
    state.elapsed[0] = 5.0
    restored = service(state)
    assert len(restored.active()) == 1
    reminder = restored.active()[0]
    assert (reminder.text, reminder.due_at, reminder.wall_due) == ("check the oven", 1805.0, "13:00")
    assert (reminder.channel, reminder.recipient) == ("SMS", "me")


def test_cancelled_reminders_stay_cancelled_and_ids_do_not_recur(state):
    first = service(state)
    assert first.add("one", minutes=1).ok
    assert first.cancel(1).ok
    second = service(state)
    assert not second.active()
    assert second.add("two", minutes=1).ok
    assert second.active()[0].id == 2
    assert second.cancel().ok
    assert not service(state).active()


def test_failed_add_does_not_change_memory_counter_banner_or_old_file(state, monkeypatch):
    first = service(state)
    assert first.add("existing", minutes=5).ok
    before, banners = state.path.read_bytes(), list(state.bus.banner.sent)
    def fail(*_, **__):
        raise OSError("synthetic disk full")
    monkeypatch.setattr(reminders, "atomic_write_text", fail)
    assert not first.add("unsaved", minutes=5).ok
    assert state.path.read_bytes() == before
    assert first._counter == 1
    assert [r.text for r in first.active()] == ["existing"]
    assert state.bus.banner.sent == banners


def test_failed_cancellation_preserves_active_reminder_and_disk(state, monkeypatch):
    first = service(state)
    first.add("existing", minutes=5)
    before = state.path.read_bytes()
    def fail(*_, **__):
        raise PermissionError("synthetic sharing lock")
    monkeypatch.setattr(reminders, "atomic_write_text", fail)
    assert not first.cancel(1).ok
    assert not first.active()[0].fired
    assert state.path.read_bytes() == before
    assert len(service(state).active()) == 1


@pytest.mark.parametrize("content", [
    "{broken", "[]", '{"version": 99}',
    '{"version": 1, "counter": true, "reminders": []}',
    '{"version": 1, "counter": 1, "reminders": [{"id": 1, "due_epoch": NaN}]}',
    '{"version": 1, "counter": 0, "reminders": [{"id": 1, "due_epoch": 1}]}',
])
def test_unreadable_or_malformed_store_is_preserved_and_not_reported_as_clear(state, content):
    state.path.write_text(content, encoding="utf-8")
    first = service(state)
    assert not first.list_text().ok
    assert not first.add("new", minutes=5).ok
    assert not first.cancel().ok
    assert state.path.read_text(encoding="utf-8") == content
    assert len(state.bus.log.sent) == 1


def test_overdue_remote_reminder_is_announced_locally_once_and_checkpointed(state):
    first = service(state)
    first.add("call the supplier", minutes=1, channel="VOICE_CALL")
    state.clock[0] += timedelta(hours=1)
    restored = service(state)
    calls = []
    restored.telephony = SimpleNamespace(call=lambda *_: calls.append(True))
    state.bus.speak_request.callback = restored.stop
    asyncio.run(restored.run())
    assert calls == []
    assert len(state.bus.speak_request.sent) == 1
    assert "offline" in state.bus.speak_request.sent[0][0]
    assert not service(state).active()


def test_checkpoint_failure_does_not_repeat_the_delivery_in_this_process(state, monkeypatch):
    first = service(state)
    first.add("one", minutes=1)
    state.elapsed[0] += 60
    def fail(*_, **__):
        raise OSError("synthetic checkpoint failure")
    monkeypatch.setattr(reminders, "atomic_write_text", fail)
    state.bus.speak_request.callback = first.stop
    asyncio.run(first.run())
    assert not first.active()
    assert first._checkpoint_error
    assert len(state.bus.speak_request.sent) == 1
    assert "could not be saved" in state.bus.log.sent[-1][0]
    # Delivery/checkpoint cannot be one atomic transaction with a speaker.
    # Keep the pending disk entry for recovery rather than claim exactly once.
    assert len(json.loads(state.path.read_text())["reminders"]) == 1


def test_slow_disk_save_does_not_block_active_reads(state, monkeypatch):
    first = service(state)
    entered, release = threading.Event(), threading.Event()
    original = reminders.atomic_write_text
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(reminders, "atomic_write_text", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        adding = pool.submit(first.add, "pending", minutes=1)
        assert entered.wait(timeout=5)
        try:
            assert pool.submit(first.active).result(timeout=0.25) == []
        finally:
            release.set()
        assert adding.result(timeout=5).ok
    assert len(first.active()) == 1


def test_parallel_adds_preserve_every_id_and_reopen_every_reminder(state):
    first = service(state)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: first.add(f"item {i}", minutes=1), range(12)))
    assert all(result.ok for result in results)
    restored = service(state)
    assert sorted(r.id for r in restored.active()) == list(range(1, 13))
    assert {r.text for r in restored.active()} == {f"item {i}" for i in range(12)}


def test_explicit_in_memory_mode_does_not_create_a_file(state):
    first = reminders.ReminderService(state.bus, path=None)
    assert first.add("ephemeral", minutes=1).ok
    assert not state.path.exists()


def test_dispatcher_offloads_reminder_saves_while_the_loop_keeps_running(state, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.bus import OrionBus
    dispatcher = OrionDispatcher(OrionBus(), SimpleNamespace(active_project="", save=lambda *_: None),
                                 None, None, None, None, None, None, None, None)
    dispatcher.reminders = service(state)
    entered, release = threading.Event(), threading.Event()
    original = reminders.atomic_write_text
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(reminders, "atomic_write_text", slow)
    async def scenario():
        task = asyncio.create_task(dispatcher.dispatch("reminder", {"text": "one", "minutes": 1}))
        try:
            while not entered.is_set() and not task.done():
                await asyncio.sleep(0.001)
            assert entered.is_set()
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        return await task
    assert asyncio.run(scenario()).ok


def test_widget_keeps_failed_input_and_reenables_controls(state, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from orion_core.bus import OrionBus
    from orion_core.gui.entrepreneur import RemindersWidget
    app = QApplication.instance() or QApplication([])
    widget = RemindersWidget(OrionBus(), service(state))
    widget._timer.stop()
    widget.text.setText("unsaved note")
    widget.mins.setText("5")
    def fail(*_, **__):
        raise OSError("synthetic save failure")
    monkeypatch.setattr(reminders, "atomic_write_text", fail)
    async def scenario():
        widget._add()
        assert not widget.text.isEnabled()
        await widget._action_task
    asyncio.run(scenario())
    assert widget.text.text() == "unsaved note" and widget.mins.text() == "5"
    assert widget.text.isEnabled() and widget._set_button.isEnabled()
    assert not widget.reminders.active()
    widget.close()
    app.processEvents()


def test_widget_saves_on_a_worker_and_ignores_duplicate_clicks(state, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from orion_core.bus import OrionBus
    from orion_core.gui.entrepreneur import RemindersWidget
    app = QApplication.instance() or QApplication([])
    widget = RemindersWidget(OrionBus(), service(state))
    widget._timer.stop()
    widget.text.setText("one")
    widget.mins.setText("5")
    entered, release = threading.Event(), threading.Event()
    threads = []
    original = reminders.atomic_write_text
    def slow(*args, **kwargs):
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(reminders, "atomic_write_text", slow)
    async def scenario():
        widget._add()
        task = widget._action_task
        try:
            while not entered.is_set() and not task.done():
                await asyncio.sleep(0.001)
            assert entered.is_set()
            widget._add()
            assert widget._action_task is task
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        await task
    asyncio.run(scenario())
    assert threads == [threads[0]] and threads[0] is not threading.main_thread()
    assert len(widget.reminders.active()) == 1
    assert not widget.text.text() and widget._clear_button.isEnabled()
    widget.close()
    app.processEvents()


def test_widget_explains_unreadable_saved_reminders_and_keeps_mutations_disabled(state):
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication
    from orion_core.bus import OrionBus
    from orion_core.gui.entrepreneur import RemindersWidget
    app = QApplication.instance() or QApplication([])
    state.path.write_text("{synthetic corrupt data", encoding="utf-8")
    events = OrionBus()
    widget = RemindersWidget(events, reminders.ReminderService(events, path=state.path))
    try:
        assert widget.list.count() == 1
        assert "unavailable" in widget.list.item(0).text().lower()
        assert not widget._set_button.isEnabled()
        assert not widget._clear_button.isEnabled()
        widget._set_busy(True)
        widget._set_busy(False)
        assert not widget._set_button.isEnabled()
        assert not widget._clear_button.isEnabled()
        assert state.path.read_text(encoding="utf-8") == "{synthetic corrupt data"
    finally:
        sip.delete(widget)
        app.processEvents()
