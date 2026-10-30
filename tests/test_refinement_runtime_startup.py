"""Provider configuration I/O yields the loop; dialog operations keep affinity."""

import asyncio
import threading

import pytest

from orion_core import app


async def _entered(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


def test_saved_provider_io_leaves_the_loop_free(monkeypatch):
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    settings = object()
    threads = []

    def read():
        threads.append(threading.get_ident())
        entered.set()
        release.wait(1)
        return settings

    monkeypatch.setattr(app, "_configured_provider_settings", read)
    monkeypatch.setattr(app, "ApiKeyDialog", lambda *_: pytest.fail("saved settings need no dialog"))

    async def scenario():
        task = asyncio.create_task(app.ensure_provider_settings_async())
        try:
            await _entered(entered)
            # The controller is an independent event-loop turn while I/O is held.
            assert not task.done(), "saved-provider I/O blocked the event loop"
            assert threads == [threads[0]] and threads[0] != owner
        finally:
            release.set()
            assert await task is settings

    asyncio.run(scenario())


class _Signal:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def disconnect(self, handler):
        self.handlers.remove(handler)

    def emit(self, result):
        for handler in list(self.handlers):
            handler(result)


def test_key_persistence_is_offloop_but_dialog_stays_on_owner(monkeypatch):
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Dialog:
        def __init__(self, _parent):
            calls.append(("create", threading.get_ident()))
            self.finished = _Signal()

        def open(self):
            calls.append(("open", threading.get_ident()))
            self.finished.emit(app.QDialog.DialogCode.Accepted)

        def key(self):
            calls.append(("key", threading.get_ident()))
            return "synthetic-key"

    def save(key):
        calls.append(("save", threading.get_ident()))
        entered.set()
        release.wait(1)
        return ("saved", key)

    monkeypatch.setattr(app, "_configured_provider_settings", lambda: None)
    monkeypatch.setattr(app, "_settings_from_key", save)
    monkeypatch.setattr(app, "ApiKeyDialog", Dialog)

    async def scenario():
        task = asyncio.create_task(app.ensure_provider_settings_async())
        try:
            await _entered(entered)
            assert not task.done(), "provider-key persistence blocked the event loop"
            assert all(thread == owner for name, thread in calls if name != "save")
            assert [thread for name, thread in calls if name == "save"] == [calls[-1][1]]
            assert calls[-1][1] != owner
        finally:
            release.set()
            assert await task == ("saved", "synthetic-key")

    asyncio.run(scenario())


def test_cancelled_setup_rejects_and_disconnects_its_dialog(monkeypatch):
    owner = threading.get_ident()
    dialogs = []

    class Dialog:
        def __init__(self, _parent):
            self.finished = _Signal()
            self.opened = asyncio.Event()
            self.rejected = 0
            dialogs.append(self)

        def open(self):
            assert threading.get_ident() == owner
            self.opened.set()

        def reject(self):
            assert threading.get_ident() == owner
            self.rejected += 1
            self.finished.emit(app.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(app, "_configured_provider_settings", lambda: None)
    monkeypatch.setattr(app, "ApiKeyDialog", Dialog)
    monkeypatch.setattr(app, "_settings_from_key", lambda *_: pytest.fail("cancelled dialog must not save"))

    async def scenario():
        task = asyncio.create_task(app.ensure_provider_settings_async())
        while not dialogs:
            await asyncio.sleep(0)
        await dialogs[0].opened.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert dialogs[0].rejected == 1
        assert dialogs[0].finished.handlers == []

    asyncio.run(scenario())


def test_cancellation_during_saved_settings_read_does_not_open_a_late_dialog(monkeypatch):
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()

    def read():
        entered.set()
        release.wait(1)
        settled.set()
        return None

    monkeypatch.setattr(app, "_configured_provider_settings", read)
    monkeypatch.setattr(app, "ApiKeyDialog", lambda *_: pytest.fail("cancelled boot must not open a dialog"))

    async def scenario():
        task = asyncio.create_task(app.ensure_provider_settings_async())
        try:
            await _entered(entered)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        assert await asyncio.to_thread(settled.wait, 3)

    asyncio.run(scenario())


def test_accepted_key_save_failure_propagates_after_worker_settles(monkeypatch):
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()

    class Dialog:
        def __init__(self, _parent):
            self.finished = _Signal()

        def open(self):
            self.finished.emit(app.QDialog.DialogCode.Accepted)

        def key(self):
            return "synthetic-key"

    def save(_key):
        entered.set()
        release.wait(1)
        settled.set()
        raise OSError("synthetic settings write failure")

    monkeypatch.setattr(app, "_configured_provider_settings", lambda: None)
    monkeypatch.setattr(app, "ApiKeyDialog", Dialog)
    monkeypatch.setattr(app, "_settings_from_key", save)

    async def scenario():
        task = asyncio.create_task(app.ensure_provider_settings_async())
        try:
            await _entered(entered)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(OSError, match="synthetic settings write failure"):
            await task
        assert settled.is_set()

    asyncio.run(scenario())


def test_cancel_during_accepted_key_write_returns_no_success_or_new_dialog(monkeypatch):
    # Acceptance already authorized the write; cancellation cannot revoke
    # filesystem work running in a thread. It must not return success, open a
    # second prompt, or invoke a second persistence operation as that work ends.
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    dialogs, writes = [], []

    class Dialog:
        def __init__(self, _parent):
            self.finished = _Signal()
            dialogs.append(self)

        def open(self):
            self.finished.emit(app.QDialog.DialogCode.Accepted)

        def key(self):
            return "synthetic-key"

    def save(key):
        writes.append(key)
        entered.set()
        release.wait(1)
        settled.set()
        return "saved"

    monkeypatch.setattr(app, "_configured_provider_settings", lambda: None)
    monkeypatch.setattr(app, "ApiKeyDialog", Dialog)
    monkeypatch.setattr(app, "_settings_from_key", save)

    async def scenario():
        task = asyncio.create_task(app.ensure_provider_settings_async())
        try:
            await _entered(entered)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not settled.is_set()
        finally:
            release.set()
        assert await asyncio.to_thread(settled.wait, 3)
        assert task.cancelled()
        assert len(dialogs) == 1 and writes == ["synthetic-key"]
        assert dialogs[0].finished.handlers == []

    asyncio.run(scenario())


def test_optional_gateway_import_yields_the_loop_and_keeps_construction_on_owner(monkeypatch):
    import importlib
    from types import SimpleNamespace
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Gateway:
        def __init__(self):
            calls.append(("construct", threading.get_ident()))

    def load(name, package):
        calls.append((name, package, threading.get_ident()))
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(RemoteGateway=Gateway)

    monkeypatch.setattr(importlib, "import_module", load)

    async def scenario():
        task = asyncio.create_task(app._load_remote_gateway_type())
        try:
            await _entered(entered)
            assert not task.done() and calls[0][:2] == (".remote", "orion_core")
            assert calls[0][2] != owner
        finally:
            release.set()
        factory = await task
        assert factory is Gateway
        factory()
        assert calls[-1] == ("construct", owner)

    asyncio.run(scenario())
