"""
The first-run API-key dialog is awaited, not run in a nested event loop.

QDialog.exec() inside the boot coroutine spins a second Qt loop; qasync then
steps other tasks from inside that coroutine's step and Python refuses with
"Cannot enter into task ... while another task ... is being executed". On a
first run the stall detector and metrics sampler, started just before the
dialog, died that way. Here a heartbeat task must keep running while the
dialog is open, and the loop must report no such error.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import qasync
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialog

from orion_core import app as app_module

_QT = None


def _qt():
    """The QApplication, kept referenced: an unreferenced one is collected at
    once, and the next widget built without it aborts the process."""
    global _QT
    _QT = QApplication.instance() or QApplication([])
    return _QT


def _run(coro_factory, answer):
    qt = _qt()
    # The real boot already has its core window. This standalone fixture's
    # dialog is its only window; accepting it must not quit Qt while the
    # provider settings are still being persisted on the worker.
    quit_on_last = qt.quitOnLastWindowClosed()
    qt.setQuitOnLastWindowClosed(False)

    class _Dialog(QDialog):
        def open(self):
            super().open()
            QTimer.singleShot(150, self.accept if answer else self.reject)

        def key(self):
            return "k-123"

    loop = qasync.QEventLoop(qt)
    errors = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    ticks = []

    async def heartbeat():
        while True:
            ticks.append(1)
            await asyncio.sleep(0.01)

    async def flow():
        beat = asyncio.ensure_future(heartbeat())
        try:
            return await coro_factory(_Dialog)
        finally:
            beat.cancel()

    try:
        try:
            with loop:
                result = loop.run_until_complete(flow())
        except SystemExit as exc:
            result = exc
    finally:
        qt.setQuitOnLastWindowClosed(quit_on_last)
    return result, ticks, errors


@pytest.fixture()
def unconfigured(monkeypatch):
    monkeypatch.setattr(app_module, "_configured_provider_settings", lambda: None)
    monkeypatch.setattr(app_module, "_settings_from_key", lambda key: ("settings", key))
    return monkeypatch


def test_an_accepted_key_dialog_keeps_other_tasks_alive(unconfigured):
    def factory(dialog_cls):
        unconfigured.setattr(app_module, "ApiKeyDialog", lambda _w: dialog_cls())
        return app_module.ensure_provider_settings_async(None)

    result, ticks, errors = _run(factory, answer=True)
    assert result == ("settings", "k-123")
    assert len(ticks) >= 5, "the heartbeat stopped while the dialog was open"
    assert not [e for e in errors if "Cannot enter into task" in str(e.get("exception"))]


def test_a_cancelled_key_dialog_exits(unconfigured):
    """Cancelling the first-run dialog ends ORION, as before. Run on a plain
    asyncio loop: raised through qasync, the SystemExit would end the test
    process itself, which is what it does to ORION."""
    _qt()

    class _Rejecting(QDialog):
        def open(self):
            self.reject()           # emits finished(Rejected) synchronously

        def key(self):
            return ""

    unconfigured.setattr(app_module, "ApiKeyDialog", lambda _w: _Rejecting())
    with pytest.raises(SystemExit) as exited:
        asyncio.run(app_module.ensure_provider_settings_async(None))
    assert exited.value.code == 0


def test_configured_settings_skip_the_dialog(monkeypatch):
    monkeypatch.setattr(app_module, "_configured_provider_settings", lambda: "saved")

    def refuse(_w):
        raise AssertionError("no dialog when a provider is configured")
    monkeypatch.setattr(app_module, "ApiKeyDialog", refuse)
    assert asyncio.run(app_module.ensure_provider_settings_async(None)) == "saved"


def test_offline_environment_skips_key_setup_without_writing_credentials(monkeypatch):
    from orion_core import providers
    saved = object()
    monkeypatch.setenv("ORION_MODE", "offline")
    monkeypatch.setattr(providers, "read_provider_settings", lambda: saved)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline bootstrap must not prompt or rewrite credentials")

    monkeypatch.setattr(app_module, "ApiKeyDialog", forbidden)
    monkeypatch.setattr(app_module, "_configured_provider_settings", forbidden)
    monkeypatch.setattr(providers, "write_api_key", forbidden)
    monkeypatch.setattr(providers, "write_provider_settings", forbidden)
    assert app_module.ensure_provider_settings() is saved
    assert asyncio.run(app_module.ensure_provider_settings_async()) is saved


def test_offline_dialog_choice_keeps_booting_without_reading_a_key(unconfigured):
    from orion_core import providers
    _qt()
    saved = object()
    unconfigured.setenv("ORION_MODE", "auto")
    unconfigured.setattr(providers, "read_provider_settings", lambda: saved)

    class Offline(QDialog):
        OFFLINE_RESULT = 2

        def open(self):
            self.done(self.OFFLINE_RESULT)

        def key(self):
            raise AssertionError("offline selection must not read or persist a key")

    unconfigured.setattr(app_module, "ApiKeyDialog", lambda _w: Offline())
    assert asyncio.run(app_module.ensure_provider_settings_async()) is saved
    assert os.environ["ORION_MODE"] == "offline"


def test_key_dialog_exposes_a_distinct_offline_choice():
    from orion_core.gui.widgets import ApiKeyDialog
    from PyQt6.QtWidgets import QPushButton
    _qt()
    dialog = ApiKeyDialog()
    try:
        button = next(button for button in dialog.findChildren(QPushButton)
                      if button.text() == "Continue Offline")
        button.click()
        assert dialog.result() == ApiKeyDialog.OFFLINE_RESULT
        assert dialog.key() == ""
    finally:
        dialog.deleteLater()
