"""
Tests for the Debugger and Docker tabs added to DevelopmentDeckView. Both
tabs call through self.dispatcher.dispatch("debugger"/"docker", ...) — the
identical path a voice/text command takes — so these tests use the same
_StubDispatcher pattern as test_development_deck.py, asserting on the
arguments/results that cross that boundary rather than reaching into
debugger.py/docker_control.py directly (those have their own test files).

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.gui.development_deck import DevelopmentDeckView


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubDispatcher:
    def __init__(self, result: ToolResult | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result = result or ToolResult("ok")

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        return self._result


# ── debugger tab ─────────────────────────────────────────────────────────────

def test_debug_start_calls_debugger_tool_with_path_and_args(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("started"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view.debug_script.setText("C:/scripts/fixture.py")
        view.debug_args.setText("--flag value")
        view._debug_start()
        await asyncio.sleep(0)
        assert dispatcher.calls == [
            ("debugger", {"action": "start", "path": "C:/scripts/fixture.py",
                          "args": ["--flag", "value"]})]
        assert "started" in view.debug_output.toPlainText()

    asyncio.run(_scenario())


def test_debug_start_does_nothing_when_script_blank(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._debug_start()
        await asyncio.sleep(0)
        assert dispatcher.calls == []

    asyncio.run(_scenario())


def test_debug_next_step_continue_buttons_send_the_right_command(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("line 3"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._debug_send("next")
        await asyncio.sleep(0)
        assert dispatcher.calls == [("debugger", {"action": "command", "command": "next"})]

    asyncio.run(_scenario())


def test_debug_command_input_sends_and_clears(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("2"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view.debug_command.setText("p x")
        view._debug_send_input()
        await asyncio.sleep(0)
        assert dispatcher.calls == [("debugger", {"action": "command", "command": "p x"})]
        assert view.debug_command.text() == ""
        assert "(Pdb) p x" in view.debug_output.toPlainText()
        assert "2" in view.debug_output.toPlainText()

    asyncio.run(_scenario())


def test_debug_command_input_does_nothing_when_blank(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._debug_send_input()
        await asyncio.sleep(0)
        assert dispatcher.calls == []

    asyncio.run(_scenario())


def test_debug_stop_calls_debugger_tool(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("Debug session stopped."))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._debug_stop()
        await asyncio.sleep(0)
        assert dispatcher.calls == [("debugger", {"action": "stop"})]
        assert "stopped" in view.debug_output.toPlainText().lower()

    asyncio.run(_scenario())


# ── docker tab ───────────────────────────────────────────────────────────────

def test_docker_refresh_fetches_status_and_list(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()

        async def _dispatch(name, args):
            dispatcher.calls.append((name, args))
            if args["action"] == "status":
                return ToolResult("Docker is available and the daemon is responding.")
            return ToolResult("Containers:\nweb  [nginx]  Up 2 hours")

        dispatcher.dispatch = _dispatch
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._docker_refresh()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ("docker", {"action": "status"}) in dispatcher.calls
        assert ("docker", {"action": "list"}) in dispatcher.calls
        assert "available" in view.docker_status.text().lower()
        assert view.docker_list.count() == 1
        assert view.docker_list.item(0).text().startswith("web")

    asyncio.run(_scenario())


def test_docker_lifecycle_requires_a_selected_container(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("ok"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._docker_lifecycle("stop")
        await asyncio.sleep(0)
        assert dispatcher.calls == []
        assert "select a container" in view.docker_output.toPlainText().lower()

    asyncio.run(_scenario())


def test_docker_lifecycle_acts_on_the_selected_container(_app):
    async def _scenario():
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QListWidgetItem

        dispatcher = _StubDispatcher(ToolResult("docker stop web: OK"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        item = QListWidgetItem("web  [nginx]  Up")
        item.setData(Qt.ItemDataRole.UserRole, "web")
        view.docker_list.addItem(item)
        view.docker_list.setCurrentItem(item)
        view._docker_lifecycle("stop")
        await asyncio.sleep(0)
        assert ("docker", {"action": "stop", "container": "web"}) in dispatcher.calls
        assert "OK" in view.docker_output.toPlainText()

    asyncio.run(_scenario())


def test_docker_show_logs_uses_the_selected_container(_app):
    async def _scenario():
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QListWidgetItem

        dispatcher = _StubDispatcher(ToolResult("log line one\nlog line two"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        item = QListWidgetItem("web  [nginx]  Up")
        item.setData(Qt.ItemDataRole.UserRole, "web")
        view.docker_list.addItem(item)
        view.docker_list.setCurrentItem(item)
        view._docker_show_logs()
        await asyncio.sleep(0)
        assert dispatcher.calls == [("docker", {"action": "logs", "container": "web", "tail": 200})]
        assert "log line one" in view.docker_output.toPlainText()

    asyncio.run(_scenario())


def test_actions_are_inert_with_no_dispatcher_attached(_app):
    view = DevelopmentDeckView(OrionBus(), dispatcher=None)
    view.debug_script.setText("x.py")
    view._debug_start()           # must not raise
    view._debug_send("next")      # must not raise
    view._debug_stop()            # must not raise
    view._docker_refresh()        # must not raise
