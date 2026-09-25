"""
Tests for DevelopmentDeckView (Mark XX design-spec, "Automation + Development
pages" follow-up) — a real workspace calling the same dev_workbench
dispatcher tool a voice/text command would (repo analysis, allow-listed
command execution, line-numbered file reading), not a decorative page or a
parallel/weaker reimplementation.

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


def test_analyse_repo_calls_dev_workbench_with_the_typed_path(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("Repository analysis: ..."))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view.repo_path.setText("C:/some/repo")
        view._analyse_repo()
        await asyncio.sleep(0)
        assert dispatcher.calls == [("dev_workbench", {"action": "analyse_repo", "path": "C:/some/repo"})]
        assert "Repository analysis" in view.repo_output.toPlainText()

    asyncio.run(_scenario())


def test_analyse_repo_defaults_to_current_directory_when_blank(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._analyse_repo()
        await asyncio.sleep(0)
        assert dispatcher.calls[0][1]["path"] == "."

    asyncio.run(_scenario())


def test_analyse_repo_shows_a_working_indicator_before_the_result_lands(_app):
    # A stub whose dispatch() completes instantly makes the "Analysing…"
    # state disappear within the same scheduling tick, so it needs a real
    # suspension point (an asyncio.Event) to observe deterministically
    # rather than a race against however fast the stub happens to return.
    async def _scenario():
        gate = asyncio.Event()

        class _SlowDispatcher:
            async def dispatch(self, name, args):
                await gate.wait()
                return ToolResult("done analysing")

        view = DevelopmentDeckView(OrionBus(), _SlowDispatcher())
        view._analyse_repo()
        await asyncio.sleep(0)   # let the task start and reach the await point
        assert "Analysing" in view.repo_output.toPlainText()
        gate.set()
        await asyncio.sleep(0)
        assert "done analysing" in view.repo_output.toPlainText()

    asyncio.run(_scenario())


def test_run_command_calls_dev_workbench_with_the_command_and_cwd(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("exit code 0\nall tests passed"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view.repo_path.setText("C:/some/repo")
        view.command_input.setText("pytest -q")
        view._run_command()
        await asyncio.sleep(0)
        assert dispatcher.calls == [
            ("dev_workbench", {"action": "run_command", "command": "pytest -q", "path": "C:/some/repo"})]
        assert "all tests passed" in view.command_output.toPlainText()

    asyncio.run(_scenario())


def test_run_command_does_nothing_when_blank(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._run_command()
        await asyncio.sleep(0)
        assert dispatcher.calls == []

    asyncio.run(_scenario())


def test_read_file_calls_dev_workbench_with_line_range(_app):
    async def _scenario():
        dispatcher = _StubDispatcher(ToolResult("    1  import os"))
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view.file_path.setText("orion_core/app.py")
        view.start_line.setText("10")
        view.line_count.setText("50")
        view._read_file()
        await asyncio.sleep(0)
        assert dispatcher.calls == [(
            "dev_workbench",
            {"action": "read_file", "path": "orion_core/app.py", "start_line": 10, "line_count": 50})]
        assert "import os" in view.code_output.toPlainText()

    asyncio.run(_scenario())


def test_read_file_falls_back_to_sane_defaults_on_bad_input(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view.file_path.setText("orion_core/app.py")
        view.start_line.setText("not a number")
        view.line_count.setText("also not a number")
        view._read_file()
        await asyncio.sleep(0)
        args = dispatcher.calls[0][1]
        assert args["start_line"] == 1
        assert args["line_count"] == 200

    asyncio.run(_scenario())


def test_read_file_does_nothing_when_path_blank(_app):
    async def _scenario():
        dispatcher = _StubDispatcher()
        view = DevelopmentDeckView(OrionBus(), dispatcher)
        view._read_file()
        await asyncio.sleep(0)
        assert dispatcher.calls == []

    asyncio.run(_scenario())


def test_actions_are_inert_with_no_dispatcher_attached(_app):
    view = DevelopmentDeckView(OrionBus(), dispatcher=None)
    view.repo_path.setText("x")
    view._analyse_repo()          # must not raise
    view.command_input.setText("git status")
    view._run_command()           # must not raise
    view.file_path.setText("x.py")
    view._read_file()             # must not raise
