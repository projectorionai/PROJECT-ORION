"""Real widget regressions for research input and asynchronous run ownership."""
import asyncio
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from orion_core import background
from orion_core.gui.research_console import ResearchConsole


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_enter_cannot_launch_a_second_research_job(qapp):
    async def scenario():
        calls = []
        gate = asyncio.Event()

        async def dispatch(tool, args):
            calls.append((tool, args))
            await gate.wait()
            return SimpleNamespace(text="Research complete")

        view = ResearchConsole(None, SimpleNamespace(dispatch=dispatch))
        view.topic.setText("Materials science")
        try:
            view._start()
            view.topic.returnPressed.emit()
            await asyncio.sleep(0)
            assert len(calls) == 1
        finally:
            gate.set()
            await background.drain(1)
            view.deleteLater()

    asyncio.run(scenario())


def test_without_a_running_loop_the_research_button_recovers(qapp):
    async def dispatch(*args):
        pytest.fail("dispatch must not run without an event loop")

    view = ResearchConsole(None, SimpleNamespace(dispatch=dispatch))
    view.topic.setText("Materials science")
    view._start()
    assert view.start_btn.isEnabled()
    assert "not started" in view.status.text().lower()
    view.deleteLater()


@pytest.mark.parametrize("before_start", [False, True])
def test_cancellation_allows_research_to_be_started_again(qapp, before_start):
    async def scenario():
        async def dispatch(*args):
            await asyncio.Event().wait()

        view = ResearchConsole(None, SimpleNamespace(dispatch=dispatch))
        view.topic.setText("Materials science")
        view._start()
        task = next(t for t in background._RUNNING if t.get_name() == "orion-research-console")
        if not before_start:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        assert view.start_btn.isEnabled()
        assert "cancelled" in view.status.text().lower()
        view.deleteLater()

    asyncio.run(scenario())


@pytest.mark.parametrize("timestamp", ["invalid", float("inf"), float("nan"), 1e300, {}])
def test_bad_event_timestamp_still_displays_the_step(qapp, timestamp):
    view = ResearchConsole(None)
    view.add_step({"kind": "reading", "message": "Reading a source", "at": timestamp})
    assert view.feed.count() == 1
    view.deleteLater()


@pytest.mark.parametrize("count", ["invalid", -1, float("inf"), float("nan"), 2**80, {}])
def test_invalid_progress_cannot_crash_a_research_event(qapp, count):
    view = ResearchConsole(None)
    view.add_step({"kind": "planned", "message": "Plan ready", "sections": count})
    view.add_step({"kind": "section_done", "message": "Section ready", "total": 4, "done": count})
    assert 0 <= view.progress.value() <= 4
    view.deleteLater()


def test_reset_ends_the_previous_paper_stream(qapp):
    view = ResearchConsole(None)
    view.add_step({"kind": "paper_start", "message": "Old paper"})
    view.reset()
    view.add_step({"kind": "reading", "message": "Old source"})
    view.add_step({"kind": "start", "message": "New investigation"})
    assert view.feed.count() == 1
    view.deleteLater()
