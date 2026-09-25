"""
A clean console on exit — without hiding anything real.

  "When ORION shuts down ... I want the shutdown to be nice and smooth in CMD
   just saying 'ORION has shutdown cleanly' - Any errors (that must be
   legitimate and affecting performance and usage) must be in a compiled
   reports folder."

The console had four distinct sources of noise and they needed four different
answers, because only two of them were noise at all:

  1. Over a hundred "Task was destroyed but it is pending!" lines, one per
     self-repair task. A REAL defect — cancel() only schedules cancellation,
     and the old shutdown cleared its task set immediately, so the loop never
     got a turn to deliver a single CancelledError. Fixed at source, not
     filtered; suppressing it would have hidden the bug.
  2. "coroutine ... was never awaited", pointing at _tasks.clear() — the same
     defect wearing a different hat, and a confusing place to be sent since
     nothing is wrong on that line.
  3. "Task exception was never retrieved" for orion-receive-realtime. Also
     real: the session loop raised on the first exceptional task and left the
     other's exception unread.
  4. google.genai and chess.engine chatter. Genuinely benign, genuinely
     third-party, and the only category that gets filtered — to a file, never
     to nowhere.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import console_hygiene  # noqa: E402


# ── the filter: benign chatter is routed, everything else prints ─────────────

@pytest.fixture
def router(tmp_path, monkeypatch):
    monkeypatch.setattr(console_hygiene, "REPORTS_DIR", tmp_path / "console")
    return console_hygiene._Router()


def _record(name: str, message: str, level: int = logging.WARNING):
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


def test_the_genai_non_data_parts_notice_is_routed(router):
    """ORION asks for transcriptions and the model thinks, so audio responses
    legitimately carry 'text' and 'thought' parts. It describes him working."""
    assert router.filter(_record(
        "google.genai.types",
        "Warning: there are non-data parts in the response: ['text', 'thought'], "
        "returning concatenated data result from data parts")) is False


def test_the_stockfish_pv_parse_notice_is_routed(router):
    """python-chess cannot parse one shape of multipv info line. The move is
    still returned and played correctly."""
    assert router.filter(_record(
        "chess.engine",
        "Exception parsing pv from info: 'depth 9 seldepth 15 multipv 1 score "
        "cp 147', position at root: rnbqkbnr/ppppp1pp", logging.ERROR)) is False


def test_a_real_error_from_the_same_library_still_prints(router):
    """Filtering by logger name alone would silence the library entirely."""
    assert router.filter(_record(
        "google.genai.types", "authentication failed: invalid API key",
        logging.ERROR)) is True


def test_orions_own_logs_are_never_touched(router):
    assert router.filter(_record("orion.forge", "something went wrong",
                                 logging.ERROR)) is True
    assert router.filter(_record("root", "unexpected", logging.ERROR)) is True


def test_an_unrecognised_message_prints(router):
    """The failure mode of an over-eager filter is the expensive one."""
    assert router.filter(_record("chess.engine", "engine process died",
                                 logging.ERROR)) is True


def test_routed_lines_are_written_to_disk_not_discarded(router, tmp_path):
    router.filter(_record("google.genai.types",
                          "Warning: there are non-data parts in the response: ['text']"))
    router.close()
    files = list((tmp_path / "console").glob("*console.log"))
    assert files, "the message was discarded rather than routed"
    text = files[0].read_text(encoding="utf-8")
    assert "non-data parts" in text
    assert "kept quiet because" in text, "no reason recorded for the suppression"


def test_every_filter_entry_carries_a_justification():
    """A filter entry nobody can justify is one that will hide something real."""
    for prefix, pattern, why in console_hygiene.BENIGN:
        assert prefix and pattern is not None
        assert len(why) > 40, f"{prefix} has no real justification: {why!r}"


def test_the_filter_never_raises_on_a_malformed_record(router):
    bad = logging.LogRecord("chess.engine", logging.ERROR, __file__, 1,
                            "%d %d", (1,), None)   # not enough args
    assert router.filter(bad) is True


def test_installing_twice_is_harmless():
    console_hygiene.install()
    console_hygiene.install()


# ── real faults reach the reports folder ─────────────────────────────────────

def test_a_fault_is_written_where_it_can_be_found(tmp_path, monkeypatch):
    monkeypatch.setattr(console_hygiene, "REPORTS_DIR", tmp_path / "console")
    try:
        raise ValueError("something genuinely broke")
    except ValueError as exc:
        path = console_hygiene.report_fault(exc, context="shutdown")
    assert path is not None and path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "something genuinely broke" in body
    assert "Traceback" in body
    assert "shutdown" in body


# ── the exit messages ────────────────────────────────────────────────────────

def _app_source() -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")


def test_no_test_here_imports_app_py():
    """A guard, because the failure it prevents is invisible: importing app.py
    pulls in QtWebEngine, which crashes the interpreter partway through a LATER
    test file — so the blame lands somewhere else entirely.

    Walks the AST rather than grepping the text: a substring check matched its
    own assertion and failed on a file that was already correct.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert "orion_core.app" not in imported, (
        "importing app.py here will crash a later Qt test file")


def test_a_clean_exit_says_exactly_what_was_asked_for():
    assert '"ORION has shutdown cleanly."' in _app_source()


def test_a_crash_does_not_dump_a_traceback_to_the_console_by_default():
    """A traceback in CMD scrolls past and is gone. One line plus a file path
    is both calmer and more useful."""
    source = _app_source()
    assert "_report_exit_fault" in source
    assert "ORION stopped after a fault" in source


def test_a_quiet_console_still_says_how_much_was_routed():
    """'Quiet' must never be mistaken for 'nothing happened'."""
    assert "_quiet_report" in _app_source()


def test_the_clean_shutdown_classifier_handles_the_qasync_message():
    """Imported from console_hygiene, NOT app.py. Importing app.py pulls in
    QtWebEngine, which hard-crashes every Qt test that runs after it in the
    same process — no traceback, just a dead interpreter partway through the
    suite. That is why the predicate lives in a Qt-free module."""
    from orion_core.console_hygiene import is_clean_shutdown

    exc = RuntimeError("Event loop stopped before Future completed.")
    assert is_clean_shutdown(exc, requested=True)
    assert is_clean_shutdown(exc, requested=False)


def test_a_genuine_runtime_error_is_still_a_fault():
    from orion_core.console_hygiene import is_clean_shutdown

    assert not is_clean_shutdown(RuntimeError("dictionary changed size"),
                                 requested=False)


# ── the defects that were fixed rather than filtered ─────────────────────────

def test_cancelled_repairs_are_awaited_not_just_cancelled():
    """cancel() only SCHEDULES cancellation. The old shutdown cleared its task
    set immediately, so the loop never got a turn to deliver a single
    CancelledError — over a hundred 'Task was destroyed but it is pending!'
    lines on one exit."""
    import inspect

    from orion_core.selfrepair import SelfRepairAgent

    shutdown = inspect.getsource(SelfRepairAgent.shutdown)
    assert "_tasks.clear()" not in shutdown, (
        "still dropping references while tasks are mid-cancellation")
    drain = inspect.getsource(SelfRepairAgent.drain)
    assert "asyncio.wait" in drain


def test_the_drain_actually_settles_cancelled_tasks():
    from orion_core.selfrepair import SelfRepairAgent

    async def main():
        agent = SelfRepairAgent.__new__(SelfRepairAgent)
        agent._tasks = set()

        async def forever():
            await asyncio.sleep(60)

        for _ in range(5):
            agent._tasks.add(asyncio.ensure_future(forever()))
        for task in agent._tasks:
            task.cancel()
        settled = await agent.drain(timeout=2.0)
        assert settled == 5
        assert agent._tasks == set()

    asyncio.run(main())


def test_the_drain_is_bounded():
    """A repair wedged in a blocking call must not hold shutdown open."""
    from orion_core.selfrepair import SelfRepairAgent

    async def main():
        agent = SelfRepairAgent.__new__(SelfRepairAgent)
        agent._tasks = {asyncio.ensure_future(asyncio.sleep(30))}
        loop = asyncio.get_running_loop()
        started = loop.time()
        await agent.drain(timeout=0.2)
        assert loop.time() - started < 2.0
        for task in list(agent._tasks) or []:
            task.cancel()

    asyncio.run(main())


def test_concurrent_repairs_are_capped():
    """Task-291 through Task-403 alive at once is a symptom, not a workload."""
    from orion_core.selfrepair import SelfRepairAgent

    assert 1 <= SelfRepairAgent.MAX_CONCURRENT_REPAIRS <= 16
    import inspect
    source = inspect.getsource(SelfRepairAgent._schedule_auto_repair)
    assert "MAX_CONCURRENT_REPAIRS" in source


def test_every_session_task_exception_is_retrieved():
    """Raising on the first exceptional task left the other's exception unread,
    which Python reports later as 'Task exception was never retrieved'."""
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    source = inspect.getsource(GenAILiveWorker)
    marker = source.index("failures = []")
    window = source[marker:marker + 400]
    assert "for task in done" in window
    assert "failures.append" in window
    assert "raise failures[0]" in window


def test_spawn_without_a_loop_closes_the_coroutine():
    """An un-awaited coroutine emits 'coroutine ... was never awaited' from
    whatever line the collector happens to run on — which is how shutdown
    printed warnings pointing at unrelated code."""
    from orion_core import background

    async def work():
        return 1

    coro = work()
    assert background.spawn(coro, name="no-loop") is None
    with pytest.raises(RuntimeError):
        coro.send(None)          # already closed
