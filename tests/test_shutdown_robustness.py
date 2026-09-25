"""
Shutdown robustness (Mark XXII):
  · a watchdog force-exits so ORION always FULLY shuts down, even if a
    background thread (audio/GL/torch) refuses to die;
  · the goodbye gets a tail so it is never clipped mid-sentence.

app.py imports QtWebEngine at module scope (crashes the offscreen Qt platform
for later tests), so its wiring is checked by reading the source, not importing.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "orion_core"


def test_shutdown_watchdog_exists_and_is_armed():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _arm_shutdown_watchdog" in src
    assert "os._exit(0)" in src                       # the last-resort kill
    assert "daemon=True" in src
    # armed the moment shutdown is requested
    assert "_arm_shutdown_watchdog()" in src
    idx_def = src.index("def request_shutdown")
    # the arm call appears inside/after the handler, not only in the definition
    assert "_arm_shutdown_watchdog()" in src[idx_def:]


def test_await_farewell_has_a_tail():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    # the farewell no longer returns the instant the pipeline reports idle
    assert "asyncio.sleep(1.3)" in src


def test_live_farewell_waits_longer_and_tails():
    """The flat 2.4s sleep cut off the longer sign-offs.

    Asserted against the wait's real defaults rather than a literal in the
    source: the ceiling moved into _await_speech when the farewell started
    coming from the MODEL, whose words arrive a second or three after the turn
    is sent. What matters is that the ceiling is generous and the tail exists,
    not which function holds the number.
    """
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    parameters = inspect.signature(GenAILiveWorker._await_speech).parameters
    assert parameters["end_limit"].default >= 10.0, "too short for a real goodbye"
    assert parameters["tail"].default >= 1.0, "no tail — the last word is clipped"
    # And he is waited for at all, from the shutdown path.
    assert "_await_speech" in inspect.getsource(
        GenAILiveWorker._shutdown_after_farewell)


def test_the_wait_covers_him_starting_as_well_as_finishing():
    """A live turn is silent for a second or three after it is sent. Polling
    only for "not busy" finds that silence, decides he has finished, and fires
    the shutdown before he has said a word."""
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    source = inspect.getsource(GenAILiveWorker._await_speech)
    assert "start_limit" in source
    assert source.index("start_limit") < source.index("end_limit")


# ── the shutdown report (orion_core/shutdown_trace.py) ──────────────────────

def _trace_with_clock():
    from orion_core.shutdown_trace import ShutdownTrace
    now = [100.0]
    return ShutdownTrace(clock=lambda: now[0]), now


def test_a_forced_report_names_the_step_it_interrupted():
    """The whole point: a quit that ran past the watchdog used to leave no
    trace of which of ~58 steps was holding it."""
    trace, now = _trace_with_clock()
    trace.begin()
    with trace.phase("farewell"):
        now[0] += 3.0
    with trace.phase("worker"):
        with trace.phase("live session close"):
            now[0] += 22.0
            report = trace.report(forced=True)
    assert report.startswith("FORCED by the shutdown watchdog after 25.0 s.")
    assert "Still inside: worker > live session close" in report
    assert trace.current is None                  # the stack unwound


def test_a_clean_report_lists_the_slowest_steps_first():
    trace, now = _trace_with_clock()
    trace.begin()
    for label, seconds in (("memory", 0.1), ("farewell", 3.8), ("worker", 0.2)):
        with trace.phase(label):
            now[0] += seconds
    lines = trace.report().splitlines()
    assert lines[0] == "Clean shutdown after 4.1 s."
    assert [line.split("s  ", 1)[1] for line in lines[2:5]] == ["farewell", "worker", "memory"]


def test_threads_the_process_must_still_wait_for_are_named():
    import threading

    from orion_core.shutdown_trace import lingering_threads
    release = threading.Event()
    busy = threading.Thread(target=release.wait, name="stuck-job")
    idle = threading.Thread(target=release.wait, name="background-daemon", daemon=True)
    busy.start()
    idle.start()
    try:
        found = lingering_threads()
    finally:
        release.set()
        busy.join()
        idle.join()
    assert any(entry.startswith("stuck-job  at threading.py") for entry in found), found
    assert not any(entry.startswith("background-daemon") for entry in found)


def test_writing_the_report_never_raises(tmp_path):
    trace, _now = _trace_with_clock()
    trace.begin()
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    assert trace.write(blocker / "last_shutdown.txt") is False


def test_he_leaves_the_screen_once_he_has_said_goodbye():
    """Nothing ever closed the windows: the last frame stayed up, frozen,
    until the interpreter finished exiting."""
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    farewell = src.index("await _await_farewell(worker.speech)")
    off_screen = src.index('_safe("off screen"')
    first_service_stop = src.index('_safe("proactive", proactive.stop)')
    assert farewell < off_screen < first_service_stop
