"""
Project-wide pytest configuration.

The suite has no async plugin (pytest-asyncio et al.) and deliberately keeps it
that way — the convention is plain sync tests that wrap coroutines in
``asyncio.run(...)``.  The one hook below makes a bare ``async def test_*``
function *also* run correctly instead of erroring out with

    "async def functions are not natively supported"

so an accidentally-async test can never silently masquerade as a failure again.
It activates only for coroutine test functions; every existing sync test is
untouched.

It also keeps generated copies of the project out of collection — see below.
"""

from __future__ import annotations

import asyncio
import inspect
import os

import pytest

#: Directories that contain COPIES of this project rather than the project.
#:
#: tools/prepare_public_source.py writes a publication export to release/, and
#: build_standalone.py writes a frozen one to dist/. Each contains a complete
#: tests/ tree whose modules have the same basenames as the real ones, so
#: pytest imports two different files as (say) `test_memory.py`, hits the
#: import-file mismatch, and aborts the whole run. A single validation export
#: took collection from 5,584 passing tests to 626 collection errors and zero
#: tests run — the suite does not fail loudly in that state, it simply stops
#: existing, which is the worst way for a safety net to break.
#:
#: These are gitignored, but .gitignore has no bearing on what pytest walks.
# ORION's real face is the Three.js one, rendered in a QWebEngineView. That
# is right for the application and wrong for this suite: every window built in
# a test would start a Chromium compositor, and several of them in one process
# wedge it — the run stops dead partway through rather than failing, which is
# the worst way for a suite to break because it looks like a slow machine.
#
# It could not be verified here anyway: WebEngine will not even grab to a
# pixmap without a real GPU compositor, so a test looking at that face would
# be looking at nothing.
#
# So the suite exercises the software-rendered face, which is deterministic
# and headless-safe, and tests/test_mark_identity.py and the renderer
# migration tests assert separately that the APPLICATION prefers the WebGL
# one. Set ORION_HOLO_HEAD=0 to run the suite against WebEngine deliberately.
os.environ.setdefault("ORION_HOLO_HEAD", "1")


collect_ignore_glob = [
    "release/*",
    "dist/*",
    "build/*",
    ".venv/*",
    "android/*",
    # ORION writes whole projects of his own under projects/, each with its
    # own tests that import from its own src/ root. Walked by this suite they
    # fail to import and abort collection — the same way a release export
    # once did, and with the same symptom: not a red suite, an ABSENT one.
    "projects/*",
]


# ── the suite must never touch ORION's real data ────────────────────────────
#
# An audit on 2026-09-23 caught the suite writing into the REAL config/ on
# every run: rewriting identity.json, teaching the chess brain test games,
# logging fake "API key failing" events into the provider diagnostics ORION
# reports from, opening (and schema-writing) the finance, decisions, study,
# focus, wellbeing, language and standing-questions databases, and — above
# all — 1,600+ synthetic incidents in the self-repair journal plus 359 proposal
# files. Two layers stop that: the fixture below points every such path at a
# temporary directory, and an audit hook REFUSES any write that still reaches
# config/, failing the run with the test's name even if the code swallows it.

_REAL_CONFIG = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(__file__), "config")))
_CONFIG_WRITES: list[tuple[str, str]] = []
_CURRENT_TEST = ["<collection>"]


def _into_real_config(path) -> bool:
    try:
        candidate = os.path.normcase(os.path.abspath(os.fspath(path)))
    except TypeError:
        return False
    return (candidate == _REAL_CONFIG or candidate.startswith(_REAL_CONFIG + os.sep)) \
        and os.sep + "__pycache__" + os.sep not in candidate


def _refuse_real_config_writes(event, args):
    if event == "open":
        mode = args[1] if len(args) > 1 else "r"
        target = args[0]
        writing = isinstance(mode, str) and any(flag in mode for flag in "wax+")
    elif event == "sqlite3.connect":
        target, writing = (args[0] if args else None), True
        if target in (None, ":memory:", "") or str(target).startswith("file::memory:"):
            return
    elif event in ("os.replace", "os.rename"):
        target, writing = args[1], True          # (src, dst, src_dir_fd, dst_dir_fd)
    elif event == "os.remove":
        target, writing = args[0], True
    else:
        return
    if writing and _into_real_config(target):
        _CONFIG_WRITES.append((_CURRENT_TEST[0], os.fspath(target)))
        raise PermissionError(
            f"test wrote into ORION's REAL config: {target} — isolate it (see conftest.py)")


import sys as _sys  # noqa: E402
_sys.addaudithook(_refuse_real_config_writes)


def pytest_runtest_setup(item):
    _CURRENT_TEST[0] = item.nodeid


# ── an exception inside a Qt callback must fail a test, not kill the run ────
#
# PyQt6 passes an exception raised in a Python override of a Qt virtual (an
# eventFilter, a paintEvent, a slot Qt invokes) to sys.excepthook — and when
# that is still Python's default it calls qFatal instead. On Windows qFatal is
# a __fastfail: the process dies with 0xC0000409, pytest prints nothing and
# faulthandler never runs. ORION never meets that, because selfrepair installs
# a hook at startup; the suite had none. On 2026-09-25 that killed full runs
# at ~93%, intermittently and with no test named: the garbage collector had
# emptied a deck's __dict__ before destroying it, and its eventFilter raised
# AttributeError into Qt. This hook reports it against the running test.

_CALLBACK_ERRORS: list[tuple[str, str]] = []
_CALLBACK_ERRORS_SEEN = [0]


def _record_callback_error(exc_type, exc, tb) -> None:
    import traceback
    detail = "".join(traceback.format_exception(exc_type, exc, tb))
    _CALLBACK_ERRORS.append((_CURRENT_TEST[0], detail))
    try:
        print(f"\nUNCAUGHT EXCEPTION IN A QT CALLBACK during {_CURRENT_TEST[0]}:\n{detail}",
              file=_sys.__stderr__, flush=True)
    except Exception:
        pass


_sys.excepthook = _record_callback_error


@pytest.hookimpl(trylast=True)       # after the test's own fixtures have torn down
def pytest_runtest_teardown(item, nextitem):
    new = _CALLBACK_ERRORS[_CALLBACK_ERRORS_SEEN[0]:]
    _CALLBACK_ERRORS_SEEN[0] = len(_CALLBACK_ERRORS)
    if new:
        pytest.fail(
            "a Qt callback raised during this test — possibly while the garbage "
            "collector destroyed objects an EARLIER test left behind. Without "
            "conftest's excepthook this kills the whole run with 0xC0000409:\n\n"
            + "\n".join(detail for _test, detail in new), pytrace=False)


def _threads_that_block_exit() -> list[str]:
    """Non-daemon threads the interpreter will wait for once pytest returns.

    Executor workers are left out: concurrent.futures registers its own exit
    hook that stops them, so they never hold the process open. A thread that
    is merely finishing gets a short grace period before it counts.
    """
    import threading
    import time as _time
    candidates = []
    for thread in threading.enumerate():
        if thread.daemon or thread is threading.main_thread():
            continue
        target = getattr(thread, "_target", None)
        if getattr(target, "__module__", "") == "concurrent.futures.thread":
            continue
        candidates.append(thread)
    deadline = _time.monotonic() + 2.0
    for thread in candidates:
        thread.join(timeout=max(0.0, deadline - _time.monotonic()))
    return [thread.name for thread in candidates if thread.is_alive()]


def pytest_sessionfinish(session, exitstatus):
    if _CONFIG_WRITES:
        lines = "\n".join(f"  {test}  ->  {path}" for test, path in sorted(set(_CONFIG_WRITES)))
        print(f"\n\nTESTS TRIED TO WRITE INTO THE REAL config/ ({len(_CONFIG_WRITES)}):\n{lines}")
        session.exitstatus = 1
    if _CALLBACK_ERRORS:                 # including any raised after the last teardown
        names = "\n".join(f"  {test}" for test in sorted({test for test, _ in _CALLBACK_ERRORS}))
        print(f"\n\nQT CALLBACKS RAISED ({len(_CALLBACK_ERRORS)}) — a fatal crash "
              f"without conftest's excepthook; tracebacks above:\n{names}")
        session.exitstatus = 1
    # A test that leaves a non-daemon thread running does not fail anything:
    # the summary prints and then the process simply never exits (a leaked
    # python-chess engine thread did exactly that on 2026-09-23). Name it.
    blocking = _threads_that_block_exit()
    if blocking:
        print(f"\n\nTHREADS STILL RUNNING — the interpreter will wait for these "
              f"and the run will not exit: {blocking}")
        session.exitstatus = 1


@pytest.fixture(scope="session")
def _test_data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("orion_config")


@pytest.fixture(autouse=True)
def _point_user_data_at_a_temp_dir(_test_data_dir):
    """Every store a test can reach without passing its own path.

    A PRIVATE MonkeyPatch, never the shared ``monkeypatch`` fixture: requesting
    that from an autouse fixture creates it first, so it tears down LAST — after
    a test module's own fixtures, which then run their cleanup while the test's
    patches are still in place (browser.reset_cache() met a patched function
    with no cache_clear in test_browser_and_autostart).
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _redirect_user_data(monkeypatch, _test_data_dir)


def _redirect_user_data(monkeypatch, _test_data_dir):
    import orion_core.chess_brain as chess_brain
    import orion_core.chess_sound as chess_sound
    import orion_core.commerce as commerce
    import orion_core.decisions as decisions
    import orion_core.finance as finance
    import orion_core.focus as focus
    import orion_core.identity as identity
    import orion_core.ingestion as ingestion
    import orion_core.language_tutor as language_tutor
    import orion_core.provider_diagnostics as provider_diagnostics
    import orion_core.standing_questions as standing_questions
    import orion_core.study as study
    import orion_core.wellbeing as wellbeing
    data = _test_data_dir
    for module in (finance, decisions, focus, study, wellbeing, ingestion):
        monkeypatch.setattr(module, "CONFIG_DIR", data)      # their own data only
    monkeypatch.setattr(standing_questions, "STORE_PATH", data / "standing_questions.db")
    monkeypatch.setattr(language_tutor, "DECK_PATH", data / "language_deck.db")
    monkeypatch.setattr(chess_brain, "BRAIN_PATH", data / "chess_brain.json")
    monkeypatch.setattr(chess_sound, "SOUND_DIR", data / "sounds")
    monkeypatch.setattr(identity, "IDENTITY_PATH", data / "identity.json")
    monkeypatch.setattr(provider_diagnostics, "DIAGNOSTICS_JOURNAL",
                        data / "diagnostics" / "provider_events.jsonl")
    monkeypatch.setattr(commerce, "COMMERCE_SIGNAL_DB", data / "commerce_signals.db")
    # MCP tool-list cache and the Live connection journal.
    import orion_core.live_worker as live_worker
    import orion_core.mcp_host as mcp_host
    monkeypatch.setattr(mcp_host, "MCP_CATALOGUE_PATH", data / "mcp_catalogue.json")
    monkeypatch.setattr(mcp_host, "MCP_CONFIG_PATH", data / "mcp_servers.json")
    monkeypatch.setattr(live_worker, "LIVE_DIAG_PATH",
                        data / "diagnostics" / "live_sessions.jsonl")
    # The learning brain and the resolver's evidence. observe_async is stubbed
    # rather than redirected: it learns on a background thread that can outlive
    # the test — and so this redirect — and would then write the real config.
    import orion_core.intent_brain as intent_brain
    import orion_core.resolver_shadow as resolver_shadow
    import orion_core.vision_rules as vision_rules
    monkeypatch.setattr(intent_brain, "BRAIN_DIR", data / "brain")
    monkeypatch.setattr(intent_brain, "_BRAIN", None)
    monkeypatch.setattr(intent_brain, "observe_async", lambda *a, **k: None)
    monkeypatch.setattr(intent_brain, "warm", lambda: None)
    monkeypatch.setattr(resolver_shadow, "DEFAULT_PATH", data / "resolver_shadow.db")
    monkeypatch.setattr(vision_rules, "RULES_PATH", data / "vision_rules.json")
    yield


@pytest.fixture(scope="session")
def _self_repair_scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("self_repair")


@pytest.fixture(autouse=True)
def _keep_self_repair_out_of_the_real_config(_self_repair_scratch):
    """No test may write into the REAL config/self_repair.

    Several did. Every full run appended synthetic incidents to the journal
    ORION reads back as his own history (1,600+ fake "captured" entries by
    2026-09-23) and left a repair proposal .md behind — 359 of them. Tests that
    want their own directory still monkeypatch these themselves; this only
    changes where a test that forgot ends up writing.
    """
    import orion_core.selfrepair as selfrepair
    with pytest.MonkeyPatch.context() as monkeypatch:     # private: see above
        monkeypatch.setattr(selfrepair, "SELF_REPAIR_DIR", _self_repair_scratch)
        monkeypatch.setattr(selfrepair, "JOURNAL_PATH", _self_repair_scratch / "journal.jsonl")
        monkeypatch.setattr(selfrepair.SelfRepairAgent, "BACKUP_DIR",
                            _self_repair_scratch / "backups")
        yield


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    test_fn = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_fn):
        return None  # ordinary sync test — let pytest handle it.
    argnames = pyfuncitem._fixtureinfo.argnames
    kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
    asyncio.run(test_fn(**kwargs))
    return True  # we drove the coroutine to completion.
