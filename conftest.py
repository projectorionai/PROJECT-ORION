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

# A gateway started by a test must never bind the real home-network address
# or ask Windows Firewall for a rule (remote_lan.py). The LAN link's own tests
# switch it back on with fakes for the address, the sockets and netsh.
os.environ.setdefault("ORION_REMOTE_LAN", "0")


collect_ignore_glob = [
    "release/*",
    "dist/*",
    "build/*",
    # Audit evidence includes pre-edit source and isolated patch replays.
    "reports/*",
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
        candidate = os.path.normcase(os.path.abspath(os.fsdecode(path)))
    except TypeError:
        return False
    return (candidate == _REAL_CONFIG or candidate.startswith(_REAL_CONFIG + os.sep)) \
        and os.sep + "__pycache__" + os.sep not in candidate


def _refuse_real_config_writes(event, args):
    targets = ()
    if event == "open":
        mode = args[1] if len(args) > 1 else "r"
        targets = (args[0],)
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(mode, str) and any(flag in mode for flag in "wax+")
        # os.open audits with mode=None and integer flags, unlike built-in open.
        writing |= isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
    elif event == "sqlite3.connect":
        target, writing = (args[0] if args else None), True
        if target in (None, ":memory:", "") or str(target).startswith("file::memory:"):
            return
        targets = (target,)
    elif event in ("os.replace", "os.rename"):
        # Moving an existing config file OUT also mutates the real store.
        targets, writing = args[:2], True       # (src, dst, src_dir_fd, dst_dir_fd)
    elif event == "os.remove":
        targets, writing = (args[0],), True
    else:
        return
    for target in targets:
        if writing and _into_real_config(target):
            _CONFIG_WRITES.append((_CURRENT_TEST[0], os.fsdecode(target)))
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


@pytest.fixture(scope="session", autouse=True)
def _keep_qt_application_alive():
    """Own the actual application through every module's fixture teardown.

    Destroying QApplication also destroys native widgets/timers whose Python
    wrappers can still exist. GUI test collection already imports QtWidgets;
    use that module without importing optional Qt for a backend-only run.
    Tests that import QtWidgets later must explicitly own their application.
    """
    qtwidgets = _sys.modules.get("PyQt6.QtWidgets")
    application_type = getattr(qtwidgets, "QApplication", None)
    if application_type is None:
        yield None
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = application_type.instance() or application_type([])
    yield application


@pytest.fixture(autouse=True)
def _point_user_data_at_a_temp_dir(_test_data_dir, tmp_path):
    """Every store a test can reach without passing its own path.

    A PRIVATE MonkeyPatch, never the shared ``monkeypatch`` fixture: requesting
    that from an autouse fixture creates it first, so it tears down LAST — after
    a test module's own fixtures, which then run their cleanup while the test's
    patches are still in place (browser.reset_cache() met a patched function
    with no cache_clear in test_browser_and_autostart).
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _redirect_user_data(monkeypatch, _test_data_dir, tmp_path / "reminders.json")


def _redirect_user_data(monkeypatch, _test_data_dir, reminders_path):
    import orion_core.chess_brain as chess_brain
    import orion_core.chess_sound as chess_sound
    import orion_core.commerce as commerce
    import orion_core.decisions as decisions
    import orion_core.finance as finance
    import orion_core.fitness as fitness
    import orion_core.focus as focus
    import orion_core.identity as identity
    import orion_core.ingestion as ingestion
    import orion_core.language_tutor as language_tutor
    import orion_core.provider_diagnostics as provider_diagnostics
    import orion_core.reminders as reminders
    import orion_core.standing_questions as standing_questions
    import orion_core.study as study
    import orion_core.wellbeing as wellbeing
    data = _test_data_dir
    from orion_core import research_runs
    monkeypatch.setattr(research_runs, "DEFAULT_PATH", reminders_path.parent / "research_runs.json")
    monkeypatch.setattr(research_runs, "_DEFAULT", None)
    monkeypatch.setattr(reminders, "REMINDERS_PATH", reminders_path)
    for module in (finance, decisions, fitness, focus, study, wellbeing, ingestion):
        monkeypatch.setattr(module, "CONFIG_DIR", data)      # their own data only
    monkeypatch.setattr("orion_core.goals.CONFIG_DIR", data)  # goals.db (and its legacy imports)
    # The owner's real fresh start would hide every test board's past rows.
    monkeypatch.setattr("orion_core.owner_tasks.FRESH_START_PATH", data / "fresh_start.json")
    monkeypatch.setattr("orion_core.gui.command_overview.BRAIN_VIEW_PATH", data / "brain_view.json")
    # Tests see no CUDA unless they ask for it: the same results on any PC,
    # and no test loads a model onto the owner's graphics card.
    monkeypatch.setattr("orion_core.accelerator._cuda", False)
    # Nor DirectML: the card may be full with training or a game, and a test
    # must not depend on it. Tests of the GPU path switch it back on themselves.
    monkeypatch.setenv("ORION_GPU", "0")
    # Agent routing by meaning depends on whether some earlier test loaded
    # the sentence encoder; tests route by keywords unless they opt in.
    monkeypatch.setattr("orion_core.agent_router.ENABLED", False)
    # A test that opens a file never waits for a real window to appear.
    monkeypatch.setattr("orion_core.file_search.VERIFY_OPENS", False)
    monkeypatch.setattr("orion_core.context_capsules.default_path",
                        lambda: reminders_path.parent / "context_capsules.db")
    monkeypatch.setattr("orion_core.experiment_notebook.default_path",
                        lambda: reminders_path.parent / "experiment_notebook.db")
    monkeypatch.setattr(standing_questions, "STORE_PATH", data / "standing_questions.db")
    monkeypatch.setattr(language_tutor, "DECK_PATH", data / "language_deck.db")
    monkeypatch.setattr(chess_brain, "BRAIN_PATH", data / "chess_brain.json")
    monkeypatch.setattr(chess_sound, "SOUND_DIR", data / "sounds")
    monkeypatch.setattr(identity, "IDENTITY_PATH", data / "identity.json")
    monkeypatch.setattr("orion_core.restart_note.NOTE_PATH", data / "restart_note.json")
    monkeypatch.setattr("orion_core.restart_note.EXIT_PATH", data / "last_exit.json")
    monkeypatch.setattr("orion_core.restart_note.SAID_PATH", data / "greeting_memory.json")
    monkeypatch.setattr("orion_core.restart_note.MARK_PATH", data / "last_mark.json")
    monkeypatch.setattr("orion_core.restart_note.EVIDENCE_PATH", data / "no_older_orion.json")
    monkeypatch.setattr("orion_core.phone_control.SWITCH_PATH", data / "phone_control.json")
    monkeypatch.setattr("orion_core.phone_control.ACTIVITY_PATH", data / "phone_activity.jsonl")
    monkeypatch.setattr("orion_core.providers.STICKY_MODELS_PATH",
                        data / "diagnostics" / "sticky_models.json")
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
    # Deck pins/recent pages and the appearance/motion preference: any deck or
    # preferences test that navigates or saves would otherwise write them.
    import orion_core.appearance as appearance
    import orion_core.gui.deck_favourites as deck_favourites
    monkeypatch.setattr(deck_favourites, "DECK_PREFERENCES_PATH",
                        data / "deck_preferences.json")
    monkeypatch.setattr(appearance, "APPEARANCE_PATH", data / "appearance.json")
    # Which brain answers (cloud / hybrid / local): never the owner's own
    # choice, and never an environment override left behind by a shell.
    # Per test (the reminders file's own tmp folder), so a test that saves a
    # mode cannot leave the next one in it.
    import orion_core.brain_mode as brain_mode
    monkeypatch.setattr(brain_mode, "SETTINGS_PATH", reminders_path.parent / "brain_mode.json")
    monkeypatch.delenv("ORION_BRAIN_MODE", raising=False)
    # Which voice (cloud Gemini voice, Kokoro voice and speed): never the
    # owner's pick, and never an environment override left in a shell.
    from orion_core import voice_settings
    monkeypatch.setattr(voice_settings, "SETTINGS_PATH", reminders_path.parent / "voice_settings.json")
    for name in voice_settings.ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    from orion_core import voice_variety
    monkeypatch.setattr(voice_variety, "STATE_PATH", data / "voice_variety.json")
    monkeypatch.setattr(voice_variety, "_VARIETY", None)
    # The learning brain and the resolver's evidence. observe_async is stubbed
    # rather than redirected: it learns on a background thread that can outlive
    # the test — and so this redirect — and would then write the real config.
    import orion_core.intent_brain as intent_brain
    import orion_core.resolver_shadow as resolver_shadow
    import orion_core.reflex_learning as reflex_learning
    import orion_core.vision_rules as vision_rules
    monkeypatch.setattr(intent_brain, "BRAIN_DIR", data / "brain")
    monkeypatch.setattr(intent_brain, "_BRAIN", None)
    monkeypatch.setattr(intent_brain, "observe_async", lambda *a, **k: None)
    monkeypatch.setattr(intent_brain, "warm", lambda: None)
    monkeypatch.setattr(resolver_shadow, "DEFAULT_PATH", data / "resolver_shadow.db")
    monkeypatch.setattr(reflex_learning, "DEFAULT_PATH", data / "reflex_learning.db")
    # Rewards and mistakes (written whenever a test worker stores the owner's words).
    import orion_core.reward_learning as reward_learning
    monkeypatch.setattr(reward_learning, "DEFAULT_PATH", data / "reward_learning.db")
    # A benchmark run is appended to its history by the diagnostics tool.
    import orion_core.self_benchmark as self_benchmark
    monkeypatch.setattr(self_benchmark, "HISTORY_PATH", data / "benchmarks" / "self_benchmark.jsonl")
    monkeypatch.setattr(vision_rules, "RULES_PATH", data / "vision_rules.json")
    # Which apps this user opens, and what "a browser" means to them.
    import orion_core.app_resolver as app_resolver
    monkeypatch.setattr(app_resolver, "USAGE_PATH", data / "app_usage.json")
    monkeypatch.setattr(app_resolver, "USAGE", app_resolver.AppUsage(data / "app_usage.json"))
    # What the user asked for that ORION could not do — written by every
    # failed dispatch, so every dispatcher test would otherwise add to it.
    import orion_core.capability_gaps as capability_gaps
    monkeypatch.setattr(capability_gaps, "GAPS_PATH", data / "capability_gaps.json")
    # An explicit path: a failure is recorded on a worker thread that can
    # outlive this redirect, and must still not reach the real file.
    monkeypatch.setattr(capability_gaps, "LEDGER",
                        capability_gaps.GapLedger(data / "capability_gaps.json"))
    # The Mission page's dock layout (a QSettings INI, written from C++, so
    # the audit hook below would never see it).
    import orion_core.gui.mission_deck as mission_deck
    monkeypatch.setattr(mission_deck, "LAYOUT_PATH", data / "mission_layout.ini")
    import orion_core.gui.chess_view as chess_view
    monkeypatch.setattr(chess_view, "BOARD_OPTIONS_PATH", data / "chess_view.ini")
    # Deleting through ORION goes to the real Windows Recycle Bin; tests get a
    # private one, or every run would leave its temp files in the owner's bin.
    import orion_core.recycle_bin as recycle_bin
    fake_bin = _FakeRecycleBin(data / "recycle_bin")
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", fake_bin.send)
    monkeypatch.setattr(recycle_bin, "find_in_recycle_bin", fake_bin.find)
    monkeypatch.setattr(recycle_bin, "restore_from_recycle_bin", fake_bin.restore)
    yield


class _FakeRecycleBin:
    """Stands in for orion_core/recycle_bin.py: same contract, private folder."""

    def __init__(self, root) -> None:
        self.root = root
        self.held: dict[str, object] = {}

    @staticmethod
    def _key(path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def send(self, path) -> bool:
        import shutil
        import uuid
        source = os.path.abspath(os.fspath(path))
        if not os.path.exists(source):
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        item = self.root / uuid.uuid4().hex
        shutil.move(source, str(item))
        self.held[self._key(source)] = item
        return True

    def find(self, path):
        item = self.held.get(self._key(path))
        return (item, item) if item is not None and item.exists() else None

    def restore(self, path) -> bool:
        import shutil
        target = os.path.abspath(os.fspath(path))
        item = self.held.get(self._key(target))
        if item is None or os.path.exists(target) or not item.exists():
            return False
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(str(item), target)
        del self.held[self._key(target)]
        return True


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
