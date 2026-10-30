"""
Tests for the destructive-action safety layer (Section 10).

Covers intent separation, ambiguity, token-bound confirmation, expiry, replay,
unrelated ("yes") replies, cancellation, the safe-by-default power primitives,
and the dispatcher gating that never executes an OS shutdown on model text.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import constants, path_policy as policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.system_guard import (
    ActionIntent,
    DecisionKind,
    SystemActionGuard,
    classify_intent,
)
from orion_core.peripherals import PeripheralController


# ── intent classification ────────────────────────────────────────────────────

def test_bare_shutdown_is_ambiguous_not_os_shutdown():
    assert classify_intent("shut down") is ActionIntent.AMBIGUOUS
    assert classify_intent("shutdown") is ActionIntent.AMBIGUOUS


def test_explicit_os_shutdown():
    assert classify_intent("shut down the computer") is ActionIntent.OS_SHUTDOWN
    assert classify_intent("power off the pc") is ActionIntent.OS_SHUTDOWN
    assert classify_intent("turn off the computer") is ActionIntent.OS_SHUTDOWN
    assert classify_intent("power down the pc") is ActionIntent.OS_SHUTDOWN


def test_self_directed_shutdown_synonyms_close_orion():
    # A bare/self-addressed power command means ORION, never the PC.
    for phrase in ("power down", "turn off now please", "shut it down"):
        assert classify_intent(phrase) is ActionIntent.AMBIGUOUS, phrase
    for phrase in ("turn yourself off", "go offline", "power yourself down",
                   "shut yourself down"):
        assert classify_intent(phrase) in (
            ActionIntent.CLOSE_APP, ActionIntent.STOP_ORION), phrase


def test_device_commands_are_not_power_commands():
    # "turn off the lights" carries an object → a device request for the model,
    # NOT a command to shut ORION (or the PC) down.
    for phrase in ("turn off the lights", "turn off the music", "turn off the tv",
                   "shut down the download", "turn off notifications",
                   "restart the download"):
        assert classify_intent(phrase) is ActionIntent.UNKNOWN, phrase


def test_the_owners_real_shutdown_requests_are_power_commands():
    """13 of 14 real requests fell through (8 Oct 2026): "Orion" is
    transcribed as "Ryan" three times in four, and the requests come wrapped
    in framing and trailing sentences."""
    for phrase in ("Ryan, shut down.", "shut down to please.",
                   "So Ryan , um , I want you to just shut down for now, please.",
                   "Okay , shut down for now. We're going to work tomorrow.",
                   "O ri on , uh shut down for now, please.",
                   "Ryan shut down again, I'm repairing you."):
        assert classify_intent(phrase.lower()) is ActionIntent.AMBIGUOUS, phrase
    for phrase in ("don't shut down", "why did you shut down?",
                   "I don't want you to shut down", "are you going to shut down?",
                   "when you shut down tonight, save my notes"):
        assert classify_intent(phrase.lower()) is ActionIntent.UNKNOWN, phrase


def test_app_and_soft_stops_are_not_destructive():
    assert classify_intent("close orion") is ActionIntent.CLOSE_APP
    assert classify_intent("stop orion") is ActionIntent.STOP_ORION
    assert classify_intent("stop speaking") is ActionIntent.STOP_SPEAKING
    assert classify_intent("stop the task") is ActionIntent.STOP_TASK
    assert classify_intent("restart orion") is ActionIntent.STOP_ORION


def test_other_os_power_intents():
    assert classify_intent("restart the computer") is ActionIntent.OS_RESTART
    assert classify_intent("reboot the machine") is ActionIntent.OS_RESTART
    assert classify_intent("log out") is ActionIntent.OS_LOGOUT
    assert classify_intent("hibernate the computer") is ActionIntent.OS_HIBERNATE
    assert classify_intent("put the computer to sleep") is ActionIntent.OS_SLEEP


# ── guard flow ────────────────────────────────────────────────────────────────

def _guard():
    ticks = {"t": 0.0}
    guard = SystemActionGuard(machine="TEST-PC", ttl_s=30.0,
                              clock=lambda: ticks["t"])
    return guard, ticks


def test_ambiguous_command_asks_for_clarification():
    guard, _ = _guard()
    d = guard.evaluate("shut down")
    assert d.kind is DecisionKind.CLARIFY
    assert d.token is None


def test_destructive_requires_confirmation_and_token_release():
    guard, _ = _guard()
    d = guard.evaluate("shut down the computer")
    assert d.kind is DecisionKind.CONFIRM
    assert d.token
    assert "TEST-PC" in d.message
    ok = guard.confirm(d.token)
    assert ok.kind is DecisionKind.EXECUTE
    assert ok.intent is ActionIntent.OS_SHUTDOWN


def test_non_destructive_executes_without_token():
    guard, _ = _guard()
    d = guard.evaluate("stop speaking")
    assert d.kind is DecisionKind.EXECUTE
    assert d.token is None


def test_expired_confirmation_is_rejected():
    guard, ticks = _guard()
    d = guard.evaluate("restart the computer")
    ticks["t"] = 31.0  # past the 30s TTL
    rejected = guard.confirm(d.token)
    assert rejected.kind is DecisionKind.REJECT


def test_replay_is_rejected_token_is_single_use():
    guard, _ = _guard()
    d = guard.evaluate("shut down the computer")
    first = guard.confirm(d.token)
    assert first.kind is DecisionKind.EXECUTE
    replay = guard.confirm(d.token)
    assert replay.kind is DecisionKind.REJECT


def test_bare_yes_does_not_confirm_when_ambiguous_pending_count():
    guard, _ = _guard()
    # No pending action → a stray "yes" confirms nothing.
    assert guard.confirm_phrase("yes").kind is DecisionKind.REJECT
    # Two pending actions → "yes" cannot be bound safely.
    guard.evaluate("shut down the computer")
    guard.evaluate("restart the computer")
    assert guard.confirm_phrase("yes").kind is DecisionKind.REJECT


def test_yes_confirms_only_single_bound_pending():
    guard, _ = _guard()
    guard.evaluate("shut down the computer")
    d = guard.confirm_phrase("yes, go ahead")
    assert d.kind is DecisionKind.EXECUTE
    assert d.intent is ActionIntent.OS_SHUTDOWN


def test_cancellation_clears_pending():
    guard, _ = _guard()
    d = guard.evaluate("shut down the computer")
    guard.cancel(d.token)
    assert guard.pending_count() == 0
    assert guard.confirm(d.token).kind is DecisionKind.REJECT


def test_unknown_token_rejected():
    guard, _ = _guard()
    assert guard.confirm("not-a-real-token").kind is DecisionKind.REJECT


# ── safe-by-default primitives ────────────────────────────────────────────────

class _Log:
    def emit(self, *_a, **_k):
        pass


class _Bus:
    log = _Log()


def test_peripheral_power_primitives_are_safe_by_default():
    # These must return a ToolResult WITHOUT executing any OS command when not
    # explicitly confirmed — including the pre-existing test that calls
    # peripheral.shutdown() with no args.
    p = PeripheralController(_Bus())
    for res in (p.shutdown(), p.restart(), p.logout(), p.sleep(), p.hibernate()):
        assert res.ok is False
        assert "confirmation" in res.text.lower()


# ── dispatcher gating (no OS shutdown from model text) ────────────────────────

class _Signal:
    def __init__(self, sink, name):
        self._sink, self._name = sink, name

    def emit(self, *payload):
        self._sink.append((self._name, payload[0] if len(payload) == 1 else payload))


class _StubBus2:
    def __init__(self):
        self.emitted = []
        self.log = _Signal(self.emitted, "log")
        self.confirm_action = _Signal(self.emitted, "confirm_action")


class _RecordingPeripherals:
    def __init__(self):
        self.executed = []

    def shutdown(self, confirm=False):
        self.executed.append(("shutdown", confirm))
        from orion_core.data import ToolResult
        return ToolResult("host shutting down", ok=True)


def _dispatcher_with_guard(bus, peripherals):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = bus
    d.peripherals = peripherals
    d.system_guard = SystemActionGuard(machine="TEST", logger=lambda *_: None)
    return d


def test_dispatcher_shutdown_arms_but_does_not_execute():
    bus = _StubBus2()
    per = _RecordingPeripherals()
    disp = _dispatcher_with_guard(bus, per)
    res = disp.peripherals_tool({"action": "shutdown"})
    assert res.ok is False                      # not executed
    assert per.executed == []                   # OS command never ran
    # A confirmation prompt was surfaced to the UI carrying a token.
    prompts = [p for (n, p) in bus.emitted if n == "confirm_action"]
    assert prompts and prompts[0]["token"]
    # Releasing with the bound token executes exactly once, with confirm=True.
    token = prompts[0]["token"]
    ok = disp.confirm_system_action(token)
    assert ok.ok is True
    assert per.executed == [("shutdown", True)]
    # Replay of the same token does nothing.
    per.executed.clear()
    disp.confirm_system_action(token)
    assert per.executed == []


def test_orions_name_is_matched_as_a_whole_word():
    """"ryan" joined the wake words (how his name is transcribed); a substring
    test would have let "Bryant" or "Ryanair" open the wake gate."""
    from orion_core.constants import WAKE_RE
    for heard in ("ryan, shut down", "orion open notepad", "oh ryan what time", "o'rion hi"):
        assert WAKE_RE.search(heard), heard
    for other in ("hello bryant", "ryanair flights", "the orionids tonight"):
        assert not WAKE_RE.search(other), other

# User-file locations and token-bound approval (13 October continuation).

@pytest.fixture
def places(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "owner"
    documents = home / "Documents"
    base = documents / "ORION-source"
    config = home / "AppData" / "ORION" / "config"
    drive = tmp_path / "data-drive"
    windows = tmp_path / "Windows"
    for root in (documents, base, config, drive, windows):
        root.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    monkeypatch.setattr(constants, "BASE_DIR", base)
    monkeypatch.setattr(constants, "CONFIG_DIR", config)
    monkeypatch.setattr(constants, "PACKAGE_DIR", base / "orion_core")
    monkeypatch.setattr(constants, "LAUNCHER_PATH", base / "orion.py")
    monkeypatch.setattr(policy, "user_roots", lambda: [("your Documents folder", documents)])
    monkeypatch.setattr(policy, "data_drives", lambda: [drive])
    monkeypatch.setattr(policy, "_system_places", lambda: [(policy._norm(windows), "inside Windows")])
    return SimpleNamespace(home=home, documents=documents, base=base, config=config,
                           drive=drive, windows=windows)


def test_ordinary_user_files_and_output_files_are_writable(places):
    for path, zone in ((places.documents / "business" / "notes.txt", policy.Zone.USER),
                       (places.drive / "work" / "notes.txt", policy.Zone.USER),
                       (places.base / "exports" / "notes.txt", policy.Zone.WORKSPACE)):
        verdict = policy.classify(path)
        assert verdict.writable and verdict.zone is zone


def test_program_settings_system_and_other_profiles_remain_closed(places):
    targets = (places.base / "orion_core" / "audio.py", places.base / "dist" / "ORION.exe",
               places.config / "memory.db", places.windows / "system.ini",
               places.home.parent / "another-owner" / "Documents" / "notes.txt",
               places.base / "README.md")
    for path in targets:
        verdict = policy.classify(path)
        assert not verdict.writable, path
        assert "Nothing was changed" in verdict.refusal()
        assert verdict.remedy


def test_parent_cannot_delete_protected_orion_descendants(places):
    verdict = policy.classify(places.documents)
    assert not verdict.writable
    assert "containing" in verdict.area
    assert not policy.classify(places.base).writable


def test_user_area_root_is_protected_even_without_orion_inside(places, monkeypatch):
    pictures = places.home / "Pictures"
    pictures.mkdir()
    monkeypatch.setattr(policy, "user_roots", lambda: [("your Pictures folder", pictures)])
    assert not policy.classify(pictures).writable
    assert policy.classify(pictures / "holiday" / "photo.jpg").writable


def test_escaping_junction_is_judged_by_its_destination(places, monkeypatch):
    requested = places.documents / "looks-safe" / "system.ini"
    actual = places.windows / "system.ini"
    original = policy.resolve
    monkeypatch.setattr(policy, "resolve", lambda path: actual if Path(path) == requested else original(path))
    verdict = policy.classify(requested)
    assert not verdict.writable and verdict.path == actual


def test_failed_resolution_does_not_use_the_safe_looking_spelling(places, monkeypatch):
    def failure(_path):
        raise OSError("junction cannot be resolved")
    monkeypatch.setattr(policy, "resolve", failure)
    verdict = policy.classify(places.documents / "looks-safe.txt")
    assert not verdict.writable and "resolve safely" in verdict.area


def test_measurement_keeps_unreadable_subtrees_explicit(tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_bytes(b"abc")
    def walk(root, onerror):
        onerror(PermissionError("unreadable subtree"))
        yield str(root), [], ["one.txt"]
    monkeypatch.setattr(policy.os, "walk", walk)
    measured = policy.measure(tmp_path)
    assert (measured.files, measured.bytes) == (1, 3)
    assert not measured.complete and "at least" in measured.describe()


def test_measurement_excludes_junction_contents(tmp_path, monkeypatch):
    child = tmp_path / "linked-system"
    child.mkdir()
    monkeypatch.setattr(Path, "is_junction", lambda path: path == child, raising=False)
    measured = policy.measure(tmp_path)
    assert measured.folders == 1 and not measured.complete


@pytest.mark.parametrize("intent_name", ["FILE_DELETE", "FILE_WRITE"])
def test_file_approval_dispatches_only_the_bound_payload_once(intent_name):
    from orion_core.data import ToolResult
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    from orion_core.system_guard import ActionIntent, SystemActionGuard
    intent = getattr(ActionIntent, intent_name)
    executed = []
    class Dispatch(DesktopDispatchMixin):
        def _run_confirmed_file_action(self, chosen, payload):
            executed.append((chosen, payload))
            return ToolResult("synthetic receipt")
    dispatch = Dispatch()
    dispatch.system_guard = SystemActionGuard(machine="TEST", logger=lambda *_: None)
    dispatch.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    dispatch.peripherals = None
    payload = {"path": "synthetic-user-file", "action": "delete"}
    decision = dispatch.system_guard.request_confirmation(intent, payload=payload)
    payload["path"] = "different-file"
    assert dispatch.confirm_system_action(decision.token).ok
    assert executed == [(intent, {"path": "synthetic-user-file", "action": "delete"})]
    assert not dispatch.confirm_system_action(decision.token).ok
    assert len(executed) == 1


def test_open_named_folder_uses_file_search_instead_of_the_application_catalogue():
    import asyncio
    from orion_core.data import ToolResult
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    calls = []
    class Dispatch(DesktopDispatchMixin):
        async def find_files(self, args):
            calls.append(args)
            return ToolResult("Two matching folders; choose one.", ok=False)
    dispatch = Dispatch()
    dispatch.desktop = SimpleNamespace(open_app=lambda *_: pytest.fail("tried to launch a folder as an app"))
    result = asyncio.run(dispatch.open_app({"app_name": "the research folder about solar energy"}))
    assert not result.ok and "choose one" in result.text
    assert calls == [{"query": "the research folder about solar energy", "type": "folder", "action": "open"}]


def _project_dispatch(tmp_path):
    from orion_core.dispatch_files import FilesDispatchMixin
    class Dispatch(FilesDispatchMixin):
        def _resolve_user_path(self, raw):
            return Path(raw)
        def _ensure_write_safe(self, path):
            path = Path(path).resolve()
            assert path.is_relative_to(tmp_path.resolve())
            return policy.Verdict(policy.Zone.USER, path, "synthetic user area")
    return Dispatch()


def test_project_scaffold_cannot_overwrite_an_existing_user_project(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    original = root / "README.md"
    original.write_bytes(b"the owner's existing project")
    result = _project_dispatch(tmp_path)._create_python_project({"path": str(root), "name": "demo"})
    assert not result.ok and "existing user folder" in result.text
    assert original.read_bytes() == b"the owner's existing project"
    assert list(root.iterdir()) == [original]


def test_project_scaffold_can_create_a_new_user_project(tmp_path):
    root = tmp_path / "new"
    result = _project_dispatch(tmp_path)._create_python_project({"path": str(root), "name": "demo"})
    assert result.ok
    assert (root / "src" / "demo" / "main.py").is_file()
    assert (root / "tests" / "test_demo.py").is_file()
    assert (root / "README.md").is_file()


def test_all_project_destinations_are_checked_before_any_write(tmp_path):
    from orion_core.security import SecurityViolation
    root = tmp_path / "new"
    dispatch = _project_dispatch(tmp_path)
    original_check = dispatch._ensure_write_safe
    def checked(path):
        if Path(path).name == "README.md":
            raise SecurityViolation("synthetic protected child")
        return original_check(path)
    dispatch._ensure_write_safe = checked
    with pytest.raises(SecurityViolation, match="protected child"):
        dispatch._create_python_project({"path": str(root), "name": "demo"})
    assert not root.exists(), "preflight changed the filesystem before a protected-child refusal"


@pytest.mark.parametrize("configured_area", ["config", "source"])
def test_resolved_private_root_cannot_be_written_via_its_user_area_target(
        places, monkeypatch, configured_area):
    """A configured junction and its direct destination have one policy."""
    alias = places.home / "AppData" / f"linked-{configured_area}"
    actual = places.documents / f"relocated-{configured_area}"
    actual.mkdir()
    if configured_area == "config":
        monkeypatch.setattr(constants, "CONFIG_DIR", alias)
        target = actual / "memory.db"
    else:
        monkeypatch.setattr(constants, "BASE_DIR", alias)
        monkeypatch.setattr(constants, "PACKAGE_DIR", alias / "orion_core")
        monkeypatch.setattr(constants, "LAUNCHER_PATH", alias / "orion.py")
        target = actual / "orion_core" / "audio.py"
    original_resolve = policy.resolve

    def resolve_junction(path):
        expanded = original_resolve(path)
        if expanded == alias or expanded.is_relative_to(alias):
            return actual / expanded.relative_to(alias)
        return expanded

    monkeypatch.setattr(policy, "resolve", resolve_junction)
    via_alias = alias / target.relative_to(actual)
    assert not policy.classify(via_alias).writable
    assert not policy.classify(target).writable
    assert policy.classify(actual.parent / "ordinary.txt").writable
