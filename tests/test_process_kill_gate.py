"""
Terminating a process needs a token a human issued — not a parameter.

What was actually wrong
-----------------------
ORION's power actions were already gated properly: SystemActionGuard issues a
random, single-use, expiring token, the GUI raises a dialog, and
``confirm_system_action(token)`` is documented as being called by that dialog
and not by the model. That design was sound and is untouched here.

``process_control`` was the exception. Its gate read:

    if not bool(args.get("confirm")):
        return ToolResult("Terminating PID 4321 requires your confirmation — "
                          "repeat the request with confirm=true and I shall "
                          "stop it.", ok=False)

``confirm`` is a tool parameter, so the MODEL fills it in. Nothing stopped it
sending ``confirm=true`` on the first call, nothing checked that a human was
ever involved, and the refusal message explained exactly how to bypass itself.
It was a convention, not a gate.

Killing a process is not undoable and takes unsaved work with it, so it belongs
behind the real token — which is what these tests pin, along with the property
that makes the token worth having: the TARGET is bound to it, so the process
the user approves on screen is necessarily the process that dies.

Offline: no Qt, no processes are harmed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.system_guard import (
    DESTRUCTIVE_INTENTS,
    ActionIntent,
    DecisionKind,
    SystemActionGuard,
)


@pytest.fixture
def guard():
    return SystemActionGuard()


# ── the intent ───────────────────────────────────────────────────────────────

def test_killing_a_process_is_classed_as_destructive():
    """It cannot be undone and takes unsaved work with it, so it belongs with
    the power actions rather than with the reversible ones."""
    assert ActionIntent.PROCESS_KILL in DESTRUCTIVE_INTENTS


def test_arming_does_not_execute(guard):
    decision = guard.request_confirmation(
        ActionIntent.PROCESS_KILL, payload={"pid": 4321})
    assert decision.kind is DecisionKind.CONFIRM
    assert decision.token


def test_the_prompt_names_what_will_happen(guard):
    decision = guard.request_confirmation(ActionIntent.PROCESS_KILL)
    assert "terminate" in decision.message.lower()


# ── the token cannot be forged ───────────────────────────────────────────────

@pytest.mark.parametrize("forged", ["", "yes", "true", "confirm", "i-approve",
                                    "0", "None"])
def test_a_model_cannot_talk_its_way_past_the_gate(guard, forged):
    """The old gate accepted the string the model chose to send. This one
    accepts a 128-bit secret the model never sees."""
    guard.request_confirmation(ActionIntent.PROCESS_KILL, payload={"pid": 1})
    assert guard.confirm(forged).kind is DecisionKind.REJECT


def test_the_real_token_releases_the_action(guard):
    armed = guard.request_confirmation(ActionIntent.PROCESS_KILL,
                                       payload={"pid": 4321})
    released = guard.confirm(armed.token)
    assert released.kind is DecisionKind.EXECUTE
    assert released.intent is ActionIntent.PROCESS_KILL


def test_a_token_is_single_use(guard):
    armed = guard.request_confirmation(ActionIntent.PROCESS_KILL, payload={"pid": 1})
    assert guard.confirm(armed.token).kind is DecisionKind.EXECUTE
    assert guard.confirm(armed.token).kind is DecisionKind.REJECT, "replayed"


def test_cancelling_leaves_nothing_to_release(guard):
    armed = guard.request_confirmation(ActionIntent.PROCESS_KILL, payload={"pid": 1})
    guard.cancel(armed.token)
    assert guard.confirm(armed.token).kind is DecisionKind.REJECT


# ── the target is bound to the token ─────────────────────────────────────────

def test_the_approved_target_comes_back_with_the_confirmation(guard):
    """THE property that makes the token worth having. The payload is bound at
    arming, so between the user reading 'Terminate PID 4321?' and the action
    running, nothing can substitute a different process."""
    armed = guard.request_confirmation(
        ActionIntent.PROCESS_KILL,
        payload={"pid": 4321, "label": "", "action": "terminate"})
    released = guard.confirm(armed.token)
    assert released.payload["pid"] == 4321
    assert released.payload["action"] == "terminate"


def test_mutating_the_payload_afterwards_cannot_change_the_target(guard):
    """The guard copies it. A caller holding the decision must not be able to
    reach back and edit what the user approved."""
    payload = {"pid": 4321}
    armed = guard.request_confirmation(ActionIntent.PROCESS_KILL, payload=payload)
    payload["pid"] = 9999
    armed.payload["pid"] = 8888
    assert guard.confirm(armed.token).payload["pid"] == 4321


def test_power_actions_still_work_without_a_payload(guard):
    """The payload is additive; the existing intents pass none."""
    armed = guard.request_confirmation(ActionIntent.OS_SHUTDOWN)
    released = guard.confirm(armed.token)
    assert released.kind is DecisionKind.EXECUTE
    assert released.payload == {}


# ── the tool no longer explains how to bypass itself ─────────────────────────

def test_the_refusal_no_longer_teaches_the_model_to_forge_approval():
    source = (Path(__file__).resolve().parents[1]
              / "orion_core" / "dispatch_desktop.py").read_text(encoding="utf-8")
    assert "repeat the request with confirm=true" not in source, (
        "the refusal message still tells the model how to bypass the gate")


def test_terminate_arms_the_guard_instead_of_reading_a_parameter():
    source = (Path(__file__).resolve().parents[1]
              / "orion_core" / "dispatch_desktop.py").read_text(encoding="utf-8")
    assert "_Intent.PROCESS_KILL" in source
    assert "confirm_action.emit" in source, "no prompt is raised for the user"


def test_the_confirmed_executor_takes_its_target_from_the_payload():
    """Not from a fresh tool call — that would reopen the hole at the last step."""
    import inspect

    from orion_core.dispatch_desktop import DesktopDispatchMixin

    body = inspect.getsource(DesktopDispatchMixin._run_confirmed_process_kill)
    assert 'payload.get("pid")' in body
    assert "args" not in body.split('"""')[-1], "the executor reads tool args"


def test_the_power_action_release_point_is_still_documented_as_human_only():
    """This was already correct before the change and must stay correct."""
    import inspect

    from orion_core.dispatch_desktop import DesktopDispatchMixin

    doc = inspect.getdoc(DesktopDispatchMixin.confirm_system_action) or ""
    assert "NOT by the model" in doc


def test_cleanup_confirmation_is_not_reachable_by_the_model():
    """confirm_cleanup takes a second_confirm flag that would escalate a
    recycle-bin deletion to a permanent one — which is only safe because the
    method is never registered as a tool."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    declared = {tool["name"] for tool in TOOL_DECLARATIONS}
    assert "confirm_cleanup" not in declared
    assert "confirm_system_action" not in declared


def test_nested_payload_cannot_change_after_approval_is_armed(guard):
    payload = {"target": {"pid": 4321}}
    armed = guard.request_confirmation(ActionIntent.PROCESS_KILL, payload=payload)
    payload["target"]["pid"] = 1
    armed.payload["target"]["pid"] = 2
    assert guard.confirm(armed.token).payload["target"]["pid"] == 4321


@pytest.fixture
def process_dispatch(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    import psutil

    dispatcher = DesktopDispatchMixin()
    dispatcher.system_guard = SystemActionGuard()
    dispatcher.peripherals = None
    prompts = []
    dispatcher.bus = SimpleNamespace(
        confirm_action=SimpleNamespace(emit=prompts.append),
        log=SimpleNamespace(emit=lambda *_: None),
    )
    process = Mock(pid=4321)
    process.name.return_value = "example.exe"
    process.create_time.return_value = 100.0
    factory = Mock(return_value=process)
    monkeypatch.setattr(psutil, "Process", factory)
    terminate = Mock(side_effect=lambda proc, results: results.append("example.exe:4321"))
    dispatcher.desktop = SimpleNamespace(_terminate_process=terminate, close_app=Mock(), open_app=Mock())
    return dispatcher, process, factory, prompts, terminate


def test_model_confirm_parameter_cannot_terminate_a_process(process_dispatch):
    dispatcher, _, _, prompts, terminate = process_dispatch
    result = dispatcher.process_governor({"action": "terminate", "pid": 4321, "confirm": True})
    assert not result.ok
    terminate.assert_not_called()
    assert len(prompts) == 1
    assert "example.exe" in prompts[0]["message"]
    assert dispatcher.confirm_system_action(prompts[0]["token"]).ok
    terminate.assert_called_once()
    assert not dispatcher.confirm_system_action(prompts[0]["token"]).ok
    assert terminate.call_count == 1


def test_recycled_pid_is_not_terminated(process_dispatch):
    dispatcher, process, _, prompts, terminate = process_dispatch
    dispatcher.process_governor({"action": "terminate", "pid": 4321})
    process.create_time.return_value = 200.0
    result = dispatcher.confirm_system_action(prompts[0]["token"])
    assert not result.ok
    terminate.assert_not_called()


@pytest.mark.parametrize("pid", [True, 0, -2, 1.5, "invalid"])
def test_invalid_pid_never_arms_or_looks_up_a_process(process_dispatch, pid):
    dispatcher, _, factory, prompts, terminate = process_dispatch
    assert not dispatcher.process_governor({"action": "terminate", "pid": pid}).ok
    assert not prompts
    factory.assert_not_called()
    terminate.assert_not_called()


def test_access_denied_reports_failure_without_restarting(process_dispatch):
    import psutil
    dispatcher, _, _, prompts, terminate = process_dispatch
    dispatcher.process_governor({"action": "restart", "pid": 4321})
    terminate.side_effect = psutil.AccessDenied(pid=4321)
    assert not dispatcher.confirm_system_action(prompts[0]["token"]).ok
    dispatcher.desktop.open_app.assert_not_called()
