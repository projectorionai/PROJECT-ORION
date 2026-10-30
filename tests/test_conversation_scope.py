"""
The owner's own turn may change ORION's own windows; nothing else changes.

26 Oct 2026, the owner's log: "show me where that is on the Command Deck" and
"cycle through your facial emotions" were both refused - "interface control
requires a foreground session". Mark XXXVI runs every tool call in a
background scope so a model call cannot take the mouse, keyboard, clipboard
or focus, and interface_control asked for the foreground, so it could never
run when asked. Now a tool call made while answering the owner runs in a
conversation scope: still a background scope for every shared effect, but
ORION's own windows (require_own_window) may change. Jobs, workflows,
plugins, schedules, phone requests and work already handed to the
background are still refused.

Fakes only: a recording bus, a fake dispatcher. No window, no ORION.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import execution_policy as policy  # noqa: E402
from orion_core.dispatch_desktop import DesktopDispatchMixin  # noqa: E402


# ── the policy ──────────────────────────────────────────────────────────────

def test_the_owners_turn_may_change_orions_own_windows():
    with policy.conversation_scope("voice") as scope:
        assert scope.origin == policy.CONVERSATION
        assert policy.require_own_window("gui_navigation") is None


def test_the_owners_turn_still_never_takes_shared_input():
    # Mark XXXX.2: his own pointer and keyboard are his to ask for; automating a
    # browser session is still never done for him.
    with policy.conversation_scope("voice"):
        assert policy.require_foreground("keyboard_input") is None
        blocked = policy.require_foreground("generic_browser_automation")
    assert blocked is not None and blocked.receipt["status"] == "requires_foreground"
    assert blocked.receipt["desktop_mutated"] is False


@pytest.mark.parametrize("origin", ["tool", "job", "workflow", "plugin", "plugin_schedule",
                                    "plugin_event", "approved_plugin", "messaging"])
def test_every_other_background_is_refused(origin):
    with policy.background_scope(origin):
        blocked = policy.require_own_window("gui_navigation")
    assert blocked is not None and blocked.receipt["status"] == "requires_foreground"


def test_a_conversation_started_inside_background_work_is_refused():
    with policy.background_scope("job", "nightly"):
        with policy.conversation_scope("voice"):
            assert policy.require_own_window("gui_navigation") is not None


def test_a_cancelled_or_detached_turn_may_no_longer_change_his_windows():
    with policy.conversation_scope("voice") as scope:
        scope.detach()
        assert scope.active and scope.detached
        assert policy.require_own_window("gui_navigation") is not None
    with policy.conversation_scope("voice") as scope:
        scope.revoke()
        assert policy.require_own_window("gui_navigation") is not None


def test_outside_any_scope_the_old_rule_holds():
    assert policy.require_own_window("gui_navigation") is None
    with policy.foreground_scope("explicit-user-button"):
        assert policy.require_own_window("gui_navigation") is None


# ── interface_control itself ────────────────────────────────────────────────

def _host():
    emitted = []
    host = SimpleNamespace(bus=SimpleNamespace(gui_command=SimpleNamespace(emit=emitted.append)),
                           page_resolver=lambda name: name.lower(),
                           page_names=lambda: ["tasks", "chess"])
    return host, emitted


def test_show_me_that_on_the_command_deck_now_opens_the_page():
    host, emitted = _host()
    with policy.conversation_scope("voice"):
        result = DesktopDispatchMixin.interface_control(host, {"action": "page", "target": "Tasks"})
    assert result.ok and emitted == [{"action": "page", "target": "tasks"}]


def test_a_dock_or_compact_request_from_the_owner_runs():
    host, emitted = _host()
    with policy.conversation_scope("typed"):
        assert DesktopDispatchMixin.interface_control(host, {"action": "show_dock"}).ok
        assert DesktopDispatchMixin.interface_control(host, {"action": "overlay"}).ok
    assert [e["action"] for e in emitted] == ["show_dock", "overlay"]


def test_a_scheduled_job_cannot_flip_orions_windows():
    host, emitted = _host()
    with policy.background_scope("job", "every-morning"):
        result = DesktopDispatchMixin.interface_control(host, {"action": "page", "target": "chess"})
    assert not result.ok and result.receipt["status"] == "requires_foreground"
    assert emitted == []


# ── the live worker puts the owner's calls in it ────────────────────────────

class _Result:
    media = None

    def response_payload(self):
        return {"ok": True, "result": "done"}


def _worker(seen):
    async def dispatch_chain(name, args, **_kw):
        scope = policy.current_scope()
        seen.append((scope.origin, scope.run_id) if scope else None)
        return _Result()

    return SimpleNamespace(dispatcher=SimpleNamespace(dispatch_chain=dispatch_chain),
                           _typed_live_turn=True, bus=SimpleNamespace(log=SimpleNamespace(emit=print)))


def test_a_live_tool_call_runs_in_the_owners_conversation():
    from orion_core.live_worker import GenAILiveWorker
    seen = []
    payload, _media = asyncio.run(
        GenAILiveWorker._dispatch_live_tool(_worker(seen), "interface_control", {"action": "page"}))
    assert payload["ok"] and seen == [(policy.CONVERSATION, "typed")]


def test_a_text_path_call_runs_in_the_conversation_but_a_phone_request_does_not():
    from orion_core import live_worker
    seen = []
    worker = _worker(seen)
    asyncio.run(live_worker.GenAILiveWorker._text_path_dispatch(worker, "interface_control", {}, "x"))

    async def remote():
        token = live_worker._TURN_ORIGIN.set("remote")
        try:
            await live_worker.GenAILiveWorker._text_path_dispatch(worker, "interface_control", {}, "x")
        finally:
            live_worker._TURN_ORIGIN.reset(token)

    asyncio.run(remote())
    assert seen == [(policy.CONVERSATION, "text"), None]


def test_a_tool_left_to_finish_in_the_background_is_detached():
    from orion_core.live_worker import _detach_task_scope

    async def flow():
        inside, release, verdict = asyncio.Event(), asyncio.Event(), []

        async def tool():
            with policy.conversation_scope("voice"):
                inside.set()
                await release.wait()
                verdict.append(policy.require_own_window("gui_navigation"))

        task = asyncio.create_task(tool())
        await inside.wait()
        _detach_task_scope(task)
        release.set()
        await task
        return verdict

    verdict = asyncio.run(flow())
    assert verdict and verdict[0] is not None
