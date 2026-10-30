"""
Tests for the remote capability policy + confirmation registry (#12).

These pin the security-critical behaviour of the full-parity remote gate:
code-execution tools are refused outright, irreversible/outward actions require
an on-phone tap, read actions run immediately, unknown tools fail safe, and the
confirmation token is single-use, time-boxed and replay-proof.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio

from orion_core.data import ToolResult
from orion_core.remote_capability import (
    ALLOW_TOOLS,
    CONFIRM_TOOLS,
    FORBID_TOOLS,
    RemoteConfirmationRegistry,
    RemoteToolGate,
    Tier,
    classify,
    describe_action,
)


class _RecDispatcher:
    def __init__(self):
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, dict(args or {})))
        return ToolResult(f"ran {name}")


# ── tier membership integrity ─────────────────────────────────────────────────

def test_tiers_are_mutually_exclusive():
    assert not (ALLOW_TOOLS & FORBID_TOOLS)
    assert not (ALLOW_TOOLS & CONFIRM_TOOLS)
    assert not (CONFIRM_TOOLS & FORBID_TOOLS)


def test_code_execution_tools_are_always_forbidden():
    for tool in ("forge", "dev_workbench", "execute_plan", "self_repair"):
        assert classify(tool) is Tier.FORBID
        # No args, action, or framing can lift a FORBID.
        assert classify(tool, {"action": "read"}) is Tier.FORBID
        assert classify(tool, {"action": "list", "confirm": True}) is Tier.FORBID


# ── classification ────────────────────────────────────────────────────────────

def test_read_only_tools_run_immediately():
    for tool in ("query_intelligence", "resource_status", "web_search",
                 "find_files", "diagnostics"):
        assert classify(tool) is Tier.ALLOW


def test_mutating_tools_need_confirmation():
    for tool in ("messaging", "peripherals", "shutdown_orion",
                 "organise_files", "backup"):
        assert classify(tool) is Tier.CONFIRM


def test_read_actions_downgrade_confirm_tools_to_allow():
    # Reading is instant; the same tool's mutating action still needs a tap.
    assert classify("outlook_mail", {"action": "read_inbox"}) is Tier.ALLOW
    assert classify("outlook_mail", {"action": "send_draft"}) is Tier.CONFIRM

    assert classify("notion_workspace", {"action": "list_tasks"}) is Tier.ALLOW
    assert classify("notion_workspace", {"action": "create_task"}) is Tier.CONFIRM

    assert classify("web_automation", {"action": "read"}) is Tier.ALLOW
    assert classify("web_automation", {"action": "go_to"}) is Tier.ALLOW
    assert classify("web_automation", {"action": "submit"}) is Tier.CONFIRM
    assert classify("web_automation", {"action": "type"}) is Tier.CONFIRM

    assert classify("desktop_control", {"action": "list_windows"}) is Tier.ALLOW
    assert classify("desktop_control", {"action": "click"}) is Tier.CONFIRM

    assert classify("peripherals", {"action": "volume"}) is Tier.ALLOW
    assert classify("peripherals", {"action": "shutdown"}) is Tier.CONFIRM


def test_unknown_tool_fails_safe_to_confirm():
    assert classify("some_new_unlisted_tool") is Tier.CONFIRM
    assert classify("") is Tier.CONFIRM


# ── plugin tier registration (Mark X.14) ───────────────────────────────────────

def test_register_plugin_tier_is_respected_by_classify(monkeypatch):
    import orion_core.remote_capability as rc
    monkeypatch.setattr(rc, "_PLUGIN_TIERS", {})
    rc.register_plugin_tier("spotify_control", "allow")
    assert classify("spotify_control") is Tier.ALLOW


def test_register_plugin_tier_confirm_and_forbid(monkeypatch):
    import orion_core.remote_capability as rc
    monkeypatch.setattr(rc, "_PLUGIN_TIERS", {})
    rc.register_plugin_tier("risky_plugin", "forbid")
    assert classify("risky_plugin") is Tier.FORBID
    rc.register_plugin_tier("cautious_plugin", "confirm")
    assert classify("cautious_plugin") is Tier.CONFIRM


def test_register_plugin_tier_ignores_invalid_tier_string(monkeypatch):
    import orion_core.remote_capability as rc
    monkeypatch.setattr(rc, "_PLUGIN_TIERS", {})
    rc.register_plugin_tier("weird_plugin", "not-a-real-tier")
    # Unregistered → falls back to the normal fail-safe.
    assert classify("weird_plugin") is Tier.CONFIRM
    assert "weird_plugin" not in rc._PLUGIN_TIERS


def test_static_tables_always_win_over_a_plugin_claiming_the_same_name(monkeypatch):
    """A plugin cannot rename itself 'forge' or 'diagnostics' to inherit or
    escape the hardcoded classification of a core tool."""
    import orion_core.remote_capability as rc
    monkeypatch.setattr(rc, "_PLUGIN_TIERS", {})
    rc.register_plugin_tier("forge", "allow")
    assert classify("forge") is Tier.FORBID  # static FORBID_TOOLS still wins
    rc.register_plugin_tier("diagnostics", "forbid")
    assert classify("diagnostics") is Tier.ALLOW  # static ALLOW_TOOLS still wins


def test_describe_action_is_human_readable():
    assert "email" in describe_action("outlook_mail", {"action": "send_draft"})
    assert describe_action("shutdown_orion") == "shut ORION down"
    msg = describe_action("messaging", {"action": "send", "contact": "+44123"})
    assert "message" in msg and "+44123" in msg
    # Generic tool+action never produces awkward "run on" grammar.
    assert " on " not in describe_action("file_controller", {"action": "delete"})


# ── confirmation registry ─────────────────────────────────────────────────────

def test_confirmation_happy_path_single_use():
    reg = RemoteConfirmationRegistry(ttl_s=60)
    conf = reg.create("messaging", {"action": "send"}, "deviceA", "send a message")
    assert conf.status == "pending"
    assert "token" not in conf.public()          # token never leaves the server
    # Correct token approves exactly once.
    resolved = reg.resolve(conf.id, conf.token, approve=True, device_id="deviceA")
    assert resolved is not None and resolved.status == "approved"
    # Replay is refused.
    assert reg.resolve(conf.id, conf.token, approve=True, device_id="deviceA") is None


def test_confirmation_rejects_wrong_token():
    reg = RemoteConfirmationRegistry(ttl_s=60)
    conf = reg.create("peripherals", {"action": "shutdown"}, "d", "shut down")
    assert reg.resolve(conf.id, "not-the-token", approve=True, device_id="d") is None
    # The real token still works afterwards (a wrong guess doesn't burn it).
    assert reg.resolve(conf.id, conf.token, approve=False, device_id="d").status == "denied"


def test_confirmation_expires():
    reg = RemoteConfirmationRegistry(ttl_s=0.01)
    conf = reg.create("backup", {}, "d", "make a backup")
    time.sleep(0.03)
    assert reg.resolve(conf.id, conf.token, approve=True, device_id="d") is None
    got = reg.get(conf.id)
    assert got is None or got.status == "expired"


def test_confirmation_rejects_other_device_even_with_token():
    reg = RemoteConfirmationRegistry(ttl_s=60)
    conf = reg.create("messaging", {"action": "send"}, "owner", "send")
    assert reg.resolve(conf.id, conf.token, approve=True, device_id="other") is None
    assert reg.resolve(conf.id, conf.token, approve=True, device_id="owner") is conf


def test_pending_for_device_and_capacity_bound():
    reg = RemoteConfirmationRegistry(ttl_s=60, capacity=3)
    for _ in range(5):
        reg.create("messaging", {"action": "send"}, "phone", "send")
    # Capacity keeps memory bounded.
    assert len(reg.pending_for("phone")) <= 3
    assert reg.pending_for("other-device") == []


# ── the gate: end-to-end tier enforcement ─────────────────────────────────────

def test_gate_allow_runs_immediately():
    reg = RemoteConfirmationRegistry()
    disp = _RecDispatcher()
    gate = RemoteToolGate(disp, reg)
    result = asyncio.run(gate.dispatch("web_search", {"query": "orion"}, device_id="d1"))
    assert result.ok
    assert disp.calls == [("web_search", {"query": "orion"})]


def test_gate_forbid_is_refused_and_never_dispatched():
    reg = RemoteConfirmationRegistry()
    disp = _RecDispatcher()
    gate = RemoteToolGate(disp, reg)
    result = asyncio.run(gate.dispatch("forge", {"action": "build"}, device_id="d1"))
    assert not result.ok
    assert disp.calls == []                       # code-exec never reached the PC


def test_gate_confirm_parks_notifies_and_defers_until_approved():
    reg = RemoteConfirmationRegistry()
    disp = _RecDispatcher()
    notified = []

    async def notify(device_id, conf):
        notified.append((device_id, conf.id))

    gate = RemoteToolGate(disp, reg, notify=notify)

    async def flow():
        parked = await gate.dispatch(
            "messaging", {"action": "send", "contact": "+44"}, device_id="phone")
        # Nothing ran yet — it's waiting for the tap.
        assert parked.ok and parked.media and "confirm" in parked.media
        assert disp.calls == []
        assert notified and notified[-1][0] == "phone"
        conf = reg.get(parked.media["confirm"]["id"])
        assert conf is not None
        assert reg.resolve(conf.id, conf.token, True, device_id="phone") is conf
        # The approval finally runs the real tool.
        out = await gate.run_approved(conf)
        assert out.ok
        assert disp.calls == [("messaging", {"action": "send", "contact": "+44"})]

    asyncio.run(flow())


def test_gate_uses_current_device_when_not_passed():
    reg = RemoteConfirmationRegistry()
    gate = RemoteToolGate(_RecDispatcher(), reg)
    gate.current_device = "gated-brain-device"
    asyncio.run(gate.dispatch("messaging", {"action": "send"}))
    pending = reg.pending_for("gated-brain-device")
    assert len(pending) == 1


def test_concurrent_remote_turns_keep_their_own_device():
    reg = RemoteConfirmationRegistry()
    gate = RemoteToolGate(_RecDispatcher(), reg)

    async def turn(device_id):
        with gate.for_device(device_id):
            await asyncio.sleep(0)
            await gate.dispatch("messaging", {"action": "send"})

    async def scenario():
        await asyncio.gather(turn("phone-a"), turn("phone-b"))

    asyncio.run(scenario())
    assert len(reg.pending_for("phone-a")) == 1
    assert len(reg.pending_for("phone-b")) == 1

# ── folded in from test_telegram_approvals.py (Mark XXXX.3: approve phone requests from Telegram with buttons.) ────

import asyncio
from types import SimpleNamespace

from orion_core.remote_capability import RemoteConfirmationRegistry
from orion_core.telegram_remote import APPROVAL_NOTE, BUTTON_NOTE, TelegramRemote

CHAT = "4242"


class FakeHttp:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, payload=None, timeout=0.0):
        self.calls.append((url.rsplit("/", 1)[-1].split("?")[0], payload or {}))
        return {"ok": True, "result": []}

    def sent(self, method):
        return [payload for name, payload in self.calls if name == method]


def _remote(pending=None, decide=None, http=None):
    async def answer(text, chat_id):
        return "That would delete a file. I've sent it to your phone to approve - tap to confirm."
    return TelegramRemote(answer, token="123:abc", chats=frozenset({CHAT}), http=http or FakeHttp(),
                          clock=lambda: 1000.0, pending=pending, decide=decide)


def _message(text, chat=CHAT, kind="private"):
    return {"update_id": 1, "message": {"chat": {"id": int(chat), "type": kind},
                                        "from": {"id": int(chat)}, "date": 1000, "text": text}}


def _tap(data, chat=CHAT, sender=CHAT, kind="private"):
    return {"update_id": 2, "callback_query": {
        "id": "tap1", "data": data, "from": {"id": int(sender)},
        "message": {"message_id": 77, "chat": {"id": int(chat), "type": kind}}}}


def test_an_approval_comes_back_with_buttons():
    http = FakeHttp()
    waiting = [{"id": "abcd1234", "token": "tok_tok_tok_tok", "summary": "delete notes.txt"}]
    remote = _remote(pending=lambda chat: waiting, decide=None, http=http)
    remote._decide = lambda *a: None                       # buttons on
    reply = asyncio.run(remote.handle(_message("delete notes.txt")))
    assert BUTTON_NOTE in reply
    offers = [p for p in http.sent("sendMessage") if "reply_markup" in p]
    assert len(offers) == 1
    keys = offers[0]["reply_markup"]["inline_keyboard"][0]
    assert [k["callback_data"] for k in keys] == ["ok:abcd1234:tok_tok_tok_tok",
                                                  "no:abcd1234:tok_tok_tok_tok"]
    asyncio.run(remote.offer_approvals(CHAT))              # never offered twice
    assert len([p for p in http.sent("sendMessage") if "reply_markup" in p]) == 1


def test_without_buttons_the_old_note_stands():
    reply = asyncio.run(_remote().handle(_message("delete notes.txt")))
    assert APPROVAL_NOTE in reply


def test_a_tap_from_the_allowed_chat_decides_and_clears_the_buttons():
    http = FakeHttp()
    decided = []

    async def decide(chat, conf_id, token, approve):
        decided.append((chat, conf_id, token, approve))
        return "Deleted notes.txt."
    remote = _remote(pending=lambda chat: [], decide=decide, http=http)
    outcome = asyncio.run(remote.handle(_tap("ok:abcd1234:tok_tok_tok_tok")))
    assert outcome == "Deleted notes.txt."
    assert decided == [(CHAT, "abcd1234", "tok_tok_tok_tok", True)]
    assert http.sent("editMessageReplyMarkup")[0]["reply_markup"] == {"inline_keyboard": []}
    assert http.sent("answerCallbackQuery")


def test_taps_from_anyone_else_are_ignored():
    decided = []

    async def decide(*args):
        decided.append(args)
        return "ran"
    remote = _remote(pending=lambda chat: [], decide=decide)
    assert asyncio.run(remote.handle(_tap("ok:abcd1234:tok_tok_tok_tok", chat="999", sender="999"))) is None
    assert asyncio.run(remote.handle(_tap("ok:abcd1234:tok_tok_tok_tok", sender="555"))) is None
    assert asyncio.run(remote.handle(_tap("ok:abcd1234:tok_tok_tok_tok", kind="group"))) is None
    assert asyncio.run(remote.handle(_tap("rm -rf:x:y"))) is None
    assert decided == []


def _gateway(registry, ran):
    from orion_core.remote import RemoteGateway

    async def run_approved(conf):
        ran.append(conf.tool)
        return SimpleNamespace(text="Done it.", ok=True)
    gateway = RemoteGateway.__new__(RemoteGateway)
    gateway.confirmations = registry
    gateway.tool_gate = SimpleNamespace(run_approved=run_approved)
    gateway.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda line: None))
    return gateway


def test_the_gateway_only_lets_the_asking_chat_approve():
    registry = RemoteConfirmationRegistry()
    conf = registry.create("file_delete", {"path": "x"}, f"telegram:{CHAT}", "delete x")
    ran: list[str] = []
    gateway = _gateway(registry, ran)
    assert gateway.pending_for_channel("") == []
    assert [c["id"] for c in gateway.pending_for_channel(f"telegram:{CHAT}")] == [conf.id]
    other = asyncio.run(gateway.decide_for_channel("telegram:999", conf.id, conf.token, True))
    assert "expired" in other and ran == []
    wrong_token = asyncio.run(gateway.decide_for_channel(f"telegram:{CHAT}", conf.id, "nope", True))
    assert "expired" in wrong_token and ran == []
    done = asyncio.run(gateway.decide_for_channel(f"telegram:{CHAT}", conf.id, conf.token, True))
    assert done == "Done it." and ran == ["file_delete"]
    again = asyncio.run(gateway.decide_for_channel(f"telegram:{CHAT}", conf.id, conf.token, True))
    assert "expired" in again and ran == ["file_delete"]


def test_declining_runs_nothing():
    registry = RemoteConfirmationRegistry()
    conf = registry.create("file_delete", {"path": "x"}, f"telegram:{CHAT}", "delete x")
    ran: list[str] = []
    out = asyncio.run(_gateway(registry, ran).decide_for_channel(
        f"telegram:{CHAT}", conf.id, conf.token, False))
    assert out.startswith("Declined") and ran == []
