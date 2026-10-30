"""Offline messaging delivery regressions. All application UI and I/O are fake."""
from contextlib import nullcontext
import asyncio
import importlib.util
import itertools
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.desktop_messaging import DesktopMessenger
from orion_core.messaging import MessagingGateway
from orion_core.execution_policy import background_scope, foreground_scope


class Node:
    ids = itertools.count()

    def __init__(self, name, kind="Text", value=None, children=()):
        self.element_info = SimpleNamespace(name=name, control_type=kind, runtime_id=[next(self.ids)])
        self.value, self.items, self.owner = value, [], None
        self.invocations, self.writes = 0, []
        self.keys = []
        self.on_invoke = lambda: None
        for child in children:
            self.add(child)

    def add(self, child):
        self.items.append(child)
        child.owner = self
        return child

    def children(self):
        return self.items[:]

    def descendants(self):
        return [item for child in self.items for item in [child, *child.descendants()]]

    def parent(self):
        return self.owner

    def is_visible(self):
        return True

    def is_enabled(self):
        return True

    def get_value(self):
        if self.value is None:
            raise ValueError("not readable")
        return self.value

    def set_edit_text(self, value):
        self.writes.append(value)
        self.value = value

    def invoke(self):
        self.invocations += 1
        self.on_invoke()

    def set_focus(self):
        pass

    def has_keyboard_focus(self):
        return True

    def type_keys(self, keys, set_foreground=True):
        self.keys.append((keys, set_foreground))
        self.on_invoke()


@pytest.fixture
def raw_app(monkeypatch):
    search = Node("Search contacts", "Edit", "")
    recipient = Node("Mum", "ListItem")
    composer = Node("Type a message", "Edit", "")
    send = Node("Send", "Button")
    pane = Node("Conversation", "Pane", children=[Node("Mum", "Header"), composer, send])
    window = Node("WhatsApp", "Window", children=[search, recipient, pane])
    control = SimpleNamespace(_guard=lambda *a, **k: None, _uia_desktop=lambda: nullcontext(object()),
                              open_application=lambda name: pytest.fail("existing window should be reused"))
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *a: None))
    sender = DesktopMessenger(control, bus)
    sender.WAIT_S = 0
    monkeypatch.setattr(sender, "_windows", lambda *a: [window])

    def submit():
        pane.add(Node("You: outgoing message", "Group", children=[Node(composer.value)]))
        composer.value = ""

    send.on_invoke = submit
    return SimpleNamespace(sender=sender, window=window, pane=pane, send=send,
                           composer=composer, search=search, recipient=recipient, control=control, bus=bus)


@pytest.fixture
def app(raw_app):
    """Native adapter cases explicitly exercise a host-authorized session."""
    with foreground_scope("native-adapter-test:user-authorized"):
        yield raw_app


def test_background_native_send_cannot_enumerate_or_mutate_app(raw_app, monkeypatch):
    monkeypatch.setattr(raw_app.control, "_uia_desktop", lambda: pytest.fail("Background must stop before UIA"))
    with background_scope("native-adapter-test"):
        result = raw_app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "requires_foreground"
    assert raw_app.search.writes == [] and raw_app.composer.writes == []
    assert raw_app.recipient.invocations == 0 and raw_app.send.invocations == 0


def test_named_contact_send_requires_and_observes_a_fresh_outgoing_message(app):
    body = "  Home soon — mañana\nSee you 🙂  "
    result = app.sender.send("whatsapp", "Mum", body)
    assert result.ok and result.receipt["status"] == "submitted"
    assert result.receipt["delivery_verified"] is False
    assert app.search.writes == ["Mum"] and app.composer.writes == [body]
    assert app.send.invocations == 1 and app.composer.value == ""


@pytest.mark.parametrize("restore_allowed", [True, False])
def test_contenteditable_focus_clearing_native_foreground_is_rechecked_before_paste(app, monkeypatch, restore_allowed):
    state = {"foreground": False, "window_focuses": 0, "clipboard": "previous", "pastes": 0}
    def window_focus():
        state["window_focuses"] += 1
        state["foreground"] = restore_allowed
    def composer_focus():
        state["foreground"] = False
    monkeypatch.setattr(app.window, "set_focus", window_focus)
    monkeypatch.setattr(app.composer, "set_focus", composer_focus)
    monkeypatch.setattr(app.composer, "set_edit_text", lambda _text: None)
    app.composer.iface_value = SimpleNamespace(SetValue=lambda _text: None)
    monkeypatch.setattr(app.sender, "_foreground_is", lambda _window: state["foreground"])
    monkeypatch.setitem(sys.modules, "pyperclip", SimpleNamespace(
        paste=lambda: state["clipboard"], copy=lambda value: state.update(clipboard=value)))
    def hotkey(keys):
        assert state["foreground"] and app.composer.has_keyboard_focus()
        assert keys == "ctrl+v"
        state["pastes"] += 1
        app.composer.value = state["clipboard"]
        return ToolResult("pasted")
    app.control.send_hotkeys = hotkey
    assert app.sender._write(app.composer, "Exact body", app.window) is restore_allowed
    assert state["window_focuses"] == 2
    assert state["pastes"] == int(restore_allowed)
    assert state["clipboard"] == "previous"
    assert app.send.invocations == 0


def test_duplicate_exact_recipients_are_rejected_before_composing(app):
    app.window.add(Node("Mum", "ListItem"))
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and "found 2" in result.text
    assert app.composer.writes == [] and app.send.invocations == 0


def test_partial_contact_name_is_never_selected(app):
    app.recipient.element_info.name = "Mum Smith"
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and "found 0" in result.text
    assert app.recipient.invocations == 0 and app.send.invocations == 0


def grid_contact(app, label):
    app.window.items.remove(app.recipient)
    cell = Node(label, "DataItem", children=[Node("row contents", "Group")])
    cell.iface_selection_item = SimpleNamespace(Select=cell.invoke)
    row = Node(label, "DataItem", children=[cell])
    grid = app.window.add(Node("Chats", "DataGrid", children=[row]))
    return grid, row, cell


def test_actual_whatsapp_nested_dataitems_resolve_one_selectable_row(app):
    _grid, row, cell = grid_contact(app, "Mum: You: Earlier message")
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert result.ok and cell.invocations == 1 and row.invocations == 0


def test_whatsapp_preview_cells_cannot_supply_a_recipient_name(app):
    _grid, _row, cell = grid_contact(app, "Different Person: Mum: Previous message")
    cell.add(Node("Mum: Old message", "DataItem"))
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and cell.invocations == 0 and app.send.invocations == 0


def test_aggregated_row_candidate_still_requires_exact_conversation_header(app):
    _grid, _row, cell = grid_contact(app, "Mum, Work: Previous message")
    app.pane.items[0].element_info.name = "Mum, Work"
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and cell.invocations == 1
    assert app.composer.writes == [] and app.send.invocations == 0


def test_self_chat_requires_both_explicit_row_and_conversation_self_markers(app):
    _grid, _row, cell = grid_contact(app, "My Account (You): Earlier message")
    cell.on_invoke = lambda: setattr(app.pane.items[0].element_info, "name", "Message yourself")
    result = app.sender.send("whatsapp", "myself", "Synthetic self message")
    assert result.ok and cell.invocations == 1 and app.send.invocations == 1


def test_self_chat_row_alone_cannot_authorize_typing_in_another_conversation(app):
    grid_contact(app, "My Account (You): Earlier message")
    result = app.sender.send("whatsapp", "myself", "Synthetic self message")
    assert not result.ok and app.composer.writes == [] and app.send.invocations == 0


def test_actual_whatsapp_self_header_and_empty_paragraph_share_sidebar_ancestor(app):
    app.window.items.clear()
    app.composer.value = "\n"
    footer = Node("", "Group", children=[Node("", "Group", children=[app.composer, app.send])])
    main = app.window.add(Node("", "Group", children=[app.search,
        Node("My Account (You) Message yourself", "Button"), footer]))
    def submit():
        main.add(Node("You: outgoing", "Group", children=[Node(app.composer.value)]))
        app.composer.value = "\n"
    app.send.on_invoke = submit
    result = app.sender.send("whatsapp", "self", "Synthetic self message")
    assert result.ok and app.search.writes == [] and app.send.invocations == 1
    assert app.composer.writes == ["Synthetic self message"]


def test_self_marker_in_contact_row_cannot_authorize_another_conversation(app):
    app.window.items.clear()
    sidebar = Node("Chats", "DataGrid", children=[Node("Own account", "DataItem", children=[
        Node("My Account (You) Message yourself", "Button")])])
    main = app.window.add(Node("", "Group", children=[app.search, sidebar, app.pane]))
    assert app.sender._conversation(app.window, "self") is None
    assert app.composer.writes == [] and app.send.invocations == 0


@pytest.mark.parametrize("row_identity,body,expected", [
    ("@My Account", "Exact body", 1),
    ("@Other Account", "Exact body", 0),
    ("@My Account", "Exact body extra", 0),
])
def test_self_chat_preview_requires_active_account_identity_and_exact_body(app, row_identity, body, expected):
    app.pane.add(Node("My Account (You) Message yourself", "Button"))
    label = f"{row_identity} (You) 21:25 {body}"
    app.pane.add(Node("Chats", "DataGrid", children=[Node(label, "DataItem", children=[Node(label, "DataItem")])]))
    assert len(app.sender._outgoing(app.pane, "Exact body")) == expected


@pytest.mark.parametrize("empty", ["\n", "\r\n", "\n\n"])
def test_contenteditable_empty_paragraph_is_read_as_empty(app, empty):
    app.composer.value = empty
    assert app.sender._value(app.composer) == ""


def test_nonempty_draft_with_terminal_newline_is_not_trimmed(app):
    app.composer.value = "  Exact existing draft\n"
    assert app.sender._value(app.composer) == "  Exact existing draft\n"


def test_selectionitem_pattern_is_used_without_falling_back_to_coordinates(app, monkeypatch):
    target = Node("selectable row", "DataItem")
    selected = []
    def unsupported():
        raise AttributeError("No InvokePattern")
    monkeypatch.setattr(target, "invoke", unsupported)
    target.iface_selection_item = SimpleNamespace(Select=lambda: selected.append(True))
    app.sender._activate(target, app.window)
    assert selected == [True]


def test_recipient_header_must_match_selected_contact(app):
    app.pane.items[0].element_info.name = "Someone Else"
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and "recipient could not be verified" in result.text
    assert app.composer.writes == [] and app.send.invocations == 0


def test_existing_user_draft_is_preserved(app):
    app.composer.value = "My unfinished message"
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and app.composer.value == "My unfinished message"
    assert app.composer.writes == [] and app.send.invocations == 0


def test_missing_send_button_leaves_a_draft_without_pressing_enter(app):
    app.send.element_info.name = "Send a gift"
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and app.composer.value == "Home"
    assert app.send.invocations == 0 and result.receipt["sent"] is False


def test_discord_enter_targets_only_verified_focused_recipient_composer(app, monkeypatch):
    app.search.element_info.name = "Where would you like to go?"
    app.send.element_info.name = "Other action"
    app.composer.element_info.name = "Message @Mum"
    app.composer.on_invoke = app.send.on_invoke
    monkeypatch.setattr(app.sender, "_foreground_is", lambda *_: True)
    result = app.sender.send("discord", "Mum", "Home")
    assert result.ok and result.receipt["status"] == "submitted"
    assert app.composer.keys == [("{ENTER}", False)] and app.send.invocations == 0


@pytest.mark.parametrize("wrong_label", [False, True])
def test_discord_enter_refuses_unfocused_or_unidentified_recipient_composer(app, monkeypatch, wrong_label):
    app.search.element_info.name = "Where would you like to go?"
    app.send.element_info.name = "Other action"
    app.composer.element_info.name = "Type a message" if wrong_label else "Message @Mum"
    monkeypatch.setattr(app.sender, "_foreground_is", lambda *_: wrong_label)
    result = app.sender.send("discord", "Mum", "Home")
    assert not result.ok and app.composer.keys == [] and app.send.invocations == 0


def test_discord_never_uses_channel_message_search_as_contact_search(app, monkeypatch):
    app.search.element_info.name = "Search"
    monkeypatch.setattr(app.sender, "_foreground_is", lambda *_: False)
    result = app.sender.send("discord", "Mum", "Home")
    assert not result.ok and app.search.writes == [] and app.send.invocations == 0


def test_discord_opens_targeted_quick_switcher_before_searching_contacts(app, monkeypatch):
    app.search.element_info.name = "Search"
    pressed = []
    def quick_switcher(keys):
        pressed.append(keys)
        app.search.element_info.name = "Where would you like to go?"
        return ToolResult("Synthetic quick switcher opened")
    app.control.send_hotkeys = quick_switcher
    monkeypatch.setattr(app.sender, "_foreground_is", lambda *_: True)
    result = app.sender.send("discord", "Mum", "Home")
    assert result.ok and pressed == ["ctrl+k"] and app.search.writes == ["Mum"]


def test_old_identical_message_is_not_evidence_of_another_send(app):
    app.pane.add(Node("You: old", "Group", children=[Node("Home")]))
    app.send.on_invoke = lambda: setattr(app.composer, "value", "")
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain"
    assert "Do not resend" in result.text and app.send.invocations == 1


def test_incoming_message_with_the_same_text_does_not_verify_send(app):
    def incoming():
        app.pane.add(Node("Mum: incoming", "Group", children=[Node("Home")]))
        app.composer.value = ""
    app.send.on_invoke = incoming
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain"
    assert app.send.invocations == 1


def test_aggregated_whatsapp_outgoing_row_verifies_exact_body_and_direction(app):
    def submit():
        app.pane.add(Node("You: Home, 12:34", "DataItem"))
        app.composer.value = ""
    app.send.on_invoke = submit
    assert app.sender.send("whatsapp", "Mum", "Home").ok


def test_aggregated_longer_body_cannot_verify_a_shorter_requested_message(app):
    def submit():
        app.pane.add(Node("You: Home later, 12:34", "DataItem"))
        app.composer.value = ""
    app.send.on_invoke = submit
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain"


def test_send_exception_is_terminal_and_cannot_invoke_another_button(app):
    def uncertain():
        raise OSError("Synthetic UI timeout")
    app.send.on_invoke = uncertain
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["sent"] is None
    assert app.send.invocations == 1


def test_recipient_is_rechecked_immediately_before_send(app, monkeypatch):
    observations = iter([(app.pane, app.composer), None])
    monkeypatch.setattr(app.sender, "_conversation", lambda *_: next(observations))
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and "changed before sending" in result.text
    assert app.send.invocations == 0


def test_control_disabled_prevents_contact_search_and_typing(app):
    app.control._guard = lambda *a, **k: ToolResult("Autonomy is disabled", ok=False)
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and app.search.writes == [] and app.send.invocations == 0


def test_partial_failed_value_write_is_not_appended_again(app, monkeypatch):
    def corrupt(value):
        app.composer.writes.append(value)
        app.composer.value = value[:2]
        raise OSError("Synthetic partial write")
    monkeypatch.setattr(app.composer, "set_edit_text", corrupt)
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and app.composer.writes == ["Home"] and app.send.invocations == 0


@pytest.mark.parametrize("app_executable", ["WhatsApp.exe", "WhatsApp.Root.exe"])
def test_process_identity_not_window_title_controls_app_selection(monkeypatch, app_executable):
    import psutil
    sender = DesktopMessenger(None, None)
    windows = [SimpleNamespace(is_visible=lambda: True, process_id=lambda pid=pid: pid) for pid in (10, 20)]
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(name=lambda: "notepad.exe" if pid == 10 else app_executable))
    assert sender._windows(SimpleNamespace(windows=lambda: windows), "whatsapp") == [windows[1]]


@pytest.mark.parametrize("limit", ["nodes", "depth"])
def test_incomplete_accessibility_traversal_cannot_prove_recipient_uniqueness(app, limit):
    if limit == "nodes":
        app.sender.MAX_NODES = 3
    else:
        app.sender.MAX_DEPTH = 1
    result = app.sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and "verify safely" in result.text
    assert app.recipient.invocations == 0 and app.send.invocations == 0


def test_gateway_prefers_verified_desktop_over_compose_link(raw_app, monkeypatch):
    gate = MessagingGateway(raw_app.bus)
    gate._desktop_sender = raw_app.sender
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    monkeypatch.setattr(gate, "_send_via_bot", lambda *_: None)
    monkeypatch.setattr("orion_core.messaging.open_url", lambda *_: pytest.fail("desktop operation must not also open a draft"))
    with foreground_scope("native-gateway-test:user-authorized"):
        assert gate.send_text("whatsapp", "Mum", "Home").ok


def test_uncertain_api_result_never_retries_via_desktop(raw_app, monkeypatch):
    gate = MessagingGateway(raw_app.bus)
    gate._desktop_sender = raw_app.sender
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    gate.bind_plugin_sender(lambda *_: ToolResult("Request timed out; acceptance unknown", ok=False,
        receipt={"kind": "messaging", "status": "uncertain", "sent": None, "attempted": True}))
    assert not gate.send_text("discord", "Mum", "Home").ok
    assert raw_app.search.writes == [] and raw_app.send.invocations == 0


@pytest.mark.parametrize("first_kind", ["chat", "email"])
@pytest.mark.parametrize("first_ok", [True, False])
def test_chat_and_email_share_inflight_guard_until_physical_worker_returns(raw_app, monkeypatch, first_kind, first_ok):
    gate = MessagingGateway(raw_app.bus)
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    entered, release = threading.Event(), threading.Event()
    calls, results = [], []
    def transport(*args):
        calls.append(args)
        entered.set()
        assert release.wait(3), "test transport was not released"
        return ToolResult("Synthetic transport result", ok=first_ok)
    monkeypatch.setattr(gate, "_send_via_bot", transport)
    monkeypatch.setattr(gate, "_send_via_smtp", transport)
    first = (lambda: gate.send_text("telegram", "Mum", "Home")) if first_kind == "chat" else (
        lambda: gate.send_email("synthetic@example.invalid", "Subject", "Body"))
    worker = threading.Thread(target=lambda: results.append(first()))
    worker.start()
    try:
        assert entered.wait(2)
        chat = gate.send_text("telegram", "Mum", "Home")
        email = gate.send_email("synthetic@example.invalid", "Subject", "Body")
        assert not chat.ok and chat.receipt["status"] == "pending"
        assert not email.ok and email.receipt["status"] == "pending"
        assert len(calls) == 1
    finally:
        release.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and results[0].ok is first_ok
    monkeypatch.setattr(gate, "_send_via_smtp", lambda *_: ToolResult("Synthetic accepted email"))
    assert gate.send_email("synthetic@example.invalid", "Subject", "Body").ok


def test_inflight_guard_releases_when_a_worker_raises(raw_app, monkeypatch):
    gate = MessagingGateway(raw_app.bus)
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    def fail(*args):
        raise RuntimeError("Synthetic programming failure")
    monkeypatch.setattr(gate, "_send_via_bot", fail)
    with pytest.raises(RuntimeError, match="Synthetic programming failure"):
        gate.send_text("telegram", "Mum", "Home")
    monkeypatch.setattr(gate, "_send_via_bot", lambda *_: ToolResult("Synthetic accepted message"))
    assert gate.send_text("telegram", "Mum", "Home").ok


def test_dry_run_never_reaches_the_desktop(raw_app):
    gate = MessagingGateway(raw_app.bus)
    gate._desktop_sender = raw_app.sender
    assert "dry-run" in gate.send_text("whatsapp", "Mum", "Home").text
    assert raw_app.search.writes == [] and raw_app.send.invocations == 0


def test_dispatch_binds_existing_control_and_accepts_to_text_aliases():
    from orion_core.dispatch_web import WebDispatchMixin
    calls = []
    control = object()
    host = SimpleNamespace(control=control, messaging=SimpleNamespace(
        bind_desktop=lambda value: calls.append(value),
        send_text=lambda *args: calls.append(args) or ToolResult("Synthetic submission")))
    assert asyncio.run(WebDispatchMixin.messaging_tool(host, {"action": "send", "platform": "whatsapp", "to": "Mum", "text": "Home"})).ok
    assert calls == [control, ("whatsapp", "Mum", "Home")]


def test_webhook_only_configuration_is_reached_and_credential_is_redacted(raw_app, monkeypatch):
    gate = MessagingGateway(raw_app.bus)
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    monkeypatch.setattr(gate, "_bot_token", lambda *_: "")
    secret = "https://example.invalid/hooks/synthetic-webhook-secret"
    monkeypatch.setattr(gate, "_discord_webhook", lambda: secret)
    invoked = []
    gate.bind_plugin_sender(lambda name, args: invoked.append((name, args)) or "Sent on Discord.")
    assert gate.send_text("discord", "team", "Home").ok
    assert invoked == [("discord_message", {"action": "send", "to": "team", "message": "Home"})]
    def fail(*args):
        raise OSError(f"Could not reach {secret}")
    gate.bind_plugin_sender(fail)
    failed = gate.send_text("discord", "team", "Home")
    assert not failed.ok and secret not in failed.text


@pytest.mark.parametrize("reply", ["I don't know this person", "Who should I message?", "What would you like me to say?",
                                  "Message not sent", "Not delivered", "Message was never sent", "Sent? No, not sent"])
def test_legacy_bot_questions_cannot_claim_success(raw_app, monkeypatch, reply):
    gate = MessagingGateway(raw_app.bus)
    monkeypatch.setattr(gate, "_bot_token", lambda *_: "synthetic-token")
    invoked = []
    gate.bind_plugin_sender(lambda name, args: invoked.append(name) or reply)
    assert not gate._send_via_bot("telegram", "Mum", "Home").ok
    assert invoked == ["telegram_message"]


def test_failed_browser_open_does_not_claim_a_draft_exists(raw_app, monkeypatch):
    gate = MessagingGateway(raw_app.bus)
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    monkeypatch.setattr(gate, "_send_via_bot", lambda *_: None)
    monkeypatch.setattr("orion_core.messaging.open_url", lambda *_: False)
    with foreground_scope("draft-launch-test:user-authorized"):
        result = gate.send_text("telegram", "Mum", "Home")
    assert not result.ok and result.receipt["sent"] is False


@pytest.mark.parametrize("target,config,allowed", [
    ("Dave", {}, False),
    ("Dave", {"contacts": {"Dave": {"user_id": "42"}}}, False),
    ("team", {"webhook_contact": "team"}, True),
    ("123", {"default_channel_id": "123"}, True),
    ("team", {"default_channel_id": "123", "contacts": {"team": {"channel_id": "123"}}}, True),
    ("other", {"default_channel_id": "123", "contacts": {"other": {"channel_id": "999"}}}, False),
    ("", {}, True),
])
def test_discord_webhook_must_be_bound_to_explicit_target(monkeypatch, target, config, allowed):
    path = Path(__file__).resolve().parents[1] / "config" / "custom_tools" / "discord_message_tool.py"
    spec = importlib.util.spec_from_file_location("synthetic_webhook_plugin", path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    posted = []
    monkeypatch.setattr(plugin, "_config", lambda: {"webhook_url": "https://example.invalid/synthetic", **config})
    monkeypatch.setattr(plugin, "_post", lambda *args, **kwargs: posted.append(args) or (True, "200"))
    reply = plugin.run(to=target, message="Home")
    assert bool(posted) is allowed
    assert reply.startswith("Sent") is allowed
