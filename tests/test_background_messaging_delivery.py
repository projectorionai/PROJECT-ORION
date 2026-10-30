"""Background messaging contracts against the applied source; no live I/O."""
from contextlib import nullcontext
import asyncio
import importlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules():
    policy = importlib.import_module("orion_core.execution_policy")
    browser = importlib.import_module("orion_core.browser_messaging")
    desktop = importlib.import_module("orion_core.desktop_messaging")
    gateway = importlib.import_module("orion_core.messaging")
    dispatch = importlib.import_module("orion_core.dispatch_web")
    return SimpleNamespace(policy=policy, browser=browser, desktop=desktop, gateway=gateway, dispatch=dispatch)


def forbidden(*args, **kwargs):
    pytest.fail("Foreground/clipboard/input must not be used")


class Nodes:
    def __init__(self, nodes=()):
        self.nodes = list(nodes)
    def count(self):
        return len(self.nodes)
    def nth(self, index):
        return Nodes([self.nodes[index]])
    def get_attribute(self, name):
        return self.nodes[0].attrs.get(name)
    def inner_text(self):
        return self.nodes[0].text
    def fill(self, text):
        self.nodes[0].text = text
        self.nodes[0].fills.append(text)
    def click(self):
        self.nodes[0].clicked += 1
        self.nodes[0].action()
    def press(self, key):
        self.nodes[0].keys.append(key)
        self.nodes[0].action()
    def is_enabled(self):
        return True
    def locator(self, selector):
        return Nodes([child for node in self.nodes for child in node.children.get(selector, [])])
    def get_by_role(self, *args, **kwargs):
        return self.locator("send")
    def wait_for(self, **kwargs):
        if not self.nodes:
            raise TimeoutError("Synthetic login wall")


class Element:
    def __init__(self, text="", attrs=None, children=None, action=lambda: None):
        self.text, self.attrs, self.children, self.action = text, attrs or {}, children or {}, action
        self.clicked, self.fills = 0, []
        self.keys = []


class Page:
    def __init__(self):
        self.contact = "Mum"
        self.header_title = Element("", {"title": "Mum"})
        self.header = Element("Mum", children={"[title]": [self.header_title]})
        self.composer, self.search = Element(), Element()
        self.outgoing = []
        self.rows = [Element(children={"[title]": [Element(attrs={"title": "Mum"})]})]
        self.send = Element(action=self.submit)
        self.footer = Element(children={"send": [self.send]})
        self.logged_in = True
        self.new_chat = Element()
        self.self_rows = []
    def submit(self):
        body = self.composer.text
        self.outgoing.append(Element(attrs={"data-id": "new-message"}, children={".selectable-text.copyable-text": [Element(body)]}))
        self.composer.text = ""
    def is_closed(self):
        return False
    def locator(self, selector):
        return Nodes({"#pane-side": [Element()] if self.logged_in else [], "#main header": [self.header],
            '#side [contenteditable="true"][role="textbox"]': [self.search],
            '#pane-side [role="row"]': self.rows,
            '#main footer [contenteditable="true"][role="textbox"]': [self.composer],
            "#main footer": [self.footer], "#main .message-out": self.outgoing}.get(selector, []))
    def get_by_role(self, *args, **kwargs):
        return Nodes([self.new_chat])
    def wait_for_timeout(self, ms):
        pass


@pytest.fixture
def browser(modules, tmp_path, monkeypatch):
    service = modules.browser.OwnedBrowserMessenger(SimpleNamespace(), tmp_path / "owned")
    page = Page()
    monkeypatch.setattr(service, "_ensure_page", lambda **kwargs: page)
    yield SimpleNamespace(service=service, page=page)
    service.close()


def test_owned_browser_exact_body_and_recipient_submit_once_without_os_input(browser):
    body = "  exact body\nUnicode 🙂  "
    result = browser.service.send("whatsapp", "Mum", body)
    assert result.ok and result.receipt["status"] == "submitted"
    assert result.receipt["desktop_input_used"] is False
    assert browser.page.composer.fills == [body] and browser.page.send.clicked == 1


def test_owned_browser_existing_draft_is_preserved(browser):
    browser.page.composer.text = "Existing user draft"
    result = browser.service.send("whatsapp", "Mum", "New body")
    assert not result.ok and result.receipt["status"] == "draft_conflict"
    assert browser.page.composer.text == "Existing user draft" and browser.page.send.clicked == 0


@pytest.mark.parametrize("already_open", [False, True])
def test_owned_browser_duplicate_recipients_refused_before_draft(browser, already_open):
    browser.page.header_title.attrs["title"] = "Mum" if already_open else "Other"
    browser.page.rows = [Element(children={"[title]": [Element(attrs={"title": "Mum"})]}) for _ in range(2)]
    result = browser.service.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "recipient_ambiguous"
    assert browser.page.composer.fills == [] and browser.page.send.clicked == 0


def test_owned_browser_login_required_never_opens_setup_automatically(browser):
    browser.page.logged_in = False
    result = browser.service.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "needs_login"
    assert browser.page.send.clicked == 0


def test_owned_browser_readiness_is_explicit_and_no_send(browser):
    assert browser.service.readiness().receipt["status"] == "ready"
    assert browser.page.composer.fills == [] and browser.page.send.clicked == 0


def test_owned_browser_self_requires_independent_account_marker(browser):
    browser.page.header.text = "Account (You) Message yourself"
    assert browser.service._header_matches(browser.page, "self")
    browser.page.header.text = "Someone named You"
    assert not browser.service._header_matches(browser.page, "self")


def test_gateway_and_browser_construction_are_lazy_and_do_not_import_playwright(modules, monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("playwright"):
            pytest.fail("Construction must not import Playwright")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    before = {thread.ident for thread in threading.enumerate()}
    gate = modules.gateway.MessagingGateway(SimpleNamespace())
    assert gate._browser_sender is None
    service = modules.browser.OwnedBrowserMessenger(SimpleNamespace(), tmp_path / "owned")
    assert service._worker is None and service._context is None and service._playwright is None
    assert {thread.ident for thread in threading.enumerate()} == before
    assert service.close().ok


def test_metadata_status_never_starts_browser_or_owner_thread(modules, tmp_path, monkeypatch):
    import orion_core.browser as browser_paths
    monkeypatch.setattr(browser_paths, "edge_path", lambda: "Synthetic Edge")
    monkeypatch.setattr(modules.browser.importlib.util, "find_spec", lambda name: object())
    service = modules.browser.OwnedBrowserMessenger(SimpleNamespace(), tmp_path / "owned")
    result = service.status()
    assert result.receipt["status"] == "needs_setup" and service._worker is None
    assert service._context is None and not (tmp_path / "owned").exists()
    assert service.close().ok


def test_owned_browser_launch_uses_only_distinct_profile_and_installed_headless_edge(modules, monkeypatch, tmp_path):
    launches = []
    class NewPage:
        def set_default_timeout(self, value):
            pass
        def goto(self, *args, **kwargs):
            pass
    context = SimpleNamespace(new_page=lambda: NewPage())
    def launch(**kwargs):
        launches.append(kwargs)
        return context
    playwright = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=lambda: SimpleNamespace(start=lambda: playwright)))
    service = modules.browser.OwnedBrowserMessenger(SimpleNamespace(), tmp_path / "orion_owned")
    service._ensure_page()
    assert launches[0]["headless"] is True and launches[0]["channel"] == "msedge"
    assert launches[0]["user_data_dir"] == str(tmp_path / "orion_owned" / "whatsapp")
    assert launches[0]["permissions"] == []
    assert {"--mute-audio", "--disable-audio-output", "--disable-audio-input", "--disable-notifications",
            "--autoplay-policy=user-gesture-required"}.issubset(launches[0]["args"])
    assert service._worker is None  # This direct unit call did not create a worker.


def test_owned_browser_recovers_manually_closed_setup_before_headless_check(modules, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=forbidden))
    class TargetClosedError(Exception):
        pass
    closed = []
    class ClosedSetup:
        def close(self):
            closed.append(True)
            raise TargetClosedError()
    page = Page()
    page.set_default_timeout = lambda *_: None
    page.goto = lambda *_args, **_kwargs: None
    launches = []
    def launch(**kwargs):
        launches.append(kwargs)
        return SimpleNamespace(new_page=lambda: page)
    service = modules.browser.OwnedBrowserMessenger(SimpleNamespace(), tmp_path / "owned")
    service._context, service._page = ClosedSetup(), SimpleNamespace(is_closed=lambda: True)
    service._headless = False
    service._playwright = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
    selected, blocked = service._ready()
    assert selected is page and blocked is None and closed == [True]
    assert len(launches) == 1 and launches[0]["headless"] is True


@pytest.mark.parametrize("closed_error", [True, False])
def test_owned_browser_preflight_recovery_is_closed_target_only_and_bounded(modules, tmp_path, monkeypatch, closed_error):
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=forbidden))
    class TargetClosedError(Exception):
        pass
    error = TargetClosedError if closed_error else RuntimeError
    closes, launches = [], []
    class FailedContext:
        def new_page(self):
            raise error("Synthetic preflight failure")
        def close(self):
            closes.append(True)
    def launch(**kwargs):
        launches.append(kwargs)
        return FailedContext()
    service = modules.browser.OwnedBrowserMessenger(SimpleNamespace(), tmp_path / "owned")
    service._context = FailedContext()
    service._playwright = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
    with pytest.raises(error):
        service._ensure_page()
    assert len(launches) == int(closed_error) and len(closes) == int(closed_error)


def test_owned_browser_self_picker_never_searches_literal_self_or_guesses_you(browser, monkeypatch):
    page = browser.page
    own = Element("Own Account (You)\nMessage yourself")
    other = Element("You\nAnother person's name")
    def selected():
        page.header.text = "Own Account (You) Message yourself"
    own.action = selected
    original = page.locator
    monkeypatch.setattr(page, "locator", lambda selector: Nodes([own, other]) if selector == '#side [role="row"], #side [role="listitem"]' else original(selector))
    assert browser.service._select_self(page)
    assert own.clicked == 1 and other.clicked == 0 and page.search.fills == []


def test_owned_browser_shutdown_rejects_new_operations_and_closes_once(browser):
    assert browser.service.readiness().ok
    assert browser.service.close().ok
    assert not browser.service.send("whatsapp", "Mum", "Home").ok
    assert browser.service.close().ok


def test_owned_browser_outgoing_send_exception_is_uncertain_and_never_repeated(browser):
    def timeout():
        raise TimeoutError("Synthetic send accepted but response lost")
    browser.page.send.action = timeout
    result = browser.service.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain"
    assert browser.page.send.clicked == 1


def test_owned_browser_old_identical_outgoing_message_is_not_fresh_evidence(browser, modules, monkeypatch):
    browser.page.submit()
    browser.page.outgoing[0].children[".selectable-text.copyable-text"][0].text = "Home"
    browser.page.send.action = lambda: setattr(browser.page.composer, "text", "")
    ticks = iter(range(0, 1000))
    monkeypatch.setattr(modules.browser.time, "monotonic", lambda: next(ticks))
    result = browser.service.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain" and browser.page.send.clicked == 1


@pytest.mark.parametrize("error,state", [(ModuleNotFoundError("playwright"), "unavailable"),
    (RuntimeError("distribution 'msedge' is not found"), "unavailable"),
    (RuntimeError("SingletonLock: profile in use"), "profile_busy")])
def test_owned_browser_environment_failures_are_actionable_before_submit(browser, monkeypatch, error, state):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(browser.service, "_ensure_page", fail)
    result = browser.service.send("whatsapp", "Mum", "Home")
    assert result.receipt["status"] == state and result.receipt["attempted"] is False
    assert browser.page.send.clicked == 0


def test_owned_browser_timeout_retains_physical_worker_ownership(browser, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow():
        entered.set()
        release.wait(3)
        return ToolResult("Synthetic completed")
    browser.service.WORKER_TIMEOUT_S = .01
    result = browser.service._call(slow)
    assert entered.wait(1) and result.receipt["status"] == "timed_out_before_submit"
    assert browser.service.send("whatsapp", "Mum", "Home").receipt["status"] == "pending"
    release.set()
    browser.service.WORKER_TIMEOUT_S = 3


def test_owned_browser_timeout_before_commit_cannot_submit_later(browser):
    entered, release = threading.Event(), threading.Event()
    commit = []
    def delayed_commit():
        entered.set()
        assert release.wait(3)
        commit.append(browser.service._begin_submit())
        return ToolResult("Synthetic stopped")
    browser.service.WORKER_TIMEOUT_S = .01
    result = browser.service._call(delayed_commit)
    assert entered.wait(1) and result.receipt["sent"] is False
    release.set()
    browser.service.WORKER_TIMEOUT_S = 3
    browser.service.close()
    assert commit == [False]


def test_owned_browser_timeout_after_commit_remains_uncertain_and_busy(browser):
    release = threading.Event()
    def submitted():
        assert browser.service._begin_submit()
        assert release.wait(3)
        return ToolResult("Synthetic completion")
    browser.service.WORKER_TIMEOUT_S = .01
    result = browser.service._call(submitted)
    assert result.receipt["status"] == "uncertain" and result.receipt["sent"] is None
    assert browser.service.send("whatsapp", "Mum", "Home").receipt["status"] == "pending"
    release.set()
    browser.service.WORKER_TIMEOUT_S = 3


def test_owned_browser_setup_requires_foreground_authorization_context(browser, modules):
    with modules.policy.background_scope("test"):
        result = browser.service.connect()
    assert not result.ok and browser.page.send.clicked == 0


def test_gateway_lock_outlives_browser_timeout_across_chat_and_email(browser, modules, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    service = browser.service
    service.WORKER_TIMEOUT_S = .02
    def blocked(*_args):
        entered.set()
        release.wait(2)
        return service._result("cancelled_before_submit", "Synthetic operation finished")
    original_finish = service._finish_operation
    def finish():
        original_finish()
        finished.set()
    monkeypatch.setattr(service, "_send", blocked)
    monkeypatch.setattr(service, "_finish_operation", finish)
    gate = modules.gateway.MessagingGateway(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    gate._browser_sender = service
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    monkeypatch.setattr(gate, "_send_via_smtp", lambda *_: ToolResult("Synthetic email accepted"))
    try:
        result = gate.send_text("whatsapp", "Mum", "Exact")
        assert entered.wait(1) and result.receipt["status"] == "timed_out_before_submit"
        assert gate.send_text("discord", "123", "Exact").receipt["status"] == "pending"
        assert gate.send_email("self@example.invalid", "Exact", "Exact").receipt["status"] == "pending"
    finally:
        release.set()
    assert finished.wait(1)
    assert gate.send_email("self@example.invalid", "Exact", "Exact").ok


def test_gateway_uses_registered_plugin_outcome_and_preserves_receipt(modules, monkeypatch):
    gate = modules.gateway.MessagingGateway(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    calls = []
    def registered(name, args):
        calls.append((name, args))
        return ToolResult("Synthetic accepted", receipt={"kind": "messaging", "status": "submitted"})
    gate.bind_plugin_sender(registered)
    result = gate.send_text("discord", "Mum", "Home")
    assert result.ok and result.receipt["status"] == "submitted"
    assert result.receipt["sender_kind"] == "configured_bot_or_webhook"
    assert "Discord bot/webhook API" in result.text
    assert calls == [("discord_message", {"action": "send", "to": "Mum", "message": "Home"})]


def test_gateway_readiness_refresh_preserves_other_vault_only_account(modules, monkeypatch):
    gate = modules.gateway.MessagingGateway(SimpleNamespace())
    monkeypatch.setattr(gate, "_bot_token", lambda _: "")
    monkeypatch.setattr(gate, "_discord_webhook", lambda: "")
    calls = []
    def invoke(name, arguments):
        calls.append(name)
        return ToolResult("Accepted", receipt={"kind": "messaging", "status": "submitted", "sent": True})
    gate.bind_plugin_sender(invoke, {"discord_message": True})
    gate.bind_plugin_sender(invoke, {"telegram_message": False})
    assert gate._send_via_bot("discord", "123", "Exact").ok
    assert calls == ["discord_message"]


@pytest.mark.parametrize("outcome", [ToolResult("Ready", receipt={"status": "ready"}),
    ToolResult("Sent", receipt={"status": "uncertain", "sent": None}),
    {"ok": True, "text": "Account configured"}, "Message not sent", "Not delivered"])
def test_gateway_requires_affirmative_delivery_evidence(modules, outcome):
    gate = modules.gateway.MessagingGateway(SimpleNamespace())
    gate.bind_plugin_sender(lambda *_: outcome)
    assert not gate._send_via_bot("discord", "123", "Exact").ok


def test_gateway_only_explicit_not_configured_can_fall_through(modules, monkeypatch):
    gate = modules.gateway.MessagingGateway(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    desktop = []
    gate._desktop_sender = SimpleNamespace(PLATFORMS={"discord"}, send=lambda *args: desktop.append(args) or ToolResult("Synthetic pattern-only send"))
    gate.bind_plugin_sender(lambda *_: ToolResult("Disabled plugin", ok=False, receipt={"status": "unavailable", "attempted": False}))
    assert not gate.send_text("discord", "Mum", "Home").ok and desktop == []
    gate.bind_plugin_sender(lambda *_: ToolResult("Missing account", ok=False,
        receipt={"kind": "messaging", "status": "not_configured", "sent": False, "attempted": False}))
    assert gate.send_text("discord", "Mum", "Home").ok and len(desktop) == 1


def test_gateway_draft_fallback_refused_in_default_no_input_scope(modules, monkeypatch):
    gate = modules.gateway.MessagingGateway(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    monkeypatch.setattr(modules.gateway, "open_url", forbidden)
    gate.bind_plugin_sender(lambda *_: ToolResult("Missing account", ok=False,
        receipt={"kind": "messaging", "status": "not_configured", "sent": False, "attempted": False}))
    assert not gate.send_text("telegram", "Mum", "Home").ok


def test_native_background_valuepattern_noop_never_uses_focus_clipboard_or_keyboard(modules):
    sender = modules.desktop.DesktopMessenger(SimpleNamespace(_guard=lambda *a, **k: None), None)
    node = SimpleNamespace(element_info=SimpleNamespace(runtime_id=[1]), get_value=lambda: "",
                           iface_value=SimpleNamespace(SetValue=lambda value: None), set_focus=forbidden)
    with modules.policy.background_scope("test"):
        with pytest.raises(modules.desktop._ForegroundRequired):
            sender._write(node, "Home", SimpleNamespace(set_focus=forbidden))


def test_native_background_blocks_even_valuepattern_mutation(modules):
    sender = modules.desktop.DesktopMessenger(SimpleNamespace(_guard=lambda *a, **k: None), None)
    values = [""]
    node = SimpleNamespace(element_info=SimpleNamespace(runtime_id=[1]), get_value=lambda: values[0],
        iface_value=SimpleNamespace(SetValue=lambda value: values.__setitem__(0, value)), set_edit_text=forbidden, set_focus=forbidden)
    with modules.policy.background_scope("test"):
        with pytest.raises(modules.desktop._ForegroundRequired):
            sender._write(node, "  Exact\nBody 🙂  ", SimpleNamespace(set_focus=forbidden))
    assert values[0] == ""


def test_native_delivery_requires_explicit_host_foreground_authorization(modules):
    sender = modules.desktop.DesktopMessenger(SimpleNamespace(_guard=lambda *a, **k: None, _uia_desktop=forbidden), None)
    result = sender.send("whatsapp", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "requires_foreground"


def test_native_background_missing_activation_pattern_never_clicks_or_focuses(modules):
    sender = modules.desktop.DesktopMessenger(SimpleNamespace(_guard=lambda *a, **k: None, click=forbidden), None)
    with modules.policy.background_scope("test"):
        with pytest.raises(modules.desktop._ForegroundRequired):
            sender._activate(SimpleNamespace(invoke=forbidden, select=forbidden), SimpleNamespace(set_focus=forbidden))


def test_async_dispatch_bridge_calls_registered_invoker_on_owner_loop(modules):
    gate = modules.gateway.MessagingGateway(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    gate.delivery_suppressed = lambda: False
    calls = []
    async def scenario():
        owner = threading.get_ident()
        async def invoke(name, arguments, timeout_s=None):
            calls.append((threading.get_ident(), modules.policy.is_background(), name))
            return ToolResult("Synthetic API submission", receipt={"kind": "messaging", "status": "submitted", "sent": True})
        host = SimpleNamespace(messaging=gate, invoke_plugin=invoke)
        result = await modules.dispatch.WebDispatchMixin.messaging_tool(host,
            {"action": "send", "platform": "discord", "contact": "Mum", "message": "Home"})
        assert result.ok and calls == [(owner, True, "discord_message")]
    asyncio.run(scenario())


class DiscordPage:
    OWNER_ID = "111111111111111111"
    OTHER_ID = "222222222222222222"
    def __init__(self):
        self.account = Element("OwnerHandle", attrs={"data-user-id": self.OWNER_ID})
        self.composer = Element(attrs={"aria-label": "Message @Mum"}, action=self.submit)
        self.messages = [self.message("old-id", "Earlier message", "OtherHandle")]
        self.links = [Element("Mum")]
        self.logged_in = True
    @staticmethod
    def message(identity, text, author, author_id=None):
        if author_id is None:
            author_id = DiscordPage.OWNER_ID if author == "OwnerHandle" else DiscordPage.OTHER_ID
        return Element(attrs={"id": identity, "data-author-id": author_id}, children={"[id^=\"message-content-\"]": [Element(text)],
            "[id^=\"message-username-\"]": [Element(author)]})
    def submit(self):
        self.messages.append(self.message("new-id", self.composer.text, "OwnerHandle"))
        self.composer.text = ""
    def is_closed(self):
        return False
    def locator(self, selector):
        return Nodes({
            '[data-list-id="private-channels"]': [Element()] if self.logged_in else [],
            'section[aria-label="User area"] [class*="username"]': [self.account] if self.account else [],
            'section[aria-label="User area"][data-user-id], section[aria-label="User area"] [data-user-id]': [self.account] if self.account and self.account.attrs.get("data-user-id") else [],
            '[role="textbox"][contenteditable="true"][data-slate-editor="true"]': [self.composer],
            '[id^="chat-messages-"]': self.messages,
            '[id^="chat-messages-"] [id^="message-username-"]': [Element("OtherHandle")],
            '[data-list-id="private-channels"] a[href^="/channels/@me/"]': self.links,
        }.get(selector, []))
    def wait_for_timeout(self, _ms):
        pass


@pytest.fixture
def discord(modules, monkeypatch, tmp_path):
    service = modules.browser.OwnedDiscordMessenger(SimpleNamespace(), tmp_path / "owned")
    page = DiscordPage()
    monkeypatch.setattr(service, "_ensure_page", lambda **kwargs: page)
    yield SimpleNamespace(service=service, page=page)
    service.close()


def test_owned_discord_personal_dm_uses_exact_dom_enter_and_fresh_self_message(discord):
    body = "  exact personal DM\nUnicode 🙂  "
    result = discord.service.send("discord", "Mum", body)
    assert result.ok and result.receipt["platform"] == "discord"
    assert discord.page.composer.fills == [body] and discord.page.composer.keys == ["Enter"]
    assert result.receipt["desktop_input_used"] is False


def test_owned_discord_stable_own_id_proves_sender_despite_different_display_name(discord):
    def submit():
        row = discord.page.message("new-own-id", discord.page.composer.text,
            "Different visible display name", author_id=DiscordPage.OWNER_ID)
        row.attrs.pop("data-author-id")
        row.children['[id^="message-username-"]'][0].attrs["data-user-id"] = DiscordPage.OWNER_ID
        discord.page.messages.append(row)
        discord.page.composer.text = ""
    discord.page.composer.action = submit
    result = discord.service.send("discord", "Mum", "Home")
    assert result.ok and result.receipt["status"] == "submitted"
    assert discord.page.composer.keys == ["Enter"]


def test_owned_discord_colliding_display_name_cannot_prove_outgoing(discord, modules, monkeypatch):
    def incoming():
        discord.page.messages.append(discord.page.message("new-other-id", "Home", "OwnerHandle", author_id=DiscordPage.OTHER_ID))
        discord.page.composer.text = ""
    discord.page.composer.action = incoming
    ticks = iter(range(100))
    monkeypatch.setattr(modules.browser.time, "monotonic", lambda: next(ticks))
    result = discord.service.send("discord", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain" and result.receipt["sent"] is None
    assert discord.page.composer.fills == ["Home"] and discord.page.composer.keys == ["Enter"]
    assert discord.service._outgoing_ids(discord.page, "Home") == set()


@pytest.mark.parametrize("missing", ["account", "author"])
def test_owned_discord_missing_stable_ids_preserves_one_send_and_uncertainty(discord, modules, monkeypatch, missing):
    if missing == "account":
        discord.page.account.attrs.clear()
    def submit():
        row = discord.page.message("new-unproven-id", discord.page.composer.text, "OwnerHandle")
        if missing == "author":
            row.attrs.pop("data-author-id")
        discord.page.messages.append(row)
        discord.page.composer.text = ""
    discord.page.composer.action = submit
    ticks = iter(range(100))
    monkeypatch.setattr(modules.browser.time, "monotonic", lambda: next(ticks))
    result = discord.service.send("discord", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain" and result.receipt["sent"] is None
    assert result.receipt["attempted"] is True
    assert discord.page.composer.fills == ["Home"] and discord.page.composer.keys == ["Enter"]


def test_owned_discord_rejects_unreadable_self_account_before_draft(discord):
    discord.page.account = None
    result = discord.service.send("discord", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "unsupported_layout"
    assert discord.page.composer.fills == [] and discord.page.composer.keys == []


@pytest.mark.parametrize("already_open", [False, True])
def test_owned_discord_ambiguous_personal_dm_never_selects_arbitrary_contact(discord, already_open):
    discord.page.composer.attrs["aria-label"] = "Message @Mum" if already_open else "Message @Other"
    discord.page.links = [Element("Mum") for _ in range(2)]
    result = discord.service.send("discord", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "recipient_ambiguous"
    assert not any(link.clicked for link in discord.page.links)
    assert discord.page.composer.fills == []


def test_owned_discord_self_dm_is_explicitly_unsupported(discord):
    result = discord.service.send("discord", "self", "Home")
    assert not result.ok and result.receipt["status"] == "unsupported_recipient"
    assert discord.page.composer.fills == [] and discord.page.composer.keys == []


def test_owned_discord_incoming_identical_body_is_not_verified_as_outgoing(discord, modules, monkeypatch):
    def incoming():
        discord.page.messages.append(discord.page.message("incoming-id", "Home", "OtherHandle"))
        discord.page.composer.text = ""
    discord.page.composer.action = incoming
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr(modules.browser.time, "monotonic", lambda: next(ticks))
    result = discord.service.send("discord", "Mum", "Home")
    assert not result.ok and result.receipt["status"] == "uncertain"
    assert discord.page.composer.keys == ["Enter"]


@pytest.mark.parametrize("platform", ["discord", "telegram"])
def test_api_response_reads_are_bounded_and_oversize_never_retries(platform, monkeypatch):
    path = ROOT / "config/custom_tools" / (platform + "_message_tool.py")
    spec = importlib.util.spec_from_file_location("mark36_bounded_" + platform, path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    reads, opens = [], []
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            reads.append(size)
            return b"x" * (plugin.MAX_RESPONSE_BYTES + 1)
    def open_response(*args, **kwargs):
        opens.append(True)
        return Response()
    monkeypatch.setattr(plugin.urllib.request, "urlopen", open_response)
    result = plugin._open_dm("synthetic-id", "synthetic-token") if platform == "discord" else plugin._send("synthetic-token", "synthetic-id", "Home")
    assert not result[0] and len(opens) == 1
    assert reads == [plugin.MAX_RESPONSE_BYTES + 1]


@pytest.mark.parametrize("platform", ["discord", "telegram"])
def test_api_error_body_reads_are_bounded_and_do_not_retry(platform, monkeypatch):
    path = ROOT / "config/custom_tools" / (platform + "_message_tool.py")
    spec = importlib.util.spec_from_file_location("mark36_bounded_error_" + platform, path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    reads, opens = [], []
    class HttpFailure(Exception):
        code = 500
        def read(self, size):
            reads.append(size)
            return b"x" * (plugin.MAX_RESPONSE_BYTES + 1)
    def open_response(*args, **kwargs):
        opens.append(True)
        raise HttpFailure()
    monkeypatch.setattr(plugin.urllib.error, "HTTPError", HttpFailure)
    monkeypatch.setattr(plugin.urllib.request, "urlopen", open_response)
    result = plugin._post("https://example.invalid/synthetic", {"content": "Home"}) if platform == "discord" else plugin._send("synthetic-token", "synthetic-id", "Home")
    assert not result[0] and "oversized" in result[1].lower() and len(opens) == 1
    assert reads == [plugin.MAX_RESPONSE_BYTES + 1]


def load_jv(name, monkeypatch):
    module_name = "orion_core.jv_plugins.mark36_" + name
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "orion_core/jv_plugins" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("body", ["  Exact first\nLast 🙂  ", " \n "])
def test_chat_takeover_preserves_final_body_and_rejects_blank(monkeypatch, body):
    module = load_jv("integrations", monkeypatch)
    calls = []
    async def native(name, arguments):
        calls.append((name, arguments))
        return ToolResult("Synthetic messaging result")
    monkeypatch.setattr(module.runtime, "native", native)
    result = asyncio.run(module.chat_takeover({"action": "send", "platform": "whatsapp", "contact": "Mum", "message": body}))
    assert result.ok is bool(body.strip())
    assert calls == ([("messaging", {"action": "send", "platform": "whatsapp", "contact": "Mum", "message": body})] if body.strip() else [])


@pytest.mark.parametrize("body", ["  Exact first\nLast 🙂  ", " \n ", " " + "x" * 4095 + " "])
def test_jv_telegram_preserves_approved_body_and_checks_original_length(monkeypatch, body):
    module = load_jv("telegram", monkeypatch)
    monkeypatch.setattr(module, "_config", lambda: {"bot_token": "synthetic", "allowed_chat_ids": {"123"}})
    calls = []
    async def api(token, method, payload):
        calls.append((token, method, payload))
        return {"ok": True, "result": {"message_id": 42}}
    monkeypatch.setattr(module, "_api", api)
    result = asyncio.run(module.telegram_remote({"action": "send", "chat_id": "123", "message": body}))
    valid = bool(body.strip()) and len(body) <= 4096
    assert result.ok is valid
    assert calls == ([("synthetic", "sendMessage", {"chat_id": "123", "text": body})] if valid else [])
    if valid:
        assert result.receipt["sender_kind"] == "configured_bot" and result.receipt["delivery_verified"] is False
