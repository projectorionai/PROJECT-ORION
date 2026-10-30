"""Mark XXXVII.5 - the owner's "implement all of it" (26 Oct 2026).

* A plan the owner starts keeps his turn's permissions once Live has answered
  and left it to finish (attended_scope), within limits.
* Charon's free tier (ten lines a day a model) is learnt and remembered; the
  warm-up pauses on it and the singer uses the offline voice first.
* One messaging gateway over the services' APIs (Telegram, Discord, push,
  IFTTT), every send behind the owner's on-screen approval.
* The smart home built in: Home Assistant and Hue, routines, a Home page; it
  never unlocks a door or opens a garage.
* The forge no longer doubles "_tool" in a module name.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from orion_core import execution_policy as policy


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# ── plans keep the owner's permissions while attended ────────────────────────

def test_a_plan_the_owner_started_survives_its_live_call_being_detached():
    with policy.conversation_scope("voice") as turn:
        with policy.attended_scope("plan", seconds=60) as plan:
            assert plan is not None
            turn.detach()                       # Live answered and left it to finish
            plan.detach()                       # ... and detached the plan's own scope
            assert policy.require_foreground("application_open") is None
            assert policy.require_own_window("gui_navigation") is None
            # his pointer and keyboard (Mark XXXX.2), never browser automation
            assert policy.require_foreground("keyboard_input") is None
            assert policy.require_foreground("generic_browser_automation") is not None


def test_an_attended_plan_ends_with_its_window_or_a_cancel():
    with policy.conversation_scope("voice") as turn:
        with policy.attended_scope("plan", seconds=0) as plan:
            plan.detach()
            assert policy.require_foreground("application_open") is not None
    with policy.conversation_scope("voice") as turn:
        with policy.attended_scope("plan", seconds=60):
            turn.revoke()                       # the owner pressed Stop
            assert policy.require_foreground("application_open") is not None


def test_a_plan_outside_the_owners_turn_gains_nothing():
    with policy.background_scope("job"):
        with policy.attended_scope("plan") as plan:
            assert plan is None
            assert policy.require_foreground("application_open") is not None


def test_the_plan_executor_runs_its_steps_attended():
    import inspect

    from orion_core.plan_executor import PlanExecutor

    assert "attended_scope" in inspect.getsource(PlanExecutor.execute)


# ── Charon's free tier ───────────────────────────────────────────────────────

_FREE_TIER_429 = ("429 RESOURCE_EXHAUSTED quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' "
                  "quotaValue: '10'")


def test_the_free_tier_is_learnt_from_a_refusal_and_remembered(tmp_path):
    from orion_core.voice_charon import CharonVoice

    voice = CharonVoice(key_source=lambda: "k", cache=tmp_path)
    assert not voice.on_free_tier()
    voice._block("gemini-3.8-flash-lite-tts", RuntimeError(_FREE_TIER_429))
    assert voice.on_free_tier()
    assert json.loads((tmp_path / "quota.json").read_text(encoding="utf-8"))["free_tier"] is True
    assert CharonVoice(key_source=lambda: "k", cache=tmp_path).on_free_tier()


def test_a_paid_key_refusal_is_not_the_free_tier(tmp_path):
    from orion_core.voice_charon import CharonVoice

    voice = CharonVoice(key_source=lambda: "k", cache=tmp_path)
    voice._block("gemini-3.8-flash-tts", RuntimeError("429 quotaId: 'GenerateRequestsPerMinute'"))
    assert not voice.on_free_tier()


def test_the_warm_up_pauses_on_the_free_tier(tmp_path):
    from orion_core.voice_charon import CharonPrewarm, CharonVoice

    voice = CharonVoice(key_source=lambda: "k", cache=tmp_path, synthesise=lambda text: b"\x00\x00")
    voice._block("gemini-3.8-flash-lite-tts", RuntimeError(_FREE_TIER_429))
    warm = CharonPrewarm(voice, lines=lambda: ["Good evening.", "Welcome back."],
                         state_path=tmp_path / "prewarm.json", ready=lambda: True,
                         online=lambda: True, start_delay_s=0, idle_s=0, interval_s=0)
    assert warm.fill() == "free-tier"
    assert not voice.is_cached("Good evening.")


def test_on_the_free_tier_the_offline_voice_sings_first(monkeypatch):
    from orion_core import singing

    monkeypatch.setattr(singing, "voices_on", lambda: True)
    monkeypatch.setattr(singing, "charon_saving", lambda: True)
    assert [v.name for v in singing.voice_chain()][:2] == ["offline voice", "Charon"]
    monkeypatch.setattr(singing, "charon_saving", lambda: False)
    assert [v.name for v in singing.voice_chain()][:2] == ["Charon", "offline voice"]


# ── one messaging gateway ────────────────────────────────────────────────────

@pytest.fixture
def channels(monkeypatch, tmp_path):
    from orion_core import message_channels as mc

    files: dict[str, dict] = {}
    sent: list[tuple[str, bytes, dict]] = []
    replies: list[mc._Reply] = []
    monkeypatch.setattr(mc, "_file", lambda name: files.get(name, {}))
    monkeypatch.setattr(mc, "_vault", lambda plugin, env: "")
    for env in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DISCORD_BOT_TOKEN", "DISCORD_WEBHOOK_URL",
                "NTFY_TOPIC", "NTFY_SERVER", "NTFY_TOKEN", "GOTIFY_URL", "GOTIFY_TOKEN",
                "PUSHOVER_TOKEN", "PUSHOVER_USER", "IFTTT_KEY"):
        monkeypatch.delenv(env, raising=False)

    def post(url, data, headers):
        sent.append((url, data, headers))
        return replies.pop(0) if replies else mc._Reply(True, 200, '{"ok": true, "id": "55"}')
    monkeypatch.setattr(mc, "_post", post)
    return SimpleNamespace(mc=mc, files=files, sent=sent, replies=replies)


@pytest.mark.parametrize("said,kind", [("Telegram", "telegram"), ("my phone", "push"),
                                       ("ntfy", "push"), ("IFTTT", "ifttt"), ("carrier pigeon", "")])
def test_channels_are_known_by_what_people_call_them(said, kind):
    from orion_core.message_channels import channel

    assert channel(said) == kind


def test_nothing_is_configured_and_each_channel_says_how(channels):
    status = channels.mc.status()
    assert "@BotFather" in status and "webhook" in status and "ntfy" in status and "IFTTT" in status
    result = channels.mc.send("telegram", "mum", "hello")
    assert not result.ok and result.receipt["attempted"] is False and channels.sent == []


def test_telegram_sends_to_a_saved_contact(channels):
    channels.files["messaging.json"] = {"telegram": {"bot_token": "123:SECRET",
                                                     "contacts": {"Mum": "4242"}}}
    result = channels.mc.send("telegram", "mum", "Running late")
    url, body, _headers = channels.sent[-1]
    assert result.ok and url.endswith("/sendMessage")
    assert json.loads(body) == {"chat_id": "4242", "text": "Running late"}
    unknown = channels.mc.send("telegram", "Sam", "Hi")
    assert not unknown.ok and unknown.receipt["status"] == "unknown_recipient"


def test_a_failure_never_repeats_the_token(channels):
    channels.files["messaging.json"] = {"telegram": {"bot_token": "123:SECRET", "default_chat_id": "1"}}
    channels.replies.append(channels.mc._Reply(False, 400, '{"description": "bad 123:SECRET"}'))
    result = channels.mc.send("telegram", "", "Hi")
    assert not result.ok and "123:SECRET" not in result.text and "[redacted]" in result.text


def test_discord_posts_to_the_channel_only_when_the_channel_was_asked_for(channels):
    channels.files["messaging.json"] = {"discord": {"webhook_url": "https://discord.example/hook"}}
    assert channels.mc.send("discord", "", "Game night at eight").ok
    assert channels.sent[-1][0] == "https://discord.example/hook"
    before = len(channels.sent)
    named = channels.mc.send("discord", "Sam", "Hi Sam")
    assert not named.ok and len(channels.sent) == before          # never posted to everyone


def test_push_prefers_an_authenticated_service(channels):
    channels.files["messaging.json"] = {"ntfy": {"topic": "orion-x"},
                                        "pushover": {"token": "T", "user": "U"}}
    result = channels.mc.send("push", "", "Your build finished", title="ORION")
    assert result.ok and channels.sent[-1][0] == channels.mc.PUSHOVER_API


def test_ifttt_triggers_the_named_event(channels):
    channels.files["ifttt.json"] = {"key": "K3Y"}
    assert channels.mc.send("ifttt", "lights on!", "evening").ok
    assert "/trigger/lightson/with/key/K3Y" in channels.sent[-1][0]


class _Guard:
    def __init__(self) -> None:
        self.armed: list = []

    def request_confirmation(self, intent, origin="", payload=None):
        from orion_core.system_guard import DecisionKind, GuardDecision
        self.armed.append((intent, origin, payload))
        return GuardDecision(DecisionKind.CONFIRM, intent, "confirm", token="tok", payload=payload)

    def cancel(self, token=None):
        pass


async def test_a_channel_message_waits_for_the_owners_approval(channels):
    from orion_core.dispatch_web import WebDispatchMixin
    from orion_core.system_guard import ActionIntent

    channels.files["messaging.json"] = {"telegram": {"bot_token": "123:SECRET",
                                                     "contacts": {"mum": "4242"}}}
    prompts: list = []
    host = WebDispatchMixin()
    host.messaging = SimpleNamespace(delivery_suppressed=lambda: False)
    host.system_guard = _Guard()
    host.bus = SimpleNamespace(confirm_action=SimpleNamespace(emit=prompts.append),
                               log=SimpleNamespace(emit=lambda *_a: None))
    with policy.conversation_scope("voice"):
        result = await host.messaging_tool({"action": "send", "platform": "telegram",
                                            "contact": "mum", "message": "Running late"})
    assert result.ok and result.receipt["status"] == "awaiting_confirmation"
    assert channels.sent == []                                     # nothing sent yet
    intent, _origin, payload = host.system_guard.armed[0]
    assert intent is ActionIntent.SEND_MESSAGE and payload["to"] == "mum"
    prompt = prompts[0]
    assert prompt["intent"] == "channel_message" and prompt["background"] is True
    assert (prompt["channel"], prompt["to"], prompt["body"]) == ("telegram", "mum", "Running late")
    assert "Running late" in prompt["message"]
    # a job asks the same way - a non-modal approval, never a send
    with policy.background_scope("job"):
        queued = await host.messaging_tool({"action": "send", "platform": "telegram",
                                            "contact": "mum", "message": "x"})
    assert queued.receipt["status"] == "awaiting_confirmation" and channels.sent == []


async def test_an_installed_messaging_plugin_keeps_its_own_path(channels):
    from orion_core.dispatch_web import WebDispatchMixin

    channels.files["messaging.json"] = {"telegram": {"bot_token": "123:SECRET"}}
    host = WebDispatchMixin()
    host.messaging = SimpleNamespace(send_text=lambda *a: "plugin path", bind_desktop=lambda *_a: None,
                                     bind_browser=lambda: None, delivery_suppressed=lambda: False)
    host.plugins = SimpleNamespace(get=lambda name: object() if name == "telegram_message" else None)
    host.system_guard = _Guard()
    result = await host.messaging_tool({"action": "send", "platform": "telegram",
                                        "contact": "mum", "message": "hi"})
    assert result == "plugin path" and host.system_guard.armed == []


def test_the_approved_message_is_exactly_what_was_approved(monkeypatch):
    from orion_core import message_channels
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    from orion_core.data import ToolResult

    calls: list = []
    monkeypatch.setattr(message_channels, "send",
                        lambda *a, **k: calls.append((a, k)) or ToolResult("Sent."))
    host = DesktopDispatchMixin()
    host.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None),
                               speak_request=SimpleNamespace(emit=lambda *_a: None))
    result = host._run_confirmed_channel({"channel": "telegram", "to": "mum",
                                          "message": "Running late", "title": "", "priority": ""})
    assert result.ok and calls == [(("telegram", "mum", "Running late"),
                                    {"title": "", "priority": ""})]


# ── the smart home ───────────────────────────────────────────────────────────

@pytest.fixture
def house(monkeypatch):
    from orion_core import smart_home as sh

    calls: list = []
    monkeypatch.setattr(sh, "ha_config", lambda: {"url": "http://ha", "token": "T"})
    monkeypatch.setattr(sh, "hue_config", lambda: {"bridge": "", "username": ""})
    monkeypatch.setattr(sh, "_file", lambda name: {})
    states = [
        {"entity_id": "light.bedroom_lamp", "state": "on",
         "attributes": {"friendly_name": "Bedroom Lamp", "brightness": 128}},
        {"entity_id": "light.kitchen", "state": "off", "attributes": {"friendly_name": "Kitchen Light"}},
        {"entity_id": "switch.fan_plug", "state": "off", "attributes": {"friendly_name": "Fan Plug"}},
        {"entity_id": "lock.front_door", "state": "locked", "attributes": {"friendly_name": "Front Door"}},
        {"entity_id": "cover.garage", "state": "closed", "attributes": {"friendly_name": "Garage Door"}},
        {"entity_id": "scene.cosy", "state": "scening", "attributes": {"friendly_name": "Cosy"}},
        {"entity_id": "sensor.temp", "state": "20", "attributes": {"friendly_name": "Temperature"}},
    ]

    def ha(path, payload=None):
        if path == "/api/states":
            return states
        calls.append((path, payload))
        return []
    monkeypatch.setattr(sh, "_ha", ha)
    return SimpleNamespace(sh=sh, calls=calls)


def test_the_house_is_read_and_described(house):
    text, rows = house.sh.describe()
    assert "2 lights, 1 on (Bedroom Lamp)" in text and "1 lock" in text
    assert {r["name"] for r in rows} >= {"Bedroom Lamp", "Kitchen Light", "Front Door", "Cosy"}
    assert all(r["kind"] != "sensor" for r in rows)


def test_lights_are_found_by_the_names_people_use(house):
    rooms, _problems = house.sh.devices()
    assert [d.name for d in house.sh.resolve("all the lights", rooms)] == ["Bedroom Lamp", "Kitchen Light"]
    assert [d.name for d in house.sh.resolve("the bedroom lamp", rooms)] == ["Bedroom Lamp"]
    assert [d.name for d in house.sh.resolve("kitchen lights", rooms)] == ["Kitchen Light"]


def test_switching_and_dimming_call_home_assistant(house):
    assert house.sh.apply("off", "bedroom lamp").ok
    assert house.calls[-1] == ("/api/services/light/turn_off", {"entity_id": "light.bedroom_lamp"})
    assert house.sh.apply("brightness", "kitchen", "40%").ok
    assert house.calls[-1] == ("/api/services/light/turn_on",
                               {"entity_id": "light.kitchen", "brightness_pct": 40})
    assert house.sh.apply("on", "cosy").ok                     # a scene by name
    assert house.calls[-1] == ("/api/services/scene/turn_on", {"entity_id": "scene.cosy"})


def test_it_never_unlocks_a_door_or_opens_the_garage(house):
    unlock = house.sh.apply("off", "front door")
    opened = house.sh.apply("on", "garage door")
    assert not unlock.ok and "won't" in unlock.text and not opened.ok
    assert all("unlock" not in path and "open" not in path for path, _ in house.calls)
    assert house.sh.apply("on", "front door").ok               # locking is fine
    assert house.calls[-1][0] == "/api/services/lock/lock"


def test_goodnight_turns_every_light_off(house):
    result = house.sh.run_routine("good night")
    assert result.ok and "Goodnight routine done" in result.text
    assert {payload["entity_id"] for path, payload in house.calls
            if path == "/api/services/light/turn_off"} == {"light.bedroom_lamp", "light.kitchen"}


def test_with_nothing_connected_it_says_how_to_set_it_up(monkeypatch):
    from orion_core import smart_home as sh

    monkeypatch.setattr(sh, "ha_config", lambda: {"url": "", "token": ""})
    monkeypatch.setattr(sh, "hue_config", lambda: {"bridge": "", "username": ""})
    result = sh.apply("off", "all lights")
    assert not result.ok and "homeassistant.json" in result.text and "Hue" in result.text


@pytest.mark.parametrize("said,args", [
    ("turn off the bedroom lights", {"action": "off", "target": "bedroom lights"}),
    ("turn on the lights", {"action": "on", "target": "all lights"}),
    ("dim the lights to 30 percent", {"action": "brightness", "value": "30", "target": "all lights"}),
    ("goodnight routine", {"action": "routine", "target": "goodnight"}),
])
def test_light_phrases_reach_the_smart_home_at_once(said, args):
    from orion_core.reflex import match_reflex

    match = match_reflex(said)
    assert match is not None and match.tool == "smart_home" and match.args == args


@pytest.mark.parametrize("said,tool", [("turn off the screen", "peripherals"),
                                       ("turn on wifi", "peripherals")])
def test_other_turn_on_phrases_keep_their_tools(said, tool):
    from orion_core.reflex import match_reflex

    assert match_reflex(said).tool == tool


def test_the_home_page_shows_each_device_with_its_own_switch(_app):
    from orion_core.data import ToolResult
    from orion_core.gui.home_deck import HomeDeckView

    view = HomeDeckView(SimpleNamespace(), SimpleNamespace())
    rows = [{"name": "Bedroom Lamp", "kind": "light", "state": "on", "brightness": 50, "on": True},
            {"name": "Front Door", "kind": "lock", "state": "locked", "on": False},
            {"name": "Cosy", "kind": "scene", "state": "", "on": False}]
    view.show_house(ToolResult("Your smart home: 1 light.", evidence=[{"smart_home_devices": rows}]))
    labels = [b.text() for b in view.grid_host.findChildren(QPushButton)]
    assert labels == ["Turn Off"]                               # the lock has no switch
    assert "Your smart home" in view.status.text()
    assert any(b.text() == "Cosy" for b in view.findChildren(QPushButton))


def test_the_home_page_is_on_the_deck_and_found_by_its_words(_app):
    from orion_core.gui.unified_dashboard import UnifiedDashboard

    bus = SimpleNamespace(**{name: _Sink() for name in ("log", "dashboard_event", "gui_command")})
    deck = UnifiedDashboard.__new__(UnifiedDashboard)
    deck._pages = [("HOME", QWidget()), ("GOALS", QWidget())]
    assert deck.resolve_page_name("lights") == "HOME" and deck.resolve_page_name("home") == "HOME"
    assert any("HOME" in names for _group, names in UnifiedDashboard.SIDEBAR_GROUPS)
    assert "HOME" in UnifiedDashboard.ZONE_PAGES["AUTOMATION"]
    del bus


class _Sink:
    def emit(self, *_a):
        pass

    def connect(self, *_a, **_k):
        pass


# ── the forge ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,name", [
    ("resource_optimizer_tool", "resource_optimizer"), ("Weather Tool", "weather"),
    ("backup_tool_tool", "backup"), ("../../x", "x"), ("tool", ""), ("_tool", ""),
    ("toolbox", "toolbox"),
])
def test_the_forge_no_longer_doubles_tool_in_a_module_name(raw, name):
    from orion_core.forge import forge_tool_name

    assert forge_tool_name(raw) == name


def test_retired_plugins_stay_in_the_source_but_are_not_shipped():
    import build_standalone
    from tools.prepare_public_source import PUBLIC_CONFIG_FILES, RETIRED_PLUGINS

    shipped = {build_standalone._plugin_of(name) for name in PUBLIC_CONFIG_FILES
               if name.startswith("config/custom_tools/") and not name.endswith("__init__.py")
               and build_standalone._plugin_of(name) not in RETIRED_PLUGINS}
    assert {"telegram_message", "philips_hue", "jv_reminder"} <= RETIRED_PLUGINS
    assert not shipped & RETIRED_PLUGINS and "jv_water_reminder" in shipped
    assert build_standalone._plugin_of("config/custom_tools/push_notify.plugin.json") == "push_notify"
