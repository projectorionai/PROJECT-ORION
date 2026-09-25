"""
The messaging plugins — Discord and Telegram (Mark XXVI).

ORION could previously only *open* WhatsApp/Telegram in a browser for the user to
press send. These plugins actually deliver a message. They are also the first
real dogfood of the plugin system: manifest, capability declaration, tier.

Nothing here touches the network — the transport is stubbed — but every branch a
user will actually hit is exercised, especially the unconfigured one, because
"it silently did nothing" is the failure that matters.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLUGIN_DIR = ROOT / "config" / "custom_tools"


def _load(name: str):
    """Import a plugin module by path (it is not part of the package)."""
    path = PLUGIN_DIR / f"{name}_tool.py"
    spec = importlib.util.spec_from_file_location(f"_plugin_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def discord(monkeypatch):
    module = _load("discord_message")
    sent: list[tuple] = []
    monkeypatch.setattr(module, "_post",
                        lambda url, payload, token="": (sent.append((url, payload, token)), (True, "200"))[1])
    monkeypatch.setattr(module, "_open_dm", lambda user_id, token: (f"dm-{user_id}", ""))
    module._sent = sent
    return module


@pytest.fixture()
def telegram(monkeypatch):
    module = _load("telegram_message")
    sent: list[tuple] = []
    monkeypatch.setattr(module, "_send",
                        lambda token, chat_id, text: (sent.append((token, chat_id, text)), (True, ""))[1])
    module._sent = sent
    return module


def _configure(monkeypatch, module, config):
    monkeypatch.setattr(module, "_config", lambda: config)


# ── the contract (both plugins are real ORION plugins) ──────────────────────

@pytest.mark.parametrize("name", ["discord_message", "telegram_message"])
def test_the_plugin_satisfies_the_orion_contract(name):
    from orion_core.plugin_registry import validate_module
    assert validate_module(PLUGIN_DIR / f"{name}_tool.py") == []


@pytest.mark.parametrize("name", ["discord_message", "telegram_message"])
def test_the_manifest_declares_what_the_code_does(name):
    from orion_core.plugin_registry import audit_permissions
    import json
    manifest = json.loads((PLUGIN_DIR / f"{name}.plugin.json").read_text(encoding="utf-8"))
    report = audit_permissions(PLUGIN_DIR / f"{name}_tool.py", manifest["permissions"])
    assert report["undeclared"] == [], f"{name} uses undeclared capabilities"
    # sending a message is outward-facing: a remote device must confirm first
    assert manifest["tier"] == "confirm"


@pytest.mark.parametrize("name", ["discord_message", "telegram_message"])
def test_no_credential_is_baked_into_the_plugin_source(name):
    source = (PLUGIN_DIR / f"{name}_tool.py").read_text(encoding="utf-8")
    for marker in ("Bot ", "xoxb-", "https://discord.com/api/webhooks/"):
        # the literal API host is fine; a real webhook/token would contain a path
        assert f'"{marker}' not in source or marker == "Bot "
    assert "bot_token" in source and "os.getenv" in source


# ── unconfigured: say exactly what to do, never fail silently ───────────────

def test_discord_unconfigured_explains_itself(discord, monkeypatch):
    _configure(monkeypatch, discord, {})
    reply = discord.run(message="hello")
    assert "not configured" in reply.lower()
    assert "messaging.json" in reply
    assert discord._sent == []


def test_telegram_unconfigured_explains_itself(telegram, monkeypatch):
    _configure(monkeypatch, telegram, {})
    reply = telegram.run(message="hello")
    assert "not configured" in reply.lower()
    assert telegram._sent == []


def test_status_reports_configuration(discord, telegram, monkeypatch):
    _configure(monkeypatch, discord, {"bot_token": "t", "contacts": {"dave": {"user_id": "1"}}})
    assert "bot token: set" in discord.run(action="status")
    _configure(monkeypatch, telegram, {})
    assert "not configured" in telegram.run(action="status").lower()


# ── sending ─────────────────────────────────────────────────────────────────

def test_discord_dms_a_saved_person(discord, monkeypatch):
    _configure(monkeypatch, discord,
               {"bot_token": "tok", "contacts": {"dave": {"user_id": "42"}}})
    reply = discord.run(to="dave", message="on my way")
    assert "sent on discord" in reply.lower() and "dave" in reply
    url, payload, token = discord._sent[-1]
    assert "channels/dm-42/messages" in url
    assert payload["content"] == "on my way"
    assert token == "tok"


def test_discord_posts_to_a_channel_id(discord, monkeypatch):
    _configure(monkeypatch, discord, {"bot_token": "tok"})
    assert "sent" in discord.run(to="99887766", message="hi").lower()
    assert "channels/99887766/messages" in discord._sent[-1][0]


def test_discord_uses_a_webhook_when_there_is_no_bot(discord, monkeypatch):
    _configure(monkeypatch, discord, {"webhook_url": "https://example.invalid/hook"})
    assert "sent" in discord.run(message="hello team").lower()
    assert discord._sent[-1][0] == "https://example.invalid/hook"


def test_discord_truncates_over_the_platform_limit(discord, monkeypatch):
    _configure(monkeypatch, discord, {"bot_token": "tok", "default_channel_id": "5"})
    discord.run(message="x" * 5000)
    assert len(discord._sent[-1][1]["content"]) <= 2000


def test_telegram_sends_to_a_saved_contact(telegram, monkeypatch):
    _configure(monkeypatch, telegram, {"bot_token": "tok", "contacts": {"mum": "123"}})
    reply = telegram.run(to="mum", message="running late")
    assert "sent on telegram" in reply.lower()
    token, chat_id, text = telegram._sent[-1]
    assert chat_id == "123" and text == "running late"


def test_telegram_accepts_a_negative_group_chat_id(telegram, monkeypatch):
    _configure(monkeypatch, telegram, {"bot_token": "tok"})
    telegram.run(to="-1001234", message="hi")
    assert telegram._sent[-1][1] == "-1001234"


def test_telegram_truncates_over_the_platform_limit(telegram, monkeypatch):
    _configure(monkeypatch, telegram, {"bot_token": "tok", "default_chat_id": "1"})
    telegram.run(message="y" * 9000)
    assert len(telegram._sent[-1][2]) <= 4096


# ── refusals that keep the user informed ────────────────────────────────────

def test_an_unknown_contact_lists_the_known_ones(discord, monkeypatch):
    _configure(monkeypatch, discord,
               {"bot_token": "tok", "contacts": {"dave": {"user_id": "42"}}})
    reply = discord.run(to="nobody", message="hi")
    assert "don't know" in reply.lower() and "dave" in reply
    assert discord._sent == []


def test_an_empty_message_is_refused(discord, telegram, monkeypatch):
    _configure(monkeypatch, discord, {"bot_token": "tok", "default_channel_id": "1"})
    assert "what would you like" in discord.run(message="  ").lower()
    _configure(monkeypatch, telegram, {"bot_token": "tok", "default_chat_id": "1"})
    assert "what would you like" in telegram.run().lower()
    assert discord._sent == [] and telegram._sent == []


def test_no_recipient_and_no_default_is_refused(discord, monkeypatch):
    _configure(monkeypatch, discord, {"bot_token": "tok"})
    reply = discord.run(message="hi")
    assert "who should i message" in reply.lower()
    assert discord._sent == []


def test_a_transport_failure_is_reported_not_swallowed(discord, monkeypatch):
    _configure(monkeypatch, discord, {"bot_token": "tok", "default_channel_id": "5"})
    monkeypatch.setattr(discord, "_post", lambda *a, **k: (False, "Discord returned 500."))
    reply = discord.run(message="hi")
    assert "could not send" in reply.lower() and "500" in reply


def test_contacts_action_lists_saved_people(discord, monkeypatch):
    _configure(monkeypatch, discord,
               {"contacts": {"dave": {"user_id": "1"}, "team": {"channel_id": "2"}}})
    listing = discord.run(action="contacts")
    assert "dave" in listing and "team" in listing
    assert "DM" in listing and "channel" in listing


# ── the shipped credential template must stay empty ─────────────────────────

def test_the_messaging_config_template_holds_no_secrets():
    import json
    config = ROOT / "config" / "messaging.json"
    if not config.is_file():
        pytest.skip("no messaging.json in this checkout")
    data = json.loads(config.read_text(encoding="utf-8"))
    for platform in ("discord", "telegram"):
        section = data.get(platform) or {}
        for field in ("bot_token", "webhook_url"):
            value = str(section.get(field, ""))
            assert value == "" or value.startswith("<"), (
                f"a real {platform} {field} must never be committed")
