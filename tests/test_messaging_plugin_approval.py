"""Real registry/SystemGuard admission; every external effect is intercepted."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher
from orion_core.execution_policy import background_scope
from orion_core.messaging import MessagingGateway
from orion_core.plugin_registry import PluginRegistry
from orion_core.system_guard import SystemActionGuard
import orion_core

ROOT = Path(orion_core.__file__).resolve().parent.parent


class Signal:
    def __init__(self):
        self.calls = []
    def emit(self, *args):
        self.calls.append(args)


@pytest.fixture(params=["discord", "telegram"])
def configured(request, tmp_path, monkeypatch):
    platform = request.param
    name = platform + "_message"
    directory = tmp_path / "plugins"
    directory.mkdir()
    for suffix in ("_tool.py", ".plugin.json"):
        shutil.copyfile(ROOT / "config/custom_tools" / (name + suffix), directory / (name + suffix))
    import orion_core.plugin_registry as registry_module
    monkeypatch.setattr(registry_module, "PLUGIN_STATE_PATH", tmp_path / "plugin-state.json")
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "1")
    monkeypatch.setenv(platform.upper() + "_BOT_TOKEN", "synthetic-account-token")
    bus = SimpleNamespace(log=Signal(), confirm_action=Signal())
    registry = PluginRegistry(bus, directory=directory)
    record = registry.get(name)
    assert record is not None and record.tier == "confirm" and "network" in record.permissions
    assert not record.problems and not record.missing_deps
    path = directory / (name + "_tool.py")
    spec = importlib.util.spec_from_file_location("synthetic_gated_" + platform, path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    plugin.run._orion_plugin_source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    effects = []
    def forbidden(effect):
        def call(*_args, **_kwargs):
            effects.append(effect)
            pytest.fail("A pre-transport permission result must not cause " + effect)
        return call
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", forbidden("HTTP"))
    monkeypatch.setattr("orion_core.messaging.open_url", forbidden("default browser"))
    host = OrionDispatcher.__new__(OrionDispatcher)
    host.bus, host.plugins, host.system_guard = bus, registry, SystemActionGuard()
    host.plugin_vault = SimpleNamespace(secrets_for=lambda *_: {})
    host._tool_handlers = {name: plugin.run}
    host.TOOL_DECLARATIONS = [plugin.get_tool_schema()]
    host.messaging = MessagingGateway(bus)
    host.messaging.delivery_suppressed = lambda: False
    host.messaging._bot_token = lambda *_: "synthetic-account-token"
    host.messaging._discord_webhook = lambda: ""
    host.messaging._browser_sender = SimpleNamespace(PLATFORMS={"discord", "whatsapp"}, send=forbidden("owned browser fallback"))
    host.messaging._desktop_sender = SimpleNamespace(PLATFORMS={"discord", "telegram"}, send=forbidden("native fallback"), control=None)
    host.control = None
    return SimpleNamespace(host=host, platform=platform, name=name, registry=registry, bus=bus,
                           effects=effects, manifest=directory / (name + ".plugin.json"))


def send(fixture, body="  Exact approved body\nUnicode 🙂  "):
    async def run():
        with background_scope("messaging-gate-integration"):
            return await fixture.host.messaging_tool({"action": "send", "platform": fixture.platform,
                                                       "contact": "123", "message": body})
    return asyncio.run(run())


def assert_blocked(fixture, result, status):
    assert not result.ok and result.receipt["status"] == status
    assert result.receipt["attempted"] is False and result.receipt["sent"] is False
    assert result.receipt["backend"] == "registered_plugin"
    assert fixture.effects == []
    assert fixture.registry.get(fixture.name).calls == 0


def test_configured_api_queues_real_guard_approval_without_transport(configured):
    body = "  Exact approved body\nUnicode 🙂  "
    result = send(configured, body)
    assert_blocked(configured, result, "awaiting_confirmation")
    assert configured.host.system_guard.pending_count() == 1
    assert len(configured.bus.confirm_action.calls) == 1
    token = configured.bus.confirm_action.calls[0][0]["token"]
    pending = configured.host.system_guard._pending[token]
    assert pending.payload["plugin"] == configured.name
    assert pending.payload["arguments"] == {"action": "send", "to": "123", "message": body}
    configured.host.system_guard.cancel(token)
    assert configured.host.system_guard.pending_count() == 0 and configured.effects == []


def test_disabled_real_registry_prevents_api_and_alternate_account(configured):
    assert configured.registry.set_enabled(configured.name, False).ok
    assert_blocked(configured, send(configured), "disabled")
    assert configured.host.system_guard.pending_count() == 0 and not configured.bus.confirm_action.calls


def test_unactivated_api_preserves_unavailable_before_transport(configured):
    configured.host._tool_handlers.clear()
    assert_blocked(configured, send(configured), "unavailable")
    assert configured.host.system_guard.pending_count() == 0 and not configured.bus.confirm_action.calls


def test_input_capable_plugin_defers_without_native_prompt(configured):
    data = json.loads(configured.manifest.read_text(encoding="utf-8"))
    data["permissions"].append("input")
    configured.manifest.write_text(json.dumps(data), encoding="utf-8")
    assert_blocked(configured, send(configured), "requires_foreground")
    assert configured.host.system_guard.pending_count() == 0 and not configured.bus.confirm_action.calls


@pytest.mark.parametrize("status", ["requires_confirmation", "awaiting_confirmation", "requires_foreground", "disabled", "unavailable", "forbidden", "rejected"])
def test_typed_pre_submit_refusal_remains_terminal(status):
    gateway = MessagingGateway(SimpleNamespace(log=Signal()))
    gateway._bot_token = lambda *_: "synthetic-account-token"
    gateway._discord_webhook = lambda: ""
    gateway.bind_plugin_sender(lambda *_: ToolResult("Typed gate refusal", ok=False, receipt={"status": status}))
    result = gateway._send_via_bot("discord", "123", "Exact")
    assert not result.ok and result.receipt["status"] == status
    assert result.receipt["attempted"] is False and result.receipt["sent"] is False


def test_post_submit_uncertainty_never_becomes_not_attempted():
    gateway = MessagingGateway(SimpleNamespace(log=Signal()))
    gateway._bot_token = lambda *_: "synthetic-account-token"
    gateway._discord_webhook = lambda: ""
    gateway.bind_plugin_sender(lambda *_: ToolResult("Transport result lost", ok=False,
        receipt={"status": "uncertain", "attempted": True, "sent": None}))
    result = gateway._send_via_bot("discord", "123", "Exact")
    assert not result.ok and result.receipt["status"] == "uncertain"
    assert result.receipt["attempted"] is True and result.receipt["sent"] is None
