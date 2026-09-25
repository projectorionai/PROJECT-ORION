"""Exercise service failure replies without network access or account actions."""
import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "config/custom_tools" / (name + "_tool.py")
    spec = importlib.util.spec_from_file_location(name, path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    return plugin


@pytest.mark.parametrize("name", ["homeassistant_call_service", "ifttt_webhook", "philips_hue", "push_notify"])
def test_unconfigured_integration_is_not_success(name, monkeypatch):
    plugin = load(name)
    monkeypatch.setattr(plugin, "_config", lambda: {})
    monkeypatch.setattr(plugin.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Network access"))
    assert not plugin.run().ok


@pytest.mark.parametrize("name", ["ifttt_webhook", "push_notify"])
def test_unknown_action_cannot_trigger_a_send(name, monkeypatch):
    plugin = load(name)
    monkeypatch.setattr(plugin, "_config", lambda: pytest.fail("Invalid action reached configuration"))
    assert not plugin.run(action="typo").ok


def test_hue_rejects_http_200_error_and_unconfirmed_state(monkeypatch):
    plugin = load("philips_hue")
    monkeypatch.setattr(plugin, "_config", lambda: {"bridge": "bridge.invalid", "username": "fixture"})
    lights = {"1": {"name": "Desk", "state": {"on": False}}}
    monkeypatch.setattr(plugin, "_lights", lambda _: lights)
    monkeypatch.setattr(plugin, "_request", lambda *a, **k: [{"error": {"type": 1}}])
    assert not plugin.run(action="on").ok
    monkeypatch.setattr(plugin, "_request", lambda *a, **k: [{"success": {"/lights/1/state/on": True}}])
    assert not plugin.run(action="on").ok  # accepted, but readback still says off
    lights["1"]["state"]["on"] = True
    assert plugin.run(action="on").ok


def test_hue_pairing_write_failure_is_not_success(monkeypatch):
    plugin = load("philips_hue")
    monkeypatch.setattr(plugin, "_config", lambda: {})
    monkeypatch.setattr(plugin, "_request", lambda *a, **k: [{"success": {"username": "fixture"}}])
    def denied(data):
        raise PermissionError("test denial")
    monkeypatch.setattr(plugin, "_save", denied)
    assert not plugin.run(action="pair", bridge="bridge.invalid").ok


def test_push_rejects_error_inside_successful_http_response(monkeypatch):
    plugin = load("push_notify")
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"status": 0, "errors": ["rejected"]}).encode()
    monkeypatch.setattr(plugin.urllib.request, "urlopen", lambda *a, **k: Response())
    assert plugin._post("https://fixture.invalid", b"test", {})[0] is False


def test_weather_rejects_invalid_input_empty_reply_and_connection_loss(monkeypatch):
    plugin = load("open_meteo_weather")
    assert not plugin.run().ok
    assert not plugin.run(place="Example", days="bad").ok
    monkeypatch.setattr(plugin, "_locate", lambda _: (0, 0, "Example"))
    monkeypatch.setattr(plugin, "_get", lambda *a, **k: {})
    assert not plugin.run(place="Example").ok
    def offline(*a, **k): raise urllib.error.URLError("offline")
    monkeypatch.setattr(plugin, "_get", offline)
    assert not plugin.run(place="Example").ok


def test_homeassistant_rejected_token_and_invalid_service_data(monkeypatch):
    plugin = load("homeassistant_call_service")
    monkeypatch.setattr(plugin, "_config", lambda: {"url": "https://fixture.invalid", "token": "fixture"})
    def rejected(*a, **k):
        raise urllib.error.HTTPError("https://fixture.invalid", 401, "unauthorised", {}, None)
    monkeypatch.setattr(plugin, "_request", rejected)
    assert not plugin.run(action="status").ok
    assert not plugin.run(domain="light", service="turn_on", data="[]").ok
