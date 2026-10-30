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


@pytest.mark.parametrize("name", ["homeassistant_call_service", "ifttt_webhook", "open_meteo_weather", "philips_hue", "push_notify"])
def test_remote_plugin_response_is_capped_before_decode_and_never_retried(monkeypatch, name):
    plugin = load(name)
    reads, requests = [], []
    class Oversized:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size=-1):
            reads.append(size)
            assert size > 0, "unbounded read of a remote body"
            return b"x" * size
    def response(*args, **kwargs):
        requests.append(1)
        return Oversized()
    monkeypatch.setattr(plugin.urllib.request, "urlopen", response)
    monkeypatch.setattr(plugin.json, "loads", lambda *_: pytest.fail("Oversized response reached JSON parser"))
    if name == "homeassistant_call_service":
        with pytest.raises(ValueError, match="size limit"):
            plugin._request({"url": "https://fixture.invalid", "token": "fixture"}, "/api/")
    elif name == "philips_hue":
        with pytest.raises(ValueError, match="size limit"):
            plugin._request("https://fixture.invalid")
    elif name == "open_meteo_weather":
        with pytest.raises(ValueError, match="size limit"):
            plugin._get("https://fixture.invalid", {})
    elif name == "push_notify":
        assert plugin._post("https://fixture.invalid", b"fixture", {})[0] is False
    else:
        monkeypatch.setattr(plugin, "_config", lambda: {"key": "fixture"})
        result = plugin.run(event="fixture")
        assert not result.ok and "size limit" in result.text
    assert reads == [plugin.MAX_RESPONSE_BYTES + 1] and requests == [1]


def test_legacy_game_http_cap_rejects_oversize_without_parsing():
    # Extract only the standalone body-limit helper as data. Importing this
    # legacy module is unnecessary for a byte-limit witness.
    import ast
    path = Path(__file__).resolve().parents[1] / "orion_core/jv_plugins/legacy_game_updater.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "_bounded_http_body") or (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MAX_RESPONSE_BYTES" for target in node.targets))]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    reads = []
    class Oversized:
        def read(self, size=-1):
            reads.append(size)
            assert size > 0
            return b"x" * size
    with pytest.raises(ValueError, match="size limit"):
        namespace["_bounded_http_body"](Oversized())
    assert reads == [namespace["MAX_RESPONSE_BYTES"] + 1]


@pytest.mark.parametrize("available,frame_ok", [(False, False), (True, False), (True, True)])
def test_camera_plugin_reports_capture_outcome_and_releases_fake_capture(monkeypatch, available, frame_ok):
    plugin = load("live_camera_analysis")
    from orion_core.plugin_results import plugin_result
    releases = []
    class Frame:
        shape = (480, 640, 3)
        def mean(self): return 150
    class Capture:
        def read(self): return frame_ok, Frame() if frame_ok else None
        def release(self): releases.append(1)
    def open_fake(self):
        self.capture = Capture()
        return available
    monkeypatch.setattr(plugin.LiveCameraAnalysis, "open", open_fake)
    result = plugin_result(plugin.run())
    assert result.ok is (available and frame_ok)
    assert result.receipt["captured"] is (available and frame_ok)
    assert releases == [1]
    if result.ok:
        assert result.text == "Captured a 640x480 frame; it is well lit (mean intensity 150 of 255)."
