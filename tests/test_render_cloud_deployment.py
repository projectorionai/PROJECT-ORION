"""Managed-container ingress stays HTTPS-advertised and pairing-protected."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy import render_entrypoint
from deploy.render_entrypoint import configure_environment
from orion_core.remote import RemoteGateway


ROOT = Path(__file__).resolve().parents[1]


def _environment(tmp_path):
    return {
        "RENDER": "true", "RENDER_EXTERNAL_URL": "https://orion-test.onrender.com",
        "PORT": "10000", "ORION_RENDER_EPHEMERAL_ACK": "1",
        "ORION_CONFIG_DIR": str(tmp_path / "cloud-data"),
        "ORION_GEMINI_API_KEY": "private-test-value",
    }


def test_launcher_maps_platform_port_origin_and_preserves_private_key(tmp_path):
    environ = _environment(tmp_path)
    configure_environment(environ)
    assert environ["ORION_REMOTE_HOST"] == "0.0.0.0"
    assert environ["ORION_REMOTE_PORT"] == "10000"
    assert environ["ORION_REMOTE_PUBLIC_URL"] == "https://orion-test.onrender.com:443/"
    assert environ["ORION_REMOTE_EXPOSURE"] == "platform-proxy"
    assert environ["ORION_REMOTE_TLS"] == "0"
    assert environ["ORION_HEADLESS"] == "1"
    assert environ["ORION_MODE"] == "cloud"
    assert environ["ORION_GEMINI_API_KEY"] == "private-test-value"
    assert not Path(environ["ORION_CONFIG_DIR"]).exists()


@pytest.mark.parametrize("name,value", [
    ("RENDER", "false"), ("ORION_RENDER_EPHEMERAL_ACK", "0"),
    ("PORT", "0"), ("PORT", "65536"), ("PORT", "1e4"), ("PORT", "１２３"),
    ("PORT", "8080\n8081"), ("ORION_CONFIG_DIR", "relative/state"),
    ("ORION_CONFIG_DIR", ""), ("ORION_REMOTE_TLS", "1"),
    ("RENDER_EXTERNAL_URL", "http://orion-test.onrender.com"),
    ("RENDER_EXTERNAL_URL", "https://orion-test.onrender.com.attacker.example"),
    ("RENDER_EXTERNAL_URL", "https://orion-test.onrender.com:8765"),
    ("RENDER_EXTERNAL_URL", "https://user:password@orion-test.onrender.com"),
    ("RENDER_EXTERNAL_URL", "https://orion-test.onrender.com/secret"),
    ("RENDER_EXTERNAL_URL", "https://orion-test.onrender.com?token=secret"),
])
def test_launcher_rejects_bad_configuration_without_partial_changes(tmp_path, name, value):
    environ = _environment(tmp_path)
    environ[name] = value
    before = dict(environ)
    with pytest.raises(ValueError):
        configure_environment(environ)
    assert environ == before


def test_launcher_replaces_process_without_exposing_key_in_arguments(monkeypatch, tmp_path, capsys):
    environ = _environment(tmp_path)
    monkeypatch.setattr(render_entrypoint.os, "environ", environ)
    calls = []

    def replace(executable, arguments, child_environment):
        calls.append((executable, arguments, child_environment))

    monkeypatch.setattr(render_entrypoint.os, "execve", replace)
    assert render_entrypoint.main() == 0
    executable, arguments, child_environment = calls[0]
    assert arguments == [executable, str(ROOT / "orion.py"), "--headless"]
    assert "private-test-value" not in repr(arguments)
    assert child_environment["ORION_GEMINI_API_KEY"] == "private-test-value"
    output = capsys.readouterr()
    assert "private-test-value" not in output.out + output.err


def test_launcher_errors_do_not_echo_a_secret_from_a_bad_origin(monkeypatch, tmp_path, capsys):
    environ = _environment(tmp_path)
    environ["RENDER_EXTERNAL_URL"] = "https://user:secret-in-url@orion-test.onrender.com"
    monkeypatch.setattr(render_entrypoint.os, "environ", environ)
    calls = []
    monkeypatch.setattr(render_entrypoint.os, "execve", lambda *args: calls.append(args))
    assert render_entrypoint.main() == 2
    assert calls == []
    output = capsys.readouterr()
    assert "configuration failed" in output.err
    assert "secret-in-url" not in output.out + output.err
    assert "private-test-value" not in output.out + output.err


def _gateway(monkeypatch, tmp_path):
    environ = _environment(tmp_path)
    configure_environment(environ)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda _message: None))
    return RemoteGateway(None, None, bus, config_dir=tmp_path / "gateway")


def test_managed_gateway_advertises_only_the_platform_https_origin(monkeypatch, tmp_path):
    gateway = _gateway(monkeypatch, tmp_path)
    assert gateway.port == 10000
    assert gateway._public_https is True
    assert gateway._serve_mode is False
    assert asyncio.run(gateway._endpoints()) == [{
        "kind": "reverse-proxy", "host": "orion-test.onrender.com", "port": 443,
        "https": True, "url": "https://orion-test.onrender.com:443/", "priority": 10,
    }]


def test_platform_mode_keeps_pairing_and_rejects_forwarded_identity(monkeypatch, tmp_path):
    async def flow():
        from aiohttp.test_utils import TestClient, TestServer

        gateway = _gateway(monkeypatch, tmp_path)
        async with TestClient(TestServer(gateway._build_app())) as client:
            forged = {"X-Forwarded-For": "127.0.0.1", "X-Forwarded-Proto": "https",
                      "Forwarded": "for=127.0.0.1;proto=https", "Authorization": "Bearer forged"}
            response = await client.get("/api/pc/power", headers=forged)
            assert response.status == 401
            response = await client.get("/api/endpoints", headers=forged)
            assert response.status == 401
            code = gateway.begin_pairing()
            response = await client.post("/v1/auth/pair", json={"code": code, "device_name": "Test phone"})
            assert response.status == 200
            paired = await response.json()
            token_response = await client.post("/v1/auth/token", json={
                "device_id": paired["device_id"], "refresh_token": paired["refresh_token"],
            })
            assert token_response.status == 200
            token = (await token_response.json())["access_token"]
            response = await client.get("/api/endpoints", headers={"Authorization": f"Bearer {token}"})
            assert response.status == 200
            body = await response.json()
            assert body["https"] is True and body["port"] == 443
            assert body["endpoints"][0]["host"] == "orion-test.onrender.com"
            assert gateway.auth.revoke_device(paired["device_id"])
            response = await client.get("/api/endpoints", headers={"Authorization": f"Bearer {token}"})
            assert response.status == 401
    asyncio.run(flow())


@pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.2", "localhost"])
def test_platform_proxy_rejects_unintended_bind(monkeypatch, host):
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "platform-proxy")
    monkeypatch.setenv("ORION_REMOTE_PUBLIC_URL", "https://orion-test.onrender.com")
    monkeypatch.setenv("ORION_REMOTE_HOST", host)
    with pytest.raises(ValueError, match="isolated container"):
        RemoteGateway(None, None, None)


def test_platform_proxy_needs_explicit_valid_https_origin(monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "platform-proxy")
    monkeypatch.setenv("ORION_REMOTE_PUBLIC_URL", "http://example.com")
    monkeypatch.setenv("ORION_REMOTE_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="HTTPS origin"):
        RemoteGateway(None, None, None)


def test_platform_does_not_trust_forwarded_https_for_key_provisioning(monkeypatch, tmp_path):
    gateway = _gateway(monkeypatch, tmp_path)
    gateway._build_app()
    request = SimpleNamespace(secure=False, remote="10.0.0.4", headers={
        "X-Forwarded-Proto": "https", "X-Forwarded-For": "127.0.0.1",
        "Forwarded": "for=127.0.0.1;proto=https",
    })
    response = asyncio.run(gateway._handle_mobile_ai_profile(request))
    assert response.status == 403


def test_render_image_has_an_unprivileged_user_and_no_private_state_copies():
    dockerfile = (ROOT / "deploy" / "Dockerfile.render").read_text(encoding="utf-8")
    copies = [line.split()[1] for line in dockerfile.splitlines() if line.startswith("COPY ")]
    assert copies == ["deploy/requirements-server.txt", "orion.py", "orion_core",
                      "config/knowledge_packs", "deploy/render_entrypoint.py"]
    assert "USER orion" in dockerfile
    assert "ENV ORION_GEMINI_API_KEY" not in dockerfile
    context = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert context.splitlines()[1] == "**"
    assert "!deploy/render_entrypoint.py" in context
    assert "!config/api_keys.json" not in context


def test_blueprint_uses_free_compute_no_paid_disk_and_a_private_key_prompt():
    import yaml

    blueprint = yaml.safe_load((ROOT / "deploy" / "render-free.yaml").read_text(encoding="utf-8"))
    assert set(blueprint) == {"services"}
    service, = blueprint["services"]
    assert service["type"] == "web" and service["runtime"] == "docker"
    assert service["plan"] == "free" and service["autoDeployTrigger"] == "off"
    assert service["healthCheckPath"] == "/api/health"
    assert "disk" not in service
    gemini = next(value for value in service["envVars"] if value["key"] == "ORION_GEMINI_API_KEY")
    assert gemini == {"key": "ORION_GEMINI_API_KEY", "sync": False}
