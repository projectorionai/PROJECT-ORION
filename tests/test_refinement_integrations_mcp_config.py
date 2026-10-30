"""A damaged saved MCP policy must not become first-run authorization."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import mcp_host as module


BAD_CONFIGS = [
    b'{"servers":', b'[]', b'null', b'\xff\xfe',
    b'{"servers":[]}', b'{"servers":{"retained":null}}',
    pytest.param(b'{"servers":{"retained":{"enabled":"false"}}}', id="ambiguous_enabled_text"),
    pytest.param(b'{"servers":{"retained":{"enabled":null}}}', id="ambiguous_enabled_null"),
    pytest.param(b'{"servers":{},"servers":{"retained":{"enabled":true}}}', id="duplicate_server_policy"),
]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    path = tmp_path / "mcp_servers.json"
    monkeypatch.setattr(module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(module, "MCP_CONFIG_PATH", path)
    monkeypatch.setattr(module, "MCP_CATALOGUE_PATH", tmp_path / "catalogue.json")
    # ORION reads only its own MCP file now; nothing is imported from elsewhere.
    imported, attempted, lines = [], [], []
    host = module.MCPHost(SimpleNamespace(log=SimpleNamespace(emit=lines.append)))
    monkeypatch.setattr(host, "_prepare", lambda _name, spec: spec)
    monkeypatch.setattr(host, "_bind_connection", lambda *_: None)
    monkeypatch.setattr(host, "_beat", lambda *_: None)

    async def start():
        return False

    async def stop(**_kwargs):
        pass

    def connection(name, spec, _bus):
        attempted.append(name)
        return SimpleNamespace(start=start, stop=stop, last_error="synthetic unavailable",
                               tools=[], proc=None, ready=False)

    monkeypatch.setattr(module, "MCPServerConn", connection)
    return SimpleNamespace(path=path, host=host, imported=imported,
                           attempted=attempted, lines=lines)


@pytest.mark.parametrize("contents", BAD_CONFIGS)
async def test_damaged_saved_policy_starts_nothing_or_imports_nothing(isolated, contents):
    isolated.path.write_bytes(contents)
    await isolated.host.connect_all()
    assert isolated.attempted == [], "damaged policy was replaced by enabled defaults"
    assert isolated.imported == [], "damaged policy admitted imported server configuration"
    assert isolated.path.read_bytes() == contents
    assert "restore or repair" in " ".join(isolated.lines).lower()


@pytest.mark.parametrize("operation", ["list", "add", "enable", "disable"])
async def test_management_preserves_unreadable_policy_and_explains_repair(isolated, operation):
    original = b'{"servers":{"retained":'
    isolated.path.write_bytes(original)
    if operation == "list":
        result = isolated.host.list_servers()
    elif operation == "add":
        result = isolated.host.add_server("new", "synthetic-only")
    else:
        result = await isolated.host.set_enabled("fetch", operation == "enable")
    assert "restore or repair" in result.lower()
    assert "mcp_servers.json" in result
    assert isolated.path.read_bytes() == original
    assert isolated.attempted == [] and isolated.imported == []


def test_configured_confirmation_cannot_disappear_when_policy_becomes_corrupt(isolated):
    isolated.path.write_bytes(b'{"servers":')
    reason = isolated.host.requires_confirmation("synthetic_bank", "update")
    assert reason and "restore or repair" in reason.lower()
    assert isolated.imported == []


async def test_existing_connection_cannot_execute_with_unknown_saved_policy(isolated):
    calls = []

    async def call_tool(*args):
        calls.append(args)
        return module.MCPReply("synthetic effect")

    isolated.host.servers["synthetic_bank"] = SimpleNamespace(
        proc=None, tools=[{"name": "update"}], call_tool=call_tool)
    isolated.path.write_bytes(b'{"servers":')
    reply = await isolated.host.call("synthetic_bank", "update", {})
    assert reply.is_error and reply.kind == "unavailable"
    assert "restore or repair" in reply.lower()
    assert calls == []


async def test_reconnect_preserves_saved_policy_and_actionable_status(isolated):
    original = b'{"servers":'
    isolated.path.write_bytes(original)
    assert not await isolated.host.reconnect("fetch")
    assert isolated.attempted == [] and isolated.imported == []
    assert "restore or repair" in isolated.host.status["fetch"].lower()
    assert isolated.path.read_bytes() == original


def test_write_helper_cannot_publish_an_unavailable_configuration(isolated):
    original = b'{"servers":'
    isolated.path.write_bytes(original)
    unavailable = module.load_config()
    unavailable["servers"]["new"] = {"enabled": False}
    assert not isolated.host._write_config(unavailable)
    assert isolated.path.read_bytes() == original


def test_temporary_read_denial_is_redacted_and_never_overwrites_file(isolated, monkeypatch):
    original = json.dumps({"_revision": module.CONFIG_REVISION, "servers": {}}).encode()
    isolated.path.write_bytes(original)
    read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == isolated.path:
            raise PermissionError("synthetic-private-credential")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    result = isolated.host.list_servers()
    assert "restore or repair" in result.lower()
    assert "synthetic-private-credential" not in result
    assert isolated.imported == [] and isolated.path.read_bytes() == original


def test_supported_repaired_config_recovers_without_discarding_saved_choice(isolated):
    isolated.path.write_bytes(b'{"servers":')
    assert "restore or repair" in isolated.host.list_servers().lower()
    repaired = module._default_config()
    repaired["servers"]["retained"] = {"enabled": False, "command": "synthetic-only"}
    isolated.path.write_text(json.dumps(repaired), encoding="utf-8")
    result = isolated.host.list_servers()
    assert "retained — disabled" in result and "restore or repair" not in result.lower()
    assert json.loads(isolated.path.read_text(encoding="utf-8")) == repaired


def test_absent_configuration_keeps_first_run_defaults(isolated):
    assert not isolated.path.exists()
    result = module.load_config()
    assert isolated.path.exists()
    assert result["servers"]["orion_world"]["enabled"] is True
    assert "imported" not in result["servers"]
    assert isolated.imported == []


def test_current_valid_configuration_keeps_custom_confirmation_policy(isolated):
    config = module._default_config()
    config["servers"]["synthetic_bank"] = {
        "enabled": True, "command": "synthetic-only", "confirm": True}
    isolated.path.write_text(json.dumps(config), encoding="utf-8")
    assert isolated.host.requires_confirmation("synthetic_bank", "update")
    assert "restore or repair" not in isolated.host.list_servers().lower()
    assert json.loads(isolated.path.read_text(encoding="utf-8")) == config
