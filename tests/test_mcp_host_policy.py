"""Exercise Windows path boundaries and the actual telemetry stdio transport."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core.mcp_host import MCPReply, MCPServerConn
from orion_core.mcp_policy import child_environment, scoped_path, validate_arguments


@pytest.mark.parametrize("name", ["../secret", "..\\secret", "x:stream", "NUL.txt", "com1",
                                  "folder. /x", "C:relative", "\\\\server\\share\\file",
                                  "\\\\?\\C:\\secret", "bad\x00path", "CONOUT$"])
def test_windows_escape_forms_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        scoped_path(name, [str(tmp_path)])


def test_paths_cannot_escape_sibling_or_symlink(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    assert scoped_path("new.txt", [str(root)]) == str(root / "new.txt")
    with pytest.raises(ValueError):
        scoped_path(str(tmp_path / "allowed_other" / "secret"), [str(root)])
    link = root / "escape"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs Windows developer mode or elevation")
    with pytest.raises(ValueError):
        scoped_path(str(link / "outside.txt"), [str(root)])


def test_reference_filesystem_roots_apply_to_all_path_fields(tmp_path):
    spec = {"args": ["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)]}
    result = validate_arguments(spec, {"paths": ["a", "b"], "destination": "c"})
    assert result["paths"] == [str(tmp_path / "a"), str(tmp_path / "b")]
    with pytest.raises(ValueError):
        validate_arguments(spec, {"source": "../../outside"})


def test_git_revision_options_and_file_escape_are_rejected(tmp_path):
    spec = {"args": ["mcp-server-git@2025.9.25", "--repository", str(tmp_path)]}
    with pytest.raises(ValueError):
        validate_arguments(spec, {"repo_path": str(tmp_path), "revision": "--output=x"})
    with pytest.raises(ValueError):
        validate_arguments(spec, {"repo_path": str(tmp_path), "files": ["../secret"]})


def test_arguments_must_be_finite_bounded_json():
    for value in ([1], {"number": float("nan")}, {"object": object()}, {"data": "a" * 270000}):
        with pytest.raises(ValueError):
            validate_arguments({}, value)


def test_child_gets_only_its_explicit_credentials():
    parent = {"PATH": "bin", "OPENAI_API_KEY": "secret1", "OTHER_TOKEN": "secret2",
              "DB_PASS": "secret3", "GITHUB_PAT": "secret4",
              "DATABASE_URL": "postgres://user:secret5@host/db",
              "SERVICE_DSN": "secret6", "REDIS_URL": "redis://:secret7@host",
              "MONGODB_URI": "mongodb://user:secret8@host", "LANG": "en_GB"}
    env = child_environment({"env": {"SERVER_TOKEN": "intended"}}, parent)
    assert env == {"PATH": "bin", "LANG": "en_GB", "SERVER_TOKEN": "intended"}
    assert child_environment({"inherit_env": ["OTHER_TOKEN"]}, parent)["OTHER_TOKEN"] == "secret2"


def test_short_named_explicit_credential_is_redacted():
    conn = MCPServerConn("test", {"env": {"SERVICE_DSN": "postgres://private"}}, None)
    assert "private" not in conn._redact("connection refused for postgres://private")


def test_redacting_credentials_preserves_error_type():
    conn = MCPServerConn("test", {"env": {"SERVER_TOKEN": "private-value"}}, None)
    reply = conn._redact(MCPReply("private-value rejected", "tool"))
    assert reply.kind == "tool" and reply.is_error
    assert "private-value" not in reply


async def test_telemetry_runs_through_real_stdio_and_stops():
    from orion_core import mcp_telemetry
    conn = MCPServerConn("telemetry", {"command": sys.executable,
                         "args": [str(Path(mcp_telemetry.__file__))]}, None)
    try:
        assert await conn.start(), conn.last_error
        reply = await conn.call_tool("system_snapshot", {})
        assert not reply.is_error
        assert json.loads(reply)["ram_total_bytes"] > 0
        assert "process_memory" in {t["name"] for t in conn.tools}
        bad = await conn.call_tool("process_memory", {"limit": 1000})
        assert bad.is_error
        proc = conn.proc
    finally:
        await conn.stop()
    assert proc.returncode is not None
    assert conn._stderr_task is None


async def test_cancelled_handshake_reaps_server(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    conn = MCPServerConn("slow", {"command": sys.executable, "args": [str(script)]}, None)
    task = asyncio.create_task(conn.start())
    for _ in range(100):
        if conn.proc is not None:
            break
        await asyncio.sleep(0.01)
    proc = conn.proc
    assert proc is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.returncode is not None
    assert conn.proc is None


def test_telemetry_is_opt_in_and_schema_pruning_preserves_discovery():
    from orion_core.mcp_host import _default_template
    from orion_core.tool_gateway import live_declarations
    spec = _default_template()["servers"]["orion_telemetry"]
    assert not spec["enabled"]
    declarations = [{"name": n} for n in ["find_tool", "use_tool", "mcp__orion_telemetry__system_snapshot"]]
    assert [d["name"] for d in live_declarations(declarations)] == ["find_tool", "use_tool"]


async def test_model_cannot_register_executable_without_human_token(monkeypatch):
    from orion_core import tasks
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard
    calls, spawned = [], []
    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    dispatcher.system_guard = SystemActionGuard()
    dispatcher.bus = SimpleNamespace(confirm_action=SimpleNamespace(emit=lambda *_: None),
                                     log=SimpleNamespace(emit=lambda *_: None))
    dispatcher.mcp_host = SimpleNamespace(add_server=lambda *args: calls.append(args) or "declared")
    request = {"action": "add", "server": "local", "command": "python",
               "arguments": ["-m", "reviewed_server"], "confirm": True}
    result = await dispatcher.mcp_tool(request)
    assert not result.ok and not calls
    pending = next(iter(dispatcher.system_guard._pending.values()))
    request["arguments"][1] = "substituted_server"
    monkeypatch.setattr(tasks, "spawn", lambda coro, **kwargs: spawned.append(asyncio.create_task(coro)) or spawned[-1])
    assert dispatcher.confirm_system_action(pending.token).ok
    await asyncio.gather(*spawned)
    assert calls[0][2] == ["-m", "reviewed_server"]
    assert not dispatcher.confirm_system_action(pending.token).ok  # one-use token


def test_frozen_build_ships_telemetry_script():
    import ast
    source = Path(__file__).parents[1] / "build_standalone.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    values = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "DATAS" for t in n.targets))
    assert ("orion_core/mcp_telemetry.py", "orion_core") in values
    assert ("orion_core/mcp_network.py", "orion_core") in values
    assert ("orion_core/mcp_world.py", "orion_core") in values
    assert ("orion_core/mcp_desktop.py", "orion_core") in values
