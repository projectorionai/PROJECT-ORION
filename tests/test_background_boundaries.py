"""Regressions for worker, native-process and queued-Qt policy boundaries."""
from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from orion_core import execution_policy as policy
from orion_core.data import ToolResult


async def test_owned_executor_copies_scope_without_leaking_it_to_next_submission():
    from orion_core.executors import DaemonThreadExecutor
    pool = DaemonThreadExecutor(1, max_pending=4, join_timeout=1)
    try:
        loop = asyncio.get_running_loop()
        with policy.background_scope("bulk-job", "bulk-1") as scope:
            observed = await loop.run_in_executor(pool, policy.current_scope)
            assert observed is scope
        assert not observed.active
        assert await loop.run_in_executor(pool, policy.current_scope) is None
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize("command", [
    'python -c "import ctypes; ctypes.windll.user32.SetForegroundWindow(123)"',
    'python -c "import ctypes; ctypes.windll.user32.SendInput(1, None, 0)"',
    'node -e "require(\'child_process\').spawn(\'notepad.exe\')"',
    'npm run user-defined-hook',
    'dotnet run --project synthetic-gui',
])
async def test_unrestricted_dev_commands_are_refused_before_spawning(monkeypatch, command):
    from orion_core import dispatch_files
    monkeypatch.setattr(dispatch_files.subprocess, "run", lambda *_a, **_k: pytest.fail("Background arbitrary host code spawned"))
    with policy.background_scope("ordinary-tool"):
        result = await dispatch_files.FilesDispatchMixin()._run_dev_command({"command": command})
    assert result.receipt["effect"] == "unrestricted_development_process"
    assert not result.ok and result.receipt["desktop_mutated"] is False


def test_debugger_and_forge_process_sinks_cannot_escape_scope(monkeypatch):
    from orion_core import debugger, sandbox
    monkeypatch.setattr(debugger.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Background Python child"))
    session = object.__new__(debugger.DebugSession)
    harness = SimpleNamespace(_isolation_mode="process")
    with policy.background_scope("workflow"):
        assert session.start().receipt["status"] == "requires_foreground"
        assert session.send("!import ctypes; ctypes.windll.user32.SendInput(1,None,0)").receipt["status"] == "requires_foreground"
        output = sandbox.SandboxVerificationHarness._run_script(harness, "synthetic.py", 1)
    assert output[2] == -1 and "foreground" in output[1]


@pytest.mark.parametrize("attached", [False, True])
async def test_outlook_background_call_can_attach_but_cannot_launch(monkeypatch, attached):
    from orion_core.outlook import OutlookService
    calls = []
    pythoncom = ModuleType("pythoncom")
    pythoncom.CoInitialize = lambda: calls.append("init")
    pythoncom.CoUninitialize = lambda: calls.append("uninit")
    package = ModuleType("win32com")
    package.__path__ = []
    client = ModuleType("win32com.client")
    outlook = SimpleNamespace(GetNamespace=lambda name: "Synthetic namespace")
    def get_active(_name):
        calls.append("attach")
        if not attached:
            raise OSError("Synthetic Outlook not running")
        return outlook
    client.GetActiveObject = get_active
    client.Dispatch = lambda *_a: pytest.fail("Background Outlook launch")
    package.client = client
    monkeypatch.setitem(sys.modules, "pythoncom", pythoncom)
    monkeypatch.setitem(sys.modules, "win32com", package)
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    owner = SimpleNamespace(_note_state=lambda *_a: None)
    with policy.background_scope("ordinary-tool"):
        result = await OutlookService._run_com(owner, lambda app, ns: ToolResult(ns), launch=True)
    assert calls == ["init", "attach", "uninit"]
    if attached:
        assert result.ok and result.text == "Synthetic namespace"
    else:
        assert not result.ok and result.receipt["effect"] == "outlook_application_launch"


async def test_gui_navigation_and_presentation_refuse_before_emitting_queued_qt_requests():
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    host = DesktopDispatchMixin()
    with policy.background_scope("ordinary-tool"):
        assert host.interface_control({"action": "page", "target": "TASKS"}).receipt["status"] == "requires_foreground"
        assert (await host.presentation_tool("display", {"content": "Synthetic"})).receipt["status"] == "requires_foreground"
        assert host.shutdown_orion({}).receipt["status"] == "requires_foreground"
        assert host.restart_orion({}).receipt["status"] == "requires_foreground"
        assert host.gesture_control({"action": "pointer_start"}).receipt["status"] == "requires_foreground"


async def test_mcp_configuration_and_spend_do_not_open_modal_prompts():
    from orion_core.dispatch_web import WebDispatchMixin
    host = WebDispatchMixin()
    host.mcp_host = SimpleNamespace(requires_confirmation=lambda *_a: "Synthetic paid action")
    host.system_guard = SimpleNamespace(request_confirmation=lambda *_a, **_k: pytest.fail("Background modal was armed"))
    with policy.background_scope("plugin"):
        assert (await host.mcp_tool({"action": "enable", "server": "custom"})).receipt["status"] == "requires_foreground"
        assert host._mcp_spend_gate("custom", "send", {}, confirmed=True).receipt["status"] == "requires_foreground"


def test_file_approval_is_deferred_before_arming_a_gui_token(tmp_path):
    from orion_core.dispatch_files import FilesDispatchMixin
    from orion_core.path_policy import Zone
    path = tmp_path / "synthetic.txt"
    path.write_text("Keep this", encoding="utf-8")
    host = SimpleNamespace(_file_references=lambda: SimpleNamespace(resolve=lambda *_a: (False, None)),
                           _resolve_user_path=lambda *_a: path,
                           _ensure_write_safe=lambda *_a: SimpleNamespace(path=path, zone=Zone.USER))
    with policy.background_scope("workflow"):
        result = FilesDispatchMixin.file_controller(host, {"action": "write_text", "path": str(path), "text": "Replace"})
    assert result.receipt["effect"] == "file_confirmation_dialog"
    assert path.read_text(encoding="utf-8") == "Keep this"


def test_research_auto_navigation_does_not_cross_the_queued_qt_boundary():
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    host = SimpleNamespace(bus=SimpleNamespace(gui_command=SimpleNamespace(
        emit=lambda *_a: pytest.fail("Background research raised or navigated the GUI"))))
    with policy.background_scope("research"):
        KnowledgeDispatchMixin._show_research_console(host)


def test_reviewed_tailscale_observation_does_not_flash_a_console(monkeypatch):
    from orion_core import mcp_network
    calls = []
    monkeypatch.setattr(mcp_network.shutil, "which", lambda *_a: "synthetic-tailscale.exe")
    monkeypatch.setattr(mcp_network.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"BackendState":"Running"}')
    monkeypatch.setattr(mcp_network.subprocess, "run", run)
    assert mcp_network._tailscale_status() == {"BackendState": "Running"}
    assert calls[0][0] == ["synthetic-tailscale.exe", "status", "--json"]
    assert calls[0][1]["creationflags"] == 0x08000000
