"""Regression overlay for the staged Mark 36 sources; no real OS transports."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import execution_policy as policy


@pytest.mark.parametrize("action", ["copy", "write", "set"])
def test_dispatcher_clipboard_writes_defer_before_looking_up_the_qt_clipboard(monkeypatch, action):
    from orion_core import dispatch_desktop
    monkeypatch.setattr(dispatch_desktop, "QApplication", SimpleNamespace(
        clipboard=lambda: pytest.fail("Background task looked up writable clipboard")))
    with policy.background_scope("ordinary-tool"):
        result = dispatch_desktop.DesktopDispatchMixin().clipboard_operate({"action": action, "text": "Synthetic"})
    assert not result.ok and result.receipt["effect"] == "clipboard_write"


def test_read_only_clipboard_observation_remains_available(monkeypatch):
    from orion_core import dispatch_desktop
    clipboard = SimpleNamespace(text=lambda: "Synthetic observed text",
                                setText=lambda *_a: pytest.fail("Read-only observation wrote clipboard"))
    monkeypatch.setattr(dispatch_desktop, "QApplication", SimpleNamespace(clipboard=lambda: clipboard))
    with policy.background_scope("ordinary-tool"):
        result = dispatch_desktop.DesktopDispatchMixin().clipboard_operate({"action": "read"})
    assert result.ok and result.text == "Synthetic observed text"


@pytest.mark.parametrize("method,args", [("open_app", ("Synthetic",)), ("close_app", ("Synthetic",)),
                                         ("window_control", ("focus", "Synthetic")),
                                         ("_minimise_all", ([],)), ("_move_to_screen", ("Synthetic", "second")),
                                         ("media_control", ("play",)), ("_start_music", ()),
                                         ("_media_via_session", ("play",))])
def test_actual_desktop_agent_mutations_return_before_any_native_state(method, args):
    from orion_core.agents import DesktopAgent
    agent = object.__new__(DesktopAgent)
    with policy.background_scope("workflow"):
        result = getattr(agent, method)(*args)
    assert not result.ok and result.receipt["status"] == "requires_foreground"


def test_native_focus_and_launch_helpers_cannot_bypass_background_policy(monkeypatch):
    from orion_core import agents
    monkeypatch.setattr(agents.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Background launch"))
    monkeypatch.setattr(agents.os, "startfile", lambda *_a, **_k: pytest.fail("Background ShellExecute"), raising=False)
    with policy.background_scope("plugin"):
        assert agents.force_foreground(123) is False
        with pytest.raises(RuntimeError, match="foreground"):
            agents.DesktopAgent._launch("synthetic.exe")
        with pytest.raises(RuntimeError, match="foreground"):
            agents.DesktopAgent._post_close(123)
        assert "foreground" in agents.DesktopAgent._open_folder(Path("synthetic"))


@pytest.mark.parametrize("action", ["open", "reveal", "show_in_folder"])
async def test_find_files_open_intent_is_typed_before_search_or_shell_launch(action):
    from orion_core.dispatch_files import FilesDispatchMixin
    with policy.background_scope("job"):
        result = await FilesDispatchMixin().find_files({"action": action, "path": "synthetic.txt"})
    assert not result.ok and result.receipt["status"] == "requires_foreground"


@pytest.mark.parametrize("method", ["open_path", "reveal_path"])
def test_final_file_sinks_refuse_without_resolving_or_opening_a_path(monkeypatch, method):
    from orion_core import file_search
    monkeypatch.setattr(file_search.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Background Explorer/Notepad"))
    monkeypatch.setattr(file_search.os, "startfile", lambda *_a, **_k: pytest.fail("Background ShellExecute"), raising=False)
    with policy.background_scope("late-worker"):
        problem = getattr(file_search, method)("synthetic-does-not-exist.txt")
    assert "foreground" in problem


@pytest.mark.parametrize("action", ["launch", "open", "fill", "read", "screenshot"])
async def test_generic_browser_dispatch_never_uses_a_headless_flag_as_an_ownership_grant(action):
    from orion_core.dispatch_web import WebDispatchMixin
    # No copilot field is supplied: policy must refuse before resolving one.
    with policy.background_scope("ordinary-tool"):
        result = await WebDispatchMixin().web_automation_tool({"action": action, "headless": True})
    assert not result.ok and result.receipt["status"] == "requires_foreground"


async def test_generic_browser_direct_sink_cannot_attach_or_send_cdp_after_scope_revocation():
    from orion_core.browser_copilot import BrowserCopilot
    browser = object.__new__(BrowserCopilot)
    with policy.background_scope("job") as scope:
        scope.revoke()
        result = await browser.ensure(background=True)
        assert result.receipt["status"] == "requires_foreground"
        assert await browser._connect_debugger() is False
        with pytest.raises(RuntimeError, match="foreground"):
            await browser._cdp("Input.dispatchKeyEvent", {"type": "keyDown"})


async def test_shared_browser_and_social_thread_bridges_refuse_before_accessing_services():
    from orion_core.web import WebController
    from orion_core.social_automation import SocialAutomationService
    web = object.__new__(WebController)
    social = object.__new__(SocialAutomationService)
    called = []
    async def action():
        called.append(True)
    coroutine = action()
    with policy.background_scope("plugin_schedule"):
        assert (await web._input("send_hotkeys", "ctrl+l")).receipt["status"] == "requires_foreground"
        assert await web._focus_browser() is False
        assert (await web.open("https://example.invalid")).receipt["status"] == "requires_foreground"
        result = await social._run_pw(coroutine)
        assert (await social._ensure_context()).receipt["status"] == "requires_foreground"
    assert not result.ok and not called and coroutine.cr_frame is None


def test_artifact_game_media_and_host_command_sinks_are_covered(monkeypatch):
    from orion_core import camera_tools, gaming, media_search, peripherals
    monkeypatch.setattr(camera_tools.os, "startfile", lambda *_a: pytest.fail("Background artifact launch"), raising=False)
    game = object.__new__(gaming.GamingClientService)
    host = object.__new__(peripherals.PeripheralController)
    with policy.background_scope("job"):
        assert camera_tools.open_file(Path("synthetic.png")) is False
        assert game.launch_app("123").receipt["status"] == "requires_foreground"
        assert host._run_shell("synthetic.exe").receipt["status"] == "requires_foreground"
        assert host.screen_off().receipt["status"] == "requires_foreground"
        assert media_search.play("Synthetic music").receipt["status"] == "requires_foreground"
        assert media_search._play_spotify("Synthetic", prefer_library=True).receipt["status"] == "requires_foreground"
        assert media_search._invoke(SimpleNamespace()) is False
