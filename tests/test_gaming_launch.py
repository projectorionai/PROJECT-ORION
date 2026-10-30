"""Gaming actions must reach a launcher and read real client manifests."""

from __future__ import annotations

from types import SimpleNamespace

from orion_core.gaming import GamingClientService
from orion_core.jv_plugins import legacy_game_updater as clients


def test_index_status_and_launch_use_installed_manifest(tmp_path, monkeypatch):
    root = tmp_path / "Steam"
    apps = root / "steamapps"
    apps.mkdir(parents=True)
    executable = root / "steam.exe"
    executable.write_bytes(b"not executed")
    (apps / "appmanifest_123.acf").write_text(
        '"AppState" {"appid" "123" "name" "Test Game" "StateFlags" "4" "SizeOnDisk" "1000"}', encoding="utf-8")
    monkeypatch.setattr(clients, "_find_steam_path", lambda: root)
    monkeypatch.setattr(clients, "_epic_games", list)
    launches = []
    monkeypatch.setattr("orion_core.gaming.subprocess.Popen", lambda command, **kwargs: launches.append((command, kwargs)))
    log = SimpleNamespace(emit=lambda *_: None)
    service = GamingClientService(SimpleNamespace(log=log))
    service._platform = "windows"
    indexed = service.index_installs()
    assert indexed.ok and "Test Game (AppID 123)" in indexed.text
    assert indexed.evidence == [{"platform": "steam", "id": "123", "name": "Test Game"}]
    assert service.update_status().ok
    result = service.launch_app("123")
    assert result.ok and "launch request" in result.text
    assert launches[0][0] == [str(executable), "steam://rungameid/123"]
    assert not service.launch_app("456").ok
    assert not service.launch_app("123 & calc.exe").ok
    assert len(launches) == 1


def test_missing_launcher_is_not_reported_as_started(tmp_path, monkeypatch):
    monkeypatch.setattr(clients, "_find_steam_path", lambda: tmp_path)
    monkeypatch.setattr(clients, "_games", lambda *_: [{"id": "123", "name": "Test Game"}])
    service = GamingClientService(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    service._platform = "windows"
    outcome = service.launch_app("123")
    assert not outcome.ok and "executable could not be found" in outcome.text


def test_malformed_settings_never_enable_licence_or_power_actions(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("Broken settings")
    monkeypatch.setattr("orion_core.jv_plugins.runtime.get_plugin_setting", fail)
    for key in ("claim_free", "accept_eula", "shutdown_when_done"):
        assert clients._setting_bool(key, True) is False


def test_frozen_schedule_uses_durable_entry_point_and_interactive_user(monkeypatch):
    calls = []
    monkeypatch.setattr(clients.sys, "frozen", True, raising=False)
    monkeypatch.setattr(clients.sys, "executable", r"C:\ORION\ORION.exe")

    def fake_process(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(clients, "_run_process", fake_process)
    result = clients._schedule_windows(3, 15)
    command = calls[-1]
    assert "scheduled for 03:15" in result
    assert command[command.index("/TR") + 1] == '"C:\\ORION\\ORION.exe" --game-updater-scheduled'
    assert "/IT" in command and "SYSTEM" not in command and "HIGHEST" not in command
