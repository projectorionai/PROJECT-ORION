from pathlib import Path

from orion_core import app_resolver, constants, providers


def _router(monkeypatch, tmp_path, folders):
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path / "installed-program")
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "user-data")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(app_resolver, "known_folder", lambda name, **kw: folders.get(name))
    return object.__new__(providers.ProviderRouter)


def test_prompt_exposes_current_and_legacy_research_locations(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, {})
    line = router._environment_line()
    assert str(constants.CONFIG_DIR / "research") in line
    assert str(constants.BASE_DIR / "research") in line
    assert "legacy" in line.lower()
    assert "saved" in line.lower() and "find_files" in line


def test_prompt_uses_redirected_shell_folders(monkeypatch, tmp_path):
    redirected = {name: tmp_path / "OneDrive" / name for name in ("desktop", "documents", "downloads")}
    router = _router(monkeypatch, tmp_path, redirected)
    line = router._environment_line()
    assert all(str(path) in line for path in redirected.values())
    assert str(tmp_path / "home" / "Desktop") not in line


def test_unresolved_shell_locations_are_not_invented(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, {})
    line = router._environment_line()
    for folder in ("Desktop", "Documents", "Downloads"):
        assert str(tmp_path / "home" / folder) not in line


def test_resolved_locations_are_reused_for_later_prompts(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, {})
    calls = []
    monkeypatch.setattr(app_resolver, "known_folder", lambda name, **kw: calls.append(name))
    first = router._environment_line()
    assert first and len(calls) == 3
    assert router._environment_line() == first
    assert len(calls) == 3
