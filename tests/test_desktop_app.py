"""
ORION as a real desktop application.

  "I want ORION to be able to be in the background apps in the taskbar, also can
   we make ORION a full on application that I can put in my taskbar and the
   desktop since he's that good now"

Three things turn a script into an app Windows treats as its own: a stable
taskbar identity (AppUserModelID), an icon, and shortcuts to launch from. The
identity in particular has a trap — it must never change once shortcuts exist,
or a pinned taskbar button goes dead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import desktop_app  # noqa: E402


# ── the icon ─────────────────────────────────────────────────────────────────

def test_the_icon_is_generated_at_multiple_sizes():
    """Windows picks 16px for the taskbar and 256px for large icons; a single
    size scaled to both looks wrong at one end."""
    pytest.importorskip("PIL")
    icon = desktop_app.ensure_icon()
    assert icon is not None and icon.exists()
    from PIL import Image
    with Image.open(icon) as im:
        sizes = set(im.ico.sizes()) if hasattr(im, "ico") else set()
    assert (16, 16) in sizes
    assert (256, 256) in sizes


def test_the_icon_is_only_generated_once(monkeypatch):
    """It must not be redrawn on every launch."""
    pytest.importorskip("PIL")
    icon = desktop_app.ensure_icon()
    assert icon is not None
    calls = {"n": 0}
    real = desktop_app._render_icon

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(desktop_app, "_render_icon", _counting)
    desktop_app.ensure_icon()          # already exists → should not render
    assert calls["n"] == 0


# ── the identity ─────────────────────────────────────────────────────────────

def test_the_app_id_is_stable_and_versionless():
    """It must never change once shortcuts exist — a version in it would break
    the taskbar pin on every release."""
    assert desktop_app.APP_USER_MODEL_ID
    assert "Mark" not in desktop_app.APP_USER_MODEL_ID
    assert not any(ch.isdigit() for ch in desktop_app.APP_USER_MODEL_ID)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows taskbar identity")
def test_setting_the_identity_succeeds():
    assert desktop_app.set_app_user_model_id() is True


def test_setting_the_identity_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(desktop_app.sys, "platform", "linux")
    assert desktop_app.set_app_user_model_id() is False


# ── the launch target ────────────────────────────────────────────────────────

def test_the_launch_target_prefers_the_exe_when_built():
    """A real ORION.exe is what Windows pins cleanly, so it wins when present."""
    target, args, _workdir = desktop_app._launch_target()
    if desktop_app.EXE_PATH.exists() or desktop_app.STANDALONE_EXE.exists():
        assert Path(target).name.lower() == "orion.exe"
        assert args == ""
    else:
        # Fallback: windowless interpreter + orion.py.
        assert "orion.py" in args
        if sys.platform == "win32" and "python" in Path(sys.executable).name.lower():
            twin = Path(sys.executable).with_name(
                Path(sys.executable).name.replace("python", "pythonw"))
            if twin.exists():
                assert "pythonw" in Path(target).name


def test_the_fallback_launch_target_points_at_the_real_entry_point(monkeypatch, tmp_path):
    """With no exe (neither the standalone build nor the launcher), it must
    still launch orion.py windowless."""
    monkeypatch.setattr(desktop_app, "STANDALONE_EXE", tmp_path / "nope-standalone.exe")
    monkeypatch.setattr(desktop_app, "EXE_PATH", tmp_path / "nope-ORION.exe")
    target, args, _workdir = desktop_app._launch_target()
    assert "orion.py" in args
    assert Path(target).name.lower() in {"pythonw.exe", "python.exe",
                                         Path(sys.executable).name.lower()}


def test_the_launcher_source_resolves_the_project_root():
    """launcher/orion_app.py must find orion.py and a windowless interpreter."""
    import sys as _sys
    launcher_dir = Path(__file__).resolve().parents[1] / "launcher"
    _sys.path.insert(0, str(launcher_dir))
    try:
        import orion_app
    finally:
        _sys.path.remove(str(launcher_dir))
    base = orion_app._base_dir()
    assert (base / "orion.py").is_file()
    assert "python" in orion_app._interpreter(base).lower()


# ── shortcuts ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcuts")
def test_shortcuts_can_be_created(tmp_path, monkeypatch):
    pytest.importorskip("win32com.client")
    monkeypatch.setattr(desktop_app, "_desktop_dir", lambda: tmp_path)
    monkeypatch.setattr(desktop_app, "_start_menu_dir", lambda: tmp_path / "sm")
    result = desktop_app.install_shortcuts()
    assert result.get("desktop") is True
    assert (tmp_path / "ORION.lnk").exists()
    assert (tmp_path / "sm" / "ORION.lnk").exists()


def test_install_off_windows_is_reported_not_crashed(monkeypatch):
    monkeypatch.setattr(desktop_app.sys, "platform", "linux")
    result = desktop_app.install_shortcuts()
    assert result == {"supported": False}


def test_status_describes_what_exists():
    st = desktop_app.status()
    assert "supported" in st and "icon" in st
    assert isinstance(desktop_app.describe_status(), str)


# ── wiring ───────────────────────────────────────────────────────────────────

def test_startup_sets_the_identity_before_the_qapplication():
    """AppUserModelID must be set BEFORE any window, or the taskbar grouping
    is already wrong by the time it runs."""
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")
    id_at = source.index("desktop_app.set_app_user_model_id()")
    app_at = source.index("app = QApplication(sys.argv)")
    assert id_at < app_at


def test_startup_applies_the_window_icon():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")
    assert "desktop_app.apply_window_icon(app)" in source


def test_the_tool_can_install_the_app():
    import inspect

    from orion_core.dispatch_desktop import DesktopDispatchMixin

    source = inspect.getsource(DesktopDispatchMixin.system_startup_tool)
    assert "install_shortcuts" in source
    assert "install_app" in source
