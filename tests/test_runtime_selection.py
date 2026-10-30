from pathlib import Path
import json
import os

from orion_core import runtime_selection as selection


def test_runtime_skips_broken_environment_and_uses_working_source_runtime(tmp_path, monkeypatch):
    attempted = []
    def check(path):
        attempted.append(path)
        return ".neural-debug-env" in path.parts and path.name == "python.exe"
    monkeypatch.delenv("ORION_PYTHON", raising=False)
    monkeypatch.setattr(selection, "interpreter_can_start", check)
    result = selection.choose_interpreter(tmp_path)
    assert Path(result) == tmp_path / ".neural-debug-env" / "Scripts" / "python.exe"
    assert not any(".venv" in path.parts for path in attempted)


def test_new_source_prevents_old_bundle_from_hiding_fixes(tmp_path):
    executable = tmp_path / "ORION.exe"
    executable.write_bytes(b"bundle")
    source = tmp_path / "orion_core" / "remote.py"
    source.parent.mkdir()
    source.write_text("# Updated gateway", encoding="utf-8")
    os.utime(executable, ns=(100, 100))
    os.utime(source, ns=(200, 200))
    assert not selection.bundle_is_current(tmp_path, executable)
    os.utime(executable, ns=(300, 300))
    assert selection.bundle_is_current(tmp_path, executable)


def test_remote_preferences_keep_binding_on_loopback_and_preserve_explicit_env(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "remote_launch.json").write_text(json.dumps({
        "enabled": True, "exposure": "quick-tunnel", "host": "0.0.0.0", "port": 80}), encoding="utf-8")
    for key in ("ORION_REMOTE_HOST", "ORION_REMOTE_PORT", "ORION_REMOTE_TLS", "ORION_REMOTE_EXPOSURE"):
        monkeypatch.delenv(key, raising=False)
    env = selection.launch_environment(tmp_path)
    assert env["ORION_REMOTE_HOST"] == "127.0.0.1" and env["ORION_REMOTE_PORT"] == "8000"
    assert env["ORION_REMOTE_TLS"] == "0" and env["PYTHONUTF8"] == "1"
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "serve")
    assert selection.launch_environment(tmp_path)["ORION_REMOTE_EXPOSURE"] == "serve"
