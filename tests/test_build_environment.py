"""A toolchain's incompatible ICU must not shadow Windows' ICU in Qt builds."""
import os
import shutil
import sys

import pytest

import build_standalone


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL resolution")
def test_windows_dll_resolution_prefers_system_icu_over_poppler(tmp_path, monkeypatch):
    bindepend = pytest.importorskip("PyInstaller.depend.bindepend")
    system_root = tmp_path / "Windows"
    system32 = system_root / "System32"
    poppler = tmp_path / "poppler"
    system32.mkdir(parents=True)
    poppler.mkdir()
    # The resolver validates the PE architecture before accepting a file.
    # Reuse a local PE without loading either synthetic DLL.
    shutil.copyfile(sys.executable, system32 / "icuuc.dll")
    shutil.copyfile(sys.executable, poppler / "icuuc.dll")
    monkeypatch.setattr(build_standalone.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(system_root))
    monkeypatch.setenv("PATH", str(poppler))
    assert bindepend.resolve_library_path("icuuc.dll") == str(poppler / "icuuc.dll")
    environment = build_standalone._build_environment()
    assert os.environ["PATH"] == str(poppler)
    monkeypatch.setenv("PATH", environment["PATH"])
    assert bindepend.resolve_library_path("icuuc.dll") == str(system32 / "icuuc.dll")


def test_non_windows_build_preserves_the_environment(monkeypatch):
    monkeypatch.setattr(build_standalone.sys, "platform", "linux")
    monkeypatch.setenv("PATH", "toolchain-path")
    assert build_standalone._build_environment()["PATH"] == "toolchain-path"


@pytest.fixture
def mocked_standalone_build(tmp_path, monkeypatch):
    """Exercise build orchestration without freezing or launching anything."""
    from types import SimpleNamespace

    entry = tmp_path / "orion.py"
    entry.write_text("# synthetic entry\n", encoding="utf-8")
    exe = tmp_path / "dist" / "ORION" / "ORION.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"synthetic build output")
    monkeypatch.setattr(build_standalone, "BASE", tmp_path)
    monkeypatch.setattr(build_standalone, "ENTRY", entry)
    monkeypatch.setattr(build_standalone, "ICON", tmp_path / "absent.ico")
    monkeypatch.setattr(build_standalone, "_has_private_config", lambda _: False)
    monkeypatch.setattr(build_standalone, "_ensure_pyinstaller", lambda: True)
    monkeypatch.setattr(build_standalone, "_ensure_icon", lambda: None)
    monkeypatch.setattr(build_standalone, "_write_version_file", lambda: None)
    monkeypatch.setattr(build_standalone, "_write_build_info", lambda: None)
    monkeypatch.setattr(build_standalone, "_ssl_runtime_inputs", lambda: ([], []))
    monkeypatch.setattr(build_standalone, "_verify_ssl_runtime_provenance", lambda *_args: {})
    events = []
    monkeypatch.setattr(build_standalone, "_prune_unused_qt_data",
                        lambda _: events.append("prune"))
    monkeypatch.setattr(build_standalone, "_seed_config",
                        lambda _: events.append("seed"))

    def run(command, **kwargs):
        events.append((list(command), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build_standalone.subprocess, "run", run)
    from tools import check_installed_build as parity

    def check(command, **kwargs):
        events.append((list(command), kwargs))
        return parity.RunResult(0, "ALL OK\n", "", 0.01)

    monkeypatch.setattr(parity, "run_bounded", check)
    return exe, events


def test_standalone_build_checks_parity_after_all_packaging_steps(mocked_standalone_build):
    exe, events = mocked_standalone_build
    assert build_standalone.build() == 0
    assert events[1:3] == ["prune", "seed"]
    assert len(events) == 4, "a successful freeze must still run the parity gate"
    command, kwargs = events[-1]
    assert command == [sys.executable, str(build_standalone.BASE / "tools" /
                                         "check_installed_build.py"),
                       str(exe), "--source", str(build_standalone.BASE),
                       "--json", str(exe.parent / "parity_report.json")]
    assert kwargs["cwd"] == build_standalone.BASE
    assert kwargs["timeout"] > 0


def test_standalone_build_bundles_the_verified_offline_voice(mocked_standalone_build, tmp_path,
                                                            monkeypatch):
    """The Kokoro model (354 MB, not in the repository) goes into
    _internal/assets/models/kokoro-v1.0 when its pinned hashes match."""
    import hashlib

    from orion_core import voice_kokoro

    _exe, events = mocked_standalone_build
    folder = tmp_path / "voice"
    folder.mkdir()
    files = {}
    for name, body in ((voice_kokoro.MODEL_FILE, b"model"), (voice_kokoro.VOICES_FILE, b"voices")):
        (folder / name).write_bytes(body)
        files[name] = voice_kokoro.ModelFile(len(body), hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(voice_kokoro, "MODEL",
                        voice_kokoro.ModelSpec("kokoro-v1.0", "https://example.invalid", files))
    assert build_standalone.build(["--kokoro-dir", str(folder)]) == 0
    command = events[0][0]
    sep = ";" if sys.platform == "win32" else ":"
    for name in files:
        assert f"{folder / name}{sep}assets/models/kokoro-v1.0" in command
    for package in ("kokoro_onnx", "espeakng_loader", "phonemizer"):
        assert command[command.index(package) - 1] == "--collect-all"


@pytest.mark.parametrize("outcome", [7, "start-error", "timeout"])
def test_standalone_build_cannot_succeed_when_parity_fails(
        mocked_standalone_build, monkeypatch, capsys, outcome):
    from types import SimpleNamespace

    from tools import check_installed_build as parity

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    def check(command, **kwargs):
        calls.append(command)
        if outcome == "start-error":
            return parity.RunResult(None, "", "", 0.0, error="synthetic launch refusal")
        if outcome == "timeout":
            return parity.RunResult(-9, "", "", kwargs["timeout"], timed_out=True)
        return parity.RunResult(outcome, "FAILED: 1\n", "", 0.01)

    monkeypatch.setattr(build_standalone.subprocess, "run", run)
    monkeypatch.setattr(parity, "run_bounded", check)
    assert build_standalone.build() != 0
    assert len(calls) == 2
    assert "parity" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("report,reason", [
    ({"exited_on_eof": False, "returncode": -9}, "stdin"),
    ({"exited_on_eof": True, "returncode": 1}, "code 1"),
])
def test_mcp_parity_rejects_a_successful_handshake_with_failed_shutdown(
        monkeypatch, report, reason):
    from tools import check_installed_build as parity

    monkeypatch.setattr(parity, "mcp_handshake", lambda *a, **k: {
        "ok": True, "tools": 4, "error": "", "protocol": "2024-11-05", **report})
    check = parity.check_mcp("orion_world", ["not-launched"], "mock host",
                             env={}, protocol="2024-11-05",
                             supported=frozenset({"2024-11-05"}), timeout=60)
    assert not check.ok
    assert reason in check.detail


def test_mcp_parity_keeps_a_clean_handshake_and_shutdown_green(monkeypatch):
    from tools import check_installed_build as parity

    monkeypatch.setattr(parity, "mcp_handshake", lambda *a, **k: {
        "ok": True, "tools": 4, "error": "", "protocol": "2024-11-05",
        "exited_on_eof": True, "returncode": 0})
    assert parity.check_mcp("orion_world", ["not-launched"], "mock host",
                            env={}, protocol="2024-11-05",
                            supported=frozenset({"2024-11-05"}), timeout=60).ok


def test_mcp_parity_fails_fast_on_a_missing_world_feed_and_reaps_the_child(monkeypatch):
    """Model the old windowed exe's traceback + modal wait, without an exe/UI."""
    import io
    import subprocess
    import threading
    import time

    from tools import check_installed_build as parity

    stopped = threading.Event()

    class SilentStdout:
        def __iter__(self):
            stopped.wait(5)
            return iter(())

    class CrashedProcess:
        stdin = io.BytesIO()
        stdout = SilentStdout()
        stderr = io.BytesIO(
            b'Traceback (most recent call last):\n'
            b'  File "orion_world.py", line 7, in <module>\n'
            b'    import orion_core.world_feeds\n'
            b"ModuleNotFoundError: No module named 'orion_core.world_feeds'\n")
        returncode = None
        pid = 123

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(["not-launched"], timeout)

    process = CrashedProcess()
    killed = []
    monkeypatch.setattr(parity.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(parity, "CRASH_GRACE_S", 0.02)

    def kill(proc):
        assert proc is process
        killed.append(proc)
        proc.returncode = -9
        stopped.set()

    monkeypatch.setattr(parity, "kill_tree", kill)
    started = time.monotonic()
    try:
        report = parity.mcp_handshake(["not-launched", "--mcp-server", "orion_world"],
                                      env={}, timeout=60.0)
    finally:
        stopped.set()
    assert time.monotonic() - started < 2.0
    assert not report["ok"]
    assert "crashed during startup" in report["error"]
    assert "ModuleNotFoundError" in report["stderr_tail"]
    assert "orion_core.world_feeds" in report["error"]
    assert report["exited_on_eof"] is False
    assert report["returncode"] == -9
    assert killed == [process]


def test_mcp_parity_rejects_a_non_object_result_without_leaking_a_child(monkeypatch):
    import io

    from tools import check_installed_build as parity

    class Process:
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'{"jsonrpc":"2.0","id":1,"result":["malformed"]}\n')
        stderr = io.BytesIO()
        returncode = None

        def wait(self, timeout):
            self.returncode = 0
            return 0

    process = Process()
    monkeypatch.setattr(parity.subprocess, "Popen", lambda *a, **k: process)
    report = parity.mcp_handshake(["not-launched"], env={}, timeout=1.0)
    assert not report["ok"]
    assert "result is not an object" in report["error"]
    assert process.stdin.closed
    assert process.returncode == 0


def test_parity_never_launches_an_archive_without_the_requested_headless_modes(
        tmp_path, monkeypatch):
    from tools import check_installed_build as parity

    exe = tmp_path / "old.exe"
    exe.write_bytes(b"synthetic archive")

    def forbidden(*args, **kwargs):
        pytest.fail("missing headless flags must never fall through to the desktop app")

    monkeypatch.setattr(parity.subprocess, "Popen", forbidden)
    checks, _ = parity.run_checks(
        exe, parity.ROOT, groups=("mcp", "face", "forge", "pip"),
        archive=parity.Archive(set(), set(), set()), work_root=tmp_path, keep=True)
    assert checks
    assert all(not check.ok for check in checks)
    assert {"mcp:orion_world", "face-host", "forge:conformance", "forge:pip"} <= {
        check.name for check in checks}


def test_parity_timeout_reaps_its_own_entire_process_tree(monkeypatch):
    import subprocess

    from tools import check_installed_build as parity

    class Process:
        returncode = None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(["not-launched"], timeout)

    process = Process()
    killed = []

    def kill(proc):
        assert proc is process
        killed.append(proc)
        proc.returncode = -9

    monkeypatch.setattr(parity, "kill_tree", kill)
    result = parity.run_bounded(["not-launched"], timeout=0.01,
                                popen=lambda *a, **k: process)
    assert result.timed_out
    assert result.returncode == -9
    assert killed == [process]


def _synthetic_ssl_runtime(tmp_path):
    from pathlib import Path
    lib = tmp_path / "python" / "Lib"
    dlls = tmp_path / "python" / "DLLs"
    lib.mkdir(parents=True)
    dlls.mkdir()
    ssl_file = lib / "ssl.py"
    ssl_file.write_text(
        "import os\n"
        "def context():\n"
        "    return os.path.join(os.path.dirname(__file__), 'runtime-trust.pem')\n",
        encoding="utf-8")
    (lib / "runtime-trust.pem").write_bytes(b"Synthetic runtime trust data")
    extension = dlls / "_ssl.pyd"
    extension.write_bytes(b"Synthetic extension, never loaded")
    for name in ("libssl-3-x64.dll", "libcrypto-3-x64.dll"):
        (dlls / name).write_bytes(("Synthetic interpreter " + name).encode())
    return ssl_file, extension


def test_ssl_runtime_pins_interpreter_dlls_and_required_adjacent_pem(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    ssl_file, extension = _synthetic_ssl_runtime(tmp_path)
    monkeypatch.setattr(build_standalone, "sys", SimpleNamespace(platform="win32"))
    binaries, datas = build_standalone._ssl_runtime_inputs(ssl_file=ssl_file, extension_file=extension)
    assert {path.name for path, _ in binaries} == {"libssl-3-x64.dll", "libcrypto-3-x64.dll"}
    assert all(path.parent == extension.parent and destination == "." for path, destination in binaries)
    assert datas == [(ssl_file.with_name("runtime-trust.pem"), ".")]


def test_non_windows_ssl_runtime_keeps_required_pem_without_windows_dlls(tmp_path, monkeypatch):
    from types import SimpleNamespace
    ssl_file, extension = _synthetic_ssl_runtime(tmp_path)
    monkeypatch.setattr(build_standalone, "sys", SimpleNamespace(platform="linux"))
    binaries, datas = build_standalone._ssl_runtime_inputs(ssl_file=ssl_file, extension_file=extension)
    assert binaries == [] and datas == [(ssl_file.with_name("runtime-trust.pem"), ".")]


@pytest.mark.parametrize("missing", ["runtime-trust.pem", "libssl-3-x64.dll", "libcrypto-3-x64.dll"])
def test_build_refuses_missing_required_ssl_runtime_inputs(tmp_path, monkeypatch, missing):
    from types import SimpleNamespace
    ssl_file, extension = _synthetic_ssl_runtime(tmp_path)
    target = ssl_file.parent / missing if missing.endswith(".pem") else extension.parent / missing
    target.unlink()
    monkeypatch.setattr(build_standalone, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(RuntimeError, match="missing|Expected one"):
        build_standalone._ssl_runtime_inputs(ssl_file=ssl_file, extension_file=extension)


def test_runtime_ssl_provenance_rejects_poppler_even_when_output_name_exists(tmp_path, monkeypatch):
    from pathlib import Path
    ssl_file, extension = _synthetic_ssl_runtime(tmp_path)
    source = extension.parent / "libssl-3-x64.dll"
    wrong = tmp_path / "poppler" / source.name
    wrong.parent.mkdir()
    wrong.write_bytes(b"Wrong native toolchain DLL")
    internal = tmp_path / "output"
    internal.mkdir()
    (internal / source.name).write_bytes(wrong.read_bytes())
    toc = tmp_path / "build" / "_standalone_work" / "ORION" / "Analysis-00.toc"
    toc.parent.mkdir(parents=True)
    toc.write_text(repr(([ (source.name, str(wrong), "BINARY") ],)), encoding="utf-8")
    monkeypatch.setattr(build_standalone, "BASE", tmp_path)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        build_standalone._verify_ssl_runtime_provenance(internal, [(source, ".")], [])


def test_runtime_ssl_provenance_preserves_pem_bytes_exactly(tmp_path, monkeypatch):
    ssl_file, _extension = _synthetic_ssl_runtime(tmp_path)
    pem = ssl_file.with_name("runtime-trust.pem")
    internal = tmp_path / "output"
    internal.mkdir()
    output = internal / pem.name
    output.write_bytes(pem.read_bytes())
    toc = tmp_path / "build" / "_standalone_work" / "ORION" / "Analysis-00.toc"
    toc.parent.mkdir(parents=True)
    toc.write_text(repr(([ (pem.name, str(pem), "DATA") ],)), encoding="utf-8")
    monkeypatch.setattr(build_standalone, "BASE", tmp_path)
    receipt = build_standalone._verify_ssl_runtime_provenance(internal, [], [(pem, ".")])
    assert receipt["ok"] and receipt["files"][0]["source_sha256"] == receipt["files"][0]["output_sha256"]
    output.write_bytes(b"Changed runtime trust policy")
    with pytest.raises(RuntimeError, match="trust data changed"):
        build_standalone._verify_ssl_runtime_provenance(internal, [], [(pem, ".")])


def test_build_supplies_explicit_ssl_inputs_before_analysis(mocked_standalone_build, monkeypatch, tmp_path):
    ssl_file, extension = _synthetic_ssl_runtime(tmp_path)
    binary = extension.parent / "libssl-3-x64.dll"
    pem = ssl_file.with_name("runtime-trust.pem")
    monkeypatch.setattr(build_standalone, "_ssl_runtime_inputs", lambda: ([(binary, ".")], [(pem, ".")]))
    _exe, events = mocked_standalone_build
    assert build_standalone.build(["--skip-parity-check"]) == 0
    command = events[0][0]
    sep = ";" if sys.platform == "win32" else ":"
    assert command[command.index(f"{binary}{sep}.") - 1] == "--add-binary"
    assert command[command.index(f"{pem}{sep}.") - 1] == "--add-data"


def test_build_cannot_finish_when_ssl_origin_verification_fails(mocked_standalone_build, monkeypatch):
    def fail(*_args):
        raise RuntimeError("Synthetic wrong DLL source")
    monkeypatch.setattr(build_standalone, "_verify_ssl_runtime_provenance", fail)
    _exe, events = mocked_standalone_build
    assert build_standalone.build(["--skip-parity-check"]) != 0
    assert len(events) == 1, "wrong SSL origin must fail before seeding or declaring a verified build"
