"""Verification retirement, output limits and settings consistency."""
from types import SimpleNamespace

import pytest

from orion_core.sandbox import SandboxVerificationHarness


BUS = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
SOURCE = "def get_tool_schema():\n    return {'name': 'demo', 'parameters': {'type': 'object', 'properties': {}}}\ndef run():\n    return 'ok'\n"


@pytest.mark.parametrize("value", [True, False, 0, -1, float("inf"), float("nan"), 91, "invalid"])
async def test_bad_timeout_refused_before_any_process(tmp_path, monkeypatch, value):
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    harness._run_script = lambda *_: pytest.fail("bad time budgets must not launch a verifier")
    result = await harness.verify_tool(SOURCE, "pass", "demo", value)
    assert not result.passed and not list(tmp_path.iterdir())


def test_fast_child_output_over_budget_cannot_pass(tmp_path, monkeypatch):
    from orion_core import sandbox
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    monkeypatch.setattr(sandbox, "MAX_OUTPUT_BYTES", 1024)
    monkeypatch.setattr(sandbox, "_MEMORY_POLL_INTERVAL_S", 5)
    path = tmp_path / "bounded.py"
    path.write_text("print('x' * 2048)\n", encoding="utf-8")
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    result = harness._run_script(path.name, 5)
    assert result[2] != 0 and "output limit" in result[1]


async def test_memory_failure_is_never_a_pass_even_with_zero_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    harness._run_script = lambda *_: ("", "", 0, False, True)
    result = await harness.verify_tool(SOURCE, "pass", "demo")
    assert not result.passed and result.memory_exceeded


async def test_one_attempt_keeps_container_mode_when_environment_changes(tmp_path, monkeypatch):
    from orion_core import sandbox
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "1")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "container")
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    monkeypatch.setenv("ORION_FORGE_CONTAINER_IMAGE", "orion-forge-verifier:local")
    commands = []
    class CompletedProcess:
        returncode = 0
        pid = -1
        def wait(self, **_):
            return 0
    def spawn(command, **kwargs):
        commands.append(command)
        monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
        monkeypatch.setenv("ORION_FORGE_CONTAINER_IMAGE", "untrusted:changed")
        if "/work/_orion_conformance.py" in command:
            kwargs["stdout"].write('{"name": "demo", "description": "Demo tool", "parameters": {"type": "object", "properties": {}}}\nconformance OK\n')
            kwargs["stdout"].flush()
        return CompletedProcess()
    monkeypatch.setattr(sandbox.subprocess, "Popen", spawn)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""))
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    result = await harness.verify_tool(SOURCE, "assert True", "demo")
    assert result.passed and result.typecheck == "passed" and result.isolation == "container"
    assert len(commands) == 3 and all(command[:2] == ["docker", "run"] for command in commands)
    assert all("orion-forge-verifier:local" in command for command in commands)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error", ["cannot connect to daemon", "permission denied"])
def test_unconfirmed_container_cleanup_surfaces_failure(tmp_path, monkeypatch, error):
    from orion_core import sandbox
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "container")
    class CompletedProcess:
        returncode = 0
        pid = -1
        def wait(self, **_):
            return 0
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess())
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr=error))
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    result = harness._run_script("demo.py", 5)
    assert result[2] != 0 and "cleanup failed" in result[1]


def test_already_removed_container_is_successful_retirement(tmp_path, monkeypatch):
    from orion_core import sandbox
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "container")
    class CompletedProcess:
        returncode = 0
        pid = -1
        def wait(self, **_):
            return 0
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess())
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="Error response from daemon: No such container: demo"))
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    assert harness._run_script("demo.py", 5)[2] == 0


# ── the installed app can verify (forge_runtime, 10 Oct) ─────────────────────

def test_a_harness_that_never_ran_is_not_sent_for_repair():
    """15 Forge sessions in the installed app failed with "A frozen ORION
    executable is not a Python verifier", classified unknown -> repair both:
    four paid "repairs" each of code that had never been executed."""
    from orion_core.forge_diagnosis import FailureClass, RepairTarget, diagnose
    found = diagnose(["Verifier failed: RuntimeError: A frozen ORION executable is not a "
                      "Python verifier."], "disk_space_analyzer")
    assert found.failure_class is FailureClass.VERIFIER_UNAVAILABLE
    assert found.target is RepairTarget.NONE and "not a Python verifier" in found.detail
    import inspect
    from orion_core import forge
    loop = inspect.getsource(forge)
    assert loop.index("diagnosis.target is RepairTarget.NONE") < loop.index(
        "if not await self._apply_repair(session, diagnosis, error_log):")


def test_orion_exe_verifies_with_its_own_interpreter(tmp_path, monkeypatch):
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path
    from orion_core import forge_runtime
    script = tmp_path / "check.py"
    script.write_text("import sys\nprint('args', sys.argv[1:])\nraise SystemExit(3)\n", encoding="utf-8")
    root = _Path(__file__).resolve().parents[1]
    done = subprocess.run([_sys.executable, str(root / "orion.py"), "--forge-verify", str(script), "a", "b"],
                          capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert done.returncode == 3 and "args ['a', 'b']" in done.stdout
    broken = tmp_path / "broken.py"
    broken.write_text("raise ValueError('boom')\n", encoding="utf-8")
    assert forge_runtime.run_script([str(broken)]) == 1
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    assert forge_runtime.verifier_command("s.py", ["x"])[1:] == ["--forge-verify", "s.py", "x"]
    monkeypatch.setattr(forge_runtime, "external_python", lambda: None)
    assert forge_runtime.pip_install_command(["psutil"]) is None
    monkeypatch.setattr(forge_runtime, "external_python", lambda: "C:/Python313/python.exe")
    command = forge_runtime.pip_install_command(["psutil"])
    assert "--target" in command and f"{_sys.version_info[0]}.{_sys.version_info[1]}" in command
