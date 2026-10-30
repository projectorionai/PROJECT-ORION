from types import SimpleNamespace

import sys

import pytest

from orion_core.forge_isolation import container_command
from orion_core.sandbox import SandboxVerificationHarness


def test_container_command_has_no_host_credentials_network_or_gpu(tmp_path):
    cmd = container_command(tmp_path, "demo_test.py", (), image="orion-forge-verifier:local",
                            name="orion-forge-" + "a" * 32)
    for flag in ["--network=none", "--read-only", "--cap-drop=ALL", "--pull=never",
                 "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=512m",
                 "--memory-swap=512m", "--cpus=1", "--user=65534:65534"]:
        assert flag in cmd
    assert cmd[cmd.index("--mount") + 1].endswith("target=/work,readonly")
    assert "--gpus" not in cmd and "--privileged" not in cmd


@pytest.mark.parametrize("image", ["--privileged", "x y", "x;whoami", "x\n--gpus=all"])
def test_invalid_container_image_cannot_add_options(tmp_path, image):
    with pytest.raises(ValueError):
        container_command(tmp_path, "demo.py", (), image=image, name="orion-forge-" + "a" * 32)


async def test_required_typechecker_fails_closed_when_missing(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "required")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: None if n == "mypy" else original(n))
    harness = SandboxVerificationHarness(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    harness.staging_dir = tmp_path
    result = await harness.verify_tool("pass", "pass", "demo")
    assert not result.passed and result.stage == "typecheck"
    assert result.typecheck == "unavailable"


async def test_container_failure_never_falls_back_to_host(tmp_path, monkeypatch):
    import orion_core.sandbox as sandbox
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "container")
    commands = []
    def absent(command, **kwargs):
        commands.append(command)
        raise FileNotFoundError("docker is not installed")
    monkeypatch.setattr(sandbox.subprocess, "Popen", absent)
    harness = SandboxVerificationHarness(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    harness.staging_dir = tmp_path
    result = await harness.verify_tool("pass", "pass", "demo")
    assert not result.passed
    assert result.isolation == "container"
    assert commands and all(c[0] == "docker" for c in commands)


async def test_frozen_executable_is_never_spawned_as_python(tmp_path, monkeypatch):
    from orion_core import sandbox
    monkeypatch.setattr(sandbox.sys, "frozen", True, raising=False)
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    calls = []
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    harness = SandboxVerificationHarness(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    harness.staging_dir = tmp_path
    result = await harness.verify_tool("pass", "pass", "demo")
    # Never as a bare interpreter (ORION.exe script.py starts a second ORION):
    # only through its explicit verifier mode (forge_runtime, 10 Oct - every
    # installed-app Forge session had failed while this refused outright).
    assert calls, "the installed app could not verify anything"
    for (command, *_rest) in calls:
        assert command[0] == sys.executable and command[1] == "--forge-verify", command
