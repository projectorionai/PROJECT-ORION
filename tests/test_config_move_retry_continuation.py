"""Synthetic directory-publication locks must recover or preserve the pointer."""
import os
import uuid
from pathlib import Path

import pytest

from tools import move_config as module


@pytest.fixture(autouse=True)
def _private_instance_lock(monkeypatch):
    """Synthetic migration must neither inspect nor take the owner's lock."""
    from orion_core import single_instance

    tag = f"-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(single_instance, "MUTEX_NAME", single_instance.MUTEX_NAME + tag)
    monkeypatch.setattr(single_instance, "SOCKET_NAME", single_instance.SOCKET_NAME + tag)
    monkeypatch.setattr(module, "_orion_process_running", lambda: False)


@pytest.mark.parametrize("fault,expected_ok,expected_attempts", [
    ("temporary", True, 2), ("persistent", False, 5), ("other_io", False, 1)])
def test_configuration_publication_has_bounded_lock_recovery(tmp_path, monkeypatch, fault, expected_ok, expected_attempts):
    monkeypatch.delenv("ORION_CONFIG_DIR", raising=False)
    monkeypatch.setattr(module, "store_python", lambda: False)
    monkeypatch.setattr(module, "_visible_to_others", lambda *_: "")
    app = tmp_path / "app"
    source = app / "config"
    source.mkdir(parents=True)
    (source / "synthetic.txt").write_bytes(b"original synthetic configuration")
    target = tmp_path / "new"
    calls = []
    sleeps = []
    replace = module.os.replace
    def publish(src, dest):
        if Path(dest) == target:
            calls.append(Path(src))
            if fault == "persistent" or (fault == "temporary" and len(calls) == 1):
                raise PermissionError("synthetic Windows directory lock")
            if fault == "other_io":
                raise OSError("synthetic unrelated publication failure")
        return replace(src, dest)
    monkeypatch.setattr(module.os, "replace", publish)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    result = module.move(app, target)
    assert (result == 0) is expected_ok
    assert len(calls) == expected_attempts
    assert len(sleeps) == (expected_attempts - 1)
    assert (source / "synthetic.txt").read_bytes() == b"original synthetic configuration"
    pointer = app / module.CONFIG_POINTER_NAME
    assert pointer.exists() is expected_ok
    assert target.exists() is expected_ok
    if expected_ok:
        assert pointer.read_text(encoding="utf-8").strip() == str(target)
        assert (target / "synthetic.txt").read_bytes() == (source / "synthetic.txt").read_bytes()


def test_configuration_publication_still_refuses_a_running_instance(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_CONFIG_DIR", raising=False)
    app = tmp_path / "app"
    source = app / "config"
    source.mkdir(parents=True)
    (source / "synthetic.txt").write_bytes(b"unchanged")
    target = tmp_path / "new"
    monkeypatch.setattr(module, "_orion_process_running", lambda: True)
    assert module.move(app, target) != 0
    assert not target.exists()
    assert not (app / module.CONFIG_POINTER_NAME).exists()
    assert (source / "synthetic.txt").read_bytes() == b"unchanged"
