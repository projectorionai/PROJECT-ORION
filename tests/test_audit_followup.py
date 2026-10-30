"""Further audit regressions: damaged metadata, logging and test-data isolation."""

from __future__ import annotations

import asyncio
import io
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import conftest
from orion_core import tasks
from orion_core.neural import MLP, SparseMLP


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("damage", [
    "header", "stats", "counter", "negative_counter", "loss", "method", "meta",
    "learning_rate", "zero_dimension",
])
def test_invalid_checkpoint_metadata_is_rejected(tmp_path, kind, damage):
    net = MLP([2, 3, 1]) if kind == "dense" else SparseMLP(8, 3, 2)
    with np.load(io.BytesIO(net.to_bytes()), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    header = json.loads(arrays["header"].tobytes())
    if damage == "header":
        header = ["damaged"]
    elif damage == "stats":
        header["stats"] = ["damaged"]
    elif damage == "counter":
        header["stats"]["steps"] = "twenty"
    elif damage == "negative_counter":
        header["stats"]["samples"] = -1
    elif damage == "loss":
        header["stats"]["last_loss"] = float("nan")
    elif damage == "method":
        header["stats"]["note"] = "overwritten"
    elif damage == "meta":
        header["meta"] = ["damaged"]
    elif damage == "learning_rate":
        header["lr"] = float("inf")
    elif damage == "zero_dimension":
        header["sizes"][0] = 0
    arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    path = tmp_path / "network.npz"
    np.savez_compressed(path, **arrays)
    assert type(net).load(path) is None


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("learning_rate", [0.0, 0.001])
def test_valid_checkpoint_metadata_remains_usable(kind, learning_rate):
    net = MLP([2, 3, 1]) if kind == "dense" else SparseMLP(8, 3, 2)
    net.lr = learning_rate
    net.stats.note(0.25, 4)
    restored, meta = type(net).from_bytes(net.to_bytes({"labels": ["one", "two"]}))
    assert meta == {"labels": ["one", "two"]}
    assert restored.stats.steps == 1
    assert restored.stats.samples == 4
    assert restored.lr == learning_rate
    restored.stats.note(0.5, 2)
    assert restored.stats.samples == 6


def test_message_less_background_exception_reaches_log(monkeypatch):
    monkeypatch.setattr(tasks, "_RUNNING", set())
    lines = []

    async def scenario():
        async def fail():
            raise TimeoutError()

        task = tasks.spawn(
            fail(), name="timed-out-work",
            bus=SimpleNamespace(log=SimpleNamespace(emit=lines.append)),
        )
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert any("timed-out-work" in line and "TimeoutError" in line for line in lines)


@pytest.fixture
def protected_config(tmp_path, monkeypatch):
    # Simulated audit events only: no real configuration file is touched.
    protected = tmp_path / "protected"
    monkeypatch.setattr(conftest, "_REAL_CONFIG", os.path.normcase(str(protected)))
    monkeypatch.setattr(conftest, "_CONFIG_WRITES", [])
    return protected / "example.json"


@pytest.mark.parametrize("flags", [os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC])
@pytest.mark.parametrize("as_bytes", [False, True])
def test_audit_hook_blocks_low_level_writes(protected_config, flags, as_bytes):
    path = os.fsencode(protected_config) if as_bytes else str(protected_config)
    with pytest.raises(PermissionError, match="REAL config"):
        conftest._refuse_real_config_writes("open", (path, None, flags))


@pytest.mark.parametrize("event", ["os.rename", "os.replace"])
def test_audit_hook_blocks_moving_configuration_out(protected_config, event, tmp_path):
    with pytest.raises(PermissionError, match="REAL config"):
        conftest._refuse_real_config_writes(
            event, (str(protected_config), str(tmp_path / "outside.json"), -1, -1))


def test_audit_hook_still_allows_reads_and_scratch_writes(protected_config, tmp_path):
    conftest._refuse_real_config_writes(
        "open", (str(protected_config), None, os.O_RDONLY))
    conftest._refuse_real_config_writes(
        "open", (str(tmp_path / "scratch.txt"), None, os.O_WRONLY | os.O_CREAT))
