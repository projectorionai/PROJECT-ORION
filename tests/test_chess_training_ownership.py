"""Continuous and desktop chess trainers cannot race on the same checkpoint."""

from __future__ import annotations

import asyncio
import json
from threading import Event

import pytest

from orion_core.grandmaster.supervisor import Grandmaster
from orion_core.grandmaster.training_owner import TrainingLease, external_training_status
from orion_core.grandmaster import training_owner


def test_lease_is_exclusive_and_released_after_close(tmp_path):
    first = TrainingLease.acquire(tmp_path)
    assert first is not None
    try:
        assert TrainingLease.acquire(tmp_path) is None
    finally:
        first.close()
    replacement = TrainingLease.acquire(tmp_path)
    assert replacement is not None
    replacement.close()


def test_desktop_start_and_force_report_busy_without_a_second_writer(tmp_path):
    held = TrainingLease.acquire(tmp_path)

    async def flow():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")
        try:
            result = await swarm.start()
            assert result["status"] == "busy"
            assert swarm._selfplay is None
            result = await swarm.force_training()
            assert result["status"] == "busy"
            assert not swarm._force.is_set()
            assert swarm._checkpoint_owner is None
        finally:
            await swarm.close()

    try:
        asyncio.run(flow())
    finally:
        held.close()


def test_direct_practice_and_training_calls_also_obey_the_directory_lease(tmp_path, monkeypatch):
    held = TrainingLease.acquire(tmp_path)

    async def flow():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")

        async def unwanted(*_args, **_kwargs):
            pytest.fail("A second checkpoint writer was started")

        monkeypatch.setattr(swarm, "_run_child", unwanted)
        try:
            assert (await swarm.practice(2))["status"] == "busy"
            assert (await swarm.train_and_validate())["status"] == "busy"
            assert not swarm._owners
        finally:
            await swarm.close()

    try:
        asyncio.run(flow())
    finally:
        held.close()


def test_running_continuous_owner_is_displayed_and_stale_state_is_ignored(tmp_path, monkeypatch):
    state = {"version": 1, "pid": 123, "phase": "arena", "arena": {"scores": [0.5] * 12}}
    (tmp_path / "improvement.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(training_owner, "_continuous_process", lambda pid, _directory: pid == 123)
    held = TrainingLease.acquire(tmp_path)
    try:
        result = external_training_status(tmp_path)
        assert result["status"] == "running"
        assert result["pid"] == 123
        assert "12/80 colour pairs" in result["reason"]
    finally:
        held.close()
    assert external_training_status(tmp_path) is None


def test_process_identity_must_match_module_and_checkpoint_directory(tmp_path, monkeypatch):
    import psutil

    class Process:
        def is_running(self):
            return True

        def cmdline(self):
            return ["python.exe", "-m", "tools.improve_chess", "--directory", str(tmp_path)]

        def cwd(self):
            return str(tmp_path.parent)

    monkeypatch.setattr(psutil, "Process", lambda _pid: Process())
    assert training_owner._continuous_process(123, tmp_path)
    assert not training_owner._continuous_process(123, tmp_path / "other")
    monkeypatch.setattr(Process, "cmdline", lambda _self: ["python.exe", "another_script.py"])
    assert not training_owner._continuous_process(123, tmp_path)
    assert not training_owner._continuous_process(True, tmp_path)


def test_closing_desktop_leaves_external_worker_running(tmp_path, monkeypatch):
    held = TrainingLease.acquire(tmp_path)
    (tmp_path / "improvement.json").write_text(json.dumps({"pid": 123, "phase": "collect"}), encoding="utf-8")
    monkeypatch.setattr(training_owner, "_continuous_process", lambda *_args: True)

    async def flow():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")
        assert swarm.status()["status"] == "running"
        await swarm.close()
        assert not (tmp_path / "improvement.stop").exists()
        assert TrainingLease.acquire(tmp_path) is None

    try:
        asyncio.run(flow())
    finally:
        held.close()


def test_explicit_stop_requests_checkpoint_and_waits_for_lease_release(tmp_path, monkeypatch):
    held = TrainingLease.acquire(tmp_path)
    (tmp_path / "improvement.json").write_text(json.dumps({"pid": 123, "phase": "collect"}), encoding="utf-8")
    monkeypatch.setattr(training_owner, "_continuous_process", lambda *_args: True)

    async def flow():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")

        async def worker():
            for _ in range(200):
                if (tmp_path / "improvement.stop").exists():
                    held.close()
                    return
                await asyncio.sleep(0.01)
            pytest.fail("The stop request never reached the worker")

        completion = asyncio.create_task(worker())
        result = await swarm.stop_training()
        await completion
        assert result["status"] == "stopped"
        assert "saved its work" in result["reason"]
        assert (tmp_path / "improvement.stop").exists()
        await swarm.close()

    try:
        asyncio.run(flow())
    finally:
        held.close()


def test_cancelled_lease_acquisition_releases_a_late_thread_result(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    acquire = TrainingLease.acquire

    def delayed(directory):
        entered.set()
        assert release.wait(2)
        return acquire(directory)

    monkeypatch.setattr(TrainingLease, "acquire", delayed)

    async def flow():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")
        task = asyncio.create_task(swarm._acquire_checkpoint_owner())
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        lease = acquire(tmp_path)
        assert lease is not None
        lease.close()
        await swarm.close()

    asyncio.run(flow())


def test_displayed_strength_names_its_actual_fast_move_contract():
    from orion_core.chess_brain import MEASURED_STRENGTH, strength_text

    result = strength_text()
    assert result == MEASURED_STRENGTH
    for expected in ("1801", "48 games", "24 colour pairs", "0.08 s", "1722", "1899", "30 Sep 2026"):
        assert expected in result
    assert "not a FIDE or Lichess rating" in result
    assert "2604" not in result


def test_bounded_cli_cannot_train_into_another_workers_directory(tmp_path, monkeypatch):
    from tools import train_chess

    owner = TrainingLease.acquire(tmp_path)
    monkeypatch.setattr(train_chess, "_run", lambda _args: pytest.fail("A second trainer ran"))
    try:
        assert train_chess.main(["--directory", str(tmp_path), "--games", "0"]) == 2
    finally:
        owner.close()


def test_remote_controls_refresh_verified_external_training_state(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from orion_core.gui.chess_remote import RemoteChessWidget
    from types import SimpleNamespace

    app = QApplication.instance() or QApplication([])
    state = {"status": "running", "external": True, "reason": "Continuous trainer: 43/80 colour pairs"}
    grandmaster = SimpleNamespace(status=lambda: state)
    view = RemoteChessWidget(SimpleNamespace(grandmaster=grandmaster))
    try:
        view._refresh_training()
        assert "43/80" in view.training_status.text()
        assert "stopped" not in view.training_status.text().lower()
        state.update(status="stopped", reason="Workers saved and stopped")
        view._refresh_training()
        assert "Workers saved and stopped" in view.training_status.text()
    finally:
        view.close()
    assert app is not None
