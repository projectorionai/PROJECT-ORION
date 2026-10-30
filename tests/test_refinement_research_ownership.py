"""Readers cannot interrupt live research or overwrite another writer's ledger."""

import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import psutil

from orion_core.research_runs import ResearchRunStore


def test_secondary_store_reads_live_run_without_interrupting_or_rewriting_it(tmp_path):
    path = tmp_path / "runs.json"
    first = ResearchRunStore(path)
    run = first.start("Still running", "dossier")
    before = path.read_bytes()
    second = ResearchRunStore(path)
    assert second.list()[0]["research_state"] == "running"
    assert path.read_bytes() == before
    first.update(run["run_id"], {"research_state": "completed"}, "Actual completion")
    assert second.list()[0]["research_state"] == "completed"


def test_secondary_writer_is_refused_without_lost_update_then_can_take_handoff(tmp_path):
    first = ResearchRunStore(tmp_path / "runs.json")
    run = first.start("First", "dossier")
    second = ResearchRunStore(first.path)
    with pytest.raises(OSError, match="owned|writer"):
        second.start("Must not execute", "paper")
    assert [row["run_id"] for row in first.list()] == [run["run_id"]]
    first.close()
    added = second.start("New owner", "paper")
    assert {row["run_id"] for row in second.list()} == {run["run_id"], added["run_id"]}
    assert second.list(run_id=run["run_id"])[0]["research_state"] == "interrupted"
    with pytest.raises(OSError, match="closed"):
        first.update(run["run_id"], {"research_state": "completed"}, "Late mutation")


def test_unreadable_ledger_is_preserved_even_when_lease_is_available(tmp_path):
    path = tmp_path / "runs.json"
    path.write_bytes(b"{broken ledger")
    store = ResearchRunStore(path)
    assert store.storage_error
    with pytest.raises(OSError):
        store.start("Must not start", "paper")
    assert path.read_bytes() == b"{broken ledger"


def test_process_crash_releases_lease_and_reader_refreshes_then_recovers(tmp_path):
    path, ready, advance, updated = (tmp_path / name for name in ("runs.json", "ready", "advance", "updated"))
    script = """
import sys,time,os,json
from pathlib import Path
from orion_core.research_runs import ResearchRunStore
path,ready,advance,updated=map(Path,sys.argv[1:])
store=ResearchRunStore(path)
row=store.start('Child work','dossier')
ready.write_text(json.dumps({'run_id':row['run_id'],'pid':os.getpid()}))
while not advance.exists(): time.sleep(0.01)
store.update(row['run_id'],{'research_state':'running','source_count':3},'Actual partial progress')
updated.write_text('done')
while True: time.sleep(0.05)
"""
    environment = dict(os.environ)
    environment.pop("ORION_CONFIG_DIR", None)
    process = subprocess.Popen([sys.executable, "-c", script, str(path), str(ready), str(advance), str(updated)],
                               env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_for(file):
        deadline = time.monotonic() + 10
        while not file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert file.exists(), "child ledger fixture did not reach its bounded checkpoint"

    reader = None
    owned_child = None
    try:
        wait_for(ready)
        owner = json.loads(ready.read_text())
        run_id = owner["run_id"]
        # Windows venv launchers may have a different PID from the interpreter
        # that owns the lease. End that exact fixture process, not just its proxy.
        owned_child = psutil.Process(owner["pid"])
        reader = ResearchRunStore(path)
        assert reader.list()[0]["research_state"] == "running"
        with pytest.raises(OSError, match="owned|writer"):
            reader.start("Refused duplicate", "paper")
        advance.write_text("continue")
        wait_for(updated)
        assert reader.list()[0]["source_count"] == 3
        assert reader.list()[0]["result"] == "Actual partial progress"
        owned_child.terminate()
        owned_child.wait(timeout=5)
        process.wait(timeout=5)
        before_recovery = path.read_bytes()
        assert reader.list(run_id=run_id)[0]["research_state"] == "interrupted"
        assert path.read_bytes() == before_recovery  # read-only owner-loss observation
        new_run = reader.start("After owner crash", "paper")
        old_run = reader.list(run_id=run_id)[0]
        assert old_run["research_state"] == "interrupted" and old_run["source_count"] == 3
        assert new_run["run_id"] != run_id and len(reader.list()) == 2
        assert path.with_name(path.name + ".lock").is_file()
    finally:
        if owned_child is not None:
            try:
                if owned_child.is_running():
                    owned_child.kill()
                    owned_child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if reader is not None:
            reader.close()


def test_lock_io_failure_refuses_admission_without_rewriting_existing_data(tmp_path, monkeypatch):
    from orion_core import research_runs
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"version": 1, "runs": []}))
    before = path.read_bytes()

    def denied(_path):
        raise PermissionError("synthetic lock access denied")

    monkeypatch.setattr(research_runs, "acquire_lease", denied)
    store = ResearchRunStore(path)
    assert store.list() == []
    with pytest.raises(OSError, match="writer lock is unavailable"):
        store.start("Never accepted", "paper")
    assert path.read_bytes() == before


def test_closed_writer_cannot_promote_or_mutate_after_handoff(tmp_path):
    first = ResearchRunStore(tmp_path / "runs.json")
    run = first.start("Before close", "paper")
    first.close()
    second = ResearchRunStore(first.path)
    before = first.path.read_bytes()
    with pytest.raises(OSError, match="closed"):
        first.start("Late start", "paper")
    with pytest.raises(OSError, match="closed"):
        first.update(run["run_id"], {"research_state": "completed"}, "Late finish")
    assert first.path.read_bytes() == before
    assert second.list()[0]["research_state"] == "interrupted"


def test_close_waits_for_active_ledger_mutation_before_releasing_lease(tmp_path, monkeypatch):
    first = ResearchRunStore(tmp_path / "runs.json")
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    original = first._save

    def save(rows):
        entered.set()
        assert release.wait(3)
        original(rows)

    def close():
        closing.set()
        first.close()

    monkeypatch.setattr(first, "_save", save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        mutation = pool.submit(first.start, "Real pending mutation", "paper")
        try:
            assert entered.wait(3)
            finishing = pool.submit(close)
            assert closing.wait(3) and not finishing.done()
            second = ResearchRunStore(first.path)
            with pytest.raises(OSError, match="owned|writer"):
                second.start("Premature handoff", "paper")
        finally:
            release.set()
        row = mutation.result(timeout=3)
        finishing.result(timeout=3)
    second.start("Safe handoff", "paper")
    assert second.list(run_id=row["run_id"])[0]["research_state"] == "interrupted"


def test_busy_ledger_refuses_dispatch_before_provider_operation(tmp_path, monkeypatch):
    from orion_core import deep_research
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    first = ResearchRunStore(tmp_path / "runs.json")
    first.start("Actual owner", "dossier")
    calls = []

    async def write(*_args, **_kwargs):
        calls.append("provider called")

    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    signal = SimpleNamespace(emit=lambda *_: None)

    class Host(KnowledgeDispatchMixin):
        pass

    host = Host()
    host.bus = SimpleNamespace(log=signal, gui_command=signal, speak_request=signal)
    host.research, host.research_director = object(), None
    host._research_run_store = ResearchRunStore(first.path)

    async def scenario():
        with pytest.raises(OSError, match="owned|writer"):
            await host.research_tool({"action": "paper", "topic": "Not admitted"})
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert calls == [] and len(first.list()) == 1
