"""Disposable hard-kill recovery, never provider replay or private-tree search."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from orion_core.research_runs import ResearchRunStore


@pytest.mark.parametrize("boundary", ["after_rename", "serialized", "mid_write", "multifile"])
def test_killed_publication_recovers_only_verified_artifacts(tmp_path, boundary):
    ledger, ready = tmp_path / "runs.json", tmp_path / "ready.json"
    output = tmp_path / "output"
    output.mkdir()
    script = r'''
import asyncio,json,os,sys,time
from pathlib import Path
from orion_core.research_runs import ResearchRunStore,publish
from orion_core.research_docx import _publish_new
from orion_core.research_publication import PublicationJournal
ledger,ready,output=map(Path,sys.argv[1:4])
boundary=sys.argv[4]
store=ResearchRunStore(ledger)
row=store.start('Synthetic killed publication','paper')
original=os.replace
def checkpoint(path):
    ready.write_text(json.dumps({'pid':os.getpid(),'run_id':row['run_id'],'path':str(path)}))
    while True: time.sleep(.01)
def replace(source,target):
    original(source,target)
    if boundary=='after_rename' and Path(target).parent==output and Path(target).suffix=='.md':
        checkpoint(target)
os.replace=replace
stage=PublicationJournal.stage
def staged(self,key,value,**kwargs):
    stage(self,key,value,**kwargs)
    if boundary=='serialized' and value=='serialized': checkpoint(kwargs['temporary'])
PublicationJournal.stage=staged
def write(path):
    path.write_text('Synthetic complete canary')
    if boundary in {'mid_write','multifile'}:
        checkpoint(output/'first.md' if boundary=='multifile' else path)
async def run():
    asyncio.current_task()._orion_research_publication_run=(store,row['run_id'])
    if boundary=='multifile':
        await publish(_publish_new,output,'first','.md',lambda p:p.write_text('Synthetic complete canary'),artifact_paths=lambda p:[p])
    await publish(_publish_new,output,'result','.md',write,artifact_paths=lambda p:[p])
asyncio.run(run())
'''
    environment = dict(os.environ, ORION_CONFIG_DIR=str(tmp_path / "config"))
    process = subprocess.Popen([sys.executable, "-c", script, str(ledger), str(ready), str(output), boundary],
                               env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = reader = None
    try:
        deadline = time.monotonic() + 12
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), ("Disposable publisher did not reach rename checkpoint: " +
                                (process.communicate(timeout=2)[1] if process.poll() is not None else "deadline"))
        checkpoint = json.loads(ready.read_text())
        child = psutil.Process(checkpoint["pid"])
        reader = ResearchRunStore(ledger)
        assert reader.list()[0]["research_state"] == "running"
        child.kill()
        child.wait(timeout=5)
        process.wait(timeout=5)
        durable_before = ledger.read_bytes()
        observed = reader.list(run_id=checkpoint["run_id"])[0]
        assert observed["research_state"] == "interrupted" and ledger.read_bytes() == durable_before
        reader.close()
        reader = ResearchRunStore(ledger)
        record = reader.list(run_id=checkpoint["run_id"])[0]
        assert record["research_state"] == "interrupted"
        if boundary == "mid_write":
            assert not record["file_paths"]
            assert record["publication_recovery"]["uncertain_count"] == 1
            return
        assert checkpoint["path"] in record["file_paths"]
        assert record["file_paths"] == observed["file_paths"]
        assert record["publication_recovery"]["serialized_partial_count"] == int(boundary == "serialized")
        assert record["publication_recovery"]["published_count"] == int(boundary != "serialized")
        assert record["publication_recovery"]["uncertain_count"] == int(boundary == "multifile")
        assert Path(checkpoint["path"]).read_text() == "Synthetic complete canary"
        if boundary == "serialized":
            assert not (output / "result.md").exists()  # reconciliation never completes a rename
    finally:
        if child is not None:
            try:
                if child.is_running():
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if reader is not None:
            reader.close()


def published_run(tmp_path):
    from orion_core.research_docx import _publish_new
    from orion_core.research_runs import bind_publication_run, publish
    store = ResearchRunStore(tmp_path / "runs.json")
    row = store.start("Synthetic known files", "paper")
    output = tmp_path / "output"
    output.mkdir()

    async def run():
        bind_publication_run(asyncio.current_task(), store, row["run_id"])
        return await publish(_publish_new, output, "result", ".md",
                             lambda p: p.write_text("canary evidence"), artifact_paths=lambda p: [p])

    artifact = asyncio.run(run())
    manifest = store.path.parent / "research_publications" / row["publication_manifest"]
    return store, row, artifact, manifest


@pytest.mark.parametrize("damage", ["contents", "identity", "missing"])
def test_recovery_refuses_changed_or_missing_output_without_deleting_it(tmp_path, damage):
    from orion_core.atomic_io import _replace
    store, row, artifact, _manifest = published_run(tmp_path)
    if damage == "contents":
        artifact.write_text("changed content")
    elif damage == "identity":
        other = artifact.with_suffix(".other")
        other.write_text("canary evidence")
        _replace(str(other), artifact)
    else:
        artifact.unlink()
    before = artifact.read_bytes() if artifact.exists() else None
    store.close()
    recovered = ResearchRunStore(store.path)
    try:
        record = recovered.list(run_id=row["run_id"])[0]
        assert record["research_state"] == "interrupted" and not record["file_paths"]
        assert record["publication_recovery"]["uncertain_count"] == 1
        assert (artifact.read_bytes() if artifact.exists() else None) == before
    finally:
        recovered.close()


@pytest.mark.parametrize("damage", ["truncated", "oversized", "wrong_run", "extra_field", "escaped", "relative"])
def test_bad_manifests_are_preserved_and_cannot_claim_unknown_files(tmp_path, damage):
    from orion_core.research_publication import MAX_MANIFEST_BYTES
    store, row, artifact, manifest = published_run(tmp_path)
    data = json.loads(manifest.read_text())
    if damage == "truncated":
        payload = b'{"version":'
    elif damage == "oversized":
        payload = b" " * (MAX_MANIFEST_BYTES + 1)
    else:
        if damage == "wrong_run":
            data["run_id"] = "another run"
        elif damage == "extra_field":
            data["artifacts"][0]["unknown"] = "unsupported schema"
        elif damage == "escaped":
            data["artifacts"][0]["destination"] = str(tmp_path / "private.md")
        else:
            data["artifacts"][0]["folder"] = "relative"
        payload = json.dumps(data).encode()
    manifest.write_bytes(payload)
    store.close()
    recovered = ResearchRunStore(store.path)
    try:
        record = recovered.list(run_id=row["run_id"])[0]
        assert record["research_state"] == "interrupted" and not record["file_paths"]
        assert record["publication_recovery"]["uncertain"] is True
        assert manifest.read_bytes() == payload and artifact.read_text() == "canary evidence"
    finally:
        recovered.close()


def test_recovery_observes_only_manifest_paths_and_never_globs_output_tree(tmp_path, monkeypatch):
    store, row, artifact, _manifest = published_run(tmp_path)
    unrelated = artifact.parent / "unrelated.md"
    unrelated.write_text("Private unrelated synthetic content")
    monkeypatch.setattr(Path, "rglob", lambda *_a, **_k: pytest.fail("Recovery searched a private tree"))
    store.close()
    recovered = ResearchRunStore(store.path)
    try:
        assert recovered.list(run_id=row["run_id"])[0]["file_paths"] == [str(artifact)]
        assert unrelated.read_text() == "Private unrelated synthetic content"
    finally:
        recovered.close()


def test_manifest_failure_refuses_writer_before_serialization(tmp_path, monkeypatch):
    from orion_core import research_publication
    from orion_core.research_docx import _publish_new
    from orion_core.research_runs import bind_publication_run, publish
    store = ResearchRunStore(tmp_path / "runs.json")
    row = store.start("Admission then denied journal", "paper")
    calls = []

    def denied(*_a, **_k):
        raise PermissionError("synthetic journal denial")

    monkeypatch.setattr(research_publication, "atomic_write_text", denied)

    async def run():
        bind_publication_run(asyncio.current_task(), store, row["run_id"])
        with pytest.raises(PermissionError, match="journal denial"):
            await publish(_publish_new, tmp_path, "result", ".md", lambda p: calls.append(p),
                          artifact_paths=lambda p: [p])

    try:
        asyncio.run(run())
        assert not calls and not list(tmp_path.glob(".paper-*")) and not (tmp_path / "result.md").exists()
    finally:
        store.close()


def test_recovery_read_budget_returns_uncertainty_without_claiming_completion(tmp_path):
    from orion_core.research_publication import reconcile
    store, row, artifact, manifest = published_run(tmp_path)
    try:
        value = reconcile(manifest, row["run_id"], key=store._rows[0]["_publication_key"], budget=[0])
        assert value["paths"] == [] and value["uncertain_count"] == 1 and value["verification_limited"]
        assert artifact.read_text() == "canary evidence"
    finally:
        store.close()


def test_scoped_status_recovers_only_requested_run_after_owner_absence(tmp_path, monkeypatch):
    from orion_core import research_publication
    store, row, artifact, _manifest = published_run(tmp_path)
    unrelated = store.start("Other active research", "dossier")
    reader = ResearchRunStore(store.path)
    before = store.path.read_bytes()
    store.close()
    calls = []
    original = research_publication.reconcile

    def reconcile(path, run_id, **kwargs):
        calls.append(run_id)
        return original(path, run_id, **kwargs)

    monkeypatch.setattr(research_publication, "reconcile", reconcile)
    try:
        record = reader.list(run_id=row["run_id"], limit=1)[0]
        assert record["file_paths"] == [str(artifact)] and calls == [row["run_id"]]
        assert store.path.read_bytes() == before  # status never promotes or reconciles disk
        admitted = reader.start("Actual handoff", "paper")
        assert admitted["run_id"] not in {row["run_id"], unrelated["run_id"]}
        assert reader.list(run_id=unrelated["run_id"])[0]["research_state"] == "interrupted"
    finally:
        reader.close()


@pytest.mark.parametrize("failure", ["transient", "permanent", "competing"])
def test_journal_retains_existing_windows_rename_guards(tmp_path, monkeypatch, failure):
    from orion_core import research_docx
    from orion_core.research_runs import bind_publication_run, publish
    store = ResearchRunStore(tmp_path / "runs.json")
    row = store.start("Guarded publication", "paper")
    original = research_docx.os.replace
    attempts, serializations = [], []
    monkeypatch.setattr(research_docx, "_WINDOWS", True)
    monkeypatch.setattr(research_docx.time, "sleep", lambda _s: None)

    def replace(source, target):
        if Path(target).name == "result.md":
            attempts.append(str(target))
            if failure == "competing":
                Path(target).write_text("Another writer's populated reservation")
            if failure != "transient" or len(attempts) == 1:
                error = PermissionError("synthetic rename denial")
                error.winerror = 5
                raise error
        return original(source, target)

    def write(path):
        serializations.append(path)
        path.write_text("Complete serialized evidence")

    monkeypatch.setattr(research_docx.os, "replace", replace)

    async def run():
        bind_publication_run(asyncio.current_task(), store, row["run_id"])
        operation = publish(research_docx._publish_new, tmp_path, "result", ".md", write,
                            artifact_paths=lambda p: [p])
        if failure == "transient":
            assert (await operation).read_text() == "Complete serialized evidence"
        else:
            with pytest.raises(PermissionError):
                await operation

    try:
        asyncio.run(run())
        assert len(serializations) == 1
        assert len(attempts) == {"transient": 2, "permanent": 5, "competing": 1}[failure]
        if failure == "competing":
            assert (tmp_path / "result.md").read_text() == "Another writer's populated reservation"
        elif failure == "permanent":
            assert not (tmp_path / "result.md").exists()
    finally:
        store.close()


def test_successful_rename_survives_final_manifest_save_failure_receipt(tmp_path, monkeypatch):
    from orion_core.research_publication import PublicationJournal
    from orion_core.research_docx import _publish_new
    from orion_core.research_runs import bind_publication_run, publication_paths, publish
    store = ResearchRunStore(tmp_path / "runs.json")
    row = store.start("Saved effect but final receipt failed", "paper")
    original = PublicationJournal.stage

    def stage(self, key, value, **kwargs):
        if value == "published":
            raise PermissionError("synthetic final manifest denial")
        return original(self, key, value, **kwargs)

    monkeypatch.setattr(PublicationJournal, "stage", stage)

    async def run():
        bind_publication_run(asyncio.current_task(), store, row["run_id"])
        with pytest.raises(PermissionError, match="final manifest denial"):
            await publish(_publish_new, tmp_path, "result", ".md", lambda p: p.write_text("Saved partial"),
                          artifact_paths=lambda p: [p])
        paths, pending = publication_paths(asyncio.current_task())
        assert paths == [str(tmp_path / "result.md")] and not pending
        assert Path(paths[0]).read_text() == "Saved partial"

    try:
        asyncio.run(run())
    finally:
        store.close()


def test_admitted_dispatch_task_is_bound_before_real_writer_runs(tmp_path):
    from types import SimpleNamespace
    from orion_core.data import ToolResult
    from orion_core.dispatch_knowledge import _spawn_research_task
    from orion_core.research_docx import _publish_new
    from orion_core.research_runs import publish
    store = ResearchRunStore(tmp_path / "runs.json")
    host = SimpleNamespace(_research_run_store=store)

    async def work():
        path = await publish(_publish_new, tmp_path, "result", ".md", lambda p: p.write_text("Admitted work"),
                             artifact_paths=lambda p: [p])
        return ToolResult("Saved", receipt={"research_state": "completed", "file_paths": [str(path)]})

    async def run():
        task, receipt = await _spawn_research_task(host, "Trusted task", "paper", work())
        assert task._orion_research_publication_run == (store, receipt["run_id"])
        await task
        await asyncio.sleep(0)
        save = host._background_research_runs[0].get("save_task")
        if save is not None:
            await save
        row = store.list(run_id=receipt["run_id"])[0]
        assert row["research_state"] == "completed"
        assert (tmp_path / "research_publications" / row["publication_manifest"]).is_file()

    try:
        asyncio.run(run())
    finally:
        store.close()


def test_integrity_key_never_leaves_private_store_records(tmp_path):
    store, row, artifact, _manifest = published_run(tmp_path)
    secret = store._rows[0]["_publication_key"]
    reader = ResearchRunStore(store.path)
    try:
        assert secret not in json.dumps(row)
        assert secret not in json.dumps(store.list())
        assert secret not in json.dumps(reader.list())
        saved = store.update(row["run_id"], {"research_state": "running", "file_paths": [str(artifact)]})
        assert secret not in json.dumps(saved)
        store.close()
        observed = reader.list()
        assert observed[0]["file_paths"] == [str(artifact)]
        assert secret not in json.dumps(observed) and "_publication_key" not in json.dumps(observed)
    finally:
        store.close()
        reader.close()


def test_missing_integrity_key_does_not_use_a_default_or_adopt_a_manifest(tmp_path):
    store, row, artifact, manifest = published_run(tmp_path)
    store.close()
    data = json.loads(store.path.read_text())
    data["runs"][0].pop("_publication_key")
    store.path.write_text(json.dumps(data))
    before = manifest.read_bytes()
    recovered = ResearchRunStore(store.path)
    try:
        record = recovered.list(run_id=row["run_id"])[0]
        assert record["research_state"] == "interrupted" and not record["file_paths"]
        assert record["publication_recovery"]["uncertain"]
        assert manifest.read_bytes() == before and artifact.read_text() == "canary evidence"
    finally:
        recovered.close()


def test_legacy_rows_without_journals_remain_supported_without_silent_keys(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"version": 1, "runs": [{"run_id": "legacy", "topic": "Earlier run",
        "research_state": "running", "file_paths": []}]}))
    store = ResearchRunStore(path)
    try:
        old = store.list()[0]
        assert old["research_state"] == "interrupted" and "publication_recovery" not in old
        assert "_publication_key" not in json.loads(path.read_text())["runs"][0]
        assert store.start("New admitted work", "paper")["run_id"] != "legacy"
    finally:
        store.close()


def test_manifest_growth_between_stat_and_open_has_a_hard_read_bound(tmp_path, monkeypatch):
    from orion_core import research_publication
    store, row, _artifact, manifest = published_run(tmp_path)
    opened = Path.open
    reads = []

    class Bounded:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, count=-1):
            reads.append(count)
            assert count == research_publication.MAX_MANIFEST_BYTES + 1
            return self.handle.read(count)

    def growing(path, mode="r", *args, **kwargs):
        if path == manifest and mode == "rb":
            with opened(path, "ab") as handle:
                handle.write(b" " * (research_publication.MAX_MANIFEST_BYTES + 10))
            return Bounded(opened(path, mode, *args, **kwargs))
        return opened(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", growing)
    try:
        with pytest.raises(ValueError, match="exceeds its limit"):
            research_publication.reconcile(manifest, row["run_id"], key=store._rows[0]["_publication_key"])
        assert reads == [research_publication.MAX_MANIFEST_BYTES + 1]
        assert manifest.stat().st_size > research_publication.MAX_MANIFEST_BYTES
    finally:
        store.close()


@pytest.mark.parametrize("change", ["corrupt", "deleted", "valid_but_changed"])
def test_live_writer_refuses_external_manifest_edits_without_losing_evidence(tmp_path, change):
    from orion_core.research_docx import _publish_new
    from orion_core.research_runs import bind_publication_run, publish
    from orion_core.research_publication import _signed
    store, row, artifact, manifest = published_run(tmp_path)
    if change == "deleted":
        manifest.unlink()
        preserved = None
    elif change == "corrupt":
        manifest.write_bytes(b"{damaged manifest")
        preserved = manifest.read_bytes()
    else:
        data = json.loads(manifest.read_text())
        data.pop("signature")
        data["artifacts"] = []
        data["signature"] = _signed(data, store._rows[0]["_publication_key"])
        manifest.write_text(json.dumps(data))
        preserved = manifest.read_bytes()
    called = []

    async def run():
        bind_publication_run(asyncio.current_task(), store, row["run_id"])
        with pytest.raises((OSError, ValueError)):
            await publish(_publish_new, artifact.parent, "second", ".md", lambda p: called.append(p),
                          artifact_paths=lambda p: [p])

    try:
        asyncio.run(run())
        assert not called and artifact.read_text() == "canary evidence"
        assert (manifest.read_bytes() if manifest.exists() else None) == preserved
    finally:
        store.close()


def test_journal_checks_existing_ancestors_before_creating_directories(tmp_path, monkeypatch):
    import threading
    from orion_core import research_publication
    folder = tmp_path / "not-created" / "run"
    journal = research_publication.PublicationJournal(folder / "manifest.json", "run", "ab" * 32,
                                                     lambda: None, threading.RLock())
    mkdirs = []

    def refused(*_a, **_k):
        raise ValueError("synthetic linked ancestor")

    monkeypatch.setattr(research_publication, "_ordinary", refused)
    monkeypatch.setattr(Path, "mkdir", lambda *_a, **_k: mkdirs.append(True))
    with pytest.raises(ValueError, match="linked ancestor"):
        journal._save()
    assert not mkdirs and not folder.exists()


def test_limit_one_status_copies_one_record_instead_of_all_history(tmp_path, monkeypatch):
    from orion_core import research_runs
    store = ResearchRunStore(tmp_path / "runs.json")
    for index in range(4):
        store.start(f"Running {index}", "dossier")
    original = research_runs._public_row
    copies = []

    def copy(row):
        copies.append(row["run_id"])
        return original(row)

    monkeypatch.setattr(research_runs, "_public_row", copy)
    try:
        result = store.list(limit=1)
        assert len(result) == 1 and copies == [result[0]["run_id"]]
    finally:
        store.close()


def test_artifact_growth_cannot_exceed_its_reserved_recovery_read_budget(tmp_path, monkeypatch):
    from orion_core import research_publication
    artifact = tmp_path / "artifact.md"
    artifact.write_bytes(b"small")
    original = Path.open
    reads = []

    class Growing:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, count=-1):
            reads.append(count)
            return self.handle.read(count)

    def opened(path, mode="r", *args, **kwargs):
        if path == artifact and mode == "rb":
            with original(path, "ab") as handle:
                handle.write(b"x" * 1024)
            return Growing(original(path, mode, *args, **kwargs))
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(ValueError, match="grew during verification"):
        research_publication._fingerprint(artifact)
    assert reads == [6]  # initial five bytes plus exactly one growth sentinel
