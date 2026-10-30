from orion_core.forge_deployment import prepare, recover_pending

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_interrupted_swap_restores_exact_previous_bytes(tmp_path):
    path = tmp_path / "demo_tool.py"
    before = b"x = 'before'\r\n"
    after = b"x = 'after'\n"
    path.write_bytes(before)
    receipt = prepare(path, before, after)
    path.write_bytes(after)  # simulate process death before activation completes
    assert recover_pending(tmp_path) == [{"file": path.name, "restored": True}]
    assert path.read_bytes() == before
    assert not receipt.exists()


def test_interrupted_new_tool_is_removed(tmp_path):
    path = tmp_path / "demo_tool.py"
    prepare(path, None, b"new")
    path.write_bytes(b"new")
    assert recover_pending(tmp_path)[0]["restored"]
    assert not path.exists()


def test_post_crash_manual_edit_is_preserved(tmp_path):
    path = tmp_path / "demo_tool.py"
    path.write_bytes(b"stable")
    receipt = prepare(path, b"stable", b"candidate")
    path.write_bytes(b"human edit")
    result = recover_pending(tmp_path)
    assert not result[0]["restored"]
    assert path.read_bytes() == b"human edit" and receipt.exists()


def test_malformed_receipt_cannot_change_files_outside_root(tmp_path):
    import json
    directory = tmp_path / ".forge_transactions"
    directory.mkdir()
    (directory / "demo.json").write_text(json.dumps({"file": "../outside.py"}))
    assert not recover_pending(tmp_path)[0]["restored"]


def test_concurrent_prepare_keeps_exactly_one_claim_and_backup(tmp_path, monkeypatch):
    path = tmp_path / "demo_tool.py"
    path.write_bytes(b"working source")
    receipt = tmp_path / ".forge_transactions" / "demo_tool.json"
    started = threading.Barrier(2)
    checked_absence = threading.Barrier(2)
    original_exists = Path.exists

    def exists(candidate):
        result = original_exists(candidate)
        if candidate == receipt and not result:
            # Force both old check-then-replace callers past the same stale
            # observation. Exclusive creation must not depend on this check.
            checked_absence.wait(timeout=3)
        return result

    monkeypatch.setattr(Path, "exists", exists)

    def attempt(label):
        started.wait(timeout=3)
        try:
            return label, prepare(path, b"working source", b"candidate " + label.encode()), None
        except RuntimeError as exc:
            return label, None, exc

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(attempt, label) for label in ("first", "second")]
        results = [future.result(timeout=5) for future in futures]

    winners = [item for item in results if item[1] is not None]
    assert len(winners) == 1, "two publishers claimed the same deployment receipt"
    assert sum(isinstance(item[2], RuntimeError) for item in results) == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    winner = winners[0][0].encode()
    assert payload["before_hash"] == hashlib.sha256(b"working source").hexdigest()
    assert payload["after_hash"] == hashlib.sha256(b"candidate " + winner).hexdigest()
    assert path.read_bytes() == b"working source"


def test_prepare_refusal_preserves_an_existing_receipt_and_source(tmp_path):
    path = tmp_path / "demo_tool.py"
    path.write_bytes(b"working source")
    receipt = prepare(path, b"working source", b"first candidate")
    original_receipt = receipt.read_bytes()

    with pytest.raises(RuntimeError, match="recovered"):
        prepare(path, b"different backup", b"second candidate")

    assert receipt.read_bytes() == original_receipt
    assert path.read_bytes() == b"working source"


def test_failed_prepare_flush_never_replaces_source_and_releases_its_claim(tmp_path, monkeypatch):
    path = tmp_path / "demo_tool.py"
    path.write_bytes(b"working source")

    def fail_flush(_):
        raise OSError("synthetic durable-write failure")

    monkeypatch.setattr(os, "fsync", fail_flush)
    with pytest.raises(OSError, match="durable-write"):
        prepare(path, b"working source", b"candidate")

    assert path.read_bytes() == b"working source"
    assert not (tmp_path / ".forge_transactions" / "demo_tool.json").exists()
    assert list((tmp_path / ".forge_transactions").iterdir()) == []


def test_prepare_refuses_a_stale_backup_before_touching_source(tmp_path):
    path = tmp_path / "demo_tool.py"
    path.write_bytes(b"newer working generation")

    with pytest.raises(RuntimeError, match="changed"):
        prepare(path, b"stale captured generation", b"candidate")

    assert path.read_bytes() == b"newer working generation"
    assert not (tmp_path / ".forge_transactions" / "demo_tool.json").exists()
