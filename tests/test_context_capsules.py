"""Durable work handoffs preserve exact user context without touching references."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.context_capsules import ContextCapsuleStore, default_path


@pytest.fixture
def store(tmp_path):
    return ContextCapsuleStore(tmp_path / "capsules.db")


def save(store, **kwargs):
    return store.handle({"action": "save", "title": "Robot arm", "intent": "Reliable grip",
                         "context": "Servo calibrated at 90 degrees.",
                         "next_action": "Measure the unloaded current.",
                         "blockers": ["Need multimeter"], "assumptions": ["5V supply"],
                         "references": ["C:/notes/build.md", "https://example.org/manual"],
                         **kwargs})


def test_restart_retains_full_handoff_and_exact_next_action(store):
    saved = save(store)
    assert saved.ok
    identifier = saved.receipt["id"]
    restored = ContextCapsuleStore(store.path).handle({"action": "show", "id": identifier})
    assert restored.ok
    assert restored.receipt["capsule"] == saved.receipt["capsule"]
    assert "Next action: Measure the unloaded current." in restored.text
    assert restored.receipt["capsule"]["references"] == ["C:/notes/build.md", "https://example.org/manual"]


def test_same_title_creates_distinct_checkpoints_and_id_patch_preserves_other_fields(store):
    first = save(store)
    second = save(store, next_action="Order the test jig.")
    assert first.receipt["id"] != second.receipt["id"]
    updated = store.handle({"action": "save", "id": first.receipt["id"], "blockers": []})
    assert updated.ok
    assert updated.receipt["capsule"]["blockers"] == []
    assert updated.receipt["capsule"]["context"] == first.receipt["capsule"]["context"]
    assert updated.receipt["capsule"]["created_at"] == first.receipt["capsule"]["created_at"]
    assert store.handle({"action": "list"}).receipt["total"] == 2


@pytest.mark.parametrize("changes", [
    {"title": " "}, {"next_action": ""}, {"context": "x" * 12001},
    {"blockers": "not a list"}, {"references": ["x"] * 25},
    {"assumptions": [42]}, {"references": ["x" * 2049]}, {"intent": "bad\x00value"},
])
def test_invalid_patch_preserves_existing_checkpoint(store, changes):
    initial = save(store)
    identifier = initial.receipt["id"]
    result = store.handle({"action": "save", "id": identifier, **changes})
    assert not result.ok
    current = store.handle({"action": "show", "id": identifier})
    assert current.receipt["capsule"] == initial.receipt["capsule"]


def test_new_checkpoint_requires_a_title_and_next_action(store):
    assert not store.handle({"action": "save", "title": "Unfinished"}).ok
    assert not store.handle({"action": "save", "next_action": "Think"}).ok
    assert not store.path.exists()


def test_archive_is_durable_hidden_by_default_and_still_retrievable(store):
    identifier = save(store).receipt["id"]
    archived = store.handle({"action": "archive", "id": identifier})
    assert archived.ok and archived.receipt["capsule"]["archived"]
    restarted = ContextCapsuleStore(store.path)
    assert restarted.handle({"action": "list"}).receipt["total"] == 0
    assert restarted.handle({"action": "list", "include_archived": True}).receipt["total"] == 1
    shown = restarted.handle({"action": "show", "id": identifier})
    assert shown.ok and "(archived)" in shown.text
    assert restarted.handle({"action": "archive", "id": identifier}).ok


def test_missing_ids_never_create_or_overwrite_checkpoints(store):
    save(store)
    missing = "cc-" + "0" * 32
    for action in ("show", "archive", "save"):
        result = store.handle({"action": action, "id": missing, "next_action": "changed"})
        assert not result.ok and result.receipt["state"] == "not_found"
    assert store.handle({"action": "list"}).receipt["total"] == 1
    assert not store.handle({"action": "show", "id": "Robot arm"}).ok


def test_search_is_literal_and_lists_are_bounded(store):
    save(store, title="100% complete", blockers=["Need unusual component"])
    for index in range(3):
        save(store, title=f"Build {index}")
    assert store.handle({"action": "list", "query": "%"}).receipt["total"] == 1
    assert store.handle({"action": "list", "query": "unusual component"}).receipt["total"] == 1
    limited = store.handle({"action": "list", "limit": 2})
    assert limited.receipt["total"] == 4 and len(limited.receipt["capsules"]) == 2
    for bad_limit in (0, 51, True, 2.5, "2"):
        assert not store.handle({"action": "list", "limit": bad_limit}).ok
    assert not store.handle({"action": "list", "include_archived": "true"}).ok


def test_concurrent_connections_preserve_distinct_checkpoints_and_disjoint_patches(store):
    identifier = save(store).receipt["id"]
    def insert(index):
        return save(ContextCapsuleStore(store.path), title=f"Checkpoint {index}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(insert, range(12)))
    assert all(result.ok for result in results)
    assert store.handle({"action": "list"}).receipt["total"] == 13
    patches = [{"context": "new evidence"}, {"intent": "new purpose"},
               {"blockers": ["new obstacle"]}, {"next_action": "new next step"}]
    def patch(fields):
        return ContextCapsuleStore(store.path).handle({"action": "save", "id": identifier, **fields})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(patch, patches))
    assert all(result.ok for result in results)
    record = store.handle({"action": "show", "id": identifier}).receipt["capsule"]
    for fields in patches:
        for field, value in fields.items():
            assert record[field] == value


def test_storage_failure_does_not_report_saved(tmp_path):
    path = tmp_path / "corrupt.db"
    path.write_bytes(b"this is not a sqlite database")
    result = save(ContextCapsuleStore(path))
    assert not result.ok and result.receipt["state"] == "failed"
    assert path.read_bytes() == b"this is not a sqlite database"


def test_default_path_is_resolved_after_module_import(monkeypatch, tmp_path):
    from orion_core import constants, context_capsules
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path)
    # Keep this test meaningful when the global isolation fixture redirects
    # the helper: restore the original lazily resolved helper for this case.
    monkeypatch.setattr(context_capsules, "default_path", default_path)
    store = ContextCapsuleStore()
    assert store.path == tmp_path / "context_capsules.db"
    assert not store.path.exists()
    assert save(store).ok
