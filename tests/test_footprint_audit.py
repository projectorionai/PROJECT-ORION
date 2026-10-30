"""Disk totals must remain honest when files are unreadable or redirected."""

from contextlib import contextmanager
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import audit_footprint


@pytest.fixture
def footprint_tree(tmp_path):
    files = {"readme.txt": 5, "runtime/main.py": 10, "runtime/native/library.dll": 20}
    for relative, size in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    return tmp_path, files


def test_nested_totals_count_files_once_without_reading_contents(footprint_tree, monkeypatch):
    root, _files = footprint_tree
    # File contents could be large, private, or online-only. Metadata suffices.
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", lambda *_a, **_k: pytest.fail("read file contents"))
        report = audit_footprint.inventory(root, top=2)
    assert report["complete"]
    assert report["size_kind"] == "logical_bytes"
    assert report["files"] == 3
    assert report["total_bytes"] == 35
    assert report["subtrees"] == {
        ".": 35, "readme.txt": 5, "runtime": 30,
        str(Path("runtime/main.py")): 10,
        str(Path("runtime/native")): 20,
    }
    assert report["largest_files"] == [
        {"bytes": 20, "path": str(Path("runtime/native/library.dll"))},
        {"bytes": 10, "path": str(Path("runtime/main.py"))},
    ]
    assert report["errors"] == []


@pytest.mark.parametrize("top, expected_count", [(0, 0), (-1, 0), (1, 1), (30, 3)])
def test_largest_file_limit_does_not_change_total_size(footprint_tree, top, expected_count):
    root, _files = footprint_tree
    report = audit_footprint.inventory(root, top=top)
    assert report["total_bytes"] == 35
    assert report["files"] == 3
    assert len(report["largest_files"]) == expected_count
    assert [entry["bytes"] for entry in report["largest_files"]] == [20, 10, 5][:expected_count]


def test_unreadable_subtree_produces_an_explicit_partial_total(tmp_path, monkeypatch):
    (tmp_path / "readable.bin").write_bytes(b"abc")
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "hidden.bin").write_bytes(b"do not silently count this")
    real_scandir = audit_footprint.os.scandir

    def scandir(directory):
        if Path(directory) == denied:
            raise PermissionError("synthetic access denial")
        return real_scandir(directory)

    monkeypatch.setattr(audit_footprint.os, "scandir", scandir)
    report = audit_footprint.inventory(tmp_path)
    assert not report["complete"]
    assert report["total_bytes"] == 3
    assert report["files"] == 1
    assert len(report["errors"]) == 1
    assert report["errors"][0]["path"] == "denied"
    assert "synthetic access denial" in report["errors"][0]["error"]


@pytest.mark.parametrize("mode, attributes", [
    (stat.S_IFLNK, 0),
    (stat.S_IFDIR, getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
    (stat.S_IFREG, getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
], ids=["symlink", "directory-reparse-point", "file-reparse-point"])
def test_redirected_entries_are_not_followed_or_counted(tmp_path, monkeypatch, mode, attributes):
    (tmp_path / "kept.bin").write_bytes(b"abc")
    real_scandir = audit_footprint.os.scandir
    with real_scandir(tmp_path) as entries:
        real_entries = list(entries)
    calls = []

    def redirected_stat(*, follow_symlinks):
        assert follow_symlinks is False
        return SimpleNamespace(st_mode=mode, st_file_attributes=attributes, st_size=1_000_000)

    redirected = SimpleNamespace(path=str(tmp_path / "redirected"), stat=redirected_stat)

    @contextmanager
    def scandir(directory):
        calls.append(Path(directory))
        assert Path(directory) == tmp_path, "inventory descended into a redirected entry"
        yield iter([*real_entries, redirected])

    monkeypatch.setattr(audit_footprint.os, "scandir", scandir)
    report = audit_footprint.inventory(tmp_path)
    assert calls == [tmp_path]
    assert report["total_bytes"] == 3
    assert report["files"] == 1
    assert not report["complete"]
    assert report["skipped_links_or_reparse_points"] == ["redirected"]


@pytest.mark.parametrize("mode, attributes", [
    (stat.S_IFLNK, 0),
    (stat.S_IFDIR, getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
])
def test_a_redirected_root_is_rejected_before_resolving_it(tmp_path, monkeypatch, mode, attributes):
    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", lambda _self: SimpleNamespace(
            st_mode=mode, st_file_attributes=attributes))
        guard.setattr(Path, "resolve", lambda *_a, **_k: pytest.fail("resolved a redirected root"))
        with pytest.raises(ValueError, match="link or reparse point"):
            audit_footprint.inventory(tmp_path)
