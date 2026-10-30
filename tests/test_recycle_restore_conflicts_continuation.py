"""Undo must preserve a replacement file without claiming the old one returned."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_undo_preserves_a_replacement_and_reports_restore_conflict(tmp_path, monkeypatch):
    from orion_core import recycle_bin, undo
    from orion_core.dispatch_files import FilesDispatchMixin
    class Files(FilesDispatchMixin):
        def _resolve_user_path(self, raw):
            return Path(raw)
        def _ensure_write_safe(self, path):
            assert Path(path).resolve().is_relative_to(tmp_path.resolve())
    target = tmp_path / "note.bin"
    target.write_bytes(b"original bytes")
    def trash(path):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        Path(path).unlink()
        return True
    monkeypatch.setattr(recycle_bin, "refusal", lambda _: None)
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", trash)
    undo.clear()
    try:
        # The step that runs once the owner approves the delete (every delete
        # asks first since 30 Oct 2026).
        assert Files()._file_controller_execute({"action": "delete", "path": str(target)}).ok
        target.write_bytes(b"replacement bytes")
        ok, detail = undo.undo_last()
        assert not ok, f"Restore was not performed, but the receipt claimed success: {detail}"
        assert target.read_bytes() == b"replacement bytes"
    finally:
        undo.clear()


@pytest.mark.parametrize("replacement", ["same_bytes", "symlink"])
def test_already_restored_bytes_are_success_but_a_foreign_symlink_is_preserved(tmp_path, monkeypatch, replacement):
    from orion_core import recycle_bin, undo
    from orion_core.dispatch_files import FilesDispatchMixin
    class Files(FilesDispatchMixin):
        def _resolve_user_path(self, raw):
            return Path(raw)
        def _ensure_write_safe(self, path):
            assert Path(path).resolve().is_relative_to(tmp_path.resolve())
    target = tmp_path / "note.bin"
    original = b"original bytes"
    target.write_bytes(original)
    def trash(path):
        Path(path).unlink()
        return True
    monkeypatch.setattr(recycle_bin, "refusal", lambda _: None)
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", trash)
    undo.clear()
    try:
        assert Files()._file_controller_execute({"action": "delete", "path": str(target)}).ok
        if replacement == "same_bytes":
            target.write_bytes(original)
        else:
            foreign = tmp_path / "foreign.bin"
            foreign.write_bytes(b"foreign bytes")
            try:
                target.symlink_to(foreign)
            except OSError:
                pytest.skip("Host does not allow unprivileged file symlinks")
        ok, detail = undo.undo_last()
        if replacement == "same_bytes":
            assert ok and "already restored" in detail
            assert target.read_bytes() == original
        else:
            assert not ok and target.is_symlink()
            assert foreign.read_bytes() == b"foreign bytes"
    finally:
        undo.clear()
