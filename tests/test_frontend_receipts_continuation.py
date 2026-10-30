"""Worker face ownership and truthful file-restore receipts."""
from pathlib import Path
from types import SimpleNamespace

import pytest

@pytest.mark.parametrize("running", [True, False])
def test_first_face_frame_does_not_paint_the_worker_owned_head_on_the_gui_thread(running):
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from orion_core.gui.face_pipeline import FrameBuffer
    from orion_core.gui.holo_head import HoloHeadPanel
    app = QApplication.instance() or QApplication(["orion-audit"])
    panel = HoloHeadPanel()
    panel.timer.stop()
    panel.resize(320, 400)
    painted = []
    panel._head.paint = lambda *_: painted.append("GUI painted the worker's head")
    panel._pipeline = SimpleNamespace(frames=FrameBuffer(), running=running, stop=lambda: True)
    try:
        image = QImage(320, 400, QImage.Format.Format_RGB32)
        panel.render(image)
        assert not painted, "No published frame caused synchronous painting of a worker-owned head"
    finally:
        panel._pipeline = None
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_failed_recycle_undo_is_not_reported_as_undone(tmp_path, monkeypatch):
    from orion_core import recycle_bin, undo
    from orion_core.dispatch_files import FilesDispatchMixin
    class Files(FilesDispatchMixin):
        def _resolve_user_path(self, raw):
            return Path(raw)
        def _ensure_write_safe(self, path):
            assert Path(path).is_relative_to(tmp_path)
    target = tmp_path / "large.bin"
    target.write_bytes(b"synthetic" * 20)
    monkeypatch.setattr("orion_core.dispatch_files.UNDO_MAX_BYTES", 16)
    monkeypatch.setattr(recycle_bin, "refusal", lambda _path: None)
    def trash(path):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        Path(path).unlink()
        return True
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", trash)
    undo.clear()
    try:
        # Every delete now waits for the owner's on-screen approval (30 Oct
        # 2026); this is the step that runs once it is given.
        assert Files()._file_controller_execute({"action": "delete", "path": str(target)}).ok
        assert not target.exists()
        monkeypatch.setattr(recycle_bin, "restore_from_recycle_bin", lambda _path: False)
        ok, detail = undo.undo_last()
        assert not ok, f"No bytes or bin entry were restored, but undo claimed success: {detail}"
    finally:
        undo.clear()
