"""Render the changed desktop pages against disposable local engines."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtGui import QFontDatabase, QPixmap
from PyQt6.QtWidgets import QApplication
from orion_core.bus import OrionBus
from orion_core.chess_engine import ChessService
from orion_core.focus import FocusEngine, FocusStore
from orion_core.gui.chess_view import ChessPanel
from orion_core.gui.cognition_deck import CognitionDeckView
from orion_core.gui.style import APP_STYLESHEET
from orion_core.gui.tasks_view import TasksView
from orion_core.job_store import JobStore
from orion_core.study import StudyEngine, StudyStore


def main():
    app = QApplication.instance() or QApplication([])
    for name in ("segoeui.ttf", "seguisb.ttf", "seguisym.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    app.setStyleSheet(APP_STYLESHEET)
    destination = ROOT / ".runtime-cache" / "debug-previews"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / ".test-runs") as scratch:
        study = StudyEngine(StudyStore(Path(scratch) / "study.db"))
        focus = FocusEngine(FocusStore(Path(scratch) / "focus.db"))
        bus = OrionBus()
        store = JobStore(None)
        cognition = CognitionDeckView(bus, study=study, focus=focus)
        tasks = TasksView(bus, jobs=SimpleNamespace(snapshot=lambda **kw: store.snapshot(**kw)))
        service = ChessService(bus)
        chess = ChessPanel(bus, service)
        for name, view in (("cognition", cognition), ("tasks", tasks), ("chess-local", chess),
                           ("chess-remote", chess)):
            if name == "chess-remote":
                chess.modes.setCurrentIndex(1)
            view.resize(1440, 840)
            view.show()
            app.processEvents()
            pixmap = QPixmap(view.size())
            view.render(pixmap)
            path = destination / f"{name}.png"
            pixmap.save(str(path))
            print(path)
            view.hide()
        for view in (cognition, tasks, chess):
            view.close()
        service.shutdown()
        store.close()
        study.store.close()
        focus.store.close()


if __name__ == "__main__":
    main()
