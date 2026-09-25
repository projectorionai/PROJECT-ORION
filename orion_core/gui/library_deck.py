"""
LibraryDeckView — the File Import / File Library page of the Command Deck.

A single control surface over the ``IngestionEngine``:

    • Import File / Import Folder buttons (native pickers);
    • a live library table (path · type · chunks · version · tags);
    • an offline semantic search box that ranks stored chunks;
    • a stats strip (documents, chunks, unique chunks, embedding-cache hits).

Ingestion runs on a background ``QThread`` worker so a large folder never
freezes the UI; each finished file streams back into the table via a signal.
The view degrades to a clear message when no engine is wired in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus


class _IngestWorker(QObject):
    """Runs a file or folder ingest off the UI thread."""

    file_done = pyqtSignal(dict)      # one IngestionResult.to_dict()
    finished = pyqtSignal(str)        # human summary line

    def __init__(self, engine: Any, target: str, is_folder: bool) -> None:
        super().__init__()
        self._engine = engine
        self._target = target
        self._is_folder = is_folder

    def run(self) -> None:
        try:
            if self._is_folder:
                batch = self._engine.ingest_folder(
                    self._target, progress=lambda r: self.file_done.emit(r.to_dict()))
                self.finished.emit(batch.summary())
            else:
                result = self._engine.ingest_file(self._target)
                self.file_done.emit(result.to_dict())
                self.finished.emit(f"{Path(self._target).name}: {result.status}")
        except Exception as exc:      # never let a worker crash take the UI down
            self.finished.emit(f"Ingestion error: {exc}")


class LibraryDeckView(QWidget):
    def __init__(self, bus: OrionBus, engine: Any | None = None) -> None:
        super().__init__()
        self.bus = bus
        self.engine = engine
        self._thread: Optional[QThread] = None
        self._worker: Optional[_IngestWorker] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QLabel("FILE LIBRARY  ·  KNOWLEDGE INGESTION")
        header.setObjectName("panelHeading")
        root.addWidget(header)

        root.addWidget(self._build_controls())
        root.addWidget(self._build_search())

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["File", "Type", "Chunks", "Ver", "Tags", "Directive"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self.stats = QLabel("")
        self.stats.setObjectName("mutedLabel")
        root.addWidget(self.stats)

        if self.engine is None:
            self._set_status("Ingestion engine not available.")
        else:
            self.refresh()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_controls(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panelFrame")
        row = QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        import_file = QPushButton("＋  Import File")
        import_file.setObjectName("primaryButton")
        import_file.clicked.connect(self._pick_file)
        row.addWidget(import_file)

        import_folder = QPushButton("📁  Import Folder")
        import_folder.clicked.connect(self._pick_folder)
        row.addWidget(import_folder)

        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)

        row.addStretch(1)
        self.status = QLabel("Ready.")
        self.status.setObjectName("mutedLabel")
        row.addWidget(self.status)
        return frame

    def _build_search(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panelFrame")
        col = QVBoxLayout(frame)
        col.setContentsMargins(10, 8, 10, 8)
        col.setSpacing(6)

        row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search ingested knowledge (offline, semantic)…")
        self.search_box.returnPressed.connect(self._run_search)
        row.addWidget(self.search_box, 1)
        go = QPushButton("Search")
        go.clicked.connect(self._run_search)
        row.addWidget(go)
        col.addLayout(row)

        self.results = QListWidget()
        self.results.setMaximumHeight(120)
        col.addWidget(self.results)
        return frame

    # ── actions ────────────────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import a file to ORION's knowledge")
        if path:
            self._start(path, is_folder=False)

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Import a folder to ORION's knowledge")
        if path:
            self._start(path, is_folder=True)

    def _start(self, target: str, is_folder: bool) -> None:
        if self.engine is None or self._thread is not None:
            return
        self._set_status(f"Ingesting {Path(target).name}…")
        self._thread = QThread(self)
        self._worker = _IngestWorker(self.engine, target, is_folder)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_file_done(self, result: dict) -> None:
        if result.get("status") in {"indexed", "updated", "skipped"}:
            self._upsert_row(result)

    def _on_finished(self, summary: str) -> None:
        self._set_status(summary)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        self.refresh()

    def _run_search(self) -> None:
        self.results.clear()
        if self.engine is None:
            return
        query = self.search_box.text().strip()
        if not query:
            return
        for hit in self.engine.search(query, limit=8):
            name = Path(hit["path"]).name
            self.results.addItem(f"[{hit['score']:.2f}] {name} — {hit['snippet']}")

    # ── table + stats ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        if self.engine is None:
            return
        docs = self.engine.library(limit=300)
        self.table.setRowCount(0)
        for doc in docs:
            self._upsert_row(doc)
        stats = self.engine.stats()
        self.stats.setText(
            f"{stats['documents']} documents · {stats['chunks']} chunks · "
            f"{stats['unique_chunks']} unique · "
            f"cache {stats['embed_hits']}/{stats['embed_hits'] + stats['embed_misses']} hits")

    def _upsert_row(self, doc: dict) -> None:
        path = doc.get("path", "")
        name = Path(path).name if path else doc.get("doc_id", "?")
        # Replace an existing row for the same path, else append.
        target_row = self.table.rowCount()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == path:
                target_row = r
                break
        else:
            self.table.insertRow(target_row)

        tags = ", ".join(doc.get("tags", [])[:5])
        directive = str(doc.get("directive", "") or "")
        cells = [
            name, str(doc.get("category", "")), str(doc.get("chunks", 0)),
            str(doc.get("version", 1)), tags, directive,
        ]
        for col, text in enumerate(cells):
            cell = QTableWidgetItem(text)
            if col == 0:
                cell.setData(Qt.ItemDataRole.UserRole, path)
                cell.setToolTip(path)
            elif col == 5 and directive:
                cell.setToolTip(directive)         # full directive on hover
            self.table.setItem(target_row, col, cell)

    def _set_status(self, text: str) -> None:
        if hasattr(self, "status"):
            self.status.setText(text)
        if self.bus is not None:
            try:
                self.bus.log.emit(f"LIBRARY: {text}")
            except Exception:
                pass
