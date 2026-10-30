"""
LibraryDeckView — the File Import / File Library page of the Command Deck.

A single control surface over the ``IngestionEngine``:

    • Import File / Import Folder buttons (native pickers);
    • a live library table (path · type · chunks · version · tags);
    • an offline semantic search box that ranks stored chunks;
    • a stats strip (documents, chunks, unique chunks, embedding-cache hits).

Ingestion, searches and library reads use the bounded bulk executor so slow
storage never freezes the UI; each imported file streams back via a signal.
The view degrades to a clear message when no engine is wired in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, Qt, pyqtSignal
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
from ..executors import bulk_executor


class _LibrarySignals(QObject):
    file_done = pyqtSignal(dict)
    completed = pyqtSignal(str, int, object, str)


def _emit(signals: _LibrarySignals, name: str, *args: Any) -> None:
    try:
        getattr(signals, name).emit(*args)
    except RuntimeError:
        pass  # the view was destroyed while a storage call was finishing


def _ingest(engine: Any, signals: _LibrarySignals, target: str, is_folder: bool) -> str:
    def progress(result: Any) -> None:
        _emit(signals, "file_done", result.to_dict())

    if is_folder:
        return engine.ingest_folder(target, progress=progress).summary()
    result = engine.ingest_file(target)
    progress(result)
    return f"{Path(target).name}: {result.status}"


def _library_snapshot(engine: Any) -> tuple:
    return engine.library(limit=300), engine.stats()


class LibraryDeckView(QWidget):
    def __init__(self, bus: OrionBus, engine: Any | None = None) -> None:
        super().__init__()
        self.bus = bus
        self.engine = engine
        self._jobs: dict[str, Any] = {}
        self._search_generation = 0
        self._search_query = ""
        self._refresh_pending = False
        self._rows: dict[str, int] = {}
        self._signals = _LibrarySignals(self)
        self._signals.file_done.connect(self._on_file_done, Qt.ConnectionType.QueuedConnection)
        self._signals.completed.connect(self._on_completed, Qt.ConnectionType.QueuedConnection)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QLabel("Library")
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

        import_folder = QPushButton("Import Folder")
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
        self.search_box.setPlaceholderText("Search the library…")
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
        path, _ = QFileDialog.getOpenFileName(self, "Import a File to ORION's Knowledge")
        if path:
            self._start(path, is_folder=False)

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Import a Folder to ORION's Knowledge")
        if path:
            self._start(path, is_folder=True)

    def _start(self, target: str, is_folder: bool) -> None:
        if self.engine is None or "import" in self._jobs:
            return
        self._set_status(f"Ingesting {Path(target).name}…")
        self._submit("import", 0, _ingest, self.engine, self._signals, target, is_folder)

    def _submit(self, kind: str, generation: int, fn: Any, *args: Any, **kwargs: Any) -> None:
        # The callback retains only signals, never the QWidget. Deleting a page
        # disconnects its slots; a running storage call may finish safely.
        signals = self._signals
        try:
            future = bulk_executor().submit(fn, *args, **kwargs)
        except Exception as exc:
            self._set_status(f"{kind.title()} error: {exc}")
            return
        self._jobs[kind] = future

        def finished(done: Any) -> None:
            try:
                result, error = done.result(), ""
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            _emit(signals, "completed", kind, generation, result, error)

        future.add_done_callback(finished)

    def _on_file_done(self, result: dict) -> None:
        if result.get("status") in {"indexed", "updated", "skipped"}:
            self._upsert_row(result)

    def _on_completed(self, kind: str, generation: int, result: Any, error: str) -> None:
        self._jobs.pop(kind, None)
        if kind == "search" and generation != self._search_generation:
            if self._search_query:
                self._begin_search()
            return
        if kind == "refresh" and self._refresh_pending:
            self._refresh_pending = False
            self.refresh()
            return
        if error:
            self._set_status(f"{kind.title()} error: {error}")
        elif kind == "import":
            self._set_status(result)
        elif kind == "search":
            for hit in result:
                name = Path(hit["path"]).name
                self.results.addItem(f"[{hit['score']:.2f}] {name} — {hit['snippet']}")
        elif kind == "refresh":
            docs, stats = result
            self.table.setUpdatesEnabled(False)
            try:
                self.table.setRowCount(0)
                self._rows.clear()
                for doc in docs:
                    self._upsert_row(doc)
            finally:
                self.table.setUpdatesEnabled(True)
            self.stats.setText(
                f"{stats['documents']} documents · {stats['chunks']} chunks · "
                f"{stats['unique_chunks']} unique · "
                f"cache {stats['embed_hits']}/{stats['embed_hits'] + stats['embed_misses']} hits")
        if kind == "import":
            self.refresh()  # include files indexed before a partial failure too

    def _run_search(self) -> None:
        self.results.clear()
        if self.engine is None:
            return
        query = self.search_box.text().strip()
        self._search_query = query
        self._search_generation += 1
        if not query:
            return
        if "search" not in self._jobs:
            self._begin_search()

    def _begin_search(self) -> None:
        self._submit("search", self._search_generation, self.engine.search,
                     self._search_query, limit=8)

    # ── table + stats ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        if self.engine is None:
            return
        if "refresh" in self._jobs:
            self._refresh_pending = True
            return
        self._submit("refresh", 0, _library_snapshot, self.engine)

    def _upsert_row(self, doc: dict) -> None:
        path = doc.get("path", "")
        name = Path(path).name if path else doc.get("doc_id", "?")
        # Replace an existing row for the same path, else append.
        target_row = self._rows.get(path)
        if target_row is None:
            target_row = self.table.rowCount()
            self._rows[path] = target_row
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
