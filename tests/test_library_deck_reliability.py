"""Library actions must leave Qt responsive and recover after failures."""
import threading
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from orion_core.gui.library_deck import LibraryDeckView


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(qapp, until):
    deadline = time.monotonic() + 3
    while not until() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.002)
    assert until(), "library operation did not finish"


class Engine:
    def library(self, **kwargs):
        return []

    def stats(self):
        return dict(documents=0, chunks=0, unique_chunks=0, embed_hits=0, embed_misses=0)

    def search(self, query, **kwargs):
        return [dict(path="source.txt", score=.8, snippet=query)]

    def ingest_file(self, target):
        return SimpleNamespace(status="indexed", to_dict=lambda: dict(path=target, status="indexed"))


def ready(qapp, engine):
    view = LibraryDeckView(None, engine)
    pump(qapp, lambda: bool(view.stats.text()))
    return view


def test_library_loading_and_search_run_off_the_gui_thread(qapp):
    main_thread = threading.get_ident()
    threads = []

    class Recording(Engine):
        def library(self, **kwargs):
            threads.append(threading.get_ident())
            return super().library(**kwargs)

        def stats(self):
            threads.append(threading.get_ident())
            return super().stats()

        def search(self, query, **kwargs):
            threads.append(threading.get_ident())
            return super().search(query, **kwargs)

    view = ready(qapp, Recording())
    view.search_box.setText("query")
    view._run_search()
    pump(qapp, lambda: view.results.count() == 1)
    assert len(threads) == 3
    assert main_thread not in threads
    view.deleteLater()


def test_a_new_query_wins_over_an_in_flight_search(qapp):
    gate = threading.Event()
    entered = threading.Event()
    queries = []

    class Slow(Engine):
        def search(self, query, **kwargs):
            queries.append(query)
            entered.set()
            gate.wait(.2)
            return super().search(query, **kwargs)

    view = ready(qapp, Slow())
    try:
        for query in ("old", "middle", "new"):
            view.search_box.setText(query)
            view._run_search()
        gate.set()
        pump(qapp, lambda: view.results.count() == 1 and "new" in view.results.item(0).text())
        assert "middle" not in queries, "superseded searches must not fill the worker queue"
    finally:
        gate.set()
        view.deleteLater()


def test_search_failure_is_visible_and_a_retry_works(qapp):
    class Broken(Engine):
        def search(self, query, **kwargs):
            if query == "bad":
                raise OSError("database temporarily unavailable")
            return super().search(query, **kwargs)

    view = ready(qapp, Broken())
    view.search_box.setText("bad")
    view._run_search()
    pump(qapp, lambda: "unavailable" in view.status.text())
    view.search_box.setText("good")
    view._run_search()
    pump(qapp, lambda: view.results.count() == 1)
    assert "good" in view.results.item(0).text()
    view.deleteLater()


def test_repeated_imports_release_worker_resources(qapp):
    view = ready(qapp, Engine())
    for i in range(5):
        view._start(f"file{i}.txt", False)
        pump(qapp, lambda: "indexed" in view.status.text())
    assert not view.findChildren(QThread), "finished import threads accumulate on the view"
    view.deleteLater()


def test_library_failure_keeps_the_existing_rows(qapp):
    engine = Engine()
    view = ready(qapp, engine)
    view._upsert_row(dict(path="keep.txt"))

    def broken(**kwargs):
        raise OSError("database unavailable")

    engine.library = broken
    view.refresh()
    pump(qapp, lambda: "unavailable" in view.status.text())
    assert view.table.rowCount() == 1
    view.deleteLater()


def test_destroying_a_page_during_import_is_safe(qapp, caplog):
    from PyQt6 import sip
    gate = threading.Event()
    entered = threading.Event()

    class Slow(Engine):
        def ingest_file(self, target):
            entered.set()
            gate.wait(2)
            return super().ingest_file(target)

    view = ready(qapp, Slow())
    view._start("document.txt", False)
    future = view._jobs["import"]
    try:
        pump(qapp, entered.is_set)
        sip.delete(view)
    finally:
        gate.set()
    pump(qapp, future.done)
    assert future.result() == "document.txt: indexed"
    assert "exception calling callback" not in caplog.text


def test_clearing_query_discards_in_flight_results(qapp):
    view = ready(qapp, Engine())
    view.search_box.setText("old")
    view._run_search()
    view.search_box.clear()
    view._run_search()
    pump(qapp, lambda: "search" not in view._jobs)
    assert view.results.count() == 0
    view.deleteLater()


def test_exception_without_a_message_is_still_reported(qapp):
    class Broken(Engine):
        def search(self, *args, **kwargs):
            raise ValueError()

    view = ready(qapp, Broken())
    view.search_box.setText("query")
    view._run_search()
    pump(qapp, lambda: "ValueError" in view.status.text())
    assert view.results.count() == 0
    view.deleteLater()


@pytest.mark.parametrize("folder", [False, True])
def test_real_ingestion_search_and_refresh_through_the_view(qapp, tmp_path, monkeypatch, folder):
    from orion_core import semantic
    from orion_core.ingestion import IngestionEngine

    monkeypatch.setattr(semantic.ENCODER, "ready", lambda: False)
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    source = source_dir / "astronomy.txt"
    source.write_text("Quasars are luminous galactic nuclei powered by accretion. " * 20,
                      encoding="utf-8")
    engine = IngestionEngine(db_path=tmp_path / "library.db")
    view = ready(qapp, engine)
    try:
        view._start(str(source_dir if folder else source), folder)
        pump(qapp, lambda: not view._jobs and view.table.rowCount() == 1)
        assert "1 documents" in view.stats.text()
        view.search_box.setText("quasars")
        view._run_search()
        pump(qapp, lambda: "search" not in view._jobs)
        assert view.results.count() >= 1
        assert "astronomy.txt" in view.results.item(0).text()
        view.refresh()
        pump(qapp, lambda: not view._jobs)
        assert view.table.rowCount() == 1
    finally:
        view.deleteLater()
        engine.close()
