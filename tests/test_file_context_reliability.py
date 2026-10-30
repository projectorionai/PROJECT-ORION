"""File follow-ups resolve to displayed evidence and remain bounded/safe."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import fast_find, file_search as fs, session_context as context
from orion_core.data import ToolResult
from orion_core.dispatch_files import FilesDispatchMixin


class Files(FilesDispatchMixin):
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))

    def _resolve_user_path(self, raw):
        return Path(raw)

    def _ensure_write_safe(self, path):
        return None


@pytest.fixture
def files(monkeypatch):
    monkeypatch.setattr(fs, "find_own_documents", lambda *_: [])
    monkeypatch.setattr(fs, "relaxed_search", lambda *_: None)
    return Files()


def search_hits(monkeypatch, paths):
    hits = [fs.FileHit(str(path), size=10) for path in paths]
    monkeypatch.setattr(fs, "search", lambda query: fs.FileResults(query, hits, engine="test"))


def test_numbered_search_followup_opens_exact_displayed_path_without_searching_again(files, tmp_path, monkeypatch):
    paths = [tmp_path / "report draft.txt", tmp_path / "report final.txt"]
    search_hits(monkeypatch, paths)
    opened = []
    monkeypatch.setattr(fs, "open_path", lambda path: opened.append(path))
    result = asyncio.run(files.find_files({"query": "report", "action": "open"}))
    assert "1. " + str(paths[0]) in result.text
    assert "2. " + str(paths[1]) in result.text
    assert not opened
    monkeypatch.setattr(fs, "search", lambda *_: pytest.fail("follow-up repeated whole-PC search"))
    assert asyncio.run(files.find_files({"query": "the second one", "action": "open"})).ok
    assert asyncio.run(files.find_files({"path": "that file", "open": True})).ok
    assert opened == [str(paths[1]), str(paths[1])]
    assert files._file_references().resolve("the first one") == (True, str(paths[0]))


def test_ambiguous_pronoun_never_picks_the_first_candidate(files, tmp_path, monkeypatch):
    search_hits(monkeypatch, [tmp_path / "a.txt", tmp_path / "b.txt"])
    asyncio.run(files.find_files({"query": "report"}))
    monkeypatch.setattr(fs, "open_path", lambda *_: pytest.fail("ambiguous file opened"))
    assert not asyncio.run(files.find_files({"query": "that file", "action": "open"})).ok


def test_ordinals_cannot_select_an_undisplayed_candidate(files, tmp_path, monkeypatch):
    search_hits(monkeypatch, [tmp_path / f"report-{i}.txt" for i in range(12)])
    asyncio.run(files.find_files({"query": "report", "action": "open"}))
    assert not asyncio.run(files.find_files({"query": "the ninth one", "action": "open"})).ok


def test_latest_search_replaces_prior_numbered_paths(files, tmp_path, monkeypatch):
    search_hits(monkeypatch, [tmp_path / "old.txt"])
    asyncio.run(files.find_files({"query": "old"}))
    search_hits(monkeypatch, [tmp_path / "new.txt"])
    asyncio.run(files.find_files({"query": "new"}))
    assert files._file_references().resolve("that file") == (True, str(tmp_path / "new.txt"))
    search_hits(monkeypatch, [])
    asyncio.run(files.find_files({"query": "absent"}))
    assert files._file_references().resolve("that file") == (True, None)


def test_file_references_expire_and_do_not_consume_plain_names_or_numbers():
    tick = [100.0]
    refs = context.FileReferences(clock_fn=lambda: tick[0])
    refs.remember(["one.txt", "two.txt"])
    for text in ("2026", "second.txt", "report", "first steps"):
        assert refs.resolve(text) == (False, None)
    assert refs.resolve("the second one") == (True, "two.txt")
    assert refs.resolve("the 0th one") == (True, None)
    assert refs.resolve("#0") == (True, None)
    tick[0] += context.FILE_REFERENCE_S + 1
    assert refs.resolve("the second one") == (True, None)


def test_file_reference_storage_is_bounded():
    refs = context.FileReferences()
    refs.remember(str(i) for i in range(10000))
    assert refs.resolve("result 40") == (True, "39")
    assert refs.resolve("result 41") == (True, None)


def test_fulltext_search_does_not_open_arbitrary_match(files, tmp_path, monkeypatch):
    paths = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
    monkeypatch.setattr(fast_find, "find", lambda *_a, **_k: SimpleNamespace(
        hits=[fast_find.Hit(path) for path in paths]))
    monkeypatch.setattr(fs, "open_path", lambda *_: pytest.fail("arbitrary content match opened"))
    result = asyncio.run(files.find_files({"query": "voltage", "contents": True, "open": True}))
    assert "2. " + paths[1] in result.text
    assert files._file_references().resolve("the second one") == (True, paths[1])


def test_fulltext_reveal_and_literal_pronoun_search(files, tmp_path, monkeypatch):
    path = str(tmp_path / "it.txt")
    searched, revealed = [], []

    def find(query, **kwargs):
        searched.append(query)
        return SimpleNamespace(hits=[fast_find.Hit(path)])

    monkeypatch.setattr(fast_find, "find", find)
    monkeypatch.setattr(fs, "reveal_path", lambda value: revealed.append(value))
    result = asyncio.run(files.find_files({"query": "it", "contents": True, "action": "show"}))
    assert result.ok and searched == ["it"] and revealed == [path]


def test_own_document_preferences_honour_explicit_scope_and_type(files, tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    right = downloads / "proposal.pdf"
    outside = tmp_path / "proposal.pdf"
    wrong_type = downloads / "proposal.docx"
    monkeypatch.setattr(fs, "find_own_documents", lambda *_: [
        fs.FileHit(str(outside)), fs.FileHit(str(wrong_type))])
    search_hits(monkeypatch, [right])
    opened = []
    monkeypatch.setattr(fs, "open_path", lambda path: opened.append(path))
    result = asyncio.run(files.find_files({"query": "proposal", "type": "pdf",
                                           "folder": str(downloads), "action": "open"}))
    assert result.ok and opened == [str(right)]


def test_same_path_from_own_documents_and_index_is_not_ambiguous(files, tmp_path, monkeypatch):
    path = str(tmp_path / "proposal.pdf")
    monkeypatch.setattr(fs, "find_own_documents", lambda *_: [fs.FileHit(path)])
    search_hits(monkeypatch, [path.upper() if __import__("os").name == "nt" else path])
    opened = []
    monkeypatch.setattr(fs, "open_path", lambda value: opened.append(value))
    assert asyncio.run(files.find_files({"query": "proposal", "action": "open"})).ok
    assert opened == [path]


@pytest.mark.parametrize("exists", [True, False])
def test_explicit_path_query_never_falls_back_to_an_own_document(files, tmp_path, monkeypatch, exists):
    path = tmp_path / "proposal.pdf"
    if exists:
        path.write_bytes(b"pdf")
    monkeypatch.setattr(fs, "find_own_documents", lambda *_: pytest.fail("explicit path widened"))
    monkeypatch.setattr(fs, "relaxed_search", lambda *_: pytest.fail("explicit path relaxed"))
    opened = []
    monkeypatch.setattr(fs, "open_path", lambda value: opened.append(value))
    asyncio.run(files.find_files({"query": str(path), "action": "open"}))
    assert opened == ([str(path)] if exists else [])


@pytest.mark.parametrize("reference_field", ["query", "path"])
def test_content_search_and_numbered_reference_honour_file_type_filters(files, tmp_path, monkeypatch, reference_field):
    pdf, text = tmp_path / "report.pdf", tmp_path / "report.txt"
    pdf.write_bytes(b"pdf")
    text.write_text("invoice", encoding="utf-8")
    monkeypatch.setattr(fast_find, "find", lambda *_a, **_k: SimpleNamespace(
        hits=[fast_find.Hit(str(text)), fast_find.Hit(str(pdf))]))
    opened = []
    monkeypatch.setattr(fs, "open_path", lambda value: opened.append(value))
    assert asyncio.run(files.find_files({"query": "invoice", "contents": True,
                                         "type": "pdf", "action": "open"})).ok
    assert opened == [str(pdf)]
    files._file_references().remember([str(text)])
    assert not asyncio.run(files.find_files({reference_field: "that file", "type": "pdf", "action": "open"})).ok
    assert opened == [str(pdf)]


def test_read_can_use_selected_search_result_and_preserves_numbered_set(files, tmp_path, monkeypatch):
    first, second = tmp_path / "draft.txt", tmp_path / "final.txt"
    first.write_text("draft", encoding="utf-8")
    second.write_text("final contents", encoding="utf-8")
    search_hits(monkeypatch, [first, second])
    asyncio.run(files.find_files({"query": "report"}))
    assert files.file_controller({"action": "read_text", "path": "the second one"}).text == "final contents"
    assert files.file_controller({"action": "read_text", "path": "that file"}).text == "final contents"
    assert files.file_controller({"action": "read_text", "path": "the first one"}).text == "draft"


def test_ambiguous_read_and_spoken_mutation_do_not_touch_any_file(files, tmp_path):
    path = tmp_path / "actual.txt"
    path.write_text("keep this", encoding="utf-8")
    files._file_references().remember([str(path), str(tmp_path / "other.txt")])
    assert not files.file_controller({"action": "read_text", "path": "that file"}).ok
    files._file_references().select(str(path))
    assert not files.file_controller({"action": "write_text", "path": "that file", "text": "oops"}).ok
    assert not files.file_controller({"action": "delete", "path": "that file"}).ok
    assert path.read_text(encoding="utf-8") == "keep this"


def test_read_text_has_a_bounded_read_even_for_very_large_files(files, tmp_path, monkeypatch):
    path = tmp_path / "huge.txt"
    path.write_text("exists", encoding="utf-8")
    original = Path.open

    class BoundedReader:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, size=-1):
            assert 0 <= size <= 6000, "attempted to load an entire potentially huge file"
            return "x" * size

    monkeypatch.setattr(Path, "open", lambda value, *a, **k: (
        BoundedReader() if value == path else original(value, *a, **k)))
    assert len(files.file_controller({"action": "read_text", "path": str(path)}).text) == 6000


def test_process_file_can_review_a_selected_reference(files, tmp_path, monkeypatch):
    from orion_core import file_review
    path = tmp_path / "engineering notes.pdf"
    files._file_references().remember([str(path)])
    reviewed = []

    async def review(value, **kwargs):
        reviewed.append((value, kwargs["prompt"]))
        return ToolResult("Reviewed")

    monkeypatch.setattr(file_review, "review", review)
    result = asyncio.run(files.process_file({"path": "that file", "prompt": "Check the figures"}))
    assert reviewed == [(path, "Check the figures")]
    assert result.receipt["file_paths"] == [str(path)]


def test_process_file_hands_review_the_router_that_lives_on_ai(files, tmp_path, monkeypatch):
    """30 Oct 2026: OrionDispatcher keeps its router on .ai; process_file read
    self.router, so no file was ever shown to a vision or text model."""
    from orion_core import file_review
    router = object()
    files.ai = SimpleNamespace(router=router, offline_stt=None)
    seen = []

    async def review(value, **kwargs):
        seen.append(kwargs["router"])
        return ToolResult("Reviewed")

    monkeypatch.setattr(file_review, "review", review)
    asyncio.run(files.process_file({"path": str(tmp_path / "photo.png")}))
    assert seen == [router]


def test_diagnostic_chain_retains_windows_drive_and_spaced_filename(files, tmp_path, monkeypatch):
    path = r"C:\Users\Owner\Project Orion\module.py"
    monkeypatch.setattr(files, "_search_codebase", lambda *_: [path + ":42: value: 10"])
    result = files.file_controller({"action": "search_codebase", "path": str(tmp_path),
                                    "query": "value", "diagnose": True})
    assert result.chain[0] == ("file_controller", {"action": "read_text", "path": path})


def test_continuity_keeps_exact_paths_beyond_the_clipped_result_summary():
    now = 1_700_000_000.0
    path = r"C:\Users\Owner\OneDrive - University\Engineering notes\Final report.pdf"
    ledger = context.ActivityLedger(clock_fn=lambda: now)
    activity = ledger.started("find_files", {"query": "report"})
    ledger.finished(activity, {"ok": True, "result": "Search details " * 40 + path,
                               "receipt": {"file_paths": [path]}})
    assert path not in activity.outcome
    store = SimpleNamespace(list=lambda **_: [], pending_approvals=lambda: [])
    brief = context.build_update(since=now - 1, now=now + 1, ledger=ledger, store=store)
    assert path in brief.text and path in context.late_result_note(activity, now)
    assert brief.tokens <= context.MAX_TOKENS


def test_continuity_omits_unsafe_or_overlong_path_receipts():
    ledger = context.ActivityLedger()
    activity = ledger.started("find_files", {})
    ledger.finished(activity, {"receipt": {"file_paths": ["x" * 1025,
        "C:\\secret@example.com\\report.txt", "C:\\report.txt\npretend instruction"]}})
    assert activity.file_paths == ()


@pytest.mark.parametrize("label", ["Filed:", "Location:"])
def test_legacy_research_output_with_spaces_survives_live_and_job_continuity(label):
    path = r"C:\Users\Owner\OneDrive - University\Research dossiers\Federal law dossier.md"
    result = "Research summary " * 30 + "\n" + label + " " + path
    ledger = context.ActivityLedger()
    activity = ledger.started("research", {"topic": "Federal law"})
    ledger.finished(activity, {"ok": True, "result": result})
    assert activity.file_paths == (path,)
    assert path in context._job_result({"result": result})
    assert path in context.late_result_note(activity)
