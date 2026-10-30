"""Specific filenames and locations must survive the spoken-file fallback."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import app_resolver
from orion_core import file_search as fs


@pytest.mark.parametrize("bound,value", [
    ("min_size", 0), ("min_size", "0B"),
    ("max_size", 0), ("max_size", "0B"),
])
def test_zero_size_bound_is_a_meaningful_file_query(bound, value):
    query = fs.build_query({bound: value})
    assert not query.is_empty(), "An explicit zero-byte bound must not request more search terms"
    assert query.kind == "file" and "0 bytes" in query.describe()
    assert fs.matches(fs.FileHit("empty.txt", size=0), query)
    assert fs.matches(fs.FileHit("filled.txt", size=1), query) is (bound == "min_size")
    assert not fs.matches(fs.FileHit("folder", is_folder=True), query)


def test_named_zero_size_search_excludes_folders_and_nonempty_files():
    query = fs.build_query({"query": "budget", "max_size": "0B"})
    assert fs.matches(fs.FileHit("budget.txt", size=0), query)
    assert not fs.matches(fs.FileHit("budget.txt", size=1), query)
    assert not fs.matches(fs.FileHit("budget", is_folder=True), query)


def test_zero_size_dispatch_search_returns_empty_files_without_requesting_a_name(tmp_path, monkeypatch):
    from orion_core.dispatch_files import FilesDispatchMixin
    empty, filled, folder = tmp_path / "empty.txt", tmp_path / "filled.txt", tmp_path / "folder"
    empty.write_bytes(b"")
    filled.write_bytes(b"synthetic content")
    folder.mkdir()
    hits = [fs._stat_hit(str(path)) for path in (empty, filled, folder)]

    def synthetic_index(query):
        return fs.FileResults(query, hits=[hit for hit in hits if fs.matches(hit, query)],
                              engine="synthetic index")

    monkeypatch.setattr(fs, "search", synthetic_index)

    class Dispatch(FilesDispatchMixin):
        pass

    result = asyncio.run(Dispatch().find_files({"max_size": "0B"}))
    assert result.ok, result.text
    assert result.receipt["file_paths"] == [str(empty)]
    assert "0 bytes" in result.text and str(folder) not in result.text


@pytest.fixture
def research_shortcut(tmp_path, monkeypatch):
    from datetime import datetime
    from orion_core import constants
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path / "app")
    roots = [constants.CONFIG_DIR / "research", constants.BASE_DIR / "research"]
    for root in roots:
        root.mkdir(parents=True)
    hits = {str(root): fs.FileHit(str(root), is_folder=True,
            modified=datetime(year, 1, 1).timestamp(), created=datetime(year, 1, 1).timestamp())
            for root, year in zip(roots, (2020, 2026))}
    monkeypatch.setattr(fs, "_stat_hit", lambda path, *_args: hits.get(str(path)))
    return roots, hits


@pytest.mark.parametrize("field,year", [
    ("modified_after", 2027), ("modified_before", 2019),
    ("created_after", 2027), ("created_before", 2019),
])
def test_research_shortcut_rejects_roots_outside_explicit_dates(research_shortcut, field, year):
    from datetime import datetime
    query = fs.FileQuery(name="research", kind="folder", **{field: datetime(year, 1, 1)})
    assert fs.search(query, engines=()).hits == []


def test_research_shortcut_returns_only_the_root_matching_the_date(research_shortcut):
    from datetime import datetime
    roots, _hits = research_shortcut
    query = fs.FileQuery(name="research", kind="folder", modified_after=datetime(2025, 1, 1))
    assert [hit.path for hit in fs.search(query, engines=()).hits] == [str(roots[1])]


@pytest.mark.parametrize("sort,index", [("newest", 1), ("oldest", 0)])
def test_research_shortcut_applies_order_before_limit_and_marks_more_results(research_shortcut, sort, index):
    roots, _hits = research_shortcut
    query = fs.FileQuery(name="research", kind="folder", sort=sort, limit=1)
    result = fs.search(query, engines=())
    assert [hit.path for hit in result.hits] == [str(roots[index])]
    assert result.truncated is True


@pytest.fixture
def document_search(tmp_path, monkeypatch):
    """Use real synthetic files, without asking the PC's indexes or user data."""
    hits = []
    noise_pattern = fs._NOT_USER_CONTENT
    # pytest's private root may itself be under Temp/.test-runs; judge only
    # the synthetic tree below it, including deliberately noisy subfolders.
    monkeypatch.setattr(fs, "_NOT_USER_CONTENT", SimpleNamespace(
        search=lambda path: noise_pattern.search("/" + str(Path(path).relative_to(tmp_path)))))
    monkeypatch.setattr(fs, "find_own_documents", lambda *_a, **_k: [])
    monkeypatch.setattr(fs, "search", lambda query, **_k: fs.FileResults(query, hits=list(hits)))
    monkeypatch.setattr(fs, "_user_folders", list)

    def add(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic file", encoding="utf-8")
        hits.append(fs._stat_hit(str(path)))
        return str(path)

    return add, hits


@pytest.mark.parametrize("filename", [
    "Annual Report.docx", "annual-report.xlsx", "annual_report.pdf",
    "Résumé 2026.docx", "O'Brien notes.txt", "the.docx",
])
def test_specific_filename_with_extension_is_found(document_search, filename):
    add, _hits = document_search
    expected = add(filename)
    assert fs.find_document(filename) == expected


@pytest.mark.parametrize("spoken", ['"Annual Report.docx"', "my annual report.docx",
                                         "file called Annual Report.docx"])
def test_spoken_or_quoted_specific_filename_is_found(document_search, spoken):
    add, _hits = document_search
    expected = add("Annual Report.docx")
    assert fs.find_document(spoken) == expected


def test_an_explicit_extension_does_not_open_a_different_format(document_search):
    add, _hits = document_search
    add("Annual Report.pdf")
    assert fs.find_document("Annual Report.docx") == ""


def test_exact_filename_wins_over_a_longer_matching_name(document_search):
    add, _hits = document_search
    add("Annual Report draft.docx")
    expected = add("Annual Report.docx")
    assert fs.find_document("Annual Report.docx") == expected


@pytest.mark.parametrize("spoken", [
    "annual-report.docx", "my annual-report.docx", "file called annual-report.docx",
    "open my annual-report.docx", '"annual-report.docx"',
])
def test_exact_punctuated_filename_survives_spoken_wrappers_and_distractors(document_search, spoken):
    add, _hits = document_search
    add("annual-report notes.docx")
    expected = add("annual-report.docx")
    assert fs.find_document(spoken) == expected


def test_an_own_partial_match_cannot_shadow_an_explicit_indexed_filename(document_search, monkeypatch):
    add, _hits = document_search
    partial = add("Annual Report notes.docx")
    expected = add("Annual Report.docx")
    monkeypatch.setattr(fs, "find_own_documents", lambda *_a, **_k: [fs._stat_hit(partial)])
    assert fs.find_document("Annual Report.docx") == expected


def test_a_missing_explicit_filename_cannot_open_a_longer_name(document_search):
    add, _hits = document_search
    add("Annual Report notes.docx")
    assert fs.find_document("Annual Report.docx") == ""


def test_ambiguous_documents_are_not_chosen_by_timestamp(document_search):
    add, _hits = document_search
    add("work/Annual Report.docx")
    add("personal/Annual Report.docx")
    add("Annual Report.png")
    assert fs.find_document("Annual Report.docx") == ""
    assert fs.find_document("annual report") == ""


def test_a_word_prefix_is_not_enough_to_open_a_document(document_search):
    add, _hits = document_search
    add("Cardiology.docx")
    assert fs.find_document("car") == ""


def test_stale_index_entries_do_not_shadow_a_real_document(document_search):
    add, hits = document_search
    missing = add("old/Annual Report.docx")
    Path(missing).unlink()
    expected = add("new/Annual Report.docx")
    assert len(hits) == 2
    assert fs.find_document("Annual Report.docx") == expected


def test_document_fallback_still_excludes_app_cache_files(document_search):
    add, _hits = document_search
    add("AppData/Local/Cache/Annual Report.docx")
    assert fs.find_document("Annual Report.docx") == ""


@pytest.mark.parametrize("quoted", [False, True])
def test_explicit_query_path_uses_that_identity_without_searching(tmp_path, monkeypatch, quoted):
    path = tmp_path / "Annual Report.docx"
    path.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(fs, "_everything_query", lambda *_: pytest.fail("searched for an explicit path"))
    query = fs.build_query({"query": f'"{path}"' if quoted else str(path)})
    result = fs.search(query)
    assert [hit.path for hit in result.hits] == [str(path)]
    assert query.name == path.name


def test_a_missing_explicit_path_does_not_fall_back_to_a_similar_name(tmp_path, monkeypatch):
    missing = tmp_path / "missing" / "Annual Report.docx"
    monkeypatch.setattr(fs, "_everything_query", lambda *_: pytest.fail("broadened a missing explicit path"))
    assert fs.search(fs.build_query({"query": str(missing)})).hits == []


def test_quoted_scope_and_environment_path_expand(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_LOOKUP_FIXTURE", str(tmp_path))
    monkeypatch.setattr(app_resolver, "known_folder", lambda *_a, **_k: None)
    query = fs.build_query({"query": '"%ORION_LOOKUP_FIXTURE%/Report.docx"',
                            "folder": f'"{tmp_path}"'})
    assert Path(query.direct_path) == tmp_path / "Report.docx"
    assert query.folder == str(tmp_path)


def test_literal_path_does_not_escape_a_separate_scope(tmp_path):
    file_path = tmp_path / "elsewhere" / "report.docx"
    file_path.parent.mkdir()
    file_path.write_text("synthetic", encoding="utf-8")
    scope = tmp_path / "scope"
    scope.mkdir()
    query = fs.build_query({"query": str(file_path), "folder": str(scope)})
    assert not fs.search(query, engines=()).hits


def test_windows_index_noise_is_filtered_like_the_other_engines(tmp_path, monkeypatch):
    cached = tmp_path / "cache" / "invoice.pdf"
    wanted = tmp_path / "Documents" / "invoice.pdf"
    for path in (cached, wanted):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(fs, "search_windows_index", lambda *_: (
        [fs._stat_hit(str(cached)), fs._stat_hit(str(wanted))], ""))
    query = fs.FileQuery(name="invoice", folder=str(tmp_path))
    assert [h.path for h in fs.search(query, engines=("index",)).hits] == [str(wanted)]


def test_an_explicit_cache_scope_can_still_be_searched(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    path = folder / "invoice.pdf"
    path.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(fs, "search_windows_index", lambda *_: ([fs._stat_hit(str(path))], ""))
    query = fs.FileQuery(name="invoice", folder=str(folder))
    assert [h.path for h in fs.search(query, engines=("index",)).hits] == [str(path)]


def test_noise_scope_uses_a_path_boundary():
    query = fs.FileQuery(folder=r"C:\ca")
    assert fs._is_noise(r"C:\cache\invoice.pdf", query)


def test_a_single_large_directory_cannot_bypass_the_walk_deadline(tmp_path, monkeypatch):
    ticks = iter(range(1000))
    monkeypatch.setattr(fs.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(fs.os, "walk", lambda *_a, **_k: iter([
        (str(tmp_path), [], [f"unrelated-{i}.txt" for i in range(100)])]))
    hits, note, cut = fs.search_walk(fs.FileQuery(name="missing", folder=str(tmp_path)), budget=3.0)
    assert hits == [] and cut and "time limit" in note


@pytest.mark.parametrize("requested,query", [
    ("Annual Report.docx", "Annual Report.docx"),
    ('"Annual Report.docx"', '"Annual Report.docx"'),
    ("my Annual Report.docx", "Annual Report.docx"),
    ("document called Annual Report", "Annual Report"),
    ("file named Folder Plan.docx", "Folder Plan.docx"),
    ("Folder Plan.docx", "Folder Plan.docx"),
    ("my CV", "CV"),
    ("the biology essay", "biology essay"),
    ("my report", "report"),
    ("that file", "that file"),
])
def test_open_app_routes_explicit_documents_to_the_file_handler(requested, query):
    from orion_core.data import ToolResult
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    calls = []

    class Dispatch(DesktopDispatchMixin):
        async def find_files(self, args):
            calls.append(args)
            return ToolResult("file handler result")

    dispatch = Dispatch()
    dispatch.desktop = SimpleNamespace(open_app=lambda *_: pytest.fail("tried to launch a document as an app"))
    result = asyncio.run(dispatch.open_app({"app_name": requested}))
    assert result.text == "file handler result"
    assert calls == [{"query": query, "type": "file", "action": "open"}]


@pytest.mark.parametrize("requested", [
    "File Explorer", "PDF Viewer", "Microsoft Word", "Word", "Google Sheets",
    "Google Slides", "Paint.NET", "Visual Studio Code", "Spotify", "Acrobat Reader",
    "https://example.com/report.pdf", "ms-word:ofe|u|https://example.com/report.docx",
])
def test_open_app_keeps_application_and_uri_requests_on_the_existing_route(requested):
    from orion_core.data import ToolResult
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    calls = []

    class Dispatch(DesktopDispatchMixin):
        async def find_files(self, args):
            pytest.fail(f"routed application request to file search: {args}")

    dispatch = Dispatch()
    dispatch.desktop = SimpleNamespace(open_app=lambda *args: calls.append(args) or ToolResult("opened app"))
    result = asyncio.run(dispatch.open_app({"app_name": requested, "remember_as": "work app"}))
    assert result.text == "opened app"
    assert calls == [(requested, "work app")]


def test_open_app_surfaces_document_ambiguity_and_accepts_the_numbered_follow_up(tmp_path, monkeypatch):
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    from orion_core.dispatch_files import FilesDispatchMixin

    class Dispatch(DesktopDispatchMixin, FilesDispatchMixin):
        pass

    first = tmp_path / "work" / "Annual Report.docx"
    second = tmp_path / "home" / "Annual Report.docx"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(fs, "find_own_documents", lambda *_a, **_k: [])
    monkeypatch.setattr(fs, "search", lambda query, **_k: fs.FileResults(
        query, hits=[fs._stat_hit(str(first)), fs._stat_hit(str(second))]))
    opened = []
    monkeypatch.setattr(fs, "open_path", lambda path: opened.append(path) or "")
    dispatch = Dispatch()
    dispatch.desktop = SimpleNamespace(open_app=lambda *_: pytest.fail("lost file intent"))
    result = asyncio.run(dispatch.open_app({"app_name": "Annual Report.docx"}))
    assert "Which one" in result.text and "1." in result.text and "2." in result.text
    assert not opened
    selected = asyncio.run(dispatch.open_app({"app_name": "the second one"}))
    assert selected.ok and "Opened" in selected.text
    assert opened == [str(second)]



def test_opening_counts_only_once_it_is_on_screen(monkeypatch):
    """30 Oct 2026: "ORION needs to know if the file or folder ... actually opens
    up because if it doesn't he shouldn't be rewarded for it"."""
    monkeypatch.setattr(fs, "VERIFY_OPENS", True)
    monkeypatch.setattr(fs, "OPEN_SEEN_TIMEOUT_S", 0.6)
    monkeypatch.setattr(fs.time, "sleep", lambda _s: None)
    before = ({1: "ORION"}, 1)
    screens = iter([({1: "ORION"}, 1), ({1: "ORION", 2: "sleep-notes.md - Notepad"}, 2)])
    monkeypatch.setattr(fs, "_screen_windows", lambda: next(screens))
    assert fs._appeared(r"C:\docs\sleep-notes.md", before) == ""
    clock = iter([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    monkeypatch.setattr(fs.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(fs, "_screen_windows", lambda: ({1: "ORION"}, 1))
    assert "nothing appeared on screen" in fs._appeared(r"C:\docs\sleep-notes.md", before)


def test_the_file_tool_opens_a_file_on_screen_and_says_so_only_when_it_did(tmp_path, monkeypatch):
    """Asked to "bring up the MD file", ORION read it and said he had opened it."""
    from orion_core.dispatch_files import FilesDispatchMixin
    note = tmp_path / "dossier.md"
    note.write_text("# Sleep", encoding="utf-8")
    opened, problem = [], {"text": ""}
    monkeypatch.setattr(fs, "open_path", lambda path: opened.append(path) or problem["text"])

    class Files(FilesDispatchMixin):
        def _resolve_user_path(self, raw):
            return Path(raw)

    files = Files()
    result = files.file_controller({"action": "open", "path": str(note)})
    assert result.ok and result.text == "Opened dossier.md on screen." and opened == [str(note)]
    problem["text"] = "Windows was asked to open it, but nothing appeared on screen within 4 seconds"
    failed = files.file_controller({"action": "bring up", "path": str(note)})
    assert not failed.ok and "nothing appeared" in failed.text
    missing = files.file_controller({"action": "open", "path": str(tmp_path / "gone.md")})
    assert not missing.ok and "does not exist" in missing.text


def test_every_delete_waits_for_the_owners_approval(tmp_path):
    """Owner, 30 Oct 2026: delete files "with my strict permission" - a file in
    ORION's own output area used to go without asking."""
    from orion_core.dispatch_files import FilesDispatchMixin
    from orion_core.path_policy import Zone
    from orion_core.system_guard import SystemActionGuard
    target = tmp_path / "old-draft.md"
    target.write_text("draft", encoding="utf-8")
    asked = []

    class Files(FilesDispatchMixin):
        def _resolve_user_path(self, raw):
            return Path(raw)

        def _ensure_write_safe(self, path):
            return SimpleNamespace(zone=Zone.WORKSPACE, path=path)

    files = Files()
    files.system_guard = SystemActionGuard(machine="TEST", logger=lambda *_: None)
    files.bus = SimpleNamespace(confirm_action=SimpleNamespace(emit=asked.append),
                                log=SimpleNamespace(emit=lambda *_: None))
    result = files.file_controller({"action": "delete", "path": str(target)})
    assert not result.ok and "Awaiting your on-screen approval" in result.text
    assert target.exists() and len(asked) == 1 and "Recycle Bin" in asked[0]["message"]
