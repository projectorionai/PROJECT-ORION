"""Reproductions found in the 15 October capability audit; all data is temporary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from orion_core import deep_research, research_docx
from orion_core.data import ToolResult
from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
from orion_core.memory import MemoryAgent, OrionMemoryMatrix
from orion_core.reminders import ReminderService
from orion_core.time_service import TIME
from orion_core.research import ResearchAgent
from orion_core.research_worker import ResearchWorker


class Signal:
    def __init__(self):
        self.sent = []

    def emit(self, *args):
        self.sent.append(args)


class Bus:
    def __getattr__(self, name):
        signal = Signal()
        setattr(self, name, signal)
        return signal


class Dispatcher(KnowledgeDispatchMixin):
    def __init__(self):
        self.bus = Bus()
        self.research = object()
        self.research_director = None


def _paper(body):
    return deep_research.Paper("Same topic", deep_research.Style.SYSTEMATIC, deep_research.Depth.BRIEF,
        sections=[deep_research.Section(1, "Findings", "brief", body, 2)],
        sources=[{"title": "Source", "url": "https://example.test"}])


@pytest.mark.parametrize("word_available", [True, False])
def test_repeated_same_day_paper_never_overwrites_prior_saved_result(tmp_path, monkeypatch, word_available):
    if word_available and not research_docx.available():
        pytest.skip("python-docx is unavailable")
    monkeypatch.setattr(research_docx, "available", lambda: word_available)
    first, first_note = research_docx.write_paper(_paper("Original result"), tmp_path)
    assert first is not None, first_note
    before = first.read_bytes()
    second, second_note = research_docx.write_paper(_paper("New result"), tmp_path)
    assert second is not None, second_note
    assert first != second
    assert first.read_bytes() == before
    assert second.is_file()


def test_parallel_papers_reserve_distinct_results_without_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(research_docx, "available", lambda: False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda i: research_docx.write_paper(_paper(f"Result {i}"), tmp_path), range(6)))
    paths = [path for path, _ in outputs]
    assert len(set(paths)) == 6
    assert all(path.is_file() for path in paths)
    assert not list(tmp_path.glob(".paper-*.tmp"))


def test_failed_paper_publication_preserves_previous_output_and_cleans_temporary(tmp_path, monkeypatch):
    monkeypatch.setattr(research_docx, "available", lambda: False)
    original, _ = research_docx.write_paper(_paper("Original result"), tmp_path)
    before = original.read_bytes()

    def denied(*_):
        raise PermissionError("synthetic publication denied")

    monkeypatch.setattr(research_docx.os, "replace", denied)
    result, note = research_docx.write_paper(_paper("New result"), tmp_path)
    assert result is None and "denied" in note
    assert original.read_bytes() == before
    assert list(tmp_path.iterdir()) == [original]


@pytest.mark.parametrize("action", ["paper", "browse"])
def test_waited_research_is_still_visible_and_cancellable(action, monkeypatch):
    dispatcher = Dispatcher()

    async def scenario():
        entered = asyncio.Event()

        async def work(*_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

        if action == "paper":
            monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: object())
            monkeypatch.setattr(dispatcher, "_finish_paper", work)
        else:
            monkeypatch.setattr(dispatcher, "_browse_research", work)
        task = asyncio.create_task(dispatcher.research_tool({"action": action, "topic": "Topic", "wait": True}))
        await entered.wait()
        status = await dispatcher.research_tool({"action": "status"})
        assert status.receipt["research_state"] == "running"
        assert status.receipt["action"] == action
        await dispatcher.research_tool({"action": "stop"})
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await dispatcher.research_tool({"action": "status"})).receipt["research_state"] == "cancelled"

    asyncio.run(scenario())


def test_completed_newer_run_does_not_hide_an_older_active_investigation(monkeypatch):
    dispatcher = Dispatcher()

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def dossier(topic, args, **kwargs):
            if topic == "Long running topic":
                entered.set()
                await release.wait()
            return ToolResult("Complete")

        monkeypatch.setattr(dispatcher, "_dossier", dossier)
        active = asyncio.create_task(dispatcher.research_tool({"action": "dossier", "topic": "Long running topic"}))
        await entered.wait()
        await dispatcher.research_tool({"action": "dossier", "topic": "Quick topic"})
        status = await dispatcher.research_tool({"action": "status"})
        try:
            assert status.receipt["research_state"] == "running"
            assert status.receipt["topic"] == "Long running topic"
        finally:
            release.set()
            await active

    asyncio.run(scenario())


@pytest.fixture
def memory(tmp_path, monkeypatch):
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path / "config", Bus())
    monkeypatch.setattr(MemoryAgent, "_new_transcript_path", lambda self: tmp_path / "transcript.jsonl")
    agent = MemoryAgent(matrix, Bus())
    yield agent, matrix
    matrix.close()


@pytest.mark.parametrize("query", ["", "retrieval needle"])
def test_tier_recall_filters_category_before_global_candidate_limits(memory, query):
    agent, matrix = memory
    agent.remember("project", "needle", "retrieval needle belongs to this project", project="Target")
    matrix.conn.execute("UPDATE intelligence SET updated_at = '2000-01-01T00:00:00Z'")
    for i in range(205):
        matrix.save("knowledge", f"noise_{i}", "retrieval needle belongs to a different category")
    result = agent.recall("project", query, project="Target")
    assert len(result) == 1 and result[0]["key_ref"] == "needle"


def test_session_recall_honours_query_and_limit(memory):
    agent, _ = memory
    agent.remember("session", "unrelated", "Other subject")
    agent.remember("session", "report_file", "Report path is in Documents")
    agent.remember("session", "report_source", "Report source is the local draft")
    found = agent.recall("session", "report", limit=1)
    assert len(found) == 1 and "report" in found[0]["key_ref"]
    assert agent.recall("session", "absent") == []


@pytest.mark.parametrize("phrase,clock,expected_hours", [
    ("submit the report tomorrow", "10am", 25),
    ("check the draft on Friday", "10am", 25),
    ("check the oven tonight", "8", 11),
])
def test_structured_clock_keeps_day_and_period_from_the_user_text(monkeypatch, phrase, clock, expected_hours):
    now = datetime(2026, 10, 15, 9, tzinfo=timezone.utc)  # Thursday
    monkeypatch.setattr(TIME, "_clock", lambda: now)
    service = ReminderService(Bus(), path=None)
    assert service.add(text=phrase, at=clock).ok
    due = service.active()[0]
    assert due.due_epoch - now.timestamp() == expected_hours * 3600


def test_dossier_preserves_builtin_source_extracts_and_passes_findings_to_summary(tmp_path, monkeypatch):
    prompts = []

    async def generate(prompt, **kwargs):
        prompts.append(prompt)
        if "Break the research topic" in prompt:
            return None, "What does the canary evidence say?"
        if "executive briefing" in prompt:
            assert "NOTE_CANARY_481" in prompt
            assert "https://example.test/canary" in prompt
            return None, "Grounded summary"
        if "3-4 things" in prompt:
            return None, "One grounded takeaway from the findings"
        assert "SOURCE_CANARY_739" in prompt
        return None, "NOTE_CANARY_481 is supported by the supplied source."

    agent = ResearchAgent(Bus(), SimpleNamespace(generate_text=generate), None)

    async def gather(_query):
        return [{"title": "Canary source", "url": "https://example.test/canary",
                 "snippet": "SOURCE_CANARY_739 is the unique source fact."}]

    monkeypatch.setattr(agent, "gather_sources", gather)
    dossier = asyncio.run(ResearchWorker(agent=agent, root=tmp_path).investigate("Synthetic canary", questions=1))
    assert dossier.summary == "Grounded summary"
    assert "NOTE_CANARY_481" in dossier.sections[0][1]
    assert dossier.sources[0].extract.startswith("SOURCE_CANARY_739")


def test_sourced_dossier_provider_failure_is_incomplete_not_success(tmp_path, monkeypatch):
    async def fail(*_a, **_k):
        raise RuntimeError("synthetic provider unavailable")

    agent = ResearchAgent(Bus(), SimpleNamespace(generate_text=fail), None)

    async def gather(_query):
        return [{"title": "A real synthetic source", "url": "https://example.test/source",
                 "snippet": "Actual retrieved facts"}]

    monkeypatch.setattr(agent, "gather_sources", gather)
    import orion_core.research_worker as worker_module
    monkeypatch.setattr(worker_module, "DOSSIER_DIR", tmp_path)
    dispatcher = Dispatcher()
    dispatcher.research = agent
    result = asyncio.run(dispatcher.research_tool({"action": "dossier", "topic": "Topic", "questions": 1}))
    assert not result.ok
    assert result.receipt["research_state"] == "failed"
    assert "incomplete" in result.text.lower()
    assert "provider unavailable" in result.text
    assert Path(result.receipt["selected_path"]).is_file(), "useful failure artifact should remain available"


def test_timed_runs_reserve_distinct_directories_without_replacing_notes(tmp_path, monkeypatch):
    agent = ResearchAgent(Bus(), None, None)
    monkeypatch.setattr(agent, "RESEARCH_ROOT", tmp_path)
    first = agent._make_folder("Same topic")
    (first / "notes" / "original.md").write_text("Original findings", encoding="utf-8")
    second = agent._make_folder("Same topic")
    assert first != second
    assert (first / "notes" / "original.md").read_text(encoding="utf-8") == "Original findings"
    assert not list((second / "notes").iterdir())


def test_browse_runs_reserve_distinct_directories_without_stale_notes(tmp_path):
    from orion_core.live_research import Investigation, LiveResearcher
    researcher = LiveResearcher(ask=None, root=tmp_path)
    run = Investigation(topic="Same topic")
    first = researcher.write_folder(run, "Original findings")
    before = (first / "SUMMARY.md").read_bytes()
    second = researcher.write_folder(run, "New findings")
    assert first != second and (first / "SUMMARY.md").read_bytes() == before


@pytest.mark.parametrize("provider_failed", [False, True])
def test_timed_run_uses_completed_notes_and_never_succeeds_on_provider_failure(tmp_path, monkeypatch, provider_failed):
    prompts = []

    async def generate(prompt, **kwargs):
        prompts.append(prompt)
        if provider_failed:
            raise RuntimeError("synthetic timed provider unavailable")
        if "executive briefing" in prompt:
            assert "NOTE_CANARY_929" in prompt
            assert "https://example.test/timed" in prompt
            return None, "Grounded final summary"
        assert "SOURCE_CANARY_828" in prompt
        return None, "NOTE_CANARY_929 is the saved finding."

    agent = ResearchAgent(Bus(), SimpleNamespace(generate_text=generate), None)

    async def questions(_topic):
        return ["What is the timed canary evidence?"]

    async def gather(_question):
        return [{"title": "Timed source", "url": "https://example.test/timed", "snippet": "SOURCE_CANARY_828"}]

    monkeypatch.setattr(agent, "_decompose", questions)
    monkeypatch.setattr(agent, "_gather_sources", gather)
    result = asyncio.run(agent._run("Synthetic topic", 1, tmp_path))
    assert result.ok is not provider_failed
    if provider_failed:
        assert result.receipt["research_state"] == "failed"
        assert "provider unavailable" in result.text
    else:
        assert "Grounded final summary" in (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")


def test_grounded_summary_bounds_notes_and_retains_source_prefix():
    seen = []

    async def generate(prompt, **kwargs):
        seen.append(prompt)
        return None, "Summary"

    agent = ResearchAgent(Bus(), SimpleNamespace(generate_text=generate), None)
    asyncio.run(agent._final_summary_from_notes("Topic", [
        (f"Question {i}", f"Source: https://example.test/{i}\n" + "x" * 20000) for i in range(30)]))
    assert len(seen[0]) < 13000 and "Source: https://example.test/0" in seen[0]


def test_dossier_summary_keeps_each_questions_own_sources_past_first_eight(tmp_path):
    summary_notes = []

    class Agent:
        async def _decompose(self, topic):
            return ["Question A", "Question B"]

        async def gather_sources(self, question):
            return [{"title": f"Source {question}-{i}", "url": f"https://example.test/{question[-1]}/{i}",
                     "snippet": f"Evidence for {question}"}
                    for i in range(8 if question == "Question A" else 1)]

        async def _synthesise(self, topic, question, sources):
            return f"Finding for {question}"

        async def _final_summary_from_notes(self, topic, notes):
            summary_notes.extend(notes)
            return "Summary grounded in each question's sources"

    asyncio.run(ResearchWorker(agent=Agent(), root=tmp_path).investigate("Topic", questions=2))
    first, second = dict(summary_notes)["Question A"], dict(summary_notes)["Question B"]
    assert "https://example.test/A/0" in first
    assert "https://example.test/B/0" in second
    assert "https://example.test/A/" not in second
    assert "https://example.test/B/" not in first


@pytest.mark.parametrize("topic,expected", [("Topic A", "Topic A"), ("Topic B", "Topic B"),
                                          ("", "Topic B"), ("Missing topic", None)])
@pytest.mark.parametrize("action", ["status", "stop"])
@pytest.mark.parametrize("tracked_b", [False, True])
def test_explicit_research_control_never_targets_an_unrelated_newer_legacy_run(topic, expected, action, tracked_b):
    class Task:
        cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    task_a, task_b = Task(), Task()
    dispatcher = Dispatcher()

    def legacy_status():
        return ToolResult("Researching Topic B", receipt={"topic": "Topic B", "research_state": "running"})

    def legacy_stop():
        task_b.cancel()
        return ToolResult("Stopping Topic B", receipt={"topic": "Topic B", "research_state": "cancelling"})

    dispatcher.research = SimpleNamespace(
        _active={"topic": "Topic B", "task": task_b, "started": 20.0},
        status=legacy_status, stop_research=legacy_stop)
    dispatcher._background_research_runs = [
        {"topic": "Topic A", "task": task_a, "started": 10.0, "result": None,
         "receipt": {"topic": "Topic A", "research_state": "running"}}]
    if tracked_b:
        dispatcher._background_research_runs.append(
            {"topic": "Topic B", "task": task_b, "started": 20.0, "result": None,
             "receipt": {"topic": "Topic B", "research_state": "running"}})
    result = asyncio.run(dispatcher.research_tool({"action": action, "topic": topic}))
    if expected is None:
        assert not result.ok and result.receipt["research_state"] == "not_found"
    else:
        assert result.receipt["topic"] == expected
    assert task_a.cancelled is (action == "stop" and expected == "Topic A")
    assert task_b.cancelled is (action == "stop" and expected == "Topic B")


# ── Mark XXXVII: news sources carry a date and the article's own words ───────
#
# RSS headlines arrived with empty snippets, so a dossier built on them read
# like the model's memory and called an old model "verified current".

_RSS = """<rss><channel>
<item><title><![CDATA[Synthetic Lab releases Model Nine]]></title>
<link>https://example.test/model-nine</link>
<pubDate>Wed, 22 Oct 2026 09:00:00 GMT</pubDate>
<description><![CDATA[<a href="x">Model Nine</a>&nbsp;<font>Synthetic Times</font>]]></description></item>
<item><title>Second synthetic story</title><link>https://example.test/unreadable</link>
<pubDate>Tue, 21 Oct 2026 08:00:00 GMT</pubDate></item>
<item><title>Third synthetic story</title><link>https://example.test/three</link></item>
<item><title>Fourth story beyond the excerpt limit</title><link>https://example.test/four</link></item>
</channel></rss>"""


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return _RSS


class _Session:
    def get(self, url, **_kw):
        assert url.startswith("https://news.google.com/rss/search?q=")
        return _Response()


def test_news_sources_carry_their_date_and_a_bounded_article_excerpt(monkeypatch):
    from orion_core import background_http, research
    fetched = []
    article = " ".join(["Synthetic Lab said Model Nine replaces Model Eight this week."] * 40)

    async def fetch_text(url, *, start_index=0, max_length=8000):
        fetched.append(url)
        if url.endswith("unreadable"):
            return ToolResult("Page retrieval failed: HTTP 403", ok=False)
        return ToolResult("Home | News | Sport\n" + article)

    monkeypatch.setattr(background_http, "fetch_text", fetch_text)
    agent = ResearchAgent(Bus(), None, None)
    items = asyncio.run(agent._news(_Session(), "model nine"))
    assert [i["title"] for i in items][:2] == ["Synthetic Lab releases Model Nine",
                                               "Second synthetic story"]
    assert sorted(fetched) == sorted(["https://example.test/model-nine",
                                      "https://example.test/unreadable",
                                      "https://example.test/three"]), \
        "only the top items are read, and never through a browser"
    first = items[0]["snippet"]
    assert first.startswith("Published Wed, 22 Oct 2026 09:00:00 GMT. Synthetic Lab said")
    assert "Home | News" not in first, "navigation lines are not article prose"
    assert len(first) <= len("Published Wed, 22 Oct 2026 09:00:00 GMT. ") \
        + research.NEWS_EXCERPT_CHARS + 2
    # An unreadable page keeps what the feed said, still dated.
    assert items[1]["snippet"] == "Published Tue, 21 Oct 2026 08:00:00 GMT."
    assert items[3]["snippet"] == "", "beyond the excerpt limit nothing is fetched"


def test_a_hung_article_cannot_hold_the_news_up(monkeypatch):
    from orion_core import background_http, research

    async def never(url, **_kw):
        await asyncio.sleep(30)

    monkeypatch.setattr(background_http, "fetch_text", never)
    monkeypatch.setattr(research, "NEWS_EXCERPT_TIMEOUT_S", 0.05)
    items = asyncio.run(asyncio.wait_for(
        ResearchAgent(Bus(), None, None)._news(_Session(), "model nine"), 2))
    assert len(items) == 4 and "Synthetic Times" in items[0]["snippet"]


# ── accuracy: research reads pages and cites them (Mark XXXX.9) ──────────────
# Owner, 30 Oct 2026: "if ORION's researching it must be more accurate". His
# log showed "11 sources via the built-in encyclopaedia and news lookup" -
# search snippets and headlines, not one page opened - and notes that cited
# nothing.

_DEEP_FACT = "A low dose of 0.5 mg of melatonin taken five hours before sleep advanced the phase."
_DEEP_PAGE = "\n".join(["Readers often write in to ask about naps, caffeine and the best pillow to buy."] * 40
                       + [_DEEP_FACT]
                       + ["This article was reviewed by our board and updated to reflect current guidance."] * 40)


def _offline_web(monkeypatch, *, search=True):
    from orion_core import live_research, research, web_search_backends

    class NoSession:
        def __init__(self, *a, **k):
            raise OSError("offline")

    async def fake_search(query, limit=8, say=None):
        if not search:
            raise web_search_backends.SearchUnavailable("fixture")
        return [web_search_backends.Hit("Melatonin timing", "https://journal.example/melatonin",
                                        "A review of melatonin for sleep timing.", "fixture"),
                web_search_backends.Hit("Cookie wall", "https://cookies.example/", "Sleep tips.", "fixture")]

    opened = []

    async def fake_read(url, timeout=0):
        opened.append(url)
        return ("Melatonin timing", _DEEP_PAGE) if "journal" in url else ("", "")

    monkeypatch.setattr(research, "ClientSession", NoSession)
    monkeypatch.setattr(web_search_backends, "search", fake_search)
    monkeypatch.setattr(live_research, "read_page", fake_read)
    return opened


def test_timed_research_reads_pages_cites_them_and_marks_what_no_source_says(tmp_path, monkeypatch):
    opened = _offline_web(monkeypatch)
    prompts = []

    async def generate(prompt, **kwargs):
        prompts.append(prompt)
        if "Break the research topic" in prompt:
            return None, "What melatonin timing helps a delayed sleep phase?"
        if "executive briefing" in prompt:
            return None, "Melatonin taken five hours before sleep advanced the phase [1]."
        assert _DEEP_FACT in prompt, "the passage that answers the question reached the writer"
        assert "[1] (page read) Melatonin timing" in prompt and "(search snippet only)" in prompt
        return None, ("A low dose taken five hours before sleep advanced the phase [1].\n"
                      "Most clinicians also recommend a fixed wake time every day [7].")

    agent = ResearchAgent(Bus(), SimpleNamespace(generate_text=generate), None)
    result = asyncio.run(agent._run("Delayed sleep phase", 1, tmp_path))
    assert result.ok and result.receipt["evidence_level"] == "pages"
    assert result.receipt["pages_read"] == 1 and "treat it as unverified" not in result.text
    assert opened[0] == "https://journal.example/melatonin"
    note = next((tmp_path / "notes").glob("*.md")).read_text(encoding="utf-8")
    assert "advanced the phase [1]." in note
    assert "[7]" not in note and "wake time every day [unsourced]." in note
    assert "**Sources cited**" in note
    assert "- [1] (page read) Melatonin timing — https://journal.example/melatonin" in note
    assert "cookies.example" not in note, "a source the note did not cite is not listed as cited"
    sources = (tmp_path / "sources.md").read_text(encoding="utf-8")
    assert "[1] (page read)" in sources and "[2] (search snippet only)" in sources
    assert "SOURCE INDEX:" in prompts[-1] and "[1] (page read)" in prompts[-1]
    assert len(prompts) == 3, "reading pages costs no extra model call"


def test_an_encyclopaedia_only_timed_run_says_so(tmp_path, monkeypatch):
    _offline_web(monkeypatch, search=False)

    async def wiki(self, session, query):
        return {"title": "Wikipedia: Circadian rhythm", "url": "https://en.wikipedia.org/wiki/Circadian_rhythm",
                "snippet": "A circadian rhythm repeats roughly every 24 hours.", "evidence": "encyclopaedia"}

    class Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    from orion_core import research
    monkeypatch.setattr(research, "ClientSession", Session)
    monkeypatch.setattr(ResearchAgent, "_wikipedia", wiki)
    monkeypatch.setattr(ResearchAgent, "_news", lambda self, session, query: asyncio.sleep(0, []))

    async def generate(prompt, **kwargs):
        if "Break the research topic" in prompt:
            return None, "What is a circadian rhythm?"
        return None, "A circadian rhythm repeats roughly every 24 hours [1]."

    result = asyncio.run(ResearchAgent(Bus(), SimpleNamespace(generate_text=generate), None)
                         ._run("Circadian rhythm", 1, tmp_path))
    assert result.receipt["evidence_level"] == "encyclopaedia_only"
    assert "encyclopaedia summary only - no web page was opened" in result.text


def test_the_last_question_of_a_short_run_is_still_asked(tmp_path, monkeypatch):
    """The pacing divided the time left by the questions left, so after the
    second-to-last question it slept the whole remaining window and the last
    question was never asked in any short run."""
    asked = []

    async def questions(_topic):
        return ["First question here?", "Second question here?", "Third question here?"]

    async def gather(question):
        asked.append(question)
        return [{"title": "Source", "url": "https://example.test"}]

    async def prose(*_):
        return "Finding"

    agent = ResearchAgent(Bus(), None, None)
    monkeypatch.setattr(agent, "_decompose", questions)
    monkeypatch.setattr(agent, "_gather_sources", gather)
    monkeypatch.setattr(agent, "_synthesise", prose)
    monkeypatch.setattr(agent, "_final_summary_from_notes", prose)
    asyncio.run(agent._run("Topic", 0.03, tmp_path))
    assert asked == ["First question here?", "Second question here?", "Third question here?"]


def test_wikipedia_is_asked_for_a_title_not_a_query_string():
    """quote_plus made "circadian rhythm" into "circadian+rhythm", a page
    Wikipedia does not have: every multi-word lookup was a 404."""
    asked = []

    class Response:
        status = 404

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class Session:
        def get(self, url, **_kw):
            asked.append(url)
            return Response()

    asyncio.run(ResearchAgent(Bus(), None, None)._wikipedia(Session(), "circadian rhythm"))
    assert asked == ["https://en.wikipedia.org/api/rest_v1/page/summary/circadian_rhythm"]


def test_the_reasoning_look_up_stays_quick_and_opens_no_page(monkeypatch):
    opened = _offline_web(monkeypatch)
    rows = asyncio.run(ResearchAgent(Bus(), None, None).gather_sources("melatonin timing"))
    assert rows and opened == [] and all(row["evidence"] == "snippet" for row in rows)
