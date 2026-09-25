"""
Tests for AgentWorkspaceView (Mark XX design-spec §6) — one shared template
(Brief/Findings/Context/Instruments), six domain fills, all calling the same
AgentManager.dispatch() the WIDGETS panel and voice/text agent_dispatch
already use.

Also covers file/folder attachment: BRIEF's "Add File(s)" / "Add Folder"
buttons run a selection through the shared IngestionEngine and fold a
one-line digest per file into the ``context`` string handed to dispatch(),
so a specialist can be pointed at material without the caller composing a
prompt by hand.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.gui.agent_workspace import AgentWorkspaceView
from orion_core.ingestion import IngestionEngine


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubAgentManager:
    def __init__(self, result: ToolResult | None = None) -> None:
        self._result = result or ToolResult("the answer")
        self.calls: list[tuple[str, str, str]] = []   # (request, agent_name, context)
        self._instruments: list[str] = []
        self._describe = [{"name": "coding", "title": "Coding Agent",
                           "focus": "software engineering", "calls": 0, "last_active": ""}]

    def describe(self):
        return self._describe

    def available_instruments(self):
        return self._instruments

    async def dispatch(self, request, agent_name="auto", context=""):
        self.calls.append((request, agent_name, context))
        return self._result


def _view(manager=None, quick_prompts=(), ingestion=None) -> AgentWorkspaceView:
    return AgentWorkspaceView(
        OrionBus(), manager or _StubAgentManager(), "coding", "Coding Agent",
        quick_prompts=quick_prompts, ingestion=ingestion)


# ── header / instruments ────────────────────────────────────────────────────

def test_header_shows_the_agents_current_activity(_app):
    manager = _StubAgentManager()
    manager._describe = [{"name": "coding", "title": "Coding Agent", "focus": "x",
                          "calls": 5, "last_active": "12:00:00"}]
    view = _view(manager)
    assert "5 call" in view.activity_label.text()
    assert "12:00:00" in view.activity_label.text()


def test_instruments_panel_lists_available_tools(_app):
    manager = _StubAgentManager()
    manager._instruments = ["memory_search", "knowledge_graph"]
    view = _view(manager)
    assert "memory_search" in view.tools_label.text()
    assert "knowledge_graph" in view.tools_label.text()


def test_instruments_panel_explains_when_none_are_attached(_app):
    view = _view()
    assert "no read-only instruments" in view.tools_label.text().lower()


# ── quick prompts ────────────────────────────────────────────────────────────

def test_quick_prompt_button_fills_the_request_box(_app):
    view = _view(quick_prompts=["Review this code for bugs"])
    view._use_quick_prompt("Review this code for bugs")
    assert view.request_input.toPlainText() == "Review this code for bugs"


# ── consulting ────────────────────────────────────────────────────────────────

def test_consult_with_no_request_reports_an_error_and_does_not_dispatch(_app):
    manager = _StubAgentManager()
    view = _view(manager)
    view._consult()
    assert manager.calls == []
    assert "type a request" in view.status_label.text().lower()


def test_consult_calls_dispatch_with_the_request_and_context(_app):
    async def _scenario():
        manager = _StubAgentManager(ToolResult("here is the fix"))
        view = _view(manager)
        view.request_input.setPlainText("why does this crash")
        view.context_input.setText("traceback attached")
        view._consult()
        await asyncio.sleep(0)
        assert manager.calls == [("why does this crash", "coding", "traceback attached")]
        assert "here is the fix" in view.findings_output.toPlainText()

    asyncio.run(_scenario())


def test_consult_renders_the_evidence_trail(_app):
    async def _scenario():
        result = ToolResult(
            "grounded answer", ok=True,
            evidence=[{"tool": "memory_search", "ok": True, "summary": "memory_search(...) 2 hits"}])
        manager = _StubAgentManager(result)
        view = _view(manager)
        view.request_input.setPlainText("what did we discuss last time")
        view._consult()
        await asyncio.sleep(0)
        text = view.findings_output.toPlainText()
        assert "Sources consulted (1)" in text
        assert "memory_search" in text

    asyncio.run(_scenario())


def test_consult_appends_to_context_history(_app):
    async def _scenario():
        manager = _StubAgentManager(ToolResult("answer one"))
        view = _view(manager)
        view.request_input.setPlainText("first question")
        view._consult()
        await asyncio.sleep(0)
        assert view.history_list.count() == 1
        assert "first question" in view.history_list.item(0).text()

    asyncio.run(_scenario())


def test_selecting_a_history_entry_reloads_it_into_findings(_app):
    async def _scenario():
        manager = _StubAgentManager(ToolResult("the historical answer"))
        view = _view(manager)
        view.request_input.setPlainText("an old question")
        view._consult()
        await asyncio.sleep(0)
        view.findings_output.setPlainText("something else entirely")
        view._on_history_selection(view.history_list.item(0))
        assert view.findings_output.toPlainText() == "the historical answer"
        assert view.request_input.toPlainText() == "an old question"

    asyncio.run(_scenario())


def test_consult_refreshes_the_header_after_answering(_app):
    async def _scenario():
        manager = _StubAgentManager(ToolResult("answer"))
        view = _view(manager)
        manager._describe = [{"name": "coding", "title": "Coding Agent", "focus": "x",
                              "calls": 1, "last_active": "13:00:00"}]
        view.request_input.setPlainText("question")
        view._consult()
        await asyncio.sleep(0)
        assert "1 call" in view.activity_label.text()

    asyncio.run(_scenario())


# ── attachments (file/folder import through IngestionEngine) ───────────────

@pytest.fixture
def ingestion_engine(tmp_path):
    eng = IngestionEngine(db_path=tmp_path / "ws_ingestion.db")
    yield eng
    eng.close()


def test_no_ingestion_engine_reports_unavailable_instead_of_opening_a_dialog(_app):
    view = _view(ingestion=None)
    view._pick_files()
    assert "not available" in view.status_label.text().lower()
    view._pick_folder()
    assert "not available" in view.status_label.text().lower()


def test_attaching_a_file_adds_its_digest_to_attachments(_app, ingestion_engine, tmp_path):
    async def _scenario():
        target = tmp_path / "brief.md"
        target.write_text(
            "# Brand Brief\n\nYou must keep the palette within two accent colours.",
            encoding="utf-8")
        view = _view(ingestion=ingestion_engine)
        await view._attach_async([str(target)], is_folder=False)
        assert len(view._attachments) == 1
        assert view._attachments[0]["path"] == str(target.resolve()) or \
            Path(view._attachments[0]["path"]).name == "brief.md"
        assert "attached" in view.status_label.text().lower()
        assert "brief.md" in view.attachments_label.text()

    asyncio.run(_scenario())


def test_attached_file_digest_is_folded_into_context(_app, ingestion_engine, tmp_path):
    async def _scenario():
        target = tmp_path / "notes.txt"
        target.write_text(
            "Always keep the logo clear space at one full letterform.", encoding="utf-8")
        view = _view(ingestion=ingestion_engine)
        await view._attach_async([str(target)], is_folder=False)
        context = view._build_context()
        assert "notes.txt" in context
        assert len(context) > 0

    asyncio.run(_scenario())


def test_manual_context_and_attachment_digest_both_reach_dispatch(_app, ingestion_engine, tmp_path):
    async def _scenario():
        target = tmp_path / "spec.txt"
        target.write_text("Ensure every export is colour-managed for print.", encoding="utf-8")
        manager = _StubAgentManager(ToolResult("noted"))
        view = _view(manager, ingestion=ingestion_engine)
        await view._attach_async([str(target)], is_folder=False)
        view.request_input.setPlainText("review the attached brief")
        view.context_input.setText("client wants a minimalist look")
        view._consult()
        await asyncio.sleep(0)
        assert len(manager.calls) == 1
        _request, _agent, context = manager.calls[0]
        assert "client wants a minimalist look" in context
        assert "spec.txt" in context

    asyncio.run(_scenario())


def test_ingesting_a_folder_attaches_every_supported_file(_app, ingestion_engine, tmp_path):
    async def _scenario():
        (tmp_path / "a.txt").write_text("First reference document about layout.", encoding="utf-8")
        (tmp_path / "b.md").write_text("# Second\nSecond reference document about type.", encoding="utf-8")
        view = _view(ingestion=ingestion_engine)
        await view._attach_async([str(tmp_path)], is_folder=True)
        assert len(view._attachments) == 2
        names = {Path(a["path"]).name for a in view._attachments}
        assert names == {"a.txt", "b.md"}

    asyncio.run(_scenario())


def test_clear_attachments_empties_the_context_contribution(_app, ingestion_engine, tmp_path):
    async def _scenario():
        target = tmp_path / "keep.txt"
        target.write_text("Reference material that should be cleared later.", encoding="utf-8")
        view = _view(ingestion=ingestion_engine)
        await view._attach_async([str(target)], is_folder=False)
        assert view._attachments
        view._clear_attachments()
        assert view._attachments == []
        assert view._build_context() == ""
        assert "no files attached" in view.attachments_label.text().lower()

    asyncio.run(_scenario())
