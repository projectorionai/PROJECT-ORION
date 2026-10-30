"""Exact observations survive model result clipping and process restart."""

import asyncio
import json
from types import SimpleNamespace

from orion_core.data import ToolResult
from orion_core import text_tools, session_context


def test_text_fallback_observes_failed_research_and_exact_artifact_after_long_prose(monkeypatch):
    path = r"C:\Research\Maritime law\incomplete dossier.docx"
    replies = iter(['<tool_call>{"name":"research","arguments":{"topic":"Maritime law"}}</tool_call>',
                    "The report is incomplete; I saved the available notes."])
    prompts = []

    async def generate(prompt, _system):
        prompts.append(prompt)
        return None, next(replies)

    async def dispatch(*_):
        return ToolResult("notes " * 900, ok=False, receipt={
            "research_state": "failed", "run_id": "run123", "selected_path": path,
            "file_paths": [path], "failed_section_count": 2})

    monkeypatch.setattr(text_tools, "choose_tools", lambda _q, decl: decl)
    asyncio.run(text_tools.run("Compile the dossier", [{"name": "research"}], generate, dispatch))
    assert path in prompts[1] or json.dumps(path)[1:-1] in prompts[1]
    assert '"research_state": "failed"' in prompts[1]
    assert '"failed_section_count": 2' in prompts[1]


def test_text_evidence_only_exposes_allowlisted_receipt_fields():
    result = ToolResult("accepted", receipt={"research_state": "running", "token": "PRIVATE_TOKEN",
                                            "arbitrary": {"password": "PRIVATE_PASSWORD"}})
    result.execution = {"state": "returned", "executor_ok": True, "args": "PRIVATE_ARGS"}
    ok, text = text_tools._result_text(result)
    assert ok
    assert '"state": "returned"' in text
    assert '"research_state": "running"' in text
    assert "PRIVATE" not in text


def test_restarted_brief_keeps_research_id_state_and_exact_path(tmp_path):
    from orion_core.research_runs import ResearchRunStore
    store = ResearchRunStore(tmp_path / "research.json")
    path = tmp_path / "a dossier with spaces.docx"
    path.write_text("incomplete notes")
    row = store.start("Maritime law", "paper")
    store.update(row["run_id"], {"research_state": "failed", "file_paths": [str(path)]}, "Partial notes")
    restarted = ResearchRunStore(store.path)
    brief = session_context.build_brief(research_store=restarted)
    assert row["run_id"] in brief.text
    assert str(path) in brief.text
    assert "failed" in brief.text
    assert "Partial notes" in brief.text
    assert "research(action='status'" in brief.text


def test_reconnect_only_includes_research_changed_after_saved_point():
    rows = [{"run_id": "old", "updated_at": 99, "research_state": "failed", "topic": "old"},
            {"run_id": "new", "updated_at": 110, "research_state": "running", "topic": "new"}]
    brief = session_context.build_update(since=100, now=120,
                                        research_store=SimpleNamespace(list=lambda **_: rows))
    assert "new" in brief.text
    assert "old" not in brief.text
    assert "not start" in brief.text


def test_continuity_research_read_failure_is_visible():
    class Broken:
        def list(self, **_):
            raise OSError("disk unavailable")
    brief = session_context.build_brief(research_store=Broken())
    assert "Research records unavailable" in brief.text


def test_plain_tool_result_contract_is_unchanged():
    assert text_tools._result_text(ToolResult("ordinary", ok=False)) == (False, "ordinary")


def test_observation_budget_never_truncates_a_selected_path():
    path = "C:/Research/" + "資料" * 1800 + ".docx"
    result = ToolResult("notes " * 900, receipt={
        "selected_path": path, "file_paths": ["C:/" + "x" * 2000] * 3,
        "research_state": "failed", "topic": "t" * 600,
        "research_runs": [{"topic": "t" * 600, "run_id": "r" * 128}] * 10})
    _, text = text_tools._result_text(result)
    observations = json.loads(text.split("OPERATION OBSERVATIONS: ")[1])
    assert observations["selected_path"] == path
    assert len(json.dumps(observations, ensure_ascii=False)) <= text_tools.OBSERVATION_CHARS


def test_research_restart_never_calls_interrupted_work_running(tmp_path):
    from orion_core.research_runs import ResearchRunStore
    store = ResearchRunStore(tmp_path / "research.json")
    row = store.start("Never replay", "paper")
    store.close()  # a second live reader is not a process restart
    restarted = ResearchRunStore(store.path)
    brief = session_context.build_brief(research_store=restarted)
    assert row["run_id"] in brief.text
    assert "[interrupted]" in brief.text
    assert "not resumed" in brief.text
    assert "[running]" not in brief.text


def test_old_active_research_survives_many_newer_finished_runs():
    rows = [{"run_id": f"ended-{i}", "updated_at": 119, "research_state": "completed", "topic": "done"}
            for i in range(40)]
    rows.append({"run_id": "still-running", "updated_at": 1, "research_state": "running", "topic": "long task"})
    brief = session_context.build_update(since=100, now=120,
                                        research_store=SimpleNamespace(list=lambda **_: rows))
    assert "still-running [running]" in brief.text
    assert "do not start again" in brief.text
