"""
Dispatch domain — Memory, research, knowledge bases, documents and reporting.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line


def _flatten_strings(value: Any) -> list[str]:
    """Every string value inside a nested dict/list — used to gather ORION's own
    secret values for the spillage guard without caring about their structure."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_strings(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_flatten_strings(v))
    return out


#: Research modes that start a second, independent run of their own.
_SECOND_RESEARCH_ACTIONS = frozenset({
    "dossier", "investigate", "unattended", "background", "paper", "write_paper", "deep",
    "deep_research", "study", "thesis", "report", "browse", "live", "read", "properly", "human"})
_TOPIC_FILLER = frozenset({"about", "with", "from", "into", "that", "this", "what", "which",
                           "practical", "information", "research", "topic"})


def _topic_words(topic: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", str(topic or "").lower())
            if word not in _TOPIC_FILLER}


def _overlapping_research(research: Any, topic: str) -> str:
    """What to say instead of starting another run while the timed research
    already covers *topic*, or "". 30 Oct 2026: a dossier on "circadian rhythm
    and sleep strategies" started 90 s into a run on "circadian rhythm phase
    delay and practical strategies for severe sleep schedule disruption" -
    two runs on one question, the second one spending the quota."""
    active = getattr(research, "_active", None) or {}
    task = active.get("task")
    if task is None or task.done():
        return ""
    asked, running = _topic_words(topic), _topic_words(str(active.get("topic") or ""))
    shorter = min(len(asked), len(running))
    if shorter < 2 or len(asked & running) / shorter < 0.6:
        return ""
    status = research.status().text
    return (f"That is already being researched. {status} I'll tell you when it is ready; "
            "say 'stop research' to cancel it.")


def _research_receipt(dispatcher: Any, topic: str, paths: list[Any], *,
                      state: str = "completed", allow_folders: bool = False) -> dict[str, Any]:
    """Publish actual saved locations, never a proposed or missing path."""
    verified: list[str] = []
    for value in paths:
        if value is None:
            continue
        try:
            path = Path(value).resolve()
            if (path.is_file() or (allow_folders and path.is_dir())) and str(path) not in verified:
                verified.append(str(path))
        except (OSError, ValueError, RuntimeError):
            continue
    selected = verified[0] if verified else None
    refs = getattr(dispatcher, "_file_references", None)
    if selected and callable(refs):
        refs().remember(verified, selected=selected)
    return {"file_paths": verified, "selected_path": selected,
            "research_state": state, "topic": topic}


async def _track_research_task(dispatcher: Any, topic: str, action: str,
                               task: asyncio.Task, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """A real background task owns a running receipt and one final result.

    Optional callbacks let the Live channel put the actual lifecycle into its
    continuity ledger even though the original tool turn has already ended.
    """
    receipt = {"file_paths": [], "selected_path": None, "research_state": "running",
               "topic": topic, "action": action, "run_id": uuid.uuid4().hex}
    from . import research_runs
    store = getattr(dispatcher, "_research_run_store", None)
    if store is None:
        store = dispatcher._research_run_store = await asyncio.to_thread(research_runs.default_store)
    from . import background
    admission = background.spawn(asyncio.to_thread(
        store.start, topic, action, run_id=receipt["run_id"], inputs=inputs,
        owner=getattr(dispatcher, "research_owner", "local_session"),
        request_id=getattr(dispatcher, "research_request_id", None)), name="orion-research-admission")
    if admission is None:
        raise OSError("Research could not be recorded and was not started: the event loop is closing")
    try:
        await asyncio.shield(admission)
        research_runs.bind_publication_run(task, store, receipt["run_id"])
        receipt["durable"] = True
    except asyncio.CancelledError:
        # Cancelling an await cannot stop an atomic write already in a thread.
        # Keep an observer until admission settles, then close the accepted
        # record without ever releasing the research operation's start gate.
        async def settle_admission() -> None:
            try:
                await admission
                await asyncio.to_thread(store.update, receipt["run_id"],
                    {"research_state": "cancelled", "error_category": "admission_cancelled"},
                    "Research was cancelled before its operation started.")
            except (OSError, ValueError):
                pass  # A failed admission has no accepted operation to replay.

        settlement = background.spawn(settle_admission(), name="orion-research-admission-settle")
        if settlement is not None:
            try:
                await asyncio.wait_for(asyncio.shield(settlement), research_runs.PUBLICATION_SETTLE_S)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass  # The owned observer continues if the disk is still busy.
        raise
    except (OSError, ValueError) as exc:
        raise OSError(f"Research could not be recorded and was not started: {first_line(exc, 240)}") from exc
    runs = getattr(dispatcher, "_background_research_runs", None)
    if runs is None:
        runs = dispatcher._background_research_runs = []
    record = {"topic": topic, "action": action, "task": task,
              "receipt": receipt, "result": None, "started": time.monotonic()}
    runs.append(record)

    def trim_history() -> None:
        while len(runs) > 16:
            old = next((item for item in runs if (item.get("receipt") or {}).get(
                "research_state") not in research_runs.ACTIVE), None)
            if old is None:
                break  # durable admission caps active records, never discard their control handles
            runs.remove(old)

    trim_history()

    def notify(name: str, payload: Any) -> bool:
        callback = getattr(dispatcher, name, None)
        if callable(callback):
            try:
                callback(topic, payload)
                return True
            except Exception:
                pass  # Continuity reporting must not cancel the research.
        return False

    notify("research_started", dict(receipt))

    def persist(result: ToolResult) -> None:
        previous = record.get("save_task")
        payload, message = dict(result.receipt or {}), result.text

        async def save() -> None:
            if previous is not None:
                await previous
            if receipt.get("durable"):
                try:
                    await asyncio.to_thread(store.update, receipt["run_id"], payload, message)
                except (OSError, ValueError) as exc:
                    record["receipt"].update(durable=False, persistence_error=first_line(exc, 240))
        from . import background
        record["save_task"] = background.spawn(save(), name="orion-research-receipt-save")

    def publish_result(result: ToolResult, *, terminal: bool = True) -> None:
        record["result"], record["task"] = result, None
        record["receipt"] = result.receipt
        trim_history()
        persist(result)
        if terminal and callable(getattr(dispatcher, "research_completion", None)):
            if not notify("research_completion", result):
                try:
                    dispatcher.bus.speak_request.emit(result.text[:600])
                except Exception:
                    pass

    def finished(done: asyncio.Task) -> None:
        if done.cancelled():
            result = ToolResult(f"Research on {topic} was cancelled before completion.", ok=False,
                                receipt={"research_state": "cancelled"})
        else:
            try:
                result = done.result()
                if not isinstance(result, ToolResult):
                    result = ToolResult(f"Research on {topic} ended without a recorded result.", ok=False)
            except Exception as exc:
                result = ToolResult(f"Research on {topic} failed: {first_line(exc, 160)}", ok=False)
        state = "cancelled" if done.cancelled() else "completed" if result.ok else "failed"
        paths, pending = research_runs.publication_paths(done)
        final = dict(receipt)
        final.update(result.receipt or {})
        final.update(_research_receipt(dispatcher, topic, [*(final.get("file_paths") or []), *paths],
                                       state=state, allow_folders=True))
        final.update(run_id=receipt["run_id"], action=action, topic=topic)
        result.receipt = final
        if done.cancelled() and paths:
            result.text += " Published artifact retained: " + "; ".join(paths)
        if done.cancelled() and pending:
            final["research_state"] = "cancel_requested"
            result.text = f"Cancellation was requested for {topic}; a file publication is still settling."
            publish_result(result, terminal=False)

            async def settle_publication() -> None:
                await asyncio.gather(*pending, return_exceptions=True)
                actual_paths, _ = research_runs.publication_paths(done)
                settled = dict(final, **_research_receipt(dispatcher, topic, actual_paths,
                                  state="cancelled", allow_folders=True))
                message = f"Research on {topic} was cancelled."
                if actual_paths:
                    message += " Published artifact retained: " + "; ".join(actual_paths)
                publish_result(ToolResult(message, ok=False, receipt=settled))

            from . import background
            record["publication_task"] = background.spawn(settle_publication(), name="orion-research-publication-settle")
        else:
            publish_result(result)

    task.add_done_callback(finished)
    if task.done():
        await asyncio.sleep(0)
    return dict(receipt)


async def _spawn_research_task(dispatcher: Any, topic: str, action: str, work: Any,
                                inputs: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Register intent durably before the coroutine can start its operation."""
    from . import background
    accepted = asyncio.Event()

    async def run() -> ToolResult:
        started = False
        try:
            await accepted.wait()
            started = True
            return await work
        finally:
            if not started:
                work.close()

    task = background.spawn(run(), name=f"orion-{action}-research")
    if task is None:
        work.close()
        return None, {}
    try:
        receipt = await _track_research_task(dispatcher, topic, action, task, inputs)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    accepted.set()
    return task, receipt


async def _await_research_task(dispatcher: Any, topic: str, action: str, work: Any) -> ToolResult:
    """Waiting for the result does not remove its running/status/stop lifecycle."""
    from . import background

    started = False

    async def finished() -> ToolResult:
        nonlocal started
        started = True
        try:
            return await work
        except Exception as exc:
            return ToolResult(f"The research on {topic} failed: {first_line(exc, 160)}", ok=False,
                              receipt=_research_receipt(dispatcher, topic, [], state="failed"))

    try:
        task, _receipt = await _spawn_research_task(dispatcher, topic, action, finished())
    except BaseException:
        if not started:
            work.close()
        raise
    if task is None:
        work.close()
        return ToolResult("Research could not start because the event loop is closing.", ok=False)
    return await task


class KnowledgeDispatchMixin:
    """Memory, research, knowledge bases, documents and reporting."""

    async def antigravity_research(self, args: dict[str, Any]) -> ToolResult:
        """Research an explicit public topic in a fresh, isolated agent sandbox."""
        from .antigravity_research import AntigravityError, AntigravityResearch
        from .providers import read_api_key

        service = getattr(self, "_antigravity_service", None)
        if service is None:
            service = AntigravityResearch()
            self._antigravity_service = service
        try:
            result = await service.run(args.get("topic"), api_key=read_api_key())
        except AntigravityError as exc:
            return ToolResult(str(exc), ok=False)
        return ToolResult(result.text, ok=result.status == "completed")

    async def semantic_compare(self, args: dict[str, Any]) -> ToolResult:
        """Compare only text supplied in this call; cloud use is per-call opt in."""
        from .gemini_embeddings import rank_texts

        try:
            ranking = await rank_texts(
                args.get("query", ""), args.get("candidates", []),
                cloud=args.get("cloud") is True,
                consent=args.get("confirm_cloud") is True,
                model=args.get("model"),
            )
        except (ValueError, PermissionError) as exc:
            return ToolResult(str(exc), ok=False)
        lines = [f"Compared with {ranking.source} embeddings."]
        if ranking.detail:
            lines.append(ranking.detail)
        lines.extend(f"{row.position}. {row.score:.4f} — {row.text}"
                     for row in ranking.rows)
        return ToolResult("\n".join(lines))

    def save_memory(self, args: dict[str, Any]) -> ToolResult:
        category = str(args.get("category") or "notes")
        key      = str(args.get("key") or args.get("key_ref") or "entry")
        value    = str(args.get("value") or args.get("fact") or "")
        result   = self.memory.save(category, key, value)
        return ToolResult(result)

    #: The user's own past statements added to a recall answer, at most.
    SAID_LIMIT = 4

    def _recall_with_statements(self, query: str, limit: int
                                ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Stored facts for *query*, and what the user said about it himself.

        Measured 17 Oct 2026 on a copy of the owner's store: half of the small
        details he gave ORION (his course, his work days, his budget, a trip)
        were never turned into a fact and lived only in his own turns, which
        this tool did not search although the system prompt says it does.
        Blocking SQLite and encoder work: called on a worker thread.
        """
        rows = self.memory.query(query, limit=limit)
        recall_said = getattr(self.memory, "recall_said", None)
        said: list[dict[str, str]] = []
        if callable(recall_said):
            try:
                said = list(recall_said(query, limit=min(self.SAID_LIMIT, max(1, limit))) or [])
            except SecurityViolation:
                raise
            except Exception:
                said = []           # extra evidence only; a fault here must not lose the facts
        return rows, said

    async def query_intelligence(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or args.get("text") or "").strip()
        limit = int(args.get("limit") or 8)
        category = str(args.get("category") or "").strip().lower()
        learned = bool(args.get("learned"))
        said: list[dict[str, str]] = []
        if query and not (category or learned):
            rows, said = await asyncio.to_thread(self._recall_with_statements, query, limit)
        else:
            # The schema promised "an empty query returns the most recent
            # entries" — but query("") returns nothing, so an empty lookup
            # always came back "No matching intelligence records". records()
            # is newest-first; the filters then narrow it.
            rows = self.memory.records(query, limit=500 if (category or learned) else limit)
            if category:
                rows = [r for r in rows if str(r.get("category", "")).lower() == category]
            if learned:
                # Facts ORION kept from conversation on his own (fact_memory).
                rows = [r for r in rows if str(r.get("key_ref", "")).startswith("learned_")]
            rows = rows[:limit]
        if not rows and not said:
            return ToolResult("No matching intelligence records.")
        lines = [
            f"{row['category']}/{row['key_ref']}: {row['value']} ({row['updated_at']})"
            for row in rows
        ]
        if said:
            lines.append("What the user said about this, in their own words (dated, best match first):")
            for turn in said:
                words = " ".join(str(turn.get("content") or "").split())
                if len(words) > 300:
                    words = words[:300].rstrip() + " [...]"
                lines.append(f"- [{str(turn.get('created_at') or '')[:10]}] {words}")
        return ToolResult("\n".join(lines))

    def recall_conversation(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or args.get("text") or "")
        limit = int(args.get("limit") or 10)
        rows  = self.memory.recall_episodes(query, limit=limit)
        if not rows:
            return ToolResult("No matching conversation history.")
        return ToolResult("\n".join(
            f"[{row['created_at']}] {row['role']}: {row['content'][:240]}" for row in rows
        ))

    async def _deep_research(self, args: dict[str, Any], topic: str) -> ToolResult:
        """Plan, write, save as Word, file into memory, and announce it.

        Replaces the old six-section paper, which was ~1,500 words because its
        prompts asked for 150-300 words a section. Depth and style are taken
        from the request so "a 50-page literature review" and "a short
        technical note" are the same call with different arguments.
        """
        if not topic:
            return ToolResult("What should the research be about?", ok=False)
        from .deep_research import DeepResearchEngine, resolve_depth, resolve_style

        depth = resolve_depth(args.get("depth") or args.get("pages")
                              or args.get("length") or "standard")
        style = resolve_style(args.get("style") or args.get("kind") or "systematic")
        engine = DeepResearchEngine(self.research, getattr(self, "bus", None))
        self._show_research_console()
        if not args.get("wait") and getattr(self, "bus", None) is not None:
            # Detached. A paper takes minutes; held inside the voice turn,
            # ORION sat silent the whole time and a long run could outlast
            # the live session's deadline. The work carries on here, the
            # console shows every step, and the finished paper is announced
            # aloud (speak_request) when it lands.
            from . import background

            async def _run() -> ToolResult:
                try:
                    result = await self._finish_paper(
                        engine, topic, style, depth,
                        announce=not callable(getattr(self, "research_completion", None)))
                except Exception as exc:
                    result = ToolResult(f"The research on {topic} failed: {first_line(exc, 120)}", ok=False)
                if not result.ok and not callable(getattr(self, "research_completion", None)):
                    try:
                        self.bus.speak_request.emit(result.text[:600])
                    except Exception:
                        pass
                return result

            task, receipt = await _spawn_research_task(self, topic, "paper", _run(), args)
            if task is not None:
                from .deep_research import DEPTHS
                pages = DEPTHS[depth][0]
                return ToolResult(
                    f"Research on '{topic}' is underway — about {pages} pages. I'm "
                    "searching and reading real sources now; every step is on the "
                    "Research console, and I'll tell you the moment the paper is "
                    "written. We can carry on meanwhile.", receipt=receipt)
        return await _await_research_task(
            self, topic, "paper", self._finish_paper(
                engine, topic, style, depth,
                announce=not callable(getattr(self, "research_completion", None))))

    async def _finish_paper(self, engine: Any, topic: str, style: Any, depth: Any, *,
                            announce: bool = True) -> ToolResult:
        """Write the paper, save it as Word, file it into memory, announce it."""
        from . import research_docx

        try:
            paper = await engine.write(topic, style=style, depth=depth)
        except Exception as exc:
            return ToolResult(f"The research run failed: {first_line(exc, 160)}",
                              ok=False, receipt=_research_receipt(self, topic, [], state="failed"))

        try:
            from .research_runs import publish
            path, note = await publish(research_docx.write_paper, paper,
                                       artifact_paths=lambda result: [result[0]] if result[0] else [])
        except Exception as exc:
            path, note = None, first_line(exc, 160)
        receipt = await asyncio.to_thread(_research_receipt, self, topic, [path])
        paper.docx_path = Path(receipt["selected_path"]) if receipt["selected_path"] else None
        if paper.docx_path is None:
            receipt["research_state"] = "failed"
            return ToolResult(f"The research on {topic} was written, but no paper was saved. "
                              + (note or "The output file is missing."), ok=False, receipt=receipt)

        sections = list(getattr(paper, "sections", ()) or ())
        incomplete = [section for section in sections if getattr(section, "error", "")
                      or not str(getattr(section, "body", "") or "").strip()]
        sourced = bool(getattr(paper, "sources", ()))
        complete = sourced and bool(sections) and not incomplete
        # Only a fully generated paper is filed as completed knowledge. The
        # saved partial artifact remains selected and available in the receipt.
        filed = self._file_research(paper) if complete else ""
        if incomplete or not sections:
            problem = (f"{len(incomplete)} section(s) failed or were empty." if sections
                       else "No completed sections were returned.")
            announcement = f"I saved an incomplete paper on {topic}. {problem} The research is not complete."
        elif not sourced:
            announcement = (f"I saved an unverified draft on {topic}. No external sources were retrieved, "
                            "so the research is not complete.")
        else:
            announcement = paper.announcement()
        if note:
            announcement += f" ({note})"
        if filed:
            announcement += f" {filed}"
        # Spoken as well as returned: a fifty-page document that appears
        # silently in a folder may as well not exist.
        try:
            if announce and complete:
                self.bus.speak_request.emit(announcement)
        except Exception:
            pass
        # The path goes in the TEXT. `media` is the live channel's typed frame
        # (bytes + mime_type): a bare path string there reached _send_media,
        # which called .get on it — "'str' object has no attribute 'get'" —
        # and the voice session was benched straight after every paper.
        if path:
            announcement += f" Location: {path}"
        receipt.update(research_state="completed" if complete else "failed",
                       source_count=len(getattr(paper, "sources", ())),
                       failed_section_count=len(incomplete), section_count=len(sections))
        return ToolResult(announcement, ok=complete, receipt=receipt)

    def _file_research(self, paper: Any) -> str:
        """Put the finished paper where ORION will find it again.

        Section by section rather than as one blob: recall works on the level
        of an idea, and a single 25,000-word record can only ever be retrieved
        whole, which is the same as not being retrievable.
        """
        stored = 0
        graph = getattr(self, "knowledge_graph", None) or getattr(self, "graph", None)
        memory = getattr(self, "memory", None)
        for section in paper.sections:
            if not section.body:
                continue
            summary = f"{paper.topic} — {section.title}: {section.body[:600]}"
            if graph is not None and hasattr(graph, "ingest_record"):
                try:
                    graph.ingest_record(kind="research", title=section.title,
                                        body=section.body, source=str(paper.docx_path or ""))
                    stored += 1
                    continue
                except Exception:
                    pass
            if memory is not None and hasattr(memory, "remember"):
                try:
                    topic_key = re.sub(r"[^a-z0-9]+", "_", paper.topic.lower()).strip("_")[:48]
                    section_key = re.sub(r"[^a-z0-9]+", "_", section.title.lower()).strip("_")[:48]
                    location = f" (source: {paper.docx_path})" if paper.docx_path else ""
                    memory.remember("knowledge", f"research_{topic_key}_{section_key}", summary + location)
                    stored += 1
                except Exception:
                    pass
        if not stored:
            return ""
        return f"I've filed {stored} section(s) into memory."

    async def _browse_research(self, topic: str, args: dict[str, Any]) -> ToolResult:
        """Search, open, read, take notes — and say so the whole way through.

        Distinct from every other research action in one respect that matters:
        what comes back is built from pages that were actually opened. The
        notes name their source, and a claim with no note behind it does not
        appear.
        """
        if not topic:
            return ToolResult("What should I look into?", ok=False)
        agent = self.research
        if agent is None or getattr(agent, "_ask", None) is None:
            return ToolResult("The research agent has no model access.", ok=False)

        from .constants import BASE_DIR
        from .deep_research import DeepResearchEngine
        from .live_research import LiveResearcher

        # The strict call: a model failure must stop a note being taken, not
        # become one (the agent's own _ask returns placeholder prose instead).
        strict = DeepResearchEngine(agent, getattr(self, "bus", None))
        self._show_research_console()
        researcher = LiveResearcher(
            ask=lambda prompt, persona="research reader":
                strict._ask(prompt, persona, max_tokens=1200),
            bus=self.bus, root=BASE_DIR / "research")
        try:
            run = await researcher.investigate(
                topic,
                questions=max(1, min(8, int(args.get("questions") or 4))),
                per_question=max(1, min(6, int(args.get("per_question") or 3))),
                rounds=max(1, min(4, int(args.get("rounds") or 2))),
                deadline_s=float(args.get("minutes") or 0) * 60.0)
        except Exception as exc:
            return ToolResult(f"The investigation failed: {first_line(exc)}",
                              ok=False)

        if not run.notes:
            # Said plainly rather than dressed up as a finding. "I read nothing"
            # and "there is nothing to find" are different statements.
            return ToolResult(
                f"I could not read anything usable about {topic}. "
                f"{len(run.failed)} page(s) would not open or carried no text.",
                ok=False)

        summary_error = ""
        try:
            summary = str(await researcher.summarise(run) or "").strip()
        except Exception as exc:
            summary, summary_error = "", first_line(exc, 200)
        from .research_runs import publish
        folder = await publish(researcher.write_folder, run, summary,
                               artifact_paths=lambda folder: [folder / "SUMMARY.md", folder] if folder else [])
        receipt = await asyncio.to_thread(_research_receipt, self, topic,
                                          [folder / "SUMMARY.md", folder] if folder else [],
                                          allow_folders=True)
        saved = bool(receipt["file_paths"])
        complete = saved and bool(summary) and (folder / "SUMMARY.md").is_file()
        if not complete:
            receipt["research_state"] = "failed"
        if not summary:
            summary = "The investigation is incomplete: no usable summary was generated. " + summary_error
        where = f"\nLocation: {receipt['selected_path']}" if saved else "\nThe research notes could not be saved."
        return ToolResult(
            f"{summary}\n\n_Read {len(run.read)} page(s) and took "
            f"{len(run.notes)} note(s) over {run.seconds / 60:.1f} minutes._"
            f"{where}", ok=complete, receipt=receipt)

    def _show_research_console(self) -> None:
        """Put the research console on screen as a run starts.

        "ORION must SHOW me his working": every search, page and note is
        already announced on the bus, but announced into a log nobody had
        open. The console is the deck's RESEARCH page; asking for it is a GUI
        navigation, so it is emitted rather than called.
        """
        # His own deck page: shown when the owner asked for the research,
        # never when research runs on its own (require_own_window).
        from .execution_policy import require_own_window
        if require_own_window("gui_research_navigation") is not None:
            return
        bus = getattr(self, "bus", None)
        if bus is None:
            return
        try:
            bus.gui_command.emit({"action": "page", "target": "RESEARCH",
                                  "quiet": True})
        except Exception:
            pass

    async def research_tool(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "start").lower().strip()
        topic = str(args.get("topic") or args.get("query") or "").strip()
        run_id = str(args.get("run_id") or "").strip()
        runs = [*(getattr(self, "_background_research_runs", ()) or ()),
                *(getattr(getattr(self, "research_director", None), "_background_research_runs", ()) or ())]
        if action in {"status", "progress", "runs", "history", "open"}:
            for run in runs:
                saving = run.get("save_task")
                if saving is not None:
                    await asyncio.shield(saving)
            from . import research_runs
            store = getattr(self, "_research_run_store", None)
            if store is None:
                store = self._research_run_store = await asyncio.to_thread(research_runs.default_store)
            if action in {"runs", "history", "open"} or (action in {"status", "progress"} and not runs) or (run_id and not any(
                    (run.get("receipt") or {}).get("run_id") == run_id for run in runs)):
                if store.storage_error:
                    return ToolResult(f"Research history could not be read: {store.storage_error}", ok=False)
                rows = await asyncio.to_thread(store.list, run_id=run_id, topic=topic,
                                               limit=20 if action in {"runs", "history"} else 1)
                if not rows:
                    return ToolResult("No matching recorded research was found.", ok=False,
                                      receipt={"research_state": "not_found", "run_id": run_id, "file_paths": []})
                if action in {"runs", "history"}:
                    return ToolResult("Recorded research (newest first):\n" + "\n".join(
                        f"- {row['run_id']} [{row['research_state']}] {row['topic']}" for row in rows),
                        receipt={"research_runs": rows})
                row = rows[0]
                available = await asyncio.to_thread(research_runs.verified_paths, row.get("file_paths"))
                receipt = dict(row, recorded_file_paths=row.get("file_paths", []), file_paths=available,
                               selected_path=available[0] if available else None)
                if action == "open":
                    if not available:
                        return ToolResult("The recorded research artifact is no longer available at its saved path.",
                                          ok=False, receipt=receipt)
                    from . import file_search
                    problem = await asyncio.to_thread(file_search.open_path, available[0])
                    if problem:
                        return ToolResult(f"Could not open the recorded artifact: {problem}", ok=False, receipt=receipt)
                    refs = getattr(self, "_file_references", None)
                    if callable(refs):
                        refs().select(available[0])
                    return ToolResult(f"Opened {available[0]}.", receipt=receipt)
                return ToolResult(row.get("result") or f"Research is {row['research_state']}.",
                                  ok=row["research_state"] == "completed", receipt=receipt)
        matching = [run for run in reversed(runs)
                    if (not topic or str(run["topic"]).casefold() == topic.casefold())
                    and (not run_id or (run.get("receipt") or {}).get("run_id") == run_id)]
        recent = next((run for run in matching if (run["task"] is not None
                       and not run["task"].done()) or (run.get("receipt") or {}).get(
                       "research_state") == "cancel_requested"), matching[0] if matching else None)
        active = getattr(getattr(self, "research", None), "_active", {})
        active_task = active.get("task")
        if (recent is not None and active_task is not None and not active_task.done()
                and active_task is not recent["task"]
                and not run_id and (not topic or str(active.get("topic") or "").casefold() == topic.casefold())
                and float(active.get("started") or 0.0) > recent["started"]):
            recent = None
        if recent is not None and action in {"status", "progress", "stop", "cancel"}:
            task = recent["task"]
            if (recent.get("receipt") or {}).get("research_state") == "cancel_requested":
                return recent["result"]
            if task is not None and task.done() and recent["result"] is None:
                await asyncio.sleep(0)  # let the registered outcome callback publish it
            if task is not None and not task.done():
                if action in {"stop", "cancel"}:
                    receipt = dict(recent["receipt"], research_state="cancel_requested")
                    task.cancel()
                    return ToolResult(f"Stopping research on {recent['topic']}.",
                                      receipt=dict(receipt, research_state="cancelling"))
                return ToolResult(f"Research on {recent['topic']} is still running; no finished "
                                  "report has been recorded yet.", receipt=dict(recent["receipt"]))
            if recent["result"] is not None and action in {"status", "progress"}:
                result = recent["result"]
                return ToolResult(result.text, ok=result.ok, receipt=result.receipt)
        if run_id and action in {"status", "progress", "stop", "cancel"}:
            return ToolResult("No active research matches that run ID; completed or interrupted work cannot be cancelled.",
                              ok=False, receipt={"run_id": run_id, "research_state": "not_found", "file_paths": []})
        if (topic and action in {"status", "progress", "stop", "cancel"}
                and str(active.get("topic") or "").casefold() != topic.casefold()):
            return ToolResult(f"No matching active research was found for '{topic}'.", ok=False,
                              receipt={"topic": topic, "research_state": "not_found", "file_paths": []})
        if action == "neuroscience_pilot":
            # A fixed, reviewed source snapshot. Reading the mission record is
            # optional; this path never changes mission, memory or research state.
            from .neuroscience_research import pilot_evidence, pilot_report

            try:
                report = pilot_report(
                    missions=getattr(self, "missions", None),
                    topic=str(args.get("topic") or args.get("query") or ""),
                )
            except ValueError as exc:
                return ToolResult(str(exc), ok=False)
            return ToolResult(report, evidence=pilot_evidence())
        if self.research is None:
            return ToolResult("Research agent is not available.", ok=False)
        if action in _SECOND_RESEARCH_ACTIONS:
            running = _overlapping_research(self.research, topic)
            if running:
                return ToolResult(running)
        if action in {"start", "conduct", "research"}:
            from .research import ResearchAgent
            accepted = asyncio.Event() if isinstance(self.research, ResearchAgent) else None
            options = {"_accepted": accepted} if accepted is not None else {}
            result = self.research.start_research(topic, minutes=float(args.get("minutes", 30)), **options)
            task = getattr(self.research, "_active", {}).get("task")
            if result.ok and isinstance(task, asyncio.Task):
                try:
                    result.receipt = await _track_research_task(self, topic, "start", task, args)
                except BaseException:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise
                if accepted is not None:
                    accepted.set()
            return result
        if action in {"paper", "write_paper", "deep", "deep_research", "study",
                      "thesis", "report"}:
            return await self._deep_research(args, topic)
        if action in {"dossier", "investigate", "unattended", "background"}:
            if not topic.strip():
                return ToolResult("What should I look into?", ok=False)
            from . import background

            async def _finish_dossier() -> ToolResult:
                callback_handles_announcement = callable(getattr(self, "research_completion", None))
                try:
                    return await self._dossier(topic, args, announce=not callback_handles_announcement)
                except Exception as exc:
                    result = ToolResult(
                        f"The research on {topic} failed: {first_line(exc, 160)}", ok=False,
                        receipt=_research_receipt(self, topic, [], state="failed"))
                    if not callback_handles_announcement:
                        try:
                            self.bus.speak_request.emit(result.text)
                        except Exception:
                            pass
                    return result

            task, _receipt = await _spawn_research_task(self, topic, "dossier", _finish_dossier(), args)
            if task is None:
                return ToolResult("Research could not start because the event loop is closing.", ok=False)
            # Dossiers retain their awaited tool contract. Their independent
            # owned task is also visible to status/stop and Live continuity.
            return await task
        if action in {"browse", "live", "read", "properly", "human"}:
            if not topic:
                return ToolResult("What should I look into?", ok=False)
            if args.get("wait"):
                return await _await_research_task(self, topic, "browse", self._browse_research(topic, args))
            # Reading several rounds of pages can outlive a Live WebSocket.
            # Return to the voice turn now; finish independently and announce
            # the requested result when it is ready.
            from . import background

            async def _finish_browse() -> ToolResult:
                try:
                    result = await self._browse_research(topic, args)
                    if (getattr(self, "bus", None) is not None
                            and not callable(getattr(self, "research_completion", None))):
                        self.bus.speak_request.emit(str(result.text)[:600])
                except Exception as exc:
                    result = ToolResult(f"The research on {topic} failed: {first_line(exc, 120)}", ok=False)
                    if (getattr(self, "bus", None) is not None
                            and not callable(getattr(self, "research_completion", None))):
                        self.bus.speak_request.emit(result.text)
                return result

            task, receipt = await _spawn_research_task(self, topic, "browse", _finish_browse(), args)
            if task is not None:
                return ToolResult(
                    f"I'm researching {topic} now. Follow the Research console; "
                    "I'll tell you what I found when the reading is done.", receipt=receipt)
            return ToolResult("Research could not start because the event loop is closing.",
                              ok=False)
        if action in {"stop", "cancel"}:
            return self.research.stop_research()
        if action in {"status", "progress"}:
            return self.research.status()
        # ── Phase 3: persistent research programme (ResearchDirector) ─────────
        if self.research_director is not None:
            if action in {"queue", "add", "later"}:
                return await self.research_director.queue_topic(
                    topic, minutes=float(args.get("minutes", 20)))
            if action in {"agenda", "programme", "list"}:
                return await self.research_director.agenda()
            if action in {"findings", "evidence", "claims"}:
                return await self.research_director.findings(topic)
            if action in {"reviewed", "mark_reviewed"}:
                return await self.research_director.mark_reviewed(topic)
            if action in {"validate", "validate_sources", "sources"}:
                return await self.research_director.validate_sources(
                    str(args.get("text") or topic))
            if action in {"opportunities", "opportunity", "gaps"}:
                return await self.research_director.opportunities()
        return ToolResult(
            "Unsupported research action. Use start, paper, stop, status, "
            "queue, agenda, findings, reviewed, validate, or opportunities.",
            ok=False)

    async def _dossier(self, topic: str, args: dict[str, Any], *, announce: bool = True) -> ToolResult:
        """An unattended investigation, filed as a dossier with its citations.

        Distinct from `start` (a timed background crawl that reports on
        request) and `paper` (a long written document): this is the one a
        scheduled plugin calls at three in the morning, so it announces itself
        when it finishes rather than waiting to be asked.
        """
        if not topic.strip():
            return ToolResult("What should I look into?", ok=False)
        from .research_worker import ResearchWorker

        try:
            questions = max(1, min(12, int(args.get("questions") or 5)))
        except (TypeError, ValueError):
            questions = 5

        worker = ResearchWorker(
            agent=self.research,
            mcp=getattr(self, "mcp", None),
            bus=getattr(self, "bus", None),
            reminders=getattr(self, "reminders", None),
            announce_results=announce,
        )
        dossier = await worker.investigate(topic, questions=questions)
        receipt = await asyncio.to_thread(_research_receipt, self, topic, [dossier.path])
        saved = bool(receipt["file_paths"])
        sourced = bool(dossier.sources)
        complete = saved and sourced and not dossier.generation_errors
        receipt.update(research_state="completed" if complete else "failed",
                       source_count=len(dossier.sources),
                       evidence_level=getattr(dossier, "evidence_level", "") or "unknown")
        where = receipt["selected_path"] if saved else "nowhere (it could not be filed)"
        lines = [
            f"Dossier on {dossier.topic} — {len(dossier.sources)} source"
            f"{'s' if len(dossier.sources) != 1 else ''} via "
            f"{dossier.gatherer or 'the built-in lookup'}.",
            f"Filed: {where}",
        ]
        if dossier.filing_error:
            lines.append(dossier.filing_error)
        elif not saved:
            lines.append("No saved dossier file could be verified.")
        if not sourced:
            lines.append("No external sources were retrieved. This is an unverified draft from "
                         "the model's own knowledge; the research request has not been completed.")
        if dossier.generation_errors:
            lines += ["The investigation is incomplete; generation failed:"]
            lines.extend(f"- {error}" for error in dossier.generation_errors[:8])
        if dossier.summary:
            lines += ["", dossier.summary.strip()]
        if dossier.takeaways:
            lines += [""] + [f"  {n}. {point}"
                             for n, point in enumerate(dossier.takeaways, 1)]
        return ToolResult("\n".join(lines), ok=complete, receipt=receipt)

    def neuro_knowledge(self, args: dict[str, Any]) -> ToolResult:
        """Authoritative neuroscience / neural-engineering facts from the local corpus."""
        if self.knowledge is None:
            return ToolResult("Knowledge base is not available.", ok=False)
        query = str(args.get("query") or args.get("topic") or "").strip()
        if not query:
            return ToolResult("Topics I can go deep on: " + ", ".join(self.knowledge.topics()))
        answer = self.knowledge.answer(query)
        if answer:
            return ToolResult(answer)
        return ToolResult(
            "No specific corpus entry matched. I can speak to neurons, glia, action "
            "potentials, synapses, plasticity, cortex, hippocampus, Hodgkin-Huxley, "
            "integrate-and-fire, cable theory, BCIs, EEG, ECoG, the Utah array, "
            "Neuralink, spike sorting, LFP, decoding, neuroprosthetics, DBS and "
            "stimulation."
        )

    def knowledge_pack(self, args: dict[str, Any]) -> ToolResult:
        if self.packs is None:
            return ToolResult("Knowledge packs are unavailable.", ok=False)
        action = str(args.get("action") or "consult").lower().strip()
        if action in {"consult", "search", "ask"}:
            return self.packs.consult(str(args.get("query") or args.get("topic") or ""))
        if action in {"list", "installed"}:
            packs = self.packs.list_packs()
            return ToolResult("Installed knowledge packs:\n" + "\n".join(
                f"- {p['title']} ({p['entries']} entries): {p['description']}" for p in packs))
        if action in {"remove", "uninstall"}:
            return self.packs.remove(str(args.get("id") or args.get("pack_id") or ""))
        if action in {"expand", "add"}:
            entries = args.get("entries") or []
            if isinstance(entries, dict):
                entries = [entries]
            return self.packs.expand(str(args.get("id") or args.get("pack_id") or ""), entries)
        return ToolResult(f"Unsupported knowledge_pack action: {action}.", ok=False)

    async def conversation_recall(self, args: dict[str, Any]) -> ToolResult:
        if self.conversation is None:
            return ToolResult("Conversation memory engine unavailable.", ok=False)
        action = str(args.get("action") or "recall").lower().strip()
        if action in {"recall", "when", "history"}:
            return await self.conversation.recall(
                str(args.get("query") or args.get("question") or ""))
        if action in {"summarise", "summarize", "summary"}:
            return await self.conversation.summarise_recent(int(args.get("turns") or 40))
        if action in {"compress"}:
            return await self.conversation.compress(int(args.get("days") or 14))
        return ToolResult(f"Unsupported conversation_recall action: {action}.", ok=False)

    async def learn_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.learning is None:
            return ToolResult("The learning service is not available.", ok=False)
        action = str(args.get("action") or "learn").lower().strip()
        if action in {"recall", "remember", "what_did_you_learn"}:
            return await self.learning.recall(str(args.get("query") or ""))
        if action in {"folder", "learn_folder", "ingest", "library", "bulk"}:
            return await self.learning.learn_folder(
                folder=str(args.get("folder") or args.get("path") or args.get("source") or ""),
                topic=str(args.get("topic") or ""),
                deep=bool(args.get("deep")),
            )
        if action in {"correct", "correction", "fix"}:
            return await self.learning.correct(
                topic=str(args.get("topic") or ""),
                correction=str(args.get("correction") or args.get("text") or args.get("source") or ""),
            )
        if action in {"forget", "unlearn", "remove"}:
            return await self.learning.forget(
                str(args.get("topic") or args.get("query") or args.get("source") or ""))
        return await self.learning.learn(
            source=str(args.get("source") or args.get("text") or args.get("url") or args.get("path") or ""),
            topic=str(args.get("topic") or ""),
        )

    def _study_generator(self):
        """An async prompt→text callable for model-written flashcards, or None so
        the study engine falls back to extractive cards. Same model path the
        document/research tools use."""
        ask = getattr(getattr(self, "research", None), "_ask", None)
        if ask is None:
            return None

        async def gen(prompt: str) -> str:
            return await ask(prompt, "study coach")
        return gen

    async def study_tool(self, args: dict[str, Any]) -> ToolResult:
        """Spaced-repetition study & recall over any subject (Mark XXIV).

        The interactive flow is standard SRS self-grading: 'review' surfaces the
        next due card as Q + reference answer; ORION asks the question, lets the
        user attempt it, reveals the answer, then 'grade' records how well they
        recalled it and advances the schedule. The engine remembers the card in
        play, so 'grade' needs no id.
        """
        engine = getattr(self, "study", None)
        if engine is None:
            from .study import StudyEngine
            engine = StudyEngine(generate=self._study_generator())
            self.study = engine
        from .study import human_interval
        action = str(args.get("action") or "review").lower().strip()
        deck = str(args.get("deck") or "").strip()

        if action in {"add", "new", "card"}:
            front = str(args.get("front") or args.get("question") or args.get("q") or "")
            back = str(args.get("back") or args.get("answer") or args.get("a") or "")
            try:
                card = engine.add(front, back, deck=deck or "General",
                                  source=str(args.get("source") or ""))
            except ValueError:
                return ToolResult("A card needs both a question and an answer.", ok=False)
            return ToolResult(f'Added to the {card.deck} deck: "{card.front}"')

        if action in {"generate", "make", "cards_from", "from"}:
            text = str(args.get("text") or args.get("material") or args.get("source") or "")
            path = str(args.get("path") or args.get("file") or "")
            if not text and path:
                # Reuse the document reader the research/library stack already
                # uses, so "make flashcards from this paper" works on a PDF or a
                # .docx, not only a plain-text file (read_text() on a PDF returns
                # binary noise and silently produces nonsense cards).
                from .document_intelligence import read_document
                source_path = Path(path).expanduser()
                if not source_path.is_file():
                    return ToolResult(f"I can't find a file at {source_path}.", ok=False)
                try:
                    text = read_document(source_path)
                except Exception as exc:
                    return ToolResult(
                        f"Couldn't read {source_path.name}: {first_line(exc, 120)}",
                        ok=False)
                if not text.strip():
                    return ToolResult(
                        f"{source_path.name} has no extractable text — if it is a "
                        "scanned PDF, ingest it first so OCR can read it.", ok=False)
            count = int(args.get("count") or args.get("n") or 8)
            cards = await engine.generate_cards(
                text, deck=deck or "General", count=count,
                source=str(args.get("source") or path))
            if not cards:
                return ToolResult("There wasn't enough material there to make cards from.",
                                  ok=False)
            preview = "\n".join(f"  • {c.front}" for c in cards[:5])
            more = f"\n  …and {len(cards) - 5} more" if len(cards) > 5 else ""
            return ToolResult(f"Made {len(cards)} cards for the "
                              f"{deck or 'General'} deck:\n{preview}{more}")

        if action in {"review", "quiz", "next", "due", "test"}:
            card = engine.next_due(deck or None)
            if card is None:
                s = engine.stats(deck or None)
                where = f" in {deck}" if deck else ""
                return ToolResult(f"Nothing's due{where} right now — {s['total']} cards "
                                  f"total, {s['mastered']} mastered. Well ahead.")
            remaining = len(engine.due(deck or None, limit=10_000))
            return ToolResult(
                f"Card {card.id} ({card.deck}) — {remaining} due.\n"
                f"Q: {card.front}\nA: {card.back}")

        if action in {"grade", "answer", "score", "mark", "rate"}:
            if args.get("quality") is not None:
                quality = int(args.get("quality"))
            elif args.get("correct") is not None:
                v = args.get("correct")
                quality = 5 if (v is True or str(v).lower() in {"1", "true", "yes", "correct"}) else 2
            else:
                quality = 4
            card_id = int(args["id"]) if str(args.get("id") or "").strip().isdigit() else None
            card = engine.grade(quality, card_id=card_id)
            if card is None:
                return ToolResult("I'm not sure which card to grade — start a review first.",
                                  ok=False)
            when = human_interval(card.interval_days)
            nxt = engine.next_due(card.deck)
            if nxt is None:
                return ToolResult(f'Noted — "{card.front}" comes back in {when}. '
                                  "That's everything due. Nicely done.")
            return ToolResult(f'Noted — back in {when}.\nNext, card {nxt.id}:\n'
                              f"Q: {nxt.front}\nA: {nxt.back}")

        if action in {"stats", "progress", "mastery"}:
            s = engine.stats(deck or None)
            ret = f"{s['retention']}% recall" if s["retention"] is not None else "no reviews yet"
            return ToolResult(
                f"{s['deck']}: {s['total']} cards — {s['due']} due, {s['new']} new, "
                f"{s['learning']} learning, {s['mastered']} mastered ({ret}).")

        if action in {"decks", "list"}:
            decks = engine.decks()
            if not decks:
                return ToolResult("No decks yet — add a card or generate some from your notes.")
            return ToolResult("\n".join(
                f"{d['deck']}: {d['total']} cards, {d['due']} due, {d['mastered']} mastered"
                for d in decks))

        return ToolResult(f"Unsupported study action: {action}.", ok=False)

    def transcript_tool(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "export").lower().strip()
        if action in {"path", "where"}:
            return ToolResult(f"This session's transcript is being recorded to "
                              f"{self.memory.transcript_path()}.")
        if action in {"recall", "search"}:
            return self.recall_conversation({"query": args.get("query") or "", "limit": args.get("limit") or 10})
        # export
        path = self.memory.export_transcript_markdown()
        if not path:
            return ToolResult("There's nothing recorded to export yet.")
        return ToolResult(f"Exported this session's verbatim transcript to {path}.")

    def programming_knowledge_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.programming is None:
            return ToolResult("The programming knowledge base is not available.", ok=False)
        query = str(args.get("query") or args.get("topic") or "")
        answer = self.programming.answer(query)
        if answer:
            return ToolResult(answer)
        # Fall back to memory search over the seeded corpus.
        rows = self.memory.query(query or "programming", limit=5)
        hits = [r.get("value", "") for r in rows if str(r.get("key_ref", "")).startswith("prog_")]
        if hits:
            return ToolResult("\n".join(f"- {h}" for h in hits))
        return ToolResult("Ask me about complexity, data structures, concurrency, patterns, "
                          "databases, security, testing or a language.")

    def cyber_knowledge_tool(self, args: dict[str, Any]) -> ToolResult:
        from . import electronics_teaching
        query = str(args.get("query") or args.get("topic") or "")
        boundary = electronics_teaching.construction_boundary(query)
        if boundary:
            return ToolResult(boundary, ok=False)
        if str(args.get("format") or "short").lower() != "full":
            from .learning_steps import resolve_plan
            reviewed_topic = electronics_teaching.resolve_topic(query) or query
            if resolve_plan(reviewed_topic):
                return KnowledgeDispatchMixin.cyber_curriculum_tool(
                    self, {**args, "action": "coach_start", "topic": reviewed_topic})
        if electronics_teaching.resolve_topic(query):
            return ToolResult(electronics_teaching.teach(query))
        if self.cyber is None:
            return ToolResult("The cybersecurity knowledge base is not available.", ok=False)
        answer = self.cyber.answer(query)
        if answer:
            return ToolResult(answer)
        rows = self.memory.query(query or "security", limit=5)
        hits = [r.get("value", "") for r in rows if str(r.get("key_ref", "")).startswith("cyber_")]
        if hits:
            return ToolResult("\n".join(f"- {h}" for h in hits))
        return ToolResult("Ask me about the CIA triad, threat modelling, OWASP risks, "
                          "encryption, authentication, malware, detection, incident response, "
                          "secure development or cloud hardening.")

    def cyber_curriculum_tool(self, args: dict[str, Any]) -> ToolResult:
        """Navigate the structured cybersecurity curriculum (Section 11): outline,
        search, per-module detail, progress, certifications, and a default-deny
        target-scope check.  Knowledge and safe-practice guidance only — it never
        performs an intrusive action, and offensive modules carry isolation and
        authorisation requirements."""
        from .cyber_curriculum import (
            CyberCurriculum, CERTIFICATION_TIMELINE, training_capability_note,
        )
        cur = getattr(self, "_cyber_curriculum", None)
        if cur is None:
            cur = CyberCurriculum()
            self._cyber_curriculum = cur
        action = str(args.get("action") or "outline").lower().strip()
        topic = str(args.get("topic") or args.get("query") or "").strip()
        from . import cyber_exercises, electronics_teaching
        boundary = electronics_teaching.construction_boundary(topic)
        if boundary:
            return ToolResult(boundary, ok=False)
        # Short, learner-controlled teaching is the default. Full reference
        # lessons remain an explicit choice rather than a lecture on every turn.
        from .learning_coach import LearningCoach
        coach_action = action.removeprefix("coach_")
        coach_controls = {"start", "next", "answer", "simpler", "repeat", "pause",
                          "resume", "status", "settings"}
        if ((action.startswith("coach_") and coach_action in coach_controls)
                or action in {"next", "simpler", "repeat", "pause", "resume"}):
            coach = getattr(self, "_learning_coach", None)
            if coach is None:
                coach = self._learning_coach = LearningCoach()
            return coach.handle(
                coach_action, user=str(args.get("user") or "default"), topic=topic,
                answer=str(args.get("answer") or ""), pace_minutes=args.get("pace_minutes"),
                steps_per_session=args.get("steps_per_session"), restart=args.get("restart", False))
        short_actions = {"path", "learning_path", "beginner", "start", "teach", "lesson",
                         "explain", "learn", "code", "example", "secure_code", "exercise",
                         "practice", "practise", "quiz"}
        if action in short_actions and str(args.get("format") or "short").lower() != "full":
            return KnowledgeDispatchMixin.cyber_curriculum_tool(
                self, {**args, "action": "coach_start", "topic": topic})
        if action in {"path", "learning_path", "beginner", "start"}:
            return ToolResult(cyber_exercises.learning_path())
        if action in {"code", "example", "secure_code"}:
            if electronics_teaching.resolve_topic(topic):
                return ToolResult(electronics_teaching.teach(topic))
            return ToolResult(cyber_exercises.code_lesson(topic),
                              ok=cyber_exercises.resolve_topic(topic) is not None)
        if action in {"exercise", "practice", "practise", "quiz"}:
            return ToolResult(cyber_exercises.exercise(topic),
                              ok=cyber_exercises.resolve_topic(topic) is not None)
        if action in {"check", "answer", "check_answer"}:
            return ToolResult(cyber_exercises.check_answer(topic, str(args.get("answer") or "")),
                              ok=cyber_exercises.resolve_topic(topic) is not None)
        # ── teaching layer: recognition-first lessons + a code identifier ─────
        # The curriculum is the syllabus; this teaches a specific technique with
        # the KEY IDENTIFIERS that let a student recognise it in the wild.
        if action in {"teach", "lesson", "explain", "learn"}:
            from . import cyber_teaching
            if electronics_teaching.resolve_topic(topic):
                return ToolResult(electronics_teaching.teach(topic))
            sections = []
            if cyber_teaching.resolve_topic(topic):
                sections.append(cyber_teaching.teach(topic))
            if cyber_exercises.resolve_topic(topic):
                sections.append(cyber_exercises.code_lesson(topic))
            if sections:
                return ToolResult("\n\n".join(sections))
            return ToolResult(cyber_teaching.teach(topic)
                              + "\nUse action='path' for a beginner sequence, 'lessons' "
                              "for all topics including electronics defence, or 'code' "
                              "for sql_injection, xss, access_control or log_analysis.")
        if action in {"identify", "recognise", "recognize", "what_is_this",
                      "analyse_code", "analyze_code"}:
            from . import cyber_teaching
            code = str(args.get("code") or args.get("snippet")
                       or args.get("query") or "")
            if not code.strip():
                return ToolResult("Paste the code or describe the behaviour and "
                                  "I'll tell you which technique it resembles.",
                                  ok=False)
            return ToolResult(cyber_teaching.identify_report(code))
        if action in {"lessons", "topics", "catalogue", "catalog"}:
            from . import cyber_teaching
            rows = cyber_teaching.catalogue()
            return ToolResult(
                "Techniques I teach (recognition-first):\n" + "\n".join(
                    f"  • {r['title']} ({r['category']}) — {r['demo_policy']}"
                    for r in rows)
                + "\n\nOffline code exercises (action='code' or 'exercise'):\n"
                + "\n".join(f"  • {r['topic']} — {r['title']}"
                            for r in cyber_exercises.catalogue())
                + "\n\nElectronics and hardware defence (action='teach'):\n"
                + "\n".join(f"  • {r['topic']} — {r['title']}"
                            for r in electronics_teaching.catalogue())
                + "\nUse action='path' for the beginner learning sequence.")
        if action in {"outline", "list", "modules"}:
            rows = cur.outline()
            lines = [f"Cybersecurity curriculum v{cur.version} — {len(rows)} modules "
                     f"({training_capability_note()})"]
            for r in rows:
                gate = " [authorised+isolated]" if r["isolation_required"] else ""
                lines.append(f"  {r['module']:>2}. {r['title']} — {r['safety']}{gate}")
            return ToolResult("\n".join(lines))
        if action in {"search", "find"}:
            hits = cur.search(str(args.get("query") or ""))
            if not hits:
                return ToolResult("No curriculum modules matched that query.", ok=False)
            return ToolResult("Matches:\n" + "\n".join(
                f"  Module {h['module']}: {h['title']} ({h['safety']})" for h in hits))
        if action in {"module", "detail"}:
            m = cur.module(int(args.get("module") or 0))
            if m is None:
                return ToolResult("No such module.", ok=False)
            d = m.as_dict()
            reminder = f"\n  ⚠ {m.legal_reminder}" if m.legal_reminder else ""
            return ToolResult(
                f"Module {m.number}: {m.title} ({m.safety.value})\n"
                f"  Topics: {', '.join(m.topics[:20])}\n"
                f"  Labs: {', '.join(l.name for l in m.labs) or '—'}\n"
                f"  Projects: {', '.join(p.name for p in m.projects) or '—'}\n"
                f"  Prerequisites: {m.prerequisites or '—'}{reminder}")
        if action in {"progress", "status"}:
            user = str(args.get("user") or "default")
            return ToolResult(json.dumps(cur.progress(user), indent=2))
        if action in {"complete", "mark_complete"}:
            user = str(args.get("user") or "default")
            try:
                p = cur.mark_complete(user, int(args.get("module") or 0))
            except ValueError as exc:
                return ToolResult(str(exc), ok=False)
            return ToolResult(f"Marked complete. Progress: {p['percent']}% "
                              f"(next: module {p['next_module']}).")
        if action in {"certifications", "certs"}:
            lines = [f"{yr}: {', '.join(certs)}" for yr, certs in CERTIFICATION_TIMELINE.items()]
            return ToolResult("Recommended certification timeline:\n" + "\n".join(lines))
        if action in {"scope_check", "authorise", "authorize"}:
            from .pentest_lab import target_allowed
            target = str(args.get("target") or "")
            ok, reason = target_allowed(target, allow_guidance=False)
            return ToolResult(f"Target '{target}': {'ALLOWED' if ok else 'DENIED'} — {reason}",
                              ok=ok)
        return ToolResult(
            "cyber_curriculum actions: outline, search, module, progress, "
            "complete, certifications, scope_check, path, teach, lessons, identify, "
            "code, exercise, check (topic + answer); coach_start, coach_next, "
            "coach_answer, coach_simpler, coach_repeat, coach_pause, coach_resume, "
            "coach_status, coach_settings. Teaching defaults to short; format='full' "
            "requests a complete reference lesson.", ok=False)

    def language_tutor_tool(self, args: dict[str, Any]) -> ToolResult:
        """The immersion tutor (CAP-01) — teach a language, Spanish by default.

        Real spaced repetition (SM-2) so words resurface as you're about to
        forget them, offline pronunciation grading against Whisper's transcript,
        and an immersion directive that keeps a session in the target language.
        Deterministic and self-contained — the conversation itself is carried by
        the normal model routing using the directive this returns.
        """
        from . import language_tutor as lt
        tutor = getattr(self, "_language_tutor", None)
        if tutor is None:
            tutor = lt.LanguageTutor()
            self._language_tutor = tutor
        action = str(args.get("action") or "start").strip().lower()
        language = args.get("language") or args.get("lang")

        if action in {"start", "session", "practice", "converse", "immersion"}:
            level = args.get("level")
            immersion = str(args.get("mode") or ("immersion" if action == "immersion"
                            else "immersion")).lower() not in {"bilingual", "both", "english"}
            session = tutor.start_session(language, level, immersion=immersion,
                                          topic=str(args.get("topic") or ""))
            stats = tutor.stats(session.language.code)
            return ToolResult(
                f"Language tutor engaged — {session.language.name} at {session.level}, "
                f"{'immersion' if session.immersion else 'bilingual'} mode. "
                f"{stats['due']} of {stats['total']} words are due for review.\n\n"
                + session.directive())

        if action in {"review", "due", "quiz", "flashcards"}:
            cards = tutor.due(language, limit=int(args.get("limit") or 12))
            if not cards:
                return ToolResult("Nothing is due right now — you're caught up. "
                                  "Add words with action='add' or start a session.")
            lang = lt.resolve_language(language)
            lines = [f"{len(cards)} {lang.name} card(s) due for review:"]
            for c in cards:
                ex = f"   e.g. {c.example}" if c.example else ""
                lines.append(f"  [#{c.id}] {c.term} — {c.translation}{ex}")
            lines.append("\nGrade each with action='grade', card=<id>, quality=0-5.")
            return ToolResult("\n".join(lines))

        if action in {"add", "add_word", "learn_word"}:
            term = str(args.get("term") or args.get("word") or "").strip()
            translation = str(args.get("translation") or args.get("meaning") or "").strip()
            if not term or not translation:
                return ToolResult("Give me the word and its meaning: "
                                  "term='gato', translation='cat'.", ok=False)
            card = tutor.add_word(term, translation, language=language,
                                  example=str(args.get("example") or ""),
                                  level=args.get("level"))
            return ToolResult(f"Added '{card.term}' → '{card.translation}' "
                              f"[#{card.id}]. It's due for its first review now.")

        if action in {"grade", "answer", "score_review"}:
            try:
                card_id = int(args.get("card") or args.get("id") or 0)
                quality = int(args.get("quality") or args.get("q") or 0)
            except (TypeError, ValueError):
                return ToolResult("Grade needs card=<id> and quality=0-5.", ok=False)
            card = tutor.grade(card_id, quality)
            if card is None:
                return ToolResult(f"No card #{card_id}.", ok=False)
            when = ("tomorrow" if card.interval_days <= 1
                    else f"in {int(round(card.interval_days))} days")
            return ToolResult(f"Noted. '{card.term}' comes back {when} "
                              f"(ease {card.ease}, {card.repetitions} in a row).")

        if action in {"pronounce", "pronunciation", "say"}:
            expected = str(args.get("expected") or args.get("phrase") or "").strip()
            heard = str(args.get("heard") or args.get("transcript")
                        or args.get("said") or "").strip()
            if not expected:
                return ToolResult("Tell me the phrase to check against "
                                  "(expected='buenos días').", ok=False)
            score = tutor.pronounce(expected, heard,
                                    accent_sensitive=bool(args.get("accent_sensitive")))
            return ToolResult(f"Pronunciation: {score.score}/100 — {score.verdict}")

        if action in {"stats", "progress", "streak"}:
            lang = lt.resolve_language(language)
            s = tutor.stats(lang.code)
            return ToolResult(f"{lang.name}: {s['total']} words, {s['due']} due now, "
                              f"{s['learned']} learned (reviewed twice or more).")

        if action in {"languages", "list_languages"}:
            return ToolResult("Languages I can tutor: " + ", ".join(
                f"{l.name} ({l.native})" for l in lt.LANGUAGES.values()))

        return ToolResult(
            "language_tutor actions: start, review, add, grade, pronounce, "
            "stats, languages. Spanish is the default language.", ok=False)

    async def read_documents_tool(self, args: dict[str, Any]) -> ToolResult:
        """Document intelligence (CAP-06) — read a folder/files and answer across
        them, with file:line citations. Retrieval is offline TF-IDF (instant,
        deterministic); the answer is grounded in the retrieved passages, and
        cited. Reads txt/md/code/csv/json and, if the optional readers are
        installed, PDF and Word."""
        from .document_intelligence import DocumentIntelligence
        engine = getattr(self, "_doc_intel", None)
        if engine is None:
            engine = DocumentIntelligence()
            self._doc_intel = engine
        action = str(args.get("action") or "").strip().lower()
        path = str(args.get("path") or args.get("folder") or args.get("file") or "").strip()

        def _generate():
            agent = getattr(self, "research", None)
            ask = getattr(agent, "_ask", None)
            if ask is None:
                return None
            async def gen(prompt: str) -> str:
                return await ask(prompt, "document analyst")
            return gen

        # Reading a folder/file into the corpus.
        if action in {"read", "ingest", "load", "scan", "open", "index"} or (
                path and not action):
            if not path:
                return ToolResult("Which folder or file should I read? Give a "
                                  "path.", ok=False)
            p = Path(path)
            recursive = str(args.get("recursive") or "true").lower() not in {"false", "no", "0"}
            # Off the loop: parsing a folder of PDFs froze the face and voice
            # for as long as it took.
            report = await asyncio.to_thread(
                lambda: engine.ingest_paths([p]) if p.is_file()
                else engine.ingest_folder(p, recursive=recursive))
            if report.files_read == 0:
                return ToolResult(f"I couldn't read anything at {path}. "
                                  + report.describe(), ok=False)
            return ToolResult(report.describe()
                              + " Ask me anything about them with action='ask'.")

        # Asking a question across what's been read.
        if action in {"ask", "question", "query", "answer"}:
            question = str(args.get("question") or args.get("query")
                           or args.get("text") or "").strip()
            if not question:
                return ToolResult("What would you like to know about the "
                                  "documents?", ok=False)
            if path and not engine.passages:      # convenience: read then ask
                p = Path(path)
                await asyncio.to_thread(
                    lambda: engine.ingest_paths([p]) if p.is_file()
                    else engine.ingest_folder(p))
            if not engine.passages:
                return ToolResult("I haven't read any documents yet. Use "
                                  "action='read' with a path first.", ok=False)
            answer = await engine.ask(question, k=int(args.get("limit") or 6),
                                      generate=_generate())
            cites = ("\n\nSources: " + "; ".join(answer.sources)) if answer.sources else ""
            return ToolResult(answer.text + cites)

        if action in {"digest", "summary", "summarise", "summarize", "overview"}:
            if not engine.passages:
                return ToolResult("I haven't read any documents yet. Use "
                                  "action='read' with a path first.", ok=False)
            return ToolResult(await engine.digest(generate=_generate()))

        if action in {"search", "find", "locate"}:
            query = str(args.get("query") or args.get("text") or "").strip()
            hits = await asyncio.to_thread(engine.search, query,
                                           int(args.get("limit") or 8))
            if not hits:
                return ToolResult("No passages matched that." if engine.passages
                                  else "No documents read yet.", ok=False)
            return ToolResult("Matches:\n" + "\n".join(
                f"  {h.passage.citation} (score {h.score}) — "
                f"{h.passage.text[:120].strip()}…" for h in hits))

        if action in {"status", "loaded"}:
            return ToolResult(f"{engine.file_count} file(s) read, "
                              f"{len(engine.passages)} passages in memory.")

        if action in {"clear", "reset", "forget"}:
            self._doc_intel = DocumentIntelligence()
            return ToolResult("Cleared the document corpus.")

        return ToolResult(
            "read_documents actions: read (path), ask (question), digest, "
            "search (query), status, clear.", ok=False)

    def privacy_guard_tool(self, args: dict[str, Any]) -> ToolResult:
        """The spillage guard (CAP-08) — check whether a piece of text is safe to
        send/post/log, or redact it. Catches API keys, private-key blocks,
        credential assignments, card numbers and ORION's OWN configured secrets
        echoed back. A live-credential finding is raised on the safety channel,
        which overrides standby."""
        from .spillage_guard import SpillageGuard
        guard = getattr(self, "_spillage_guard", None)
        if guard is None:
            guard = SpillageGuard(known_secrets=self._own_secret_values())
            self._spillage_guard = guard
        action = str(args.get("action") or "scan").strip().lower()
        text = str(args.get("text") or args.get("content") or args.get("message") or "")
        if action in {"redact", "sanitise", "sanitize", "clean"}:
            if not text:
                return ToolResult("Give me the text to redact.", ok=False)
            return ToolResult(guard.redact(text))
        if action in {"scan", "check", "is_safe", "guard", "inspect"}:
            if not text:
                return ToolResult("Give me the text to check for secrets.", ok=False)
            report = guard.guard(text, bus=getattr(self, "bus", None))
            return ToolResult(report.describe(), ok=report.safe)
        return ToolResult("privacy_guard actions: scan (text), redact (text).",
                          ok=False)

    async def standing_questions_tool(self, args: dict[str, Any]) -> ToolResult:
        """Standing questions (CAP-05) — interests ORION revisits on a cadence,
        reporting only what's NEW since last time. 'add' registers one (with a
        cadence); 'list' shows them; 'remove' drops one; 'check' researches the
        ones that are due now and reports the delta; 'due' lists what's ready to
        revisit. Designed so a standby loop can call 'check' as permitted
        background work."""
        from .standing_questions import StandingQuestions
        store = getattr(self, "_standing_questions", None)
        if store is None:
            store = StandingQuestions()
            self._standing_questions = store
        action = str(args.get("action") or "list").strip().lower()
        text = str(args.get("question") or args.get("text") or args.get("topic") or "").strip()

        if action in {"add", "register", "watch", "follow", "track"}:
            if not text:
                return ToolResult("What should I keep an eye on? Give me the "
                                  "question or interest.", ok=False)
            q = store.add(text, args.get("cadence"))
            return ToolResult(f"Standing question registered [#{q.id}]: '{q.text}' "
                              f"— I'll revisit it every {int(q.cadence_hours)}h and "
                              "tell you only what's new.")
        if action in {"list", "show", "all"}:
            items = store.list()
            if not items:
                return ToolResult("No standing questions yet. Add one with "
                                  "action='add'.")
            return ToolResult("Standing questions:\n" + "\n".join(
                f"  [#{q.id}] {q.text} (every {int(q.cadence_hours)}h"
                + (", due now" if q.is_due() else "") + ")" for q in items))
        if action in {"remove", "delete", "stop", "unwatch"}:
            try:
                qid = int(args.get("id") or args.get("question_id") or 0)
            except (TypeError, ValueError):
                qid = 0
            return (ToolResult(f"Stopped watching #{qid}.") if store.remove(qid)
                    else ToolResult(f"No standing question #{qid}.", ok=False))
        if action in {"due", "ready"}:
            due = store.due()
            if not due:
                return ToolResult("Nothing is due to be revisited right now.")
            return ToolResult("Due to revisit:\n" + "\n".join(
                f"  [#{q.id}] {q.text}" for q in due))
        if action in {"check", "run", "update", "brief", "refresh"}:
            researcher = self._standing_researcher()
            if researcher is None:
                return ToolResult("I can't research right now (no model access), "
                                  "so I can't refresh the standing questions.", ok=False)
            deltas = await store.run_due(researcher)
            if not deltas:
                return ToolResult("Nothing was due, or nothing new came back.")
            news = [d for d in deltas if d.has_news]
            if not news:
                return ToolResult(f"Revisited {len(deltas)} standing question(s) — "
                                  "no new developments.")
            return ToolResult("Standing-question update:\n\n"
                              + "\n\n".join(d.describe() for d in news))
        return ToolResult(
            "standing_questions actions: add, list, remove, due, check.", ok=False)

    def catch_up_tool(self, args: dict[str, Any]) -> ToolResult:
        """Situation report (Mark XXIII) — "catch me up" / "where do things
        stand". Fuses what's NEW on the user's standing questions with what was
        recently DECIDED (from the Rewind transcripts) into one prioritised,
        sectioned brief, worst first. Reads existing local state — it does not
        research or call a model here, so it's instant."""
        from . import situation_report as sr
        # Recent decisions from the verbatim transcripts (Rewind).
        decisions: list[dict[str, Any]] = []
        try:
            from .rewind import RewindTimeline
            timeline = getattr(self, "_rewind", None)
            if timeline is None:
                directory = None
                memory = getattr(self, "memory", None)
                tpath = getattr(memory, "transcript_path", None)
                if callable(tpath):
                    try:
                        directory = tpath().parent
                    except Exception:
                        directory = None
                if directory is None:
                    from .constants import BASE_DIR
                    directory = BASE_DIR / "conversations"
                timeline = RewindTimeline(directory)
                self._rewind = timeline
            timeline.reload()
            for turn in timeline.decisions(limit=5):
                decisions.append({"text": turn.content[:160], "when": turn.clock})
        except Exception:
            pass
        # The standing questions themselves (what he's watching), plus any that
        # are due to be revisited — surfaced as light 'watch' lines.
        watch: list[dict[str, Any]] = []
        try:
            store = getattr(self, "_standing_questions", None)
            if store is None:
                from .standing_questions import StandingQuestions
                store = StandingQuestions()
                self._standing_questions = store
            due = {q.id for q in store.due()}
            for q in store.list():
                pts = ["due to revisit"] if q.id in due else []
                watch.append({"question": q.text, "points": pts})
        except Exception:
            pass

        # Mark XXV — the daily brief also carries what's waiting in the learning
        # loop: flashcards due for review, and a focus block that has run past
        # its length. Read-only: the stores are only opened if they already
        # exist, so a user who has never studied gets no phantom line.
        tasks: list[dict[str, Any]] = []
        from .constants import CONFIG_DIR
        try:
            engine = getattr(self, "study", None)
            if engine is None and (CONFIG_DIR / "study.db").exists():
                from .study import StudyEngine
                engine = StudyEngine()
                self.study = engine
            if engine is not None:
                n = len(engine.due(limit=10_000))
                if n:
                    tasks.append({"text": f"{n} flashcard{'s' if n != 1 else ''} "
                                  "due for review", "overdue": n > 20})
        except Exception:
            pass
        try:
            journal = getattr(self, "decisions", None)
            if journal is None and (CONFIG_DIR / "decisions.db").exists():
                from .decisions import DecisionJournal
                journal = DecisionJournal()
                self.decisions = journal
            if journal is not None:
                ready = journal.due()
                if ready:
                    tasks.append({"text": f"{len(ready)} decision(s) ready to review "
                                  "— did they turn out as predicted?",
                                  "overdue": len(ready) > 3})
        except Exception:
            pass
        try:
            fe = getattr(self, "focus", None)
            if fe is None and (CONFIG_DIR / "focus.db").exists():
                from .focus import FocusEngine
                fe = FocusEngine()
                self.focus = fe
            if fe is not None:
                s = fe.active()
                if s is not None and s.is_break_due():
                    tasks.append({"text": f'focus block "{s.label}" is up — '
                                  "time for a break", "overdue": True})
        except Exception:
            pass

        report = sr.build_report(watch_deltas=watch, decisions=decisions, tasks=tasks)
        return ToolResult(report.render(greeting=str(args.get("greeting") or "")))

    async def rewind_tool(self, args: dict[str, Any]) -> ToolResult:
        """Rewind (CAP-04) — the scrubbable day. Reads ORION's own verbatim
        transcripts and answers 'what did we say/decide about X', 'recap
        yesterday', 'what happened today'. Deterministic — it reads the
        conversation logs, it does not re-imagine them."""
        from .rewind import RewindTimeline
        timeline = getattr(self, "_rewind", None)
        if timeline is None:
            directory = None
            memory = getattr(self, "memory", None)
            tpath = getattr(memory, "transcript_path", None)
            if callable(tpath):
                try:
                    directory = tpath().parent
                except Exception:
                    directory = None
            if directory is None:
                from .constants import BASE_DIR
                directory = BASE_DIR / "conversations"
            timeline = RewindTimeline(directory)
            self._rewind = timeline
        # Transcripts grow during the session, so every call re-reads them —
        # off the event loop: parsing months of history stalled voice replies.
        await asyncio.to_thread(timeline.refresh)
        action = str(args.get("action") or "recap").strip().lower()
        query = str(args.get("query") or args.get("text") or args.get("topic") or "").strip()
        day = args.get("day") or args.get("when")

        if action in {"search", "find", "when_did"}:
            if not query:
                return ToolResult("What should I search the history for?", ok=False)
            hits = await asyncio.to_thread(
                timeline.search, query, limit=int(args.get("limit") or 12))
            if not hits:
                # Fall back to relevance ranking so a near-miss still surfaces.
                hits = await asyncio.to_thread(
                    timeline.relevant, query, limit=int(args.get("limit") or 12))
            if not hits:
                return ToolResult(f"I found nothing in our history about '{query}'.")
            return ToolResult(f"Found {len(hits)} mention(s) of '{query}':\n"
                              + "\n".join(t.line()[:200] for t in hits))
        if action in {"recall", "relevant", "what_did_we_say"}:
            if not query:
                return ToolResult("What should I recall from our history?", ok=False)
            hits = await asyncio.to_thread(
                timeline.relevant, query, limit=int(args.get("limit") or 10))
            if not hits:
                return ToolResult(f"I don't recall anything about '{query}'.")
            return ToolResult(f"Most relevant to '{query}' in our history:\n"
                              + "\n".join(t.line()[:220] for t in hits))
        if action in {"decisions", "decide", "decided", "what_did_we_decide"}:
            hits = timeline.decisions(query, limit=int(args.get("limit") or 12))
            if not hits:
                topic = f" about '{query}'" if query else ""
                return ToolResult(f"I don't have a recorded decision{topic}.")
            return ToolResult("Decisions on record:\n"
                              + "\n".join(f"  • {t.line()[:200]}" for t in hits))
        if action in {"timeline", "day", "history", "show"}:
            turns = timeline.timeline(day, limit=int(args.get("limit") or 40))
            if not turns:
                return ToolResult("Nothing recorded for that period.")
            return ToolResult("\n".join(t.line()[:200] for t in turns))
        if action in {"days", "when"}:
            days = timeline.days()
            if not days:
                return ToolResult("No conversation history recorded yet.")
            return ToolResult("Days with recorded conversation: "
                              + ", ".join(d.isoformat() for d in days))
        # default: recap
        researcher = None
        agent = getattr(self, "research", None)
        ask = getattr(agent, "_ask", None)
        if ask is not None:
            async def researcher(prompt: str) -> str:   # noqa: E306
                return await ask(prompt, "recap")
        return ToolResult(await timeline.recap(day, generate=researcher))

    def _standing_researcher(self):
        """A concise researcher for delta checks, from the research agent's model
        access. Returns None when no model is available."""
        agent = getattr(self, "research", None)
        ask = getattr(agent, "_ask", None)
        if ask is None:
            return None
        async def researcher(question: str) -> str:
            prompt = (
                "Give a concise, bulleted update on the CURRENT state of this "
                "topic — the most recent, concrete developments only, one per "
                f"line, no preamble:\n\n{question}")
            return await ask(prompt, "standing-question analyst")
        return researcher

    def _own_secret_values(self) -> list[str]:
        """ORION's own configured secret values, so echoing one back is caught.
        Values only — never their names, never logged. Best-effort and quiet."""
        values: list[str] = []
        try:
            import json
            from .constants import CONFIG_DIR
            path = CONFIG_DIR / "api_keys.json"
            if path.is_file():
                from .key_seal import load_json
                data = load_json(path)
                for v in _flatten_strings(data):
                    if isinstance(v, str) and len(v) >= 12:
                        values.append(v)
        except Exception:
            pass
        return values

    def muscle_memory_tool(self, args: dict[str, Any]) -> ToolResult:
        """Muscle memory (CAP-07) — record a browser task once, replay it with
        new inputs. 'record' starts capturing; 'stop' generalises what was
        captured into a named, parameterised skill; 'list' shows skills; 'run'
        produces the concrete plan for a skill with given parameters (executed
        through the browser co-pilot); 'forget' deletes one. The values you type
        become the skill's parameters."""
        from .muscle_memory import MuscleMemory
        mm = getattr(self, "_muscle_memory", None)
        if mm is None:
            mm = MuscleMemory()
            self._muscle_memory = mm
        action = str(args.get("action") or "").strip().lower()
        name = str(args.get("name") or args.get("skill") or "").strip()

        if action in {"record", "start", "learn"}:
            if not name:
                return ToolResult("What should I call this skill? Give it a name, "
                                  "then do the task.", ok=False)
            return ToolResult(mm.start_recording(name))
        if action in {"step", "add"} and mm.recording:
            mm.record(str(args.get("op") or "").strip().lower(),
                      str(args.get("target") or ""), str(args.get("value") or ""))
            return ToolResult("Step recorded.")
        if action in {"stop", "save", "finish"}:
            skill = mm.stop_recording()
            if skill is None:
                return ToolResult("Nothing was recorded, so there's no skill to save.",
                                  ok=False)
            return ToolResult(f"Learned it.\n{skill.describe()}")
        if action in {"list", "skills", "show"}:
            skills = mm.skills()
            if not skills:
                return ToolResult("No skills recorded yet. Record one with "
                                  "action='record', name='...'.")
            return ToolResult("Recorded skills:\n" + "\n".join(
                f"  • {s.name}" + (f" ({', '.join(s.params)})" if s.params else "")
                for s in skills))
        if action in {"run", "replay", "do"}:
            params = args.get("params") if isinstance(args.get("params"), dict) else {}
            # also accept flat key=value params passed alongside
            for k, v in args.items():
                if k not in {"action", "name", "skill", "params", "op", "target", "value"}:
                    params.setdefault(k, v)
            skill, steps, missing = mm.run(name, params)
            if skill is None:
                return ToolResult(f"I don't have a skill called '{name}'.", ok=False)
            if missing:
                return ToolResult(f"Skill '{skill.name}' needs: {', '.join(missing)}. "
                                  "Provide them and I'll run it.", ok=False)
            plan_text = "\n".join(
                f"  {i}. {s['op']}" + (f" → {s['target']}" if s['target'] else "")
                + (f" = {s['value']}" if s['value'] else "")
                for i, s in enumerate(steps, 1))
            return ToolResult(f"Plan for '{skill.name}':\n{plan_text}\n\n"
                              "Say 'go' and I'll run it through the browser co-pilot.")
        if action in {"forget", "delete", "remove"}:
            return (ToolResult(f"Forgot '{name}'.") if mm.forget(name)
                    else ToolResult(f"No skill called '{name}'.", ok=False))
        return ToolResult(
            "muscle_memory actions: record (name), stop, list, run (name + params), "
            "forget.", ok=False)

    def pentest_lab_tool(self, args: dict[str, Any]) -> ToolResult:
        """The proving ground (CAP-09) — a guided, lawful, recognition-first
        walk-through of a technique on a legal range. It TEACHES and gates; it
        never attacks anything. Dangerous categories (malware/ransomware) have no
        lab, by design. Educational only."""
        from .pentest_lab import PentestLab, target_allowed
        lab = getattr(self, "_pentest_lab", None)
        if lab is None:
            lab = PentestLab()
            self._pentest_lab = lab
        action = str(args.get("action") or "").strip().lower()
        topic = str(args.get("topic") or args.get("technique") or args.get("query") or "").strip()
        target = str(args.get("target") or "").strip()

        if target:
            ok, reason = target_allowed(target)
            if not ok:
                return ToolResult(f"I won't guide a lab against '{target}' — {reason}.",
                                  ok=False)

        if action in {"list", "labs", "catalogue", "catalog"} or (not action and not topic):
            rows = lab.catalogue()
            return ToolResult(
                "Guided labs (lawful ranges only, recognition-first):\n"
                + "\n".join(f"  • {r['topic']} — {r['title']} ({r['steps']} steps)"
                            for r in rows)
                + "\nStart one with action='start', topic='sql_injection'.")
        if action in {"start", "begin", "practice", "practise"} or (topic and not action):
            ok, text = lab.start(topic or "")
            return ToolResult(text, ok=ok)
        if action in {"step", "current", "where"}:
            return ToolResult(lab.current_step_text())
        if action in {"next", "continue", "done"}:
            return ToolResult(lab.next_step())
        if action in {"hint", "help", "stuck"}:
            return ToolResult(lab.hint())
        if action in {"reset", "stop", "quit"}:
            return ToolResult(lab.reset())
        return ToolResult(
            "pentest_lab actions: list, start (topic), step, next, hint, reset. "
            "Educational, lawful ranges only.", ok=False)

    async def security_recon_tool(self, args: dict[str, Any]) -> ToolResult:
        """Real cybersecurity/pentesting execution — for the user's OWN
        authorized targets, labs and CTFs only. scan_host/packet_capture/
        craft_packet all route through cyber_curriculum.gate_security_action
        (default-deny; see security_recon.py). cve_lookup is a public,
        read-only NVD query and needs no authorization. write_tool delegates
        to the existing Forge pipeline (writing code touches no target)."""
        recon = getattr(self, "security_recon", None)
        if recon is None:
            return ToolResult("Security recon tooling is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action in {"authorize_target", "authorize"}:
            return recon.authorize_target(str(args.get("target") or ""))
        if action in {"revoke_target", "revoke"}:
            return recon.revoke_target(str(args.get("target") or ""))
        if action in {"list_authorized", "list"}:
            return recon.list_authorized()
        if action in {"lan_devices", "devices", "network_devices"}:
            # Passive (reads the neighbour table, sends nothing), so it needs
            # no authorised target: "scan my network" from someone at home.
            from .lan_devices import describe
            return ToolResult(await asyncio.to_thread(describe))
        if action == "scan_host":
            return await recon.scan_host(
                str(args.get("target") or ""), str(args.get("ports") or ""))
        if action == "packet_capture":
            return await recon.packet_capture(
                str(args.get("target") or ""),
                int(args.get("count") or 10), float(args.get("timeout") or 10.0))
        if action == "craft_packet":
            return await recon.craft_packet(str(args.get("target") or ""))
        if action == "cve_lookup":
            return await recon.cve_lookup(str(args.get("query") or ""))
        if action == "write_tool":
            description = str(args.get("description") or "").strip()
            if not description:
                return ToolResult("Describe the security tool to write.", ok=False)
            from .electronics_teaching import construction_boundary
            boundary = construction_boundary(description)
            if boundary:
                return ToolResult(boundary, ok=False)
            return await self.forge_tool({
                "action": "forge",
                "tool_name": str(args.get("tool_name") or "security_tool"),
                "tool_plan": (
                    "A cybersecurity tool for the user's own authorized "
                    f"testing/learning: {description}"
                ),
            })
        return ToolResult(
            "security_recon actions: authorize_target, revoke_target, "
            "list_authorized, lan_devices, scan_host, packet_capture, craft_packet, "
            "cve_lookup, write_tool.", ok=False)

    def expand_mind(self, args: dict[str, Any]) -> ToolResult:
        if self.corpus is None:
            return ToolResult("Knowledge corpus builder is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"search", "recall", "consult"}:
            query = str(args.get("query") or args.get("topic") or "")
            hits = self.corpus.search(query, limit=int(args.get("limit", 6)))
            if not hits:
                return ToolResult(f"Nothing in the study corpus matches '{query}'.")
            return ToolResult("From my expanded knowledge:\n" + "\n".join(f"- {h}" for h in hits))
        status = self.corpus.status()
        return ToolResult(
            f"Knowledge corpus: {'built' if status['built'] else 'not built'} — "
            f"{status['bytes']:,} bytes ({status['bytes']/1024/1024:.1f} MiB) across "
            f"{status['shards']} shard(s)."
        )

    async def second_brain_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.graph is None:
            return ToolResult("The knowledge graph is not available.", ok=False)
        action = str(args.get("action") or "recall").lower().strip()
        query = str(args.get("query") or args.get("text") or "")
        if action in {"recall", "search", "retrieve"}:
            answer = await asyncio.to_thread(self.graph.answer_offline, query)
            return ToolResult(answer)
        if action in {"timeline", "history"}:
            events = await asyncio.to_thread(
                self.graph.timeline_reconstruction, query, "", "",
                int(args.get("limit") or 12))
            if not events:
                return ToolResult("No matching events on the graph timeline.")
            return ToolResult("Timeline:\n" + "\n".join(
                f"- {e.at[:19]} [{e.source_type}] {e.title}: {e.text[:140]}" for e in events))
        if action in {"entity", "neighbourhood", "related"}:
            rows = await asyncio.to_thread(
                self.graph.entity_neighbourhood, str(args.get("name") or query))
            if not rows:
                return ToolResult("No such entity (or it has no relationships yet).", ok=False)
            return ToolResult("Relationships:\n" + "\n".join(
                f"- {r['source_name']} —{r['kind']}→ {r['target_name']}" for r in rows[:15]))
        if action in {"ingest", "remember", "record"}:
            event = await asyncio.to_thread(
                self.graph.ingest_record,
                str(args.get("source_type") or "conversation"),
                str(args.get("title") or query[:80]),
                query,
            )
            return ToolResult(
                f"Recorded on the graph: '{event.title}' "
                f"({len(event.entity_ids)} linked entities).")
        if action in {"stats", "status"}:
            stats = self.graph.stats()
            return ToolResult(
                f"Second brain: {stats['entities']} entities, {stats['events']} "
                f"events, {stats['relationships']} relationships.")
        return ToolResult(
            f"Unsupported second_brain action: {action}. Use recall, timeline, "
            "entity, ingest, or stats.",
            ok=False,
        )

    async def awareness_tool(self, args: dict[str, Any]) -> ToolResult:
        # "I've done all those tasks" arrived as action=clear_completed, which
        # no branch knew, so nothing was marked done (action_aliases).
        from .action_aliases import normalise_args
        args = normalise_args("awareness", args)
        action = str(args.get("action") or "situation").lower().strip()
        if action in {"situation", "report", "status"}:
            if self.cognitive_loop is None:
                return ToolResult("The cognitive loop is not available.", ok=False)
            report = self.cognitive_loop.situation_report()
            # The same task view the briefing reads (owner_tasks): the report
            # alone knew only ORION's own list, so it could deny the overdue
            # Notion tasks the briefing had just read out (25 Oct 2026).
            view_fn = getattr(self, "owner_task_view", None)
            if report.ok and callable(view_fn):
                try:
                    from dataclasses import replace
                    # Bounded: a slow board must not hold up the whole report.
                    view = await asyncio.wait_for(view_fn(), timeout=4.0)
                    report = replace(report, text=report.text + "\n- " + view.summary_line())
                except Exception:
                    pass
            return report
        if self.cognition is None:
            return ToolResult("Cognitive state is not available.", ok=False)
        if action in {"add_priority", "priority"}:
            priorities = await asyncio.to_thread(
                self.cognition.add_priority, str(args.get("text") or ""))
            return ToolResult("Priorities noted: " + "; ".join(priorities[-5:]))
        title = str(args.get("title") or args.get("text") or "")
        if action in {"add_task", "task"}:
            task = await asyncio.to_thread(
                self.cognition.add_task, title,
                str(args.get("project") or ""), str(args.get("due") or ""),
                bool(args.get("restore")))
            if task.get("dismissed"):
                return ToolResult(
                    f"Not re-added: '{task.get('title', title)}' was {task.get('reason', 'removed')} "
                    f"on {str(task.get('at', ''))[:10]}. Only add it back if the user asks for "
                    "that task back by name (restore=true).", ok=False)
            return ToolResult(f"Task tracked: {task.get('title', '?')}.")
        if action in {"complete_task", "done", "complete_tasks", "mark_done", "finish_tasks",
                      "completed"}:
            # One path for "I've done that/those": ORION's list AND the Notion
            # board the briefing reads (17 Oct 2026: completing only here left
            # the board's six tasks to be read out again next morning).
            return await self.complete_owner_tasks(
                [args.get("titles"), title], all_listed=bool(args.get("all_listed")))
        if action in {"list_tasks", "tasks"}:
            # ORION's list AND the board the briefing reads, one overdue rule.
            if callable(getattr(self, "list_owner_tasks", None)):
                return await self.list_owner_tasks()
            tasks = await asyncio.to_thread(self.cognition.list_tasks)
            if not tasks:
                return ToolResult("No open tasks.")
            await asyncio.to_thread(self.cognition.note_mentioned_tasks,
                                    [str(t.get("title") or "") for t in tasks[:30]], "awareness")
            return ToolResult("Open tasks:\n" + "\n".join(
                f"- {t.get('title', '?')}"
                + (f" (due {t['due']}{', OVERDUE' if t.get('overdue') else ''})" if t.get("due") else "")
                for t in tasks[:30]))
        if action in {"remove_task", "delete_task", "dismiss_task", "drop_task"}:
            removed = await asyncio.to_thread(self.cognition.dismiss_task, title)
            return (ToolResult(f"Removed '{removed}'. It will not come back unless you ask for it.")
                    if removed else ToolResult("No matching open task.", ok=False))
        if action in {"clear_overdue", "remove_overdue", "clear_overdue_tasks"}:
            if callable(getattr(self, "clear_overdue_owner_tasks", None)):
                return await self.clear_overdue_owner_tasks()
            removed = await asyncio.to_thread(self.cognition.clear_overdue)
            if not removed:
                return ToolResult("Nothing is overdue.")
            return ToolResult(f"Removed {len(removed)} overdue task(s): " + "; ".join(removed)
                              + ". They will not be raised again.")
        if action in {"goals", "list_goals"}:
            goals = await asyncio.to_thread(self.cognition.goals.list_goals)
            if not goals:
                return ToolResult("No goals are currently tracked.")
            return ToolResult("Goals:\n" + "\n".join(
                f"- [{g.status}] {g.title}" for g in goals[:12]))
        return ToolResult(
            f"Unsupported awareness action: {action}. Use situation, add_priority, "
            "add_task, complete_task, list_tasks, remove_task, clear_overdue, or goals.",
            ok=False,
        )

    def patch_notes_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.changelog is None:
            return ToolResult("The changelog is not available.", ok=False)
        action = str(args.get("action") or "latest").lower().strip()
        if action in {"all", "history", "full"}:
            return ToolResult(self.changelog.render_all())
        version = str(args.get("version") or "").strip()
        if version:
            release = self.changelog.find(version)
            return ToolResult(release.speak() if release else f"No release matches '{version}'.",
                              ok=release is not None)
        count = int(args.get("count", 1))
        return ToolResult(self.changelog.render_latest(max(1, count)))

    def self_changes_tool(self, args: dict[str, Any]) -> ToolResult:
        """File-level change-awareness: which of ORION's own modules changed.

        Distinct from patch_notes (curated prose) — this is the ground truth
        computed from the source tree itself.  'latest' (default) reports the
        change detected at this boot; 'scan' re-checks the tree right now;
        'history' recounts the last several update events.
        """
        tracker = getattr(self, "change_tracker", None)
        if tracker is None:
            return ToolResult("My change-awareness tracker is not available.", ok=False)
        action = str(args.get("action") or "latest").lower().strip()
        if action in {"history", "log", "recent", "updates"}:
            count = args.get("count") or args.get("limit") or 5
            try:
                limit = max(1, int(count))
            except (TypeError, ValueError):
                limit = 5
            return ToolResult(tracker.render_history(limit))
        if action in {"scan", "check", "now", "refresh"}:
            return ToolResult(tracker.render(tracker.detect(commit=False)))
        return ToolResult(tracker.render_latest())

    def code_changes_tool(self, args: dict[str, Any]) -> ToolResult:
        """What ORION actually CHANGED in himself, read from his own source.

        The third and most useful of the three "what's new?" answers:

            patch_notes    curated prose somebody wrote by hand
            self_changes   which FILES changed — true, but not an answer
            code_changes   which functions and classes changed, and what the
                           code itself says each one is for

        'latest' (default) reports the semantic diff since the last snapshot;
        'explain' describes a named part of ORION in its own words; 'snapshot'
        re-baselines without reporting.
        """
        from .code_changelog import CodeChangelog

        root = Path(__file__).resolve().parent
        changelog = CodeChangelog(root)
        action = str(args.get("action") or "latest").lower().strip()

        if action in {"explain", "describe", "what is", "whatis"}:
            target = str(args.get("name") or args.get("target") or "").strip()
            return ToolResult(changelog.explain(target), ok=bool(target))

        if action in {"snapshot", "baseline", "remember"}:
            index = changelog.snapshot()
            symbols = sum(len(m.symbols) for m in index.modules.values())
            return ToolResult(
                f"Noted how my code stands right now: {len(index.modules)} "
                f"modules, {symbols} definitions. I'll compare against this "
                "the next time you ask what I changed.")

        spoken = bool(args.get("spoken"))
        # update=False so ASKING does not consume the answer — the user can ask
        # twice, or ask and then ask for the spoken form, and get the same
        # truth both times. Re-baselining is an explicit 'snapshot'.
        return ToolResult(changelog.report(spoken=spoken, update=False))

    def speaker_id_tool(self, args: dict[str, Any]) -> ToolResult:
        """Report who ORION last heard — a male or female voice (#14).

        Input-side recognition by vocal pitch; ORION's own locked voice is never
        affected.  'who'/'status' reports the current inference; 'reset' clears it.
        """
        tracker = getattr(self, "speaker_tracker", None)
        if tracker is None:
            return ToolResult(
                "Speaker recognition isn't active (set ORION_SPEAKER_ID=1).",
                ok=False)
        action = str(args.get("action") or "who").lower().strip()
        if action in {"reset", "clear", "forget"}:
            tracker.reset()
            return ToolResult("Cleared what I'd inferred about the speaker.")
        return ToolResult(tracker.describe())

    async def document_export_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.exporter is None:
            return ToolResult("The document exporter is not available.", ok=False)
        action = str(args.get("action") or "report").lower().strip()
        title = str(args.get("title") or args.get("topic") or "")
        sections = args.get("sections")
        sections = sections if isinstance(sections, list) else None
        if action in {"brief", "docx", "docx_brief"}:
            return await self.exporter.compile_docx_brief(
                title, sections, str(args.get("summary") or args.get("body") or ""))
        if action in {"deck", "html_deck", "presentation", "slides"}:
            slides = args.get("slides")
            return await self.exporter.compile_presentation_deck(
                title, slides if isinstance(slides, list) else sections)
        if action in {"report", "export_report"}:
            return await self.exporter.export_report(
                title, str(args.get("body") or args.get("summary") or ""), sections)
        if action in {"history", "list"}:
            return self.exporter.get_export_history(int(args.get("limit") or 20))
        return ToolResult(
            f"Unsupported document_export action: {action}. Use brief, deck, "
            "report, or history.",
            ok=False,
        )

    async def draft_report_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.reports is None:
            return ToolResult("The report drafter is not available.", ok=False)
        charts = args.get("charts")
        sections = args.get("sections")
        return await self.reports.draft(
            topic=str(args.get("topic") or args.get("title") or ""),
            brief=str(args.get("brief") or args.get("notes") or ""),
            charts=charts if isinstance(charts, list) else None,
            sections=sections if isinstance(sections, list) else None,
        )

    async def proactive_report_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.reporting is None:
            return ToolResult("The reporting service is not available.", ok=False)
        kind = str(args.get("kind") or args.get("action") or "daily_business")
        return await self.reporting.generate(kind)

    async def literature_vault_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.literature is None:
            return ToolResult("The literature vault is not available.", ok=False)
        action = str(args.get("action") or "ingest_paper").lower().strip()
        if action in {"ingest_paper", "ingest", "read"}:
            return await self.literature.ingest_paper(
                path=str(args.get("path") or args.get("file") or ""),
                title=str(args.get("title") or ""),
            )
        if action in {"query_mechanisms", "query", "mechanisms"}:
            return await self.literature.query_mechanisms(
                str(args.get("query") or args.get("topic") or "")
            )
        if action in {"generate_citation_summary", "citations", "citation_summary"}:
            return await self.literature.generate_citation_summary(str(args.get("slug") or ""))
        return ToolResult(
            "Unsupported literature_vault action. Use ingest_paper, query_mechanisms, "
            "or generate_citation_summary.", ok=False,
        )

    async def ingest_tool(self, args: dict[str, Any]) -> ToolResult:
        """Unified file/folder ingestion into ORION's knowledge base.

        Actions: ``file`` (default), ``folder``, ``search``, ``library``,
        ``directives`` (the per-file directive manifest — what each ingested
        file is for). The engine is synchronous and SQLite-backed, so work runs
        in a thread to keep the event loop free.
        """
        if self.ingestion is None:
            return ToolResult("The ingestion engine is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        path = str(args.get("path") or args.get("file") or args.get("folder") or "").strip()
        query = str(args.get("query") or args.get("q") or "").strip()
        tags = args.get("tags") if isinstance(args.get("tags"), list) else None
        if not action:
            action = "search" if (query and not path) else "file"

        if action in {"search", "find"}:
            hits = await asyncio.to_thread(self.ingestion.search, query, 8)
            if not hits:
                return ToolResult(f"No ingested knowledge matched '{query}'.")
            lines = [f"• {Path(h['path']).name}: {h['snippet']}" for h in hits]
            return ToolResult(f"Top matches for '{query}':\n" + "\n".join(lines))

        if action in {"directives", "manifest", "catalog", "remembered"}:
            try:
                limit = max(1, int(args.get("count") or args.get("limit") or 30))
            except (TypeError, ValueError):
                limit = 30
            catalog = await asyncio.to_thread(self.ingestion.catalog, limit)
            if not catalog:
                return ToolResult("I have not ingested any files yet.")
            lines = [f"• {c['filename']} [{c['category']}] — {c['directive']}"
                     for c in catalog]
            return ToolResult(
                f"I've ingested {len(catalog)} file(s); here is what each is for:\n"
                + "\n".join(lines))

        if action in {"library", "list", "stats"}:
            stats = await asyncio.to_thread(self.ingestion.stats)
            return ToolResult(
                f"Knowledge library: {stats['documents']} documents, "
                f"{stats['chunks']} chunks ({stats['unique_chunks']} unique). "
                "Ask for 'ingest directives' to hear what each file is for.")

        if not path:
            return ToolResult("Provide a file or folder path to ingest.", ok=False)

        if action in {"folder", "dir", "directory"}:
            batch = await asyncio.to_thread(
                lambda: self.ingestion.ingest_folder(path, recursive=True, tags=tags))
            return ToolResult(f"Ingested {Path(path).name}: {batch.summary()}")

        result = await asyncio.to_thread(self.ingestion.ingest_file, path, tags)
        if not result.ok:
            return ToolResult(f"Could not ingest {Path(path).name}: {result.note}", ok=False)
        entities = (" Linked entities: " + ", ".join(result.entities[:6]) + ".") \
            if result.entities else ""
        directive = f" Directive: {result.directive}" if result.directive else ""
        return ToolResult(
            f"Ingested {Path(path).name} [{result.category}] — {result.chunks} chunk(s), "
            f"v{result.version}, tags: {', '.join(result.tags[:6])}.{directive}{entities}")

    def forget_engine(self) -> Any:
        """The cross-store forget engine over ORION's live stores (forget.py)."""
        from .forget import ForgetEngine
        log = getattr(getattr(self, "bus", None), "log", None)
        return ForgetEngine(
            memory=getattr(self, "memory", None), companion=getattr(self, "companion", None),
            graph=getattr(self, "graph", None), cognition=getattr(self, "cognition", None),
            goals=getattr(self, "goal_tracker", None),
            log=(log.emit if log is not None else None))

    async def request_forget(self, topic: str) -> ToolResult:
        """Survey what ORION holds about *topic* and raise the on-screen
        confirmation. Nothing is removed here: only the owner's approval of
        that prompt (confirm_system_action -> run_confirmed_forget) removes
        exactly the records listed, so a mishearing cannot erase anything."""
        if not str(topic or "").strip():
            return ToolResult("What should I forget? Name the topic or the item.", ok=False)
        from .execution_policy import require_foreground
        blocked = require_foreground("forget_confirmation_dialog")
        if blocked is not None:
            return blocked
        engine = self.forget_engine()
        plan = await asyncio.to_thread(engine.survey, topic)
        if not plan.patterns:
            return ToolResult(f"'{topic}' is too vague to forget safely; name it more exactly.", ok=False)
        if not plan.matches:
            return ToolResult(plan.summary())
        guard = getattr(self, "system_guard", None)
        if guard is None:
            return ToolResult("Forgetting needs the on-screen confirmation, which is not "
                              "available right now; nothing was removed.", ok=False)
        from .system_guard import ActionIntent
        decision = guard.request_confirmation(
            ActionIntent.MEMORY_FORGET, origin="forget",
            payload={"topic": topic, "approved": plan.identities(), "summary": plan.summary()})
        token = getattr(decision, "token", "") or ""
        if not token:
            return ToolResult("The confirmation could not be raised; nothing was removed.", ok=False)
        self.bus.confirm_action.emit({
            "token": token, "intent": "memory_forget",
            "message": f"Forget everything about {plan.label}? {plan.summary()}",
            "items": [m.preview for m in plan.matches[:20]],
        })
        return ToolResult(
            f"{plan.summary()} Approve the on-screen confirmation to forget it; nothing has "
            "been removed yet.", ok=False,
            receipt={"pending_confirmation": True, "records": len(plan.matches),
                     "stores": plan.counts()})

    def run_confirmed_forget(self, payload: dict[str, Any]) -> ToolResult:
        """Released ONLY by the owner's approval (the guard token). Re-surveys the
        approved topic and removes the approved records - never more."""
        topic = str((payload or {}).get("topic") or "")
        approved = [tuple(str(x) for x in ident) for ident in (payload or {}).get("approved") or []]
        if not topic or not approved:
            return ToolResult("The approved forget request lost its details; nothing was removed.", ok=False)
        engine = self.forget_engine()

        def _work() -> ToolResult:
            plan = engine.survey(topic)
            receipt = engine.execute(plan, approved)
            removed = sum(receipt["removed"].values())
            redacted = sum(receipt["redacted"].values())
            where = ", ".join(f"{name} ({n})" for name, n in
                              {**receipt["removed"], **{f"{k} redacted": v
                                                        for k, v in receipt["redacted"].items()}}.items())
            text = (f"Forgotten: {removed} record(s) removed and {redacted} redacted"
                    + (f" - {where}" if where else "") + ". No copy was kept.")
            if receipt["problems"]:
                text += " Could not finish: " + "; ".join(receipt["problems"]) + "."
            return ToolResult(text, ok=not receipt["problems"], receipt=receipt)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _work()

        async def _release() -> None:
            try:
                result = await asyncio.to_thread(_work)
                self.bus.log.emit(f"SYS: {result.text}")
                self.bus.dashboard_event.emit("companion", {"forgotten": True})
            except Exception as exc:
                self.bus.log.emit(f"SYS: forgetting failed - {exc}")

        from . import background
        if background.spawn(_release(), name="orion-forget") is None:
            return _work()
        return ToolResult("Forgetting it now - I will confirm what was removed.")

    def troubleshooting_guide_tool(self, args: dict[str, Any]) -> ToolResult:
        """ORION's own manual (self_manual.py): how he works and how to fix it."""
        from . import self_manual
        question = str(args.get("question") or args.get("topic") or "").strip()
        if not question:
            return ToolResult(self_manual.contents())
        found = self_manual.lookup(question)
        if not found:
            return ToolResult(f"The manual has nothing on that. {self_manual.contents()}", ok=False)
        return ToolResult("\n\n".join(f"{title}: {answer}" for title, answer in found))

    def away_mode_tool(self, args: dict[str, Any]) -> ToolResult:
        """Keep the PC awake for the phone while the owner is out (away_mode.py)."""
        from .away_mode import AwayMode
        away = getattr(self, "_away", None)
        if away is None:
            log = getattr(getattr(self, "bus", None), "log", None)
            away = self._away = AwayMode(log.emit if log is not None else None)
        action = str(args.get("action") or "status").strip().lower()
        if action in ("on", "start", "enable"):
            return ToolResult(away.start())
        if action in ("off", "stop", "disable"):
            return ToolResult(away.stop())
        return ToolResult(away.status())

    async def erase_my_data_tool(self, args: dict[str, Any]) -> ToolResult:
        """Erase what ORION keeps about the owner (erase_all.py): survey, then
        the on-screen confirmation, then the erase at the next start."""
        from . import erase_all
        from .execution_policy import require_foreground
        action = str(args.get("action") or "survey").strip().lower()
        scope = str(args.get("scope") or "personal").strip().lower()
        try:
            if action == "status":
                waiting = erase_all.pending()
                return ToolResult(f"An erase ({waiting}) is waiting for the next start."
                                  if waiting else "No erase is waiting.")
            if action == "cancel":
                return ToolResult("The waiting erase is cancelled." if erase_all.cancel()
                                  else "No erase was waiting.")
            plan = await asyncio.to_thread(erase_all.survey, scope)
            if action == "survey":
                return ToolResult(plan.summary(), receipt={"items": len(plan.paths),
                                                           "bytes": plan.total_bytes, "scope": plan.scope})
            if action != "erase":
                return ToolResult(f"Unknown action '{action}' (survey, erase, cancel or status).", ok=False)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        blocked = require_foreground("erase_confirmation_dialog")
        if blocked is not None:
            return blocked
        guard = getattr(self, "system_guard", None)
        if guard is None:
            return ToolResult("Erasing needs the on-screen confirmation, which is not available "
                              "right now; nothing was changed.", ok=False)
        from .system_guard import ActionIntent
        decision = guard.request_confirmation(ActionIntent.DATA_ERASE, origin="erase",
                                              payload={"scope": plan.scope})
        token = getattr(decision, "token", "") or ""
        if not token:
            return ToolResult("The confirmation could not be raised; nothing was changed.", ok=False)
        self.bus.confirm_action.emit({
            "token": token, "intent": "data_erase",
            "message": f"Erase your data? {plan.summary()}",
            "items": [p.name for p in plan.paths[:20]],
        })
        return ToolResult(f"{plan.summary()} Approve the on-screen confirmation to erase it; nothing "
                          "has changed yet.", ok=False, receipt={"pending_confirmation": True})

    def run_confirmed_erase(self, payload: dict[str, Any]) -> ToolResult:
        """Released ONLY by the owner's approval: arms the erase and restarts,
        so it runs before any store is opened."""
        from . import erase_all
        scope = str((payload or {}).get("scope") or "")
        if scope not in erase_all.SCOPES:
            return ToolResult("The approved erase lost its scope; nothing was changed.", ok=False)
        plan = erase_all.schedule(scope)
        self.bus.log.emit(f"PRIVACY: erase approved ({scope}, {len(plan.paths)} items); restarting to run it.")
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1500, self.bus.request_restart.emit)
        except Exception:
            pass
        return ToolResult(f"Erasing {len(plan.paths)} item(s) to the Recycle Bin; restarting now to finish.")

    async def companion_tool(self, args: dict[str, Any]) -> ToolResult:
        """Companion continuity: brief, goals, habits, achievements, activity log."""
        if self.companion is None:
            return ToolResult("The companion engine is not available.", ok=False)
        action = str(args.get("action") or "brief").lower().strip()
        title = str(args.get("title") or args.get("name") or args.get("subject") or "").strip()

        if action in {"brief", "continuity", "catchup", "catch_up"}:
            brief = await asyncio.to_thread(self.companion.continuity_brief)
            if brief:
                return ToolResult(brief)
            # Not "nothing tracked" while wins are on record (25 Oct 2026:
            # the panel said so above three starred achievements).
            wins = await asyncio.to_thread(self.companion.achievements, 3)
            if wins:
                return ToolResult("No recent activity is tracked. Recorded wins: "
                                  + "; ".join(str(w.get("title") or "") for w in wins) + ".")
            return ToolResult("Nothing tracked yet — I will keep notes as you work.")

        if action in {"forget", "erase", "forget_topic", "remove", "delete", "forget_item"}:
            topic = str(args.get("topic") or args.get("query") or title or "").strip()
            return await self.request_forget(topic)

        if action in {"log", "activity", "log_activity"}:
            if not title:
                return ToolResult("Tell me what you worked on.", ok=False)
            kind = str(args.get("kind") or "worked_on")
            rec = await asyncio.to_thread(
                self.companion.log_activity, kind, title, str(args.get("detail") or ""))
            return ToolResult(f"Noted — {kind.replace('_', ' ')} {rec.get('subject', title)}.")

        if action in {"goal", "set_goal", "goal_progress"}:
            if not title:
                return ToolResult("Name the goal.", ok=False)
            progress = args.get("progress")
            goal = await asyncio.to_thread(
                self.companion.upsert_goal, title, str(args.get("target") or ""),
                int(progress) if progress is not None else None)
            if goal.status == "achieved":
                return ToolResult(f"Goal '{goal.title}' achieved — well done.")
            return ToolResult(f"Goal '{goal.title}' tracked at {goal.progress}%.")

        if action in {"goals", "list_goals"}:
            goals = await asyncio.to_thread(self.companion.goals)
            if not goals:
                return ToolResult("No open goals tracked.")
            lines = [f"• {g.title} — {g.progress}%" + (f" (target {g.target})" if g.target else "")
                     for g in goals[:8]]
            return ToolResult("Open goals:\n" + "\n".join(lines))

        if action in {"habit", "mark_habit"}:
            if not title:
                return ToolResult("Name the habit.", ok=False)
            rec = await asyncio.to_thread(self.companion.mark_habit, title)
            return ToolResult(f"Habit '{rec['name']}' marked — {rec['streak']}-day streak.")

        if action in {"habits", "list_habits"}:
            habits = await asyncio.to_thread(self.companion.habits)
            if not habits:
                return ToolResult("No habits tracked.")
            lines = [f"• {h['name']} — {h['streak']}-day streak ({h['marks']} total)"
                     for h in habits[:8]]
            return ToolResult("Habits:\n" + "\n".join(lines))

        if action in {"achievement", "record_achievement"}:
            if not title:
                return ToolResult("Name the achievement.", ok=False)
            await asyncio.to_thread(
                self.companion.record_achievement, title, str(args.get("detail") or ""))
            return ToolResult(f"Achievement recorded: {title}.")

        if action in {"achievements", "list_achievements"}:
            records = await asyncio.to_thread(self.companion.achievements)
            if not records:
                return ToolResult("No achievements recorded yet.")
            return ToolResult("Achievements:\n" + "\n".join(f"• {a['title']}" for a in records[:8]))

        if action in {"status", "stats"}:
            s = await asyncio.to_thread(self.companion.status)
            return ToolResult(
                f"Companion ledger: {s['activities']} activities, {s['open_goals']} open goal(s), "
                f"{s['habits']} habit(s), {s['achievements']} achievement(s).")

        return ToolResult(
            "Unsupported companion action. Use brief, log, goal, goals, habit, habits, "
            "achievement, achievements, or status.", ok=False)
