"""
Tool dispatcher — routes model function-calls (and manual GUI commands) to
the owning service, and declares the Gemini function-calling schema.

Mark VIII layout: the dispatcher no longer *implements* desktop control,
vision, mail or workspace logic itself — it validates the payload through the
security layer and delegates:

    open_app / close_app / window_control / media_control  → DesktopAgent
    vision_analyse / capture_screen                        → VisionAgent
    outlook_mail                                           → OutlookService
    notion_workspace                                       → NotionService
    agent_dispatch                                         → AgentManager
    morning_briefing                                       → live worker hook
    save_memory / query_intelligence / recall_conversation → MemoryAgent

File, process, browser and dev-workbench primitives remain local to the
dispatcher because they are thin, stateless wrappers over the standard
library.  Tool chaining (`dispatch_chain`) is unchanged from Mark VII.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .agents import AgentManager, DesktopAgent
from .briefing import MorningBriefingService
from .bus import OrionBus
from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .memory import MemoryAgent
from .notion import NotionService
from .outlook import OutlookService
from .security import SecuritySanitiser, SecurityViolation
from .vision import LocalFileIntelligence, VisionAgent, VolatileScreenGrabber


class OrionDispatcher:
    def __init__(
        self,
        bus: OrionBus,
        memory: MemoryAgent,
        grabber: VolatileScreenGrabber,
        file_intel: LocalFileIntelligence,
        desktop: DesktopAgent,
        vision: VisionAgent,
        outlook: OutlookService,
        notion: NotionService,
        agent_manager: AgentManager,
        briefing: MorningBriefingService,
        control: Any | None = None,
        verifier: Any | None = None,
        web: Any | None = None,
        workspace: Any | None = None,
        copilot: Any | None = None,
        selfrepair: Any | None = None,
        proactive: Any | None = None,
        display: Any | None = None,
        telemetry: Any | None = None,
        knowledge: Any | None = None,
        packs: Any | None = None,
        conversation: Any | None = None,
        commerce: Any | None = None,
        community: Any | None = None,
        hub: Any | None = None,
        ai: Any | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.grabber = grabber
        self.file_intel = file_intel
        self.desktop = desktop
        self.vision = vision
        self.outlook = outlook
        self.notion = notion
        self.agent_manager = agent_manager
        self.briefing = briefing
        # Mark IX managers (optional so Mark VIII construction still works).
        self.control = control
        self.verifier = verifier
        self.web = web
        self.workspace = workspace
        self.copilot = copilot
        self.selfrepair = selfrepair
        self.proactive = proactive
        self.display = display
        self.telemetry = telemetry
        self.knowledge = knowledge
        # Dual-mode / entrepreneurial managers (optional).
        self.packs = packs                # KnowledgePackManager
        self.conversation = conversation  # ConversationMemoryEngine
        self.commerce = commerce          # CommerceSuite (agents + advisor)
        self.community = community         # CommunityHub
        self.hub = hub                     # EcommerceHub
        self.ai = ai                       # AIModeInfo (router + connectivity + ollama)
        # Mark X managers (optional).
        self.research = None               # ResearchAgent (set post-construction)
        self.corpus = None                 # KnowledgeCorpusBuilder
        # JARVIS layer (optional, set post-construction).
        self.protocols = None              # ProtocolManager
        self.reminders = None              # ReminderService
        self.sentinel = None               # SentinelAgent
        # Studio deck services (optional, set post-construction).
        self.audio_studio = None           # AudioStudioService
        self.literature = None             # LiteratureIntakeService
        self.pipeline = None               # AgencyPipelineService
        # Realistic-improvements batch (optional, set post-construction).
        self.plan_executor = None          # PlanExecutor
        self.file_organiser = None         # FileOrganiser
        self.security = None               # SecuritySentinel
        self.backup = None                 # BackupManager
        self.reports = None                # ReportDrafter
        self.diagnostics = None            # DiagnosticsEngine
        self.cursor_overlay = None         # CursorOverlay
        # Mark IX Living-Memory batch (optional, set post-construction).
        self.changelog = None              # Changelog (patch notes)
        self.learning = None               # LearningService
        self.programming = None            # ProgrammingKnowledgeBase
        self.cyber = None                  # CyberKnowledgeBase
        # Mark X.5 — AI Operating System layer (optional, set post-construction).
        self.exporter = None               # DocumentExporterService
        self.reporting = None              # ProactiveReportingService
        self.cognition = None              # CognitiveStateManager
        self.cognitive_loop = None         # CognitiveLoopManager
        self.graph = None                  # KnowledgeGraphEngine (second brain)
        self.executive = None              # ExecutiveAssistantMode
        self.momentum = None               # MomentumEngine (shipping coach)
        self.emotion = None                # EmotionStateManager (Mark X.7)
        self.geo = None                    # GeoIntelligenceEngine (Phase 8)
        # JARVIS subsystems (optional, set post-construction).
        self.web_automation = None         # WebAutomationService
        self.peripherals = None            # PeripheralController
        self.messaging = None              # MessagingGateway
        self.gaming = None                 # GamingClientService
        self.entertainment = None          # EntertainmentService
        # Mark X.7+: Forge self-improvement engine (optional, set post-construction).
        self.forge = None                  # ForgeOrchestrationManager (capability forging)
        # Rolling record of recent tool executions for the Command Centre.
        self.recent_tools: deque[dict[str, Any]] = deque(maxlen=60)
        self.active_tools = 0
        # Set by the live worker: coroutine factory that (re)delivers the
        # briefing through the active voice channel.
        self.on_briefing_request: Callable[[], Awaitable[None]] | None = None

    @property
    def news_articles(self) -> list[dict[str, str]]:
        """Cached briefing stories (owned by the briefing service)."""
        return self.briefing.articles

    # ── chaining ──────────────────────────────────────────────────────────────

    async def dispatch_chain(
        self, name: str, args: dict[str, Any] | None, max_depth: int = 4
    ) -> ToolResult:
        pending: list[tuple[str, dict[str, Any]]] = [(name, dict(args or {}))]
        transcript: list[str]       = []
        aggregate_ok                = True
        media: dict[str, Any] | None = None
        seen: set[str]              = set()
        depth                       = 0
        while pending and depth < max(1, min(8, max_depth)):
            current_name, current_args = pending.pop(0)
            signature = json.dumps([current_name, current_args], sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            result = await self.dispatch(current_name, current_args)
            aggregate_ok = aggregate_ok and result.ok
            if media is None and result.media is not None:
                media = result.media
            transcript.append(f"[{current_name}] {result.text}")
            for chained_name, chained_args in self._derive_chain(current_name, current_args, result):
                pending.append((chained_name, chained_args))
            if result.chain:
                pending.extend(result.chain)
            depth += 1
        return ToolResult("\n\n".join(transcript), ok=aggregate_ok, media=media)

    async def dispatch(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        name = SecuritySanitiser.guard_text(str(name or ""), "tool.name")
        args = SecuritySanitiser.guard_payload(dict(args or {}), f"tool.{name}")
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "open_app":           self.open_app,
            "close_app":          self.close_app,
            "web_search":         self.web_search,
            "open_news":          self.open_news,
            "browser_control":    self.browser_control,
            "window_control":     self.window_control,
            "media_control":      self.media_control,
            "find_files":         self.find_files,
            "dev_workbench":      self.dev_workbench,
            "file_controller":    self.file_controller,
            "process_file":       self.process_file,
            "image_processor":    self.process_file,
            "save_memory":        self.save_memory,
            "query_intelligence": self.query_intelligence,
            "recall_conversation": self.recall_conversation,
            "execute_plan":       self.execute_plan,
            "capture_screen":     self.capture_screen,
            "vision_analyse":     self.vision_analyse,
            "outlook_mail":       self.outlook_mail,
            "notion_workspace":   self.notion_workspace,
            "agent_dispatch":     self.agent_dispatch,
            "morning_briefing":   self.morning_briefing,
            "clipboard_operate":  self.clipboard_operate,
            "process_governor":   self.process_governor,
            "system_notify":      self.system_notify,
            "shutdown_orion":     self.shutdown_orion,
            # ── Mark IX ────────────────────────────────────────────────────
            "desktop_control":    self.desktop_control,
            "vision_verify":      self.vision_verify,
            "web_control":        self.web_control,
            "workspace_control":  self.workspace_control,
            "codebase_copilot":   self.codebase_copilot,
            "self_repair":        self.self_repair,
            "proactive_check":    self.proactive_check,
            "display_info":       self.display_info,
            "neuro_knowledge":    self.neuro_knowledge,
            # ── Dual-mode / entrepreneurial (Mark IX+) ─────────────────────
            "ai_mode":            self.ai_mode,
            "knowledge_pack":     self.knowledge_pack,
            "conversation_recall": self.conversation_recall,
            "product_research":   self.product_research,
            "tiktok_intel":       self.tiktok_intel,
            "instagram_intel":    self.instagram_intel,
            "founder_knowledge":  self.founder_knowledge,
            "business_advisor":   self.business_advisor,
            "commerce_hub":       self.commerce_hub,
            "community_share":    self.community_share,
            # ── Mark X ─────────────────────────────────────────────────────
            "research":           self.research_tool,
            "globe":              self.globe_tool,
            "expand_mind":        self.expand_mind,
            # ── JARVIS layer ───────────────────────────────────────────────
            "protocol":           self.protocol_tool,
            "reminder":           self.reminder_tool,
            "sentinel":           self.sentinel_tool,
            # ── Studio deck ────────────────────────────────────────────────
            "audio_studio":       self.audio_studio_tool,
            "literature_vault":   self.literature_vault_tool,
            "campaign_pipeline":  self.campaign_pipeline_tool,
            # ── Realistic-improvements batch ──────────────────────────────
            "autoplan":           self.autoplan_tool,
            "organise_files":     self.organise_files_tool,
            "security_watch":     self.security_watch_tool,
            "backup":             self.backup_tool,
            "draft_report":       self.draft_report_tool,
            "diagnostics":        self.diagnostics_tool,
            "cursor_overlay":     self.cursor_overlay_tool,
            # ── Living Memory batch ────────────────────────────────────────
            "patch_notes":        self.patch_notes_tool,
            "learn":              self.learn_tool,
            "transcript":         self.transcript_tool,
            "programming_knowledge": self.programming_knowledge_tool,
            "cyber_knowledge":    self.cyber_knowledge_tool,
            # ── Mark X.5: AI Operating System layer ────────────────────────
            "document_export":    self.document_export_tool,
            "proactive_report":   self.proactive_report_tool,
            "awareness":          self.awareness_tool,
            "second_brain":       self.second_brain_tool,
            "executive":          self.executive_tool,
            "momentum":           self.momentum_tool,
            "competitor_intel":   self.competitor_intel_tool,
            "brand_growth":       self.brand_growth_tool,
            "emotion":            self.emotion_tool,
            "geo":                self.geo_tool,
            "audio_devices":      self.audio_devices_tool,
            # ── JARVIS subsystems ──────────────────────────────────────────
            "web_automation":     self.web_automation_tool,
            "peripherals":        self.peripherals_tool,
            "messaging":          self.messaging_tool,
            "gaming":             self.gaming_tool,
            "entertainment":      self.entertainment_tool,
            # ── Mark X.7+: Forge self-improvement engine ─────────────────────
            "forge":              self.forge_tool,
        }
        handler = handlers.get(name)
        if handler is None:
            return ToolResult(f"Unknown dispatch target: {name}.", ok=False)
        started = time.perf_counter()
        self.active_tools += 1
        ok = True
        try:
            result = handler(args)
            if asyncio.iscoroutine(result):
                result = await result
            ok = result.ok
            return result
        except SecurityViolation:
            ok = False
            raise
        except Exception as exc:
            ok = False
            return ToolResult(f"{name} failed: {exc}", ok=False)
        finally:
            self.active_tools = max(0, self.active_tools - 1)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.recent_tools.append({
                "tool": name, "ok": ok, "ms": round(elapsed_ms, 1),
                "at": time.strftime("%H:%M:%S"),
            })
            if self.telemetry is not None:
                self.telemetry.metrics.observe(f"tool.{name}.ms", elapsed_ms)
                self.telemetry.metrics.incr("tool.calls")
                if not ok:
                    self.telemetry.metrics.incr("tool.failures")

    # ── desktop delegation ────────────────────────────────────────────────────

    def open_app(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.open_app(str(args.get("app_name") or args.get("name") or ""))

    def close_app(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.close_app(str(args.get("app_name") or args.get("name") or ""))

    def window_control(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.window_control(
            str(args.get("action") or "list"), str(args.get("title") or "")
        )

    def media_control(self, args: dict[str, Any]) -> ToolResult:
        return self.desktop.media_control(
            str(args.get("action") or "play_pause"), int(args.get("steps") or 2)
        )

    # ── web / news ────────────────────────────────────────────────────────────

    def web_search(self, args: dict[str, Any]) -> ToolResult:
        query = SecuritySanitiser.guard_text(str(args.get("query") or ""), "web_search.query")
        if not query:
            return ToolResult("No search query supplied.", ok=False)
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        webbrowser.open(url)
        return ToolResult(f"Secure web search opened for: {query}.")

    def open_news(self, args: dict[str, Any]) -> ToolResult:
        if not self.news_articles:
            return ToolResult("No briefing stories are cached yet.", ok=False)
        query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("topic") or args.get("title") or ""),
            "open_news.query",
        ).strip().lower()
        chosen: list[dict[str, str]] = []
        index = args.get("index")
        if index is not None:
            try:
                position = int(index) - 1
                if 0 <= position < len(self.news_articles):
                    chosen = [self.news_articles[position]]
            except (TypeError, ValueError):
                pass
        if not chosen and query:
            chosen = [
                article for article in self.news_articles
                if query in article["title"].lower() or query in article["topic"].lower()
            ]
        if not chosen:
            return ToolResult(
                "No cached briefing story matches that request. Cached stories:\n"
                + "\n".join(
                    f"{i}. ({a['topic']}) {a['title']}"
                    for i, a in enumerate(self.news_articles, 1)
                ),
                ok=False,
            )
        opened: list[str] = []
        for article in chosen[:3]:
            if article.get("url"):
                webbrowser.open(article["url"])
                opened.append(article["title"])
        if not opened:
            return ToolResult("The matched stories carry no usable link.", ok=False)
        return ToolResult("Opened in browser: " + "; ".join(opened))

    def browser_control(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "go_to").lower().strip()
        if action in {"go_to", "open", "new_tab"}:
            url = str(args.get("url") or args.get("target") or "").strip()
            if not url:
                return ToolResult("No URL supplied.", ok=False)
            url = self._normalise_url(SecuritySanitiser.guard_text(url, "browser_control.url"))
            webbrowser.open(url)
            return ToolResult(f"Browser opened: {url}.")
        if action in {"search", "web_search"}:
            return self.web_search({"query": args.get("query") or args.get("text") or ""})
        return ToolResult(f"Browser action '{action}' is not supported by the native dispatcher.", ok=False)

    # ── vision ────────────────────────────────────────────────────────────────

    async def vision_analyse(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "describe").lower().strip()
        prompt = str(args.get("prompt") or args.get("question") or "")
        path   = str(args.get("path") or "").strip()
        if action in {"describe", "screen", "analyse_screen", "analyze_screen", "look"}:
            return await self.vision.analyse_screen(prompt)
        if action in {"ocr", "read_text", "extract_text"}:
            return await self.vision.ocr(path)
        if action in {"find_errors", "detect_errors", "errors", "diagnose"}:
            return await self.vision.detect_errors()
        if action in {"analyse_image", "analyze_image", "image"}:
            if not path:
                return ToolResult("An image path is required for analyse_image.", ok=False)
            return await self.vision.analyse_image(path, prompt)
        return ToolResult(
            f"Unsupported vision action: {action}. "
            "Use describe, ocr, find_errors, or analyse_image.",
            ok=False,
        )

    async def capture_screen(self, args: dict[str, Any]) -> ToolResult:
        quality      = int(args.get("quality") or 78)
        max_side     = int(args.get("max_side") or 1024)
        # mss grab + PIL encode block for tens of milliseconds; keep the GUI
        # event loop clear by offloading to a worker thread.
        image_bytes  = await asyncio.to_thread(
            self.grabber.capture_jpeg, max_side=max_side, quality=quality
        )
        return ToolResult(
            f"Captured primary monitor in volatile memory: {len(image_bytes)} JPEG bytes.",
            media={"data": image_bytes, "mime_type": "image/jpeg"},
        )

    # ── outlook ───────────────────────────────────────────────────────────────

    async def outlook_mail(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "read_inbox").lower().strip()
        limit = int(args.get("limit") or 10)
        # Tool calls are direct user requests, so Outlook may be started here
        # (launch=True); passive surveys elsewhere remain attach-only.
        if action in {"read_inbox", "read", "inbox", "check"}:
            return await self.outlook.read_inbox(
                limit=limit, unread_only=bool(args.get("unread_only")), launch=True
            )
        if action in {"priority", "priority_emails", "important"}:
            return await self.outlook.priority_emails(limit=limit, launch=True)
        if action in {"read_email", "open_email", "full_body"}:
            entry_id = str(args.get("entry_id") or "").strip()
            if not entry_id:
                return ToolResult("An entry_id from a previous inbox read is required.", ok=False)
            return await self.outlook.read_email_body(entry_id)
        if action in {"draft", "create_draft", "compose"}:
            return await self.outlook.create_draft(
                to=str(args.get("to") or ""),
                subject=str(args.get("subject") or ""),
                body=str(args.get("body") or ""),
                cc=str(args.get("cc") or ""),
            )
        if action in {"send_draft", "send"}:
            return await self.outlook.send_draft(
                draft_ref=str(args.get("draft_ref") or args.get("draft") or ""),
                confirm=bool(args.get("confirm")),
            )
        if action in {"discard_draft", "discard"}:
            return await self.outlook.discard_draft(
                str(args.get("draft_ref") or args.get("draft") or "")
            )
        if action in {"pending_drafts", "drafts"}:
            drafts = self.outlook.pending_drafts()
            if not drafts:
                return ToolResult("No drafts are awaiting approval.")
            return ToolResult(
                "Drafts awaiting approval:\n" + "\n".join(
                    f"- {d['ref']}: to {d['to']}, subject '{d['subject']}'" for d in drafts
                )
            )
        return ToolResult(
            f"Unsupported outlook action: {action}. Use read_inbox, priority, "
            "read_email, draft, send_draft, discard_draft, or pending_drafts.",
            ok=False,
        )

    # ── notion ────────────────────────────────────────────────────────────────

    async def notion_workspace(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list_tasks").lower().strip()
        if action in {"list_tasks", "tasks", "todo"}:
            return await self.notion.list_tasks(
                limit=int(args.get("limit") or 12),
                include_done=bool(args.get("include_done")),
            )
        if action in {"create_task", "add_task", "new_task"}:
            return await self.notion.create_task(
                title=str(args.get("title") or args.get("task") or ""),
                due=str(args.get("due") or args.get("date") or ""),
                notes=str(args.get("notes") or ""),
            )
        if action in {"complete_task", "finish_task", "done"}:
            return await self.notion.complete_task(
                str(args.get("query") or args.get("title") or "")
            )
        if action in {"upcoming_events", "calendar", "schedule", "agenda"}:
            return await self.notion.upcoming_events(
                days=int(args.get("days") or 7), limit=int(args.get("limit") or 10)
            )
        if action in {"create_event", "schedule_event", "book"}:
            return await self.notion.create_event(
                title=str(args.get("title") or ""),
                start=str(args.get("start") or args.get("date") or ""),
                end=str(args.get("end") or ""),
            )
        if action in {"projects", "project_overview"}:
            return await self.notion.project_overview(limit=int(args.get("limit") or 10))
        return ToolResult(
            f"Unsupported notion action: {action}. Use list_tasks, create_task, "
            "complete_task, upcoming_events, create_event, or projects.",
            ok=False,
        )

    # ── specialist agents / briefing ──────────────────────────────────────────

    async def agent_dispatch(self, args: dict[str, Any]) -> ToolResult:
        return await self.agent_manager.dispatch(
            request=str(args.get("request") or args.get("query") or ""),
            agent_name=str(args.get("agent") or "auto"),
            context=str(args.get("context") or ""),
        )

    async def morning_briefing(self, args: dict[str, Any]) -> ToolResult:
        if self.on_briefing_request is not None:
            asyncio.create_task(self.on_briefing_request())
            return ToolResult("Briefing underway — composing the intelligence picture now.")
        briefing = await self.briefing.compose_source_material()
        return ToolResult(briefing)

    # ── Mark IX: autonomous desktop control ───────────────────────────────────

    async def desktop_control(self, args: dict[str, Any]) -> ToolResult:
        """
        Cursor/keyboard/window control with optional visual verification.
        Every mutating action runs off the event loop (to_thread) and, when
        'verify' is set (default for clicks), is confirmed by vision.
        """
        if self.control is None:
            return ToolResult("Autonomous control layer is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        verify = bool(args.get("verify", True)) and self.verifier is not None

        def _run(fn: Callable[..., ToolResult], *a: Any, **k: Any) -> Any:
            return asyncio.to_thread(fn, *a, **k)

        # Element-targeted click with self-correcting coordinates.
        if action in {"click_text", "click_element"}:
            if self.verifier is None:
                return ToolResult("Verification engine unavailable for element clicks.", ok=False)
            return await self.verifier.click_text(str(args.get("text") or args.get("query") or ""))
        if action == "move_cursor":
            return await _run(self.control.move_cursor, int(args.get("x", 0)), int(args.get("y", 0)),
                              args.get("monitor"))
        if action in {"click", "left_click"}:
            call = lambda: self.control.click(args.get("x"), args.get("y"), monitor=args.get("monitor"))
            return await self._maybe_verify(call, verify)
        if action == "double_click":
            call = lambda: self.control.double_click(args.get("x"), args.get("y"), monitor=args.get("monitor"))
            return await self._maybe_verify(call, verify)
        if action == "right_click":
            return await _run(self.control.right_click, args.get("x"), args.get("y"), args.get("monitor"))
        if action == "drag":
            return await _run(self.control.drag_cursor, int(args.get("x1", 0)), int(args.get("y1", 0)),
                              int(args.get("x2", 0)), int(args.get("y2", 0)), args.get("monitor"))
        if action == "scroll":
            return await _run(self.control.scroll, int(args.get("amount", 0)),
                              args.get("x"), args.get("y"), args.get("monitor"))
        if action in {"smooth_scroll", "scroll_smooth", "glide_scroll"}:
            return await _run(self.control.smooth_scroll, int(args.get("amount", -10)),
                              args.get("x"), args.get("y"), args.get("monitor"),
                              float(args.get("duration", 0.8)))
        if action == "type_text":
            return await _run(self.control.type_text, str(args.get("text") or ""))
        if action in {"edit_text", "write_text", "write_to", "set_text"}:
            return await _run(self.control.edit_text, str(args.get("text") or ""),
                              str(args.get("title") or args.get("window") or ""),
                              bool(args.get("replace")))
        if action in {"hotkey", "send_hotkeys"}:
            return await _run(self.control.send_hotkeys, args.get("keys") or args.get("hotkey") or "")
        if action in {"open_app", "open_application"}:
            return await _run(self.control.open_application, str(args.get("app_name") or args.get("name") or ""))
        if action in {"close_app", "close_application"}:
            return await _run(self.control.close_application, str(args.get("app_name") or args.get("name") or ""))
        if action in {"focus_window", "focus", "switch"}:
            return await _run(self.control.focus_window, str(args.get("title") or ""))
        if action in {"resize_window", "resize"}:
            return await _run(self.control.resize_window, str(args.get("title") or ""),
                              int(args.get("width", 800)), int(args.get("height", 600)))
        if action in {"move_window"}:
            return await _run(self.control.move_window, str(args.get("title") or ""),
                              int(args.get("x", 0)), int(args.get("y", 0)), args.get("monitor"))
        if action in {"minimise_window", "minimize_window"}:
            return await _run(self.control.minimise_window, str(args.get("title") or ""))
        if action in {"maximise_window", "maximize_window"}:
            return await _run(self.control.maximise_window, str(args.get("title") or ""))
        if action in {"list_windows", "windows"}:
            return await _run(self.control.list_windows)
        return ToolResult(
            f"Unsupported desktop_control action: {action}. Use click, click_text, "
            "move_cursor, double_click, right_click, drag, scroll, type_text, "
            "edit_text (reliable write into Notepad/editors with a 'title'), hotkey, "
            "open_app, close_app, focus_window, resize_window, move_window, "
            "minimise_window, maximise_window, or list_windows.",
            ok=False,
        )

    async def _maybe_verify(self, call: Callable[[], ToolResult], verify: bool) -> ToolResult:
        if verify and self.verifier is not None:
            return (await self.verifier.verify_action(call)).to_tool_result()
        return await asyncio.to_thread(call)

    # ── Mark IX: vision-guided detection (extends vision_analyse) ─────────────

    async def vision_verify(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "elements").lower().strip()
        if action in {"elements", "ui"}:
            return await self.vision.detect_elements(str(args.get("kinds") or "all"))
        if action in {"dialogs", "popups"}:
            return await self.vision.detect_dialogs()
        return await self.vision_analyse(args)

    # ── Mark IX: web control ──────────────────────────────────────────────────

    async def web_control(self, args: dict[str, Any]) -> ToolResult:
        if self.web is None:
            return ToolResult("Web controller is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        w = self.web
        if action in {"open", "go_to"}:
            return await w.open(str(args.get("url") or ""))
        if action == "navigate":
            return await w.navigate(str(args.get("url") or ""))
        if action == "new_tab":
            return await w.new_tab(str(args.get("url") or ""))
        if action == "close_tab":
            return await w.close_tab()
        if action == "switch_tab":
            return await w.switch_tab(int(args.get("index", 1)))
        if action in {"accept_cookies", "accept"}:
            return await w.accept_cookies()
        if action in {"reject_cookies", "reject"}:
            return await w.reject_cookies()
        if action in {"close_popup", "dismiss"}:
            return await w.close_popup()
        if action in {"fill_form", "form"}:
            fields = args.get("fields")
            if not isinstance(fields, dict):
                return ToolResult("fill_form requires a 'fields' object of label→value.", ok=False)
            return await w.fill_form(fields, submit=bool(args.get("submit")))
        if action in {"read_page", "read"}:
            return await w.read_page()
        if action in {"browse", "narrated_scroll", "read_down", "explore"}:
            return await w.narrated_scroll(
                url=str(args.get("url") or ""),
                steps=int(args.get("steps", 6)),
                dwell=float(args.get("dwell", 1.6)),
            )
        if action in {"summarise", "summarize", "evaluate", "review"}:
            return await w.summarise_current(url=str(args.get("url") or ""))
        if action in {"download"}:
            return await w.download_current()
        if action in {"file_dialog", "upload", "save_as"}:
            return await w.handle_file_dialog(str(args.get("path") or ""))
        return ToolResult(
            f"Unsupported web_control action: {action}. Use open, navigate, new_tab, "
            "close_tab, switch_tab, accept_cookies, reject_cookies, close_popup, "
            "fill_form, read_page, browse (narrated slow-scroll), summarise, "
            "download, or file_dialog.",
            ok=False,
        )

    # ── Mark X: research, globe, mind expansion ───────────────────────────────

    async def research_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.research is None:
            return ToolResult("Research agent is not available.", ok=False)
        action = str(args.get("action") or "start").lower().strip()
        topic = str(args.get("topic") or args.get("query") or "")
        if action in {"start", "conduct", "research"}:
            return self.research.start_research(topic, minutes=float(args.get("minutes", 30)))
        if action in {"paper", "write_paper"}:
            return await self.research.write_paper(topic)
        if action in {"stop", "cancel"}:
            return self.research.stop_research()
        if action in {"status", "progress"}:
            return self.research.status()
        return ToolResult("Unsupported research action. Use start, paper, stop, or status.", ok=False)

    def globe_tool(self, args: dict[str, Any]) -> ToolResult:
        place = str(args.get("place") or args.get("location") or args.get("query") or "").strip()
        if not place:
            return ToolResult("Where should I take you on the globe, sir?", ok=False)
        # Drive the GUI globe over the bus (the GlobeView geocodes + fetches news).
        self.bus.globe_request.emit(place)
        return ToolResult(f"Taking you to {place} on the globe, sir — fetching the "
                          "regional news and footage now.")

    def expand_mind(self, args: dict[str, Any]) -> ToolResult:
        if self.corpus is None:
            return ToolResult("Knowledge corpus builder is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"search", "recall", "consult"}:
            query = str(args.get("query") or args.get("topic") or "")
            hits = self.corpus.search(query, limit=int(args.get("limit", 6)))
            if not hits:
                return ToolResult(f"Nothing in the study corpus matches '{query}', sir.")
            return ToolResult("From my expanded knowledge, sir:\n" + "\n".join(f"- {h}" for h in hits))
        status = self.corpus.status()
        return ToolResult(
            f"Knowledge corpus: {'built' if status['built'] else 'not built'} — "
            f"{status['bytes']:,} bytes ({status['bytes']/1024/1024:.1f} MiB) across "
            f"{status['shards']} shard(s)."
        )

    # ── JARVIS layer: protocols, reminders, sentinel ─────────────────────────

    async def protocol_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.protocols is None:
            return ToolResult("Protocols are not available.", ok=False)
        action = str(args.get("action") or "run").lower().strip()
        name = str(args.get("name") or args.get("protocol") or "")
        if action in {"run", "engage", "execute"}:
            return await self.protocols.run(name or str(args.get("query") or ""))
        if action in {"list", "protocols"}:
            return ToolResult(self.protocols.list_text())
        if action in {"create", "save", "define"}:
            steps = args.get("steps")
            if not isinstance(steps, list):
                return ToolResult("create requires a 'steps' list of {tool, args}.", ok=False)
            return self.protocols.create(name, steps, str(args.get("description") or ""))
        if action in {"delete", "remove"}:
            return self.protocols.delete(name)
        return ToolResult("Unsupported protocol action. Use run, list, create, or delete.", ok=False)

    def reminder_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.reminders is None:
            return ToolResult("Reminders are not available.", ok=False)
        action = str(args.get("action") or "add").lower().strip()
        if action in {"add", "set", "create", "remind"}:
            minutes = args.get("minutes")
            return self.reminders.add(
                text=str(args.get("text") or args.get("task") or ""),
                minutes=float(minutes) if minutes is not None else None,
                at=str(args.get("at") or ""),
                phrase=str(args.get("phrase") or args.get("query") or ""),
            )
        if action in {"list", "show"}:
            return self.reminders.list_text()
        if action in {"cancel", "clear", "delete"}:
            rid = args.get("id")
            return self.reminders.cancel(int(rid) if rid is not None else None)
        return ToolResult("Unsupported reminder action. Use add, list, or cancel.", ok=False)

    def sentinel_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.sentinel is None:
            return ToolResult("The system sentinel is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "report", "check"}:
            return self.sentinel.status()
        if action in {"enable", "on"}:
            self.sentinel.set_enabled(True)
            return ToolResult("Ambient monitoring enabled, sir.")
        if action in {"disable", "off"}:
            self.sentinel.set_enabled(False)
            return ToolResult("Ambient monitoring disabled, sir.")
        return ToolResult("Unsupported sentinel action. Use status, enable, or disable.", ok=False)

    # ── Studio deck: audio, literature, pipeline ──────────────────────────────

    async def audio_studio_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.audio_studio is None:
            return ToolResult("The audio studio is not available.", ok=False)
        action = str(args.get("action") or "index_assets").lower().strip()
        if action in {"index_assets", "index", "scan"}:
            return await self.audio_studio.index_assets()
        if action in {"process_vocal_take", "process", "normalise", "normalize"}:
            return await self.audio_studio.process_vocal_take(
                path=str(args.get("path") or args.get("file") or ""),
                target_dbfs=float(args["target_dbfs"]) if args.get("target_dbfs") is not None else None,
                convert_to=str(args.get("convert_to") or "wav"),
            )
        if action in {"export_stem_package", "export", "package"}:
            return await self.audio_studio.export_stem_package(str(args.get("name") or ""))
        return ToolResult(
            "Unsupported audio_studio action. Use index_assets, process_vocal_take, "
            "or export_stem_package.", ok=False,
        )

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

    def campaign_pipeline_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.pipeline is None:
            return ToolResult("The campaign pipeline is not available.", ok=False)
        action = str(args.get("action") or "get_pipeline_snapshot").lower().strip()
        ref = str(args.get("campaign") or args.get("name") or args.get("ref") or "")
        if action in {"get_pipeline_snapshot", "snapshot", "status", "board"}:
            return self.pipeline.snapshot_text()
        if action in {"create", "create_campaign", "new"}:
            return self.pipeline.create_campaign(
                name=str(args.get("name") or ""), brand=str(args.get("brand") or ""),
                value=float(args.get("value") or 0.0), deadline=str(args.get("deadline") or ""),
                notes=str(args.get("notes") or ""),
            )
        if action in {"update_stage", "stage", "move"}:
            return self.pipeline.update_stage(ref, str(args.get("stage") or ""))
        if action in {"log_performance", "log", "performance"}:
            return self.pipeline.log_performance(
                ref, str(args.get("metric") or "engagement"), float(args.get("value") or 0.0)
            )
        if action in {"schedule_content", "content", "schedule"}:
            return self.pipeline.schedule_content(
                ref, str(args.get("title") or ""), str(args.get("platform") or ""),
                str(args.get("scheduled") or ""),
            )
        if action in {"delete", "remove", "delete_campaign"}:
            return self.pipeline.delete_campaign(ref, bool(args.get("confirm")))
        return ToolResult(
            "Unsupported campaign_pipeline action. Use get_pipeline_snapshot, create, "
            "update_stage, log_performance, schedule_content, or delete.", ok=False,
        )

    # ── Realistic-improvements batch ──────────────────────────────────────────

    async def autoplan_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.plan_executor is None:
            return ToolResult("The plan executor is not available.", ok=False)
        steps = args.get("steps")
        if not isinstance(steps, list):
            return ToolResult("autoplan requires a 'steps' list of {tool, args, on_fail}.", ok=False)
        return await self.plan_executor.execute(
            steps, objective=str(args.get("objective") or ""),
            max_retries=int(args.get("max_retries", 2)),
        )

    async def organise_files_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.file_organiser is None:
            return ToolResult("The file organiser is not available.", ok=False)
        action = str(args.get("action") or "preview").lower().strip()
        folder = str(args.get("folder") or "Downloads")
        by_month = bool(args.get("by_month"))
        if action in {"preview", "plan", "dry_run"}:
            return await self.file_organiser.preview(folder, by_month=by_month)
        if action in {"apply", "organise", "organize", "run"}:
            return await self.file_organiser.organise(folder, apply=True, by_month=by_month)
        if action in {"undo", "reverse"}:
            return await self.file_organiser.undo()
        return ToolResult("Unsupported organise_files action. Use preview, apply, or undo.", ok=False)

    def security_watch_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.security is None:
            return ToolResult("The security sentinel is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "report", "check"}:
            return self.security.status()
        if action in {"enable", "on"}:
            self.security.set_enabled(True)
            return ToolResult("Cybersecurity monitoring enabled, sir.")
        if action in {"disable", "off"}:
            self.security.set_enabled(False)
            return ToolResult("Cybersecurity monitoring disabled, sir.")
        return ToolResult("Unsupported security_watch action. Use status, enable, or disable.", ok=False)

    async def backup_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.backup is None:
            return ToolResult("The backup manager is not available.", ok=False)
        action = str(args.get("action") or "backup").lower().strip()
        if action in {"backup", "create", "now"}:
            return await self.backup.backup(str(args.get("note") or ""))
        if action in {"list", "show"}:
            return self.backup.list_backups()
        if action in {"restore"}:
            return await self.backup.restore(str(args.get("archive") or ""))
        return ToolResult("Unsupported backup action. Use backup, list, or restore.", ok=False)

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

    # ── Mark IX: workspace ────────────────────────────────────────────────────

    async def workspace_control(self, args: dict[str, Any]) -> ToolResult:
        if self.workspace is None:
            return ToolResult("Workspace manager is not available.", ok=False)
        action = str(args.get("action") or "snapshot").lower().strip()
        if action in {"snapshot", "capture"}:
            snap = await self.workspace.snapshot_workspace()
            return ToolResult(f"Workspace snapshot: {snap.summary()} (active: {snap.active_window}).")
        if action in {"save", "save_state"}:
            return await self.workspace.save_workspace_state(str(args.get("name") or ""))
        if action in {"restore", "restore_state", "resume"}:
            return await self.workspace.restore_workspace_state(str(args.get("name") or ""))
        if action in {"track", "changes", "track_changes"}:
            return await self.workspace.track_changes()
        if action in {"resume_context", "context"}:
            return ToolResult(self.memory.resume_context(str(args.get("project") or "")) or
                              "No prior context to resume.")
        if action in {"set_project", "project"}:
            name = self.memory.set_active_project(str(args.get("project") or args.get("name") or ""))
            return ToolResult(f"Active project set to '{name or 'none'}'.")
        return ToolResult(
            f"Unsupported workspace action: {action}. Use snapshot, save, restore, "
            "track_changes, resume_context, or set_project.",
            ok=False,
        )

    # ── Mark IX: developer copilot ────────────────────────────────────────────

    async def codebase_copilot(self, args: dict[str, Any]) -> ToolResult:
        if self.copilot is None:
            return ToolResult("Developer Copilot is not available.", ok=False)
        action = str(args.get("action") or "analyse").lower().strip()
        path = str(args.get("path") or "")
        if action in {"analyse", "analyze", "index", "overview"}:
            return await self.copilot.analyse_repository(path)
        if action in {"find_symbol", "symbol"}:
            return await self.copilot.find_symbol(str(args.get("name") or args.get("query") or ""), path)
        if action in {"dependencies", "deps", "impact"}:
            return await self.copilot.dependency_report(str(args.get("module") or ""), path)
        if action in {"task", "refactor", "review", "bughunt", "tests", "docs"}:
            return await self.copilot.engineering_task(
                str(args.get("task") or action), path, str(args.get("focus") or "")
            )
        return ToolResult(
            f"Unsupported copilot action: {action}. Use analyse, find_symbol, "
            "dependencies, or task.",
            ok=False,
        )

    # ── Mark IX: self-repair ──────────────────────────────────────────────────

    async def self_repair(self, args: dict[str, Any]) -> ToolResult:
        if self.selfrepair is None:
            return ToolResult("Self-repair agent is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "incidents", "list"}:
            incidents = self.selfrepair.incidents()
            if not incidents:
                return ToolResult("No incidents captured; the runtime is healthy.")
            return ToolResult("Captured incidents:\n" + "\n".join(i.summary() for i in incidents[-10:]))
        if action in {"propose", "fix", "propose_fix"}:
            return await self.selfrepair.propose_fix(str(args.get("incident_id") or ""))
        if action in {"test", "run_tests", "verify"}:
            return await self.selfrepair.run_tests(str(args.get("path") or ""))
        if action in {"repair", "repair_file", "apply"}:
            # Approval-gated code fix: draft with confirm=false, apply with confirm=true.
            return await self.selfrepair.repair_file(
                incident_id=str(args.get("incident_id") or ""),
                confirm=bool(args.get("confirm")),
            )
        if action in {"revert", "undo"}:
            return self.selfrepair.revert_last()
        if action in {"diagnose", "diagnostics", "diagnostic", "full_check"} and self.diagnostics is not None:
            return await self.diagnostics.run_full()
        return ToolResult(
            f"Unsupported self_repair action: {action}. Use status, propose, run_tests, "
            "repair (confirm=true to apply), revert, or diagnose.",
            ok=False,
        )

    async def diagnostics_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.diagnostics is None:
            return ToolResult("The diagnostics engine is not available.", ok=False)
        return await self.diagnostics.run_full()

    def cursor_overlay_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.cursor_overlay is None:
            return ToolResult("The cursor overlay is not available.", ok=False)
        action = str(args.get("action") or "toggle").lower().strip()
        if action in {"show", "on", "enable"}:
            self.cursor_overlay.start()
            return ToolResult("Cursor halo on, sir — you'll see where I move the mouse.")
        if action in {"hide", "off", "disable"}:
            self.cursor_overlay.stop()
            return ToolResult("Cursor halo hidden, sir.")
        state = self.cursor_overlay.toggle()
        return ToolResult(f"Cursor halo {'on' if state else 'off'}, sir.")

    # ── Living Memory: patch notes, learning, transcript, programming ─────────

    def patch_notes_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.changelog is None:
            return ToolResult("The changelog is not available.", ok=False)
        action = str(args.get("action") or "latest").lower().strip()
        if action in {"all", "history", "full"}:
            return ToolResult(self.changelog.render_all())
        version = str(args.get("version") or "").strip()
        if version:
            release = self.changelog.find(version)
            return ToolResult(release.speak() if release else f"No release matches '{version}', sir.",
                              ok=release is not None)
        count = int(args.get("count", 1))
        return ToolResult(self.changelog.render_latest(max(1, count)))

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

    def transcript_tool(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "export").lower().strip()
        if action in {"path", "where"}:
            return ToolResult(f"This session's transcript is being recorded to "
                              f"{self.memory.transcript_path()}, sir.")
        if action in {"recall", "search"}:
            return self.recall_conversation({"query": args.get("query") or "", "limit": args.get("limit") or 10})
        # export
        path = self.memory.export_transcript_markdown()
        if not path:
            return ToolResult("There's nothing recorded to export yet, sir.")
        return ToolResult(f"Exported this session's verbatim transcript to {path}, sir.")

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
                          "databases, security, testing or a language, sir.")

    def cyber_knowledge_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.cyber is None:
            return ToolResult("The cybersecurity knowledge base is not available.", ok=False)
        query = str(args.get("query") or args.get("topic") or "")
        answer = self.cyber.answer(query)
        if answer:
            return ToolResult(answer)
        rows = self.memory.query(query or "security", limit=5)
        hits = [r.get("value", "") for r in rows if str(r.get("key_ref", "")).startswith("cyber_")]
        if hits:
            return ToolResult("\n".join(f"- {h}" for h in hits))
        return ToolResult("Ask me about the CIA triad, threat modelling, OWASP risks, "
                          "encryption, authentication, malware, detection, incident response, "
                          "secure development or cloud hardening, sir.")

    # ── Mark IX: proactive + display ──────────────────────────────────────────

    async def proactive_check(self, args: dict[str, Any]) -> ToolResult:
        if self.proactive is None:
            return ToolResult("Proactive intelligence is not available.", ok=False)
        return await self.proactive.check_now()

    def display_info(self, args: dict[str, Any]) -> ToolResult:
        if self.display is None:
            return ToolResult("Display topology manager is not available.", ok=False)
        if bool(args.get("refresh")):
            self.display.refresh()
        return ToolResult(self.display.summary())

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

    # ── dual-mode / entrepreneurial intelligence ──────────────────────────────

    def ai_mode(self, args: dict[str, Any]) -> ToolResult:
        """Report the active intelligence mode (cloud vs offline) and models."""
        if self.ai is None:
            return ToolResult("Mode information is unavailable.", ok=False)
        return self.ai.report()

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

    async def product_research(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Commerce intelligence unavailable.", ok=False)
        action = str(args.get("action") or "score").lower().strip()
        name = str(args.get("name") or args.get("product") or "")
        desc = str(args.get("description") or args.get("notes") or "")
        if action in {"score", "evaluate"}:
            return await self.commerce.dropship.score_product(name, desc, args.get("metrics"))
        if action in {"discover", "discovery", "find"}:
            return await self.commerce.product.discover(
                str(args.get("niche") or name or "home products"),
                int(args.get("count") or 5))
        if action in {"validate", "validation"}:
            return await self.commerce.dropship.validate(name, desc)
        if action in {"competition", "saturation"}:
            return await self.commerce.dropship.analyse_competition(
                str(args.get("niche") or name))
        if action in {"log", "research_log"}:
            rows = self.commerce.dropship.research_log()
            return ToolResult("Product research log:\n" + "\n".join(
                f"- {r.get('key_ref')}: {r.get('value', '')[:120]}" for r in rows) or "empty")
        return ToolResult(f"Unsupported product_research action: {action}.", ok=False)

    async def tiktok_intel(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Commerce intelligence unavailable.", ok=False)
        action = str(args.get("action") or "trend").lower().strip()
        if action in {"trend", "trends", "report"}:
            return await self.commerce.tiktok.trend_report(str(args.get("niche") or ""))
        if action in {"product"}:
            return await self.commerce.tiktok.product_report(str(args.get("product") or ""))
        if action in {"virality", "score"}:
            return self.commerce.tiktok.score_virality(args.get("signals") or {})
        return ToolResult(f"Unsupported tiktok_intel action: {action}.", ok=False)

    async def instagram_intel(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Commerce intelligence unavailable.", ok=False)
        action = str(args.get("action") or "discover").lower().strip()
        if action in {"discover", "discovery"}:
            return await self.commerce.instagram.discover(str(args.get("niche") or "home products"))
        if action in {"influencer", "influencers"}:
            return await self.commerce.instagram.influencer_strategy(str(args.get("brand") or "Hausables"))
        if action in {"weekly", "report"}:
            return await self.commerce.instagram.weekly_report(str(args.get("niche") or "home products"))
        return ToolResult(f"Unsupported instagram_intel action: {action}.", ok=False)

    # ── Mark X.5: AI Operating System layer ───────────────────────────────────

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

    async def proactive_report_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.reporting is None:
            return ToolResult("The reporting service is not available.", ok=False)
        kind = str(args.get("kind") or args.get("action") or "daily_business")
        return await self.reporting.generate(kind)

    async def awareness_tool(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "situation").lower().strip()
        if action in {"situation", "report", "status"}:
            if self.cognitive_loop is None:
                return ToolResult("The cognitive loop is not available.", ok=False)
            return self.cognitive_loop.situation_report()
        if self.cognition is None:
            return ToolResult("Cognitive state is not available.", ok=False)
        if action in {"add_priority", "priority"}:
            priorities = await asyncio.to_thread(
                self.cognition.add_priority, str(args.get("text") or ""))
            return ToolResult("Priorities noted: " + "; ".join(priorities[-5:]))
        if action in {"add_task", "task"}:
            task = await asyncio.to_thread(
                self.cognition.add_task, str(args.get("title") or args.get("text") or ""),
                str(args.get("project") or ""), str(args.get("due") or ""))
            return ToolResult(f"Task tracked: {task.get('title', '?')}.")
        if action in {"complete_task", "done"}:
            ok = await asyncio.to_thread(
                self.cognition.complete_task, str(args.get("title") or args.get("text") or ""))
            return ToolResult("Task marked complete." if ok else "No matching open task.", ok=ok)
        if action in {"goals", "list_goals"}:
            goals = await asyncio.to_thread(self.cognition.goals.list_goals)
            if not goals:
                return ToolResult("No goals are currently tracked.")
            return ToolResult("Goals:\n" + "\n".join(
                f"- [{g.status}] {g.title}" for g in goals[:12]))
        return ToolResult(
            f"Unsupported awareness action: {action}. Use situation, add_priority, "
            "add_task, complete_task, or goals.",
            ok=False,
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
                return ToolResult("No such entity (or it has no relationships yet).")
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

    async def executive_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.executive is None:
            return ToolResult("Executive assistant mode is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "overview"}:
            return await self.executive.status()
        if action in {"prioritise", "prioritize", "priorities"}:
            return await self.executive.prioritise()
        if action in {"schedule", "book"}:
            return await self.executive.schedule(
                str(args.get("title") or args.get("text") or ""),
                str(args.get("when") or args.get("due") or ""),
                str(args.get("notes") or ""))
        if action in {"meeting_summary", "summarise_meeting", "minutes"}:
            return await self.executive.summarise_meeting(
                str(args.get("transcript") or args.get("text") or ""),
                str(args.get("title") or ""))
        if action in {"plan", "workflow", "plan_workflow"}:
            return await self.executive.plan_workflow(
                str(args.get("objective") or args.get("text") or ""))
        if action in {"progress", "monitor"}:
            return await self.executive.progress()
        if action in {"track", "track_project"}:
            return await self.executive.track_project(
                str(args.get("project") or args.get("name") or ""),
                str(args.get("notes") or ""))
        return ToolResult(
            f"Unsupported executive action: {action}. Use status, prioritise, "
            "schedule, meeting_summary, plan, progress, or track.",
            ok=False,
        )

    async def momentum_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.momentum is None:
            return ToolResult("The momentum engine is not available.", ok=False)
        action = str(args.get("action") or "focus").lower().strip()
        if action in {"standup", "stand_up", "status"}:
            return self.momentum.standup()
        if action in {"plan", "milestones", "roadmap"}:
            return await self.momentum.plan(
                str(args.get("project") or ""), str(args.get("goal") or args.get("text") or ""))
        return self.momentum.focus()

    async def competitor_intel_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None or getattr(self.commerce, "competitor", None) is None:
            return ToolResult("Competitor intelligence is not available.", ok=False)
        action = str(args.get("action") or "store").lower().strip()
        target = str(args.get("target") or args.get("store") or args.get("competitor") or "")
        if not target:
            return ToolResult("A competitor/store name is required.", ok=False)
        if action in {"store", "store_analysis"}:
            return await self.commerce.competitor.store_analysis(target)
        if action in {"offer", "offer_analysis"}:
            return await self.commerce.competitor.offer_analysis(
                target, str(args.get("offer") or ""))
        if action in {"funnel", "funnel_analysis"}:
            return await self.commerce.competitor.funnel_analysis(target)
        return ToolResult(
            f"Unsupported competitor_intel action: {action}. Use store, offer, or funnel.",
            ok=False,
        )

    async def brand_growth_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None or getattr(self.commerce, "growth", None) is None:
            return ToolResult("The brand growth agent is not available.", ok=False)
        action = str(args.get("action") or "strategy").lower().strip()
        if action in {"strategy"}:
            return await self.commerce.growth.strategy(str(args.get("focus") or ""))
        if action in {"conversion", "cro", "conversion_optimisation"}:
            return await self.commerce.growth.conversion_optimisation(
                str(args.get("page") or "store"))
        if action in {"positioning", "position"}:
            return await self.commerce.growth.positioning(
                str(args.get("product") or ""))
        if action in {"retention", "retention_systems"}:
            return await self.commerce.growth.retention_systems()
        return ToolResult(
            f"Unsupported brand_growth action: {action}. Use strategy, conversion, "
            "positioning, or retention.",
            ok=False,
        )

    def audio_devices_tool(self, args: dict[str, Any]) -> ToolResult:
        """List, inspect or choose ORION's microphone and speaker."""
        from . import audio_devices as ad
        action = str(args.get("action") or "status").lower().strip()
        if action in {"list", "devices"}:
            return ToolResult(ad.list_devices())
        if action in {"status", "current", "describe"}:
            return ToolResult(ad.describe())
        if action in {"set_output", "output", "speaker", "set_speaker"}:
            return ToolResult(ad.set_device("output", str(args.get("device") or "")))
        if action in {"set_input", "input", "mic", "set_mic", "microphone"}:
            return ToolResult(ad.set_device("input", str(args.get("device") or "")))
        return ToolResult(
            f"Unsupported audio_devices action: {action}. Use list, status, "
            "set_output, or set_input.",
            ok=False,
        )

    async def geo_tool(self, args: dict[str, Any]) -> ToolResult:
        """Phase 8: worldwide geospatial intelligence — locate any place, list
        the settlements near it, or describe its administrative hierarchy."""
        if self.geo is None:
            return ToolResult("The geospatial engine is not available.", ok=False)
        action = str(args.get("action") or "locate").lower().strip()
        query = str(args.get("query") or args.get("place") or "")
        if not query:
            return ToolResult("A place is required, sir.", ok=False)
        if action in {"locate", "find", "fly"}:
            return await self.geo.locate(query)
        if action in {"nearby", "towns_near", "near", "within"}:
            radius = float(args.get("radius_km") or args.get("radius") or 50.0)
            return await self.geo.towns_near(query, radius)
        if action in {"describe", "admin", "administrative", "detail"}:
            return await self.geo.describe(query)
        return ToolResult(
            f"Unsupported geo action: {action}. Use locate, nearby, or describe.",
            ok=False,
        )

    def emotion_tool(self, args: dict[str, Any]) -> ToolResult:
        """Mark X.7: inspect or pin ORION's emotional rendering."""
        if self.emotion is None:
            return ToolResult("The emotion engine is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "current", "describe"}:
            desc = self.emotion.describe()
            return ToolResult(
                f"Emotional state: {desc['current']} (baseline {desc['baseline']}"
                + (f", pinned {desc['manual']}" if desc.get("manual") else "")
                + f"). Last sentiment: {desc['last_sentiment']['sentiment']} "
                f"@ {desc['last_sentiment']['confidence']:.2f}."
            )
        if action in {"set", "pin", "express"}:
            return ToolResult(self.emotion.set_emotion(str(args.get("name") or "")))
        if action in {"auto", "clear", "release"}:
            return ToolResult(self.emotion.set_emotion("auto"))
        return ToolResult(
            f"Unsupported emotion action: {action}. Use status, set (with name), or auto.",
            ok=False,
        )

    async def founder_knowledge(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Founder knowledge unavailable.", ok=False)
        action = str(args.get("action") or "profile").lower().strip()
        if action in {"list"}:
            return ToolResult("Founders in the knowledge base: "
                              + ", ".join(self.commerce.founder.list_founders()))
        if action in {"profile"}:
            return self.commerce.founder.profile(str(args.get("name") or ""))
        if action in {"learn", "apply"}:
            return await self.commerce.founder.learn_from(
                str(args.get("name") or ""), str(args.get("question") or ""))
        return ToolResult(f"Unsupported founder_knowledge action: {action}.", ok=False)

    async def business_advisor(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Business advisor unavailable.", ok=False)
        action = str(args.get("action") or "advise").lower().strip()
        advisor = self.commerce.advisor
        if action in {"advise", "advice"}:
            return await advisor.advise(str(args.get("topic") or args.get("question") or ""))
        if action in {"brand", "brand_strategy"}:
            return await advisor.brand_strategy()
        if action in {"growth", "growth_plan"}:
            return await advisor.growth_plan(str(args.get("horizon") or "next 90 days"))
        if action in {"store", "optimise", "optimize", "cro"}:
            return await advisor.store_optimisation()
        return ToolResult(f"Unsupported business_advisor action: {action}.", ok=False)

    def commerce_hub(self, args: dict[str, Any]) -> ToolResult:
        if self.hub is None:
            return ToolResult("E-commerce hub unavailable.", ok=False)
        return self.hub.report()

    def community_share(self, args: dict[str, Any]) -> ToolResult:
        if self.community is None:
            return ToolResult("Community layer unavailable.", ok=False)
        action = str(args.get("action") or "list").lower().strip()
        if action in {"export_pack"}:
            return self.community.export_pack(str(args.get("id") or args.get("pack_id") or ""),
                                              str(args.get("privacy") or "shared"))
        if action in {"import_pack"}:
            return self.community.import_pack(str(args.get("path") or ""))
        if action in {"export_research"}:
            return self.community.export_research(str(args.get("privacy") or "shared"),
                                                  bool(args.get("anonymise", True)))
        if action in {"import_research"}:
            return self.community.import_research(str(args.get("path") or ""))
        if action in {"list", "bundles"}:
            return ToolResult("Community bundles:\n" + "\n".join(self.community.list_bundles()) or "none")
        return ToolResult(f"Unsupported community_share action: {action}.", ok=False)

    # ── files ─────────────────────────────────────────────────────────────────

    async def find_files(self, args: dict[str, Any]) -> ToolResult:
        query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("name") or ""), "find_files.query"
        ).strip().lower()
        if not query:
            return ToolResult("No search query supplied.", ok=False)
        open_first = bool(args.get("open") or args.get("open_first"))
        matches = await asyncio.to_thread(self._scan_user_files, query)
        if not matches:
            return ToolResult(f"No files or folders matching '{query}' were found.")
        if open_first:
            try:
                os.startfile(matches[0])  # type: ignore[attr-defined]
            except Exception as exc:
                return ToolResult(
                    f"Found {len(matches)} match(es) but could not open the first: {exc}\n"
                    + "\n".join(matches[:20]),
                    ok=False,
                )
            others = "\n".join(matches[1:15]) or "none"
            return ToolResult(f"Opened {matches[0]}.\nOther matches:\n{others}")
        return ToolResult(f"{len(matches)} match(es):\n" + "\n".join(matches[:25]))

    def _scan_user_files(self, query: str) -> list[str]:
        """Time-boxed name search across the user's common folders (runs off-thread)."""
        home = Path.home()
        roots: list[Path] = []
        try:
            for entry in home.iterdir():
                if entry.is_dir() and (
                    entry.name in {"Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"}
                    or entry.name.startswith("OneDrive")
                ):
                    roots.append(entry)
        except Exception:
            pass
        roots.append(BASE_DIR)
        ignored = {"appdata", "node_modules", "__pycache__", ".git", ".venv", "venv"}
        matches: list[str] = []
        seen: set[str] = set()
        deadline = time.monotonic() + 8.0
        for root in roots:
            if time.monotonic() > deadline or len(matches) >= 40:
                break
            try:
                for path in root.rglob("*"):
                    if time.monotonic() > deadline or len(matches) >= 40:
                        break
                    if any(part.lower() in ignored or part.startswith(".") for part in path.parts):
                        continue
                    if query in path.name.lower():
                        resolved = str(path)
                        if resolved not in seen:
                            seen.add(resolved)
                            matches.append(resolved)
            except Exception:
                continue
        # Folders first, then the tightest name matches.
        matches.sort(key=lambda m: (0 if Path(m).is_dir() else 1, len(Path(m).name)))
        return matches

    def file_controller(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        path   = self._resolve_user_path(str(args.get("path") or args.get("directory") or BASE_DIR))
        if action in {"search_codebase", "search", "grep"}:
            query = SecuritySanitiser.guard_text(
                str(args.get("query") or args.get("text") or ""), "file_controller.query"
            )
            if not query:
                return ToolResult("No codebase search query supplied.", ok=False)
            if not path.exists() or not path.is_dir():
                return ToolResult(f"Search root not found: {path}", ok=False)
            matches = self._search_codebase(path, query)
            if not matches:
                return ToolResult(f"No codebase matches for: {query}.")
            text  = "\n".join(matches[:80])
            chain: list[tuple[str, dict[str, Any]]] = []
            if args.get("diagnose") or args.get("inspect"):
                first_path = matches[0].split(":", 1)[0]
                chain.append(("file_controller", {"action": "read_text", "path": first_path}))
                chain.append((
                    "save_memory",
                    {"category": "projects", "key": "last_codebase_search",
                     "value": f"{query}: {matches[0][:240]}"}
                ))
            return ToolResult(text, chain=chain)
        if action in {"list", "dir", "inventory"}:
            if not path.exists():
                return ToolResult(f"Path not found: {path}", ok=False)
            if path.is_file():
                return ToolResult(f"File: {path} ({path.stat().st_size} bytes)")
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return ToolResult("\n".join(entries[:200]) or "Directory is empty.")
        if action in {"mkdir", "create_dir", "structure_build", "build_structure"}:
            self._ensure_write_safe(path)
            path.mkdir(parents=True, exist_ok=True)
            return ToolResult(f"Directory structure ready: {path}")
        if action in {"write_text", "append_text"}:
            self._ensure_write_safe(path)
            text = SecuritySanitiser.guard_text(
                str(args.get("text") or args.get("content") or ""), "file_controller.text"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if action == "append_text" else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(text)
            return ToolResult(f"File {'appended' if mode == 'a' else 'written'}: {path}")
        if action in {"read", "read_text"}:
            if not path.exists() or not path.is_file():
                return ToolResult(f"File not found: {path}", ok=False)
            text = path.read_text(encoding="utf-8", errors="replace")
            return ToolResult(text[:6000])
        if action in {"delete", "remove"}:
            self._ensure_write_safe(path)
            if not path.exists():
                return ToolResult(f"Path already absent: {path}")
            if path.is_dir():
                if any(path.iterdir()):
                    return ToolResult("Refusing to remove a non-empty directory.", ok=False)
                path.rmdir()
            else:
                path.unlink()
            return ToolResult(f"Removed: {path}")
        return ToolResult(f"Unsupported file action: {action}", ok=False)

    async def process_file(self, args: dict[str, Any]) -> ToolResult:
        raw_path = str(args.get("path") or args.get("file_path") or "")
        if not raw_path:
            return ToolResult("No file path supplied.", ok=False)
        prompt = str(args.get("prompt") or args.get("question") or args.get("instruction") or "")
        path   = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        if not path.is_absolute():
            path = BASE_DIR / path
        # inspect_async offloads all synchronous file I/O via asyncio.to_thread()
        result = await self.file_intel.inspect_async(path, prompt=prompt)
        if result.ok:
            self.bus.log.emit(f"FILE: scanned {path.name}")
        return result

    # ── software engineering workbench ────────────────────────────────────────

    DEV_COMMANDS = {
        "python", "py", "pytest", "pip", "node", "npm", "npx", "tsc",
        "cargo", "go", "dotnet", "git", "rustc",
    }
    CODE_SUFFIX_LANGS = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
        ".jsx": "JavaScript", ".cs": "C#", ".rs": "Rust", ".go": "Go",
        ".html": "HTML", ".css": "CSS", ".json": "JSON", ".md": "Markdown",
        ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML", ".sql": "SQL",
    }

    async def dev_workbench(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").lower().strip()
        if action in {"analyse", "analyse_repo", "analyze", "analyze_repo"}:
            root = self._resolve_user_path(str(args.get("path") or BASE_DIR))
            return await asyncio.to_thread(self._analyse_repo, root)
        if action in {"read", "read_file", "read_code"}:
            path  = self._resolve_user_path(str(args.get("path") or ""))
            start = max(1, int(args.get("start_line") or 1))
            count = max(1, min(400, int(args.get("line_count") or 200)))
            return await asyncio.to_thread(self._read_code, path, start, count)
        if action in {"run", "run_command", "test", "run_tests"}:
            return await self._run_dev_command(args)
        if action in {"create_python_project", "new_project", "scaffold"}:
            return self._create_python_project(args)
        return ToolResult(
            f"Unsupported workbench action: {action}. "
            "Use analyse_repo, read_file, run_command, or create_python_project.",
            ok=False,
        )

    def _analyse_repo(self, root: Path) -> ToolResult:
        if not root.is_dir():
            return ToolResult(f"Repository root not found: {root}", ok=False)
        ignored = {
            ".git", "__pycache__", ".venv", "venv", "node_modules", "target",
            "bin", "obj", ".mypy_cache", ".pytest_cache", "dist", "build",
        }
        key_names = {
            "readme.md", "pyproject.toml", "package.json", "cargo.toml",
            "go.mod", "requirements.txt", "setup.py", "tsconfig.json",
        }
        languages: dict[str, int] = {}
        line_totals: dict[str, int] = {}
        key_files: list[str] = []
        todo_count = 0
        scanned = 0
        for path in root.rglob("*"):
            if scanned >= 4000:
                break
            if any(part in ignored for part in path.parts):
                continue
            if not path.is_file():
                continue
            scanned += 1
            if path.name.lower() in key_names or path.suffix.lower() == ".csproj":
                key_files.append(str(path.relative_to(root)))
            language = self.CODE_SUFFIX_LANGS.get(path.suffix.lower())
            if language is None:
                continue
            languages[language] = languages.get(language, 0) + 1
            try:
                if path.stat().st_size < 600_000:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    line_totals[language] = line_totals.get(language, 0) + text.count("\n") + 1
                    todo_count += len(re.findall(r"(?i)\b(?:todo|fixme|hack)\b", text))
            except Exception:
                continue
        if not languages:
            return ToolResult(f"No recognised source files under {root}.")
        summary = ", ".join(
            f"{lang}: {count} file(s), ~{line_totals.get(lang, 0)} lines"
            for lang, count in sorted(languages.items(), key=lambda kv: kv[1], reverse=True)
        )
        return ToolResult(
            f"Repository analysis: {root}\n"
            f"Languages: {summary}.\n"
            f"Key files: {', '.join(key_files[:12]) or 'none detected'}.\n"
            f"Open TODO/FIXME markers: {todo_count}.  Files scanned: {scanned}."
        )

    def _read_code(self, path: Path, start: int, count: int) -> ToolResult:
        if not path.is_file():
            return ToolResult(f"File not found: {path}", ok=False)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(len(lines), start - 1 + count)
        if start > len(lines):
            return ToolResult(f"{path} has only {len(lines)} lines.", ok=False)
        numbered = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        return ToolResult(f"{path} lines {start}-{end} of {len(lines)}:\n{numbered[:8000]}")

    async def _run_dev_command(self, args: dict[str, Any]) -> ToolResult:
        command = SecuritySanitiser.guard_text(
            str(args.get("command") or ""), "dev.command"
        ).strip()
        if not command:
            return ToolResult("No development command supplied.", ok=False)
        parts = command.split()
        binary = Path(parts[0]).name.lower().removesuffix(".exe")
        if binary not in self.DEV_COMMANDS:
            return ToolResult(
                f"Command '{parts[0]}' is not on the development allowlist "
                f"({', '.join(sorted(self.DEV_COMMANDS))}).",
                ok=False,
            )
        cwd = self._resolve_user_path(str(args.get("path") or BASE_DIR))
        if cwd.is_file():
            cwd = cwd.parent
        if not cwd.is_dir():
            return ToolResult(f"Working directory not found: {cwd}", ok=False)

        def _execute() -> str:
            completed = subprocess.run(
                parts, cwd=str(cwd), capture_output=True, text=True,
                timeout=120, shell=False,
            )
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            output = f"exit code {completed.returncode}\n{stdout[-5000:]}"
            if stderr:
                output += f"\nSTDERR:\n{stderr[-2000:]}"
            return output

        try:
            output = await asyncio.to_thread(_execute)
        except subprocess.TimeoutExpired:
            return ToolResult("Development command timed out after 120 seconds.", ok=False)
        except FileNotFoundError:
            return ToolResult(f"Command not found on this host: {parts[0]}", ok=False)
        return ToolResult(f"$ {command}  (cwd: {cwd})\n{output}")

    def _create_python_project(self, args: dict[str, Any]) -> ToolResult:
        raw_name = str(args.get("name") or args.get("project") or "new_project")
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip().lower()).strip("_") or "new_project"
        root = self._resolve_user_path(str(args.get("path") or (BASE_DIR / "projects" / name)))
        self._ensure_write_safe(root)
        package = root / "src" / name
        tests   = root / "tests"
        package.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(f'"""{name} package."""\n', encoding="utf-8")
        (package / "main.py").write_text(
            "def main() -> None:\n"
            f'    print("Hello from {name}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n",
            encoding="utf-8",
        )
        (tests / f"test_{name}.py").write_text(
            f"from src.{name}.main import main\n\n\n"
            "def test_main_runs():\n"
            "    main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            f"# {raw_name.strip() or name}\n\nScaffolded by O.R.I.O.N.\n", encoding="utf-8"
        )
        (root / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{name.replace("_", "-")}"\n'
            'version = "0.1.0"\n'
            'requires-python = ">=3.10"\n',
            encoding="utf-8",
        )
        (root / ".gitignore").write_text("__pycache__/\n.venv/\n*.pyc\n", encoding="utf-8")
        return ToolResult(
            f"Python project scaffolded at {root} (src/{name}, tests, pyproject.toml, README)."
        )

    # ── memory and agentic tools ──────────────────────────────────────────────

    def save_memory(self, args: dict[str, Any]) -> ToolResult:
        category = str(args.get("category") or "notes")
        key      = str(args.get("key") or args.get("key_ref") or "entry")
        value    = str(args.get("value") or args.get("fact") or "")
        result   = self.memory.save(category, key, value)
        return ToolResult(result)

    def query_intelligence(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or args.get("text") or "")
        limit = int(args.get("limit") or 8)
        rows  = self.memory.query(query, limit=limit)
        if not rows:
            return ToolResult("No matching intelligence records.")
        lines = [
            f"{row['category']}/{row['key_ref']}: {row['value']} ({row['updated_at']})"
            for row in rows
        ]
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

    async def execute_plan(self, args: dict[str, Any]) -> ToolResult:
        """Agentic plan runner: execute steps sequentially, verify, report."""
        steps = args.get("steps") or []
        objective = str(args.get("objective") or "").strip()
        if not isinstance(steps, list) or not steps:
            return ToolResult("No plan steps supplied.", ok=False)
        report: list[str] = [f"OBJECTIVE: {objective}"] if objective else []
        succeeded = failed = 0
        for number, step in enumerate(list(steps)[:8], 1):
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or step.get("name") or "").strip()
            raw_args = step.get("args")
            if not isinstance(raw_args, dict):
                try:
                    raw_args = json.loads(str(step.get("args_json") or "{}"))
                except Exception:
                    raw_args = {}
            if tool in {"execute_plan", "shutdown_orion"}:
                report.append(f"{number}. {tool}: skipped - not permitted inside a plan.")
                continue
            try:
                result = await self.dispatch(tool, raw_args if isinstance(raw_args, dict) else {})
            except SecurityViolation as exc:
                failed += 1
                report.append(f"{number}. {tool}: BLOCKED - {exc}")
                continue
            if result.ok:
                succeeded += 1
            else:
                failed += 1
            line = result.text.splitlines()[0][:220] if result.text else ""
            report.append(f"{number}. {tool}: {'OK' if result.ok else 'FAILED'} - {line}")
        report.append(f"VERIFICATION: {succeeded} step(s) succeeded, {failed} failed.")
        return ToolResult("\n".join(report), ok=failed == 0)

    # ── host utilities ────────────────────────────────────────────────────────

    def clipboard_operate(self, args: dict[str, Any]) -> ToolResult:
        action    = str(args.get("action") or "read").lower().strip()
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return ToolResult("Clipboard unavailable.", ok=False)
        if action == "read":
            text = clipboard.text()
            return ToolResult(text if text else "Clipboard is empty.")
        if action in {"copy", "write", "set"}:
            text = SecuritySanitiser.guard_text(str(args.get("text") or ""), "clipboard.text")
            clipboard.setText(text)
            return ToolResult("Clipboard updated.")
        return ToolResult(f"Unsupported clipboard action: {action}", ok=False)

    def process_governor(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        # ── the biggest power users (what the user asks ORION to police) ──────
        if action in {"top", "heavy", "hogs", "power", "high_power"}:
            metric = str(args.get("metric") or "cpu").lower()
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    procs.append((int(info["pid"]), str(info.get("name") or "?"),
                                  float(info.get("cpu_percent") or 0.0),
                                  float(info.get("memory_percent") or 0.0)))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            key = 3 if metric in {"ram", "memory", "mem"} else 2
            procs.sort(key=lambda p: -p[key])
            lines = [f"Top processes by {'RAM' if key == 3 else 'CPU'}:"]
            for pid, name, cpu, ram in procs[:12]:
                lines.append(f"  PID {pid:>6}  {name[:30]:<30}  CPU {cpu:5.1f}%  RAM {ram:4.1f}%")
            try:
                from . import gpu_stats
                g = gpu_stats.sample()
                if g.get("available"):
                    lines.append(f"GPU {g['name']}: {g['util']:.0f}% util, "
                                 f"{g['mem_percent']:.0f}% VRAM, {g['temp_c']}°C")
            except Exception:
                pass
            lines.append("Say 'terminate PID <n>' or 'close <name>' and confirm to stop one.")
            return ToolResult("\n".join(lines))
        if action in {"list", "inventory"}:
            rows = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    rows.append(
                        f"{info['pid']:>6}  {info.get('name') or 'unknown'}  "
                        f"CPU {info.get('cpu_percent') or 0:.1f}%  "
                        f"RAM {info.get('memory_percent') or 0:.1f}%"
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return ToolResult("\n".join(rows[:80]))
        if action in {"terminate", "kill", "stop", "restart"}:
            pid   = args.get("pid")
            label = SecuritySanitiser.guard_text(
                str(args.get("name") or args.get("label") or ""), "process.name"
            )
            # Permission gate: destructive, so require explicit confirmation.
            if not bool(args.get("confirm")):
                target = f"PID {pid}" if pid is not None else f"'{label}'"
                return ToolResult(
                    f"Terminating {target} requires your confirmation, sir — "
                    "repeat the request with confirm=true and I shall stop it.",
                    ok=False,
                )
            if pid is not None:
                try:
                    proc = psutil.Process(int(pid))
                except (psutil.NoSuchProcess, ValueError):
                    return ToolResult(f"No process with PID {pid}.", ok=False)
                name = proc.name()
                terminated: list[str] = []
                self.desktop._terminate_process(proc, terminated)
                result = f"Terminated PID {pid} ({name})."
                if action == "restart":
                    restarted = self.desktop.open_app(name)
                    result += f" Restart: {restarted.text}"
                return ToolResult(result)
            if label:
                closed = self.desktop.close_app(label)
                if action == "restart" and closed.ok:
                    reopened = self.desktop.open_app(label)
                    return ToolResult(f"{closed.text} Restart: {reopened.text}")
                return closed
            return ToolResult("No process PID or label supplied.", ok=False)
        return ToolResult(
            f"Unsupported process action: {action}. Use top (biggest power "
            "users), list, terminate, or restart.",
            ok=False,
        )

    def system_notify(self, args: dict[str, Any]) -> ToolResult:
        message  = SecuritySanitiser.guard_text(
            str(args.get("message") or args.get("text") or "System event."), "notify.message"
        )
        priority = int(args.get("priority") or 1)
        self.bus.banner.emit(message, priority)
        self.bus.log.emit(f"ALERT: {message}")
        return ToolResult("System notification projected.")

    def shutdown_orion(self, args: dict[str, Any]) -> ToolResult:
        self.bus.log.emit("SYS: shutdown directive accepted.")
        self.bus.request_shutdown.emit()
        return ToolResult("Orion shutdown initiated.")

    # ── chaining logic ────────────────────────────────────────────────────────

    def _derive_chain(
        self, name: str, args: dict[str, Any], result: ToolResult
    ) -> list[tuple[str, dict[str, Any]]]:
        if not result.ok:
            return []
        action = str(args.get("action") or "").lower()
        chain: list[tuple[str, dict[str, Any]]] = []
        if name == "save_memory":
            key = str(args.get("key") or args.get("key_ref") or "")
            if key:
                chain.append(("query_intelligence", {"query": key, "limit": 3}))
        elif name == "file_controller" and action in {"list", "dir", "inventory"}:
            query = str(args.get("query") or args.get("contains") or "").strip()
            if query:
                chain.append((
                    "file_controller",
                    {"action": "search_codebase",
                     "path": args.get("path") or args.get("directory") or str(BASE_DIR),
                     "query": query,
                     "diagnose": bool(args.get("diagnose"))},
                ))
        elif name == "query_intelligence" and "No matching intelligence" in result.text:
            remember_query = str(args.get("query") or "").strip()
            if remember_query:
                chain.append((
                    "save_memory",
                    {"category": "notes", "key": "unresolved_query", "value": remember_query[:300]},
                ))
        return chain

    # ── helpers ───────────────────────────────────────────────────────────────

    def _search_codebase(self, root: Path, query: str) -> list[str]:
        query_lower  = query.lower()
        allowed      = {".py", ".txt", ".md", ".json", ".html", ".css", ".js", ".ts",
                        ".yml", ".yaml", ".toml"}
        ignored_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                        ".mypy_cache", ".pytest_cache"}
        matches: list[str] = []
        for file_path in root.rglob("*"):
            if len(matches) >= 160:
                break
            if any(part in ignored_dirs for part in file_path.parts):
                continue
            if not file_path.is_file() or file_path.suffix.lower() not in allowed:
                continue
            if is_protected_path(file_path):
                continue
            try:
                if file_path.stat().st_size > 1_200_000:
                    continue
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if query_lower in line.lower():
                            matches.append(f"{file_path}:{line_number}: {line.strip()[:220]}")
                            break
            except Exception:
                continue
        return matches

    def _normalise_url(self, url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme not in {"http", "https"}:
            raise SecurityViolation(
                "blocked unsafe browser URL: only HTTP and HTTPS are permitted"
            )
        return parsed.geturl()

    def _resolve_user_path(self, raw: str) -> Path:
        SecuritySanitiser.guard_text(raw, "path")
        expanded = os.path.expandvars(os.path.expanduser(raw))
        path     = Path(expanded)
        if not path.is_absolute():
            path = BASE_DIR / path
        resolved = path.resolve()
        # Reading core code is permitted (dev_workbench introspection);
        # every write path re-checks through _ensure_write_safe.
        return resolved

    def _ensure_write_safe(self, path: Path) -> None:
        resolved = path.resolve()
        if is_protected_path(resolved):
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        if BASE_DIR not in resolved.parents and resolved != BASE_DIR:
            raise SecurityViolation(
                "blocked unsafe file operation: write path outside workspace"
            )

    # ── JARVIS subsystem tool handlers ─────────────────────────────────────

    async def web_automation_tool(self, args: dict[str, Any]) -> ToolResult:
        """Playwright-driven browser automation."""
        if self.web_automation is None:
            return ToolResult("Web automation service is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "launch":
            browser = str(args.get("browser") or "chrome").lower()
            headless = bool(args.get("headless", False))
            return await self.web_automation._launch_browser_async(browser, headless)
        if action == "go_to":
            url = str(args.get("url") or "")
            return await self.web_automation.go_to(url)
        if action in {"click", "click_selector"}:
            selector = str(args.get("selector") or "")
            return await self.web_automation.click_selector(selector)
        if action in {"type", "type_text"}:
            selector = str(args.get("selector") or "")
            text = str(args.get("text") or "")
            return await self.web_automation.type_text(selector, text)
        if action == "form":
            values = dict(args.get("values") or {})
            return await self.web_automation.fill_form(values)
        if action == "smart_click":
            element_text = args.get("target")
            role = args.get("role")
            return await self.web_automation.smart_click(element_text, role)
        if action == "close":
            return await self.web_automation.close_async()
        return ToolResult(f"Unsupported web_automation action: {action}.", ok=False)

    def peripherals_tool(self, args: dict[str, Any]) -> ToolResult:
        """Hardware and OS peripheral control."""
        if self.peripherals is None:
            return ToolResult("Peripheral control is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "volume":
            level = float(args.get("level") or 0.5)
            return self.peripherals.set_volume(level)
        if action == "mute":
            return self.peripherals.toggle_mute()
        if action == "shutdown":
            return self.peripherals.shutdown()
        if action == "restart":
            return self.peripherals.restart()
        if action in {"lock", "lock_screen"}:
            return self.peripherals.lock_screen()
        if action == "wifi":
            return self.peripherals.toggle_wifi()
        if action == "ethernet":
            return self.peripherals.toggle_ethernet()
        return ToolResult(f"Unsupported peripherals action: {action}.", ok=False)

    def messaging_tool(self, args: dict[str, Any]) -> ToolResult:
        """Real-time messaging gateway (WhatsApp, Telegram)."""
        if self.messaging is None:
            return ToolResult("Messaging gateway is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "send":
            platform = str(args.get("platform") or "").lower()
            contact = str(args.get("contact") or "")
            message = str(args.get("message") or "")
            return self.messaging.send_text(platform, contact, message)
        return ToolResult(f"Unsupported messaging action: {action}.", ok=False)

    def gaming_tool(self, args: dict[str, Any]) -> ToolResult:
        """Local gaming client discovery and launch."""
        if self.gaming is None:
            return ToolResult("Gaming client service is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "index":
            return self.gaming.index_installs()
        if action == "launch":
            app_id = str(args.get("app_id") or "")
            return self.gaming.launch_app(app_id)
        if action == "status":
            return self.gaming.update_status()
        return ToolResult(f"Unsupported gaming action: {action}.", ok=False)

    def entertainment_tool(self, args: dict[str, Any]) -> ToolResult:
        """Entertainment discovery and media analysis (YouTube, trends)."""
        if self.entertainment is None:
            return ToolResult("Entertainment service is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action == "summarise":
            url = str(args.get("url") or "")
            return self.entertainment.summarise_video(url)
        if action == "channel":
            channel = str(args.get("channel") or "")
            return self.entertainment.channel_priority(channel)
        if action == "trending":
            region = str(args.get("region") or "GB")
            return self.entertainment.trending(region)
        return ToolResult(f"Unsupported entertainment action: {action}.", ok=False)

    # ── Mark X.7+: Forge self-improvement engine ─────────────────────────────

    async def forge_tool(self, args: dict[str, Any]) -> ToolResult:
        """
        Dynamically forge and activate new tools on the fly.

        Orchestrates: Plan → Code Gen → Sandbox (with self-heal) →
                     Dependency Check → Live Activation.
        """
        if self.forge is None:
            return ToolResult("The Forge capability-forging engine is unavailable.", ok=False)
        action = str(args.get("action") or "forge").lower().strip()

        if action in {"forge", "build", "create_tool"}:
            tool_name = str(args.get("tool_name") or args.get("name") or "").strip()
            tool_plan = str(args.get("tool_plan") or args.get("plan") or "").strip()
            if not tool_name or not tool_plan:
                return ToolResult(
                    "forge requires 'tool_name' and 'tool_plan'. "
                    "Example: forge(tool_name='summarise_video', tool_plan='...')",
                    ok=False,
                )
            return await self.forge.forge_tool(tool_name, tool_plan)

        if action in {"batch", "forge_batch", "create_tools"}:
            tool_specs = args.get("tool_specs") or args.get("specs") or []
            if not isinstance(tool_specs, list):
                return ToolResult("batch requires 'tool_specs' (list of name + plan dicts).", ok=False)
            return await self.forge.forge_batch(tool_specs)

        if action in {"session", "status", "session_status"}:
            session_id = str(args.get("session_id") or "")
            if not session_id:
                return self.forge.list_sessions()
            return self.forge.session_status(session_id)

        return ToolResult(
            f"Unsupported forge action: {action}. Use 'forge' (single tool), "
            "'batch' (multiple), or 'session' (track progress).",
            ok=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TOOL DECLARATIONS (Gemini function-calling schema)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "open_app",
        "description": "Launch a trusted host application, well-known web app (notion, gmail, github), or secure URL.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Application label, executable, path, web-app name, or URL."},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "web_search",
        "description": "Open a secure web search for the supplied query.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "open_news",
        "description": "Open one of the cached briefing news stories in the system browser. Match by topic keyword, headline fragment, or 1-based story index.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Topic or headline fragment, e.g. 'neuralink' or part of the story title."},
                "index": {"type": "INTEGER", "description": "1-based story number from the briefing cache."},
            },
        },
    },
    {
        "name": "close_app",
        "description": "Close a running application by process or window name.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Application or process label, e.g. 'notepad'."},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "window_control",
        "description": "List, focus, minimise, maximise or close desktop windows.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, focus, minimise, maximise, or close."},
                "title":  {"type": "STRING", "description": "Window title fragment to match."},
            },
        },
    },
    {
        "name": "media_control",
        "description": "Control system media playback and volume.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play_pause, next, previous, stop, volume_up, volume_down, or mute."},
                "steps":  {"type": "INTEGER", "description": "Volume steps for volume actions (1-10)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "vision_analyse",
        "description": "ORION's eyes: 'describe' analyses the current screen and attaches the frame; 'ocr' extracts on-screen or image text; 'find_errors' sweeps the desktop for error dialogs and crash text; 'analyse_image' inspects an image file.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "describe, ocr, find_errors, or analyse_image."},
                "path":   {"type": "STRING", "description": "Image path for ocr/analyse_image (omit for the live screen)."},
                "prompt": {"type": "STRING", "description": "Optional focus question."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "outlook_mail",
        "description": "Outlook email control. read_inbox/priority list messages; draft composes (saved, never sent); send_draft transmits ONLY with confirm=true after the user explicitly approves; pending_drafts lists what awaits approval.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "read_inbox, priority, read_email, draft, send_draft, discard_draft, or pending_drafts."},
                "to":        {"type": "STRING", "description": "Recipient address(es) for draft."},
                "subject":   {"type": "STRING", "description": "Subject for draft."},
                "body":      {"type": "STRING", "description": "Body text for draft."},
                "cc":        {"type": "STRING", "description": "Optional CC address(es)."},
                "draft_ref": {"type": "STRING", "description": "Draft reference (e.g. 'draft-1') for send/discard."},
                "confirm":   {"type": "BOOLEAN", "description": "Must be true to actually send — only after explicit user approval."},
                "entry_id":  {"type": "STRING", "description": "Message entry_id from read_inbox for read_email."},
                "limit":     {"type": "INTEGER", "description": "Message count for reads."},
                "unread_only": {"type": "BOOLEAN", "description": "Restrict read_inbox to unread mail."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "notion_workspace",
        "description": "Notion productivity control: list_tasks, create_task, complete_task, upcoming_events (calendar), create_event (scheduling), or projects (project tracking).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list_tasks, create_task, complete_task, upcoming_events, create_event, or projects."},
                "title":  {"type": "STRING", "description": "Task or event title."},
                "due":    {"type": "STRING", "description": "ISO due date for create_task, e.g. 2026-07-03."},
                "notes":  {"type": "STRING", "description": "Optional body notes for create_task."},
                "query":  {"type": "STRING", "description": "Title fragment for complete_task."},
                "start":  {"type": "STRING", "description": "ISO start datetime for create_event."},
                "end":    {"type": "STRING", "description": "Optional ISO end datetime for create_event."},
                "days":   {"type": "INTEGER", "description": "Look-ahead window for upcoming_events."},
                "limit":  {"type": "INTEGER", "description": "Maximum rows returned."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "agent_dispatch",
        "description": "Consult a specialist agent: marketing (digital marketing/growth), coding (software engineering/debugging), design (design & art/UI/UX), fashion (styling), entertainment (film/music/games), research (research, analysis, comparison & evaluation), or auto to route by content.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "agent":   {"type": "STRING", "description": "marketing, coding, design, fashion, entertainment, research, or auto."},
                "request": {"type": "STRING", "description": "The question or task for the specialist."},
                "context": {"type": "STRING", "description": "Optional extra context."},
            },
            "required": ["request"],
        },
    },
    {
        "name": "morning_briefing",
        "description": "Deliver the daily intelligence briefing on demand: AI news, Neuralink, economy, stock market, crypto, calendar, tasks and priority email.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "find_files",
        "description": "Search the user's Desktop, Documents, Downloads, OneDrive and workspace for files or folders by name; optionally open the best match.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "File or folder name fragment."},
                "open":  {"type": "BOOLEAN", "description": "Open the best match with its default application."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "dev_workbench",
        "description": "Software engineering workbench: analyse a repository, read code with line numbers, run allow-listed development commands (python, pytest, node, npm, cargo, go, dotnet, git), or scaffold a new Python project inside the workspace.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "analyse_repo, read_file, run_command, or create_python_project."},
                "path":       {"type": "STRING", "description": "Repository, file, or project path."},
                "command":    {"type": "STRING", "description": "Development command for run_command."},
                "start_line": {"type": "INTEGER", "description": "First line for read_file."},
                "line_count": {"type": "INTEGER", "description": "Line count for read_file (max 400)."},
                "name":       {"type": "STRING", "description": "Project name for create_python_project."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "recall_conversation",
        "description": "Search past conversation history (episodic memory) for what was previously discussed and when.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "What to look for in past conversations."},
                "limit": {"type": "INTEGER", "description": "Maximum entries."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_plan",
        "description": "Run a multi-step plan of tool calls sequentially, verify each step, and return a consolidated report. Use for autonomous multi-part tasks.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "objective": {"type": "STRING", "description": "One-line goal of the plan."},
                "steps": {
                    "type": "ARRAY",
                    "description": "Ordered steps (max 8).",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "tool":      {"type": "STRING", "description": "Tool name to call."},
                            "args_json": {"type": "STRING", "description": "JSON object of arguments for the tool."},
                        },
                        "required": ["tool"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
    {
        "name": "browser_control",
        "description": "Open a secure URL or search in the system browser.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "go_to, open, new_tab, or search."},
                "url":    {"type": "STRING", "description": "HTTP or HTTPS URL."},
                "query":  {"type": "STRING", "description": "Search query."},
            },
        },
    },
    {
        "name": "file_controller",
        "description": "Access ANY file or folder on the user's computer by absolute path (~ and environment variables are expanded): list directories and read_text any file freely — you have full read access. Writes (write_text/append_text) and delete are permitted anywhere EXCEPT ORION's own program files, and should be done on the user's explicit instruction.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, mkdir, write_text, append_text, read_text, delete."},
                "path":   {"type": "STRING", "description": "Any absolute path on the PC (e.g. C:/Users/you/Documents/x.txt), or workspace-relative."},
                "text":   {"type": "STRING", "description": "Text for write operations."},
            },
        },
    },
    {
        "name": "process_file",
        "description": "Inspect ANY local file on the user's computer by absolute path — text, JSON, CSV, TSV, binary, PDF, or image. Full read access; use it to read documents, code, data or images the user points you to.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path":   {"type": "STRING", "description": "Path to the file to scan."},
                "prompt": {"type": "STRING", "description": "Optional focus question for the review."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "save_memory",
        "description": "Commit a durable user fact into the local SQLite FTS5 intelligence matrix.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "description": "Category such as identity, preferences, projects."},
                "key":      {"type": "STRING", "description": "Snake case memory key."},
                "value":    {"type": "STRING", "description": "Fact value."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "query_intelligence",
        "description": "Search local SQLite FTS5 memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Full text search query."},
                "limit": {"type": "INTEGER", "description": "Maximum records."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "capture_screen",
        "description": "Capture the primary monitor as volatile in-memory JPEG bytes (prefer vision_analyse for analysis).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "quality":  {"type": "INTEGER", "description": "JPEG quality 45-95."},
                "max_side": {"type": "INTEGER", "description": "Maximum image side, default 1024."},
            },
        },
    },
    {
        "name": "clipboard_operate",
        "description": "Read or copy text through the native Qt clipboard.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "read or copy."},
                "text":   {"type": "STRING", "description": "Text to copy."},
            },
        },
    },
    {
        "name": "process_governor",
        "description": "Monitor and manage host processes. 'top' lists the biggest CPU or RAM power users (plus GPU utilisation) — use this when the user asks what's using too much power; 'list' shows all; 'terminate' stops a PID or named process; 'restart' stops then relaunches it. Terminate/restart are destructive and require confirm=true (the user's permission).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "top, list, terminate, or restart."},
                "metric":  {"type": "STRING", "description": "For 'top': cpu (default) or ram."},
                "pid":     {"type": "INTEGER", "description": "Process ID to terminate/restart."},
                "name":    {"type": "STRING", "description": "Process/app name to terminate/restart."},
                "confirm": {"type": "BOOLEAN", "description": "Must be true to actually terminate/restart — only after the user explicitly approves."},
            },
        },
    },
    {
        "name": "system_notify",
        "description": "Project a high-priority alert banner into the HUD.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "message":  {"type": "STRING", "description": "Alert text."},
                "priority": {"type": "INTEGER", "description": "Priority from 0 to 5."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "shutdown_orion",
        "description": "Safely disconnect the live session and terminate O.R.I.O.N.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    # ── Mark IX autonomous OS tools ───────────────────────────────────────────
    {
        "name": "desktop_control",
        "description": "Autonomously control the machine: move/click the cursor, type, send hotkeys, and manage windows. To WRITE text into Notepad, an editor or a form, use action 'edit_text' with the target window 'title' — it focuses the window and uses a reliable native/clipboard/keyboard fallback chain (this is the correct way to edit a text file, not type_text). Prefer click_text over raw click. Clicks are visually verified before continuing.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "click_text, click, double_click, right_click, move_cursor, drag, scroll, type_text, edit_text, hotkey, open_app, close_app, focus_window, resize_window, move_window, minimise_window, maximise_window, or list_windows."},
                "text":   {"type": "STRING", "description": "Visible label to click (click_text), or the text to write (edit_text)."},
                "replace": {"type": "BOOLEAN", "description": "For edit_text: replace all existing content (select-all first) instead of appending."},
                "x":      {"type": "INTEGER", "description": "X coordinate (virtual-desktop, or monitor-local with 'monitor')."},
                "y":      {"type": "INTEGER", "description": "Y coordinate."},
                "x1": {"type": "INTEGER"}, "y1": {"type": "INTEGER"},
                "x2": {"type": "INTEGER"}, "y2": {"type": "INTEGER"},
                "monitor": {"type": "INTEGER", "description": "Monitor index for monitor-local coordinates."},
                "amount": {"type": "INTEGER", "description": "Scroll amount (+up/-down)."},
                "text_value": {"type": "STRING"},
                "keys":   {"type": "STRING", "description": "Hotkey combo e.g. 'ctrl+c'."},
                "title":  {"type": "STRING", "description": "Window title fragment for window actions."},
                "app_name": {"type": "STRING", "description": "Application to open/close."},
                "width": {"type": "INTEGER"}, "height": {"type": "INTEGER"},
                "verify": {"type": "BOOLEAN", "description": "Visually confirm the action (default true for clicks)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "vision_verify",
        "description": "Detect on-screen UI structure: 'elements' lists interactive controls (buttons, menus, fields) in the foreground window via the accessibility tree; 'dialogs' finds open dialogs and pop-ups.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "elements, dialogs, describe, ocr, or find_errors."},
                "kinds":  {"type": "STRING", "description": "For elements: all, button, menu, input, link, or dialog."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "web_control",
        "description": "Operate a browser like a human: navigate, tabs, accept/reject cookies, close pop-ups, fill forms, read page contents, download, and complete file dialogs. Uses the accessibility tree + vision to find controls reliably.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "open, navigate, new_tab, close_tab, switch_tab, accept_cookies, reject_cookies, close_popup, fill_form, read_page, download, or file_dialog."},
                "url":    {"type": "STRING", "description": "URL for open/navigate/new_tab."},
                "index":  {"type": "INTEGER", "description": "Tab number for switch_tab."},
                "fields": {"type": "OBJECT", "description": "For fill_form: an object of field-label → value."},
                "submit": {"type": "BOOLEAN", "description": "Submit the form after filling."},
                "path":   {"type": "STRING", "description": "File path for file_dialog (upload/save)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "workspace_control",
        "description": "Persistent workspace awareness: snapshot the desktop, save/restore a named workspace to resume work, track what changed, recall resume context, or set the active project.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "snapshot, save, restore, track_changes, resume_context, or set_project."},
                "name":    {"type": "STRING", "description": "Workspace name for save/restore."},
                "project": {"type": "STRING", "description": "Project name for resume_context/set_project."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "codebase_copilot",
        "description": "Whole-repository engineering: analyse (index + dependency graph + hotspots + cycles), find_symbol, dependencies (impact analysis), or task (refactor/review/bughunt/tests/docs grounded in the live index).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "analyse, find_symbol, dependencies, or task."},
                "path":   {"type": "STRING", "description": "Repository path (defaults to the workspace)."},
                "name":   {"type": "STRING", "description": "Symbol name for find_symbol."},
                "module": {"type": "STRING", "description": "Module for dependency/impact analysis."},
                "task":   {"type": "STRING", "description": "The engineering task for 'task'."},
                "focus":  {"type": "STRING", "description": "Optional focus area or file."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "self_repair",
        "description": "Self-healing runtime. 'status' lists captured incidents; 'propose' drafts a fix diff for review; 'run_tests' runs the verify suite; 'diagnose' runs a full self-diagnostic; 'repair' rewrites the failing source file (confirm=false drafts + validates it compiles; confirm=true backs up the original and applies it — a restart then loads it); 'revert' undoes the last applied repair. Code is only ever changed with explicit confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "status, propose, run_tests, diagnose, repair, or revert."},
                "incident_id": {"type": "STRING", "description": "Incident id (defaults to the latest)."},
                "confirm":     {"type": "BOOLEAN", "description": "For 'repair': true actually writes the fix to the source (after backup)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "diagnostics",
        "description": "Run a full self-diagnostic across ORION: compile every module, import checks, dependency audit, database integrity, config validity, tool registry, component health, resource headroom and permissions. Returns a PASS/WARN/FAIL report.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "cursor_overlay",
        "description": "The visible cursor halo that shows where ORION moves the mouse. 'show'/'hide'/'toggle' control it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"action": {"type": "STRING", "description": "show, hide, or toggle."}},
        },
    },
    {
        "name": "patch_notes",
        "description": "Recount ORION's own recent system updates, patch-notes style. 'latest' (default) gives the newest release; 'all' the full history; or pass a version.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "latest or all."},
                "count":   {"type": "INTEGER", "description": "How many recent releases for 'latest'."},
                "version": {"type": "STRING", "description": "A specific version, e.g. 9.7."},
            },
        },
    },
    {
        "name": "learn",
        "description": "Feed ORION information to learn permanently. Actions: 'learn' distils facts from raw text, a local file (incl. PDF/DOCX) or a URL; 'folder' bulk-ingests an entire directory of documents (PDF/DOCX/Markdown/text/HTML/code) into memory — the way to teach ORION gigabytes of a subject (e.g. neuroscience, cybersecurity, programming); 'recall' retrieves learned facts; 'correct' records an authoritative correction; 'forget' removes previously-learned facts on a topic (built-in knowledge is never removed).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "learn, folder, recall, correct, or forget."},
                "source": {"type": "STRING", "description": "Text to learn, a file path, or a URL."},
                "folder": {"type": "STRING", "description": "Directory to bulk-ingest (for action=folder)."},
                "deep":   {"type": "BOOLEAN", "description": "folder only: distil each file with the model (slower, higher quality). Default false = fast extractive."},
                "topic":  {"type": "STRING", "description": "Optional topic label (or the topic to correct/forget)."},
                "correction": {"type": "STRING", "description": "The authoritative correction text (for action=correct)."},
                "query":  {"type": "STRING", "description": "What to recall (for recall)."},
            },
        },
    },
    {
        "name": "cyber_knowledge",
        "description": "Consult ORION's cybersecurity knowledge base (defensive, educational): CIA triad, threat modelling, OWASP risks, injection/XSS/CSRF/SSRF, cryptography, authentication/MFA, network defence, malware, MITRE ATT&CK, detection/SIEM, incident response, secure development, cloud/container and supply-chain security.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Security topic or question."}},
            "required": ["query"],
        },
    },
    {
        "name": "transcript",
        "description": "The verbatim recording of this conversation. 'export' writes a Markdown transcript of the session; 'path' reports where it is being recorded; 'recall' searches past conversation history.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "export, path, or recall."},
                "query":  {"type": "STRING", "description": "Search text for recall."},
            },
        },
    },
    {
        "name": "programming_knowledge",
        "description": "Consult ORION's extensive software-engineering knowledge base (complexity, data structures, algorithms, concurrency, design patterns, databases, security, testing, languages).",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Programming topic or question."}},
            "required": ["query"],
        },
    },
    {
        "name": "proactive_check",
        "description": "Run the proactive survey now (priority email, task deadlines, today's calendar, repository state) and report items for attention.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "display_info",
        "description": "Report the monitor topology: resolutions, positions, DPI scaling and cursor location.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "refresh": {"type": "BOOLEAN", "description": "Force a topology refresh first."},
            },
        },
    },
    {
        "name": "neuro_knowledge",
        "description": "Retrieve authoritative neuroscience / neural-engineering facts from ORION's resident corpus (neurons, synapses, plasticity, BCIs, EEG/ECoG, Utah array, Neuralink, spike sorting, decoding, neuroprosthetics, DBS). Omit query to list topics.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Topic or question, e.g. 'how does the Utah array work'."},
            },
        },
    },
    {
        "name": "ai_mode",
        "description": "Report ORION's active intelligence mode: cloud-enhanced (MODE A) vs fully offline (MODE B), internet status, and which cloud/local models are available. Use when asked 'are you online', 'what model are you using', or about offline capability.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "knowledge_pack",
        "description": "Consult installed offline knowledge packs (entrepreneurship, dropshipping, tiktok shop, marketing, copywriting, sales psychology, business, coding, AI, personal development). Actions: consult (default, needs query), list, remove (id), expand (id + entries).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "consult, list, remove, or expand."},
                "query":  {"type": "STRING", "description": "Question/topic for consult."},
                "id":     {"type": "STRING", "description": "Pack id for remove/expand."},
            },
        },
    },
    {
        "name": "conversation_recall",
        "description": "Long-term conversation memory. recall answers time-scoped questions ('what did we discuss three weeks ago about TikTok Shop?'); summarise digests recent turns; compress rolls old turns into durable summaries. Works offline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "recall, summarise, or compress."},
                "query":    {"type": "STRING", "description": "The time-scoped question for recall."},
                "turns":    {"type": "INTEGER", "description": "Turns to summarise."},
                "days":     {"type": "INTEGER", "description": "Age threshold for compress."},
            },
        },
    },
    {
        "name": "product_research",
        "description": "Dropshipping product research. score computes a 0-100 Product Opportunity Score from a name/description (virality, demand, competition, margin, shipping, returns, seasonality); validate adds a test plan; competition analyses a niche's saturation; log lists prior research. Works offline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "score, validate, competition, or log."},
                "name":        {"type": "STRING", "description": "Product name."},
                "description": {"type": "STRING", "description": "Product description / notes."},
                "niche":       {"type": "STRING", "description": "Niche for competition analysis."},
            },
        },
    },
    {
        "name": "tiktok_intel",
        "description": "TikTok Shop intelligence: trend reports, product assessments, and a deterministic virality-velocity score from supplied signals (avg_views, creators_posting, buy_intent_comments, new_videos_per_day).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "trend, product, or virality."},
                "niche":   {"type": "STRING", "description": "Niche for a trend report."},
                "product": {"type": "STRING", "description": "Product to assess."},
            },
        },
    },
    {
        "name": "instagram_intel",
        "description": "Instagram commerce intelligence: product discovery for a niche, influencer partnership strategy, or a weekly opportunity report.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "discover, influencer, or weekly."},
                "niche":  {"type": "STRING", "description": "Niche."},
                "brand":  {"type": "STRING", "description": "Brand for influencer strategy."},
            },
        },
    },
    {
        "name": "founder_knowledge",
        "description": "Structured founder/operator profiles (Hormozi, Vaynerchuk, Blakely, Bezos): their strategies, frameworks and lessons, with applied analysis. Actions: list, profile (name), learn (name + question).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "list, profile, or learn."},
                "name":     {"type": "STRING", "description": "Founder name."},
                "question": {"type": "STRING", "description": "What to extract/apply."},
            },
        },
    },
    {
        "name": "business_advisor",
        "description": "Personal business advisor for the user's brand Hausables (home products; TikTok Shop, Instagram, dropshipping). Actions: advise (topic), brand, growth (horizon), store (CRO).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "advise, brand, growth, or store."},
                "topic":   {"type": "STRING", "description": "The advice topic for advise."},
                "horizon": {"type": "STRING", "description": "Time horizon for growth."},
            },
        },
    },
    {
        "name": "commerce_hub",
        "description": "The E-commerce Intelligence Hub: an aggregated snapshot of scored product opportunities, research log and knowledge packs for the active brand.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "community_share",
        "description": "ORION Network: export/import knowledge packs and product research as privacy-controlled bundles (private items excluded, optional anonymisation). Actions: export_pack (id), import_pack (path), export_research, import_research (path), list.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "export_pack, import_pack, export_research, import_research, or list."},
                "id":      {"type": "STRING", "description": "Pack id to export."},
                "path":    {"type": "STRING", "description": "Bundle path to import."},
                "privacy": {"type": "STRING", "description": "public, shared, or private."},
            },
        },
    },
    {
        "name": "research",
        "description": "Autonomous research: 'start' researches a topic on ORION's own for a set number of minutes and writes organised notes into a dated folder (say 'research X for 30 minutes'); 'paper' writes a structured research paper; 'status'/'stop' manage a run.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "start, paper, status, or stop."},
                "topic":   {"type": "STRING", "description": "The research topic."},
                "minutes": {"type": "NUMBER", "description": "How long to research (for start)."},
            },
        },
    },
    {
        "name": "globe",
        "description": "Fly the on-screen 3-D globe to a place and show that region's news and footage. Use when the user asks to see/travel to a location or wants regional news.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"place": {"type": "STRING", "description": "City, country or region."}},
            "required": ["place"],
        },
    },
    {
        "name": "expand_mind",
        "description": "Consult ORION's 50 MB offline study corpus. 'search' returns curated knowledge on a topic; 'status' reports the corpus size.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "search or status."},
                "query":  {"type": "STRING", "description": "Topic to look up."},
            },
        },
    },
    {
        "name": "protocol",
        "description": "JARVIS-style named protocols (macros that run a sequence of actions on one command). 'run' engages a protocol by name (morning, focus, wind_down, situation_report, or a user one); 'list' shows them; 'create' defines a new one from steps; 'delete' removes a user protocol.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "run, list, create, or delete."},
                "name":   {"type": "STRING", "description": "Protocol name."},
                "steps": {
                    "type": "ARRAY",
                    "description": "For create: ordered steps, each an object with 'tool' and 'args'.",
                    "items": {"type": "OBJECT", "properties": {
                        "tool": {"type": "STRING"},
                    }},
                },
                "description": {"type": "STRING", "description": "Optional description for create."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "reminder",
        "description": "Set spoken reminders and alarms. 'add' schedules one — pass a natural phrase (e.g. 'remind me in 20 minutes to check the ad campaign') or structured minutes/at; 'list' shows pending; 'cancel' clears one or all.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "add, list, or cancel."},
                "text":    {"type": "STRING", "description": "What to be reminded of."},
                "minutes": {"type": "NUMBER", "description": "Delay in minutes."},
                "at":      {"type": "STRING", "description": "Absolute time, e.g. '15:00' or '3pm'."},
                "phrase":  {"type": "STRING", "description": "A full natural-language reminder phrase."},
                "id":      {"type": "INTEGER", "description": "Reminder id to cancel."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "sentinel",
        "description": "The ambient system sentinel that watches host health and warns proactively. 'status' gives a situation report (CPU/RAM/disk/battery); 'enable'/'disable' toggle spoken monitoring.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, enable, or disable."},
            },
        },
    },
    {
        "name": "audio_studio",
        "description": "Creative Audio Workspace: index_assets scans the audio_studio/raw folder; process_vocal_take gain-stages (loudness-normalises) a vocal WAV/MP3 and writes a processed stem; export_stem_package collects processed stems into a dated package with a manifest.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "index_assets, process_vocal_take, or export_stem_package."},
                "path":       {"type": "STRING", "description": "Audio file for process_vocal_take (name under raw/ or an absolute path)."},
                "target_dbfs": {"type": "NUMBER", "description": "Target peak level in dBFS (default -3)."},
                "convert_to": {"type": "STRING", "description": "Output format, e.g. wav (ffmpeg needed for others)."},
                "name":       {"type": "STRING", "description": "Package name for export_stem_package."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "literature_vault",
        "description": "Academic intake: ingest_paper deep-reads a PDF/text paper, extracting citations, tables and biophysical mechanisms and seeding them into the KNOWLEDGE memory; query_mechanisms returns mechanism-first findings; generate_citation_summary lists a paper's citations.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "ingest_paper, query_mechanisms, or generate_citation_summary."},
                "path":   {"type": "STRING", "description": "Path to the paper (PDF/txt) for ingest_paper."},
                "title":  {"type": "STRING", "description": "Optional title override for ingest_paper."},
                "query":  {"type": "STRING", "description": "Mechanism/topic to look up for query_mechanisms."},
                "slug":   {"type": "STRING", "description": "Paper slug/title fragment for generate_citation_summary (omit for the latest)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "campaign_pipeline",
        "description": "Creator agency pipeline (offline): get_pipeline_snapshot shows the kanban; create adds a campaign; update_stage moves it (Lead→Negotiation→Contracted→In Production→Scheduled→Published→Paid); log_performance records engagement; schedule_content plans a drop; delete permanently removes a campaign (e.g. 'Hausables x BrandX') and requires confirm=true after the user approves.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "get_pipeline_snapshot, create, update_stage, log_performance, schedule_content, or delete."},
                "confirm":  {"type": "BOOLEAN", "description": "Must be true to actually delete a campaign — only after the user approves."},
                "name":     {"type": "STRING", "description": "Campaign name (for create)."},
                "campaign": {"type": "STRING", "description": "Campaign reference for update_stage/log_performance/schedule_content."},
                "brand":    {"type": "STRING", "description": "Brand partner name."},
                "value":    {"type": "NUMBER", "description": "Deal value, or the metric value for log_performance."},
                "stage":    {"type": "STRING", "description": "Target stage for update_stage."},
                "metric":   {"type": "STRING", "description": "Metric name for log_performance (views, likes, engagement)."},
                "title":    {"type": "STRING", "description": "Content title for schedule_content."},
                "platform": {"type": "STRING", "description": "Platform for schedule_content."},
                "scheduled": {"type": "STRING", "description": "ISO date for schedule_content."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "autoplan",
        "description": "Run a multi-step autonomous plan with self-verification: each step is checked (desktop_control/web_control/vision_verify steps are visually confirmed via screen diff, others by result), retried on failure, and reported honestly.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "objective": {"type": "STRING", "description": "One-line goal of the plan."},
                "max_retries": {"type": "INTEGER", "description": "Retries per step (default 2)."},
                "steps": {
                    "type": "ARRAY",
                    "description": "Ordered steps.",
                    "items": {"type": "OBJECT", "properties": {
                        "tool": {"type": "STRING", "description": "Dispatcher tool name."},
                        "on_fail": {"type": "STRING", "description": "retry, continue, or abort."},
                    }, "required": ["tool"]},
                },
            },
            "required": ["steps"],
        },
    },
    {
        "name": "organise_files",
        "description": "Contextual autonomous file organisation. 'preview' (default) is a DRY RUN reporting what would move; 'apply' sorts a folder (Downloads/Desktop/Documents or a path) into typed sub-folders and writes an undo log; 'undo' reverses the last run. Never overwrites; never touches ORION's own files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "preview, apply, or undo."},
                "folder":   {"type": "STRING", "description": "Downloads, Desktop, Documents, or a path."},
                "by_month": {"type": "BOOLEAN", "description": "Also bucket by modified month."},
            },
        },
    },
    {
        "name": "security_watch",
        "description": "Proactive cybersecurity posture. 'status' reports running processes and externally-listening ports; 'enable'/'disable' toggle the background monitor that warns about new open ports, suspicious processes and external drives.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"action": {"type": "STRING", "description": "status, enable, or disable."}},
        },
    },
    {
        "name": "backup",
        "description": "Back up ORION's settings and memory to a timestamped zip (default under OneDrive so it cloud-syncs). 'backup' creates one; 'list' shows archives; 'restore' unpacks one into a review folder (never overwrites live files).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "backup, list, or restore."},
                "note":    {"type": "STRING", "description": "Optional note stored in the archive."},
                "archive": {"type": "STRING", "description": "Archive name fragment for restore (omit for latest)."},
            },
        },
    },
    {
        "name": "draft_report",
        "description": "Draft a structured professional report (Markdown + self-contained HTML, plus DOCX when python-docx is present) into a dated reports/ folder, with optional pure-SVG bar charts. Narrative is written by the model from the brief.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic":    {"type": "STRING", "description": "Report title/topic."},
                "brief":    {"type": "STRING", "description": "Source notes/data to ground the report."},
                "sections": {"type": "ARRAY", "description": "Section headings.", "items": {"type": "STRING"}},
                "charts": {
                    "type": "ARRAY",
                    "description": "Optional charts, each {title, data:{label:number}}.",
                    "items": {"type": "OBJECT", "properties": {"title": {"type": "STRING"}}},
                },
            },
            "required": ["topic"],
        },
    },
    # ── Mark X.5: AI Operating System layer ──────────────────────────────────
    {
        "name": "document_export",
        "description": "Executive document production (offline): 'brief' compiles a DOCX executive brief; 'deck' assembles a responsive HTML presentation; 'report' exports Markdown+HTML+DOCX; 'history' lists everything exported.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "brief, deck, report, or history."},
                "title":   {"type": "STRING", "description": "Document title."},
                "summary": {"type": "STRING", "description": "Executive summary / body text."},
                "sections": {
                    "type": "ARRAY",
                    "description": "Sections, each {heading, content}.",
                    "items": {"type": "OBJECT", "properties": {
                        "heading": {"type": "STRING"}, "content": {"type": "STRING"}}},
                },
                "slides": {
                    "type": "ARRAY",
                    "description": "Deck slides, each {heading, content} (bullet lines).",
                    "items": {"type": "OBJECT", "properties": {
                        "heading": {"type": "STRING"}, "content": {"type": "STRING"}}},
                },
                "limit":   {"type": "INTEGER", "description": "History rows for 'history'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "proactive_report",
        "description": "Generate and store a scheduled intelligence report on demand: daily_business, weekly_product (product intelligence), or monthly_growth. Reports also generate automatically on schedule.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "kind": {"type": "STRING", "description": "daily_business, weekly_product, or monthly_growth."},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "awareness",
        "description": "The continuous cognitive loop: 'situation' reports what ORION is currently aware of (projects, focus, deadlines, priorities, intents); add_priority/add_task/complete_task/goals maintain the durable cognitive state. Awareness only — never executes actions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "situation, add_priority, add_task, complete_task, or goals."},
                "text":    {"type": "STRING", "description": "Priority text or task title."},
                "title":   {"type": "STRING", "description": "Task title for add_task."},
                "project": {"type": "STRING", "description": "Project the task belongs to."},
                "due":     {"type": "STRING", "description": "ISO due date/time, e.g. 2026-07-05 17:00."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "second_brain",
        "description": "ORION's local knowledge graph (works fully offline): 'recall' answers from stored history; 'timeline' reconstructs event order; 'entity' shows an entity's relationships; 'ingest' records new material onto the graph; 'stats' sizes it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "recall, timeline, entity, ingest, or stats."},
                "query":       {"type": "STRING", "description": "Question, topic, or text to ingest."},
                "name":        {"type": "STRING", "description": "Entity name for 'entity'."},
                "title":       {"type": "STRING", "description": "Title for 'ingest'."},
                "source_type": {"type": "STRING", "description": "conversation, file, project, research, email, meeting…"},
                "limit":       {"type": "INTEGER", "description": "Row limit for timeline."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "executive",
        "description": "JARVIS executive-assistant mode: status (the executive picture), prioritise (urgency-ranked task queue), schedule (task+reminder+Notion), meeting_summary (minutes from a transcript), plan (workflow planning), progress (goal/workflow monitoring), track (put a project under tracking).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "status, prioritise, schedule, meeting_summary, plan, progress, or track."},
                "title":      {"type": "STRING", "description": "Title for schedule / meeting_summary."},
                "when":       {"type": "STRING", "description": "Date/time for schedule, e.g. 15:30 or 2026-07-05 09:00."},
                "transcript": {"type": "STRING", "description": "Meeting transcript/notes for meeting_summary."},
                "objective":  {"type": "STRING", "description": "Objective for plan."},
                "project":    {"type": "STRING", "description": "Project name for track."},
                "notes":      {"type": "STRING", "description": "Optional notes."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "momentum",
        "description": "Shipping coach for finishing projects. 'focus' gives the single next action for the active project plus blockers and a focus block; 'standup' gives a cross-project shipping stand-up (in-progress, do-next, overdue); 'plan' breaks a project/goal into milestones with a definition-of-done and immediate next actions. Use when the user asks what to do next, what to work on, how to finish/ship something, or for a stand-up.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "focus (default), standup, or plan."},
                "project": {"type": "STRING", "description": "Project name (for plan; defaults to the active project)."},
                "goal":    {"type": "STRING", "description": "The goal to plan toward (for plan)."},
            },
        },
    },
    {
        "name": "competitor_intel",
        "description": "Competitor intelligence: 'store' dissects a rival store, 'offer' breaks down their offer stack, 'funnel' maps their sales funnel. Findings persist to long-term memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "store, offer, or funnel."},
                "target": {"type": "STRING", "description": "Competitor/store name."},
                "offer":  {"type": "STRING", "description": "Specific offer to analyse (optional)."},
            },
            "required": ["action", "target"],
        },
    },
    {
        "name": "brand_growth",
        "description": "Brand growth strategist for Hausables: 'strategy' (positioning + channel priorities), 'conversion' (CRO plan), 'positioning' (product positioning), 'retention' (retention systems). Findings persist to long-term memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "strategy, conversion, positioning, or retention."},
                "focus":   {"type": "STRING", "description": "Strategy focus area (optional)."},
                "page":    {"type": "STRING", "description": "Page for conversion analysis (default store)."},
                "product": {"type": "STRING", "description": "Product for positioning."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "audio_devices",
        "description": "Inspect or choose ORION's microphone and speaker. 'list' shows every device with its index; 'status' says which mic and speaker are in use; 'set_output'/'set_input' select a device by index or name fragment (e.g. 'headphones'). Use this when the user says they can't hear ORION or wants a specific mic/speaker.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, status, set_output, or set_input."},
                "device": {"type": "STRING", "description": "Device index or name fragment for set_output/set_input; 'default' to reset."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "geo",
        "description": "Worldwide geospatial intelligence over OpenStreetMap: 'locate' resolves any place (country, region, county, state, city, town, village, district) and flies the globe there; 'nearby' lists every settlement within a radius (e.g. towns within 50 km of Bristol); 'describe' gives the administrative hierarchy, population and coordinates.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "locate, nearby, or describe."},
                "query":     {"type": "STRING", "description": "Place name, e.g. 'Ashford, Kent' or 'Fukuoka'."},
                "radius_km": {"type": "NUMBER", "description": "Search radius in km for 'nearby' (default 50)."},
            },
            "required": ["action", "query"],
        },
    },
    {
        "name": "emotion",
        "description": "ORION's facial emotion engine: 'status' reports the current emotional rendering; 'set' pins an expression (neutral, thinking, listening, speaking, happy, excited, concerned, sad, frustrated, alert, critical); 'auto' returns expression to automatic sentiment-driven control.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, set, or auto."},
                "name":   {"type": "STRING", "description": "Emotion name for 'set'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "web_automation",
        "description": "Playwright-driven browser automation: 'launch' opens a browser (chrome, edge, firefox); 'go_to' navigates to a URL; 'click' finds and clicks an element by CSS selector; 'type' inputs text into a field; 'form' fills multiple fields at once; 'smart_click' clicks by element text or ARIA role; 'close' shuts the browser.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "launch, go_to, click, type, form, smart_click, or close."},
                "url":      {"type": "STRING", "description": "URL for 'go_to'."},
                "selector": {"type": "STRING", "description": "CSS selector for 'click' or 'type'."},
                "text":     {"type": "STRING", "description": "Text to type for 'type'."},
                "values":   {"type": "OBJECT", "description": "Field name→value map for 'form'."},
                "target":   {"type": "STRING", "description": "Element text or ARIA role for 'smart_click'."},
                "browser":  {"type": "STRING", "description": "chrome, edge, or firefox for 'launch' (default chrome)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "peripherals",
        "description": "Hardware and OS control: 'volume' sets master volume (0.0–1.0); 'mute' toggles mute; 'shutdown', 'restart', 'lock' trigger power commands; 'wifi' toggles Wi-Fi; 'ethernet' toggles Ethernet.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "volume, mute, shutdown, restart, lock, wifi, or ethernet."},
                "level":  {"type": "NUMBER", "description": "Volume level (0.0–1.0) for 'volume'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "messaging",
        "description": "Send text alerts over WhatsApp or Telegram: 'send' routes a message (WhatsApp/Telegram) to a contact. Contact is a phone number for WhatsApp (+44 prefix for UK) or Telegram username.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "send."},
                "platform": {"type": "STRING", "description": "whatsapp or telegram."},
                "contact": {"type": "STRING", "description": "Phone number (+44...) for WhatsApp; username for Telegram."},
                "message": {"type": "STRING", "description": "Text message to send."},
            },
            "required": ["action", "platform", "contact", "message"],
        },
    },
    {
        "name": "gaming",
        "description": "Local gaming client control: 'index' lists installed Steam and Epic Games installations; 'launch' starts a game by AppID; 'status' checks update/download progress for installed titles.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "index, launch, or status."},
                "app_id":  {"type": "STRING", "description": "Steam/Epic AppID for 'launch'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "entertainment",
        "description": "YouTube media discovery and analysis: 'summarise' extracts a transcript and summary from a YouTube video; 'channel' looks up a channel's priority/type; 'trending' fetches the trending chart for a region (default GB).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "summarise, channel, or trending."},
                "url":     {"type": "STRING", "description": "YouTube URL for 'summarise'."},
                "channel": {"type": "STRING", "description": "Channel name for 'channel'."},
                "region":  {"type": "STRING", "description": "Region code for 'trending' (default GB)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "forge",
        "description": "Dynamically create, test, and activate new Python tools on the fly — the Forge self-improvement engine. Plan → Code Gen → Sandbox (with self-healing fix loop, up to 3 attempts) → Dependency Install → Live Activation. Actions: 'forge' (single tool), 'batch' (multiple tools), 'session' (track progress).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "forge, batch, or session."},
                "tool_name": {"type": "STRING", "description": "Name of the tool to forge (for 'forge')."},
                "tool_plan": {"type": "STRING", "description": "High-level spec of the tool's purpose (for 'forge'), e.g. 'Analyse sentiment from text using transformers'."},
                "tool_specs": {
                    "type": "ARRAY",
                    "description": "For 'batch': list of {'name': str, 'plan': str} dicts.",
                    "items": {"type": "OBJECT", "properties": {
                        "name": {"type": "STRING"},
                        "plan": {"type": "STRING"},
                    }},
                },
                "session_id": {"type": "STRING", "description": "Session ID to track (for 'session'). Omit to list all sessions."},
            },
            "required": ["action"],
        },
    },
]
