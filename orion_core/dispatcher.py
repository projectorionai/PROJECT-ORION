"""
Tool dispatcher — routes model function-calls (and manual GUI commands) to
the owning service, and declares the Gemini function-calling schema.

This module is the **thin core router**. Following the July 2026 improvement
pass (Priority 2.1), the ~90 tool handlers that once lived here were split by
domain into mixin modules that ``OrionDispatcher`` inherits, so no call site
or ``self.*`` reference changed:

    dispatch_desktop.py      DesktopDispatchMixin      apps, windows, cursor,
                                                       clipboard, processes,
                                                       peripherals, system
    dispatch_vision.py       VisionDispatchMixin       capture, analyse, verify
    dispatch_web.py          WebDispatchMixin          browser, mail, notion,
                                                       MCP, messaging, geo
    dispatch_files.py        FilesDispatchMixin        files, dev workbench,
                                                       self-repair, forge
    dispatch_knowledge.py    KnowledgeDispatchMixin    memory, research, docs
    dispatch_commerce.py     CommerceDispatchMixin     product/market advisory
    dispatch_productivity.py ProductivityDispatchMixin agents, briefing, globe,
                                                       telemetry, planning
    dispatch_schema.py       TOOL_DECLARATIONS         function-calling schema

The core retains construction, the ``handler_table`` routing map, the
``dispatch`` / ``dispatch_chain`` loop, and the cross-cutting helpers. Every
payload is still validated through the security layer before delegation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

if TYPE_CHECKING:          # names used only in annotations (PEP 563 strings)
    # Importing agents here would drag it onto the startup path even though
    # this module never touches it at runtime — the dispatcher is handed an
    # already-built manager. See orion_core/lazy_import.py.
    from .agents import AgentManager, DesktopAgent
from .briefing import MorningBriefingService
from .bus import OrionBus
from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .memory import MemoryAgent
from .notion import NotionService
from .outlook import OutlookService
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line
from .vision import LocalFileIntelligence, VisionAgent, VolatileScreenGrabber



# ── domain dispatch mixins (Priority 2.1 split) ───────────────────────────────
from .dispatch_commerce import CommerceDispatchMixin
from .dispatch_desktop import DesktopDispatchMixin
from .dispatch_files import FilesDispatchMixin
from .dispatch_knowledge import KnowledgeDispatchMixin
from .dispatch_productivity import ProductivityDispatchMixin
from .dispatch_schema import TOOL_DECLARATIONS  # re-exported for existing import sites
from .dispatch_vision import VisionDispatchMixin
from .dispatch_web import WebDispatchMixin


class OrionDispatcher(
    CommerceDispatchMixin,
    DesktopDispatchMixin,
    FilesDispatchMixin,
    KnowledgeDispatchMixin,
    ProductivityDispatchMixin,
    VisionDispatchMixin,
    WebDispatchMixin,
):
    """Thin core router. Tool handlers live in the per-domain mixins above;
    this class owns construction, the handler table and the dispatch loop.
    """

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
        self.ingestion = None              # IngestionEngine (files/folders)
        self.companion = None              # CompanionEngine (continuity ledger)
        self.companion_agent = None        # CompanionAgent (passive observer)
        self.pipeline = None               # AgencyPipelineService
        # Realistic-improvements batch (optional, set post-construction).
        self.plan_executor = None          # PlanExecutor
        self.jobs = None                   # JobManager (background work)
        self.file_organiser = None         # FileOrganiser
        self.security = None               # SecuritySentinel
        self.breach_monitor = None         # BreachMonitor (HIBP)
        self.antivirus = None              # AntivirusMonitor (Windows Defender awareness)
        self.live_camera_window = None     # LiveCameraWindow ("show me what you see")
        self.backup = None                 # BackupManager
        self.reports = None                # ReportDrafter
        self.diagnostics = None            # DiagnosticsEngine
        self.cursor_overlay = None         # CursorOverlay
        # Mark IX Living-Memory batch (optional, set post-construction).
        self.changelog = None              # Changelog (patch notes)
        self.change_tracker = None         # SourceChangeTracker (file-level self-awareness)
        self.speaker_tracker = None        # SpeakerGenderTracker (male/female voice recognition)
        self.learning = None               # LearningService
        self.programming = None            # ProgrammingKnowledgeBase
        self.cyber = None                  # CyberKnowledgeBase
        self.study = None                  # StudyEngine (Mark XXIV spaced-repetition)
        self.focus = None                  # FocusEngine (Mark XXV deep-focus blocks)
        self.finance = None                # FinanceEngine (Mark XXVI money/runway)
        self.decisions = None              # DecisionJournal (Mark XXVI calibration)
        self.wellbeing = None              # WellbeingEngine (Mark XXVI check-ins)
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
        self.last_flight_search = None     # FlightSearch (for follow-up questions)
        self.web_copilot = None            # BrowserCopilot (real visible-browser co-pilot, #11)
        self.peripherals = None            # PeripheralController
        self.messaging = None              # MessagingGateway
        self.social = None                 # SocialAutomationService (real-account TikTok/Instagram)
        self.gestures = None               # GestureEngine (webcam hand-gesture PC control)
        self.security_recon = None         # SecurityReconService (authorized pentest tooling)
        self.chess = None                  # ChessService (Command Deck chess + Stockfish)
        self.gaming = None                 # GamingClientService
        self.entertainment = None          # EntertainmentService
        # Mark X.7+: Forge self-improvement engine (optional, set post-construction).
        self.forge = None                  # ForgeOrchestrationManager (capability forging)
        # MCP host — connects ORION to Model Context Protocol servers (Gmail,
        # Google Calendar, filesystem, …), set post-construction in app.py.
        self.mcp_host = None               # MCPHost
        self.telephony = None              # TelephonyGateway: real calls/SMS
        # Mark XXI, Track E1: a per-INSTANCE copy of the module-level tool
        # schema, so a connected MCP server's tools (and a freshly forged
        # capability — dynamic_loader.py's activate_tool already expected
        # exactly this attribute, but it never existed, so that path was
        # silently dead) can be appended at runtime without mutating the
        # shared module list every other dispatcher/test instance also
        # imports. live_worker.py reads THIS list when building the live
        # session's tool schema, falling back to the static module list
        # when absent (e.g. a bare test dispatcher).
        self.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
        self._mcp_routes: dict[str, tuple[str, str]] = {}   # "mcp__x__y" -> (server, tool)
        self._mcp_schemas: dict[str, dict[str, Any]] = {}    # "mcp__x__y" -> original inputSchema
        # Phase 3 — Personal AI Operating System layer (set post-construction).
        self.executive_core = None         # ExecutiveCore (decision/priority/strategy)
        self.research_director = None      # ResearchDirector (persistent agenda)
        self.evidence = None               # EvidenceEngine (claims + provenance)
        self.skills = None                 # SkillManager
        self.plugins = None                # PluginRegistry (code plugins)
        self.plugin_events = None          # PluginEventBridge (Mark XXVI hooks)
        self.automation = None             # AutomationManager (workflow facade)
        self.creator_intel = None          # CreatorIntelSuite (Creator Studio)
        self.missions = None               # MissionEngine (mission operating model)
        self.avatar = None                 # AvatarController (avatar system)
        self.face_tracker = None           # FaceTracker (webcam head tracking)
        self.briefing_engine = None        # DynamicBriefingEngine
        self.registries = None             # SystemRegistries
        self.reasoning = None              # ReasoningEngine (Track C deliberation)
        self.strategy = None               # StrategyEngine (Track D option search)
        self.perception = None             # PerceptionLoop (Track E continuous vision)
        self.debugger = None               # DebuggerService (real pdb-backed debug sessions)
        # ── the two speaker tools, which sound alike and are not ─────────
        #
        # voiceprint_service  ->  the `voice_speaker_id` tool.
        #     A trained 256-dimension speaker embedding compared against
        #     people who have enrolled. Answers WHICH PERSON is speaking, and
        #     backs the only-my-voice gate.
        #
        # speaker_tracker     ->  the `speaker_id` tool.
        #     A pitch-based guess at whether a voice sounded male or female.
        #     Answers nothing about identity and cannot be made to.
        #
        # This attribute used to be called speaker_id, which made it the exact
        # opposite of the tool of that name.
        self.voiceprint_service = None
        self.navigation_trace = None       # NavigationTrace (UI-targeting attempt log)
        self.pattern_detector = None       # WorkflowPatternDetector (repeated tool sequences)
        self.swarm_view = None             # SwarmDeckView (Track B5 live-activity pulses)
        # Rolling record of recent tool executions for the Command Centre.
        self.recent_tools: deque[dict[str, Any]] = deque(maxlen=60)
        self.active_tools = 0
        # Deterministic policy layer for destructive OS actions (Section 10):
        # OS shutdown/restart/logout/sleep/hibernate are never executed on model
        # text alone — they are armed here and released only by a token-bound
        # human confirmation routed through confirm_system_action().
        from .system_guard import SystemActionGuard
        import platform as _platform
        self.system_guard = SystemActionGuard(
            machine=f"this computer ({_platform.node() or 'host'})",
            logger=self.bus.log.emit,
        )
        # Set by the live worker: coroutine factory that (re)delivers the
        # briefing through the active voice channel. Takes the requested
        # period ("morning" for an overnight catch-up, "general" otherwise).
        self.on_briefing_request: Callable[[str], Awaitable[None]] | None = None

    @property
    def news_articles(self) -> list[dict[str, str]]:
        """Cached briefing stories (owned by the briefing service)."""
        return self.briefing.articles

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

    # ── MCP tools as first-class, namespaced tools (Mark XXI, Track E1) ────────
    #
    # Previously an MCP tool was only reachable through the generic `mcp`
    # dispatcher tool (list a server's tools, then call one by name) — an
    # indirection the model had to already know to use. This makes each
    # connected server's tools appear directly in the schema under
    # "mcp__<server>__<tool>", so they get chosen naturally alongside every
    # native tool. Routing is a small name->{server,tool} lookup, not a
    # rewrite of dispatch()'s own resolution order.

    # The server's schema is arbitrary JSON Schema — the Notion server's is
    # generated from OpenAPI ($defs/$ref, oneOf, const, additionalProperties).
    # It used to be copied through with only the type casing changed, and
    # Gemini's LiveConnectConfig forbids every keyword it does not know, so
    # one such tool failed EVERY Live connect with "35 validation errors" —
    # the "live voice channel is unstable" outage. gemini_schema translates
    # it into the subset ORION's own declarations use; the original is kept
    # so a value the model had to send as JSON text is decoded on the way out.

    @staticmethod
    def _mcp_function_name(prefix: str, tool_name: str) -> str:
        """A Gemini-legal function name ([A-Za-z0-9_.-], <= 64 chars)."""
        from .gemini_schema import valid_function_name
        name = f"{prefix}{tool_name}"
        if valid_function_name(name):
            return name
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
        if len(safe) > 64:
            digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
            safe = f"{safe[:55]}_{digest}"
        return safe

    def _plugin_gate(self, name: str, args: dict[str, Any]) -> ToolResult | None:
        """None when a plugin tool may run now; otherwise the reply to give.

        Plugin tiers used to be enforced only for requests from the phone — a
        'confirm'-tier plugin called by ORION himself simply ran. The line
        drawn here is what the plugin DOES: a confirm-tier plugin that uses the
        network (sends a Discord/Telegram message, fires a webhook, drives
        Home Assistant) acts in the outside world, so it goes behind the same
        human-issued, single-use token as phone calls and MCP spending. A
        confirm-tier plugin that only reads or writes local files, and every
        'allow' plugin, runs as before. 'forbid' is refused. Mutates *args*
        (removes 'confirm') so the plugin never sees the gate's flag.
        ORION_PLUGIN_CONFIRM=0 turns the gate off.
        """
        confirmed = bool(args.pop("confirm", False))
        if os.getenv("ORION_PLUGIN_CONFIRM", "1").strip().lower() in {"0", "false", "no", "off"}:
            return None
        registry = getattr(self, "plugins", None)
        record = None
        if registry is not None:
            try:
                record = registry.get(name)
            except Exception:
                record = None
        if record is None:
            return None
        tier = str(getattr(record, "tier", "") or "").lower()
        if tier == "forbid":
            return ToolResult(f"The '{name}' plugin is set to 'forbid'; I won't run it.",
                              ok=False)
        uses = set(getattr(record, "permissions", ()) or ()) | set(
            getattr(record, "undeclared", ()) or ())
        if tier != "confirm" or "network" not in uses:
            return None
        reason = f"'{name}' acts outside this computer (it uses the network)"
        if not confirmed:
            return ToolResult(f"{reason}. Tell the user exactly what it will send or do; "
                              "if they agree, call again with confirm=true and approve "
                              "the on-screen prompt.", ok=False)
        guard = getattr(self, "system_guard", None)
        if guard is None:
            return None
        from .system_guard import ActionIntent, DecisionKind

        decision = guard.request_confirmation(
            ActionIntent.SEND_MESSAGE, origin=f"plugin:{name}",
            payload={"plugin": name, "arguments": dict(args)})
        if decision.kind is DecisionKind.CONFIRM:
            try:
                self.bus.confirm_action.emit({
                    "token": decision.token, "intent": "plugin_action",
                    "message": f"Run {name}? {reason}."})
            except Exception:
                pass
            return ToolResult(f"{reason}. Approve the on-screen prompt to go ahead.",
                              ok=False)
        if decision.kind is not DecisionKind.EXECUTE:
            return ToolResult(f"{reason}. Not permitted: {decision.message}", ok=False)
        return None

    def resolve_tool_declarations(self, query: str = "", state: Any = None) -> list[dict[str, Any]]:
        """The tool declarations to expose for a turn (Mark XXVI, Phase 2).

        A stable seam for callers: it returns the FULL declaration list unless the
        ``ORION_TOOL_RESOLVER`` flag is set, so wiring it in changes nothing until
        the resolver is deliberately switched on. When active it pre-filters the
        132-tool surface to the turn's most relevant subset. Never called at
        Live-session connect time (which has no query — that path uses capability
        facades); this is for the turn-based reasoning path."""
        declarations = list(getattr(self, "TOOL_DECLARATIONS", None) or TOOL_DECLARATIONS)
        try:
            from . import tool_resolver as tr
            if not tr.resolver_enabled():
                return declarations
            return tr.resolve(query, state, declarations)
        except Exception:
            return declarations          # a resolver fault must never strand a turn

    def _coerce_to_schema(self, name: str, args: dict[str, Any]
                          ) -> tuple[dict[str, Any], str | None]:
        """Normalise *args* to *name*'s declared parameter types (tool_args)."""
        from .tool_args import coerce_args
        declarations = getattr(self, "TOOL_DECLARATIONS", None) or TOOL_DECLARATIONS
        declaration = next((d for d in declarations if d.get("name") == name), None)
        if declaration is None:
            return args, None
        try:
            return coerce_args(args, declaration)
        except Exception:
            return args, None            # normalising must never itself break a call

    def register_mcp_tools(self, server: str, tools: list[dict[str, Any]]) -> int:
        """Append *server*'s tools to self.TOOL_DECLARATIONS under
        "mcp__<server>__<tool>" and route them to mcp_host.call(...).
        Idempotent — re-registering the same server (e.g. a reconnect)
        first drops its previous entries so the list never accumulates
        stale duplicates. Returns how many tools were registered."""
        # The supervisor's reconnect hook passes the CONNECTION (name, conn),
        # not its tool list. Iterating a connection raised TypeError after the
        # server's old tools had already been removed — so every supervised
        # reconnect silently deleted that server's tools for the session.
        if not isinstance(tools, (list, tuple)):
            tools = list(getattr(tools, "tools", None) or [])
        if not hasattr(self, "TOOL_DECLARATIONS"):
            self.TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
        if not hasattr(self, "_mcp_routes"):
            self._mcp_routes = {}
        prefix = f"mcp__{server}__"
        self.TOOL_DECLARATIONS[:] = [
            d for d in self.TOOL_DECLARATIONS if not str(d.get("name", "")).startswith(prefix)
        ]
        if not hasattr(self, "_mcp_schemas"):
            self._mcp_schemas = {}
        for name in list(self._mcp_routes):
            if name.startswith(prefix):
                del self._mcp_routes[name]
                self._mcp_schemas.pop(name, None)
        from .gemini_schema import to_gemini_parameters
        registered = 0
        for tool in tools:
            tool_name = str(tool.get("name") or "").strip()
            if not tool_name:
                continue
            full_name = self._mcp_function_name(prefix, tool_name)
            if full_name in self._mcp_routes:
                continue                 # two tools collapsed onto one legal name
            raw_schema = tool.get("inputSchema")
            params = to_gemini_parameters(raw_schema or {"type": "OBJECT", "properties": {}})
            if isinstance(raw_schema, dict):
                self._mcp_schemas[full_name] = raw_schema
            self.TOOL_DECLARATIONS.append({
                "name": full_name,
                "description": (str(tool.get("description") or "")[:900]
                               or f"'{tool_name}' from the MCP server '{server}'."),
                "parameters": params,
            })
            self._mcp_routes[full_name] = (server, tool_name)
            registered += 1
        return registered

    async def _dispatch_mcp(self, name: str, args: dict[str, Any]) -> ToolResult | None:
        """Route an "mcp__<server>__<tool>" name, or None if *name* isn't
        one (so dispatch() can fall through to its other resolution tiers)."""
        route = getattr(self, "_mcp_routes", {}).get(name)
        if route is None or self.mcp_host is None:
            return None
        server, tool = route
        args = dict(args or {})
        confirmed = bool(args.pop("confirm", False))
        gate = getattr(self, "_mcp_spend_gate", None)
        if gate is not None:
            blocked = gate(server, tool, args, confirmed=confirmed)
            if blocked is not None:
                return blocked
        schema = getattr(self, "_mcp_schemas", {}).get(name)
        if schema is not None:
            from .gemini_schema import decode_json_args
            try:
                args = decode_json_args(args, schema)
            except Exception:
                pass                     # decoding is a courtesy, never a failure
        text = await self.mcp_host.call(server, tool, args)
        from .dispatch_web import mcp_call_failed
        return ToolResult(text, ok=not mcp_call_failed(text))

    def handler_table(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        """The live tool-name → handler routing table (one entry per tool)."""
        return {
            "open_app":           self.open_app,
            "close_app":          self.close_app,
            "web_search":         self.web_search,
            "fetch_url":          self.fetch_url,
            "flight_search":      self.flight_search,
            "open_news":          self.open_news,
            "browser_control":    self.browser_control,
            "window_control":     self.window_control,
            "media_control":      self.media_control,
            "find_files":         self.find_files,
            "dev_workbench":      self.dev_workbench,
            "docker":             self.docker_tool,
            "debugger":           self.debugger_tool,
            "undo":               self.undo_tool,
            "file_controller":    self.file_controller,
            "process_file":       self.process_file,
            "image_processor":    self.process_file,
            "save_memory":        self.save_memory,
            "query_intelligence": self.query_intelligence,
            "recall_conversation": self.recall_conversation,
            "execute_plan":       self.execute_plan,
            "capture_screen":     self.capture_screen,
            "vision_analyse":     self.vision_analyse,
            "sound_sense":        self.sound_sense_tool,
            "outlook_mail":       self.outlook_mail,
            "notion_workspace":   self.notion_workspace,
            "agent_dispatch":     self.agent_dispatch,
            "reason":             self.reason_tool,
            "strategy":           self.strategy_tool,
            "perception":         self.perception_tool,
            "morning_briefing":   self.morning_briefing,
            "clipboard_operate":  self.clipboard_operate,
            "process_governor":   self.process_governor,
            "system_notify":      self.system_notify,
            "shutdown_orion":     self.shutdown_orion,
            "restart_orion":      self.restart_orion,
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
            "max_zoom_in_globe":  self.max_zoom_in_globe_tool,
            "resource_status":    self.resource_status_tool,
            "cleanup_review":     self.cleanup_review_tool,
            "token_usage":        self.token_usage_tool,
            "cyber_curriculum":   self.cyber_curriculum_tool,
            "language_tutor":     self.language_tutor_tool,
            "read_documents":     self.read_documents_tool,
            "privacy_guard":      self.privacy_guard_tool,
            "standing_questions": self.standing_questions_tool,
            "rewind":             self.rewind_tool,
            "catch_up":           self.catch_up_tool,
            "study":              self.study_tool,
            "focus":              self.focus_tool,
            "finance":            self.finance_tool,
            "decision":           self.decision_tool,
            "wellbeing":          self.wellbeing_tool,
            "screen_read":        self.screen_read_tool,
            "pentest_lab":        self.pentest_lab_tool,
            "muscle_memory":      self.muscle_memory_tool,
            "security_recon":     self.security_recon_tool,
            "chess":              self.chess_tool,
            "voice_tone":         self.voice_tone_tool,
            "system_startup":     self.system_startup_tool,
            "interface_control":  self.interface_control,
            "phone_action":       self.phone_action,
            "mcp":                self.mcp_tool,
            "telephony":          self.telephony_tool,
            "expand_mind":        self.expand_mind,
            # ── JARVIS layer ───────────────────────────────────────────────
            "protocol":           self.protocol_tool,
            "reminder":           self.reminder_tool,
            "sentinel":           self.sentinel_tool,
            # ── Studio deck ────────────────────────────────────────────────
            "audio_studio":       self.audio_studio_tool,
            "literature_vault":   self.literature_vault_tool,
            "ingest":             self.ingest_tool,
            "companion":          self.companion_tool,
            "campaign_pipeline":  self.campaign_pipeline_tool,
            # ── Realistic-improvements batch ──────────────────────────────
            "autoplan":           self.autoplan_tool,
            "organise_files":     self.organise_files_tool,
            "security_watch":     self.security_watch_tool,
            "breach_check":       self.breach_check_tool,
            "antivirus":          self.antivirus_tool,
            "backup":             self.backup_tool,
            "draft_report":       self.draft_report_tool,
            "diagnostics":        self.diagnostics_tool,
            "cursor_overlay":     self.cursor_overlay_tool,
            # ── Living Memory batch ────────────────────────────────────────
            "patch_notes":        self.patch_notes_tool,
            "code_changes":       self.code_changes_tool,
            "self_changes":       self.self_changes_tool,
            "speaker_id":         self.speaker_id_tool,
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
            "aviation":           self.aviation_tool,
            "audio_devices":      self.audio_devices_tool,
            "elevenlabs_voice":   self.elevenlabs_voice_tool,
            "voice_speaker_id":   self.voice_speaker_id_tool,
            "navigation_trace":   self.navigation_trace_tool,
            # ── JARVIS subsystems ──────────────────────────────────────────
            "web_automation":     self.web_automation_tool,
            "peripherals":        self.peripherals_tool,
            "messaging":          self.messaging_tool,
            "social_media":       self.social_media_tool,
            "gesture_control":    self.gesture_control,
            "gaming":             self.gaming_tool,
            "entertainment":      self.entertainment_tool,
            # ── Mark X.7+: Forge self-improvement engine ─────────────────────
            "forge":              self.forge_tool,
            "job":                self.job_tool,
            # ── Phase 3: Personal AI Operating System ────────────────────────
            "skill":              self.skill_tool,
            "plugin":             self.plugin_tool,
            "workflow":           self.workflow_tool,
            "creator_intel":      self.creator_intel_tool,
            "briefing":           self.briefing_tool,
            "capabilities":       self.capabilities_tool,
            "find_tool":          lambda args: _gateway().find_tool(self, args),
            "use_tool":           lambda args: _gateway().use_tool(self, args),
            "workflow_patterns":  self.workflow_patterns_tool,
            "mission":            self.mission_tool,
            "avatar":             self.avatar_tool,
        }

    async def dispatch(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        name = SecuritySanitiser.guard_text(str(name or ""), "tool.name")
        args = SecuritySanitiser.guard_payload(dict(args or {}), f"tool.{name}")
        handler = self.handler_table().get(name)
        if handler is None:
            # Forged tools: the ReflectiveModuleLoader registers each verified
            # custom tool's run() into _tool_handlers — route them here so a
            # freshly forged capability is callable the moment it activates.
            forged = getattr(self, "_tool_handlers", {}).get(name)
            if forged is not None:
                args = dict(args or {})
                gate = getattr(self, "_plugin_gate", None)
                blocked = gate(name, args) if gate is not None else None
                if blocked is not None:
                    return blocked
                # Count the call and record any failure against the plugin's
                # health, so 'plugin doctor' reports what a plugin has actually
                # been doing rather than only whether it managed to load.
                registry = getattr(self, "plugins", None)
                if registry is not None:
                    try:
                        registry.record_call(name)
                    except Exception:
                        pass
                try:
                    # Bounded: a hung plugin held the whole turn indefinitely.
                    try:
                        outcome = await asyncio.wait_for(
                            asyncio.to_thread(forged, **dict(args or {})),
                            timeout=float(os.getenv("ORION_FORGED_TOOL_TIMEOUT_S", "120")))
                    except asyncio.TimeoutError:
                        if registry is not None:
                            try:
                                registry.mark_loaded(name, True, "timed out")
                            except Exception:
                                pass
                        return ToolResult(f"'{name}' did not finish in time and was "
                                          "abandoned; it may still be running.", ok=False)
                    if asyncio.iscoroutine(outcome):
                        outcome = await outcome
                    from .plugin_results import plugin_result
                    result = plugin_result(outcome)
                    if registry is not None and not result.ok:
                        try:
                            registry.mark_loaded(name, True, first_line(result.text, 200))
                        except Exception:
                            pass
                    return result
                except Exception as exc:
                    detail = first_line(exc, 200)
                    if registry is not None:
                        try:
                            registry.mark_loaded(name, True, detail)
                        except Exception:
                            pass
                    return ToolResult(
                        f"Forged tool '{name}' failed: {detail}", ok=False)
            # MCP tools registered as first-class "mcp__<server>__<tool>"
            # names (Track E1) — checked after native/forged tools so an
            # MCP server can never shadow a real ORION tool by coincidence.
            mcp_result = await self._dispatch_mcp(name, dict(args or {}))
            if mcp_result is not None:
                return mcp_result
            return ToolResult(f"Unknown dispatch target: {name}.", ok=False)
        # Normalise against the schema the model was given — null, numeric
        # strings, and above all string booleans: bool("false") is True, and
        # handlers gate confirm/consent/submit on bool(args.get(...)). Only
        # native tools: forged and MCP tools carry their own schemas.
        args, arg_error = self._coerce_to_schema(name, args)
        if arg_error:
            return ToolResult(f"{name}: {arg_error}", ok=False)
        started = time.perf_counter()
        self.active_tools += 1
        # A bridge is opened BEFORE the handler runs, not after it. The packet
        # is then visibly travelling — and then visibly WAITING on whatever it
        # reached — for exactly as long as the real work takes, which is the
        # only place in the interface that shows latency as duration rather
        # than as a number after the fact.
        flight = self._swarm_launch(name, args)
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
            # Feed the latency tracker from the timing that already exists here,
            # so diagnostics(action='latency') can report p50/p95 per tool
            # without measuring anything twice. recent_tools is a fixed-shape
            # ring with two GUI readers; this is the percentile view.
            try:
                from .latency import TRACKER
                TRACKER.record(f"tool:{name}", elapsed_ms,
                               "" if ok else "failed")
            except Exception:
                pass
            # Separate from recent_tools, which has two live GUI readers and a
            # fixed shape: the detector needs an epoch timestamp and an
            # argument-key signature that recent_tools deliberately doesn't
            # carry. Pure in-memory append; never allowed to break a dispatch.
            detector = getattr(self, "pattern_detector", None)
            if detector is not None:
                try:
                    detector.record(name, args, ok)
                except Exception:
                    pass
            swarm = getattr(self, "swarm_view", None)
            if swarm is not None:
                try:
                    # tool/args are passed through so the swarm can also
                    # attribute this call to the EDGE it travelled (Mark XXII,
                    # Phase 3) rather than only flashing the endpoint node.
                    target = self._swarm_pulse_target(name, args)
                    try:
                        # The Brain page draws the call's real duration.
                        swarm.pulse("core", target, tool=name, args=args,
                                    flight=flight, ok=ok, elapsed_ms=elapsed_ms)
                    except TypeError:
                        # A view that predates it must still resolve the
                        # call's bridge, so it gets the original arguments.
                        swarm.pulse("core", target, tool=name, args=args,
                                    flight=flight, ok=ok)
                except Exception:
                    pass
            if self.telemetry is not None:
                self.telemetry.metrics.observe(f"tool.{name}.ms", elapsed_ms)
                # Aggregate + per-tool usage counters (Priority 3.4 audit).
                self.telemetry.record_tool_call(name, ok)

    def _swarm_launch(self, name: str, args: dict[str, Any]) -> Any:
        """Open an in-flight packet for this dispatch, if the swarm is up.

        Returns an opaque handle (or None) that the finally block hands back
        to resolve it. Never allowed to affect the dispatch: a visualisation
        that could fail a tool call would be worse than no visualisation."""
        swarm = getattr(self, "swarm_view", None)
        if swarm is None:
            return None
        try:
            return swarm.launch_bridge(name, args)
        except Exception:
            return None

    @staticmethod
    def _swarm_pulse_target(name: str, args: dict[str, Any]) -> str:
        """Best-effort mapping from a dispatched tool call to the swarm
        node it most represents (Track B5's live-activity pulse). An
        unmatched tool just pulses the core node against itself — the JS
        side already no-ops a flash() on an id with no live mesh, so a
        wrong guess here is silent, never an error."""
        if name.startswith("mcp__"):
            parts = name.split("__", 2)
            if len(parts) >= 2 and parts[1]:
                return f"mcp:{parts[1]}"
        if name in ("agent_dispatch", "reason") and args.get("agent"):
            return f"agent:{args['agent']}"
        if name == "workflow" and args.get("name"):
            return f"workflow:{args['name']}"
        return "core"

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


__all__ = ["OrionDispatcher", "TOOL_DECLARATIONS"]


def _gateway() -> Any:
    """The Live tool gateway module (find_tool / use_tool), imported lazily."""
    from . import tool_gateway
    return tool_gateway
