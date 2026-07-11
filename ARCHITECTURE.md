# O.R.I.O.N. Mark IX — Architecture

**Open Resolution Intelligence Overt Network** — a modular, autonomous AI
operating system. Launch with `python orion.py` exactly as before; the Mark VII
single-file build is archived at `legacy/orion_mark7_monolith.py`.

> Mark IX adds a full **autonomous control stack** (desktop, vision-verified
> automation, web), an **event-driven audio state machine**, a **structured
> telemetry core**, **workspace persistence**, a **developer copilot**,
> **self-healing** and **proactive intelligence**, plus a third window — the
> **Command Centre**. See the "Mark IX" section at the end for the phase map.
> The Mark VIII sections below still describe the unchanged foundation.

---

## Layout

```
orion.py                     Thin launcher (dependency gate → orion_core.app)
orion_core/
├── constants.py             App constants, paths, palette, frozen VOICE_PROFILE
├── utils.py                 Shared helpers (no internal dependencies)
├── security.py              SecuritySanitiser — regex/AST firewall for all OS payloads
├── data.py                  ToolResult (Qt-free shared data structures)
├── bus.py                   OrionBus — Qt signal hub decoupling every subsystem
├── memory.py                OrionMemoryMatrix (SQLite FTS5) + MemoryAgent
├── audio.py                 Capture, VAD, Vosk STT, playback, SpeechQueueManager
├── vision.py                Screen grabber, file intelligence, VisionAgent
├── providers.py             Provider profiles, config I/O, ProviderRouter
├── outlook.py               OutlookService — COM bridge, approval-gated send
├── notion.py                NotionService — REST tasks/calendar/projects
├── agents.py                BaseAgent, AgentManager, DesktopAgent, 5 specialists
├── briefing.py              MorningBriefingService (concurrent multi-feed)
├── dispatcher.py            OrionDispatcher + Gemini TOOL_DECLARATIONS
├── live_worker.py           GenAILiveWorker — Gemini Live session + offline loop
├── remote.py                RemoteGateway — opt-in browser/mobile uplink
├── app.py                   Composition root (all wiring happens here, only here)
└── gui/
    ├── style.py             Shared stylesheet
    ├── widgets.py           MetricBar, MiniOrb, HolographicToggle, ApiKeyDialog
    ├── hud.py               CentralHud — Liquid Vector Orb
    ├── views.py             HUD / Log / Memory / Telemetry stacked views
    ├── core_window.py       ORION Core Window (+ fullscreen & overlay modes)
    └── dashboard.py         Widget Dashboard Window
```

Dependency direction is strictly downward: `gui` and `app` know the services;
services know the bus, security and constants; nothing imports upward. All
cross-subsystem communication rides `OrionBus` Qt signals — audio threads,
agents and integrations never touch widgets.

---

## Voice system (objective 1)

- **Permanent voice** — `constants.VOICE_PROFILE` is a *frozen* dataclass.
  Gemini Live is locked to the professional male "Charon" voice;
  the local pyttsx3/SAPI fallback selects a male voice once at startup from a
  deterministic search order (Ryan → George → David → …) and never re-selects.
  The PowerShell fallback applies a `Male` voice hint. Nothing at runtime can
  mutate the profile (enforced by the dataclass, verified by test).
- **SpeechQueueManager** (`audio.py`) — the single authority on
  "is ORION speaking?". Both channels feed through it: native PCM chunks into
  the unbounded `AudioPlaybackThread` queue, fallback utterances into the
  FIFO `SpeechSynthesiser` queue. A monitor thread publishes `bus.speaking`
  transitions (the VOICE ● LED in both windows) and a `state_cb` used by the
  live worker.
- **No cut-offs** — playback queue is unbounded (dropped chunks are what
  time-compressed speech in older builds); `is_active()` extends SPEAKING
  through a device-buffer tail (`PLAYBACK_TAIL_SECONDS`) so the final syllable
  is never clipped by a state flip; `turn_complete` no longer forces
  LISTENING while audio is still draining.
- **Speak-then-listen (half-duplex)** — while the queue manager reports
  active output, `AudioGateThread` forwards nothing to the live channel and
  feeds nothing to local recognition; on each speaking↔listening transition
  the raw microphone queue is drained so stale audio (including ORION's own
  voice) can never masquerade as a user turn. Legacy voice barge-in is
  available with `ORION_ALLOW_BARGE_IN=1`.

## Agents (objectives 2, 6)

- **DesktopAgent** — owns host control: Start-Menu app index, allow-listed
  executables, well-known web apps (Notion, Gmail, GitHub, Outlook Web),
  Win32 window management, media keys, process termination. The dispatcher
  delegates `open_app`/`close_app`/`window_control`/`media_control` here.
- **AgentManager** — registry + dynamic router. Scores requests against each
  specialist's keyword profile; `agent_dispatch` tool exposes it to the live
  model, and the dashboard's SPECIALIST AGENTS panel exposes it to the user.
- **Specialists** — Digital Marketing, Coding, Design & Art, Fashion,
  Entertainment. Each is a persona over `ProviderRouter.generate_text(...,
  system_extra=persona)`. With no text provider configured they return their
  specialist *brief* so the live model can adopt the role directly.

## Vision (objective 3)

`VisionAgent` (`vision.py`) — `vision_analyse` tool:
- `describe` — screen capture + structural analysis + OCR excerpt + attached
  JPEG frame for the multimodal channel;
- `ocr` — pytesseract text extraction from screen or image file (optional
  dependency; degrades to multimodal reading with an install hint);
- `find_errors` — desktop error sweep: Win32 window-title scan + OCR keyword
  heuristics + attached frame for visual confirmation;
- `analyse_image` — file analysis via `LocalFileIntelligence` + OCR.

## Dual-window GUI, fullscreen & overlay (objectives 4, 5)

- **ORION Core Window** — conversation console, HUD orb, voice-activity LED,
  system health (CPU/RAM/network), state indicators, memory browser,
  telemetry. Memory is injected at construction (the Mark VII dummy-memory
  workaround is gone).
- **Widget Dashboard Window** — SPECIALIST AGENTS, OUTLOOK COMMAND (inbox,
  priority mail, drafts-awaiting-approval with confirmed SEND), NOTION
  WORKSPACE (tasks/agenda/quick-add) and MORNING BRIEFING panels. Panels
  render from `bus.dashboard_event` — services publish, panels subscribe.
- **Modes** — `F11` fullscreen; `Ctrl+Shift+O` overlay (compact frameless
  always-on-top orb, drag to move, Esc to exit); `Ctrl+D` toggles the
  dashboard; the floating ORION pill toggles the core window.

## Memory (objective 7)

`MemoryAgent` fronts two horizons: a volatile session layer (rolling turn
window + pinned session notes) and the persistent SQLite FTS5 matrix (facts +
episodic history). `prompt_context()` merges both into every system
instruction, so the model always sees the immediate conversation *and*
long-term knowledge. The full matrix API is forwarded, so the agent is a
drop-in replacement anywhere the matrix was used.

## Outlook (objective 8)

`OutlookService` — pywin32 COM bridge (each call in its own COM apartment via
`asyncio.to_thread`). Reading: inbox digests, priority (unread/high-importance)
mail, full bodies for summarisation. Writing: `create_draft` saves a real
Outlook draft under a speakable reference (`draft-1`); **transmission requires
`send_draft(confirm=True)`** — reached only after explicit user approval by
voice or the dashboard's confirmation dialog. Nothing is ever auto-sent.

## Notion (objective 9)

`NotionService` — official REST API, token + database IDs from
`config/api_keys.json → integrations.notion` (env overrides:
`ORION_NOTION_TOKEN`, `ORION_NOTION_TASKS_DB`, `ORION_NOTION_CALENDAR_DB`,
`ORION_NOTION_PROJECTS_DB`). Schema-tolerant: discovers each database's
title/status/date properties at call time. Tasks (list/create/complete),
calendar (upcoming events, event creation), project overview.

## Morning Briefing (objective 10)

`MorningBriefingService` — AI news, Neuralink, economy (Google News RSS),
stock indices (Yahoo Finance: S&P 500, Nasdaq, FTSE 100), crypto (CoinGecko),
plus Notion calendar + tasks and Outlook priority mail when configured. All
feeds fetched **concurrently**; any feed may fail without sinking the
briefing. Delivered by voice at startup, on demand via the `morning_briefing`
tool ("give me my briefing") or the dashboard button; headline sources are
cached so `open_news` opens the exact story that was read out.

## Concurrency & latency (objective 12)

- PortAudio callbacks only copy bytes; VAD/STT/queue work happens on
  `AudioGateThread`; playback on its own thread; TTS on its own thread.
- Every blocking operation (PIL, OCR, CSV sniffing, COM, subprocess, screen
  grabs) runs through `asyncio.to_thread` — the qasync GUI loop never blocks.
- Briefing feeds run concurrently under one `ClientSession`.
- State signals emit only on change; the HUD throttles to 4 fps when hidden;
  the static HUD background is cached in a QPixmap.
- Live-channel (websocket) hygiene: bounded mic queue with drop-oldest,
  session resumption handles, model rotation on 404, provider cooldowns
  scaled to failure class (quota 300 s, auth 1800 s, other 45 s).

## Configuration

`config/api_keys.json` — providers (unchanged schema, now `mark_viii`),
`provider_order`, and the new `integrations` section (preserved verbatim on
every read/merge/write). Environment switches:
`ORION_WAKE_MODE`, `ORION_ALLOW_BARGE_IN`, `ORION_REMOTE_ACCESS`,
`ORION_REMOTE_PORT`, `ORION_VOSK_MODEL`, plus the provider/Notion keys above.

## Dependencies

Required: `PyQt6 qasync aiohttp sounddevice google-genai pillow mss psutil`
Optional: `pyttsx3` (local voice), `pywin32` (Outlook), `pytesseract` +
Tesseract (local OCR), `vosk` (wake word / offline STT), `pypdf` (PDF text),
`torch` + `silero-vad` (neural VAD). Every optional dependency degrades
gracefully with an actionable log line.

---

# Mark IX — Autonomous AI Operating System

Mark IX is additive: every Mark VIII module is intact; the following modules and
wiring layer autonomy, observability and resilience on top.

## New modules

```
orion_core/
├── telemetry.py       Telemetry facade: StructuredLogger + MetricsRegistry + HealthRegistry
├── audio_state.py     AudioStateMachine — event-driven speaking-state authority
├── display.py         DisplayTopologyManager — monitors, DPI, coordinate translation
├── control.py         AutonomousControlLayer — cursor/keyboard/window control
├── verification.py    VisualVerificationEngine — act→capture→verify→retry
├── web.py             WebController — hybrid (accessibility + vision + keyboard) browser control
├── workspace.py       WorkspaceManager — snapshot/save/restore/track the desktop
├── copilot.py         ProjectIndexer, DependencyMapper, SystemGraph, CodebaseMemory, DeveloperCopilot
├── selfrepair.py      SelfRepairAgent — capture → analyse → propose patch (approval-gated)
├── proactive.py       ProactiveIntelligence — background survey of email/tasks/calendar/repos
└── gui/command_centre.py   CommandCentreWindow — the persistent operating dashboard (window 3)
```

## Phase → implementation map

| Phase | Deliverable | Where |
|------|-------------|-------|
| 1  | Voice reconstruction / audio state machine | `audio_state.py`, `audio.py` (rebuilt playback + TTS), `live_worker.py` |
| 2  | True desktop autonomy | `control.py` (+ `desktop_control` tool) |
| 3  | Vision-guided automation + Visual Verification Engine | `vision.py` (UIA/region/array), `verification.py` (+ `vision_verify`) |
| 4  | Advanced web control | `web.py` (+ `web_control`) |
| 5  | Workspace memory | `workspace.py` (+ `workspace_control`) |
| 6  | Self-healing runtime | `selfrepair.py` (+ `self_repair`) |
| 7  | Developer copilot | `copilot.py` (+ `codebase_copilot`) |
| 8  | Proactive intelligence | `proactive.py` (+ `proactive_check`) |
| 9  | Multi-monitor intelligence | `display.py` (+ `display_info`) |
| 10 | Command Centre | `gui/command_centre.py` |
| 11 | Memory architecture (7 tiers) | `memory.py` (`MemoryTier`, `remember`/`recall`/`resume_context`) |
| 12 | Production requirements | `telemetry.py` + type hints/async/health throughout |

## Phase 1 — audio state machine (root-cause fixes)

The Mark VIII review found the "random pauses / clipped audio" came from a
polled speaking-state (80 ms lag, two disagreeing truths), an empty-queue
TOCTOU on turn-completion, event-loop-blocking SQLite writes, and a
cross-thread pyttsx3 `stop()`. Mark IX fixes each at the root:

- **`AudioStateMachine`** is the single, event-driven truth. The playback and
  TTS threads push `native_started/stopped` and `tts_started/stopped`; the
  machine sets one atomic flag that both the microphone gate and the HUD read,
  and fires `bus.speaking` on the transition — **zero poll latency, no
  desync.** `SpeechQueueManager` no longer runs a monitor thread.
- **Deterministic buffering**: `AudioPlaybackThread` prebuffers a few chunks
  before the first write (no cold-start clip) and owns its drain **tail** — it
  waits for more audio itself before declaring the source stopped, so there is
  no external empty-queue race. The queue stays unbounded (never drops model
  audio) with a high-water gauge.
- **No event-loop blocking**: the worker persists episodes via
  `asyncio.to_thread` (`_persist_episode`), so SQLite writes never stall the
  receive loop; `mss` is cached per thread.
- **Safe interrupt**: TTS speaks sentence-by-sentence and honours the interrupt
  flag between sentences — the cross-thread `engine.stop()` hazard is gone, and
  the voice stays locked to `VOICE_PROFILE`.
- **Telemetry**: queue depth, playback write latency (p50/p95), speech state and
  active-stream count are published to the Command Centre.

## Autonomy stack (Phases 2, 3, 9)

`DisplayTopologyManager` makes the process per-monitor-DPI-aware and translates
virtual ↔ monitor-local ↔ fractional coordinates, so clicks land correctly on
any monitor at any scale. `AutonomousControlLayer` drives cursor/keyboard
(pyautogui + `keyboard` SendInput Unicode) and windows (pygetwindow), clamping
every coordinate and recording a region + expectation per action. The
`VisualVerificationEngine` is the only place control meets vision: it captures
before/after (cv2 pixel-diff), confirms the screen actually reacted, and on
failure **re-locates the target from the live UIA tree** (self-correcting
coordinates) before retrying. Prefer `desktop_control` action `click_text` —
it finds a control by visible label and verifies it.

Safety: autonomy honours an `ORION_AUTONOMY` kill-switch and a Command Centre
toggle; typed text passes the SecuritySanitiser; pyautogui FAILSAFE (screen
corner) is a hard physical abort.

## Persistence, resilience, proactivity (Phases 5, 6, 8, 11)

- **Memory tiers**: `MemoryTier` formalises short-term, session, conversation,
  long-term, knowledge, project and workspace horizons; `resume_context()`
  assembles project facts + last workspace + recent conversation so ORION picks
  up where it left off.
- **WorkspaceManager** snapshots windows/tabs/documents/dev-sessions, persists
  them through the WORKSPACE tier, restores by relaunching apps, and diffs live
  state for change tracking.
- **SelfRepairAgent** installs a `sys.excepthook` + asyncio exception handler,
  captures the failing module/line + logs, and — on request — proposes a patch
  (diff + rationale) saved under `config/self_repair/` for review. **It never
  auto-applies** and never mutates ORION's source in place.
- **ProactiveIntelligence** runs one low-priority background survey (email,
  Notion deadlines/calendar, git status), de-duplicated with a cooldown, and
  surfaces findings to the Command Centre and HUD without ever speaking
  unprompted (`proactive_check` returns them on demand).

## Command Centre (Phase 10)

`CommandCentreWindow` (window 3, `Ctrl+Shift+C`) pulls the telemetry snapshot on
a 1 s timer and renders system metrics (CPU/RAM/GPU/network), audio state,
memory tiers, workspace, component health, the active-tool queue with per-tool
timings, and the live structured log — all read-only, never on a hot path.

## New tools

`desktop_control`, `vision_verify`, `web_control`, `workspace_control`,
`codebase_copilot`, `self_repair`, `proactive_check`, `display_info` — 33 tools
total. Each dispatch is timed and recorded into the tool-execution telemetry.

## New / notable dependencies

Required as before, plus already-present: `pyautogui`, `pygetwindow`,
`pywinauto` (UIA element detection), `screeninfo`, `keyboard`, `opencv-python`
(cv2), `numpy`. All degrade gracefully. Optional: `nvidia-smi` for GPU telemetry.

## Environment switches (additions)

`ORION_AUTONOMY` (default on) gates the control layer; everything from Mark VIII
still applies.

---

# Mark IX+ — Neuroscience expertise, offline brain, pause control

Three further enhancements layered on Mark IX.

## Resident neuroscience / neural-engineering expertise (`knowledge.py`)

`NeuroKnowledgeBase` holds a curated, mechanism-first corpus (25 entries:
neurons, glia, action potentials, synapses, neurotransmitters, plasticity,
cortex, hippocampus, Hodgkin–Huxley, integrate-and-fire, cable theory, BCIs,
EEG/ECoG, Utah array, Neuralink, spike sorting, LFP, decoding,
neuroprosthetics, DBS, connectomics, fMRI, stimulation). At startup it is
**seeded idempotently into the persistent KNOWLEDGE memory tier**, so it
surfaces in `prompt_context` for the live model *and* is queryable offline.
Its `PERSONA_BOOST` reinforces the domain persona in the system instruction,
and the `neuro_knowledge` tool exposes authoritative retrieval to the model.

## Offline conversational brain (`local_brain.py`)

`LocalBrain` is the safety net so ORION is **never a mute tool-runner when the
cloud API quota is exhausted**. With zero API calls it: greets and makes small
talk in ORION's register; answers time/date/system/weather (from cached
readings); does arithmetic; stores and recalls memory; discusses neuroscience
from the corpus; and routes "help me do X" ("open …", "search …", "what's on
my screen", "find my …", "save my workspace", "my tasks/email") straight to the
real local tool dispatcher and reports the outcome. The worker's
`_submit_text_fallback` now tries cloud/local text providers first and, when
none are available *or they fail*, hands the turn to the LocalBrain — spoken
through the offline local voice. If a local LLM (LM Studio/Ollama) is running it
is used for open-ended chat, but nothing depends on it.

## Smooth speech + pause/resume (`audio.py`, `live_worker.py`, core window)

- **Gapless local voice** — the local TTS speaks each reply in a single
  synthesis pass, removing the inter-sentence gaps; the native Gemini path was
  already smoothed in Phase 1. Prompt interruption for a *deliberate* pause
  uses `engine.stop()` (a rare, user-initiated event, not per-chunk polling).
- **Pause control** — a header **PAUSE/RESUME button** (`Ctrl+Space`) and the
  spoken word **"pause"** put ORION into a silent, listening-only PAUSED state
  (speech interrupted, microphone forwarding gated, native audio dropped). He
  stays quiet until you **"zone back in"**: saying **"resume"/"continue"**, the
  **wake word "Orion"**, or typing a command brings him back. Resume words are
  in `constants.RESUME_WORDS`. The PAUSED state holds on the HUD/state indicator
  and is broadcast on `bus.paused` to both windows.

Tool count is now **34**.

---

## Dual-Mode Offline Intelligence (MODE A / MODE B)

ORION now runs cloud-enhanced **or fully offline**, switching automatically.

### New modules
```
connectivity.py         ConnectivityMonitor — TCP probe → auto MODE A/B switch
local_models.py         OllamaManager (auto-discovers Ollama + models) + AIModeInfo
knowledge_packs.py      KnowledgePackManager + 10 built-in packs (offline)
conversation_memory.py  Summariser / Compression / Retrieval / SessionRecall
commerce.py             ProductOpportunityScore + 5 entrepreneurial agents
community.py            CommunityHub (export/import) + EcommerceHub (Phase 8)
speech_offline.py       OfflineTranscriber (faster-whisper → whisper → vosk)
```

### Phase 1 — dual-mode provider router
- `ConnectivityMonitor` probes port 443 on anycast hosts, caches for 6 s,
  refreshes every 15 s, and flips ORION between **MODE A (cloud-enhanced)** and
  **MODE B (fully offline)** with a bus banner.
- `OllamaManager` auto-detects a running Ollama server, lists pulled models,
  and enables the `local_ollama` provider with the strongest model
  (preference: Qwen → Llama → DeepSeek → Mistral → Gemma). **Verified doing
  real local inference** with `llama3.1`.
- The `ProviderRouter` is now mode-aware: `is_online()`, `current_mode()`,
  `local_text_profiles()`, `select_text_profiles()` (orders cloud vs local by
  connectivity, `ORION_PREFER`, and task complexity), and `generate_text_offline()`.
  Env: `ORION_MODE` (auto/cloud/offline), `ORION_PREFER` (local/cloud),
  `ORION_OLLAMA_HOST`. Cloud failures always fall through to local — ORION is
  never mute.

### Phase 2 — offline voice
- Output is already offline (pyttsx3/SAPI, frozen male `VOICE_PROFILE`).
- `OfflineTranscriber` adds offline dictation, preferring faster-whisper →
  openai-whisper → vosk (openai-whisper is installed; vosk is not). Piper/Coqui
  documented as optional higher-quality male TTS engines.

### Phase 3 — Knowledge Packs
Ten installable, offline-consultable packs (entrepreneurship, dropshipping,
tiktok shop, marketing, copywriting, sales psychology, business, coding, AI,
personal development), stored as JSON in `config/knowledge_packs/` and indexed
into the KNOWLEDGE tier. Install / update / remove / expand / consult via the
`knowledge_pack` tool; used to ground the LLM in MODE B.

### Phase 4 — Conversation memory
`ConversationMemoryEngine`: model-written-or-extractive summaries, compression
of old turns into LONG_TERM, context retrieval, and **time-scoped recall** —
"what did we discuss three weeks ago about TikTok Shop?" resolves the window
(word-numbers included) and answers offline. Tool: `conversation_recall`.

### Phases 5-7, 9, 11 — entrepreneurial agents
- **ProductOpportunityScore** — deterministic 0-100 over virality, demand,
  competition, margin, shipping, returns, seasonality (risk metrics inverted),
  inferable from a plain description; works fully offline.
- **DropshippingResearchAgent** (`product_research`), **TikTokShopAgent**
  (`tiktok_intel`, incl. virality-velocity), **InstagramCommerceAgent**
  (`instagram_intel`), **FounderKnowledgeAgent** (`founder_knowledge`),
  **BusinessAdvisorAgent** (`business_advisor`) — the last understands the
  user's brand **Hausables** (home products). LLM advice grounds in the
  knowledge packs and prior product research; deterministic paths run offline.

### Phases 8, 10 — Hub + Community
- `EcommerceHub` (`commerce_hub`) aggregates scored opportunities, research log
  and packs into one snapshot.
- `CommunityHub` (`community_share`) exports/imports knowledge packs and
  product research as privacy-controlled bundles (private items excluded,
  optional anonymisation) — the ORION Network foundation.

Tool count is now **44**. Config dirs added: `config/knowledge_packs/`,
`config/community/`.

---

## Mark X.5 — Autonomous AI Operating System (2026-07-04)

Mark X.5 turns ORION from a reactive assistant into a continuously present AI
operating system. All subsystem communication remains on the OrionBus; all
blocking work runs through `asyncio.to_thread()`; no service imports a widget.

### True voice interruption
- `voice_interrupt.py` — **VoiceInterruptManager**: a grammar-constrained Vosk
  listener (sharing the loaded model, near-zero cost) that stays live even
  while ORION speaks. Phrases: *ORION pause / stop / wait / hold on / silence*
  → hold; *ORION continue / resume* → resume.
- `audio.py` — `AudioPlaybackThread` writes in ~85 ms slices and gains
  `hold()/resume_playback()`: the queue AND the unwritten remainder are
  preserved, so resume continues from the exact interruption point (no
  regeneration). `SpeechSynthesiser` tracks word boundaries and re-speaks the
  utterance from the interrupted word. `SpeechQueueManager.hold_all()/
  resume_all()` is the non-destructive counterpart of `interrupt_all()`.
- `live_worker.py` — `pause()` is now a hold; model audio streamed while
  paused is preserved silently; the gate feeds captured chunks to the
  interruption listener during speech (fixing the half-duplex deafness that
  made interruption unreliable).

### AI operating-system layer
- `identity.py` — **IdentityManager**: frozen core persona + persisted
  preferences (`config/identity.json`); one persona injected into every
  channel; drift-detecting signature. `ProviderRouter.attach_identity()`.
- `providers.py` — multi-model orchestration: SMALL/MEDIUM/LARGE workload
  tiers, per-provider latency EMA, strength-aware ordering.
- `cognition.py` — **CognitiveStateManager / GoalManager / IntentTracker**:
  durable goals, projects, workflows, tasks, priorities
  (`config/cognitive_state.json`), restored on launch.
- `cognitive_loop.py` — **CognitiveLoopManager**: the continuous awareness
  loop (workspace focus, fresh conversation, deadlines) — monitors and
  remembers, NEVER acts. `awareness` tool.
- `knowledge_graph.py` — **KnowledgeGraphEngine**: local SQLite FTS5 graph of
  entities/events/relationships with timeline reconstruction and offline
  answering (`config/knowledge_graph.db`). `second_brain` tool.
- `exporter.py` — **DocumentExporterService**: DOCX executive briefs,
  responsive HTML decks, MD+HTML+DOCX reports in the crimson palette; export
  history at `config/exports/history.json`. `document_export` tool.
- `reporting.py` — **ProactiveReportingService**: scheduled Daily Business /
  Weekly Product Intelligence / Monthly Growth reports, stored via the
  exporter; slot bookkeeping in `config/reporting.json`. `proactive_report`.
- `executive.py` — **ExecutiveAssistantMode** (JARVIS mode): status,
  prioritisation, scheduling, meeting minutes, workflow planning, progress.
  `executive` tool.
- `workspace.py` — **DesktopMemoryManager** extends WorkspaceManager with
  named workspace restoration ("restore marketing workspace").
- `commerce.py` — Entrepreneurial Intelligence Division:
  **ProductResearchAgent**, **CompetitorIntelligenceAgent** (store/offer/
  funnel), **BrandGrowthAgent** (Hausables strategy/CRO/positioning/
  retention). `competitor_intel` + `brand_growth` tools.
- `gui/diagnostics_centre.py` — **DiagnosticsCentreView**: live health, CPU/
  RAM, speech state + queue depth + hold flag, tool ledger, bus traffic —
  new DIAGNOSTICS page in the Command Deck.

### Startup + integration changes
- Dual-screen startup: core window maximised on monitor 0, Command Deck
  maximised on monitor 1 (Qt QScreen placement, DisplayTopologyManager for
  topology + logging); single monitor docks the deck right-half as a panel.
- Outlook lazy connection: `GetActiveObject` first; `Dispatch` (which starts
  Outlook) only on explicit user requests (`launch=True` from the tool);
  dashboard/proactive/briefing are attach-only. Connection state transitions
  logged (`DISCONNECTED → ATTACHED | LAUNCHED`).

Tool count is now **71**. New background tasks: `orion-cognitive-loop`,
`orion-reporting`. Version: `10.5.0` — codename *Autonomous AI Operating
System*.

---

## Mark X.6 — Holographic orb, futuristic HUD, smarter agents (2026-07-04)

A UI/UX and intelligence refresh layered on Mark X.5. No architectural
changes: the bus, `asyncio.to_thread` discipline and no-widget-in-services
rules all hold. British-English identifiers throughout.

### Reliability fixes (root-caused from a live session)
- **Invisible-Unicode window matching** — real window titles embed zero-width
  characters (Edge renders *"Microsoft​ Edge"* with a U+200B), which silently
  defeated `close`/`focus`/`click_text` matching. `utils.fold_title()` now
  strips invisible code points, collapses whitespace and lowercases; it is the
  single matcher used by `agents.DesktopAgent.window_control`/`close_app`,
  `control.AutonomousControlLayer._find_window` and
  `vision.VisionAgent._find_element_sync`. `close_app` also resolves spoken
  aliases → executables ("Microsoft Edge" → `msedge.exe`) and falls back to a
  polite window close; `window_control close` closes *all* matching windows.
- **Conversation recall never dead-ends** — with no time scope and no keyword
  hit, `conversation_memory.SessionRecall.recall` now summarises the current
  session instead of replying "I have no recorded discussion", and always
  points at the verbatim `transcript` tool.

### Futuristic orb — ORION's avatar (`gui/hud.py`, `gui/widgets.py`)
The flat stack of ellipses is replaced by a depth-shaded holographic entity,
same two-pass QPixmap caching and 16–33 ms tick:
- **Sphere-shaded body** — radial gradient offset up-left gives true volume;
  an **electric-cyan Fresnel rim** (lower-right) plus a crimson counter-rim.
- **Camera-iris aperture** — six blades that dilate with voice amplitude and
  active state (`iris` eased 0→1), revealing more of the incandescent core.
- **Orbiting satellites** — three tilted gyroscopic planes, each with a glowing
  node; nodes are split in-front-of / behind the orb by projected depth for
  real occlusion, and dim when behind.
- **Parallax starfield** baked into the cached background (fixed seed → no
  flicker), crimson core-glow + offset cyan key-light, vignette.
- **Holographic readouts** frame the orb: a left status stack
  (STATE/ROT/SCAN/IRIS) and a right **VOX amplitude ladder**, in the cyan
  data-hue. Corona is now a soft radial bloom, not a flat disc.
- **MiniOrb** shares the sphere shading, cyan rim and an orbiting node so the
  compact orb reads as the same being.

### Futuristic shell (`constants.py`, `gui/style.py`, `gui/core_window.py`)
- **Palette** gains an electric-cyan accent family (`ACCENT`/`ACCENT_DIM`/
  `ACCENT_DEEP`), brighter crimson (`PRI_HI`), `CORE`, glass surfaces
  (`PANEL_HI`, `INK`), status colours (`GOOD`/`WARN`/`BAD`) and RGB tuples for
  painter code. Crimson stays the *identity* hue; cyan is the *interaction*
  hue (focus rings, selections, active tabs, data lines).
- **Stylesheet** — top-lit glass panels, cyan focus/selection accents,
  monospace clock/state chips, and opt-in `accentButton` / `ghostButton`
  variants used to give the header visual hierarchy. Header title corrected to
  **MARK X.5**.

### Smarter specialist workforce (`agents.py`)
- **Weighted routing** — each agent declares high-signal `primary` patterns
  (weight 3) and supporting `keywords` (weight 1). `AgentManager` exposes
  `route_with_confidence()` and only prefers a specialist above a
  `ROUTE_THRESHOLD`, so a lone brushing keyword ("email") no longer hijacks a
  request — it falls through to ORION's general capacity.
- **Shared `OPERATING_METHOD`** — every specialist now reasons like a senior
  practitioner (restate objective → state assumptions → reason from first
  principles/evidence → specific actionable answer → next action + risks),
  injected into each system instruction so even small local models lift.
- **Deeper personas** for all five specialists, plus a new **Research &
  Analysis Agent** (`research`) for synthesis, comparison and evaluation —
  wired into the `agent_dispatch` tool declaration. Six specialists total.
- Verified offscreen: weighted routing picks the right specialist across six
  domains, weak signals fall through, and both orbs render without error.

---

## Mark X.7 — Cloud/mobile readiness, street-level globe (2026-07-04)

Makes ORION deployable off the desktop — a headless brain for **Oracle Cloud**
and an installable **private Android app** — without touching the desktop build.
Additive and backward-compatible: `python orion.py` is unchanged.

### Headless brain (`bus.py`, `server.py`, `orion.py`)
- **`OrionBus` is now a `QObject`, not a `QWidget`.** It was only ever a signal
  hub, but subclassing `QWidget` forced a full GUI `QApplication` and a hidden
  top-level widget. As a `QObject` the signal hub runs under a
  `QCoreApplication`, which is what unlocks headless operation. Only site of
  construction (`app.py`) is unaffected; nothing used it as a widget.
- **`orion_core/server.py`** composes the *portable* half of ORION — bus →
  telemetry → memory → identity → connectivity → Ollama → `ProviderRouter` →
  conversation memory → knowledge/pack seeds → `LearningService` → remote
  uplink — under `QCoreApplication` + qasync, **no GUI, audio, screen-grab or
  Windows COM**. Runs on a display-less Ubuntu/Oracle-Linux VM. Graceful
  SIGINT/SIGTERM shutdown.
- **`orion.py`** now branches early: `--headless` / `--server` / `--cloud` (or
  `ORION_HEADLESS=1`) takes a *lighter* dependency gate (aiohttp + qasync +
  PyQt6-Core only) and enters `server.main()`; the desktop path is unchanged.

### Remote uplink → the full brain + a real app (`remote.py`)
- **Full-brain routing** — remote turns were answered by raw
  `router.generate_text`; they now go through `_answer()`: identity- and
  memory-grounded language model when reachable, with the tool-capable offline
  `LocalBrain` as fallback. A phone gets the *real* ORION, memory-synced.
- **Installable PWA** — the page ships a `manifest.webmanifest`, a cache-first
  **service worker** (`/sw.js`, network-first for `/api/*`), and a
  PIL-generated crimson-orb icon (`/icon-192.png` `/icon-512.png`, SVG
  fallback). "Add to Home screen" on Android/iOS gives a full-screen,
  own-icon app that opens offline — no store, private via token.
- **Hardening for internet exposure** — constant-time token check
  (`hmac.compare_digest`), per-client sliding-window rate limit (30/min,
  self-pruning), CSP + `nosniff` + `X-Frame-Options` + `Referrer-Policy` on
  every response, richer `/api/health` (mode, uptime). TLS is expected from a
  reverse proxy, not terminated here.
- Constructor now takes `router, memory, bus, *, identity, local_brain,
  conversation, telemetry` (was `worker, memory, bus`); `app.py` updated. Works
  with or without a live worker, so both the desktop and the headless node
  reuse it.

### Deployment scaffolding (`deploy/`)
`requirements-server.txt` (brain-only deps), `Dockerfile` (multi-arch
`python:slim`, `QT_QPA_PLATFORM=offscreen`, healthcheck), `orion-node.service`
(hardened systemd unit), `Caddyfile` (one-line auto-HTTPS reverse proxy), and
`README_ORACLE_CLOUD.md` — a step-by-step for an Always-Free VM: ingress + host
firewall, cloud key *or* local Ollama, service install, TLS, and installing the
app on the phone.

### Street-level globe ↔ Google Maps (`gui/globe.py`)
- Flying to a place now also opens a **keyless Google Maps embed**
  (`output=embed`) driven by the *raw query*, so Google resolves an exact
  street/address/apartment rather than the city centroid — built for research.
  **Street View** and **Open-in-Google-Maps** buttons hand off to the system
  browser through the existing navigation interceptor; the card hides on Reset.
- Denser detail: the offline city-marker set roughly tripled (~70 cities);
  centred hint line so the map card never collides with it.

### Trainability & self-repair (assessment, no code change)
- **Trainable by data, not code** *today*: `LearningService` (`learn` tool)
  ingests text/file/URL → distils atomic facts → **KNOWLEDGE tier** →
  recalled online/offline; **Knowledge Packs** add installable JSON expertise;
  identity/preferences persist in `config/`. So most "teach ORION X" needs no
  edit — see the follow-up list for closing the loop (correction learning,
  remote learning endpoint).
- **Self-repair extent**: `SelfRepairAgent` captures faults
  (`sys.excepthook` + asyncio handler), reads the failing source window, drafts
  a fix via the model, **compile-validates it**, and — *with explicit approval*
  (`repair_file(confirm=True)`) — backs up and rewrites its own file inside
  `orion_core`, auto-reverting if the write fails to compile; `revert_last`
  undoes it. It is approval-gated and restart-to-load by design, never
  autonomous. Diff-only proposals for large files land in `config/self_repair/`.

---

## Mark X.8 — Bulk learning, cyber corpus, momentum, small cursor (2026-07-04)

Entrepreneur-facing capability + the knowledge expansion path, plus two direct
UX asks. All additive; British-English identifiers; tools grow to reflect it.

### A normal cursor (`gui/cursor_overlay.py`)
The large crimson "scope" halo (66 px ring + crosshair) is replaced by a small
(34 px) **black arrow pointer** with a white outline for contrast, its tip
anchored exactly on the OS cursor. ORION's autonomous moves still read at a
glance via a *small* soft flare ring at the tip on `control_activity`, but it no
longer dominates the screen.

### Learn gigabytes — bulk ingestion (`learning.py`, `memory.py`)
- **`LearningService.learn_folder(folder, topic, deep)`** walks an entire
  directory (recursively) and ingests every readable document — PDF, DOCX,
  Markdown, text, HTML and source code (`TEXT_SUFFIXES`/`DOC_SUFFIXES`) — into
  the KNOWLEDGE tier. Default is **fast extractive** (no API cost, scales to
  very large libraries); `deep=True` distils each file with the model. Walking
  and extraction run off the event loop; progress is emitted on the bus. This is
  the realistic path to "gigabytes": supply the corpus locally, ORION indexes
  it, and it's recalled online or offline.
- **Correction learning** — `correct(topic, correction)` records an
  authoritative override; `forget(query)` removes previously-learned facts
  (built-in corpora are never touched). Backed by a new
  `OrionMemoryMatrix.forget(category, key_prefix, contains)` (FTS-synced DELETE,
  never wipes everything without criteria) surfaced on `MemoryAgent.forget`.
- The `learn` tool gains actions `folder` / `correct` / `forget` alongside
  `learn` / `recall`.

### Cybersecurity knowledge base (`cyber_knowledge.py`)
A curated, **defensive/educational** corpus (33 entries: CIA triad, defence in
depth, zero trust, threat modelling/STRIDE, the OWASP risks, injection/XSS/
CSRF/SSRF, cryptography, auth/MFA, network defence, malware & ransomware,
privilege escalation, MITRE ATT&CK, detection/SIEM, incident response, secure
SDLC, cloud/container and supply-chain security). Mirrors the neuroscience and
programming bases: seeded idempotently into the KNOWLEDGE tier (desktop **and**
the headless cloud node), a `CYBER_PERSONA_BOOST` injected into the system
instruction next to the neuro/programming boosts, and a `cyber_knowledge` tool.
Neuroscience, programming and cybersecurity are now ORION's three resident
technical corpora, all offline-recallable and all expandable by `learn_folder`.

### Momentum — a shipping coach (`momentum.py`)
`MomentumEngine` turns tracked cognition state into *pressure to finish* — the
gap between "tracks tasks" (cognition) and "monitors them" (awareness loop):
- `focus` — the single highest-leverage next action for the active project,
  plus blockers/overdue and a concrete focus block;
- `standup` — a cross-project shipping stand-up (in-progress / do-next / overdue);
- `plan` — breaks a project/goal into ordered milestones with a definition-of-
  done and immediate next actions (model-drafted, template fallback offline).
Read-only over cognition (never fights the awareness loop); exposed via the
`momentum` tool. Aimed squarely at an entrepreneur getting projects *shipped*.

### Navigation — scroll-to-find (`verification.py`)
`click_text` now **scrolls the page to find** a target that is below the fold or
not yet realised in the accessibility tree (the usual reason a first attempt
can't find a section title), retrying across interactive roles at each step
before giving up — self-correcting web/desktop navigation.

Verified: full package byte-compiles; the new modules import cleanly; the cyber
corpus loads (33 entries). GUI/audio not launched in this environment.

---

## Mark X.9 — Overlay restore, clean shutdown, HUD polish (2026-07-04)

Three usability fixes surfaced from real use, plus a further futuristic pass.

### You can always get ORION back (`gui/core_window.py`, `gui/style.py`)
Overlay mode hid the header — and with it every control — so there was **no way
to un-shrink ORION**. Now:
- floating glass **restore (⤢) and quit (⏻) chips** (`#overlayChip`) sit over the
  compact orb, shown on enter / hidden on exit and repositioned on resize;
- **double-clicking the orb restores** the full window;
- entering overlay shows a 3-second banner spelling out every way back
  (double-click · ⤢ · Esc · Ctrl+Shift+O).

### Clean shutdown — no more lingering process (`gui/core_window.py`, `app.py`)
Launching with `py orion.py` left ORION running until reboot, because the core
window's X only hid it while the floating pill and Command Deck kept the app
alive. Fixed at three levels:
- a header **Quit (⏻) button** and **Ctrl+Q**, both emitting
  `bus.request_shutdown`;
- **`closeEvent`** now requests a full shutdown, so closing the main window
  stops every window and background task;
- **terminal Ctrl+C** works: a `SIGINT` handler calls `app.quit()` and a 250 ms
  no-op `QTimer` keeps the interpreter responsive so the signal is delivered
  even while the Qt/asyncio loop is in C code (the Windows gotcha).

### More futuristic HUD (`gui/hud.py`, `gui/style.py`)
- A soft **CRT-style cyan scanline** drifts continuously down the HUD — a subtle
  holographic 'refresh' over the orb.
- The shutdown control is styled as a real **power button** (transparent →
  crimson on hover); overlay chips are cyan-rimmed glass.

---

## Mark X.10 — Navigation cleanup: one control language (2026-07-04)

The GUI read as cluttered because the core header crammed ten controls into one
row using *two* navigation paradigms (a dropdown for views + five loose window
buttons), while the deck used yet another (plain tab buttons). Unified into a
single, consistent control language with clear visual zones.

### Shared controls (`gui/style.py`)
- **Segmented control** (`QFrame#segmented` + `QPushButton#segItem`) — one
  scannable switcher used for *both* the core views and the deck pages, so
  navigation looks and behaves identically everywhere. Active segment is
  crimson (identity); hover lifts to white.
- **Icon cluster** (`QFrame#controlCluster` + `QPushButton#iconButton`) —
  compact single-glyph buttons grouped in a recessed panel, replacing rows of
  loose labelled buttons.

### Core window header (`gui/core_window.py`)
Rebuilt into clear left→right zones:
`brand · [⬡ HUD · ⌨ LOG · ⊞ MEMORY · ◈ TELEMETRY] ——— ⏸ Pause · [⧉ ◉ ◱ ⛶] · VOICE · STATE · CLOCK · ⏻`
- The **view dropdown is gone**, replaced by the segmented switcher
  (`_select_view` keeps it in sync; overlay entry reuses it).
- The four window-mode buttons are now an **icon cluster** with tooltips
  instead of five full-width labelled buttons — far less horizontal noise.
- Status chips (voice/state/clock) are grouped as a unit before the power
  button.

### Command Deck (`gui/unified_dashboard.py`)
The tab bar now uses the **same segmented control**, with a concise glyph per
page (⧉ Widgets · ⚒ Toolkit · ♪ Studio · ◉ Centre · ◈ Diagnostics · ◍ Globe) and
`iconButton` ‹ › arrows, centred between the title and the swipe hint. "COMMAND
CENTRE" shortens to "CENTRE" on the tab (full name still resolves in
`show_page_named`).

Net effect: the same operations, a fraction of the visual weight, and one
navigation model the eye learns once. Verified: package byte-compiles; GUI
modules import cleanly.

---

## Mark X.11 — Speaking face + offline Whisper ears (2026-07-04)

### The cyberpunk speaking face (`gui/face.py`, `gui/core_window.py`)
`HologramFace` is a second avatar: a **Detroit-style holographic skull** drawn
entirely in QPainter vectors, in the same crimson-and-cyan palette as the orb —
machined cranium panels edged in crimson, cyan circuit seams, angular glowing
eyes, an expressive brow, a forehead sensor, and an **articulated jaw whose
mouth opens in time with ORION's voice** (a voice-equaliser glows inside the
mouth while he talks).
- Driven by the same signals as the orb: `set_amplitude` (jaw/mouth),
  `set_state` (expression) and `set_speaking` (luminance). Expression channels
  (eye openness, brow tilt, glow) ease toward per-state targets — wide cyan eyes
  when **listening**, furrowed focus when **processing**, bright crimson when
  **speaking**, calm dim when **idle** — plus timed blinks so it feels alive.
- **Auto-morph on speech**: it is a `FACE` view (index 1) in the segmented
  switcher, and while ORION speaks the core window automatically morphs the orb
  HUD into the face and back — but only when you're on the orb view (never
  yanks you out of log/memory/telemetry) and only reverses a switch it made
  (`_auto_face`/`_auto_faced`). Frame-rate is throttled for whichever avatar
  isn't on screen.

### Offline Whisper ears (`speech_offline.py`, `requirements.txt`)
`OfflineTranscriber` already fell back faster-whisper → openai-whisper → vosk;
the engines are now actually installed. **faster-whisper** (ctranslate2, int8
CPU, no torch) is installed and detected as the preferred engine; **openai-
whisper** is added as the heavier torch-based alternative. Model size is
`ORION_WHISPER_MODEL` (default `base.en`); the model downloads on first use.
This gives ORION fully offline dictation to pair with his already-offline male
voice — internet-free ears and mouth.

Verified: package byte-compiles; `gui.face` and `gui.core_window` import
cleanly; `import faster_whisper` succeeds and is first in the engine order.
