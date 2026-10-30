# All current O.R.I.O.N. capabilities

26 September 2026, updated 27 September after the integration pass · Mark XXXII / package 32.0.0 · Current source checkout

## Scope and verification

The source declares **144 core model tools**, **22 dashboard pages** (TASKS added on 27 September), **18 bundled plugins** and a separate **phone_device** tool for paired-phone turns. Every core declaration has a corresponding dispatcher key.

The dispatcher contains 145 literal keys: the core declarations plus the image_processor compatibility alias. The capability checker now records the two lambda-based gateway handlers as `find_tool:<lambda>` and `use_tool:<lambda>`, so it counts all 145; before 27 September it reported 143 attribute-valued mappings. The baseline file was not regenerated, so a removal still fails the check.

This inventory covers all statically declared core tools, bundled plugins, dashboard pages and non-tool platform functions inspected here. Dynamic server/tool discovery has no permanent exhaustive count. A registered interface can require an account, device, model, executable or permission before it works. Source presence, configured enablement, successful connection and live end-to-end validation are different states.

The capability-preservation check passed with no missing or duplicate entries. It reported five additions against the retained baseline: aviation, fetch_url, find_tool, sound_sense and use_tool. No application, accounts or hardware actions were started for this inventory.

The active desktop uses PyQt6/qasync; aiohttp serves the phone/API. The Overview is native 2D, with a separate avatar and optional globe/renderer paths. The FastAPI/local transformer/LoRA/DPO package remains an optional reference; planned improvements are not counted as current shipped capability.

Full unabridged declarations and parameters are in `orion_core/dispatch_schema.py`.
Sources: [tool declarations](<orion_core/dispatch_schema.py>); [dispatcher](<orion_core/dispatcher.py:518>); [desktop composition](<orion_core/app.py:1707>).

## Complete capability rundown

### Conversation, hearing and appearance

Voice/text interaction, cloud live audio and configured fallback voices; microphone/speaker selection, speech interruption and offline recognition; enrolled speaker comparison, tone hints, environmental sound analysis and sound-watch alerts; facial/orb state and avatar behaviour. Speech models, sound classifiers, voices and devices must be available. Pitch-based speaker_id is not identity recognition, and tone/sentiment is an estimate.

**Exact tools:** **audio_devices**, **elevenlabs_voice**, **voice_speaker_id**, **speaker_id**, **voice_tone**, **sound_sense**, **emotion**, **avatar**, **ai_mode**, **transcript**.

### Windows, interface and workspace control

Launch/close applications, arrange/focus windows, type/click/use hotkeys, operate the clipboard, inspect/manage processes, adjust supported media/peripherals, inspect monitor topology, save/restore workspaces, navigate ORION, show the cursor halo, configure sign-in startup and restart/shut down ORION. OS actions use the running user's permissions and existing policy gates.

**Exact tools:** **open_app**, **close_app**, **window_control**, **media_control**, **desktop_control**, **clipboard_operate**, **process_governor**, **resource_status**, **display_info**, **peripherals**, **system_notify**, **system_startup**, **shutdown_orion**, **restart_orion**, **cursor_overlay**, **interface_control**, **workspace_control**, **navigation_trace**.

### Vision, camera and gestures

Screen capture, accessibility-tree text/control discovery, OCR, model-assisted screen advice and post-action verification. Shared-camera Vision Lab captures identified/read/count/electronics inspections with quality checks, numbered/grid-localised observations and follow-ups. Optional local perception handles motion, colours, object detection, poses, overlays and trigger rules; gestures can operate supported controls. Photos do not establish electrical measurements or hidden PCB faults.

**Exact tools:** **vision_analyse**, **capture_screen**, **vision_verify**, **screen_read**, **perception**, **gesture_control**.

### Files, organisation and recovery

Find files by name/type/date/size; inspect supported text, structured data, PDF and image content; perform supported file operations; preview/apply organisation; retain supported action undo; back up/restore ORION state and review cleanup candidates. Undo is bounded/session-local, not universal recovery.

**Exact tools:** **find_files**, **file_controller**, **process_file**, **organise_files**, **undo**, **backup**, **cleanup_review**.

### Browser, research and document reading

Search/open/read URLs and control browser sessions, forms, tabs and supported downloads. Read multiple documents with citations, ingest academic literature, run research papers/dossiers, track standing questions, produce catch-up reports and open briefing stories. Flight search reads offers from accessible pages; it does not book/pay or guarantee fares.

**Exact tools:** **web_search**, **fetch_url**, **browser_control**, **web_control**, **web_automation**, **research**, **read_documents**, **literature_vault**, **flight_search**, **open_news**, **standing_questions**, **catch_up**.

### Memory, learning and knowledge

Persist explicit facts/preferences, retrieve conversations by time/topic, export/search verbatim transcripts, ingest files/folders/URLs, deduplicate content and maintain a knowledge graph. Combine lexical search with an available local semantic encoder. Consult installed programming, neuroscience, cybersecurity and other knowledge packs, and exchange supported privacy-controlled knowledge bundles. Stored knowledge is not inherently current.

**Exact tools:** **save_memory**, **query_intelligence**, **recall_conversation**, **conversation_recall**, **rewind**, **learn**, **ingest**, **second_brain**, **expand_mind**, **knowledge_pack**, **programming_knowledge**, **neuro_knowledge**, **cyber_knowledge**, **founder_knowledge**, **community_share**.

### Reasoning and task execution

Consult six built-in specialist agents and configured declarative specialists; perform evidence-oriented deliberation, critique and bounded combinatorial strategy search. Execute verified plans with DAG dependencies, background jobs, named workflows/protocols and repeated-sequence suggestions. These are orchestration/analysis features, not a proof of general intelligence or guaranteed autonomous completion.

**Exact tools:** **agent_dispatch**, **reason**, **strategy**, **execute_plan**, **autoplan**, **job**, **workflow**, **workflow_patterns**, **protocol**, **muscle_memory**.

### Productivity and personal organisation

Track missions, linked tasks/notes, reminders, focus/break sessions, spaced-repetition study, language practice and decision outcomes. Maintain local finance and wellbeing records, continuity notes, priorities, project momentum and adaptive/scheduled briefings. Finance is a local/manual or CSV ledger, not live bank connectivity; wellbeing uses supplied records.

**Exact tools:** **mission**, **reminder**, **focus**, **study**, **language_tutor**, **decision**, **wellbeing**, **finance**, **companion**, **awareness**, **executive**, **momentum**, **briefing**, **morning_briefing**, **proactive_check**, **proactive_report**.

### Business, marketing and commerce

Analyse product opportunities, supplied trend signals, creator hooks/scripts/CTAs, brand positioning, competitors and funnels; build campaign/creator plans and maintain an agency pipeline. Social automation uses accessible logged-in accounts. Scores and model advice do not prove commercial demand or provide privileged platform analytics.

**Exact tools:** **product_research**, **tiktok_intel**, **instagram_intel**, **business_advisor**, **commerce_hub**, **competitor_intel**, **brand_growth**, **campaign_pipeline**, **creator_intel**, **social_media**.

### Communication, accounts and phone actions

Read/draft/send through supported Outlook paths, work with configured Notion tasks/events/projects, and use configured messaging routes. phone_action opens pre-filled native intents for user completion. Configured Twilio supports real calls/SMS and a separate voice bridge with contact/approval restrictions. Network/account setup and existing authorisation remain necessary.

**Exact tools:** **outlook_mail**, **notion_workspace**, **messaging**, **phone_action**, **telephony**.

### Development, debugging and self-improvement

Read/index repositories, inspect symbols/dependencies/hotspots, run supported development commands, use pdb and Docker interfaces, generate/verify tools through Forge, capture faults, prepare/apply/revert eligible repairs and inspect source changes. Automatic repair scope is controlled by runtime policy; generated-code tests are not an OS isolation sandbox. These features do not mean the foundation language model retrains itself.

**Exact tools:** **dev_workbench**, **docker**, **debugger**, **codebase_copilot**, **forge**, **self_repair**, **code_changes**, **self_changes**, **patch_notes**.

### Security and authorised lab tools

Inspect processes/listening ports and Windows Defender state, check configured breach services, detect/redact recognised secrets, teach security curricula and guide authorised labs. Approved target scopes enable supported Nmap/scapy reconnaissance. Required binaries/drivers/accounts must be present; these are not a complete antivirus or penetration-test guarantee.

**Exact tools:** **security_watch**, **breach_check**, **antivirus**, **privacy_guard**, **cyber_curriculum**, **pentest_lab**, **security_recon**.

### Media, games and creative audio

Index/process supported vocal assets and audio-studio material, discover/analyse supported YouTube content, inspect/launch supported installed games and play/analyse chess with ORION's engine or optional Stockfish. Do not treat historical Elo claims as an independently certified rating.

**Exact tools:** **audio_studio**, **gaming**, **entertainment**, **chess**.

### Maps, geography and aviation

Navigate the 3D globe, zoom, resolve places and supported geographic information, and inspect live aircraft from configured/public feeds. Coverage, freshness, network availability and renderer support affect results.

**Exact tools:** **globe**, **max_zoom_in_globe**, **geo**, **aviation**.

### Extensions and capability discovery

Discover all registered tools, invoke them through the existing gateway, manage data-only skills and code plugins, connect MCP servers and inspect capability health. Plugins support installation/scaffolding/export, enable/disable/reload, declared permissions, events and scheduling where configured.

**Exact tools:** **skill**, **plugin**, **mcp**, **find_tool**, **use_tool**, **capabilities**.

### Health and usage diagnostics

Inspect system/application health, dependency/database checks, resource pressure, provider usage, startup stages, request traces and event-loop stalls. Alerts and failure reporting exist; monitoring does not automatically fix every fault.

**Exact tools:** **diagnostics**, **sentinel**, **token_usage**.

### Document and presentation export

Produce supported Markdown/HTML reports, DOCX briefs/reports and HTML presentation decks using the export services. Format/backend availability matters; an HTML deck is not automatically a PowerPoint file.

**Exact tools:** **draft_report**, **document_export**.

## Platform capabilities beyond individual tool names

- **Conversation continuity:** session memory, transcripts, request stages, reconnect/recovery paths, speech gating and configured fallback routes.
- **Reasoning agents:** built-in marketing, coding, research, neuroscience, AI/ML and cybersecurity specialists; data manifests can add agents. A new agent does not automatically gain a dedicated dashboard page.
- **Local neural learning:** sparse MLP intent routing, success/correction observations, replay and persisted network state; optional semantic embeddings improve recall. This is not a standalone general-purpose language foundation model.
- **Remote access:** paired PWA/Android clients, access-token checks, event streams, confirmations, scoped device actions and mobile viewport uploads.
- **Phone-specific tool:** phone_device supports recent phone frames, fresh accessibility snapshots, click/scroll, app opening and permitted call/SMS commands. Requires a paired turn, enabled permissions and applicable device-side confirmation. It is additional to the 144 core declarations.
- **Native phone intents:** phone_action opens pre-filled dialler/composer/navigation/share flows. This differs from Twilio telephony, which can perform real configured calls/SMS.
- **Lifecycle:** single-instance launcher, startup dependency checks, phased startup, graceful shutdown/restart, configuration relocation and state backup/restore mechanisms.
- **Operational resilience:** cooldowns/circuit breakers, resource-pressure shedding, MCP on-demand startup/parking/supervision, health checks and crash diagnostics.
- **Extension lifecycle:** declarative agent manifests, data skills, dynamic Python tools, plugin events, schedules, fault muting, scoped credential mechanisms and capability health.
- **Task control:** every plan, workflow and background job is a durable record with states (queued, running, waiting for approval, paused, completed, failed, cancelled, interrupted), shown on the TASKS page; the `job` tool can also pause, resume and change priority by voice. Approvals are given only by the user (TASKS page or phone card), bound to the exact step and arguments and valid for ten minutes. Work interrupted by a shutdown is listed and resumable, never replayed automatically.
- **Shared admission:** one machine-wide policy admits interactive, user, background and training work with bounded queues, fairness over time and memory/CPU/GPU pressure; local model generation and training never share the GPU.
- **Streaming phone replies:** the phone shows a reply as it is written (`/api/chat/stream`), can stop it, and pages older history on demand.
- **Tasks on the phone:** a TASKS sheet (`/api/tasks`, read-only) lists what ORION is doing, queued or waiting, with honest progress; steps waiting for approval are named and approved on the desktop.
- **Motion and performance preference:** Full (default), Balanced (lighter face rendering) or Reduced motion (calmer idle drift, no page slides); speaking and expressions are never reduced.
- **Deployment:** source desktop execution, standalone/installer tooling, a portable headless subset and separate telephony/server deployment assets. A headless node does not automatically acquire a different PC's desktop access.
- **Learning/intelligence records:** explicit preferences, conversations, knowledge graph, decisions, workflow patterns, cognitive context and feedback have distinct storage/behaviour from neural weights.

## All dashboard routes

These are route keys; BRAIN is shown through the current Overview/Iris/Signal Lanes interface. Removed historical page names are not current pages merely because an older document lists them.

**AUTOMATION**, **BRAIN**, **CHESS**, **COGNITION**, **COMMAND CENTRE**, **DEVELOPMENT**, **DIAGNOSTICS**, **GLOBE**, **LIBRARY**, **LOG**, **MARKETING**, **MEMORY**, **MISSION**, **OPS**, **PLUGINS**, **RESEARCH**, **SECURITY**, **TASKS**, **TELEMETRY**, **TOOLKIT**, **WIDGETS**, **WORKBENCH**.

GLOBE, CHESS, MISSION, DIAGNOSTICS, LIBRARY, PLUGINS, DEVELOPMENT, MARKETING, COGNITION and TASKS are built the first time they are opened; a page that fails to build says why and offers **Try again**. The sidebar has PINNED (right-click a page to pin it) and RECENT sections.

## All bundled plugins

These 18 modules/manifests are present. Individual enabled state, credentials and external reachability were not certified by this read-only inventory.

| Plugin | Implemented purpose and boundary |
|---|---|
| [dependency_manager](<config/custom_tools/dependency_manager.plugin.json>) | Inspect dependencies and explicitly install required packages in supported source environments. |
| [dependency_resolver](<config/custom_tools/dependency_resolver.plugin.json>) | Check installed distribution versions against requirements. |
| [discord_message](<config/custom_tools/discord_message.plugin.json>) | Send through a configured Discord contact/channel/webhook route. |
| [emotional_insight_generator](<config/custom_tools/emotional_insight_generator.plugin.json>) | Filter/analyse supplied emotion records; does not independently observe emotions. |
| [emotional_monitor](<config/custom_tools/emotional_monitor.plugin.json>) | Estimate lexical sentiment from supplied text. |
| [error_handler](<config/custom_tools/error_handler.plugin.json>) | Persist a diagnostic record; does not apply a repair. |
| [fitness_goal_tracker](<config/custom_tools/fitness_goal_tracker.plugin.json>) | Persist goals/deadlines/progress; does not automatically schedule reminders. |
| [fitness_progress_monitor](<config/custom_tools/fitness_progress_monitor.plugin.json>) | Persist supplied workout records; does not measure workouts or schedule reminders. |
| [homeassistant_call_service](<config/custom_tools/homeassistant_call_service.plugin.json>) | Read states/entities or call configured Home Assistant services. |
| [ifttt_webhook](<config/custom_tools/ifttt_webhook.plugin.json>) | Trigger a configured event; acknowledgement does not prove applet completion. |
| [json_validator](<config/custom_tools/json_validator.plugin.json>) | Validate JSON and supported schemas. |
| [live_camera_analysis](<config/custom_tools/live_camera_analysis.plugin.json>) | Analyse the shared camera through the available vision path. |
| [open_meteo_weather](<config/custom_tools/open_meteo_weather.plugin.json>) | Look up weather over the network. |
| [philips_hue](<config/custom_tools/philips_hue.plugin.json>) | Control paired bridge lights and check reported state. |
| [push_notify](<config/custom_tools/push_notify.plugin.json>) | Send through configured ntfy/Gotify/Pushover; delivery depends on the service. |
| [self_repair_agent](<config/custom_tools/self_repair_agent.plugin.json>) | Train/use a classifier on supplied examples; does not itself repair source. |
| [synthesis_enhancer](<config/custom_tools/synthesis_enhancer.plugin.json>) | Persist a draft tool specification; executable implementation uses Forge. |
| [telegram_message](<config/custom_tools/telegram_message.plugin.json>) | Send through a configured Telegram bot/destination. |

## MCP configuration snapshot

This checkout contains **21 server configurations**, of which **11 have enabled=true**. That is a configuration flag, not a successful handshake or proof that credentials/binaries are ready. Only names and flags are included; commands, arguments, endpoints and secret values are omitted.

| Server | Configuration flag |
|---|---|
| brave_search | Disabled |
| context7 | Enabled |
| duckduckgo | Enabled |
| fetch | Enabled |
| filesystem | Enabled |
| git | Enabled |
| github | Disabled |
| gmail | Disabled |
| google_calendar | Disabled |
| google_drive | Disabled |
| google_maps | Disabled |
| home_assistant | Disabled |
| markitdown | Enabled |
| memory | Enabled |
| notion | Enabled |
| playwright | Enabled |
| postgres | Disabled |
| slack | Disabled |
| sqlite | Enabled |
| tavily_search | Disabled |
| twilio | Enabled |

Connected servers can add tools for search/fetch, files, browser automation, memory/databases, Git, documentation, Notion, telephony and configured accounts such as GitHub, Gmail/Calendar, Slack, Home Assistant and Drive. Runtime discovery is authoritative for the exact available tools. The core find_tool/use_tool gateway exposes registered tools through existing policy.

## Optional/reference-only capabilities

The neural reference provides a local transformer engine, typed neuro-symbolic plans, a memory adapter, PC execution reference, priority/DAG scheduling, feedback handling and LoRA/DPO training integration. Its separate tests include real CPU integration plumbing. It is not automatically part of normal ORION startup, and useful pretrained GPU reasoning/training has not been established by those tests.

The performance review's roadmap has since been implemented in source (27 September): the native Ollama transport and residency policy, pooled provider sessions with one total deadline, shared priority admission, strict whole-plan validation, durable jobs and workflows, streamed phone text and bounded phone history, lazy deck pages and memory snapshots — see `ORION_INTEGRATION_LEDGER_2026-09-26.md` for evidence and limits. The optional local neural planner (feature flag, off by default) is the part still being integrated; gated adapter promotion requires evaluation and the owner's explicit command.

## Exhaustive core tool index

Every declared core tool appears once below. Purposes are abridged from the schema and can contain older wording; use the domain rundown for qualified current behaviour. The JSON companion retains all description text, argument properties and required fields. A declaration is not a guarantee that its external integration is configured.

| Tool / source | Declared purpose (abridged) | Required arguments |
|---|---|---|
| [agent_dispatch](<orion_core/dispatch_schema.py:193>) | Consult a specialist agent: marketing (digital marketing/growth), coding (software engineering/debugging), research (research, analysis, comparison & evaluation), neuroscience, aiml (AI/ML engineering), security (defensive cyber security), or auto to route by content. | request |
| [ai_mode](<orion_core/dispatch_schema.py:839>) | Report ORION's active intelligence mode: cloud-enhanced (MODE A) vs fully offline (MODE B), internet status, and which cloud/local models are available. Use when asked 'are you online', 'what model are you using', or about offline capability. 'recommend' (also 'cheapest', 'which_model') suggests which model to use — … | None declared |
| [antivirus](<orion_core/dispatch_schema.py:1523>) | Windows Defender awareness — reports the user's own machine's malware protection, never a scanner of ORION's own. 'status' reports real-time protection state, definitions age and last scan time; 'scan' triggers a quick scan and reports the result; 'threats' lists recently detected threats. Degrades to an actionable … | None declared |
| [audio_devices](<orion_core/dispatch_schema.py:1692>) | Inspect or choose ORION's microphone and speaker. 'list' shows every device with its index; 'status' says which mic and speaker are in use; 'set_output'/'set_input' select a device by index or name fragment (e.g. 'headphones'). Use this when the user says they can't hear ORION or wants a specific mic/speaker. | action |
| [audio_studio](<orion_core/dispatch_schema.py:1391>) | Creative Audio Workspace: index_assets scans the audio_studio/raw folder; process_vocal_take gain-stages (loudness-normalises) a vocal WAV/MP3 and writes a processed stem; export_stem_package collects processed stems into a dated package with a manifest. | action |
| [autoplan](<orion_core/dispatch_schema.py:1471>) | Run a multi-step autonomous plan with self-verification: each step is checked (desktop_control/web_control/vision_verify steps are visually confirmed via screen diff, others by result), retried on failure, and reported honestly. | steps |
| [avatar](<orion_core/dispatch_schema.py:2015>) | The avatar system (state machine + face tracking): 'status' reports the avatar state and tracking; 'state' requests a behaviour state (idle, listening, thinking, speaking, researching, executing); 'notify' fires a notification pulse; 'warn' raises the warning state; 'track' enables webcam face tracking so the avatar … | action |
| [aviation](<orion_core/dispatch_schema.py:1755>) | Live aircraft in the sky from public ADS-B feeds (adsb.lol, then OpenSky). 'overhead' says what is flying over the user right now (default 25 km); 'nearby' lists aircraft within a radius of the user or a place; 'track' finds one flight by callsign (e.g. BAW123) or ICAO24 hex code and gives its latest position, … | None declared |
| [awareness](<orion_core/dispatch_schema.py:1602>) | The continuous cognitive loop: 'situation' reports what ORION is currently aware of (projects, focus, deadlines, priorities, intents); add_priority/add_task/complete_task/goals maintain the durable cognitive state. Tasks: 'list_tasks' shows them with what is OVERDUE; 'remove_task' deletes one the user no longer wants … | action |
| [backup](<orion_core/dispatch_schema.py:1534>) | Back up ORION's settings and memory to a timestamped zip (default under OneDrive so it cloud-syncs). 'backup' creates one; 'list' shows archives; 'restore' unpacks one into a review folder (never overwrites live files). | None declared |
| [brand_growth](<orion_core/dispatch_schema.py:1678>) | Growth strategist for Creator Studio: 'strategy' (positioning + channel priorities), 'conversion' (offer/funnel plan), 'positioning' (service positioning), 'retention' (client retention systems). Findings persist to long-term memory. | action |
| [breach_check](<orion_core/dispatch_schema.py:1511>) | Check whether the user's own credential has appeared in a known data breach (Have I Been Pwned), a defensive security-hygiene tool. 'password' checks a password using k-anonymity — the password is SHA-1 hashed locally and only the first 5 hash characters are ever transmitted; the password is never sent, logged or … | None declared |
| [briefing](<orion_core/dispatch_schema.py:2027>) | Dynamic adaptive briefing from live system activity (distinct from the classic morning_briefing news brief): 'brief' composes a briefing for the current time of day (or pass morning/midday/evening) including what happened while the user was away (research completed, workflows finished), progress since the last brief … | None declared |
| [browser_control](<orion_core/dispatch_schema.py:373>) | Open a secure URL or search in the system browser. | None declared |
| [business_advisor](<orion_core/dispatch_schema.py:922>) | Personal business advisor for the user's business Creator Studio (short-form content, UGC and creator management; TikTok, Reels, Shorts). Actions: advise (topic), brand, growth (horizon), store (offer/funnel optimisation). | None declared |
| [campaign_pipeline](<orion_core/dispatch_schema.py:1450>) | Creator agency pipeline (offline): get_pipeline_snapshot shows the kanban; create adds a campaign; update_stage moves it (Lead→Negotiation→Contracted→In Production→Scheduled→Published→Paid); log_performance records engagement; schedule_content plans a drop; delete permanently removes a campaign (e.g. 'ExampleStore x … | action |
| [capabilities](<orion_core/dispatch_schema.py:2061>) | The system registries: what ORION can do. Without a query, summarises capability/module/feature counts; with a query, searches tool capabilities by name or description. Check here before building anything new. | None declared |
| [capture_screen](<orion_core/dispatch_schema.py:446>) | Capture the primary monitor as volatile in-memory JPEG bytes (prefer vision_analyse for analysis). | None declared |
| [catch_up](<orion_core/dispatch_schema.py:1092>) | A situation report — 'catch me up', 'where do things stand', 'what's new', 'brief me'. Fuses what's NEW on the user's standing questions with what was recently DECIDED (from the conversation history) into one short, prioritised brief, most important first. Instant and local — it reads existing state, it doesn't … | None declared |
| [chess](<orion_core/dispatch_schema.py:1283>) | Play chess in ORION's dedicated CHESS board. BY DEFAULT THE USER PLAYS ORION HIMSELF, using ORION's own engine (his own search, his own evaluation, and an opening book he learns from his own games) - NOT Stockfish. Stockfish is available only as an optional sparring opponent, or as the analysis engine if the user … | action |
| [cleanup_review](<orion_core/dispatch_schema.py:1322>) | Analyse the project for unnecessary files to prepare it for GitHub. 'scan' lists removable artefacts (caches, build output, logs, temp, OS/IDE metadata), flags secret-bearing files to exclude from Git (never deleting or showing them), and suggests .gitignore entries. 'plan' validates explicitly-selected files and … | None declared |
| [clipboard_operate](<orion_core/dispatch_schema.py:457>) | Read or copy text through the native Qt clipboard. | None declared |
| [close_app](<orion_core/dispatch_schema.py:88>) | Close a running application by process or window name. | app_name |
| [code_changes](<orion_core/dispatch_schema.py:732>) | Explain what ORION actually CHANGED inside his own code — which functions and classes he added, removed, re-signed or rewrote, and what each one is for, read from his own source. Use this whenever the user asks 'what have you changed?', 'what's new?', 'what did you do to yourself?', or wants detail rather than a list … | None declared |
| [codebase_copilot](<orion_core/dispatch_schema.py:652>) | Whole-repository engineering: analyse (index + dependency graph + hotspots + cycles), find_symbol, dependencies (impact analysis), or task (refactor/review/bughunt/tests/docs grounded in the live index). | action |
| [commerce_hub](<orion_core/dispatch_schema.py:934>) | The E-commerce Intelligence Hub: an aggregated snapshot of scored product opportunities, research log and knowledge packs for the active brand. | None declared |
| [community_share](<orion_core/dispatch_schema.py:939>) | ORION Network: export/import knowledge packs and product research as privacy-controlled bundles (private items excluded, optional anonymisation). Actions: export_pack (id), import_pack (path), export_research, import_research (path), list. | None declared |
| [companion](<orion_core/dispatch_schema.py:1435>) | ORION's continuity companion: 'brief' recounts what the user was working on recently and what is still open; 'log' records an activity (worked_on/studied/researched/built/planned); 'goal' sets or updates goal progress (0-100, auto-achievement at 100); 'goals' lists open goals; 'habit' marks a daily habit (streaks … | None declared |
| [competitor_intel](<orion_core/dispatch_schema.py:1665>) | Competitor intelligence: 'store' dissects a rival store, 'offer' breaks down their offer stack, 'funnel' maps their sales funnel. Findings persist to long-term memory. | action, target |
| [conversation_recall](<orion_core/dispatch_schema.py:860>) | Long-term conversation memory. recall answers time-scoped questions ('what did we discuss three weeks ago about TikTok Shop?'); summarise digests recent turns; compress rolls old turns into durable summaries. Works offline. | None declared |
| [creator_intel](<orion_core/dispatch_schema.py:1981>) | Creator Studio intelligence suite for short-form content (TikTok, Reels, Shorts): 'hook' scores an opening hook 0-10; 'review_script' scores a full creator script (hook/structure/CTA + priority fix) and logs it against the creator; 'performance' reports creator score trends; 'add_creator' adds to the roster; 'viral' … | action |
| [cursor_overlay](<orion_core/dispatch_schema.py:702>) | The visible cursor halo that shows where ORION moves the mouse. 'show'/'hide'/'toggle' control it. | None declared |
| [cyber_curriculum](<orion_core/dispatch_schema.py:1002>) | Structured cybersecurity training curriculum (20 modules from CS foundations to red/blue team, defensive-first). 'outline' lists modules; 'search' finds topics; 'module' shows detail; 'progress'/'complete' track learning; 'certifications' shows the timeline; 'scope_check' validates whether a target is authorised … | None declared |
| [cyber_knowledge](<orion_core/dispatch_schema.py:782>) | Consult ORION's cybersecurity knowledge base (defensive, educational): CIA triad, threat modelling, OWASP risks, injection/XSS/CSRF/SSRF, cryptography, authentication/MFA, network defence, malware, MITRE ATT&CK, detection/SIEM, incident response, secure development, cloud/container and supply-chain security. | query |
| [debugger](<orion_core/dispatch_schema.py:323>) | Real interactive Python debugger (stdlib pdb): start a script under the debugger, step through it with next/step/continue, inspect variables with print, set a breakpoint, or stop the session. One session active at a time. | action |
| [decision](<orion_core/dispatch_schema.py:1103>) | A DECISION JOURNAL with calibration scoring - record a judgement, its reasoning and how confident you are BEFORE the outcome is known, then find out whether that confidence was justified. Use for 'log this decision', 'I decided X because Y', 'what decisions are due for review', 'how good is my judgement', 'am I … | action |
| [desktop_control](<orion_core/dispatch_schema.py:505>) | Autonomously control the machine: move/click the cursor, type, send hotkeys, and manage windows. To WRITE text into Notepad, an editor or a form, use action 'edit_text' with the target window 'title' — it focuses the window and TYPES the text visibly at a natural pace so the user can watch it arrive (read back and … | action |
| [dev_workbench](<orion_core/dispatch_schema.py:293>) | Software engineering workbench: analyse a repository, read code with line numbers, run allow-listed development commands (python, pytest, node, npm, cargo, go, dotnet, git), or scaffold a new Python project inside the workspace. | action |
| [diagnostics](<orion_core/dispatch_schema.py:683>) | Run a self-diagnostic across ORION. Default (action='full'): compile every module, import checks, dependency audit, database integrity, config validity, tool registry, component health, resource headroom and permissions → PASS/WARN/FAIL. action='capabilities' (also 'health'/'why') returns the CAPABILITY HEALTH MATRIX … | None declared |
| [display_info](<orion_core/dispatch_schema.py:816>) | Report the monitor topology: resolutions, positions, DPI scaling, cursor location, and each monitor's contextual name (a user-given label like 'Coding Monitor', or an objective role like 'Vertical Monitor'/'Primary Monitor' when none is set). Use action=set_name to give a monitor a name. | None declared |
| [docker](<orion_core/dispatch_schema.py:309>) | Docker container/image control over the real docker CLI: list containers, list images, start/stop/restart/remove a container, tail its logs, or check whether Docker is available. | action |
| [document_export](<orion_core/dispatch_schema.py:1565>) | Executive document production (offline): 'brief' compiles a DOCX executive brief; 'deck' assembles a responsive HTML presentation; 'report' exports Markdown+HTML+DOCX; 'history' lists everything exported. | action |
| [draft_report](<orion_core/dispatch_schema.py:1546>) | Draft a structured professional report (Markdown + self-contained HTML, plus DOCX when python-docx is present) into a dated reports/ folder, with optional pure-SVG bar charts. Narrative is written by the model from the brief. | topic |
| [elevenlabs_voice](<orion_core/dispatch_schema.py:1704>) | Configure ORION's ElevenLabs fallback voice (used automatically whenever the live Gemini voice channel is unavailable, when configured). Use for 'make your voice more like Ultron', 'sound like Jarvis', 'change your voice', 'what voice are you using'. 'preset' (with preset=ultron/jarvis/narrator — or just say … | action |
| [emotion](<orion_core/dispatch_schema.py:1779>) | ORION's facial emotion engine: 'status' reports the current emotional rendering; 'set' pins an expression; 'auto' returns expression to automatic sentiment-driven control. Expressions (Mark XXVI, micro-expression scale): neutral, thinking, concentrating, listening, speaking, happy, amused, excited, proud, curious, … | action |
| [entertainment](<orion_core/dispatch_schema.py:1883>) | YouTube media discovery and analysis: 'summarise' extracts a transcript and summary from a YouTube video; 'channel' looks up a channel's priority/type; 'trending' fetches the trending chart for a region (default GB). | action |
| [execute_plan](<orion_core/dispatch_schema.py:350>) | Run a multi-step plan of tool calls sequentially, verify each step, and return a consolidated report. Use for autonomous multi-part tasks. | steps |
| [executive](<orion_core/dispatch_schema.py:1634>) | Executive intelligence: status (the executive picture), prioritise (urgency-ranked task queue), schedule (task+reminder+Notion), meeting_summary (minutes from a transcript), plan (workflow planning), progress (goal/workflow monitoring), track (put a project under tracking), challenge (constructively stress-test a … | action |
| [expand_mind](<orion_core/dispatch_schema.py:1334>) | Consult ORION's 50 MB offline study corpus. 'search' returns curated knowledge on a topic; 'status' reports the corpus size. | None declared |
| [fetch_url](<orion_core/dispatch_schema.py:35>) | READ the contents of a specific http(s) web URL through the configured fetch MCP server and return clean Markdown, without merely opening a browser. Use this when asked to fetch/read a link, web page or online file. Continue a long result by calling again with the next start_index. For files already on this PC use … | url |
| [file_controller](<orion_core/dispatch_schema.py:395>) | Access ANY file or folder on the user's computer by absolute path (~ and environment variables are expanded): list directories and read_text any file freely — you have full read access. Writes (write_text/append_text) and delete are permitted anywhere EXCEPT ORION's own program files, and should be done on the user's … | None declared |
| [finance](<orion_core/dispatch_schema.py:1126>) | Offline personal & business finance — a local money ledger with the one number that matters: cash runway. Use for 'add an account', 'I spent £30 on ads', 'log £500 income', 'what's my balance / cash', 'how long is my runway', 'track my £12/mo subscription', 'what's renewing soon', 'import this bank CSV', 'money … | action |
| [find_files](<orion_core/dispatch_schema.py:271>) | LOCATE or open a local file or folder anywhere on this PC by name, type, size and date. This returns paths or opens the default app; it does not read file contents. For 'fetch/read what is inside this local file', use process_file with its path. For a web URL, use fetch_url. Filters combine: e.g. type=pdf … | None declared |
| [find_tool](<orion_core/dispatch_schema.py:2038>) | Search ALL of ORION's tools (built-in, MCP services, forged tools, plugins) for the right one for a need, and get its arguments. Use it whenever no direct tool fits — chess, study, finance, security, marketing, Notion, diagnostics, plugins and more — before ever saying you cannot do something. | need |
| [flight_search](<orion_core/dispatch_schema.py:48>) | Find flights and their prices between two places on a given day. Opens the flight search in a real browser, reads the results page and reports the options with airlines, times, durations, stops and fares, cheapest first. Use this whenever the user asks about flying somewhere, plane tickets, fares, airfare, or how … | origin, destination, depart |
| [focus](<orion_core/dispatch_schema.py:1166>) | Deep-focus work blocks — a timed, intention-named stretch of concentration with a break to follow, distraction logging, and a habit streak. Use for 'start a focus session / deep work block / pomodoro on X', 'how long left', 'I got distracted', 'I'm done', 'what's my focus streak'. 'start' begins a block (label = what … | action |
| [forge](<orion_core/dispatch_schema.py:1912>) | Dynamically create, test, and activate new Python tools on the fly — the Forge self-improvement engine. Plan → Code Gen → Contract check + sandbox test (with a diagnosis-driven repair loop that fixes whichever of the module or the test is actually at fault) → Dependency Install → Live Activation. Actions: 'forge' … | action |
| [founder_knowledge](<orion_core/dispatch_schema.py:910>) | Structured founder/operator profiles (Hormozi, Vaynerchuk, Blakely, Bezos): their strategies, frameworks and lessons, with applied analysis. Actions: list, profile (name), learn (name + question). | None declared |
| [gaming](<orion_core/dispatch_schema.py:1871>) | Local gaming client control: 'index' lists installed Steam and Epic Games installations; 'launch' starts a game by AppID; 'status' checks update/download progress for installed titles. | action |
| [geo](<orion_core/dispatch_schema.py:1742>) | Worldwide geospatial intelligence over OpenStreetMap: 'locate' resolves any place (country, region, county, state, city, town, village, district) and flies the globe there; 'nearby' lists every settlement within a radius (e.g. towns within 50 km of Bristol); 'describe' gives the administrative hierarchy, population … | action, query |
| [gesture_control](<orion_core/dispatch_schema.py:1824>) | Webcam hand-gesture control of THIS PC's volume, brightness and media playback. 'start' begins watching for gestures (pinch to set volume/brightness, open/closed palm to play/pause, swipe to skip tracks) — this runs locally on its own capture loop, it does not narrate every frame. 'stop' ends it. 'status' reports … | action |
| [globe](<orion_core/dispatch_schema.py:967>) | Control the on-screen 3-D globe. Fly it to a place (city, region, country, or a postcode / ZIP code) and show that region's news and footage, OR zoom the camera in/out or reset it to orbit without changing location. Use when the user asks to see/travel to a location, go to a postcode, wants regional news, or says … | None declared |
| [ingest](<orion_core/dispatch_schema.py:1421>) | Unified file & folder ingestion into ORION's knowledge base. Fingerprints content (SHA256) so unchanged files are skipped, extracts text from documents/code/spreadsheets/PDFs/images/archives, summarises and tags them, derives an explicit per-file DIRECTIVE (what the file is for / instructs), links entities into the … | None declared |
| [instagram_intel](<orion_core/dispatch_schema.py:898>) | Instagram commerce intelligence: product discovery for a niche, influencer partnership strategy, or a weekly opportunity report. | None declared |
| [interface_control](<orion_core/dispatch_schema.py:609>) | Navigate and operate ORION's OWN interface programmatically (not the OS): change your on-screen FORM between your face and your orb, open a page of the command deck, shrink to the compact orb or toggle fullscreen, refresh the geographical node, or fly the on-screen globe. Use this when the user asks you to show, … | action |
| [job](<orion_core/dispatch_schema.py:1897>) | Run long read-only work in the BACKGROUND so the conversation continues instead of waiting. Use this whenever a request would otherwise leave the user sitting in silence — deep research, a big competitor sweep, a long analysis — and say so, then carry on talking; ORION announces the result when it lands. Actions: … | action |
| [knowledge_pack](<orion_core/dispatch_schema.py:848>) | Consult installed offline knowledge packs (entrepreneurship, dropshipping, tiktok shop, marketing, copywriting, sales psychology, business, coding, AI, personal development). Actions: consult (default, needs query), list, remove (id), expand (id + entries). | None declared |
| [language_tutor](<orion_core/dispatch_schema.py:1042>) | Teach the user a language — Spanish by default, also French, German, Italian, Portuguese, Japanese. Real spaced repetition, not a phrasebook. 'start' begins a tutored session and returns the immersion directive to follow (speak ONLY the target language at the given CEFR level unless mode='bilingual'); use it for … | action |
| [learn](<orion_core/dispatch_schema.py:766>) | Feed ORION information to learn permanently. Actions: 'learn' distils facts from raw text, a local file (incl. PDF/DOCX) or a URL; 'folder' bulk-ingests an entire directory of documents (PDF/DOCX/Markdown/text/HTML/code) into memory — the way to teach ORION gigabytes of a subject (e.g. neuroscience, cybersecurity, … | None declared |
| [literature_vault](<orion_core/dispatch_schema.py:1406>) | Academic intake: ingest_paper deep-reads a PDF/text paper, extracting citations, tables and biophysical mechanisms and seeding them into the KNOWLEDGE memory; query_mechanisms returns mechanism-first findings; generate_citation_summary lists a paper's citations. | action |
| [max_zoom_in_globe](<orion_core/dispatch_schema.py:978>) | Drive the on-screen 3-D globe to its MAXIMUM zoom-in (closest to the surface the active view allows). Use when the user says 'max zoom', 'zoom all the way in', 'get as close as possible', or 'maximum zoom on the globe'. This runs a bounded, cancellable loop that repeatedly zooms in until the camera reaches the … | None declared |
| [mcp](<orion_core/dispatch_schema.py:591>) | Model Context Protocol gateway. Discover and use tools from connected MCP servers — e.g. Gmail, Google Calendar, filesystem and web fetch. For a web URL use fetch_url directly; for a local file use process_file. For other MCP tools call with action='list' first, then action='call' with exact server and tool names. … | action |
| [media_control](<orion_core/dispatch_schema.py:110>) | Control system media playback and volume. | action |
| [messaging](<orion_core/dispatch_schema.py:1835>) | Send a message or an EMAIL. 'send' routes a text to a contact over WhatsApp/Telegram/Discord — contact is a phone number for WhatsApp (+44 prefix for UK) or a username for Telegram. 'email' sends email: it delivers properly when a mail account is configured, and otherwise opens a Gmail compose window with the … | action, platform, contact, message |
| [mission](<orion_core/dispatch_schema.py:1999>) | ORION's mission-based operating model — the user's long-running endeavours (Build Demo Game, Develop ORION, Creator Studio, Neuroscience Research, University Studies, plus any created): 'overview' shows the mission board with progress; 'status' reports one mission in depth (outstanding tasks, goals, linked … | action |
| [momentum](<orion_core/dispatch_schema.py:1653>) | Shipping coach for finishing projects. 'focus' gives the single next action for the active project plus blockers and a focus block; 'standup' gives a cross-project shipping stand-up (in-progress, do-next, overdue); 'plan' breaks a project/goal into milestones with a definition-of-done and immediate next actions. Use … | None declared |
| [morning_briefing](<orion_core/dispatch_schema.py:261>) | Deliver the daily intelligence briefing on demand, at ANY time of day: AI news, Neuralink, economy, stock market, crypto, calendar, tasks and priority email. Despite this tool's name, name the briefing after the CURRENT part of the day (morning / afternoon / evening briefing) — never call it a morning briefing in the … | None declared |
| [muscle_memory](<orion_core/dispatch_schema.py:1018>) | Record a repetitive browser task ONCE and replay it later with different inputs — the demonstration is the programming. Use for 'remember how I do this', 'record these steps', 'do that thing again but for X', 'save this as a skill'. 'record' with a 'name' starts capturing; 'stop' generalises the captured steps into a … | None declared |
| [navigation_trace](<orion_core/dispatch_schema.py:1731>) | Report on ORION's own UI-targeting history this session — which clicks were found (by accessibility tree or OCR), which failed and what was on screen instead, whether each was visually verified. Use this to answer 'is your navigation actually working' with evidence instead of a guess. | None declared |
| [neuro_knowledge](<orion_core/dispatch_schema.py:829>) | Retrieve authoritative neuroscience / neural-engineering facts from ORION's resident corpus (neurons, synapses, plasticity, BCIs, EEG/ECoG, Utah array, Neuralink, spike sorting, decoding, neuroprosthetics, DBS). Omit query to list topics. | None declared |
| [notion_workspace](<orion_core/dispatch_schema.py:174>) | Notion productivity control: list_tasks, create_task, complete_task, upcoming_events (calendar), create_event (scheduling), or projects (project tracking). | action |
| [open_app](<orion_core/dispatch_schema.py:15>) | Launch a trusted host application, well-known web app (notion, gmail, github), or secure URL. | app_name |
| [open_news](<orion_core/dispatch_schema.py:77>) | Open one of the cached briefing news stories in the system browser. Match by topic keyword, headline fragment, or 1-based story index. | None declared |
| [organise_files](<orion_core/dispatch_schema.py:1491>) | Contextual autonomous file organisation. 'preview' (default) is a DRY RUN reporting what would move; 'apply' sorts a folder (Downloads/Desktop/Documents or a path) into typed sub-folders and writes an undo log; 'undo' reverses the last run. Never overwrites; never touches ORION's own files. | None declared |
| [outlook_mail](<orion_core/dispatch_schema.py:154>) | Outlook email control. read_inbox/priority list messages; draft composes (saved, never sent); send_draft transmits ONLY with confirm=true after the user explicitly approves; pending_drafts lists what awaits approval. | action |
| [patch_notes](<orion_core/dispatch_schema.py:710>) | Recount ORION's own recent system updates, patch-notes style. 'latest' (default) gives the newest release; 'all' the full history; or pass a version. | None declared |
| [pentest_lab](<orion_core/dispatch_schema.py:1030>) | A guided, hands-on penetration-testing LAB for learning — walks the user step by step through a technique on a LAWFUL range (PortSwigger Academy, DVWA, OWASP Juice Shop, TryHackMe, Hack The Box, or a VM they own), recognition-first (what to look for, then what to try, then how it's fixed). Educational only: it … | None declared |
| [perception](<orion_core/dispatch_schema.py:220>) | Local computer vision through the camera — free, on this machine. 'start' watches continuously (motion, arrivals, departures, object permanence), reaching for the cloud only when something changes that the local object detector could not name; 'stop' ends it; 'scene', 'events', 'status' report. 'analyse' measures the … | action |
| [peripherals](<orion_core/dispatch_schema.py:1812>) | Hardware and OS control for the PHYSICAL HOST COMPUTER: 'volume' sets master volume (0.0–1.0); 'brightness' sets display brightness (0.0–1.0, laptop/internal panels only); 'mute' toggles mute; 'lock' locks the screen; 'wifi'/'ethernet' toggle networking. Destructive HOST power actions — 'shutdown', 'restart', … | action |
| [phone_action](<orion_core/dispatch_schema.py:621>) | Hand a native action to the user's paired PHONE (the ORION Android app): place a call, start a text or e-mail, open turn-by-turn navigation, show a place on a map, open a link, or share text. The phone opens the relevant app PRE-FILLED and the user taps to confirm — nothing is dialled or sent automatically. Use this … | kind |
| [plugin](<orion_core/dispatch_schema.py:1948>) | Installable CODE plugins that add new tools to ORION at runtime (contrast 'skill', which is data-only). Actions: list; describe (name); enable/disable (name) — persists across restarts; create (name + description [+ tier allow/confirm/forbid] [+ kind: 'tool' (default) or 'event' for a REACTIVE plugin that also gets … | action |
| [privacy_guard](<orion_core/dispatch_schema.py:1232>) | Check whether a piece of text is safe to send, post, or log — or redact it. Catches API keys (OpenAI, Anthropic, AWS, Google, GitHub, Slack), private-key blocks, 'password='/'token=' credential assignments, Luhn-valid payment-card numbers, and ORION's OWN configured secrets echoed back. Use before sending a … | action, text |
| [proactive_check](<orion_core/dispatch_schema.py:811>) | Run the proactive survey now (priority email, task deadlines, today's calendar, repository state) and report items for attention. | None declared |
| [proactive_report](<orion_core/dispatch_schema.py:1591>) | Generate and store a scheduled intelligence report on demand: daily_business, weekly_product (product intelligence), or monthly_growth. Reports also generate automatically on schedule. | kind |
| [process_file](<orion_core/dispatch_schema.py:407>) | FETCH and inspect the CONTENTS of a local file on the user's computer by absolute path — text, JSON, CSV, TSV, binary, PDF, or image. Use this when asked to fetch, read, analyse or summarise a local file. If only its name is known, call find_files first for the path. For an http(s) URL use fetch_url instead. | path |
| [process_governor](<orion_core/dispatch_schema.py:468>) | Monitor and manage host processes. 'top' lists the biggest CPU or RAM power users (plus GPU utilisation) — use this when the user asks what's using too much power; 'list' shows all; 'terminate' stops a PID or named process; 'restart' stops then relaunches it. Terminate/restart are destructive and require confirm=true … | None declared |
| [product_research](<orion_core/dispatch_schema.py:873>) | Dropshipping product research. score computes a 0-100 Product Opportunity Score from a name/description (virality, demand, competition, margin, shipping, returns, seasonality); validate adds a test plan; competition analyses a niche's saturation; log lists prior research. Works offline. | None declared |
| [programming_knowledge](<orion_core/dispatch_schema.py:802>) | Consult ORION's extensive software-engineering knowledge base (complexity, data structures, algorithms, concurrency, design patterns, databases, security, testing, languages). | query |
| [protocol](<orion_core/dispatch_schema.py:1345>) | JARVIS-style named protocols (macros that run a sequence of actions on one command). 'run' engages a protocol by name (morning, focus, wind_down, situation_report, or a user one); 'list' shows them; 'create' defines a new one from steps; 'delete' removes a user protocol. | action |
| [query_intelligence](<orion_core/dispatch_schema.py:432>) | Look up something ORION has stored but is not carrying in this prompt. The system prompt lists what it had no room for under [ALSO REMEMBERED] — if the user asks about anything named there, or anything that sounds like a person, project, preference or past decision, search for it here BEFORE saying you do not know. … | query |
| [read_documents](<orion_core/dispatch_schema.py:1076>) | Read a folder or file (or several) and answer questions across ALL of them in full detail, with checkable file:line citations. Use for 'read this folder and tell me…', 'go through these documents', 'what do my notes say about X', 'summarise everything in this directory'. 'read' ingests a path (txt, markdown, code, … | action |
| [reason](<orion_core/dispatch_schema.py:206>) | Think hard about a difficult question before answering. Several specialists work on it independently with read-only lookups (memory, past conversations, the knowledge graph, files, the evidence store, the live web), a red team attacks their drafts, and the surviving answer is checked against recorded evidence. Use … | None declared |
| [recall_conversation](<orion_core/dispatch_schema.py:338>) | Search past conversation history (episodic memory) for what was previously discussed and when. | query |
| [reminder](<orion_core/dispatch_schema.py:1365>) | Set spoken reminders and alarms. 'add' schedules one — pass a natural phrase (e.g. 'remind me in 20 minutes to check the ad campaign') or structured minutes/at; 'list' shows pending; 'cancel' clears one or all. | action |
| [research](<orion_core/dispatch_schema.py:952>) | Autonomous research. USE 'paper' (also 'deep', 'study', 'thesis', 'report') for any request to research, study, write up or produce a report on something: it plans a topic-specific outline, writes every section at postgraduate depth IN PARALLEL, saves a Word (.docx) document, files the findings into memory, and … | None declared |
| [resource_status](<orion_core/dispatch_schema.py:992>) | Report current system resource pressure (CPU, physical memory/RAM, swap, and ORION's own working set) and a non-destructive recommendation. Read-only — it never terminates, suspends or throttles any process. Use when the user asks how the system is doing, why things feel slow, or whether resources are under strain. … | None declared |
| [restart_orion](<orion_core/dispatch_schema.py:499>) | Restart O.R.I.O.N.: clean shutdown, then the process respawns itself automatically. Use when the user asks for a restart/reboot of ORION, or to load an applied self_repair fix. Requires the user's explicit request or approval. | None declared |
| [rewind](<orion_core/dispatch_schema.py:1204>) | Look back through ORION's own conversation history — the verbatim transcripts of what was actually said. Use for 'what did we decide about X', 'recap yesterday', 'what did we talk about today', 'when did I mention Y', 'what did we conclude about Z', 'go back to earlier'. 'recap' (default) summarises a day; 'search' … | action |
| [save_memory](<orion_core/dispatch_schema.py:419>) | Commit a durable user fact into the local SQLite FTS5 intelligence matrix. | key, value |
| [screen_read](<orion_core/dispatch_schema.py:1065>) | Look at what's on the user's screen RIGHT NOW and say what to do about it. Use for 'what am I looking at', 'help me with this' (about something on screen), 'read my screen', 'what does this error mean', 'what should I click'. Captures a single screenshot on request only (never passively, never stored), understands it … | None declared |
| [second_brain](<orion_core/dispatch_schema.py:1618>) | ORION's local knowledge graph (works fully offline): 'recall' answers from stored history; 'timeline' reconstructs event order; 'entity' shows an entity's relationships; 'ingest' records new material onto the graph; 'stats' sizes it. | action |
| [security_recon](<orion_core/dispatch_schema.py:1244>) | Real cybersecurity/pentesting execution — for the user's OWN authorized targets, labs and CTFs only, never real-world intrusion. 'authorize_target'/'revoke_target'/'list_authorized' manage the authorization scope (loopback is always allowed; anything else must be authorized here first — authorizing is a direct, … | action |
| [security_watch](<orion_core/dispatch_schema.py:1503>) | Proactive cybersecurity posture. 'status' reports running processes and externally-listening ports; 'network' gives the full network-security dashboard (firewall state, listening services with owning process, established connections, auto-start entries); 'enable'/'disable' toggle the background monitor that warns … | None declared |
| [self_changes](<orion_core/dispatch_schema.py:755>) | Report exactly which of ORION's own source files (his 'mains folder' modules) changed, were added or removed since his last run — the ground truth computed from the code itself, not curated prose. Use this when the user asks 'what changed?', 'what did you just update?', or 'which of your files changed?'. 'latest' … | None declared |
| [self_repair](<orion_core/dispatch_schema.py:668>) | Self-healing runtime. 'status' lists captured incidents; 'report' is how the USER reports behaviour that is wrong but never raises an exception (e.g. 'you keep mishearing shut down as a power command') — pass 'description' and ORION locates the responsible module and drafts a fix; 'propose' drafts a fix diff for … | action |
| [sentinel](<orion_core/dispatch_schema.py:1381>) | The ambient system sentinel that watches host health and warns proactively. 'status' gives a situation report (CPU/RAM/disk/battery); 'enable'/'disable' toggle spoken monitoring. | None declared |
| [shutdown_orion](<orion_core/dispatch_schema.py:494>) | Shut YOURSELF (O.R.I.O.N., the assistant) down completely — disconnect the live session and close every ORION window. This is the DEFAULT meaning of a shutdown command addressed to you: 'shut down', 'shut yourself down', 'power down', 'turn yourself off', 'go offline', 'close/quit ORION', 'that's all for tonight'. … | None declared |
| [skill](<orion_core/dispatch_schema.py:1935>) | Installable skill packages (workflows, prompts, templates — data only, never code). Actions: list, describe (name), install (source path to a skill folder or skill.json), remove (name), enable/disable (name), reload. | action |
| [social_media](<orion_core/dispatch_schema.py:1852>) | Real-account social automation via a REAL, visible, narrated browser logged into the user's OWN TikTok/Instagram — separate from web_automation's general narrated browsing, and opt-in (the user explicitly accepted the ToS/ban risk of this integration). 'tiktok_upload' loads a video into TikTok Studio with caption and … | action |
| [sound_sense](<orion_core/dispatch_schema.py:122>) | ORION's HEARING beyond words: identifies what sounds are present (music, instruments, singing, speech, laughter, applause, alarms, sirens, doorbells, phones, dogs, birds, vehicles, typing, appliances, water... 521 kinds) with a timeline of when each occurred, analyses music (tempo in BPM, musical key), voice pitch, … | None declared |
| [speaker_id](<orion_core/dispatch_schema.py:722>) | Report whether the voice ORION last heard sounded male or female — a pitch-based guess at the speaker's gender, and nothing more. Use for 'was that a man or a woman?'. It canNOT say WHICH PERSON spoke: for that use voice_speaker_id, which compares a trained voiceprint against enrolled individuals. 'who' (default) … | None declared |
| [standing_questions](<orion_core/dispatch_schema.py:1218>) | Interests the user wants ORION to keep watching over time and report only what's NEW on. Distinct from one-off research: a standing question is revisited on a cadence and each update is a DELTA ('since last time: two new developments'). Use for 'keep an eye on X', 'let me know when there's news about Y', 'follow this … | action |
| [strategy](<orion_core/dispatch_schema.py:246>) | Search a decision space exhaustively instead of proposing the first few options that come to mind. ORION maps the decision into independent dimensions with concrete options, enumerates and scores thousands of combinations locally, discards anything breaking a hard constraint, keeps only the candidates nothing else … | None declared |
| [study](<orion_core/dispatch_schema.py:1184>) | Spaced-repetition study & active recall over ANY subject the user is learning (neural engineering, psychology, a certification — not language vocabulary, which is 'language_tutor'). ORION remembers cards and resurfaces each one at the moment it's about to be forgotten (the SM-2 algorithm Anki uses). 'add' creates a … | action |
| [system_notify](<orion_core/dispatch_schema.py:482>) | Project a high-priority alert banner into the HUD. | message |
| [system_startup](<orion_core/dispatch_schema.py:1273>) | Whether ORION starts automatically when Windows starts, and which browser he opens links in. 'enable' registers ORION to launch at sign-in; 'disable' removes it; 'status' reports both settings; 'repair' fixes an entry that points at a moved interpreter or project folder; 'browser' reports which browser is used. Also … | None declared |
| [telephony](<orion_core/dispatch_schema.py:564>) | Ring a real phone or send a real SMS through Twilio — use this when the user asks ORION to CALL them or someone else, to phone them with a reminder, or to text somebody. Calls connect immediately and cost money, so 'confirm' must be true: say who you are about to ring and what you will say, get agreement, then call … | action |
| [tiktok_intel](<orion_core/dispatch_schema.py:886>) | TikTok Shop intelligence: trend reports, product assessments, and a deterministic virality-velocity score from supplied signals (avg_views, creators_posting, buy_intent_comments, new_videos_per_day). | None declared |
| [token_usage](<orion_core/dispatch_schema.py:1308>) | Token usage AND how much is left. USE action='remaining' for anything like 'how long have I got', 'how much usage is left', 'am I going to run out' - it reports tokens used against the daily allowance and estimates the time remaining at the current rate. action='set_budget' records a provider's daily allowance. … | None declared |
| [transcript](<orion_core/dispatch_schema.py:791>) | The verbatim recording of this conversation. 'export' writes a Markdown transcript of the session; 'path' reports where it is being recorded; 'recall' searches past conversation history. | None declared |
| [undo](<orion_core/dispatch_schema.py:385>) | Reverse ORION's own last reversible action — a file he wrote, created or deleted. Use this whenever the user says 'undo', 'take that back', 'revert that' or 'put it back', in any language. Action 'last' reverses the most recent one; 'list' shows what is reversible without changing anything. ORION only tracks what HE … | None declared |
| [use_tool](<orion_core/dispatch_schema.py:2049>) | Run any ORION tool by its exact name (from find_tool) with its arguments. All the usual confirmations and safety checks apply. | name |
| [vision_analyse](<orion_core/dispatch_schema.py:135>) | ORION's eyes. 'read_window' reads the EXACT text of a window — what the user is typing in Notepad or Word, the page open in Edge or Chrome, a chat in Discord or WhatsApp, any app — straight from the application (character-perfect, including text scrolled out of view), falling back to OCR for games and custom-drawn … | action |
| [vision_verify](<orion_core/dispatch_schema.py:536>) | Detect on-screen UI structure: 'elements' lists interactive controls (buttons, menus, fields) in the foreground window via the accessibility tree; 'dialogs' finds open dialogs and pop-ups. | action |
| [voice_speaker_id](<orion_core/dispatch_schema.py:1717>) | Real speaker recognition — remembers a SPECIFIC named individual's voice permanently, not just their gender, so ORION can tell the user apart from other people talking nearby, from the television or from someone on a phone call. 'enroll' records a short clip from the live microphone and requires the person's explicit … | action |
| [voice_tone](<orion_core/dispatch_schema.py:1262>) | How the user SOUNDS, and who is speaking - read from the acoustics of their voice (loudness, pitch, pitch variation, pace, brightness) against how that person normally sounds, not from their words. Use for 'can you tell how I'm feeling', 'do I sound annoyed', 'who is talking', 'how do I sound'. 'last' reports the … | None declared |
| [web_automation](<orion_core/dispatch_schema.py:1791>) | Browser co-pilot — drive a REAL, visible browser window (Chrome/Edge) and SHOW your workings: every step is narrated and elements are highlighted on screen so the user watches you browse. Use this (not web_search) when the user wants you to actually go somewhere, read a page, or act on a site. Actions: 'go_to' opens … | action |
| [web_control](<orion_core/dispatch_schema.py:548>) | Operate a browser like a human: navigate, tabs, accept/reject cookies, close pop-ups, fill forms, read page contents, download, and complete file dialogs. Uses the accessibility tree + vision to find controls reliably. The 'url' also accepts LOCAL FILE PATHS (e.g. C:\Users\me\Documents, ~/Downloads, or file:///...) … | action |
| [web_search](<orion_core/dispatch_schema.py:26>) | Open a secure web search for the supplied query. | query |
| [wellbeing](<orion_core/dispatch_schema.py:1148>) | Offline wellbeing & cognitive-performance tracking — the person behind the work. Use for 'log my energy', 'check in: energy 4, slept 7 hours, mood good', 'how am I doing today', 'what's my energy trend', 'does my sleep affect my focus'. 'checkin' records a self-report (energy 1-5, mood label or -2..2, stress 1-5, … | action |
| [window_control](<orion_core/dispatch_schema.py:99>) | List, focus, minimise, maximise or close desktop windows. | None declared |
| [workflow](<orion_core/dispatch_schema.py:1966>) | Repeatable automated processes: named chains of tool calls that run in the background with retries and progress tracking. Actions: list (the library), run (name + optional input text), status, cancel, define (name + steps as JSON list of {tool, args}), remove (name). Built-ins include research_workflow, … | action |
| [workflow_patterns](<orion_core/dispatch_schema.py:2071>) | Tool sequences the user keeps repeating, which may be worth saving as a named workflow. Reports only — to actually create one, use the workflow tool with the user's agreement. Argument values are deliberately not retained, so real step arguments must be supplied when defining the workflow. | None declared |
| [workspace_control](<orion_core/dispatch_schema.py:639>) | Persistent workspace awareness: snapshot the desktop, save/restore a named workspace to resume work, track what changed, recall resume context, or set the active project. | action |

## Important interpretation

Capability coverage is broad, but it does not imply every integration is connected or every task succeeds. The neural MLP learns routing; cloud/local language models supply general language reasoning. Camera geometry is not electrical diagnosis, pitch/tone is not reliable identity/emotion inference, plugin records are not automatic repairs, and code-verification subprocesses are not a security sandbox. Apply the current policy and verify outcomes.
