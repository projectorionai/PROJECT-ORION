# O.R.I.O.N. Mark XXVI — Engineering Roadmap & Workload Specification

**Status:** Draft for execution · **Baseline:** Mark XXV, 4,378 tests green · **Author role:** Lead AI Architect / Principal Systems Engineer

This specification is biased towards the vulnerabilities enumerated in §7 of the Mark XXV capability inventory. It is written as modular extensions over the existing bus/dispatcher/engine substrate; nothing here rewrites a working subsystem. Every deliverable states its conceptual test strategy, its offline-degradation path, its `qasync` safety posture, and — for visual work — its headless CI verification method.

---

## Governing invariants (apply to every deliverable)

1. **Zero regression.** New behaviour is additive and feature-flagged. The full 132-tool `handler_table` and `TOOL_DECLARATIONS` remain intact and individually dispatchable; any new routing layer is a *pre-filter*, not a replacement, and defaults to pass-through when its flag is off.
2. **Offline-first degradation.** Every cloud call sits behind a capability probe with a deterministic local fallback: Gemini Live → SAPI5/Whisper + local TTS; Claude/Gemini reasoning → Ollama (Llama-3.1 / Qwen2.5); cloud embeddings → local MiniLM or BM25.
3. **`qasync` safety.** All long-running work is an `asyncio` task or an existing `JobManager` job. The PyQt6 paint thread is never blocked; cross-thread state moves over the `OrionBus` signal layer only. (The qasync loop *is* the GUI thread, so a single shared engine instance is safe — no cross-thread SQLite handles required.)
4. **Visual CI.** Qt-widget surfaces render to a `QImage` buffer for structural assertion + tolerance pixel-diff; WebGL surfaces are verified by JS-state inspection through an offscreen `QWebEnginePage` plus baseline pixel-diff. No visual change is merged on headless-blind faith.

---

## Phase 1 — UI & Integration Catch-up

**Focus:** connect the orphaned `study` (Mark XXIV) and `focus` (Mark XXV) engines to the PyQt6 Command Deck as a first-class page.

### 1.1 The COGNITION deck page

**Placement.** Introduce a new zone `LEARNING` in `UnifiedDashboard.ZONE_PAGES`, page name `COGNITION`. It is a Qt-widgets surface (not WebGL), so it is **excluded** from `_NATIVE_SURFACE_PAGES` and takes the standard `QGraphicsEffect` fade like every non-GL page. Page count moves 23 → 24.

**Composition (three panels in a `QSplitter`):**
- **Focus panel** — a `QPainter` ring showing the active block's elapsed/remaining (`FocusSession.remaining_minutes`), start/break/cancel controls, live streak (`FocusEngine.streak`), and distraction counter. Break-due state (`FocusSession.is_break_due`) flips the ring to an amber "take your break" affordance.
- **Study panel** — due count (`StudyEngine.due`), deck list (`StudyEngine.decks`), and an inline review flow: prompt → *reveal* → grade buttons (Again/Hard/Good/Easy → SM-2 quality 1/3/4/5) driving `StudyEngine.grade`.
- **Insight panel** — mastery distribution (new/learning/mastered) and a 30-day retention sparkline, drawn with `QPainter` or the pure-SVG chart helper already used by `draft_report` (no charting dependency).

**State management across the bus.** The page **shares the single lazy engine instance** already held on the dispatcher (`dispatcher.study`, `dispatcher.focus`) — legitimate because qasync unifies the loop and GUI threads. New bus signals `bus.focus_changed` and `bus.study_changed` are emitted by `FocusEngine`/`StudyEngine` on any state mutation (start/grade/complete); the page subscribes and refreshes its read-model. A `QTimer` advances only the focus ring, **gated by `isVisible()`** per the established render-loop discipline, and paced by `AnimationBudget` so an idle page drops to idle-Hz and never taxes the audio deadline.

**Offline posture.** Fully local; the page reads SQLite (`config/study.db`, `config/focus.db`) and never touches the network.

**Test strategy (zero-regression).**
- *Pure reducer test:* `engine-state → view-model` is a free function; unit-test with synthetic sessions/cards (no Qt).
- *Headless render test:* construct the page offscreen (`QT_QPA_PLATFORM=offscreen`), inject a stub engine, `render()` to a `QImage`; assert (a) non-uniform output (it drew), (b) expected labels present via a `__probe_state__()` inspection hook, (c) tolerance pixel-diff against a committed baseline PNG.
- *Visibility-gating test:* assert the ring `QTimer` does not fire while `isVisible()` is `False` (reuse the pattern in `test_gui_visibility_guards`).

### 1.2 Wiring & migration
- Register `COGNITION` in `ZONE_PAGES`; add its builder to the page-construction loop; add the zone to the navigation ring.
- No tool changes — the page is a *view* over existing engines, so the tool suite and its tests are untouched.

---

## Phase 2 — Routing & Autonomy Overhaul

### 2.1 The 132-tool routing problem

**Root constraint (measured in code).** For the primary voice path, `live_worker.py` passes the *entire* `TOOL_DECLARATIONS` list into the Gemini **Live** `connect()` config once per session (`tools=[{"function_declarations": declarations}]`). Live sessions do **not** support cheap per-turn toolset mutation, so naïve semantic retrieval (which assumes a fresh tool list per request) cannot be applied to the Live channel without a session reconnect. The architecture must therefore differ by channel.

**Two-tier resolution architecture.**

**Tier A — Capability Facades (persistent Live session).** Collapse the Live model's *visible* surface from 132 concrete tools to ~18 stable **facade** declarations, one per domain (`desktop`, `web`, `vision`, `knowledge`, `research`, `learning`, `engineering`, `automation`, `business`, `security`, `creative`, `comms`, `system`, `memory`, `geo`, `finance`, `wellbeing`, plus a `route` escape hatch). Each facade takes `{intent: str, ...}`. When the Live model calls a facade, a **local sub-dispatcher** resolves `intent` to a concrete tool within that domain's 5–15 handlers by (i) a deterministic keyword/alias map first, then (ii) a fast **local router model** (Ollama) constrained to that domain's declarations only. The concrete handler in `handler_table` is then invoked unchanged. This keeps the Live prompt small and *stable* (no reconnect churn) and pushes fan-out to an offline-capable second stage.

**Tier B — Semantic Tool Retrieval (turn-based/text path + the sub-dispatcher).** Build a local embedding index of all 132 tool descriptions at start-up into `config/tool_index.db` (embeddings via a local sentence-transformer or an Ollama embed model; rebuilt only when the schema hash changes). At turn time, embed `query ⊕ recent-context ⊕ active-deck-domain`, retrieve top-k (k≈15), and pass only those declarations to the turn-based reasoner (Claude/Gemini/Ollama via the router). **Deterministic fallback:** BM25/keyword scan over descriptions when embeddings are unavailable — preserving offline-first.

**Context-aware pre-filter (applies to both tiers).** A `ToolResolver` composes hard filters *before* retrieval: MODE B hides cloud-only tools; a live focus block boosts `focus`/`study`; the foreground deck page boosts its domain; `security_*` execution tools are withheld unless an authorised target exists. The resolver is a **pure function** `resolve(query, state) -> list[declaration]`, making it fully unit-testable.

**Routing pipeline — flow sequence (turn-based path):**

```
user utterance ─▶ ToolResolver.resolve(query, state)
    1. state snapshot   = {mode, active_page, focus_active, auth_targets, recent_tools}
    2. hard_filter      = drop tools excluded by mode/authorisation
    3. boosts           = +score for domain of active_page, running focus, recent tools
    4. retrieve         = semantic_topk(query⊕context, candidates, k=15)   # BM25 if offline
    5. always_include   = {catch_up, reason, help}                        # safety floor
    6. declarations     = dedupe(retrieve ∪ boosts ∪ always_include)[:MAX_TOOLS]
       ─▶ reasoner(declarations)  ─▶ tool_call
       ─▶ handler_table[name](args)          # unchanged concrete handler
```

**Live path flow:** `Live(session_tools = 18 facades)` → model calls `learning{intent:"quiz me on action potentials"}` → `SubDispatcher["learning"].resolve(intent)` → deterministic map ∪ Ollama-constrained pick → `handler_table["study"]({action:"review"})`.

**Test strategy.**
- *Golden retrieval set:* a corpus of `(query → expected-tool ∈ top-k)` assertions; CI fails if recall@k on the golden set drops below threshold.
- *Purity/zero-regression:* with the resolver flag OFF, `resolve()` returns the full 132 (identical to today) — an explicit test pins this pass-through.
- *Facade coverage:* assert every concrete tool is reachable through exactly one facade sub-dispatcher (no orphans), computed from `handler_table` at test time.
- *Offline path:* force embeddings unavailable → assert BM25 fallback still returns the expected tool.

### 2.2 The Continuous Proactivity Engine

**Substrate.** Extend, do not replace, `proactive.py`. Today its loop emits `bus.banner`/`bus.log`/`bus.dashboard_event` and is gated by `proactive_policy.should_speak(kind, urgency) -> Decision`. Mark XXVI adds a **voice-capable** proactivity path.

**Design.** A single background `asyncio` loop (`ProactivityEngine`, `DEFAULT_INTERVAL_S` cadence, same shape as `CognitiveLoopManager`). Each tick it gathers **triggers** from existing state, no new polling of the OS:
- `focus.active().is_break_due()` → *break-due*
- `len(study.due()) ≥ N` → *review-backlog*
- `standing_questions.due()` → *watch-delta*
- `reminders` / calendar imminence → *time-critical*
- `sentinel` / `security_watch` alerts → *urgent*

Each candidate → `should_speak(kind, urgency)` → `Decision ∈ {SPEAK, BANNER, SUPPRESS}` honouring FOCUS state, quiet hours, the standby permission ledger, and a **per-kind cooldown** (debounce, so nothing nags). Approved `SPEAK` items are **coalesced** through `situation_report.build_report()` into *one* spoken digest, then emitted on a new `bus.proactive_utterance(text, priority)` signal. The voice pipeline consumes it and speaks via the **current** channel — Gemini Live `say` when connected, else `speech.speak_text` (local TTS) — so proactivity works fully offline.

**Interruption etiquette (hard rules).** Never speak while the microphone is capturing user speech or ORION is mid-utterance; queue and coalesce; obey a global "focus/do-not-disturb" flag; escalate SPEAK→BANNER when suppressed by etiquette so the signal is never silently lost.

**`qasync` safety.** The loop `await asyncio.sleep(interval)`; utterance emission is a non-blocking signal; TTS runs on the existing audio path. No paint-thread work.

**Test strategy.**
- *Pure decision function:* `decide(state_snapshot, clock) -> list[Action]`; unit-test synthetic states (focus overrun; 25 cards due; quiet hours; mic-active) → assert Actions, coalescing, and that a repeat within cooldown is suppressed.
- *Routing test:* assert an approved SPEAK emits `bus.proactive_utterance` and that, with Live down, the fallback TTS path is selected.
- *No real timers/audio in tests* — inject clock and a spy voice sink.

---

## Phase 3 — Deep Research & Advanced Capabilities

### 3.1 Avatar viseme synchronisation (`face3d.py`)

**Insight — the offline path is the *higher-fidelity* path.** Fidelity is bounded by whether we know the phonemes ahead of the audio:
- **Local TTS path (SAPI5 / Windows):** SAPI emits **real-time `SPEI_VISEME` events** during synthesis. Subscribe to them (via the TTS driver), producing an exact, time-aligned viseme stream at zero inference cost. This is the accurate, fully-offline path.
- **Local neural TTS without native visemes:** derive phonemes by grapheme-to-phoneme (bundled CMUdict + a compact rule-based g2p fallback) → map to a reduced **12-viseme set** (sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, and vowels aa/E/ih/oh/ou) → schedule against the TTS word/phoneme timings.
- **Gemini Live path (cloud, audio-only, no text-ahead):** estimate visemes from the PCM frames already feeding the jaw — formant tracking (F1/F2 → vowel class) plus energy/zero-crossing-rate → consonant class. Lower fidelity, but strictly better than amplitude-only, and it degrades to today's spectral jaw if analysis fails.

**Transport & rendering.** A unified `VisemeScheduler` emits `(viseme_id, weight, t)` onto the bus; the WebGL avatar consumes it through the existing JS bridge (`view.page().runJavaScript("window.orionViseme(id, w)")`, the same pattern Spellscape/Globe use). The Three.js layer interpolates between visemes in its `requestAnimationFrame` loop with co-articulation smoothing (weighted blend of the previous/next viseme), so the mouth never snaps. If `face3d.py` exposes no mouth morph targets, add a parametric mouth (jaw-open × lip-round × lip-spread) driven by the viseme's three coordinates.

**`qasync` safety.** Viseme extraction runs on the audio thread and yields a lightweight event stream; the GL interpolation happens in-browser (RAF), never on the Python paint thread.

**Test strategy.**
- *Pure mapping:* `text → viseme timeline` golden tests ("hello" → sil,HH…,EH,L,OH sequence); g2p and viseme-map are free functions.
- *SAPI path:* mock `SPEI_VISEME` callbacks → assert bus events + ordering.
- *Formant path:* synthesise vowels as known F1/F2 sine mixes → assert vowel classification accuracy on a fixture set.
- *Render:* offscreen `QWebEnginePage`; after feeding a timeline, `runJavaScript("return window.__lastViseme")` → assert the avatar received the last id; optional baseline pixel-diff of an "open A" vs "closed M" frame.

### 3.2 `literature_vault` → `study` automated flashcard pipeline

**Flow.** `literature_vault.ingest_paper` / `ingestion` already extract text, mechanisms and citations into KNOWLEDGE memory. Add a `CardProposer` stage: extracted mechanism/definition units → the existing `StudyEngine._generate` model hook (local Ollama first for offline, cloud when available) → `Q:/A:` pairs (`study.parse_generated_cards`) → dedup against the target deck → a **proposed deck** named for the paper.

**Human-in-the-loop (no fakes / user control).** Proposals land in a **review queue**, not the live deck; the COGNITION page (Phase 1) surfaces accept/reject. Accepted pairs enter via `StudyEngine.add_many`. This preserves correctness and user agency.

**Fallback cascade.** cloud model (best questions) → local Ollama → `study.extractive_cards` (definition/cloze). All three are implemented; the pipeline always yields *real* cards.

**Async.** Generation is long-running → dispatched as a `JobManager` job so the conversation continues; progress on the bus; results to the review queue.

**Test strategy.** Fixture paper text → assert (a) proposals produced with a stubbed generator *and* with extractive fallback, (b) dedup removes near-duplicates, (c) nothing is auto-committed to `study.db`.

### 3.3 New first-class domains — Wellbeing & Finance (offline-capable)

Both follow the proven `study.py`/`focus.py` scaffold: `@dataclass` records + a `Store` (SQLite under `CONFIG_DIR`) + an `Engine` + a dispatcher tool + a lazy dispatcher slot. Both are manual/CSV-first (no third-party API dependency), so they are offline by construction; connectors are a later, optional layer.

**`finance.db`**
```
accounts(id, name, kind[current|savings|credit|cash], currency, opening_balance, created_at)
transactions(id, account_id, at, amount, direction[in|out], category, merchant, note, source[manual|csv|rule])
subscriptions(id, name, amount, cadence_days, next_due, category, active)
categories(id, name, kind[income|expense], budget_monthly)
snapshots(id, at, net_worth, cash, monthly_burn, runway_months)   -- derived cache
```
Tool `finance`: `add_txn`, `import_csv`, `balance`, `runway`, `subscriptions`, `budget`, `report`.
Derived logic: `runway_months = liquid_cash / trailing_3mo_avg_net_burn`; subscription renewal detection from `next_due`.
Proactivity hooks: *renewal in N days*, *runway < threshold* → Proactivity Engine triggers.

**`wellbeing.db`**
```
checkins(id, at, energy[1-5], mood_valence[-2..2], stress[1-5], sleep_hours, note)
factors(id, checkin_id, tag)                     -- caffeine|exercise|social|deadline...
correlations(id, computed_at, pair, coefficient, n)   -- derived cache
```
Tool `wellbeing`: `checkin`, `today`, `trend`, `correlate`.
Cross-domain integration (the value multiplier): correlate `checkins.energy` against `focus.sessions.completed` and `checkins.sleep_hours` against `study` review retention — grounded in the tables Marks XXIV/XXV already own. Psychology-appropriate framing, matching the developer's domain.
Proactivity hooks: prompt an evening check-in; flag a sustained downward energy trend.

**Fallback handling.** No cloud dependency at all; a later "insight narration" step (turn a correlation into prose) uses the router with the Ollama fallback.

**Test strategy.** Engine unit tests (runway maths on fixture ledgers; streak/trend; correlation coefficient over synthetic paired series); CSV-import parsing + dedup; tool wiring (schema + `handler_table`); source-check tests for registration.

---

## Risk Mitigation & Packaging Strategy

**The frozen-`.exe` snapshot problem.** Introduce a **non-GUI self-test mode** in `orion.py`: `--self-test` boots the core headless (offscreen Qt), runs the existing `diagnostics` capability (module compile, import audit, DB integrity, tool-registry completeness, component health, resource headroom), prints a JSON report, and exits with a status code. `build_standalone.py` gains a `--smoke` step that, after freezing, launches `dist/ORION/ORION.exe --self-test` and asserts exit 0 + parses the report — turning "does the freeze actually run?" into a gate rather than a hope. Add `--render-face out.png` (boots minimally, renders `HologramFace` to PNG, exits) so CI pixel-diffs a frozen-render baseline and catches freeze-only rendering breakage.

**CI shape.** The base pytest suite remains the source of truth for logic (fast, cross-platform-ish). A dedicated Windows CI job builds the standalone and runs `--self-test` + `--render-face`; it is allowed to be slow and is the *only* thing that guards the freeze. This keeps the 4,378-test suite untouched and green while adding freeze-integrity coverage.

**Heavy dependencies (PyTorch / MediaPipe).** Keep the lean base freeze (they remain in `EXCLUDES`). Ship them as an **optional CV Extension Pack**: an independently-built artefact (a sidecar directory or user-writable venv) that the frozen app discovers at runtime and prepends to `sys.path`; the already-guarded optional imports light up when present and degrade silently when absent. Extend the `capabilities`/`ai_mode` reporting into a **capability manifest** that states which packs are installed, so the user (and ORION) always know the active feature envelope. A first-run optional installer can pip the CV extras into the sidecar env on demand — avoiding a ~2 GB base bundle while keeping emotion/gesture available to anyone who wants them.

**Rollback & flags.** Tool-routing, proactive-voice, and viseme paths each ship behind an env/config flag defaulting to *off* in the same PR that introduces them, then flip on in a follow-up once the golden tests and a manual live pass are green — preserving the zero-regression mandate at every step.

---

## First five branches/PRs to open (ordered: foundational & low-risk first)

- [ ] **`feat/tool-resolver-core`** — the pure `ToolResolver.resolve(query, state) -> declarations` with hard-filters, BM25 fallback, and a feature flag whose *off* state is provably identical to today's full-132 behaviour. Ships the golden retrieval test set. Unblocks all of Phase 2, touches no handler. *(Highest leverage, lowest risk.)*
- [ ] **`feat/cognition-deck-page`** — the `COGNITION` Command-Deck page (Phase 1) wrapping the existing `study`/`focus` engines, with `bus.focus_changed`/`bus.study_changed` signals and the offscreen render + reducer tests. High daily value, fully isolated from routing.
- [ ] **`feat/proactivity-engine`** — the `ProactivityEngine` loop, `should_speak` integration, `bus.proactive_utterance`, offline-TTS routing, and the pure `decide()` test suite. Turns reactive → proactive without touching the tool layer.
- [ ] **`feat/sapi-viseme-bridge`** — SAPI5 `SPEI_VISEME` → `VisemeScheduler` → `face3d.py` JS bridge, with the g2p/viseme mapping as tested pure functions and the formant path stubbed behind the offline flag. Self-contained; the accurate offline lip-sync lands first.
- [ ] **`feat/finance-wellbeing-scaffold`** — the two SQLite stores + engines + `finance`/`wellbeing` tools + lazy dispatcher slots + unit/wiring tests (no GUI yet), mirroring the `study`/`focus` scaffolding. Additive new domains, ready for a later COGNITION-style surface.

*Sequence rationale:* (1) is the architectural keystone and provably regression-free; (2) converts the largest latent user value already sitting in the codebase; (3) depends only on existing signals; (4) and (5) are independent, additive tracks that can proceed in parallel once (1)–(3) are merged.
