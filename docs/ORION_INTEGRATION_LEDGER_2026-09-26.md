# ORION integration pass — progress ledger

Started 26 September 2026. Tracks the phased roadmap in
`ORION_ARCHITECTURE_PERFORMANCE_REVIEW_2026-09-26.md`. Each item is **implemented**, **verified**
(with the evidence named), **pending** or **blocked** (with the missing prerequisite).

## Baseline (before any change in this pass)

| Check | Result |
|---|---|
| Full suite (`.venv`, `QT_QPA_PLATFORM=offscreen`, `-p no:cacheprovider`) | 7,117 passed, 3 skipped, 7 subtests passed, 0 failed — 371.7 s |
| `tools/check_capabilities.py` | ok; no missing or duplicate entries. Added since the baseline file: tools `aviation`, `fetch_url`, `find_tool`, `sound_sense`, `use_tool`; handlers `aviation`, `fetch_url`, `sound_sense` |
| Startup (headless boot of a copied config, earlier today) | 3.12 s / 3.15 s process start to ready (was 12.8 s cold / 6.3 s warm before the start-up fixes) |

Skips: the two optional neural modules (need the separate neural environment) and one symlink test
(host privilege).

## Phase 0 — measurement

- **Implemented:** `request_trace.current()/bound()/mark()` — a context-scoped current request
  that follows `await`, tasks and worker threads, with bounded, label-sanitised measurements
  (no prompts or keys can enter a trace).
- **Implemented:** `StartupBudget.milestone()` and `startup_budget.milestone()` — separate
  `first_paint`, `command_ready`, `microphone_ready`, `provider_ready` readiness, written to
  `diagnostics/last_boot.txt` and rewritten as later milestones arrive.
- **Implemented:** `first_audio` mark on the voice request trace; `REPLY_DELIVERED` completes
  non-voice requests (a phone turn no longer reads as "stalled" after 45 s).
- **Implemented and run:** `tools/perf_scenarios.py` — `startup` (fresh temporary config, dummy
  local provider, `ORION_NO_CAPTURE=1`, process-tree private/resident memory including WebEngine
  children, idle CPU), `local-command` (routing decision latency, nothing executed) and
  `retrieval` (delegates to `tools/bench_retrieval.py`).
- **Measured (26 September, machine under heavy load — five test workloads running, 1.8–2.7 GB
  free):** warm routing p95 0.54 ms for repeated phrasings and 11.8 ms for novel sentences with
  the ONNX encoder (target 250 ms); warm boots 4.65–5.79 s, first-ever launch 20.8 s. A py-spy
  profile of the first launch attributed 4.5 s to source fingerprinting and ~1.6 s to
  `shutil.which("ffmpeg")`, both on the Qt thread — both moved off it (below). Process tree after
  15 s idle: ~1.4 GB private, ~0.82 GB resident (python + one QtWebEngineProcess).
- **Re-measured after all changes (27 September 22:10, same method, machine at 11% CPU with
  1.9 GB free):** first-ever launch **20.8 s → 7.92 s** (first paint 1.06 s, commands ready
  7.53 s; this run also started two MCP servers because a fresh config has no tool catalogue);
  warm launches **3.42 s and 2.93 s** (first paint 0.80/0.64 s). Warm idle process tree
  ~1.45 GB private / ~0.84 GB resident; idle CPU mean 5.4% of the whole machine (p95 13%) —
  offscreen, so the face renders in software; a visible session with GPU rendering was not
  measured.
- Not measured: `microphone_ready` (capture disabled by design in measurement runs),
  `provider_ready` with a real provider, visible-desktop frame times.

## Phases 1–8

The original ledger was not filled in as the later work landed. The
[27 September source reconciliation](ORION_COMPREHENSIVE_AUDIT_2026-09-27.md#roadmap-reconciliation)
records the implementation now present: provider pooling/deadlines, shared
admission, phone streaming, durable jobs, lazy activation/measurement, retrieval
benchmarks and dependency groups. It also separates implementation from the
remaining hardware and integration acceptance work. Treat that audit as the
current status; this historical ledger alone understates what has been built.

Per-phase detail from this integration pass (27 September):

### Phase 1 — provider transport and local memory control — implemented, verified

- Confirmed on real hardware: through `/v1/chat/completions`, `keep_alive: 0` and `num_ctx: 2048`
  were ignored (the model stayed resident at context 4096, 5.27 GB VRAM, ~5 min expiry).
- `provider_transport.py`: pooled per-event-loop sessions with bounded connectors (no reuse
  across loops), a stale-connection retry only before any response, the shared SSE decoder
  (with the stream-integrity rules from the second security pass), a native Ollama transport
  (`/api/chat` NDJSON, images, `keep_alive`, `num_ctx`, `num_predict`, usage and load time),
  a cached `/api/version` probe, and a priority endpoint gate.
- `providers.py`: one total deadline (`deadline_s`, `budget` interactive 90 s / research 900 s)
  through selection, admission, generation, healing and cooldown waits; per-profile admission
  (cloud 6, background 4, local 1); residency (120 s after conversation, 30 s after background,
  0 under pressure, never while answering); `on_delta`/`on_restart` streaming with no splicing;
  `aclose()`, `unload_local_models()`, memory-pressure hooks (wired into the governor and
  shutdown); `ProviderDeadlineError` never charged to a provider, including a saturated GPU
  lease; context default 4096 (the 2048 cap had never applied and would have cut the system
  prompt). Rollback: `ORION_OLLAMA_NATIVE=0`.
- Evidence: 1,011 focused tests passed; opt-in live test passed against Ollama 0.32.6 with
  `llama3.1:latest` (effective context 4096, unload verified via `/api/ps`); real-router
  streaming from a local server: first fragment 18 ms, whole answer 593 ms.
- **Cold versus warm local latency (measured 27 September, real router, native transport,
  Ollama 0.32.6, `llama3.1:latest` 8B Q4_K_M, RTX 3060 Ti; short one-sentence answers):** cold
  (model unloaded before each of 3 samples) first token median **8,364 ms**, of which model load
  7.1–7.4 s, total 8,495 ms; warm (5 samples) first token median **307 ms**, total **465 ms**
  (the first warm sample 1,043 ms). The 120 s conversational keep-alive is what turns an 8 s
  wait into a 0.3 s one. Ollama was started for the measurement and stopped afterwards.
- Not measured: cloud connection reuse against real providers.

### Phase 2 — shared priorities and bounded work — implemented, verified

- `admission.py`: classes interactive > user > background > training with per-class running and
  pending bounds, a shared budget, ageing (one class per 20 s), coalescing, `Busy` instead of
  blocking, hysteresis on memory/CPU/GPU (unknown GPU is never "free"), an exclusive GPU lease
  with cooperative preemption, an optional-loop registry, `note_foreground` (wired to voice).
- Bounded executor queue (`ExecutorBusy`), bounded intent-training queues (drop-oldest lesson),
  `tasks.spawn` tasks registered for shutdown drain.
- Audio: a real bug fixed — cancelling a paused reply left stale audio that replayed later;
  playback chunks now carry an interruption generation. No speech is ever dropped otherwise.
- Evidence: `tests/test_admission.py` (26), executor/audio/task additions; burst of 10,000 submits
  held depth at 16; grant/release 11.5 µs.

### Phase 3 — frontend and remote responsiveness — implemented, verified (no device run)

- Phone: `/api/chat/stream` (start/delta/restart/done/error, request ids, rising seq), cancel
  route and disconnect detection, per-device history route, bounded SSE subscribers, PWA painting
  once per frame, DOM bounded to 100 messages at the bottom with paging, approval cards never
  evicted. First visible text ~67 ms versus ~1,256 ms for the whole reply (headless Chromium,
  fake provider, n=10). Long session: 501 → 141 log children.
- Desktop: six stateless deck pages built on first open with honest loading and
  failed-with-retry states; `bus.last_value` so late pages show the live state; Motion and
  performance preference (Full default; the active frame rate is never lowered — face rendered
  in headless Chromium: steady 16.7 ms frames, no errors); pinned and recent pages; a TASKS page
  over the durable job store (states in words, real or honestly indeterminate progress,
  approvals, pause/resume/cancel/priority, recent actions, capacity).
- Not done: a real Android/WebGL device run.

### Voice reliability (found while working the backlog) — implemented, verified by tests

- The installed build's own Live journal (`diagnostics/live_sessions.jsonl`, 26–27 September)
  shows sessions dropping with "1008 Requested entity was not found" 152–172 s in, then walking
  retired models before reconnecting after ~12 s. The installed `ORION.exe` (built 25 September)
  predates the reconnect fixes, which is why the model walk still appeared.
- Root cause of the mid-session 1008: the tool call was awaited inside the receive loop, so the
  server's `tool_call_cancellation` (sent when the user interrupts) was not read until the tool
  finished — and ORION then answered a call the server had already forgotten. Tool calls now run
  beside the loop (one message at a time, in order); cancelled calls are not started, never
  answered, and an answer is never sent to a session that reconnected meanwhile. Four regression
  tests, including one that drives the real receive loop.
- The phone gained a read-only TASKS sheet (`GET /api/tasks`: states, progress, notes, results of
  finished work; no arguments or checkpoints), rendered and checked in headless Chromium at phone
  width.

### Phase 4 — durable orchestration and strict plans — implemented, verified

- Whole-plan validation before any dispatch (duplicate ids, missing/self prerequisites, cycles,
  unknown or forbidden tools, schema-invalid arguments, oversize) — the old test that expected
  unknown edges to be dropped was replaced deliberately; 17 invalid-plan cases dispatch nothing.
- `job_store.py` (durable SQLite, uuid ids, legal transitions, interrupted on restart, approvals
  bound to job + step + hash of tool and exact arguments, 10-minute expiry, single use);
  `jobs.py` controls and checkpoints; workflows durable and resumable; side-effecting steps never
  retried blindly. Wired into `app.py` start-up and shutdown.

### Phase 5 — memory and data placement — implemented, verified

- Immutable semantic snapshots, partial top-k, generation counters and change log, re-check of
  every meaning match against the current row (an edited fact was previously found by its old
  words), per-row indexing on save. 50k records: warm query p50 13.5 → 3.5 ms, first question
  after a save 394 → 5.6 ms, recall unchanged (1.0; real-encoder quality identical). ANN not
  justified.
- `tools/move_config.py` rewritten: real source resolution, running-instance refusal, holds
  ORION's lock, SQLite backup API copies, verification, atomic pointer switch, journal,
  `--dry-run`, `--rollback`. A moved checkout keeps using the installed key store; an
  `ORION_CONFIG_DIR` run never reaches it. Not run against real data (needs the owner's consent).

### Phase 6 — modularity and packaging — implemented (install groups), partly pending (service
decomposition)

- `requirements/` groups (core, speech, vision, research, integrations, dev, neural-training);
  `requirements.txt` is the full install built from them; a test proves `orion_core.app` imports
  and 145 routes build with every optional group's modules blocked.
- Pending: the service registry and incremental decomposition of `app.py`, `providers.py`,
  `live_worker.py` and `remote.py`.

### Phase 7 — optional local neural planner — in progress

See `docs/ORION_NEURAL_ARCHITECTURE_GUIDE.md` (Phase 7 section) when it lands.

### Phase 8 — capability preservation — implemented

- The checker now lists lambda handlers (`find_tool:<lambda>`, `use_tool:<lambda>`), explaining
  the old 143-of-145 count; the baseline file was not regenerated, so removals still fail.
  Added since the baseline: tools `aviation`, `fetch_url`, `find_tool`, `sound_sense`,
  `use_tool`; page `TASKS`.
