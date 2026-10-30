# O.R.I.O.N. frontend and backend assessment

26 September 2026

## Overall assessment

ORION has a substantial assistant platform: desktop and phone interfaces, live voice, multi-provider routing, local intent learning, semantic memory, verified tool execution and dependency-aware plans. Its strongest foundation is the resilience and observability already present.

The main architectural weakness is fragmented control of shared resources. Individual components have useful limits, but UI work, speech, model calls, indexing and background services do not share one consistent admission and priority policy. Making those components cooperate should precede adding another always-running model.

This is a source and focused-test assessment, with a hardware snapshot. It is not a measured end-to-end session with the avatar, microphone, phone and model running together. Benefits below are expected mechanisms, not measured speed-up percentages. Production behaviour was not changed during this assessment.

## Actual architecture

The production desktop uses **Python, PyQt6 and qasync**. The remote HTTP interface uses **aiohttp**. The optional FastAPI/transformer/DPO implementation under the neural reference documentation is a separate integration example; it is not the production backend.

~~~mermaid
flowchart TD
    Desktop["Desktop: PyQt6, native Overview, WebEngine avatar/globe"]
    Phone["Phone: PWA / Android WebView"]
    Voice["Microphone and live voice"]
    Loop["Desktop qasync loop and OrionBus"]
    Gateway["aiohttp RemoteGateway"]
    Routing["Command, local intent and provider routing"]
    Models["Gemini Live / cloud text / local Ollama"]
    Tools["Dispatcher, policy and verification"]
    Plans["Plans, workflows and background jobs"]
    Memory["SQLite FTS5 + ONNX semantic retrieval"]
    Workers["I/O workers and separate bulk pool"]
    Desktop <--> Loop
    Voice <--> Loop
    Phone <--> Gateway
    Gateway --> Routing
    Loop --> Routing
    Routing <--> Models
    Routing <--> Memory
    Routing --> Tools
    Plans --> Tools
    Tools --> Workers
~~~

The headless entry point composes a smaller portable service set. It still uses Qt Core/qasync. Introducing FastAPI would be an API/deployment decision, not an automatic speed improvement.

The active BRAIN route now creates a **native 2D CommandOverview**. The older swarm renderer remains a conditional fallback. Its retained source should not be mistaken for another WebGL scene that always runs.

Evidence: [desktop composition](<orion_core/app.py:1666>), [remote routes](<orion_core/remote.py:1070>), [headless entry point](<orion_core/server.py:1>).

## Evidence and limits

- **Fresh validation:** 314 tests passed in 18.28 seconds across 14 modules covering lazy startup/pages, animation, WebGL gating, executors, parallel plans, memory, database scale, routing and latency.
- **Previous validation:** the earlier full audit reported 7,117 passed, 3 skipped and 7 subtests passed. That complete suite was not rerun for this assessment. [Audit record](<docs/ORION_BUG_AUDIT_2026-09-26.md>)
- **Hardware:** six physical / twelve logical CPU cores; 15.92 GiB usable RAM, with 5.17 GiB available at the sample; RTX 3060 Ti with 8,192 MiB VRAM, of which 1,522 MiB was in use.
- These resource readings describe the whole machine. No ORION/python desktop process appeared in the process snapshot; the readings cannot establish ORION's idle footprint.
- Local Ollama answered its read-only version endpoint with **0.32.6**. Its loaded-model list was empty. No model was loaded for this review.
- A saved startup timing report was absent from this checkout's diagnostics directory. Cold-start duration, real voice latency, frame-time percentiles and local tokens/second remain unmeasured here.
- Frontend findings concern implementation and interaction structure, not a visual usability study or a real Android-device rendering test.

## Strengths worth preserving

| Area | Existing strength | Practical value |
|---|---|---|
| Desktop | Native Overview, lazy-page support, keyboard search/navigation, visibility gating and idle animation controls | Avoids some unnecessary construction and rendering |
| Voice | Dedicated audio handling, bounded microphone capture, playback telemetry and request-stage traces | Useful seams for protecting speech and diagnosing silence |
| Routing | Capability/budget selection, latency estimates, cooldowns, circuit breakers, retry healing and local fallback | One failing provider need not stop the assistant |
| Memory | SQLite FTS5, optional quantised ONNX semantic retrieval, cached embeddings and vector matrices | Local recall without another database server |
| Execution | Read/write concurrency classification, conservative defaults, verification and DAG dependencies | Independent reads overlap while desktop actions retain order |
| Resource handling | Separate bulk workers, memory governor, deferred warm-up and idle model/MCP parking | A resource-management foundation already exists |
| Diagnostics | Startup timing, percentile metrics, loop-stall sampling and bounded request traces | Optimisation can build on existing measurements |
| Tests | Isolated regression tests, scale checks and failure-path coverage | Refactoring has protection, though mocks do not prove external behaviour |

The semantic encoder already limits ONNX intra-op and inter-op threads to one. Adding more ML threads indiscriminately could undo a deliberate protection for the UI and voice.

Evidence: [concurrency policy](<orion_core/concurrency.py:1>), [DAG execution](<orion_core/plan_executor.py:197>), [semantic threading](<orion_core/semantic.py:170>), [latency monitoring](<orion_core/latency.py:1>), [request tracing](<orion_core/request_trace.py:1>).

## Frontend weaknesses and improvements

### Shared UI and service scheduling

Qt painting and qasync coroutine scheduling share a thread. A synchronous store lookup, expensive paint or graph composition can delay command handling even when called from a coroutine.

**Improve:** measure frame duration and loop stalls with the face visible, the deck on a second monitor, and research active. Move the specific expensive operations identified into workers; keep widget updates on the Qt thread.

Preserve the native Overview and visibility controls. Add a unified reduced-motion/performance preset covering avatar quality, idle cadence and optional visual updates. The WebEngine face already targets 60 FPS when active and defaults to 30 when idle. Assess further reductions for visual quality rather than assuming lower FPS is always better. [Face cadence](<orion_core/gui/face3d.py:1495>)

### Lazy construction is not universal

UnifiedDashboard accepts factories, but the composition root also constructs many concrete panels before the desktop is fully wired. Some expensive work is deferred; substantial service/panel construction remains.

**Improve:** classify panels as essential at launch, needed when opened, or optional. Extend the existing factory mechanism and initialise associated services on demand. Expose first paint, command readiness and voice readiness separately: a painted face does not establish that the microphone/provider is ready.

Evidence: [page factories](<orion_core/gui/unified_dashboard.py:53>), [dashboard assembly](<orion_core/app.py:1707>), [startup budget](<orion_core/startup_budget.py:1>).

### Phone text chat waits for the full answer

The gateway has SSE state/events, but its chat handler awaits the complete answer and then returns JSON. Ordinary compatible-provider text requests disable streaming. Gemini Live audio and the thought stream already stream; the gap is normal remote text chat.

**Improve:** add request-scoped answer deltas, completion/error events and cancellation. Batch UI updates rather than repainting each token. Preserve the final answer for memory and verification. This improves time to first visible text, not the model's underlying generation rate.

Evidence: [chat handler](<orion_core/remote.py:1811>), [text payload](<orion_core/providers.py:2094>).

### Long phone sessions accumulate UI history

The embedded PWA appends messages and activity details without an evident eviction/pagination policy. Layout and memory costs can therefore grow over a long session.

**Improve:** bound the visible history, load older messages on demand, and coalesce status updates. Keep unresolved approval cards visible. Add a long-session browser test with realistic message sizes.

Evidence: [message insertion](<orion_core/remote.py:177>), [activity insertion](<orion_core/remote.py:439>).

### Navigation must accommodate considerable breadth

The desktop registers 21 pages. Search, zones and a status strip already help.

**Add:** favourites/recent tools and a task view distinguishing queued, running, awaiting approval, paused and failed work. Show measured progress or an honest indeterminate state. This is a recommendation from the structure, not a claim that the interface was visually observed to be confusing.

## Backend weaknesses and improvements

### 1. Confirmed contract defect: local memory controls do not reach Ollama as intended

Both compatible chat paths send **keep_alive: 0** and **options.num_ctx** through **/v1/chat/completions**. The installed Ollama version's upstream request type and conversion do not carry those fields through. These requests therefore do not enforce ORION's intended unload/context policy.

Ollama's compatibility documentation directs context configuration to a Modelfile. Its native chat API documents options and keep_alive. ORION's current tests assert the outgoing JSON keys, so their success does not establish effective memory control.

**Fix first:** implement an explicit native Ollama transport, or configure and verify the model/server settings for the compatibility path. Keep other compatible servers separate: locality does not establish support for Ollama fields. Test effective context and residency against the installed server.

Then select an intentional residency policy: briefly keep the model loaded for repeated local turns; unload earlier under pressure. Reloading every turn trades memory for latency.

Evidence: [ordinary local settings](<orion_core/providers.py:2111>), [streaming equivalent](<orion_core/providers.py:1985>), [mock assertions](<tests/test_local_provider_memory.py:90>), [Ollama 0.32.6 conversion](https://github.com/ollama/ollama/blob/v0.32.6/openai/openai.go#L98), [compatibility documentation](https://docs.ollama.com/api/openai-compatibility), [native chat API](https://docs.ollama.com/api/chat).

### 2. New HTTP sessions for each compatible-provider call

Both chat transports create and close an aiohttp ClientSession per call. Connections cannot be pooled across those calls.

**Improve:** manage reusable sessions within the router's event-loop lifecycle, with bounded connectors and per-request authentication/timeouts. Explicitly close them during desktop/headless shutdown. Measure connection setup separately from inference time. [aiohttp guidance](https://docs.aiohttp.org/en/stable/client_quickstart.html)

Evidence: [streaming session](<orion_core/providers.py:1993>), [ordinary session](<orion_core/providers.py:2128>).

### 3. Local limits do not form a global work budget

The default executor bounds threads but has an unbounded pending SimpleQueue. Intent training also has unbounded queues. Elsewhere, bulk workers default to two, jobs to four active jobs, and tool batches to six parallel calls. Those independent limits can overlap.

The resource governor samples RAM and sheds selected components. The explicit optional-work gate is used by Forge; it is not universal admission for background work.

**Improve:** introduce shared admission for interactive, background and training work. Bound pending work and report busy/queued status. Reserve user-turn capacity, age waiting jobs to prevent starvation, and coalesce obsolete telemetry/index refreshes. Include CPU load, available RAM and GPU memory in admission decisions.

Do not blindly cap audio playback by dropping chunks: the implementation preserves speech deliberately. Apply backpressure upstream and discard stale utterances only through explicit interruption semantics.

Evidence: [executor queue](<orion_core/executors.py:68>), [training queue](<orion_core/intent_brain.py:196>), [job limits](<orion_core/jobs.py:45>), [governor integration](<orion_core/app.py:2003>).

### 4. Text routing lacks one total request deadline

Text generation can try several profiles in sequence and perform a healing retry. Each attempt has its own timeout. The patience_s deadline limits waiting for cooldown recovery; it does not wrap all attempts in one total deadline.

**Improve:** pass a remaining deadline through selection, connection, generation, healing and retries. Give interactive and research work different budgets. Apply per-provider concurrency admission across foreground and background users.

Evidence: [text routing](<orion_core/providers.py:1655>), [healing retry](<orion_core/providers.py:1089>).

### 5. This checkout defaults writable data into OneDrive

No configuration pointer exists in this checkout, and no config override was present in the inspected environment. The resolver therefore selects its in-project config directory. Migration support already exists.

**Improve:** use the existing migration workflow to keep live databases, caches and model state outside the synced project; synchronise consistent backups. This is primarily a durability improvement, with possible reduction in sync-related I/O contention. No sync-induced delay was measured. Another installed executable may use a different data location.

SQLite's backup API provides consistent snapshots; uncheckpointed WAL is part of the database's persistent state. [SQLite backup API](https://sqlite.org/backup.html), [WAL documentation](https://sqlite.org/wal.html)

Evidence: [resolver](<orion_core/constants.py:305>), [migration tool](<tools/move_config.py:1>).

### 6. Orchestration has several implementations

Plans already support DAG dependencies. Workflows have their own step loops, jobs manage detached calls, and two modules provide background-task helpers. Cancellation, persistence, progress and limits consequently have several owners.

PlanExecutor also logs and removes unknown prerequisites, allowing the dependent step to execute. A test explicitly preserves that behaviour. For consequential steps, a misspelt prerequisite should invalidate the plan rather than relax its preconditions. This is not evidence that the dispatcher's separate permissions are bypassed.

**Improve:** validate all IDs/dependencies before any execution. Add a shared durable job record, cooperative pause/checkpoints and priority admission behind existing interfaces. Resume read-only/idempotent steps after restart; reconcile outcomes before retrying side effects. Cancelling an await does not guarantee that a native thread or OS action has stopped.

Evidence: [unknown prerequisites](<orion_core/plan_executor.py:205>), [workflow loop](<orion_core/workflow_engine.py:193>), [background helper](<orion_core/background.py>), [task helper](<orion_core/tasks.py>).

### 7. Semantic retrieval has a scaling cost

Warm lookup computes similarities against the cached matrix, sorts scores and reconstructs an ID-to-position map. Index changes can force matrix rebuilding. Query caching and incremental embedding already exist.

**Improve in order:** benchmark representative corpus sizes; cache the ID map with its matrix; use partial top-k selection where full ordering is unnecessary; rebuild and replace snapshots away from interactive work. Introduce approximate-nearest-neighbour indexing only when measurements justify it. A second vector database is not automatically faster for modest local memory.

Evidence: [similarity ranking](<orion_core/memory.py:456>), [matrix cache](<orion_core/memory.py:572>).

### 8. Large modules and broad dependencies raise maintenance cost

Current source sizes include 3,543 lines in live_worker, 2,802 in audio, 2,632 in providers and 2,511 in app. Size is not itself a performance defect, but complicates lifecycle and ownership changes.

Requirements combine core UI/audio with research, vision, ML and specialised tools, including both preferred and alternative Whisper backends.

**Improve:** separate transport from routing policy and voice session state from orchestration. Introduce a declarative service registry with start/stop/readiness hooks. Package optional capabilities as extras and lock supported environments reproducibly. Smaller installations improve build/upgrade cost; RAM improves only when unnecessary imports and active services also disappear.

Evidence: [composition root](<orion_core/app.py>), [requirements](<requirements.txt:1>).

## Ranked implementation roadmap

| Order | Change/addition | Expected benefit | Acceptance evidence |
|---|---|---|---|
| 1 | Correct Ollama controls and verify the actual adapter contract | Reliable local memory policy | Effective context and residency match policy |
| 2 | Reuse HTTP sessions; enforce total deadlines | Less repeated setup; bounded failure latency | Reused connections and bounded slow-provider scenarios |
| 3 | Shared priority admission/resource budgets | Responsive UI/voice during background activity | Queue age/depth and user-turn p95 under load |
| 4 | Stream remote text; bound visible history | Earlier answers and stable phone sessions | First-delta timing and DOM size over long sessions |
| 5 | Extend lazy activation; add a performance preset | Lower startup/idle cost | Readiness timings, process-tree memory and frame times |
| 6 | Durable job state and strict plan validation | Predictable autonomy and recovery | Restart/cancellation without duplicated side effects |
| 7 | Benchmark retrieval and optimise cache/ranking | Fast recall as memory grows | Latency and recall quality at 1k/10k/50k records |
| 8 | Capability packages and smaller service boundaries | Easier upgrades; fewer integration regressions | Minimal-install checks plus the existing suite |

Extend the existing telemetry panel with queue delay, first-token/first-audio time, provider-attempt breakdown, GPU residency and cold/warm model-load time. ORION already has request IDs and a stall detector; connect these measurements rather than creating a competing diagnostics system.

## Starting hardware policy

These are proposed settings to benchmark, not established optimums:

- One local generation active at a time until measurements justify concurrency.
- Begin local model evaluation with a quantised 3–4B-class candidate, short bounded context and tool/reasoning quality tests. Parameter count does not establish fit or speed.
- Keep semantic embeddings and the lightweight intent learner on the CPU; preserve the conservative ONNX thread settings.
- Retain the small bulk pool initially. More workers can increase contention on six physical cores.
- Reserve VRAM for the desktop and avatar instead of allocating against the entire 8 GB. Check actual CPU/GPU placement and peak context memory. [Ollama inspection guidance](https://docs.ollama.com/faq#how-can-i-tell-if-my-model-was-loaded-onto-the-gpu)
- Defer training during user turns and heavy tasks. Compare cold loading against short warm residency before setting the default.
- Consider additional RAM only after representative workloads show sustained pressure or paging. The current single sample does not establish a hardware shortage.

The active neural foundation is a sparse MLP intent classifier with replay memory. It is useful for routing; general-purpose language reasoning comes from the model providers. The transformer/adapter architecture in the neural reference should enter production through offline evaluations, versioned adapters and rollback. Continuous in-place weight updates would add load and make failures harder to reproduce.

## How to measure improvement

Before and after each substantial change, repeat: cold/warm desktop start, idle face/deck, short local command, remote answer, provider failure, research during conversation, growing memory retrieval and phone long-session use.

Record first paint, command/voice readiness, frame p95, loop stalls, process-tree private memory, GPU memory, queued/running work, first-token/first-audio time, total response time, cancellations and success. Include renderer/worker children rather than only the Python process.

Initial engineering targets could be: local UI acknowledgement within 100 ms at p95, warm local command routing within 250 ms at p95, no repeated loop stalls over 250 ms, and the existing five-second startup budget with separate voice-readiness reporting. These are proposed targets, not measured current capabilities.

Document background load, repeat runs and retain failures. Check tool accuracy, memory relevance and duplicate side effects alongside speed. Faster completion matters only when ORION completes the intended task.

## Implementation handover and complete capabilities

These findings were then carried out as phased changes with acceptance tests, capability-preservation requirements and rollback.

The [complete current capability rundown](<docs/ORION_CAPABILITIES_2026-09-26.md>) covers all 144 core declarations, the paired-phone tool, 21 pages, 18 bundled plugins and configured MCP integrations. It includes an exact tool index and distinguishes reference-only features from the active architecture.
