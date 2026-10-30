# ORION comprehensive audit — 27 September 2026

This pass reviewed the current dirty checkout, preserved existing work, and
implemented fixes in automation, scheduling, persistence and release preparation.
The evidence is in `reports/comprehensive_audit_2026_09_27/`.

## Coverage and what the results mean

The starting inventory contains **954 source, test, build and documentation
files**: 290 modified tracked files, 491 unchanged tracked files and 173 new
files. All **850 Python files** in that inventory parsed successfully. The
inventory includes local custom-tool source, Android source/build configuration,
deployment files and the optional neural reference. It excludes generated builds,
third-party vendored assets, executable/model binaries and private runtime data.
Git status cannot establish which person or assistant authored an existing edit;
the inventory distinguishes changed from unchanged files, not authorship.

Repository-wide checks cover syntax, critical static errors, test collection,
registered capabilities and source publication. Manual inspection concentrated on
the prior audit findings, provider/phone integration seams, workflow and job
recovery, schedulers, sync stores, Android credentials, packaging and the old
roadmaps. This is **not a claim of line-by-line manual review of every file**, a
penetration test, or proof that every hardware/cloud capability works.

Static resolution also checked **1,418 internal module imports**, with no
missing module paths. Removed modules have no remaining static import sites.
The TODO/placeholder scan mostly found intentional UI waiting states and base
interfaces, such as `FlightProvider.area`; those are not treated as missing
implementations. Dynamic plugin imports still depend on behavioral checks.

The initial regression file reproduced 28 failures before implementation.
Expanded tests exposed two further boundary cases. There are now 36 new
reliability cases, one optional-dependency recovery case and three new
source-publication cases.

## Implemented improvements

| Area | Defect or missing behavior | Result |
|---|---|---|
| Plugin cron validation | Wrapped ranges bypassed bounds checking; month names could appear in minute/weekday fields; `0-7` collapsed to Sunday alone. | Validate both endpoints before wrapping, restrict names to their field, normalize Sunday after range expansion. |
| Plugin interval validation | NaN, infinities, negative durations and unrepresentably large values could silently disable scheduling or raise during registration. | Reject them at parsing and manifest loading with a readable validation error. Zero still means no interval. |
| Next scheduled occurrence | An impossible February date scanned more than two million individual minutes on the caller's thread; datetime overflow escaped. | Skip ineligible days, then consider scheduled hours/minutes; return no future occurrence at the datetime boundary. The four-year search horizon is retained. |
| Scheduled plugin lifecycle | A plugin removed, replaced, muted or rescheduled while another awaited run executed remained in the old due-list and could still run. | Recheck current registration and due time immediately before dispatch. Existing once-on-wake catch-up behavior is preserved. |
| Workflow persistence | Create, replace and remove reported success even when writing `workflows.json` failed, leaving memory and disk inconsistent. | Save the proposed library before publishing it in memory; return failure and preserve the previous definition on write errors. |
| Workflow ownership | Caller-owned nested arguments and live library edits could alter a queued/running workflow. | Deep-copy definitions at creation and execution. Each run owns its original plan. |
| Workflow restart | Resume compared only tool names. Changed arguments or retry policies could silently alter an interrupted run. | New jobs persist a definition digest covering every step and retry setting. Resume refuses mismatches; legacy jobs compare recorded tools and arguments. Incomplete job arguments are refused. |
| Nested workflow arguments | Placeholders only expanded at the top level; input containing `{prev}` was expanded again as a template. | Expand string values recursively through dictionaries/lists in one pass, keeping inserted user text literal and the source template unchanged. |
| Failed job admission | A failed job-store write could retain a running/queued admission slot. | Release the ticket and report that the workflow was not started. |
| Durable job creation | Failure halfway through step insertion left a parent and some steps inside an open transaction, later committable by unrelated work. | Parent, steps and retention pruning commit or roll back together. |
| Sync journal atomicity | A malformed or interrupted remote batch left partial entries and logical-clock changes pending. A failed local write also advanced the pending clock. | Roll back the whole batch or local write. Reserve the SQLite writer before reading a local clock; separate connections preserve every increment. |
| Source release gate | An unignored old test tree introduced 955 invalid release candidates; the plan and two generated images were also rejected. | Ignore only the known root audit scratch pattern; approve `PLANS.md`; admit the two reviewed procedural PNGs by exact SHA-256. Altered images still fail closed. |
| Optional Grandmaster chat | The full suite assumed the optional `berserk` adapter was installed; missing it raised an unhelpful import error at runtime. | Missing chat support raises an actionable `LichessError` naming the optional requirements file. The actual-adapter test explicitly requires that package; a separate test always exercises its absence and worker cleanup. |

No runtime dependency was added. No user database was migrated, production model
trained, provider contacted, outbound message sent or installed executable rebuilt.
Changes are in source; running/frozen ORION installations need their normal
restart/rebuild process to pick them up.

## Measured scheduling improvement

Three runs per case, same Python interpreter and PC, saved original module versus
the changed module. These measurements ran while the suite was active, so their
wall times are indicative. Both implementations returned the same result.

| Search from 27 September 2026 | Before median | After median |
|---|---:|---:|
| Impossible 31 February | 2,408.27 ms | 1.27 ms |
| Next leap day | 890.93 ms | 0.45 ms |

The structural guarantee matters more than the timing: impossible-date lookup
now checks days rather than every minute. A deterministic test enforces a bounded
match-call count without a machine-speed threshold. Samples are recorded in
`schedule-benchmark.json`; the reproduction script is `benchmark_schedule.py`.

## Roadmap reconciliation

The 26 September integration ledger stopped after phase 0, but the checkout
contains substantial later implementation. Do not schedule these as wholly new
features again:

| Earlier proposal | Current evidence | Remaining acceptance work |
|---|---|---|
| Provider pooling and total deadlines | `provider_transport.py`, provider call scopes and deadline propagation; provider regression suites. | Real provider p95 under cold connections, throttling and intermittent networks. |
| Shared resource admission | `admission.py`, `resource_governor.py`, bounded executors and bounded intent-learning queues. | Sustained mixed voice, UI, inference and background workload on the target PC. |
| Stream phone text and bound history | `/api/chat/stream`, per-stream IDs, bounded chat/trace DOM logic in `remote.py`. | Long physical-phone sessions, reconnects, Android background/foreground transitions. |
| Durable workflow/plan jobs | `job_store.py`, job controls, explicit interrupted state and bound replay approvals. | Broader power-loss and filesystem-fault testing; this pass closes the concrete defects above. |
| Startup/lazy work and measurement | Startup readiness marks, lazy page factories, `tools/perf_scenarios.py`. | Repeated visible desktop boots and microphone-ready latency, with independent hardware measurements. |
| Retrieval benchmarks | `tools/bench_retrieval.py`, cached semantic snapshots and recall fixtures. | Repeat 1k/10k/50k measurements after retrieval changes; larger independently labelled holdouts. |
| Capability dependency groups | `requirements/core.txt`, speech/vision/research/integrations groups, developer/training groups and a pinned Windows lock. | Clean environment installation and optional-dependency removal tests; requirements grouping is not module decomposition. |

## Actual unfinished work and recommended order

These are scoped follow-on features, not claims that this pass implemented them.

1. **Complete desktop/cloud sync transport.** The journal and deterministic
   merge core exist, and app/server opt in via `ORION_SYNC_JOURNAL=1`. The current
   gateway has no `/v1/sync/journal` route. Add authenticated, bounded push/pull,
   per-peer cursors, reconnect reconciliation and visible sync health. Before
   exposing it to network input, validate entry shapes, hash integrity and
   Lamport bounds. Acceptance: two isolated nodes converge after interrupted and
   repeated batches without unauthorized memory changes. Keep the current
   journal opt-in until its consumer exists.
2. **Finish durable Android proactive delivery.** Screen capture and sentinel
   foreground services exist; reliable assistant notifications while the WebView
   is closed remain separate work. Add an explicit notification permission and
   delivery policy, deduplication, and reconnect recovery. Acceptance: background,
   process death, revoked permissions and Doze tested on a real phone.
3. **Protect Android refresh credentials with Keystore.** `Prefs.kt` still stores
   refresh credentials in ordinary private SharedPreferences. Backups are
   excluded and certificate pins exist, but those address different boundaries.
   Include migration, invalidated-key recovery and re-pairing tests; never make
   an undecryptable credential silently look like a valid session.
4. **Publish a reproducible hardware acceptance run.** Record desktop first
   paint/command/microphone readiness, first text/audio latency, long phone-session
   behavior, camera handoff, GPU residency and memory pressure. Existing unit
   tests and the scenario harness cannot establish this end-to-end experience.
5. **Move live generated tools behind a process boundary.** The opt-in container
   verifier is distinct from execution of a loaded Python plugin. Plan an RPC
   worker with explicit data/capability access and cancellation. Acceptance:
   process failure is contained, secrets are not inherited unintentionally, and
   approved tool capabilities continue to work. In-process execution is not
   hostile-code isolation.
6. **Separate large composition and routing modules incrementally.** Extract
   lifecycle-owned services from `app.py`, `providers.py`, `live_worker.py` and
   `remote.py` while preserving public interfaces and capability gates. Start
   with one service and its teardown contract, not a whole-application rewrite.
7. **Strengthen scheduling policy and operational evidence.** The existing
   scheduler performs one catch-up run after sleep; this pass corrects older
   comments describing a skipped run. Make catch-up, daylight-saving fold/gap handling,
   timezone choice and failure notifications explicit settings before adding
   more unattended schedules. Do not silently change existing wake behavior.
8. **Graduate optional neural work through offline evaluation.** Keep training
   separate, compare held-out routing/retrieval quality, stage versioned models
   and prove rollback before production promotion. CPU examples do not establish
   full-size GPU training or a better assistant.
9. **Bound cleanup review cost and isolate its dispatcher test.** The first full
   run spent 77.31 seconds in the empty-argument `cleanup_review` contract case.
   That case scans the real project; directory classification recursively totals
   the sizes of environments/build output, even where the main walk prunes them.
   Give the test a synthetic project, then add scan budgets, cancellation and
   explicit partial-size reporting. Acceptance: large generated trees cannot
   monopolize a request, and incomplete size estimates are never presented as
   exact. This observation is from test timing and source inspection, not a
   measurement of visible UI latency.

## Verification and reproduction

- Focused scheduling/workflow/sync/job/admission/parallel suites: **256 passed**,
  including all 36 new reliability cases (`focused-final.xml`).
- Packaging and icon suites after the final export fix: **40 passed**, including
  three new publication cases (`public-tests.xml`).
- Grandmaster suite: **25 passed, one skipped** without the optional chat adapter
  (`grandmaster.xml`). The real-adapter timeout/thread/redirect test also
  **passed** using `berserk` already installed in the debug environment, with
  HTTP mocked and no packages installed (`adapter.xml`).
- Optional neural reference suite in its existing debug environment: **46
  passed, seven subtests passed**, in 58.82 seconds (`neural.xml`). Tests exercise
  temporary tiny CPU models, including an adapter update, not production weights.
  Three warnings concern Starlette/httpx deprecation and missing optional model
  metadata; they are retained in `neural.log`.
- Capability gate: passed; no missing declarations or duplicate entries.
- Public-source audit after repair: passed, **1,032 source files**, no archive
  created. This count also includes approved assets excluded from the text/code
  inventory; it is a different scope.
- Critical Ruff rules `E9,F63,F7,F82`: passed. This does not claim all style rules
  or repository-wide type checking passed.
- Final syntax scan: **851 Python files**, zero syntax errors. The extra file
  is this pass's new reliability regression module.
- First complete regression suite: **7,651 passed, one failed, six skipped**,
  with seven subtests passed in 492.64 seconds (`full.xml`). The failure was
  the absent optional Grandmaster chat adapter, addressed above.
- **Final complete regression suite: 7,655 passed, seven skipped, zero failures,
  with seven subtests passed in 506.93 seconds** (`final.xml`, `final.log`).
  This run includes the publication fixes and optional-dependency recovery.
  Skips: two optional neural modules and the optional chat adapter (all exercised
  separately above), two Windows symlink-privilege cases, one opt-in live Ollama
  check and one obsolete matplotlib-dependency check. Skips are not passes.

`changes.patch` and `changes.json` identify this pass's edits relative to saved
pre-edit files, rather than the already-dirty Git base. Pre-edit hashes match the
starting inventory. The patch normalizes line endings; the edited ignore file's
mixed line endings were normalized too. A reverse applicability check verifies
the patch against the current source without applying or reversing it here.

The first attempted baseline used an inaccessible temporary directory and was
stopped. A fresh-directory baseline was also stopped before production edits;
neither partial run is a completed baseline result. The before-fix evidence is
the isolated failing regression run. The final complete suite was launched after
all runtime fixes and publication-tool changes. Only documentation/evidence and
ignore-file line-ending normalization changed while that final run was active.

From the repository root, with a fresh temporary directory name for each run:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe -m pytest -q -ra -p no:cacheprovider --basetemp reports/audit-verification-tmp
.venv\Scripts\python.exe tools/check_capabilities.py
.venv\Scripts\python.exe tools/prepare_public_source.py
```

Tests use isolated state with the repository's real-config write guard. No live
phone, microphone/camera, paid-provider account, outbound messaging, Docker
isolation runtime or production GPU training was exercised. Android was reviewed
at source level and was not rebuilt in this pass. Existing audit evidence and
unrelated dirty files remain in place; nothing was committed or pushed.
