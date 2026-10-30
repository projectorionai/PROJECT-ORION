# ORION Mark XXXIII — completion and audit record

Date: 30 September 2026. Current source identity: **Mark XXXIII**, package version **33.0.0**.

This record distinguishes implemented fixes, measured results and remaining deployment or device checks. Syntax coverage and passing tests do not establish that every account integration or physical device is fault-free.

## Chronological work

### 1. Persistent task and event-loop faults

- Added durable, exact-title dismissal records for Blog post, Opinion piece, Product launch, Wake Up, Work, Travel buffer, Chill out, Gym buffer and Gym. Synchronisation cannot silently recreate a dismissed task.
- Archived **30 confirmed overdue Notion entries** with those exact titles. Duplicate titles are handled only by the explicit overdue-only operation; future dated entries and partial title matches are preserved.
- Moved CPU, memory and network counter sampling into one bounded worker. The GUI receives the last complete snapshot immediately. A slow Windows network adapter query cannot hold the GUI/asyncio thread, spawn overlapping refreshes or publish data from a previous reset.
- Added `python -m tools.retire_tasks` with dry-run behaviour and explicit apply/Notion options. This is reproducible maintenance, rather than hiding overdue warnings.

### 2. Supplied JV capabilities

Installed twelve native adapters, with reviewed manifests, dependency checks, application startup binding and cancellation on shutdown. The original files use a different plugin contract, so they were adapted rather than advertised as directly loadable.

The capabilities cover document review, quizzes, Pomodoro sessions, intake/reminders, real Excel workbooks, nutrition estimates, launcher maintenance, reviewed chat drafts, Telegram remote requests, ORION cloud delegation, native reminders and Forge trials. See [the complete setup and limits](JV_PLUGIN_ADAPTERS_MARK_XXXIII.md).

Concrete additional fixes include:

- Native gaming launch formerly reported success without launching a client. It now reads client metadata, validates identities and requests the actual OS launch URI.
- Potentially blocking desktop gaming operations run off the event loop.
- Supplied scheduled launcher work no longer runs as SYSTEM or passes a Python script path to the frozen ORION executable. The frozen entry point is handled before the GUI singleton.
- Gemini schema conversion accepts the reviewed free-form tool parameters without sending incompatible schema fields.
- Workbook creation requires supplied data, refuses formula injection, and verifies the resulting file and chart.
- Telegram requests are restricted to configured private chats and require single-use local review. Imports do not start polling, send messages or open a webcam.
- Packaging allowlists include the adapters; their private runtime settings and credentials remain excluded.
- Reserved Forge trials now pass an enforced analysis-source admission policy at generation, every LLM/mechanical repair, verification, dependency handling, publication and persisted loading. It rejects I/O, reflection, dynamic callables, unsafe imports, unsafe generated tests, dependency installation and replacement of other tools. Persisted loading compiles the same byte snapshot that was admitted; quarantine rehabilitation also fails closed.

Live Telegram sends, game acquisition, conversation sends and cloud deployment are separate account operations and were not performed by offline tests. Missing credentials remain visible as setup requirements.

### 3. Learning and chess

The action classifier had exhausted its 256 output slots. Its capacity is now 512 with migration of learned weights, optimiser state and labels. Unused outputs no longer dilute known-action probabilities.

Two bounded isolated training candidates were evaluated against 197 held-out routing cases:

| Candidate | Top-1 before → after | Top-3 before → after | Top-10 before → after | Previously correct top-1 losses | Promotion |
| --- | --- | --- | --- | --- | --- |
| Original head, 3 epochs | 83 → 100 | 112 → 131 | 149 → 154 | 4 | Rejected |
| Expanded head, 1 epoch | 83 → 101 | 112 → 119 | 149 → 142 | 6 | Rejected |

The second candidate learned labels for 287 actions, but lost top-10 coverage. The active weights were preserved. These are local language-to-action classifiers, not training of Google's hosted model or evidence of increased general intelligence.

The chess benchmark completed **48 games**, arranged as 24 colour pairs against Stockfish 18 at configured strengths 1320, 1600 and 1900. Both engines had **0.08 seconds per move**. Results: **28 wins, 10 draws, 10 losses**; fitted strength **1,801**, bootstrap 95% interval **1,722–1,899**. This is a calibration on Stockfish's UCI Elo scale; it is not a FIDE or Lichess rating. See [benchmark conditions](CHESS_BENCHMARK_MARK_XXXIII.md).

Continuous training and desktop/swarm training now share a lease, preventing concurrent checkpoint writers. Desktop status recognises an independently running worker. Closing the desktop preserves that worker; an explicit stop waits for its release. Rejected candidates do not replace the champion. The retained continuous worker has been generating games, but that alone is not measured improvement.

Remote play has no configured Lichess BOT token on this installation. Its transport and state tests run, but no authenticated online game or public rating can be claimed until that credential is configured.

### 4. Appearance, speech and navigation

- Removed competing face/orb materialisation ownership. Explicit Face or Orb selection stays pinned while listening/speaking state frames update colour and amplitude.
- Batched renderer bridge updates, restored current state on attachment, preserved LISTENING during microphone amplitude updates, and replaced dropped cadence with elapsed-time/deadline scheduling.
- Removed the visible drag/rotate/scatter gesture legend and collapsed the empty Inner Voice panel. Real populated content remains available.
- Removed the unused duplicate Security Centre implementation from Operations. Security owns trust, permissions, device pairing and incidents; Operations owns execution, workflows and results.
- Added shared reply PCM playback and bounded synthesis prefetch on the phone and desktop. Paragraph/device boundaries no longer deliberately recreate an output device for each sentence.
- Gemini Live producer completion and device-tail completion are now distinct. Short native audio gaps cannot reopen the microphone mid-answer.
- Explicit interruption rejects late provider PCM from an old generation. Pausing preserved shared PCM does not resynthesise and replay the whole section.

The first ORION **3.11** phone update was installed wirelessly and opened successfully, with no package-specific AndroidRuntime crash observed. The user then reported continuing paragraph/reading latency. The follow-up update **3.12** groups paragraph speech, conservatively caps generated near-silence at 260 ms and adds 30 validated Gemini choices plus explicit Phone voice in Options. The selection is stored through Android Keystore and forwarded to warm Live sessions and authenticated host speech. The user explicitly permits Android's normal voice when the selected cloud voice is unavailable; that fallback is now implemented. A partially spoken section is drained without blindly replaying it, and later untouched sections can use the phone voice. Speaking/microphone gating and cancellation remain held through the stock voice's completion callback.

The follow-up passed **74 Android tests** and **27 focused Python tests**. Full installed-device latency and selection persistence across a restart remain device checks; offline gates are not proof of satisfactory microphone-to-speaker latency.

Approved live provider measurements used fixed diagnostic text, retained the key only in memory and logged timings rather than speech content:

| Route | Setup | First audio after turn/request | Result |
| --- | --- | --- | --- |
| Gemini 2.5 Flash preview TTS | Included in request | 4.149 s | Complete 3.611 s clip; three chunks arrived together; 216 words/minute |
| Gemini 3.8 Live, the app's configured route | 0.308 s | 0.807 s | Complete PCM reply |
| Gemini 2.5 native-audio preview, compatibility probe | 0.335 s | No audio | Turn completed with zero PCM; must be handled as unavailable |

These are single provider probes, not a phone microphone-to-speaker benchmark or a promise of constant latency. They explain why repeated short TTS requests can exhaust a playback buffer despite application prefetching.

Three complete Command Deck revisions — Console, Workspace and Observatory — were prepared for the user's selection. The preview's primary interactions and widths 320, 736 and 1024 were checked. The user subsequently set aside this broader redesign; its previews are retained as historical work, with no further redesign active.

### 5. Clean-up and credential protection

- Removed the unused `projects/advancedneuralnetwork` Hello World scaffold. The projects directory is now empty.
- Removed the generated `build/_exe` and `build/_dist` directories. The current source runtime and root launcher are preserved.
- Removed publication copies r3, r4 and r5 only after verifying every one of their 886 source files byte for byte against retained ZIP archives.
- Removed 22 further obsolete APK/signature/check copies. Thirty-one test directories were preserved after access-denied results; their permissions were not altered.
- Preserved the first and r2 publication directories: their extra or changed files are not fully covered by the retained archive.
- Preserved `dist/ORION`: it contains browser profiles, databases, device pairing, conversations and learned state. Age does not establish that those files are disposable.
- The separate JARVIS project is already source-only. Its source, assets, audio credits and package lockfiles are functional; no application folder was confirmed redundant.
- Redacted a credential-shaped test fixture from the older JUnit report before staging source. The source publication scan subsequently passed for **1,249 files**, including comparison against known local credentials without logging their values.

The initial broad clean-up was rejected by automatic approval review because its contents and recovery were insufficiently established. The narrower operation was approved after archive verification and the personal-data discovery. Current APKs, environments, training data, keys and audit evidence are retained.

### 6. Globe, official feeds and World MCP

The preceding reliability work was committed as `e7a5f50` before this stage began. The existing Globe is CesiumJS embedded in `gui/globe.py`; there is no separate `Globe.js` file in this checkout.

- Added one shared, on-demand official feed service and enabled the local `orion_world` MCP. It exposes five read-only tools for CelesTrak space station predictions, USGS earthquakes, NOAA tsunami bulletins, Met Office UK warning summaries and cache health. Existing explicit disabled choices survive migration.
- Satellite positions use SGP4 and carry the orbital epoch and prediction time. NOAA information statements are not represented as active tsunami warnings. Unlocated bulletins remain inspectable reports rather than invented map locations.
- Persistent transactional request reservations prevent the GUI and MCP processes from downloading the same orbital dataset independently. Downloads are bounded, use provider backoff and preserve the last successful data with a visible stale indication. A genuine empty feed and an unavailable source are distinct states.
- Fixed retained event handlers and camera callbacks across renderer rebuilds, outdated navigation results, hidden-page feed tasks, unsafe source/news links, reset races and hidden attribution. Page shutdown cancels owned work before the shared service closes.
- Added restrained grey/crimson layer controls and source details. Request rendering replaces continuous idle rotation; hidden pages stop rendering and polling.

The live feed check returned **22 valid satellite predictions, 222 earthquakes, two NOAA bulletins and a successful empty Met Office warning feed**. The actual MCP subprocess negotiated its protocol, exposed five annotated tools and returned matching cached counts. These are observations at the test time, not permanent feed contents.

Real Cesium 1.115 browser verification loaded imagery and all selected layers with zero JavaScript/console errors. It recorded **one rendered frame in three idle seconds and zero in three hidden seconds**. Widths 1,280, 736 and 320 pixels had no horizontal overflow. The combined feed, renderer, zoom and postcode gate passed **93 tests**; the existing restricted pytest cache emitted one write warning. See [source coverage and deployment details](WORLD_FEEDS_MARK_XXXIII_2026-09-30.md).

### 7. Optional remote PC shutdown from the phone

ORION 3.13 adds **Options → Connections → PC controls**. Native pairing uses the PC's current HTTPS address while retaining independent phone AI. The authenticated endpoint requires a fresh device-bound confirmation and retains request IDs to prevent a lost connection from executing a duplicate shutdown. The phone can cancel its acknowledged 60-second countdown.

Windows forces application closure when its own shutdown timeout is positive. ORION therefore manages the countdown itself and invokes the fixed `shutdown.exe /s /t 0` command without `/f`. Unsaved work can prevent shutdown. Closing or restarting the host cancels a pending countdown, and an interrupted OS request is never replayed.

Tests inject the OS runner; this PC is not powered off during verification. Independent review also found an unbounded process-reaping wait and an inaccurate persistent `executing` state after a pre-command save failure. Both were corrected and covered by the final 29 power tests. A cancellation request ID cannot be reused for a different countdown. The first Android build caught a nested dialogue callback shadowing its `BottomSheetDialog`; the named callback correction passed the subsequent release build. See [the endpoint and use guide](REMOTE_PC_POWER_MARK_XXXIII.md).

The final **3.13 / versionCode 35** update was installed and opened wirelessly, preserving the existing app data. Its SHA-256 is `38C751A96120F5127B10607ABA29AA78C9DD23CD59AFE6F9A6FB2E4BFDDD5494`. The matching phone certificate and separate production certificate were both verified before installation. A separate production-signed 3.13 APK is retained; it cannot replace the phone's existing debug-signed package without a migration.

### 8. Additional hosted AI, MCP transport, offline routes and prepared cloud

The preceding Globe/power phase was committed as `a6e414e`. This integration phase was built and installed before the user deferred further phone work.

- Added seven phone text-provider profiles, giving eleven in total: Gemini, Groq, OpenRouter, Cloudflare, Cerebras, Mistral, Hugging Face, NVIDIA NIM, SambaNova, Together and DeepSeek. Each service uses its own encrypted customer credential and a reviewed endpoint. Selecting a service does not provision an account or promise free allowance. Desktop defaults now use the documented Cerebras/Together models and endpoint whilst preserving explicitly saved choices.
- Corrected hosted reply streaming that inherited a 4,096-event speech limit. Long text streams are now byte-bounded and stop immediately at `[DONE]`, including when the server retains its connection. Incomplete token-truncated tool calls cannot execute. Tool receipt identifiers survive the round trip.
- AI configuration changes cancel the active console/background turn and speech generation. Revision checks reject old captured segments and queued callbacks; configuration changes do not restart the service or create a competing microphone owner.
- Added current stateless Streamable HTTP MCP support and negotiated 2025 revisions. Discovery, schemas, IDs, pagination and payloads are bounded. Failed tool calls are not silently replayed. OAuth, old two-endpoint SSE, resumable session streams and unsupported multi-step workflows remain documented limitations; no newly configured live MCP account is claimed.
- Added native BRouter coordinate routes, manual coordinate planning, saved places and GPX restoration. Parsing rejects external entities, invalid geometry and oversized/decompression payloads. Route progress uses measured GPS and reports stale/off-route conditions. Satellite imagery is not bundled.
- Installed BRouter 1.7.9 and seven official UK road-graph cells, totalling **294,157,904 bytes**, with each phone copy SHA-256 checked before activation. The native map displayed the installed UK packs and a fresh measured position. It correctly refused a later stale indoor fix. Actual route calculation, route restoration and driving remain device acceptance checks.
- Prepared a validated, non-root Render trial image, Blueprint and headless entry point. Container ingress and local reverse-proxy ingress have distinct binding rules; forwarded headers do not grant authentication. Private stores and credentials are excluded from the image context. The user requested preparation only: no deployment, source push or cloud credential transfer occurred. Docker Desktop's daemon was unavailable, so the Linux image build remains unverified. Render's free service sleeps and loses disposable pairing/memory state; it is not an always-on customer deployment.
- Corrected source-export allowlists for the new AIDL interface and Render Dockerfile. Ignored generated Kotlin session files rather than including them in the distributable source.

The final **3.14 / versionCode 36** compatible APK was built, installed wirelessly and opened, retaining the existing app data. SHA-256: `C7B4F7A2F58BD495B609E8D3FB7149D9CCEB658CFE0502B6EE92B26444160E0B`. A separately signed production APK is retained with SHA-256 `407E89227DE688468705C06232EADB20135B2147A927DCE20A1C9A1AB27A8548`; it cannot replace the existing debug-signed installation without a deliberate data/Keystore migration.

The user then deferred **all phone fixes, installations and device checks until returning home**. No subsequent phone acceptance is claimed. See [phone integration details](PHONE_INTEGRATIONS_2026-09-30.md), [MCP compatibility](PHONE_MCP_INTEROPERABILITY_2026-09-30.md), [offline route limits](OFFLINE_CAR_ROUTES_2026-09-30.md) and [prepared cloud trial](../deploy/README_RENDER_FREE.md).

### 9. Final desktop/backend and remote-chess review

- Headless startup now owns partially opened stores, sessions and background tasks within one cleanup path. Gateway bind/configuration failures and cancelled startup release acquired resources. Shutdown retires HTTP ingress before closing the provider sessions used by remote turns, then closes stores and performs the final checkpoint.
- A fatal headless exception now exits with status **1**, allowing a service manager to detect failure. A requested console interruption remains a normal stop.
- Recognised official DeepSeek/Cerebras requests now apply supported lower-thinking settings for routine replies. Explicit `ORION_MODEL_THINKING=1` preserves the operator's policy. Custom hosts/models are unaffected by those service-specific fields. No live provider speed claim is inferred from request tests.
- Expired outgoing chess invitations no longer leave the remote-game slot permanently occupied. Administrative stop requests have separate pending ownership from authoritative terminal states; a failed or cancelled abort releases that ownership and reconciles a quiet game stream with fresh server state. Successful/terminal states still prevent late moves.
- Independent review reproduced a second invitation fault: the shared 30-second reservation could expire whilst a challenge HTTP request was still waiting. Incoming acceptance and outgoing challenges now keep non-expiring, token-owned reservations until HTTP/cancellation settles. Only confirmed invitation IDs expire; duplicate incoming events cannot send conflicting accept/decline requests. Regressions cover requests exceeding 30 seconds and delayed cancellation cleanup.
- Remote resignation no longer reports success after both resign/abort requests fail. Partial failure produces a failed receipt with counts, without disclosing private transport exceptions or claiming a game has finished before its final state arrives.
- Rechecked all twelve JV adapter manifests and source contracts and the Forge loading, admission, permission and rollback paths. Tests are local; no Telegram message, Lichess challenge or external account action was sent.

A real isolated headless composition used a fresh disposable store and **127.0.0.1:18080**, with existing provider keys removed from that subprocess. It returned **HTTP 200 in 1.016 seconds**, then cancelled and closed cleanly with the listener retired. Sampled process RSS was **57.9 MiB** on this Windows machine; this is neither a Linux memory ceiling nor free-cloud acceptance. The production PC gateway was stopped during the final read-only check; the isolated test did not launch the user's desktop assistant or create a public route.

### 10. Desktop targeting, prompt speed and original plugins

Following the user's account replies, work continued on the desktop/backend,
with phone work still deferred. Three agents prepared independent targeting,
provider/memory and Forge/plugin changes; the primary review then verified
their combined source, regressions and real local boundaries.

- Named desktop targeting now refuses equal-ranked ambiguity, uses observed
  automation IDs, and checks foreground/runtime identity, control state,
  geometry and hit-testing before input and after pointer travel. Failed or
  uncertain actions are not repeated. Invalid/disconnected monitor targets
  and physical screen gaps are refused. Reversible worker DPI scopes and
  same-capture screenshot scale/origin metadata address mixed-DPI geometry.
- Installed the four-tool, read-only `orion_desktop` MCP. Its real stdio
  subprocess listed all tools, negotiated `2025-11-25`, reported accessibility
  available and exited cleanly. No live click or screenshot was issued by
  that acceptance. Added missing desktop/World script data to frozen-build
  configuration; no new complete frozen desktop build is claimed.
- Memory context now reads bounded indexed values and recall keys in one
  snapshot, preserving stored knowledge and user-category priority. It also
  corrects omission of the two newest bulk-category facts. On 50,000 synthetic
  facts, seven paired repetitions measured median preparation **457.948 ms →
  3.458 ms**; this is not total provider/phone reply time.
- Provider context preparation runs off the live loop within the request's
  existing deadline. Account/endpoint-scoped catalogue reads are shared for
  five minutes. Completed streams briefly discard bounded trailing data
  instead of waiting another half second. Recognised official Gemini requests
  use supported lower-thinking fields; explicit operator thinking stays valid.
- Added original `orion_latency_report` and `orion_batch_planner` plugins.
  They calculate supplied timing statistics and dependency schedules through
  the native module/manifest contract. Receipts disclose that neither measures
  live services nor executes the planned actions.
- Customer Forge mode requires a restricted container and passed type check,
  refuses generated dependency installation/host activation, and saves inert
  JSON review candidates outside the loader directory. Snapshotted execution
  settings, finite time budgets, post-exit output enforcement, memory failures
  and unconfirmed container cleanup have regression coverage.
- Started the existing local Docker engine and built the verifier image.
  A real candidate passed type checking, conformance and arithmetic assertions
  in **7.516 seconds**, with no host import/activation or dependency install.
  A separate same-command probe confirmed UID 65534, denied networking and
  mounted-file writes, and 512 MiB/64 PID/one CPU cgroup limits.
- Built the prepared Render Linux image locally with no private state or keys.
  Loopback-only acceptance returned health **HTTP 200 in 1.204 seconds** and
  unauthenticated power **HTTP 401**, running as UID 10001. Sampled container
  memory was **38.21 MiB under a 512 MiB ceiling**; the container was retired.
  No cloud resource, source push, provider request or credential transfer was
  involved. This does not prove Render cold-start or phone compatibility.
- Applied the user's Google AI Studio-only account preference by disabling
  the previously enabled OpenRouter, Groq and thoughts profiles, preserving
  every saved credential/model/integration value. Google profiles remain on.
- Recorded the user's Lichess account and JV redistribution permission. No
  licence terms or attribution were invented. Lichess token/BOT status and
  live rated play remain unverified.

The new focused gate passed **219 tests with one optional environment skip**.
Its first attempt exposed an oversized synthetic pytest ID exceeding Windows'
environment-string limit; bounded IDs corrected that harness fault. The first
new whole-project run found two World feed performance omissions: eager
`aiohttp` import and a SQLite connection missing shared performance pragmas.
Both were fixed; real WAL/settings and fresh-process import regressions were
added, and the affected database/startup/World MCP/feed gate passed **113 tests**.
The final stable run completed in **578.39 seconds: 8,622 passed, 21 skipped,
zero failures**, with four warnings and seven additional passed subtests. A
separate permitted browser run then passed **34 tests**, including all twelve
previously skipped because Chromium could not launch in the restricted process.
The nine remaining skip conditions are two Windows symlink permissions, two
Windows Search index permissions, two opt-in live Ollama checks, unavailable
MCP SDK v2 and two conditional dependency checks. Counts from overlapping
gates are not added as a unique project total.
See [targeting, speed and Forge details](DESKTOP_ACCURACY_AND_SPEED_2026-09-30.md).

## Verification ledger

The initial whole-suite run collected 8,123 tests: 8,098 passed, 17 skipped and eight failed. Failures exposed obsolete Mark/version assertions, source-inspection races while agents were editing speech modules and a renderer timing failure during concurrent heavy work. Identity assertions and documentation were corrected, source was stabilised, and the five affected suites then passed **37 tests**.

The final stable whole-project run, including the Forge and speech follow-ups, completed in **543.02 seconds: 8,219 passed, 17 skipped, zero failures**, with one `audioop` deprecation warning. Its XML and complete skip reasons are retained in `reports/mark_xxxiii_full_stable.xml` and `.log`. Skips are optional environment/live integrations, not counted as functional passes. The completed Gradle 9.5 daemon was confirmed idle and stopped before the renderer tests, releasing its retained memory; available memory subsequently measured about 5.7 GiB. Ongoing chess training was preserved.

That run exposed missing declared `python-docx` and Python Playwright dependencies in the active environment. They were installed, and the complete document/browser suites then passed **68 tests**, including all eight previously skipped checks. `pip check` reports no broken requirements. The nine remaining skip conditions concern Windows symlink/index permissions, intentionally disabled live Ollama tests, SDK v2 availability and conditional dependency tests. The follow-up emitted one pytest-cache write warning under the restricted run; it did not affect the functional results.

Completed focused evidence includes:

| Gate | Result |
| --- | --- |
| Git-visible Python syntax scan | 971 files; zero syntax errors after Globe/power follow-up |
| Backend inventory | 493 files; zero syntax errors; 67 classified IPv4 literals; zero direct blocking calls detected in async functions |
| Critical Ruff errors (`E9,F63,F7,F821`) | Passed after follow-up |
| Stable whole-project Python gate | 8,219 passed; 17 skipped; zero failures |
| Declared document/browser dependency follow-up | 68 passed; eight formerly skipped checks resolved; `pip check` clean |
| Intent migration, durable task dismissal and metrics | 69 passed |
| Avatar/UI regressions | 223 passed |
| Chess regressions | 55 passed, plus 11 final ownership regressions |
| Actual renderer JavaScript checks | 5 passed |
| Speech regressions | 101 Python tests passed |
| Android unit tests | 77 passed after PC controls; zero failures/errors/skips |
| JV adapters and packaging | 116 focused tests passed |
| Forge analysis trial admission | 79 dedicated regressions passed; broader Forge/security/plugin gate 484 passed |
| Capability preservation | Passed; no missing capability or duplicate identity |
| Android release build and lint | Final 3.13 compatible and production builds succeeded; zero lint errors, 121 warnings |
| Requested design preview | Syntax, interactions and three responsive widths passed |
| Wireless installation | 3.13, versionCode 35, successful retained-data update |
| Globe, World MCP and related remote/networking follow-up | 510 passed, two environment skips, zero failures; one pytest-cache write warning |
| Remote PC power | 29 dedicated injected-runner tests; 141 related gateway regressions |
| Credential/publication scan after follow-up | 1,274 source files passed; local credentials compared without disclosure |
| Final integration syntax scan | 975 Git-visible Python files; zero syntax errors |
| Final backend source inventory | 497 files; zero syntax errors; 69 classified IPv4 locations; zero direct blocking calls detected in async functions |
| Hosted provider, gateway and integration contracts | 205 passed; zero failures |
| Runtime/startup, telemetry, task and shutdown follow-up | 116 passed; zero failures |
| Headless lifecycle, Render/proxy and provider policy final gate | 116 passed; zero failures |
| Final Android integration unit tests | 125 passed; zero failures/errors/skips |
| Source export/publication regressions | 35 passed; zero failures |
| Final plugin/Forge loading, verification and permissions | 155 passed; zero failures |
| Final chess, dispatch and credential-redirect regressions | 129 passed; zero failures, including all 15 administrative regressions |
| Independent final chess review | Reproduced and verified invitation ownership fix; all 15 administrative regressions passed independently |
| Final capability preservation | 154 tools, 155 handlers, 22 pages, 30 public plugins, 8 local plugins, 5 entry points and 7 fallbacks; no missing identity or duplicate |
| Final credential/publication audit | 1,294 source files passed; known local credentials compared without disclosure; no archive/publication created |
| Final critical Ruff and diff whitespace | Passed |
| Final integration release build and lint | 3.14 compatible and production builds succeeded; zero lint errors, 137 warnings |
| Final integration wireless installation | 3.14, versionCode 36; successful retained-data update before phone deferral |
| Real isolated headless composition | HTTP 200 in 1.016 s; clean cancellation/shutdown; listener retired |
| Targeting/speed/Forge full stable project gate | 8,622 passed, 21 skipped, zero failures; 578.39 s; seven passed subtests |
| Browser permission follow-up | 34 passed; all twelve Chromium permission skips resolved; no phone installation or device test |
| New targeting/speed/Forge/plugin regressions | 219 passed, one optional skip |
| World feed performance corrections | 113 database/startup/feed/World MCP tests passed |
| Real customer Forge container | Typecheck/contract/assertions passed; inert receipt; denied network and mounted-file writes; resource caps confirmed |
| Real prepared Linux trial container | HTTP 200 in 1.204 s; unauthenticated power HTTP 401; UID 10001; retired cleanly; no cloud deployment |
| Desktop targeting MCP subprocess | Four tools discovered; accessibility available; clean exit; no clicks/screenshots |
| Latest Git-visible Python syntax | 986 files; zero syntax errors |
| Latest backend inventory | 499 files; zero syntax errors; 69 classified IPv4 locations; zero detected direct blocking async calls |
| Latest capability preservation | 154 tools, 155 handlers, 22 pages, 32 public plugins, 8 local plugins, 5 entry points, 7 fallbacks; no missing/duplicate identities |
| Latest critical Ruff/dependencies/publication | Critical Ruff passed across core/tools/tests; pip check clean; 1,308 source files credential-audited before final staging |

These are separate recorded gates and some cover the same tests; their counts must not be added as a unique project total. The earlier 8,219-test whole-project run predates the subsequent Globe, power and 3.14/backend additions. The backend inventory is a static diagnostic. It cannot detect every indirect blocking call, account error, driver fault or callback race. Full manual review of every line is not claimed.

## Remaining concerns and completion gates

1. **Speech latency:** paragraph grouping, warm selected-voice sessions, pause filtering and later stream/configuration fixes are implemented. Provider synthesis can still arrive slower than playback; installed 3.14 needs microphone-to-speaker timing and listening checks. The user deferred those phone checks until returning home. Phone voice is an explicit alternative and permitted fallback.
2. **Customer APK identity:** the existing phone uses the Android debug certificate. A matching certificate was needed to update without losing its Keystore-bound credential. A production-signed 3.14 APK is retained and verified separately, but customers need a stable production signing identity and a deliberate migration strategy.
3. **Account acceptance:** the user confirmed Lichess and Google AI Studio accounts. Lichess still has no configured BOT token/status or live rated acceptance. Non-Google AI profiles were disabled with credentials preserved. Telegram and external account MCPs need their own configuration; no live success is inferred from test doubles.
4. **Redistribution:** the user confirmed permission for the JV collection on 30 September. The collection still lacks stated licence/author metadata; no terms were invented or attribution removed. Retained launcher code has 72 legacy style/broad-catch findings to refactor before customer distribution.
5. **Forge containment:** restricted container verification and customer verification-only receipts now have real local acceptance. Source admission and model-written tests still do not establish general correctness. Personal live activation retains its existing policy; customer mode deliberately does not activate generated code, and separate execution/publication remains a review decision.
6. **Build maintenance:** the final Android integration lint has 137 warnings, and Gradle reports APIs deprecated for Gradle 10. These are tracked maintenance work, not a failed current build. Further phone work remains deferred.
7. **Clean-up permissions:** 31 old test directories and the unenumerated baseline are retained. The legacy installation requires a deliberate data migration before any whole-directory removal.
8. **Offline driving:** BRouter and seven UK graph cells are installed, and native coordinate planning is implemented. Actual route calculation, saved-route restoration and driving remain unverified. Native spoken turns, automatic rerouting, offline address search and locked-screen route following are absent; OsmAnd remains the full guidance option. The app does not contain offline satellite imagery. See the [route guide](OFFLINE_CAR_ROUTES_2026-09-30.md) and existing [map audit](OFFLINE_UK_MAP_2026-09-29.md).
9. **Visual scope:** the user set aside the broader Command Deck redesign on 30 September. The existing first window and Command Deck remain in place; requested appearance defects and clutter removals were implemented independently.
10. **Globe coverage:** satellite positions are predictions for the bounded space stations group. NOAA coverage is limited to its two bulletin feeds; Met Office warnings cover the UK. The imagery renderer still needs online tiles and the Cesium CDN. This desktop Globe does not expand the phone's offline routing capability.
11. **Remote power:** the current PC HTTPS address and pairing must be configured. Temporary Cloudflare addresses change after tunnel restarts. Tests validate the command contract and prevent duplicate execution; actual power-off remains a deliberate user action.
12. **Prepared cloud:** deployment is explicitly deferred. A local Linux build and isolated health/authentication startup passed; actual Render hosting, cold start and paired-phone operation remain unverified. Render Free's sleep and disposable pairing/memory state make this a single-owner trial, not a continuously available customer service.
13. **Deferred phone prompt:** the AI prompt still describes a blanket Desktop Mode requirement, although explicit PC power controls are available in Options → Connections. Reconcile that wording in the deferred phone phase; no phone source was changed after the user's instruction to pause it.

## Local evidence

Reproducible tools and persistent regression tests are committed as source. Local execution logs, candidate weights, device APKs and clean-up inventories remain under ignored `reports/mark_xxxiii_2026_09_30` and `releases/android`, preventing private runtime data from entering Git.

The first source phase is committed as `e7a5f50`; Globe/remote power as `a6e414e`; integration/backend as `6e4a2c7`; targeting/speed/Forge as `903e034`. The subsequent Globe accuracy/BOT setup phase has its own focused gates and local commit. No push or public release is implied. APKs, reports, private stores and credentials remain outside committed source.

## Globe accuracy and BOT setup follow-up

The later request identified a need for clearer Lichess BOT setup and pinpoint
Globe navigation. The [accuracy record](GLOBE_ACCURACY_2026-09-30.md) documents
the camera latitude offset, inconsistent map geocoding, silent ambiguous-name
selection, missing-coordinate cache default, coarser reverse-address cache,
queued old navigation and unlocated report selection faults. Exact coordinate
input now bypasses geocoding/news; source provenance and result choice remain
visible without inventing a source accuracy radius.

Real Cesium 1.115 acceptance used six supplied positions, including both poles
and both date-line sides. Marker position error was zero, the largest projected
camera-centre error was 0.003247 pixels, and page JavaScript errors were zero.
Imagery requests were blocked, so this is a renderer maths measurement rather
than an imagery/geocoder accuracy claim. The isolated native Qt chooser and
bounded offline browser regressions passed. The final combined Globe/BOT gate
passed **199 tests with zero failures or skips in 49.78 seconds**. Its 25 account
helper cases use mocked requests. Critical Ruff and whitespace passed;
capability preservation found no missing or duplicate identity; the known-key
publication scan passed 1,316 source files. No new whole-project total is inferred.

The new Lichess helper defaults to account verification. Conversion requires
an explicit user-driven `-Upgrade` and exact authenticated username after an
irreversible-conversion warning. Tests replace both HTTP and interactive input;
no account was converted, token provisioned or game started during this phase.
Phone work was deferred at that stage. The latest request resumes it; cloud
deployment remains prepared only.

## Recursive Forge, reflections and resumed phone phase

The [Forge record](FORGE_RECURSIVE_IMPROVEMENT_2026-09-30.md) documents independent
benchmarks, bounded generations, evidence-led feedback, correctness/resource
comparison, immutable contained execution and durable review candidates. The
review fixed cancelled-write ordering, installed-wrapper admission, required
type annotation, schema-provenance and local-image resolution faults. No candidate was activated
and no hosted weights were trained during acceptance.

Cognition now owns bounded completed reflections with recorded context and
evidence. Operations no longer contains its invisible duplicate. Routine
reflections, successful maintenance observations and heartbeat analysis do not
spam the terminal. Genuine diagnostics remain visible, with backoff and repeat
suppression. External notes survive late page construction and are journalled
once; shutdown rejects late deliveries.

The [3.15 phone record](PHONE_VOICE_ROUTE_FIXES_3_15_2026-09-30.md) covers the fixed
capture waits, overlong-command tail defect, shared route-write race and PC
prompt inconsistency. Both APK variants are built and signature-checked. The
owner update matches 3.14's certificate and preserves the existing data path.
Wireless ADB and mDNS still find no device: this update has not been installed,
and microphone-to-speaker timing, real BRouter calculation and physical driving
remain unverified.

| Gate | Actual result |
| --- | --- |
| Real restricted reference benchmarks | 135 independently scored observations; both references correct; required typing and containment passed |
| Real recursive fixture | First faulty candidate rejected; second corrected generation qualified; zero model requests or activation |
| Initial Forge/reflections integration | 214 passed |
| Heartbeat, retry, context and UI follow-up | 74 passed |
| Whole-project Python sweep | 8,864 passed, 29 skipped, one obsolete UI wording assertion failed; 645.16 seconds |
| Final correction and related follow-up | 159 passed; obsolete assertion updated to the delivered evidence heading; no remaining failure in that gate |
| Final callable-contract, schema and experiment-budget follow-up | 237 passed; printed benchmark/test dictionaries cannot replace the actual container schema, parameter changes cannot qualify as improvements and parallel experiments cannot stack resource budgets |
| Authorised Chromium permission follow-up | 43 passed, zero skips/failures; resolves all twenty browser skips; one pytest-cache write warning |
| Android final unit/build/lint | 139 passed, zero failures/errors/skips; debug/release builds succeeded; zero lint errors, 137 warnings |
| Git-visible Python syntax | 1,001 files, zero syntax errors |
| Capability preservation | Passed; no missing or duplicate identities |
| Known-credential source audit | 1,331 source files passed after final documentation; no archive or publication |
| Critical Ruff and new module Ruff | Passed |

The full sweep imported the earlier UI assertion before its correction; its
failure is retained in the XML rather than rewriting history as a clean whole
run. The corrected actual Qt widget and related Forge/Cognition/thought tests
passed in the final follow-up. Counts are separate gates, not an additive
unique-project total. Browser-permission skips receive a separate authorised
Chromium follow-up, which passed all 43 tests. Nine remaining skips cover host
symlink/Search restrictions, absent optional MCP v2/local-model services,
environment-specific training refusal and removed plotting dependencies;
remaining optional/account/environment checks are kept
explicit. Python also reports the existing `audioop` deprecation, requiring
maintenance before moving the telephony bridge to Python 3.13.

The former phone deferral and PC-prompt concern above are historical: the
latest user request resumes phone acceptance and the prompt is repaired in
3.15. Small local neural components can learn curated task/chess outcomes;
that does not retrain Gemini. Existing legacy personal Forge activation,
customer review/execution decisions, missing native spoken-turn/rerouting/
address/locked-following functions, account tokens, image licensing, provider
quota/latency and requested later Globe additions remain practical limits.

## Offline guidance and voice concurrency follow-up — 3.16

The [3.16 record](PHONE_NAVIGATION_VOICE_3_16_2026-09-30.md) documents the next
phone source phase. Its new location foreground service reads supplied turns
offline, retains GPS following whilst locked and recalculates sustained
deviations for the current BRouter car journey. It starts explicitly from the
visible app with precise location granted and has a Stop notification. It
does not restart automatically, infer turns or claim that imported tracks are
car routes. Native offline address search/satellite photography remain absent.

The voice audit found blocked stop-only capture during streamed replies,
unnecessary idle Live-session resets, unguarded queued speech, delayed stop
results applying to newer playback and navigation speech entering dictation.
These are repaired with separate stop recognition, exact playback generations,
active-only stopping, bounded interruption admission and shared audio leases.
Provider inference/synthesis and Bluetooth/vehicle latency still need device
timings. No additional hosted-model request was made during these checks.

The combined gate passed **174 Android unit tests** with zero failures/errors/
skips, and **180 related Python phone/backend tests** in 56.15 seconds. Both
Android variants compiled/assembled; lint has zero errors, with existing
style/localisation/maintenance warnings retained. Final packaging and signature
evidence is in the 3.16 record. The capability-preservation check again found
no missing/duplicate identities. These counts remain separate gates, without
claiming another clean whole-project run.

The desktop was brought back up with the existing checked launcher. The
gateway listens on `127.0.0.1:8000` and its local health endpoint returned
`ok=true`. No old temporary HTTPS address is thereby verified. Wireless ADB
and mDNS still found no phone, so no 3.16 installation, actual microphone/
Charon timing, screen-lock service or driving success is claimed. These
physical gates precede the retained later Globe/chess additions.
