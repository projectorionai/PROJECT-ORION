# O.R.I.O.N.

Current desktop release: [Mark XXXXI Ascendant 41.1.1](docs/MARK_XXXXI_RELEASE_2026-10-29.md).
Previous desktop release: [Mark XXXX Aegis 40.8.0](docs/MARK_XXXX_RELEASE_2026-10-27.md).
Earlier desktop release: [Mark XXXIX Sentinel 39.0.0](docs/MARK_XXXIX_RELEASE_2026-10-27.md).
Earlier desktop release: [Mark XXXVIII Clean Deck 38.0.0](docs/MARK_XXXVIII_RELEASE_2026-10-27.md).
Earlier desktop release: [Mark XXXVII Screen Face 37.5.0](docs/MARK_XXXVII_RELEASE_2026-10-24.md).
Earlier performance release: [Mark XXXV performance refinement 35.0.2](docs/MARK_XXXV_PERFORMANCE_2026-10-15.md).
Earlier reliability release: [Mark XXXV 35.0.0](docs/MARK_XXXV_RELEASE_2026-10-15.md).
Earlier source audit: [Mark XXXIII audit](docs/MARK_XXXIII_AUDIT_2026-09-30.md).
Earlier release and plugin review: [23 September release review](docs/RELEASE_REVIEW_2026-09-23.md).

**Open Resolution Intelligence Overt Network** is a Python desktop assistant
with a PyQt6 interface, voice interaction, persistent memory, desktop tools and
a camera-based Vision Lab. It combines local services with optional cloud or
locally hosted language and vision models.

**Package version: 41.1.1 · Mark XXXXI**

## What changed in Mark XXXV

- Research has durable run IDs, explicit status/history/open selection, restart
  interruption records, and cancellation that retains late-written partial files.
- Research programme queues advance from settled outcomes. Failed or empty
  synthesis remains incomplete even when useful notes were saved.
- Text fallback and reconnect context retain structured outcomes and exact paths.
- Native window operations run off the event loop and verify OS state before
  reporting success. Empty video evidence cannot establish completed analysis.
- OCR reports unknown confidence honestly, and optional local OCR setup does not
  silently download models. Capability health distinguishes dependency checks
  from timestamped operation evidence.
- Messaging preserves failed/partial outcomes; synchronous delivery runs on a
  worker. Scheduled plugin vault values use owner-scoped access instead of
  process-wide environment injection, with a documented helper migration.
- Optional metrics database startup and sampling run outside the interface loop.

The release guide records validation, boundaries and remaining work. Cloud
accounts, physical devices and provider quality require their own acceptance
checks; local regressions do not establish those results.

## What changed in Mark XXXIII

- A persistent FACE/ORB selection and one materialisation timeline; listening
  amplitude no longer makes the avatar report that ORION is speaking.
- Telemetry and Security posture scans run outside the Qt/asyncio thread.
- Continuous reply playback, sentence prefetch and turn-level microphone
  gating keep speech sections together without enabling echo between them.
- Supplied JARVIS capabilities use ORION's plugin contract and native services.
  Credentials, device permissions and online chess access remain explicit.
- Obsolete tasks have durable removal records and exact-title Notion archival.
- A reproducible chess calibration and guarded intent-training candidates
  replace unqualified claims of strength or intelligence. See the audit for
  the measurement conditions, results and limitations.
- Official satellite predictions, earthquakes, tsunami bulletins and UK
  weather warnings share a cached World service and read-only MCP. The Globe
  renders on demand and suspends hidden-page work.
- The phone has selectable cloud voices with normal phone speech as fallback,
  and optional paired PC controls with a cancellable shutdown countdown.

The wider Command Deck layout revisions await design approval. See the
[World feed guide](docs/WORLD_FEEDS_MARK_XXXIII_2026-09-30.md) and
[remote PC controls](docs/REMOTE_PC_POWER_MARK_XXXIII.md) for current coverage
and configuration.

Read the [current capabilities summary](docs/CAPABILITIES_2026-09-18.md) for
implemented features, dependencies and practical limits. Older architecture
notes and roadmaps describe development history; they are not release guarantees.

## What changed in Mark XXXII

- **The Brain, redrawn.** The Command Deck's Brain page is now two native
  instruments: the Iris (ORION's state, load and busy systems) over the Signal
  Lanes (each request stepping through hear, think, act, speak and recall over
  the last twenty seconds, with the measured time to his first word). No
  WebGL page, and nothing is drawn while he is idle.
- **Navigation that follows the screen.** Arrows and Ctrl+Tab move in sidebar
  order and wrap; Alt+Left/Right and the mouse's side buttons go back and
  forward; Ctrl+1–9 jump straight to a page; Esc returns to the Brain; the
  header shows where you are.
- **A Live channel that stays up.** A dropped connection resumes the same
  conversation within half a second instead of standing Live down for 45 s
  and forgetting it; a tool that runs long is answered and finishes in the
  background, so no single tool can freeze him. Every connect and drop is
  journalled in `diagnostics/live_sessions.jsonl`.
- **Lighter on memory.** MCP servers start when first used (about 1 GB less at
  boot), GPU meters read NVML in-process instead of launching `nvidia-smi`,
  and the face reloads itself if Windows reclaims its renderer.
- **Ten MCP servers** — filesystem, Playwright, memory, SQLite, git, Context7,
  MarkItDown and more — with GitHub, Home Assistant, Postgres and Google Drive
  ready for credentials. Anything that writes, pushes or spends asks first.
- **Chess analysis.** Plain-language move explanations and keyboard and voice
  navigation. Historical strength claims require their original match evidence;
  the Mark XXXIII audit supplies a reproducible current calibration instead.

## What changed in Mark XXXI

- **Research that reads the web.** Every run searches (Google via the Gemini
  key, then the DuckDuckGo MCP server, Bing and others), opens and reads each
  page, and takes notes with the URL attached — shown live, step by step, on
  the Command Deck's Research console.
- **Model routing that heals itself.** Gemini answers text work through its
  OpenAI-compatible endpoint; a provider that retires a model is moved to one
  it still serves, and an account out of credit moves to free models rather
  than truncating replies.
- **Exact screen reading.** Windows UI Automation reads the text of Notepad,
  Word, Edge, Chrome, Discord and most other apps character for character;
  Windows OCR (read at 2x) and DXGI capture cover games and custom surfaces.
- **Whole-PC file search** by name, type, size and date, with open/reveal.
- **Self-repair that produces patches**: exact search/replace edits, compiled
  and tested in a sandbox copy, applied only with approval.
- **MCP servers that run**: fetch, DuckDuckGo and Notion out of the box.
- **Face or orb, on request**, and a compact orb that is its own window.

## Vision Lab

Choose a camera by name, start it and frame anything — an object, a document,
a room, a part or a circuit board — then capture an inspection. Modes: identify
anything, read text, count items, or electronics. A labelled grid over the view
is also drawn on the image the vision model sees, and every localised finding
is pinned to its grid cells. Numbered observations stay attached to the
captured image; follow-up questions reuse that capture.

Visual observations cannot establish electrical function, hidden faults or
anything that is not visible. Live contours indicate geometry, not confirmed
identifications.

## Run from source

Windows is the primary desktop platform. The current development environment
uses Python 3.13. Some optional dependencies require additional models, software
or hardware; see the comments in [requirements.txt](requirements.txt).

From PowerShell in a clean source directory:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe orion.py
```

Configure provider credentials through the app's settings or your process
environment. [.env.example](.env.example) lists optional environment variables;
ORION does not automatically load a `.env` file. Cloud features need a configured
provider. Local language-model features need a running compatible local server
and suitable installed models. Camera and microphone features need the
corresponding devices and OS permissions.

For Cloudflare Workers AI and other hosted text providers, see the
[hosted provider setup guide](docs/HOSTED_AI_PROVIDERS.md).

The full requirements file includes heavy optional packages. First-time package
and model downloads are separate from normal application startup. Headless
deployment has its own [dependency list](deploy/requirements-server.txt) and
[deployment guide](deploy/README_ORACLE_CLOUD.md).

## Tests

```powershell
.venv\Scripts\python.exe -m pip install pytest
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe -B -m pytest -q
.venv\Scripts\python.exe -B tools/prepare_public_source.py
```

The publication check examines current source, including new files Git has not
yet staged. It blocks unexpected paths and recognised credential formats.
It does not audit old commits or replace review of personal content.

## Build and publish

Use the [publication guide](docs/PUBLISHING.md) before uploading a working copy
or distributing a build. A live installation contains private state and must
not be uploaded as a complete folder. Source exports exclude Git history,
credentials, databases, recordings, browser profiles and installed executables.

The standalone builder seeds only public defaults. It refuses to overwrite an
existing installation containing personal configuration. Build from a clean
source export in a separate directory:

```powershell
.venv\Scripts\python.exe build_standalone.py
```

The result is `dist/ORION/ORION.exe` with its supporting files. Generated
executables are not included in source. Existing installations may need their
desktop/taskbar shortcuts recreated after adopting the neutral public app identity.

## Project map

| Path | Purpose |
| --- | --- |
| `orion.py` | Desktop/headless entry point |
| `orion_core/` | Services, providers, dispatch, GUI and supporting engines |
| `tests/` | Regression tests |
| `tools/` | Publication and development helpers |
| `config/knowledge_packs/` | Reviewed generic knowledge packs |
| `config/custom_tools/` | Explicitly reviewed messaging, JSON validation and camera helper plugins; other local plugins remain ignored |
| `android/` | Optional Android companion source |
| `deploy/` | Optional headless/container deployment |
| `docs/` | Guides, dated capability summary and historical design notes |

See [security and private data](SECURITY.md) for storage and reporting guidance.

For beginner cybersecurity code exercises, scoped penetration-testing labs and
electronics defence lessons, see the [security learning guide](docs/SECURITY_LEARNING.md).
Teaching defaults to one small step with pause/resume support; see
[teaching at your pace](docs/SHORT_STEP_TEACHING.md).

## Licence

O.R.I.O.N. is free software under the
[GNU Affero General Public License v3.0](LICENSE).

You may read, run, study and modify it. If you distribute a modified version —
or run one as a service that others interact with over a network — you must
make your source available under the same terms. Section 13 is what makes the
network case explicit, and it is the reason this licence was chosen over a
permissive one.

Copyright (C) 2026 Project ORION contributors.

Holding the copyright, the author remains free to offer the same code under
different terms; the AGPL governs what everyone else may do with it. For a
commercial licence, ask.

Third-party material redistributed here — the MediaPipe canonical face mesh
under Apache-2.0 — is listed in [NOTICE](NOTICE), along with an
acknowledgement of the MARK LIV release notes, whose *approach* informed
ORION's avatar, lip-sync, echo guard and audio-device probing. No code from
that project is included; those subsystems were written independently for this
codebase.
