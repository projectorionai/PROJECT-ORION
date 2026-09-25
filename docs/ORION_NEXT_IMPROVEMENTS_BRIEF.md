# O.R.I.O.N. — Next Improvements Brief (Mark X.12 candidate)

**Purpose of this document:** a single, comprehensive working brief for the
next development pass. It is organised the same way the existing `docs/*_AUDIT.md`
files are — audit → gap → phased roadmap — but consolidates **UI, new
capability, mobile, cloud and optimisation** work into one brief instead of
five separate ones, because several items touch more than one axis (e.g. the
Command Deck redesign is both a UI and a performance item).

Read alongside, before starting: `ARCHITECTURE.md`, `docs/ORION_CLOUD_ROADMAP.md`,
`docs/ORION_PHASE_UPGRADE_AUDIT.md`, `docs/ORION_REMOTE_ACCESS.md`,
`docs/ULTRON_ANALYSIS.md`, `docs/DEPENDENCY_AUDIT.md`, `android/README.md`.
Do not re-derive what those already establish — extend it.

**Non-negotiable invariants (kept here so this document is self-contained):** strict downward dependency
(`gui`/`app` → services → `bus`/`security`/`constants`, never upward); all
cross-subsystem communication over `OrionBus` signals, never direct widget
access; every blocking call wrapped in `asyncio.to_thread()`; every OS-bound
payload routed through `SecuritySanitiser`; `VOICE_PROFILE` stays frozen;
optional dependencies always degrade gracefully with an actionable log line;
no autonomous self-rewrite without explicit approval. Any change that would
violate one of these needs to be flagged and discussed, not silently done.

---

## 0. How to work this brief

1. Treat each numbered section below as an independent phase — ship, test and
   commit one before starting the next, exactly as the Mark X.x history does.
2. For every phase: state the root cause / gap first (from the audits above
   where one exists), then the smallest correct implementation, then how it
   was verified (unit test, byte-compile, or a described manual check — GUI
   and audio paths can't run headless in CI, say so honestly rather than
   claiming a check that didn't happen).
3. Update `ARCHITECTURE.md` with a new dated section per phase, the same way
   Mark X.7 → X.11 are documented, and update the tool count.
4. Keep British English identifiers and comments throughout, matching the
   existing codebase.
5. Where a phase duplicates or overlaps an existing subsystem (the audit
   already flags `sentinel.py` / `security_sentinel.py` / `system_guard.py`
   and `web.py` / `web_automation.py` as soft-duplicates) — resolve the
   overlap as part of the work, don't add a third one.

---

## 1. UI / UX

### 1.1 Finish the Phase C UI backlog (already specced, not built)
- **Thought-stream typewriter UI + confidence chips** in the Command Centre —
  `thought_stream.py` backend exists; the panel is still a static dump. Add
  incremental reveal (char/word cadence tied to `QTimer`, not per-frame
  string-slicing that reflows layout) and a confidence chip (colour-coded
  GOOD/WARN/BAD from the existing status palette) per reasoning step.
- **Mobile QR pairing render** on the desktop, replacing the manual code
  entry — `qrcode` (pure Python, no native deps) rendered to a `QPixmap` in
  the pairing dialog; the PWA/Android app already parses a pairing code, so
  this only needs a `data:` URI or short pairing URL encoded in the QR.
- **Webcam continuous monitor UI** — once `vision.capture_live_frame` grows
  a background loop (see §2.2), the Command Centre needs an opt-in toggle
  and a live thumbnail with a "scene changed" event ticker, off by default.

### 1.2 Command Deck: from tab bar to real workspace
The deck (`gui/unified_dashboard.py`) is still fundamentally a tabbed
document window (Mark X.10 unified the tab language, which was the right
call, but panels still can't be rearranged). Next step:
- **Dockable/resizable panels** (Qt `QDockWidget` or a lightweight custom
  dock manager) so OUTLOOK COMMAND, NOTION WORKSPACE, DIAGNOSTICS etc. can be
  arranged per the user's workflow, and the **layout persists** to
  `config/` the same way workspace snapshots do. The ULTRON analysis already
  flags "command centre, not chat app" with a 14-panel dockable MISSION deck
  as the direction — make the general Command Deck follow the same pattern,
  not just the Mission page.
- **Command palette** (`Ctrl+K`) — fuzzy-searchable list of every tool/action
  ORION exposes (all ~90 dispatcher tools), so power users don't need to
  phrase a voice/text request for something with a direct action (e.g.
  "restore marketing workspace", "run breach check"). This also doubles as
  living documentation of what ORION can do.
- **Notification centre** — CONFIRM-tier remote actions already produce an
  approval card on the phone (per `ORION_REMOTE_ACCESS.md`); the desktop has
  no equivalent single place to see pending approvals, proactive findings,
  and self-repair proposals together. One bell icon, one list, one place.

### 1.3 Accessibility pass (currently absent from every audit)
- Keyboard-only operability audit of both windows — tab order, focus rings
  (the cyan focus accent already exists in the stylesheet, confirm it's
  applied to every interactive widget, not just buttons).
- Screen-reader labels (`QWidget.setAccessibleName/-Description`) on the
  HUD/orb, status chips and dashboard panels — currently these are purely
  visual and convey state (VOICE ●, STATE) with no text alternative.
- A "reduce motion" toggle that dampens the orb/face idle animation and HUD
  scanline to near-static, for users sensitive to constant motion, and for
  low-power/lite mobile contexts (ties into §4.2).
- Font-scale / high-contrast theme variant, reusing the existing palette
  tokens in `constants.py`/`gui/style.py` rather than a parallel stylesheet.

### 1.4 Onboarding
There is currently no first-run experience — a new machine needs
`config/api_keys.json` hand-edited and dependencies installed from a printed
error message. A guided first-run wizard (desktop): detect missing
required/optional deps against `docs/DEPENDENCY_AUDIT.md`'s table, offer to
`pip install` the required set, walk through provider API keys, and end by
generating the first pairing QR for the phone. This is the single biggest
lever on "would this be usable by someone other than its author".

---

## 2. New capabilities

### 2.1 Close the Phase C backlog first (specced, not built — `ORION_PHASE_UPGRADE_AUDIT.md` §6)
- Browser reading-mode extraction + smooth/incremental scroll — resolve the
  `web.py`/`web_automation.py` duplication as part of this, pick one owner.
- Area-intelligence report assembler over `knowledge_graph.py` + `geo.py` +
  `research.py` — the pieces exist, the composed report doesn't.
- Security-posture hardening (Phase D): per-capability tests for the Phase C
  work, a startup-time budget check, the sentinel-surface merge.

### 2.2 Continuous perception (bigger than the "webcam monitor" line item)
- Background frame-diff loop off `vision.capture_live_frame`, bus event on
  material scene change, throttled and opt-in (already specced) — but extend
  it to **presence-aware behaviour**: if `presence.py` already tracks
  user-present/away, wire continuous monitor to arm only while away (a
  "someone's at my desk" security use, distinct from `security_sentinel.py`'s
  network/breach posture) and disarm the moment presence returns, logged, never
  silent.
- Scene-change events should feed `proactive.py`'s survey rather than becoming
  a fourth independent alerting path — one proactive surface, not several.

### 2.3 Correction & feedback loop for LearningService
`learning.py` already has `correct()`/`forget()`; there's no UI or voice
affordance for "no, that's wrong" during an actual conversation — today the
user has to know the tool exists and invoke it explicitly. Add: after any
factual answer sourced from KNOWLEDGE/long-term memory, a lightweight
thumbs-down voice/text pattern ("that's wrong", "no, actually...") that
routes straight to `correct()` with the prior answer as context, instead of
just being logged as a normal turn.

### 2.4 Calendar-aware scheduling, not just calendar-reading
Notion/Outlook integration currently reads and creates events on request.
The natural next step, consistent with `momentum.py`'s "shipping coach"
framing, is **conflict-aware scheduling**: "find me 90 minutes this week for
X" should cross-reference Notion calendar + Outlook calendar + tracked
project deadlines and propose slots, not just create a blind event. Keep it
CONFIRM-tier (never auto-books).

### 2.5 Multi-user / household mode (design-only for now, don't build blind)
ORION is single-user by design (frozen identity, one persona). If the user
wants ORION addressable by more than one person (family, in ExampleStore
context), that's a genuine architectural fork — voice-print or simple
"who's speaking" disambiguation, per-user memory partitioning, per-user
Notion/Outlook credential scoping. **Flag this as a decision point, don't
implement speculatively** — it changes the identity model at the root.

### 2.6 Plugin/skill marketplace parity with Knowledge Packs
`knowledge_packs.py` already gives a clean "installable JSON expertise"
pattern. Extend the same pattern to **tool plugins**: a manifest format
(name, dispatcher hook, capability tier, required deps) that lets a new
dispatcher capability be dropped into `config/custom_tools/` (already exists
as a directory per the file listing — check whether it's wired to anything
yet or just scaffolded) and registered without editing `dispatcher.py`
directly. This is what makes "teach ORION a new skill" mean *code*, not just
*facts*, without every addition being a manual dispatcher edit.

---

## 3. Phone / mobile compatibility

### 3.1 Android native gaps (from `android/README.md`'s own "next steps")
- **Foreground service + push notifications** for proactive messages while
  the app is closed — currently ORION can only reach the phone while the app
  is open and connected to the SSE stream. This is the single biggest gap
  between "companion app" and "assistant that can actually interrupt you".
  Use Android's `NotificationManager` + a foreground service holding the
  uplink connection, or (better, lower battery cost) a lightweight polling
  `WorkManager` job hitting `/v1/health` or a new `/v1/notifications/pending`
  endpoint.
- **Native `SpeechRecognizer`** so voice input works over plain HTTP (today
  it needs `ORION_REMOTE_TLS=1` because it's the *browser's* recogniser
  inside the WebView). This also removes the record-clip-then-transcribe
  round trip currently used inside the app per `ORION_REMOTE_ACCESS.md` §1.
- **Home-screen widget** — quick voice/text entry + last-briefing glance,
  without opening the full app. High value for a JARVIS-style assistant;
  Android widgets are a well-trodden API.
- **Wear OS / quick-tile companion** (stretch) — "ask ORION" from a watch or
  a quick-settings tile, text-only round trip, reuses the existing
  `/v1/converse` endpoint.

### 3.2 iOS parity
Right now the mobile story is Android-native + generic PWA. iOS Safari PWAs
have real limits (no true background push without a native wrapper, stricter
mic permission model). Decide explicitly: (a) PWA-only for iOS and document
the limitations, or (b) a thin SwiftUI/WebView wrapper mirroring the Android
app's native bridge (`window.OrionNative` equivalent via `WKScriptMessageHandler`).
Don't let this default by omission — the user should choose, then it gets
built to the same bar as Android.

### 3.3 Lite-mode completeness
`ORION_REMOTE_ACCESS.md` documents `?lite=1` dropping the Three.js face for
a CSS orb on metered connections. Audit whether **every** heavy asset (not
just the face) respects lite mode — Command Centre charts, the globe's
Google Maps embed, any future dockable-panel graphics from §1.2 — and make
lite-mode detection a single shared client-side flag rather than a per-page
check, so new features inherit it for free instead of needing to remember it.

### 3.4 Reliability of the remote uplink itself
- Reconnection/backoff on the SSE stream (phone loses signal in a lift,
  tunnel, etc.) — confirm the app resumes cleanly rather than needing a
  manual reconnect, and that pending CONFIRM approvals aren't lost mid-drop.
- Offline queue on the phone side: if a message is sent while the PC is
  asleep/offline, queue it client-side and flag "will deliver when ORION is
  reachable" rather than silently failing.

---

## 4. Cloud / server

### 4.1 Advance the Cloud Roadmap to C2 (`docs/ORION_CLOUD_ROADMAP.md`)
C1 (pairing auth, capability manifest) is done. Next:
- **Sync journal + replicator (C2)** — the roadmap already specs the schema
  (`node_id, lamport_ts, tier, category, key, value_hash, payload`,
  append-only, last-writer-wins except `long_term`/`knowledge` which merge
  additively). Implement the journal hook in `OrionMemoryMatrix` and
  `KnowledgeGraphEngine` first (one seam each, as noted), then the
  push/pull replicator. This is the prerequisite for the desktop and a
  cloud node to actually agree on what ORION knows.
- **PWA voice loop + push (C3)** — cloud-side STT (faster-whisper, already a
  desktop dependency, reuse the model) and TTS (Piper matched to
  `VOICE_PROFILE`, or Edge-TTS as the documented fallback) so the *cloud*
  node — not just the desktop uplink — can carry a full voice conversation
  when the desktop is off. Distinguish clearly from the existing desktop
  `remote.py` uplink in docs so it's obvious which node is answering.
- **Remote agent execution with audit log (C4)** — `RemoteAgentQueue` is
  seeded with a whitelist; extend to a durable audit log (who/when/what
  tool/what result) surfaced in the Command Centre, not just a log line.

### 4.2 Deployment hardening
- `deploy/` has `Dockerfile`, `Caddyfile`, `orion-node.service`, an Oracle
  Cloud walkthrough. Missing: a **provisioning script** (the roadmap's
  `provision.sh` — Python, Ollama, certs on a fresh VM) so C1 deployment is
  one command, not a manual walkthrough every time.
- **Backup/restore for the cloud node's SQLite stores** — `config/*.db`
  files (memory, knowledge graph, evidence graph, ingestion) currently have
  no documented backup strategy on the cloud node. A cron'd `sqlite3 .backup`
  to object storage (or the VM's block storage snapshot) is cheap insurance
  before C2 sync makes cloud data genuinely load-bearing.
- **Health/alerting** — `/api/health` exists; nothing currently *watches* it.
  A minimal external check (systemd `Restart=on-failure` is already
  presumably in `orion-node.service` — confirm — plus an optional
  uptime-ping to a free service like healthchecks.io) so a crashed cloud
  node is noticed before "why isn't ORION answering my phone" is the first
  signal.

### 4.3 Cost and resource ceiling
Oracle's Always-Free tier has hard limits (CPU/RAM/egress). Before C2/C3 add
real load (sync traffic, STT/TTS compute), profile the headless node's
baseline footprint (`server.py` idle RSS, CPU) and set a documented ceiling
in `docs/ORION_CLOUD_ROADMAP.md` so a future feature doesn't silently blow
the free tier — Ollama on a Free-tier VM in particular is a likely offender
if a heavier model is ever pulled cloud-side.

---

## 5. Optimisation & smoothness

### 5.1 Startup-time budget (flagged in the Phase C/D roadmap, never closed)
ORION now seeds neuroscience, programming and cybersecurity corpora, ten
knowledge packs, cognitive state, identity, and (per `ARCHITECTURE.md`)
dual-screen window placement — all at launch. Add an actual **startup-time
budget test**: log a timestamp at each major composition-root step in
`app.py`/`server.py`, assert (or at least report) the total against a target
(e.g. < 5 s desktop, < 2 s headless), and treat any step over ~200 ms as a
candidate for lazy/deferred init.

### 5.2 Database housekeeping
`config/ingestion.db` is already 5 MB with a 4 MB WAL file; several DBs carry
substantial `-wal`/`-shm` siblings. None of the docs mention a `VACUUM`/WAL
checkpoint schedule. Add a periodic (weekly, low-priority background task
alongside the existing cognitive-loop/reporting tasks) `PRAGMA wal_checkpoint`
+ `VACUUM` pass across the SQLite stores, and confirm FTS5 indexes aren't
silently bloating past what's actually queried.

### 5.3 Thread-pool and concurrency audit
`asyncio.to_thread()` uses the default executor (unbounded-ish, sized by
`min(32, os.cpu_count()+4)`). With OCR, COM calls, subprocess launches,
screen grabs, web automation and file ingestion all potentially in flight,
confirm there's no head-of-line blocking between latency-sensitive calls
(audio persistence, bus-critical work) and bulk ones (`learn_folder` walking
a large directory). Consider a **dedicated bounded executor** for bulk
ingestion so it can never starve the default pool that audio/voice depend on.

### 5.4 Render-path profiling
The HUD/orb/face already throttle to 4 fps when hidden and cache backgrounds
in `QPixmap` — good discipline. Extend the same rigour to the newer surfaces
(Command Centre live telemetry redraw, globe embed, any dockable panels from
§1.2): confirm each animated widget has a measured frame budget and an
idle/hidden throttle, not just the two avatars. A simple frame-time overlay
(toggle-only, dev build) would make regressions visible instead of
subjective ("does it feel smooth").

### 5.5 Test and CI coverage
`docs/ORION_PHASE_UPGRADE_AUDIT.md` notes ~556 passing tests across ~35
modules, and one known-broken root-level test needing `pytest-asyncio` fixed
or relocated under `tests/`. Before adding more surface area (§2, §3, §4):
fix that test, add a CI job (GitHub Actions, `.github/` already exists for
the Android APK workflow) that runs `pytest` + `py_compile` on every push,
so "verified: package byte-compiles" stops being a manual claim in each
Mark X.x section and becomes an actual gate.

### 5.6 Dependency trim pass
`docs/DEPENDENCY_AUDIT.md` is thorough but from July 2026 — re-run it after
this brief's work lands, and specifically decide on the optional PyTorch-scale
upgrades (EasyOCR/PaddleOCR/silero-vad/Coqui) rather than leaving them
permanently "optional, not installed" — either commit to one OCR/VAD upgrade
path with a clear justification, or explicitly close the question so it
doesn't get re-asked every audit cycle.

---

## 6. Suggested sequencing

Not a hard order, but a sane one given dependencies between sections:

1. §5.5 (fix CI/tests) — do this first, everything else needs a safety net.
2. §2.1 (close existing Phase C backlog) — smallest, already designed, clears
   the board before new design work.
3. §1.1 → §1.3 (finish specced UI, then accessibility) — user-facing, keeps
   momentum visible.
4. §3.1 (Android push/foreground service) — highest-leverage mobile gap.
5. §4.1 C2 (sync journal) — prerequisite for everything else in cloud.
6. §5.1–5.4 (perf/smoothness pass) — do once the above have landed, so it's
   profiling the *current* surface, not a moving target.
7. §1.2, §1.4, §2.3–2.6, §3.2–3.4, §4.2–4.3, §5.6 — everything else, roughly
   in value order, revisit priority once the above ships.

§2.5 (multi-user mode) stays a flagged decision point, not a scheduled phase,
until the user confirms they want it.
