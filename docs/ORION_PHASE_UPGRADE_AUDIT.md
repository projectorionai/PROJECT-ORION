# ORION Phase Upgrade — Architecture Audit, Gap Report & Roadmap

*Prepared before any code was changed, per the phase-upgrade brief (analyse →
map → report → identify → plan → then code). Every claim below is grounded in a
module that exists in this tree as of this pass.*

---

## 1. Architecture audit (what actually exists)

ORION is **not** an early-stage project — it is a ~38,000-line, ~110-module
autonomous AI operating system (`orion_core/`) with a Qt desktop shell, a
headless cloud brain, a mobile uplink, and ~63 test modules (827 collected,
826 passing + 1 skipped, all inside CI as of Mark X.12 §5.5).
The `ARCHITECTURE.md` "Mark" history (VII → X.11) documents the evolution.

### Layering (enforced, one direction, downward)

```
orion.py                     launcher → desktop app  OR  --headless → server.py
orion_core/
  constants / utils / data   dependency-free primitives (paths, ToolResult, helpers)
  security.py                SecuritySanitiser — AST/regex firewall on every OS payload
  bus.py                     OrionBus (QObject signal hub — every subsystem decouples here)
  memory.py                  OrionMemoryMatrix (SQLite FTS5) + MemoryAgent (7 tiers)
  providers.py               ProviderRouter — cloud/local tiered routing, MODE A/B
  audio.py / audio_state.py  capture→VAD→STT→playback→TTS state machine
  vision.py / ocr_engine.py  screen grab, file intelligence, single webcam frame
  control.py / verification.py  autonomous desktop control + act→verify→retry
  web.py / web_automation.py    hybrid browser control
  agents.py                  AgentManager + specialist personas
  dispatcher.py + dispatch_*.py  thin router + per-domain tool mixins (~90 tools)
  live_worker.py             Gemini Live session + offline loop
  remote.py                  RemoteGateway — paired, token-authed mobile/browser uplink
  server.py                  headless composition root (no GUI/audio/COM)
  app.py                     desktop composition root (all wiring, only here)
  gui/                       core_window, unified_dashboard, command_centre, face3d, globe…
```

**Design invariants (verified in code, must be preserved):**
- All cross-subsystem messaging rides `OrionBus` Qt signals; **no service imports a widget**.
- Every blocking call (PIL/OCR/COM/subprocess/screen-grab/SQLite-write) goes through `asyncio.to_thread` so the qasync GUI loop never stalls.
- Every OS-bound payload passes `SecuritySanitiser`; autonomy honours the `ORION_AUTONOMY` kill-switch + pyautogui FAILSAFE.
- `VOICE_PROFILE` is a frozen dataclass; nothing at runtime mutates the voice.
- Remote origins are constrained by `REMOTE_TOOL_WHITELIST` / `REMOTE_HARD_DENY` (host-mutating tools hard-denied even when named).

## 2. Dependency map (subsystem level)

```
constants ─┬─> (everything)
utils ─────┤
data ──────┘
security ──> dispatch_* , control , web , forge          (payload firewall)
bus ───────> (every subsystem; the only cross-talk channel)
memory ────> providers , agents , conversation_memory , learning , knowledge*
providers ─> live_worker , agents , commerce , reports , forge(brain) , remote , server
audio ─────> live_worker , audio_state
vision ────> verification , dispatch_vision            (ocr_engine plugs in)
control ───> verification , dispatch_desktop , plan_executor
telemetry ─> ~all services (health/metrics/log facade)
dispatcher ─> ALL capability services (composition target)
app.py ────> constructs & wires EVERYTHING (desktop)     ← single wiring site
server.py ─> constructs the portable subset (headless)
```

Direction is strictly downward; `gui/` and `app.py` know services, services know
`bus`/`security`/`constants`, nothing imports upward. New capabilities attach as
`dispatcher.<name> = <service>` in `app.py` and expose a schema entry — no call
site changes (the mixin pattern from the dispatcher split).

## 3. Capability matrix (status by requested area)

| Area (from brief)          | Module(s)                                   | Status | Real gap |
|----------------------------|---------------------------------------------|--------|----------|
| Self-repair                | `selfrepair.py`                             | ✅ Working | compile-validate + isolated tests + approval gate all present; auto-apply is deliberately narrow |
| Autonomous PC control      | `control.py`, `agents.py` (DesktopAgent)    | ✅ Working | kill-switch, clamp, verify present |
| Human-like typing          | `control.py` `type_text`/`edit_text`        | ⚠️ Partial | **uniform delay only — no WPM/jitter/typo sim** *(fixed this pass)* |
| Browser scrolling          | `verification.py` (`click_text` scroll), `web.py` | ⚠️ Partial | scroll-to-find exists; smooth/incremental/reading-mode thin |
| Real-time thinking stream  | `thought_stream.py` + `gui/command_centre`  | ⚠️ Partial | backend + panel exist; typewriter/confidence UI thin |
| Webcam intelligence        | `vision.py` `capture_live_frame`, `presence.py` | ⚠️ Partial | **single frame only — no continuous monitor / scene loop** |
| Security sentinel          | `security_sentinel.py`, `sentinel.py`, `system_guard.py` | ⚠️ Partial | ports/procs/drives watched; **no breach monitoring, thin network report** *(fixed this pass)* |
| Knowledge graph / OSINT    | `knowledge_graph.py`, `geo.py`, `research.py` | ✅/⚠️ | graph + entity/timeline exist; area-intel report assembles sources |
| Command deck / ingestion   | `gui/unified_dashboard.py`, `learning.py` `learn_folder` | ✅/⚠️ | drag-drop ingest via learn tool; project-workspace UI thin |
| Mobile access / pairing    | `remote.py` (pairing + tokens), PWA + SW     | ✅ Working | QR-code display is the main missing nicety |
| Agent system               | `agents.py`, `plan_executor.py`, `cognition.py` | ✅ Working | shared bus, routing, delegation present |
| Performance/telemetry      | `telemetry.py`, `metrics_history.py`, `resource_monitor.py` | ✅ Working | rolling SQLite trends + per-tool audit |

**Legend:** ✅ working · ⚠️ partial (works but below the brief's bar) · ❌ missing.

## 4. Unfinished / duplicated / risk inventory

- **Overlapping "sentinel" surfaces**: `sentinel.py` (SentinelAgent — perf/battery),
  `security_sentinel.py` (SecuritySentinel — posture), `system_guard.py`. Not
  duplicates but the boundary is subtle; the security posture surface is the
  thinnest and the natural home for breach + network-dashboard work.
- **Two web layers**: `web.py` (WebController) and `web_automation.py` — confirm
  a single owner for scroll/reading-mode before extending.
- **Root-level `test_jarvis_subsystems.py`** — *resolved (2026-07-22).* Rewritten as
  ordinary sync pytest tests (async paths wrapped in `asyncio.run`) with a root
  `conftest.py` hook that also drives bare `async def test_*`, so no
  `pytest-asyncio` dependency is needed. It still lives at repo root (it doubles
  as a standalone `python test_jarvis_subsystems.py` script), so CI's test step
  was changed from `pytest tests` to bare `pytest` to collect it — the file is
  now inside the gate. See `ARCHITECTURE.md` → Mark X.12 §5.5.
- **Uncommitted working tree**: HEAD is a single "sterilised" commit; all prior
  passes live only in the working tree. `git stash` is hazardous here (OneDrive
  file locks + sterilised HEAD can destroy uncommitted work) — never stash.

## 5. Security assessment

- **Payload firewall** (`SecuritySanitiser`) gates every OS-bound string — strong.
- **Remote exposure** (`remote.py`) is hardened: pairing → revocable refresh
  token (SHA-256 at rest) → 15-min HMAC access token, constant-time compares,
  per-client rate limits, CSP/nosniff/frame headers, capability hard-deny.
  TLS is expected from a reverse proxy — do **not** expose localhost raw.
- **New risk surface introduced this pass** (breach monitoring) is designed to
  the same bar: the Pwned-Passwords check uses **k-anonymity** — only the first
  5 hex chars of a local SHA-1 hash ever leave the machine; the password and the
  full hash are never transmitted, never logged, never persisted. Opt-in, and
  the *passive* sentinel loop stays fully offline (network calls are on-demand
  only), preserving its "never touches the network" invariant.

## 6. Roadmap (phased, value-ordered)

**Phase A — Security posture (this pass, shipped + tested):**
1. HIBP Pwned-Passwords k-anonymity + breached-account check (`breach_monitor.py`).
2. Network security dashboard: firewall state, listening services + owning
   process, established-connection summary, startup entries.
3. `breach_check` tool + `security_watch action=network`.

**Phase B — Human interaction (this pass, shipped + tested):**
4. `human_typing.py` — deterministic keystroke planner (WPM, jitter, typo sim).
5. `type_text(human=True, wpm=…, typos=…)` path + desktop-tool args.

**Phase C — Next (specced, not built this pass):**
6. Webcam continuous monitor: background frame loop → scene diff → event on
   change, opt-in, off by default, reuse `vision.capture_live_frame` + bus events.
7. Thought-stream typewriter UI + confidence chips in the command centre.
8. Browser reading-mode extraction + smooth/incremental scroll in the single
   chosen web layer.
9. Mobile QR pairing render on top of the existing pairing-code flow.
10. Area-intelligence report assembler over `knowledge_graph` + `geo` + `research`.

**Phase D — Hardening:** per-capability tests for Phase C, a startup-time budget
check, and a duplication pass to merge the two web layers.

## 7. Verification method

Headless where possible: pure-logic units (keystroke plans, k-anonymity range
parsing, dashboard formatting) are deterministic and unit-tested with mocked
`aiohttp`/`psutil`/`subprocess`; GUI/camera paths degrade gracefully and log an
actionable hint when their optional dependency or hardware is absent.
