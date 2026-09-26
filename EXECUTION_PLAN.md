# ORION execution plan — 25 September 2026

This plan records what an audit of the current source found, what has been
changed, and what remains. It was written against the code, not against the
brief that started it: where the brief assumed something the code does not
contain, that is stated and the underlying intent is applied to what is
actually there.

## What the code actually is

| Brief assumed | Code has | Consequence |
| --- | --- | --- |
| A FastAPI backend with Pydantic models | aiohttp: the phone gateway (`orion_core/remote.py`) and the Twilio bridge (`orion_core/telephony_server.py`); the desktop is PyQt6 on qasync | No FastAPI migration. Request validation, size limits and loop hygiene were applied to the aiohttp handlers. Pydantic only arrives through `google-genai`, which the headless install treats as optional, so validation uses a small typed schema instead of a new hard dependency. |
| YAMNet, YuNet and voiceprint pipelines to be built | All three exist: `sound_sense.py`, `face_tracking.py`, `voiceprint.py` | Extended rather than rebuilt: a background sound watch, and an owner-voice guard for sensitive actions. |
| MCP servers served by ORION over stdio | ORION is an MCP *client* (`mcp_host.py`) that spawns servers | The stdout/stderr concern is inverted: the host must survive a server's noise. It already drains stderr and skips non-JSON lines; this is now pinned by a real-subprocess test. Error typing was the real gap. |
| An Android client that polls | A WebView shell around the desktop's uplink page, which already streams over authenticated SSE with a 20 s keepalive | No native rewrite (no Android SDK here to build or verify one). One security finding recorded below. |
| A Three.js canvas in the Command Deck | The Deck's Brain page is already native Qt (the WebGL renderer was removed in the previous audit); Three.js still draws the face (window 1), the globe and the phone's face page | The Deck gained telemetry. `assets/three/` stays: deleting it would break the face and globe, which are in use. |

## Defects found and fixed

Order is the order of implementation. Every item has regression tests.

| # | Defect | Files | Commit subject |
| --- | --- | --- | --- |
| 1 | Every JSON endpoint on the phone gateway, including the unauthenticated pairing and token doors, parsed bodies up to 8 MB (the ceiling was raised for audio uploads). | `orion_core/remote.py` | fix(remote): cap JSON request bodies at 64 KB |
| 2 | `audioop` left the standard library in Python 3.13 (the target interpreter) and no requirements file listed `audioop-lts`; the telephony bridge could not convert call audio. | `requirements.txt`, `deploy/requirements-server.txt`, `orion_core/telephony_bridge.py` | fix(deps): require audioop-lts on Python 3.13+ |
| 3 | Transient provider faults (5xx, timeouts, refused connections) took a flat 45 s cooldown forever, so a dead endpoint cost its full timeout every 45 s; no half-open probing; out-of-memory strikes never reset. | `orion_core/providers.py` | feat(router): add a circuit breaker for transient provider faults |
| 4 | A timing-dependent test (`warm()` idempotency) failed under parallel load. | `tests/test_startup_performance.py` | test(startup): make the warm-up idempotency check deterministic |
| 5 | The rewind tool re-read and parsed every conversation transcript on the event loop (the Qt GUI thread under qasync) on each call; `learn_folder` walked a whole tree on the loop before applying its cap. | `orion_core/dispatch_knowledge.py`, `orion_core/rewind.py`, `orion_core/learning.py` | fix(async): move transcript loading and folder walks off the event loop |
| 6 | MCP replies were plain strings and failure was guessed from wording; JSON-RPC errors surfaced as raw dict reprs. | `orion_core/mcp_host.py`, `orion_core/dispatch_web.py` | feat(mcp): type protocol rejections and carry each reply's outcome |
| 7 | The audio capture loop re-read and re-parsed both voiceprint JSON files on every chunk of speech; a spoken go-ahead for a sensitive action was accepted from anyone in the room. | `orion_core/voiceprint.py`, `orion_core/speaker_gate.py` (new), `orion_core/audio.py`, `orion_core/speech_offline.py`, `orion_core/live_presence.py`, `orion_core/live_worker.py`, `orion_core/dispatcher.py`, `orion_core/dispatch_desktop.py`, `orion_core/dispatch_schema.py` | feat(voice): require the owner's voice for spoken go-aheads on sensitive actions |
| 8 | Sound classification ran only on request; nothing noticed a smoke alarm, siren or knock unprompted. | `orion_core/sound_watch.py` (new), `orion_core/dispatch_vision.py`, `orion_core/dispatch_schema.py` | feat(ml): add a background YAMNet sound watch for alarms and the door |
| 9 | The Command Deck did not show which text model answered, its latency, GPU load (already in each telemetry sample) or sound alerts. | `orion_core/providers.py`, `orion_core/gui/command_overview.py` | feat(deck): show the text model, its latency, GPU load and sound alerts |
| 10 | One live-research question failing (a search backend fault) failed the whole run while its sibling kept working as an orphan. | `orion_core/live_research.py` | fix(research): contain a failing question instead of sinking the run |
| 11 | Gateway fields were `str()`-coerced, so an object sent as a chat message or a boolean sent as a pairing code was processed. | `orion_core/remote.py` | feat(remote): validate gateway request fields against per-endpoint schemas |
| 12 | With no OCR backend installed the capability matrix said DEGRADED (installed but failing) instead of MISSING; the spoken browser name broke on a Windows path read on another OS; several Windows-only tests failed rather than skipped elsewhere. | `orion_core/capability_health.py`, `orion_core/browser.py`, `orion_core/remote.py`, four test files | fix: report a missing OCR backend as MISSING and make tests portable |
| 13 | The headless node discarded its whole log, including the first-run pairing code, so a fresh cloud node could not be paired; the Oracle guide still described the retired static token. Found by booting the node. | `orion_core/server.py`, `orion_core/remote.py`, `deploy/README_ORACLE_CLOUD.md` | fix(server): print the headless node's log so a fresh node can be paired |
| 14 | Every single-instance test claimed the machine-global lock, so parallel test workers (and a running ORION) broke them. | `tests/test_single_instance.py` | test(single-instance): give each test a private lock name |
| 15 | The Android app accepted any certificate a known host name presented, so anyone on the same network could impersonate ORION and collect the pairing token. | `android/.../MainActivity.kt`, `Prefs.kt`, `SetupActivity.kt`, `strings.xml` | fix(android): pin ORION's certificate instead of trusting any for his hosts |
| 16 | The Android APK workflow could not build: it requested the retired SDK `tools` package, then regenerated the Gradle wrapper at 8.7 over the committed 9.5.0 pin that AGP 9.3 requires. It now builds, including #15. | `.github/workflows/android.yml` | ci(android): two commits |
| 17 | On every first run the API-key dialog's `exec()` nested a Qt event loop inside the boot coroutine; qasync then killed the event-loop stall detector and the metrics sampler at boot. Found by booting the desktop app offscreen with asyncio debug on. | `orion_core/app.py` | fix(boot): await the first-run API-key dialog instead of nesting a Qt loop |
| 18 | The briefing header read the machine's clock while the greeting read ORION's, so on a machine in another zone they named different parts of the day and the wrong local time. | `orion_core/briefing.py` | fix(briefing): read ORION's clock, not the machine's, for the header |
| 19 | The same mismatch elsewhere: "remind me at 3pm" fired an hour out on a machine in another zone; briefing-engine headers mixed both clocks in one line; the late-night farewell checked the machine's hour. | `orion_core/reminders.py`, `orion_core/briefing_engine.py`, `orion_core/live_worker.py` | fix(time): tell the time from ORION's clock in reminders, briefings and farewells |

### Second round: the desktop's Mark XXXII snapshot

`main` was republished from the desktop as a single-commit snapshot holding
everything above plus new work: MCP credential redaction and Gmail/Calendar
confirmation rules, the parked-server reconnect fix, sanitised MCP route
removal, refusing `confirm=true` without a human guard, QR pairing, phone
reconnection and approval catch-up, per-device phone actions and chess
evaluation tables. The branch merges it (tree identical to `main`); these were
then found and fixed on top:

| # | Defect | Files | Commit subject |
| --- | --- | --- | --- |
| 20 | Credential redaction turned MCP replies back into plain strings, losing their outcome for every server with a credential; short settings under secret-looking keys (`AUTH_ENABLED=on`) were redacted wherever the letters appeared. | `orion_core/mcp_host.py` | fix(mcp): keep each reply's outcome through credential redaction |
| 21 | Scanning the pairing QR code also popped the manual pairing-code prompt: start-up refreshed access while the scanned code was still being redeemed. Found by driving the page in headless Chromium. | `orion_core/remote.py`, `tests/test_phone_page_browser.py` | fix(remote): redeem a QR pairing code before the start-up refresh |
| 22 | Any phone reopening its event stream (every 15 minutes, and after any drop) became "last active", so an idle phone took ORION's own phone actions from the one in use. | `orion_core/remote.py` | fix(remote): count a phone as last active on use, not on reconnect |
| 23 | The proactivity engine's task was never cancelled on shutdown; housekeeping and diagnostics were cancelled but never awaited. | `orion_core/app.py` | fix(shutdown): cancel and await every background task ORION starts |
| 24 | One HTTP 503 benched the only provider for about 40 s and the phone was told no model was reachable. A first transient fault is now retried once after a short pause (not timeouts), the cooldown starts at 5 s, and a success clears it. Found by running a real headless node against a stand-in model. | `orion_core/providers.py`, `tests/test_headless_node_e2e.py` | fix(router): ride out a single overload blip instead of failing the turn |

Housekeeping: a stray editor configuration file and its tool-specific ignore
entries were removed; the publication checker's generic dot-directory rule
covers them.

## Using the new behaviour

- **Owner voice for sensitive actions:** say "only accept my voice for
  actions" (tool `voice_speaker_id`, action `guard_actions`; `unguard_actions`
  to turn it off). Needs an enrolled voice. ORION keeps listening to everyone;
  only a spoken confirm/consent/submit from a confidently different voice is
  refused. Typed requests are never checked.
- **Sound watch:** "keep an ear out for alarms" (tool `sound_sense`, action
  `watch`; `watch_off`, `watch_status`). Local only; nothing is recorded.
- **Provider health:** `ProviderRouter.diagnostics()` and the snapshot now
  include each provider's breaker state (`closed`, `open`, `half_open`).

## Findings not changed here

1. **First contact on Android.** Pinning (#15) trusts the first certificate
   the phone sees, so an attacker present at that very first connection is
   still accepted. Closing that needs the fingerprint delivered out of band,
   for example shown beside the pairing code on the desktop.
2. **Phone approvals and tools with their own confirmation.** An action
   approved on the phone runs with the model's original arguments. A tool with
   its own `confirm=true` step can then ask again, and an MCP tool that spends
   money raises an on-screen prompt on the desktop, which the phone user
   cannot see. Whether a phone tap should count as that confirmation is a
   policy decision; it predates the snapshot and was left unchanged.
3. **Refresh token at rest (Android).** Stored in plain SharedPreferences.
   Backups and device transfer are already excluded; consider wrapping it with
   an Android Keystore key.
4. **Boot-time loop stalls.** Under asyncio debug in this container, several
   boot steps and background tasks exceeded 100 ms. The largest were trivial
   steps (the stall detector's own heartbeat) waiting for the GIL while the
   import-warming thread runs, so they are not evidence of blocking code; judge
   them on real hardware with ORION's own stall detector.
5. **Timing tests.** `test_gui_hot_paths::test_cursor_pos_is_cheap_enough_for_33_hz`
   and `test_holo_head::test_paint_stays_within_the_frame_budget` measure wall
   time and fail on a loaded machine.

## Verification

Linux, Python 3.13, Qt offscreen, before and after this work:

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Before | 6,787 | 22 | 33 |
| After | 6,894 | 2 | 39 |
| After the snapshot merge and round two | 6,923 | 2 | 39 |

The 2 remaining failures are the wall-clock timing tests listed above. Eleven
telephony failures were fixed by the `audioop-lts` requirement; Windows-only
tests now skip elsewhere (hence 6 more skips); the work added 107 tests. The
Android APK builds in GitHub Actions, including the pinning change. The publication check, byte-compilation and the
capability gate pass. No real microphone, camera, phone, provider account or
Twilio call was exercised.

Run from the repository root with the Qt offscreen platform:

```
python -B tools/prepare_public_source.py
python -m compileall -q orion_core orion.py
python -B tools/check_capabilities.py
python -B -m pytest -q
```
