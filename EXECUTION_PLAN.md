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

Housekeeping: an agent prompt file and tool-specific ignore entries were
removed; the publication checker's generic dot-directory rule covers them.

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

1. **Android certificate trust (security).** `MainActivity.onReceivedSslError`
   proceeds for any certificate presented by a known host name, so an attacker
   on the same network could intercept the pairing token. Pin the desktop's
   certificate fingerprint: report its SHA-256 in the `/v1/auth/pair` response,
   store it in `Prefs`, and compare `SslError.certificate` against it. Needs an
   Android build to verify.
2. **Refresh token at rest (Android).** Stored in plain SharedPreferences.
   Backups and device transfer are already excluded; consider wrapping it with
   an Android Keystore key.
3. **Windows-only tests on Linux.** In a Linux run, 3 firewall tests (they
   build a `WindowsPath`), 3 autostart tests, 2 config-move tests (they call
   `cmd`), and the OCR/mediapipe availability tests fail for platform or
   optional-package reasons. CI runs on Windows, where they apply.
4. **Timing tests.** `test_gui_hot_paths::test_cursor_pos_is_cheap_enough_for_33_hz`
   and `test_holo_head::test_paint_stays_within_the_frame_budget` measure wall
   time and fail on a loaded machine.

## Verification

Linux, Python 3.13, Qt offscreen, before and after this work:

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Before | 6,787 | 22 | 33 |
| After | 6,882 | 11 | 33 |

The 11 remaining failures are exactly the platform and timing cases listed
above; 11 telephony failures were fixed by the `audioop-lts` requirement, and
the new work added 95 tests. The publication check, byte-compilation and the
capability gate pass. No real microphone, camera, phone, provider account or
Twilio call was exercised.

Run from the repository root with the Qt offscreen platform:

```
python -B tools/prepare_public_source.py
python -m compileall -q orion_core orion.py
python -B tools/check_capabilities.py
python -B -m pytest -q
```
