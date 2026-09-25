# ORION — System Audit, Root-Cause Report & Architecture Notes
**Date:** 2026-08-08 · **Baseline:** 3068 passing / 2 pre-existing failures · **After:** 3261 passing / 2 pre-existing failures

---

## A. Root-cause report

### A1 · "ORION thinks it is a summer daytime at 12:31 AM"

| | |
|---|---|
| **Problem** | At 00:31 ORION greeted with "Good morning" and described "a clear summer day". |
| **Cause** | Seven modules each decided the time of day for themselves, using **three different band tables**. The weather clause in `temporal.py` asked `hour < 12` → "morning", and called *everything else* "day". At 00:31, `0 < 12` is true. Season came straight from the month, so August was always summery regardless of the hour. |
| **Affected** | `temporal.py`, `providers.py`, `live_worker.py`, `local_brain.py`, `briefing.py`, `briefing_engine.py` |
| **Why it happened** | No authoritative clock. Each site was written independently and each was individually "reasonable"; the boundaries diverged (17:00 in briefing/live_worker, 18:00 in local_brain) so subsystems could name different parts of the same minute. |
| **Fix** | New `orion_core/time_service.py` — one clock, one band table, one season. Bands: night 22:00–04:59, morning 05:00–11:59, afternoon 12:00–17:59, evening 18:00–21:59. Crucially it separates `time_of_day()` (the true period — "night") from `greeting_period()` (what you *say* — "evening", since English has no greeting "good night"), and adds `is_daytime()`, the predicate the weather clause never had. Timezone-aware, DST-correct via the OS clock, with `zoneinfo` preferred when tzdata exists (it does **not** on this machine — verified `ZoneInfoNotFoundError`, hence the guard). Nothing is cached, so midnight, a DST change and a clock correction are all picked up on the next call. |
| **Files** | `orion_core/time_service.py` (new), `temporal.py`, `providers.py`, `live_worker.py`, `local_brain.py`, `briefing.py`, `briefing_engine.py`, `tests/test_time_service.py` (new, 50 tests) |
| **Testing** | All eight clock times from §37 asserted. The literal bug pinned: 00:31 in August → period `night`, `is_daytime()` False, season `summer`, greeting `evening`. Every consumer asserted to read the shared table. |

### A2 · Standby: "he comes back alive but does nothing"

Three independent defects, all in the same five lines.

| | |
|---|---|
| **Problem** | After waking from standby ORION could be unable to speak, unable to hear, or apparently alive and unresponsive. |
| **Cause 1 — the mute that was never lifted** | Entering standby sets `_drop_live_output = True` so a mid-flight reply goes quiet. It is cleared **only** by `_mark_turn()` (text turns) and `turn_complete`. A *spoken* turn on the live channel passes through neither on the way in — so every audio chunk of the **first reply after waking was discarded**. ORION heard you, answered, and you heard silence. |
| **Cause 2 — the microphone that was never re-enabled** | The wake path guarded on `prior_mic is not None`, where `prior_mic = getattr(self, "_microphone_before_standby", None)`. That attribute is only ever set by the *spoken* standby path. Enter standby any other way (the orb, a `gui_command`) and it does not exist → `getattr` returns `None` → the guard fails → **the microphone is never re-enabled**. |
| **Cause 3 — nothing verified the devices** | The wake path never touched the audio devices at all. A hold left set on the renderer, a dead output stream, or a closed capture stream all survived the wake untouched. |
| **Why it happened** | Standby state lived as five loose booleans across two objects (`standby_mode`, `_standby`, `_microphone_before_standby`, `_drop_live_output`, `microphone_enabled`) with no single owner. Nothing could see whether the combination was even legal. |
| **Fix** | New `orion_core/audio_recovery.py` owns both halves. Entry records state unconditionally; wake runs the brief's exact ladder — cancel standby → verify output → verify input → reopen streams → reset playback/interruption/mic flags → restart listening → confirm. It is idempotent (a second wake is a no-op; repeated wakes never stack streams, listeners or tasks) and returns a `RecoveryReport` that is spoken and logged. |
| **Files** | `audio_recovery.py` (new), `audio_devices.py`, `audio_state.py`, `live_worker.py`, `tests/test_audio_recovery.py` (new, 18 tests) |
| **Testing** | Each of the three causes has a named regression test. Idempotence, concurrent-wake collapse, missing-mic-engine, and total device failure all covered. |

> **Important, and not a bug:** neither **XRocker** nor **Fifine** is currently attached to this PC. Enumerated all 59 PortAudio devices — the outputs are HyperX Cloud III S, Pico Streaming, NVIDIA HDMI, MFDriver Virtual and Bluetooth headsets. `config/audio.json` already requests `"output": "Xrocker"` / `"input": "fifine"`, and `resolve()` silently fell back to the system default with no log line. The recovery sequence now **states this out loud** ("the XRocker isn't connected, so I'm using the system default") instead of pretending it restored hardware that is not there. When you plug them in, `preferred_index()` finds them by name fragment and they are selected and verified automatically — no config change needed.

### A3 · Speech stutter

| | |
|---|---|
| **Problem** | ORION's speech breaks up. Never reproducible on a bench. |
| **Cause** | `AudioPlaybackThread._write()` sliced the buffer at 4 KB (~85 ms) and, **between every slice**, computed an amplitude (a 512-sample pure-Python loop), an FFT spectral profile, and emitted two Qt signals. Six extra GIL acquisitions per 85 ms of audio, on the critical path. |
| **Measured** | Idle: a 341 ms buffer takes 322 ms to write — correct, the write blocks on the device consuming it. With three Python threads competing for the GIL — which is exactly what the 3-D swarm, the face renderer and the telemetry loop are — the **same buffer took 2311 ms mean, peaking at 2884 ms**. The renderer was being descheduled between slices while PortAudio's buffer drained to empty. That is the stutter, and it is why it only ever happens in the real app. |
| **Fix** | Two changes, no sleeps. (1) A dedicated `_VisualiserThread` samples the newest buffer at 20 Hz and does the amplitude/FFT/emit work off the audio path; the write loop's only added cost is one reference assignment. (2) An explicit `PLAYBACK_DEVICE_LATENCY_S` (0.30 s, `ORION_AUDIO_LATENCY` to override) so PortAudio holds enough buffered audio to ride out a GIL stall the renderer cannot prevent. Plus a Windows thread-priority boost and a shorter GIL switch interval. |
| **Result** | Under the identical 3-thread load: **2311 ms → 316 ms** per buffer, tracking the 333 ms of audio it contains. Writes per run went 5 → 15. |
| **Files** | `orion_core/audio.py`, `orion_core/constants.py` |

### A4 · ElevenLabs was architecturally wrong for the target design

| | |
|---|---|
| **Problem** | The requested architecture is *TTS stream → audio queue → dedicated worker → device*. The implementation was neither streaming nor shared. |
| **Cause** | `synthesize_pcm()` fetched the **entire** utterance before a single sample played, then `sd.play(..., blocking=True)` opened a **brand-new output stream per utterance**. Three consequences: silence proportional to reply length; a device open/close between every sentence (tens–hundreds of ms of dead air on Windows); and because `sd.play` uses PortAudio's *default* device, ElevenLabs audio could come out of a different speaker than the one ORION had just verified. Blocking also meant an interruption could not cut it. |
| **Fix** | New `stream_pcm()` uses the streaming endpoint and hands each chunk to a callback as it lands; `SpeechSynthesiser` feeds those straight into `AudioPlaybackThread`'s queue. One persistent, verified, deep-buffered stream shared with the native channel — with the prebuffer, coalescing, tail-drain and hold/resume already proven there. `should_stop` is polled between chunks so barge-in cuts the network read immediately. A truncated utterance is never cached. |
| **Files** | `voice_elevenlabs.py`, `audio.py`, `tests/test_elevenlabs_streaming.py` (new, 12 tests) |
| **Status** | Implemented and tested against a mocked endpoint. **Not exercised against the live API** — no API key or voice_id is configured (verified). See §F. |

### A5 · The globe "stopped working" — and the Command Deck navigation

**These are one defect, not two.**

| | |
|---|---|
| **Problem** | "The globe function does not work anymore" and "cannot navigate smoothly through the Command Deck". |
| **Investigation** | Built the real `GlobeView` in a Qt harness and drove it for 24 s. Result: `WEBENGINE_OK=True`, Cesium loaded, viewer created, `globe.show=true`, `tilesLoaded=true`, 2 imagery layers, **zero JS errors**, `loadFinished=True`. The CDN and Esri tile endpoints all return 200. **The globe is not broken.** |
| **Actual cause** | `app.py` called `deck.set_swarm_navigation(True)` **by default**, which hides the deck's zone row and page row entirely and makes the 3-D swarm graph the only way to open a page. `GlobeView` builds its renderer lazily on first `showEvent` — so a page you cannot reliably reach is a page that never initialises. Hunting for a node in a rotating point cloud is also the exact opposite of "instant, predictable, smooth". |
| **Fix** | Tab bars restored as the default; swarm-only navigation is now opt-in via `ORION_SWARM_NAV=1`. The swarm keeps every node and every bit of its functionality — it is simply no longer the *only* way in. |
| **Second defect found** | `_fade_in()` applied a `QGraphicsOpacityEffect` to every incoming page. On pages backed by a native surface (the WebEngine globe, the GL swarm) this forces the subtree through software compositing, which those surfaces do not participate in — the page can go black or blank. Now skipped for native-surface pages, and the previous animation is stopped before a new one starts (a fast run through pages left multiple animations driving effects on already-cleared widgets). |
| **Files** | `app.py`, `gui/unified_dashboard.py` |

### A6 · The Command Palette was decorative

| | |
|---|---|
| **Problem** | "Largely decorative." |
| **Cause** | Choosing *any* entry only ever typed its name into the message box. An entry labelled "System Scan" that types the words "system scan" into a text field is a label pretending to be a control. |
| **Fix** | New `orion_core/command_router.py` implements the brief's chain: *UI Action → Command ID → Command Router → Handler → Subsystem → Result → Event → UI*. Commands are only registered when their backend genuinely exists, so the palette can never list something that fails when clicked. Results return to the UI as a `dashboard_event`, never by touching a widget. Destructive commands confirm first. The GUI thread never awaits — it hands the coroutine to the loop via `run_soon()`, which also keeps a task reference (a task with no live reference can be garbage-collected mid-flight, which is one of the ways a command silently did nothing). |
| **Files** | `command_router.py` (new), `gui/command_palette.py`, `gui/core_window.py`, `app.py` |

### A7 · Protocols

Preserved all four originals (`morning`, `focus`, `wind_down`, `situation_report`) and added `afternoon`, `evening`, `emergency`.

- **Time awareness (§7):** a protocol declares the period it is *for*; the manager compares that against the shared clock at run time. Running the morning protocol at 23:10 still runs in full, prefixed with "it is currently night, not morning — ten past eleven at night. Running it anyway." It never asserts a time it has not checked.
- **Bounded (§10):** per-step timeouts (90 s; 30 s for emergency), so one hung feed cannot hold the protocol open for ever. Rate limiting and a re-entrancy guard collapse double triggers.
- **Emergency is not a macro.** Deciding whether something is an emergency, how reliable the source is and whether it actually reaches the user is a judgement no list of tool calls can express. `orion_core/emergency.py` gathers, classifies and scores, with confidence derived from the source (official domain → CONFIRMED, recognised outlet → REPORTED, low-authority → POSSIBLE, unattributed → SPECULATIVE **and severity downgraded**), location-based relevance, per-source timeouts, cancellation, rate limiting and fingerprint deduplication so the same storm is not re-announced. Ordinary news classifies as *not an emergency*; with no sources the honest answer — "I found no active emergencies affecting you" — is what it says.

### A8 · "Why did ORION ignore me?"

A voice request crosses eight subsystems on three threads and an event loop, and none of them shared an identifier — so every failure looked identical: silence. `orion_core/request_trace.py` assigns `ORION_REQ_000123` and stamps each stage. A request that stops is one whose last stage is not `LISTENING_RESTORED`, and the stage that *is* last names the owning subsystem. The worst case is called out explicitly: stopping after `PLAYBACK_COMPLETE` means "ORION finished speaking but listening was never restored — this is the state where he appears alive and ignores everything you say." Available in the palette as **"Why Did You Ignore Me?"**.

### A9 · A forged tool that opens the webcam at import — the boot blocker

| | |
|---|---|
| **Problem** | Startup never reached the Command Deck in a verification run, and went silent with no error. |
| **Why it was invisible** | `_boot_phase()` called `budget.mark(name)` but **never logged it**. A phase that stalled produced no output at all, so the last thing visible was whatever happened to log just before it. |
| **Cause** | `config/custom_tools/live_camera_analysis_tool.py` ended with `camera_analysis = LiveCameraAnalysis()` at module level, and that constructor calls `cv2.VideoCapture(0)`. ORION forges his own tools and the loader imports **every persisted one during boot** — so *importing* the tool opened the webcam. On Windows the probe blocks or fails hard, producing the `cv::obsensor … Camera index out of range` line present in **all three** of the user's own `orion_startup*.log` files. |
| **Fix** | Two layers. (1) The tool opens the camera lazily inside `run()` and always releases it; its `analyze_frame` now returns real measurements instead of the constant string `"Analyzed frame content"` it returned regardless of the image. (2) `ReflectiveModuleLoader` bounds every tool import (`IMPORT_TIMEOUT_S = 8.0`): a tool that will not import promptly is quarantined with a message naming the cause, and startup carries on. Self-written module-level code on the boot path is a single point of failure no individual tool can be trusted to avoid. |
| **Caveat** | A thread running an import cannot be killed, so the timeout frees *startup*, not the worker. Quarantining the file is what stops it recurring next launch. |
| **Also added** | Boot-phase timing. Slow phases (≥1.5 s) always log; `ORION_BOOT_TRACE=1` logs every phase with its duration. This is what located the stall in one run. |
| **Files** | `dynamic_loader.py`, `config/custom_tools/live_camera_analysis_tool.py`, `app.py`, `tests/test_forge_import_bound.py` (new, 7 tests) |

---

## B. Architecture

**Audio.** One output stream for ORION's entire voice, opened with explicit device latency and owned by `AudioPlaybackThread`. Both the native Gemini PCM channel and the ElevenLabs/local TTS path feed the same queue. Analysis for the avatar runs on a separate visualiser thread. `AudioStateMachine` is the single authority on "is ORION speaking?", now extended with `PROCESSING`, `STANDBY`, `RECOVERING` and `ERROR`, an enforced legal-transition table (illegal transitions are refused *and logged*), and `illegal_combination()` which names impossible flag states.

**Providers.** Left substantially alone — `ProviderRouter` in `providers.py` already implements profile abstraction, local/cloud selection, key and model rotation, degraded mode and failover. It did not need rewriting; it needed its time grounding fixed (A1). The Live *voice* channel remains Gemini-specific because native speech-to-speech is not an interchangeable interface (see §C).

**Events.** Subsystems publish on `OrionBus`; the Spellscape and the palette consume events and never reach into each other.

**Health.** `health_model.py` probes each subsystem cheaply, in isolation, and treats `UNKNOWN` as a first-class answer — an indicator that is green because nothing checked it is exactly the decorative indicator the brief forbids. A subsystem that was never wired in is *absent*, not green.

---

## C. API comparison

Figures gathered 2026-08-08; verify before committing spend.

| Option | Audio cost | Latency | Streaming | Tools | Best for |
|---|---|---|---|---|---|
| **Gemini Live** (current) | ~$3/M audio in, ~$12/M audio out; 25 tok/s audio | Native speech-to-speech, no ASR→LLM→TTS hop | Yes | Yes | **Keep as primary.** Barge-in, affective dialog and 70 languages are already wired, and it is 5–10× cheaper per audio minute than the OpenAI equivalent |
| **OpenAI Realtime** (`gpt-realtime-2.1`) | $32/M audio in, $64/M audio out (~$0.18–0.46/min uncached, $0.05–0.10/min cached) | Comparable | Yes | Yes | A failover peer, not a replacement — materially more expensive |
| `gpt-realtime-2.1-mini` | $10/M in, $20/M out (~$0.016/min base) | Comparable | Yes | Yes | Viable budget failover |
| **ElevenLabs Flash v2.5** | ~$50/M characters | ~75 ms inference (model only) | Yes (SSE) | n/a | **Voice layer** — now implemented |
| **Local** (Ollama + Whisper + Piper) | Free | 1–2 s end-to-end on an RTX 3060 with an 8B model | Yes | Limited | Offline fallback; already present via `local_brain.py` |

**Recommendation: do not switch primary providers.** Your reported latency problem was not Gemini — it was measured GIL starvation in the local playback path (A3) and a non-streaming TTS path (A4). Both are now fixed. The other JARVIS-style app on your PC feels more responsive largely because it does not run a Python-side 3-D renderer competing for the same GIL. Re-measure before spending: `TRACES.bottlenecks()` now reports mean ms per pipeline stage, which will tell you whether any remaining latency is the model or the machine.

---

## D. ElevenLabs plan — implemented

Config lives in `config/api_keys.json` → `"elevenlabs": {"api_key": ..., "voice_id": ...}`, resolved fresh per call. **No key is in source.** To enable:

1. `set_voice(api_key=..., voice_id=...)` (or edit the config section).
2. That is all — `available()` flips true and the streaming path activates on the next utterance.

Behaviour: streams chunk-by-chunk into the shared renderer; falls back to pyttsx3/PowerShell per-utterance on failure; a mid-stream failure keeps what already played rather than re-speaking the sentence in a second voice; barge-in cuts the network read; utterances cache on disk (200-file cap) and cache hits are delivered in the same chunked way.

---

## E. Test report

**Suite: 3261 passed, 2 failed, 1 skipped.** Both failures (`test_overlay_orb_face.py`) are **pre-existing** and present in the pre-change baseline — unrelated to this work.

New: `test_time_service.py` (50), `test_audio_recovery.py` (18), `test_request_trace.py` (31), `test_protocols_and_emergency.py` (33), `test_command_router_and_health.py` (26), `test_elevenlabs_streaming.py` (12), `test_spellscape.py` (22), `test_forge_import_bound.py` (7) — **199 new tests**.

Verified beyond unit tests:
- **Globe** — rendered in a real Qt harness, screenshotted, JS console captured. Working.
- **Audio playback** — benchmarked on real hardware, idle and under GIL contention, before and after.
- **Devices** — all 59 PortAudio devices enumerated; `verify()` opens and closes real streams.
- **Spellscape** — rendered at four sizes, screenshotted, frame rate measured (15 fps sustained), static-layer caching asserted.

Updated two existing tests whose assertions encoded behaviour this brief explicitly changes: the 17:00 evening boundary (§37 requires 17:59 afternoon / 18:00 evening) and a deck fixture missing the new page.

---

## F. Remaining problems

| Item | Status | Detail |
|---|---|---|
| **XRocker / Fifine absent** | Non-blocking | Neither is attached. The code selects and verifies them by name the moment they appear; until then it uses a verified default and says so. **Cannot be tested end-to-end until you plug them in.** |
| **ElevenLabs not live-tested** | Non-blocking | No API key configured. Logic tested against a mocked endpoint; the real round-trip is unverified. |
| **Live voice conversation not exercised** | Blocking for full sign-off | I cannot speak to ORION. The 10–20 consecutive voice interactions, real barge-in and live standby cycles in §37 need you at the microphone. The request tracing exists specifically so that if something does drop, it will now name the subsystem. |
| **Boot never observed reaching the deck** | Open | In my harness — stdout piped, no console — startup consistently ends just after the `mcp host + diagnostics` phase, before the Command Deck is built. Every constructor in the following phase is instant when timed individually, and no Python traceback is produced, so this reads as the app exiting rather than hanging. It may be an artefact of the headless harness rather than a fault in normal use, but I could not confirm the deck being constructed in a live run and do not claim it. Run with `ORION_BOOT_TRACE=1` to see exactly how far your own launch gets. |
| **2 pre-existing test failures** | Non-blocking | `test_overlay_orb_face.py` — present before this work, untouched, out of scope. |
| **Gemini Live remains provider-specific** | By design | Native speech-to-speech is not an interchangeable interface. Text generation is already fully abstracted behind `ProviderRouter`. |
| **Spellscape brain stem** | Cosmetic | The stem/cerebellum reads adequately but is not photorealistic. Reliability was prioritised. |
