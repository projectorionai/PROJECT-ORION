# ORION release review — 23 September 2026

This review covers the current Mark XXX source, following the historical
[18 September inventory](CAPABILITIES_2026-09-18.md). It records what was checked,
what changed, and the limits of the automated evidence.

## Capability preservation

The release gate records **139 declared tools, 140 handler mappings, 26 dashboard
pages, 18 public plugins, 5 entry points and 7 fallback symbols**. These counts
represent interfaces, not independently certified integrations. The extra
handler mapping is not counted as another declared model tool.

Run `python -B tools/check_capabilities.py` before a release. CI also runs this
gate. Removed or duplicated capabilities fail the check; additions are reported
for review. The baseline does not replace behavioural tests.

Current capabilities include voice/text conversation and tools, local memory
and retrieval, research and document workflows, desktop and browser automation,
camera/PCB inspection, chess, plugin management, specialist workspaces, remote
access and optional messaging, home-automation and telephony integrations.
GLOBE, CHESS and MISSION construct their dashboard pages on first use. BRAIN
remains available immediately because other services use it.

## Reliability and cost changes

- Plugin dispatch preserves explicit failure states and structured evidence.
  Invalid or absent results no longer become successful tool results.
- Fitness goals and workout records are committed and read back locally.
  Diagnostic records and tool drafts explicitly distinguish storage from an
  actual repair, reminder, or generated tool.
- Both ordinary and streamed background thoughts honour the local-only policy.
  Paid background thoughts require the existing `ORION_PAID_THOUGHTS=1` opt-in.
  Foreground provider choices still determine conversation costs.
- A failed deferred page can be retried; it is not marked as constructed.
- Audio queued for an old connection is discarded after reconnection or shutdown.
- Plugin enable/disable changes are atomically written and verified. Write
  failure restores the previous in-memory state and reports failure.
- Dependency checks inspect installed distribution versions. Explicit installs
  use the running Python interpreter, one requirements-file invocation, and a
  120-second timeout. A frozen executable cannot invoke itself as Python/pip.
- The repair-prediction plugin loads scikit-learn only when invoked. Its
  classifier predicts from supplied examples; it does not repair source code.
- Integration plugins report configuration, transport and service errors as
  failures. Hue checks bridge error replies and reads back reported light state.
  Push and IFTTT acknowledgements do not claim delivery or applet completion.

## Plugins available in the source release

All 18 plugin modules imported and registered with ORION's runtime loader in an
isolated checkout. No external actions or dependency installations were performed
by that check. The manifests declare required packages; configuration and
credentials are deliberately excluded from the release.

| Plugin | What is available / what is still needed |
| --- | --- |
| dependency_manager | Inspect requirements; explicitly requested installation in a source Python environment |
| dependency_resolver | Check installed package versions without importing or installing them |
| json_validator | Local JSON validation; jsonschema dependency |
| live_camera_analysis | Shared-camera analysis; camera and configured vision route when used |
| emotional_monitor | Local lexical estimate from supplied text; not a verified emotional state |
| emotional_insight_generator | Filter supplied emotion records; not independent emotion detection |
| error_handler | Store a diagnostic record; actual repair remains a separate operation |
| synthesis_enhancer | Store a proposed tool specification; forging/activation remain separate |
| fitness_goal_tracker | Durable goals, progress and deadlines; does not silently schedule reminders |
| fitness_progress_monitor | Durable supplied workout records and metrics |
| self_repair_agent | Local classifier using supplied training data; scikit-learn on first use |
| open_meteo_weather | Weather lookup; network required, no account key required by the plugin |
| homeassistant_call_service | Home Assistant instance URL and token |
| philips_hue | Local bridge and pairing; reported device state is checked after changes |
| ifttt_webhook | Configured Webhooks key and event; acknowledgement only |
| push_notify | Configured ntfy, Gotify or Pushover route |
| discord_message | Configured Discord route/credentials |
| telegram_message | Configured Telegram bot and destination |

The Plugins page controls enabled state. A plugin importing successfully does
not establish that an external account, device or endpoint is configured.
See the plugin manifests and [messaging documentation](MESSAGING.md) for setup.
Restart an updated source instance to load the revised modules. An older frozen
executable does not acquire core source changes automatically.

## Automated evidence

**6,842 passed, 4 skipped, 0 failed** in 196.81 seconds. After the run, a
packaging-test isolation adjustment was checked separately: all 3 fingerprint
tests passed. Production code did not change after the full run.

The checks use Windows, Python 3.13 and Qt's offscreen platform. They include
failure paths for persisted outcomes, plugin state writes, paid-thought policy,
plugin loading, interrupted/reconnected audio, camera lifecycle, deferred page
recovery, and synthetic PCB images with blur, darkness, glare and rotation.
A captured still with no successful OCR must not invent readable markings.

Synthetic boards are repeatable regression fixtures, not evidence of accuracy
on real electronics. No camera, microphone, physical board, live account action
or cloud conversation was exercised in this audit. Full physical voice sessions
and real-board inspection remain validation tasks.

## Measured startup components

Three fresh subprocesses per case; median values on the audit machine:

| Component | Eager / before | Deferred / after |
| --- | ---: | ---: |
| Repair-prediction plugin import | 1.114 s | 0.0016 s |
| Minimal LOG/MISSION dashboard construction | 0.0553 s | 0.0215 s |

The second measurement uses real widgets with service engines absent and an
offscreen Qt platform. Work is deferred to first use, not eliminated. Importing
`orion_core.app` alone took a median 0.243 s. These are component measurements;
they are not full application startup or real voice-readiness measurements.
Hardware readiness was not measured under the automated-only scope.

## Source privacy and publication

The source export excludes Git history, credentials, private configuration,
databases, browser data, development environments and installed binaries. The
publication audit checks recognised credential patterns, supplied private terms
and known local credential values. It compiles every exported Python file and
verifies the ZIP CRC and SHA-256 manifest.

Personal identifiers were removed from public prose/test examples. The local
speaker encoder contained identifying export metadata: 71 metadata entries were
removed, reclaiming 30,271 bytes. Model weights were unchanged, ONNX validation
passed, and two deterministic inference inputs produced exactly equal outputs
before and after sanitisation. Reviewed model digests are pinned by the exporter.
Third-party model licence texts and provenance are included with the assets.

These checks do not prove that every possible unknown secret or private passage
is absent. Existing Git history is a separate publication risk. Start a fresh
repository from the audited source export; do not push the live installation's
old history. Rotate credentials that appeared in historical commits. Follow
[Publishing ORION](PUBLISHING.md).

## Executable validation

The existing installed executable differs from the audited source: 314 matching
core modules, 9 differing modules and 10 missing modules. A separate rebuilt
candidate contains **all 333 core modules**, with matching compiled code, and
48 reviewed public configuration/source defaults. This comparison ignores
source paths, line tables and incidental string interning, but compares bytecode,
constants, signatures and exception tables.

The candidate has not been launched, promoted into the installed directory or
connected to hardware/accounts. Compiled-code agreement is packaging evidence,
not an end-to-end runtime test. Keep its whole folder together: it is a folder
build, not a single self-contained executable. The standalone build deliberately
excludes Torch/MediaPipe/Resemblyzer extras; use the source environment for those
optional paths. ONNX voiceprint and OCR models are bundled.

Use `python -B tools/inspect_installed_build.py path/to/ORION.exe` to repeat the
comparison. Build in a clean separate directory; the builder refuses to overwrite
an installation containing private state.
