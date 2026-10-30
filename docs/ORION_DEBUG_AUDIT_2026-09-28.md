# ORION debugging audit — 28 September 2026

## Scope and architecture

This pass audited the current, already modified checkout. It preserved the
existing working tree and concentrated on the desktop-to-phone gateway, Android
transport and background audio, hardware telemetry, deployment scripts, build
gates and repository-wide regressions. The active remote gateway is `aiohttp`
in `orion_core/remote.py`; it is not a FastAPI `main.py` service. The desktop
application is Qt-based. The earlier system-wide findings and unfinished
features remain recorded in `ORION_COMPREHENSIVE_AUDIT_2026-09-27.md`.

This is a broad automated and targeted manual audit, not a line-by-line proof
that every hardware, provider and mobile operating-system path is defect-free.

## Defects repaired

| Area | Reproduced or source-confirmed fault | Repair |
|---|---|---|
| Android microphone service | Lint rejected `AudioRecord` construction without a worker-thread permission check. A revocation or constructor failure could strand the foreground notification with no recording worker. | Recheck permission on the recording thread, handle construction and foreground-start failures, and stop the service when its worker exits unexpectedly. Use the microphone foreground type only from API 30. |
| Android TLS handoff | The native command channel treated a previously approved self-signed LAN certificate as an exclusive pin. It rejected Tailscale Serve's valid MagicDNS certificate after a handoff. Background audio used a separate system-only TLS client and could not use the approved self-signed endpoint. | Both channels now share one client. It accepts a valid system CA chain with a matching host name, or the saved certificate fingerprint for a configured ORION host. |
| Android keep-alive | A fixed-rate ping task could bunch up when Android resumed a cached process. | Schedule the next ping after the previous task finishes. |
| Audio WebSocket | The receive loop awaited transcription and a provider reply. A slow reply prevented `aiohttp` from reading protocol PONG frames, so its 15-second heartbeat could close a healthy connection. | Process one committed utterance in a separate task while the receive loop continues handling protocol frames. Refuse a second commit while a reply is active and cancel the task on disconnect. An accelerated heartbeat regression covers this path. |
| Hardware dashboard | A stalled WMI sensor held a lock while each polling request queued another worker. Missing loads were converted from JSON `null` to chart value zero. | Return a marked stale reading immediately when a sample is already in progress. The dashboard labels delayed data and does not plot missing or stale loads. |
| Android CI | The APK workflow built without lint, so the microphone permission fault did not fail CI. | Run `lintDebug` before `assembleDebug`. |

## Verification

- Complete Python regression suite against the repaired source: **7,867 passed,
  9 skipped, 7 subtests passed**, zero failures, in 570.30 seconds. Skips cover
  optional neural, Grandmaster, MCP and live-model dependencies, Windows
  symlink privileges, and one obsolete dependency check.
- Focused remote gateway, endpoint and audio suite: **68 passed**. The new
  regressions cover a slow audio reply under a short WebSocket heartbeat and a
  hardware sampler already occupied by a sensor call.
- Android `lintDebug assembleDebug`: **successful**, with **0 errors and 54
  warnings**. Remaining warnings mainly concern Android style, localisation
  and Gradle age; the deliberate certificate trust path also attracts static
  security warnings. The debug APK is at
  `android/app/build/outputs/apk/debug/app-debug.apk`.
- **879 Python files** parsed without syntax errors. The embedded mobile page
  JavaScript passed Node's syntax check. Both Tailscale deployment PowerShell
  scripts parsed with zero syntax errors.
- Capability registry validation passed with no missing or duplicate entries.
  The source publication check passed for **1,071 files**, without creating an
  archive.

One concurrent full-suite run showed a single failure around 15%, but was
interrupted before its traceback was captured. An isolated complete rerun
passed through that point and finished cleanly. The exact cause of that
transient result is unknown; no claim is made that timing faults are excluded.

## Live deployment state and acceptance gaps

Tailscale itself reports `Running` and the PC's MagicDNS name as
`your-pc.your-tailnet.ts.net`. `tailscale serve status --json` returned `{}`,
and no ORION listener was found on ports 8000 or 8765 during this audit. The
source and build checks therefore do not establish a working phone connection
on cellular data. Start ORION and Tailscale Serve using
`deploy/start_tailscale_remote.ps1`, then test Wi-Fi-to-cellular handoff on a
physical phone with the updated APK.

The repository has no iOS mobile client. Physical-device background microphone
behaviour, wake-word quality, TTS playback, certificate migration and hardware
sensor availability remain to be measured. The earlier comprehensive audit
also records outstanding sync transport, proactive Android delivery and live
generated-tool isolation work. Those are feature and acceptance gaps, not
results established by this test run.
