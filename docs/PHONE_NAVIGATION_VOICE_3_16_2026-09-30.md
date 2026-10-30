# ORION 3.16 — offline guidance and voice concurrency

This continues the resumed phone phase after the measured Forge/reflections
work and the 3.15 source fixes. It keeps the owner's existing data/signing
identity. Cloud deployment and the later Globe/chess additions remain queued
under their existing acceptance conditions.

## Faults found and corrected

| Fault or missing function | Correction |
| --- | --- |
| Background capture discarded every frame whilst the AI reply was still streaming | A separate, single-flight recogniser accepts stop-only clips during echo-cancelled playback; it cannot execute actions or replace the main request's HTTP call |
| A delayed interruption could cancel a newer foreground/background reply | Shared playback epochs, segmentation reset on epoch changes, accepted-request checks and per-voice conditional cancellation |
| Queued speech had no cancellation identity until its main-thread callback ran | Reserve its generation and pending state before scheduling; cancellation rejects the queued callback |
| Foreground submission closed every idle prepared Live speech session | Stop active or scheduled output synchronously for normal submissions; keep full reset for explicit cancellation, configuration changes and closure |
| Foreground/remote speech bypassed the background service's own acoustic-tail guard | Detect aggregate playback transitions and preserve the three-second tail |
| Navigation directions could be transcribed as new instructions | Shared utterance leases block all three capture paths, including echo-cancelled capture and the actual playback tail; explicit tap dictation may interrupt an announcement |
| Native map stopped route following whilst paused and had no spoken turns | Explicitly started GPS foreground service, installed offline UK speech, notification reopening/Stop and complete lifecycle cleanup |
| Deviations required manual route recalculation | Sustained accurate off-route evidence, current-BRouter-only recalculation, original-destination checks, single-flight routing and a sixty-second attempt cooldown |
| A reroute or restored route could acquire a racing journey's provenance | Read route/revision together; start against an exact revision; reserve a reroute only whilst its source remains current |
| Repeating the last fix could either repeat a prompt or stop valid speech | Fresh unchanged watchdog repeats retain valid progress without actions; stale/invalid fixes suspend speech |
| Map progress and speech warnings disappeared behind generic route text | Current instruction/distance, explicit unavailable-speech status and retained stop/failure feedback in the existing route status area |

Interrupt recognition is capped at five clips per minute and shares the
existing provider cooldowns. Only exact recognised stop phrases can cancel a
reply; background capture with no echo cancellation remains paused during
playback. This is a source correction, not a measured end-to-end latency claim.
The requested cloud voice and authorised ordinary phone-voice fallback remain
available for general assistant replies.

## Offline guidance contract

Choose **Maps → Routes → Start guidance** after selecting the journey. The
visible Activity requests precise Location/notification permission and starts
an unexported location foreground service. There is no boot receiver,
automatic background restart or background-location permission request. An
explicit coordinate-navigation action follows the same path after its route
is ready whilst the Activity remains visible.

Directions use supplied GPX manoeuvres and actual along-route distances, with
deduplicated approach stages at 500/180/45 metres. No turn is inferred from an
unannotated track. GPS must be fresh and have reported accuracy within fifty
metres. Arrival requires the route endpoint and along-route progress, with
crossing/backtracking checks. Invalid, stale or missing fixes pause guidance.
Notification and location updates continue when the screen is locked until
the user stops navigation, it arrives, permissions change or Android ends the
service. A terminated journey must be started again from the map.

Automatic recalculation applies only to the currently calculated BRouter car
journey, after three distinct off-route fixes spanning at least eight seconds.
Attempts are at least sixty seconds apart. Imported or reopened GPX routes
retain their geometry/turns but cannot silently become a car route: calculate
them with BRouter again to enable automatic recalculation. Routing uses the
existing installed `car-fast` profile and downloaded graph regions.

The installed local speech voice must be non-network `en-GB`; directions make
no AI/provider/MCP request. Missing offline speech is shown explicitly while
visual guidance remains available. Navigation audio uses transient ducking,
bounded announcements, stale-callback leases and a three-second acoustic tail.
An old turn/distance is discarded after its position is passed or it becomes
stale. Other ORION speech takes precedence. Vehicle/Bluetooth behaviour still
needs physical measurements.

The implementation follows Android's [foreground service types](https://developer.android.com/develop/background-work/services/fgs/service-types),
[background-start restrictions](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start)
and [voice API](https://developer.android.com/reference/android/speech/tts/Voice).
This does not provide native offline address search, traffic or satellite
photography. The integrated OsmAnd hand-off remains available for its own
maps/address search/guidance. Road graph freshness and restrictions depend on
the installed dataset.

## Verification and installation status

The combined Android unit gate passed **174 tests**, with zero failures,
errors or skips, including twenty-seven new guidance cases. The related Python
phone/backend gate passed **180 tests** in 56.15 seconds. Both Android variants
compile and assemble. Final lint/build and APK hashes are recorded below after
the compatibility follow-up. The capability-preservation check found no missing
or duplicate identity. These gates are separate; their counts are not a new
whole-project total.

The final offline Gradle gate completed successfully in **1 minute 33 seconds**.
All 174 unit tests passed again; lint reports **zero errors and 141 warnings**.
The navigation notification now guards its Android 28 category on Android
26/27 and uses the current Icon action constructor. Remaining warnings concern
style/localisation and existing maintenance; Gradle 10 migration is still due.
The known-credential audit passed 1,341 source files, without producing an
archive or publishing anything.

| APK | SHA-256 |
| --- | --- |
| `releases/android/orion-companion-3.16-debug-compatible.apk` | `49FA8762C96C394657575C6E7BCF6247079B81940C50F541F6F9E5491FA95158` |
| `releases/android/orion-companion-3.16-production.apk` | `03E29BF4F415F327BEAE08E4EBA7BAF413D37D6954BC7B5A7C830E90780FCCC3` |

Both final signatures were verified. The existing download alias
`releases/android/orion-companion.apk` contains the compatible owner APK with
the same SHA-256. Local `/download/orion-companion.apk?v=3.16` returned HTTP 200
and `Cache-Control: no-store`; this is a loopback check, not a working public
HTTPS route or an installed-device result.

The owner update must retain certificate
`e4ba6e4bab286fa16c9606867b524255c22c3442bcd67351a1f2019ddbfe4127`.
The separate production certificate is
`3b78cd7b606d222b091e7497525fe2b79235b3d62de32a19499969294821fe02`.
Use only the compatible update for the existing installation; never uninstall,
clear storage or replace Keystore to bypass a mismatch.

Wireless ADB/mDNS found no connected or discoverable phone. Version 3.16 is
not yet installed during this phase. Actual microphone-to-first-text/speech
timing, stop during a long streamed reply, offline TTS, Android screen-lock
lifecycle, real BRouter calculation, route restoration and physical driving
remain device acceptance gates. The prior BRouter/map installation and unit
tests do not constitute those checks. A journey observation should come from
the owner without operating controls whilst driving.

The updated desktop was started with the existing deployment launcher;
`http://127.0.0.1:8000/api/health` returned `ok=true`. This confirms local
startup, not reachability of an old temporary HTTPS address.
