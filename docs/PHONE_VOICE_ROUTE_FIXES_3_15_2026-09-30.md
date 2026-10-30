# ORION 3.15 — resumed phone acceptance

The latest user instruction resumes phone and driving checks. The previous
deferral remains historical context, not a current restriction.

## Source faults fixed

| Fault | Correction | Verification |
| --- | --- | --- |
| Background capture always waited 1,000 ms after speech | Shared 600 ms endpoint timing | Quiet/short speech, half-second breath and partial-frame regressions |
| Tap dictation waited 1,200 ms and required 1,500 ms capture | Same 600 ms endpoint with an 800 ms minimum | Monotonic-clock and short-request timing cases |
| A request exceeding ten seconds could be discarded and its tail treated as a new instruction | Bound complete requests to thirty seconds; discard an over-budget background utterance until a full quiet pause | Twenty-second complete request, continuous overflow, breath, recovery and explicit reset cases |
| Map screens shared `route.gpx.pending` without a shared revision/lock | Shared per-file route store, latest-request revision, bounded restoration and atomic replacement | Cross-instance stale writes/clear, cancellation, corruption, invalid UTF-8 and reopen cases |
| AI prompt demanded Desktop Mode for every PC task despite explicit PC controls | Name paired PC power controls and native confirmation separately | Presentation/request contracts and compiled prompt wiring |

The endpoint reduction is a source-level reduction in the fixed capture wait,
not a measured improvement in cloud inference or Charon synthesis. A human
pause longer than the endpoint threshold may still end a request. The existing
three-second anti-echo cooldown remains; microphone, speaker and vehicle noise
need physical-device measurements.

## Build and signing

Version **3.15 / versionCode 37** builds offline with the installed Android
toolchain. The combined gate passed **139 unit tests**, with zero failures,
errors or skips. Debug and release APKs built successfully. Lint reports zero
errors and **137 warnings**; Gradle's existing deprecated APIs remain maintenance
work before Gradle 10.

Local APKs, kept outside Git:

| Variant | Local path | SHA-256 |
| --- | --- | --- |
| Owner update | `releases/android/orion-companion-3.15-debug-compatible.apk` | `461652450095C1644EF776CD49979C54E2C3E68AB2680A4864B37580B883A4C6` |
| Fresh customer release | `releases/android/orion-companion-3.15-production.apk` | `332F45C8DDEE630CD78158923EE6934619E00B1F3454CD75E57BB466CCF650F5` |

The owner update has the same certificate as the installed 3.14 family:
`e4ba6e4bab286fa16c9606867b524255c22c3442bcd67351a1f2019ddbfe4127`.
The production certificate is different:
`3b78cd7b606d222b091e7497525fe2b79235b3d62de32a19499969294821fe02`.
Both signatures were verified. Do not uninstall, clear storage or replace the
Keystore to force a mismatched update; encrypted keys, pairing, conversations
and downloaded maps must be preserved.

## Device checks still required

ADB and mDNS found zero connected or discoverable devices after the build.
Wireless debugging needs the phone/PC on the same Wi-Fi and the current
connection address/port, plus a pairing address/port/code if this PC is no
longer paired. The APK has **not** been installed during this resumed phase.

Once connected, install the matching owner APK with `adb install -r`, verify
versionCode 37, and open the normal launcher. Record actual capture,
transcription, first reply text and first speech timings. Test background
conversation without the wake name, long replies, stop, selected cloud voice,
normal phone-voice fallback and the retained Face/Orb choice.

Whilst stationary, plan between two explicit UK coordinates using the installed
BRouter service and real graph regions. Verify returned car-route geometry and
instruction offsets, reopen/restoration, moving measured GPS, stale-fix refusal
and operation with mobile data disabled. Do not infer driving acceptance from
unit tests or injected coordinates. A physical journey requires the owner's
observations without interacting with controls whilst driving.

The native map still lacks spoken turns, automatic rerouting, offline address
search and locked-screen following. OsmAnd remains the integrated full spoken
guidance option after its own maps are downloaded. ORION's vector maps are not
offline satellite imagery. See the [routing guide](OFFLINE_CAR_ROUTES_2026-09-30.md).
