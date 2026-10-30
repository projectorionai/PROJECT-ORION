# ORION Android source audit — 29 September 2026

## Scope and evidence

The first pass covers all 38 Kotlin production files, the Android manifest,
network security settings, resources, build configuration and 11 Android unit
test files. The combined 2.8 release compiled and packaged successfully;
49 JVM unit tests passed. Android lint finished with no errors and 80 warnings.
The signed APK is distributed through the installer-only HTTPS route. This is
source and build verification, not a claim that every device-specific
operation has been exercised on a phone.

## Production-file register

| File | Area checked | Result |
| --- | --- | --- |
| `GeminiStream.kt` | Streaming event parser and termination | Unit tests pass; no confirmed new fault. |
| `IntentRouter.kt` | Native action validation and receipts | Restricted hands-free exemption to local phone mode; corrected contact insert intent. |
| `LiveTurnGuard.kt` | Turn lifetime and cancellation | Unit tests pass; no confirmed new fault. |
| `MainActivity.kt` | Desktop WebView, reconnect, TLS and lifecycle | Cancelled invalid TLS, rejected failed main-frame loads, gated native bridge, stopped background UI ticker. |
| `NativeBridge.kt` | JavaScript to native boundary | Added trusted-page check before native methods. |
| `NativeDictation.kt` | Tap-to-dictate capture | Audited with voice path; no confirmed new fault. |
| `OrionAccessibilityService.kt` | Snapshot, gesture and e-mail controls | Added gesture completion receipt and timeout; sensitive actions retain review. |
| `OrionAudioService.kt` | Desktop-mode background audio socket | Discarded stale socket events and cleared pending reply on network loss. |
| `OrionFrameTransport.kt` | Paired uplink and HTTPS | Removed custom self-signed certificate acceptance; uses platform TLS. |
| `OrionProjectionService.kt` | Screen capture lifecycle | Audited asynchronous start; the UI now waits for active capture before reporting success. |
| `OrionSentinelService.kt` | Floating controls and notification | Audited start/stop, overlay and foreground hand-offs; device check remains. |
| `PcmFrames.kt` | PCM framing | Unit tests pass; no confirmed new fault. |
| `PcmSpeechSegmenter.kt` | Voice activity segmentation | Unit tests pass; no confirmed new fault. |
| `PhoneActionCoordinator.kt` | Action queue and expiry | Enforces expiry at execution even when a timer is delayed. |
| `PhoneActionRules.kt` | App matching and input limits | Requires matching package and label; ambiguous apps remain a choice. |
| `PhoneAiProfile.kt` | Key provisioning and HTTPS | Client uses system TLS and disables redirects for credential requests. |
| `PhoneBrain.kt` | Gemini tools, retry and model routing | Audited tool turns and response handling; existing HTTP 400 regression remains covered. |
| `PhoneCommandContract.kt` | Allowed phone command names | Audited against action router; no confirmed new fault. |
| `PhoneConsoleActivity.kt` | Phone UI, approvals and lifecycle | Recovery clears transient voice error; screen sharing waits for capture; UI meter stops while paused. |
| `PhoneControls.kt` | Manual action controls | Deduplicated launcher packages. |
| `PhoneConversationStore.kt` | Bounded encrypted history | Audited size and role handling; no confirmed new fault. |
| `PhoneHostedChat.kt` | Optional hosted provider stream | Reviewed endpoint, response and redirect behaviour; no live third-party key test. |
| `PhoneHostedProfile.kt` | Provider selection and stored keys | Reviewed key and model validation; no live third-party key test. |
| `PhoneListeningService.kt` | Wake listening and replies | Emits recovery only after capture resumes following transient failure. |
| `PhoneLiveProtocol.kt` | Live audio messages | Unit tests pass; no confirmed new fault. |
| `PhoneLiveVoice.kt` | Gemini Live speech | Disabled WebSocket redirects that could expose its credential; phone check remains. |
| `PhoneMcpClient.kt` | Remote MCP configuration and calls | Reviewed HTTPS and per-call authorisation; no live MCP server test. |
| `PhoneMediaProtocol.kt` | Audio packet validation | Unit tests pass; no confirmed new fault. |
| `PhoneSecretStore.kt` | Encrypted phone settings | Reviewed Keystore storage and failure handling. |
| `PhoneVoice.kt` | Charon speech and model fallback | Tries another TTS model after format failure; model-failure cache is key-scoped. |
| `Prefs.kt` | Saved addresses and pairing | Accepts only clean HTTPS origins and rejects credentials, paths, queries and fragments. |
| `RemoteSpeech.kt` | Desktop speech streaming | Tries another saved HTTPS endpoint when no audio has played. |
| `SecureStore.kt` | Pairing-token encryption | Reviewed Keystore encryption and migration handling. |
| `SentinelActionContract.kt` | Explicit overlay actions | Checked supported-action allowlist. |
| `SetupActivity.kt` | Initial host entry | Checked HTTPS enforcement and changed-host credential reset. |
| `SlideToConfirmView.kt` | Native approval gesture | Checked review boundary; retained for sensitive actions. |
| `SpeechText.kt` | Speech text formatting | Long replies now split within the desktop speech endpoint's 1,200-character limit. |
| `TactileWaveformView.kt` | Listening and speaking indicator | Checked state updates and layout; phone visual check remains. |

## Material limits still to verify

- Android can block launching another app from the background or lock screen.
  A safe request made while ORION is paused currently waits behind a
  notification; opening an app entirely hands-free while locked is not proven.
- Gmail sending, SMS permissions, wake recognition, the microphone path,
  Bluetooth audio, OsmAnd hand-off and screen projection require a real phone.
- `addJavascriptInterface` is visible to frames in Android WebView. The app
  gates it to the trusted top-level page, blocks untrusted frame navigation,
  and the served presentation iframe is sandboxed without scripts. Migrating
  to a frame-aware WebMessage listener would further strengthen isolation.
- A self-signed direct-tailnet certificate no longer works with the current
  app. Tailscale Serve or another Android-trusted HTTPS route is required.
- The 80 remaining lint warnings are mostly style, localisation and legacy
  Gradle API notices; no lint error remains. The one static-context warning
  refers to an application context retained by the uplink singleton.

## Release evidence

- Version 3.2 (`versionCode` 24) records phone-only response timings in
  Options, honours the selected Gemini text model for every question,
  prepares Charon whilst work runs, and adds a reviewed battery/connection
  action plus a capability and readiness panel. First-run setup offers a
  customer's own key before private PC provisioning. The signed APK passed
  52 JVM tests and lint with no errors; its SHA-256 is
  `851c42c6cb96e113f54f14436231f296b62dcfe15b8599b855fb2a86cfd66b0c`
  (1,549,640 bytes). Its installer-only HTTPS GET matched that hash and
  `/api/health` returned 404. Real-phone latency timing is still pending.
- Version 3.1 (`versionCode` 23) adds ordered, cancellable Charon speech from
  complete streamed phrases on the terminal and in background listening.
  Requests that could invoke phone actions still wait for their action receipt.
  Android compilation, 52 unit tests and lint passed with no failures or lint
  errors. The signed APK SHA-256 is
  `f13f09f0c51dc555d56d5b6b513e78c5ba09259d4f133e49c7547336336b0fb0`
  (1,544,772 bytes). `apksigner` verified the existing signing certificate.
  A GET through the installer-only HTTPS route returned the same SHA-256 and
  `/api/health` returned 404. The user confirmed that version 3.0 answered
  aloud and opened an app with the PC off; version 3.1 still needs a phone
  timing check for the improvement in speech onset.
- Version: `3.0` (`versionCode` 22).
- Signing certificate SHA-256:
  `e4ba6e4bab286fa16c9606867b524255c22c3442bcd67351a1f2019ddbfe4127`.
- APK SHA-256:
  `e99f4b7391710be60121cb694423984f95a9c4e0dace69fc7efc11c16b7e7632`.
- APK size: 1,542,452 bytes.
