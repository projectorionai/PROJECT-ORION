# ORION Android: latency, capabilities and customer readiness

## Evidence and current release

Version 3.2 (`versionCode` 24) compiles, passes 52 JVM unit tests and Android
lint without errors. The signed APK is 1,549,640 bytes with SHA-256
`851c42c6cb96e113f54f14436231f296b62dcfe15b8599b855fb2a86cfd66b0c`.
The temporary installer-only HTTPS route returned matching bytes and returned
404 for `/api/health`. The user confirmed version 3.0 could answer aloud and
open an app with the PC off. Version 3.2 has not yet been timed on the user's
phone, so no latency reduction is claimed as measured.

Version 3.3 (`versionCode` 25) passes 53 JVM unit tests and lint without
errors. The signed installer is 1,551,400 bytes, SHA-256
`23df819825f047a7ebfe5a3cff5c8b05a404ee7fb3db4f7fadf91078ef60ac28`,
and retains the signing certificate of 3.2. The installer-only HTTPS route
returned these exact bytes and 404 for `/api/health`. The first physical-phone
timing sample is recorded below; a full listening and action acceptance pass
is still outstanding.

Version 3.4 (`versionCode` 26) passes 53 JVM unit tests and lint without
errors. Its 1,551,408-byte APK has SHA-256
`0a7d0b541e73d939f73f71c25282c798a3b52c832d6e765fb53282bb41c3ed25`
and the same signing certificate. It installed over 3.3 via wireless ADB,
opened on the phone, and the installer-only HTTPS route returned identical
bytes with 404 for `/api/health`.

## Where the waiting occurs

| Stage | Code | Current behaviour |
| --- | --- | --- |
| Foreground speech capture | `NativeDictation.kt` | Waits for 1.2 seconds of silence before ending a clip. |
| Background speech capture | `PcmSpeechSegmenter.kt` | Waits for approximately one second of silence. |
| Transcription | `PhoneBrain.kt:190` | Sends audio to Gemini before text generation can begin. |
| Reply preparation | `PhoneBrain.kt:233` | Loads up to 20 past turns and 128,000 characters, then may discover one configured MCP server synchronously. Discovery has a six-second budget when its cache expires. |
| Model request | `PhoneBrain.kt:274` | Sends the system instruction, conversation and tool declarations; phone actions may need a second request after their receipt. |
| Speech | `PhoneVoice.kt`, `PhoneLiveVoice.kt` | Charon Live can wait for session setup and its first audio; direct TTS is the fallback. |

Version 3.2 adds an **Options → Performance diagnostics** panel containing only
local durations and a model identifier. It stores no prompt, transcript, key
or audio. It records speech capture, transcription, AI preparation, request to
first text, complete reply and voice to first audio. Compare a short typed
question, a long typed question, a microphone request and background listening
on Wi-Fi and mobile data before tuning timeouts or changing providers.

The default Gemini Lite model is now used first for long and short questions.
The previous rule silently tried several larger models first for a long
question. A user-selected non-Lite model is used alone, preventing unrequested
model upgrades on a customer's key. The Gemini text model can be changed in
Options. Charon is prepared while text processing runs.

## Native capability ledger

The `PhoneCommandContract`, `IntentRouter` and `PhoneControls` currently expose
the following outcomes. An `ok` receipt means Android accepted the described
step, not that an external service finished it.

| Capability | Actual result |
| --- | --- |
| Open installed app, URL or settings | Opens the selected app, link or settings page. |
| Phone battery and connection | Reads battery, charging, network transport and power saving after native review; added in 3.2. |
| Call | Requests a call; connection is unverified. |
| SMS | Submits if Android grants the restricted permission; otherwise opens a Messages draft. Delivery is unverified. |
| E-mail | Gmail with Accessibility can attempt reviewed sending and return its receipt. Other clients open a draft. |
| Contact search and installed apps | Reads the specific contact or app list after review. |
| Alarm, timer, calendar event, contact creation | Opens the relevant app or draft for the user's final check. |
| Offline navigation | Hands a destination to OsmAnd. An address opens its search; ORION does not confirm a started car route. |
| Popular videos | Opens YouTube's Great Britain chart. |
| Share | Opens Android's share sheet; nothing is sent until the user chooses a recipient. |
| Screen controls | Accessibility can capture a semantic screen tree, click, scroll, set text, go Back or Home under review. |
| MCP | One configured HTTPS server; each tool call requires native review. |
| PC control | Separate Desktop mode while the PC is online. Standalone phone mode does not claim a Windows action ran. |

Version 3.2 adds **Options → Capabilities & readiness**, showing permission and
integration status together with these completion limits. The phone's front
page remains focused on listening and the terminal.

### Capability work still required

1. Add a local reminder store and notification scheduler with reboot recovery,
   explicit review and tests for late or revoked notifications. Alarms and
   calendar drafts are not a substitute for completed reminders.
2. Add media playback controls only after testing an Android-supported media
   session path on real devices; do not advertise play/pause as working until
   the active player can be identified and a receipt observed.
3. Replace the single MCP slot with a named, scoped server registry for home
   and customer services. Each server needs its own encrypted credential,
   health state, tool allowlist, timeout and approval receipt.
4. Add an explicit PC action bridge from standalone mode if the user wants a
   single conversation to command the home PC. It must identify the target,
   require the PC to be online and return a verified action result.
5. Embedded offline car routing would require downloaded map and routing data,
   storage management, turn instructions and driving tests. The current
   OsmAnd handoff must remain labelled as a handoff.

## Customer API-key and security gates

- First run now offers **Use my own Gemini key** before private PC pairing.
  Manual key entry reports **Key stored**, since syntax checks alone cannot
  prove the key or selected model works. Customers should use their own key.
- Private PC provisioning copies the PC owner's raw provider key to an
  authenticated phone and stores it in Android Keystore. Unpairing cannot
  revoke that copied provider key. This is suitable only for the owner's
  trusted devices; rotate the provider key if a device is lost. A customer
  release must not use one shared owner key.
- Groq, OpenRouter and Cloudflare Workers AI are text providers in the
  current app. Microphone transcription and Charon still require Gemini.
- A customer release needs a bounded provider check before saying a key is
  ready, model capability checks for function calls, per-key usage visibility,
  a user-set request or spending budget, and explicit handling for exhausted
  quota. Long history and continuation calls currently have no phone-side
  budget. These are release gates, not features claimed complete in 3.2.
- Version 3.3 checks a manually supplied Gemini key against Google's
  `models.get` endpoint and checks that the chosen text model supports
  `generateContent` before replacing a working setting. A text-capability
  response alone does not prove function-call support.
- Keys already use Android Keystore, Android backup is disabled, and provider
  endpoints are fixed HTTPS hosts. Device-side testing is still required for
  Keystore invalidation, app upgrades and revoked permissions.

## Verification still needed on physical devices

Record the 3.3 timing panel after cold and warm requests on Wi-Fi and mobile
data. Test long answers to their final sentence, background listening with the
screen locked, review expiry, Gemini quota errors, a bad customer key, Gmail
sending, SMS draft or submission, OsmAnd search and a real MCP outage. JVM
tests cannot establish Android microphone, AudioTrack, carrier or app handoff
behaviour.

## Version 3.3 continuous listening

Once the user starts Background listening, each completed speech segment is a
request without a mandatory wake word. Saying “ORION” remains optional.
**Options → Require wake word** restores the earlier mode if wanted. Exact
“stop” or “stop speaking” interrupts speech, and “stop listening” ends the
foreground service. Clips captured while ORION speaks can only stop playback;
the playback tail has a three-second cooldown. Sensitive phone actions retain
their native review. Continuous listening can still transcribe ambient speech,
use the customer's API quota and consume battery, so its persistent Android
notification and Stop action must stay visible.

The app still waits for an utterance to end and runs transcription, text
generation and Charon speech in sequence. Removing the wake-word gate does
not shorten those provider stages. The duration panel distinguishes them and
should guide the next latency fix.

### First connected-phone timing

After wireless installation of 3.3 on the owner's Galaxy S23 Ultra, one
background request recorded 1.169 s for transcription, 0.005 s for AI
preparation, 0.629 s to first text, 0.636 s to complete the reply, and 3.263 s
from voice start to first audio. The chosen text model was
`gemini-3.5-flash-lite` in one attempt. Live Charon and the no-wake setting
were active. The speech segmenter additionally waits for roughly one second
of silence; this is not included in those stored durations. Charon is the
largest measured stage. This is one real-device sample, not a median or a
guarantee across networks or answer lengths.

The owner's phone then selected Direct TTS. A typed short reply on 3.3 reached
first sound in 1.352 s; a second typed short reply on 3.4 reached first sound
in 1.02 s using `gemini-3.8-flash-lite-tts`, with first text in 0.85 s.
These are separate single samples, so they show that Direct TTS can start
faster on this phone, but do not prove a reliable average improvement. The
3.8 Flash TTS route returned HTTP 429 during a separate test. Version 3.4
therefore attempts Live Charon if all direct models are unavailable before
any audio plays, and records the actual speech route and model in the timing
panel. The fallback code passed compilation and JVM tests; quota exhaustion
has not yet been reproduced on the device.
