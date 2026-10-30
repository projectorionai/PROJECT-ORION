# ORION Companion — native Android app

A native Kotlin/Android app that works independently on the phone by default:
Gemini, Charon speech, the terminal, background listening, Sentinel and
reviewed Android actions run on the phone after its AI key is provisioned.
The PC can then be switched off, provided the phone has mobile data or Wi-Fi.
**PC connection** retains the paired HTTPS uplink, desktop memory and PC tools;
choose it explicitly from Options when the host is online.

---

## What you get

- **Phone screen** — the terminal and listening box stay in front. A dark
  Options sheet groups native controls, permissions, MCP connection and desktop
  mode while keeping the visual choice immediately accessible.
- **Screenshots** — Android's normal screenshot controls work in the phone
  console. **Options → Phone → Save screenshot** also captures the current
  screen and opens Android's file picker so you choose where it is stored.
- **ORION's face** — the PC's current Screen Face (the Sentinel design, the
  same expressions, timing and crimson-and-silver colours), drawn in 2-D and
  bundled in the APK, so it works with the PC switched off. There is no orb
  and no Three.js any more. While the paired PC is reachable the face follows
  the PC's state and emotions (and takes its design and colours, remembered for
  offline use); otherwise it follows the phone's listening, thinking and
  speaking and reads the feeling in what is said with the PC's sentiment rules.
  The speaking waveform follows the audio sent to the phone speaker.
- **Independent voice** — Gemini Flash-Lite transcribes phone microphone audio.
  ORION streams Charon speech from a compatible Gemini TTS model directly to
  the phone. Long replies are grouped to conserve speech requests and played in
  order. If cloud speech is unavailable before any audio for a section, the normal
  phone voice reads that section. A partially spoken section is never replayed;
  subsequent untouched sections may use phone speech. Voice selection and the
  explicit Phone option are in Options. Stop cancels speech and pending work.
- **Native actions** — installed apps and games, contacts, calls, SMS, reviewed Gmail
  sending, drafts, links, maps, navigation, share sheet, settings, alarms and
  calendar/contact drafts return explicit receipts. Sending an e-mail requires
  Accessibility, exact composer verification and a local approval slide.
- **Offline maps and car routes** — **Options → Phone → Offline UK map** opens
  ORION's native Mapsforge map. Its Great Britain and Ireland map packs show
  roads and the measured phone position. **Places** saves named coordinates;
  **Routes** plans between coordinates, imports GPX and calculates local car
  routes through installed BRouter and its separate downloaded road regions.
  Route geometry, supplied manoeuvres and along-route distance survive reopening.
  The latest requested journey owns the saved route across map screens; cancelled
  or superseded calculations cannot replace it. These are vector maps, not
  satellite photos. **Routes → Start guidance** uses downloaded UK phone speech,
  follows fresh GPS whilst locked and recalculates sustained deviations from a
  freshly calculated BRouter route. Imported/restored tracks do not acquire a
  car-routing claim. Offline address lookup remains unavailable. OsmAnd supplies
  its own address search and guidance when journey maps are downloaded. Prepare routes
  whilst parked; a map-screen launch does not confirm that routing succeeded.
- **Faster context** — recent encrypted phone history remains on the phone;
  each AI turn sends a bounded excerpt with the current request intact and
  the tail of long previous replies for follow-up questions.
- **SMS on sideloaded Android** — if direct SMS access is available, ORION
  submits the reviewed message. Where Android withholds that permission, ORION
  opens the exact recipient and text in the phone's messaging app and reports
  that the draft still needs the user's Send tap.
- **Listening and Sentinel** — a user-started foreground service accepts
  natural speech whilst the phone is locked, subject to Android microphone and battery
  rules. “ORION” is optional; Options can require it again. The floating Sentinel opens dictation, listening, screen sharing and
  phone controls. Screen sharing needs Android's MediaProjection consent.
  Both dictation and background capture use a 600 ms end-of-speech pause, with
  short-word and breath protection. That reduces the previous fixed capture
  wait; provider processing and audio delivery still need device measurements.
  Captured requests are bounded to 30 seconds including pre-roll and the final
  pause. Over-budget background speech is discarded until a full quiet pause;
  its remaining words cannot become a separate instruction.
- **Phone MCP** — connect an HTTPS MCP endpoint in Options. Discovered tools
  require a separate native approval for each call. An MCP hosted on the PC
  needs that PC online; independent hosted services can work with it off.
- **Local memory** — the phone encrypts a bounded conversation history with
  Android Keystore. Desktop mode keeps its separate host memory.

---

## Build it

After editing `orion_core/gui/screen_face.py`, `orion_core/emotion.py` or
`tools/phone_face_template.html`, run `python tools/export_phone_avatar.py` from
the repository root before building Android. It fills the Canvas 2-D face page
(`app/src/main/assets/orion_face.html`) with the desktop face's own tables;
`tests/test_phone_avatar_export.py` fails if the page drifts from the PC.

You need a recent Android Studio **or** a local JDK 17+ (Android Studio's bundled
`jbr` works) + Android SDK with platform 34.

### Option A — Android Studio (easiest)
1. `File ▸ Open…` and select this `android/` folder.
2. Studio installs the matching Gradle wrapper and SDK packages on first sync.
3. Press **Run ▶**, or `Build ▸ Build Bundle(s) / APK(s) ▸ Build APK(s)`
   and download the installable APK to the phone wirelessly.

### Option B — command line
From `android/`:
```bash
gradle wrapper --gradle-version 9.5.0   # once, to generate ./gradlew
./gradlew assembleDebug testDebugUnitTest lintDebug
```
The APK lands in `app/build/outputs/apk/`.

> The Gradle **wrapper jar** is intentionally not committed. Android Studio
> creates it on import; on the command line `gradle wrapper` generates it. AGP 9
> needs Gradle 9, so pin `--gradle-version 9.5.0` (an older system Gradle would
> otherwise generate a wrapper that cannot load the plugin).

Versions used: AGP 9.3.0, Gradle 9.5.0, Kotlin 2.2.10, compileSdk 34, minSdk 26.

### Release build and signing
`assembleRelease` shrinks the app with R8 and resource shrinking. It is signed only when a release key is
found outside the repository:

- `~/.orion/android/keystore.properties` (or the file named by
  `ORION_SIGNING_PROPERTIES`) with `storeFile`, `storePassword`, `keyAlias` and
  `keyPassword`; a relative `storeFile` is resolved next to that file.
- Or the environment variables `ORION_KEYSTORE_FILE`, `ORION_KEYSTORE_PASSWORD`,
  `ORION_KEY_ALIAS` and `ORION_KEY_PASSWORD`, which override the file.

With neither, the build still succeeds and writes `app-release-unsigned.apk`.
Back the key up: Android requires a compatible signing identity for in-place
updates. Changing that identity needs a separately planned migration; app-private
credentials and conversation data must not be deleted to work around an update error.

The owner's distributed 2.4–3.14 installers share this PC's original Android
debug-keystore certificate. Keep that signing identity for in-place updates;
the separately configured release keystore has a different certificate and
cannot update those installs without a deliberate signing-key migration.

For the owner's existing installation, use the **debug-compatible** 3.16 APK
built with this PC's original debug keystore. The separately signed **release**
3.16 APK is for fresh customer installations under the stable release identity.
Do not uninstall, clear app storage or replace Keystore to force that release
APK onto the owner's existing installation. A signing mismatch leaves the
existing app intact and needs investigation.

Build both variants with `assembleDebug assembleRelease`. Local checked copies
are named `releases/android/orion-companion-3.16-debug-compatible.apk` and
`releases/android/orion-companion-3.16-production.apk` at the repository root. Once wireless
ADB is connected and the signing identity is verified, update the owner with:
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The launcher icons are rendered by `tools/make_brand_icons.py` at the
repository root; edit and re-run that script rather than the image files.

---

## Set up ORION on the phone

1. Install version 3.16 with the same signing identity as the existing APK.
   An already provisioned phone retains its pairing, encrypted key and history;
   it does not need provisioning again. A new customer phone can supply its own key.
2. Keep the PC and its temporary HTTPS tunnel running for one provisioning
   request. The already paired phone authenticates to `/v1/device/ai-profile`.
   The PC owner must include that exact device ID in the private
   `config/mobile_ai_permissions.json` file. The existing Gemini key is
   transferred only to the authenticated native client, and sealed with
   Android Keystore. It is never compiled into the APK or handed to a WebView.
3. On the phone, open ORION and wait for its setup confirmation. You may now
   switch off the PC. Ask a question, dictate with Microphone or start Background
   listening in Options. Android will ask for microphone and notification
   permission. Enable Accessibility only for reviewed screen controls and
   Gmail sending; approve screen capture only when you intend to share it.

For a separate hosted text provider, open **Options → Text AI provider** and
select Groq, OpenRouter or Cloudflare Workers AI. Enter that provider's own API
key; Cloudflare also needs its account ID. Keys are sealed with Android
Keystore. Gemini remains the speech and transcription provider, so keep its
key if you want Charon voice or background listening. Calls, messages, e-mail,
Accessibility, sharing and other sensitive actions keep native approval
whichever text provider is selected. Version 3.2 opens a uniquely resolved
installed app or OsmAnd destination directly while ORION is active, so those
low-risk requests do not require a slide.

For informational replies, version 3.1 starts Charon after the first complete
phrase arrives and keeps speaking as the rest of the answer streams. Requests
that may control the phone wait for the verified action receipt before speech.
Version 3.2 keeps the chosen Gemini text model for long questions, prepares
Charon whilst a request runs, and shows local response timings in Options →
Performance diagnostics. Options → Capabilities & readiness describes what
each phone function actually finishes. A new phone can enter its own Gemini
key directly on first run; PC key provisioning is for the owner's devices.
Version 3.3 and later accept ordinary speech after Background listening is switched on.
Say “stop listening” or tap Stop in its notification to end it. Options →
Require wake word restores the earlier trigger. Version 3.3 also checks a
manually entered Gemini key and text model before replacing working settings.

The phone still needs an internet connection to reach Gemini. If the
temporary HTTPS address changes before provisioning, choose Options → Change
provisioning address and paste the new HTTPS URL. The manual key option in
phone Options can also initialise a new phone.

Google AI Studio applies separate limits to each model. The phone uses
`gemini-3.5-flash-lite` first for everyday speech recognition and replies.
It can fall back to another compatible Lite model when unavailable, but never
silently upgrades a chosen text model. The Charon voice setting now names its
actual route: **Voice: Live** or **Voice: Direct TTS**. Version 3.4 falls back
to the other route if the first is unavailable before any audio has played.
Partial speech is never replayed. Updates preserve the selected engine.
Options → Performance diagnostics reports the actual route; no current
preference or latency is inferred from an earlier host-only measurement.
Live remains available when the small TTS allowance is exhausted.
When a model is refused before it produces output, ORION can try another
compatible model without repeating any phone action. A `0/0` allowance in AI
Studio means that model is unavailable to the project; listing it does not
grant access.

## Optional desktop connection

### At home: the PC directly over Wi-Fi (no Tailscale)

With phone access on (`ORION_REMOTE_ACCESS=1`, or ask ORION to "turn on phone
access"), ORION also listens on the PC's home network address, port 8765, over
HTTPS (`orion_core/remote_lan.py`; `ORION_REMOTE_LAN=0` turns it off,
`ORION_REMOTE_LAN_PORT` moves it). Its certificate is ORION's own, so the phone
pins it:

1. On the PC open **SECURITY → PAIR PHONE**. It shows the pairing code and, for
   this Wi-Fi, the address (`https://192.168.x.y:8765`) and a 16-character
   certificate code.
2. In the app choose **Options → PC controls → Pair or change PC**, enter that
   address, the pairing code and the certificate code. (Desktop mode's
   **Change server** asks for the certificate code the same way.) The phone reads the certificate the
   address presents and refuses to send anything unless it matches the code.
3. A phone already paired through Tailscale learns the home address and its
   certificate automatically on its next connection: nothing to type.

The phone then tries the home address first and Tailscale (if set up) only when
the home address does not answer. If the PC's address changes, a paired phone on
the same Wi-Fi finds it again by its pinned certificate. Windows needs one
inbound rule, Private networks and the local subnet only; ORION adds it when it
runs as Administrator and otherwise prints the exact command in its log:

```powershell
netsh advfirewall firewall add rule name="ORION Phone LAN (HTTPS)" dir=in action=allow protocol=TCP localport=8765 profile=private remoteip=localsubnet
```

Set the home Wi-Fi to **Private** in Windows (Settings → Network). The link
answers only devices with private addresses, and on a network Windows treats as
Public the port stays closed.

### Away from home: Tailscale (optional)

1. Install Tailscale on the PC and phone, sign in to the same tailnet, and
   enable MagicDNS and HTTPS certificates for that tailnet.
2. On the PC run `deploy/start_tailscale_remote.ps1`. This binds ORION to
   `127.0.0.1:8000` and publishes its HTTPS endpoint through Tailscale Serve.
3. Enter the `https://<host>.<tailnet>.ts.net/` URL printed by
   `tailscale serve status` in the app. Port 443 and HTTPS are required.
4. Pair once with the code in ORION's Security Centre. The app retains the
   pairing across network changes.

If Serve is unavailable, run `deploy/start_orion.ps1 -Background` followed by
`deploy/start_quick_https_tunnel.ps1`. Enter the generated `https://…trycloudflare.com`
address and port **443** in the native app. Keep the PC and tunnel running.
The temporary address changes if the tunnel is restarted. This address is
needed for pairing, initial key provisioning, optional PC controls or Desktop mode.

The app probes its HTTPS endpoints, retries with capped exponential backoff,
and reconnects on Android network changes. The native **Listen** control starts
an ongoing microphone foreground service; say “ORION” before a request. Its
notification has a Stop action. The service can keep listening while the phone
is locked after being started from the visible app.

## Use ORION when you're away from home

Keep Tailscale connected on the phone for tailnet-only Serve access. If the
phone must connect without its VPN, explicitly run
`deploy/start_tailscale_remote.ps1 -Public` on the PC to use Tailscale Funnel.
Funnel exposes the pairing gateway to the public internet. The Android app
continues to use the HTTPS MagicDNS name; it never needs a raw `100.x` address.
See [the network audit](../docs/REMOTE_NETWORK_AUDIT_2026-09-28.md) for the
exact commands, firewall options and sensor requirements.

## Native phone actions (more than the HTML)

In the paired Android app, ask ORION to "text Mum saying I'm on my way",
"call Mum", or "open Calculator on my phone". Phone commands pass through the
`phone_device` capability to the native command channel. Opening one clearly
matched installed app or an OsmAnd destination runs directly while ORION is
active. Calls, messages, e-mail and screen controls still show the exact target
and payload for approval.

Under the hood, ORION's web UI drives the phone's own apps through a native
bridge (`window.OrionNative`) and through link schemes:

- **Call** — `OrionNative.call("+441234567890")` requests native approval, then
  initiates a call. A `tel:` link opens the dialer.
- **Text** — `OrionNative.sms("+44…", "on my way")` requests native approval,
  then sends the text. An `smsto:` link opens the messaging composer.
- **App** — `OrionNative.openApp("com.android.chrome")` requests approval, then
  launches the installed package.
- **E-mail** — `OrionNative.email("user@example.com", "subject", "body")` or `mailto:`.
- **Navigate** — `OrionNative.navigate("Birmingham New Street")` opens OsmAnd
  with the requested destination when it is installed and the region's map has
  been downloaded. Check the route in OsmAnd before setting off.
- **Share / open** — `OrionNative.share(text)`, `OrionNative.openUrl(url)`.

Calls and SMS require a deliberate native slide before execution. Calls ask for
`CALL_PHONE` after that slide. `SEND_SMS` is restricted by Android's installer;
where it has not been granted, the SMS command opens a draft in the installed
messaging app and does not claim it has sent. The complete SMS body appears in
the approval deck. Contact names need `READ_CONTACTS`, and ambiguous matches
are refused.
Remote command approvals expire after 80 seconds, close if the uplink drops,
and are checked again after Android permission prompts before an action runs.

## Device viewport and sentinel

- Choose **Options → Share phone screen** to start an Android screen-sharing consent session.
  The foreground notification has a Stop action. Frames are scaled to at most
  960 pixels on the long edge and sent as JPEG at no more than 2 frames/s.
- Choose **Options → Phone accessibility** to enable ORION's Accessibility Service in Android settings.
  ORION can request a bounded, redacted UI tree and can click or scroll only a
  node from a recent snapshot, after a separate local slide approval.
- Choose **Options → Start Sentinel overlay** to grant overlay access and enable the draggable crimson
  ring. Whilst another app is open, authorisation cards appear as temporary
  overlays; use the sentinel's persistent notification to stop it.
- The binary device uplink needs HTTPS/WSS and a paired phone.
- When the encrypted uplink is unavailable, capture skips JPEG compression;
  it resumes automatically when the connection returns.
- Reconfiguring to a different server clears the old pairing credentials and
  certificate pin. Pair again with the new server.

## Build a real APK without Android Studio

Push to GitHub and the **Android APK** workflow (`.github/workflows/android.yml`)
compiles an installable `app-debug.apk` and attaches it to the run. Download it
from **Actions ▸ Android APK ▸ (latest run) ▸ Artifacts**, copy it to the phone,
and install (allow "install unknown apps" once). Or trigger it by hand from
**Actions ▸ Android APK ▸ Run workflow**.

## Offline UK map (3.5)

Open **Options → Offline UK map**. The native map renders local Mapsforge road
data without a PC or network connection. Use **Map packs** to download on
unmetered Wi-Fi or import existing `.map` files. Great Britain and Ireland
including Northern Ireland are separate packs. Keep the app open during a
download; an interrupted transfer leaves the previous usable pack in place.
The map files are stored in the app's external files directory under `maps/`,
outside the APK. Removing the app removes these files too.

The map asks for foreground Location permission when opened. **Follow** keeps
the map centred on the phone's measured position. The marker shows available
accuracy, speed and heading; an old fix is marked stale. Map-only location updates
stop when the map is closed. Explicitly started navigation continues through a
location foreground service with a visible Stop notification. A coordinate given to the `map` phone action is a map
centre, never a claimed phone position.

This is an offline **vector road map**, not satellite imagery. Version 3.14
adds BRouter coordinate car routes, explicit planning between coordinates,
GPX import and saved places. BRouter and its separate road regions must be
installed; a Mapsforge map pack alone does not supply a routing graph. The
native screen still lacks offline address search. Version 3.16 adds supplied
spoken turns, conservative rerouting of the current BRouter car journey and
locked-screen GPS following. It requires an installed offline UK speech voice;
no cloud key is used for directions. The existing reviewed OsmAnd
hand-off remains available for its address search and guidance; download the
appropriate OsmAnd maps before travelling. Do not rely on the ORION map as the
only means of navigation. Country-wide detailed satellite imagery requires a
separate licence and substantially more storage; no online tile service is
silently cached.

The source packs are [Great Britain](https://download.mapsforge.org/maps/v5/europe/great-britain.map)
and [Ireland and Northern Ireland](https://download.mapsforge.org/maps/v5/europe/ireland-and-northern-ireland.map).
They are based on OpenStreetMap data (© OpenStreetMap contributors, ODbL).
The Mapsforge community server asks developers not to use it for mass
downloads; a customer release needs a separately hosted, updated and
integrity-checked pack feed.

See [native routing setup and exact limits](../docs/OFFLINE_CAR_ROUTES_2026-09-30.md)
and [the 3.14 integration record](../docs/PHONE_INTEGRATIONS_2026-09-30.md).

## Screen Face, home Wi-Fi and offline (3.17)

The orb and the Three.js face are replaced by the PC's Screen Face (see
"ORION's face" above). The app reaches the PC directly on the home Wi-Fi with a
pinned certificate, Tailscale only away from home. With no internet the phone
answers time, date and battery itself, keeps up to five requests (30 minutes,
memory only) and sends them when the connection returns, uses Android's
offline speech recogniser for the Mic when one is installed, and shows the
offline face; Options shows the PC and internet status. The full lists are in
[docs/OFFLINE_CAPABILITIES.md](../docs/OFFLINE_CAPABILITIES.md).

Install it like the earlier builds: copy `app/build/outputs/apk/debug/app-debug.apk`
to `releases/android/orion-companion.apk` and the phone downloads it from
`/download/orion-companion.apk` on the PC's uplink (the Tailscale Serve address,
`deploy/start_apk_download_tunnel.ps1`, or the home address, where the phone's
browser will warn about ORION's own certificate). The debug-compatible build
keeps the existing installation's signing identity, pairing and history.

## Phone controls (3.6)

The voice and terminal remain the main screen. The compact controls below the
waveform start or stop background listening, open the installed offline map,
and open phone actions directly. The listening control reflects the actual
foreground service state. Conversation cards are visually separated by role.

## Reliability fixes (3.7)

Terminal history is read away from the UI thread, and streamed replies update
the screen in batches. Background listening clears failed start requests and
waits out voice-provider quota cooldowns. Direct TTS remains in the selected
mode. Map pack validation, GPS freshness, Wi-Fi-only downloads, MCP cancellation
and duplicate remote command handling have also been tightened. See
[`docs/DEBUG_PASS_2026-09-29.md`](../docs/DEBUG_PASS_2026-09-29.md) for the
verification results and remaining live-device checks.

## Notes & next steps

- The app uses **HTTPS only**. At home it uses the PC's home Wi-Fi link; for
  cellular and external Wi-Fi, expose the loopback uplink through Tailscale
  Serve and use its MagicDNS address.
- Named addresses (Tailscale Serve, a tunnel, a reverse proxy) need a
  certificate Android trusts, exactly as before. The only self-signed
  certificate accepted is ORION's own on a private home-network address, and
  only the exact one pinned at pairing (`LanTrust.kt`). See
  [what works offline](../docs/OFFLINE_CAPABILITIES.md).
- The microphone permission is asked for the first time you tap the mic, not
  at launch, and again later if it was refused.
- App data is excluded from backups and phone-to-phone transfer, because it
  holds this phone's pairing credential. From version 1.5 the long-lived part
  of that credential is stored only as ciphertext under a key held in the
  phone's Android Keystore; a copy of the app's files is useless on its own.
  If the key is ever lost (a Keystore reset, or data restored onto another
  phone) the app forgets the credential and asks you to pair again, rather
  than presenting a half-read one. Earlier plain-text credentials are sealed
  the first time 1.5 starts.
- Background listening uses `AudioRecord`, a bounded speech segmenter, a
  microphone foreground notification and a renewable partial wake lock.
  WebSockets send keep-alive frames every 15 seconds and reconnect with
  exponential backoff and jitter after a network handoff.
- The PC API key never enters the APK. Access tokens are held in memory for
  speech; the long-lived pairing credential remains sealed in Android Keystore.
