# ORION Companion — native Android app

A native Kotlin/Android app that puts **ORION on your phone**: his crimson voxel
orb/face, his voice, and his memory — everything the desktop already serves over
the local-network uplink, wrapped in a first-class Android shell (native setup,
microphone permission, self-signed-TLS handling, keep-awake, immersive
full-screen).

It is a **WebView shell** around the ORION uplink rather than a from-scratch
reimplementation, and that is deliberate: the orb, the live emotion mirror, the
chat, the spoken replies and the episodic-memory sync are already produced by
the uplink (`orion_core/remote.py`) and rendered identically to the desktop. The
app makes that a real installed app you launch from your home screen.

---

## What you get

- **His face** — the same Three.js voxel orb, mirrored live over Server-Sent
  Events (state, amplitude, emotion) so the phone orb moves exactly as the
  desktop one does.
- **His voice** — replies are spoken aloud (with the orb lip-syncing).
- **His memory** — every phone turn is logged into the *same* episodic memory as
  your desktop conversations, and answers are memory- and identity-grounded.
- **Talk to him** — the microphone button uses on-device speech (needs HTTPS —
  see below); text chat and voice *output* work over plain HTTP too.

---

## Build it

You need a recent Android Studio **or** a local JDK 17+ (Android Studio's bundled
`jbr` works) + Android SDK with platform 34.

### Option A — Android Studio (easiest)
1. `File ▸ Open…` and select this `android/` folder.
2. Studio installs the matching Gradle wrapper and SDK packages on first sync.
3. Plug in your phone (USB debugging on) and press **Run ▶**, or
   `Build ▸ Build Bundle(s) / APK(s) ▸ Build APK(s)` for an installable file.

### Option B — command line
From `android/`:
```bash
gradle wrapper --gradle-version 9.5.0   # once, to generate ./gradlew
./gradlew assembleDebug                 # or assembleRelease
```
The APK lands in `app/build/outputs/apk/`.

> The Gradle **wrapper jar** is intentionally not committed. Android Studio
> creates it on import; on the command line `gradle wrapper` generates it. AGP 9
> needs Gradle 9, so pin `--gradle-version 9.5.0` (an older system Gradle would
> otherwise generate a wrapper that cannot load the plugin).

Versions used: AGP 9.3.0, Gradle 9.5.0, Kotlin 2.2.10, compileSdk 34, minSdk 26.

---

## Connect it to ORION

1. On the PC, make sure ORION is running with **phone access on**. It is off by
   default: ask ORION to *"turn on phone access"*, or start him with
   `ORION_REMOTE_ACCESS=1`. He then prints a line like
   *"on your phone (same Wi-Fi) open http://192.168.1.182:8765"*.
2. **For voice input**, start ORION with `ORION_REMOTE_TLS=1` so the uplink
   serves HTTPS — browsers (and Android WebView) only allow the microphone on a
   secure origin. You'll be asked to trust his self-signed certificate once.
3. Open the app, enter the PC address (e.g. `192.168.1.182`) and port (`8765`),
   leave **Use HTTPS** on, and tap **Connect**.
   A full address pasted into **PC address** (`https://192.168.1.182:8765/`)
   is split into host, port and scheme for you.
4. Pair once: ORION shows a short code in his Security Centre (**Pair phone**);
   the app asks for it in a *Pair this phone* dialog. The pairing is kept by the
   app and shared across all of ORION's addresses (LAN and Tailscale), so
   switching between them never asks again.

On launch the app checks every address it knows in parallel (a couple of
seconds at most) and opens the best one that answers, showing *Connecting to
ORION…* meanwhile; if none answer it says which addresses it tried and why. It
re-checks when you come back to it, so walking out of your Wi-Fi moves it onto
Tailscale by itself. If it can't connect, check ORION is running with phone
access on, the address is right, and the firewall allows TCP 8765 (ORION tries
to open this automatically on private/domain networks).

---

---

## Use ORION when you're away from home

The app reaches ORION over the network, so "away from home" just needs a network
path to the PC that works from anywhere. The clean, secure, zero-config way:

1. Install **Tailscale** (free) on the PC and on the phone, signed into the same
   account. It builds a private encrypted mesh between your own devices — no
   port-forwarding, no exposing ORION to the public internet.
2. On the PC, note its **Tailscale IP** (looks like `100.x.y.z`).
3. In the app's setup screen, put that `100.x.y.z` in **PC address**. Leave the
   PC's LAN IP for when you're at home, or just always use the Tailscale IP —
   it works both at home and away.

Now, as long as ORION is running on the PC, you can open the app from anywhere,
see his face, talk to him, and have him help with news, navigation, messaging
and calls. (Alternatives to Tailscale: a Cloudflare Tunnel or any reverse proxy
that gives ORION a stable hostname — put that hostname in **PC address**.)

## Native phone actions (more than the HTML)

This is wired end to end: just **ask** ORION ("text Mum I'm on my way", "navigate
to New Street", "call the dentist"). On the PC he runs the `phone_action` tool,
which pushes an action to your phone over the live event stream; the app opens
the dialer / messaging / Maps / mail **pre-filled** and you tap to confirm. In a
plain browser (no app) the same action appears as a tappable link instead.

Under the hood, ORION's web UI drives the phone's own apps through a native
bridge (`window.OrionNative`) and through link schemes:

- **Call** — `OrionNative.call("+441234567890")` or a `tel:` link → opens the
  dialer pre-filled (you press call).
- **Text** — `OrionNative.sms("+44…", "on my way")` or an `smsto:` link.
- **E-mail** — `OrionNative.email("user@example.com", "subject", "body")` or `mailto:`.
- **Navigate** — `OrionNative.navigate("Birmingham New Street")` or a `geo:` /
  `google.navigation:` link → turn-by-turn in Maps.
- **Share / open** — `OrionNative.share(text)`, `OrionNative.openUrl(url)`.

Nothing is dialled or sent without your final tap (ORION uses `ACTION_DIAL` /
`ACTION_SENDTO`, never a silent send), so the app needs **no** call/SMS
permissions. News articles and other `http(s)` links open **inside** the app.

## Build a real APK without Android Studio

Push to GitHub and the **Android APK** workflow (`.github/workflows/android.yml`)
compiles an installable `app-debug.apk` and attaches it to the run. Download it
from **Actions ▸ Android APK ▸ (latest run) ▸ Artifacts**, copy it to the phone,
and install (allow "install unknown apps" once). Or trigger it by hand from
**Actions ▸ Android APK ▸ Run workflow**.

## Notes & next steps

- **Cleartext HTTP** is allowed (LAN + Tailscale) so text + voice output work
  without TLS; enable TLS for in-page voice input.
- The self-signed certificate is accepted **only** for ORION's own hosts (the
  one you configured plus the endpoints he reported), in `onReceivedSslError`.
- The microphone permission is asked for the first time you tap the mic, not
  at launch, and again later if it was refused.
- App data is excluded from backups and phone-to-phone transfer, because it
  holds this phone's pairing credential.
- Possible future native additions (not yet built): a foreground service +
  push notifications for proactive messages while the app is closed, and native
  Android `SpeechRecognizer` so voice input works over plain HTTP.
