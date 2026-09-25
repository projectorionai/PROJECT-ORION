# ORION Remote Access — Tailscale, the App, and What Works Away From Home

**Goal:** talk to and *command* ORION from your phone on mobile data, away from
home, without typing an IP or scanning a QR code every time — and without turning
his uplink into something the wider internet can poke at.

There are two independent parts, and it helps to keep them separate:

1. **Reachability** — getting the phone to actually *reach* the PC across the
   internet. This is a networking fact, not an app setting: your PC sits behind
   your router's NAT, so its LAN address (`192.168.x`) only works on your home
   Wi-Fi. You need a *stable, reachable* address. We use **Tailscale**.
2. **Capability** — how much ORION will *do* when you command him remotely. This
   is the full-parity gate: read/benign runs instantly, irreversible/outward
   actions ask you to tap **Approve** on the phone, and code-execution tools are
   refused outright.

---

## 1. One-time setup: Tailscale (≈5 minutes)

Tailscale is a free, encrypted mesh VPN (WireGuard under the hood). Both devices
dial *out* to it, so there are **no ports to forward** and it works even behind
carrier-grade NAT.

1. **On the PC:** install Tailscale (<https://tailscale.com/download>), sign in
   (Google/Microsoft/GitHub — a personal account is free). Note the PC's
   **Tailscale name**, e.g. `orion-pc.tailXXXX.ts.net` (see it with
   `tailscale status`, or in the Tailscale admin console).
2. **On the phone:** install the Tailscale app from the Play Store, sign in with
   the **same account**. Leave it running (it's a lightweight always-on VPN).
3. That's it. The PC is now reachable at that Tailscale name from anywhere the
   phone has data — encrypted end-to-end.

ORION detects Tailscale automatically (`tailscale status --json`) and reports the
name to the phone at pairing time — you rarely need to type it yourself. If the
CLI is somewhere unusual, set `ORION_TAILSCALE_PATH` to the `tailscale` binary.

> **TLS for voice input:** phone voice **input** needs a secure context. Set
> `ORION_REMOTE_TLS=1` on the PC so the uplink serves HTTPS; you'll accept the
> self-signed certificate once per device. Voice *replies* work either way.
>
> Tap the 🎙 button and speak. In a desktop/mobile Chrome browser this uses the
> browser's own speech recogniser. **Inside the Android app** — whose WebView has
> no in-browser recogniser — the button instead records a short clip and ORION
> transcribes it on the PC (offline, via Whisper/Vosk) over `POST /api/transcribe`,
> then answers as if you'd typed it. Either way you only need HTTPS on and the
> microphone allowed for ORION. No app rebuild is required — the phone loads the
> voice code fresh from the PC, so just restart ORION and reopen the app.

---

## 2. The app: pair once, then it just connects

The Android app (`android/`, build it in Android Studio → *Build ▸ Build APK(s)*)
is a native shell around ORION's uplink. The connection flow is now frictionless:

- **First run:** enter the PC's **Tailscale name** on the setup screen (a LAN IP
  also works but only at home), then **pair once** with the 6–8 char code ORION
  shows on the desktop. Pairing hands the phone a long-lived, revocable refresh
  token **and** the PC's endpoint list (Tailscale + LAN).
- **Every launch after:** the app auto-connects — no IP, no QR, no code. It tries
  its known endpoints in order (Tailscale first, so it works home *and* away) and
  **fails over** automatically: if the LAN address doesn't answer (you're out),
  it silently rolls to the Tailscale name. The endpoint list refreshes itself on
  each launch, so a changed Tailscale name or LAN IP is picked up without
  re-pairing.
- **Revoking a device** on the desktop kills both its tokens within ~15 minutes.

### Mobile-data (lite) mode

On a **metered** connection the app appends `?lite=1` automatically (it also
honours the browser's `Save-Data` hint). Lite mode:

- **drops the Three.js voxel face** — its 3-D modules are pulled from a CDN and
  are by far the biggest cellular cost — and shows a self-contained CSS orb
  instead;
- rides on **gzip** compression, which the uplink now applies to all text/JSON.

Force it on/off with the `lite` preference (`auto` | `on` | `off`).

---

## 3. What you can command remotely (capability parity)

Remote chat now runs ORION's **full brain with tools**, through a capability gate.
Every tool is one of three tiers:

| Tier | Behaviour | Examples |
|---|---|---|
| **ALLOW** | Runs instantly | ask/reason, recall memory, briefings, web search, **read** your email, **list** Notion tasks, read a web page, read the screen, find files, resource/diagnostics status |
| **CONFIRM** | Parks and asks you to tap **Approve** on the phone before it runs | **send** an email / WhatsApp / Telegram, create/edit Notion, desktop control (mouse/keyboard/windows), drive the browser co-pilot (click/type/submit), file writes/deletes, organise files, clipboard writes, shutdown/restart ORION, peripheral power, backups, launch games/media |
| **FORBID** | Refused outright — desktop-only, forever | `forge`, `dev_workbench`, `execute_plan`, `self_repair`, `process_governor`, `security_watch`, `interface_control` |

**Why the FORBID list can't be run remotely (you asked for the list):** these
either **execute or rewrite code on your PC** (`forge`, `dev_workbench`,
`execute_plan`, `self_repair`), **kill arbitrary processes** (`process_governor`),
**disable ORION's own security guard** (`security_watch`), or **reconfigure his
desktop UI** (`interface_control`) — pointless from a phone and dangerous as an
internet-reachable surface. Exposing any of them would make the uplink a
remote-code-execution target no matter how well authenticated. Everything *else*
is available; irreversible/outward actions just need your tap.

**The confirmation tap** is the desktop's on-screen approval gate, moved to where
you actually are. When ORION wants to do something in the CONFIRM tier, a card
appears in the app: *"Approve: send an email?"* with **Approve** / **Deny**. The
action carries a single-use, time-boxed (3 min) token delivered over the
authenticated event stream; nothing fires until you tap.

### Read vs. write is action-aware

A tool that can both read and write is judged on the *action*: `read my email`
runs instantly (ALLOW), but `send that reply` prompts a tap (CONFIRM). Same for
Notion (list vs. create), the browser co-pilot (read/scroll/screenshot vs.
click/type/submit), and the clipboard (read vs. write).

---

## 4. Security notes

- **Transport:** all remote traffic is inside the Tailscale WireGuard tunnel
  (encrypted, device-authenticated). Add `ORION_REMOTE_TLS=1` for HTTPS on top.
- **Auth:** one-time pairing code → long-lived **revocable** refresh token
  (stored only hashed on the PC) → 15-minute HMAC access tokens. No static token.
- **The agent-task API stays locked down.** The separate `/v1/agent/tasks` queue
  keeps its original tiny read-only whitelist — full parity lives *only* on the
  authenticated conversational path with the confirm gate.
- **Fail-safe default:** any tool not explicitly classified is treated as
  CONFIRM, so a new capability can never run remotely without you seeing it.

---

## Environment flags

| Flag | Effect |
|---|---|
| `ORION_REMOTE_ACCESS=0` | Keep the session desktop-only (no uplink). |
| `ORION_REMOTE_TLS=1` | Serve HTTPS (needed for phone voice input). |
| `ORION_REMOTE_PORT` | Uplink port (default `8765`). |
| `ORION_TAILSCALE_PATH` | Explicit path to the `tailscale` binary if not found. |
