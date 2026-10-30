# ORION Remote Access — Tailscale, the App, and What Works Away From Home

**Android 3.2:** ORION runs Gemini text and audio, Charon speech, wake
listening, reviewed native actions and configured HTTPS MCP tools on the phone.
It needs mobile data or Wi-Fi, but the PC may be switched off after one
authenticated AI-key provisioning request. Options → Change provisioning
address accepts a new temporary HTTPS URL. Desktop mode is an explicit option
for PC controls, memory and telemetry whilst the host is online. Phone actions
in desktop mode target the phone unless the instruction names the PC.
Long informational answers can begin speaking after the first complete phrase
while later text is still arriving. Phone actions wait for their receipt.

**Goal:** talk to and *command* ORION from your phone on mobile data, away from
home, without typing an IP or scanning a QR code every time — and without turning
his uplink into something the wider internet can poke at.

Current deployment and audit details are in
[REMOTE_NETWORK_AUDIT_2026-09-28.md](REMOTE_NETWORK_AUDIT_2026-09-28.md).
The gateway binds to `127.0.0.1:8000` by default and is reached through
Tailscale Serve when tailnet HTTPS is enabled. Run
`deploy/start_tailscale_remote.ps1` on the PC, then enter the HTTPS MagicDNS URL
in the Android app. The optional `-Public` switch uses Funnel and makes the
endpoint reachable without the phone's Tailscale VPN.

### HTTPS when Tailscale Serve is unavailable

If the tailnet administrator has not enabled Serve, run this in PowerShell
from the repository root:

```powershell
.\deploy\start_tailscale_remote.ps1 -DirectTailnetHttps -MagicDnsHost your-pc.your-tailnet.ts.net -Headless
```

This legacy route serves a self-signed certificate. Current Android builds
require Android-trusted HTTPS and will reject that certificate, so use
Tailscale Serve or a trusted HTTPS reverse proxy for the app. The command is
retained only for clients that explicitly install and trust its certificate.

If the phone times out, run the following once in an **elevated** PowerShell:

```powershell
.\deploy\configure_tailscale_firewall.ps1 -AllowDirectTailnetHttps
```

That rule allows inbound TCP 443 only on the Tailscale virtual adapter, at
this PC's tailnet address, from tailnet IPv4 peers. No router port forwarding
is needed. A mobile carrier's `100.x` address may still interfere with a
particular phone's route; a successful Wi-Fi test does not prove cellular
handoff. Test the phone on mobile data while Tailscale remains connected.

If ORION already has a pinned `config/remote_tls/cert.pem`, the gateway keeps
it instead of silently issuing a replacement. A certificate issued before
`ORION_REMOTE_DNS_NAME` was configured may lack the MagicDNS name; replacing
it requires deliberate re-trust on each phone.

### Temporary public HTTPS URL without a tailnet administrator

To share an Android update without publishing ORION's remote controls, use the
installer-only route instead. It serves the signed APK and returns 404 for all
other paths:

```powershell
.\deploy\start_apk_download_tunnel.ps1
```

Append `/download/orion-companion.apk` to the temporary HTTPS address printed
by Cloudflare. The address changes when the tunnel restarts. This route cannot
pair a phone or control the PC; it is solely for installing an update.

For a short connection test without Tailscale on the phone, the bundled
Cloudflare Quick Tunnel launcher can expose ORION's loopback uplink. From the
repository root, download and verify the pinned client once:

```powershell
.\deploy\download_cloudflared_quick.ps1
```

In **PowerShell window 1**, start the desktop ORION process with these settings:

```powershell
$env:ORION_REMOTE_ACCESS = '1'
$env:ORION_REMOTE_HOST = '127.0.0.1'
$env:ORION_REMOTE_PORT = '8000'
$env:ORION_REMOTE_TLS = '0'
$env:ORION_REMOTE_EXPOSURE = 'quick-tunnel'
$env:ORION_REMOTE_PAIR_ON_START = '1'
& .\.venv\Scripts\python.exe orion.py
```

In **PowerShell window 2**:

```powershell
.\deploy\start_quick_https_tunnel.ps1
```

In the Android app, open **Change server**, put the printed
`https://...trycloudflare.com:443` URL in **PC address**, and set **Port** to
`443`. Including `:443` in the address overrides any old port saved by a
previous app installation. Tap **Connect**. On the desktop, choose
**Controls (⚙) → REMOTE CONTROL → SECURITY → PAIR PHONE**, then enter the
fresh code in the app's pairing dialog. The URL provides the optional desktop
route and the one-time transfer of the approved Gemini key to Android Keystore.
Phone-mode replies and native actions then run without the PC uplink.

This
is a **public, temporary** address that changes on tunnel restart. ORION still
requires its one-time pairing code and authenticated tokens for private API
actions. Quick Tunnels are temporary and may buffer SSE. Desktop mode uses
bounded NDJSON text streaming on this route and polls approval cards whilst
the page is open. Live orb events work best on a private tailnet or named
tunnel. Keep both PowerShell windows open during the desktop-mode test.
`ORION_REMOTE_PAIR_ON_START=1` prints a fresh ten-minute pairing code in the
desktop log, including when this PC already has other paired devices.

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
3. Run `tailscale serve --bg --https=443 http://127.0.0.1:8000` on the PC.
   The app then reaches ORION at the HTTPS MagicDNS name from anywhere the phone
   has data.

ORION detects Tailscale automatically (`tailscale status --json`) and reports the
name to the phone at pairing time — you rarely need to type it yourself. If the
CLI is somewhere unusual, set `ORION_TAILSCALE_PATH` to the `tailscale` binary.

> **TLS for voice input:** Tailscale Serve supplies a trusted HTTPS certificate.
> Leave `ORION_REMOTE_TLS=0` for the loopback upstream; the phone sees HTTPS.
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

- **First run:** enter the PC's HTTPS Tailscale name or temporary tunnel URL on the setup screen,
  then **pair once** with the 6–8 character code ORION
  shows on the desktop. Pairing hands the phone a long-lived, revocable refresh
  token **and** the PC's HTTPS endpoint list.
- **Every launch after:** the app auto-connects — no IP, no QR, no code. It tries
  its known HTTPS endpoint and **reconnects** across Wi-Fi and mobile data.
  The endpoint list refreshes itself on each launch, so a changed Tailscale name is picked up without
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

- **Transport:** Serve terminates HTTPS within the tailnet. Funnel terminates
  HTTPS for public access when explicitly enabled. ORION remains on loopback.
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
| `ORION_REMOTE_TLS=0` | Loopback HTTP upstream; Tailscale Serve terminates HTTPS. |
| `ORION_REMOTE_HOST` | Bind address (default `127.0.0.1`). |
| `ORION_REMOTE_PORT` | Uplink port (default `8000`). |
| `ORION_REMOTE_EXPOSURE` | `serve`, `funnel` or `tailnet-direct` advertises only the HTTPS MagicDNS origin. |
| `ORION_REMOTE_DNS_NAME` | Full MagicDNS name included in a newly issued direct HTTPS certificate. |
| `ORION_TAILSCALE_PATH` | Explicit path to the `tailscale` binary if not found. |
