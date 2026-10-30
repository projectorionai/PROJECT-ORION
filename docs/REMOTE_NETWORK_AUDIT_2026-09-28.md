# Remote network and mobile audit — 28 September 2026

## Active architecture

The remote gateway is `aiohttp` in `orion_core/remote.py`. There is no active
FastAPI `main.py` or Uvicorn entry point. `orion.py` starts the desktop or
headless application and constructs the gateway. Replacing this with a second
server would duplicate pairing, memory, approvals and tool capability policy.

## Findings and corrections

| Finding | Location | Correction |
|---|---|---|
| Public bind (`0.0.0.0:8765`) was the gateway default. It exposed a cleartext or self-signed endpoint directly and bypassed Serve TLS. | `orion_core/remote.py` gateway initialisation | Default is `127.0.0.1:8000`; Serve or Funnel supplies HTTPS on 443. |
| Endpoint discovery advertised raw `100.x` and LAN IPs. Those can collide with cellular CGNAT and bypass Serve. | `orion_core/remote_endpoints.py` | Loopback mode advertises only the HTTPS MagicDNS name on 443. Direct bind remains an explicit legacy mode. |
| Android remembered legacy ports and cleartext endpoints. | `Prefs.kt`, `SetupActivity.kt`, Android network configuration | `.ts.net` endpoints migrate to HTTPS 443; cleartext and raw `100.x` candidates are rejected. |
| The WebView probed once and only checked again on resume. Its main-frame failure had no automatic retry. | `MainActivity.kt` | A network callback now reconnects on handoff; failures use capped exponential backoff with jitter. |
| The native viewport socket had 30-second pings. | `OrionFrameTransport.kt`, `remote_mobile_ws.py` | Both peers use 15-second WebSocket heartbeats. |
| The remote HTML's pairing, token, health, history, tasks and transcription requests had uneven timeouts. SSE reconnect already backed off; chat SSE already streamed model deltas. | `orion_core/remote.py` embedded page | A bounded request helper now covers these paths and retries safe GET requests with jitter. Chat POSTs are timed while waiting for response headers but are never automatically replayed because a tool action may have run. |
| First access to the Three.js face walked files and imported the GUI in an async handler. | `orion_core/remote.py` face routes | These calls now run in a worker thread. Other blocking operations, including transcription and endpoint detection, were already offloaded. |
| No authenticated host sensor endpoint or mobile chart existed. | Gateway and mobile page | `/api/hardware` returns CPU, GPU, VRAM and temperature readings, with null for unavailable temperature sensors. |

The target CPU and GPU are read from actual sensors. On Windows, CPU temperature
needs LibreHardwareMonitor or OpenHardwareMonitor to publish a CPU package sensor
through WMI. NVIDIA GPU load, VRAM and temperature use NVML or `nvidia-smi`.
The interface never invents a temperature when the sensor is absent.

The literal-IP scan of runtime code found these further addresses. They are
separate from the phone uplink: `1.1.1.1`, `8.8.8.8` and `208.67.222.222` in
`connectivity.py` are outbound internet probes; `8.8.8.8` in
`remote_endpoints.py` and `10.255.255.255` in `remote.py` are socket-based
local interface probes; `127.0.0.1:11434` and `127.0.0.1:1234` are local
Ollama and LM Studio clients; `127.0.0.1:8790` is the separate telephony
service; and `127.0.0.1` in `browser_copilot.py` is the local browser debugger.
The disabled Vast.ai provider still carries `http://0.0.0.0:8000/v1` as a
placeholder: that is not a usable client destination and needs an explicit
`ORION_VASTAI_BASE_URL` before the provider can run. The old `100.64.0.0/10`
literal remains only in direct-bind firewall compatibility code, which
loopback Serve mode skips. Example IPs in tests and comments are not active
destinations.

## Deployment

On this PC, Tailscale reports `Running` with MagicDNS enabled and the host name
`your-pc.your-tailnet.ts.net`. `tailscale serve status` reports no Serve
configuration, neither port 8000 nor 8765 was listening during the audit, and
the Android peer was offline. The HTTPS endpoint therefore still needs the
deployment command below and a running ORION process before a cellular handoff
can be checked on the phone.
The tailnet also advertises a separate `svc:orion` service on TCP 8765;
the new Serve configuration does not change that service. Use the host name
printed by `tailscale serve status` for this mobile client.

On the PC, sign in to Tailscale and enable MagicDNS and HTTPS certificates for
the tailnet. From the repository root, in PowerShell:

```powershell
$env:ORION_REMOTE_ACCESS = '1'
$env:ORION_REMOTE_HOST = '127.0.0.1'
$env:ORION_REMOTE_PORT = '8000'
$env:ORION_REMOTE_TLS = '0'
tailscale serve --bg --https=443 http://127.0.0.1:8000
tailscale serve status
& .\.venv\Scripts\python.exe orion.py
```

`tailscale serve --bg 8000` is the equivalent shorthand for the default HTTPS
443 proxy to the local service on port 8000.

The phone uses the `https://<host>.<tailnet>.ts.net/` URL printed by Serve.
Keep Tailscale connected on the phone. MagicDNS and Serve fix the URL and TLS,
but Serve still routes through the tailnet's `100.64.0.0/10` address space.
If a cellular route or the phone's VPN interferes with that path, explicitly
choose public Funnel instead; Funnel reaches the HTTPS name without the phone's
Tailscale VPN:

```powershell
tailscale funnel --bg --https=443 http://127.0.0.1:8000
tailscale funnel status
```

`tailscale funnel --bg 443` means **forward to local port 443** and is incorrect
for this gateway. Funnel makes the HTTPS endpoint public; the pairing and
capability gates still apply. `deploy/start_tailscale_remote.ps1` performs the
Serve or Funnel setup and starts ORION. Tailscale's own proxy accepts the
external connection, so the gateway's loopback TCP port needs no inbound rule.
Serve and Funnel cannot both own HTTPS 443 on the same node; the last command
sets whether that port is private or public.

`deploy/configure_tailscale_firewall.ps1` classifies the Tailscale adapter as
Private. Its optional `-AllowDirectPeerUdp` switch adds inbound UDP 41641 for
direct Tailscale peers. Its optional `-AllowDirectLanTcp` switch adds a
Private-profile, local-subnet-only inbound TCP 8000 rule for an explicitly
configured direct LAN bind. Neither rule is needed for loopback Serve.

## Audio and platform limits

The Android Listen button starts a user-visible microphone foreground service.
It sends bounded PCM utterances over a paired WebSocket, requires the spoken
word “ORION”, and speaks replies through Android text-to-speech. The service
can continue while the screen is locked after the user starts it from the
visible app. Android may still stop it under battery or device policy. No iOS
application exists in this repository, so an iOS background task was not
implemented. The Android service uses energy-based utterance detection rather
than a dedicated on-device wake-word model; recognition and response depend on
the host's offline speech engine.

The existing remote capability gate still refuses arbitrary code execution and
self-modification from a paired phone. The mobile page can render a typed
`script_result` card, but a permitted, named script catalogue and its approval
policy would be needed before script execution could match the PC. This is an
intentional remaining parity gap, alongside the absent iOS application.

## Verification on real devices

1. Pair through the Serve URL, then test a text turn and the hardware cards.
2. Start Listen while the app is visible, lock the phone, and say “ORION” and a
   short request; check the foreground notification and spoken reply.
3. Switch between home Wi-Fi and mobile data while the page and Listen service
   are running. Check reconnection without changing the address.
4. Repeat with Funnel only if public exposure is intended. Revoke a phone and
   confirm its existing sockets close after the next authentication check.
