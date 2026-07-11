# Putting O.R.I.O.N. on Oracle Cloud (and on your phone)

This guide takes the **headless node** (`python orion.py --headless`) and runs it
on an Oracle Cloud VM, then connects your Android phone to it as a private,
installable app. Nothing here turns ORION into a Play-Store app — it stays
**private to you**, protected by a token, reachable over the internet or mobile
data.

> **What runs in the cloud:** ORION's *brain* — the language-model router
> (cloud API and/or local Ollama), identity, memory, conversation recall,
> knowledge, learning, and the remote uplink.
> **What does NOT run in the cloud:** voice, screen control, vision, Outlook —
> those need a real desktop and stay on your PC. The cloud node is the
> always-on conversational ORION you carry in your pocket.

---

## 0. Architecture at a glance

```
 Android phone (PWA)  ──HTTPS──▶  Caddy (443, TLS)  ──▶  ORION node (127.0.0.1:8765)
                                                          │
                                       Gemini cloud API ◀─┤ (MODE A, online)
                                       local Ollama     ◀─┘ (MODE B, offline-capable)
                                                          │
                                                    config/  (SQLite memory, token, keys)
```

---

## 1. Create the VM (Always Free tier)

1. Oracle Cloud console → **Compute → Instances → Create Instance**.
2. Image: **Canonical Ubuntu 22.04**. Shape: **VM.Standard.A1.Flex**
   (Ampere/arm64 — up to 4 OCPU / 24 GB RAM free) or an **E2.1.Micro** x86.
3. Add your SSH public key. Create.
4. Note the **public IP**.

## 2. Open the network

**Two firewalls must both allow traffic** — this is the usual gotcha.

*Oracle security list / NSG* (console → VCN → Security Lists → add Ingress):
- `0.0.0.0/0` TCP **80** and **443** (for Caddy/TLS).
- Do **not** expose 8765 publicly — ORION listens only on localhost behind Caddy.

*Host firewall* (on the VM):
```bash
sudo iptables -I INPUT -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save        # persist across reboots
```

## 3. Install ORION

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git curl
sudo useradd -m -d /opt/orion -s /bin/bash orion || true
sudo mkdir -p /opt/orion && sudo chown orion:orion /opt/orion

# Copy the project to /opt/orion (scp from your PC, or git clone your private repo)
#   scp -r "ORION Backup" ubuntu@<IP>:/tmp/orion && sudo mv /tmp/orion/* /opt/orion/

sudo -u orion bash -lc '
  cd /opt/orion
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r deploy/requirements-server.txt
'
```

### Give it a brain
Pick **either** path (or both — it prefers cloud when online, falls back to local):

- **Cloud (simplest):** put your Gemini key in `/opt/orion/config/api_keys.json`
  (same schema the desktop uses), or export `ORION_MODE=cloud` with the key.
- **Local & fully offline:** install Ollama on the VM and pull a small model:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5:3b        # 3B fits comfortably on the A1 free tier
  ```
  ORION auto-detects Ollama and runs in **MODE B** with no API cost.

## 4. Run it as a service

```bash
sudo cp /opt/orion/deploy/orion-node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-node
journalctl -u orion-node -f            # watch it boot; note the token line
```

The first line of the log prints where the access token lives:
`/opt/orion/config/remote_token.txt`. **Copy that token** — the phone needs it.

## 5. HTTPS (needed for the phone app)

The PWA service worker and secure token entry require `https://`. Easiest is
**Caddy**, which fetches a free auto-renewing certificate:

1. Point a hostname at the VM IP. No domain? Use a free one from
   [DuckDNS](https://www.duckdns.org) (e.g. `myorion.duckdns.org`).
2. Edit `deploy/Caddyfile`, replacing `orion.example.com` with your hostname.
3. Install and start Caddy:
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/caddy-stable-archive-keyring.gpg] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update && sudo apt install -y caddy
   sudo cp /opt/orion/deploy/Caddyfile /etc/caddy/Caddyfile
   sudo systemctl restart caddy
   ```
4. Make sure the node binds localhost only: the service file already sets
   `ORION_REMOTE_HOST=127.0.0.1`.

> **No domain at all?** You can still reach it at `http://<public-IP>:8765` by
> opening 8765 in both firewalls, but the browser will refuse to *install* the
> PWA (and the token travels unencrypted). Fine for a quick test, not for daily
> use. A one-command alternative is a **Cloudflare Tunnel** (`cloudflared`) which
> gives you HTTPS without opening any inbound port.

## 6. Put ORION on your Android phone

1. On the phone (mobile data or Wi-Fi), open **Chrome** and go to
   `https://myorion.duckdns.org`.
2. It prompts for the **access token** — paste the one from step 4.
3. Chrome menu → **Add to Home screen** → *Install*. It now behaves like a
   native app (own icon, full-screen, no address bar) and opens instantly.
4. Talk to ORION. Every exchange is written into the same memory the desktop
   uses, so context follows you.

> iPhone: Safari → Share → *Add to Home Screen* works the same way.

## 7. Keeping it private & safe

- The token is the only key — treat it like a password. Rotate it by deleting
  `config/remote_token.txt` and restarting; re-enter the new one on the phone.
- Rate limiting (30 req/min per client) and hardening headers are built in.
- Keep 8765 closed to the world; only 443 (Caddy) is public.
- To share ORION with someone later, you'd add per-user tokens — noted as a
  follow-up in the patch notes, not built yet (it stays single-user for now).

## 8. Optional: keep desktop and cloud in sync

Point both nodes at the **same `config/` directory** (e.g. an rclone-synced
volume or a small object-storage mount) and they share one memory + token.
Otherwise each node keeps its own memory and they diverge — which is fine if you
just want a standalone pocket ORION.

---

### Quick local test before you deploy
On your PC you can dry-run the exact cloud node:
```powershell
$env:ORION_HEADLESS = "1"
python orion.py --headless
# then open http://localhost:8765 and paste the token from config/remote_token.txt
```
