# ORION on a Hostinger VPS

This is a **paid VPS** deployment for the headless conversation and remote
gateway. It does not run the desktop UI, Windows controls, camera, microphone,
scheduled desktop plugins or an audio device. Phone actions still run on the
phone. The separate Twilio bridge is present but cannot yet provide a spoken
conversation in this standalone deployment.

The setup script supports Ubuntu 24.04 and Debian 12. It installs dependencies,
creates the service account, writes a starter environment file, installs the
systemd units, adds HTTP/HTTPS rules to UFW and configures Caddy. It preserves
existing UFW rules and never enables an inactive firewall. Review SSH and
firewall access before enabling UFW yourself. The script does not start ORION
or create any provider or Twilio credentials.

## 1. Copy the source

Copy the project to `/opt/orion` on the VPS. Exclude the local `config/`
directory, `.env`, databases, logs, recordings, virtual environments and
build outputs. Check the uploaded files before running the setup script.

```bash
rsync -av --exclude .venv --exclude .git --exclude config \
  --exclude .env --exclude dist --exclude build --exclude logs \
  ./ root@YOUR_VPS_IP:/opt/orion/
ssh root@YOUR_VPS_IP
bash /opt/orion/deploy/hostinger_setup.sh orion.yourdomain.com
```

Point the hostname's DNS A record at the VPS. In the provider firewall, allow
TCP 80 and 443. Keep 8765 and 8790 closed: both services bind to loopback.

## 2. Configure and start

Edit `/opt/orion/config/orion.env`. The setup script writes
`ORION_REMOTE_PUBLIC_URL=https://orion.yourdomain.com`; keep it aligned with
the Caddy hostname. Add a hosted provider key such as `ORION_GEMINI_API_KEY` if
you want cloud model replies. Alternatively, configure a local model separately.
The environment file is readable only by the `orion` account.

```bash
nano /opt/orion/config/orion.env
systemctl start orion caddy
systemctl status orion caddy --no-pager
curl --fail --silent --show-error http://127.0.0.1:8765/api/health
curl --fail --silent --show-error https://orion.yourdomain.com/api/health
journalctl -u orion -n 80 --no-pager
```

The first run shows a one-time pairing code in the ORION journal. Keep journal
access private while the code is valid. Pair the Android client with the stable
HTTPS hostname. A successful health check proves the uplink is responding; a
real paired conversation still needs to be checked on the intended phone.

## 3. Twilio bridge status

The standalone `orion-telephony.service` starts `TelephonyServer` without an
audio engine. It can validate signed requests and accept a media WebSocket,
but it sends no conversational audio. **Do not route a live Twilio number to
it yet.** The bridge needs an audio engine connected to the headless node and
an end-to-end call test before it can be used for calls.

If the bridge is developed further, `TWILIO_AUTH_TOKEN` and
`ORION_PUBLIC_HOST` will be required. The public host is the hostname alone,
without `https://`. Caddy routes its `/media` WebSocket to port 8790.

## 4. Updating and data

Stop the node before replacing source files or databases, then repeat the
source copy and setup steps. The setup script retains an existing
`config/orion.env`. If the hostname changes, update
`ORION_REMOTE_PUBLIC_URL` there and re-run setup with the new hostname.

Keep the cloud node's SQLite databases separate from the desktop's live files.
To transfer selected memory, create a consistent SQLite backup first and
restore it while the node is stopped. Do not share a live `config/` directory
over a network filesystem.

The `orion.service` unit permits writes to `config/`, `research/`,
`reports/`, `logs/` and `conversations/`. It limits the headless process to
4 GB RAM and two CPU cores. Set a lower memory limit for a smaller VPS.
