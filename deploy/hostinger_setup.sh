#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# O.R.I.O.N. — Hostinger VPS setup (Ubuntu 24.04 / Debian 12)
#
# Run as root on a fresh VPS:
#     bash deploy/hostinger_setup.sh yourdomain.example.com
#
# Idempotent: safe to run again after a failure or an upgrade. Every step
# checks before it acts, so a second run repairs rather than duplicates.
#
# What it does NOT do, on purpose:
#   * It never writes credentials. You fill config/orion.env yourself, so no
#     key is ever in shell history or in this file.
#   * It does not open port 8765 or 8790 to the world. Those stay on
#     loopback; Caddy terminates TLS and proxies to them.
#   * It does not start ORION. Review the config first, then
#     `systemctl start orion`.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DOMAIN="${1:-}"
ORION_USER="orion"
ORION_HOME="/opt/orion"
PYTHON_MIN="3.11"

say()  { printf '\n\033[1;31m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ! \033[0m%s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  echo "Run this as root: sudo bash $0 ${DOMAIN}" >&2
  exit 1
fi

# ── 1. packages ──────────────────────────────────────────────────────────────
say "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3 python3-venv python3-dev build-essential \
  git curl ca-certificates ufw rsync \
  libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1 \
  libasound2t64 alsa-utils
# libgl1/libxkbcommon/libfontconfig: PyQt6 links against them even under
# QT_QPA_PLATFORM=offscreen. Without them `import PyQt6.QtCore` fails with a
# missing shared object and the unit restart-loops with a cryptic message.
# libasound2t64 is the Ubuntu 24.04 name; on Debian 12 it is libasound2.
ok "packages installed"

# ── 2. the orion user ────────────────────────────────────────────────────────
say "Creating the service account"
if id -u "$ORION_USER" >/dev/null 2>&1; then
  ok "user ${ORION_USER} already exists"
else
  # No login shell and no password: this account exists to run one process.
  adduser --system --group --home "$ORION_HOME" --shell /usr/sbin/nologin \
          "$ORION_USER"
  ok "created ${ORION_USER}"
fi

# ── 3. silent audio ──────────────────────────────────────────────────────────
say "Configuring a null audio device"
# A VPS has no sound card. ORION's audio threads open a device at start-up and
# throw "ALSA: No such device" without one — which is fatal to the thread and
# leaves him unable to speak on a call. snd-dummy provides a real ALSA device
# that goes nowhere, which is exactly what a headless node wants.
if ! grep -q '^snd-dummy' /etc/modules 2>/dev/null; then
  echo 'snd-dummy' >> /etc/modules
fi
modprobe snd-dummy 2>/dev/null || warn "snd-dummy unavailable (container?) — using the ALSA null plugin instead"
cat > /etc/asound.conf <<'ASOUND'
# ORION headless node: send everything to the null plugin. Capture and
# playback both succeed and move no audio, so the audio threads initialise
# cleanly instead of raising on a missing device.
pcm.!default { type null }
ctl.!default { type null }
ASOUND
ok "audio will initialise without hardware"

# ── 4. the code ──────────────────────────────────────────────────────────────
say "Preparing ${ORION_HOME}"
mkdir -p "$ORION_HOME"/{config,research/dossiers,reports,logs}
if [[ ! -f "$ORION_HOME/orion.py" ]]; then
  warn "No ORION source at ${ORION_HOME}."
  warn "Copy it up, then run this script again:"
  warn "  rsync -av --exclude .venv --exclude config ./ root@HOST:${ORION_HOME}/"
fi
chown -R "$ORION_USER:$ORION_USER" "$ORION_HOME"
ok "directories ready"

# ── 5. the virtualenv ────────────────────────────────────────────────────────
say "Building the virtualenv"
if [[ ! -x "$ORION_HOME/.venv/bin/python" ]]; then
  sudo -u "$ORION_USER" python3 -m venv "$ORION_HOME/.venv"
fi
sudo -u "$ORION_USER" "$ORION_HOME/.venv/bin/pip" install --quiet --upgrade pip
if [[ -f "$ORION_HOME/deploy/requirements-server.txt" ]]; then
  sudo -u "$ORION_USER" "$ORION_HOME/.venv/bin/pip" install --quiet \
    -r "$ORION_HOME/deploy/requirements-server.txt"
  ok "server requirements installed"
else
  warn "deploy/requirements-server.txt not found — skipping"
fi

# ── 6. the environment file ──────────────────────────────────────────────────
say "Preparing config/orion.env"
ENV_FILE="$ORION_HOME/config/orion.env"
if [[ -f "$ENV_FILE" ]]; then
  ok "orion.env already exists — leaving it alone"
else
  cat > "$ENV_FILE" <<'ENVEOF'
# O.R.I.O.N. node environment. Read by systemd; never committed.
# Fill in what you use and leave the rest blank.

# ── language model ──────────────────────────────────────────────────────────
# GEMINI_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=

# ── telephony ───────────────────────────────────────────────────────────────
# Either TWILIO_AUTH_TOKEN, or the API key pair. Not both.
# TWILIO_ACCOUNT_SID=
# TWILIO_AUTH_TOKEN=
# TWILIO_FROM_NUMBER=+44...
# The public HTTPS host Twilio fetches TwiML from. Must match your Caddyfile.
# ORION_PUBLIC_HOST=orion.example.com

# ── remote uplink ───────────────────────────────────────────────────────────
# Shared secret for the phone/PWA pairing. Generate one with:
#   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# ORION_REMOTE_TOKEN=
ENVEOF
  ok "wrote a template — fill it in before starting"
fi
chown "$ORION_USER:$ORION_USER" "$ENV_FILE"
# Credentials. Owner-only, and the owner is a no-login account.
chmod 600 "$ENV_FILE"

# ── 7. systemd ───────────────────────────────────────────────────────────────
say "Installing systemd units"
for unit in orion.service orion-telephony.service; do
  if [[ -f "$ORION_HOME/deploy/$unit" ]]; then
    cp "$ORION_HOME/deploy/$unit" "/etc/systemd/system/$unit"
    ok "installed $unit"
  else
    warn "deploy/$unit not found — skipped"
  fi
done
systemctl daemon-reload
systemctl enable orion.service >/dev/null 2>&1 || warn "could not enable orion.service"
ok "units installed (not started — review the config first)"

# ── 8. firewall ──────────────────────────────────────────────────────────────
say "Configuring the firewall"
# Default deny inbound. Only SSH and HTTP(S) are reachable; ORION's own ports
# stay on loopback and are only reachable through Caddy.
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp comment 'ssh' >/dev/null
ufw allow 80/tcp comment 'http (ACME challenge, redirects to https)' >/dev/null
ufw allow 443/tcp comment 'https (Caddy -> ORION)' >/dev/null
ufw --force enable >/dev/null
ok "ufw: 22, 80, 443 in; everything else denied"
warn "8765 (uplink) and 8790 (telephony) are NOT exposed — that is deliberate."

# ── 9. Caddy ─────────────────────────────────────────────────────────────────
say "Installing Caddy"
if ! command -v caddy >/dev/null 2>&1; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update -qq && apt-get install -y -qq caddy
fi
if [[ -n "$DOMAIN" && -f "$ORION_HOME/deploy/Caddyfile.hostinger" ]]; then
  sed "s/orion.example.com/${DOMAIN}/g" \
    "$ORION_HOME/deploy/Caddyfile.hostinger" > /etc/caddy/Caddyfile
  systemctl reload caddy 2>/dev/null || systemctl restart caddy
  ok "Caddy serving ${DOMAIN} (certificate issues on first request)"
else
  warn "No domain given — Caddy installed but not configured."
  warn "Re-run as: bash $0 yourdomain.example.com"
fi

# ── done ─────────────────────────────────────────────────────────────────────
say "Done"
cat <<NEXT
  Next:
    1. nano ${ENV_FILE}                    # keys, Twilio, public host
    2. nano ${ORION_HOME}/config/telephony_contacts.json
                                           # numbers ORION may dial
    3. systemctl start orion
       systemctl start orion-telephony     # only if you want calls
    4. journalctl -u orion -f              # watch it come up

  Check it:
    systemctl status orion --no-pager
    curl -s localhost:8765/health || echo "(not answering yet)"
NEXT
