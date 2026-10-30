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

say()  { printf '\n\033[1;31m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ! \033[0m%s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  echo "Run this as root: sudo bash $0 ${DOMAIN}" >&2
  exit 1
fi
if [[ ! $DOMAIN =~ ^([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$ ]]; then
  echo "Supply a DNS hostname, for example: orion.example.com" >&2
  exit 1
fi
if [[ ! -f "$ORION_HOME/orion.py" || ! -f "$ORION_HOME/deploy/requirements-server.txt" ||
      ! -f "$ORION_HOME/deploy/Caddyfile.hostinger" ]]; then
  echo "Copy the ORION source to $ORION_HOME before running this script." >&2
  exit 1
fi

source /etc/os-release
case "${ID}:${VERSION_ID}" in
  ubuntu:24.04|debian:12) ;;
  *) echo "Supported systems: Ubuntu 24.04 and Debian 12." >&2; exit 1 ;;
esac

# ── 1. packages ──────────────────────────────────────────────────────────────
say "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3 python3-venv python3-dev build-essential \
  git curl ca-certificates ufw rsync sudo gnupg \
  libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1
# libgl1/libxkbcommon/libfontconfig: PyQt6 links against them even under
# QT_QPA_PLATFORM=offscreen. Without them `import PyQt6.QtCore` fails with a
# missing shared object and the unit restart-loops with a cryptic message.
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

# ── 3. the code ──────────────────────────────────────────────────────────────
say "Preparing ${ORION_HOME}"
mkdir -p "$ORION_HOME"/{config,research/dossiers,reports,logs,conversations}
chown -R "$ORION_USER:$ORION_USER" "$ORION_HOME"
ok "directories ready"

# ── 4. the virtualenv ────────────────────────────────────────────────────────
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

# ── 5. the environment file ──────────────────────────────────────────────────
say "Preparing config/orion.env"
ENV_FILE="$ORION_HOME/config/orion.env"
if [[ -f "$ENV_FILE" ]]; then
  ok "orion.env already exists — leaving it alone"
else
  cat > "$ENV_FILE" <<ENVEOF
# O.R.I.O.N. node environment. Read by systemd; never committed.
# Fill in what you use and leave the rest blank.

# ── language model ──────────────────────────────────────────────────────────
# ORION_GEMINI_API_KEY=
# ORION_OPENAI_API_KEY=
# ORION_ANTHROPIC_API_KEY=

# ── telephony ───────────────────────────────────────────────────────────────
# The inbound bridge validates Twilio signatures with the auth token.
# TWILIO_AUTH_TOKEN=
# The public HTTPS host Twilio fetches TwiML from. Must match your Caddyfile.
# ORION_PUBLIC_HOST=orion.example.com

# ── remote uplink ───────────────────────────────────────────────────────────
# The gateway advertises this HTTPS origin to paired phones.
ORION_REMOTE_PUBLIC_URL=https://${DOMAIN}
ENVEOF
  ok "wrote a template — fill it in before starting"
fi
if ! grep -q '^ORION_REMOTE_PUBLIC_URL=' "$ENV_FILE"; then
  printf '\nORION_REMOTE_PUBLIC_URL=https://%s\n' "$DOMAIN" >> "$ENV_FILE"
  ok "added the public HTTPS origin to the existing environment file"
elif ! grep -Fxq "ORION_REMOTE_PUBLIC_URL=https://${DOMAIN}" "$ENV_FILE"; then
  warn "orion.env has a different ORION_REMOTE_PUBLIC_URL; check it matches ${DOMAIN}"
fi
chown "$ORION_USER:$ORION_USER" "$ENV_FILE"
# Credentials. Owner-only, and the owner is a no-login account.
chmod 600 "$ENV_FILE"

# ── 6. systemd ───────────────────────────────────────────────────────────────
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

# ── 7. firewall ──────────────────────────────────────────────────────────────
say "Configuring the firewall"
# Preserve the existing SSH and firewall policy. Resetting UFW on a remote VPS
# can remove an operator's non-standard SSH allowance and lock them out.
ufw allow 80/tcp comment 'http (ACME challenge, redirects to https)' >/dev/null
ufw allow 443/tcp comment 'https (Caddy -> ORION)' >/dev/null
if ufw status | grep -q '^Status: active'; then
  ok "ufw is active; existing rules were preserved and 80/443 allowed"
else
  warn "ufw is inactive; verify SSH access and enable it yourself if required"
fi
warn "8765 (uplink) and 8790 (telephony) are NOT exposed — that is deliberate."

# ── 8. Caddy ─────────────────────────────────────────────────────────────────
say "Installing Caddy"
if ! command -v caddy >/dev/null 2>&1; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
    /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy
fi
candidate=$(mktemp /etc/caddy/Caddyfile.orion.XXXXXX)
trap 'rm -f "$candidate"' EXIT
sed "s/orion.example.com/${DOMAIN}/g" \
  "$ORION_HOME/deploy/Caddyfile.hostinger" > "$candidate"
caddy validate --config "$candidate"
cp -n /etc/caddy/Caddyfile /etc/caddy/Caddyfile.before-orion 2>/dev/null || true
install -m 0644 "$candidate" /etc/caddy/Caddyfile
systemctl enable caddy >/dev/null
systemctl reload caddy 2>/dev/null || systemctl restart caddy
ok "Caddy serving ${DOMAIN} (certificate issues on first request)"

# ── done ─────────────────────────────────────────────────────────────────────
say "Done"
cat <<NEXT
  Next:
    1. nano ${ENV_FILE}                    # keys, Twilio, public URL
    2. systemctl start orion
    3. journalctl -u orion -f              # watch it come up

  Check it:
    systemctl status orion --no-pager
    curl --fail --silent --show-error http://127.0.0.1:8765/api/health || echo "(not answering yet)"
NEXT
