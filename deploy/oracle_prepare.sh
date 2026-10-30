#!/usr/bin/env bash
# Prepare a dedicated Ubuntu 24.04 Oracle Always Free VM for the headless node.
# Run as root after copying the repository to /opt/orion. This script never
# handles provider credentials or opens the ORION application port publicly.
set -euo pipefail

ORION_DIR=/opt/orion
HOSTNAME_ARG=${1:-}

if [[ ${EUID} -ne 0 ]]; then
    printf 'Run with sudo on the VM.\n' >&2
    exit 1
fi
if [[ ! -f ${ORION_DIR}/orion.py || ! -f ${ORION_DIR}/deploy/orion-node.service ]]; then
    printf 'Copy the ORION source to %s before running this script.\n' "${ORION_DIR}" >&2
    exit 1
fi
if [[ ! ${HOSTNAME_ARG} =~ ^([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$ ]]; then
    printf 'Supply a DNS hostname, for example: orion-example.duckdns.org\n' >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates curl caddy \
    libgl1 libglib2.0-0 libegl1 libxkbcommon0

if ! id orion >/dev/null 2>&1; then
    useradd --system --user-group --home-dir "${ORION_DIR}" \
        --shell /usr/sbin/nologin orion
fi
for path in config research reports logs conversations; do
    install -d -o orion -g orion -m 0700 "${ORION_DIR}/${path}"
done
# The source archive can contain reviewed, public knowledge packs. They must
# be writable when ORION refreshes them on first start.
chown -R orion:orion "${ORION_DIR}/config"
ENV_FILE="${ORION_DIR}/config/orion-cloud.env"
if [[ -f ${ENV_FILE} ]]; then
    # Keep any operator-provided settings on a repeat run. Only the hostname
    # selected for this run changes.
    sed -i '/^ORION_REMOTE_PUBLIC_URL=/d' "${ENV_FILE}"
fi
printf 'ORION_REMOTE_PUBLIC_URL=https://%s\n' "${HOSTNAME_ARG}" >> "${ENV_FILE}"
chown orion:orion "${ENV_FILE}"
chmod 0600 "${ENV_FILE}"
if [[ ! -x ${ORION_DIR}/.venv/bin/python ]]; then
    install -d -o orion -g orion -m 0755 "${ORION_DIR}/.venv"
    runuser -u orion -- python3 -m venv "${ORION_DIR}/.venv"
fi
runuser -u orion -- "${ORION_DIR}/.venv/bin/python" -m pip install --no-cache-dir \
    -r "${ORION_DIR}/deploy/requirements-server.txt"

install -m 0644 "${ORION_DIR}/deploy/orion-node.service" \
    /etc/systemd/system/orion-node.service
candidate=$(mktemp /etc/caddy/Caddyfile.orion.XXXXXX)
trap 'rm -f "$candidate"' EXIT
sed "s/orion.example.com/${HOSTNAME_ARG}/g" \
    "${ORION_DIR}/deploy/Caddyfile" > "$candidate"
caddy validate --config "$candidate"
cp -n /etc/caddy/Caddyfile /etc/caddy/Caddyfile.before-orion 2>/dev/null || true
install -m 0644 "$candidate" /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable orion-node.service caddy.service

printf '\nPrepared ORION for %s.\n' "${HOSTNAME_ARG}"
printf 'Next: set the DNS A record, open TCP 80/443 in the OCI security list,\n'
printf 'enter the Gemini key privately (see deploy/README_ORACLE_CLOUD.md),\n'
printf 'then run: sudo systemctl start orion-node caddy\n'
