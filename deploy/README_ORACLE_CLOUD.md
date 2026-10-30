# ORION on Oracle Cloud Always Free

This deploys ORION's existing headless Python node so it can answer when the
PC is switched off. The Android app's own phone actions still run on the phone.
Cloud hosting does not grant a Linux VM control over Windows, Android, or
their hardware.

Oracle's [current Always Free compute allowance](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
for a free tenancy is **2 Ampere A1 OCPUs and 12 GB RAM** in its home region.
Capacity is not guaranteed, and Oracle may reclaim idle Always Free instances.
The Gemini API's free quota is separate from Oracle's hosting allowance.

## 1. Create the VM

In the Oracle Cloud Console, choose **Compute → Instances → Create instance**:

| Setting | Value |
| --- | --- |
| Image | Ubuntu 24.04 ARM64 |
| Shape | `VM.Standard.A1.Flex`, 2 OCPUs, 12 GB memory, marked Always Free eligible |
| Region | Your tenancy's home region |
| Network | Public subnet with public IPv4 |
| Boot volume | Use an Always Free eligible size within the account's remaining allowance |
| SSH | Add your own public key; keep the private key on your PC |

If A1 reports **out of host capacity**, retry an availability domain or later.
The `VM.Standard.E2.1.Micro` option has only 1 GB RAM, so A1 is the recommended
fit for ORION's Python and PyQt6 headless process. Check the Console's cost
estimate before creating the instance; stop if it is not Always Free eligible.
Oracle account creation requires a payment verification method.

## 2. DNS and ingress

Create a free hostname at [DuckDNS](https://www.duckdns.org/) and point it to
the VM's public IPv4 address. In the VM's Oracle VCN security list or network
security group, allow inbound TCP **80** and **443** from the internet. Restrict
TCP **22** to your own current public IP where possible. Do not open TCP 8765:
ORION binds to `127.0.0.1:8765`, and Caddy provides HTTPS.

On the VM, allow the same ports in its host firewall if one is enabled. Test
SSH access before changing any firewall rule. A certificate needs the hostname
to resolve to this VM and ports 80/443 to be reachable.

## 3. Copy and prepare ORION

Make a source-only archive from the ORION project on your PC. Review the
archive contents before upload. Do not send a live `config/` directory,
`.env`, databases, credentials, recordings, logs or caches.

```powershell
tar -cf orion-cloud.tar orion.py orion_core deploy config/knowledge_packs
tar -tf orion-cloud.tar
scp -i <SSH_KEY> orion-cloud.tar ubuntu@<VM_IP>:/home/ubuntu/
ssh -i <SSH_KEY> ubuntu@<VM_IP>
```

On the VM, after replacing the example hostname:

```bash
sudo mkdir -p /opt/orion
sudo tar -xf /home/ubuntu/orion-cloud.tar -C /opt/orion
sudo bash /opt/orion/deploy/oracle_prepare.sh orion-example.duckdns.org
```

The setup script installs Python dependencies and Caddy, creates a dedicated
`orion` service account, and prepares systemd. It does **not** start ORION,
handle credentials or open a network port. It writes the public HTTPS address
to `/opt/orion/config/orion-cloud.env`; check that hostname before starting
the service.

## 4. Configure a provider privately

ORION needs a hosted AI API key for cloud replies. The current Gemini key can
be entered on the VM without placing it in shell history:

```bash
cd /opt/orion
sudo -u orion env ORION_CONFIG_DIR=/opt/orion/config \
  /opt/orion/.venv/bin/python -c \
  'from getpass import getpass; from orion_core.providers import write_api_key; write_api_key(getpass("Gemini API key: "))'
sudo chmod 600 /opt/orion/config/api_keys.json
```

The key is then stored on the VM in `/opt/orion/config/api_keys.json`; treat
VM access as credential access. Do not add it to the service file, Caddy
configuration, Git, command-line arguments or support logs. The cloud VM has
its own memory and pairing state unless an explicit sync mechanism is later
configured; copying a desktop database over a running node is unsafe.

## 5. Start and check

```bash
sudo systemctl start orion-node caddy
sudo systemctl status orion-node caddy --no-pager
curl --fail --silent --show-error http://127.0.0.1:8765/api/health
curl --fail --silent --show-error https://orion-example.duckdns.org/api/health
sudo journalctl -u orion-node -n 80 --no-pager
```

On a fresh node, the journal shows a **one-time pairing code**, valid for ten
minutes. Enter it in the Android app's server pairing screen. Keep journal
access private while the code is valid. A restart opens a new code if no phone
has been paired. Use the same stable HTTPS hostname in the app; a random
`trycloudflare.com` Quick Tunnel changes and is intended for testing only.

## 6. What to expect

- The cloud node can host ORION's conversation and memory while the PC is off.
- The native Android app may continue to use its on-phone Gemini mode for
  phone actions. Pointing it at this node requires separate client testing,
  including pairing and action routing.
- Windows applications and host GPU/CPU sensors remain unavailable when the
  PC is off. Phone permissions and approvals remain on the phone.
- Oracle hosting may be free within its allowance; Gemini usage and any
  usage beyond free quotas have their own terms.

Official references: [Oracle Always Free](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm),
[DuckDNS](https://www.duckdns.org/about.jsp),
[Caddy automatic HTTPS](https://caddyserver.com/docs/automatic-https), and
[Cloudflare Quick Tunnel limits](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).
