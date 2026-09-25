# O.R.I.O.N. on a Hostinger VPS

A 24/7 node: no display, no sound card, no keyboard. It answers the phone,
runs scheduled plugins, compiles research while you are asleep, and talks to
the paired phone app over HTTPS.

Tested on **Ubuntu 24.04** and **Debian 12**. A 2 vCPU / 4 GB plan is the
sensible floor — the unit caps ORION at 4 GB and two cores' worth of CPU
precisely so a runaway cannot take the box with it.

---

## 1. Put the code on the VPS

From your desktop, excluding everything machine-specific:

```bash
rsync -av --delete \
  --exclude .venv --exclude .git --exclude dist --exclude build \
  --exclude config --exclude release --exclude reports \
  ./ root@YOUR_VPS_IP:/opt/orion/
```

`config/` is excluded deliberately. It holds your keys, your memory
databases and your browser profile; the VPS gets its own.

## 2. Run the setup script

```bash
ssh root@YOUR_VPS_IP
bash /opt/orion/deploy/hostinger_setup.sh orion.yourdomain.com
```

It is idempotent — safe to run again after a failure or an upgrade. It
installs packages, creates the `orion` service account, configures a null
audio device, builds the virtualenv, installs both systemd units, locks the
firewall down to 22/80/443 and sets Caddy up for your domain.

It does **not** write any credential and does **not** start ORION.

## 3. Fill in the configuration

```bash
nano /opt/orion/config/orion.env              # keys, Twilio, public host
nano /opt/orion/config/telephony_contacts.json # numbers ORION may dial
```

The contact book is default-deny: a number that is not listed cannot be
called, however ORION is asked.

## 4. Start it

```bash
systemctl start orion
systemctl start orion-telephony     # only if you want calls
journalctl -u orion -f
```

---

## Verification checklist

Each of these fails loudly if the thing it checks is broken. Run them in
order; a later one assumes the earlier ones passed.

### Headless operation

```bash
# 1. It starts, and keeps running.
systemctl status orion --no-pager
#    Want: "active (running)". If it restart-loops, the reason is here:
journalctl -u orion -n 50 --no-pager

# 2. No display was touched. A Qt platform error means QT_QPA_PLATFORM
#    did not take, and the unit will be restart-looping.
journalctl -u orion --no-pager | grep -i "qt.qpa\|xcb\|could not connect to display"
#    Want: no output.

# 3. The uplink answers — on loopback only.
curl -s localhost:8765/health
curl -s --max-time 5 http://YOUR_VPS_IP:8765/health
#    Want: the first answers, the SECOND times out. If the second answers,
#    ORION is listening on a public interface; check ORION_REMOTE_HOST.

# 4. Through Caddy, with a real certificate.
curl -sI https://orion.yourdomain.com | head -3
```

### The boot roll-call

```bash
journalctl -u orion --no-pager | grep '^\[Tools\]' | tail -3
#    Want: "Tool discovery complete: N active." with N around 140.
journalctl -u orion --no-pager | grep '^\[Plugins\]'
#    Want: "Plugin discovery complete: N active, 0 rejected".
#    A non-zero rejected count names a plugin that failed its audit.
journalctl -u orion --no-pager | grep '^\[Audio\]'
#    Want: a line naming the host API. On a VPS this is the null device,
#    which is correct — it means the audio threads initialised.
```

### Audio decoupling (desktop only — the VPS has no face)

```bash
# The face must paint identically off the GUI thread, and cost the loop
# almost nothing. Both are asserted, with pixel comparison:
.venv/Scripts/python.exe -m pytest tests/test_face_pipeline.py -v

# To see the numbers rather than a pass/fail:
.venv/Scripts/python.exe -c "
import os, sys, time; os.environ.pop('QT_QPA_PLATFORM', None); sys.path.insert(0, '.')
from PyQt6.QtWidgets import QApplication; a = QApplication([])
from PyQt6.QtGui import QPixmap
from orion_core.gui.holo_head import HoloHeadPanel
p = HoloHeadPanel(); p.resize(900, 760); p.show(); a.processEvents()
time.sleep(1); pix = QPixmap(p.size())
t = time.perf_counter()
for _ in range(30): p.render(pix)
print(f'GUI paint: {(time.perf_counter()-t)*1000/30:.2f} ms/frame')
print(f'worker:    {p._pipeline.paint_ms:.1f} ms/frame on its own thread')
p.stop_pipeline()"
#    Want: GUI paint under 1 ms. Around 12 ms means the thread did not
#    start — check ORION_FACE_THREAD.
```

### Scheduled plugins

```bash
# Cron arithmetic, the vault and the supervisor:
.venv/bin/python -m pytest tests/test_plugin_schedule.py -q

# What is actually scheduled on the running node:
journalctl -u orion --no-pager | grep '^\[Schedule\]'
#    Want: one "'name' scheduled — cron(...) — next ..." per scheduled plugin.
#    "muted" means it failed three times running; the reason is on the line
#    above it.
```

### Outbound telephony

```bash
# The refusals, offline — no call is placed by any of these:
.venv/bin/python -m pytest tests/test_telephony.py tests/test_telephony_bridge.py -q

# Is a telephony server actually connected?
journalctl -u orion --no-pager | grep -i twilio

# A real call. This DIALS and COSTS MONEY. The number must already be in
# telephony_contacts.json or it is refused.
.venv/bin/python -c "
import asyncio, sys; sys.path.insert(0, '/opt/orion')
from orion_core.telephony import ContactBook, TelephonyGateway
from pathlib import Path
book = ContactBook(Path('/opt/orion/config/telephony_contacts.json'))
print('numbers I may dial:', book.names())
"
#    Then, in ORION: 'remind me in one minute about the news and call me'
```

### Two-way calling

```bash
# Twilio must be able to reach the bridge, with a valid certificate.
curl -sI https://orion.yourdomain.com/twilio/voice | head -1
#    Want: a 200 or a 403. A 403 is CORRECT here — it means the signature
#    check rejected an unsigned request, which is exactly its job.
#    A 502 means orion-telephony is not running.

systemctl status orion-telephony --no-pager
journalctl -u orion-telephony --no-pager | grep '^\[Call\]'
```

### Resolver telemetry

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '/opt/orion')
from orion_core.resolver_shadow import ShadowEvaluator
print(ShadowEvaluator('/opt/orion/config/resolver_shadow.db').report())"
#    Want, eventually: 'SAFE TO ENABLE — 200 turns, perfect recall'.
#    Until 200 turns it says NOT YET, and that is the correct answer —
#    recall on a small sample is not evidence.
```

### Resource limits are real

```bash
systemctl show orion -p MemoryMax -p CPUQuotaPerSecUSec -p Restart
#    Want: MemoryMax=4294967296, CPUQuotaPerSecUSec=2s, Restart=always
systemd-cgtop -1 -n1 | grep orion
```

---

## Keeping the desktop and the VPS in step

Both nodes read the same SQLite databases. SQLite does not support two
writers over a network filesystem, and pretending otherwise corrupts the
file — so this is a **one-way pull**, on a schedule, not a live mirror.

```bash
# On the desktop, nightly: push the knowledge ORION accumulated.
rsync -az --delete \
  ~/ORION/config/second_brain.db \
  ~/ORION/config/knowledge_graph.db \
  ~/ORION/config/ingestion.db \
  root@YOUR_VPS_IP:/opt/orion/config/
```

Stop ORION on the VPS first, or copy a consistent snapshot rather than the
live file:

```bash
ssh root@YOUR_VPS_IP systemctl stop orion
rsync -az ...
ssh root@YOUR_VPS_IP systemctl start orion
```

The alternative — `sqlite3 source ".backup snapshot.db"` on the desktop and
shipping the snapshot — avoids the downtime and is what to do once the node
matters. What you must not do is mount `config/` over NFS or SSHFS and point
both at it: SQLite's locking does not work over those, and the failure is
silent corruption rather than an error.

Which databases are safe to share, and which are not:

| Database | Share it? | Why |
| --- | --- | --- |
| `second_brain.db`, `knowledge_graph.db` | yes | Facts. The same on both. |
| `ingestion.db` | yes | What has been read already; avoids re-reading. |
| `resolver_shadow.db` | no | Per-node evidence about that node's usage. |
| `orion_core.db`, `focus.db`, `study.db` | no | Desktop session state. |
| `plugin_vault.json` | **no** | DPAPI-sealed to a Windows account; it will not decrypt on Linux. Re-enter the secrets on the VPS. |

---

## When something is wrong

| Symptom | Cause |
| --- | --- |
| Unit restart-loops immediately | Missing shared library. `journalctl -u orion -n 30` names it; usually `libgl1` or `libxkbcommon0`. |
| `ALSA: No such device` | `snd-dummy` did not load and `/etc/asound.conf` is missing. Re-run the setup script. |
| Calls connect then go silent after 30s | Caddy's default read timeout. Use `Caddyfile.hostinger`, which sets 3600s on `/media/*`. |
| Twilio gets 403 on every request | `TWILIO_AUTH_TOKEN` in `orion.env` does not match the account. That refusal is the signature check working. |
| ORION will not dial a number | It is not in `telephony_contacts.json`. That is default-deny, not a fault. |
| Scheduled plugin never fires | `journalctl -u orion \| grep Schedule`. Muted after three failures, or the machine was asleep at the cron time — a missed window is a missed run, by design. |
