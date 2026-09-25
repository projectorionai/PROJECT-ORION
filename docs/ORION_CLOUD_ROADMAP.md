# ORION Cloud — Architecture Roadmap (Phase 7)

**Status:** design document · **Author:** Principal Systems Architect · **Date:** July 2026

ORION Cloud extends the desktop operating system into a local-first, multi-device
mesh. This is an *evolution*, not an invention: `orion_core/server.py` already
boots a headless brain (bus → telemetry → memory → identity → router →
RemoteGateway with an installable Android PWA), and desktop + cloud nodes
already share the SQLite memory schema. The roadmap below hardens that seed
into a product-grade architecture.

## Principles

1. **Local-first.** The desktop remains fully functional with zero cloud —
   MODE B is sacred. Cloud adds reach, never dependency.
2. **The desktop is the source of truth** for memory; the cloud node is a
   replica that can accept writes and reconcile.
3. **One identity everywhere.** The frozen persona + `identity.json`
   signature must match across nodes; drift is surfaced, never silently merged.
4. **Everything through the bus.** Remote requests enter as bus events on the
   receiving node; no service grows a second, cloud-only entry path.

## Node topology

```
┌────────────────────┐   mTLS / token    ┌─────────────────────┐
│  DESKTOP NODE      │◄─────────────────►│  CLOUD NODE          │
│  full GUI + audio  │   sync protocol   │  server.py --headless│
│  Windows control   │                   │  Oracle/any Linux VM │
│  source of truth   │                   │  RemoteGateway + PWA │
└─────────┬──────────┘                   └──────────┬──────────┘
          │ LAN (optional direct)                   │ HTTPS
          └──────────────►┌──────────────┐◄─────────┘
                          │ MOBILE PWA   │  voice + chat + briefing
                          │ (companion)  │  push notifications
                          └──────────────┘
```

## Folder structure (target)

```
orion_cloud/
├── gateway/            # aiohttp API + PWA static bundle (from remote.py)
│   ├── api.py          #   versioned JSON API (/v1/…)
│   ├── auth.py         #   token issue/verify, device registry
│   └── pwa/            #   installable companion app
├── sync/
│   ├── journal.py      #   append-only change journal (memory + graph)
│   ├── replicator.py   #   push/pull reconciliation engine
│   └── conflict.py     #   last-writer-wins + tier-aware merge rules
├── agents/
│   └── remote_exec.py  #   queued agent tasks with capability manifests
└── deploy/
    ├── orion-cloud.service   # systemd unit
    └── provision.sh          # VM bootstrap (Python, Ollama, certs)
```

## Service architecture

| Service | Node | Responsibility |
|---|---|---|
| RemoteGateway | cloud + desktop (opt-in) | HTTPS API, PWA hosting, push |
| SyncJournal | both | append-only log of memory/graph mutations with vector stamps |
| Replicator | both | exchange journal segments; idempotent apply |
| RemoteAgentQueue | cloud | accept task → run through dispatcher whitelist → return transcript |
| VoiceRelay | cloud | Piper/Edge-TTS synthesis for the PWA; STT via faster-whisper |

## API architecture (v1)

```
POST /v1/auth/pair            device pairing (QR one-time code from desktop)
POST /v1/auth/token           refresh short-lived access token
POST /v1/converse             text turn → routed reply (+sentiment tag)
GET  /v1/briefing             latest intelligence briefing (composed cloud-side)
POST /v1/agent/tasks          enqueue remote agent task (whitelisted tools only)
GET  /v1/agent/tasks/{id}     task status + transcript
GET  /v1/sync/journal?since=  pull journal segments
POST /v1/sync/journal         push journal segments
GET  /v1/health               node health + identity signature
```

## Security model

- **Transport:** TLS everywhere (Let's Encrypt on the VM; certificate pinning
  in the PWA). LAN sync may use mTLS with locally-issued certs.
- **AuthN:** device pairing via one-time QR code displayed on the desktop →
  long-lived refresh token per device (revocable) → 15-minute access tokens
  (HMAC-signed, `itsdangerous`-style, no third-party IdP required).
- **AuthZ:** capability manifests. A remote task may only call dispatcher
  tools on an explicit whitelist (no `desktop_control`, no `file_controller`
  writes, no `send_draft` — external actions always require the desktop's
  approval gate).
- **Secrets:** API keys never sync. Each node holds its own `api_keys.json`;
  the journal explicitly excludes the `config/` credential files.
- **Identity integrity:** every sync handshake exchanges the persona
  signature; mismatch → sync pauses and surfaces a banner, never auto-merges.

## Synchronisation strategy

1. Every memory/graph write already funnels through `OrionMemoryMatrix` /
   `KnowledgeGraphEngine` — add a journal hook there (one seam each).
2. Journal entries: `(node_id, lamport_ts, tier, category, key, value_hash,
   payload)` — append-only SQLite table, segment-shipped over `/v1/sync`.
3. Conflict rule: last-writer-wins per key, EXCEPT `long_term` and
   `knowledge` tiers, which merge additively (both values kept, newest
   pointed-to). Episodes never conflict (append-only by design).
4. Cadence: push-on-write when online, full reconcile on reconnect; the
   mobile PWA is read-mostly and needs no journal of its own.

## Voice continuity

The PWA records → cloud STT (faster-whisper) → router → reply text → cloud
TTS (Piper voice matched to the desktop SAPI profile) → audio stream back.
The same `VOICE_PROFILE` constants ship to the cloud node so ORION sounds
like himself on the move; Edge-TTS stands in where Piper's model is absent.

## Delivery phases

1. **C1 (now viable):** harden `server.py` deployment (systemd unit,
   provision script, TLS), pairing-based auth replacing the static token.
   *Status July 2026: pairing auth DONE in `remote.py` (`/v1/auth/pair`,
   `/v1/auth/token`, 15-min HMAC access tokens, revocable hashed refresh
   tokens); the C4 capability manifest is also seeded (`RemoteAgentQueue`
   with whitelist + hard-deny). Deployment scaffolding still pending.*
2. **C2:** sync journal + replicator; desktop⇄cloud memory continuity.
   *Status July 2026: the deterministic core is DONE in `orion_core/sync/`
   (`journal.py` — append-only `(node_id, lamport_ts, tier, category, key,
   value_hash, payload)` store with a Lamport clock + idempotent apply;
   `conflict.py` — last-writer-wins except additive `long_term`/`knowledge`;
   `replicator.py` — reconcile + `materialise_into` the memory matrix). Hooked
   into `OrionMemoryMatrix.save()` behind `ORION_SYNC_JOURNAL=1` (opt-in until
   the transport lands). Convergence is unit-tested (two nodes exchanging
   segments reconcile to identical state). Both seams are done: the
   `OrionMemoryMatrix.save()` hook and the `KnowledgeGraphEngine.upsert_entity()`
   hook (graph writes journal as knowledge-tier → additive). Remaining: the
   `/v1/sync/journal` push/pull endpoints on `RemoteGateway` and the
   identity-signature handshake gate.*
3. **C3:** PWA voice loop (STT/TTS relay) + push notifications for
   reminders/sentinel alerts.
4. **C4:** remote agent execution with capability manifests + audit log.
