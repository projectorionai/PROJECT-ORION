#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# O.R.I.O.N. — knowledge sync between the desktop and the VPS node
#
#     bash deploy/sync_state.sh push root@1.2.3.4     # desktop -> node
#     bash deploy/sync_state.sh pull root@1.2.3.4     # node -> desktop
#     bash deploy/sync_state.sh status root@1.2.3.4   # what differs
#
# Why this is not a live mirror
# -----------------------------
# SQLite's locking does not work over NFS or SSHFS. Pointing two ORION nodes
# at one shared config directory does not fail with an error — it corrupts the
# database, silently, and you find out weeks later when a table will not read.
#
# So this ships CONSISTENT SNAPSHOTS, one direction at a time. Each database is
# copied with SQLite's own `.backup`, which takes a read lock and produces a
# file that is valid even if the source is being written to mid-copy. A plain
# `cp` or `rsync` of a live SQLite file can capture a torn page and the result
# is a database that opens and then fails on a query.
#
# The directive this implements asks for bi-directional hydration. It is
# bi-directional in the sense that knowledge flows both ways — but never at
# once, and never as a mount. That restriction is not squeamishness: it is the
# difference between a sync and a corruption.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-status}"
REMOTE="${2:-}"
REMOTE_HOME="${ORION_REMOTE_HOME:-/opt/orion}"
LOCAL_HOME="${ORION_LOCAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Where the desktop's data actually is — the same order constants.py uses:
# ORION_CONFIG_DIR, then the config_location.txt pointer that
# tools/move_config.py writes (the recommended way to get live SQLite out of a
# synced folder), then the in-project config/. Hard-coding config/ meant a
# moved install pushed nothing and pulled into a folder ORION no longer reads.
resolve_local_config() {
  if [[ -n "${ORION_CONFIG_DIR:-}" ]]; then
    printf '%s\n' "$ORION_CONFIG_DIR"; return
  fi
  local pointer="$LOCAL_HOME/config_location.txt" target=""
  if [[ -f "$pointer" ]]; then
    target="$(head -n 1 "$pointer" | tr -d '\r' | sed 's/[[:space:]]*$//')"
    if [[ -n "$target" && -d "$target" ]]; then
      printf '%s\n' "$target"; return
    fi
  fi
  printf '%s\n' "$LOCAL_HOME/config"
}
LOCAL_CONFIG="$(resolve_local_config)"

# Databases that describe the WORLD. The same facts are true on both nodes, so
# these are worth carrying across. (second_brain.db was listed here, but no
# version of ORION writes it: the second brain IS knowledge_graph.db. The
# evidence store — each claim's source and confidence — belongs with it.)
SHARED=(
  knowledge_graph.db
  evidence_graph.db
  ingestion.db
)

# Deliberately NOT synced, and why:
#   orion_core.db, focus.db, study.db  — session state. A focus streak from
#       the desktop means nothing on a node with no keyboard.
#   resolver_shadow.db  — evidence about THAT node's own tool usage. Merging
#       two nodes' observations would produce a recall figure describing
#       neither of them, which is worse than a small sample.
#   plugin_vault.json   — DPAPI-sealed to a Windows account. It will not
#       decrypt on Linux, and copying it there just leaves a file that throws.
#   browser_profile/    — logged-in sessions and cookies. Copying them shares
#       your live authentication with another machine.

say()  { printf '\n\033[1;31m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ! \033[0m%s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }

usage() {
  cat >&2 <<USAGE
Usage: $0 {push|pull|status} user@host

  push    desktop -> node   (send what you have learned here)
  pull    node -> desktop   (collect what it learned overnight)
  status  compare, change nothing

Environment:
  ORION_REMOTE_HOME  default /opt/orion
  ORION_LOCAL_HOME   default the repository this script is in
  ORION_CONFIG_DIR   the desktop's data folder (else config_location.txt, else config/)
  ORION_REMOTE_UNIT  the node's systemd unit (default: whichever of orion, orion-node exists)
  ORION_FORCE=1      push even if the node could not be stopped (risks corruption)

  $0 where          print the local data folder this script would use
USAGE
}

if [[ "$MODE" == "where" ]]; then
  printf '%s\n' "$LOCAL_CONFIG"
  exit 0
fi

if [[ -z "$REMOTE" ]]; then
  usage
  exit 1
fi

# The Oracle guide installs the node as orion-node.service, the Hostinger
# setup as orion.service. Stopping a unit that does not exist "failed", the
# script warned and carried on, and then swapped databases under a RUNNING
# node — the exact corruption the header above describes.
remote_unit() {
  if [[ -n "${ORION_REMOTE_UNIT:-}" ]]; then
    printf '%s\n' "$ORION_REMOTE_UNIT"; return
  fi
  ssh "$REMOTE" 'for u in orion orion-node; do
      systemctl cat "$u.service" >/dev/null 2>&1 && { echo "$u"; exit 0; }
    done; exit 1' || true
}

have_sqlite() { command -v sqlite3 >/dev/null 2>&1; }

snapshot_local() {
  # A consistent copy of a live database. `.backup` takes a read lock and
  # copies page by page; a plain file copy can catch a torn page and produce
  # a database that opens and then fails on a query.
  local name="$1" out="$2"
  if have_sqlite; then
    sqlite3 "$LOCAL_CONFIG/$name" ".backup '$out'"
  else
    warn "sqlite3 not found — falling back to a plain copy of $name."
    warn "Stop ORION first, or this may capture a torn page."
    cp "$LOCAL_CONFIG/$name" "$out"
  fi
}

case "$MODE" in

  status)
    say "Comparing (nothing will be changed)"
    printf '  %-24s %12s  %12s\n' "database" "local" "node"
    for db in "${SHARED[@]}"; do
      local_size=$(stat -c%s "$LOCAL_CONFIG/$db" 2>/dev/null || echo 0)
      remote_size=$(ssh "$REMOTE" "stat -c%s $REMOTE_HOME/config/$db 2>/dev/null || echo 0")
      printf '  %-24s %12s  %12s\n' "$db" "$local_size" "$remote_size"
    done
    echo
    ssh "$REMOTE" "systemctl is-active orion" 2>/dev/null \
      | sed 's/^/  node orion: /' || true
    ;;

  push)
    say "Desktop -> node"
    staging=$(mktemp -d)
    trap 'rm -rf "$staging"' EXIT
    for db in "${SHARED[@]}"; do
      if [[ ! -f "$LOCAL_CONFIG/$db" ]]; then
        warn "$db is not here — skipping"
        continue
      fi
      snapshot_local "$db" "$staging/$db"
      ok "snapshotted $db"
    done

    # Stopped while the files are replaced. A running ORION holding an open
    # handle to a database that is swapped underneath it will keep reading
    # the old inode until it restarts anyway — and may write to it, losing
    # exactly what was just sent.
    say "Stopping the node"
    unit="$(remote_unit)"
    if [[ -z "$unit" ]] || ! ssh "$REMOTE" "systemctl stop $unit"; then
      if [[ "${ORION_FORCE:-}" != "1" ]]; then
        warn "could not stop the node (unit: ${unit:-not found}). Nothing was sent:"
        warn "replacing a database under a running node corrupts it."
        warn "Set ORION_REMOTE_UNIT, or ORION_FORCE=1 to send anyway."
        exit 1
      fi
      warn "could not stop it — continuing because ORION_FORCE=1"
    fi
    rsync -az --info=stats1 "$staging"/ "$REMOTE:$REMOTE_HOME/config/"
    ssh "$REMOTE" "chown -R orion:orion $REMOTE_HOME/config" || true
    if [[ -n "$unit" ]]; then
      say "Starting the node"
      ssh "$REMOTE" "systemctl start $unit"
    fi
    ok "done"
    ;;

  pull)
    say "Node -> desktop"
    staging=$(mktemp -d)
    trap 'rm -rf "$staging"' EXIT

    # Snapshot on the far side, with its own sqlite3, for the same reason.
    for db in "${SHARED[@]}"; do
      ssh "$REMOTE" "command -v sqlite3 >/dev/null && \
        sqlite3 $REMOTE_HOME/config/$db \".backup '/tmp/orion-$db'\" \
        || cp $REMOTE_HOME/config/$db /tmp/orion-$db" 2>/dev/null \
        || { warn "$db could not be snapshotted on the node — skipping"; continue; }
      rsync -az "$REMOTE:/tmp/orion-$db" "$staging/$db"
      ssh "$REMOTE" "rm -f /tmp/orion-$db" || true
      ok "collected $db"
    done

    # The desktop's copies are backed up before being replaced. This script
    # overwrites months of accumulated knowledge; it should be possible to
    # change your mind about that.
    backup="$LOCAL_CONFIG/pre-sync-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup"
    for db in "${SHARED[@]}"; do
      [[ -f "$LOCAL_CONFIG/$db" ]] && cp "$LOCAL_CONFIG/$db" "$backup/"
    done
    ok "previous copies kept in $(basename "$backup")"

    warn "Stop ORION on this machine before continuing, then press Enter."
    read -r _
    cp "$staging"/* "$LOCAL_CONFIG/" 2>/dev/null || true
    ok "done"
    ;;

  *)
    echo "Unknown mode: $MODE (use push, pull or status)" >&2
    exit 1
    ;;
esac
