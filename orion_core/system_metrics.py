"""
CPU, memory and network, sampled once for everyone who asks.

Two loops were reading the same three counters independently — the core
window's telemetry every 0.75 s and the command centre's system strip every
1.0 s — and one of those counters is far more expensive than it looks.

Measured on this machine, with eight network interfaces:

    psutil.net_io_counters()     7.471 ms
    psutil.cpu_percent(None)     0.078 ms
    psutil.virtual_memory()      0.024 ms

``net_io_counters`` enumerates every adapter, including the virtual ones that
Tailscale, WSL and Hyper-V leave behind, so it costs a hundred times what the
other two cost together. Two callers at those intervals spent **17.45 ms per
second** on it — and they spent it on the qasync thread, which in this
application is also the thread that paints the face and feeds the audio
pipeline. It showed up in a sampled profile of a running ORION as 4.5% of all
samples, doing nothing but asking Windows the same question twice.

Here it is asked once. Throughput is also allowed to be staler than load:
a bar showing bytes per second does not become wrong when it is refreshed
every 1.5 s instead of every 0.75 s, because the rate is computed over the
elapsed time actually measured rather than over the interval someone hoped
for. Load and memory stay fresh, because those do move meaningfully between
one second and the next.

The counters now run on a single worker. Cached reads on the Qt/asyncio thread
return immediately even when Windows adapter enumeration stalls.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

#: How stale each figure is allowed to be. Load and memory change from second
#: to second and are nearly free to read; throughput is expensive to read and
#: is displayed as a rate, which staleness does not distort.
CPU_TTL_S = 0.40
NETWORK_TTL_S = 1.50

#: What counts as a full bar: 100 Mbit/s, in bytes.
FULL_SCALE_BYTES_PER_S = 12_500_000.0


@dataclass(frozen=True)
class Snapshot:
    """One reading. Percentages, so callers need no units."""

    cpu: float = 0.0
    ram: float = 0.0
    network: float = 0.0
    #: Bytes per second across all interfaces, for anything wanting the raw
    #: figure rather than a bar.
    bytes_per_second: float = 0.0


_cpu_at = 0.0
_net_at = 0.0
_last_total = 0
_last_at = 0.0
_latest = Snapshot()
_lock = Lock()
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orion-metrics")
_pending: Future | None = None
_generation = 0


def _refresh(generation: int) -> None:
    """Read Windows counters on the worker and publish one complete snapshot."""
    global _cpu_at, _net_at, _last_total, _last_at, _latest
    with _lock:
        cpu_at, net_at, last_total, last_at = _cpu_at, _net_at, _last_total, _last_at
        latest = _latest
    now = time.monotonic()
    cpu, ram = latest.cpu, latest.ram
    network, rate = latest.network, latest.bytes_per_second

    try:
        import psutil
    except Exception:                       # pragma: no cover - psutil is core
        return

    if now - cpu_at >= CPU_TTL_S:
        cpu_at = now
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            ram = float(psutil.virtual_memory().percent)
        except Exception:
            pass

    if now - net_at >= NETWORK_TTL_S:
        net_at = now
        try:
            counters = psutil.net_io_counters()
            # Adapter enumeration can take hundreds of milliseconds. Use the
            # time the counter returned, rather than the requested refresh time.
            measured_at = time.monotonic()
            total = int(counters.bytes_sent) + int(counters.bytes_recv)
            if last_at:
                elapsed = max(1e-3, measured_at - last_at)
                # max(0, ...): counters reset when an adapter is disabled or
                # a VPN comes up, and a negative delta would read as a full
                # bar rather than as nothing happening.
                rate = max(0.0, (total - last_total) / elapsed)
                network = min(100.0, rate / FULL_SCALE_BYTES_PER_S * 100.0)
            last_total, last_at, net_at = total, measured_at, measured_at
        except Exception:
            pass

    with _lock:
        if generation == _generation:
            _cpu_at, _net_at = cpu_at, net_at
            _last_total, _last_at = last_total, last_at
            _latest = Snapshot(cpu, ram, network, rate)


def sample() -> Snapshot:
    """Return cached telemetry immediately and request a refresh when due.

    Hardware counters and imports run on one worker. A slow adapter cannot
    block Qt painting, audio or the asyncio event loop, and repeated callers
    cannot queue overlapping reads.
    """
    global _pending
    with _lock:
        now = time.monotonic()
        if (_pending is None or _pending.done()) and (
                now - _cpu_at >= CPU_TTL_S or now - _net_at >= NETWORK_TTL_S):
            _pending = _worker.submit(_refresh, _generation)
        return _latest


def prime() -> None:
    """Request the first reading without holding up application startup."""
    sample()


def reset() -> None:
    """Forget everything. For tests, which must not inherit a live machine's
    readings from whichever test ran first."""
    global _cpu_at, _net_at, _last_total, _last_at, _latest, _generation
    with _lock:
        _generation += 1
        _cpu_at = _net_at = _last_at = 0.0
        _last_total = 0
        _latest = Snapshot()


__all__ = ["CPU_TTL_S", "FULL_SCALE_BYTES_PER_S", "NETWORK_TTL_S",
           "Snapshot", "prime", "reset", "sample"]
