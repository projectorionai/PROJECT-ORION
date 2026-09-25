"""One reading of the machine, shared by everything that displays it.

Found by sampling a running ORION with py-spy: ``net_io_counters`` was 4.5% of
every sample taken. Not because it is called often, but because it is
expensive — it enumerates every network adapter, and on this machine there are
eight once Tailscale, WSL and Hyper-V have left theirs behind:

    psutil.net_io_counters()     7.471 ms
    psutil.cpu_percent(None)     0.078 ms
    psutil.virtual_memory()      0.024 ms

Two loops were paying it independently, on the qasync thread — which in this
application is also the thread that paints the face and feeds audio. 17.45 ms
of every second went on asking Windows the same question twice.

These tests hold the three properties that make the fix a fix rather than a
rearrangement: the expensive call is actually skipped, the figures stay
correct when it is, and nothing here can take down the loop that asked.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import system_metrics  # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    """No test inherits another's readings, or the live machine's."""
    system_metrics.reset()
    yield
    system_metrics.reset()


class FakeCounters:
    def __init__(self, sent, recv):
        self.bytes_sent, self.bytes_recv = sent, recv


class FakePsutil:
    """Counts how often each figure is actually asked for."""

    def __init__(self, total=0):
        self.net_calls = self.cpu_calls = 0
        self.total = total

    def net_io_counters(self):
        self.net_calls += 1
        return FakeCounters(self.total, 0)

    def cpu_percent(self, interval=None):
        self.cpu_calls += 1
        return 12.5

    def virtual_memory(self):
        return type("Memory", (), {"percent": 44.0})()


@pytest.fixture()
def fake(monkeypatch):
    stub = FakePsutil()
    monkeypatch.setitem(sys.modules, "psutil", stub)
    return stub


# ── the expensive call is genuinely skipped ──────────────────────────────────

def test_repeated_callers_share_one_reading(fake):
    """The whole point. Two loops asking within a TTL must cost one call."""
    for _ in range(50):
        system_metrics.sample()
    assert fake.net_calls == 1, (
        f"the network counter was read {fake.net_calls} times for 50 samples")


def test_load_refreshes_sooner_than_throughput(fake, monkeypatch):
    """Load moves between one second and the next and is nearly free to read.
    Throughput is displayed as a rate, which staleness does not distort, and
    costs a hundred times more."""
    assert system_metrics.CPU_TTL_S < system_metrics.NETWORK_TTL_S

    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    system_metrics.sample()
    clock[0] += system_metrics.CPU_TTL_S + 0.01
    system_metrics.sample()
    assert fake.cpu_calls == 2
    assert fake.net_calls == 1, "throughput refreshed on the load interval"


def test_the_expensive_call_happens_when_it_is_due(fake, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    system_metrics.sample()
    clock[0] += system_metrics.NETWORK_TTL_S + 0.01
    system_metrics.sample()
    assert fake.net_calls == 2


# ── the figures stay right ───────────────────────────────────────────────────

def test_throughput_is_a_rate_over_the_time_actually_measured(fake,
                                                              monkeypatch):
    """Not over the interval someone hoped for.

    This is why a slower sample is still a correct one: half a megabyte in
    two seconds is 250 KB/s whether it was measured over two seconds or one.
    """
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    fake.total = 0
    system_metrics.sample()

    clock[0] += 2.0
    fake.total = 500_000
    reading = system_metrics.sample()
    assert reading.bytes_per_second == pytest.approx(250_000.0, rel=0.01)


def test_a_counter_reset_reads_as_idle_not_as_a_full_bar(fake, monkeypatch):
    """Counters go backwards when an adapter is disabled or a VPN comes up.
    A negative delta divided by elapsed would peg the bar at 100%."""
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    fake.total = 9_000_000
    system_metrics.sample()

    clock[0] += 2.0
    fake.total = 12_000                 # the adapter came back from zero
    reading = system_metrics.sample()
    assert reading.bytes_per_second == 0.0
    assert reading.network == 0.0


def test_the_bar_never_exceeds_full(fake, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    system_metrics.sample()
    clock[0] += 2.0
    fake.total = 10 ** 10
    assert system_metrics.sample().network == 100.0


def test_load_and_memory_come_through(fake):
    reading = system_metrics.sample()
    assert reading.cpu == 12.5
    assert reading.ram == 44.0


# ── it cannot take down the loop that asked ──────────────────────────────────

def test_a_failing_counter_returns_the_last_known_reading(monkeypatch, fake):
    """These are decorations. An exception here would kill the telemetry loop
    or the GUI timer that asked, which is a real fault caused by a cosmetic
    one."""
    good = system_metrics.sample()

    class Broken(FakePsutil):
        def net_io_counters(self):
            raise OSError("the adapter went away")

        def cpu_percent(self, interval=None):
            raise OSError("no")

    monkeypatch.setitem(sys.modules, "psutil", Broken())
    system_metrics.reset()
    system_metrics.sample()
    later = system_metrics.sample()
    assert isinstance(later, system_metrics.Snapshot)
    assert later.cpu == 0.0 and later.network == 0.0


def test_no_psutil_at_all_is_survivable(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("not installed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    assert isinstance(system_metrics.sample(), system_metrics.Snapshot)


# ── both displays actually use it ────────────────────────────────────────────

def test_neither_display_reads_the_counters_itself():
    """Otherwise this module is an extra call rather than a shared one.

    Checked as source rather than by running the GUI, because constructing a
    QApplication inline in a test hangs this suite.
    """
    for name in ("core_window.py", "command_centre.py"):
        source = (ROOT / "orion_core" / "gui" / name).read_text(
            encoding="utf-8")
        assert "system_metrics" in source, f"{name} does not share the reading"
        assert "psutil.net_io_counters()" not in source, (
            f"{name} still reads the network counters itself, so the "
            "expensive call happens twice per cycle after all")
