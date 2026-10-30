"""Mark XXXXI: the security notes of 29 Oct 2026.

Between 10:07 and 10:10 the owner heard six spoken notes, each repeating
"an unknown program has started accepting connections from the network on
port N - I couldn't tell which program. If you didn't expect that, block it in
Windows Firewall." once per port. Windows hands out a new port each time, and
the finding was keyed by the port, so every 20-second pass was "new".
"""
from __future__ import annotations

import os
import types

import psutil
import pytest

from orion_core import security_sentinel as ss
from orion_core import threat_guard as tg
from orion_core.threat_guard import Finding, SeenStore, ThreatGuard, compose


class _Bus:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.banners: list[str] = []
        self.logs: list[str] = []
        self.speak_request = types.SimpleNamespace(emit=self.spoken.append)
        self.banner = types.SimpleNamespace(emit=lambda text, _level: self.banners.append(text))
        self.log = types.SimpleNamespace(emit=self.logs.append)


class _Clock:
    now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def _sentinel(tmp_path, bus=None, clock=None):
    guard = ThreatGuard(seen=SeenStore(tmp_path / "seen.json"), scan=lambda _p: tg.Verdict("clean"),
                        signature=lambda _p: "unsigned", downloads=tmp_path / "none",
                        startup=lambda: {}, hosts=tmp_path / "hosts", clock=clock or _Clock())
    return ss.SecuritySentinel(bus or _Bus(), guard=guard)


def _unreadable(monkeypatch, *, gone=()):
    """Processes whose path Windows will not reveal (and whose name, through
    the old one-line read, was lost with it); pids in *gone* have exited."""
    class _Proc:
        def __init__(self, pid):
            if pid in gone:
                raise psutil.NoSuchProcess(pid)
            self.pid = pid

        def name(self):
            return "" if self.pid >= 900 else f"svc{self.pid}.exe"

        def exe(self):
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(ss.psutil, "Process", _Proc)


def test_programs_that_cannot_be_named_are_never_spoken_however_many_ports(tmp_path, monkeypatch):
    _unreadable(monkeypatch)
    bus = _Bus()
    sentinel = _sentinel(tmp_path, bus)
    for pass_ports in ([49730, 49746, 50171], [55634, 55642], [49705, 55607, 60023]):
        for port in pass_ports:
            finding = sentinel._port_finding(port, 900 + port % 7)
            assert finding is not None and finding.severity == "info"
            sentinel._findings.append(finding)
        sentinel._flush()
    assert bus.spoken == []
    assert len(bus.banners) == 1               # once, then remembered for the week


def test_a_name_without_a_path_is_kept_and_only_shown(tmp_path, monkeypatch):
    _unreadable(monkeypatch)
    finding = _sentinel(tmp_path)._port_finding([63677, 64030], 12)
    assert finding.severity == "info" and "svc12.exe" in finding.message
    assert "ports 63677 and 64030" in finding.message and finding.key == "listen:svc12"


def test_a_program_already_gone_is_not_reported(tmp_path, monkeypatch):
    _unreadable(monkeypatch, gone={7})
    assert _sentinel(tmp_path)._port_finding(56643, 7) is None


def test_orions_own_processes_are_left_alone(tmp_path, monkeypatch):
    sentinel = _sentinel(tmp_path)
    sentinel._baseline = True
    conn = types.SimpleNamespace(status=psutil.CONN_LISTEN, laddr=types.SimpleNamespace(ip="0.0.0.0", port=50001),
                                 pid=os.getpid())
    monkeypatch.setattr(ss.psutil, "net_connections", lambda kind="inet": [conn])
    sentinel._check_ports()
    assert sentinel._findings == []


def test_advice_is_said_once_after_everything_it_follows():
    said = compose(["a.exe has started accepting connections from the network on port 1",
                    "b.exe has started accepting connections from the network on port 2"],
                   ["If you didn't expect that, block it in Windows Firewall"] * 2)
    assert said.count("block it in Windows Firewall") == 1
    assert said.startswith("Security note: 2 things.")
    assert compose(["same thing", "same thing."]) == "Security note: same thing."


def test_only_one_spoken_note_per_half_hour_unless_it_is_a_threat(tmp_path, monkeypatch):
    bus = _Bus()
    sentinel = _sentinel(tmp_path, bus)
    sentinel._findings.append(Finding("proc:one", "warn", "one.exe started from a temp folder"))
    sentinel._flush()
    sentinel._findings.append(Finding("proc:two", "warn", "two.exe started from a temp folder"))
    sentinel._flush()
    assert len(bus.spoken) == 1 and "one.exe" in bus.spoken[0]
    assert any("two.exe" in b for b in bus.banners)            # still shown
    assert any("shown, not spoken" in line for line in bus.logs)
    sentinel._findings.append(Finding("proc:evil", "threat", "Defender identifies evil.exe as Trojan"))
    sentinel._flush()
    assert len(bus.spoken) == 2 and bus.spoken[1].startswith("Security alert:")


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_store_apps_count_as_signed():
    root = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "WindowsApps")
    path = os.path.join(root, "5319275A.WhatsAppDesktop_2.2540.6.0_x64__cv1g1gvanyjgm", "WhatsApp.Root.exe")
    assert tg.is_store_app_file(path)
    assert tg.signature_state(path) == "signed"
    assert not tg.is_store_app_file(r"C:\Users\me\Downloads\WhatsApp.Root.exe")
