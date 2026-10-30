"""Mark XXXX security: the threat guard (Defender-backed antivirus layer), one
message per sentinel pass, the phone master switch and its record, and the
scored security audit."""
from __future__ import annotations

import asyncio
import os
import time
import types
from pathlib import Path

import pytest

from orion_core import phone_control
from orion_core.security_audit import (Check, check_listening, check_remote_desktop, check_uac,
                                       format_audit)
from orion_core.threat_guard import (Finding, SeenStore, ThreatGuard, Verdict, command_target,
                                     family, parse_scan)


class _Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _guard(tmp_path, *, signature="unsigned", verdict=Verdict("clean"), startup=None,
           clock=None, exists=lambda _p: True):
    scans: list[str] = []

    def scan(path):
        scans.append(str(path))
        return verdict

    guard = ThreatGuard(seen=SeenStore(tmp_path / "seen.json"), scan=scan,
                        signature=lambda _p: signature, downloads=tmp_path / "Downloads",
                        startup=startup or (lambda: {}), hosts=tmp_path / "hosts",
                        clock=clock or _Clock(), exists=exists)
    return guard, scans


def test_a_program_already_gone_is_not_judged_and_a_file_is_scanned_once(tmp_path):
    guard, scans = _guard(tmp_path, exists=lambda _p: False)
    assert guard.judge_processes([("sfx.exe", r"C:\Temp\sfx.exe")]) == []
    assert scans == []
    helper = tmp_path / "Temp" / "helper.exe"
    helper.parent.mkdir()
    helper.write_bytes(b"MZ")
    guard, scans = _guard(tmp_path / "b")
    for _ in range(3):                                    # it keeps restarting
        guard.judge_processes([("helper.exe", str(helper))])
    assert scans == [str(helper)]


def test_one_installer_is_one_program_however_it_is_spelt():
    # 27 Oct 2026: Torvis-setup.exe and Torvis-setup.tmp were two spoken notes.
    assert family("Torvis-setup.exe") == family("Torvis-setup.tmp") == "torvis-setup"
    assert family("Torvis-setup (1).exe") == "torvis-setup"
    assert family("is-ABC12.tmp") == "is-abc12"


def test_a_clean_unsigned_installer_from_temp_is_shown_not_spoken(tmp_path):
    guard, scans = _guard(tmp_path)
    started = [("Torvis-setup.exe", r"C:\Users\me\AppData\Local\Temp\is-1.tmp\Torvis-setup.exe"),
               ("Torvis-setup.tmp", r"C:\Users\me\AppData\Local\Temp\is-2.tmp\Torvis-setup.tmp")]
    found = guard.judge_processes(started)
    assert len(found) == 1 and len(scans) == 1           # one family, one scan
    said, shown = guard.to_tell(found)
    assert said == "" and "Defender found nothing" in shown[0]


def test_a_threat_is_spoken_once_and_remembered_across_restarts(tmp_path):
    clock = _Clock()
    guard, _ = _guard(tmp_path, verdict=Verdict("threat", "Trojan:Win32/Wacatac"), clock=clock)
    found = guard.judge_processes([("evil.exe", r"C:\Users\me\Downloads\evil.exe")])
    said, _ = guard.to_tell(found)
    assert said.startswith("Security alert:") and "Trojan:Win32/Wacatac" in said
    # A new ORION (restart) with the same store: not told again within a week.
    again, _ = _guard(tmp_path, verdict=Verdict("threat", "Trojan:Win32/Wacatac"), clock=clock)
    assert again.to_tell(again.judge_processes([("evil.exe", r"C:\Users\me\Downloads\evil.exe")]))[0] == ""
    clock.now += 8 * 86400
    later, _ = _guard(tmp_path, verdict=Verdict("threat", "Trojan:Win32/Wacatac"), clock=clock)
    assert later.to_tell(later.judge_processes([("evil.exe", r"C:\Users\me\Downloads\evil.exe")]))[0]


def test_signed_programs_are_not_scanned_and_unscannable_ones_are_told(tmp_path):
    guard, scans = _guard(tmp_path, signature="signed")
    found = guard.judge_processes([("setup.exe", r"C:\Temp\setup.exe")])
    assert found[0].severity == "info" and scans == []
    guard, _ = _guard(tmp_path / "b", verdict=Verdict("unknown"))
    assert guard.judge_processes([("tool.exe", r"C:\Temp\tool.exe")])[0].severity == "warn"


def test_programs_outside_temp_and_downloads_are_left_alone(tmp_path):
    guard, scans = _guard(tmp_path)
    assert guard.judge_processes([("app.exe", r"C:\Program Files\App\app.exe")]) == []
    assert scans == []


def test_a_new_download_is_scanned_once_it_has_settled(tmp_path):
    clock = _Clock(time.time())
    guard, scans = _guard(tmp_path, verdict=Verdict("threat", "Virus:DOS/EICAR_Test_File"),
                          clock=clock)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "notes.txt").write_text("x")
    assert guard.check_downloads() == []                  # the first look is the baseline
    fresh = downloads / "game-crack.exe"
    fresh.write_bytes(b"MZ")
    assert guard.check_downloads() == [] and scans == []  # still being written
    clock.now += 30
    found = guard.check_downloads()
    assert [f.severity for f in found] == ["threat"] and scans == [str(fresh)]
    assert guard.check_downloads() == []                  # unchanged: not scanned again


def test_a_new_unsigned_autostart_is_told_even_when_clean(tmp_path):
    entries = {"HKCU Run: OneDrive": r'"C:\Program Files\OneDrive\OneDrive.exe" /background'}
    target = tmp_path / "updater.exe"
    target.write_bytes(b"MZ")
    guard, _ = _guard(tmp_path, startup=lambda: dict(entries))
    assert guard.check_startup() == []                    # baseline
    entries["HKCU Run: Updater"] = f'"{target}" --silent'
    found = guard.check_startup()
    assert len(found) == 1 and found[0].severity == "warn"
    assert "start with Windows" in found[0].message
    assert guard.check_startup() == []


def test_a_changed_hosts_file_is_told(tmp_path):
    guard, _ = _guard(tmp_path)
    (tmp_path / "hosts").write_text("127.0.0.1 localhost\n")
    assert guard.check_hosts() == []
    (tmp_path / "hosts").write_text("127.0.0.1 localhost\n6.6.6.6 mybank.com\n")
    assert [f.severity for f in guard.check_hosts()] == ["warn"]


def test_defender_scan_results_and_autostart_commands_are_read():
    assert parse_scan(0, "found no threats") == Verdict("clean")
    out = "LIST OF DETECTED THREATS\nThreat                  : Virus:DOS/EICAR_Test_File\n"
    assert parse_scan(2, out) == Verdict("threat", "Virus:DOS/EICAR_Test_File")
    assert parse_scan(5, "") == Verdict("unknown")
    assert command_target(r'"C:\A B\x.exe" --flag') == r"C:\A B\x.exe"
    assert command_target(r"C:\Riot Games\Riot Client\RiotClientServices.exe --launch") == \
        r"C:\Riot Games\Riot Client\RiotClientServices.exe"
    assert command_target("rundll32 something") == ""


def test_one_sentinel_pass_says_one_thing():
    from orion_core.security_sentinel import SecuritySentinel
    spoken, banners = [], []
    bus = types.SimpleNamespace(
        speak_request=types.SimpleNamespace(emit=spoken.append),
        banner=types.SimpleNamespace(emit=lambda text, _level: banners.append(text)),
        log=types.SimpleNamespace(emit=lambda _m: None))
    from orion_core.threat_guard import Telling
    guard = types.SimpleNamespace(tell=lambda findings: Telling(
        ("tool.exe started from a temp folder",), (), False, ("tool.exe started from a temp folder",)))
    sentinel = SecuritySentinel(bus, guard=guard)
    sentinel._alert("port:9012", "a service is now listening on port 9012")
    sentinel._findings.append(Finding("proc:tool", "warn", "tool.exe started from a temp folder"))
    sentinel._flush()
    assert len(spoken) == 1 and spoken[0].startswith("Security note: 2 things.")
    assert len(banners) == 1
    sentinel._flush()
    assert len(spoken) == 1                               # nothing new, nothing said


def test_the_phone_master_switch_stops_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_control, "SWITCH_PATH", tmp_path / "phone_control.json")
    monkeypatch.setattr(phone_control, "ACTIVITY_PATH", tmp_path / "phone_activity.jsonl")
    from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate

    class _Dispatcher:
        async def dispatch(self, name, args):
            from orion_core.data import ToolResult
            return ToolResult(f"ran {name}")

    gate = RemoteToolGate(_Dispatcher(), RemoteConfirmationRegistry())
    assert phone_control.enabled()
    assert asyncio.run(gate.dispatch("diagnostics", {})).text == "ran diagnostics"
    assert "nothing from the phone" in phone_control.set_enabled(False)
    refused = asyncio.run(gate.dispatch("diagnostics", {}))
    assert not refused.ok and refused.text == phone_control.OFF_REPLY
    steps = phone_control.recent(1)
    assert [s["outcome"] for s in steps] == ["ran", "off"]
    assert all(set(s) == {"at", "tool", "tier", "outcome", "device"} for s in steps)
    summary = phone_control.summary(1)
    assert summary.startswith("Phone control is OFF") and "2 requests" in summary
    phone_control.set_enabled(True)
    assert phone_control.enabled()


def test_phone_steps_older_than_ninety_days_are_dropped(tmp_path):
    path = tmp_path / "activity.jsonl"
    now = 1_800_000_000.0
    phone_control.record("web_search", "allow", "ran", path=path, now=now - 100 * 86400)
    phone_control.record("web_search", "allow", "ran", path=path, now=now)
    assert phone_control.prune(path=path, now=now) == 1
    assert len(phone_control.recent(365, path=path, now=now)) == 1


def test_the_switch_is_desktop_only():
    from orion_core.remote_capability import Tier, classify
    assert classify("security_watch", {"action": "phone_on"}) is Tier.FORBID


def test_the_audit_scores_and_names_the_fix():
    def reg(values):
        return lambda _hive, path, name: values.get(name)
    assert check_uac(reg({"EnableLUA": 1, "ConsentPromptBehaviorAdmin": 5})).ok
    off = check_uac(reg({"EnableLUA": 0}))
    assert off.ok is False and off.fix
    assert check_remote_desktop(reg({"fDenyTSConnections": 0})).ok is False
    windows_only = [{"port": 445, "proc": "System"}, {"port": 135, "proc": "svchost.exe"}]
    assert check_listening(windows_only).ok
    apps = [{"port": 9000 + i, "proc": f"app{i}.exe"} for i in range(7)]
    assert check_listening(apps).ok is False
    report = format_audit([Check("Firewall", True, "on."), Check("Remote Desktop", False, "on.", "switch it off."),
                           Check("SMBv1", None, "not readable")])
    assert report.splitlines()[0] == "Security audit: 1 of 2 checks pass."
    assert "Fix: switch it off." in report and "Not checked here: SMBv1." in report


@pytest.mark.skipif(os.name != "nt", reason="Authenticode is Windows-only")
def test_windows_own_files_count_as_signed():
    from orion_core.threat_guard import signature_state
    root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    assert signature_state(root / "System32" / "notepad.exe") == "signed"
    assert signature_state(root / "no-such-file.exe") == "unknown"


def test_phone_control_off_stops_phone_conversation_too(tmp_path, monkeypatch):
    # Mark XXXX.1: the switch blocked tools but the phone could still chat.
    import inspect
    from orion_core.remote import RemoteGateway
    src = inspect.getsource(RemoteGateway._answer)
    assert src.index("phone_control.enabled()") < src.index("phone|mobile|android")
    assert "phone-control-off" in src


def test_a_new_network_port_names_its_program_and_spares_windows(monkeypatch):
    from orion_core import security_sentinel as ss
    from orion_core import threat_guard as tg

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def name(self):
            return {1: "svchost.exe", 2: "Spotify.exe", 3: "rat.exe"}[self.pid]

        def exe(self):
            return {1: r"C:\Windows\System32\svchost.exe", 2: r"C:\Apps\Spotify.exe",
                    3: r"C:\Users\me\AppData\Roaming\rat.exe"}[self.pid]

    monkeypatch.setattr(ss.psutil, "Process", _Proc)
    monkeypatch.setattr(tg, "signature_state",
                        lambda path: "signed" if "Spotify" in str(path) else "unsigned")
    sentinel = ss.SecuritySentinel(types.SimpleNamespace(), guard=object())
    assert sentinel._port_finding(135, 1) is None
    spotify = sentinel._port_finding(57621, 2)
    assert spotify.severity == "info" and "Spotify.exe" in spotify.message
    rat = sentinel._port_finding(4444, 3)
    assert rat.severity == "warn" and "rat.exe" in rat.message and "4444" in rat.message


def test_next_face_design_moves_on_in_order(monkeypatch):
    from orion_core import appearance
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    emitted = []
    host = types.SimpleNamespace(bus=types.SimpleNamespace(
        gui_command=types.SimpleNamespace(emit=emitted.append)))
    monkeypatch.setattr(appearance, "face_design", lambda: "visor")
    source = DesktopDispatchMixin.interface_control
    result = source(host, {"action": "face_design", "target": "next"})
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    order = appearance.FACE_DESIGNS
    assert emitted[-1]["target"] == order[(order.index("visor") + 1) % len(order)]


def test_an_all_clear_never_silences_a_later_warning_about_the_same_name(tmp_path):
    """Mark XXXX.4: a signed setup.exe (info) used to silence an unsigned,
    unscanned setup.exe from Temp for the rest of the week."""
    guard, _scans = _guard(tmp_path, signature="signed")
    temp = r"C:\Users\me\AppData\Local\Temp\setup.exe"
    first = guard.judge_processes([("setup.exe", temp)])
    assert first[0].severity == "info"
    guard.to_tell(first)
    guard._signature = lambda _p: "unsigned"
    guard._scan = lambda _p: Verdict("unknown")
    second = guard.judge_processes([("setup.exe", temp)])
    spoken, _shown = guard.to_tell(second)
    assert second[0].severity == "warn" and "setup.exe" in spoken


def test_the_phone_switch_fails_closed_on_a_damaged_file(tmp_path):
    """Mark XXXX.4: only a switch never set reads as on; a damaged or locked
    one must not let the phone through."""
    from orion_core import phone_control
    path = tmp_path / "phone_control.json"
    assert phone_control.enabled(path) is True
    path.write_text("{not json", encoding="utf-8")
    assert phone_control.enabled(path) is False
    phone_control.set_enabled(True, path)
    assert phone_control.enabled(path) is True


def test_phone_control_off_refuses_the_agent_task_route(monkeypatch):
    import asyncio
    from orion_core import phone_control
    from orion_core.remote import RemoteAgentQueue
    ran = []

    class Dispatcher:
        async def dispatch(self, name, args):
            ran.append(name)

    monkeypatch.setattr(phone_control, "enabled", lambda path=None: False)
    queue = RemoteAgentQueue(Dispatcher())
    task = asyncio.run(queue.run("recall_conversation", {}, device_id="phone-1"))
    assert task["status"] == "denied" and ran == []


def test_the_downloads_scan_stops_in_time_and_says_how_far_it_got(tmp_path, monkeypatch):
    """Mark XXXX.4: 40 Defender scans one after another outlasted the 45 s a
    live turn waits; the scan now stops at its deadline and says so."""
    from pathlib import Path
    from orion_core import dispatch_files, threat_guard
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    for index in range(5):
        (downloads / f"tool{index}.exe").write_bytes(b"MZ")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(threat_guard, "defender_cli", lambda: "MpCmdRun.exe")
    seen = []
    result = dispatch_files._scan_downloads(deadline_s=0.0,
                                            scan=lambda p: seen.append(p) or Verdict("clean"))
    assert len(seen) == 1                     # one scan always runs
    assert "4 older ones not reached in time" in result.text



class _NoteBus:
    def __init__(self):
        self.spoken, self.lines = [], []
        self.log = types.SimpleNamespace(emit=self.lines.append)
        self.banner = types.SimpleNamespace(emit=lambda *a: None)
        self.speak_request = types.SimpleNamespace(emit=self.spoken.append)


def test_a_note_held_in_the_quiet_gap_is_spoken_after_it():
    """A warn inside the 30-minute gap was marked told and never spoken -
    "Defender real-time protection is OFF" could be lost for good."""
    from orion_core.security_sentinel import SecuritySentinel
    bus = _NoteBus()
    watch = SecuritySentinel(bus)
    watch._spoken_at = time.monotonic()
    watch._notes.append("real-time protection is off in Windows Security")
    watch._flush()
    assert bus.spoken == []
    watch._spoken_at -= watch.SPOKEN_GAP_S + 1
    watch._flush()                                   # nothing new this pass
    assert len(bus.spoken) == 1 and "real-time protection is off" in bus.spoken[0]
    watch._flush()
    assert len(bus.spoken) == 1, "a held note is spoken once"


def test_programs_orion_launches_are_port_checked_but_his_own_files_are_not(monkeypatch):
    from orion_core import security_sentinel, threat_guard
    child_app = types.SimpleNamespace(pid=11, exe=lambda: r"C:\Program Files\App\app.exe")
    child_own = types.SimpleNamespace(pid=12, exe=lambda: r"C:\ORION\_internal\QtWebEngineProcess.exe")
    me = types.SimpleNamespace(pid=10, children=lambda recursive=True: [child_app, child_own])
    monkeypatch.setattr(security_sentinel.psutil, "Process", lambda *a: me)
    monkeypatch.setattr(threat_guard, "is_own_file", lambda exe: "ORION" in exe)
    assert security_sentinel.SecuritySentinel._own_pids() == {10, 12}


@pytest.mark.parametrize("name", ["MsMpEng.exe", "explorer.exe"])
def test_windows_own_listeners_are_not_security_notes(monkeypatch, name):
    from orion_core import security_sentinel
    proc = types.SimpleNamespace(name=lambda: name, exe=lambda: rf"C:\Windows\{name}")
    monkeypatch.setattr(security_sentinel.psutil, "Process", lambda pid: proc)
    watch = security_sentinel.SecuritySentinel(_NoteBus())
    assert watch._port_finding(51810, 4242) is None
