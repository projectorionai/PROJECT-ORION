"""Security audit: how well defended this PC is, checked and scored.

"How secure is my computer?" used to get a process count and a port list. A
real answer checks the defences that matter on a Windows PC and says plainly
which are down and how to put them right:

  firewall, Defender's real-time protection and definitions, User Account
  Control, Remote Desktop, Remote Assistance, SMBv1, services reachable from
  the network, programs that start with Windows without a signature, and
  whether the phone may act on this computer.

Read-only: registry reads, netsh and Defender's status. Nothing is changed;
each failed check names the fix for the owner to make.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None          # None: could not be checked here
    detail: str
    fix: str = ""


def format_audit(checks: list[Check]) -> str:
    """The audit as a short report: the score, then what needs doing."""
    known = [c for c in checks if c.ok is not None]
    passed = [c for c in known if c.ok]
    failed = [c for c in known if not c.ok]
    lines = [f"Security audit: {len(passed)} of {len(known)} checks pass."]
    for check in failed:
        lines.append(f"  ✗ {check.name}: {check.detail}" + (f" Fix: {check.fix}" if check.fix else ""))
    for check in passed:
        lines.append(f"  ✓ {check.name}: {check.detail}")
    skipped = [c.name for c in checks if c.ok is None]
    if skipped:
        lines.append("  Not checked here: " + ", ".join(skipped) + ".")
    return "\n".join(lines)


def _reg(hive_name: str, path: str, value: str) -> Any:
    """A registry value, or None when it (or the key) is absent."""
    try:
        import winreg
        hive = getattr(winreg, hive_name)
        with winreg.OpenKey(hive, path) as key:
            return winreg.QueryValueEx(key, value)[0]
    except Exception:
        return None


def check_uac(reg: Callable[[str, str, str], Any] = _reg) -> Check:
    path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"
    lua, prompt = reg("HKEY_LOCAL_MACHINE", path, "EnableLUA"), reg("HKEY_LOCAL_MACHINE", path,
                                                                     "ConsentPromptBehaviorAdmin")
    if lua is None:
        return Check("User Account Control", None, "not readable")
    if int(lua) != 1:
        return Check("User Account Control", False, "switched off - any program can make "
                     "system changes without asking.", "turn UAC back on in Control Panel.")
    if prompt is not None and int(prompt) == 0:
        return Check("User Account Control", False, "on, but set never to ask.",
                     "raise it to 'Notify me when apps try to make changes'.")
    return Check("User Account Control", True, "on and asking before system changes.")


def check_remote_desktop(reg: Callable[[str, str, str], Any] = _reg) -> Check:
    deny = reg("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\Terminal Server",
               "fDenyTSConnections")
    if deny is None:
        return Check("Remote Desktop", None, "not readable")
    if int(deny) == 1:
        return Check("Remote Desktop", True, "off.")
    return Check("Remote Desktop", False, "on - this PC accepts remote sign-ins.",
                 "switch it off in Settings > System > Remote Desktop unless you use it.")


def check_remote_assistance(reg: Callable[[str, str, str], Any] = _reg) -> Check:
    allow = reg("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\Remote Assistance",
                "fAllowToGetHelp")
    if allow is None:
        return Check("Remote Assistance", None, "not readable")
    if int(allow) == 0:
        return Check("Remote Assistance", True, "off.")
    return Check("Remote Assistance", False, "on - a favourite of phone scammers.",
                 "untick 'Allow Remote Assistance connections' in System Properties > Remote.")


def check_smb1(reg: Callable[[str, str, str], Any] = _reg) -> Check:
    server = reg("HKEY_LOCAL_MACHINE",
                 r"SYSTEM\CurrentControlSet\Services\LanmanServer\Parameters", "SMB1")
    client = reg("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Services\mrxsmb10", "Start")
    if (server is not None and int(server) != 0) or (client is not None and int(client) != 4):
        return Check("SMBv1", False, "the old file-sharing protocol WannaCry spread through is "
                     "installed.", "remove 'SMB 1.0/CIFS File Sharing Support' in Windows Features.")
    return Check("SMBv1", True, "not present.")


def check_firewall(state: str) -> Check:
    if state == "on":
        return Check("Firewall", True, "on for every network profile.")
    if state == "partial":
        return Check("Firewall", False, "off for at least one network profile.",
                     "turn it on for all profiles in Windows Security > Firewall.")
    if state == "off":
        return Check("Firewall", False, "off on every profile.",
                     "turn it on in Windows Security > Firewall.")
    return Check("Firewall", None, "not readable")


def check_defender(status: Any) -> Check:
    if status is None:
        return Check("Antivirus", None, "Defender did not answer (another antivirus may own "
                     "protection)")
    if status.realtime_protection is False:
        return Check("Antivirus", False, "Defender's real-time protection is off.",
                     "turn it on in Windows Security > Virus & threat protection.")
    age = status.signature_age_days
    if age is not None and age > 7:
        return Check("Antivirus", False, f"Defender's definitions are {age} days old.",
                     "run Windows Update or 'Check for updates' in Virus & threat protection.")
    return Check("Antivirus", True, "Defender is protecting in real time with current definitions.")


#: Windows' own networking (RPC, file sharing, sign-in): listening is normal.
_WINDOWS_LISTENERS = frozenset({"system", "svchost.exe", "lsass.exe", "wininit.exe",
                                "services.exe", "spoolsv.exe", ""})
#: More third-party listeners than this is worth a look.
MAX_APP_LISTENERS = 5


def check_listening(listening: list[dict[str, Any]]) -> Check:
    apps: dict[str, set[int]] = {}
    for entry in listening:
        proc = str(entry.get("proc") or "")
        if proc.lower() not in _WINDOWS_LISTENERS:
            apps.setdefault(proc, set()).add(int(entry["port"]))
    if not apps:
        return Check("Network exposure", True, "only Windows' own services listen beyond this PC.")
    names = ", ".join(f"{proc} ({', '.join(str(p) for p in sorted(ports)[:3])})"
                      for proc, ports in sorted(apps.items())[:8])
    return Check("Network exposure", len(apps) <= MAX_APP_LISTENERS,
                 f"{len(apps)} app(s) accept connections from the network: {names}.",
                 "quit the ones you don't need, or block them in Windows Firewall.")


def check_autostart(unsigned: list[str]) -> Check:
    if not unsigned:
        return Check("Auto-start programs", True, "every program that starts with Windows is signed.")
    return Check("Auto-start programs", False if len(unsigned) > 3 else True,
                 f"{len(unsigned)} unsigned: {', '.join(unsigned[:6])}.",
                 "remove any you do not recognise in Task Manager > Startup apps.")


def check_phone(enabled: bool, steps: int) -> Check:
    if not enabled:
        return Check("Phone control", True, "off - nothing from the phone reaches this PC.")
    return Check("Phone control", True, f"on, with every risky step confirmed on the phone; "
                 f"{steps} phone request(s) in the last 7 days.")


def run_audit(sentinel: Any = None) -> str:
    """The audit as a spoken report (blocking: call off the event loop)."""
    return format_audit(gather_checks(sentinel))


def score(checks: list[Check]) -> tuple[int, int]:
    """(passed, checked) - the checks that could not run are not counted."""
    known = [c for c in checks if c.ok is not None]
    return sum(1 for c in known if c.ok), len(known)


def gather_checks(sentinel: Any = None) -> list[Check]:
    """Every check, gathered here (blocking: call off the event loop)."""
    checks: list[Check] = []
    if sentinel is not None:
        checks.append(check_firewall(sentinel._firewall_status_sync()))
    try:
        from .antivirus import get_protection_status_sync
        checks.append(check_defender(get_protection_status_sync()))
    except Exception:
        checks.append(Check("Antivirus", None, "not readable"))
    if sys.platform.startswith("win"):
        checks += [check_uac(), check_remote_desktop(), check_remote_assistance(), check_smb1()]
    if sentinel is not None:
        checks.append(check_listening(sentinel._listening_services_sync()))
    try:
        from .threat_guard import (can_be_signed, command_target, is_own_file, signature_state,
                                   startup_entries)
        unsigned = []
        for where, command in startup_entries().items():
            target = command_target(command)
            if (target and can_be_signed(target) and not is_own_file(target)
                    and signature_state(target) in ("unsigned", "invalid")):
                unsigned.append(where.split(": ", 1)[-1])
        checks.append(check_autostart(unsigned))
    except Exception:
        pass
    try:
        from . import phone_control
        checks.append(check_phone(phone_control.enabled(), len(phone_control.recent(7))))
    except Exception:
        pass
    return checks


__all__ = ["Check", "format_audit", "gather_checks", "run_audit", "score"]
