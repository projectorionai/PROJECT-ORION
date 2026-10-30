"""Threat guard: ORION's antivirus layer, built on Windows Defender's engine.

ORION has no malware engine of its own and never will (antivirus.py). What he
adds is judgement about WHEN to ask the real one and WHAT to tell the owner:

  * a program starting from a temp or downloads folder is checked for an
    Authenticode signature (WinVerifyTrust, offline); an unsigned one is
    scanned by Defender (MpCmdRun custom scan, report only);
  * new risky files in Downloads (programs, scripts, archives, shortcuts)
    are scanned the same way;
  * a new entry that makes a program start with Windows (Run keys, the
    Startup folder) - the commonest way malware stays - is checked and
    scanned;
  * a change to the hosts file is reported.

What he says: a threat Defender confirms, always; something he could not
clear (Defender unreachable, a new unsigned auto-start, the hosts file), once.
A clean result is shown on the banner and logged, not spoken - the owner
heard "Security note: 'Torvis-setup.exe' is running from a temp directory"
twice for an installer he had just run himself (27 Oct 2026): an Inno Setup
installer runs as setup.exe plus setup.tmp, and each got its own note. Now
the findings of one pass make one message, a program family ("torvis-setup")
is one finding, and what was told is remembered across restarts.

Nothing here deletes, quarantines or blocks: Defender's own real-time
protection does that, and a scan the owner asks for (antivirus.py) may let
Defender remediate.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .constants import CONFIG_DIR

SEEN_PATH = CONFIG_DIR / "security_seen.json"
#: A finding already told is not told again for this long.
TOLD_FOR_S = 7 * 86400.0
#: Remembered findings older than this are forgotten.
FORGET_AFTER_S = 30 * 86400.0
SCAN_TIMEOUT_S = 180.0
#: Files Defender is asked about when they land in Downloads.
RISKY_EXTENSIONS = frozenset({
    ".exe", ".msi", ".msix", ".appx", ".bat", ".cmd", ".com", ".scr", ".pif",
    ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".hta", ".jar", ".lnk",
    ".dll", ".iso", ".img", ".vhd", ".zip", ".7z", ".rar", ".cab",
})
#: A download still being written is left until its size settles.
SETTLE_S = 8.0
TEMP_PATH_RE = re.compile(r"(?i)\\temp\\|\\tmp\\|\\appdata\\local\\temp\\|/tmp/")
DOWNLOADS_PATH_RE = re.compile(r"(?i)\\downloads\\")
HOSTS_PATH = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "drivers" / "etc" / "hosts"


@dataclass(frozen=True)
class Finding:
    """One thing the guard noticed. ``severity``: info (banner and log),
    warn (told once) or threat (told, Defender confirmed it). ``advice`` is
    said once after everything in a message, however many findings share it."""
    key: str
    severity: str
    message: str
    advice: str = ""


@dataclass(frozen=True)
class Telling:
    """What one pass has to say: the parts to speak and the advice after them
    (each once), whether any is a confirmed threat, and the banner lines."""
    spoken: tuple[str, ...]
    advice: tuple[str, ...]
    threat: bool
    shown: tuple[str, ...]


def compose(parts: Iterable[str], advice: Iterable[str] = (), *, threat: bool = False) -> str:
    """One spoken message: each different thing said once, then each piece of
    advice once. Three ports opened by programs he could not name used to be
    the same sentence three times, advice and all, every 20 seconds (29 Oct 2026)."""
    said = list(dict.fromkeys(str(p).strip().rstrip(".") for p in parts if str(p).strip()))
    if not said:
        return ""
    lead = "Security alert" if threat else "Security note"
    body = f"{said[0]}." if len(said) == 1 else f"{len(said)} things. " + " ".join(f"{s}." for s in said)
    tips = " ".join(f"{a}." for a in dict.fromkeys(str(a).strip().rstrip(".") for a in advice
                                                    if str(a).strip()))
    return f"{lead}: {body}" + (f" {tips}" if tips else "")


@dataclass(frozen=True)
class Verdict:
    """Defender's answer about one file: clean, threat or unknown."""
    state: str
    threat: str = ""


# ── names ───────────────────────────────────────────────────────────────────

def family(name: str) -> str:
    """One name for one program however it is spelt on disk: "Torvis-setup.exe",
    "Torvis-setup.tmp" and "Torvis-setup (1).exe" are all "torvis-setup"."""
    stem = Path(str(name or "")).name.lower()
    while True:
        base, ext = os.path.splitext(stem)
        if ext and len(ext) <= 5 and base:
            stem = base
            continue
        break
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return stem.strip() or str(name or "").lower()


# ── signatures (WinVerifyTrust, offline) ─────────────────────────────────────

def signature_state(path: str | Path) -> str:
    """'signed' (valid Authenticode), 'unsigned', 'invalid' (a signature that
    fails: tampered, revoked, untrusted root) or 'unknown'. No network: the
    revocation check is off and URL retrieval is cache-only."""
    if not sys.platform.startswith("win"):
        return "unknown"
    target = Path(path)
    if is_store_app_file(target):
        return "signed"
    if not target.is_file():
        return "unknown"            # a Run entry left behind by an uninstalled app
    if is_windows_file(target):
        # Windows signs its own files through catalogs, not inside the file,
        # so notepad.exe reads "unsigned" here; they are protected by Windows
        # Resource Protection instead.
        return "signed"
    try:
        return _win_verify_trust(str(target))
    except Exception:
        return "unknown"


def is_own_file(path: str | Path) -> bool:
    """One of ORION's own files (his install folder, or the source tree)."""
    from .constants import BASE_DIR
    root = os.path.normcase(str(BASE_DIR)).rstrip("\\/") + os.sep
    return os.path.normcase(os.path.abspath(str(path))).startswith(root)


def can_be_signed(path: str | Path) -> bool:
    """A shortcut (.lnk) or a script carries no Authenticode signature of its
    own, so "unsigned" says nothing about it."""
    return Path(str(path)).suffix.lower() in {".exe", ".dll", ".msi", ".sys", ".scr", ".com",
                                               ".ocx", ".cpl", ".msix", ".appx"}


def is_store_app_file(path: str | Path) -> bool:
    """Under Program Files\\WindowsApps: a Microsoft Store app (WhatsApp, say),
    signed through its package's catalog rather than inside each file, so it
    read "unsigned" here, and protected from change by Windows."""
    root = os.path.normcase(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                         "WindowsApps")).rstrip("\\") + "\\"
    return os.path.normcase(os.path.abspath(str(path))).startswith(root)


def is_windows_file(path: str | Path) -> bool:
    """Under the Windows folder, and not in its writable Temp."""
    root = os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows")).rstrip("\\") + "\\"
    full = os.path.normcase(os.path.abspath(str(path)))
    return full.startswith(root) and not full.startswith(root + "temp\\")


def _win_verify_trust(path: str) -> str:
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    class FileInfo(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pcwszFilePath", wintypes.LPCWSTR),
                    ("hFile", wintypes.HANDLE), ("pgKnownSubject", ctypes.c_void_p)]

    class TrustData(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pPolicyCallbackData", ctypes.c_void_p),
                    ("pSIPClientData", ctypes.c_void_p), ("dwUIChoice", wintypes.DWORD),
                    ("fdwRevocationChecks", wintypes.DWORD), ("dwUnionChoice", wintypes.DWORD),
                    ("pFile", ctypes.POINTER(FileInfo)), ("dwStateAction", wintypes.DWORD),
                    ("hWVTStateData", wintypes.HANDLE), ("pwszURLReference", wintypes.LPCWSTR),
                    ("dwProvFlags", wintypes.DWORD), ("dwUIContext", wintypes.DWORD),
                    ("pSignatureSettings", ctypes.c_void_p)]

    action = GUID(0x00AAC56B, 0xCD44, 0x11D0,
                  (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE))
    info = FileInfo(ctypes.sizeof(FileInfo), path, None, None)
    data = TrustData()
    data.cbStruct = ctypes.sizeof(TrustData)
    data.dwUIChoice = 2                 # WTD_UI_NONE
    data.fdwRevocationChecks = 0        # WTD_REVOKE_NONE
    data.dwUnionChoice = 1              # WTD_CHOICE_FILE
    data.pFile = ctypes.pointer(info)
    data.dwStateAction = 1              # WTD_STATEACTION_VERIFY
    data.dwProvFlags = 0x10 | 0x1000    # no revocation check, cache-only URLs
    verify = ctypes.WinDLL("wintrust").WinVerifyTrust
    verify.argtypes = [wintypes.HWND, ctypes.POINTER(GUID), ctypes.c_void_p]
    verify.restype = ctypes.c_long
    result = verify(None, ctypes.byref(action), ctypes.byref(data)) & 0xFFFFFFFF
    data.dwStateAction = 2              # WTD_STATEACTION_CLOSE
    verify(None, ctypes.byref(action), ctypes.byref(data))
    if result == 0:
        return "signed"
    if result in (0x800B0100, 0x800B0003):   # TRUST_E_NOSIGNATURE, SUBJECT_FORM_UNKNOWN
        return "unsigned"
    return "invalid"


# ── Defender (MpCmdRun custom scan) ──────────────────────────────────────────

def defender_cli() -> str | None:
    """Defender's command-line scanner, newest platform first."""
    if not sys.platform.startswith("win"):
        return None
    platform = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft" / \
        "Windows Defender" / "Platform"
    try:
        found = sorted(platform.glob("*/MpCmdRun.exe"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
    except OSError:
        found = []
    fallback = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Defender" / "MpCmdRun.exe"
    for candidate in (*found, fallback):
        if candidate.is_file():
            return str(candidate)
    return None


_THREAT_LINE = re.compile(r"(?im)^\s*Threat\s*:\s*(.+?)\s*$")


def parse_scan(returncode: int, output: str) -> Verdict:
    """MpCmdRun's exit code 0 is clean and 2 is a threat (named in its list)."""
    if returncode == 0:
        return Verdict("clean")
    if returncode == 2:
        names = _THREAT_LINE.findall(output or "")
        return Verdict("threat", names[0] if names else "a threat")
    return Verdict("unknown")


def scan_file(path: str | Path, timeout: float = SCAN_TIMEOUT_S) -> Verdict:
    """Ask Defender about one file. Report only (-DisableRemediation): what to
    do about a threat is Defender's real-time protection's call, or the
    owner's."""
    cli = defender_cli()
    target = Path(path)
    if cli is None or not target.is_absolute() or not target.is_file():
        return Verdict("unknown")
    try:
        done = subprocess.run(
            [cli, "-Scan", "-ScanType", "3", "-File", str(target), "-DisableRemediation"],
            capture_output=True, text=True, errors="replace", timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return Verdict("unknown")
    return parse_scan(done.returncode, done.stdout)


# ── what was told, across restarts ───────────────────────────────────────────

class SeenStore:
    """Keys of findings already told (with when) and the baselines that need
    to survive a restart: auto-start entries and the hosts file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or SEEN_PATH)
        self._lock = threading.Lock()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}
        self.data.setdefault("told", {})

    def told_recently(self, key: str, now: float) -> bool:
        at = self.data["told"].get(key)
        return isinstance(at, (int, float)) and now - at < TOLD_FOR_S

    def mark(self, key: str, now: float) -> None:
        self.data["told"][key] = round(now, 1)

    def get(self, name: str, default: Any = None) -> Any:
        return self.data.get(name, default)

    def put(self, name: str, value: Any) -> None:
        self.data[name] = value

    def save(self, now: float) -> None:
        told = self.data["told"]
        for key in [k for k, at in told.items()
                    if not isinstance(at, (int, float)) or now - at > FORGET_AFTER_S]:
            told.pop(key, None)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data), encoding="utf-8")
                tmp.replace(self.path)
            except OSError:
                pass


# ── auto-start entries ───────────────────────────────────────────────────────

def startup_entries() -> dict[str, str]:
    """Programs set to start with Windows: Run/RunOnce keys of both hives and
    the user's Startup folder, as {where: command}."""
    entries: dict[str, str] = {}
    if not sys.platform.startswith("win"):
        return entries
    try:
        import winreg
        keys = [(winreg.HKEY_CURRENT_USER, "HKCU", r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_CURRENT_USER, "HKCU", r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
                (winreg.HKEY_LOCAL_MACHINE, "HKLM", r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_LOCAL_MACHINE, "HKLM", r"Software\Microsoft\Windows\CurrentVersion\RunOnce")]
        for hive, label, sub in keys:
            try:
                with winreg.OpenKey(hive, sub) as key:
                    index = 0
                    while True:
                        try:
                            name, value, _kind = winreg.EnumValue(key, index)
                        except OSError:
                            break
                        entries[f"{label} {sub.rsplit(chr(92), 1)[-1]}: {name}"] = str(value)
                        index += 1
            except OSError:
                continue
    except Exception:
        pass
    folder = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / \
        "Programs" / "Startup"
    try:
        for item in folder.iterdir():
            if item.name.lower() != "desktop.ini":
                entries[f"Startup folder: {item.name}"] = str(item)
    except OSError:
        pass
    return entries


_COMMAND_EXE = re.compile(
    r'^\s*"([^"]+)"'
    r"|^\s*((?:[a-z]:\\|\\\\).+?\.(?:exe|com|bat|cmd|scr|ps1|vbs|js|lnk))(?=\s|$)", re.I)


def command_target(command: str) -> str:
    """The program an auto-start command runs ("C:\\x\\a.exe" --flag -> the path)."""
    match = _COMMAND_EXE.search(os.path.expandvars(str(command or "")))
    return (match.group(1) or match.group(2)) if match else ""


# ── the guard ────────────────────────────────────────────────────────────────

def _version_tag(path: str) -> str:
    """A short tag for one version of one file (path, size, modified time)."""
    try:
        stat = os.stat(path)
        text = f"{os.path.normcase(str(path))}|{stat.st_size}|{stat.st_mtime}"
    except OSError:
        text = os.path.normcase(str(path))
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


class ThreatGuard:
    """Judges what the sentinel notices and remembers what it has told."""

    def __init__(self, *, seen: SeenStore | None = None,
                 scan: Callable[[str], Verdict] = scan_file,
                 signature: Callable[[str], str] = signature_state,
                 downloads: Path | None = None,
                 startup: Callable[[], dict[str, str]] = startup_entries,
                 hosts: Path | None = None,
                 clock: Callable[[], float] = time.time,
                 exists: Callable[[str], bool] = os.path.isfile) -> None:
        self.seen = seen or SeenStore()
        self._scan_file = scan
        self._signature = signature
        self._exists = exists
        #: Defender's verdict per file version (path, size, modified), so a
        #: helper that starts again and again is scanned once.
        self._verdicts: dict[tuple[str, int, float], Verdict] = {}
        self.downloads = downloads if downloads is not None else Path.home() / "Downloads"
        self._startup = startup
        self.hosts = hosts if hosts is not None else HOSTS_PATH
        self._clock = clock
        self._download_marks: dict[str, tuple[int, float]] | None = None
        self._pending: set[str] = set()     # downloads still being written

    # One pass of the guard's own checks.
    def sweep(self) -> list[Finding]:
        found = [*self.check_downloads(), *self.check_startup(), *self.check_hosts()]
        self.seen.save(self._clock())
        return found

    # A program that started from a temp or downloads folder.
    def judge_processes(self, started: Iterable[tuple[str, str]]) -> list[Finding]:
        """``started``: (name, exe path) of programs that just started."""
        by_family: dict[str, tuple[str, str]] = {}
        for name, exe in started:
            if exe and (TEMP_PATH_RE.search(exe) or DOWNLOADS_PATH_RE.search(exe)) \
                    and not is_own_file(exe):
                by_family.setdefault(family(name or exe), (name, exe))
        found = []
        for fam, (name, exe) in by_family.items():
            finding = self._judge_program(f"proc:{fam}", name or Path(exe).name, exe,
                                          where="from a temp or downloads folder")
            if finding is not None:
                found.append(finding)
        if found:
            self.seen.save(self._clock())
        return found

    def _scan(self, path: str) -> Verdict:
        try:
            stat = os.stat(path)
            version = (os.path.normcase(str(path)), stat.st_size, stat.st_mtime)
        except OSError:
            return self._scan_file(path)
        if version not in self._verdicts:
            if len(self._verdicts) > 512:
                self._verdicts.clear()
            self._verdicts[version] = self._scan_file(path)
        return self._verdicts[version]

    def _judge_program(self, key: str, label: str, path: str, *, where: str) -> Finding | None:
        if not self._exists(path):
            return None                    # already gone: nothing left to judge
        state = self._signature(path)
        if state == "signed":
            return Finding(key, "info", f"{label} started {where}; it is properly signed")
        verdict = self._scan(path)
        if verdict.state == "threat":
            # Keyed by the FILE: a second malicious program of the same
            # family within the week is told too (Mark XXXX.4).
            return Finding(f"{key}:threat:{_version_tag(path)}", "threat",
                           f"Defender identifies {label} as {verdict.threat}. It started {where}")
        if verdict.state == "clean":
            signed = "its signature fails" if state == "invalid" else "it is unsigned"
            severity = "warn" if state == "invalid" else "info"
            return Finding(key, severity,
                           f"{label} started {where}; {signed}, and Defender found nothing in it")
        return Finding(key, "warn", f"{label} started {where}, it is not signed, and "
                       "Defender could not check it")

    # New risky files in Downloads.
    def check_downloads(self) -> list[Finding]:
        now = self._clock()
        marks: dict[str, tuple[int, float]] = {}
        try:
            with os.scandir(self.downloads) as items:
                for item in items:
                    if item.is_file() and Path(item.name).suffix.lower() in RISKY_EXTENSIONS:
                        stat = item.stat()
                        marks[item.path] = (stat.st_size, stat.st_mtime)
        except OSError:
            return []
        previous, self._download_marks = self._download_marks, marks
        if previous is None:
            return []                      # the first look is the baseline
        found = []
        for path, (size, mtime) in marks.items():
            if previous.get(path) == (size, mtime) and path not in self._pending:
                continue
            if now - mtime < SETTLE_S:
                self._pending.add(path)    # still downloading: next pass
                continue
            self._pending.discard(path)
            name = Path(path).name
            key = f"download:{hashlib.sha1(f'{path}|{size}|{mtime}'.encode()).hexdigest()[:16]}"
            if self.seen.told_recently(key, now):
                continue
            verdict = self._scan(path)
            if verdict.state == "threat":
                found.append(Finding(key, "threat",
                                     f"Defender identifies {name} in Downloads as {verdict.threat}"))
            elif verdict.state == "clean":
                found.append(Finding(key, "info", f"Scanned {name} in Downloads: clean"))
            else:
                found.append(Finding(key, "info", f"{name} arrived in Downloads; Defender "
                                     "could not scan it"))
        return found

    # New auto-start entries, against a baseline that survives restarts.
    def check_startup(self) -> list[Finding]:
        current = self._startup()
        known = self.seen.get("startup")
        self.seen.put("startup", current)
        if not isinstance(known, dict):
            return []                      # first run: this is the baseline
        found = []
        for where, command in current.items():
            if known.get(where) == command:
                continue
            target = command_target(command)
            label = where.split(": ", 1)[-1]
            key = f"startup:{hashlib.sha1(f'{where}|{command}'.encode()).hexdigest()[:16]}"
            if not target or not Path(target).is_file() or is_own_file(target) \
                    or not can_be_signed(target):
                found.append(Finding(key, "info", f"{label} was set to start with Windows"))
                continue
            judged = self._judge_program(key, label, target, where="with Windows")
            if judged is None:
                continue
            if judged.severity == "info" and self._signature(target) != "signed":
                # A new unsigned program that starts itself with Windows is the
                # commonest way malware stays: worth one spoken word even when
                # Defender has nothing on it.
                judged = Finding(judged.key, "warn",
                                 f"{label} has set itself to start with Windows; it is "
                                 "unsigned, though Defender found nothing in it")
            found.append(judged)
        return found

    # The hosts file can quietly send a bank's name to another address.
    def check_hosts(self) -> list[Finding]:
        try:
            digest = hashlib.sha256(self.hosts.read_bytes()).hexdigest()
        except OSError:
            return []
        known = self.seen.get("hosts")
        self.seen.put("hosts", digest)
        if not known or known == digest:
            return []
        return [Finding(f"hosts:{digest[:16]}", "warn",
                        "the Windows hosts file has changed - it decides where web "
                        "addresses lead, so check that you or an app you trust changed it")]

    # Which findings are new enough to tell, and how.
    def tell(self, findings: Iterable[Finding]) -> Telling:
        """Each finding is told once per TOLD_FOR_S; a threat is spoken, a
        warn spoken once, info only shown."""
        now = self._clock()
        spoken, advice, shown, threat = [], [], [], False
        for finding in findings:
            # An all-clear is remembered apart from warnings (Mark XXXX.4): a
            # signed setup.exe told as "info" used to silence an unsigned,
            # unscanned setup.exe for the rest of the week.
            memory = finding.key + (":info" if finding.severity == "info" else "")
            if self.seen.told_recently(memory, now):
                continue
            self.seen.mark(memory, now)
            shown.append(finding.message + (f". {finding.advice}" if finding.advice else ""))
            if finding.severity in ("threat", "warn"):
                spoken.append(finding.message)
                if finding.advice:
                    advice.append(finding.advice)
                threat = threat or finding.severity == "threat"
        self.seen.save(now)
        return Telling(tuple(spoken), tuple(advice), threat, tuple(shown))

    def to_tell(self, findings: Iterable[Finding]) -> tuple[str, list[str]]:
        """(one spoken message, banner lines)."""
        told = self.tell(findings)
        return compose(told.spoken, told.advice, threat=told.threat), list(told.shown)


__all__ = ["Finding", "SeenStore", "Telling", "ThreatGuard", "Verdict", "command_target", "compose",
           "defender_cli", "family", "is_store_app_file", "parse_scan", "scan_file", "signature_state",
           "startup_entries"]
