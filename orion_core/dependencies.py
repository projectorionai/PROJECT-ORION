"""
Dynamic Package Resolver.

Accepts a list of required package names, diffs against installed packages,
and executes non-blocking background pip install for any missing entries.

Validates completion before handing control back to the installation loop.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .data import ToolResult


@dataclass
class DependencyResolutionOutcome:
    """Structured result of dependency resolution and installation."""

    succeeded: bool
    required_packages: list[str]
    missing_packages: list[str]
    installed_packages: list[str] = field(default_factory=list)
    error_log: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "required_packages": self.required_packages,
            "missing_packages": self.missing_packages,
            "installed_packages": self.installed_packages,
            "error_log": self.error_log,
            "duration_ms": self.duration_ms,
        }


class DynamicPackageResolver:
    """Handles package detection, installation, and validation."""

    # Standard library module names (to ignore)
    STDLIB_MODULES = {
        "abc", "aifc", "argparse", "array", "ast", "asyncio", "atexit", "audioop",
        "base64", "bdb", "binascii", "binhex", "bisect", "builtins", "bz2",
        "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs",
        "codeop", "collections", "colorsys", "compileall", "concurrent", "configparser",
        "contextlib", "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
        "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib",
        "dis", "distutils", "doctest", "dummy_thread", "dummy_threading", "email",
        "encodings", "ensurepip", "enum", "errno", "faulthandler", "fcntl", "filecmp",
        "fileinput", "fnmatch", "formatter", "fractions", "ftplib", "functools",
        "gc", "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
        "hashlib", "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
        "imp", "importlib", "inspect", "io", "ipaddress", "itertools", "json",
        "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
        "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder", "msilib",
        "msvcrt", "multiprocessing", "netrc", "nis", "nntplib", "numbers", "operator",
        "optparse", "os", "ossaudiodev", "parser", "pathlib", "pdb", "pickle",
        "pickletools", "pipes", "pkgutil", "platform", "plistlib", "poplib",
        "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
        "pyclbr", "pydoc", "queue", "quopri", "random", "readline", "reprlib",
        "resource", "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
        "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr",
        "socket", "socketserver", "spwd", "sqlite3", "ssl", "stat", "statistics",
        "string", "stringprep", "struct", "subprocess", "sunau", "symbol", "symtable",
        "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
        "termios", "test", "textwrap", "threading", "time", "timeit", "tkinter",
        "token", "tokenize", "tomllib", "trace", "traceback", "tracemalloc", "tty",
        "turtle", "turtledemo", "types", "typing", "typing_extensions", "unicodedata",
        "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
        "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
        "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
    }

    def __init__(self, bus: OrionBus) -> None:
        """
        Initialise the package resolver.

        Args:
            bus: OrionBus for logging resolution states.
        """
        self.bus = bus

    async def resolve_and_install(
        self,
        required_packages: list[str],
        timeout_seconds: float = 120.0,
    ) -> DependencyResolutionOutcome:
        """
        Resolve missing packages and install them via pip.

        Parses package entries, filters stdlib names, diffs against installed
        packages, then executes non-blocking pip install for any missing
        entries. Validates completion before returning.

        Args:
            required_packages: List of package names (e.g., ['matplotlib', 'gTTS'])
            timeout_seconds: pip install timeout (default 120 seconds)

        Returns:
            DependencyResolutionOutcome with installed/missing lists and status
        """
        self.bus.log.emit("[FORGE] Resolver: starting dependency resolution")
        outcome = DependencyResolutionOutcome(
            succeeded=False,
            required_packages=required_packages,
            missing_packages=[],
        )
        start_time = asyncio.get_running_loop().time()

        try:
            # Parse and filter packages
            parsed = self._parse_packages(required_packages)

            # Find missing packages
            missing = await asyncio.to_thread(self._find_missing, parsed)
            outcome.missing_packages = missing

            if not missing:
                outcome.succeeded = True
                self.bus.log.emit("[FORGE] Resolver: all packages already installed")
                return outcome

            # Install missing packages
            installed = await asyncio.to_thread(
                self._install_packages,
                missing,
                timeout_seconds,
            )
            outcome.installed_packages = installed

            # Validate installation
            still_missing = await asyncio.to_thread(self._find_missing, parsed)
            if still_missing:
                outcome.error_log.append(f"Still missing after install: {still_missing}")
                self.bus.log.emit(
                    f"[FORGE] Resolver: ✗ still missing: {still_missing}"
                )
            else:
                outcome.succeeded = True
                self.bus.log.emit(
                    f"[FORGE] Resolver: ✓ installed {len(installed)} packages"
                )

        except Exception as e:
            outcome.error_log.append(f"{type(e).__name__}: {str(e)}")
            self.bus.log.emit(f"[FORGE] Resolver: ✗ exception: {e}")

        finally:
            end_time = asyncio.get_running_loop().time()
            outcome.duration_ms = (end_time - start_time) * 1000.0

        return outcome

    def _parse_packages(self, packages: list[str]) -> list[str]:
        """
        Parse and normalise package names.

        Filters out stdlib names and normalises naming conventions.

        Args:
            packages: List of package names

        Returns:
            Filtered list of third-party package names
        """
        parsed: list[str] = []
        for pkg in packages:
            if not isinstance(pkg, str):
                continue
            pkg = pkg.strip().lower()
            if not pkg:
                continue
            # Skip stdlib
            base_name = pkg.split("[")[0].split("==")[0].split(">")[0].split("<")[0].strip()
            if base_name in self.STDLIB_MODULES:
                continue
            parsed.append(pkg)
        return parsed

    def _find_missing(self, packages: list[str]) -> list[str]:
        """
        Check which packages are not installed.

        Args:
            packages: Normalised list of package names

        Returns:
            List of missing package names
        """
        missing: list[str] = []
        for pkg in packages:
            base_name = pkg.split("[")[0].split("==")[0].split(">")[0].split("<")[0].strip()
            if not self._is_installed(base_name):
                missing.append(pkg)
        return missing

    def _is_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed and importable.

        Args:
            package_name: Package name (with possible extras or version specs stripped)

        Returns:
            True if the package can be found/imported, False otherwise
        """
        # Try importlib.util.find_spec (most reliable)
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            pass

        # Try fallback: common package → module name mapping
        fallback_mapping = {
            "pillow": "PIL",
            "pyyaml": "yaml",
            "pycryptodome": "Crypto",
            "beautifulsoup4": "bs4",
            "pymongo": "pymongo",
            "sqlalchemy": "sqlalchemy",
        }
        if package_name in fallback_mapping:
            try:
                spec = importlib.util.find_spec(fallback_mapping[package_name])
                return spec is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                pass

        return False

    def _install_packages(
        self,
        packages: list[str],
        timeout_seconds: float,
    ) -> list[str]:
        """
        Install packages via pip.

        Args:
            packages: List of package specs to install
            timeout_seconds: pip install timeout

        Returns:
            List of successfully installed package names
        """
        if not packages:
            return []
        vetted, refused = [], []
        for spec in packages:
            reason = vet_package(spec)
            (refused if reason else vetted).append((spec, reason))
        if refused:
            detail = "; ".join(f"{spec}: {why}" for spec, why in refused)
            try:
                self.bus.log.emit(f"DEPS: not installing — {detail}")
            except Exception:
                pass
        packages = [spec for spec, _ in vetted]
        if not packages:
            raise RuntimeError("refused to install: " + "; ".join(
                f"{spec} ({why})" for spec, why in refused))

        try:
            cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + packages
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if result.returncode == 0:
                return packages
            else:
                raise RuntimeError(f"pip failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"pip install timeout after {timeout_seconds}s")

    def tool_result(self, outcome: DependencyResolutionOutcome) -> ToolResult:
        """
        Convert DependencyResolutionOutcome to ToolResult for dispatcher.

        Args:
            outcome: Resolution outcome from resolve_and_install()

        Returns:
            ToolResult with installation summary
        """
        if outcome.succeeded:
            msg = (
                f"✓ Dependency resolution succeeded ({outcome.duration_ms:.1f}ms).\n"
                f"Required: {len(outcome.required_packages)}\n"
                f"Missing: {len(outcome.missing_packages)}\n"
                f"Installed: {len(outcome.installed_packages)}"
            )
            if outcome.installed_packages:
                msg += f"\n  • {', '.join(outcome.installed_packages[:5])}"
                if len(outcome.installed_packages) > 5:
                    msg += f"\n  • ... and {len(outcome.installed_packages) - 5} more"
            return ToolResult(msg, ok=True)
        else:
            msg = (
                f"✗ Dependency resolution failed. Errors:\n"
                + "\n".join(outcome.error_log)
            )
            return ToolResult(msg, ok=False)


# ── vetting what the model asks to install ──────────────────────────────────
#
# Package names reach _install_packages from GENERATED code (forged tools,
# plugins, repairs). Installing one runs its setup code with ORION's rights, so
# a misspelt name is not a harmless miss: typosquats ("reqeusts", "numpyy") are
# published precisely to be installed by mistake. ORION still fixes his own
# dependencies unattended — he just will not install a name that does not
# exist, is days old, or is one keystroke away from a famous package.

#: Very widely used packages: a name ONE edit away from these (and not itself
#: one of them) is treated as a probable typo.
_FAMOUS = frozenset("""
requests numpy pandas scipy matplotlib pillow opencv-python beautifulsoup4 lxml
httpx aiohttp urllib3 certifi idna charset-normalizer pyyaml toml tomli jinja2
flask django fastapi uvicorn pydantic sqlalchemy psutil pytest setuptools wheel
six python-dateutil pytz tzdata cryptography pyopenssl paramiko boto3 botocore
selenium playwright scikit-learn torch tensorflow keras transformers tokenizers
openai anthropic google-genai rich click tqdm colorama pyperclip pyautogui
keyboard mouse pywin32 comtypes onnxruntime sounddevice soundfile pyaudio
websockets websocket-client redis pymongo psycopg2 mysqlclient markdown pypdf
python-docx openpyxl xlrd reportlab qrcode chess numba sympy networkx
""".split())

_VET_CACHE: dict[str, str] = {}


def _edit_distance_one(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diffs) == 1 or (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                                   and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def vet_package(spec: str, *, min_age_days: float = 30.0) -> str:
    """"" when *spec* may be installed; otherwise the reason not to."""
    import re
    import time

    name = re.split(r"[<>=!~\[; ]", str(spec).strip(), maxsplit=1)[0].strip().lower()
    name = re.sub(r"[-_.]+", "-", name)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", name or ""):
        return "not a valid package name"
    if name in _VET_CACHE:
        return _VET_CACHE[name]
    if name not in _FAMOUS and any(_edit_distance_one(name, f) for f in _FAMOUS):
        near = next(f for f in _FAMOUS if _edit_distance_one(name, f))
        reason = f"one letter away from '{near}' — probably a typo, possibly a typosquat"
        _VET_CACHE[name] = reason
        return reason
    try:
        import httpx
        response = httpx.get(f"https://pypi.org/pypi/{name}/json", timeout=8.0)
    except Exception as exc:
        return f"could not check PyPI ({type(exc).__name__})"
    if response.status_code == 404:
        reason = "no such package on PyPI"
    elif response.status_code != 200:
        return f"PyPI answered {response.status_code}"
    else:
        reason = ""
        try:
            releases = response.json().get("releases") or {}
            stamps = []
            for files in releases.values():
                for f in files or []:
                    when = str(f.get("upload_time_iso_8601") or f.get("upload_time") or "")
                    if when:
                        stamps.append(when)
            if stamps:
                from datetime import datetime, timezone
                first = min(stamps).replace("Z", "+00:00")
                born = datetime.fromisoformat(first)
                if born.tzinfo is None:
                    born = born.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - born).total_seconds() / 86400
                if age < min_age_days:
                    reason = (f"first published {age:.0f} days ago — too new to trust "
                              "unattended")
        except Exception:
            reason = ""
    _VET_CACHE[name] = reason
    return reason
