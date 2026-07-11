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
        start_time = asyncio.get_event_loop().time()

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
            end_time = asyncio.get_event_loop().time()
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
