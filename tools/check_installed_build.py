"""
Installed-app parity check: run a built ORION.exe in every headless mode.

"Works from source, broken in the installed exe" happened four times, and each
time nothing said so until someone used the feature in the installed app:

  1. the bundled MCP server orion_world died on its first import in the frozen
     app (the host ran the PC's python on a loose .py whose imports live inside
     the executable's archive) - fixed by ``ORION.exe --mcp-server <name>``;
  2. faster-whisper's package data (assets/silero_vad_v6.onnx) was not bundled,
     so every offline transcription failed;
  3. the Forge verifier refused to run in the installed app ("A frozen ORION
     executable is not a Python verifier") - fixed by ``--forge-verify``;
  4. the frozen app could not pip-install a forged tool's dependencies - fixed
     by installing through a real Python into config/forge_packages.

This harness runs the executable the way the app runs itself and reports one
line per check, then ``ALL OK`` or ``FAILED: n`` (exit code 0 / 1):

  modes        the entry script dispatches every headless mode the source has
  mcp:<name>   each bundled MCP server completes initialize + tools/list,
               launched with the command the FROZEN host itself would use
  face-host,   ``--face-host`` and ``--page-host <kind>`` (every kind but face;
  page-host:*  the face IS ``--face-host``) report every window they build; each
               must be a live top-level Qt window (class Qt6*QWindow*) of the
               host process - what ORION embeds with QWindow.fromWinId - and the
               host must stay up while stdin is open, then quit when it closes
  forge:*      ``--forge-verify`` passes the Forge's own conformance probe and
               its pytest wrapper, inside the Forge's time budget
  imports:*    every orion_core module, and every module ORION imports lazily or
               by name (derived from the source by AST at check time), imports
               inside the frozen interpreter
  data:*       build_standalone's DATAS, its --collect-all packages' data files,
               every resource_path() the source asks for, and named data the app
               has been bitten by, are in the bundle - and load at run time
  forge:pip    the Forge's package-install path installs a local wheel offline
               into a TEMP config, and a fresh verification can import it

Run it with the BUILD interpreter (it reads the executable's archive with
PyInstaller, and compares --collect-all data against that environment):

    .neural-debug-env\\Scripts\\python.exe tools\\check_installed_build.py [exe]
        [--json PATH|-] [--only modes,mcp,face,forge,imports,data,pip] [-v]

Safety. A mode is only launched after the executable's entry script is shown to
handle it: an older build given ``--forge-verify`` would start the full app.
Every launch gets ORION_CONFIG_DIR set to a fresh temporary folder, never the
owner's data; secrets are withheld from the children's environment; every child
is bounded by a timeout and its process tree killed on expiry. A child that
fails reports how: orion.py's headless guard writes the traceback to stderr and
exits 1, and that exception leads the failing line. A child that prints a
traceback and then neither answers nor exits (an older windowed build holding an
"Unhandled exception" dialog) is killed after a few seconds and fails. This
script imports only the standard library (and PyInstaller's archive reader).
"""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
import marshal
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "dist" / "ORION" / "ORION.exe"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

#: Environment names withheld from every child (the sandbox's own rule).
SECRETISH = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
             "AUTH", "COOKIE", "PRIVATE", "_SID")
#: Interpreter state of whoever runs this script, which a desktop launch of
#: ORION would not have (a venv would also redirect the `py` launcher).
INTERPRETER_ENV = ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
                   "__PYVENV_LAUNCHER__", "CONDA_PREFIX")

#: Lazily imported modules that may be absent from the installed app BY DESIGN.
#: Matched on the dotted prefix. build_standalone's EXCLUDES are added at run
#: time. Anything not listed here that fails to import in the frozen app is a
#: failure: a new lazy import forces a decision instead of slipping through.
OPTIONAL_BY_DESIGN: dict[str, str] = {
    "torch": "excluded from the lean build (build_standalone.EXCLUDES); chess training only",
    "resemblyzer": "excluded from the build and marked sys_platform != 'win32' in "
                   "requirements/speech.txt; the voiceprint runs on ONNX instead",
    "silero_vad": "torch-based VAD package, never a requirement; faster-whisper ships its ONNX VAD",
    "whisper": "openai-whisper (torch), never a requirement; faster-whisper is the shipped engine",
    "easyocr": "torch OCR fallback, never a requirement; rapidocr + Windows OCR ship instead",
    "paddleocr": "paddle OCR fallback, never a requirement; rapidocr + Windows OCR ship instead",
    "pytesseract": "needs the separate Tesseract binary and is in no requirements file; "
                   "rapidocr + Windows OCR ship instead",
    "PyPDF2": "legacy name tried after pypdf (requirements/research.txt ships pypdf)",
    "send2trash": "never a requirement; cleanup falls back to ORION's own recycle bin",
    "mypy": "the Forge's type check is optional (ORION_FORGE_TYPECHECK=auto skips it when absent)",
    "piper": "prepared local voice adapter; piper-tts is optional and no engine/model "
             "is downloaded (not installed in this build environment)",
}

#: Data the installed app has been bitten by, checked by name (glob under
#: _internal) on top of everything derived from build_standalone and the source.
CRITICAL_DATA: tuple[tuple[str, str], ...] = (
    ("faster_whisper/assets/*.onnx",
     "offline Whisper's VAD model: every transcription runs with vad_filter=True"),
    ("tzdata/zoneinfo/Europe/London",
     "world time and Europe/London - Windows ships no tz database"),
    ("onnxruntime/capi/onnxruntime*.dll",
     "runs the voiceprint, semantic encoder, OCR and VAD models"),
    ("orion_core/forge_contract.py",
     "the Forge's conformance probe ships this source (sandbox.probe_source)"),
    ("build_info.json", "build freshness (orion_core/build_freshness.py)"),
    ("assets/models/kokoro-v1.0/kokoro-v1.0.onnx",
     "ORION's offline voice model (voice_kokoro; build_standalone --kokoro-dir)"),
    ("assets/models/kokoro-v1.0/voices-v1.0.bin", "the offline voice's speaker styles"),
    ("espeakng_loader/espeak-ng*", "espeak-ng, which turns the offline voice's text into phonemes"),
    ("espeakng_loader/espeak-ng-data/phontab", "espeak-ng's phoneme tables"),
    ("onnxruntime/capi/DirectML.dll", "the DirectML runtime of onnxruntime-directml (GPU routing)"),
)

#: Data loaded at run time INSIDE the frozen app (keys of the probe's DATA table).
DATA_PROBES: dict[str, str] = {
    "tz-europe-london": "zoneinfo.ZoneInfo('Europe/London') from the bundled tzdata",
    "onnxruntime": "onnxruntime and its native DLLs load",
    "whisper-vad": "faster_whisper.vad.get_vad_model() opens assets/silero_vad_v6.onnx",
    "voiceprint-encoder": "the speaker voiceprint encoder (assets/voiceprint) opens in onnxruntime",
    "three-js": "the face finds its complete vendored three.js graph (face3d.local_three)",
    "forge-contract": "sandbox.probe_source() reads the shipped forge_contract.py",
    "jsonschema-metaschemas": "jsonschema checks a schema against its bundled metaschema",
    "rapidocr-models": "RapidOCR loads its bundled recognition models",
    "kokoro-voice": "the offline voice: kokoro_onnx and espeakng_loader import, espeak-ng "
                    "makes phonemes, and the bundled model files have their pinned sizes "
                    "(voice_kokoro.self_check; no model loaded, no audio)",
    "onnxruntime-directml": "onnxruntime lists DmlExecutionProvider (the GPU route)",
}

#: Functions that import a module named by a string literal: name -> (index of
#: the module argument, index of an attribute argument or None).
DYNAMIC_IMPORT_CALLS: dict[str, tuple[int, int | None]] = {
    "import_module": (0, None), "__import__": (0, None), "find_spec": (0, None),
    # orion_core.lazy_import
    "LazyModule": (0, 1), "LazyCallable": (0, 1), "lazy_attr": (0, 1),
    # capability probes (capability_health, diagnostics)
    "_importable": (0, None), "_module_present": (0, None), "_simple": (1, None),
}

PIP_PROBE_PACKAGE = "orion_parity_probe"
PIP_PROBE_VALUE = "installed-by-orion-parity-check"


# ── results ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Check:
    """One line of the report."""

    name: str
    ok: bool
    detail: str
    seconds: float = 0.0
    data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def line(self, width: int = 30) -> str:
        timing = f" ({self.seconds:.1f}s)" if self.seconds else ""
        return f"{'PASS' if self.ok else 'FAIL'}  {self.name:<{width}} {self.detail}{timing}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def summary_line(checks: Sequence[Check]) -> str:
    failed = sum(1 for check in checks if not check.ok)
    return "ALL OK" if not failed else f"FAILED: {failed}"


def render(checks: Sequence[Check], *, verbose: bool = False) -> list[str]:
    width = max([len(check.name) for check in checks] + [12])
    lines = []
    for check in checks:
        lines.append(check.line(width))
        if verbose:
            for item in check.data.get("failures", []):
                lines.append(f"      - {item}")
            for item in check.data.get("notes", []):
                lines.append(f"      . {item}")
    return lines


def _brief(items: Sequence[str], limit: int = 4) -> str:
    shown = "; ".join(items[:limit])
    return shown + (f"; ...and {len(items) - limit} more" if len(items) > limit else "")


def _one_line(text: str, limit: int = 300) -> str:
    flat = " | ".join(line.strip() for line in str(text or "").splitlines() if line.strip())
    return flat if len(flat) <= limit else "..." + flat[-limit:]


# ── bounded children ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    error: str = ""
    mechanism: str = ""


def child_env(config_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of every executable launch: a desktop-like environment,
    no secrets, and ORION's writable data redirected to *config_dir*."""
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in INTERPRETER_ENV
           and not any(mark in key.upper() for mark in SECRETISH)}
    env["ORION_CONFIG_DIR"] = str(config_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra or {})
    return env


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and everything it started (QtWebEngine helpers, pip, ...)."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=15, creationflags=CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


#: A protocol child that printed an unhandled traceback and then fell silent
#: this long is treated as crashed. A WINDOWED frozen exe does not exit on an
#: unhandled exception: it shows the traceback in a modal dialog and waits.
CRASH_GRACE_S = 3.0


def fatal_exception(lines: Sequence[str]) -> str:
    """The final exception line if *lines* end with an unhandled Python
    traceback (``Traceback (most recent call last):`` ... ``NameError: x``)."""
    import re
    text = [line.rstrip() for line in lines if line.strip()]
    if not text or not re.match(r"^[A-Za-z_][\w.]*(Error|Exception)(:|$)", text[-1]):
        return ""
    return text[-1] if any(line.startswith("Traceback (most recent call last)")
                           for line in text[-80:]) else ""


class ChildCrashed(RuntimeError):
    """A protocol child printed an unhandled traceback and went silent."""


class LineCollector:
    """Drains a child's pipe on a thread (so it can never fill and stall the
    child), keeping its lines and when the last one arrived."""

    def __init__(self, stream: Any) -> None:
        self.lines: list[str] = []
        self.changed = time.monotonic()
        if stream is not None:
            threading.Thread(target=self._run, args=(stream,), daemon=True,
                             name="parity-stderr").start()

    def _run(self, stream: Any) -> None:
        try:
            for raw in stream:
                self.lines.append(_decode(raw).rstrip("\r\n"))
                self.changed = time.monotonic()
        except (OSError, ValueError):
            pass

    def crashed(self, grace_s: float | None = None) -> str:
        """The exception line, once a fatal traceback has been followed by silence."""
        grace = CRASH_GRACE_S if grace_s is None else grace_s
        fatal = fatal_exception(self.lines[-120:])
        return fatal if fatal and time.monotonic() - self.changed >= grace else ""

    def tail(self, limit: int = 400) -> str:
        return _one_line("\n".join(self.lines[-40:]), limit)


def child_failure(returncode: int | None, stderr_lines: Sequence[str]) -> str:
    """How a child failed, led by its exception: orion.py's headless guard
    (``_headless``) writes the traceback to fd 2 and exits with code 1."""
    fatal = fatal_exception(stderr_lines)
    code = "still running" if returncode is None else f"exit {returncode}"
    return f"{code}: {fatal}" if fatal else code


def run_bounded(cmd: Sequence[Any], *, cwd: Path | None = None, env: dict[str, str] | None = None,
                timeout: float = 60.0, popen: Callable[..., subprocess.Popen] | None = None,
                after_start: Callable[[subprocess.Popen], None] | None = None,
                crash_grace_s: float = 10.0) -> RunResult:
    """Run *cmd* with stdout captured to a file and stderr drained live (no
    pipe can fill and stall it). Its process tree is killed if it outlives
    *timeout*, or if it printed an unhandled traceback and then sat silent for
    *crash_grace_s* without exiting - a windowed build would be holding that
    traceback up in a dialog on the owner's screen. Both count as failures."""
    started = time.monotonic()
    with tempfile.TemporaryFile() as out:
        try:
            proc = (popen or subprocess.Popen)(
                [str(part) for part in cmd], cwd=str(cwd) if cwd else None, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW)
        except OSError as exc:
            return RunResult(None, "", "", 0.0, error=f"could not start: {exc}")
        err = LineCollector(getattr(proc, "stderr", None))
        if after_start is not None:
            try:
                after_start(proc)
            except Exception as exc:          # the child is still bounded below
                err.lines.append(f"[parity] feeding the child failed: {exc}")
        timed_out, error = False, ""
        deadline = started + timeout
        while True:
            try:
                proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            crash = err.crashed(crash_grace_s)
            if crash:
                kill_tree(proc)
                error = (f"crashed: {crash} - still running {crash_grace_s:.0f}s after its "
                         "traceback (a windowed build shows it in a dialog); killed")
                break
            if time.monotonic() >= deadline:
                timed_out = True
                kill_tree(proc)
                break
        seconds = time.monotonic() - started
        out.seek(0)
        time.sleep(0.05)                      # let the drain thread take the last lines
        return RunResult(proc.returncode, _decode(out.read()), "\n".join(err.lines),
                         seconds, timed_out, error)


# ── the executable's archive ─────────────────────────────────────────────────

@dataclasses.dataclass
class Archive:
    strings: set[str]
    names: set[str]
    pyz: set[str] | None
    error: str = ""


def walk_code(code: Any, strings: set[str], names: set[str]) -> None:
    """Every string constant and name in *code* and its nested code objects."""
    names.update(code.co_names)
    for const in code.co_consts:
        if isinstance(const, str):
            strings.add(const)
        elif isinstance(const, (tuple, frozenset)):
            strings.update(item for item in const if isinstance(item, str))
        elif hasattr(const, "co_consts"):
            walk_code(const, strings, names)


def read_archive(exe: Path) -> Archive:
    """The entry script's constants/names and the PYZ module list - read from
    the file, without launching it."""
    try:
        from PyInstaller.archive.readers import CArchiveReader
    except Exception as exc:
        return Archive(set(), set(), None,
                       f"PyInstaller is not importable here ({type(exc).__name__}) - "
                       "run this with the build interpreter (.neural-debug-env)")
    try:
        reader = CArchiveReader(str(exe))
        strings: set[str] = set()
        names: set[str] = set()
        entries = 0
        for name, entry in reader.toc.items():
            if entry[4] == "s" and not name.startswith("pyi"):
                walk_code(marshal.loads(reader.extract(name)), strings, names)
                entries += 1
        if not entries:
            raise ValueError("no entry script in the archive")
        pyz_name = next((name for name, entry in reader.toc.items() if entry[4] == "z"), None)
        pyz = set(reader.open_embedded_archive(pyz_name).toc) if pyz_name else None
        return Archive(strings, names, pyz)
    except Exception as exc:
        return Archive(set(), set(), None,
                       f"could not read the executable's archive: {type(exc).__name__}: {exc}")


def entry_modes(archive: Archive, flags: Iterable[str]) -> dict[str, bool]:
    """Which headless flags the entry script handles, and whether it can run a
    multiprocessing child (freeze_support) - the only safe ways to launch it."""
    modes = {flag: flag in archive.strings for flag in flags}
    modes["spawn"] = "freeze_support" in archive.names
    return modes


# ── facts read from the source tree (AST; nothing is imported) ──────────────

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_constants(path: Path, *names: str) -> dict[str, Any]:
    """Literal module-level constants of *path* (missing ones are omitted)."""
    found: dict[str, Any] = {}
    try:
        tree = _parse(path)
    except (OSError, SyntaxError, UnicodeError):
        return found
    for node in tree.body:
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "frozenset" \
                        and value.args:
                    value = value.args[0]
                try:
                    found[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    return found


def build_config(source: Path) -> dict[str, Any]:
    """build_standalone's package lists and data, as the build will use them.

    ``X.append(...)`` at module level is unconditional; inside an ``if``/``try``
    it is conditional (mediapipe when installed, torch for a training build)."""
    lists = {"COLLECT_ALL": [], "HIDDEN_IMPORTS": [], "EXCLUDES": [], "DATAS": []}
    conditional: dict[str, list[Any]] = {name: [] for name in lists}
    tree = _parse(source / "build_standalone.py")
    lists.update({k: list(v) for k, v in module_constants(source / "build_standalone.py",
                                                            *lists).items()})
    top_level = {id(node.value) for node in tree.body
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in lists
                and node.func.attr in {"append", "extend"} and node.args):
            continue
        try:
            value = ast.literal_eval(node.args[0])
        except (ValueError, TypeError, SyntaxError):
            continue
        values = list(value) if node.func.attr == "extend" else [value]
        (lists if id(node) in top_level else conditional)[node.func.value.id].extend(values)
    return {"collect_all": lists["COLLECT_ALL"], "hidden_imports": lists["HIDDEN_IMPORTS"],
            "excludes": lists["EXCLUDES"], "datas": [tuple(item) for item in lists["DATAS"]],
            "conditional": {k.lower(): v for k, v in conditional.items() if v}}


def expected_collect_all(config: dict[str, Any],
                         installed: Callable[[str], bool] | None = None) -> list[str]:
    """--collect-all packages this build carries: the literal list, plus a
    conditional one that is installed here, minus anything excluded."""
    installed = installed or _installed
    excluded = set(config["excludes"])
    wanted = list(config["collect_all"])
    wanted += [name for name in config["conditional"].get("collect_all", [])
               if name not in wanted and installed(name)]
    return [name for name in wanted if name not in excluded]


def _installed(module: str) -> bool:
    """Is *module* installed in this interpreter? Finds the top-level spec,
    then walks the package folders on disk - nothing (not even a parent
    package, which namespace packages would need) is imported to answer."""
    import importlib.machinery
    parts = module.split(".")
    try:
        spec = importlib.util.find_spec(parts[0])
        locations = [str(path) for path in (spec.submodule_search_locations or [])] if spec else []
    except (ImportError, ValueError, KeyError):
        return False
    if spec is None:
        return False
    suffixes = importlib.machinery.all_suffixes()
    for index, part in enumerate(parts[1:], start=1):
        last = index == len(parts) - 1
        folders = [Path(location) / part for location in locations
                   if (Path(location) / part).is_dir()]
        if last and (folders or any((Path(location) / (part + suffix)).is_file()
                                    for location in locations for suffix in suffixes)):
            return True
        if not folders:
            return False
        locations = [str(folder) for folder in folders]
    return True


def mode_flags(source: Path) -> dict[str, str]:
    """The flag each headless mode is dispatched on (each module's FLAG)."""
    defaults = {"mcp": "--mcp-server", "face": "--face-host", "page": "--page-host",
                "forge": "--forge-verify"}
    files = {"mcp": "orion_core/mcp_bundled.py", "face": "orion_core/gui/face_host.py",
             "page": "orion_core/gui/page_host.py", "forge": "orion_core/forge_runtime.py"}
    return {key: module_constants(source / files[key], "FLAG").get("FLAG", defaults[key])
            for key in defaults}


def page_kinds(source: Path) -> dict[str, list[str]]:
    """Each page-host kind (page_host._KINDS) and the window parts it reports:
    the keys of the ``HostPage({...})`` its ``_build_<kind>`` returns."""
    path = source / "orion_core" / "gui" / "page_host.py"
    try:
        tree = _parse(path)
    except (OSError, SyntaxError, UnicodeError):
        return {}
    kinds: list[str] = []
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else \
            (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if getattr(target, "id", None) == "_KINDS" and isinstance(node.value, ast.Dict):
            kinds = [key.value for key in node.value.keys
                     if isinstance(key, ast.Constant) and isinstance(key.value, str)]
    parts: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_build_"):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "HostPage" \
                        and call.args and isinstance(call.args[0], ast.Dict):
                    parts[node.name[len("_build_"):]] = [
                        key.value for key in call.args[0].keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)]
    return {kind: parts.get(kind, []) for kind in kinds}


def bundled_servers(source: Path) -> list[str]:
    return sorted(module_constants(source / "orion_core" / "mcp_bundled.py", "BUNDLED")
                  .get("BUNDLED", {}))


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def orion_modules(source: Path) -> list[str]:
    """Every orion_core module, as a dotted name (never a __main__)."""
    names = {module_name(path, source) for path in (source / "orion_core").rglob("*.py")
             if "__pycache__" not in path.parts and path.stem != "__main__"}
    return sorted(names)


@dataclasses.dataclass
class ImportNeed:
    module: str
    names: set[str] = dataclasses.field(default_factory=set)
    kinds: set[str] = dataclasses.field(default_factory=set)
    sites: list[str] = dataclasses.field(default_factory=list)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
        (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _resolve_relative(package: str, level: int, module: str | None) -> str | None:
    if not package:
        return None
    parts = package.split(".")
    if level > len(parts):
        return None
    base = parts[:len(parts) - level + 1]
    return ".".join(base + ([module] if module else []))


def _valid_module(name: str) -> bool:
    return bool(name) and all(part.isidentifier() for part in name.split("."))


def scan_imports(tree: ast.AST, *, site: str, package: str, needs: dict[str, ImportNeed],
                 everything: bool = False) -> None:
    """Record the imports in *tree* that the freezer's entry graph does not
    prove: imports inside functions, imports guarded by try (a failure there
    is swallowed), and imports named by a string literal. *everything* also
    records plain module-level imports (a plugin is loaded from a file, so
    none of its imports is in the frozen graph).  TYPE_CHECKING blocks never
    run and are skipped."""

    def need(module: str | None, names: Iterable[str], kind: str, line: int) -> None:
        if not module or not _valid_module(module) or module == "__future__":
            return
        entry = needs.setdefault(module, ImportNeed(module))
        entry.names.update(name for name in names if name and name != "*")
        entry.kinds.add(kind)
        if len(entry.sites) < 5:
            entry.sites.append(f"{site}:{line}")

    def walk(node: ast.AST, in_function: bool, guarded: bool) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.orelse:
                walk(child, in_function, guarded)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kind = ("plugin" if everything else "function" if in_function
                    else "guarded" if guarded else "")
            if kind:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        need(alias.name, (), kind, node.lineno)
                elif node.level:
                    need(_resolve_relative(package, node.level, node.module),
                         (alias.name for alias in node.names), kind, node.lineno)
                else:
                    need(node.module, (alias.name for alias in node.names), kind, node.lineno)
            return
        if isinstance(node, ast.Call):
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            spec = DYNAMIC_IMPORT_CALLS.get(called or "")
            if spec is not None:
                index, attr_index = spec
                if len(node.args) > index and isinstance(node.args[index], ast.Constant) \
                        and isinstance(node.args[index].value, str) \
                        and not node.args[index].value.startswith("."):
                    attrs: list[str] = []
                    if attr_index is not None:
                        candidates = node.args[attr_index:attr_index + 1] + \
                            [kw.value for kw in node.keywords if kw.arg == "attr"]
                        attrs = [c.value for c in candidates
                                 if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                    need(node.args[index].value, attrs, "dynamic", node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            in_function = True
        try_types = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
        child_guard = guarded or isinstance(node, try_types)
        for child in ast.iter_child_nodes(node):
            walk(child, in_function, child_guard)

    walk(tree, False, False)


def lazy_imports(source: Path, plugin_files: Iterable[Path] = ()) -> dict[str, ImportNeed]:
    """Everything ORION imports lazily or by name, derived from the source."""
    needs: dict[str, ImportNeed] = {}
    files = [source / "orion.py"] + sorted(path for path in (source / "orion_core").rglob("*.py")
                                           if "__pycache__" not in path.parts)
    for path in files:
        if not path.is_file():
            continue
        try:
            tree = _parse(path)
        except (SyntaxError, UnicodeError) as exc:
            raise ValueError(f"cannot parse {path}: {exc}") from exc
        name = module_name(path, source)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        scan_imports(tree, site=path.relative_to(source).as_posix(), package=package, needs=needs)
    for path in plugin_files:
        try:
            tree = _parse(path)
        except (OSError, SyntaxError, UnicodeError):
            continue
        scan_imports(tree, site=f"shipped_plugins/{path.name}", package="", needs=needs,
                     everything=True)
    return needs


def platform_absent_stdlib(module: str) -> bool:
    """A standard-library module this platform does not have (fcntl, termios...)."""
    top = module.split(".")[0]
    return top in getattr(sys, "stdlib_module_names", ()) and not _installed(top)


def optional_reason(module: str | None, excludes: Iterable[str] = (), *,
                    installed: Callable[[str], bool] = _installed) -> str:
    """Why *module* may be absent from the installed app, or '' if it must not be."""
    if not module:
        return ""
    if module.split(".", 1)[0] == "piper" and installed("piper"):
        # Prepared-only when no engine was available to the builder. An
        # installed engine must survive freezing, including its native data.
        return ""
    table = dict(OPTIONAL_BY_DESIGN)
    for name in excludes:
        table.setdefault(name, "excluded from the build on purpose (build_standalone.EXCLUDES)")
    parts = module.split(".")
    for length in range(len(parts), 0, -1):
        reason = table.get(".".join(parts[:length]))
        if reason:
            return reason
    return ""


def resource_requests(source: Path) -> list[tuple[tuple[str, ...], str]]:
    """Every resource_path(...) in orion_core whose arguments are constants
    (also ``*NAME`` of a module tuple and ``*NAME.split('/')`` of a module str)."""
    found: list[tuple[tuple[str, ...], str]] = []
    for path in sorted((source / "orion_core").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = _parse(path)
        except (OSError, SyntaxError, UnicodeError):
            continue
        constants: dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError, SyntaxError):
                            pass
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and
                    (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
                    == "resource_path"):
                continue
            parts: list[str] = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(arg.value)
                elif isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name) and \
                        isinstance(constants.get(arg.value.id), (tuple, list)):
                    parts.extend(str(p) for p in constants[arg.value.id])
                elif (isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Call)
                      and isinstance(arg.value.func, ast.Attribute)
                      and arg.value.func.attr == "split"
                      and isinstance(arg.value.func.value, ast.Name)
                      and isinstance(constants.get(arg.value.func.value.id), str)
                      and len(arg.value.args) == 1
                      and isinstance(arg.value.args[0], ast.Constant)):
                    parts.extend(constants[arg.value.func.value.id].split(arg.value.args[0].value))
                else:
                    parts = []
                    break
            if parts:
                found.append((tuple(parts), f"{path.relative_to(source).as_posix()}:{node.lineno}"))
    return found


def three_entry_files(source: Path) -> list[str]:
    """The three.js modules the face page imports itself (face3d._ENTRY_MODULES)."""
    found = module_constants(source / "orion_core" / "gui" / "face3d.py",
                             "THREE_DIR", "_ENTRY_MODULES")
    base = found.get("THREE_DIR", "assets/three")
    return [f"{base}/{item}" for item in found.get("_ENTRY_MODULES", ())]


def forge_texts(source: Path) -> dict[str, Any]:
    """What the Forge stages, from sandbox.py: the conformance probe's main
    block, its pytest wrapper, and its time budgets."""
    sandbox = source / "orion_core" / "sandbox.py"
    found = module_constants(sandbox, "_PROBE_MAIN", "CONFORMANCE_SCRIPT_NAME",
                             "BASE_TIMEOUT_S", "CONFORMANCE_TIMEOUT_CAP_S")
    wrapper = next((node.value for node in ast.walk(_parse(sandbox))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith("import pytest")), None)
    # The shared fixtures the sandbox stages beside every tool (forge_fixtures.
    # fixture_source()); the probe and the pytest wrapper import them, so a
    # check that leaves them out fails where the Forge itself would not.
    fixtures = module_constants(source / "orion_core" / "forge_fixtures.py",
                                "FIXTURE_MODULE_NAME", "_SOURCE")
    return {"fixture_name": fixtures.get("FIXTURE_MODULE_NAME"),
            "fixture_source": str(fixtures.get("_SOURCE") or "").lstrip("\n"),
            "probe_main": found.get("_PROBE_MAIN"), "pytest_wrapper": wrapper,
            "conformance_name": found.get("CONFORMANCE_SCRIPT_NAME", "_orion_conformance.py"),
            "budget_s": float(found.get("BASE_TIMEOUT_S", 15.0)),
            "conformance_cap_s": float(found.get("CONFORMANCE_TIMEOUT_CAP_S", 30.0))}


def mcp_protocol(source: Path) -> tuple[str, frozenset[str]]:
    found = module_constants(source / "orion_core" / "mcp_host.py",
                             "_PROTOCOL_VERSION", "_SUPPORTED_LEGACY_VERSIONS")
    version = found.get("_PROTOCOL_VERSION", "2024-11-05")
    return version, frozenset(found.get("_SUPPORTED_LEGACY_VERSIONS", {version}))


def host_start_limit_s(source: Path) -> float:
    """How long the app lets a page/face host take to show its window
    (RemoteSurface.START_TIMEOUT_MS) before it falls back to drawing in-process."""
    for name in ("page_host.py", "face_host.py"):
        try:
            tree = _parse(source / "orion_core" / "gui" / name)
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "START_TIMEOUT_MS"
                                                    for t in node.targets):
                try:
                    return float(ast.literal_eval(node.value)) / 1000.0
                except (ValueError, TypeError):
                    break
    return 15.0


# ── data in the bundle (static) ──────────────────────────────────────────────

def package_data_files(package: str) -> tuple[list[str], str]:
    """Non-Python files of *package* in THIS interpreter's environment, as paths
    relative to site-packages - what --collect-all is asked to carry."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError) as exc:
        return [], f"cannot inspect ({type(exc).__name__})"
    if spec is None:
        return [], "not installed in this interpreter"
    locations = list(spec.submodule_search_locations or [])
    if not locations:
        return [], "a single module, no package data"
    files: set[str] = set()
    for location in locations:
        base = Path(location)
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts \
                    and path.suffix not in {".py", ".pyc", ".pyo"}:
                files.add(path.relative_to(base.parent).as_posix())
    return sorted(files), ""


def check_datas(config: dict[str, Any], source: Path, internal: Path) -> Check:
    missing: list[str] = []
    differing: list[str] = []
    total = 0
    for src, dest in config["datas"]:
        origin = source / src
        if origin.is_dir():
            pairs = [(path, Path(dest) / path.relative_to(origin))
                     for path in origin.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix != ".pyc"]
        elif origin.is_file():
            pairs = [(origin, Path(dest) / origin.name)]
        else:
            missing.append(f"{src} (missing from the SOURCE tree)")
            continue
        for path, relative in pairs:
            total += 1
            target = internal / relative
            if not target.is_file():
                missing.append(relative.as_posix())
            elif target.stat().st_size != path.stat().st_size:
                differing.append(relative.as_posix())
    notes = ([f"{len(differing)} file(s) differ in size from the source (changed since "
              f"the build): {_brief(differing)}"] if differing else [])
    if missing:
        return Check("data:build-datas", False,
                     f"{len(missing)} of {total} DATAS file(s) missing: {_brief(missing)}",
                     data={"failures": missing, "notes": notes})
    return Check("data:build-datas", True,
                 f"{total} file(s) from {len(config['datas'])} DATAS entr(ies) present"
                 + (f"; {len(differing)} changed since the build" if differing else ""),
                 data={"notes": notes})


def check_collect_all(packages: Sequence[str], internal: Path,
                      lister: Callable[[str], tuple[list[str], str]] = package_data_files) -> Check:
    missing: list[str] = []
    notes: list[str] = []
    total = 0
    for package in packages:
        files, why = lister(package)
        if why:
            notes.append(f"{package}: {why}")
            continue
        absent = [name for name in files if not (internal / name).exists()]
        total += len(files)
        if absent:
            missing.append(f"{package}: {len(absent)} of {len(files)} missing "
                           f"(e.g. {', '.join(absent[:3])})")
    if missing:
        return Check("data:collect-all", False, _brief(missing),
                     data={"failures": missing, "notes": notes})
    return Check("data:collect-all", True,
                 f"{total} data file(s) of {len(packages) - len(notes)} --collect-all "
                 f"package(s) present" + (f"; {len(notes)} not checked" if notes else ""),
                 data={"notes": notes})


def check_resources(source: Path, internal: Path) -> Check:
    requests = resource_requests(source)
    wanted = [(tuple(item.split("/")), "face3d._ENTRY_MODULES") for item in three_entry_files(source)]
    missing = []
    for parts, site in requests + wanted:
        if not ((internal.joinpath(*parts)).exists() or (internal.parent.joinpath(*parts)).exists()):
            missing.append(f"{'/'.join(parts)} ({site})")
    if missing:
        return Check("data:resources", False, _brief(missing), data={"failures": missing})
    return Check("data:resources", True,
                 f"{len(requests)} resource_path() request(s) + {len(wanted)} three.js entry "
                 f"module(s) present")


def check_critical(internal: Path, exe_dir: Path) -> Check:
    missing = []
    found = []
    for pattern, why in CRITICAL_DATA:
        matches = sorted(internal.glob(pattern))
        if matches:
            found.append(pattern)
        else:
            missing.append(f"{pattern} - {why}")
    notes = []
    stockfish = exe_dir / "engines" / "stockfish.exe"
    notes.append("Stockfish is not bundled by design (chess_engine downloads it into "
                 f"engines/ on first use): {'present' if stockfish.is_file() else 'absent'}")
    notes.append("the sentence encoder is not bundled by design (semantic.py downloads it "
                 "into CONFIG_DIR/models on first use)")
    if missing:
        return Check("data:critical", False, _brief(missing),
                     data={"failures": missing, "notes": notes})
    return Check("data:critical", True, f"{len(found)} named data item(s) present",
                 data={"notes": notes})


# ── code that runs INSIDE the frozen interpreter ────────────────────────────

PROBE_PRELUDE = r'''
import io, json, os, sys, time

def _ensure_stdio():
    # A windowed executable may start with no Python-level std streams.
    for fd, name in ((1, "stdout"), (2, "stderr")):
        if getattr(sys, name, None) is None:
            try:
                stream = io.TextIOWrapper(io.FileIO(fd, "wb", closefd=False),
                                          encoding="utf-8", errors="replace",
                                          line_buffering=True)
            except (OSError, ValueError):
                stream = open(os.devnull, "w", encoding="utf-8")
            setattr(sys, name, stream)

_ensure_stdio()

def _error(exc):
    return ("%s: %s" % (type(exc).__name__, exc))[:600]

def _finish(path, record):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)          # never wait on threads an imported module started
'''

IMPORT_PROBE = PROBE_PRELUDE + r'''
import importlib

def _import(entry):
    module = importlib.import_module(entry["module"])
    missing = []
    for name in entry.get("names") or ():
        if hasattr(module, name):
            continue
        try:
            importlib.import_module(entry["module"] + "." + name)
        except Exception as exc:
            missing.append("%s (%s)" % (name, _error(exc)))
    if missing:
        raise ImportError("cannot import name " + "; ".join(missing))
    return ""

DATA = {}

def data(key):
    def register(fn):
        DATA[key] = fn
        return fn
    return register

@data("tz-europe-london")
def _tz():
    import datetime, zoneinfo
    zone = zoneinfo.ZoneInfo("Europe/London")
    return "BST offset %s" % datetime.datetime(2026, 7, 1, 12, tzinfo=zone).utcoffset()

@data("onnxruntime")
def _ort():
    import onnxruntime
    return "%s, %s" % (onnxruntime.__version__, ",".join(onnxruntime.get_available_providers()))

@data("whisper-vad")
def _vad():
    from faster_whisper.vad import get_vad_model
    return type(get_vad_model()).__name__

@data("voiceprint-encoder")
def _voiceprint():
    import onnxruntime
    from orion_core.voiceprint import model_path
    path = model_path()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return "%s inputs=%s" % (path.name, [item.name for item in session.get_inputs()])

@data("three-js")
def _three():
    from orion_core.gui.face3d import local_three
    root = local_three()
    if root is None:
        raise RuntimeError("the vendored three.js graph is incomplete (the face would use a CDN)")
    return str(root)

@data("forge-contract")
def _contract():
    from orion_core.sandbox import probe_source
    text = probe_source()
    if "def contract_problems" not in text:
        raise RuntimeError("probe_source() has no contract rules")
    return "%d chars" % len(text)

@data("jsonschema-metaschemas")
def _jsonschema():
    import jsonschema
    jsonschema.Draft202012Validator.check_schema({"type": "object"})
    return "Draft 2020-12 metaschema loaded"

@data("rapidocr-models")
def _rapidocr():
    from rapidocr_onnxruntime import RapidOCR
    RapidOCR()
    return "models loaded"

@data("kokoro-voice")
def _kokoro():
    from orion_core.voice_kokoro import self_check
    return self_check(require_bundled=True)

@data("onnxruntime-directml")
def _directml():
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    if "DmlExecutionProvider" not in providers:
        raise RuntimeError("DmlExecutionProvider is not available: %s" % ",".join(providers))
    return ",".join(providers)

def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        job = json.load(stream)
    out = open(sys.argv[2], "a", encoding="utf-8")
    try:
        import faulthandler
        faulthandler.enable(file=open(sys.argv[2] + ".fault", "a"))
    except Exception:
        pass
    entries = job["entries"]
    for index in range(int(job.get("start", 0)), len(entries)):
        entry = entries[index]
        out.write(json.dumps({"i": index, "begin": True}) + "\n")
        out.flush()
        try:
            # A heartbeat: a traceback an imported module merely LOGS is then
            # not the last thing on stderr, so it is never taken for a crash.
            sys.stderr.write("[parity] entry %d\n" % index)
            sys.stderr.flush()
        except Exception:
            pass
        started = time.perf_counter()
        try:
            if entry["kind"] == "data":
                note = DATA[entry["key"]]()
            else:
                note = _import(entry)
            record = {"i": index, "ok": True, "note": str(note or "")[:300]}
        except BaseException as exc:
            missing = exc.name if isinstance(exc, ImportError) else None
            record = {"i": index, "ok": False, "error": _error(exc), "missing": missing}
        record["s"] = round(time.perf_counter() - started, 3)
        out.write(json.dumps(record) + "\n")
        out.flush()
    out.write(json.dumps({"done": True}) + "\n")
    out.close()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)

main()
'''

SPEC_PROBE = PROBE_PRELUDE + r'''
result = {"executable": sys.executable, "frozen": bool(getattr(sys, "frozen", False))}
try:
    from orion_core import mcp_bundled
    result["bundled"] = {name: [mcp_bundled.launch_spec(name)[0],
                                list(mcp_bundled.launch_spec(name)[1])]
                         for name in mcp_bundled.BUNDLED}
except BaseException as exc:
    result["bundled_error"] = _error(exc)
try:
    from orion_core import mcp_host
    servers = mcp_host.load_config().get("servers") or {}
    result["host"] = {name: {"command": spec.get("command"), "args": list(spec.get("args") or []),
                             "env": {k: str(v) for k, v in (spec.get("env") or {}).items()}}
                      for name, spec in servers.items()
                      if isinstance(spec, dict) and name in result.get("bundled", {})}
except BaseException as exc:
    result["host_error"] = _error(exc)
_finish(sys.argv[1], result)
'''

PIP_PROBE = PROBE_PRELUDE + r'''
import importlib, subprocess
out, wheels, package = sys.argv[1], sys.argv[2], sys.argv[3]
result = {"frozen": bool(getattr(sys, "frozen", False))}
try:
    from orion_core import forge_runtime
except BaseException as exc:
    result["error"] = "this build has no orion_core.forge_runtime (%s)" % _error(exc)
    try:
        from orion_core import dependencies
        owner = next(value for value in vars(dependencies).values()
                     if isinstance(value, type) and hasattr(value, "_install_packages"))
        code = owner._install_packages.__code__
        strings = set()
        def walk(code):
            for const in code.co_consts:
                if isinstance(const, str):
                    strings.add(const)
                elif hasattr(const, "co_consts"):
                    walk(const)
        walk(code)
        if {"-m", "pip"} <= strings:
            result["legacy"] = ("its Forge installs with [sys.executable, '-m', 'pip', ...] - "
                                "in the installed app that is ORION.exe, not a Python with pip")
    except BaseException as exc:
        result["legacy"] = "could not inspect orion_core.dependencies: " + _error(exc)
    _finish(out, result)
config = os.path.realpath(os.environ.get("ORION_CONFIG_DIR", ""))
target = os.path.realpath(str(forge_runtime.packages_dir()))
result["target"] = target
if not config or os.path.commonpath([config, target]) != config:
    result["error"] = "refusing: the package folder %s is not inside the temporary config" % target
    _finish(out, result)
result["python"] = forge_runtime.external_python()
command = forge_runtime.pip_install_command([package])
result["command"] = command
if command is None:
    result["error"] = "no Python found on this PC to install with (set ORION_FORGE_PYTHON)"
    _finish(out, result)
# Never install anywhere but the temporary config (a non-frozen runtime would
# install into its own environment, which this check must not touch).
where = command[command.index("--target") + 1] if "--target" in command else ""
if not where or os.path.commonpath([config, os.path.realpath(where)]) != config:
    result["error"] = "refusing: the install command does not --target the temporary config"
    _finish(out, result)
command = list(command) + ["--no-index", "--find-links", wheels,
                           "--disable-pip-version-check", "--no-cache-dir"]
started = time.perf_counter()
try:
    done = subprocess.run(command, capture_output=True, text=True, errors="replace",
                          timeout=150, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result["returncode"] = done.returncode
    result["pip_tail"] = (done.stderr or done.stdout or "")[-800:]
except BaseException as exc:
    result["error"] = "pip did not run: " + _error(exc)
result["pip_s"] = round(time.perf_counter() - started, 2)
if "error" not in result:
    try:
        importlib.invalidate_caches()
        forge_runtime.add_packages_to_path()
        module = importlib.import_module(package)
        result["imported"] = module.__file__
        result["value"] = getattr(module, "VALUE", None)
    except BaseException as exc:
        result["import_error"] = _error(exc)
_finish(out, result)
'''

IMPORT_CHECK = PROBE_PRELUDE + r'''
import importlib
result = {}
try:
    module = importlib.import_module(sys.argv[2])
    result = {"ok": True, "file": module.__file__, "value": getattr(module, "VALUE", None)}
except BaseException as exc:
    result = {"ok": False, "error": _error(exc)}
_finish(sys.argv[1], result)
'''


class FrozenRunner:
    """Runs a Python script inside the executable's own interpreter.

    ``--forge-verify`` is the Forge's own route and is preferred. An older
    build without it can still run code as a multiprocessing child (its entry
    script calls freeze_support() before anything else), which is exactly how
    the app's own multiprocessing children run. With neither, the build is
    never launched: anything else would start the full application."""

    def __init__(self, launcher: Sequence[str], modes: dict[str, bool], forge_flag: str) -> None:
        #: How the app is started: [ORION.exe], or anything handling the same flags.
        self.launcher = [str(part) for part in launcher]
        self.forge_flag = forge_flag
        self.mechanism = ("forge-verify" if modes.get(forge_flag) else
                          "spawn" if modes.get("spawn") and len(self.launcher) == 1 else "")

    def run(self, script: Path, args: Sequence[str], *, cwd: Path, env: dict[str, str],
            timeout: float) -> RunResult:
        if self.mechanism == "forge-verify":
            result = run_bounded([*self.launcher, self.forge_flag, script, *args],
                                 cwd=cwd, env=env, timeout=timeout)
        elif self.mechanism == "spawn":
            result = self._spawn(script, args, cwd=cwd, env=env, timeout=timeout)
        else:
            return RunResult(None, "", "", 0.0, mechanism="",
                             error="this build has no way to run code headlessly "
                                   "(no --forge-verify, no freeze_support) - not launched")
        result.mechanism = self.mechanism
        return result

    def _spawn(self, script: Path, args: Sequence[str], *, cwd: Path, env: dict[str, str],
               timeout: float) -> RunResult:
        """The multiprocessing 'spawn' handshake by hand: the child steals the
        read end of a pipe from this process and unpickles its preparation
        data and a Process whose target is runpy.run_path(script)."""
        import _winapi
        import msvcrt
        import multiprocessing
        import runpy
        from multiprocessing import context, reduction

        rhandle, whandle = _winapi.CreatePipe(None, 0)
        process = multiprocessing.get_context("spawn").Process(
            target=runpy.run_path, args=(str(script),), kwargs={"run_name": "__main__"},
            name="orion-parity-probe")
        # Deliberately minimal: no sys_path (the child keeps the bundle's own
        # path, so nothing resolves from this interpreter) and no main module.
        prep = {"name": "orion-parity-probe", "authkey": bytes(process.authkey),
                "dir": str(cwd), "sys_argv": [str(script), *map(str, args)]}
        command = [self.launcher[0], "--multiprocessing-fork", f"parent_pid={os.getpid()}",
                   f"pipe_handle={int(rhandle)}"]

        def feed(_proc: subprocess.Popen) -> None:
            with open(msvcrt.open_osfhandle(whandle, 0), "wb", closefd=True) as to_child:
                context.set_spawning_popen(self)        # lets the authkey pickle
                try:
                    reduction.dump(prep, to_child)
                    reduction.dump(process, to_child)
                finally:
                    context.set_spawning_popen(None)

        return run_bounded(command, cwd=cwd, env=env, timeout=timeout, after_start=feed)


def run_probe_script(runner: FrozenRunner, source_text: str, *, work: Path, name: str,
                     args: Sequence[str], env: dict[str, str], timeout: float,
                     ) -> tuple[RunResult, dict[str, Any] | None]:
    """Write *source_text* as a script, run it frozen, read its JSON result."""
    work.mkdir(parents=True, exist_ok=True)
    script = work / f"_orion_parity_{name}.py"
    script.write_text(source_text, encoding="utf-8")
    out = work / f"{name}.result.json"
    result = runner.run(script, [str(out), *args], cwd=work, env=env, timeout=timeout)
    try:
        return result, json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result, None


def _run_failure(result: RunResult) -> str:
    if result.error:
        return result.error
    if result.timed_out:
        return f"did not finish within {result.seconds:.0f}s (killed with its process tree)"
    summary = child_failure(result.returncode, result.stderr.splitlines())
    tail = _one_line(result.stderr or result.stdout, 400)
    return summary + (f"; output: {tail}" if tail else " with no output")


# ── frozen imports ───────────────────────────────────────────────────────────

def import_jobs(source: Path, plugin_files: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """The probe's work list: data probes, every orion_core module, then every
    lazy/dynamic import (minus stdlib modules this platform does not have)."""
    entries: list[dict[str, Any]] = [{"kind": "data", "key": key, "group": "data"}
                                     for key in DATA_PROBES]
    entries += [{"kind": "import", "module": name, "names": [], "group": "orion_core"}
                for name in orion_modules(source)]
    skipped = []
    for module, need in sorted(lazy_imports(source, plugin_files).items()):
        if platform_absent_stdlib(module):
            skipped.append(module)
            continue
        entries.append({"kind": "import", "module": module, "names": sorted(need.names),
                        "group": "dynamic", "kinds": sorted(need.kinds), "sites": need.sites})
    return entries, skipped


def run_import_probe(runner: FrozenRunner, entries: list[dict[str, Any]], *, work: Path,
                     env: dict[str, str], budget_s: float, max_restarts: int = 6,
                     ) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Run the import probe, resuming after a crash or hang at the entry that
    caused it (recorded as that entry's failure). Returns one result per entry."""
    work.mkdir(parents=True, exist_ok=True)
    script = work / "_orion_parity_imports.py"
    script.write_text(IMPORT_PROBE, encoding="utf-8")
    results: list[dict[str, Any] | None] = [None] * len(entries)
    notes: list[str] = []
    deadline = time.monotonic() + budget_s
    start = 0
    for attempt in range(max_restarts + 1):
        remaining = deadline - time.monotonic()
        if start >= len(entries) or remaining <= 5:
            break
        job = work / f"imports.job{attempt}.json"
        job.write_text(json.dumps({"entries": entries, "start": start}), encoding="utf-8")
        out = work / f"imports.out{attempt}.jsonl"
        run = runner.run(script, [str(job), str(out)], cwd=work, env=env, timeout=remaining)
        current, done = None, False
        try:
            lines = out.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("done"):
                done = True
            elif record.get("begin"):
                current = record["i"]
            elif "i" in record:
                results[record["i"]] = record
                current = None
        notes.append(f"run {attempt + 1} ({run.mechanism or 'not run'}): "
                     f"{'completed' if done else _run_failure(run)} in {run.seconds:.1f}s")
        if done or run.returncode is None and run.error.startswith("could not start"):
            break
        if current is None:
            # Died before or between entries: nothing to blame, nothing to resume.
            break
        fault = Path(str(out) + ".fault")
        crash = _one_line(fault.read_text(encoding="utf-8", errors="replace"), 300) \
            if fault.is_file() else ""
        results[current] = {"i": current, "ok": False, "missing": None,
                            "error": ("hung - the probe was killed" if run.timed_out else
                                      run.error if run.error else
                                      f"the process died during this import (exit "
                                      f"{run.returncode}){': ' + crash if crash else ''}")}
        start = current + 1
    return results, notes


def _source_file(source: Path, module: str) -> Path | None:
    base = source / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def newer_than_build(source: Path, module: str, names: Iterable[str],
                     built_at: float | None) -> bool:
    """Did the orion_core source this import needs change after the build?
    Then a failure is the build being older than the source (drift), not a
    packaging fault - still a parity gap, labelled so real faults stand out."""
    if not built_at or not module.startswith("orion_core"):
        return False
    files = [_source_file(source, module)] + [_source_file(source, f"{module}.{name}")
                                              for name in names]
    return any(path is not None and path.stat().st_mtime > built_at for path in files)


def classify_imports(entries: list[dict[str, Any]], results: list[dict[str, Any] | None],
                     *, excludes: Iterable[str], archive: Archive, built_at: float | None,
                     source: Path, installed: Callable[[str], bool] = _installed,
                     ) -> dict[str, dict[str, Any]]:
    """Group probe results into data / orion_core / dynamic outcomes."""
    groups: dict[str, dict[str, Any]] = {
        name: {"total": 0, "ok": 0, "failures": [], "drift": [], "optional": [],
               "not_run": 0, "slow": []}
        for name in ("data", "orion_core", "dynamic")}
    for entry, result in zip(entries, results):
        group = groups[entry["group"]]
        group["total"] += 1
        label = entry.get("key") or entry["module"]
        if result is None:
            group["not_run"] += 1
            continue
        if result.get("ok"):
            group["ok"] += 1
            if result.get("s", 0) >= 5:
                group["slow"].append(f"{label} {result['s']:.1f}s")
            continue
        error = result.get("error", "")
        missing = result.get("missing")
        if entry["kind"] == "data":
            group["failures"].append(f"{label}: {error}")
            continue
        # An installed Piper engine is part of this build. Its failed import
        # stays failed even when a transitive dependency is optional elsewhere.
        piper_required = entry["module"].split(".", 1)[0] == "piper" and installed("piper")
        why = "" if piper_required else (
            optional_reason(entry["module"], excludes, installed=installed)
            or optional_reason(missing, excludes, installed=installed))
        if why:
            group["optional"].append(f"{label}: {why}")
            continue
        notes = []
        if missing and not missing.startswith("orion_core") and not installed(missing):
            notes.append("not installed in the build environment either")
        drift = newer_than_build(source, entry["module"], entry.get("names") or (), built_at)
        if drift:
            notes.append("source newer than this build")
        elif entry["group"] == "orion_core" and archive.pyz is not None \
                and entry["module"] not in archive.pyz:
            notes.append("not in the executable's archive")
        if entry.get("sites"):
            notes.append("used at " + entry["sites"][0])
        line = f"{label}: {error}" + (f" [{'; '.join(notes)}]" if notes else "")
        group["drift" if drift else "failures"].append(line)
    return groups


def import_checks(groups: dict[str, dict[str, Any]], *, mechanism: str, run_notes: list[str],
                  skipped: list[str], seconds: float) -> list[Check]:
    titles = {"data": ("data:runtime", "frozen data probe(s) loaded"),
              "orion_core": ("imports:orion_core", "orion_core module(s) import"),
              "dynamic": ("imports:lazy", "lazy/dynamic import(s) resolve")}
    checks = []
    for key, (name, what) in titles.items():
        group = groups[key]
        # Packaging faults first, then the build being older than the source.
        failures = list(group["failures"]) + list(group["drift"])
        if group["not_run"]:
            failures.append(f"{group['not_run']} entr(ies) never ran ({_brief(run_notes, 2)})")
        notes = [f"optional, absent: {item}" for item in group["optional"]]
        notes += [f"slow: {item}" for item in group["slow"]]
        if key == "dynamic" and skipped:
            notes.append(f"not on this platform (stdlib): {', '.join(skipped)}")
        notes += [f"mechanism: {mechanism}"] + run_notes
        if failures:
            drift = (f" ({len(group['drift'])} only because the source is newer than this build)"
                     if group["drift"] else "")
            detail = f"{len(failures)} of {group['total']} failed{drift}: {_brief(failures, 3)}"
        else:
            detail = f"{group['ok']} of {group['total']} {what}"
            if group["optional"]:
                detail += f"; {len(group['optional'])} optional absent"
        checks.append(Check(name, not failures, detail, seconds,
                            data={"failures": failures, "notes": notes}))
    return checks


# ── MCP ──────────────────────────────────────────────────────────────────────

def mcp_handshake(cmd: Sequence[str], *, env: dict[str, str], cwd: Path | None = None,
                  timeout: float = 60.0, protocol: str = "2024-11-05",
                  supported: frozenset[str] = frozenset({"2024-11-05"}),
                  exit_timeout: float = 10.0) -> dict[str, Any]:
    """initialize -> notifications/initialized -> tools/list over stdio
    (newline-delimited JSON-RPC), as ORION's MCP host does it."""
    started = time.monotonic()
    report: dict[str, Any] = {"ok": False, "tools": 0, "error": ""}
    try:
        proc = subprocess.Popen([str(part) for part in cmd], cwd=str(cwd) if cwd else None,
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
    except OSError as exc:
        report["error"] = f"could not start: {exc}"
        return report
    err = LineCollector(proc.stderr)
    lines: queue.Queue = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=reader, daemon=True, name="mcp-read").start()
    deadline = started + timeout
    state = {"malformed": 0}

    def send(message: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        proc.stdin.flush()

    def reply(request_id: int) -> dict[str, Any]:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no reply within {timeout:.0f}s")
            try:
                raw = lines.get(timeout=min(left, 0.25))
            except queue.Empty:
                crash = err.crashed()
                if crash:
                    raise ConnectionError(f"the server crashed during startup ({crash})")
                continue
            if raw is None:
                raise ConnectionError("server closed the connection")
            try:
                message = json.loads(raw.decode("utf-8", errors="replace").strip())
            except ValueError:
                state["malformed"] += 1
                continue
            if not isinstance(message, dict) or "method" in message:
                continue                        # a notification or a server request
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"error reply: {message['error']}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise TypeError("protocol reply result is not an object")
            return result

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": protocol, "capabilities": {},
                         "clientInfo": {"name": "ORION parity check", "version": "1.0"}}})
        result = reply(1)
        version = result.get("protocolVersion")
        if version not in supported:
            raise RuntimeError(f"unsupported protocolVersion {version!r}")
        if not isinstance(result.get("capabilities"), dict):
            raise RuntimeError("initialize result has no capabilities object")
        server = result.get("serverInfo") or {}
        if not isinstance(server, dict):
            raise TypeError("initialize serverInfo is not an object")
        report["protocol"] = version
        report["server"] = server.get("name")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = reply(2)
        tools = listed.get("tools")
        if not isinstance(tools, list) or any(
                not isinstance(tool, dict) or not str(tool.get("name") or "").strip()
                for tool in tools):
            raise RuntimeError("tools/list returned an invalid tool catalogue")
        report.update(ok=True, tools=len(tools),
                      tool_names=sorted(str(tool["name"]) for tool in tools))
    except (OSError, TimeoutError, RuntimeError, TypeError, ValueError) as exc:
        report["error"] = str(exc) + (f" ({state['malformed']} non-JSON line(s) on stdout)"
                                      if state["malformed"] else "")
    report["seconds"] = round(time.monotonic() - started, 2)
    # Close stdin, as the host does at shutdown; a server should then exit.
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=0.5 if report["error"] else exit_timeout)
        report["exited_on_eof"] = True
    except subprocess.TimeoutExpired:
        report["exited_on_eof"] = False
        kill_tree(proc)                      # also dismisses a windowed crash dialog
    report["returncode"] = proc.returncode
    time.sleep(0.05)                          # let the drain thread take the last lines
    report["exit"] = child_failure(proc.returncode, err.lines)
    report["stderr_tail"] = err.tail()
    return report


def check_mcp(name: str, cmd: Sequence[str], spec_note: str, *, env: dict[str, str],
              protocol: str, supported: frozenset[str], timeout: float,
              exit_timeout: float = 10.0) -> Check:
    report = mcp_handshake(cmd, env=env, timeout=timeout, protocol=protocol, supported=supported,
                           exit_timeout=exit_timeout)
    notes = [spec_note, f"command: {' '.join(map(str, cmd))}"]
    if report.get("ok"):
        if not report.get("exited_on_eof"):
            report.update(ok=False, error=f"did not exit within {exit_timeout:.0f}s of stdin "
                                          "closing (killed with its process tree)")
        elif report.get("returncode") != 0:
            report.update(ok=False, error=f"server exited with code {report.get('returncode')}")
        else:
            return Check(f"mcp:{name}", True, f"{report['tools']} tool(s), protocol "
                         f"{report.get('protocol')}", report.get("seconds", 0.0),
                         data={"notes": notes, "report": report})
    detail = f"handshake failed - {report['error']}"
    if report.get("exit") and report["exit"] not in report["error"]:
        detail += f" [{report['exit']}]"
    if report.get("stderr_tail"):
        detail += f"; stderr: {report['stderr_tail']}"
    return Check(f"mcp:{name}", False, detail, report.get("seconds", 0.0),
                 data={"failures": [detail], "notes": notes, "report": report})


# ── page / face hosts ────────────────────────────────────────────────────────

#: The native window class Qt 6 gives a top-level QWindow on Windows
#: (e.g. Qt6100QWindowToolSaveBits) - what QWindow.fromWinId can embed.
QT_WINDOW_CLASS = "Qt6*QWindow*"


def window_facts(hwnd: int) -> dict[str, Any]:
    """What Windows itself says about native window *hwnd*: whether it exists,
    its class, its owning process, and whether it is a top-level window."""
    if os.name != "nt":
        return {"exists": None}
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetDesktopWindow.argtypes = []
    user32.GetDesktopWindow.restype = wintypes.HWND
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    handle = wintypes.HWND(int(hwnd))
    if not user32.IsWindow(handle):
        return {"exists": False}
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(handle, buffer, 256)
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    ws_child = 0x40000000
    style = user32.GetWindowLongW(handle, -16) & 0xFFFFFFFF          # GWL_STYLE
    parent = user32.GetAncestor(handle, 1)                             # GA_PARENT
    desktop = user32.GetDesktopWindow()
    return {"exists": True, "class": buffer.value, "pid": int(pid.value),
            "top_level": not (style & ws_child) and (parent or 0) == (desktop or 0),
            "visible": bool(user32.IsWindowVisible(handle))}


def parent_pids() -> dict[int, int]:
    """pid -> parent pid for every process (a Windows Toolhelp snapshot)."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        getattr(kernel32, name).argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        getattr(kernel32, name).restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)             # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return {}
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(ProcessEntry)
    parents: dict[int, int] = {}
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return parents


def descends_from(pid: int, ancestor: int, parents: dict[int, int], depth: int = 4) -> bool:
    """Is *pid* the process *ancestor*, or started by it (a venv's python.exe
    runs the real interpreter as its child; a frozen onedir exe does not)?"""
    for _ in range(depth + 1):
        if pid == ancestor:
            return True
        pid = parents.get(pid, 0)
        if not pid:
            return False
    return False


def window_problems(facts: dict[str, Any], pid: int) -> list[str]:
    """Why a reported window id could not be embedded by ORION ([] = it can):
    QWindow.fromWinId needs a live top-level Qt window of the host process."""
    import fnmatch
    if facts.get("exists") is None:
        return []                                       # not Windows: cannot ask
    if not facts.get("exists"):
        return ["is not a window"]
    problems = []
    if not fnmatch.fnmatchcase(str(facts.get("class") or ""), QT_WINDOW_CLASS):
        problems.append(f"has class {facts.get('class')!r}, not {QT_WINDOW_CLASS}")
    if facts.get("pid") != pid:
        problems.append(f"belongs to pid {facts.get('pid')}, not the host ({pid})")
    if not facts.get("top_level"):
        problems.append("is not a top-level window")
    return problems


def host_check(name: str, cmd: Sequence[str], *, env: dict[str, str], start_limit_s: float,
               parts: Sequence[str] = (), state_query: bool = False,
               hard_timeout: float = 60.0, page_wait_s: float = 12.0, hold_s: float = 3.0,
               exit_timeout: float = 10.0,
               inspect: Callable[[int], dict[str, Any]] = window_facts,
               processes: Callable[[], dict[int, int]] = parent_pids) -> Check:
    """Start a page host as ORION does, wait for every window it reports
    (``@@{"t":"wid",...}``), prove each is a live top-level Qt window of that
    process (embeddable), require an explicit alive render-state reply when
    queried, require the host to stay up while ORION is "there", then close
    stdin (ORION gone) and require it to quit by itself."""
    started = time.monotonic()
    notes: list[str] = []
    expected = list(parts)
    try:
        proc = subprocess.Popen([str(part) for part in cmd], env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=CREATE_NO_WINDOW)
    except OSError as exc:
        return Check(name, False, f"could not start: {exc}")
    err = LineCollector(proc.stderr)
    try:
        messages: queue.Queue = queue.Queue()
        logs: list[str] = []

        def reader() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("@@"):
                    try:
                        messages.put(json.loads(line[2:]))
                    except ValueError:
                        logs.append(line)
                elif line:
                    logs.append(line)
            messages.put(None)

        threading.Thread(target=reader, daemon=True, name=f"{name}-read").start()

        def next_message(until: float) -> dict[str, Any] | None:
            while True:
                left = until - time.monotonic()
                if left <= 0:
                    raise TimeoutError
                try:
                    return messages.get(timeout=min(left, 0.25))
                except queue.Empty:
                    crash = err.crashed()
                    if crash:
                        raise ChildCrashed(crash)
                    continue

        def send(message: dict[str, Any]) -> bool:
            try:
                assert proc.stdin is not None
                proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
                proc.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

        failure = ""
        windows: dict[str, dict[str, Any]] = {}       # part -> its LATEST wid message
        replaced: dict[str, list[int]] = {}           # part -> every handle it announced
        window_s = 0.0
        page = ""
        exited = False

        def take(message: dict[str, Any]) -> str:
            """Fold one host message in; returns a failure reason or ''."""
            nonlocal page
            kind = message.get("t")
            if kind == "wid":
                part = str(message.get("part") or "main")
                handles = replaced.setdefault(part, [])
                if not handles or handles[-1] != int(message.get("wid") or 0):
                    handles.append(int(message.get("wid") or 0))
                windows[part] = message
            elif kind == "failed":
                return f"reported failure: {message.get('reason')}"
            elif kind == "reply":
                page = str(message.get("value"))
            elif kind == "log":
                logs.append(str(message.get("line") or ""))
            return ""

        # 1. Every window the kind builds (a page may have several parts).
        try:
            settle_until: float | None = None
            while not (expected and all(part in windows for part in expected)):
                until = settle_until if (settle_until and not expected) else started + hard_timeout
                try:
                    message = next_message(until)
                except TimeoutError:
                    if windows and not expected:
                        break                     # no parts known: what arrived is all
                    raise
                if message is None:
                    exited = True
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    time.sleep(0.05)
                    failure = ("ended before reporting its window(s) "
                               f"({child_failure(proc.poll(), err.lines)})")
                    break
                failure = take(message)
                if failure:
                    break
                if message.get("t") == "wid":
                    window_s = time.monotonic() - started
                    settle_until = time.monotonic() + 1.0
        except TimeoutError:
            missing = [part for part in expected if part not in windows] or ["any"]
            failure = f"no window reported for {', '.join(missing)} within {hard_timeout:.0f}s"
        except ChildCrashed as exc:
            failure = f"crashed before reporting its window(s) ({exc})"
        if not failure and window_s > start_limit_s:
            failure = (f"windows took {window_s:.1f}s; ORION gives a host "
                       f"{start_limit_s:.0f}s before falling back to in-process")
        # 2. While ORION is "there" (stdin open): the host stays up, and a window
        #    it replaces (WebEngine's GPU surface re-creates the native window) is
        #    announced again - ORION re-embeds the newest handle.
        if not failure:
            hold_until = time.monotonic() + hold_s
            page_until = time.monotonic() + page_wait_s
            next_query, rid = 0.0, 0
            while True:
                now = time.monotonic()
                querying = state_query and page not in {"alive", "failed"} and now < page_until
                if now >= hold_until and not querying:
                    break
                if querying and now >= next_query:
                    rid += 1
                    if not send({"m": "check_state", "a": [], "id": rid}):
                        break
                    next_query = now + 1.0
                try:
                    message = messages.get(timeout=0.2)
                except queue.Empty:
                    crash = err.crashed()
                    if crash:
                        failure = f"crashed while running ({crash})"
                        break
                    continue
                if message is None:
                    exited = True
                    break
                failure = take(message)
                if failure:
                    break
            if not failure and state_query and page != "alive":
                if page == "failed":
                    failure = "its page reports it failed to render (WebGL / three.js)"
                elif page:
                    failure = (f"its page did not become render-ready within {page_wait_s:g}s "
                               f"(last state: {page})")
                else:
                    failure = ("its page did not answer render-state queries within "
                               f"{page_wait_s:g}s")
            if not failure and (exited or proc.poll() is not None):
                failure = ("ended while its stdin was still open "
                           f"({child_failure(proc.poll(), err.lines)})")
        # 3. Each part's LATEST handle is a live top-level Qt window of the host
        #    process: what QWindow.fromWinId can embed. A dead latest handle is a
        #    window that was replaced and never announced again.
        described = []
        if not failure:
            # A replacement may be in flight (old window gone, new handle not
            # yet announced): give it a moment before calling the handle dead.
            grace_until = time.monotonic() + 2.0
            while time.monotonic() < grace_until and any(
                    inspect(int(m.get("wid") or 0)).get("exists") is False
                    for m in windows.values()):
                try:
                    message = messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                if message is None:
                    break
                failure = take(message)
                if failure:
                    break
        if not failure:
            tree: dict[int, int] | None = None
            for part, message in sorted(windows.items()):
                wid = int(message.get("wid") or 0)
                host_pid = int(message.get("pid") or 0)
                if host_pid != proc.pid:
                    tree = tree if tree is not None else processes()
                    if not descends_from(host_pid, proc.pid, tree):
                        failure = failure or (f"window {part}: reported pid {host_pid} is neither "
                                              f"the launched process ({proc.pid}) nor its child")
                        continue
                    notes.append(f"{part}: host pid {host_pid} is a child of launched pid {proc.pid}")
                facts = inspect(wid) if wid else {"exists": False}
                problems = window_problems(facts, host_pid)
                if problems == ["is not a window"]:
                    problems = ["is no longer a window (destroyed or replaced, and no new handle "
                                "announced), so ORION would embed a dead window"]
                if problems:
                    failure = failure or f"window {part} (0x{wid:x}) {'; '.join(problems)}"
                described.append(f"{part} 0x{wid:x}" + (f" {facts.get('class')}" if facts.get("class") else ""))
                if len(replaced.get(part, [])) > 1:
                    notes.append(f"{part}: native window replaced and re-announced "
                                 + " -> ".join(f"0x{handle:x}" for handle in replaced[part]))
        # 4. ORION gone: stdin closes and the host must quit by itself.
        closed_at = time.monotonic()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        exit_s = None
        try:
            proc.wait(timeout=exit_timeout)
            exit_s = time.monotonic() - closed_at
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            if not failure:
                failure = f"did not quit within {exit_timeout:.0f}s of stdin closing (killed)"
        stderr_tail = err.tail(300)
    finally:
        if proc.poll() is None:              # a harness fault must never leak a host
            kill_tree(proc)
    seconds = time.monotonic() - started
    if logs:
        notes.append("log: " + _one_line("\n".join(logs[-5:]), 300))
    if failure:
        if stderr_tail:
            failure += f"; stderr: {stderr_tail}"
        return Check(name, False, failure, seconds, data={"failures": [failure], "notes": notes})
    swaps = sum(len(handles) - 1 for handles in replaced.values())
    return Check(name, True,
                 f"{', '.join(described)}: live top-level window(s) of the host, reported in "
                 f"{window_s:.1f}s (limit {start_limit_s:.0f}s)"
                 + (f", {swaps} replaced and re-announced" if swaps else "")
                 + (f"; page {page or 'not answering'}" if state_query else "")
                 + f"; alive until stdin closed, quit {exit_s:.1f}s after", seconds,
                 data={"notes": notes, "handles": {part: [hex(h) for h in handles]
                                                   for part, handles in replaced.items()}})


# ── Forge verification ───────────────────────────────────────────────────────

SAMPLE_TOOL = '''"""Forge-shaped sample tool used by ORION's installed-build parity check."""


def get_tool_schema():
    return {
        "name": "parity_probe",
        "description": "Echo a message back (installed-build parity check).",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Text to echo."}},
            "required": ["message"],
        },
    }


def run(message: str) -> str:
    return "echo: " + message
'''

SAMPLE_TEST = '''from parity_probe_tool import get_tool_schema, run


def test_run_echoes():
    assert run(message="hi") == "echo: hi"


def test_schema_names_the_tool():
    assert get_tool_schema()["name"] == "parity_probe"
'''


def forge_env(staging: Path, config_dir: Path) -> dict[str, str]:
    """sandbox.SandboxVerificationHarness._child_env, plus the temp config."""
    return child_env(config_dir, {
        "PYTHONPATH": str(staging), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})


def parse_schema_line(stdout: str) -> dict[str, Any] | None:
    """sandbox._parse_schema_from_output: the first JSON object line."""
    for line in (stdout or "").strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def forge_checks(launcher: Sequence[str], modes: dict[str, bool], flag: str, source: Path,
                 internal: Path, work: Path) -> list[Check]:
    """Stage and run the Forge's two verification steps exactly as
    sandbox._run_script launches them in the installed app."""
    texts = forge_texts(source)
    if not modes.get(flag):
        why = not_in_build(flag) + " - the Forge cannot verify any tool in this installed app"
        return [Check("forge:conformance", False, why), Check("forge:pytest", False, why)]
    staging = work / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    contract = internal / "orion_core" / "forge_contract.py"
    if not contract.is_file() or not texts["probe_main"] or not texts["pytest_wrapper"]:
        why = ("forge_contract.py is not shipped in _internal/orion_core - sandbox.probe_source() "
               "would raise in the installed app" if not contract.is_file() else
               "could not read the conformance probe / pytest wrapper from sandbox.py")
        return [Check("forge:conformance", False, why), Check("forge:pytest", False, why)]
    for name in ("parity_probe_tool.py", "parity_probe.py"):
        (staging / name).write_text(SAMPLE_TOOL, encoding="utf-8")
    (staging / "parity_probe_test.py").write_text(SAMPLE_TEST, encoding="utf-8")
    conformance = staging / texts["conformance_name"]
    conformance.write_text(contract.read_text(encoding="utf-8") + texts["probe_main"],
                           encoding="utf-8")
    wrapper = staging / "_orion_pytest.py"
    wrapper.write_text(texts["pytest_wrapper"], encoding="utf-8")
    if texts.get("fixture_name") and texts.get("fixture_source"):
        (staging / f"{texts['fixture_name']}.py").write_text(texts["fixture_source"],
                                                             encoding="utf-8")
    env = forge_env(staging, work / "config")
    budget = texts["budget_s"]
    conformance_budget = min(budget, texts["conformance_cap_s"])
    checks = []
    # Stage 1, exactly as _run_script launches it: [exe, flag, <abs script>, tool_name].
    run = run_bounded([*launcher, flag, conformance, "parity_probe"], cwd=staging, env=env,
                      timeout=max(60.0, conformance_budget * 3))
    schema = parse_schema_line(run.stdout)
    if run.returncode == 0 and not run.timed_out and schema and \
            schema.get("name") == "parity_probe" and "conformance OK" in run.stdout:
        ok = run.seconds <= conformance_budget
        checks.append(Check("forge:conformance", ok,
                            ("exit 0, schema line parsed, 'conformance OK'" if ok else
                             f"passed but took {run.seconds:.1f}s; the Forge kills it at "
                             f"{conformance_budget:.0f}s"), run.seconds))
    else:
        contract_lines = [line for line in run.stderr.splitlines() if line.startswith("CONTRACT:")]
        checks.append(Check("forge:conformance", False,
                            _run_failure(run) if not contract_lines else _brief(contract_lines),
                            run.seconds))
    # Stage 2: the test file has test_ functions, so the Forge runs its pytest wrapper.
    run = run_bounded([*launcher, flag, wrapper, "parity_probe_test.py"], cwd=staging, env=env,
                      timeout=max(60.0, budget * 3))
    summary = next((line.strip() for line in reversed(run.stdout.splitlines())
                    if " passed" in line or " failed" in line or " error" in line), "")
    if run.returncode == 0 and not run.timed_out and "2 passed" in summary:
        ok = run.seconds <= budget
        checks.append(Check("forge:pytest", ok,
                            (f"exit 0, pytest: {summary}" if ok else
                             f"passed but took {run.seconds:.1f}s; the Forge kills it at "
                             f"{budget:.0f}s"), run.seconds))
    else:
        checks.append(Check("forge:pytest", False,
                            (f"pytest: {summary}; " if summary else "") + _run_failure(run),
                            run.seconds))
    return checks


# ── Forge package installs ───────────────────────────────────────────────────

def build_probe_wheel(directory: Path, name: str = PIP_PROBE_PACKAGE, version: str = "1.0") -> Path:
    """A minimal, valid pure-Python wheel (py3-none-any), built offline with
    zipfile so the pip check never needs the network or a build backend."""
    directory.mkdir(parents=True, exist_ok=True)
    dist_info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": (f'"""Installed by ORION\'s parity check into a temporary '
                                f'folder."""\nVALUE = "{PIP_PROBE_VALUE}"\n'),
        f"{dist_info}/METADATA": (f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\n"
                                  f"Version: {version}\nSummary: ORION parity probe\n"),
        f"{dist_info}/WHEEL": ("Wheel-Version: 1.0\nGenerator: orion-parity-check\n"
                               "Root-Is-Purelib: true\nTag: py3-none-any\n"),
    }
    record = []
    for path, text in files.items():
        data = text.encode("utf-8")
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(data)}")
    record.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = "\n".join(record) + "\n"
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel


def pip_check(runner: FrozenRunner, modes: dict[str, bool], flag: str, work: Path) -> Check:
    started = time.monotonic()
    if not runner.mechanism:
        return Check("forge:pip", False, "no way to run code in this build (not launched)")
    wheels = work / "wheels"
    build_probe_wheel(wheels)
    config = work / "config"
    env = child_env(config)
    run, result = run_probe_script(runner, PIP_PROBE, work=work, name="pip",
                                   args=[str(wheels), PIP_PROBE_PACKAGE], env=env, timeout=180)
    seconds = time.monotonic() - started
    if result is None:
        return Check("forge:pip", False, f"the probe did not report: {_run_failure(run)}", seconds)
    if result.get("error"):
        detail = result["error"] + (f"; {result['legacy']}" if result.get("legacy") else "")
        python = str(result.get("python") or "")
        if "WindowsApps" in python and "AppData" in str(work):
            detail += ("; NOTE a Store Python redirects AppData writes - rerun with "
                       "--work-dir outside AppData")
        return Check("forge:pip", False, detail, seconds, data={"result": result})
    if result.get("returncode") != 0:
        return Check("forge:pip", False, f"pip failed (exit {result.get('returncode')}): "
                     f"{_one_line(result.get('pip_tail', ''))}", seconds, data={"result": result})
    landed = config / "forge_packages" / PIP_PROBE_PACKAGE / "__init__.py"
    if result.get("value") != PIP_PROBE_VALUE or not landed.is_file():
        detail = (result.get("import_error") or "installed package not importable") + \
            ("" if landed.is_file() else f"; {landed} is not visible outside the installer "
             "(a Store Python sandbox redirects AppData writes)")
        return Check("forge:pip", False, detail, seconds, data={"result": result})
    notes = [f"python: {result.get('python')}", f"command: {' '.join(result.get('command') or [])}"]
    # A forged tool imports it at verification time: a fresh --forge-verify run.
    if modes.get(flag):
        check_run, check = run_probe_script(
            FrozenRunner(runner.launcher, {flag: True}, flag), IMPORT_CHECK, work=work / "verify",
            name="import", args=[PIP_PROBE_PACKAGE], env=env, timeout=60)
        seconds = time.monotonic() - started
        if not check or not check.get("ok"):
            why = (check or {}).get("error") or _run_failure(check_run)
            return Check("forge:pip", False, f"installed, but a fresh verification cannot "
                         f"import it: {why}", seconds, data={"result": result, "notes": notes})
        notes.append(f"fresh verification imported it from {check.get('file')}")
    return Check("forge:pip", True,
                 f"installed offline via {Path(str(result.get('python'))).name} into the temp "
                 f"config in {result.get('pip_s')}s; importable in a fresh verification",
                 seconds, data={"result": result, "notes": notes})


# ── orchestration ────────────────────────────────────────────────────────────

GROUPS = ("modes", "mcp", "face", "forge", "imports", "data", "pip")


def _build_info(internal: Path) -> dict[str, Any]:
    try:
        return json.loads((internal / "build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def not_in_build(flag: str) -> str:
    return (f"not in this build: its entry script does not handle {flag} (launching it "
            "would start the full app; not launched)")


def run_checks(exe: Path, source: Path, *, groups: Sequence[str] = GROUPS,
               work_root: Path | None = None, keep: bool = False,
               import_budget_s: float = 150.0, archive: Archive | None = None,
               launcher: Sequence[str] | None = None) -> tuple[list[Check], dict[str, Any]]:
    """Every check against *exe*. *archive* and *launcher* default to reading
    and running that executable (tests inject stand-ins)."""
    started = time.monotonic()
    exe = exe.resolve()
    internal = exe.parent / "_internal"
    launcher = [str(part) for part in launcher] if launcher else [str(exe)]
    work = Path(tempfile.mkdtemp(prefix="orion-parity-", dir=str(work_root) if work_root else None))
    info: dict[str, Any] = {"exe": str(exe), "source": str(source), "work": str(work),
                            "build_info": _build_info(internal)}
    flags = mode_flags(source)
    archive = archive if archive is not None else read_archive(exe)
    modes = entry_modes(archive, flags.values())
    info["modes"] = modes
    config = build_config(source)
    results: dict[str, list[Check]] = {}
    try:
        if "modes" in groups:
            if archive.error:
                results["modes"] = [Check("modes", False, archive.error)]
            else:
                absent = [flag for flag in flags.values() if not modes[flag]]
                results["modes"] = [Check(
                    "modes", not absent,
                    ("entry script handles " + ", ".join(flags.values())) if not absent else
                    (f"entry script does not handle {', '.join(absent)} (those modes would "
                     "start the full app; not launched)"),
                    data={"modes": modes})]
        runner = FrozenRunner(launcher, modes, flags["forge"])
        info["frozen_mechanism"] = runner.mechanism or "none"

        def mcp_group() -> list[Check]:
            servers = bundled_servers(source)
            if archive.error:
                return [Check(f"mcp:{name}", False, archive.error) for name in servers]
            if not modes[flags["mcp"]]:
                return [Check(f"mcp:{name}", False, not_in_build(flags["mcp"]))
                        for name in servers]
            protocol, supported = mcp_protocol(source)
            spec_work = work / "mcp"
            specs: dict[str, Any] | None = None
            spec_note = ""
            if runner.mechanism:
                run, specs = run_probe_script(runner, SPEC_PROBE, work=spec_work, name="specs",
                                              args=[], env=child_env(spec_work / "config"),
                                              timeout=90)
                if specs is None:
                    spec_note = f"could not ask the frozen host for its commands: {_run_failure(run)}"
            else:
                spec_note = "no way to ask the frozen host for its commands"

            def one(name: str) -> Check:
                cmd: list[str] = [*launcher, flags["mcp"], name]
                note = spec_note or "launched with the frozen host's own command"
                extra_env: dict[str, str] = {}
                if specs:
                    host = (specs.get("host") or {}).get(name)
                    bundled = (specs.get("bundled") or {}).get(name)
                    chosen = host or ({"command": bundled[0], "args": bundled[1]} if bundled else None)
                    if chosen:
                        cmd = [str(chosen["command"]), *map(str, chosen["args"])]
                        extra_env = dict(chosen.get("env") or {})
                    else:
                        note = (f"the frozen app does not know server {name!r} "
                                f"({specs.get('bundled_error') or specs.get('host_error') or 'absent'})")
                server_config = spec_work / name / "config"
                return check_mcp(name, cmd, note, env=child_env(server_config, extra_env),
                                 protocol=protocol, supported=supported, timeout=60.0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                return list(pool.map(one, servers))

        def face_group() -> list[Check]:
            """The face host and every other page-host kind, one at a time
            (each is a Qt process; overlapping them would only skew timings)."""
            kinds = page_kinds(source)
            hosts = [("face-host", [flags["face"]], kinds.get("face") or ["main"], True)]
            hosts += [(f"page-host:{kind}", [flags["page"], kind], parts, False)
                      for kind, parts in kinds.items() if kind != "face"]
            limit = host_start_limit_s(source)
            checks = []
            for name, args, parts, query in hosts:
                if archive.error:
                    checks.append(Check(name, False, archive.error))
                elif not modes[args[0]]:
                    checks.append(Check(name, False, not_in_build(args[0])))
                else:
                    checks.append(host_check(
                        name, [*launcher, *args], env=child_env(work / name.replace(":", "-") / "config"),
                        start_limit_s=limit, parts=parts, state_query=query))
            return checks

        def forge_group() -> list[Check]:
            if archive.error:
                return [Check("forge:conformance", False, archive.error),
                        Check("forge:pytest", False, archive.error)]
            return forge_checks(launcher, modes, flags["forge"], source, internal, work / "forge")

        def imports_group() -> list[Check]:
            began = time.monotonic()
            if not runner.mechanism:
                why = archive.error or "no way to run code in this build (not launched)"
                return [Check(name, False, why) for name in
                        ("data:runtime", "imports:orion_core", "imports:lazy")]
            plugins = sorted((internal / "shipped_plugins").glob("*.py"))
            entries, skipped = import_jobs(source, plugins)
            probe_work = work / "imports"
            results_, notes = run_import_probe(runner, entries, work=probe_work,
                                               env=child_env(probe_work / "config"),
                                               budget_s=import_budget_s)
            grouped = classify_imports(entries, results_, excludes=config["excludes"],
                                       archive=archive,
                                       built_at=info["build_info"].get("built_at"), source=source)
            return import_checks(grouped, mechanism=runner.mechanism, run_notes=notes,
                                 skipped=skipped, seconds=time.monotonic() - began)

        def data_group() -> list[Check]:
            if not internal.is_dir():
                return [Check("data:bundle", False, f"{internal} does not exist")]
            return [check_datas(config, source, internal),
                    check_collect_all(expected_collect_all(config), internal),
                    check_resources(source, internal),
                    check_critical(internal, exe.parent)]

        def pip_group() -> list[Check]:
            return [pip_check(runner, modes, flags["forge"], work / "pip")]

        jobs = {"mcp": mcp_group, "face": face_group, "forge": forge_group,
                "imports": imports_group, "data": data_group, "pip": pip_group}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {name: pool.submit(job) for name, job in jobs.items() if name in groups}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as exc:          # a harness bug must still be a red line
                    results[name] = [Check(name, False, f"harness error: {type(exc).__name__}: {exc}")]
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    order = ["modes", "mcp", "face", "forge", "pip", "imports", "data"]
    checks = [check for name in order for check in results.get(name, [])]
    # data:runtime comes from the import probe; list it with the other data lines.
    checks.sort(key=lambda check: check.name.startswith("data:"))
    info["seconds"] = round(time.monotonic() - started, 1)
    return checks, info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exe", nargs="?", type=Path, default=DEFAULT_EXE,
                        help="the built ORION.exe (default dist/ORION/ORION.exe)")
    parser.add_argument("--source", type=Path, default=ROOT, help="the source tree it was built from")
    parser.add_argument("--json", metavar="PATH", help="also write a JSON report ('-' prints only JSON)")
    parser.add_argument("--only", help="comma-separated groups: " + ",".join(GROUPS))
    parser.add_argument("--work-dir", type=Path, help="where temporary folders go (default: TEMP)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every failure and note")
    args = parser.parse_args(argv)
    groups = tuple(g.strip() for g in args.only.split(",")) if args.only else GROUPS
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        parser.error(f"unknown group(s): {', '.join(unknown)}")
    if not args.exe.is_file():
        print(f"FAIL  executable not found: {args.exe}")
        print("FAILED: 1")
        return 2
    checks, info = run_checks(args.exe, args.source.resolve(), groups=groups,
                              work_root=args.work_dir, keep=args.keep)
    report = {**info, "checks": [check.to_dict() for check in checks],
              "failed": sum(1 for check in checks if not check.ok),
              "summary": summary_line(checks)}
    if args.json and args.json != "-":
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n",
                                   encoding="utf-8")
    if args.json == "-":
        print(json.dumps(report, indent=2, default=str))
    else:
        for line in render(checks, verbose=args.verbose):
            print(line)
        print(f"{len(checks)} checks of {info['exe']} in {info['seconds']}s "
              f"(frozen code via {info.get('frozen_mechanism')})")
        print(summary_line(checks))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
