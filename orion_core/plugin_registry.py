"""
plugin_registry.py — the plugin ECOSYSTEM around ORION's plugin runtime.

``plugin_manifest.py`` gave ORION a plugin *runtime*: it validates a
``*.plugin.json`` manifest, dependency-checks it, and hands the module to the
``ReflectiveModuleLoader`` for live registration.  What was missing was
everything a runtime needs to actually become an ecosystem:

  * nothing could be turned OFF without deleting its files;
  * ORION had no tool for plugins at all (``mcp`` and ``skill`` existed, but a
    code plugin could not be listed, described, installed or disabled);
  * writing a plugin meant hand-authoring two files against an undocumented
    contract, so in practice ZERO manifests were ever written;
  * the forged tools in ``config/custom_tools/`` load through the code-only
    forge path, so they carry NO declared capability tier — the remote gate
    then has to guess at them;
  * a plugin whose dependency was missing was skipped with a log line and no
    offer to fix it;
  * a broken module could only be discovered by trying to load it.

This module supplies that layer.  Its rules follow the rest of ORION:

  * it NEVER raises to the caller — every entry point returns a value or a
    ``ToolResult``, because a plugin manager that crashes the assistant is
    worse than no plugin manager;
  * it never IMPORTS a candidate module to inspect it.  Introspection is
    static (``ast``), so merely listing or validating an untrusted plugin
    cannot execute its code — only a deliberate load does that;
  * disabled state lives in ``config/plugins.json`` and is authoritative:
    the loader consults it, so disabling survives restarts.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import shutil
import zipfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR
from .data import ToolResult
from .plugin_manifest import (
    CAPABILITY_TIERS,
    MANIFEST_SUFFIX,
    PluginManifest,
    discover_manifests,
    unmet_requirements,
)
from .atomic_io import atomic_write_text

PLUGIN_STATE_PATH = CONFIG_DIR / "plugins.json"
CUSTOM_TOOLS_DIR = CONFIG_DIR / "custom_tools"
_DEFAULT_TIER = "confirm"          # safest tier for anything undeclared
_TOOL_SUFFIX = "_tool.py"


# ──────────────────────────────────────────────────────────────────────────────
# RECORDS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PluginRecord:
    """One plugin as the registry sees it — manifest facts plus live health."""

    name: str
    description: str = ""
    module: str = ""
    tier: str = _DEFAULT_TIER
    requires: tuple[str, ...] = ()
    enabled: bool = True
    has_manifest: bool = False
    module_exists: bool = False
    missing_deps: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    #: Declared version, and the capability disclosure (Mark XXVI).
    version: str = "0.0.0"
    permissions: tuple[str, ...] = ()
    #: Capabilities the CODE exercises that the manifest never declared. Not a
    #: fault (so it does not flip `healthy`) — it is a disclosure gap the user
    #: should see before trusting a plugin.
    undeclared: tuple[str, ...] = ()
    # Live health, populated as ORION runs.
    loaded: bool = False
    last_error: str = ""
    calls: int = 0

    @property
    def healthy(self) -> bool:
        return (self.module_exists and not self.missing_deps
                and not self.problems and not self.last_error)

    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.module_exists:
            return "missing"
        if self.missing_deps:
            return "needs-deps"
        if self.problems:
            return "invalid"
        if self.last_error:
            return "error"
        return "loaded" if self.loaded else "ready"

    def summary(self) -> str:
        bits = [f"{self.name} [{self.status()}]"]
        if self.version and self.version != "0.0.0":
            bits.append(f"v{self.version}")
        if self.tier:
            bits.append(f"tier={self.tier}")
        if self.permissions:
            bits.append("needs: " + ", ".join(self.permissions))
        if self.undeclared:
            bits.append("UNDECLARED: " + ", ".join(self.undeclared))
        if self.missing_deps:
            bits.append(f"missing: {', '.join(self.missing_deps)}")
        if self.problems:
            bits.append(self.problems[0])
        if self.last_error:
            bits.append(f"error: {self.last_error[:80]}")
        if self.calls:
            bits.append(f"{self.calls} call(s)")
        line = "  •  ".join(bits)
        if self.description:
            line += f"\n     {self.description[:150]}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "module": self.module, "tier": self.tier,
            "requires": list(self.requires), "enabled": self.enabled,
            "has_manifest": self.has_manifest, "status": self.status(),
            "missing_deps": list(self.missing_deps),
            "problems": list(self.problems), "loaded": self.loaded,
            "last_error": self.last_error, "calls": self.calls,
            "version": self.version, "permissions": list(self.permissions),
            "undeclared": list(self.undeclared),
        }


# ──────────────────────────────────────────────────────────────────────────────
# STATIC VALIDATION  (never imports the candidate module)
# ──────────────────────────────────────────────────────────────────────────────


#: Module/attribute fingerprints that reveal a capability, keyed by permission.
#: Coarse on purpose: this answers "does this code touch the network at all?",
#: which is what a user deciding whether to trust a dropped-in plugin needs.
_CAPABILITY_IMPORTS: dict[str, tuple[str, ...]] = {
    "network": ("socket", "requests", "urllib", "urllib3", "http", "aiohttp",
                "httpx", "ftplib", "smtplib", "telnetlib", "websocket",
                "websockets", "paramiko"),
    "filesystem": ("shutil", "pathlib", "os.path", "glob", "tempfile", "zipfile",
                   "tarfile", "csv", "sqlite3"),
    "subprocess": ("subprocess", "multiprocessing", "pty", "asyncio.subprocess"),
    "input": ("pyautogui", "keyboard", "mouse", "pynput", "pydirectinput",
              "win32api", "ctypes.windll.user32"),
    "screen": ("mss", "PIL.ImageGrab", "ImageGrab", "pygetwindow", "pywinauto"),
    "audio": ("sounddevice", "pyaudio", "pyttsx3", "speech_recognition", "wave",
              "vosk", "whisper", "simpleaudio"),
}

#: Calls that imply a capability even without a telltale import, and mean only
#: one thing wherever they appear. Nothing but a path object has write_bytes.
_CAPABILITY_CALLS: dict[str, tuple[str, ...]] = {
    "filesystem": ("unlink", "rmtree", "mkdir", "write_text", "write_bytes"),
    "subprocess": ("Popen", "check_output", "execv", "spawn"),
}

#: Calls whose names are ordinary English and belong to many types.
#: ``"a b".replace(...)``, ``items.remove(...)`` and ``tasks.run(...)`` are not
#: filesystem writes or subprocesses, and reporting them as such taught people
#: to wave the audit through. These count only with corroboration — see
#: ``_ambiguous_call_counts``.
_AMBIGUOUS_CALLS: dict[str, tuple[str, ...]] = {
    "filesystem": ("open", "remove", "rename", "replace"),
    "subprocess": ("system", "popen", "run", "call"),
}

#: Receivers that make an ambiguous call unambiguous. Either a module that is
#: about files or processes, or a variable named for one.
_PATHISH_NAMES = frozenset({"os", "shutil", "path", "pathlib", "Path",
                            "subprocess", "sp", "file", "f", "dest", "target",
                            "source", "src"})
_PATHISH_WORDS = ("path", "file", "dir", "folder", "archive")


def _ambiguous_call_counts(func: "ast.AST", name: str) -> bool:
    """Whether an ordinary-word call is really the dangerous one.

    True when it is called bare, as the builtin ``open`` is, or on something
    that is plainly a path or a filesystem/process module. False for
    ``some_string.replace(...)``, which is what this exists to stop counting.
    """
    if isinstance(func, ast.Name):
        # open(...), not x.open(...). The builtin really is that call.
        return name in {"open"}
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value
    if isinstance(owner, ast.Name):
        target = owner.id
    elif isinstance(owner, ast.Attribute):
        target = owner.attr
    elif isinstance(owner, ast.Call):
        # Path(...).replace(...) — the constructor names the type.
        inner = owner.func
        target = getattr(inner, "id", None) or getattr(inner, "attr", "") or ""
    else:
        return False
    lowered = str(target).lower()
    return (target in _PATHISH_NAMES
            or any(word in lowered for word in _PATHISH_WORDS))


#: Capabilities that make an UNATTENDED (tier=allow) plugin a poor idea: each
#: can act on the machine or reach off it without the user present to see it.
_UNATTENDED_RISK: frozenset[str] = frozenset({
    "subprocess", "input", "network", "filesystem",
})


def _module_roots(tree: "ast.AST") -> set[str]:
    """Every module path imported by the source, dotted-prefix included."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module)
    return roots


def detect_capabilities(path: Path) -> set[str]:
    """Statically infer which PERMISSIONS a plugin module actually exercises.

    Pure ``ast`` — the module is never imported, so auditing an untrusted plugin
    cannot execute it. Heuristic by design: it can over-report (a module that
    imports ``pathlib`` for a read-only path join still counts as filesystem),
    which is the safe direction for a disclosure check.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    found: set[str] = set()
    roots = _module_roots(tree)
    for permission, needles in _CAPABILITY_IMPORTS.items():
        for needle in needles:
            if any(root == needle or root.startswith(needle + ".") for root in roots):
                found.add(permission)
                break
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None) or ""
        for permission, needles in _CAPABILITY_CALLS.items():
            if name in needles:
                found.add(permission)
        for permission, needles in _AMBIGUOUS_CALLS.items():
            if name in needles and _ambiguous_call_counts(func, name):
                found.add(permission)
    return found


def audit_permissions(path: Path, declared: "tuple[str, ...] | list[str]") -> dict[str, list[str]]:
    """Compare what a plugin DOES against what it DECLARES.

    Returns ``{"used": [...], "undeclared": [...], "unused": [...]}``.
    ``undeclared`` is the one that matters: capabilities the code exercises that
    the manifest never mentioned.
    """
    used = detect_capabilities(path)
    declared_set = {str(d).strip().lower() for d in (declared or ())}
    return {
        "used": sorted(used),
        "undeclared": sorted(used - declared_set),
        "unused": sorted(declared_set - used),
    }


def compare_versions(left: str, right: str) -> int:
    """-1 / 0 / +1 comparing dotted version strings, tolerant of junk.

    Non-numeric segments compare as 0 rather than raising, because a plugin
    author's version string is untrusted input like everything else here.
    """
    def parts(value: str) -> list[int]:
        out: list[int] = []
        for chunk in str(value or "0").strip().lstrip("vV").split("."):
            digits = "".join(ch for ch in chunk if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    a, b = parts(left), parts(right)
    while len(a) < len(b):
        a.append(0)
    while len(b) < len(a):
        b.append(0)
    return (a > b) - (a < b)


def validate_module(path: Path) -> list[str]:
    """Statically check that *path* satisfies ORION's plugin contract.

    Returns a list of human-readable problems — empty means it conforms.  This
    parses the file with ``ast``; it does NOT import it, so validating a
    plugin can never run its code.
    """
    problems: list[str] = []
    path = Path(path)
    if not path.is_file():
        return [f"module file '{path.name}' not found"]
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:                      # unreadable file
        return [f"cannot read {path.name}: {exc}"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    top_level = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "get_tool_schema" not in top_level:
        problems.append("missing a top-level get_tool_schema() function")
    if "run" not in top_level:
        problems.append("missing a top-level run() function")
    problems.extend(_schema_literal_problems(tree))
    return problems


def _schema_literal_problems(tree: "ast.AST") -> list[str]:
    """Catch the schema mistakes the LOADER would reject, statically.

    The loader's contract requires ``parameters.type`` to be lowercase
    ``'object'`` (forge_contract). A plugin declaring Gemini-style ``'OBJECT'``
    passes every other check, is reported healthy, and then silently refuses to
    activate — so the registry must see it here rather than promise a plugin
    that will not load. Only literal dicts are inspected; a schema built
    dynamically is left to the loader.
    """
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "type"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value.lower() == "object"
                    and value.value != "object"):
                problems.append(
                    "schema parameters 'type' must be lowercase 'object' "
                    f"(found {value.value!r}) — the loader will refuse it")
                return problems
    return problems


def extract_description(path: Path) -> str:
    """Best-effort description for a module WITHOUT importing it: the module
    docstring's first sentence, else a literal 'description' in the schema."""
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    doc = (ast.get_docstring(tree) or "").strip()
    if doc:
        first = doc.split("\n\n")[0].replace("\n", " ").strip()
        return first[:300]
    # Fall back to a "description": "..." literal anywhere in the file.
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "description"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)):
                    return value.value.strip()[:300]
    return ""


def extract_parameters(path: Path) -> dict[str, Any]:
    """Best-effort JSON-schema 'parameters' block from a module's
    get_tool_schema(), read statically via ``ast.literal_eval``."""
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "parameters"):
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return {}
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# SCAFFOLDING
# ──────────────────────────────────────────────────────────────────────────────


_TEMPLATE = '''"""
{description}

An ORION plugin.  The contract is two top-level functions:

    get_tool_schema()  ->  the tool declaration ORION shows the model
    run(**kwargs)      ->  the implementation; return a string

Edit the body of run() to do the real work.  ORION loads this file live —
use  plugin reload {name}  after editing, no restart needed.
"""

from __future__ import annotations


def get_tool_schema() -> dict:
    """Declare this plugin as a callable ORION tool."""
    return {{
        "name": "{name}",
        "description": "{description}",
        "parameters": {{
            "type": "object",
            "properties": {{
                "query": {{
                    "type": "STRING",
                    "description": "What to act on.",
                }},
            }},
            "required": [],
        }},
    }}


def run(**kwargs) -> str:
    """Execute the plugin.  Return a short string for ORION to speak/show."""
    query = str(kwargs.get("query") or "").strip()
    if not query:
        return "{name}: ready, sir — give me something to work with."
    return f"{name} received: {{query}}"
'''



#: The REACTIVE template — a plugin that also watches ORION. Generated when the
#: user asks for kind="event"/"listener", so the hook contract is discoverable
#: rather than something you have to already know exists.
_TEMPLATE_LISTENER = '''"""
{description}

A REACTIVE ORION plugin.  It is a normal tool (get_tool_schema/run) AND it
watches ORION through declared events:

    on_event(event, payload)   <- called when a subscribed event fires

The events it receives are the ones listed in its manifest ("events": [...]).
Only observational events can be subscribed to, and a hook runs on ORION's OWN
thread — so keep on_event FAST (well under 50 ms) and hand slow work off.
"""

from __future__ import annotations


def get_tool_schema() -> dict:
    """Declare this plugin as a callable ORION tool."""
    return {{
        "name": "{name}",
        "description": "{description}",
        "parameters": {{
            "type": "object",
            "properties": {{
                "query": {{
                    "type": "STRING",
                    "description": "What to act on.",
                }},
            }},
            "required": [],
        }},
    }}


def run(**kwargs) -> str:
    """Execute the plugin.  Return a short string for ORION to speak/show."""
    query = str(kwargs.get("query") or "").strip()
    return f"{name} received: {{query}}" if query else "{name}: ready, sir."


#: The last few events seen — replace with whatever this plugin should DO.
_recent: list = []


def on_event(event: str, payload=None) -> None:
    """React to ORION.  Keep this fast: it runs on ORION's own thread."""
    _recent.append((event, payload))
    del _recent[:-20]
'''

def scaffold(name: str, description: str = "", tier: str = _DEFAULT_TIER,
             directory: Path | None = None, version: str = "1.0.0",
             permissions: "tuple[str, ...] | list[str] | None" = None,
             kind: str = "tool",
             events: "tuple[str, ...] | list[str] | None" = None) -> tuple[Path, Path]:
    """Write a WORKING plugin (module + manifest) and return both paths.

    The generated plugin conforms to the contract immediately, so it loads and
    is callable the moment it is created — the user then edits the body rather
    than assembling boilerplate against an undocumented interface.
    """
    directory = Path(directory or CUSTOM_TOOLS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(name)
    description = (description or f"The {safe.replace('_', ' ')} plugin.").strip()
    tier = tier if tier in CAPABILITY_TIERS else _DEFAULT_TIER

    module_path = directory / f"{safe}{_TOOL_SUFFIX}"
    manifest_path = directory / f"{safe}{MANIFEST_SUFFIX}"
    # Escape quotes so a description containing " cannot break the template.
    reactive = str(kind or "tool").strip().lower() in {
        "event", "events", "listener", "reactive", "watcher"}
    template = _TEMPLATE_LISTENER if reactive else _TEMPLATE
    body = template.format(name=safe, description=description.replace('"', "'"))
    module_path.write_text(body, encoding="utf-8")
    # A new plugin is born WITH a version and an honest capability declaration:
    # if none is supplied we audit the freshly-written module and declare what it
    # actually uses, so a scaffold never starts life with a disclosure gap.
    declared = [str(perm).strip().lower() for perm in (permissions or [])]
    if not declared:
        declared = sorted(detect_capabilities(module_path))
    declared_events = [str(e).strip() for e in (events or []) if str(e).strip()]
    if reactive and not declared_events:
        # A reactive plugin with no declared events would silently never fire;
        # give it the two most useful, least noisy ones to start from.
        declared_events = ["state", "connection_state"]
    manifest = {
        "name": safe,
        "description": description,
        "module": module_path.name,
        "tier": tier,
        "version": str(version or "1.0.0"),
        "permissions": declared,
        # Declare bus events here (see plugin_events.SUBSCRIBABLE) and add an
        # on_event(event, payload) function to react to them, not just be called.
        "events": declared_events,
        "requires": [],
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "What to act on."},
            },
            "required": [],
        },
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2), encoding="utf-8")
    return module_path, manifest_path


def _safe_name(name: str) -> str:
    """A filesystem/identifier-safe plugin name."""
    cleaned = "".join(
        c if (c.isalnum() or c == "_") else "_" for c in str(name or "").strip().lower()
    ).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "plugin"


# ──────────────────────────────────────────────────────────────────────────────
# SHIPPED PLUGINS
# ──────────────────────────────────────────────────────────────────────────────

#: Remembers which version of each shipped plugin was last delivered, so a
#: plugin the user removed stays removed and one they edited is left alone.
SHIPPED_LEDGER = ".shipped.json"


def _digest(path: Path) -> str:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def seed_shipped(target: Path, source: Path | None = None) -> list[str]:
    """Deliver the plugins bundled with ORION into the live plugin folder.

    The build copied the reviewed plugins beside the exe, but the app reads
    its data through a pointer to a folder outside it, so nothing shipped ever
    arrived: the installed app ran July's forged versions and had no Home
    Assistant, Hue, weather, push, Discord, Telegram or IFTTT tool at all.

    Each plugin (``<name>_tool.py`` plus its manifest) is handled as a unit:

      * missing and never delivered: installed;
      * delivered before and since deleted: left deleted;
      * still exactly the version delivered last time: upgraded;
      * edited after delivery: left alone;
      * an older copy predating this ledger: kept in ``_superseded`` and
        replaced by the reviewed version.

    Returns the names installed or upgraded. Never raises.
    """
    try:
        from .constants import resource_path
        source = Path(source) if source else resource_path("shipped_plugins")
        target = Path(target)
        if not source.is_dir() or source.resolve() == target.resolve():
            return []
        target.mkdir(parents=True, exist_ok=True)
        ledger_path = target / SHIPPED_LEDGER
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if not isinstance(ledger, dict):
                ledger = {}
        except (OSError, ValueError):
            ledger = {}
        delivered: list[str] = []
        for module in sorted(source.glob(f"*{_TOOL_SUFFIX}")):
            name = module.name[: -len(_TOOL_SUFFIX)]
            files = [module]
            manifest = source / f"{name}{MANIFEST_SUFFIX}"
            if manifest.is_file():
                files.append(manifest)
            shipped = {f.name: _digest(f) for f in files}
            last = ledger.get(name)
            if last == shipped:
                continue
            live = target / module.name
            quarantined = (target / "_quarantine" / f"{module.name}.broken").exists()
            if last and not live.exists() and not quarantined:
                continue                      # the user removed it
            current = {f: _digest(target / f) for f in shipped if (target / f).exists()}
            if current == shipped:
                ledger[name] = shipped
                continue
            if last and any(current.get(f) not in ("", last.get(f)) for f in current):
                continue                      # edited since delivery
            if current and not last:
                backup = target / "_superseded"
                backup.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                for f in current:
                    shutil.copy2(target / f, backup / f"{f}.{stamp}")
            for f in files:
                shutil.copy2(f, target / f.name)
            ledger[name] = shipped
            delivered.append(name)
        atomic_write_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True),
                          encoding="utf-8")
        return delivered
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# THE REGISTRY
# ──────────────────────────────────────────────────────────────────────────────


class PluginRegistry:
    """Discovery, lifecycle, persistence and health for ORION's code plugins."""

    def __init__(self, bus: Any = None, directory: Path | None = None,
                 telemetry: Any = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.directory = Path(directory or CUSTOM_TOOLS_DIR)
        self._state: dict[str, Any] = {"disabled": [], "meta": {}}
        self._health: dict[str, dict[str, Any]] = {}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._load_state()

    # ── logging (never fatal) ────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        try:
            if self.bus is not None:
                self.bus.log.emit(message)
        except Exception:
            pass

    # ── persistent state ─────────────────────────────────────────────────────

    def _load_state(self) -> None:
        try:
            if PLUGIN_STATE_PATH.is_file():
                data = json.loads(PLUGIN_STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    disabled = data.get("disabled")
                    meta = data.get("meta")
                    self._state = {
                        "disabled": list(disabled) if isinstance(disabled, list) else [],
                        "meta": dict(meta) if isinstance(meta, dict) else {},
                    }
        except Exception as exc:
            self._log(f"PLUGIN: state unreadable, starting fresh - {exc}")

    def _save_state(self) -> bool:
        temporary = None
        try:
            PLUGIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                    dir=PLUGIN_STATE_PATH.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self._state, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, PLUGIN_STATE_PATH)
            return json.loads(PLUGIN_STATE_PATH.read_text(encoding="utf-8")) == self._state
        except Exception as exc:
            self._log(f"PLUGIN: could not save state - {exc}")
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def is_enabled(self, name: str) -> bool:
        return _safe_name(name) not in set(self._state.get("disabled") or [])

    def set_enabled(self, name: str, enabled: bool) -> ToolResult:
        safe = _safe_name(name)
        known = {record.name for record in self.scan()}
        if safe not in known:
            return ToolResult(f"No plugin named '{name}'.", ok=False)
        previous = list(self._state.get("disabled") or [])
        disabled = set(previous)
        if enabled:
            disabled.discard(safe)
        else:
            disabled.add(safe)
        self._state["disabled"] = sorted(disabled)
        if not self._save_state():
            self._state["disabled"] = previous
            return ToolResult(f"Could not verify the saved setting for plugin '{safe}'.", ok=False)
        word = "enabled" if enabled else "disabled"
        self._log(f"PLUGIN: '{safe}' {word}.")
        note = "" if enabled else " It stays off until you enable it again."
        restart = "" if enabled else " (already-loaded tools clear on restart)"
        return ToolResult(f"Plugin '{safe}' {word}.{note}{restart}")

    # ── discovery ────────────────────────────────────────────────────────────

    def scan(self) -> list[PluginRecord]:
        """Every plugin ORION can see: manifest-declared ones first, then any
        orphan ``*_tool.py`` that has no manifest yet (the forged tools)."""
        records: dict[str, PluginRecord] = {}
        manifests, errors = discover_manifests(self.directory)

        for manifest in manifests:
            record = self._record_from_manifest(manifest)
            records[record.name] = record

        # Orphans: a *_tool.py with no manifest beside it still loads through
        # the forge path, so it must be visible here — that is exactly the set
        # that currently carries no declared capability tier.
        for module_path in sorted(self.directory.glob(f"*{_TOOL_SUFFIX}")):
            name = module_path.name[: -len(_TOOL_SUFFIX)]
            if name in records:
                continue
            record = PluginRecord(
                name=name,
                description=extract_description(module_path),
                module=module_path.name,
                tier=_DEFAULT_TIER,
                enabled=self.is_enabled(name),
                has_manifest=False,
                module_exists=True,
                problems=tuple(validate_module(module_path)),
            )
            records[name] = record

        for manifest_name, reason in errors.items():
            stem = manifest_name.replace(MANIFEST_SUFFIX, "")
            records[stem] = PluginRecord(
                name=stem, enabled=self.is_enabled(stem),
                has_manifest=True, problems=(reason,))

        for record in records.values():
            health = self._health.get(record.name)
            if health:
                record.loaded = bool(health.get("loaded"))
                record.last_error = str(health.get("error") or "")
                record.calls = int(health.get("calls") or 0)
        return sorted(records.values(), key=lambda r: r.name)

    def _record_from_manifest(self, manifest: PluginManifest) -> PluginRecord:
        module_path = manifest.module_path
        exists = bool(module_path and module_path.is_file())
        problems = validate_module(module_path) if exists and module_path else []
        return PluginRecord(
            name=manifest.name,
            description=manifest.description,
            module=manifest.module,
            tier=manifest.tier,
            requires=tuple(manifest.requires),
            enabled=self.is_enabled(manifest.name),
            has_manifest=True,
            module_exists=exists,
            missing_deps=tuple(unmet_requirements(manifest)),
            problems=tuple(problems),
            version=manifest.version,
            permissions=tuple(manifest.permissions),
            undeclared=tuple(
                audit_permissions(module_path, manifest.permissions)["undeclared"]
            ) if exists and module_path else (),
        )

    def get(self, name: str) -> PluginRecord | None:
        safe = _safe_name(name)
        for record in self.scan():
            if record.name == safe:
                return record
        return None

    # ── health (fed by the loader / dispatcher) ──────────────────────────────

    def mark_loaded(self, name: str, ok: bool = True, error: str = "") -> None:
        entry = self._health.setdefault(_safe_name(name), {})
        entry["loaded"] = bool(ok)
        entry["error"] = str(error or "")
        if self.telemetry is not None:
            try:
                self.telemetry.health.beat(
                    f"plugin.{_safe_name(name)}",
                    "OK" if ok else "DOWN", (error or "")[:80])
            except Exception:
                pass

    def record_call(self, name: str) -> None:
        entry = self._health.setdefault(_safe_name(name), {})
        entry["calls"] = int(entry.get("calls") or 0) + 1

    # ── lifecycle ────────────────────────────────────────────────────────────

    def create(self, name: str, description: str = "",
               tier: str = _DEFAULT_TIER, version: str = "1.0.0",
               permissions: "tuple[str, ...] | list[str] | None" = None,
               kind: str = "tool",
               events: "tuple[str, ...] | list[str] | None" = None) -> ToolResult:
        """Scaffold a new, immediately-working plugin."""
        safe = _safe_name(name)
        if not safe or safe == "plugin":
            return ToolResult("Give the plugin a name, sir.", ok=False)
        if (self.directory / f"{safe}{_TOOL_SUFFIX}").exists():
            return ToolResult(
                f"A plugin called '{safe}' already exists. Pick another name, "
                "or remove it first.", ok=False)
        try:
            module_path, manifest_path = scaffold(
                safe, description, tier, self.directory,
                version=str(version or "1.0.0"), permissions=permissions,
                kind=kind, events=events)
        except Exception as exc:
            return ToolResult(f"Could not create the plugin: {exc}", ok=False)
        self._state.setdefault("meta", {})[safe] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "scaffold"}
        self._save_state()
        self._log(f"PLUGIN: scaffolded '{safe}' → {module_path.name}")
        return ToolResult(
            f"Created plugin '{safe}' (tier={tier}).\n"
            f"  module   : {module_path}\n"
            f"  manifest : {manifest_path}\n"
            "It already conforms to the contract and will load on the next "
            f"'plugin reload'. Edit run() in the module to give it real work.")

    def install(self, source: str) -> ToolResult:
        """Install a plugin from a path to a ``*_tool.py`` (its manifest is
        copied too when one sits beside it, and generated when it does not)."""
        raw = str(source or "").strip().strip('"').strip("'")
        if not raw:
            return ToolResult("Give me a path to the plugin file, sir.", ok=False)
        # A portable bundle is one file rather than two loose ones - route it to
        # the bundle installer, which validates every member before extracting.
        if raw.lower().endswith((".orionplugin", ".zip")):
            return self.install_bundle(raw)
        src = Path(raw).expanduser()
        if not src.is_file():
            return ToolResult(f"No file at '{src}'.", ok=False)
        if src.suffix != ".py":
            return ToolResult("A plugin module must be a .py file.", ok=False)

        problems = validate_module(src)
        if problems:
            return ToolResult(
                f"'{src.name}' does not satisfy the plugin contract:\n  - "
                + "\n  - ".join(problems)
                + "\nIt needs top-level get_tool_schema() and run() functions.",
                ok=False)

        stem = src.name[: -len(_TOOL_SUFFIX)] if src.name.endswith(_TOOL_SUFFIX) \
            else src.stem
        safe = _safe_name(stem)
        target = self.directory / f"{safe}{_TOOL_SUFFIX}"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        except Exception as exc:
            return ToolResult(f"Could not copy the plugin in: {exc}", ok=False)

        # Bring its manifest along, or synthesise one so it gains a tier.
        sibling = src.parent / f"{stem}{MANIFEST_SUFFIX}"
        manifest_target = self.directory / f"{safe}{MANIFEST_SUFFIX}"
        if sibling.is_file():
            try:
                shutil.copy2(sibling, manifest_target)
            except Exception:
                self._write_manifest(safe, target)
        else:
            self._write_manifest(safe, target)

        self._state.setdefault("meta", {})[safe] = {
            "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src)}
        self._save_state()
        self._log(f"PLUGIN: installed '{safe}' from {src}")
        return ToolResult(
            f"Installed plugin '{safe}' (validated against the contract). "
            "Run 'plugin reload' to bring it live.")

    def remove(self, name: str) -> ToolResult:
        safe = _safe_name(name)
        record = self.get(safe)
        if record is None:
            return ToolResult(f"No plugin named '{name}'.", ok=False)
        removed: list[str] = []
        for path in (self.directory / f"{safe}{_TOOL_SUFFIX}",
                     self.directory / f"{safe}{MANIFEST_SUFFIX}"):
            try:
                if path.is_file():
                    path.unlink()
                    removed.append(path.name)
            except Exception as exc:
                return ToolResult(f"Could not remove {path.name}: {exc}", ok=False)
        self._state.setdefault("meta", {}).pop(safe, None)
        disabled = set(self._state.get("disabled") or [])
        disabled.discard(safe)
        self._state["disabled"] = sorted(disabled)
        self._save_state()
        self._health.pop(safe, None)
        self._log(f"PLUGIN: removed '{safe}'.")
        if not removed:
            return ToolResult(f"Nothing to remove for '{safe}'.", ok=False)
        return ToolResult(
            f"Removed plugin '{safe}' ({', '.join(removed)}). "
            "It stays registered until the next restart.")

    def _write_manifest(self, name: str, module_path: Path,
                        tier: str = _DEFAULT_TIER) -> Path | None:
        """Generate a manifest for a module that has none."""
        safe = _safe_name(name)
        manifest_path = self.directory / f"{safe}{MANIFEST_SUFFIX}"
        payload = {
            "name": safe,
            "description": extract_description(module_path)
                           or f"The {safe.replace('_', ' ')} plugin.",
            "module": module_path.name,
            "tier": tier if tier in CAPABILITY_TIERS else _DEFAULT_TIER,
            "requires": [],
            "parameters": extract_parameters(module_path),
        }
        try:
            atomic_write_text(manifest_path, json.dumps(payload, indent=2), encoding="utf-8")
            return manifest_path
        except Exception as exc:
            self._log(f"PLUGIN: could not write manifest for '{safe}' - {exc}")
            return None

    def audit(self, name: str = "") -> ToolResult:
        """Capability disclosure for one plugin, or every plugin.

        Answers "what does this code actually touch, and did it say so?" — the
        question a user needs before trusting a dropped-in or forged plugin.
        Static only: nothing is imported to produce this.
        """
        records = self.scan()
        if name:
            records = [r for r in records if r.name == str(name).strip()]
            if not records:
                return ToolResult(f"I have no plugin called '{name}', sir.", ok=False)
        if not records:
            return ToolResult("There are no plugins installed yet.")
        lines: list[str] = []
        flagged = 0
        for record in sorted(records, key=lambda r: r.name):
            used = ", ".join(record.permissions) or "nothing declared"
            line = f"  {record.name} (v{record.version}, tier={record.tier}) — declares: {used}"
            if record.undeclared:
                flagged += 1
                line += ("\n      ⚠ uses undeclared: "
                         + ", ".join(record.undeclared))
            lines.append(line)
        header = (f"Capability audit of {len(records)} plugin(s) — "
                  + (f"{flagged} with undeclared capabilities:"
                     if flagged else "every declaration matches the code:"))
        return ToolResult(header + "\n" + "\n".join(lines))

    def check_update(self, name: str, candidate_version: str) -> ToolResult:
        """Compare an installed plugin against a candidate version."""
        record = self.get(name)
        if record is None:
            return ToolResult(f"I have no plugin called '{name}', sir.", ok=False)
        order = compare_versions(candidate_version, record.version)
        if order > 0:
            return ToolResult(
                f"'{name}' is v{record.version}; v{candidate_version} is newer — "
                "install it with plugin install <path> to upgrade.")
        if order == 0:
            return ToolResult(f"'{name}' is already at v{record.version}.")
        return ToolResult(
            f"'{name}' is v{record.version}, which is NEWER than v{candidate_version} "
            "— installing would be a downgrade.")

    # -- portable bundles (share a plugin) ------------------------------------

    def export_bundle(self, name: str, destination: str = "") -> ToolResult:
        """Zip a plugin (module + manifest) into a portable .orionplugin bundle.

        A plugin was previously only shareable by telling someone which two files
        to copy. A bundle is one file that install() accepts directly.
        """
        record = self.get(name)
        if record is None:
            return ToolResult(f"I have no plugin called '{name}', sir.", ok=False)
        module_path = self.directory / (record.module or f"{record.name}{_TOOL_SUFFIX}")
        if not module_path.is_file():
            return ToolResult(f"'{name}' has no module file to export.", ok=False)
        manifest_path = self.directory / f"{record.name}{MANIFEST_SUFFIX}"
        target = Path(destination).expanduser() if str(destination or "").strip() else (
            self.directory / "bundles" / f"{record.name}.orionplugin")
        if target.is_dir():
            target = target / f"{record.name}.orionplugin"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(module_path, module_path.name)
                if manifest_path.is_file():
                    bundle.write(manifest_path, manifest_path.name)
                else:
                    # Export is not a place to lose the capability tier - write a
                    # manifest into the bundle rather than shipping a bare module.
                    bundle.writestr(
                        f"{record.name}{MANIFEST_SUFFIX}",
                        json.dumps(self._manifest_dict(record), indent=2))
        except Exception as exc:
            return ToolResult(f"Could not write the bundle: {exc}", ok=False)
        return ToolResult(
            f"Exported '{record.name}' (v{record.version}, tier={record.tier}) to\n"
            f"  {target}\n"
            "Anyone can add it with: plugin install <path to that file>")

    def _manifest_dict(self, record: "PluginRecord") -> dict[str, Any]:
        return {
            "name": record.name,
            "description": record.description,
            "module": record.module or f"{record.name}{_TOOL_SUFFIX}",
            "tier": record.tier,
            "version": record.version,
            "permissions": list(record.permissions),
            "events": [],
            "requires": list(record.requires),
        }

    def install_bundle(self, source: str) -> ToolResult:
        """Install a plugin from a .orionplugin/.zip bundle.

        Every member is checked before anything is written: only a plain
        ``*_tool.py`` / ``*.plugin.json`` at the archive ROOT is accepted, so a
        malicious bundle cannot path-traverse out of the plugin directory.
        """
        raw = str(source or "").strip().strip('"').strip("'")
        src = Path(raw).expanduser()
        if not src.is_file():
            return ToolResult(f"I can't find a bundle at {src}.", ok=False)
        try:
            with zipfile.ZipFile(src) as bundle:
                members = bundle.namelist()
                safe = [
                    m for m in members
                    if not m.endswith("/")
                    and "/" not in m and "\\" not in m and not m.startswith("..")
                    and (m.endswith(_TOOL_SUFFIX) or m.endswith(MANIFEST_SUFFIX))
                ]
                if not safe:
                    return ToolResult(
                        "That bundle contains no plugin module - expected a "
                        f"*{_TOOL_SUFFIX} at the top level.", ok=False)
                rejected = [m for m in members if m not in safe]
                self.directory.mkdir(parents=True, exist_ok=True)
                for member in safe:
                    bundle.extract(member, self.directory)
        except zipfile.BadZipFile:
            return ToolResult("That file is not a readable plugin bundle.", ok=False)
        except Exception as exc:
            return ToolResult(f"Could not install the bundle: {exc}", ok=False)
        installed = sorted({m[: -len(_TOOL_SUFFIX)] for m in safe
                            if m.endswith(_TOOL_SUFFIX)})
        note = ""
        if rejected:
            note = (f"\n  (ignored {len(rejected)} unexpected file(s) in the bundle: "
                    + ", ".join(rejected[:3]) + ")")
        for plugin_name in installed:
            self._state.setdefault("meta", {})[plugin_name] = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "bundle"}
        self._save_state()
        if not installed:
            return ToolResult("The bundle held a manifest but no module." + note,
                              ok=False)
        return ToolResult(
            f"Installed {', '.join(installed)} from the bundle.{note}\n"
            "Run 'plugin audit' to see what it can touch, then 'plugin reload'.")

    # -- tier vs capability coherence -----------------------------------------

    def risky_grants(self) -> list[tuple[str, str, list[str]]]:
        """Plugins whose capability TIER looks too generous for what they DO.

        The tier decides how a paired remote device may call a plugin: ``allow``
        means unattended, no confirmation. A plugin that spawns processes, drives
        the keyboard/mouse or reaches the network is not something that should run
        unattended from a phone just because nobody thought about its tier — and
        until the capability disclosure existed there was no way to notice.

        Returns (name, tier, dangerous-capabilities) triples. It reports; it never
        silently downgrades a tier the user chose.
        """
        risky: list[tuple[str, str, list[str]]] = []
        for record in self.scan():
            if record.tier != "allow":
                continue
            touches = sorted(set(record.permissions) | set(record.undeclared))
            dangerous = [c for c in touches if c in _UNATTENDED_RISK]
            if dangerous:
                risky.append((record.name, record.tier, dangerous))
        return risky

    def manifestless(self) -> list[str]:
        """Plugins still lacking a manifest AFTER a backfill — i.e. the ones
        skipped because their module does not satisfy the contract.  Reported
        separately so 'nothing was written' is never mistaken for 'all fine'."""
        return [record.name for record in self.scan() if not record.has_manifest]

    def backfill_manifests(self) -> list[str]:
        """Give every manifest-less ``*_tool.py`` a manifest.

        This is the security fix for the forged tools: without a manifest they
        load through the code-only path carrying NO declared capability tier,
        so the remote gate has nothing to classify them by.  Backfilling at the
        safe default (``confirm``) means an unattended remote request must be
        confirmed rather than silently allowed.
        """
        written: list[str] = []
        for module_path in sorted(self.directory.glob(f"*{_TOOL_SUFFIX}")):
            name = module_path.name[: -len(_TOOL_SUFFIX)]
            if (self.directory / f"{name}{MANIFEST_SUFFIX}").is_file():
                continue
            if validate_module(module_path):
                continue            # malformed — leave it for doctor() to report
            if self._write_manifest(name, module_path) is not None:
                written.append(name)
        if written:
            self._log(
                f"PLUGIN: generated {len(written)} manifest(s) at tier="
                f"{_DEFAULT_TIER} — {', '.join(written)}")
        return written

    # ── dependencies ─────────────────────────────────────────────────────────

    async def install_dependencies(self, name: str) -> ToolResult:
        """Install a plugin's missing dependencies via the Forge's resolver,
        instead of skipping the plugin with a log line the user never sees."""
        record = self.get(name)
        if record is None:
            return ToolResult(f"No plugin named '{name}'.", ok=False)
        if not record.missing_deps:
            return ToolResult(f"'{record.name}' has everything it needs.")
        try:
            from .dependencies import DynamicPackageResolver
            resolver = DynamicPackageResolver(self.bus)
            outcome = await resolver.resolve_and_install(list(record.missing_deps))
        except Exception as exc:
            return ToolResult(
                f"Could not install dependencies for '{record.name}': {exc}", ok=False)
        if getattr(outcome, "succeeded", False):
            return ToolResult(
                f"Installed {', '.join(record.missing_deps)} for '{record.name}'. "
                "Run 'plugin reload' to bring it live.")
        still = ", ".join(getattr(outcome, "missing_packages", None)
                          or record.missing_deps)
        return ToolResult(
            f"Could not install every dependency for '{record.name}' "
            f"(outstanding: {still}).", ok=False)

    # ── reporting ────────────────────────────────────────────────────────────

    def list_plugins(self) -> ToolResult:
        records = self.scan()
        if not records:
            return ToolResult(
                "No plugins installed. Create one with:  plugin create "
                "<name> — I'll scaffold a working plugin you can edit.")
        live = sum(1 for r in records if r.enabled and r.healthy)
        lines = [f"PLUGINS — {len(records)} installed, {live} healthy:", ""]
        lines += [f"  {record.summary()}" for record in records]
        lines.append("")
        lines.append("Actions: describe / audit / enable / disable / reload / "
                     "create / install / export / remove / deps / hooks / "
                     "unmute / update / backfill / doctor.")
        return ToolResult("\n".join(lines))

    def describe(self, name: str) -> ToolResult:
        record = self.get(name)
        if record is None:
            return ToolResult(f"No plugin named '{name}'.", ok=False)
        lines = [
            f"PLUGIN: {record.name}",
            f"  status      : {record.status()}",
            f"  description : {record.description or '(none)'}",
            f"  module      : {record.module or '(none)'}",
            f"  version     : {record.version}",
            f"  capability  : {record.tier}",
            f"  can touch   : {', '.join(record.permissions) or '(nothing declared)'}",
            f"  manifest    : {'yes' if record.has_manifest else 'no (using defaults)'}",
            f"  requires    : {', '.join(record.requires) or '(nothing)'}",
            f"  calls       : {record.calls}",
        ]
        if record.undeclared:
            lines.append(f"  UNDECLARED  : {', '.join(record.undeclared)}")
            lines.append("                (it uses these without declaring them)")
        if record.missing_deps:
            lines.append(f"  MISSING     : {', '.join(record.missing_deps)}")
            lines.append("                → 'plugin deps <name>' installs them.")
        for problem in record.problems:
            lines.append(f"  PROBLEM     : {problem}")
        if record.last_error:
            lines.append(f"  LAST ERROR  : {record.last_error[:200]}")
        return ToolResult("\n".join(lines))

    def doctor(self, events: Any = None) -> ToolResult:
        """One health pass over the whole plugin surface."""
        records = self.scan()
        problems: list[str] = []
        for record in records:
            if not record.module_exists:
                problems.append(f"{record.name}: module file missing")
            for problem in record.problems:
                problems.append(f"{record.name}: {problem}")
            if record.missing_deps:
                problems.append(
                    f"{record.name}: needs {', '.join(record.missing_deps)} "
                    f"→ 'plugin deps {record.name}'")
            if record.last_error:
                problems.append(f"{record.name}: {record.last_error[:120]}")
        no_manifest = [r.name for r in records if not r.has_manifest]
        quarantined = self._quarantined()

        lines = [f"PLUGIN DOCTOR — {len(records)} plugin(s) inspected.", ""]
        if problems:
            lines.append(f"{len(problems)} problem(s):")
            lines += [f"  ✗ {problem}" for problem in problems]
        else:
            lines.append("  ✓ No problems found.")
        if no_manifest:
            lines.append("")
            lines.append(
                f"  ! {len(no_manifest)} plugin(s) have no manifest, so they carry "
                f"no declared capability tier: {', '.join(no_manifest)}")
            lines.append("    → 'plugin backfill' writes safe manifests for them.")
        if quarantined:
            lines.append("")
            lines.append(f"  ! {len(quarantined)} quarantined module(s): "
                         + ", ".join(quarantined))
        undeclared = [r for r in records if r.undeclared]
        if undeclared:
            lines.append("")
            lines.append(f"  ! {len(undeclared)} plugin(s) use capabilities they "
                         "never declared:")
            for record in undeclared[:8]:
                lines.append(f"      {record.name} -> {', '.join(record.undeclared)}")
            lines.append("    -> 'plugin audit' for the full disclosure.")
        risky = self.risky_grants()
        if risky:
            lines.append("")
            lines.append(f"  ! {len(risky)} plugin(s) run UNATTENDED (tier=allow) "
                         "while touching sensitive capabilities:")
            for name, _tier, caps in risky[:8]:
                lines.append(f"      {name} -> " + ", ".join(caps))
            lines.append("    -> consider: plugin create/edit its manifest tier as "
                         "'confirm' so a remote call asks first.")
        if events is not None:
            try:
                hooked = events.hooked()
                muted = [n for n, s in getattr(events, "stats", {}).items() if s.muted]
            except Exception:
                hooked, muted = {}, []
            lines.append("")
            if hooked:
                lines.append(f"  * {len(hooked)} plugin(s) hooked to ORION events: "
                             + ", ".join(sorted(hooked)))
            else:
                lines.append("  * No plugin is subscribed to any event.")
            if muted:
                lines.append("    ! MUTED (faulting or too slow): " + ", ".join(muted))
        return ToolResult("\n".join(lines), ok=not problems)

    def _quarantined(self) -> list[str]:
        try:
            index = self.directory / "_quarantine" / "index.json"
            if not index.is_file():
                return []
            data = json.loads(index.read_text(encoding="utf-8"))
            return sorted(data.keys()) if isinstance(data, dict) else []
        except Exception:
            return []

    def snapshot(self) -> dict[str, Any]:
        """Machine-readable state for the GUI panel and diagnostics."""
        records = self.scan()
        return {
            "total": len(records),
            "enabled": sum(1 for r in records if r.enabled),
            "healthy": sum(1 for r in records if r.enabled and r.healthy),
            "plugins": [r.to_dict() for r in records],
        }
