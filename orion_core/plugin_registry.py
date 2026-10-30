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
import inspect
import json
import os
import tempfile
import shutil
import zipfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from threading import RLock
from typing import Any, Callable

from .constants import CONFIG_DIR
from .data import ToolResult
from .plugin_manifest import (
    CAPABILITY_TIERS,
    MANIFEST_SUFFIX,
    PluginManifest,
    discover_manifests,
    load_manifest,
    plugin_identity,
    unmet_requirements,
)
from .atomic_io import atomic_write_text, atomic_write_bytes, _replace

PLUGIN_STATE_PATH = CONFIG_DIR / "plugins.json"
CUSTOM_TOOLS_DIR = CONFIG_DIR / "custom_tools"
_DEFAULT_TIER = "confirm"          # safest tier for anything undeclared
_TOOL_SUFFIX = "_tool.py"
MAX_PLUGIN_FILE_BYTES = 8 * 1024 * 1024
MAX_PLUGIN_BUNDLE_BYTES = 32 * 1024 * 1024


class _InstallRecoveryRequired(RuntimeError):
    """Publication failed and its staging backups must be retained."""


def detach_runtime(dispatcher: Any, name: str) -> None:
    """Retire future registrations after a successful removal receipt.

    Python objects held by in-flight calls may outlive their registration.
    The registry's saved disabled tombstone also holds those cached calls.
    """
    safe = _safe_name(name)
    retired = {name, safe}
    handlers = getattr(dispatcher, "_tool_handlers", {})
    if isinstance(handlers, dict):
        retired.update(key for key in handlers if _safe_name(key) == safe)
    scheduler = getattr(dispatcher, "plugin_scheduler", None)
    if scheduler is not None:
        entries = getattr(scheduler, "entries", {})
        if isinstance(entries, dict):
            retired.update(key for key in entries if _safe_name(key) == safe)
        for key in retired:
            try:
                scheduler.remove(key)
            except Exception:
                pass
    removed_handlers = set()
    if isinstance(handlers, dict):
        for key in retired:
            if handlers.pop(key, None) is not None:
                removed_handlers.add(key)
    rollbacks = getattr(dispatcher, "_forge_rollbacks", {})
    if isinstance(rollbacks, dict):
        for key in list(rollbacks):
            if _safe_name(key) == safe:
                rollbacks.pop(key, None)
    declarations = getattr(dispatcher, "TOOL_DECLARATIONS", None)
    if removed_handlers and isinstance(declarations, list):
        # A native name collision must not remove the built-in declaration.
        table = getattr(dispatcher, "handler_table", None)
        try:
            native = table() if callable(table) else {}
        except Exception:
            return  # Keep declarations if native ownership cannot be checked.
        connected = getattr(dispatcher, "_mcp_routes", {})
        declarations[:] = [item for item in declarations
                           if item.get("name") not in removed_handlers
                           or item.get("name") in native or item.get("name") in connected]


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
    manifest_file: str = ""
    secrets: tuple[str, ...] = ()
    source_sha256: str = ""
    #: Why its module sits in custom_tools/_quarantine (the loader moved it
    #: there); the manifest alone stays active. 25 Oct 2026: this showed red
    #: as "module file not found", as if the plugin had been lost.
    quarantined: str = ""
    #: The built-in tool this plugin duplicates. Held back on purpose, so it
    #: is a neutral state, not a load error (it showed red as one).
    superseded_by: str = ""

    @property
    def healthy(self) -> bool:
        return (self.module_exists and not self.missing_deps
                and not self.problems and not self.last_error)

    def status(self) -> str:
        if self.superseded_by:
            return "superseded"
        if self.quarantined:
            return "quarantined"
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
        if self.superseded_by:
            bits.append(f"superseded by built-in {self.superseded_by}")
        if self.quarantined:
            bits.append(f"quarantined: {self.quarantined[:80]} (restore or remove it)")
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
            "secrets": list(self.secrets),
            "source_sha256": self.source_sha256,
            "quarantined": self.quarantined,
            "superseded_by": self.superseded_by,
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


#: Static audits by content: (kind, file name, sha256) -> result. The Plugins
#: page rescans every few seconds, and re-parsing two dozen unchanged plugins
#: each time held the interpreter lock for 50-370 ms (28 Oct 2026). Keyed by
#: the content's hash, re-read on every call, so any edit is audited afresh.
_AUDITS: dict[tuple[str, str, str], Any] = {}


def _audited(kind: str, path: Path, compute: Callable[[Path], Any]) -> Any:
    import hashlib
    path = Path(path)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return compute(path)
    key = (kind, path.name, digest)
    if key not in _AUDITS:
        if len(_AUDITS) > 512:
            _AUDITS.clear()
        _AUDITS[key] = compute(path)
    return _AUDITS[key]


def detect_capabilities(path: Path) -> set[str]:
    """detect_capabilities_uncached, once per version of the file."""
    return set(_audited("capabilities", path, detect_capabilities_uncached))


def detect_capabilities_uncached(path: Path) -> set[str]:
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
    """validate_module_uncached, once per version of the file."""
    return list(_audited("contract", path, validate_module_uncached))


def validate_module_uncached(path: Path) -> list[str]:
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
    """Apply the loader's shared contract to a plain, literal schema.

    Only a direct literal return is inspected: computed or decorated exports
    remain the runtime loader's responsibility. Ordinary data dictionaries are
    not tool declarations. Defaults and annotations are represented by inert
    placeholders, so validating a candidate never executes its code.
    """
    from .forge_contract import contract_problems, schema_problems

    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # Conditional definitions, imports and other module-scope bindings can
    # replace a literal stub. Their resulting contract belongs to the loader.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _module_scope_rebinds_export(node):
            return []
    schema_fn = functions.get("get_tool_schema")
    if schema_fn is None or schema_fn.decorator_list:
        return []
    body = schema_fn.body[1:] if ast.get_docstring(schema_fn) is not None else schema_fn.body
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return []
    try:
        schema = ast.literal_eval(body[0].value)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return []

    try:
        problems, _properties = schema_problems(schema)
    except (TypeError, ValueError) as exc:
        return [f"CONTRACT: invalid literal tool schema: {exc}"]
    run_fn = functions.get("run")
    if run_fn is not None and not run_fn.decorator_list:
        def placeholder():
            pass

        try:
            placeholder.__signature__ = _static_function_signature(run_fn)
        except ValueError as exc:
            problems.append(f"run() has an invalid signature: {exc}")
        else:
            problems = contract_problems(schema, placeholder)
    return [f"CONTRACT: {problem}" for problem in problems]


def _module_scope_rebinds_export(node: ast.AST) -> bool:
    """Conservatively identify export bindings without entering local scopes."""
    exports = {"get_tool_schema", "run"}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name in exports  # the body has its own scope
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return node.id in exports
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any((alias.asname or alias.name.split(".")[0]) in exports
                   or alias.name == "*" for alias in node.names)
    if isinstance(node, ast.ExceptHandler) and node.name in exports:
        return True
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in exports:
        return True
    if isinstance(node, ast.MatchMapping) and node.rest in exports:
        return True
    return any(_module_scope_rebinds_export(child) for child in ast.iter_child_nodes(node))


def _static_function_signature(function: Any) -> inspect.Signature:
    """Mirror an AST signature without evaluating defaults or annotations."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults_start = len(positional) - len(arguments.defaults)
    parameters = [
        inspect.Parameter(
            argument.arg,
            inspect.Parameter.POSITIONAL_ONLY if index < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if index < defaults_start else None,
        )
        for index, argument in enumerate(positional)
    ]
    if arguments.vararg is not None:
        parameters.append(inspect.Parameter(arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    parameters.extend(
        inspect.Parameter(argument.arg, inspect.Parameter.KEYWORD_ONLY,
                          default=inspect.Parameter.empty if default is None else None)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
    )
    if arguments.kwarg is not None:
        parameters.append(inspect.Parameter(arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


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
    description = str(description or "").strip() or f"The {safe.replace('_', ' ')} plugin."
    tier = tier if tier in CAPABILITY_TIERS else _DEFAULT_TIER

    module_path = directory / f"{safe}{_TOOL_SUFFIX}"
    manifest_path = directory / f"{safe}{MANIFEST_SUFFIX}"
    # The same escaped content is valid inside the module docstring and schema
    # string. Preserve detailed purposes, including newlines and Windows paths,
    # without turning their quotes/backslashes into executable Python syntax.
    escaped_description = json.dumps(description, ensure_ascii=False)[1:-1]
    reactive = str(kind or "tool").strip().lower() in {
        "event", "events", "listener", "reactive", "watcher"}
    template = _TEMPLATE_LISTENER if reactive else _TEMPLATE
    body = template.format(name=safe, description=escaped_description)
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
    return plugin_identity(name)


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


def _forget_quarantined(target: Path, module_name: str) -> None:
    """Remove a quarantined copy a delivered version has replaced. If the new
    one fails to load, the loader quarantines it again with a fresh reason."""
    qdir = target / "_quarantine"
    broken = qdir / f"{module_name}.broken"
    try:
        broken.unlink(missing_ok=True)
        index_path = qdir / "index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(index, dict) and index.pop(broken.name, None) is not None:
                atomic_write_text(index_path, json.dumps(index, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def _drop_quarantine_entry(qdir: Path, broken_name: str) -> None:
    """Remove *broken_name*'s line from the quarantine index (the file itself
    has already been restored or archived)."""
    index_path = qdir / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(index, dict) and index.pop(broken_name, None) is not None:
        atomic_write_text(index_path, json.dumps(index, indent=2), encoding="utf-8")


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
            if quarantined:
                _forget_quarantined(target, module.name)
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
        self._committed_disabled: frozenset[str] = frozenset()
        self._runtime_holds: dict[str, str] = {}
        self._ambiguous_identities: frozenset[str] = frozenset()
        self._state_error = ""
        self._file_lock = RLock()
        self._state_lock = RLock()
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
            data = json.loads(PLUGIN_STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("saved plugin state must be an object")
            disabled = data.get("disabled", [])
            meta = data.get("meta", {})
            if (not isinstance(disabled, list)
                    or any(not isinstance(name, str) or not name.strip() for name in disabled)):
                raise ValueError("disabled plugins must be a list of non-empty names")
            if not isinstance(meta, dict):
                raise ValueError("plugin metadata must be an object")
            self._state = {"disabled": list(disabled), "meta": dict(meta)}
            self._committed_disabled = frozenset(_safe_name(name) for name in disabled)
            self._state_error = ""
        except FileNotFoundError:
            # A missing file is the supported first-run state. An unreadable
            # existing file is not permission to forget saved disable choices.
            self._state = {"disabled": [], "meta": {}}
            self._committed_disabled = frozenset()
            self._state_error = ""
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            if reason != self._state_error:
                self._log(f"PLUGIN: saved state unavailable; plugins held - {reason}")
            self._state_error = reason

    @property
    def state_error(self) -> str:
        return self._state_error

    def _unavailable_state(self) -> ToolResult:
        return ToolResult(
            f"Saved plugin preferences are unavailable: {self._state_error}. "
            f"Plugins are held and {PLUGIN_STATE_PATH} has been preserved. "
            "Restore or repair that file, then restart ORION.", ok=False)

    def _save_state(self) -> bool:
        with self._state_lock:
            return self._save_state_locked()

    def _save_state_locked(self) -> bool:
        if self._state_error:
            return False
        temporary = backup = None
        previous = None
        published = keep_backup = False
        try:
            PLUGIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                previous = PLUGIN_STATE_PATH.read_bytes()
            except FileNotFoundError:
                pass
            if previous is not None:
                with tempfile.NamedTemporaryFile(mode="wb", dir=PLUGIN_STATE_PATH.parent,
                        suffix=".recovery.tmp", delete=False) as stream:
                    backup = Path(stream.name)
                    stream.write(previous)
                    stream.flush()
                    os.fsync(stream.fileno())
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                    dir=PLUGIN_STATE_PATH.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self._state, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, PLUGIN_STATE_PATH)
            published = True
            if json.loads(PLUGIN_STATE_PATH.read_text(encoding="utf-8")) != self._state:
                raise ValueError("saved preferences did not match the requested state")
            return True
        except Exception as exc:
            if published:
                try:
                    if backup is None:
                        PLUGIN_STATE_PATH.unlink(missing_ok=True)
                    else:
                        atomic_write_bytes(PLUGIN_STATE_PATH, previous)
                        if PLUGIN_STATE_PATH.read_bytes() != previous:
                            raise OSError("previous preferences could not be verified")
                except Exception as recovery:
                    keep_backup = True
                    self._state_error = (
                        f"saved preference recovery failed: {recovery}; "
                        f"previous bytes retained at {backup}" if backup is not None
                        else f"first-run preference recovery failed: {recovery}")
            self._log(f"PLUGIN: could not save state - {exc}")
            return False
        finally:
            for path in (temporary, None if keep_backup else backup):
                if path is None:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def is_enabled(self, name: str) -> bool:
        safe = _safe_name(name)
        return (not self._state_error and safe not in self._committed_disabled
                and safe not in self._runtime_holds and safe not in self._ambiguous_identities)

    def set_enabled(self, name: str, enabled: bool) -> ToolResult:
        with self._state_lock:
            return self._set_enabled_locked(name, enabled)

    def _set_enabled_locked(self, name: str, enabled: bool) -> ToolResult:
        if self._state_error:
            return self._unavailable_state()
        safe = _safe_name(name)
        if safe in self._runtime_holds:
            return ToolResult(self._runtime_holds[safe] + " Restore the retained files and restart ORION.", ok=False)
        known = {_safe_name(record.name) for record in self.scan()}
        if safe in self._ambiguous_identities:
            return ToolResult(f"Plugin '{name}' has ambiguous identities; resolve the metadata files first.", ok=False)
        if safe not in known:
            return ToolResult(f"No plugin named '{name}'.", ok=False)
        previous = list(self._state.get("disabled") or [])
        disabled = {_safe_name(item) for item in previous}
        if enabled:
            disabled.discard(safe)
        else:
            disabled.add(safe)
        self._state["disabled"] = sorted(disabled)
        if not self._save_state():
            self._state["disabled"] = previous
            detail = f" {self._state_error}" if self._state_error else ""
            return ToolResult(f"Could not verify the saved setting for plugin '{safe}'.{detail}", ok=False)
        self._committed_disabled = frozenset(disabled)
        word = "enabled" if enabled else "disabled"
        self._log(f"PLUGIN: '{safe}' {word}.")
        note = "" if enabled else " It stays off until you enable it again."
        return ToolResult(f"Plugin '{safe}' {word}.{note}")

    # ── discovery ────────────────────────────────────────────────────────────

    def scan(self) -> list[PluginRecord]:
        """Every plugin ORION can see: manifest-declared ones first, then any
        orphan ``*_tool.py`` that has no manifest yet (the forged tools)."""
        records: dict[str, PluginRecord] = {}
        manifests, errors = discover_manifests(self.directory)
        owned_modules = {manifest.module for manifest in manifests}

        for manifest in manifests:
            record = self._record_from_manifest(manifest)
            identity = _safe_name(record.name)
            if identity in records:
                record.problems += ("Duplicate manifest identity; resolve the metadata files before loading or removal",)
            records[identity] = record

        # Orphans: a *_tool.py with no manifest beside it still loads through
        # the forge path, so it must be visible here — that is exactly the set
        # that currently carries no declared capability tier.
        for module_path in sorted(self.directory.glob(f"*{_TOOL_SUFFIX}")):
            name = module_path.name[: -len(_TOOL_SUFFIX)]
            if module_path.name in owned_modules:
                continue
            identity = _safe_name(name)
            if identity in records:
                records[identity].problems += ("Duplicate manifest identity; an orphan module has the same management name",)
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
                undeclared=tuple(sorted(detect_capabilities(module_path))),
                source_sha256=_digest(module_path),
            )
            records[identity] = record

        for manifest_name, reason in errors.items():
            stem = manifest_name.replace(MANIFEST_SUFFIX, "")
            identity = _safe_name(stem)
            problem = (reason,)
            if identity in records:
                records[identity].problems += ("Duplicate manifest identity; invalid metadata has the same management name",)
                continue
            records[identity] = PluginRecord(
                name=stem, enabled=self.is_enabled(stem),
                has_manifest=True, problems=problem, manifest_file=manifest_name)

        self._ambiguous_identities = frozenset(
            identity for identity, record in records.items()
            if any("Duplicate manifest identity" in problem for problem in record.problems))
        for record in records.values():
            record.enabled = self.is_enabled(record.name)
            health = self._health.get(_safe_name(record.name))
            if health:
                record.loaded = bool(health.get("loaded"))
                record.last_error = str(health.get("error") or "")
                record.calls = int(health.get("calls") or 0)
                record.superseded_by = str(health.get("superseded_by") or "")
            if record.quarantined and record.last_error.startswith("module file"):
                record.last_error = ""      # the quarantine IS the explanation
            if _safe_name(record.name) in self._runtime_holds:
                record.last_error = self._runtime_holds[_safe_name(record.name)]
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
            manifest_file=manifest.source_path.name if manifest.source_path else "",
            secrets=tuple(manifest.secrets),
            source_sha256=_digest(module_path) if exists and module_path else "",
            quarantined="" if exists else self.quarantine_reason_for_module(manifest.module),
        )

    def get(self, name: str) -> PluginRecord | None:
        safe = _safe_name(name)
        for record in self.scan():
            if _safe_name(record.name) == safe:
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

    # ── quarantined and superseded plugins (25 Oct 2026, owner report) ───────
    #
    # The Plugins page showed three gesture plugins in red as load ERRORS
    # ("held back: it duplicates ORION's built-in tool 'gesture_control'") and
    # jv_telegram_remote in red as "module file not found". Both are deliberate
    # states, not faults: a duplicate is superseded by the built-in, and the
    # Telegram module had been quarantined by the loader, its manifest left
    # behind. They are now shown for what they are, a duplicate is archived
    # out of the active set (a timestamped copy in _superseded/, never
    # deleted), and a quarantined plugin can be restored or removed.

    SUPERSEDED_DIR = "_superseded"
    SUPERSEDED_INDEX = "index.json"

    def quarantine_reason_for_module(self, module: str) -> str:
        """Why *module* sits in _quarantine, or '' when it does not."""
        module = str(module or "").strip()
        if not module or Path(module).name != module:
            return ""
        qdir = self.directory / "_quarantine"
        broken = qdir / f"{module}.broken"
        try:
            if not broken.is_file():
                return ""
        except OSError:
            return ""
        reason = ""
        try:
            index = json.loads((qdir / "index.json").read_text(encoding="utf-8"))
            if isinstance(index, dict) and isinstance(index.get(broken.name), dict):
                reason = str(index[broken.name].get("reason") or "")
        except (OSError, ValueError):
            pass
        return reason or "it failed to load and was set aside"

    def quarantine_reason(self, name: str) -> str:
        """The quarantine reason for the plugin *name* (by its manifest)."""
        safe = _safe_name(name)
        manifests, _errors = discover_manifests(self.directory)
        for manifest in manifests:
            if _safe_name(manifest.name) == safe:
                module_path = manifest.module_path
                if module_path is not None and module_path.is_file():
                    return ""
                return self.quarantine_reason_for_module(manifest.module)
        return ""

    def _archive_files(self, paths: list[Path], receipt: dict[str, Any]) -> list[str]:
        """Move *paths* into _superseded/ as ``<name>.<stamp>`` copies and log
        a receipt in its index. Each copy is verified byte-for-byte before the
        original is removed, so nothing is ever lost. Returns archived names."""
        archive = self.directory / self.SUPERSEDED_DIR
        archive.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        moved: list[str] = []
        for path in paths:
            data = path.read_bytes()
            target = archive / f"{path.name}.{stamp}"
            serial = 1
            while target.exists():
                target = archive / f"{path.name}.{stamp}-{serial}"
                serial += 1
            shutil.copy2(path, target)
            if target.read_bytes() != data:
                raise OSError(f"the archived copy of {path.name} did not verify")
            path.unlink()
            moved.append(target.name)
        index_path = archive / self.SUPERSEDED_INDEX
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(index, list):
                index = []
        except (OSError, ValueError):
            index = []
        index.append({**receipt, "files": moved,
                      "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        atomic_write_text(index_path, json.dumps(index[-200:], indent=2), encoding="utf-8")
        return moved

    def _plugin_files(self, record: "PluginRecord") -> list[Path]:
        root = self.directory.resolve()
        names = [record.module] if record.module else []
        if record.has_manifest:
            names.append(record.manifest_file or f"{_safe_name(record.name)}{MANIFEST_SUFFIX}")
        files: list[Path] = []
        for filename in names:
            if (not filename or Path(filename).name != filename
                    or PureWindowsPath(filename).name != filename or ":" in filename):
                raise ValueError("plugin files must be plain sibling filenames")
            path = root / filename
            if path.is_symlink():
                raise ValueError(f"refusing to move {filename} through a link")
            if path.is_file():
                files.append(path)
        return files

    def supersede(self, name: str, builtin: str, reason: str = "") -> ToolResult:
        """A plugin held back because it duplicates the built-in *builtin*:
        recorded as superseded (neutral, not an error) and archived out of the
        active set. Only the loader's exact-duplicate verdict calls this."""
        with self._file_lock:
            safe = _safe_name(name)
            entry = self._health.setdefault(safe, {})
            entry.update({"loaded": False, "error": "", "superseded_by": str(builtin)})
            record = self.get(safe)
            if record is None:
                return ToolResult(f"No plugin named '{name}'.", ok=False)
            try:
                files = self._plugin_files(record)
                if not files:
                    return ToolResult(f"'{safe}' has no files left to archive.")
                moved = self._archive_files(files, {
                    "name": record.name, "superseded_by": str(builtin),
                    "reason": reason or f"duplicates the built-in tool '{builtin}'"})
            except Exception as exc:
                self._log(f"PLUGIN: '{safe}' is superseded by the built-in '{builtin}' "
                          f"(left in place: {exc}).")
                return ToolResult(f"'{safe}' is superseded by '{builtin}' but could not be "
                                  f"archived: {exc}", ok=False)
            self._log(f"PLUGIN: '{safe}' is superseded by the built-in '{builtin}' - archived "
                      f"to custom_tools/{self.SUPERSEDED_DIR}/ ({', '.join(moved)}).")
            return ToolResult(f"Archived '{safe}' (superseded by the built-in {builtin}).",
                              receipt={"archived": moved, "superseded_by": str(builtin)})

    def superseded_archive(self) -> list[dict[str, Any]]:
        """Receipts of plugins archived as superseded, newest first."""
        try:
            index = json.loads((self.directory / self.SUPERSEDED_DIR / self.SUPERSEDED_INDEX)
                               .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(index, list):
            return []
        return [dict(item) for item in reversed(index)
                if isinstance(item, dict) and item.get("superseded_by")]

    def restore_quarantined(self, name: str) -> ToolResult:
        """Put a quarantined plugin's module back, if it would pass the forge's
        safety gate today. The quarantined copy is archived, not deleted; a
        module that still fails to load is quarantined again on the next load."""
        with self._file_lock:
            if self._state_error:
                return self._unavailable_state()
            record = self.get(name)
            if record is None:
                return ToolResult(f"No plugin named '{name}'.", ok=False)
            if not record.quarantined:
                return ToolResult(f"Plugin '{record.name}' is not quarantined.", ok=False)
            qdir = self.directory / "_quarantine"
            broken = qdir / f"{record.module}.broken"
            target = self.directory / record.module
            try:
                source = broken.read_text(encoding="utf-8", errors="replace")
                from .security import SecuritySanitiser, SecurityViolation
                trial = record.module[: -len(_TOOL_SUFFIX)] if record.module.endswith(_TOOL_SUFFIX) else record.name
                try:
                    SecuritySanitiser.guard_forged_source(source, f"restore.{broken.name}",
                                                          trial_name=trial)
                except SecurityViolation as exc:
                    return ToolResult(
                        f"'{record.name}' cannot be restored as it is: {exc}. Remove it, or ask the "
                        "forge to rebuild it.", ok=False, receipt={"restored": False})
                if target.exists():
                    return ToolResult(f"A live '{record.module}' already exists; nothing restored.", ok=False)
                shutil.copy2(broken, target)
                if target.read_bytes() != broken.read_bytes():
                    target.unlink(missing_ok=True)
                    raise OSError("the restored copy did not verify")
                self._archive_files([broken], {"name": record.name, "restored_from_quarantine": True,
                                               "reason": record.quarantined})
                _drop_quarantine_entry(qdir, broken.name)
            except Exception as exc:
                return ToolResult(f"Could not restore '{record.name}': {exc}", ok=False)
            self._health.pop(_safe_name(record.name), None)
            self._log(f"PLUGIN: restored '{record.name}' from quarantine - it loads on the next reload.")
            return ToolResult(
                f"Restored '{record.name}'. Say 'reload plugins' (or restart) to load it; if it "
                f"still fails ({record.quarantined}) the loader sets it aside again.",
                receipt={"restored": True, "module": record.module})

    # ── lifecycle ────────────────────────────────────────────────────────────

    def create(self, name: str, description: str = "",
               tier: str = _DEFAULT_TIER, version: str = "1.0.0",
               permissions: "tuple[str, ...] | list[str] | None" = None,
               kind: str = "tool",
               events: "tuple[str, ...] | list[str] | None" = None) -> ToolResult:
        """Scaffold a new, immediately-working plugin."""
        with self._file_lock:
            return self._create_locked(name, description, tier, version, permissions, kind, events)

    def _create_locked(self, name: str, description: str, tier: str, version: str,
                       permissions: Any, kind: str, events: Any) -> ToolResult:
        if self._state_error:
            return self._unavailable_state()
        safe = _safe_name(name)
        if not safe or safe == "plugin":
            return ToolResult("Give the plugin a name, sir.", ok=False)
        if self.get(safe) is not None or (self.directory / f"{safe}{_TOOL_SUFFIX}").exists():
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
        self._record_install(safe, "scaffold")
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
        with self._file_lock:
            return self._install_locked(source)

    def _install_locked(self, source: str) -> ToolResult:
        if self._state_error:
            return self._unavailable_state()
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

        stem = src.name[: -len(_TOOL_SUFFIX)] if src.name.endswith(_TOOL_SUFFIX) \
            else src.stem
        safe = _safe_name(stem)
        try:
            files = {f"{safe}{_TOOL_SUFFIX}": self._candidate_bytes(src)}
            sibling = src.parent / f"{stem}{MANIFEST_SUFFIX}"
            if sibling.exists():
                payload = json.loads(self._candidate_bytes(sibling).decode("utf-8"))
                manifest = PluginManifest.from_dict(payload, source_path=sibling)
                if manifest.module != src.name:
                    raise ValueError("The manifest must describe the selected module")
                payload["module"] = f"{safe}{_TOOL_SUFFIX}"
                files[f"{safe}{MANIFEST_SUFFIX}"] = json.dumps(payload, indent=2).encode("utf-8")
            installed = self._install_candidates(files)
        except Exception as exc:
            return ToolResult(f"Could not install the plugin: {exc}", ok=False)
        saved = self._record_installs(installed, str(src))
        self._log(f"PLUGIN: installed '{safe}' from {src}")
        return ToolResult(
            f"Installed plugin '{safe}' (validated against the contract). "
            "Run 'plugin reload' to bring it live."
            + ("" if saved else " Install metadata could not be saved."))

    @staticmethod
    def _candidate_bytes(path: Path) -> bytes:
        with path.open("rb") as stream:
            data = stream.read(MAX_PLUGIN_FILE_BYTES + 1)
        if len(data) > MAX_PLUGIN_FILE_BYTES:
            raise ValueError(f"{path.name} exceeds the plugin file budget")
        return data

    def _record_install(self, name: str, source: str) -> bool:
        return self._record_installs([name], source)

    def _record_installs(self, names: list[str], source: str) -> bool:
        with self._state_lock:
            for name in names:
                self._state.setdefault("meta", {})[_safe_name(name)] = {
                    "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source}
            return self._save_state()

    def _install_candidates(self, files: dict[str, bytes]) -> list[str]:
        """Validate a complete batch, then publish with rollback on IO failure.

        Each replacement is atomic, but this is not a multi-file transaction
        across a process crash or an unrelated concurrent filesystem reader.
        """
        modules = [name for name in files if name.endswith(".py")]
        if len({name.casefold() for name in files}) != len(files):
            raise ValueError("Plugin filenames collide on a case-insensitive filesystem")
        if not modules:
            raise ValueError("The candidate contains no plugin module")
        if sum(map(len, files.values())) > MAX_PLUGIN_BUNDLE_BYTES:
            raise ValueError("The candidate exceeds the expanded plugin bundle budget")
        self.directory.mkdir(parents=True, exist_ok=True)
        root = self.directory.resolve()
        staging = Path(tempfile.mkdtemp(prefix=".install-stage-", dir=root))
        keep_recovery = False
        try:
            for name, data in files.items():
                if (Path(name).name != name or PureWindowsPath(name).name != name
                        or ":" in name or any(ord(char) < 32 for char in name)):
                    raise ValueError("Plugin members must be plain sibling filenames")
                if len(data) > MAX_PLUGIN_FILE_BYTES:
                    raise ValueError(f"{name} exceeds the plugin file budget")
                (staging / name).write_bytes(data)
            supplied, errors = discover_manifests(staging)
            if errors:
                raise ValueError("; ".join(errors.values()))
            owned_modules = {manifest.module for manifest in supplied}
            for name in modules:
                path = staging / name
                problems = validate_module(path)
                if problems:
                    raise ValueError(f"{name}: " + "; ".join(problems))
                if name not in owned_modules:
                    if not name.endswith(_TOOL_SUFFIX):
                        raise ValueError(f"{name} must have an explicit plugin manifest")
                    if self._write_manifest(name[:-len(_TOOL_SUFFIX)], path, directory=staging) is None:
                        raise OSError("Could not prepare the plugin manifest")
            prepared = []
            for manifest_path in staging.glob(f"*{MANIFEST_SUFFIX}"):
                manifest = load_manifest(manifest_path)
                module_path = manifest.module_path
                if module_path is None or not module_path.is_file():
                    raise ValueError(f"{manifest_path.name} refers to a module outside this candidate")
                prepared.append(manifest)
            identities = [_safe_name(manifest.name) for manifest in prepared]
            if len(set(identities)) != len(identities):
                raise ValueError("Duplicate manifest identity in the plugin candidate")
            if len({manifest.module.casefold() for manifest in prepared}) != len(prepared):
                raise ValueError("Each plugin module must have exactly one manifest owner")
            for record in self.scan():
                for manifest in prepared:
                    if _safe_name(record.name) != _safe_name(manifest.name):
                        continue
                    replacing_owner = record.manifest_file == manifest.source_path.name
                    replacing_orphan = not record.has_manifest and record.module in modules
                    if not replacing_owner and not replacing_orphan:
                        raise ValueError(f"Plugin identity '{manifest.name}' conflicts with existing '{record.name}'")
            candidates = sorted(path for path in staging.iterdir() if path.is_file())
            previous = staging / ".previous"
            previous.mkdir()
            backups: dict[Path, Path | None] = {}
            for path in candidates:
                target = root / path.name
                if target.is_symlink() or target.resolve().parent != root:
                    raise ValueError(f"Refusing an installation through {target.name}")
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f"{target.name} is not a regular file")
                    old = previous / target.name
                    shutil.copy2(target, old)
                    backups[target] = old
                else:
                    backups[target] = None
            published = []
            try:
                for path in sorted(candidates, key=lambda item: (not item.name.endswith(_TOOL_SUFFIX), item.name)):
                    target = root / path.name
                    _replace(str(path), target)
                    published.append(target)
            except Exception as exc:
                recovery_errors = []
                for target in reversed(published):
                    try:
                        old = backups[target]
                        if old is None:
                            target.unlink(missing_ok=True)
                        else:
                            _replace(str(old), target)
                    except OSError as recovery:
                        recovery_errors.append(str(recovery))
                if recovery_errors:
                    keep_recovery = True
                    raise _InstallRecoveryRequired(
                        f"Installation failed ({exc}); rollback could not finish. "
                        f"Previous files are retained in {previous}: {'; '.join(recovery_errors)}") from exc
                raise
            return sorted(manifest.name for manifest in prepared)
        finally:
            # This exact directory was created within the configured plugin
            # root. Preserve it if rollback needs manual recovery.
            if not keep_recovery and staging.resolve().is_relative_to(root):
                shutil.rmtree(staging, ignore_errors=True)

    def remove(self, name: str) -> ToolResult:
        with self._file_lock, self._state_lock:
            return self._remove_locked(name)

    def _remove_locked(self, name: str) -> ToolResult:
        if self._state_error:
            return self._unavailable_state()
        safe = _safe_name(name)
        record = self.get(safe)
        if record is None:
            return ToolResult(f"No plugin named '{name}'.", ok=False)
        if any("Duplicate manifest identity" in problem for problem in record.problems):
            return ToolResult(f"Plugin '{safe}' has ambiguous manifests; resolve those files before removal.", ok=False)
        if record.quarantined:
            return self._remove_quarantined_locked(record)
        staging = None
        removed: list[Path] = []
        previous_state = {"disabled": list(self._state.get("disabled") or []),
                          "meta": dict(self._state.get("meta") or {})}
        keep_recovery = False
        root = self.directory.resolve()
        try:
            filenames = [record.module or f"{safe}{_TOOL_SUFFIX}"]
            if record.has_manifest:
                filenames.append(record.manifest_file or f"{safe}{MANIFEST_SUFFIX}")
            paths = []
            for filename in filenames:
                if (Path(filename).name != filename or PureWindowsPath(filename).name != filename
                        or ":" in filename or any(ord(char) < 32 for char in filename)):
                    raise ValueError("Plugin removal requires plain sibling filenames")
                path = root / filename
                if path.is_symlink() or path.resolve().parent != root:
                    raise ValueError(f"Refusing removal through {filename}")
                if path.exists():
                    if not path.is_file():
                        raise ValueError(f"{filename} is not a regular file")
                    paths.append(path)
            if not paths:
                return ToolResult(f"Nothing to remove for '{safe}'.", ok=False)
            staging = Path(tempfile.mkdtemp(prefix=".remove-recovery-", dir=root))
            for path in paths:
                shutil.copy2(path, staging / path.name)
            for path in paths:
                path.unlink()
                removed.append(path)
            for key in list(self._state["meta"]):
                if _safe_name(key) == safe:
                    self._state["meta"].pop(key, None)
            # Keep a saved tombstone: a cached handler or a later reinstall
            # cannot silently revive the removed generation.
            disabled = {_safe_name(item) for item in self._state.get("disabled") or []} | {safe}
            self._state["disabled"] = sorted(disabled)
            if not self._save_state():
                raise OSError("Could not verify the saved removal preference" +
                              (f": {self._state_error}" if self._state_error else ""))
            self._committed_disabled = frozenset(disabled)
        except Exception as exc:
            self._state = previous_state
            recovery_errors = []
            for path in reversed(removed):
                try:
                    _replace(str(staging / path.name), path)
                except OSError as recovery:
                    recovery_errors.append(str(recovery))
            if recovery_errors:
                keep_recovery = True
                reason = (f"Removal rollback could not finish; '{safe}' is held. "
                          f"Previous files retained at {staging}: {'; '.join(recovery_errors)}")
                self._runtime_holds[safe] = reason
                return ToolResult(f"Could not remove '{safe}': {exc}. {reason}", ok=False)
            return ToolResult(f"Could not remove '{safe}': {exc}", ok=False)
        finally:
            if staging is not None and not keep_recovery and staging.resolve().is_relative_to(root):
                shutil.rmtree(staging, ignore_errors=True)
        self._health.pop(safe, None)
        self._log(f"PLUGIN: removed '{safe}'.")
        return ToolResult(
            f"Removed plugin '{safe}' ({', '.join(path.name for path in removed)}). "
            "Cached calls are held. Reinstall and explicitly enable it to use it again.",
            evidence=[{"kind": "plugin_removed", "name": record.name, "identity": safe}])

    def _remove_quarantined_locked(self, record: "PluginRecord") -> ToolResult:
        """Retire a quarantined plugin: its manifest and the quarantined module
        are archived to _superseded/ (verified copies, nothing deleted), its
        quarantine line is dropped and a saved tombstone keeps it from
        reviving. The owner can still move the files back by hand."""
        safe = _safe_name(record.name)
        qdir = self.directory / "_quarantine"
        broken = qdir / f"{record.module}.broken"
        previous = list(self._state.get("disabled") or [])
        try:
            files = self._plugin_files(record) + ([broken] if broken.is_file() else [])
            if not files:
                return ToolResult(f"Nothing to remove for '{safe}'.", ok=False)
            disabled = {_safe_name(item) for item in previous} | {safe}
            self._state["disabled"] = sorted(disabled)
            if not self._save_state():
                self._state["disabled"] = previous
                return ToolResult(f"Could not save the removal of '{safe}'; nothing was moved.", ok=False)
            self._committed_disabled = frozenset(disabled)
            moved = self._archive_files(files, {"name": record.name, "removed_quarantined": True,
                                                "reason": record.quarantined})
            _drop_quarantine_entry(qdir, broken.name)
        except Exception as exc:
            return ToolResult(f"Could not remove '{safe}': {exc}", ok=False)
        self._health.pop(safe, None)
        self._log(f"PLUGIN: removed quarantined '{safe}' - archived to custom_tools/"
                  f"{self.SUPERSEDED_DIR}/ ({', '.join(moved)}).")
        return ToolResult(
            f"Removed the quarantined plugin '{safe}'. Its files are kept in custom_tools/"
            f"{self.SUPERSEDED_DIR}/ in case you want them back.",
            evidence=[{"kind": "plugin_removed", "name": record.name, "identity": safe}],
            receipt={"archived": moved})

    def _write_manifest(self, name: str, module_path: Path,
                        tier: str = _DEFAULT_TIER, *, directory: Path | None = None) -> Path | None:
        """Generate a manifest for a module that has none."""
        safe = _safe_name(name)
        manifest_path = (directory or self.directory) / f"{safe}{MANIFEST_SUFFIX}"
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
        with self._file_lock:
            return self._export_bundle_locked(name, destination)

    def _export_bundle_locked(self, name: str, destination: str) -> ToolResult:
        try:
            return self._write_export_bundle(name, destination)
        except Exception as exc:
            return ToolResult(f"Could not write the bundle: {exc}", ok=False)

    def _write_export_bundle(self, name: str, destination: str) -> ToolResult:
        record = self.get(name)
        if record is None:
            return ToolResult(f"I have no plugin called '{name}', sir.", ok=False)
        if any("Duplicate manifest identity" in problem for problem in record.problems):
            return ToolResult(f"Plugin '{name}' has ambiguous manifests; resolve them before export.", ok=False)
        root = self.directory.resolve()
        filenames = [record.module or f"{_safe_name(record.name)}{_TOOL_SUFFIX}"]
        if record.has_manifest:
            filenames.append(record.manifest_file or f"{_safe_name(record.name)}{MANIFEST_SUFFIX}")
        sources = []
        for filename in filenames:
            if (Path(filename).name != filename or PureWindowsPath(filename).name != filename
                    or ":" in filename or any(ord(char) < 32 for char in filename)):
                raise ValueError("Plugin export requires plain sibling filenames")
            path = root / filename
            if path.is_symlink() or path.resolve().parent != root or not path.is_file():
                raise ValueError(f"Cannot export unavailable or redirected plugin file '{filename}'")
            sources.append(path)
        bundle_name = f"{_safe_name(record.name)}.orionplugin"
        target = Path(destination).expanduser() if str(destination or "").strip() else (
            self.directory / "bundles" / bundle_name)
        if target.is_dir():
            target = target / bundle_name
        if target.is_symlink():
            raise ValueError("Refusing to replace an export destination through a symbolic link")
        # Protect this plugin's sources, including hard-link aliases, and the
        # other plugin modules/metadata in the same registry.
        protected = [*root.glob("*.py"), *root.glob(f"*{MANIFEST_SUFFIX}")]
        for source in protected:
            if (target.resolve() == source.resolve()
                    or (target.exists() and source.exists() and target.samefile(source))):
                raise ValueError("The export destination would overwrite a plugin source or manifest")
        temporary = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".export.tmp", dir=target.parent)
            with os.fdopen(descriptor, "w+b") as stream:
                with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for source in sources:
                        bundle.write(source, source.name)
                    if not record.has_manifest:
                        bundle.writestr(
                            f"{_safe_name(record.name)}{MANIFEST_SUFFIX}",
                            json.dumps(self._manifest_dict(record), indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            _replace(temporary, target)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
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
            "secrets": list(record.secrets),
        }

    def install_bundle(self, source: str) -> ToolResult:
        """Install a plugin from a .orionplugin/.zip bundle.

        Every member is checked before anything is written: only a plain
        plugin ``*.py`` / ``*.plugin.json`` at the archive ROOT is accepted, so a
        malicious bundle cannot path-traverse out of the plugin directory.
        """
        with self._file_lock:
            return self._install_bundle_locked(source)

    def _install_bundle_locked(self, source: str) -> ToolResult:
        if self._state_error:
            return self._unavailable_state()
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
                    and ":" not in m and not any(ord(char) < 32 for char in m)
                    and (m.endswith(".py") or m.endswith(MANIFEST_SUFFIX))
                ]
                if not safe:
                    return ToolResult(
                        "That bundle contains no plugin module - expected a "
                        f"*{_TOOL_SUFFIX} at the top level.", ok=False)
                rejected = [m for m in members if m not in safe]
                if len(safe) != len({name.casefold() for name in safe}):
                    raise ValueError("Duplicate filenames in a plugin bundle")
                files = {}
                expanded = 0
                for member in safe:
                    info = bundle.getinfo(member)
                    expanded += info.file_size
                    if info.file_size > MAX_PLUGIN_FILE_BYTES or expanded > MAX_PLUGIN_BUNDLE_BYTES:
                        raise ValueError("The candidate exceeds the expanded plugin bundle budget")
                    with bundle.open(info) as stream:
                        data = stream.read(MAX_PLUGIN_FILE_BYTES + 1)
                    if len(data) > MAX_PLUGIN_FILE_BYTES:
                        raise ValueError(f"{member} exceeds the plugin file budget")
                    files[member] = data
                installed = self._install_candidates(files)
        except zipfile.BadZipFile:
            return ToolResult("That file is not a readable plugin bundle.", ok=False)
        except Exception as exc:
            return ToolResult(f"Could not install the bundle: {exc}", ok=False)
        note = ""
        if rejected:
            note = (f"\n  (ignored {len(rejected)} unexpected file(s) in the bundle: "
                    + ", ".join(rejected[:3]) + ")")
        saved = self._record_installs(installed, "bundle")
        if not installed:
            return ToolResult("The bundle held a manifest but no module." + note,
                              ok=False)
        return ToolResult(
            f"Installed {', '.join(installed)} from the bundle.{note}\n"
            "Run 'plugin audit' to see what it can touch, then 'plugin reload'."
            + ("" if saved else " Install metadata could not be saved."))

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
        with self._file_lock:
            return self._backfill_manifests_locked()

    def _backfill_manifests_locked(self) -> list[str]:
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
        if self._state_error:
            return self._unavailable_state()
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
        if self._state_error:
            return self._unavailable_state()
        records = self.scan()
        problems: list[str] = []
        set_aside = [r for r in records if r.quarantined or r.superseded_by]
        records = [r for r in records if not (r.quarantined or r.superseded_by)]
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

        lines = [f"PLUGIN DOCTOR — {len(records) + len(set_aside)} plugin(s) inspected.", ""]
        if problems:
            lines.append(f"{len(problems)} problem(s):")
            lines += [f"  ✗ {problem}" for problem in problems]
        else:
            lines.append("  ✓ No problems found.")
        for record in set_aside:
            if record.superseded_by:
                lines.append(f"  · {record.name}: superseded by the built-in "
                             f"{record.superseded_by} (held back on purpose)")
            else:
                lines.append(f"  · {record.name}: quarantined - {record.quarantined[:120]} "
                             f"→ 'plugin restore {record.name}' or 'plugin remove {record.name}'")
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
            if not isinstance(data, dict):
                return []
            # An entry whose tool is live again is history, not a fault.
            return sorted(name for name in data
                          if not (self.directory / name[: -len(".broken")]).exists())
        except Exception:
            return []

    def snapshot(self) -> dict[str, Any]:
        """Machine-readable state for the GUI panel and diagnostics."""
        records = self.scan()
        return {
            "total": len(records),
            "state_error": self._state_error,
            "enabled": sum(1 for r in records if r.enabled),
            "healthy": sum(1 for r in records if r.enabled and r.healthy),
            "quarantined": sum(1 for r in records if r.quarantined),
            "superseded": sum(1 for r in records if r.superseded_by),
            "archived": [{"name": str(item.get("name") or ""),
                          "superseded_by": str(item.get("superseded_by") or ""),
                          "at": str(item.get("at") or "")}
                         for item in self.superseded_archive()[:12]],
            "plugins": [r.to_dict() for r in records],
        }
