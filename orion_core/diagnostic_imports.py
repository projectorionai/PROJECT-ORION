"""One-shot import diagnostics, outside the live interpreter and personal state.

The existing Forge verifier supplies the frozen headless entry and bounded
process runner. This is fault containment, not an OS sandbox for native code.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any

MAX_MODULES = 2048
TIMEOUT_S = 60.0
_CANCELLED: ContextVar[Any] = ContextVar("diagnostic_cancelled", default=None)
_MARKER = "@@ORION_IMPORT_DIAGNOSTIC@@"
_WORKER_PACKAGES = frozenset({"torch", "tensorflow", "jax", "transformers", "sentence_transformers"})


@contextmanager
def cancellation_scope(event: threading.Event):
    token = _CANCELLED.set(event)
    try:
        yield
    finally:
        _CANCELLED.reset(token)


def cancelled() -> bool:
    event = _CANCELLED.get()
    return bool(event is not None and event.is_set())


def inventory(package_dir: Path) -> dict[str, Any]:
    """Enumerate sources or actual bootloader membership, without imports."""
    package_dir = Path(package_dir)
    if getattr(sys, "frozen", False):
        loader = sys.modules.get("pyimod02_importers")
        archive = getattr(loader, "pyz_archive", None)
        table = getattr(archive, "toc", None)
        if not isinstance(table, dict):
            return {"error": "Frozen PYZ module inventory is unavailable; coverage was not established"}
        modules = sorted(name for name in table if isinstance(name, str)
                         and (name == "orion_core" or name.startswith("orion_core.")))
        kind, sources = "frozen", {}
    else:
        sources = {}
        for path in package_dir.rglob("*.py"):
            if len(sources) >= MAX_MODULES:
                return {"error": "Source module inventory exceeds the diagnostic limit"}
            parts = path.relative_to(package_dir.parent).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            sources[".".join(parts)] = str(path.absolute())
        modules, kind = sorted(sources), "source"
    if not modules or len(modules) > MAX_MODULES:
        return {"error": "No complete bounded module inventory is available"}
    return {"kind": kind, "modules": modules, "sources": sources,
            "source_parent": str(package_dir.parent.absolute()),
            "worker_packages": sorted(_WORKER_PACKAGES)}


def _environment(work: Path) -> dict[str, str]:
    keep = {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "PATH",
            "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "NUMBER_OF_PROCESSORS"}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    for name in ("config", "home", "appdata", "localappdata", "temp"):
        (work / name).mkdir()
    env.update(ORION_CONFIG_DIR=str(work / "config"), USERPROFILE=str(work / "home"),
               HOME=str(work / "home"), APPDATA=str(work / "appdata"),
               LOCALAPPDATA=str(work / "localappdata"), TEMP=str(work / "temp"),
               TMP=str(work / "temp"), PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen",
               ORION_NO_CAPTURE="1", ORION_START_MUTED="1", ORION_START_OLLAMA="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ORION_SEMANTIC="0")
    return env


# Kept as a literal so the frozen app never needs a loose copy of this module.
# Imports and any native initializers execute only in the owned verifier child.
_WORKER = r'''
import dis, importlib, importlib.machinery, importlib.util, json, os
from pathlib import Path
import socket, sys, threading, types

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if job["kind"] == "source":
    sys.path.insert(0, job["source_parent"])
elif not getattr(sys, "frozen", False):
    raise RuntimeError("Frozen diagnostics did not run in a frozen verifier")

# No accounts or services are needed to import application modules. Permit only
# the interpreter's own socketpair, used by Windows asyncio internals.
local = threading.local()
original_pair = socket.socketpair
def pair(*args, **kwargs):
    local.pair = True
    try:
        return original_pair(*args, **kwargs)
    finally:
        local.pair = False
socket.socketpair = pair
def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"} and not getattr(local, "pair", False):
        raise OSError("Network access is disabled in the import diagnostic")
sys.addaudithook(audit)

training = set(job["worker_packages"])
codes, failures, imported, checked = {}, [], [], []
archive = getattr(sys.modules.get("pyimod02_importers"), "pyz_archive", None)
for name in job["modules"]:
    try:
        if job["kind"] == "frozen":
            code = archive.extract(name)
        else:
            path = Path(job["sources"][name])
            with path.open("rb") as handle:
                source = handle.read(4 * 1024 * 1024 + 1)
            if len(source) > 4 * 1024 * 1024:
                raise ValueError("Source exceeds the per-module diagnostic limit")
            code = compile(source, str(path), "exec", dont_inherit=True)
        if not isinstance(code, types.CodeType):
            raise ValueError("No Python code object is available")
        codes[name] = code
        checked.append(name)
    except BaseException as exc:
        failures.append(name + ": " + type(exc).__name__ + ": " + str(exc)[:160])

def imports(code, module):
    names = set()
    is_package = (module == "orion_core" or any(other.startswith(module + ".") for other in codes))
    package = module if is_package else module.rpartition(".")[0]
    instructions = [item for item in dis.get_instructions(code)
                    if item.opname not in {"EXTENDED_ARG", "CACHE"}]
    for index, instruction in enumerate(instructions):
        if instruction.opname == "IMPORT_NAME":
            base = instruction.argval
            level = instructions[index-2].argval if index >= 2 else 0
            fromlist = instructions[index-1].argval if index else None
            if isinstance(level, int) and level:
                try:
                    base = importlib.util.resolve_name("." * level + base, package)
                except ImportError:
                    # A statically present, unexecuted optional branch can
                    # contain an invalid relative import. Actual import below
                    # determines failure; dependency discovery must not crash.
                    continue
            names.add(base)
            if isinstance(fromlist, tuple):
                names.update(base + "." + item for item in fromlist if item != "*")
    # Class bodies execute at import; function bodies do not (CO_NEWLOCALS).
    for value in code.co_consts:
        if isinstance(value, types.CodeType) and not value.co_flags & 2:
            names.update(imports(value, module))
    return names

dependencies = {name: imports(code, name) for name, code in codes.items()}
for name, deps in dependencies.items():
    parent = name.rpartition(".")[0]
    if parent in codes:
        deps.add(parent)  # importing a child also executes its package initializer
worker_only = {name: {dep.split(".")[0] for dep in deps} & training
               for name, deps in dependencies.items()}
changed = True
while changed:
    changed = False
    for name, deps in dependencies.items():
        for dep in deps:
            extra = worker_only.get(dep, set()) - worker_only[name]
            if extra:
                worker_only[name].update(extra)
                changed = True
dependency_only, excluded = [], []
for name in job["modules"]:
    if name not in codes:
        continue
    if worker_only[name]:
        missing = sorted(dep for dep in worker_only[name]
                         if importlib.util.find_spec(dep) is None)
        dependency_only.append({"module": name, "missing": missing})
        continue
    if name == "orion_core.app":
        excluded.append(name)
        continue
    print("IMPORTING " + name, flush=True)
    try:
        importlib.import_module(name)
        imported.append(name)
    except BaseException as exc:
        detail = str(exc).splitlines()
        failures.append(name + ": " + type(exc).__name__ + ": " + (detail[0] if detail else "")[:160])
print("@@ORION_IMPORT_DIAGNOSTIC@@" + json.dumps({
    "kind": job["kind"], "inventory_count": len(job["modules"]),
    "code_checked": len(checked), "imported": len(imported),
    "dependency_only": dependency_only, "excluded": excluded, "failures": failures}), flush=True)
'''


def check(package_dir: Path, *, timeout: float | None = None) -> dict[str, Any]:
    job = inventory(package_dir)
    if "error" in job:
        return {"status": "WARN", "detail": job["error"]}
    from .sandbox import SandboxVerificationHarness
    from .session_log import redact
    event = _CANCELLED.get() or threading.Event()
    with tempfile.TemporaryDirectory(prefix="orion-import-diagnostic-") as temporary:
        work = Path(temporary)
        env = _environment(work)
        (work / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (work / "imports.py").write_text(_WORKER, encoding="utf-8")
        # Reuse only the existing bounded verifier runner, not Forge staging,
        # model generation, ambient environment or configurable container mode.
        runner = SandboxVerificationHarness.__new__(SandboxVerificationHarness)
        runner.staging_dir = work
        runner._isolation_mode = "process"
        runner._cancelled = event
        runner._child_env = lambda: env
        budget = TIMEOUT_S if timeout is None else max(0.1, min(TIMEOUT_S, float(timeout)))
        out, err, code, timed_out, memory_exceeded = runner._run_script("imports.py", budget, str(work / "job.json"))
    if event.is_set():
        return {"status": "WARN", "detail": "Import diagnostic cancelled; no complete result was reported"}
    progress = [line for line in out.splitlines() if line.startswith("IMPORTING ")]
    last = ("; last " + progress[-1][10:]) if progress else ""
    if timed_out or memory_exceeded or code != 0:
        reason = (f"timeout after {budget:g}s" if timed_out else
                  "verifier memory limit exceeded" if memory_exceeded else
                  f"verifier exit {code}")
        detail = redact(err.strip().splitlines()[-1])[:180] if err.strip() else ""
        return {"status": "FAIL", "detail": f"Isolated imports incomplete: {reason}{last}" + ("; " + detail if detail else "")}
    try:
        lines = [line[len(_MARKER):] for line in out.splitlines() if line.startswith(_MARKER)]
        result = json.loads(lines[-1])
        expected = len(job["modules"])
        fields = {"kind", "inventory_count", "code_checked", "imported", "failures", "dependency_only", "excluded"}
        if (not isinstance(result, dict) or set(result) != fields
                or result["kind"] != job["kind"] or type(result["inventory_count"]) is not int
                or result["inventory_count"] != expected
                or type(result["code_checked"]) is not int or not 0 <= result["code_checked"] <= expected
                or type(result["imported"]) is not int or not 0 <= result["imported"] <= expected
                or not isinstance(result["failures"], list) or any(not isinstance(v, str) for v in result["failures"])
                or not isinstance(result["dependency_only"], list) or not isinstance(result["excluded"], list)):
            raise ValueError("Incomplete verifier inventory")
        worker_names = []
        for item in result["dependency_only"]:
            if (not isinstance(item, dict) or set(item) != {"module", "missing"}
                    or item["module"] not in job["modules"] or not isinstance(item["missing"], list)
                    or any(name not in _WORKER_PACKAGES for name in item["missing"])):
                raise ValueError("Invalid dependency-only coverage")
            worker_names.append(item["module"])
        if (len(set(worker_names)) != len(worker_names)
                or len(set(result["excluded"])) != len(result["excluded"])
                or any(name != "orion_core.app" or name not in job["modules"] for name in result["excluded"])
                or set(worker_names) & set(result["excluded"])):
            raise ValueError("Invalid or overlapping excluded modules")
    except (ValueError, KeyError, IndexError, TypeError):
        return {"status": "FAIL", "detail": "Isolated import verifier returned no valid complete report"}
    failures = result["failures"]
    detail = (f"Isolated {job['kind']} inventory: {expected} modules; "
              f"{result['code_checked']} code objects checked; {result['imported']} imported; "
              f"{len(result['dependency_only'])} dependency-only; {len(result['excluded'])} excluded entry points")
    if failures:
        return {"status": "FAIL", "detail": detail + "; " + redact("; ".join(str(v) for v in failures[:3]))[:420], **result}
    if result["code_checked"] != expected or result["imported"] + len(result["dependency_only"]) + len(result["excluded"]) != expected:
        return {"status": "FAIL", "detail": detail + "; incomplete coverage", **result}
    status = "WARN" if result["dependency_only"] else "PASS"
    if result["dependency_only"]:
        detail += "; worker-only modules were not imported"
        missing = sorted({name for item in result["dependency_only"] for name in item["missing"]})
        if missing:
            detail += "; worker dependencies absent: " + ", ".join(missing)
    return {"status": status, "detail": detail, **result}
