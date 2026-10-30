"""
Reflective Module Loader.

Dynamically loads newly written Python modules from config/custom_tools/
without restarting ORION. Uses importlib to hook modules into memory and
updates the tool declarations array inside OrionDispatcher instantly.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .bus import OrionBus
from .data import ToolResult
from .forge_contract import contract_problems
from .atomic_io import atomic_write_text


# ── plugins that duplicate a built-in tool ───────────────────────────────────
#
# Three forged plugins (gesture_control_interface, gesture_control_manager and
# gesture_control_tool) sat beside the built-in gesture_control and were
# imported at every boot, with no recorded call. A forged plugin whose name,
# once words for a tool's kind (manager, tool, interface ...) are set aside,
# names exactly what a built-in tool names is not imported or activated.
# Plugins delivered with ORION are reviewed adapters and are never judged this
# way; the plugin files themselves are never touched.

#: The delivery ledger plugin_registry.seed_shipped keeps beside the plugins.
_SHIPPED_LEDGER = ".shipped.json"


@functools.lru_cache(maxsize=4)
def _concept_index(builtins: frozenset[str]) -> dict[frozenset[str], str]:
    from .forge_triage import name_concepts
    index: dict[frozenset[str], str] = {}
    for name in sorted(builtins):
        concepts = frozenset(name_concepts(name))
        if concepts:
            index.setdefault(concepts, name)
    return index


def shadowed_builtin(name: str, builtins: Iterable[str]) -> str:
    """The built-in tool *name* duplicates, or ''.

    An exact name counts, and so does the same name with only kind words
    added: 'gesture_control_manager' duplicates 'gesture_control', while
    'find_files_by_hash' names something more than 'find_files' and does not.
    """
    name = str(name or "").strip().lower()
    names = frozenset(str(item) for item in (builtins or ()) if item)
    if not name or not names:
        return ""
    if name in names:
        return name
    from .forge_triage import name_concepts
    concepts = frozenset(name_concepts(name))
    if not concepts:
        return ""
    return _concept_index(names).get(concepts, "")


@functools.lru_cache(maxsize=1)
def _bundled_plugins() -> frozenset[str] | None:
    """The plugins bundled with this build, or None outside a built app."""
    try:
        from .constants import resource_path
        folder = resource_path("shipped_plugins")
        if not folder.is_dir():
            return None
        return frozenset(path.name[: -len("_tool.py")] for path in folder.glob("*_tool.py"))
    except Exception:
        return None


@dataclass
class ModuleLoadOutcome:
    """Structured result of reflective module loading."""

    succeeded: bool
    module_name: str
    module_path: Path | None = None
    tool_schema: dict[str, Any] | None = None
    error_log: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    handler_fn: Callable[..., Any] | None = None
    # Exact import identity for transaction recovery, including rejected
    # contracts. This stays local and is deliberately absent from to_dict().
    module_obj: Any = field(default=None, repr=False, compare=False)
    #: The built-in tool this module duplicates when it was held back for
    #: that (a deliberate state the registry shows as "superseded").
    superseded_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "module_name": self.module_name,
            "module_path": str(self.module_path) if self.module_path else None,
            "tool_schema": self.tool_schema,
            "error_log": self.error_log,
            "duration_ms": self.duration_ms,
        }


class ReflectiveModuleLoader:
    """Dynamically loads and registers custom tools without restart."""

    #: How long a forged tool gets to finish importing before it is treated as
    #: hung and quarantined.  Generous — a legitimate tool importing numpy or
    #: opencv can take a second or two on a cold start — but finite, because
    #: this import sits on ORION's startup path and self-written module-level
    #: code is not reviewed by anyone before it runs.
    IMPORT_TIMEOUT_S = 8.0

    #: How long to allow a dependency install before giving up on it.
    INSTALL_TIMEOUT_S = 120.0

    #: Packages that must never be auto-installed from a tool's import line.
    #: A forged tool asking for one of these is either confused or being used
    #: as a delivery mechanism, and neither deserves a pip install.
    INSTALL_DENYLIST = frozenset({
        "os", "sys", "subprocess", "ctypes", "pickle", "setuptools", "pip",
    })

    async def _install(self, package: str) -> bool:
        """pip-install *package* into the running interpreter. Never raises."""
        name = str(package or "").strip()
        if not name or name in self.INSTALL_DENYLIST or not name.replace(
                "_", "").replace("-", "").isalnum():
            return False
        import asyncio as _asyncio
        import subprocess

        def _run() -> bool:
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet",
                     "--disable-pip-version-check", name],
                    capture_output=True, timeout=self.INSTALL_TIMEOUT_S)
                return completed.returncode == 0
            except Exception:
                return False

        try:
            return await _asyncio.wait_for(
                _asyncio.to_thread(_run), timeout=self.INSTALL_TIMEOUT_S + 15)
        except Exception:
            return False

    def __init__(self, bus: OrionBus) -> None:
        """
        Initialise the reflective module loader.

        Args:
            bus: OrionBus for logging load states.
        """
        self.bus = bus
        # CONFIG_DIR, like the plugin registry. Path(__file__)/../config is the
        # same folder from a source checkout but, in the frozen app, a folder
        # INSIDE the bundle — rebuilt (and emptied) with every new .exe.
        from .constants import CONFIG_DIR
        self.custom_tools_dir = CONFIG_DIR / "custom_tools"
        self.custom_tools_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_modules: dict[str, Any] = {}
        # Tools that imported cleanly but whose schema and run() disagree.
        # They are NOT quarantined — the file is valid Python and worth
        # re-forging or fixing — but they must never reach the dispatcher,
        # because the first call would raise TypeError inside a live turn.
        self.contract_failures: dict[str, list[str]] = {}
        # Packages already attempted this session, so a package that genuinely
        # does not exist is tried once rather than on every tool that wants it.
        self._dependency_attempts: set[str] = set()
        #: The built-in tool names, when known (the Forge sets this). A plugin
        #: that duplicates one is then refused before it is imported.
        self.native_names: Callable[[], Iterable[str]] | None = None
        self._reviewed_cache: dict[Path, tuple[int, frozenset[str] | None]] = {}

    # ── plugins that duplicate a built-in tool ──────────────────────────────

    def _native_names(self) -> frozenset[str]:
        source = getattr(self, "native_names", None)
        if source is None:
            return frozenset()
        try:
            return frozenset(source())
        except Exception:
            return frozenset()

    def _reviewed(self, directory: Path) -> frozenset[str] | None:
        """The plugins delivered with ORION for this plugin folder: its
        delivery ledger plus the plugins bundled with this build. None when
        neither can be read; then nothing is judged a duplicate, rather than
        risk holding back a reviewed plugin."""
        ledger = Path(directory) / _SHIPPED_LEDGER
        known: frozenset[str] | None = None
        try:
            stamp = ledger.stat().st_mtime_ns
        except OSError:
            stamp = None
        if stamp is not None:
            cache = self.__dict__.setdefault("_reviewed_cache", {})
            cached = cache.get(ledger)
            if cached is None or cached[0] != stamp:
                try:
                    data = json.loads(ledger.read_text(encoding="utf-8"))
                    names = frozenset(str(key) for key in data) if isinstance(data, dict) else None
                except (OSError, ValueError):
                    names = None
                cached = (stamp, names)
                cache[ledger] = cached
            known = cached[1]
        bundled = _bundled_plugins()
        if bundled is not None:
            known = bundled | (known or frozenset())
        return known

    def duplicate_reason(self, tool_name: str, module_path: Path | None,
                         builtins: Iterable[str]) -> str:
        """Why the plugin *tool_name* at *module_path* is held back as a
        duplicate of a built-in tool, or ''."""
        builtin = shadowed_builtin(tool_name, builtins) if module_path is not None else ""
        if not builtin:
            return ""
        reviewed = self._reviewed(Path(module_path).parent)
        if reviewed is None or tool_name in reviewed:
            return ""
        return (f"held back: it duplicates ORION's built-in tool '{builtin}' (use {builtin}); "
                "the plugin file is left untouched")

    async def load_and_register(
        self,
        module_path: Path,
    ) -> ModuleLoadOutcome:
        """
        Load a Python module and register its tool into the dispatcher.

        Uses importlib.util.spec_from_file_location to dynamically import
        the module. Expects the module to export:
          - get_tool_schema() → dict with OpenAI function schema
          - run(**kwargs) → tool execution endpoint

        Args:
            module_path: Full path to the .py module file

        Returns:
            ModuleLoadOutcome with success status and handler function
        """
        # Decided before the import, so a duplicate's module-level code never
        # runs. The caller reports the outcome once, in its own log line.
        natives = self._native_names()
        duplicate = self.duplicate_reason(
            module_path.stem.removesuffix("_tool"), module_path, natives)
        if duplicate:
            return ModuleLoadOutcome(
                succeeded=False, module_name=module_path.stem, module_path=module_path,
                error_log=[duplicate],
                superseded_by=shadowed_builtin(module_path.stem.removesuffix("_tool"), natives))
        self.bus.log.emit(
            f"[FORGE] Loader: starting load of '{module_path.name}'"
        )
        outcome = ModuleLoadOutcome(
            succeeded=False,
            module_name=module_path.stem,
            module_path=module_path,
        )
        start_time = asyncio.get_running_loop().time()

        try:
            # Load module via thread (I/O operation), BOUNDED.
            #
            # A forged tool's module-level code runs at import. ORION forges
            # his own tools, so that code is not reviewed by anyone before it
            # executes here, during startup.
            #
            # Observed on this machine: 'live_camera_analysis_tool.py' ends with
            #
            #     camera_analysis = LiveCameraAnalysis()
            #
            # whose __init__ calls cv2.VideoCapture(0) — so *importing* the tool
            # opens the webcam. On Windows that probe can block for a long time
            # or fail hard (the "cv::obsensor ... Camera index out of range"
            # line that appears in every orion_startup*.log), and because this
            # import sits on the startup path, ORION never reached the Command
            # Deck at all: boot went silent immediately after this loader
            # logged the tool's name.
            #
            # A tool that will not import promptly is quarantined and reported,
            # and startup carries on. An unbounded import of self-written code
            # is a boot-time single point of failure, and no amount of care in
            # any individual tool removes that.
            module = await asyncio.wait_for(
                asyncio.to_thread(self._load_module_file, module_path),
                timeout=self.IMPORT_TIMEOUT_S,
            )
            outcome.module_obj = module

            # Extract schema and handler
            schema = self._extract_schema(module)
            handler_fn = self._extract_handler(module)

            if schema is None or handler_fn is None:
                outcome.error_log.append(
                    "Module must export get_tool_schema() and run() function"
                )
                self.bus.log.emit(
                    f"[FORGE] Loader: ✗ missing required exports in {module_path.name}"
                )
            elif (module_path.stem.startswith("trial_")
                  and schema.get("name") != module_path.stem.removesuffix("_tool")):
                outcome.error_log.append(
                    "Analysis trial schema must retain its reserved file/tool name")
                self.bus.log.emit(
                    f"[FORGE] Loader: ✗ trial namespace mismatch in {module_path.name}")
            elif (problems := contract_problems(schema, handler_fn)):
                # The same gate the sandbox applies to a freshly forged tool,
                # applied here to EVERY tool including ones forged before that
                # gate existed.  Such a tool loads perfectly and then raises
                # TypeError the first time the dispatcher calls it — a failure
                # that surfaces mid-conversation rather than at boot.
                outcome.error_log.extend(f"CONTRACT: {p}" for p in problems)
                self.contract_failures[outcome.module_name] = list(problems)
                self.bus.log.emit(
                    f"[FORGE] Loader: ✗ '{module_path.name}' breaks the tool contract "
                    f"and will not be activated — {problems[0]}"
                )
            else:
                outcome.succeeded = True
                outcome.tool_schema = schema
                outcome.handler_fn = handler_fn
                self.contract_failures.pop(outcome.module_name, None)
                self.loaded_modules[outcome.module_name] = {
                    "module": module,
                    "schema": schema,
                    "handler": handler_fn,
                }
                self.bus.log.emit(
                    f"[FORGE] Loader: ✓ loaded '{outcome.module_name}' "
                    f"({schema.get('name', 'unnamed')})"
                )

        except asyncio.TimeoutError:
            # The import is still running on a worker thread and cannot be
            # killed; quarantining the FILE is what stops it happening again
            # on the next start, which is the outcome that matters.
            sys.modules.pop(module_path.stem, None)
            reason = (f"import did not complete within {self.IMPORT_TIMEOUT_S:.0f}s "
                      "— module-level code is blocking (hardware access at "
                      "import time is the usual cause)")
            outcome.error_log = [reason]
            self.bus.log.emit(
                f"[FORGE] Loader: ✗ {module_path.name} timed out during import "
                f"after {self.IMPORT_TIMEOUT_S:.0f}s — quarantined so it cannot "
                "stall startup again. Tools must not touch hardware at import."
            )
            self._quarantine_module(module_path, reason)

        except SyntaxError as e:
            outcome.error_log = [f"Syntax error: {e}", traceback.format_exc()]
            self.bus.log.emit(
                f"[FORGE] Loader: ✗ syntax error in {module_path.name}: {e}"
            )
            self._quarantine_module(module_path, f"syntax error: {e}")

        except ModuleNotFoundError as e:
            # A MISSING DEPENDENCY IS NOT A BROKEN TOOL.
            #
            # 'json_validator_tool' spent days in quarantine reading
            # "ModuleNotFoundError: No module named 'jsonschema'". The tool was
            # perfectly good — it just needed a package installing, which is an
            # environment fix, not a code fault. Condemning it meant ORION
            # permanently lost a capability he had correctly written, and the
            # diagnostics panel carried a scary "1 quarantined" for something
            # a single pip install resolves.
            #
            # So: try to install it and load again. Quarantine only if the
            # package genuinely cannot be had.
            sys.modules.pop(module_path.stem, None)
            missing = (e.name or "").split(".")[0]
            if (missing and not module_path.stem.startswith("trial_")
                    and missing not in self._dependency_attempts):
                self._dependency_attempts.add(missing)
                self.bus.log.emit(
                    f"[FORGE] Loader: {module_path.name} needs '{missing}' — "
                    "installing it rather than quarantining a working tool.")
                if await self._install(missing):
                    self.bus.log.emit(
                        f"[FORGE] Loader: installed '{missing}'; retrying "
                        f"{module_path.name}.")
                    return await self.load_and_register(module_path)
            outcome.error_log = [
                f"{type(e).__name__}: {str(e)}", traceback.format_exc()]
            self.bus.log.emit(
                f"[FORGE] Loader: ✗ {module_path.name} needs '{missing}', "
                "which could not be installed.")
            self._quarantine_module(
                module_path,
                f"missing package '{missing}' (install it and re-forge)")

        except Exception as e:
            # A half-initialised module can linger in sys.modules and poison
            # later imports; drop it before quarantining the broken file.
            sys.modules.pop(module_path.stem, None)
            outcome.error_log = [
                f"{type(e).__name__}: {str(e)}",
                traceback.format_exc(),
            ]
            self.bus.log.emit(
                f"[FORGE] Loader: ✗ exception loading {module_path.name}: {e}"
            )
            self._quarantine_module(module_path, f"{type(e).__name__}: {e}")

        finally:
            end_time = asyncio.get_running_loop().time()
            outcome.duration_ms = (end_time - start_time) * 1000.0

        return outcome

    def _quarantine_module(self, module_path: Path, reason: str) -> None:
        """Move a module that will not load into config/custom_tools/_quarantine/
        so a single broken artefact stops erroring on every boot (the root cause
        of the repeating 'name code is not defined' loader failure).  The forge
        can always regenerate the tool cleanly.  Never raises.

        The reason is recorded in an index beside the file. Without it a
        quarantined tool was completely invisible — it vanished from the load
        path with no record of what it was or why it failed, so a graveyard
        could accumulate silently and nobody would ever know to re-forge it.
        """
        try:
            if not module_path.is_file():
                return
            qdir = module_path.parent / "_quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            target = qdir / f"{module_path.name}.broken"
            if target.exists():
                target = qdir / f"{module_path.stem}.{int(time.time())}.py.broken"
            module_path.replace(target)
            self._record_quarantine(qdir, target.name, module_path.stem, reason)
            self.bus.log.emit(
                f"[FORGE] Loader: quarantined {module_path.name} → _quarantine/ "
                f"({str(reason)[:80]}); it will not be retried on boot."
            )
        except Exception:
            pass

    def _record_quarantine(self, qdir: Path, filename: str, tool: str,
                           reason: str) -> None:
        """Append to the quarantine index. Failure-contained by design."""
        index_path = qdir / "index.json"
        try:
            index = {}
            if index_path.exists():
                loaded = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    index = loaded
            index[filename] = {
                "tool": tool,
                "reason": str(reason)[:400],
                "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            atomic_write_text(index_path,
                json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    async def rehabilitate_quarantined(self) -> list[str]:
        """Give quarantined tools a second chance when the world has changed.

        A tool quarantined for a missing package stayed ``.broken`` forever,
        even after the package was installed — which is exactly how
        ``self_repair_agent_tool`` sat in quarantine reading "No module named
        'sklearn'" long after sklearn was available. The loader learned to
        install-and-retry a missing dependency at LOAD time, but nothing ever
        went back to the graveyard to check whether an old inmate could now
        walk free.

        This does. For each quarantined file it re-checks, bounded by the same
        import timeout: does it parse, does it import now (installing a named
        missing package if it is still absent), and does it satisfy the tool
        contract. If all three hold, the file is restored to custom_tools/ under
        its proper name and removed from the quarantine index; the caller's
        normal load pass then picks it up. Anything that still fails is left
        exactly where it was, with its reason intact.

        Returns the tool names restored, for the boot log.
        """
        qdir = self.custom_tools_dir / "_quarantine"
        if not qdir.is_dir():
            return []
        restored: list[str] = []
        for path in sorted(qdir.glob("*.broken")):
            tool = path.stem.split(".")[0]        # strip .py/.timestamp/.broken
            target = self.custom_tools_dir / f"{tool}.py"
            if target.exists():
                continue          # a live version already exists; do not clobber
            try:
                if await self._can_rehabilitate(path):
                    path.replace(target)
                    self._forget_quarantine(qdir, path.name)
                    restored.append(tool)
                    self.bus.log.emit(
                        f"[FORGE] Loader: rehabilitated '{tool}' from quarantine "
                        "— its blocker is resolved; restoring it.")
            except Exception:
                pass              # a stubborn inmate stays put; never fatal
        return restored

    async def _can_rehabilitate(self, broken: Path) -> bool:
        """True if a quarantined file would load cleanly now. Never raises."""
        try:
            source = await asyncio.to_thread(broken.read_text, encoding="utf-8", errors="replace")
        except OSError:
            return False
        try:
            import ast
            ast.parse(source)
        except SyntaxError:
            return False          # genuinely broken code, not an environment gap

        # Rehabilitation EXECUTES the file to see whether it would load now, so
        # it re-applies the same gate the forge applied before writing it. The
        # forge checks guard_forged_source at creation; a quarantined file has
        # since sat on disk where anything could have edited it, and "it was
        # vetted once" is not the same as "it is safe to run now".
        try:
            from .security import SecuritySanitiser, SecurityViolation
            trial_name = broken.name.split(".", 1)[0]
            if trial_name.endswith("_tool"):
                trial_name = trial_name[:-5]
            SecuritySanitiser.guard_forged_source(
                source, f"rehab.{broken.name}", trial_name=trial_name)
        except SecurityViolation as exc:
            self.bus.log.emit(
                f"[FORGE] Loader: refusing to rehabilitate {broken.name} — {exc}")
            return False
        except Exception:
            if broken.name.startswith("trial_"):
                return False      # reserved analysis trials fail closed
            pass                  # the ordinary guard must not block boot

        def _probe() -> tuple[bool, str]:
            namespace: dict[str, Any] = {}
            try:
                exec(compile(source, str(broken), "exec"), namespace)
            except ModuleNotFoundError as exc:
                return False, (exc.name or "").split(".")[0]
            except Exception:
                return False, ""
            schema_fn = namespace.get("get_tool_schema")
            run_fn = namespace.get("run")
            if not callable(schema_fn) or not callable(run_fn):
                return False, ""
            try:
                schema = schema_fn()
                if (broken.name.startswith("trial_")
                        and schema.get("name") != trial_name):
                    return False, ""
                if contract_problems(schema, run_fn):
                    return False, ""
            except Exception:
                return False, ""
            return True, ""

        try:
            ok, missing = await asyncio.wait_for(
                asyncio.to_thread(_probe), timeout=self.IMPORT_TIMEOUT_S)
        except asyncio.TimeoutError:
            return False
        if ok:
            return True
        # Importable except for one named package we have not yet tried: install
        # it and probe once more. This is what actually frees an sklearn/jsonschema
        # inmate.
        if (missing and not broken.name.startswith("trial_")
                and missing not in self._dependency_attempts):
            self._dependency_attempts.add(missing)
            if await self._install(missing):
                try:
                    ok, _ = await asyncio.wait_for(
                        asyncio.to_thread(_probe), timeout=self.IMPORT_TIMEOUT_S)
                    return bool(ok)
                except asyncio.TimeoutError:
                    return False
        return False

    def _forget_quarantine(self, qdir: Path, filename: str) -> None:
        """Remove a rehabilitated file from the quarantine index."""
        index_path = qdir / "index.json"
        try:
            if not index_path.exists():
                return
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(index, dict) and index.pop(filename, None) is not None:
                atomic_write_text(index_path,
                    json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def quarantined(self) -> list[dict[str, Any]]:
        """Every quarantined tool, newest first, with the reason it failed.

        Reads the index where available and falls back to listing the files, so
        artefacts quarantined before the index existed are still reported.
        """
        qdir = self.custom_tools_dir / "_quarantine"
        if not qdir.is_dir():
            return []
        index: dict[str, Any] = {}
        try:
            index_path = qdir / "index.json"
            if index_path.exists():
                loaded = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    index = loaded
        except Exception:
            index = {}
        rows: list[dict[str, Any]] = []
        for path in sorted(qdir.glob("*.broken"), key=lambda p: -p.stat().st_mtime):
            # A working copy of the same tool is live again (a reviewed
            # version was delivered, or it was re-forged): the loader moves a
            # tool OUT of the folder when it fails, so this copy is history.
            # Reporting it kept diagnostics saying "2 quarantined" for tools
            # that were running, and the self-improvement loop spent the day's
            # rate limit forging a tool to fix them.
            if (self.custom_tools_dir / path.name[: -len(".broken")]).exists():
                continue
            record = index.get(path.name) or {}
            rows.append({
                "file": path.name,
                "tool": str(record.get("tool") or path.stem.split(".")[0]),
                "reason": str(record.get("reason") or "reason not recorded"),
                "at": str(record.get("at") or ""),
                "path": str(path),
            })
        return rows

    def _load_module_file(self, module_path: Path) -> Any:
        """
        Load a Python file as a module using importlib.

        Args:
            module_path: Full path to the .py file

        Returns:
            Loaded module object

        Raises:
            FileNotFoundError: If module_path does not exist
            SyntaxError: If the file contains syntax errors
        """
        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")

        module_name = module_path.stem
        # Check the exact snapshot subsequently compiled: an edited trial or a
        # stale bytecode cache cannot bypass its generation-time admission gate.
        source = module_path.read_bytes()
        if module_name.startswith("trial_"):
            from .security import SecuritySanitiser
            trial_name = module_name[:-5] if module_name.endswith("_tool") else module_name
            SecuritySanitiser.guard_forged_source(
                source.decode("utf-8"), f"load.{module_name}", trial_name=trial_name)
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module

        try:
            # Timestamp/size .pyc caches can serve the OLD generation when a
            # same-size replacement lands within one second. Compile exactly
            # the bytes that were published, independent of those caches.
            exec(compile(source, str(module_path), "exec"), module.__dict__)
        except BaseException:
            if sys.modules.get(module_name) is module:
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous
            raise

        # Bind execution metadata to the exact bytes compiled above. A later
        # file edit cannot give a cached handler a different effect audit.
        handler = getattr(module, "run", None)
        try:
            handler._orion_plugin_source_sha256 = hashlib.sha256(source).hexdigest()
        except (AttributeError, TypeError):
            pass  # Non-annotatable callables require a separate isolation path.
        return module

    def _extract_schema(self, module: Any) -> dict[str, Any] | None:
        """
        Extract the OpenAI-compatible function schema from a module.

        Calls get_tool_schema() if available.

        Args:
            module: Loaded module object

        Returns:
            Schema dict, or None if not available
        """
        if not hasattr(module, "get_tool_schema"):
            return None

        try:
            schema = module.get_tool_schema()
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass

        return None

    def _extract_handler(self, module: Any) -> Callable[..., Any] | None:
        """
        Extract the runtime execution handler (run function) from a module.

        Args:
            module: Loaded module object

        Returns:
            Callable, or None if not available
        """
        if not hasattr(module, "run"):
            return None

        handler = getattr(module, "run")
        if callable(handler):
            return handler

        return None

    async def activate_tool(
        self,
        outcome: ModuleLoadOutcome,
        dispatcher: Any,
    ) -> ToolResult:
        """
        Register a successfully loaded tool into the dispatcher.

        Updates dispatcher's tool routing dictionary and TOOL_DECLARATIONS.
        Optionally registers the tool description into FTS5 KNOWLEDGE memory.

        Args:
            outcome: ModuleLoadOutcome from load_and_register()
            dispatcher: OrionDispatcher instance to register tool

        Returns:
            ToolResult indicating success or failure
        """
        if outcome.succeeded is not True or outcome.tool_schema is None or outcome.handler_fn is None:
            return ToolResult(
                f"Cannot activate failed load: {', '.join(outcome.error_log)}",
                ok=False,
            )

        try:
            # A successful load is mutable data, not an admission token. Keep
            # one detached contract/callable generation through activation.
            schema = copy.deepcopy(outcome.tool_schema)
            handler_fn = outcome.handler_fn
            problems = contract_problems(schema, handler_fn)
            if problems:
                return ToolResult("Cannot activate a tool with an invalid contract: "
                                  + "; ".join(problems), ok=False)
            tool_name = schema["name"]
            schema_digest = hashlib.sha256(json.dumps(
                schema, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest()

            def conflict() -> str:
                table = getattr(dispatcher, "handler_table", None)
                native = table() if callable(table) else {}
                if tool_name in native:
                    return "that name belongs to a native tool"
                if tool_name in getattr(dispatcher, "_mcp_routes", {}):
                    return "that name belongs to a connected MCP tool"
                return ""

            reason = conflict()
            if reason:
                return ToolResult(f"Cannot activate '{tool_name}': {reason}. Choose a unique plugin name.", ok=False)
            # The schema's name, judged as load_and_register judged the file
            # name: a plugin that duplicates a built-in tool stays inactive.
            table = getattr(dispatcher, "handler_table", None)
            try:
                natives = set(table()) if callable(table) else set()
            except Exception:
                natives = set()
            duplicate = self.duplicate_reason(
                tool_name, getattr(outcome, "module_path", None), natives)
            if duplicate:
                getattr(self, "loaded_modules", {}).pop(getattr(outcome, "module_name", ""), None)
                return ToolResult(f"Cannot activate '{tool_name}': {duplicate}.", ok=False,
                                  receipt={"operation": "activate_tool", "state": "duplicate",
                                           "tool": tool_name, "registry_changed": False})

            missing = object()
            previous_handler = getattr(dispatcher, "_tool_handlers", {}).get(tool_name, missing)
            previous_schemas = [item for item in getattr(dispatcher, "TOOL_DECLARATIONS", [])
                                if item.get("name") == tool_name]
            previous_values = copy.deepcopy(previous_schemas)

            # Indexing is optional metadata work. Finish its cancellable await
            # before publication so cancellation cannot leave the caller with
            # an active handler but no capability-tier/scheduler integration.
            memory = getattr(dispatcher, "memory", None)
            if memory is not None:
                try:
                    await asyncio.to_thread(self._register_in_knowledge,
                                            memory, tool_name, copy.deepcopy(schema))
                except Exception:
                    pass
            reason = conflict()  # connected routes may have changed during indexing
            if reason:
                return ToolResult(f"Cannot activate '{tool_name}': {reason}. Choose a unique plugin name.", ok=False)
            current_schemas = [item for item in getattr(dispatcher, "TOOL_DECLARATIONS", [])
                               if item.get("name") == tool_name]
            if (getattr(dispatcher, "_tool_handlers", {}).get(tool_name, missing) is not previous_handler
                    or len(current_schemas) != len(previous_schemas)
                    or any(current is not previous for current, previous
                           in zip(current_schemas, previous_schemas))
                    or current_schemas != previous_values):
                return ToolResult(
                    f"Cannot activate '{tool_name}': a newer registration replaced the captured generation; "
                    "the newer tool was preserved.", ok=False,
                    receipt={"operation": "activate_tool", "state": "superseded",
                             "tool": tool_name, "registry_changed": False})
            problems = contract_problems(schema, handler_fn)
            if problems:
                return ToolResult("Tool contract changed before activation: " + "; ".join(problems), ok=False)

            # Dispatcher readers run on this loop. Publish both references
            # without yielding, so callers see one generation's contract.
            # In-flight calls retain their local reference to the old handler.
            handlers = dict(getattr(dispatcher, "_tool_handlers", {}))
            declarations = list(getattr(dispatcher, "TOOL_DECLARATIONS", []))
            try:
                self._register_handler(dispatcher, tool_name, handler_fn)
                self._register_declaration(dispatcher, tool_name, schema)
            except BaseException:
                dispatcher._tool_handlers = handlers
                if hasattr(dispatcher, "TOOL_DECLARATIONS"):
                    dispatcher.TOOL_DECLARATIONS[:] = declarations
                raise

            # Optional observers cannot turn a completed publication into a
            # reported failure and cause the caller to retry it.
            self._activation_log(f"[FORGE] Loader: ✓ activated tool '{tool_name}' in dispatcher")

            return ToolResult(
                f"✓ Tool '{tool_name}' activated and registered. "
                f"Schema: {schema['description']}",
                ok=True,
                receipt={"operation": "activate_tool", "state": "activated",
                         "tool": tool_name, "schema_sha256": schema_digest},
            )

        except Exception as e:
            self._activation_log(f"[FORGE] Loader: ✗ activation failed: {e}")
            return ToolResult(
                f"✗ Failed to activate tool: {type(e).__name__}: {e}",
                ok=False,
            )

    def _activation_log(self, message: str) -> None:
        try:
            self.bus.log.emit(message)
        except Exception:
            pass

    def _register_handler(
        self,
        dispatcher: Any,
        tool_name: str,
        handler_fn: Callable[..., Any],
    ) -> None:
        """
        Register a handler function in the dispatcher's routing dictionary.

        Args:
            dispatcher: OrionDispatcher instance
            tool_name: Name of the tool
            handler_fn: Callable to execute the tool
        """
        if not hasattr(dispatcher, "_tool_handlers"):
            dispatcher._tool_handlers = {}
        dispatcher._tool_handlers[tool_name] = handler_fn

    def _register_declaration(
        self,
        dispatcher: Any,
        tool_name: str,
        schema: dict[str, Any],
    ) -> None:
        """
        Register a tool declaration in TOOL_DECLARATIONS.

        Args:
            dispatcher: OrionDispatcher instance
            tool_name: Name of the tool
            schema: OpenAI-compatible schema dict
        """
        if not hasattr(dispatcher, "TOOL_DECLARATIONS"):
            return

        # Replace an existing declaration rather than keep it: a re-forged or
        # reloaded tool with NEW arguments used to keep advertising its OLD
        # schema, so the model called it with arguments it no longer takes.
        for index, tool in enumerate(dispatcher.TOOL_DECLARATIONS):
            if tool.get("name") == tool_name:
                dispatcher.TOOL_DECLARATIONS[index] = schema
                return
        dispatcher.TOOL_DECLARATIONS.append(schema)

    def _register_in_knowledge(
        self,
        memory_agent: Any,
        tool_name: str,
        schema: dict[str, Any],
    ) -> None:
        """
        Register tool description in FTS5 KNOWLEDGE memory tier.

        Args:
            memory_agent: MemoryAgent instance
            tool_name: Name of the tool
            schema: OpenAI-compatible schema dict
        """
        if not hasattr(memory_agent, "save_intelligence"):
            return

        description = schema.get("description", "Custom forge-generated tool")
        key_ref = f"forge_tool_{tool_name}"

        try:
            memory_agent.save_intelligence(
                category="forge_tools",
                key_ref=key_ref,
                value=description,
            )
        except Exception:
            # Silently fail; knowledge registration is optional
            pass

    def tool_result(self, outcome: ModuleLoadOutcome) -> ToolResult:
        """
        Convert ModuleLoadOutcome to ToolResult for dispatcher.

        Args:
            outcome: Module load outcome

        Returns:
            ToolResult with load summary
        """
        if outcome.succeeded:
            schema_name = outcome.tool_schema.get("name", "unknown") if outcome.tool_schema else "unknown"
            msg = (
                f"✓ Module '{outcome.module_name}' loaded successfully "
                f"({outcome.duration_ms:.1f}ms).\n"
                f"Tool: {schema_name}"
            )
            return ToolResult(msg, ok=True)
        else:
            msg = (
                f"✗ Failed to load module '{outcome.module_name}'. Errors:\n"
                + "\n".join(outcome.error_log[-10:])
            )
            return ToolResult(msg, ok=False)
