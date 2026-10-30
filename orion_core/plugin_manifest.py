"""
Plugin manifests (Mark X.12 §2.6).

The forge and `ReflectiveModuleLoader` already load a custom tool (`*_tool.py`
exporting `get_tool_schema()` + `run()`) from `config/custom_tools/` and register
it into the dispatcher without editing `dispatcher.py`. What that code-only path
can't express is the *declarative* metadata a dropped-in third-party skill needs:

  * **capability tier** — how the remote gate should treat it (ALLOW / CONFIRM /
    FORBID), so a plugin can't silently grant itself unattended remote power;
  * **required dependencies** — so a plugin that needs a library which isn't
    installed is *skipped with an actionable log*, never crash-loaded and
    quarantined.

A manifest is a small JSON file `<name>.plugin.json` sitting beside its module:

    {
      "name": "spotify_control",
      "description": "Play/pause and search the local Spotify client.",
      "module": "spotify_control_tool.py",
      "tier": "confirm",
      "requires": ["spotipy"],
      "parameters": { "type": "OBJECT", "properties": { ... } }
    }

This module parses, validates and dependency-checks manifests, and plans which
are loadable. Actually importing the module and registering the handler reuses
the existing `ReflectiveModuleLoader`; wiring the two together (and feeding the
tier into the remote gate) is the integration step on top of this core.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .bus import OrionBus
    from .dynamic_loader import ReflectiveModuleLoader

# The capability tiers understood by the remote gate (ALLOW/CONFIRM/FORBID).
CAPABILITY_TIERS = frozenset({"allow", "confirm", "forbid"})
_DEFAULT_TIER = "confirm"   # safest sensible default for an unknown plugin
MANIFEST_SUFFIX = ".plugin.json"


def plugin_identity(name: str) -> str:
    """Management identity; callable/schema names remain exactly as declared."""
    cleaned = "".join(
        char if (char.isalnum() or char == "_") else "_"
        for char in str(name or "").strip().lower()
    ).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "plugin"


def _module_filename(module: str) -> bool:
    """Manifest modules are sibling files, including on a different OS."""
    return (isinstance(module, str) and module.endswith(".py") and module != ".py"
            and Path(module).name == module and PureWindowsPath(module).name == module
            and ":" not in module and not any(ord(char) < 32 for char in module))


#: The capabilities a plugin may declare it needs. Deliberately coarse — this is
#: a *disclosure* vocabulary (what does this code touch?), not a sandbox: ORION
#: does not yet enforce them at runtime, and claiming otherwise would be fake
#: security. audit_permissions() checks a module's AST against its declaration,
#: so an undeclared capability is surfaced before the plugin is imported.
PERMISSIONS: frozenset[str] = frozenset({
    "network",      # sockets, requests, urllib, aiohttp …
    "filesystem",   # open(), pathlib writes, shutil …
    "subprocess",   # spawning processes
    "input",        # synthetic keyboard/mouse (pyautogui, keyboard …)
    "screen",       # screen capture
    "audio",        # microphone / speaker
})


class ManifestError(ValueError):
    """A manifest is structurally invalid (bad JSON, missing or illegal fields)."""


@dataclass(frozen=True)
class PluginManifest:
    name: str
    description: str
    module: str                       # filename of the tool module, beside the manifest
    tier: str = _DEFAULT_TIER
    requires: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    #: Semantic-ish version string. Optional and defaulted, so every existing
    #: manifest keeps loading unchanged; it exists so an installed plugin can be
    #: compared against a newer copy (see PluginRegistry.check_update).
    version: str = "0.0.0"
    #: Declared capabilities this plugin needs — see PERMISSIONS. Purely
    #: declarative: audit_permissions() statically compares the declaration
    #: against what the module actually does, so an UNDECLARED capability is
    #: visible before the plugin is ever imported.
    permissions: tuple[str, ...] = ()
    #: Bus events this plugin wants delivered to its optional on_event() handler
    #: (see plugin_events.SUBSCRIBABLE). Empty = a passive, tool-only plugin.
    events: tuple[str, ...] = ()
    #: When this plugin runs on its own: a cron expression ("cron(0 6 * * *)")
    #: or the @daily family. Mutually exclusive with interval_seconds.
    schedule: str = ""
    #: The other way to say it: every N seconds from start-up. Minimum 30 —
    #: anything faster is a loop, not a schedule.
    interval_seconds: float = 0.0
    #: Keys requested from config/plugin_vault.json for this plugin's run.
    #: The filtered mapping is not isolation from the host process or files.
    secrets: tuple[str, ...] = ()
    source_path: Path | None = None   # where this manifest was read from

    @property
    def parsed_schedule(self) -> Any:
        """This plugin's schedule, or None if it only runs on request."""
        from .plugin_schedule import parse_schedule

        return parse_schedule(self.schedule, self.interval_seconds)

    @property
    def is_scheduled(self) -> bool:
        return bool(self.schedule) or self.interval_seconds > 0

    @property
    def module_path(self) -> Path | None:
        if self.source_path is None or not _module_filename(self.module):
            return None
        try:
            root = self.source_path.parent.resolve()
            candidate = root / self.module
            # A sibling symlink must not redirect import/quarantine elsewhere.
            return candidate if candidate.resolve().parent == root else None
        except (OSError, RuntimeError, ValueError):
            return None

    @classmethod
    def from_dict(cls, data: dict, *, source_path: Path | None = None) -> "PluginManifest":
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ManifestError("manifest is missing a non-empty 'name'")
        module = str(data.get("module") or "").strip()
        if not _module_filename(module):
            raise ManifestError(f"{name}: 'module' must be a .py filename beside the manifest")
        tier = str(data.get("tier") or _DEFAULT_TIER).strip().lower()
        if tier not in CAPABILITY_TIERS:
            raise ManifestError(
                f"{name}: 'tier' must be one of {sorted(CAPABILITY_TIERS)}, got {tier!r}")
        requires_raw = data.get("requires") or []
        if not isinstance(requires_raw, (list, tuple)):
            raise ManifestError(f"{name}: 'requires' must be a list")
        requires = tuple(str(r).strip() for r in requires_raw if str(r).strip())
        params = data.get("parameters") or {}
        if not isinstance(params, dict):
            raise ManifestError(f"{name}: 'parameters' must be an object")
        version = str(data.get("version") or "0.0.0").strip() or "0.0.0"
        events_raw = data.get("events") or []
        if not isinstance(events_raw, (list, tuple)):
            raise ManifestError(f"{name}: 'events' must be a list")
        events = tuple(str(e).strip() for e in events_raw if str(e).strip())
        perms_raw = data.get("permissions") or []
        if not isinstance(perms_raw, (list, tuple)):
            raise ManifestError(f"{name}: 'permissions' must be a list")
        permissions = tuple(
            str(perm).strip().lower() for perm in perms_raw if str(perm).strip())
        unknown = [perm for perm in permissions if perm not in PERMISSIONS]
        if unknown:
            raise ManifestError(
                f"{name}: unknown permission(s) {unknown}; "
                f"valid: {sorted(PERMISSIONS)}")

        secrets_raw = data.get("secrets") or []
        if not isinstance(secrets_raw, (list, tuple)):
            raise ManifestError(f"{name}: 'secrets' must be a list")
        secrets = tuple(str(s).strip() for s in secrets_raw if str(s).strip())

        # A schedule is validated HERE, at load, rather than at the moment it
        # would first fire. A plugin that says "cron(0 6 * *)" should be
        # rejected while someone is looking at it, not at six tomorrow morning.
        schedule = str(data.get("schedule") or "").strip()
        try:
            interval = float(data.get("interval_seconds") or 0)
        except (TypeError, ValueError, OverflowError):
            raise ManifestError(f"{name}: 'interval_seconds' must be a number")
        if schedule or interval:
            from .plugin_schedule import ScheduleError, parse_schedule

            try:
                parse_schedule(schedule, interval)
            except ScheduleError as exc:
                raise ManifestError(f"{name}: {exc}") from exc

        return cls(
            name=name,
            description=str(data.get("description") or "").strip(),
            module=module,
            tier=tier,
            requires=requires,
            parameters=params,
            version=version,
            permissions=permissions,
            events=events,
            schedule=schedule,
            interval_seconds=interval,
            secrets=secrets,
            source_path=source_path,
        )


def load_manifest(path: Path) -> PluginManifest:
    """Parse and validate one `*.plugin.json` file."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path.name}: invalid JSON — {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"{path.name}: could not read manifest — {exc}") from exc
    except (ValueError, RecursionError) as exc:
        # Decoder limits (for example huge integers or deep nesting) do not
        # use JSONDecodeError. One such file must not abort all discovery.
        raise ManifestError(f"{path.name}: invalid JSON — {exc}") from exc
    return PluginManifest.from_dict(data, source_path=path)


def unmet_requirements(manifest: PluginManifest) -> list[str]:
    """Return the manifest's required dependencies that are NOT importable."""
    missing: list[str] = []
    for dep in manifest.requires:
        top = dep.split(".")[0].split("==")[0].split(">=")[0].strip()
        if not top:
            continue
        try:
            found = importlib.util.find_spec(top) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            found = False
        if not found:
            missing.append(dep)
    return missing


@dataclass
class LoadPlan:
    loadable: list[PluginManifest] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # name -> reason


def discover_manifests(custom_tools_dir: Path) -> tuple[list[PluginManifest], dict[str, str]]:
    """Parse every `*.plugin.json` under *custom_tools_dir*. Returns (manifests,
    errors) — a bad manifest becomes an error entry, never an exception."""
    manifests: list[PluginManifest] = []
    errors: dict[str, str] = {}
    directory = Path(custom_tools_dir)
    try:
        paths = sorted(path for path in directory.iterdir() if path.name.endswith(MANIFEST_SUFFIX))
    except (FileNotFoundError, NotADirectoryError):
        return manifests, errors
    except OSError as exc:
        errors[str(directory)] = f"could not read plugin directory — {exc}"
        return manifests, errors
    for path in paths:
        try:
            manifests.append(load_manifest(path))
        except ManifestError as exc:
            errors[path.name] = str(exc)
    return manifests, errors


#: Plugins ORION has outgrown, by identity: (the built-in that does the job,
#: why). Mark XXXIX (owner, 27 Oct 2026: "ORION's got 5 plugins rejected so
#: this must be fixed as they are now obsolete"). Each was switched off and
#: counted at every boot as "rejected"; each is archived on sight now - a
#: verified copy in custom_tools/_superseded, never deleted - like a duplicate.
OBSOLETE_PLUGINS: dict[str, tuple[str, str]] = {
    "json_validator": ("diagnostics", "rebuilt into ORION: diagnostics 'config' checks "
                       "settings files and JSON (with or without a schema), no extra package"),
    "thought_model_monitor": ("ai_mode", "it reported simulated figures; the real local "
                              "brain's status is ai_mode 'brain'"),
    "dependency_manager": ("forge", "the forge resolves a tool's packages itself, with "
                           "its own confirmation; this installed from any requirements file"),
    "viseme_argument_adapter": ("emotion", "made for the retired 3-D face; the Screen Face "
                                "takes its mouth from his voice and its expressions from emotion"),
    "viseme_debugger": ("emotion", "made for the retired 3-D face (QuantumFace3D.set_viseme)"),
}


def retire_obsolete(custom_tools_dir: Path, registry: Any) -> dict[str, str]:
    """Archive every OBSOLETE_PLUGINS plugin found under *custom_tools_dir*
    through the registry. {name: skipped-reason} for the ones archived now.
    Blocking (file moves): call it off the event loop."""
    supersede = getattr(registry, "supersede", None)
    if not callable(supersede):
        return {}
    manifests, _errors = discover_manifests(custom_tools_dir)
    retired: dict[str, str] = {}
    for manifest in manifests:
        known = OBSOLETE_PLUGINS.get(plugin_identity(manifest.name))
        if known is None:
            continue
        builtin, why = known
        try:
            outcome = supersede(manifest.name, builtin, why)
        except Exception:
            continue
        if getattr(outcome, "ok", False):
            retired[manifest.name] = f"superseded by the built-in tool '{builtin}'"
    return retired


def plan_load(custom_tools_dir: Path) -> LoadPlan:
    """Decide which discovered plugins can load: manifest valid, module file
    present, and every required dependency importable. Everything else is
    recorded in ``skipped`` with a reason instead of failing."""
    plan = LoadPlan()
    manifests, errors = discover_manifests(custom_tools_dir)
    plan.skipped.update(errors)
    identities: dict[str, int] = {}
    for manifest in manifests:
        identity = plugin_identity(manifest.name)
        identities[identity] = identities.get(identity, 0) + 1
    for manifest in manifests:
        if identities[plugin_identity(manifest.name)] > 1:
            key = manifest.source_path.name if manifest.source_path else manifest.name
            plan.skipped[key] = f"duplicate manifest identity '{manifest.name}'; not importing an ambiguous tool"
            continue
        module_path = manifest.module_path
        try:
            present = module_path is not None and module_path.is_file()
        except OSError as exc:
            plan.skipped[manifest.name] = f"could not read module '{manifest.module}' — {exc}"
            continue
        if not present:
            plan.skipped[manifest.name] = f"module file '{manifest.module}' not found"
            continue
        missing = unmet_requirements(manifest)
        if missing:
            plan.skipped[manifest.name] = f"missing dependencies: {', '.join(missing)}"
            continue
        plan.loadable.append(manifest)
    return plan


async def load_plugins(
    loader: "ReflectiveModuleLoader",
    dispatcher: Any,
    custom_tools_dir: Path,
    bus: "OrionBus",
    registry: Any = None,
    events: Any = None,
    scheduler: Any = None,
) -> LoadPlan:
    """The integration step the module docstring calls out: discover every
    manifest under *custom_tools_dir*, load and register the loadable ones
    into the live dispatcher, and feed each one's declared capability tier
    into the remote gate.

    Reuses the SAME ReflectiveModuleLoader the Forge uses (pass forge.loader)
    rather than constructing a second one — a loaded plugin then shows up in
    forge.health()/snapshot() too, so there is only one place tracking
    "what's loaded and what's held back", not two.

    When a :class:`~orion_core.plugin_registry.PluginRegistry` is supplied it
    becomes authoritative for two things: a plugin the user has DISABLED is
    never loaded (the disable would otherwise last only until the next
    restart), and every load outcome is recorded against that plugin's health
    so ``plugin doctor`` can report what actually happened.  The parameter is
    optional so existing call sites keep working unchanged.
    """
    from .remote_capability import register_plugin_tier

    retired: dict[str, str] = {}
    if registry is not None:
        try:
            retired = await asyncio.to_thread(retire_obsolete, custom_tools_dir, registry)
        except Exception as exc:
            bus.log.emit(f"PLUGIN: could not archive obsolete plugins - {exc}")
    plan = await asyncio.to_thread(plan_load, custom_tools_dir)
    quarantined: set[str] = set()
    quarantine_reason = getattr(registry, "quarantine_reason", None)
    for name, reason in plan.skipped.items():
        # A module the loader quarantined leaves its manifest behind. That is
        # a known, recoverable state - not a lost file (25 Oct 2026, owner).
        held = ""
        if callable(quarantine_reason) and reason.startswith("module file"):
            try:
                held = await asyncio.to_thread(quarantine_reason, name)
            except Exception:
                held = ""
        if held:
            bus.log.emit(f"PLUGIN: '{name}' is quarantined ({held}) - the Plugins page can "
                         "restore or remove it.")
            _mark(registry, name, False, "")
            quarantined.add(name)
            continue
        bus.log.emit(f"PLUGIN: skipped '{name}' — {reason}")
        _mark(registry, name, False, reason)

    still_loadable: list[PluginManifest] = []
    for manifest in plan.loadable:
        if registry is not None and not registry.is_enabled(manifest.name):
            state_error = getattr(registry, "state_error", "")
            reason = (f"saved plugin state unavailable — {state_error}" if state_error
                      else "disabled by the user")
            plan.skipped[manifest.name] = reason
            bus.log.emit(f"PLUGIN: '{manifest.name}' held — {reason}; not loading.")
            continue
        still_loadable.append(manifest)
    plan.loadable = still_loadable

    active: list[PluginManifest] = []
    for manifest in list(plan.loadable):
        module_path = manifest.module_path
        if module_path is None:
            reason = "module path is unavailable or leaves the manifest directory"
            plan.skipped[manifest.name] = reason
            bus.log.emit(f"PLUGIN: '{manifest.name}' failed to load — {reason}")
            _mark(registry, manifest.name, False, reason)
            continue
        outcome = await loader.load_and_register(module_path)
        superseded = str(getattr(outcome, "superseded_by", "") or "")
        if not outcome.succeeded and superseded and callable(getattr(registry, "supersede", None)):
            # Held back on purpose: the built-in does this job. Neutral, and
            # archived out of the active set (registry.supersede keeps a copy).
            plan.skipped[manifest.name] = f"superseded by the built-in tool '{superseded}'"
            try:
                await asyncio.to_thread(registry.supersede, manifest.name, superseded,
                                        "; ".join(outcome.error_log))
            except Exception as exc:
                bus.log.emit(f"PLUGIN: '{manifest.name}' is superseded by '{superseded}' "
                             f"(could not archive it: {exc})")
            continue
        if not outcome.succeeded:
            reason = "; ".join(outcome.error_log) or "unknown error"
            plan.skipped[manifest.name] = reason
            bus.log.emit(f"PLUGIN: '{manifest.name}' failed to load — {reason}")
            _mark(registry, manifest.name, False, reason)
            continue
        schema = getattr(outcome, "tool_schema", None)
        if isinstance(schema, dict) and schema.get("name") != manifest.name:
            reason = (f"module declares tool '{schema.get('name')}', but the manifest "
                      f"declares '{manifest.name}'; capability and schedule names must agree")
            plan.skipped[manifest.name] = reason
            bus.log.emit(f"PLUGIN: '{manifest.name}' held — {reason}")
            _mark(registry, manifest.name, False, reason)
            continue
        activation = await loader.activate_tool(outcome, dispatcher)
        if not activation.ok:
            plan.skipped[manifest.name] = activation.text
            bus.log.emit(f"PLUGIN: '{manifest.name}' failed to activate — {activation.text}")
            _mark(registry, manifest.name, False, activation.text)
            continue
        register_plugin_tier(manifest.name, manifest.tier)
        active.append(manifest)
        # Mark XXVI: bind the plugin's declared event hooks (if any). Guarded —
        # a plugin that cannot be hooked is still a perfectly good tool.
        if events is not None:
            try:
                # The imported module object lives on the loader, not on the
                # outcome (ModuleLoadOutcome carries the schema/handler only).
                entry = getattr(loader, "loaded_modules", {}).get(outcome.module_name) or {}
                if manifest.events:
                    events.subscribe(manifest.name, entry.get("module"), manifest.events)
                else:
                    events.forget(manifest.name)
            except Exception as exc:
                bus.log.emit(f"PLUGIN: could not hook '{manifest.name}' - {exc}")
        _mark(registry, manifest.name, True, "")
        bus.log.emit(f"PLUGIN: '{manifest.name}' loaded and live (tier={manifest.tier}).")
        # Mark XXVI: say what it can touch AT LOAD TIME. A disclosure gap that
        # only appears when somebody thinks to run 'plugin audit' is a disclosure
        # nobody sees; this puts it in the ordinary log.
        try:
            _disclose(manifest, module_path, bus)
        except Exception:
            pass

        # A manifest that declared a schedule starts being watched the moment
        # it loads, rather than at the next restart.
        if scheduler is not None:
            try:
                if getattr(manifest, "is_scheduled", False):
                    scheduler.add(manifest)
                else:
                    scheduler.remove(manifest.name)
            except Exception as exc:
                bus.log.emit(f"PLUGIN: '{manifest.name}' could not be "
                             f"scheduled - {exc}")

    plan.loadable = active
    plan.skipped.update(retired)
    # The boot roll-call's closing line. Plugins load in the background, so
    # this cannot be printed with the tool list — it is printed when the load
    # has actually finished, which is the only moment the counts are true.
    # Only a FAILURE is "rejected" (Mark XXXIX): five plugins the owner had
    # switched off were reported every boot as "5 rejected".
    try:
        from .startup_report import PLUGINS, say

        loaded = len(getattr(plan, "loadable", ()) or ())
        skipped = dict(getattr(plan, "skipped", {}) or {})
        off = sum(1 for reason in skipped.values() if str(reason) == "disabled by the user")
        superseded = sum(1 for reason in skipped.values()
                         if str(reason).startswith("superseded by"))
        parked = sum(1 for name in skipped if name in quarantined)
        rejected = len(skipped) - off - superseded - parked
        extra = [f"{n} {label}" for n, label in ((off, "switched off"),
                                                 (superseded, "retired"),
                                                 (parked, "quarantined")) if n]
        say(f"{PLUGINS} Plugin discovery complete: {loaded} active, {rejected} rejected"
            + (f" ({', '.join(extra)})" if extra else "")
            + f", {loaded + len(skipped)} total.", bus)
    except Exception:
        pass
    return plan


def _disclose(manifest: "PluginManifest", module_path: Path, bus: "OrionBus") -> None:
    """Log what a freshly-loaded plugin can touch, and flag two things worth
    seeing without being asked: capabilities it never declared, and an
    unattended (tier=allow) plugin holding a sensitive capability."""
    from .plugin_registry import _UNATTENDED_RISK, audit_permissions

    report = audit_permissions(module_path, manifest.permissions)
    undeclared = report["undeclared"]
    if undeclared:
        bus.log.emit(
            f"PLUGIN: '{manifest.name}' uses UNDECLARED capabilities: "
            + ", ".join(undeclared) + " (see 'plugin audit')")
    touches = set(manifest.permissions) | set(undeclared)
    if manifest.tier == "allow":
        dangerous = sorted(touches & _UNATTENDED_RISK)
        if dangerous:
            bus.log.emit(
                f"PLUGIN: '{manifest.name}' runs UNATTENDED (tier=allow) while "
                "touching " + ", ".join(dangerous)
                + " — consider tier 'confirm' in its manifest.")


def _mark(registry: Any, name: str, ok: bool, error: str) -> None:
    """Record a load outcome against the registry — never fatal."""
    if registry is None:
        return
    try:
        registry.mark_loaded(name, ok, error)
    except Exception:
        pass
