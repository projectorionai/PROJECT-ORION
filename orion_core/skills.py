"""
Skills system (Phase 3) — installable capability packages.

A skill is a DATA package, not code: a directory under ``<workspace>/skills/``
holding a ``skill.json`` manifest with prompts, workflow references, tool
lists, templates and seed memory. Skills teach ORION how to approach a class
of work (research method, creator coaching rubric, content strategy) without
the dynamic-code risks of the Forge — nothing in a skill is ever executed.

    SkillLoader     — reads + validates manifests from disk
    SkillRegistry   — the in-memory index (name → manifest, path, enabled)
    SkillInstaller  — install from a directory/JSON file, remove, update
    SkillManager    — facade the dispatcher talks to; also contributes the
                      active skills' system-prompt fragments to the model

Built-in starter skills are seeded on first run so the system is useful out
of the box. All manifest text passes through the security sanitiser.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line, utc_stamp
from .atomic_io import atomic_write_text

SKILLS_ROOT = BASE_DIR / "skills"

_MANIFEST = "skill.json"
_TEXT_FIELDS = ("name", "version", "description", "author")
_MAX_PROMPT = 4000
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_FILE_BYTES = 1024 * 1024
_MAX_PACKAGE_BYTES = 2 * 1024 * 1024
_MAX_PACKAGE_FILES = 64
_MAX_SCAN_ENTRIES = 512
_MAX_CONTEXT_CHARS = 12000
_WINDOWS = os.name == "nt"
_RENAME_DELAYS = (0.025, 0.05, 0.1, 0.15)


def _present(path: Path) -> bool:
    """Unlike exists(), retain broken links as occupied destinations."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _directory_snapshot(path: Path) -> tuple:
    """Bounded identity/content receipt for a directory we are about to move.

    Directory timestamps are excluded because a rename can change them. File
    identities and bytes, and every directory identity/name, must be retained.
    The extra manifest allowance covers our installed_at checkpoint.
    """
    path = _safe_path(path)
    rows = []
    pending = [path]
    entries = files = total = 0
    while pending:
        directory = pending.pop()
        _safe_path(directory)
        before = directory.stat()
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("skill move source is not a regular directory")
        identity = (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode))
        rows.append((directory.relative_to(path).as_posix(), identity))
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > _MAX_SCAN_ENTRIES:
                    raise ValueError("skill recovery directory contains too many entries")
                item = _safe_path(directory / child.name)
                info = item.stat()
                if stat.S_ISDIR(info.st_mode):
                    pending.append(item)
                    continue
                files += 1
                if not stat.S_ISREG(info.st_mode) or files > _MAX_PACKAGE_FILES:
                    raise ValueError("skill recovery contains unsupported or too many files")
                limit = _MAX_MANIFEST_BYTES if item.name == _MANIFEST else _MAX_FILE_BYTES
                data = _bounded_bytes(item, limit)
                total += len(data)
                if total > _MAX_PACKAGE_BYTES + _MAX_MANIFEST_BYTES:
                    raise ValueError("skill recovery directory exceeds its byte limit")
                after = item.stat()
                signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError("skill recovery file changed during verification")
                rows.append((item.relative_to(path).as_posix(),
                             signature + (hashlib.sha256(data).hexdigest(),)))
        after = directory.stat()
        if identity != (after.st_dev, after.st_ino, stat.S_IFMT(after.st_mode)):
            raise ValueError("skill recovery directory changed during verification")
    return tuple(sorted(rows))


def _identity(name: str) -> str:
    """A package identity is one directory name, never a filesystem address."""
    clean = str(name or "").strip().lower()
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10))}
    if (not clean or len(clean) > 60 or not clean[0].isalnum()
            or not clean[-1].isalnum()
            or any(not (c.isalnum() or c == "-") for c in clean)
            or clean in reserved):
        raise ValueError("invalid skill identity; use its listed name, not a path")
    return clean


def _safe_path(path: Path) -> Path:
    """Reject links in the item and its ancestors before reading or writing."""
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ValueError("skill paths must not contain symbolic links or junctions")
    return path.resolve()


def _bounded_bytes(path: Path, limit: int) -> bytes:
    path = _safe_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("skill file is not regular or exceeds its byte limit")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(data) > limit or (before.st_size, before.st_mtime_ns)
            != (after.st_size, after.st_mtime_ns)):
        raise ValueError("skill file changed while reading or exceeds its byte limit")
    return data


def _manifest_text(data: dict[str, Any]) -> str:
    text = json.dumps(data, indent=2)
    if len(text.encode("utf-8")) > _MAX_MANIFEST_BYTES:
        raise ValueError("serialized skill manifest exceeds its byte limit")
    return text


@dataclass
class Skill:
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "ORION"
    prompts: dict[str, str] = field(default_factory=dict)
    workflows: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    templates: dict[str, str] = field(default_factory=dict)
    memory: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    path: str = ""
    installed_at: str = ""

    def summary(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return (f"{self.name} v{self.version} [{state}] — {self.description[:90]}"
                + (f" (tools: {', '.join(self.tools[:4])})" if self.tools else ""))


class SkillLoader:
    """Reads and validates skill manifests. Data only — never executes."""

    @staticmethod
    def load_manifest(path: Path) -> Skill:
        raw = json.loads(_bounded_bytes(path, _MAX_MANIFEST_BYTES).decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("skill manifest root must be an object")
        name = SecuritySanitiser.guard_text(str(raw.get("name") or ""), "skill.name")
        name = _identity("".join(c for c in name.lower().replace(" ", "-")
                                if c.isalnum() or c == "-").strip("-"))
        skill = Skill(name=name, path=str(path.parent))
        for fld in _TEXT_FIELDS[1:]:
            value = str(raw.get(fld) or getattr(skill, fld))
            setattr(skill, fld, SecuritySanitiser.guard_text(value, f"skill.{fld}")[:300])
        prompts = raw.get("prompts") or {}
        if isinstance(prompts, dict):
            skill.prompts = {
                str(k)[:60]: SecuritySanitiser.guard_text(str(v), "skill.prompt")[:_MAX_PROMPT]
                for k, v in list(prompts.items())[:12] if str(v).strip()
            }
        for list_field in ("workflows", "tools"):
            items = raw.get(list_field) or []
            if isinstance(items, list):
                setattr(skill, list_field,
                        [SecuritySanitiser.guard_text(str(i), f"skill.{list_field}")[:80]
                         for i in items[:20] if str(i).strip()])
        for map_field in ("templates", "memory"):
            data = raw.get(map_field) or {}
            if isinstance(data, dict):
                setattr(skill, map_field, {
                    str(k)[:80]: SecuritySanitiser.guard_text(str(v), f"skill.{map_field}")[:2000]
                    for k, v in list(data.items())[:20] if str(v).strip()
                })
        skill.enabled = bool(raw.get("enabled", True))
        skill.installed_at = str(raw.get("installed_at") or "")
        return skill

    @staticmethod
    def scan(root: Path) -> list[Skill]:
        skills: list[Skill] = []
        root = _safe_path(root)
        if not root.exists():
            return skills
        # Only a finite set is materialised/sorted. Recovery directories are
        # deliberately not packages and cannot replace the active registry.
        packages = []
        with os.scandir(root) as entries:
            for _ in range(_MAX_SCAN_ENTRIES):
                entry = next(entries, None)
                if entry is None:
                    break
                try:
                    if _identity(entry.name) == entry.name and entry.is_dir(follow_symlinks=False):
                        packages.append(root / entry.name)
                except ValueError:
                    continue
        for package in sorted(packages):
            try:
                skill = SkillLoader.load_manifest(package / _MANIFEST)
                if skill.name == package.name:
                    skills.append(skill)
            except Exception:
                continue  # a broken package must never break the registry
        return skills


class SkillRegistry:
    """In-memory index of installed skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._lock = threading.RLock()

    def put(self, skill: Skill) -> None:
        with self._lock:
            self._skills[skill.name] = copy.deepcopy(skill)

    def remove(self, name: str) -> Skill | None:
        with self._lock:
            return self._skills.pop(str(name or "").strip().lower(), None)

    def get(self, name: str) -> Skill | None:
        with self._lock:
            return copy.deepcopy(self._skills.get(str(name or "").strip().lower()))

    def all(self) -> list[Skill]:
        with self._lock:
            return copy.deepcopy(sorted(self._skills.values(), key=lambda s: s.name))

    def enabled(self) -> list[Skill]:
        return [s for s in self.all() if s.enabled]


class SkillInstaller:
    """Copies validated skill packages into the skills root."""

    def __init__(self, root: Path = SKILLS_ROOT) -> None:
        self.root = root
        self.retained: list[Path] = []
        self._lock = threading.RLock()

    def _target(self, name: str) -> Path:
        root = _safe_path(self.root)
        target = _safe_path(root / _identity(name))
        if target.parent != root:
            raise ValueError("skill target is outside its root")
        return target

    def _rename_owned_directory(self, source: Path, target: Path, expected: tuple) -> None:
        """Retry only a verified, denied rename; never recopy or replay install.

        Windows rename refuses an occupied destination atomically. On other
        platforms the prechecks are defensive checks, not a filesystem CAS.
        """
        root = _safe_path(self.root)
        source, target = Path(source).absolute(), Path(target).absolute()

        def checked_paths() -> None:
            for path in (source, target):
                resolved = _safe_path(path)
                if resolved == root or not resolved.is_relative_to(root):
                    raise ValueError("skill rename is outside its owned root")
            if _safe_path(self.root) != root:
                raise ValueError("skill root changed during publication")

        def completed() -> bool:
            checked_paths()
            return (not _present(source) and _present(target)
                    and _directory_snapshot(target) == expected)

        for attempt in range(len(_RENAME_DELAYS) + 1):
            checked_paths()
            if _present(target):
                raise FileExistsError("skill rename destination is already occupied")
            if _directory_snapshot(source) != expected:
                raise ValueError("skill rename source changed; recovery copies retained")
            try:
                # MoveFile on Windows fails if a competing destination appears.
                (os.rename if _WINDOWS else os.replace)(source, target)
            except OSError as exc:
                if completed():
                    return  # the operation completed before its error was observed
                if (not _WINDOWS or getattr(exc, "winerror", None) not in {5, 32, 33}
                        or attempt == len(_RENAME_DELAYS)):
                    raise
                # Do not sleep/retry over a changed stage or another writer.
                checked_paths()
                if _present(target):
                    raise FileExistsError("skill rename destination became occupied") from exc
                if _directory_snapshot(source) != expected:
                    raise ValueError("skill rename source changed; recovery copies retained") from exc
                time.sleep(_RENAME_DELAYS[attempt])
            else:
                if not completed():
                    raise OSError("skill directory rename could not be verified")
                return

    @staticmethod
    def _package(source: Path, manifest: Path) -> dict[str, bytes]:
        if not source.is_dir():
            return {_MANIFEST: _bounded_bytes(manifest, _MAX_MANIFEST_BYTES)}
        files: dict[str, bytes] = {}
        total = 0
        with os.scandir(source) as entries:
            for index, entry in enumerate(entries):
                if index >= _MAX_SCAN_ENTRIES:
                    raise ValueError("skill package contains too many entries")
                path = _safe_path(source / entry.name)
                if not entry.is_file(follow_symlinks=False) or path.suffix.lower() not in {".json", ".md", ".txt"}:
                    continue
                if len(files) >= _MAX_PACKAGE_FILES:
                    raise ValueError("skill package contains too many data files")
                limit = _MAX_MANIFEST_BYTES if path.name == _MANIFEST else _MAX_FILE_BYTES
                data = _bounded_bytes(path, min(limit, _MAX_PACKAGE_BYTES - total))
                total += len(data)
                files[path.name] = data
        if _MANIFEST not in files:
            raise ValueError("skill package has no regular skill.json")
        return files

    def install(self, source: Path) -> Skill:
        """Install from a skill directory or a bare skill.json file."""
        with self._lock:
            return self._install(source)

    def _install(self, source: Path) -> Skill:
        self.retained = []
        source = _safe_path(source)
        manifest = source / _MANIFEST if source.is_dir() else source
        if not manifest.exists():
            raise FileNotFoundError(f"no {_MANIFEST} at {source}")
        skill = SkillLoader.load_manifest(manifest)  # validate BEFORE copying
        target = self._target(skill.name)
        existing = target / _MANIFEST
        if existing.exists():
            current = SkillLoader.load_manifest(existing)
            if current.version == skill.version:
                raise ValueError(
                    f"skill '{skill.name}' v{skill.version} is already installed")
        files = self._package(source, manifest)
        root = _safe_path(self.root)
        root.mkdir(parents=True, exist_ok=True)
        pending = root / (".skill-" + uuid.uuid4().hex[:12])
        pending.mkdir()
        self.retained.append(pending)
        inputs, candidate = pending / "inputs", pending / "package"
        inputs.mkdir()
        candidate.mkdir()
        # Snapshot bounded actual bytes first. copy2 never reads a growing
        # external file; its inputs are our inert, capped private snapshot.
        for name, data in files.items():
            captured = inputs / name
            captured.write_bytes(data)
            shutil.copy2(captured, candidate / name)
            if _bounded_bytes(candidate / name, len(data)) != data:
                raise ValueError("skill copy did not preserve the captured bytes")
        copied = SkillLoader.load_manifest(candidate / _MANIFEST)
        if (copied.name, copied.version) != (skill.name, skill.version):
            raise ValueError("skill identity changed during installation")
        stored = json.loads(files[_MANIFEST].decode("utf-8"))
        stored["installed_at"] = utc_stamp()
        atomic_write_text(candidate / _MANIFEST, _manifest_text(stored), encoding="utf-8")
        installed = SkillLoader.load_manifest(candidate / _MANIFEST)
        candidate_snapshot = _directory_snapshot(candidate)
        target = self._target(skill.name)                # recheck immediately before rename
        previous = None
        previous_snapshot = None
        if target.exists():
            previous = root / (".previous-" + uuid.uuid4().hex[:12])
            previous_snapshot = _directory_snapshot(target)
            try:
                self._rename_owned_directory(target, previous, previous_snapshot)
            finally:
                if _present(previous):
                    self.retained.append(previous)
        try:
            self._rename_owned_directory(candidate, target, candidate_snapshot)
        except BaseException:
            if previous is not None and not _present(target):
                self._rename_owned_directory(previous, target, previous_snapshot)
                self.retained.remove(previous)
            raise
        installed.path = str(target)
        # Successful staging leftovers can be recycled, never hard-deleted.
        # A refusal keeps a named recovery copy and does not undo publication.
        from . import recycle_bin
        try:
            if recycle_bin.refusal(pending) is None and recycle_bin.send_to_recycle_bin(pending):
                self.retained.remove(pending)
        except Exception:
            pass
        return installed

    def uninstall(self, name: str) -> bool:
        with self._lock:
            target = self._target(name)
            if not (target / _MANIFEST).exists():
                return False
            _safe_path(target / _MANIFEST)
            from . import recycle_bin
            refused = recycle_bin.refusal(target)
            if refused:
                raise ValueError("Recycle Bin refused skill removal: " + refused)
            if not recycle_bin.send_to_recycle_bin(target) or target.exists():
                raise OSError("skill removal was not verified in the Recycle Bin")
            return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        with self._lock:
            manifest = self._target(name) / _MANIFEST
            if not manifest.exists():
                return False
            data = json.loads(_bounded_bytes(manifest, _MAX_MANIFEST_BYTES).decode("utf-8"))
            loaded = SkillLoader.load_manifest(manifest)
            if loaded.name != manifest.parent.name:
                raise ValueError("skill manifest identity does not match its directory")
            data["enabled"] = bool(enabled)
            atomic_write_text(manifest, _manifest_text(data), encoding="utf-8")
            return True


class SkillManager:
    """Facade over loader/registry/installer; owns the built-in seed set."""

    def __init__(self, bus: OrionBus, root: Path = SKILLS_ROOT,
                 memory: Any | None = None, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.root = root
        self.memory = memory
        self.telemetry = telemetry
        self._mutation_lock = threading.RLock()
        self.registry = SkillRegistry()
        self.installer = SkillInstaller(root)
        self._seed_builtins()
        self.reload()
        if telemetry is not None:
            telemetry.health.register("skills")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reload(self) -> int:
        with self._mutation_lock:
            registry = SkillRegistry()
            for skill in SkillLoader.scan(self.root):
                registry.put(skill)
            self.registry = registry  # publish only the complete immutable snapshot
            return len(registry.all())

    def install(self, source: str) -> ToolResult:
        with self._mutation_lock:
            try:
                if not str(source or "").strip():
                    raise ValueError("a skill package path is required")
                skill = self.installer.install(Path(str(source).strip()))
            except Exception as exc:
                retained = self.installer.retained
                note = (" Recovery copies retained: " + ", ".join(str(p) for p in retained)
                        if retained else "")
                return ToolResult(f"Skill install failed: {first_line(exc)}" + note, ok=False)
            self.registry.put(skill)
            self._seed_memory(skill)
            self.bus.log.emit(f"SKILLS: installed {skill.name} v{skill.version}")
            retained = self.installer.retained
            note = (" Recovery copies retained: " + ", ".join(str(p) for p in retained)
                    if retained else "")
            return ToolResult(f"Skill installed — {skill.summary()}" + note)

    def remove(self, name: str) -> ToolResult:
        with self._mutation_lock:
            try:
                if not self.installer.uninstall(name):
                    return ToolResult(f"No installed skill named '{name}'.", ok=False)
            except Exception as exc:
                return ToolResult(f"Skill removal failed: {first_line(exc)}", ok=False)
            self.registry.remove(name)
            return ToolResult(f"Skill '{name}' recycled.")

    def set_enabled(self, name: str, enabled: bool) -> ToolResult:
        with self._mutation_lock:
            try:
                if not self.installer.set_enabled(name, enabled):
                    return ToolResult(f"No installed skill named '{name}'.", ok=False)
                self.reload()
            except Exception as exc:
                return ToolResult(f"Skill update failed: {first_line(exc)}", ok=False)
            state = "enabled" if enabled else "disabled"
            return ToolResult(f"Skill '{name}' {state}.")

    # ── queries ───────────────────────────────────────────────────────────────

    def list_skills(self) -> ToolResult:
        skills = self.registry.all()
        if not skills:
            return ToolResult("No skills installed yet.")
        return ToolResult("Installed skills:\n"
                          + "\n".join(f"- {s.summary()}" for s in skills))

    def describe(self, name: str) -> ToolResult:
        skill = self.registry.get(name)
        if skill is None:
            return ToolResult(f"No installed skill named '{name}'.", ok=False)
        lines = [skill.summary()]
        if skill.prompts:
            lines.append("Prompts: " + ", ".join(skill.prompts))
        if skill.workflows:
            lines.append("Workflows: " + ", ".join(skill.workflows))
        if skill.templates:
            lines.append("Templates: " + ", ".join(skill.templates))
        return ToolResult("\n".join(lines))

    def prompt_context(self, limit: int = 3000) -> str:
        """System-prompt fragments from enabled skills, budget-capped."""
        limit = max(0, min(int(limit), _MAX_CONTEXT_CHARS))
        parts: list[str] = []
        used = 0
        for skill in self.registry.enabled():
            fragment = skill.prompts.get("system", "").strip()
            if not fragment:
                continue
            line = f"[SKILL {skill.name}] {fragment}"
            cost = len(line) + bool(parts)
            if used + cost > limit:
                continue
            parts.append(line)
            used += cost
        return "\n".join(parts)

    def template(self, skill_name: str, template_name: str) -> str:
        skill = self.registry.get(skill_name)
        if skill is None:
            return ""
        return skill.templates.get(str(template_name or "").strip(), "")

    # ── built-ins ─────────────────────────────────────────────────────────────

    def _seed_memory(self, skill: Skill) -> None:
        if self.memory is None or not skill.memory:
            return
        for key, value in list(skill.memory.items())[:10]:
            try:
                self.memory.remember("knowledge", f"skill_{skill.name}_{key}", value)
            except Exception:
                pass

    def _seed_builtins(self) -> None:
        """First-run seeding: write starter skills that don't exist yet."""
        for manifest in _BUILTIN_SKILLS:
            try:
                target = self.installer._target(manifest["name"])
                if (target / _MANIFEST).exists():
                    _safe_path(target / _MANIFEST)
                    continue
                target.mkdir(parents=True, exist_ok=True)
                data = dict(manifest)
                data["installed_at"] = utc_stamp()
                atomic_write_text((target / _MANIFEST),
                    _manifest_text(data), encoding="utf-8")
            except Exception:
                continue


_BUILTIN_SKILLS: list[dict[str, Any]] = [
    {
        "name": "research-method",
        "version": "1.0.0",
        "description": "Structured research: decompose, gather, corroborate, conclude with confidence levels.",
        "prompts": {"system": (
            "When researching: decompose the topic into sub-questions, seek at "
            "least two independent sources per claim, state confidence levels, "
            "flag contradictions explicitly, and always separate evidence from "
            "inference.")},
        "workflows": ["research_workflow"],
        "tools": ["research", "second_brain"],
    },
    {
        "name": "creator-management",
        "version": "1.0.0",
        "description": "Creator coaching: hook/script/CTA scoring rubric and feedback etiquette.",
        "prompts": {"system": (
            "When reviewing creator content: score the hook (first 2 seconds), "
            "structure, pacing and CTA separately; lead feedback with what "
            "works, give one prioritised improvement, and tie every note to a "
            "retention or conversion mechanism.")},
        "templates": {
            "feedback": ("HOOK: {hook_score}/10 — {hook_note}\n"
                         "STRUCTURE: {structure_note}\nCTA: {cta_note}\n"
                         "PRIORITY FIX: {priority_fix}"),
        },
        "tools": ["creator_intel"],
    },
    {
        "name": "content-strategy",
        "version": "1.0.0",
        "description": "Short-form content strategy for TikTok, Reels and Shorts.",
        "prompts": {"system": (
            "For short-form strategy: anchor recommendations in the first-frame "
            "hook, watch-time retention, native platform style, posting cadence "
            "and iteration on winners. Prefer testing plans over opinions.")},
        "tools": ["creator_intel", "tiktok_intel", "instagram_intel"],
    },
    {
        "name": "business-analysis",
        "version": "1.0.0",
        "description": "Decision support: unit economics, risk framing, cheapest-validating-test thinking.",
        "prompts": {"system": (
            "For business questions: quantify unit economics where possible, "
            "name the riskiest assumption, propose the cheapest test that would "
            "validate it, and challenge weak reasoning constructively.")},
        "tools": ["executive", "business_advisor"],
    },
]


__all__ = ["Skill", "SkillInstaller", "SkillLoader", "SkillManager",
           "SkillRegistry", "SKILLS_ROOT"]
