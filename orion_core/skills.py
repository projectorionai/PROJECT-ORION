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

import json
import shutil
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
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("skill manifest root must be an object")
        name = SecuritySanitiser.guard_text(str(raw.get("name") or ""), "skill.name")
        name = "".join(c for c in name.lower().replace(" ", "-")
                       if c.isalnum() or c == "-").strip("-")[:60]
        if not name:
            raise ValueError("skill manifest needs a name")
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
        if not root.exists():
            return skills
        for manifest in sorted(root.glob(f"*/{_MANIFEST}")):
            try:
                skills.append(SkillLoader.load_manifest(manifest))
            except Exception:
                continue  # a broken package must never break the registry
        return skills


class SkillRegistry:
    """In-memory index of installed skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def put(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def remove(self, name: str) -> Skill | None:
        return self._skills.pop(str(name or "").strip().lower(), None)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(str(name or "").strip().lower())

    def all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def enabled(self) -> list[Skill]:
        return [s for s in self.all() if s.enabled]


class SkillInstaller:
    """Copies validated skill packages into the skills root."""

    def __init__(self, root: Path = SKILLS_ROOT) -> None:
        self.root = root

    def install(self, source: Path) -> Skill:
        """Install from a skill directory or a bare skill.json file."""
        source = Path(source)
        manifest = source / _MANIFEST if source.is_dir() else source
        if not manifest.exists():
            raise FileNotFoundError(f"no {_MANIFEST} at {source}")
        skill = SkillLoader.load_manifest(manifest)  # validate BEFORE copying
        target = self.root / skill.name
        existing = target / _MANIFEST
        if existing.exists():
            current = SkillLoader.load_manifest(existing)
            if current.version == skill.version:
                raise ValueError(
                    f"skill '{skill.name}' v{skill.version} is already installed")
        target.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for item in source.iterdir():
                if item.is_file() and item.suffix.lower() in {".json", ".md", ".txt"}:
                    shutil.copy2(item, target / item.name)
        else:
            shutil.copy2(manifest, target / _MANIFEST)
        # Stamp the installation time into the stored manifest.
        stored = json.loads((target / _MANIFEST).read_text(encoding="utf-8"))
        stored["installed_at"] = utc_stamp()
        atomic_write_text((target / _MANIFEST),
            json.dumps(stored, indent=2), encoding="utf-8")
        return SkillLoader.load_manifest(target / _MANIFEST)

    def uninstall(self, name: str) -> bool:
        clean = "".join(c for c in str(name or "").lower() if c.isalnum() or c == "-")
        target = (self.root / clean).resolve()
        if not clean or self.root.resolve() not in target.parents:
            return False
        if not (target / _MANIFEST).exists():
            return False
        shutil.rmtree(target, ignore_errors=True)
        return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        manifest = self.root / str(name or "").strip().lower() / _MANIFEST
        if not manifest.exists():
            return False
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["enabled"] = bool(enabled)
        atomic_write_text(manifest, json.dumps(data, indent=2), encoding="utf-8")
        return True


class SkillManager:
    """Facade over loader/registry/installer; owns the built-in seed set."""

    def __init__(self, bus: OrionBus, root: Path = SKILLS_ROOT,
                 memory: Any | None = None, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.root = root
        self.memory = memory
        self.telemetry = telemetry
        self.registry = SkillRegistry()
        self.installer = SkillInstaller(root)
        self._seed_builtins()
        self.reload()
        if telemetry is not None:
            telemetry.health.register("skills")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reload(self) -> int:
        self.registry = SkillRegistry()
        for skill in SkillLoader.scan(self.root):
            self.registry.put(skill)
        return len(self.registry.all())

    def install(self, source: str) -> ToolResult:
        try:
            skill = self.installer.install(Path(str(source or "").strip()))
        except Exception as exc:
            return ToolResult(f"Skill install failed: {first_line(exc)}", ok=False)
        self.registry.put(skill)
        self._seed_memory(skill)
        self.bus.log.emit(f"SKILLS: installed {skill.name} v{skill.version}")
        return ToolResult(f"Skill installed — {skill.summary()}")

    def remove(self, name: str) -> ToolResult:
        if not self.installer.uninstall(name):
            return ToolResult(f"No installed skill named '{name}'.", ok=False)
        self.registry.remove(name)
        return ToolResult(f"Skill '{name}' removed.")

    def set_enabled(self, name: str, enabled: bool) -> ToolResult:
        if not self.installer.set_enabled(name, enabled):
            return ToolResult(f"No installed skill named '{name}'.", ok=False)
        self.reload()
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
        parts: list[str] = []
        used = 0
        for skill in self.registry.enabled():
            fragment = skill.prompts.get("system", "").strip()
            if not fragment:
                continue
            if used + len(fragment) > limit:
                break
            parts.append(f"[SKILL {skill.name}] {fragment}")
            used += len(fragment)
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
            target = self.root / manifest["name"]
            if (target / _MANIFEST).exists():
                continue
            try:
                target.mkdir(parents=True, exist_ok=True)
                data = dict(manifest)
                data["installed_at"] = utc_stamp()
                atomic_write_text((target / _MANIFEST),
                    json.dumps(data, indent=2), encoding="utf-8")
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
