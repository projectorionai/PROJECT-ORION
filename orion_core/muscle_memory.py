"""
Muscle memory (CAP-07) — record a task once, replay it forever.

    "record a sequence once, ORION generalises it into a reusable workflow."

You do a thing once with the visible browser co-pilot watching — open a site,
click a link, fill a couple of fields, submit. Muscle memory turns that recorded
sequence into a NAMED, PARAMETERISED skill you can run again with different
inputs: the demonstration is the programming.

Two ideas do the work:

* **The values you typed become the parameters.** A recording of "search for
  *wireless headphones*" generalises to a skill with a ``search`` parameter, so
  next time you can run it for *running shoes* without re-recording. Everything
  structural — which link, which button, the order — is kept; only the literal
  inputs float up into parameters.
* **The steps are already robust.** The co-pilot locates things by their visible
  text and field labels, not brittle CSS paths, so a replayed skill survives the
  small layout changes that break selector-based macros. Generalisation keeps
  that vocabulary, so a skill's plan maps straight back onto the co-pilot.

Recording and replay against a live browser are the co-pilot's job; this module
is the memory in between — record, generalise, store, and produce the concrete
plan — and all of that is pure and tested without a browser.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from .constants import CONFIG_DIR
from .utils import utc_stamp
from .atomic_io import atomic_write_text

SKILLS_PATH = CONFIG_DIR / "skills.json"

# The operation vocabulary — the same verbs browser_copilot speaks, so a plan is
# directly executable by it.
OPS = frozenset({"navigate", "click", "fill", "submit", "scroll", "read", "wait"})
# Ops whose typed value should be lifted into a parameter.
PARAM_OPS = frozenset({"fill"})


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return s or "value"


@dataclass
class RecordedAction:
    op: str
    target: str = ""          # url (navigate) / visible text (click) / field label (fill)
    value: str = ""           # the literal text typed (fill), if any
    note: str = ""


@dataclass
class SkillAction:
    op: str
    target: str = ""
    value: str = ""           # may be a "{param}" placeholder

    def as_dict(self) -> dict[str, Any]:
        return {"op": self.op, "target": self.target, "value": self.value}


@dataclass
class Skill:
    name: str
    actions: list[SkillAction]
    params: list[str] = field(default_factory=list)
    origin: str = ""          # the first navigated URL, for context
    created_at: str = field(default_factory=utc_stamp)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "origin": self.origin,
                "created_at": self.created_at,
                "actions": [a.as_dict() for a in self.actions]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        return cls(
            name=str(data.get("name") or ""),
            params=list(data.get("params") or []),
            origin=str(data.get("origin") or ""),
            created_at=str(data.get("created_at") or utc_stamp()),
            actions=[SkillAction(op=a.get("op", ""), target=a.get("target", ""),
                                 value=a.get("value", ""))
                     for a in (data.get("actions") or [])])

    def describe(self) -> str:
        head = [f"Skill '{self.name}'"
                + (f" (parameters: {', '.join(self.params)})" if self.params else "")
                + ":"]
        for i, a in enumerate(self.actions, 1):
            bit = a.op + (f" → {a.target}" if a.target else "")
            if a.value:
                bit += f" = {a.value}"
            head.append(f"  {i}. {bit}")
        return "\n".join(head)


# ── generalisation ───────────────────────────────────────────────────────────

def generalise(name: str, actions: Iterable[RecordedAction]) -> Skill:
    """Turn a raw recording into a parameterised skill.

    Literal values from PARAM_OPS become ``{param}`` placeholders named after
    the field they were typed into; identical field names get numbered so two
    different inputs never collide on one parameter.
    """
    skill_actions: list[SkillAction] = []
    params: list[str] = []
    used: dict[str, int] = {}
    origin = ""
    for raw in actions:
        op = str(raw.op or "").strip().lower()
        if op not in OPS:
            continue
        if op == "navigate" and not origin:
            origin = raw.target
        if op in PARAM_OPS and raw.value:
            base = _slug(raw.target or "value")
            used[base] = used.get(base, 0) + 1
            param = base if used[base] == 1 else f"{base}_{used[base]}"
            params.append(param)
            skill_actions.append(SkillAction(op=op, target=raw.target,
                                             value=f"{{{param}}}"))
        else:
            skill_actions.append(SkillAction(op=op, target=raw.target, value=raw.value))
    return Skill(name=_slug(name), actions=skill_actions, params=params, origin=origin)


# ── planning (params → concrete steps) ───────────────────────────────────────

_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


def plan(skill: Skill, params: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Substitute parameters into a skill and return (concrete steps, missing).

    ``missing`` lists any parameter the skill needs that wasn't supplied — the
    caller shows it rather than silently running with an empty field.
    """
    params = {str(k): str(v) for k, v in (params or {}).items()}
    steps: list[dict[str, Any]] = []
    missing: list[str] = []
    for a in skill.actions:
        value = a.value

        def _sub(match: "re.Match") -> str:
            key = match.group(1)
            if key in params:
                return params[key]
            missing.append(key)
            return match.group(0)

        value = _PLACEHOLDER.sub(_sub, value)
        steps.append({"op": a.op, "target": a.target, "value": value})
    # de-dupe missing, preserve order
    seen: set[str] = set()
    missing = [m for m in missing if not (m in seen or seen.add(m))]
    return steps, missing


# ── storage ──────────────────────────────────────────────────────────────────

class SkillStore:
    """A tiny JSON library of skills — private, local, human-readable."""

    def __init__(self, path: Any = SKILLS_PATH) -> None:
        self._lock = RLock()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> dict[str, dict]:
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}

    def _write_all(self, data: dict[str, dict]) -> None:
        with self._lock:
            atomic_write_text(self.path, json.dumps(data, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    def save(self, skill: Skill) -> None:
        data = self._load_all()
        data[skill.name] = skill.as_dict()
        self._write_all(data)

    def get(self, name: str) -> Skill | None:
        data = self._load_all().get(_slug(name))
        return Skill.from_dict(data) if data else None

    def list(self) -> list[Skill]:
        return [Skill.from_dict(d) for d in self._load_all().values()]

    def remove(self, name: str) -> bool:
        data = self._load_all()
        if _slug(name) in data:
            del data[_slug(name)]
            self._write_all(data)
            return True
        return False


# ── the engine ───────────────────────────────────────────────────────────────

class MuscleMemory:
    """Record a task, generalise it into a skill, replay it with new inputs."""

    def __init__(self, store: SkillStore | None = None) -> None:
        self.store = store or SkillStore()
        self._recording_name: str | None = None
        self._buffer: list[RecordedAction] = []

    # -- recording --
    def start_recording(self, name: str) -> str:
        self._recording_name = name
        self._buffer = []
        return f"Recording '{name}'. Do the task; I'll remember the steps."

    @property
    def recording(self) -> bool:
        return self._recording_name is not None

    def record(self, op: str, target: str = "", value: str = "", note: str = "") -> None:
        if self._recording_name is None:
            return
        self._buffer.append(RecordedAction(op=op, target=target, value=value, note=note))

    def stop_recording(self) -> Skill | None:
        if self._recording_name is None or not self._buffer:
            self._recording_name = None
            return None
        skill = generalise(self._recording_name, self._buffer)
        self.store.save(skill)
        self._recording_name = None
        self._buffer = []
        return skill

    # -- replay --
    def run(self, name: str, params: dict[str, Any] | None = None
            ) -> tuple[Skill | None, list[dict[str, Any]], list[str]]:
        skill = self.store.get(name)
        if skill is None:
            return None, [], []
        steps, missing = plan(skill, params)
        return skill, steps, missing

    def skills(self) -> list[Skill]:
        return self.store.list()

    def forget(self, name: str) -> bool:
        return self.store.remove(name)


__all__ = ["MuscleMemory", "Skill", "SkillAction", "RecordedAction", "SkillStore",
           "generalise", "plan", "OPS"]
