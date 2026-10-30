"""
agent_registry.py — declarative specialist agents (Mark XXI, Track E3).

The nine built-in specialists (agents.py) are hardcoded Python classes;
adding a new one meant editing that file and AgentManager's construction
tuple. This lets a new specialist be added with a single
config/agents/<name>.json manifest instead — no code change needed for it
to be real, routable, and dispatchable (agent_dispatch, the `reason` tool,
AgentManager.describe() all see it exactly like a built-in). A dedicated
Command Deck workspace PAGE for a declarative agent is not generated
automatically — app.py's page list is still a fixed tuple — so a new
manifest agent is reachable by name/auto-routing today, not yet visually
represented as its own page (a natural follow-up, out of this pass).

A manifest is DATA, not code — name/title/expertise/regex routing signals/
persona text and an optional work_contract — matching this codebase's existing
"skills are data, not code" boundary (skills.py makes the same distinction).
build_agent() constructs a real BaseAgent subclass at load time from that
data; nothing here ever executes manifest content as Python.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agents import AgentWorkContract, BaseAgent
from .constants import CONFIG_DIR

AGENTS_DIR = CONFIG_DIR / "agents"

_REQUIRED_STRING_FIELDS = ("name", "title", "expertise", "persona")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_regex_list(value: Any, field: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"'{field}' must be a list of strings")
        return []
    out: list[str] = []
    for pattern in value:
        if not isinstance(pattern, str):
            errors.append(f"'{field}' entries must be strings")
            continue
        try:
            re.compile(pattern)
        except re.error as exc:
            errors.append(f"'{field}' entry {pattern!r} is not a valid regex: {exc}")
            continue
        out.append(pattern)
    return out


def manifest_errors(data: Any, label: str = "manifest") -> list[str]:
    """Validate file and direct-registration inputs through the same boundary.

    Work contracts describe the requested contribution. They never grant tool
    permissions; their optional numeric limits can only reduce runtime ceilings.
    Existing manifests without a contract keep the default specialist contract.
    """
    if not isinstance(data, dict):
        return [f"{label}: manifest must be a JSON object"]
    errors: list[str] = []
    for field in _REQUIRED_STRING_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: required field '{field}' must be a non-empty string")
    name = data.get("name")
    if isinstance(name, str) and name and not _NAME_RE.fullmatch(name):
        errors.append(f"{label}: 'name' must be lowercase snake_case, got {name!r}")
    _validate_regex_list(data.get("primary"), "primary", errors)
    _validate_regex_list(data.get("keywords"), "keywords", errors)
    if "work_contract" not in data:
        return errors
    contract = data["work_contract"]
    if not isinstance(contract, dict):
        errors.append(f"{label}: 'work_contract' must be an object")
        return errors
    allowed = {"purpose", "inputs", "deliverable", "acceptance", "owner",
               "cancellation", "limits"}
    for field in contract.keys() - allowed:
        errors.append(f"{label}: unknown work_contract field {field!r}")
    for field in ("purpose", "deliverable", "owner"):
        value = contract.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: work_contract.{field} must be a non-empty string")
    for field in ("inputs", "acceptance"):
        value = contract.get(field)
        if (not isinstance(value, list) or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)):
            errors.append(f"{label}: work_contract.{field} must be a non-empty list of strings")
    if contract.get("cancellation", "propagate") != "propagate":
        errors.append(f"{label}: work_contract.cancellation must be 'propagate'")
    limits = contract.get("limits", {})
    if not isinstance(limits, dict):
        errors.append(f"{label}: work_contract.limits must be an object")
    else:
        for field, value in limits.items():
            if field not in {"max_steps", "max_readings"}:
                errors.append(f"{label}: unknown work_contract.limits field {field!r}")
                continue
            minimum = 1 if field == "max_steps" else 0
            if type(value) is not int or value < minimum:
                errors.append(
                    f"{label}: work_contract.limits.{field} must be an integer >= {minimum}")
    return errors


def load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Read and validate one manifest file. Returns (manifest, errors) —
    manifest is None whenever validation failed. Never raises: a malformed
    file is reported, not a crash."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"could not read/parse {path.name}: {exc}"]
    errors = manifest_errors(data, path.name)
    if errors:
        return None, errors
    return data, []


def build_agent(manifest: dict[str, Any], router: Any, bus: Any) -> BaseAgent:
    """A real BaseAgent subclass built dynamically from *manifest* — the
    exact same scoring/handle/investigate machinery every built-in
    specialist uses; none of it is duplicated here."""
    errors = manifest_errors(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    name = str(manifest["name"])
    attrs = {
        "name": name,
        "title": str(manifest["title"]),
        "expertise": str(manifest["expertise"]),
        "primary": tuple(str(p) for p in (manifest.get("primary") or [])),
        "keywords": tuple(str(k) for k in (manifest.get("keywords") or [])),
        "persona": str(manifest["persona"]),
    }
    contract = manifest.get("work_contract")
    if contract is not None:
        limits = contract.get("limits", {})
        attrs["work_contract"] = AgentWorkContract(
            purpose=contract["purpose"], inputs=tuple(contract["inputs"]),
            deliverable=contract["deliverable"], acceptance=tuple(contract["acceptance"]),
            owner=contract["owner"], max_steps=limits.get("max_steps"),
            max_readings=limits.get("max_readings"),
        )
    cls = type(f"Declarative_{name.title().replace('_', '')}Agent", (BaseAgent,), attrs)
    return cls(router, bus)


def load_all(bus: Any | None = None) -> list[dict[str, Any]]:
    """Every valid manifest under config/agents/*.json, in filename order.
    An invalid manifest is logged (when a bus is given) and skipped — one
    bad file must never stop the others, or the app, from loading."""
    manifests: list[dict[str, Any]] = []
    if not AGENTS_DIR.is_dir():
        return manifests
    for path in sorted(AGENTS_DIR.glob("*.json")):
        manifest, errors = load_manifest(path)
        if manifest is None:
            if bus is not None:
                try:
                    bus.log.emit(
                        f"AGENTS: skipped invalid manifest {path.name} — {'; '.join(errors)}")
                except Exception:
                    pass
            continue
        manifests.append(manifest)
    return manifests
