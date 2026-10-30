"""
Is that JSON sound? - the json_validator plugin, rebuilt into ORION (Mark XXXIX).

The plugin ("Validates the system's configuration files and JSON data") never
loaded: it imported ``jsonschema``, which is not installed, so the forge
quarantined it on 20 Sep 2026 and it sat on the Plugins page as one of the
"5 rejected". The job is worth doing - a settings file truncated by a crash
or a hand edit with a stray comma silently becomes "defaults" in most of
ORION's stores - so it is built in, with no new dependency:

  * ``check_config()`` - every ``*.json`` ORION keeps in CONFIG_DIR (and one
    level of folders under it), parsed: which are broken, where, and which are
    empty. Read-only; caches and large files are skipped.
  * ``validate(data, schema)`` - JSON Schema checking for the common keywords
    (type, required, properties, additionalProperties, items, enum, const,
    minimum/maximum, minLength/maxLength, minItems/maxItems, pattern) - and
    ``jsonschema`` itself when it happens to be importable.
  * ``check_text(text, schema_text)`` - what the diagnostics tool calls.

Blocking file I/O: callers run it off the event loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

#: Folders under CONFIG_DIR that hold caches, models or someone else's files.
SKIP_DIRS = frozenset({"cache", "voice_cache", "models", "browser_profile", "knowledge_corpus",
                       "ingest_cache", "_migration_backups", "custom_tools", "crash_reports",
                       "exports", "brain", "pipeline", "community"})
#: A settings file is small; anything bigger is data, not configuration.
MAX_BYTES = 2_000_000
MAX_ERRORS = 12
#: Built-in 'pattern' checks only run on short patterns and short strings.
MAX_PATTERN = 200
MAX_PATTERN_INPUT = 2_000

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def _is_type(value: Any, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _TYPES.get(kind)
    return expected is not None and isinstance(value, expected)


def _builtin_errors(value: Any, schema: Any, path: str, out: list[str], depth: int = 0) -> None:
    if len(out) >= MAX_ERRORS or not isinstance(schema, dict) or depth > 32:
        return
    where = path or "the document"
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        if not any(_is_type(value, str(k)) for k in kinds):
            out.append(f"{where} should be {' or '.join(map(str, kinds))}")
            return
    if "enum" in schema and value not in list(schema.get("enum") or []):
        out.append(f"{where} must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        out.append(f"{where} must be {schema['const']!r}")
    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{where} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{where} is above the maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            out.append(f"{where} is shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            out.append(f"{where} is longer than {schema['maxLength']} characters")
        pattern = schema.get("pattern")
        # Python's re has no timeout: a pathological pattern from a schema the
        # model was handed must not pin a worker on a huge value.
        if isinstance(pattern, str) and len(pattern) <= MAX_PATTERN and len(value) <= MAX_PATTERN_INPUT:
            try:
                if re.search(pattern, value) is None:
                    out.append(f"{where} does not match the pattern {pattern}")
            except re.error:
                out.append(f"the schema's pattern for {where} is not a valid expression")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            out.append(f"{where} needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            out.append(f"{where} allows at most {schema['maxItems']} items")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _builtin_errors(item, items, f"{path}[{index}]", out, depth + 1)
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                out.append(f"{where} is missing '{key}'")
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key, sub in properties.items():
            if key in value:
                _builtin_errors(value[key], sub, f"{path}.{key}" if path else key, out, depth + 1)
        extra = schema.get("additionalProperties", True)
        for key in value:
            if key in properties:
                continue
            if extra is False:
                out.append(f"{where} has an unexpected key '{key}'")
            elif isinstance(extra, dict):
                _builtin_errors(value[key], extra, f"{path}.{key}" if path else key, out, depth + 1)


def validate(value: Any, schema: Any) -> list[str]:
    """Plain-English problems with *value* against *schema* ([] = valid)."""
    if not isinstance(schema, dict):
        return ["the schema must be a JSON object"]
    try:
        import jsonschema  # type: ignore[import-not-found]
    except Exception:
        jsonschema = None
    if jsonschema is not None:
        try:
            validator = jsonschema.validators.validator_for(schema)(schema)
            return [f"{'.'.join(map(str, e.absolute_path)) or 'the document'}: {e.message}"
                    for e in list(validator.iter_errors(value))[:MAX_ERRORS]]
        except Exception as exc:                     # a broken schema
            return [f"the schema itself is invalid: {str(exc).splitlines()[0][:160]}"]
    errors: list[str] = []
    _builtin_errors(value, schema, "", errors)
    return errors


def _parse_error(exc: json.JSONDecodeError) -> str:
    return f"line {exc.lineno}, column {exc.colno}: {exc.msg}"


def check_text(text: str, schema_text: str = "") -> tuple[str, bool]:
    """Check *text* is JSON (and fits *schema_text*, when given)."""
    try:
        value = json.loads(str(text or ""))
    except json.JSONDecodeError as exc:
        return f"That isn't valid JSON - {_parse_error(exc)}.", False
    if not str(schema_text or "").strip():
        return "That's valid JSON.", True
    try:
        schema = json.loads(str(schema_text))
    except json.JSONDecodeError as exc:
        return f"The schema isn't valid JSON - {_parse_error(exc)}.", False
    problems = validate(value, schema)
    if not problems:
        return "Valid JSON, and it matches the schema.", True
    return ("Valid JSON, but it doesn't match the schema: " + "; ".join(problems) + "."), False


def _config_files(root: Path) -> list[Path]:
    files: list[Path] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return files
    for entry in entries:
        try:
            if entry.is_file() and entry.suffix.lower() == ".json":
                files.append(entry)
            elif entry.is_dir() and entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                files.extend(sorted(p for p in entry.glob("*.json") if p.is_file()))
        except OSError:
            continue
    return files


def check_config(root: Path | None = None) -> tuple[str, bool]:
    """Parse every settings file ORION keeps. (report, all sound)."""
    base = Path(root or CONFIG_DIR)
    broken: list[str] = []
    empty: list[str] = []
    checked = 0
    for path in _config_files(base):
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            raw = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            broken.append(f"{path.relative_to(base)} (unreadable: {type(exc).__name__})")
            continue
        checked += 1
        if not raw.strip():
            empty.append(str(path.relative_to(base)))
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            broken.append(f"{path.relative_to(base)} ({_parse_error(exc)})")
    if checked == 0 and not broken:
        return "I found no settings files to check.", True
    parts = [f"Checked {checked} settings files."]
    if broken:
        parts.append(f"{len(broken)} broken: " + "; ".join(broken[:MAX_ERRORS]) + ".")
    if empty:
        parts.append(f"{len(empty)} empty: " + ", ".join(empty[:MAX_ERRORS]) + ".")
    if not broken and not empty:
        parts.append("All of them are sound.")
    return " ".join(parts), not broken


__all__ = ["check_config", "check_text", "validate"]
