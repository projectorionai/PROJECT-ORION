"""
Translate arbitrary JSON Schema into the subset a Gemini function declaration
accepts — the single boundary every tool must pass before it reaches Live.

Why this exists (Sep 24 2026): the Notion MCP server's tools are generated from
an OpenAPI spec, so their input schemas carry ``$defs``/``$ref``, ``oneOf``,
``const``, ``additionalProperties`` and ``exclusiveMinimum``.  The MCP bridge
copied them through untouched, and ``types.LiveConnectConfig`` forbids every
key it does not know — so the *whole* connect failed with "35 validation
errors" before a socket was opened.  MCP servers register a few seconds after
boot, which is why the FIRST Live session usually worked and every reconnect
after it (Gemini ends each connection after ~10 minutes, ~2 with camera
frames) failed forever: "the live voice channel is unstable at the moment".

The output uses only the keywords ORION's own 139 built-in declarations use —
the subset the Live server is proven to accept: ``type``, ``description``,
``properties``, ``required``, ``items``, plus ``enum`` (strings) and
``nullable``.  Everything else is either translated (``$ref`` inlined,
``oneOf``/``anyOf`` collapsed, ``const`` -> single-value enum) or folded into
the description as plain words, so the model still learns the constraint.

Deep or self-referencing structures (a Notion block can contain blocks) are
cut at ``MAX_DEPTH`` into a JSON-encoded STRING; :func:`decode_json_args`
turns such strings back into objects before the call reaches the server.

Pure: no I/O, no SDK import, never raises on malformed input.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Gemini function names: letter/underscore first, then [A-Za-z0-9_.-], <= 64.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
_TYPES = {
    "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}
# Levels of OBJECT/ARRAY nesting kept below the parameters object. Built-in
# tools reach 3; anything deeper becomes a JSON string (see module doc).
MAX_DEPTH = 4
DESCRIPTION_LIMIT = 1024
JSON_STRING_NOTE = "Give this as a JSON-encoded string."


def valid_function_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_NAME_RE.match(name))


def _gemini_type(raw: Any) -> tuple[str | None, bool]:
    """(Gemini type or None, nullable) for a JSON-Schema ``type`` value."""
    if isinstance(raw, list):
        nullable = any(str(t).lower() == "null" for t in raw)
        rest = [t for t in raw if str(t).lower() != "null"]
        if not rest:
            return None, nullable
        if len(rest) > 1:
            return "STRING", nullable      # "string or number": a string covers both
        return _TYPES.get(str(rest[0]).lower()), nullable
    if raw is None:
        return None, False
    text = str(raw).lower()
    if text == "null":
        return None, True
    return _TYPES.get(text), False


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Follow a local JSON pointer ("#/$defs/X", "#/definitions/X", "#/a/b")."""
    if not ref.startswith("#"):
        return None
    node: Any = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _deref(node: Any, root: dict[str, Any], seen: frozenset[str]
           ) -> tuple[dict[str, Any], frozenset[str], bool]:
    """Inline ``$ref`` chains. Returns (node, seen, cyclic)."""
    if not isinstance(node, dict):
        return {}, seen, False
    hops = 0
    while isinstance(node.get("$ref"), str) and hops < 16:
        ref = node["$ref"]
        if ref in seen:
            return node, seen, True
        target = _resolve_ref(ref, root)
        seen = seen | {ref}
        hops += 1
        if target is None:
            rest = {k: v for k, v in node.items() if k != "$ref"}
            return rest or {"type": "string"}, seen, False
        # Sibling keywords (a description beside the $ref) win over the target's.
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    return node, seen, False


def _clip(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= DESCRIPTION_LIMIT else text[: DESCRIPTION_LIMIT - 1] + "…"


def _describe(node: dict[str, Any], notes: list[str]) -> str:
    base = str(node.get("description") or node.get("title") or "").strip()
    extra = " ".join(n for n in notes if n)
    if base and extra and base[-1] not in ".!?:":
        base += "."
    return _clip(f"{base} {extra}".strip())


def _is_objectish(node: dict[str, Any]) -> bool:
    kind, _ = _gemini_type(node.get("type"))
    return kind == "OBJECT" or (kind is None and isinstance(node.get("properties"), dict))


def _is_plain_string(node: dict[str, Any]) -> bool:
    kind, _ = _gemini_type(node.get("type"))
    return kind == "STRING" and "enum" not in node and "const" not in node


def _is_free_form(node: dict[str, Any]) -> bool:
    """An object that names no fields but allows any (``additionalProperties``
    true or a schema). Gemini's function calling cannot fill one, so it
    travels as JSON text. ORION's own declarations never set the keyword,
    so they are untouched by this."""
    return (_is_objectish(node) and not node.get("properties")
            and bool(node.get("additionalProperties")))


def _field_hint(shapes: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    required: list[str] = []
    for shape in shapes:
        for key in (shape.get("properties") or {}) if isinstance(shape.get("properties"), dict) else {}:
            if key not in fields:
                fields.append(str(key))
        for key in shape.get("required") or []:
            if key not in required:
                required.append(str(key))
    if not fields:
        return ""
    hint = "Fields: " + ", ".join(fields[:16]) + ("…" if len(fields) > 16 else "") + "."
    if required:
        hint += " Required: " + ", ".join(required[:8]) + "."
    return hint


def _json_string(node: dict[str, Any], notes: list[str],
                 shapes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    hint = _field_hint(shapes if shapes is not None else [node])
    return {"type": "STRING",
            "description": _describe(node, [*notes, hint, JSON_STRING_NOTE])}


def _flatten_variants(variants: list[Any], root: dict[str, Any], seen: frozenset[str],
                      depth: int = 0) -> list[dict[str, Any]] | None:
    """Resolve each alternative, splicing in nested oneOf/anyOf lists (Notion's
    ``parent`` is anyOf[$ref -> oneOf[...], string]). None on a cycle."""
    out: list[dict[str, Any]] = []
    for variant in variants:
        v, v_seen, cyclic = _deref(variant, root, seen)
        if cyclic:
            return None
        nested = v.get("oneOf") or v.get("anyOf")
        if isinstance(nested, list) and nested and not v.get("properties") and depth < 4:
            inner = _flatten_variants(nested, root, v_seen, depth + 1)
            if inner is None:
                return None
            out.extend(inner)
        else:
            out.append(v)
    return out


def _merge_objects(variants: list[dict[str, Any]], alternatives: bool
                   ) -> dict[str, Any]:
    """Union of several object shapes: every property once, and ``required``
    only where EVERY alternative requires it (oneOf/anyOf) or where ANY part
    does (allOf). Differing ``const`` values on the
    same key (a discriminator like ``type``) become one enum."""
    properties: dict[str, Any] = {}
    consts: dict[str, list[str]] = {}
    required_sets: list[set[str]] = []
    for variant in variants:
        props = variant.get("properties") if isinstance(variant.get("properties"), dict) else {}
        for key, sub in props.items():
            if isinstance(sub, dict) and "const" in sub:
                consts.setdefault(key, [])
                value = str(sub["const"])
                if value not in consts[key]:
                    consts[key].append(value)
            properties.setdefault(key, sub)
        req = variant.get("required")
        required_sets.append(set(req) if isinstance(req, list) else set())
    for key, values in consts.items():
        if len(values) > 1:
            original = properties[key] if isinstance(properties[key], dict) else {}
            properties[key] = {"type": "string", "enum": values,
                               "description": original.get("description", "")}
    required = set.intersection(*required_sets) if required_sets and alternatives else (
        set.union(*required_sets) if required_sets else set())
    merged: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        merged["required"] = sorted(required)
    return merged


def _convert(node: Any, root: dict[str, Any], depth: int, seen: frozenset[str]
             ) -> dict[str, Any]:
    node, seen, cyclic = _deref(node, root, seen)
    notes: list[str] = []
    if cyclic:
        return _json_string(node, notes)

    # ── combinators: collapse to one shape ────────────────────────────────
    nullable = False
    for key in ("oneOf", "anyOf"):
        variants = node.get(key)
        if not isinstance(variants, list) or not variants:
            continue
        resolved = _flatten_variants(variants, root, seen)
        if resolved is None:
            return _json_string(node, notes)
        concrete = []
        for v in resolved:
            kind, v_null = _gemini_type(v.get("type"))
            if v.get("type") is not None and kind is None and v_null:
                nullable = True             # a bare {"type": "null"} branch
                continue
            concrete.append(v)
        outer = {k: v for k, v in node.items() if k not in ("oneOf", "anyOf")}
        keep = {k: v for k, v in outer.items() if k in ("description", "title")}
        objects = [v for v in concrete if _is_objectish(v)]
        shaped = [v for v in objects if v.get("properties")]
        if not concrete:
            node = {**outer, "type": outer.get("type", "string")}
        elif len(concrete) == 1:
            node = {**concrete[0], **{k: v for k, v in outer.items() if k != "type"}}
        elif shaped and all(_is_objectish(v) or _is_plain_string(v) for v in concrete):
            # One or more real object shapes, perhaps also "or the same thing
            # as a JSON string" (Notion): keep the structure — the server
            # takes the object form too, and the model fills it far better.
            merged = _merge_objects(objects, alternatives=True)
            if len(shaped) > 1:
                notes.append("Several shapes are accepted; include only the "
                             "fields for the one you mean.")
            node = {**merged, **keep}
        elif all("const" in v or "enum" in v for v in concrete):
            values: list[str] = []
            for v in concrete:
                for item in ([v["const"]] if "const" in v else list(v.get("enum") or [])):
                    if str(item) not in values:
                        values.append(str(item))
            node = {**outer, "type": "string", "enum": values}
        else:
            # Genuinely mixed (text OR a free-form object, say): ask for JSON
            # text, which decode_json_args turns back into the right thing.
            base = {**concrete[0], **keep}
            return _json_string(base, notes, shapes=objects)
        break

    all_of = node.get("allOf")
    if isinstance(all_of, list) and all_of:
        parts = []
        for part in all_of:
            p, _, p_cyclic = _deref(part, root, seen)
            if p_cyclic:
                return _json_string(node, notes)
            parts.append(p)
        merged = _merge_objects([p for p in parts if _is_objectish(p)] or [{}], alternatives=False)
        node = {**{k: v for k, v in node.items() if k != "allOf"}, **merged}

    kind, type_null = _gemini_type(node.get("type"))
    nullable = nullable or type_null or node.get("nullable") is True
    if kind is None:
        if isinstance(node.get("properties"), dict):
            kind = "OBJECT"
        elif "items" in node:
            kind = "ARRAY"
        elif "const" in node or "enum" in node:
            kind = "STRING"
        else:
            kind = "STRING"
            if node.get("type") is None and not node.get("description"):
                notes.append("Any value.")

    # Constraints the subset cannot express become words the model reads.
    fmt = node.get("format")
    if isinstance(fmt, str) and fmt:
        notes.append(f"Format: {fmt}.")
    for key, label in (("minimum", "Minimum"), ("exclusiveMinimum", "Greater than"),
                       ("maximum", "Maximum"), ("exclusiveMaximum", "Less than"),
                       ("minLength", "Min length"), ("maxLength", "Max length"),
                       ("minItems", "Min items"), ("maxItems", "Max items")):
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            notes.append(f"{label} {value}.")
    if "default" in node and isinstance(node["default"], (str, int, float, bool)):
        notes.append(f"Default {json.dumps(node['default'])}.")

    out: dict[str, Any] = {"type": kind}

    if "const" in node:
        if kind == "STRING":
            out["enum"] = [str(node["const"])]
        else:
            notes.append(f"Must be {json.dumps(node['const'])}.")
    elif isinstance(node.get("enum"), list) and node["enum"]:
        values = [v for v in node["enum"] if v is not None]
        if any(v is None for v in node["enum"]):
            nullable = True
        if kind == "STRING" and values:
            out["enum"] = list(dict.fromkeys(str(v) for v in values))
        elif values:
            notes.append("One of: " + ", ".join(json.dumps(v) for v in values[:20]) + ".")

    if kind in ("OBJECT", "ARRAY") and depth >= MAX_DEPTH:
        return _json_string(node, notes)
    if kind == "OBJECT" and depth > 0 and _is_free_form(node):
        return _json_string(node, notes)

    if kind == "OBJECT":
        props = node.get("properties")
        if isinstance(props, dict):
            # Only when the source had them: an OBJECT without properties is
            # left exactly as ORION's own declarations write it.
            out["properties"] = {
                str(name): _convert(sub, root, depth + 1, seen)
                for name, sub in props.items() if isinstance(name, str)
            }
        required = node.get("required")
        if isinstance(required, list) and out.get("properties"):
            kept = [r for r in required if r in out["properties"]]
            if kept:
                out["required"] = list(dict.fromkeys(kept))
    elif kind == "ARRAY":
        items = node.get("items")
        if isinstance(items, list):          # draft-4 tuple form
            items = items[0] if items else {}
        out["items"] = _convert(items if isinstance(items, dict) else {},
                                root, depth + 1, seen)

    description = _describe(node, notes)
    if description:
        out["description"] = description
    if nullable:
        out["nullable"] = True
    return out


def to_gemini_parameters(schema: Any) -> dict[str, Any]:
    """Translate one tool's input schema. Always returns an OBJECT."""
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}
    try:
        converted = _convert(schema, schema, 0, frozenset())
    except (RecursionError, TypeError, ValueError):
        return {"type": "OBJECT", "properties": {}}
    if converted.get("type") != "OBJECT":
        return {"type": "OBJECT", "properties": {}}
    converted.setdefault("properties", {})
    converted.pop("nullable", None)
    return converted


def sanitise_declaration(declaration: Any) -> dict[str, Any] | None:
    """One declaration in the Live-safe shape, or None when it cannot be
    (no valid name). Idempotent on ORION's built-in declarations."""
    if not isinstance(declaration, dict):
        return None
    name = declaration.get("name")
    if not valid_function_name(name):
        return None
    out: dict[str, Any] = {
        "name": name,
        "description": str(declaration.get("description") or name),
    }
    params = declaration.get("parameters")
    if params is None:
        params = declaration.get("parameters_json_schema") or declaration.get("input_schema")
    out["parameters"] = to_gemini_parameters(params if params is not None
                                             else {"type": "OBJECT", "properties": {}})
    return out


def sanitise_declarations(declarations: list[Any]
                          ) -> tuple[list[dict[str, Any]], list[str]]:
    """(clean declarations, names dropped). Duplicate names keep the first —
    the server rejects a repeated function name outright."""
    clean: list[dict[str, Any]] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for declaration in declarations or []:
        result = sanitise_declaration(declaration)
        name = declaration.get("name") if isinstance(declaration, dict) else None
        if result is None or result["name"] in seen:
            dropped.append(str(name or "?"))
            continue
        seen.add(result["name"])
        clean.append(result)
    return clean, dropped


# ── the way back: JSON strings -> objects for the real server ──────────────

def _expects_structure(node: Any, root: dict[str, Any], seen: frozenset[str]) -> bool:
    node, seen, cyclic = _deref(node, root, seen)
    if not isinstance(node, dict):
        return False
    kind, _ = _gemini_type(node.get("type"))
    if kind in ("OBJECT", "ARRAY"):
        return True
    if kind is None and (isinstance(node.get("properties"), dict) or "items" in node):
        return True
    for key in ("oneOf", "anyOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list) and not cyclic:
            if any(_expects_structure(v, root, seen) for v in variants):
                return True
    return False


def _accepts_string(node: Any, root: dict[str, Any], seen: frozenset[str]) -> bool:
    node, seen, _ = _deref(node, root, seen)
    if not isinstance(node, dict):
        return False
    raw = node.get("type")
    kinds = raw if isinstance(raw, list) else [raw]
    if any(str(k).lower() == "string" for k in kinds if k is not None):
        return True
    for key in ("oneOf", "anyOf"):
        variants = node.get(key)
        if isinstance(variants, list) and any(_accepts_string(v, root, seen) for v in variants):
            return True
    return False


def _sub_schema(node: Any, key: str | None, root: dict[str, Any], seen: frozenset[str]
                ) -> Any:
    """The schema for property *key* (or array items when key is None),
    looking through $ref and combinators."""
    node, seen, _ = _deref(node, root, seen)
    if not isinstance(node, dict):
        return None
    if key is None:
        items = node.get("items")
        if isinstance(items, dict):
            return items
    else:
        props = node.get("properties")
        if isinstance(props, dict) and key in props:
            return props[key]
    for combinator in ("oneOf", "anyOf", "allOf"):
        for variant in node.get(combinator) or []:
            found = _sub_schema(variant, key, root, seen)
            if found is not None:
                return found
    return None


def decode_json_args(args: Any, schema: Any) -> Any:
    """Undo the JSON-string collapse: wherever the ORIGINAL schema wants an
    object/array (and does not also accept a plain string) and the model sent
    a string, parse it. Anything that does not parse is passed through for
    the server to judge."""
    if not isinstance(schema, dict):
        return args
    return _decode(args, schema, schema, frozenset(), 0)


def _decode(value: Any, node: Any, root: dict[str, Any], seen: frozenset[str],
            depth: int) -> Any:
    if depth > 32 or node is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        if (text[:1] in "{[" and _expects_structure(node, root, seen)
                and not _accepts_string(node, root, seen)):
            try:
                parsed = json.loads(text)
            except ValueError:
                return value
            return _decode(parsed, node, root, seen, depth + 1)
        return value
    if isinstance(value, dict):
        return {k: _decode(v, _sub_schema(node, k, root, seen), root, seen, depth + 1)
                for k, v in value.items()}
    if isinstance(value, list):
        item_schema = _sub_schema(node, None, root, seen)
        return [_decode(v, item_schema, root, seen, depth + 1) for v in value]
    return value
