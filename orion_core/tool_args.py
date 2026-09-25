"""
Bring model-supplied tool arguments into line with the types the schema declares.

Handlers read their arguments with plain Python conversions — ``int(args.get(
"limit") or 10)``, ``bool(args.get("confirm"))`` — which is correct only while
every value arrives with the JSON type the schema asked for. Gemini Live does
send typed values. The other two routes into the dispatcher do not have to:
tool calls parsed out of a local model's TEXT (``agents.py``) and calls from
the phone/remote gateway (``remote.py``). Small local models routinely quote
values. The audit of 2026-09-23 found:

* **``bool("false")`` is True.** 23 handlers read a declared BOOLEAN with
  ``bool(args.get(...))``, among them ``confirm`` on deletes and web actions,
  ``consent``, and ``submit`` on form filling. A quoted ``"false"`` CONFIRMED.
* **93 unguarded ``int()``/``float()``** on numeric arguments, so ``null`` or
  ``"5"`` from a local model failed the tool with a raw Python TypeError.

Rather than patch 116 call sites, every native tool call is normalised once,
here, against the schema the model was given:

* ``null`` means "not supplied": the key is dropped, so the handler's own
  default applies exactly as if the model had omitted it.
* A numeric string becomes a number; an INTEGER that is whole becomes an int.
* A string that is NOT a number is passed through untouched. Several handlers
  deliberately fall back to a default on junk (``camera_index="oops"`` -> 0,
  pinned by test_camera_vision), and that is theirs to decide.
* ``"true"/"yes"/"on"/"1"`` and ``"false"/"no"/"off"/"0"`` become booleans,
  and any other string for a boolean is REFUSED with a clear message rather
  than guessed. That asymmetry is deliberate: an unreadable ``confirm`` must
  not pass a confirmation gate, and the old behaviour read every string as yes.

Values already of the right type pass through untouched, and parameters the
schema does not describe are never changed.
"""

from __future__ import annotations

import math
from typing import Any

_TRUE = frozenset({"true", "yes", "y", "on", "1"})
_FALSE = frozenset({"false", "no", "n", "off", "0", ""})


def declared_types(declaration: dict[str, Any] | None) -> dict[str, str]:
    """``{param: TYPE}`` from a function declaration, upper-cased."""
    parameters = (declaration or {}).get("parameters") or {}
    properties = parameters.get("properties") or {}
    return {str(name): str((spec or {}).get("type") or "").upper()
            for name, spec in properties.items()}


def _as_number(value: Any, whole: bool) -> Any:
    """A number when *value* reads as one; otherwise *value* unchanged."""
    if isinstance(value, float) and value.is_integer() and whole:
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("_", "")
        try:
            parsed = float(text)
        except ValueError:
            return value             # not a number: the handler decides, as before
        if not math.isfinite(parsed):
            return value             # "nan"/"inf" are words here, not numbers
        return int(parsed) if whole and parsed.is_integer() else parsed
    # bools, ints, other floats, lists, dicts: left exactly as they were.
    return value


def _as_bool(value: Any) -> tuple[bool, Any]:
    if isinstance(value, (bool, int, float)):
        return True, value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE:
            return True, True
        if word in _FALSE:
            return True, False
        return False, None
    return True, value


def coerce_args(args: dict[str, Any], declaration: dict[str, Any] | None
                ) -> tuple[dict[str, Any], str | None]:
    """Return ``(normalised_args, None)`` or ``(args, error)``. Never mutates."""
    types = declared_types(declaration)
    if not types or not args:
        return dict(args or {}), None
    out: dict[str, Any] = {}
    for key, value in args.items():
        kind = types.get(key)
        if kind in {"INTEGER", "NUMBER", "BOOLEAN"} and value is None:
            continue                           # null == not supplied
        if kind in {"INTEGER", "NUMBER"}:
            out[key] = _as_number(value, whole=kind == "INTEGER")
        elif kind == "BOOLEAN":
            ok, parsed = _as_bool(value)
            if not ok:
                return dict(args), f"'{key}' must be true or false, but got {value!r}."
            out[key] = parsed
        else:
            out[key] = value
    return out, None


__all__ = ["coerce_args", "declared_types"]
