"""
The tool contract, in one place.

Two very different callers need to answer the same question — "does this module
honour the tool contract?":

  * the sandbox's conformance probe, which runs as a standalone script inside
    the staging directory and cannot import ``orion_core`` (that isolation is
    deliberate: generated code must not be able to reach ORION's own package);
  * ``ReflectiveModuleLoader``, which has already imported the module in-process
    and must refuse to activate a tool whose schema and ``run()`` disagree.

Duplicating the rules in both places would guarantee drift, so this module is
the single implementation and the sandbox ships it into the staging directory
**as text** (see ``sandbox.probe_source``). That means it must stay standalone:
no imports from ``orion_core``, no relative imports, standard library only.

The rules exist because of a specific class of silent failure. A tool could pass
verification with exit code 0 and still explode the first time the dispatcher
called it, because nothing ever compared the advertised schema against the
function that would actually receive those arguments.
"""

from __future__ import annotations

import inspect
from typing import Any


def schema_problems(schema: Any) -> tuple[list[str], dict]:
    """Validate the schema's shape.

    Returns ``(problems, properties)`` — the properties mapping is returned even
    when there are problems, so signature checking can still run and the caller
    gets the complete picture in one pass rather than one error at a time.
    """
    problems: list[str] = []
    properties: dict = {}

    if not isinstance(schema, dict):
        return ([f"get_tool_schema() must return a dict, got "
                 f"{type(schema).__name__}"], properties)

    if not str(schema.get("name") or "").strip():
        problems.append("the schema needs a non-empty 'name'")
    if not str(schema.get("description") or "").strip():
        problems.append("the schema needs a non-empty 'description'")

    params = schema.get("parameters")
    if params is None:
        problems.append("the schema needs a 'parameters' object")
        return problems, properties
    if not isinstance(params, dict):
        problems.append("schema 'parameters' must be an object")
        return problems, properties

    if params.get("type") not in (None, "object"):
        problems.append("schema parameters 'type' must be 'object'")

    raw_properties = params.get("properties", {})
    if not isinstance(raw_properties, dict):
        problems.append("schema parameters 'properties' must be an object")
    else:
        properties = raw_properties

    required = params.get("required", [])
    if not isinstance(required, (list, tuple)):
        problems.append("schema parameters 'required' must be a list")
    else:
        for key in required:
            if key not in properties:
                problems.append(
                    "'%s' is listed as required but is not in properties" % key)

    return problems, properties


def signature_problems(run_fn: Any, properties: dict) -> list[str]:
    """Check that ``run()`` can actually be called with the schema's parameters.

    The dispatcher always invokes a forged tool as ``run(**kwargs)``, so:
      * a positional-only parameter can never be supplied;
      * a parameter with no default that the schema does not declare will always
        be missing;
      * a declared property that ``run()`` does not accept raises TypeError —
        unless ``run()`` takes ``**kwargs``, which absorbs anything.
    """
    problems: list[str] = []
    if not callable(run_fn):
        return ["the module must export a callable run(**kwargs)"]

    try:
        signature = inspect.signature(run_fn)
    except (TypeError, ValueError):
        return problems      # builtins and C callables cannot be introspected

    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    named = set()

    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            problems.append(
                "run() parameter '%s' is positional-only, but the dispatcher "
                "calls run(**kwargs)" % name)
        elif parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                inspect.Parameter.KEYWORD_ONLY):
            named.add(name)
            if parameter.default is inspect.Parameter.empty and name not in properties:
                problems.append(
                    "run() requires argument '%s', but the schema does not "
                    "declare it in properties" % name)

    if not accepts_kwargs:
        for key in properties:
            if key not in named:
                problems.append(
                    "the schema declares parameter '%s', but run() does not "
                    "accept it (add it, or accept **kwargs)" % key)

    return problems


def contract_problems(schema: Any, run_fn: Any) -> list[str]:
    """Every way *schema* and *run_fn* fail the tool contract, as plain English.

    An empty list means the tool is safe for the dispatcher to call.
    """
    problems, properties = schema_problems(schema)
    problems.extend(signature_problems(run_fn, properties))
    return problems


__all__ = ["contract_problems", "schema_problems", "signature_problems"]
