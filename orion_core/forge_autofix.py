"""
Deterministic repair of tool-contract violations — no model call required.

Why this exists
---------------
From a real session log:

    17:53:27  Sandbox: ✗ contract check failed for 'url_parser' (exit 1)
    17:53:27  ✗ attempt 2 — contract_violation → repair module
    17:53:27  NET: provider groq cooled for 300s - HTTP 429 …
              Limit 6000, Used 5981, Requested 1332
    17:53:27  ✗ forge session failed: all text providers failed

Two things went wrong there and only one of them is about contracts.

The forge fired plan → generate → verify → repair → verify → repair inside four
seconds, against a 6000-tokens-per-minute budget. It exhausted the quota, every
provider went into a five-minute cooldown, and self-improvement stopped dead —
not because the code was unfixable, but because there was nobody left to ask.

The thing is, almost none of those repairs needed asking. A contract violation
is nearly always MECHANICAL:

    run() does not accept a parameter the schema declares   → add **kwargs
    run() takes positional-only parameters                  → drop the ``/``
    run() requires an argument the schema never declares    → give it a default

Each of those has exactly one correct fix, derivable from the code itself.
Sending them to a language model is slow, costs tokens ORION cannot spare, is
non-deterministic, and — as the log shows — can fail outright for reasons that
have nothing to do with the problem.

So they are fixed here: locally, instantly, for free, and identically every
time. The model is then reserved for the failures that genuinely need judgement
(a wrong algorithm, a bad assertion), which is what a rate-limited budget should
actually be spent on.

Everything here is AST-driven rather than regex-driven, because a signature can
span lines, carry defaults containing commas and brackets, and be decorated.
Nothing is rewritten unless the result still parses.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from .forge_contract import schema_problems


@dataclass
class AutofixResult:
    """What the deterministic pass managed to do."""

    code: str
    fixed: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.fixed)

    @property
    def complete(self) -> bool:
        """True when nothing is left that a model would need to look at."""
        return bool(self.fixed) and not self.remaining

    def summary(self) -> str:
        parts = []
        if self.fixed:
            parts.append("fixed locally: " + "; ".join(self.fixed))
        if self.remaining:
            parts.append("needs the model: " + "; ".join(self.remaining))
        return " | ".join(parts) or "nothing to do"


def _find_run(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            return node
    return None


def _signature_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    """Character offsets of the parameter list inside ``def run(...)``.

    Located by scanning from the ``def`` line for the matching close paren, so
    defaults containing commas, brackets or nested calls cannot confuse it.
    """
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    if node.lineno - 1 >= len(starts):
        return None
    offset = starts[node.lineno - 1]
    open_at = source.find("(", offset)
    if open_at == -1:
        return None
    depth = 0
    for index in range(open_at, len(source)):
        char = source[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return open_at + 1, index
    return None


def _has_kwargs(node: ast.AST) -> bool:
    return getattr(node.args, "kwarg", None) is not None


def _declared_names(node: ast.AST) -> set[str]:
    args = node.args
    named = [a.arg for a in args.args] + [a.arg for a in args.kwonlyargs]
    return set(named)


def schema_from_source(code: str) -> Any:
    """Recover the declared schema by READING get_tool_schema, not running it.

    This is what makes the autofix work at all in practice. The sandbox only
    records ``outcome.schema`` once the conformance probe has PASSED — so on a
    contract violation, which is exactly when the autofix is wanted, the caller
    has no schema to hand it. Without one there are no declared properties, so
    every mechanical fix was silently skipped and the forge burned all five
    attempts and a rate-limited provider on repairs it could have done for
    free. (Observed: five identical failures naming the same three parameters.)

    ``ast.literal_eval`` rather than executing the module: a schema is a plain
    dict literal in every forged tool, and running untrusted generated code to
    find out what it thinks its own arguments are is not a trade worth making.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "get_tool_schema":
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.Return) and statement.value is not None:
                try:
                    value = ast.literal_eval(statement.value)
                except (ValueError, SyntaxError, TypeError):
                    return None
                return value if isinstance(value, dict) else None
    return None


def autofix_contract(code: str, schema: Any,
                     problems: list[str] | None = None) -> AutofixResult:
    """Repair mechanically-fixable contract violations in *code*.

    *schema* is the tool's declared schema. It is routinely None here — see
    :func:`schema_from_source` — so the source is read as a fallback rather
    than giving up, which is what turns this from a no-op into the thing that
    actually saves the forge session.
    """
    result = AutofixResult(code=code)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # Not a contract problem at all — the model has to fix this.
        result.remaining.append(f"the source does not parse ({exc.msg})")
        return result

    if schema is None:
        schema = schema_from_source(code)

    run = _find_run(tree)
    if run is None:
        result.remaining.append("the module does not define run()")
        return result

    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "get_tool_schema" for n in tree.body):
        result.remaining.append("the module does not define get_tool_schema()")

    properties: dict = {}
    if schema is not None:
        schema_issues, properties = schema_problems(schema)
        # A malformed schema is a content problem, not a shape we can invent.
        result.remaining.extend(schema_issues)

    span = _signature_span(code, run)
    if span is None:
        result.remaining.append("run()'s signature could not be located")
        return result

    start, end = span
    params = code[start:end]
    new_params = params
    notes: list[str] = []

    # ── 1. positional-only parameters ────────────────────────────────────────
    # The dispatcher always calls run(**kwargs), so a positional-only parameter
    # can never be supplied. Dropping the marker makes them keyword-accepting;
    # nothing else about the function changes.
    if run.args.posonlyargs:
        stripped = _drop_posonly_marker(new_params)
        if stripped is not None:
            new_params = stripped
            notes.append("made positional-only parameters keyword-accepting")

    # ── 2. required parameters the schema never declares ─────────────────────
    # These can never arrive, so the call raises TypeError on the first use.
    # A default makes the function callable exactly as the dispatcher calls it.
    # posonlyargs MUST be included. Fix 1 above may have just removed the "/"
    # marker, but ast still reports those parameters under posonlyargs because
    # the tree was parsed from the ORIGINAL source. Scanning only args and
    # kwonlyargs missed them entirely, and the result was worse than a miss:
    # the pass reported `complete` — "nothing left for a model to look at" —
    # on a module that still failed three contract checks, so the caller would
    # skip the real repair on the strength of a fix that had not happened.
    undeclared = [
        a.arg for a in (list(run.args.posonlyargs) + list(run.args.args)
                        + list(run.args.kwonlyargs))
        if a.arg not in properties
    ]
    defaults_count = len(run.args.defaults)
    # Defaults apply to the tail of posonlyargs + args combined.
    positional = list(run.args.posonlyargs) + list(run.args.args)
    has_default = {a.arg for a in positional[len(positional) - defaults_count:]}
    has_default |= {
        a.arg for a, d in zip(run.args.kwonlyargs, run.args.kw_defaults) if d is not None
    }
    needs_default = [name for name in undeclared if name not in has_default]
    if needs_default and properties:
        patched = _add_defaults(new_params, needs_default)
        if patched is not None:
            new_params = patched
            notes.append("gave undeclared required parameter(s) a default: "
                         + ", ".join(needs_default))

    # ── 3. schema properties run() cannot accept ─────────────────────────────
    # **kwargs absorbs anything, which is exactly the contract the dispatcher
    # needs and the single most common violation.
    if not _has_kwargs(run):
        unaccepted = [key for key in properties if key not in _declared_names(run)]
        if unaccepted:
            widened = _add_kwargs(new_params)
            if widened is not None:
                new_params = widened
                notes.append("added **kwargs so run() accepts every declared "
                             f"parameter ({', '.join(sorted(unaccepted))})")

    if new_params != params:
        candidate = code[:start] + new_params + code[end:]
        try:
            ast.parse(candidate)          # never hand back source that will not parse
        except SyntaxError:
            result.remaining.append(
                "a mechanical signature fix did not parse; left unchanged")
            return result
        result.code = candidate
        result.fixed.extend(notes)

    return result


def _drop_posonly_marker(params: str) -> str | None:
    """Remove the bare ``/`` positional-only marker from a parameter list."""
    parts = _split_top_level(params)
    kept = [p for p in parts if p.strip() != "/"]
    if len(kept) == len(parts):
        return None
    return ", ".join(p.strip() for p in kept if p.strip())


def _add_kwargs(params: str) -> str | None:
    """Append ``**kwargs`` to a parameter list.

    The trailing comma matters: a multi-line signature is conventionally
    written ``def run(\\n    url,\\n)``, and naively appending produced
    ``url,, **kwargs``, which does not parse. Caught by the round-trip parse
    guard rather than shipped.
    """
    body = params.strip().rstrip(",").strip()
    if "**" in body:
        return None
    return (body + ", **kwargs") if body else "**kwargs"


def _add_defaults(params: str, names: list[str]) -> str | None:
    """Give each named parameter ``=None`` if it has no default already."""
    parts = _split_top_level(params)
    if not parts:
        return None
    wanted = set(names)
    out: list[str] = []
    changed = False
    for part in parts:
        stripped = part.strip()
        if not stripped or stripped in ("/", "*"):
            out.append(stripped)
            continue
        # name, name: annotation, name=default, name: ann = default
        name = stripped.split(":", 1)[0].split("=", 1)[0].strip().lstrip("*")
        if name in wanted and "=" not in stripped:
            out.append(f"{stripped}=None")
            changed = True
        else:
            out.append(stripped)
    if not changed:
        return None
    return ", ".join(p for p in out if p)


def _split_top_level(params: str) -> list[str]:
    """Split a parameter list on commas that are not inside brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in params:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


__all__ = ["AutofixResult", "autofix_contract"]
