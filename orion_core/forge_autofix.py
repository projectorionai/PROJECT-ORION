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
import builtins as _builtins
import copy
import importlib.util as _importlib_util
import pprint as _pprint
import re as _re
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
    # A declared but optional property may be omitted by the caller. Choosing a
    # meaningful default, or changing that property's requiredness, needs the
    # tool's intent. Preserve the fault instead of inventing None and reporting
    # a partial mechanical repair as complete.
    parameters = schema.get("parameters") if isinstance(schema, dict) else None
    if isinstance(parameters, dict):
        required = parameters.get("required", [])
        if not isinstance(required, (list, tuple)):
            required = []
        for argument in positional + list(run.args.kwonlyargs):
            name = argument.arg
            if name in properties and name not in required and name not in has_default:
                result.remaining.append(
                    "run() requires argument '%s', but the schema does not mark "
                    "it as required; choose a meaningful default or make the "
                    "schema parameter required" % name)
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

    A parameter that is itself called ``kwargs`` made the widened signature a
    duplicate-argument SyntaxError, so the safe name falls back to ``_extra``.
    """
    body = params.strip().rstrip(",").strip()
    if "**" in body:
        return None
    names = {part.strip().split(":", 1)[0].split("=", 1)[0].strip().lstrip("*")
             for part in _split_top_level(body)}
    catch_all = "**_extra" if "kwargs" in names else "**kwargs"
    return (body + ", " + catch_all) if body else catch_all


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


# ══════════════════════════════════════════════════════════════════════════════
# Forge Mark III (W4.1): more repairs that never need a model.
#
# Measured from the lesson corpus (forge_lessons.jsonl, 16 Oct 2026, by hits):
#   contract_violation  47  — 37 of them "run() requires argument X, but the
#                             schema does not declare it" (max_results 11,
#                             time_period 11, config 4, ...); 8 a test calling
#                             get_tool_schema() it never imported
#   runtime NameError    7  — 'sys' 2, 'json' 2 (filed as assertion_failed),
#                             'get_tool_schema' in tests
# Each has one correct answer derivable from the source, so each is fixed here
# for free, before the first verification and again after any failure.
# ══════════════════════════════════════════════════════════════════════════════

#: Modules a generated tool or test uses under their own name.
_MODULE_IMPORTS = frozenset({
    "json", "os", "sys", "re", "math", "time", "shutil", "subprocess", "random",
    "hashlib", "base64", "uuid", "csv", "io", "tempfile", "itertools",
    "functools", "statistics", "string", "textwrap", "platform", "logging",
    "traceback", "collections", "struct", "zlib", "gzip", "zipfile", "tarfile",
    "fnmatch", "heapq", "bisect", "decimal", "fractions", "operator", "secrets",
    "html", "calendar", "locale", "inspect", "asyncio", "socket", "sqlite3",
    "unicodedata", "difflib", "ipaddress", "mimetypes", "shlex", "pathlib",
    "typing", "dataclasses", "enum", "queue", "contextlib", "getpass",
    "warnings", "unittest", "argparse", "string", "binascii", "codecs",
})
#: Names that are normally imported FROM a module.
_NAME_IMPORTS = {
    "Path": "from pathlib import Path", "PurePath": "from pathlib import PurePath",
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque", "namedtuple": "from collections import namedtuple",
    "dataclass": "from dataclasses import dataclass", "asdict": "from dataclasses import asdict",
    "Enum": "from enum import Enum", "IntEnum": "from enum import IntEnum",
    "Decimal": "from decimal import Decimal", "Fraction": "from fractions import Fraction",
    "partial": "from functools import partial", "lru_cache": "from functools import lru_cache",
    "reduce": "from functools import reduce", "wraps": "from functools import wraps",
    "timedelta": "from datetime import timedelta", "timezone": "from datetime import timezone",
    "date": "from datetime import date", "MagicMock": "from unittest.mock import MagicMock",
    "patch": "from unittest.mock import patch", "mock": "from unittest import mock",
    **{name: f"from typing import {name}" for name in (
        "Any", "Dict", "List", "Optional", "Tuple", "Union", "Callable", "Iterable",
        "Iterator", "Sequence", "Mapping", "Set", "FrozenSet", "Type", "Literal",
        "TypedDict", "cast", "NamedTuple")},
}
#: Names that are a module when used as ``x.attr`` and a class/function when called.
_DUAL_IMPORTS = {
    "datetime": ({"datetime", "date", "timedelta", "timezone", "time", "MINYEAR", "MAXYEAR"},
                 "import datetime", "from datetime import datetime"),
    "pprint": ({"pprint", "pformat", "PrettyPrinter"}, "import pprint", "from pprint import pprint"),
    "glob": ({"glob", "iglob", "escape"}, "import glob", "from glob import glob"),
    "copy": ({"copy", "deepcopy"}, "import copy", "from copy import copy"),
}
#: Third-party aliases, only ever added when the package is importable here.
_THIRD_PARTY = {
    "np": ("numpy", "import numpy as np"), "pd": ("pandas", "import pandas as pd"),
    "requests": ("requests", "import requests"), "psutil": ("psutil", "import psutil"),
    "yaml": ("yaml", "import yaml"), "BeautifulSoup": ("bs4", "from bs4 import BeautifulSoup"),
}
#: What the staged fixtures module exports (see forge_fixtures).
_FIXTURE_NAMES = ("tmp_workdir", "make_file", "fake_http", "block_network", "freeze_time",
                  "FakeBus", "FakeResponse", "NetworkBlocked", "coerce_arguments",
                  "check_result", "run_examples")
_MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__loader__",
                   "__package__", "__builtins__", "__annotations__", "__dict__"}
_NAME_ERROR_RE = _re.compile(r"NameError: name '([A-Za-z_][A-Za-z0-9_]*)' is not defined")


def free_names(tree: ast.AST) -> set[str] | None:
    """Names read somewhere that nothing in the module ever binds.

    Scope-insensitive on purpose: a name bound in ANY scope counts as bound,
    so this can only under-report. Adding ``import json`` for a ``json`` that
    is genuinely never bound is always correct. None when a star import makes
    the answer unknowable.
    """
    bound: set[str] = set(dir(_builtins)) | _MODULE_DUNDERS
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif hasattr(ast, "MatchAs") and isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name:
                bound.add(node.name)
        elif hasattr(ast, "MatchMapping") and isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    return loaded - bound


def _attribute_uses(tree: ast.AST, name: str) -> set[str]:
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == name}


def _import_line(tree: ast.AST, name: str) -> str | None:
    if name in _DUAL_IMPORTS:
        module_attrs, as_module, as_name = _DUAL_IMPORTS[name]
        uses = _attribute_uses(tree, name)
        return as_module if uses & module_attrs else as_name
    if name == "urllib":
        subs = sorted(_attribute_uses(tree, "urllib") & {"parse", "request", "error"}) or ["parse"]
        return "; ".join(f"import urllib.{sub}" for sub in subs)
    if name in _MODULE_IMPORTS:
        return f"import {name}"
    if name in _NAME_IMPORTS:
        return _NAME_IMPORTS[name]
    if name in _THIRD_PARTY:
        package, line = _THIRD_PARTY[name]
        try:
            return line if _importlib_util.find_spec(package) is not None else None
        except (ImportError, ValueError):
            return None
    return None


def _insert_top_lines(code: str, lines: list[str]) -> str:
    """Insert *lines* after the module docstring and ``__future__`` imports."""
    tree = ast.parse(code)
    after = 0
    body = list(tree.body)
    index = 0
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        after, index = body[0].end_lineno or 0, 1
    while (index < len(body) and isinstance(body[index], ast.ImportFrom)
           and body[index].module == "__future__"):
        after, index = body[index].end_lineno or after, index + 1
    source = code.splitlines(keepends=True)
    head = "".join(source[:after])
    if head and not head.endswith("\n"):
        head += "\n"
    return head + "".join(line + "\n" for line in lines) + "".join(source[after:])


def add_missing_imports(code: str, error_log: Any = None, *,
                        test_of: str = "") -> AutofixResult:
    """Import the standard-library names *code* uses but never imported.

    Names come from static analysis and from any ``NameError`` in
    *error_log* (a name imported inside one function and used in another is
    bound somewhere, so only the traceback reveals it). With ``test_of`` set
    the source is a TEST of that tool: a free ``run``/``get_tool_schema`` is
    imported from the staged module, and a free fixture name from the staged
    fixtures. A name with no known home is left for the model.
    """
    result = AutofixResult(code=code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.remaining.append(f"the source does not parse ({exc.msg})")
        return result
    names = free_names(tree)
    if names is None:
        return result               # a star import: nothing can be said safely
    text = "\n".join(str(line) for line in (error_log or [])) if not isinstance(
        error_log, str) else error_log
    from_log = set(_NAME_ERROR_RE.findall(text or ""))
    wanted = sorted(names | from_log)
    lines: list[str] = []
    added: list[str] = []
    entry_points = [n for n in ("get_tool_schema", "run") if n in names] if test_of else []
    if entry_points:
        lines.append(f"from {test_of}_tool import {', '.join(entry_points)}")
        added.extend(entry_points)
    if test_of:
        for module_name in (f"{test_of}_tool", test_of):
            if module_name in names:
                lines.append(f"import {module_name}")
                added.append(module_name)
        fixtures = [n for n in _FIXTURE_NAMES if n in names]
        if fixtures:
            lines.append(f"from orion_forge_fixtures import {', '.join(fixtures)}")
            added.extend(fixtures)
    for name in wanted:
        if name in added:
            continue
        line = _import_line(tree, name)
        if line is None:
            if name in from_log:
                result.remaining.append(f"name '{name}' is not defined and has no known import")
            continue
        if line not in lines:
            lines.append(line)
        added.append(name)
    if not lines:
        return result
    candidate = _insert_top_lines(code, lines)
    try:
        ast.parse(candidate)
    except SyntaxError:
        result.remaining.append("adding imports did not parse; left unchanged")
        return result
    result.code = candidate
    result.fixed.append("added missing import(s): " + ", ".join(added))
    return result


# ── run() and the schema, reconciled in both directions ─────────────────────

_ANNOTATION_TYPES = {
    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
    "list": "array", "List": "array", "tuple": "array", "Tuple": "array",
    "set": "array", "Set": "array", "Sequence": "array", "Iterable": "array",
    "dict": "object", "Dict": "object", "Mapping": "object",
}


def _annotation_type(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return _ANNOTATION_TYPES.get(node.id, "")
    if isinstance(node, ast.Attribute):
        return _ANNOTATION_TYPES.get(node.attr, "")
    if isinstance(node, ast.Subscript):
        outer = node.value.id if isinstance(node.value, ast.Name) else getattr(node.value, "attr", "")
        if outer == "Optional":
            return _annotation_type(node.slice)
        if outer == "Union" and isinstance(node.slice, ast.Tuple):
            kinds = [_annotation_type(item) for item in node.slice.elts
                     if not (isinstance(item, ast.Constant) and item.value is None)]
            return kinds[0] if kinds else ""
        return _ANNOTATION_TYPES.get(outer, "")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):   # X | None
        return _annotation_type(node.left) or _annotation_type(node.right)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):  # "int"
        return _ANNOTATION_TYPES.get(node.value, "")
    return ""


def _default_type(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return "array"
    if isinstance(node, ast.Dict):
        return "object"
    return ""


def _run_parameters(run: ast.AST) -> list[tuple[str, ast.AST | None, ast.AST | None, bool]]:
    """(name, annotation, default, has_default) for every keyword-callable parameter."""
    args = run.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.AST | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = [(a.arg, a.annotation, d, d is not None) for a, d in zip(positional, defaults)]
    out += [(a.arg, a.annotation, d, d is not None)
            for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    return [row for row in out if row[0] not in {"self", "cls"}]


def _schema_return(tree: ast.Module) -> tuple[ast.AST | None, ast.Dict | None]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_tool_schema":
            returns = [s for s in ast.walk(node) if isinstance(s, ast.Return) and s.value is not None]
            if len(returns) == 1 and isinstance(returns[0].value, ast.Dict):
                return node, returns[0].value
            return node, None
    return None, None


def _replace_span(code: str, node: ast.AST, replacement: str) -> str:
    lines = code.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    # ast offsets are UTF-8 byte offsets within a line; convert per line.
    def offset(lineno: int, col: int) -> int:
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8", "ignore"))
    begin = offset(node.lineno, node.col_offset)
    finish = offset(node.end_lineno, node.end_col_offset)
    return code[:begin] + replacement + code[finish:]


def _format_literal(value: Any, column: int) -> str:
    text = _pprint.pformat(value, indent=1, width=max(40, 96 - column), sort_dicts=False)
    pad = " " * column
    return text.replace("\n", "\n" + pad)


def reconcile_schema(code: str, tool_name: str, *, description: str = "") -> AutofixResult:
    """Make get_tool_schema() and run() describe the same call.

    Both directions, because the corpus shows both:
      * run() needs a parameter the schema never mentions (37 hits) → declare
        it, typed from its annotation or default, and list it as required
        when run() gives it no default — so the model is told to send it,
        instead of the old ``=None`` default that made run() crash later on
        ``int(None)``;
      * the schema lists a ``required`` key with no property → add the
        property;
      * run() has no default for a property the schema calls optional →
        mark it required. The code that executes is the ground truth; the
        call it demands was going to fail anyway when the argument was left
        out.
    Also: the schema's ``name`` becomes the requested tool name (the Forge
    refuses to activate a mismatch — 'resource_optimizer_tool' has been held
    at every boot since 6 Oct for exactly that), and an absent
    get_tool_schema() is synthesised from run()'s signature.

    Only a literal schema is rewritten; a computed one is reported, not guessed.
    """
    result = AutofixResult(code=code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.remaining.append(f"the source does not parse ({exc.msg})")
        return result
    run = _find_run(tree)
    if run is None:
        result.remaining.append("the module does not define run()")
        return result
    parameters = _run_parameters(run)
    function, literal = _schema_return(tree)
    doc = (ast.get_docstring(run) or "").strip().splitlines()
    summary = description.strip() or (doc[0].strip() if doc else "") or \
        f"{tool_name.replace('_', ' ')} tool"

    if function is None:
        schema: dict[str, Any] = {"name": tool_name, "description": summary[:300],
                                  "parameters": {"type": "object", "properties": {}, "required": []}}
        original = None
    else:
        if literal is None:
            result.remaining.append("get_tool_schema() is computed, not a literal; not rewritten")
            return result
        try:
            original = ast.literal_eval(literal)
        except (ValueError, SyntaxError, TypeError):
            result.remaining.append("get_tool_schema() is computed, not a literal; not rewritten")
            return result
        if not isinstance(original, dict):
            result.remaining.append("get_tool_schema() does not return a dict")
            return result
        schema = copy.deepcopy(original)

    notes: list[str] = []
    if schema.get("name") != tool_name:
        notes.append(f"schema name {schema.get('name')!r} → {tool_name!r}")
        schema["name"] = tool_name
    if not str(schema.get("description") or "").strip():
        schema["description"] = summary[:300]
        notes.append("added a description")
    params = schema.get("parameters")
    if not isinstance(params, dict):
        params = schema["parameters"] = {"type": "object", "properties": {}, "required": []}
        notes.append("added a parameters object")
    if params.get("type") is None:
        params["type"] = "object"
    properties = params.get("properties")
    if not isinstance(properties, dict):
        properties = params["properties"] = {}
    required = params.get("required")
    if not isinstance(required, list):
        required = params["required"] = [] if required is None else [
            str(item) for item in required] if isinstance(required, tuple) else []
    by_name = {name: (annotation, default, has) for name, annotation, default, has in parameters}

    def kind_for(name: str) -> str:
        annotation, default, _has = by_name.get(name, (None, None, False))
        return _annotation_type(annotation) or _default_type(default) or "string"

    for key in list(required):
        if isinstance(key, str) and key not in properties:
            properties[key] = {"type": kind_for(key), "description": key.replace("_", " ")}
            notes.append(f"declared required '{key}' in properties")
    for name, _annotation, _default, has_default in parameters:
        if name not in properties:
            properties[name] = {"type": kind_for(name), "description": name.replace("_", " ")}
            notes.append(f"declared run() parameter '{name}'")
        if not has_default and name not in required:
            required.append(name)
            notes.append(f"marked '{name}' required (run() gives it no default)")

    if not notes:
        return result
    if function is None:
        lines = ["", "", "def get_tool_schema():",
                 "    return " + _format_literal(schema, 11), "", ""]
        lines_text = "\n".join(lines)
        source = code.splitlines(keepends=True)
        at = run.lineno - 1 - len(getattr(run, "decorator_list", []) or [])
        candidate = "".join(source[:at]) + lines_text.lstrip("\n") + "\n" + "".join(source[at:])
        notes.insert(0, "synthesised get_tool_schema() from run()'s signature")
    else:
        candidate = _replace_span(code, literal, _format_literal(schema, literal.col_offset))
    try:
        if schema_from_source(candidate) != schema:
            raise SyntaxError("round trip mismatch")
    except SyntaxError:
        result.remaining.append("the schema rewrite did not round-trip; left unchanged")
        return result
    result.code = candidate
    result.fixed.append("reconciled schema and run(): " + "; ".join(notes))
    return result


# ── arguments arrive as the voice channel sends them ────────────────────────

_COERCE_HELPER = '''

def _orion_coerce(value, kind):
    """Convert one argument to its declared JSON type (added by ORION's Forge:
    voice calls deliver every number as a float, and models send "10")."""
    try:
        if value is None:
            return value
        if kind == "integer":
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                import re as _orion_re
                text = value.strip()
                if _orion_re.fullmatch(r"[+-]?[0-9]+(?:[.]0*)?", text):
                    return int(text.split(".", 1)[0])
        elif kind == "number":
            if isinstance(value, str):
                return float(value)
        elif kind == "boolean":
            if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1", "on", "false", "no", "0", "off"):
                return value.strip().lower() in ("true", "yes", "1", "on")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
                return bool(value)
        elif kind == "array":
            if isinstance(value, str):
                text = value.strip()
                if text.startswith("["):
                    import json as _json
                    return _json.loads(text)
                return [part.strip() for part in text.split(",") if part.strip()]
            if isinstance(value, tuple):
                return list(value)
    except (ValueError, TypeError):
        return value
    return value
'''

_COERCIBLE = {"integer", "number", "boolean", "array"}


def add_argument_coercion(code: str, schema: Any = None) -> AutofixResult:
    """Convert run()'s typed arguments on entry, inside the tool itself.

    Gemini Live passes tool arguments through a protobuf Struct, where every
    number is a double: an INTEGER ``max_results`` arrives as 20.0, and
    ``files[:20.0]`` or ``range(20.0)`` raises TypeError in a tool that passed
    every test. The conversion lives in the generated module (it cannot import
    ORION) and only touches parameters the schema types as integer, number,
    boolean or array. Idempotent.
    """
    result = AutofixResult(code=code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.remaining.append(f"the source does not parse ({exc.msg})")
        return result
    if "_orion_coerce" in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}:
        return result
    if schema is None:
        schema = schema_from_source(code)
    try:
        properties = (schema or {}).get("parameters", {}).get("properties", {}) or {}
    except AttributeError:
        return result
    run = _find_run(tree)
    if run is None or not run.body or not isinstance(properties, dict):
        return result
    kinds = {key: str((spec or {}).get("type") or "").lower() for key, spec in properties.items()
             if isinstance(spec, dict)}
    named = [name for name, *_rest in _run_parameters(run) if kinds.get(name) in _COERCIBLE]
    kwarg = run.args.kwarg.arg if run.args.kwarg is not None else ""
    extra = {key: kind for key, kind in kinds.items()
             if kind in _COERCIBLE and key not in {n for n, *_r in _run_parameters(run)}}
    if not named and not (kwarg and extra):
        return result
    first = run.body[0]
    if first.lineno == run.lineno:
        result.remaining.append("run()'s body shares the def line; coercion not added")
        return result
    anchor = first
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        if len(run.body) < 2:
            return result
        anchor = run.body[1]
        insert_at = first.end_lineno or first.lineno
    else:
        insert_at = first.lineno - 1
    pad = " " * anchor.col_offset
    prologue = [f"{pad}{name} = _orion_coerce({name}, {kinds[name]!r})" for name in named]
    if kwarg and extra:
        prologue.append(f"{pad}{kwarg} = {{_k: _orion_coerce(_v, {extra!r}.get(_k)) "
                        f"for _k, _v in {kwarg}.items()}}")
    lines = code.splitlines(keepends=True)
    if insert_at and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] += "\n"
    body = "".join(lines[:insert_at]) + "".join(p + "\n" for p in prologue) + "".join(lines[insert_at:])
    # The helper goes right before run(), so it is defined at import time.
    tree2 = ast.parse(body)
    run2 = _find_run(tree2)
    at = (run2.lineno - 1 - len(run2.decorator_list)) if run2 is not None else len(body.splitlines())
    source = body.splitlines(keepends=True)
    candidate = "".join(source[:at]) + _COERCE_HELPER.lstrip("\n") + "\n\n" + "".join(source[at:])
    try:
        ast.parse(candidate)
    except SyntaxError:
        result.remaining.append("argument coercion did not parse; left unchanged")
        return result
    result.code = candidate
    result.fixed.append("run() now converts typed arguments on entry: "
                        + ", ".join(named + sorted(extra) if kwarg else named))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# W4.2: a test policy the model can meet, checked BEFORE anything runs.
#
# Real-path and network failures were 18+ lesson hits; each was discovered by
# running the test, then "repaired" by rewriting the MODULE. A test that will
# touch the owner's disk or the internet is rejected statically, with the exact
# reason, and only the TEST goes back for repair.
# ══════════════════════════════════════════════════════════════════════════════

_NETWORK_CALLS = {
    ("requests", "get"), ("requests", "post"), ("requests", "put"), ("requests", "delete"),
    ("requests", "patch"), ("requests", "head"), ("requests", "request"), ("requests", "Session"),
    ("httpx", "get"), ("httpx", "post"), ("httpx", "Client"), ("httpx", "AsyncClient"),
    ("httpx", "request"), ("urllib.request", "urlopen"), ("request", "urlopen"),
    ("socket", "socket"), ("socket", "create_connection"), ("aiohttp", "ClientSession"),
    ("http.client", "HTTPConnection"), ("http.client", "HTTPSConnection"),
    ("client", "HTTPConnection"), ("client", "HTTPSConnection"),
}
_ABSOLUTE_PATH_RE = _re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|etc|var|usr|root|mnt|opt|private)\b|~[\\/]|%[A-Z]+%)")
_HOME_CALLS = {("Path", "home"), ("path", "expanduser"), ("os.path", "expanduser"),
               ("Path", "expanduser")}
_HOME_ENV = {"USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "HOMEPATH", "ONEDRIVE"}


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def check_test_policy(test_code: str) -> list[str]:
    """Why *test_code* may not run in the verifier, or [] when it may."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []                 # a syntax error is reported by verification itself
    problems: list[str] = []
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    fakes_network = bool({"fake_http", "block_network", "patch", "mock", "MagicMock",
                          "monkeypatch"} & (names | attrs))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            owner, _, attr = dotted.rpartition(".")
            if (owner, attr) in _NETWORK_CALLS or dotted == "urlopen":
                if not fakes_network:
                    problems.append(f"it makes a real network call ({dotted}) — wrap it in "
                                    "fake_http({...}) from orion_forge_fixtures")
            if (owner, attr) in _HOME_CALLS or dotted in {"expanduser", "os.path.expanduser"}:
                problems.append(f"it reads the real home folder ({dotted}) — use tmp_workdir()")
            if dotted in {"time.sleep", "sleep"}:
                arg = node.args[0] if node.args else None
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float))
                        and arg.value <= 0.2):
                    problems.append("it sleeps — verification has a time budget; call run() directly")
            if dotted == "input":
                problems.append("it waits for input() — nobody is there to type")
            if dotted in {"os.getenv", "os.environ.get"} and node.args and isinstance(
                    node.args[0], ast.Constant) and str(node.args[0].value).upper() in _HOME_ENV:
                problems.append(f"it reads the real {node.args[0].value} folder — use tmp_workdir()")
        elif isinstance(node, ast.Subscript) and _dotted(node.value) == "os.environ":
            key = node.slice
            if isinstance(key, ast.Constant) and str(key.value).upper() in _HOME_ENV:
                problems.append(f"it reads the real {key.value} folder — use tmp_workdir()")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if "\n" in value or len(value) > 260 or "://" in value:
                continue
            if _ABSOLUTE_PATH_RE.match(value) or value.startswith(("../", "..\\")):
                problems.append(f"it uses a real path {value[:60]!r} — create files under "
                                "tmp_workdir() instead")
    seen: list[str] = []
    for problem in problems:
        if problem not in seen:
            seen.append(problem)
    return seen[:6]


__all__ = ["AutofixResult", "autofix_contract", "add_missing_imports", "reconcile_schema",
           "add_argument_coercion", "check_test_policy", "free_names", "schema_from_source"]
