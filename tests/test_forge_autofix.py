"""
Contract violations get fixed locally, not by burning a rate-limited model.

From the reported session: the forge fired plan → generate → verify → repair →
verify → repair in four seconds, exhausted a 6000-tokens-per-minute budget, and
every provider went into a five-minute cooldown. Most of those repairs were
mechanical and never needed a model at all.
"""

from __future__ import annotations

import ast

import pytest

from orion_core.forge_autofix import autofix_contract
from orion_core.forge_contract import contract_problems

SCHEMA = {
    "name": "url_parser",
    "description": "Parse a URL into parts.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}, "strict": {"type": "boolean"}},
        "required": ["url"],
    },
}


def _load(code: str):
    """Exec the module and return (schema, run) as the loader would."""
    namespace: dict = {}
    exec(compile(code, "<autofix>", "exec"), namespace)  # noqa: S102 - test harness
    return namespace.get("get_tool_schema", lambda: None)(), namespace.get("run")


BASE_SCHEMA_FN = '''
def get_tool_schema():
    return {
        "name": "url_parser",
        "description": "Parse a URL into parts.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "strict": {"type": "boolean"}},
            "required": ["url"],
        },
    }
'''


# ── the single most common violation ──────────────────────────────────────────

def test_run_that_cannot_accept_a_declared_parameter_gets_kwargs():
    code = BASE_SCHEMA_FN + '''
def run(url):
    return url
'''
    schema, run = _load(code)
    assert contract_problems(schema, run), "precondition: this must be a violation"

    result = autofix_contract(code, schema)
    assert result.changed
    assert "**kwargs" in result.code
    assert "strict" in result.summary()

    schema2, run2 = _load(result.code)
    assert contract_problems(schema2, run2) == [], (
        "the autofix must actually satisfy the contract, not merely change code")


def test_the_fix_is_complete_so_no_model_call_is_needed():
    code = BASE_SCHEMA_FN + '''
def run(url):
    return url
'''
    schema, _ = _load(code)
    result = autofix_contract(code, schema)
    assert result.complete is True, (
        "a fully-mechanical fix must not consume a model attempt")


# ── positional-only parameters ────────────────────────────────────────────────

def test_positional_only_parameters_are_made_keyword_accepting():
    code = BASE_SCHEMA_FN + '''
def run(url, /, strict=False):
    return url
'''
    schema, run = _load(code)
    assert any("positional-only" in p for p in contract_problems(schema, run))

    result = autofix_contract(code, schema)
    assert result.changed
    schema2, run2 = _load(result.code)
    assert contract_problems(schema2, run2) == []


# ── a required argument the schema never declares ─────────────────────────────

def test_an_undeclared_required_argument_gets_a_default():
    code = BASE_SCHEMA_FN + '''
def run(url, strict, timeout):
    return url
'''
    schema, run = _load(code)
    problems = contract_problems(schema, run)
    assert any("timeout" in p for p in problems)

    result = autofix_contract(code, schema)
    schema2, run2 = _load(result.code)
    assert contract_problems(schema2, run2) == []
    assert "timeout=None" in result.code


# ── things it must NOT touch ──────────────────────────────────────────────────

def test_a_conforming_tool_is_left_completely_alone():
    code = BASE_SCHEMA_FN + '''
def run(url=None, strict=False, **kwargs):
    return url
'''
    schema, run = _load(code)
    assert contract_problems(schema, run) == []

    result = autofix_contract(code, schema)
    assert result.changed is False
    assert result.code == code


def test_a_syntax_error_is_handed_to_the_model_not_guessed_at():
    result = autofix_contract("def run(:\n    pass", None)
    assert result.changed is False
    assert any("does not parse" in r for r in result.remaining)


def test_a_missing_run_is_reported_as_needing_the_model():
    result = autofix_contract(BASE_SCHEMA_FN, SCHEMA)
    assert result.changed is False
    assert any("does not define run()" in r for r in result.remaining)


def test_a_missing_schema_function_is_reported():
    code = '''
def run(**kwargs):
    return "x"
'''
    result = autofix_contract(code, SCHEMA)
    assert any("get_tool_schema" in r for r in result.remaining)


def test_a_malformed_schema_is_not_invented(monkeypatch):
    code = BASE_SCHEMA_FN + '''
def run(**kwargs):
    return "x"
'''
    bad = {"name": "", "description": "", "parameters": {"type": "object"}}
    result = autofix_contract(code, bad)
    assert result.remaining, "an empty name/description needs real content"
    assert any("name" in r for r in result.remaining)


# ── signatures that are awkward to parse ──────────────────────────────────────

def test_a_multi_line_signature_is_handled():
    # 'strict' is declared in the schema but absent from run() — a real
    # violation, spread over several lines so the span finder has to work.
    code = BASE_SCHEMA_FN + '''
def run(
    url,
):
    return url
'''
    schema, _ = _load(code)
    result = autofix_contract(code, schema)
    assert result.changed
    ast.parse(result.code)
    schema2, run2 = _load(result.code)
    assert contract_problems(schema2, run2) == []


def test_defaults_containing_commas_do_not_confuse_the_splitter():
    code = BASE_SCHEMA_FN + '''
def run(url, strict=False, opts={"a": 1, "b": 2}):
    return url
'''
    schema, _ = _load(code)
    result = autofix_contract(code, schema)
    ast.parse(result.code)
    schema2, run2 = _load(result.code)
    assert contract_problems(schema2, run2) == []


def test_an_annotated_signature_survives():
    code = BASE_SCHEMA_FN + '''
def run(url: str, strict: bool = False) -> dict:
    return {"url": url}
'''
    schema, _ = _load(code)
    result = autofix_contract(code, schema)
    ast.parse(result.code)
    schema2, run2 = _load(result.code)
    assert contract_problems(schema2, run2) == []


def test_an_async_run_is_handled():
    code = BASE_SCHEMA_FN + '''
async def run(url):
    return url
'''
    schema, _ = _load(code)
    result = autofix_contract(code, schema)
    assert result.changed
    ast.parse(result.code)


def test_the_output_always_parses():
    """Never hand back source that cannot be imported."""
    for body in (
        "def run(url): return url",
        "def run(url, /): return url",
        "def run(url, *, strict): return url",
        "def run(**kw): return kw",
    ):
        code = BASE_SCHEMA_FN + "\n" + body + "\n"
        schema, _ = _load(code)
        result = autofix_contract(code, schema)
        ast.parse(result.code)


# ── the forge uses it ─────────────────────────────────────────────────────────

def test_the_forge_tries_the_local_fix_before_calling_a_model():
    import inspect

    from orion_core import forge

    # forge_tool validates the name and takes the per-name lock; the
    # pipeline itself runs in _forge_tool_locked.
    source = inspect.getsource(forge.ForgeOrchestrationManager._forge_tool_locked)
    autofix_at = source.find("autofix_contract")
    repair_at = source.find("_apply_repair")
    assert autofix_at != -1, "the forge must attempt the free repair"
    assert autofix_at < repair_at, (
        "the deterministic fix has to run BEFORE the model call, or it saves "
        "no tokens at all")


def test_the_forge_logs_the_specific_contract_problems():
    import inspect

    from orion_core import forge

    # forge_tool validates the name and takes the per-name lock; the
    # pipeline itself runs in _forge_tool_locked.
    source = inspect.getsource(forge.ForgeOrchestrationManager._forge_tool_locked)
    assert 'CONTRACT:' in source, (
        "the log said a violation happened but never what it was")


# ── the schema is recovered from source, and `complete` never lies ───────────
#
# From the user's log, five identical failures in a row:
#
#   [FORGE] Manager: x attempt 5 - contract_violation -> repair module
#   [FORGE]   -> CONTRACT: run() requires argument 'data', but the schema does
#              not declare it in properties
#   (and 'test_size', and 'max_results')
#
# The autofix had diagnosed all three correctly and then done nothing, because
# sandbox.py only records outcome.schema once the conformance probe has PASSED.
# On a contract violation - exactly when the autofix is wanted - the caller has
# no schema to pass, so `properties` was empty and every mechanical fix was
# skipped. The forge burned all five attempts and a rate-limited provider on
# repairs that were free.

def test_the_schema_is_recovered_from_the_source_when_none_is_supplied():
    from orion_core.forge_autofix import schema_from_source

    recovered = schema_from_source(
        'def get_tool_schema():\n'
        '    return {"name": "t", "description": "d", "parameters": {"type": "object",\n'
        '            "properties": {"q": {"type": "string"}}}}\n')
    assert recovered is not None
    assert recovered["parameters"]["properties"]["q"]["type"] == "string"


def test_the_schema_is_read_not_executed():
    """Running untrusted generated code to ask what it thinks its own
    arguments are is not a trade worth making."""
    from orion_core.forge_autofix import schema_from_source

    marker = []
    import builtins
    original = builtins.__import__

    def _trap(name, *args, **kwargs):
        if name == "os":
            marker.append(name)
        return original(name, *args, **kwargs)

    builtins.__import__ = _trap
    try:
        schema_from_source(
            'import os\n'
            'os.environ["PWNED"] = "1"\n'
            'def get_tool_schema():\n'
            '    return {"parameters": {"properties": {}}}\n')
    finally:
        builtins.__import__ = original
    assert not marker, "the module was executed"


def test_a_computed_schema_is_declined_rather_than_guessed():
    from orion_core.forge_autofix import schema_from_source

    assert schema_from_source(
        'def get_tool_schema():\n'
        '    return build_schema()\n') is None


@pytest.mark.parametrize("source", [
    # The exact shape from the log.
    'def get_tool_schema():\n'
    '    return {"name": "t", "description": "d", "parameters": {"type": "object",\n'
    '            "properties": {"q": {"type": "string"}}, "required": ["q"]}}\n'
    'def run(q, data, test_size, max_results):\n'
    '    return q\n',
    # The same, with positional-only parameters.
    'def get_tool_schema():\n'
    '    return {"name": "t", "description": "d", "parameters": {"type": "object",\n'
    '            "properties": {"q": {"type": "string"}}, "required": ["q"]}}\n'
    'def run(q, data, test_size, max_results, /):\n'
    '    return q\n',
    # A declared property run() cannot accept.
    'def get_tool_schema():\n'
    '    return {"name": "t", "description": "d", "parameters": {"type": "object",\n'
    '            "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}}\n'
    'def run(q):\n'
    '    return q\n',
])
def test_complete_never_lies(source):
    """`complete` means "nothing left for a model to look at", and the forge
    skips the model call on the strength of it. Reporting it while the module
    still fails the contract is worse than doing nothing at all — and that is
    what happened with positional-only parameters, which ast reports under
    posonlyargs where the undeclared-parameter scan never looked.
    """
    from orion_core.forge_autofix import autofix_contract
    from orion_core.forge_contract import contract_problems

    result = autofix_contract(source, None)     # None, as the sandbox passes
    namespace: dict = {}
    exec(compile(result.code, "<autofixed>", "exec"), namespace)
    remaining = contract_problems(namespace["get_tool_schema"](), namespace["run"])
    if result.complete:
        assert not remaining, (
            f"reported complete but the contract still fails: {remaining}")
    assert not remaining, f"the mechanical fix did not hold: {remaining}"
