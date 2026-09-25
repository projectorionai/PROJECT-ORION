"""
What the sanitiser actually stops — and, just as importantly, what it does not.

``SecuritySanitiser.guard_text`` is two layers: a regex denylist over shell-ish
text, and an AST pass that rejects destructive Python calls. The AST pass
compared a **literal dotted name** against a denylist, which meant it caught

    os.remove(path)                     # blocked

and missed the two ways people ordinarily write the same thing:

    import os as o;         o.remove(path)          # allowed
    from shutil import rmtree;  rmtree(path)        # allowed

Neither is an exotic evasion, so import aliases are now resolved before the
denylist comparison.

**The honest scope.** This is defence in depth over text that is *not executed
as Python*. `guard_text` is applied to app names, chess moves, task titles,
research topics — short user/model-supplied strings — and nothing in ORION
``exec``s a payload that has passed it. Code that genuinely runs takes a
different path: ``guard_forged_source`` plus the forge sandbox. So the value
here is catching a model that tries to smuggle destructive instructions into a
text field, not defeating an attacker who already has code execution.

The ``test_known_limit_*`` cases below record constructs that still get through.
They are deliberately written as assertions rather than comments: if one starts
failing, the guard has become stronger and the record should be updated on
purpose, rather than the limits quietly drifting out of date in a docstring.
Closing them all would be an arms race (``eval``, ``getattr`` with a built
string, ``importlib``) on a layer whose consequence is bounded — that judgement
is recorded here so the next reader does not have to re-derive it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.security import SecuritySanitiser, SecurityViolation  # noqa: E402

#: A harmless-looking literal, so the REGEX layer cannot fire and the test is
#: genuinely exercising the AST pass. Using "format c:" here would have every
#: case "pass" for the wrong reason — which is exactly what the first version
#: of this probe did.
TARGET = "'/home/user/notes'"


def _blocked(payload: str) -> bool:
    try:
        SecuritySanitiser.guard_text(payload, "test")
        return False
    except SecurityViolation:
        return True


# ── the literals must not be doing the work ─────────────────────────────────

def test_the_probe_target_is_not_itself_suspicious():
    """Guards the guard: if TARGET tripped the regex layer, every case below
    would pass without testing the AST pass at all."""
    assert not _blocked("a note about " + TARGET)


# ── baseline: the direct forms ──────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    "import os\nos.remove(%s)" % TARGET,
    "import shutil\nshutil.rmtree(%s)" % TARGET,
    "import os\nos.system('echo hello')",
    "import os\nos.unlink(%s)" % TARGET,
    "import os\nos.rmdir(%s)" % TARGET,
])
def test_a_direct_destructive_call_is_blocked(payload):
    assert _blocked(payload)


# ── the gap this file closed ────────────────────────────────────────────────

def test_an_aliased_module_is_resolved():
    """`import os as o` then `o.remove(...)` — ordinary Python, previously
    walked straight through."""
    assert _blocked("import os as o\no.remove(%s)" % TARGET)


def test_a_from_import_is_resolved():
    assert _blocked("from os import remove\nremove(%s)" % TARGET)


def test_an_aliased_from_import_is_resolved():
    assert _blocked("from shutil import rmtree as nuke\nnuke(%s)" % TARGET)


def test_a_dotted_import_alias_is_resolved():
    assert _blocked("import os.path as p\nimport os\nos.remove(%s)" % TARGET)


def test_aliases_do_not_create_false_positives():
    """Resolving aliases must not start blocking innocent code that happens to
    bind a harmless name."""
    assert not _blocked("import json as os_like\nos_like.dumps({'a': 1})")
    assert not _blocked("from pathlib import Path\np = Path(%s)\nprint(p.name)" % TARGET)
    assert not _blocked("import os\nprint(os.getcwd())")
    assert not _blocked("from os import getcwd\nprint(getcwd())")


def test_ordinary_prose_is_untouched():
    """guard_text runs on task titles and research topics; it must not become a
    filter that rejects normal English."""
    for text in ("remind me to tidy the downloads folder",
                 "research how transformers handle long context",
                 "start a focus block on the dissertation",
                 "what did we discuss about the agency rate"):
        assert not _blocked(text), text


# ── the regex layer ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    "rm -rf / --no-preserve-root",
    "del /s /q C:\\\\Windows",
    "format c:",
    "diskpart",
    "reg delete HKLM\\\\Software",
    "takeown /f C:\\\\Windows",
    "shutdown /s /t 0",
])
def test_destructive_shell_text_is_blocked(payload):
    assert _blocked(payload)


def test_core_mutation_is_blocked():
    assert _blocked("please delete orion_core/app.py")
    assert _blocked("overwrite orion.py with this")


# ── known limits, asserted on purpose ───────────────────────────────────────
#
# Each of these still gets through. They are recorded rather than fixed: the
# layer protects text that is never executed, and closing them is an arms race.
# If one of these fails, the guard has been strengthened — update this record
# deliberately rather than deleting the test.

def test_known_limit_eval_of_a_string_is_not_inspected():
    assert not _blocked("eval(\"__import__('shutil').rmtree('/x')\")")


def test_known_limit_getattr_with_a_built_name_is_not_resolved():
    assert not _blocked("import os\ngetattr(os, 'rem'+'ove')(%s)" % TARGET)


def test_known_limit_importlib_is_not_followed():
    assert not _blocked(
        "import importlib\nimportlib.import_module('shutil').rmtree(%s)" % TARGET)


def test_known_limit_a_bound_method_is_not_traced():
    assert not _blocked("import shutil\nf = shutil.rmtree\nf(%s)" % TARGET)


def test_known_limit_the_ast_pass_is_skipped_above_the_size_budget():
    """Deliberate and already covered by test_security_sanitiser.py: parsing is
    bounded because it runs on the qasync loop (measured 22 ms at 47 KB, 80 ms
    at 142 KB). The regex layer still applies above the budget."""
    padded = "# " + "A" * 12500 + "\nimport os\nos.remove(%s)" % TARGET
    assert not _blocked(padded)
    # ...but the regex layer is NOT skipped, whatever the size
    assert _blocked("# " + "A" * 12500 + "\nformat c:")


# ── the scope claim in the docstring must stay true ─────────────────────────

#: Every place in orion_core that turns text into running code, with the reason
#: it is allowed to. A plugin system must execute plugin files — that is what it
#: is — so the invariant is not "nothing execs", it is "nothing execs that has
#: not been deliberately accepted".
SANCTIONED_EXECUTION = {
    # The forged-tool loader. Both its paths run files from
    # config/custom_tools/, which the forge writes only after
    # guard_forged_source approves them (forge.py), and the rehabilitation path
    # re-applies that guard before running a file that has sat in quarantine.
    "dynamic_loader.py",
}


def test_no_unsanctioned_code_execution_path_exists():
    """`guard_text`'s known limits are acceptable *because* approved text is
    never executed. Code that IS executed takes the forge path, with its own
    guard. A new exec/eval anywhere else breaks that reasoning, so it must be a
    deliberate, reviewed addition rather than a quiet one.
    """
    import re
    offenders = []
    for path in (ROOT / "orion_core").rglob("*.py"):
        if path.name in SANCTIONED_EXECUTION:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        for match in re.finditer(r"(?<![\w.])(?:exec|eval)\s*\(", source):
            line_no = source[:match.start()].count("\n") + 1
            line = lines[line_no - 1]
            if ".exec(" in line or "dialog" in line.lower():
                continue                      # Qt's QDialog.exec()
            offenders.append("%s:%d %s" % (path.name, line_no, line.strip()[:70]))
    assert not offenders, (
        "a new code-execution path appeared outside the sanctioned loader — if "
        "it is intentional, add the file to SANCTIONED_EXECUTION with the "
        "reason: " + "; ".join(offenders))


def test_the_sanctioned_loader_really_does_execute_code():
    """Guards the guard: an allowlist naming a file that no longer execs would
    be protecting nothing."""
    import re
    source = (ROOT / "orion_core" / "dynamic_loader.py").read_text(encoding="utf-8")
    assert re.search(r"(?<![\w.])exec\s*\(", source) or "exec_module" in source


def test_rehabilitating_a_quarantined_tool_re_applies_the_forge_guard():
    """The rehabilitation probe EXECUTES a quarantined file to see whether it
    would load now. It was vetted when the forge wrote it, but it has since sat
    on disk — "vetted once" is not "safe to run now"."""
    import inspect
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    src = inspect.getsource(ReflectiveModuleLoader._can_rehabilitate)
    assert "guard_forged_source" in src, (
        "quarantined code is executed without re-checking the forge guard")
    assert src.index("guard_forged_source") < src.index("exec(compile"), (
        "the guard must run BEFORE the code does")


def test_the_forge_guard_is_separate_and_narrower_on_purpose():
    """Forged code DOES run, so it takes a different path — and that one has no
    size budget."""
    import inspect
    src = inspect.getsource(SecuritySanitiser.guard_forged_source)
    assert "12000" not in src, "the forge guard must not inherit a size budget"
    assert "orion_core" in src, "the forge guard no longer blocks self-reference"


def test_guard_payload_walks_nested_structures():
    """Tool arguments arrive as dicts and lists, not bare strings."""
    with pytest.raises(SecurityViolation):
        SecuritySanitiser.guard_payload({"cmd": {"inner": ["format c:"]}}, "test")
    assert SecuritySanitiser.guard_payload({"a": ["fine", 3, None]}, "test") == {
        "a": ["fine", 3, None]}
