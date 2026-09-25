"""
Forge artefact transport (Mark II).

The Forge's original code generator asked the model to return a whole Python
module *inside a JSON string value*.  That is the single most fragile thing you
can ask a language model to do: every newline must become ``\\n``, every quote
``\\"``, and one slip anywhere in a 200-line module kills the entire session
with ``Expecting ',' delimiter``.  ``forge._repair_json`` exists purely to paper
over that self-inflicted wound.

This module removes the wound instead.  Source travels in **fenced code
blocks**, which models emit reliably because it is how they are trained to
write code, and which parse deterministically with no escaping rules at all:

    === TOOL ===
    ```python
    <module source>
    ```
    === TEST ===
    ```python
    <test source>
    ```
    === REQUIREMENTS ===
    ```
    httpx
    ```

``extract_artefacts`` accepts that layout, tolerates the ways models drift from
it (different marker punctuation, missing markers, a lone unlabelled block,
prose around the fences), and falls back to the legacy JSON form so responses
from an older prompt — or a model that ignores the instruction — still work.

Everything here is pure and synchronous: no I/O, no model calls, no bus.  That
makes the fragile part of the Forge exhaustively testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# A fenced block: ```python\n<body>\n```  (language tag optional).
_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)(?:```|\Z)",
    re.DOTALL,
)

# Section markers, in the many shapes models actually produce:
#   === TOOL ===   ### TOOL   ## Tool   **TOOL**   TOOL:   # TOOL MODULE
# The keyword is what matters; the decoration around it is noise.
_MARKER_RE = re.compile(
    r"(?im)^[ \t]*[#=*\-]{0,6}[ \t]*"
    r"(?P<key>tool|module|implementation|test|tests|test[ _-]?code|"
    r"requirements?|dependencies|deps|packages|notes?)"
    r"[ \t]*(?:code|module|script|harness)?[ \t]*[#=*\-]{0,6}[ \t]*:?[ \t]*$"
)

_SLOT_BY_KEYWORD = {
    "tool": "tool",
    "module": "tool",
    "implementation": "tool",
    "test": "test",
    "tests": "test",
    "testcode": "test",
    "test_code": "test",
    "requirement": "requirements",
    "requirements": "requirements",
    "dependencies": "requirements",
    "deps": "requirements",
    "packages": "requirements",
    "note": "notes",
    "notes": "notes",
}

# Languages that mean "this fence holds Python source".
_PYTHON_TAGS = {"", "python", "py", "python3"}

# Lines that are never a real pip requirement.
_REQ_NOISE_RE = re.compile(r"(?i)^(none|n/?a|nothing|no requirements?|-+|standard library.*)$")
# A plausible pip requirement token: name plus optional extras/version pin.
_REQ_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?"
                           r"(?:\s*(?:[<>=!~]=?|==)\s*[0-9][A-Za-z0-9._*+-]*)?$")


@dataclass
class ForgeArtefacts:
    """The three things a forge turn must produce."""

    tool_code: str = ""
    test_code: str = ""
    requirements: list[str] = field(default_factory=list)
    notes: str = ""
    # How the artefacts were recovered — "blocks", "json" or "bare".  Recorded
    # on the session so a drift back to the fragile path is visible in logs
    # rather than silent.
    transport: str = "blocks"

    @property
    def complete(self) -> bool:
        return bool(self.tool_code.strip())


def strip_fences(text: str) -> str:
    """Remove a single wrapping code fence, if present.

    Models sometimes double-wrap: a fenced block whose body is itself fenced.
    The extractor calls this on every recovered body so the source that reaches
    ``ast.parse`` is never prefixed with `````python``.
    """
    body = str(text or "").strip()
    match = _FENCE_RE.fullmatch(body)
    if match:
        return match.group(2).strip("\n")
    # A body that merely *starts* with a fence line (unterminated).
    if body.startswith("```"):
        lines = body.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        return "\n".join(lines).strip("\n")
    return body


def _slot_for(marker_text: str) -> str | None:
    """Map a marker keyword onto an artefact slot."""
    key = re.sub(r"[^a-z]", "", str(marker_text or "").lower())
    return _SLOT_BY_KEYWORD.get(key)


def _last_marker_before(text: str) -> str | None:
    """The slot named by the LAST section marker in *text*, if any.

    Only the last one counts: the preamble before a fence may mention several
    words ("here is the TOOL, and below it the TEST"), and the marker nearest
    the fence is the one that labels it.
    """
    slot: str | None = None
    for match in _MARKER_RE.finditer(text):
        candidate = _slot_for(match.group("key"))
        if candidate is not None:
            slot = candidate
    return slot


def parse_requirements(text: str) -> list[str]:
    """Recover a pip requirement list from free text.

    Accepts a fenced list, a comma-separated line, a bulleted list or a JSON
    array, and discards the ways models say "there are none".
    """
    body = strip_fences(text)
    if not body.strip():
        return []
    # A JSON array is unambiguous — take it verbatim.
    stripped = body.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return _clean_requirements(str(item) for item in parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    tokens: list[str] = []
    for line in body.splitlines():
        line = line.strip().strip("`")
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        if not line:
            continue
        tokens.extend(part.strip() for part in line.split(",") if part.strip())
    return _clean_requirements(tokens)


def _clean_requirements(tokens: Any) -> list[str]:
    out: list[str] = []
    for raw in tokens:
        token = str(raw or "").strip().strip("\"'`,")
        if not token or _REQ_NOISE_RE.match(token):
            continue
        if not _REQ_TOKEN_RE.match(token):
            continue
        if token not in out:
            out.append(token)
    return out


def _looks_like_module(code: str) -> bool:
    """True when the source exposes the tool contract's entry points."""
    return "def get_tool_schema" in code and "def run" in code


def _looks_like_test(code: str) -> bool:
    """True when the source reads as a verification harness rather than a tool."""
    if _looks_like_module(code):
        return False
    markers = ("assert", "def test_", "import unittest", "pytest",
               "get_tool_schema()", "importlib")
    return any(marker in code for marker in markers)


def extract_from_blocks(raw: str) -> ForgeArtefacts:
    """Recover artefacts from a fenced-block response.

    Each fence is labelled by the last section marker appearing between it and
    the previous fence.  Unlabelled fences are assigned by *shape*: a block
    exporting ``get_tool_schema``/``run`` is the module, a block full of
    assertions is the test.  Anything still unassigned fills the first empty
    slot in order, which is what a model that simply emitted two bare blocks
    meant anyway.
    """
    text = str(raw or "")
    artefacts = ForgeArtefacts(transport="blocks")
    cursor = 0
    labelled: dict[str, str] = {}
    unlabelled: list[tuple[str, str]] = []      # (language, body)

    for match in _FENCE_RE.finditer(text):
        preamble = text[cursor:match.start()]
        cursor = match.end()
        language = (match.group(1) or "").strip().lower()
        body = match.group(2).strip("\n")
        if not body.strip():
            continue
        slot = _last_marker_before(preamble)
        if slot in {"tool", "test", "requirements", "notes"} and slot not in labelled:
            labelled[slot] = body
        else:
            # Either unlabelled, or a repeat of a section already claimed —
            # keep it as a candidate so shape-based assignment can still place
            # it rather than silently dropping a block the model did produce.
            unlabelled.append((language, body))

    # Shape-based assignment for anything the markers did not claim.
    for language, body in unlabelled:
        if language not in _PYTHON_TAGS:
            continue
        if "tool" not in labelled and _looks_like_module(body):
            labelled["tool"] = body
        elif "test" not in labelled and _looks_like_test(body):
            labelled["test"] = body
    for language, body in unlabelled:
        if body in labelled.values():
            continue
        if language in _PYTHON_TAGS and "tool" not in labelled:
            labelled["tool"] = body
        elif language in _PYTHON_TAGS and "test" not in labelled:
            labelled["test"] = body
        elif "requirements" not in labelled and language not in _PYTHON_TAGS:
            labelled["requirements"] = body

    artefacts.tool_code = strip_fences(labelled.get("tool", ""))
    artefacts.test_code = strip_fences(labelled.get("test", ""))
    artefacts.requirements = parse_requirements(labelled.get("requirements", ""))
    artefacts.notes = strip_fences(labelled.get("notes", ""))[:800]
    return artefacts


def extract_from_json(raw: str) -> ForgeArtefacts:
    """Legacy transport: a JSON object carrying the source as string values.

    Retained so a model that ignores the block instruction, or a stored
    response from the previous prompt, still forges.  The mechanical repair for
    unescaped newlines lives in ``forge.LlmForgeBrain._repair_json``; this
    function only deals with well-formed or brace-wrapped JSON.
    """
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        return ForgeArtefacts(
            tool_code=strip_fences(str(data.get("tool_code") or "")),
            test_code=strip_fences(str(data.get("test_code") or "")),
            requirements=_clean_requirements(data.get("requirements") or []),
            notes=str(data.get("notes") or "")[:800],
            transport="json",
        )
    return ForgeArtefacts(transport="json")


def extract_artefacts(raw: str) -> ForgeArtefacts:
    """Recover forge artefacts from a model response, however it is shaped.

    Order of preference:
      1. fenced blocks (the format the prompt asks for — deterministic);
      2. legacy JSON (an older prompt, or a model that insisted on JSON);
      3. bare source (no fences at all, but the contract's entry points are
         visibly present, so the whole response *is* the module).
    """
    text = str(raw or "")
    if not text.strip():
        return ForgeArtefacts(transport="blocks")

    if "```" in text:
        artefacts = extract_from_blocks(text)
        # Anything recovered counts, not just a module: a repair aimed at the
        # TEST legitimately returns a test block and no module, and discarding
        # that as "incomplete" threw away a perfectly correct response.
        if artefacts.tool_code.strip() or artefacts.test_code.strip():
            return artefacts

    if "{" in text and '"tool_code"' in text:
        artefacts = extract_from_json(text)
        if artefacts.tool_code.strip() or artefacts.test_code.strip():
            return artefacts

    if _looks_like_module(text):
        return ForgeArtefacts(tool_code=text.strip(), transport="bare")

    # Nothing usable — return the empty artefacts so the caller reports a clean
    # "the model returned no tool module" rather than a parser stack trace.
    return ForgeArtefacts(transport="blocks")


def synthesise_test(tool_name: str) -> str:
    """A minimal loader-level test, used when the model supplies none.

    It exercises exactly what the loader will do at activation time — import
    the staged module, read the schema, confirm the entry point is callable —
    so a missing test harness never aborts an otherwise sound forge.
    """
    return (
        "import importlib.util, pathlib, sys\n"
        f"path = pathlib.Path('{tool_name}_tool.py')\n"
        "if not path.exists():\n"
        f"    path = pathlib.Path('{tool_name}.py')\n"
        f"spec = importlib.util.spec_from_file_location('{tool_name}_probe', path)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "schema = mod.get_tool_schema()\n"
        "assert isinstance(schema, dict), 'get_tool_schema() must return a dict'\n"
        "assert schema.get('name'), 'schema needs a name'\n"
        "assert callable(mod.run), 'run must be callable'\n"
        "print('schema + entry point OK')\n"
    )


__all__ = [
    "ForgeArtefacts",
    "extract_artefacts",
    "extract_from_blocks",
    "extract_from_json",
    "parse_requirements",
    "strip_fences",
    "synthesise_test",
]
