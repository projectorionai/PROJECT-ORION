"""
Security layer — the regex/AST firewall every operating-system action payload
passes through before execution.

Migrated intact from Mark VII; the core-mutation guard now also protects the
orion_core package directory, not just the orion.py launcher.
"""

from __future__ import annotations

import ast
import re
from typing import Any


class SecurityViolation(Exception):
    """Raised when a command payload violates the local execution policy."""


class SecuritySanitiser:
    """Defence in depth over tool payloads — not a sandbox, and not a proof.

    Two layers. A regex denylist catches destructive *shell* text, and an AST
    pass rejects destructive *Python* calls, resolving import aliases first so
    that ``import os as o; o.remove(x)`` and ``from shutil import rmtree`` are
    treated the same as the literal ``os.remove(x)``.

    **What this is for.** ``guard_text`` runs on short model- and user-supplied
    strings — app names, chess moves, task titles, research topics — none of
    which ORION executes as Python. Its job is to stop a confused or
    prompt-injected model smuggling destructive instructions into a text field.
    It is *not* a containment boundary for code that runs: forged tools take a
    different path (``guard_forged_source`` plus the sandbox), and the only
    sanctioned execution point in the codebase is the forged-tool loader.

    **Known limits, on purpose.** Deliberately obfuscated indirection still
    passes: ``eval`` of a string, ``getattr(os, 'rem' + 'ove')``,
    ``importlib.import_module``, and a call through a bound method. So does any
    payload above the AST size budget, because parsing runs on the qasync loop
    (measured 22 ms at 47 KB, 80 ms at 142 KB) — the regex layer still applies
    at any size. Closing those would be an arms race on a layer whose
    consequence is bounded; ``tests/test_security_sanitiser_adversarial.py``
    records each one as an explicit assertion so the limits cannot quietly
    drift out of date in this docstring.
    """

    DANGEROUS_PATTERNS = (
        re.compile(r"(?i)\b(?:rm|del|erase|rmdir|rd)\b\s+(?:/s|/q|/f|-r|-rf|--recursive)"),
        re.compile(r"(?i)\bformat\b\s+[a-z]:"),
        re.compile(r"(?i)\bdiskpart\b"),
        re.compile(r"(?i)\bbcdedit\b"),
        re.compile(r"(?i)\bmkfs(?:\.[a-z0-9]+)?\b"),
        re.compile(r"(?i)\bdd\b\s+.*\bof\s*=\s*(?:/dev/|\\\\\.\\PhysicalDrive)"),
        re.compile(r"(?i)\breg\b\s+(?:delete|add|import|restore|save)\b"),
        re.compile(r"(?i)\btakeown\b"),
        re.compile(r"(?i)\bicacls\b\s+.*\b(?:grant|deny|reset|remove)\b"),
        re.compile(r"(?i)\bshutdown\b\s+/(?:s|r|g|p|h)"),
        re.compile(
            r"(?i)\bpowershell(?:\.exe)?\b.*\b(?:Remove-Item|Clear-Content|Set-ExecutionPolicy|Stop-Computer)\b"
        ),
        re.compile(r"(?i)\bcmd(?:\.exe)?\b\s*/c\s*(?:del|erase|rd|rmdir|format)\b"),
        re.compile(r"(?i)\bwmic\b\s+.*\bdelete\b"),
        re.compile(r"(?i)>\s*\\\\\.\\PhysicalDrive\d+"),
        # No leading \b before the slash: a word boundary can never sit between
        # a space and '/', so "\b/s" would silently never match "cipher /w:C".
        re.compile(r"(?i)\b(?:attrib|compact|cipher)\b\s+.*(?:/s|/w)\b"),
    )
    # Both the launcher (orion.py) and any file inside orion_core/ are core code.
    # The verb group accepts inflected forms (deleted, deleting, removes …) so
    # "orion.py should be deleted" is caught, not just the bare imperative.
    _MUTATION_VERBS = (
        r"(?:del|erase|rm|move|ren|rename|copy|write|append|truncate"
        r"|overwrite|remove|delete|replace)(?:d|s|ed|es|ing)?"
    )
    CORE_MUTATION_RE = re.compile(
        rf"(?i)\b{_MUTATION_VERBS}\b"
        r".*\b(?:orion\.py|orion_core)\b"
        rf"|\b(?:orion\.py|orion_core)\b"
        rf".*\b{_MUTATION_VERBS}\b"
    )
    # Every CORE_MUTATION_RE match must contain one of these tokens, so when the
    # token is absent the full pattern cannot match and is skipped. Measured on
    # a 55 KB document: 0.40 ms for this versus 4.08 ms for the full pattern,
    # and this guard runs ON the qasync loop. It is a regex with the SAME (?i)
    # flag on purpose, never str.lower(): Python's IGNORECASE also folds the
    # dotless 'ı' and dotted 'İ' onto 'i', so "orıon_core" matches the pattern
    # but is not found by "orion_core" in text.lower() — a lower() prefilter
    # would be a bypass. Joining DANGEROUS_PATTERNS into one alternation was
    # measured as well and gained nothing (13.46 vs 13.41 ms), so they stay.
    _CORE_TOKEN_RE = re.compile(r"(?i)orion(?:\.py|_core)")

    @classmethod
    def guard_text(cls, text: str, context: str = "payload") -> str:
        if not isinstance(text, str):
            return text
        candidate = text.strip()
        if not candidate:
            return text
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(candidate):
                raise SecurityViolation(
                    f"blocked unsafe {context}: destructive shell pattern detected"
                )
        if cls._CORE_TOKEN_RE.search(candidate) and cls.CORE_MUTATION_RE.search(candidate):
            raise SecurityViolation(
                f"blocked unsafe {context}: core script mutation attempt detected"
            )
        cls._guard_python_ast(candidate, context)
        return text

    @classmethod
    def _guard_python_ast(cls, candidate: str, context: str) -> None:
        if not candidate or len(candidate) > 12000:
            return
        try:
            tree = ast.parse(candidate, mode="exec")
        except SyntaxError:
            return
        destructive_calls = {
            "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
            "shutil.rmtree", "pathlib.Path.unlink", "pathlib.Path.rmdir",
            "subprocess.Popen", "subprocess.run", "subprocess.call",
            "subprocess.check_call", "subprocess.check_output",
            "os.system", "os.popen",
        }
        aliases = cls._import_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = cls._resolve_call_name(cls._ast_call_name(node.func), aliases)
                if name in destructive_calls:
                    if name.startswith("subprocess."):
                        joined = " ".join(
                            lit.value for lit in ast.walk(node)
                            if isinstance(lit, ast.Constant) and isinstance(lit.value, str)
                        )
                        if any(p.search(joined) for p in cls.DANGEROUS_PATTERNS) or cls.CORE_MUTATION_RE.search(joined):
                            raise SecurityViolation(
                                f"blocked unsafe {context}: destructive subprocess payload detected"
                            )
                        if any(
                            kw.arg == "shell"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                            for kw in node.keywords
                        ):
                            raise SecurityViolation(
                                f"blocked unsafe {context}: shell-enabled subprocess call detected"
                            )
                    else:
                        raise SecurityViolation(
                            f"blocked unsafe {context}: destructive Python call detected"
                        )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value.strip()
                if cls.CORE_MUTATION_RE.search(literal) or (
                    re.search(r"(?i)\b(?:orion\.py|orion_core)\b", literal)
                    and re.search(r"(?i)\b(?:write|delete|remove|unlink|rename|replace|truncate)\b", candidate)
                ):
                    raise SecurityViolation(
                        f"blocked unsafe {context}: core script mutation attempt detected"
                    )

    @classmethod
    def _import_aliases(cls, tree: ast.AST) -> dict[str, str]:
        """Local name -> the dotted path it really refers to.

        The call check compares a literal dotted name against a denylist, so
        ``os.remove(...)`` was caught and ``import os as o; o.remove(...)`` was
        not — nor was ``from shutil import rmtree; rmtree(...)``. Neither is an
        exotic evasion; they are how people ordinarily write Python, so a model
        producing destructive code would plausibly produce either.
        """
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return aliases

    @classmethod
    def _resolve_call_name(cls, name: str, aliases: dict[str, str]) -> str:
        """Expand a call's leading segment through the import aliases."""
        if not name:
            return name
        head, _, tail = name.partition(".")
        target = aliases.get(head)
        if not target:
            return name
        return f"{target}.{tail}" if tail else target

    @classmethod
    def _ast_call_name(cls, node: ast.AST) -> str:
        parts: list[str] = []
        current: ast.AST | None = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    @classmethod
    def guard_forged_source(cls, source: str, context: str = "forge") -> str:
        """Narrow firewall for AI-GENERATED tool source (forge.py): forged
        tools are meant to be additive and standalone, so this rejects code
        that imports orion_core directly or references orion.py/orion_core
        alongside a mutating verb — the same CORE_MUTATION_RE the rest of the
        app's OS-action payloads are checked against.

        Deliberately narrower than guard_text/_guard_python_ast: it does NOT
        ban os.remove/shutil.rmtree/subprocess wholesale, because a forged
        tool legitimately cleaning up its own scratch files is normal, not a
        core-mutation attempt. Only the orion_core self-reference is in
        scope here.
        """
        if not isinstance(source, str) or not source.strip():
            return source
        if cls.CORE_MUTATION_RE.search(source):
            raise SecurityViolation(
                f"blocked unsafe {context}: references orion_core/orion.py "
                "alongside a mutating verb — forged tools must stay additive"
            )
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError:
            return source
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "orion_core" or alias.name.startswith("orion_core."):
                        raise SecurityViolation(
                            f"blocked unsafe {context}: imports orion_core directly — "
                            "forged tools must stay additive and standalone"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level > 0 or mod == "orion_core" or mod.startswith("orion_core."):
                    raise SecurityViolation(
                        f"blocked unsafe {context}: imports from orion_core directly — "
                        "forged tools must stay additive and standalone"
                    )
        return source

    @classmethod
    def guard_payload(cls, payload: Any, context: str = "payload") -> Any:
        if isinstance(payload, str):
            return cls.guard_text(payload, context)
        if isinstance(payload, dict):
            return {
                cls.guard_text(str(k), context): cls.guard_payload(v, f"{context}.{k}")
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [cls.guard_payload(v, f"{context}[{i}]") for i, v in enumerate(payload)]
        if isinstance(payload, tuple):
            return tuple(cls.guard_payload(v, f"{context}[{i}]") for i, v in enumerate(payload))
        return payload
