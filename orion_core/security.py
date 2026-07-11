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
    """Regex firewall for all operating system action payloads."""

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
        re.compile(r"(?i)\b(?:attrib|compact|cipher)\b\s+.*\b(?:/s|/w)\b"),
    )
    # Both the launcher (orion.py) and any file inside orion_core/ are core code.
    CORE_MUTATION_RE = re.compile(
        r"(?i)\b(?:del|erase|rm|move|ren|rename|copy|write|append|truncate|overwrite|remove|delete|replace)\b"
        r".*\b(?:orion\.py|orion_core)\b"
        r"|\b(?:orion\.py|orion_core)\b"
        r".*\b(?:del|erase|rm|move|ren|rename|copy|write|append|truncate|overwrite|remove|delete|replace)\b"
    )

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
        if cls.CORE_MUTATION_RE.search(candidate):
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = cls._ast_call_name(node.func)
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
