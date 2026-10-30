"""Reproducible source inventory. Does not read credentials or execute tools."""

from __future__ import annotations

import ast
import ipaddress
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
BLOCKING = {"time.sleep", "subprocess.run", "subprocess.check_output", "subprocess.call",
            "subprocess.check_call", "os.system", "urllib.request.urlopen", "urlopen"}


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


class AsyncCalls(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function = ""
        self.results: list[dict] = []
        self.calls: list[ast.Call] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        previous, self.function = self.function, node.name
        for child in node.body:
            self.visit(child)
        self.function = previous

    def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
        # A nested synchronous helper executes when called, not when defined.
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if any(dotted(call.func).endswith(("to_thread", "run_in_executor")) for call in self.calls):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted(node.func)
        if self.function and (name in BLOCKING or re.fullmatch(
                r"requests\.(?:get|post|put|patch|delete|head|request)", name)):
            self.results.append({"line": node.lineno, "function": self.function, "call": name})
        self.calls.append(node)
        self.generic_visit(node)
        self.calls.pop()


def audit() -> dict:
    files = sorted({*ROOT.joinpath("orion_core").rglob("*.py"),
                    *ROOT.joinpath("android", "app", "src", "main").rglob("*.kt"),
                    *ROOT.joinpath("android", "app", "src", "main").rglob("*.xml"),
                    *ROOT.joinpath("deploy").glob("*.ps1")})
    addresses, calls, syntax = [], [], []
    for path in files:
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        for line, text in enumerate(source.splitlines(), 1):
            for match in IPV4.finditer(text):
                try:
                    ip = ipaddress.ip_address(match.group())
                except ValueError:
                    continue
                kind = ("loopback" if ip.is_loopback else "wildcard" if ip.is_unspecified
                        else "CGNAT" if ip in ipaddress.ip_network("100.64.0.0/10")
                        else "private" if ip.is_private else "public or example")
                addresses.append({"file": relative, "line": line, "address": str(ip), "kind": kind})
        if path.suffix == ".py":
            try:
                tree = ast.parse(source, filename=relative)
            except SyntaxError as exc:
                syntax.append({"file": relative, "line": exc.lineno, "error": exc.msg})
                continue
            visitor = AsyncCalls()
            visitor.visit(tree)
            calls.extend({"file": relative, **finding} for finding in visitor.results)
    return {"files_checked": len(files), "syntax_errors": syntax,
            "hardcoded_ipv4": addresses, "direct_blocking_calls_in_async_functions": calls,
            "scope": "Direct calls only; review indirect synchronous helpers separately. Package versions are not IP addresses."}


def main() -> int:
    result = audit()
    destination = ROOT / "docs" / "BACKEND_SOURCE_INVENTORY_2026-09-28.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "files_checked": result["files_checked"],
                      "syntax_errors": len(result["syntax_errors"]),
                      "address_locations": len(result["hardcoded_ipv4"]),
                      "blocking_calls": result["direct_blocking_calls_in_async_functions"]}, indent=2))
    return bool(result["syntax_errors"])


if __name__ == "__main__":
    sys.exit(main())
