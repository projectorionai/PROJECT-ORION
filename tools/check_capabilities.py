"""Fail a release when a recorded tool, page, plugin or fallback disappears.

Static inspection never starts ORION, loads plugins or accesses user accounts.
This inventory complements behavioural tests; presence is not proof of function.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/capability_baseline.json"


def _tree(root: Path, name: str) -> ast.Module:
    return ast.parse((root / name).read_bytes(), filename=name)


def inventory(root: Path, reference: dict) -> dict[str, list[str]]:
    from tools.prepare_public_source import PUBLIC_CONFIG_FILES
    found = {key: [] for key in ("tools", "handlers", "pages", "plugins", "entrypoints", "fallbacks", "local_plugins")}
    tree = _tree(root, "orion_core/dispatch_schema.py")
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "TOOL_DECLARATIONS":
            found["tools"] = [row["name"] for row in ast.literal_eval(node.value)]
    for node in ast.walk(_tree(root, "orion_core/dispatcher.py")):
        if isinstance(node, ast.FunctionDef) and node.name == "handler_table":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    found["handlers"].extend(f"{key.value}:{value.attr}" for key, value in zip(child.keys, child.values)
                        if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Attribute))
    for node in ast.walk(_tree(root, "orion_core/app.py")):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "UnifiedDashboard":
            if len(node.args) > 1 and isinstance(node.args[1], (ast.List, ast.Tuple)):
                found["pages"] = [item.elts[0].value for item in node.args[1].elts
                                  if isinstance(item, (ast.Tuple, ast.List)) and isinstance(item.elts[0], ast.Constant)]
    public_manifests = {p for p in PUBLIC_CONFIG_FILES if p.endswith(".plugin.json")}
    for path in sorted((root / "config/custom_tools").glob("*.plugin.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        name, module = data.get("name"), data.get("module")
        if not isinstance(name, str) or not isinstance(module, str):
            raise ValueError(f"Invalid plugin identity: {path.name}")
        target = path.parent / module
        if target.resolve().parent != path.parent.resolve():
            raise ValueError(f"Plugin escapes its directory: {path.name}")
        if not target.is_file():
            continue
        functions = {n.name for n in ast.parse(target.read_bytes()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not {"get_tool_schema", "run"} <= functions:
            continue
        key = "plugins" if path.relative_to(root).as_posix() in public_manifests else "local_plugins"
        found[key].append(f"{name}:{module}")
    for path in reference.get("entrypoints", []):
        if (root / path).is_file():
            _tree(root, path)
            found["entrypoints"].append(path)
    for item in reference.get("fallbacks", []):
        path, symbol = item.split(":", 1)
        if not (root / path).is_file():
            continue
        tree = _tree(root, path)
        parts = symbol.split(".")
        scope = tree.body
        for part in parts:
            match = next((n for n in scope if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part), None)
            if match is None:
                break
            scope = match.body
        else:
            found["fallbacks"].append(item)
    return {key: sorted(values) for key, values in found.items()}


def compare(expected: dict, actual: dict, *, include_local=False) -> dict:
    missing, added, duplicates = {}, {}, {}
    for key in actual:
        if key == "local_plugins" and not include_local:
            continue
        before, after = set(expected.get(key, [])), set(actual[key])
        if before - after: missing[key] = sorted(before - after)
        if after - before: added[key] = sorted(after - before)
        repeated = sorted(value for value in after if actual[key].count(value) > 1)
        if repeated: duplicates[key] = repeated
    return {"ok": not missing and not duplicates, "missing": missing, "added": added, "duplicates": duplicates}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--include-local", action="store_true", help="Also require retained local-only plugins")
    parser.add_argument("--write-candidate", type=Path, help="Write a separate proposed inventory for review; does not change the baseline")
    args = parser.parse_args(argv)
    try:
        expected = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
        current = inventory(args.root.resolve(), expected)
        result = compare(expected, current, include_local=args.include_local)
        if args.write_candidate:
            with args.write_candidate.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(current, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": f"Capability inventory could not be read: {type(exc).__name__}"}))
        return 2


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
