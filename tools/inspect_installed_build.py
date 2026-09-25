"""Inspect a PyInstaller build without launching it or accessing devices/accounts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import types


def fingerprint(code: types.CodeType) -> bytes:
    """Ignore source paths/line tables; compare actual compiled implementation."""
    def normalise(value):
        if isinstance(value, types.CodeType):
            fields = ("co_code", "co_consts", "co_names", "co_varnames",
                      "co_freevars", "co_cellvars", "co_flags", "co_argcount",
                      "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
                      "co_stacksize", "co_exceptiontable", "co_name", "co_qualname")
            return ["code", [[key, normalise(getattr(value, key))] for key in fields]]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if isinstance(value, tuple):
            return ["tuple", [normalise(v) for v in value]]
        if isinstance(value, frozenset):
            return ["frozenset", sorted((normalise(v) for v in value), key=repr)]
        if isinstance(value, float):
            return ["float", value.hex()]
        if isinstance(value, complex):
            return ["complex", value.real.hex(), value.imag.hex()]
        if value is Ellipsis:
            return ["ellipsis"]
        return [type(value).__name__, value]
    # marshal encodes object-sharing/interning too. Identical code loaded from
    # a PYZ can have different sharing from compile(), so hash canonical values.
    return hashlib.sha256(json.dumps(normalise(code), ensure_ascii=True,
                                     separators=(",", ":")).encode()).digest()


def inspect_build(executable: Path, root: Path) -> dict:
    from PyInstaller.archive.readers import CArchiveReader
    reader = CArchiveReader(str(executable))
    pyz_name = next((name for name in reader.toc if name.endswith(".pyz")), None)
    if pyz_name is None:
        raise ValueError("The executable has no embedded Python module archive")
    pyz = reader.open_embedded_archive(pyz_name)
    matching, different, missing, unreadable = [], [], [], []
    for path in sorted((root / "orion_core").rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        if module not in pyz.toc:
            missing.append(module)
            continue
        try:
            archived = pyz.extract(module)
            digest = fingerprint(archived)
            options = [fingerprint(compile(path.read_bytes(), "", "exec", optimize=level,
                                           dont_inherit=True)) for level in (0, 1, 2)]
            (matching if digest in options else different).append(module)
        except (ValueError, TypeError, EOFError):
            unreadable.append(module)
    return {"executable": executable.name, "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "launched": False, "hardware_tested": False,
            "source_matches": not different and not missing and not unreadable,
            "matching_modules": matching, "different_modules": different,
            "missing_modules": missing, "unreadable_modules": unreadable}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_build(args.executable, args.source)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k:len(v) if isinstance(v, list) else v for k,v in result.items()}, indent=2))
        return 0 if result["source_matches"] else 1
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps({"source_matches": False, "error": type(exc).__name__, "launched": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
