"""
Reading a written dataset back, and checking it has not been altered.

Standard library only (the trainer runs in a separate environment that has
none of ORION's dependencies, and imports this module for exactly that reason).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_FORMAT = "orion-local-brain-dataset-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rows(path: Path | str) -> list[dict[str, Any]]:
    """JSON Lines -> rows. A missing file is an empty list (a build with no
    validation split writes an empty val.jsonl, and a preferences file may be empty)."""
    target = Path(path)
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_manifest(directory: Path | str) -> dict[str, Any]:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def resolve_dataset(value: str | Path, root: Path | None = None) -> Path:
    """A dataset directory from a path, or from 'latest' under *root*."""
    if str(value) == "latest":
        if root is None:
            raise ValueError("'latest' needs the training directory")
        pointer = Path(root) / "latest.txt"
        try:
            name = pointer.read_text(encoding="utf-8").strip()
        except OSError:
            raise ValueError(f"no dataset has been built under {root} yet") from None
        return Path(root) / name
    return Path(value)


def verify_dataset(directory: Path | str) -> list[str]:
    """Problems with a written dataset; an empty list means every file matches its manifest."""
    directory = Path(directory)
    try:
        manifest = read_manifest(directory)
    except (OSError, ValueError):
        return ["no readable manifest.json"]
    problems: list[str] = []
    if manifest.get("format") != DATASET_FORMAT:
        problems.append("not an orion-local-brain dataset")
    files = manifest.get("files") or {}
    for name, info in files.items():
        try:
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError:
            problems.append(f"{name} is missing")
            continue
        if digest != (info or {}).get("sha256"):
            problems.append(f"{name} was changed after it was built")
    if manifest.get("data_hash") != sha256_text(canonical({k: v.get("sha256") for k, v in files.items()})):
        problems.append("data_hash does not match the files")
    return problems


__all__ = ["DATASET_FORMAT", "canonical", "load_rows", "read_manifest", "resolve_dataset",
           "sha256_text", "verify_dataset"]
