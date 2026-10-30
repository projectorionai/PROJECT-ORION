"""
Packaging a trained adapter as a CANDIDATE that ``tools/evaluate_planner.py``
accepts.

That tool is ORION's one promotion gate: it registers a candidate as a new
Ollama tag, evaluates it against the current model, and only a person can then
promote it. It accepts only a complete, unaltered directory:

    <candidate id>/
        adapter/adapter_config.json
        adapter/adapter_model.safetensors     (the LoRA weights)
        training/receipt.json                 (what was trained, how, on what)
        training/dataset_manifest.json        (counts and hashes; no text)
        candidate.json                        (written LAST: every file's hash)

Every file is hashed into ``candidate.json`` and made read-only; the
``adapter_sha256`` there is the hash of that file list, which the registry and
the promotion step re-check. Writing ``candidate.json`` last means a candidate
that exists is complete.

Standard library only: this runs inside the training environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Any

CANDIDATE_FORMAT = "orion-planner-candidate-v1"      # the format evaluate_planner reads
AGENT_KIND = "local-brain-agent"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def new_candidate_id(data_hash: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"loc-{stamp}-{str(data_hash)[:8]}-{uuid.uuid4().hex[:6]}"


def freeze(directory: Path) -> dict[str, str]:
    """Hash every file under *directory* and make each read-only."""
    hashes: dict[str, str] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        hashes[path.relative_to(directory).as_posix()] = sha256_file(path)
        path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return hashes


def make_writable(directory: Path) -> None:
    """Undo ``freeze`` so a candidate can be deleted (read-only files resist rmtree on Windows)."""
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IWRITE)


def write_candidate(root: Path, adapter_dir: Path, *, base_model_id: str, data_hash: str,
                    receipt: dict[str, Any], dataset_manifest: dict[str, Any],
                    plumbing_only: bool = False, candidate_id: str | None = None,
                    extra: dict[str, Any] | None = None) -> Path:
    """Create ``root/<id>/`` from a saved LoRA adapter folder and return it.

    The adapter folder must hold ``adapter_config.json`` and a ``.safetensors``
    file; anything else in it is copied too (and hashed).
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(f"{adapter_dir} has no adapter_config.json")
    if not list(adapter_dir.glob("*.safetensors")):
        raise ValueError(f"{adapter_dir} has no adapter weights (*.safetensors)")
    if not str(base_model_id or "").strip():
        raise ValueError("record which base model this adapter was trained on")
    cid = candidate_id or new_candidate_id(data_hash)
    output = Path(root) / cid
    output.mkdir(parents=True, exist_ok=False)            # reserved: never reused
    try:
        shutil.copytree(adapter_dir, output / "adapter")
        (output / "training").mkdir()
        (output / "training" / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, default=str), encoding="utf-8")
        (output / "training" / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        files = freeze(output)
        record = {
            "format": CANDIDATE_FORMAT, "kind": AGENT_KIND, "candidate_id": cid,
            "created_at": time.time(), "base_model_id": base_model_id.strip(),
            "data_hash": data_hash, "adapter_sha256": sha256_text(_canonical(files)),
            "files": files, "plumbing_only": bool(plumbing_only),
            **(extra or {}),
        }
        temporary = output / f".candidate-{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        target = output / "candidate.json"
        os.replace(temporary, target)
        target.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        make_writable(output)
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


__all__ = ["AGENT_KIND", "CANDIDATE_FORMAT", "freeze", "make_writable", "new_candidate_id",
           "sha256_file", "sha256_text", "write_candidate"]
