"""Small write-ahead receipts for interrupted single-tool deployments.

A receipt is durable before source replacement. It is removed only after the
new handler and audit record are published. Startup restores incomplete swaps,
provided nobody has subsequently edited the file by hand.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes


def _hash(value: bytes | None) -> str:
    return hashlib.sha256(value).hexdigest() if value is not None else ""


def prepare(path: Path, before: bytes | None, after: bytes) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_]+_tool\.py", path.name) or path.is_symlink():
        raise ValueError("deployment target must be a regular named Forge tool")
    directory = path.parent / ".forge_transactions"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / (path.stem + ".json")
    payload = {"file": path.name, "before": base64.b64encode(before).decode() if before is not None else None,
               "before_hash": _hash(before), "after_hash": _hash(after)}
    encoded = json.dumps(payload).encode("utf-8")
    # This file also claims publication ownership. An exists/replace pair lets
    # two publishers overwrite each other's rollback backup after both observe
    # absence. Exclusive creation has exactly one winner, even across processes.
    try:
        stream = receipt.open("xb")
    except FileExistsError as exc:
        raise RuntimeError("an interrupted deployment must be recovered before replacing this tool") from exc
    claimed_identity: tuple[int, int] | None = None
    try:
        with stream:
            stat = os.fstat(stream.fileno())
            claimed_identity = stat.st_dev, stat.st_ino
            # The caller may have captured its backup while another publisher
            # still held this claim. Refuse a stale preimage after admission.
            if path.is_symlink():
                raise ValueError("deployment target must be a regular named Forge tool")
            current = path.read_bytes() if path.exists() else None
            if current != before:
                raise RuntimeError("deployment source changed since its backup was captured")
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A failed write owns only the file it exclusively created. If recovery
        # or an external publisher replaced it, preserve that newer receipt.
        try:
            stat = receipt.lstat()
            if claimed_identity == (stat.st_dev, stat.st_ino):
                receipt.unlink()
        except OSError:
            pass
        raise
    return receipt


def recover_pending(root: Path, history: Any = None) -> list[dict[str, Any]]:
    results = []
    for receipt in sorted((root / ".forge_transactions").glob("*.json")):
        try:
            if receipt.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("deployment receipt exceeds size limit")
            data = json.loads(receipt.read_text(encoding="utf-8"))
            name = str(data.get("file", ""))
            if not re.fullmatch(r"[a-zA-Z0-9_]+_tool\.py", name) or receipt.stem != Path(name).stem:
                raise ValueError("invalid deployment target")
            target = root / name
            if target.is_symlink():
                raise ValueError("deployment target became a symlink")
            before = base64.b64decode(data["before"], validate=True) if data["before"] is not None else None
            if _hash(before) != data["before_hash"]:
                raise ValueError("deployment backup hash mismatch")
            current = target.read_bytes() if target.exists() else None
            if _hash(current) not in {data["before_hash"], data["after_hash"]}:
                raise ValueError("tool changed after interruption; manual edit preserved")
            if before is None:
                target.unlink(missing_ok=True)
            else:
                atomic_write_bytes(target, before)
            if history is not None:
                history.record(module=f"custom_tools/{name}",
                    before=current.decode("utf-8", errors="replace") if current is not None else None,
                    after=before.decode("utf-8", errors="replace") if before is not None else None,
                    status="rolled_back", reasoning="Recovered an interrupted deployment at startup.",
                    verification={"backup_hash_valid": True})
            receipt.unlink()
            results.append({"file": name, "restored": True})
        except Exception as exc:
            results.append({"file": receipt.stem + ".py", "restored": False, "error": str(exc)})
    return results
