"""Commands for an opt-in Linux-container verification boundary.

The trusted image is built separately and never pulled implicitly. No source
checkout, credentials, Docker socket or GPU is mounted into the verifier.
"""
import re
import os
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_text


def customer_verification_mode() -> bool:
    """Require an explicit, valid choice before customer source is handled.

    This gate enables verification and review artefacts only. It cannot make
    host imports or later execution of generated code safe.
    """
    value = os.getenv("ORION_FORGE_CUSTOMER_MODE", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("ORION_FORGE_CUSTOMER_MODE must be 0 or 1")
    return value == "1"


def save_review_candidate(directory: Path, name: str, source: str, tests: str,
                          verification: dict[str, Any]) -> Path:
    """Keep a verified candidate as inert JSON, outside the tool load path."""
    if not re.fullmatch(r"[a-z0-9_]{1,60}", name):
        raise ValueError("invalid Forge review candidate name")
    if directory.is_symlink():
        raise ValueError("Forge review directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{uuid.uuid4().hex}.json"
    receipt = {
        "format": "orion-forge-review-v1", "tool_name": name,
        "activated": False, "dependencies_installed": False,
        "source": source, "tests": tests,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "tests_sha256": hashlib.sha256(tests.encode("utf-8")).hexdigest(),
        "verification": {key: verification.get(key) for key in (
            "passed", "stage", "isolation", "typecheck", "exit_code", "duration_ms",
            "timed_out", "memory_exceeded", "timeout_s", "examples_verified", "plan")},
    }
    atomic_write_text(path, json.dumps(receipt, ensure_ascii=False, indent=2))
    return path


def container_command(stage: Path, script: str, args: tuple[str, ...], *,
                      image: str, name: str, memory_mb: int = 512) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,250}", image):
        raise ValueError("invalid Forge verifier image")
    if not re.fullmatch(r"orion-forge-[a-f0-9]{32}", name):
        raise ValueError("invalid verifier container name")
    root = str(stage.resolve())
    if any(c in root for c in (",", "\n", "\r")):
        raise ValueError("staging path cannot contain Docker mount delimiters")
    if Path(script).name != script or not re.fullmatch(r"[A-Za-z0-9_]+\.py", script):
        raise ValueError("verifier script must be a staged Python filename")
    cap = max(128, min(int(memory_mb), 2048))
    return ["docker", "run", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--pids-limit=64",
            f"--memory={cap}m", f"--memory-swap={cap}m", "--cpus=1",
            "--user=65534:65534", "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m",
            "--mount", f"type=bind,source={root},target=/work,readonly",
            "--workdir=/work", "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=PYTHONPATH=/work", "--env=PYTHONIOENCODING=utf-8",
            "--env=PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", image,
            "python", f"/work/{script}", *args]
