"""Host-side limits for local MCP arguments and subprocess environments.

These checks constrain requests to trusted servers. They do not sandbox a
malicious server, nor eliminate the filesystem check/use race across processes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any

# Include the short names and connection URLs routinely used for credentials.
# Plain URL/PATH names are deliberately left alone for subprocess operation.
_SECRET = re.compile(
    r"key|token|secret|password|passwd|credential|auth|cookie|private|sid|"
    r"(?:^|_)(?:pass|pat|dsn|connection_string|connection_uri|"
    r"(?:database|db|redis|mongo|mongodb|postgres|postgresql|mysql|amqp|broker)_(?:url|uri))"
    r"(?:_|$)",
    re.I,
)
_DEVICE = re.compile(r"^(?:con|conin\$|conout\$|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)", re.I)


def child_environment(spec: dict[str, Any], parent: dict[str, str]) -> dict[str, str]:
    """Explicit server credentials win; unrelated provider secrets stay local."""
    inherited = spec.get("inherit_env") or []
    if not isinstance(inherited, list) or not all(isinstance(k, str) for k in inherited):
        raise ValueError("inherit_env must be a list of environment variable names")
    result = {k: v for k, v in parent.items() if not _SECRET.search(k) or k in inherited}
    explicit = spec.get("env") or {}
    if not isinstance(explicit, dict):
        raise ValueError("server env must be an object")
    result.update({str(k): str(v) for k, v in explicit.items()})
    if any(not k or "=" in k or "\0" in k or "\0" in v for k, v in result.items()):
        raise ValueError("invalid subprocess environment")
    return result


def scoped_path(value: str, roots: list[str]) -> str:
    """Canonical containment, rejecting Windows device/ADS/UNC ambiguities."""
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("path must be a non-empty string without control characters")
    win = PureWindowsPath(value)
    if value.startswith(("\\\\", "//")) or (win.drive and not win.root):
        raise ValueError("UNC, device and drive-relative paths are not permitted")
    tail = value[len(win.drive):] if win.drive else value
    if ":" in tail:
        raise ValueError("alternate data streams are not permitted")
    for part in re.split(r"[/\\]", tail):
        if part == ".." or part.rstrip(" .") != part or _DEVICE.match(part):
            raise ValueError("path traversal or reserved Windows path component")
    bases = [Path(root).expanduser().resolve() for root in roots]
    if not bases:
        raise ValueError("no permitted path roots configured")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if len(bases) != 1:
            raise ValueError("relative paths require exactly one permitted root")
        path = bases[0] / path
    resolved = path.resolve()
    if not any(resolved.is_relative_to(base) for base in bases):
        raise ValueError("path is outside the permitted roots")
    return str(resolved)


def path_policy(spec: dict[str, Any]) -> tuple[str, list[str]]:
    args = [str(a) for a in spec.get("args", [])]
    kind = str(spec.get("path_policy") or "")
    roots = spec.get("allowed_roots") or []
    if not isinstance(roots, list) or not all(isinstance(r, str) and r for r in roots):
        raise ValueError("allowed_roots must contain directory paths")
    if not kind:
        if any(a.split("@", 2)[0] == "mcp-server-git" for a in args):
            kind = "git"
        elif any(a.startswith("@modelcontextprotocol/server-filesystem") for a in args):
            kind = "filesystem"
    if not roots and kind == "filesystem":
        for index, arg in enumerate(args):
            if arg.startswith("@modelcontextprotocol/server-filesystem"):
                roots = args[index + 1:]
                break
    if not roots and kind == "git" and "--repository" in args:
        at = args.index("--repository") + 1
        roots = args[at:at + 1]
    return kind, roots


def validate_arguments(spec: dict[str, Any], arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("MCP arguments must be an object")
    try:
        encoded = json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("MCP arguments must be finite JSON data") from exc
    if len(encoded.encode("utf-8")) > 256 * 1024:
        raise ValueError("MCP arguments exceed 256 KiB")
    kind, roots = path_policy(spec)
    result = dict(arguments)
    if kind not in {"", "filesystem", "git"}:
        raise ValueError("unknown MCP path policy")
    if kind == "filesystem":
        for key in ("path", "source", "destination"):
            if key in result:
                result[key] = scoped_path(result[key], roots)
        if "paths" in result:
            if not isinstance(result["paths"], list):
                raise ValueError("paths must be a list")
            result["paths"] = [scoped_path(p, roots) for p in result["paths"]]
    if kind == "git":
        if "repo_path" in result:
            result["repo_path"] = scoped_path(result["repo_path"], roots)
        for key in ("revision", "target", "branch_name", "base_branch"):
            if key in result and (not isinstance(result[key], str)
                                  or result[key].startswith("-") or "\0" in result[key]):
                raise ValueError("invalid Git revision or branch argument")
        if "files" in result:
            repo = result.get("repo_path")
            if not repo or not isinstance(result["files"], list):
                raise ValueError("Git files require a scoped repo_path and a list")
            result["files"] = [scoped_path(p, [repo]) for p in result["files"]]
    return result
