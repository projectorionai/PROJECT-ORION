"""Audit the current source and optionally export it without Git history.

This is a conservative publication check, not a proof that content is public.
It never prints matched values. Review the documentation and any new source
before publication. Use --help for local credential/private-term checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

# Reviewed model weights: only these exact bytes may bypass text-only auditing.
BINARY_ASSETS = {
    'assets/face/face_detection_yunet_2023mar.onnx': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    'assets/voiceprint/encoder.onnx': '5374e37c9a7bee1edaf4e4a63f77106a323474b104dac9a06b37ee4132264d14',
    'assets/yamnet/yamnet.onnx': 'd4459cb7fb460e44252ac0408b3718c02ec1286c467a14abaf2ef412fbc0194c',
    'assets/yamnet/yamnet.data': '0b4de0b14bf2b528d51e3cd84e2d3e2ae826b942fbec6001c654605f1f2ceb60',
}

PUBLIC_CONFIG_FILES = frozenset({
    "config/custom_tools/__init__.py",
    "config/custom_tools/homeassistant_call_service_tool.py",
    "config/custom_tools/homeassistant_call_service.plugin.json",
    "config/custom_tools/ifttt_webhook_tool.py",
    "config/custom_tools/ifttt_webhook.plugin.json",
    "config/custom_tools/open_meteo_weather_tool.py",
    "config/custom_tools/open_meteo_weather.plugin.json",
    "config/custom_tools/philips_hue_tool.py",
    "config/custom_tools/philips_hue.plugin.json",
    "config/custom_tools/push_notify_tool.py",
    "config/custom_tools/push_notify.plugin.json",
    "config/custom_tools/dependency_manager_tool.py",
    "config/custom_tools/dependency_manager.plugin.json",
    "config/custom_tools/dependency_resolver_tool.py",
    "config/custom_tools/dependency_resolver.plugin.json",
    "config/custom_tools/emotional_insight_generator_tool.py",
    "config/custom_tools/emotional_insight_generator.plugin.json",
    "config/custom_tools/self_repair_agent_tool.py",
    "config/custom_tools/self_repair_agent.plugin.json",
    "config/telephony_contacts.example.json",
    "config/custom_tools/emotional_monitor_tool.py",
    "config/custom_tools/emotional_monitor.plugin.json",
    "config/custom_tools/error_handler_tool.py",
    "config/custom_tools/error_handler.plugin.json",
    "config/custom_tools/synthesis_enhancer_tool.py",
    "config/custom_tools/synthesis_enhancer.plugin.json",
    "config/custom_tools/fitness_goal_tracker_tool.py",
    "config/custom_tools/fitness_goal_tracker.plugin.json",
    "config/custom_tools/fitness_progress_monitor_tool.py",
    "config/custom_tools/fitness_progress_monitor.plugin.json",
    # Reviewed source plugins; credentials and user-generated tools stay ignored.
    "config/custom_tools/discord_message_tool.py",
    "config/custom_tools/discord_message.plugin.json",
    "config/custom_tools/telegram_message_tool.py",
    "config/custom_tools/telegram_message.plugin.json",
    "config/custom_tools/json_validator_tool.py",
    "config/custom_tools/json_validator.plugin.json",
    "config/custom_tools/live_camera_analysis_tool.py",
    "config/custom_tools/live_camera_analysis.plugin.json",
    *(f"config/knowledge_packs/{name}.json" for name in (
        "ai", "business", "coding", "copywriting", "dropshipping",
        "entrepreneurship", "marketing", "personal_development",
        "sales_psychology", "tiktok_shop",
    )),
})
SOURCE_DIRECTORIES = frozenset({
    ".github", "android", "assets", "deploy", "docs", "launcher",
    "orion_core", "skills", "tests", "tools",
})
ROOT_FILES = frozenset({
    ".env.example", ".gitignore", ".dockerignore", "README.md", "SECURITY.md",
    "LICENSE", "LICENSE.md", "ARCHITECTURE.md", "EXECUTION_PLAN.md", "JARVIS_SUBSYSTEMS.md",
    "ORION_CAPABILITIES.md", "orion.py", "requirements.txt", "conftest.py",
    "test_jarvis_subsystems.py", "build_exe.py", "build_setup.py",
    "build_standalone.py",
    # Third-party attribution for redistributed material (Apache-2.0 assets).
    "NOTICE",
})
TEXT_SUFFIXES = frozenset({
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini",
    ".gradle", ".properties", ".pro", ".xml", ".kt", ".service",
    ".html", ".css", ".js", ".sh", ".ps1", ".bat",
})
FORBIDDEN_PARTS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".idea", ".gradle", "build", "dist", "release", "node_modules",
    "conversations", "browser_profile", "exports", "reports",
})
# High-confidence formats. No broad exclusions for tests or documentation.
SECRET_PATTERNS = {
    "provider API key": re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key material": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\s]{64,}"),
}


def eligible_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        return False
    if name in BINARY_ASSETS or name == "assets/three/build/three.module.js":
        return True
    if any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
        return False
    if name in PUBLIC_CONFIG_FILES or name in ROOT_FILES:
        return True
    if len(path.parts) < 2 or path.parts[0] not in SOURCE_DIRECTORIES:
        return False
    if any(part.startswith(".") and part != ".github" for part in path.parts[:-1]):
        return False
    if path.name.lower() in {"api_keys.json", "credentials.json", "identity.json", "remote_devices.json"}:
        return False
    # Reviewed distributable assets. Kept as an explicit allowlist rather than
    # a suffix rule so a new binary cannot slip in by having a familiar
    # extension — the point of this audit is that additions are noticed.
    if name in {
        "assets/orion.ico",
        # MediaPipe's canonical face geometry, Apache-2.0, attributed in NOTICE.
        # Plain text (vertex and face lines), reviewed, carries no personal data.
        "assets/canonical_face_model.obj",
    }:
        return True
    if name in {"deploy/Caddyfile", "deploy/Caddyfile.hostinger", "deploy/Dockerfile"}:
        return True
    if path.name == ".gitignore":
        return True
    if path.name.startswith(".") or path.name.lower() in {"local.properties", "gradle.properties"}:
        # The checked-in Gradle flags are reviewed separately below.
        return name == "android/gradle.properties"
    return path.suffix.lower() in TEXT_SUFFIXES


def candidate_paths(root: Path) -> list[str]:
    """Include uncommitted source changes, never a parent repository's files."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                         capture_output=True, text=True, check=True)
    if Path(top.stdout.strip()).resolve() != root.resolve():
        raise ValueError("Run this helper at the root of its own Git repository.")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, check=True,
    )
    return sorted(set(result.stdout.decode("utf-8").strip("\0").split("\0")) - {""})


def local_secrets(paths: list[Path]) -> list[str]:
    """Read supplied local secret stores in memory, without logging values."""
    values: set[str] = set()

    def walk(value, sensitive=False):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, sensitive or bool(re.search(
                    r"key|token|password|secret|credential|webhook", str(key), re.I)))
        elif isinstance(value, list):
            for child in value:
                walk(child, sensitive)
        elif sensitive and isinstance(value, str) and len(value.strip()) >= 12:
            values.add(value.strip())

    for path in paths:
        walk(json.loads(path.read_text(encoding="utf-8-sig")))
    return sorted(values)


def audit(root: Path, names: list[str], *, secrets=(), private_terms=()):
    """Return exactly the scanned bytes and value-free findings (fail closed)."""
    root = root.resolve()
    payload: dict[str, bytes] = {}
    findings: list[dict] = []
    for name in sorted(set(names)):
        path = root / name
        # Deleted tracked files are not part of the current source snapshot.
        if not path.exists() and not path.is_symlink():
            continue
        if not eligible_path(name):
            findings.append({"file": name, "kind": "not an approved source path"})
            continue
        if path.is_symlink() or path.resolve() != root / name or not path.is_file():
            findings.append({"file": name, "kind": "redirected or non-regular file"})
            continue
        binary_limit_mb = 16 if name == "assets/yamnet/yamnet.data" else 8
        if path.stat().st_size > (binary_limit_mb if name in BINARY_ASSETS else 5) * 1024 * 1024:
            findings.append({"file": name, "kind": "unexpectedly large source file"})
            continue
        data = path.read_bytes()
        if name in BINARY_ASSETS:
            if hashlib.sha256(data).hexdigest() != BINARY_ASSETS[name]:
                findings.append({"file": name, "kind": "reviewed model digest changed"})
                continue
            for label, values in (("known local credential", secrets), ("private term", private_terms)):
                if any(value and value.encode("utf-8").lower() in data.lower() for value in values):
                    findings.append({"file": name, "kind": label})
            payload[name] = data
            continue
        if name == "assets/orion.ico":
            payload[name] = data
            continue
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            findings.append({"file": name, "kind": "source is not UTF-8 text"})
            continue
        if "\0" in content:
            findings.append({"file": name, "kind": "binary content in text source"})
            continue
        for kind, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(content):
                findings.append({"file": name, "line": content.count("\n", 0, match.start()) + 1,
                                 "kind": kind})
        for label, values in (("known local credential", secrets), ("private term", private_terms)):
            for value in values:
                matched = (value.casefold() in content.casefold() if label == "known local credential"
                           else re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", content, re.I))
                if value and matched:
                    findings.append({"file": name, "kind": label})
        payload[name] = data
    return payload, findings


def snapshot_changes(root: Path, payload: dict[str, bytes], names: list[str]) -> list[str]:
    """Catch edits arriving during an audit before freezing a mixed release."""
    changes = set()
    existing = {name for name in names if (root / name).exists() or (root / name).is_symlink()}
    changes.update(existing.symmetric_difference(payload))
    for name, data in payload.items():
        path = root / name
        try:
            if path.is_symlink() or path.resolve() != root.resolve() / name or path.read_bytes() != data:
                changes.add(name)
        except OSError:
            changes.add(name)
    return sorted(changes)


def write_archive(output: Path, payload: dict[str, bytes]) -> None:
    """Write only previously scanned bytes; refuse to overwrite another file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                archive.writestr("ORION/" + name, data)
            manifest = {name: hashlib.sha256(data).hexdigest()
                        for name, data in sorted(payload.items())}
            archive.writestr("SOURCE_MANIFEST.json",
                             json.dumps(manifest, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new source ZIP after a clean audit")
    parser.add_argument("--secrets-file", action="append", type=Path, default=[],
                        help="Local JSON credential store to check for verbatim leaks; never exported")
    parser.add_argument("--private-terms-file", type=Path,
                        help="Local JSON list of private names/emails to check; never exported")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        terms = json.loads(args.private_terms_file.read_text(encoding="utf-8-sig")) if args.private_terms_file else []
        if not isinstance(terms, list) or not all(isinstance(t, str) and t.strip() for t in terms):
            raise ValueError("Private terms must be a JSON list of non-empty strings.")
        payload, findings = audit(root, candidate_paths(root),
                                  secrets=local_secrets(args.secrets_file), private_terms=terms)
        if findings:
            print(json.dumps({"status": "blocked", "findings": findings}, indent=2))
            return 1
        if not payload or "orion.py" not in payload:
            raise ValueError("The source snapshot is empty or missing orion.py.")
        changed = snapshot_changes(root, payload, candidate_paths(root))
        if changed:
            print(json.dumps({"status": "blocked", "reason": "source changed during audit",
                              "files": changed}, indent=2))
            return 1
        if args.output:
            write_archive(args.output, payload)
        print(json.dumps({"status": "passed", "source_files": len(payload),
                          "git_history_included": False, "archive_created": bool(args.output)}))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError):
        # Exception messages can contain a supplied private filename or value.
        print("Publication check failed. Verify the Git root, input files and unused output path.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
