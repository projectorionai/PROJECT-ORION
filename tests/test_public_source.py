"""Publication boundaries are tested with synthetic secrets and temporary files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

import build_standalone
from tools.prepare_public_source import (
    PUBLIC_CONFIG_FILES, audit, candidate_paths, eligible_path,
    local_secrets, snapshot_changes, write_archive,
)


def _write(root: Path, name: str, value: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", [
    "config/api_keys.json", "config/memory.db", "config/identity.json",
    "config/knowledge_packs/personal.json", "conversations/session.txt",
    "orion_core/__pycache__/x.pyc", "orion_core/secrets.env",
    ".git/config", "engines/stockfish.exe", "dist/ORION/ORION.exe",
    "android/local.properties", "android/app/build/outputs/app.apk",
    "../outside.py", "C:/outside.py", "orion_core/../../outside.py",
    "orion_core\\module.py",
    "orion_core/.private/module.py", "tests/credentials.json",
])
def test_private_and_generated_paths_are_rejected(name):
    assert not eligible_path(name)


def test_public_defaults_and_entrypoints_are_allowed():
    assert all(eligible_path(p) for p in PUBLIC_CONFIG_FILES)
    assert all(eligible_path(p) for p in [
        "orion.py", ".env.example", ".dockerignore", "assets/orion.ico",
        "deploy/Dockerfile", "deploy/Caddyfile", "tests/test_public_source.py",
    ])


def test_audit_redacts_credentials_and_names_even_in_tests(tmp_path):
    key = "sk-" + "z" * 40
    known = "unique-local-credential-12345"
    name = "tests/test_example.py"
    _write(tmp_path, name, f"# example\nvalue = '{key}'\n# {known}\n# Private Person\n")
    _, findings = audit(tmp_path, [name], secrets=[known], private_terms=["Private Person"])
    assert {f["kind"] for f in findings} == {
        "provider API key", "known local credential", "private term",
    }
    assert findings[0]["line"] == 2
    report = json.dumps(findings)
    assert key not in report and known not in report and "Private Person" not in report


def test_private_terms_do_not_match_unrelated_identifier_parts(tmp_path):
    _write(tmp_path, "orion.py", "_EDGE_MARGIN = 0.06")
    assert audit(tmp_path, ["orion.py"], private_terms=["Margin"])[1] == []


def test_non_utf8_and_binary_source_fail_closed(tmp_path):
    (tmp_path / "orion.py").write_bytes(b"\xff\xfe\0")
    assert audit(tmp_path, ["orion.py"])[1]
    (tmp_path / "orion.py").write_bytes(b"a\0b")
    assert audit(tmp_path, ["orion.py"])[1]


def test_deleted_tracked_files_are_excluded(tmp_path):
    assert audit(tmp_path, ["old_file.py"]) == ({}, [])


def test_export_uses_scanned_bytes_and_has_no_history(tmp_path):
    source = _write(tmp_path, "orion.py", "print('public')\n")
    _write(tmp_path, ".git/config", "private remote")
    payload, findings = audit(tmp_path, ["orion.py"])
    assert not findings
    source.write_text("changed after scanning", encoding="utf-8")
    output = tmp_path / "source.zip"
    write_archive(output, payload)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"ORION/orion.py", "SOURCE_MANIFEST.json"}
        assert archive.read("ORION/orion.py") == payload["orion.py"]
        manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        assert manifest["orion.py"] == hashlib.sha256(payload["orion.py"]).hexdigest()
    with pytest.raises(FileExistsError):
        write_archive(output, payload)


def test_edits_additions_and_removals_during_audit_block_snapshot(tmp_path):
    _write(tmp_path, "orion.py", "original")
    _write(tmp_path, "README.md", "original")
    payload, findings = audit(tmp_path, ["orion.py", "README.md"])
    assert not findings
    assert snapshot_changes(tmp_path, payload, ["orion.py", "README.md"]) == []
    _write(tmp_path, "orion.py", "updated")
    _write(tmp_path, "SECURITY.md", "new")
    (tmp_path / "README.md").unlink()
    assert snapshot_changes(tmp_path, payload, ["orion.py", "README.md", "SECURITY.md"]) == [
        "README.md", "SECURITY.md", "orion.py",
    ]


def test_known_secret_store_only_extracts_sensitive_values(tmp_path):
    path = _write(tmp_path, "secrets.json", json.dumps({
        "providers": {"example": {"api_key": "local-key-value-1234", "model": "example-model-long-name"}},
        "token": "short", "webhook_url": "https://example.invalid/private-hook",
        "api_keys": ["another-key-value-5678"],
    }))
    assert set(local_secrets([path])) == {
        "local-key-value-1234", "another-key-value-5678", "https://example.invalid/private-hook",
    }


def test_candidates_include_new_source_but_respect_ignored_local_data(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    _write(tmp_path, ".gitignore", "config/\n")
    _write(tmp_path, "orion.py", "pass\n")
    _write(tmp_path, "config/api_keys.json", "{}")
    assert candidate_paths(tmp_path) == [".gitignore", "orion.py"]
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    with pytest.raises(ValueError):
        candidate_paths(subfolder)


def test_packager_seeds_only_public_defaults(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in PUBLIC_CONFIG_FILES:
        _write(source, name, "{}" if name.endswith(".json") else "")
    for name in ("config/api_keys.json", "config/memory.db", "config/custom_tools/private.py",
                 "config/knowledge_packs/private.json", "config/browser_profile/cookies"):
        _write(source, name, "private")
    monkeypatch.setattr(build_standalone, "BASE", source)
    output = tmp_path / "app"
    build_standalone._seed_config(output)
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == PUBLIC_CONFIG_FILES
    assert not build_standalone._has_private_config(output)
    _write(output, "config/identity.json", "private")
    assert build_standalone._has_private_config(output)


def test_build_refuses_personalised_installation_before_running_tools(tmp_path, monkeypatch):
    _write(tmp_path, "dist/ORION/config/api_keys.json", "{}")
    monkeypatch.setattr(build_standalone, "BASE", tmp_path)
    monkeypatch.setattr(build_standalone, "_ensure_pyinstaller",
                        lambda: pytest.fail("Build must stop before installing/building anything"))
    assert build_standalone.build() == 1


def test_docker_context_config_allowlist_matches_packager():
    root = Path(__file__).resolve().parents[1]
    rules = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert rules[1] == "**"
    allowed = {line[1:] for line in rules if line.startswith("!config/") and not line.endswith("/")}
    assert allowed == PUBLIC_CONFIG_FILES
