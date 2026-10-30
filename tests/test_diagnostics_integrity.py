"""Regression checks for diagnostics that inspect a running backend."""

from __future__ import annotations

from types import SimpleNamespace

from orion_core import diagnostics


def _engine(dispatcher=None):
    return diagnostics.DiagnosticsEngine(bus=None, memory=None, dispatcher=dispatcher)


def test_syntax_check_does_not_write_bytecode_into_the_install(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "PACKAGE_DIR", tmp_path)
    module = tmp_path / "example.py"
    module.write_text("answer = 42\n", encoding="utf-8")

    assert _engine()._check_compile().status == "PASS"
    assert not (tmp_path / "__pycache__").exists()

    module.write_text("def broken(:\n", encoding="utf-8")
    check = _engine()._check_compile()
    assert check.status == "FAIL"
    assert "example.py" in check.detail


def test_live_tool_check_reports_a_declared_tool_with_no_route():
    dispatcher = SimpleNamespace(
        TOOL_DECLARATIONS=[{"name": "available"}, {"name": "orphan"}],
        handler_table=lambda: {"available": lambda _: None},
    )
    check = _engine(dispatcher)._check_tools()

    assert check.status == "FAIL"
    assert "orphan" in check.detail


def test_permission_probe_preserves_a_preexisting_file(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "CONFIG_DIR", tmp_path)
    existing = tmp_path / ".diag_write_test"
    existing.write_text("do not replace", encoding="utf-8")

    assert _engine()._check_permissions().status == "PASS"
    assert existing.read_text(encoding="utf-8") == "do not replace"
    assert list(tmp_path.glob(".diag_write_test.*")) == []
