"""Invasive diagnostics need explicit coverage and a disposable interpreter."""
import asyncio
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from orion_core import diagnostic_imports as isolated
from orion_core import diagnostics


@pytest.fixture
def package(tmp_path, monkeypatch):
    directory = tmp_path / "synthetic_diagnostic_package"
    directory.mkdir()
    (directory / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(diagnostics, "PACKAGE_DIR", directory)
    yield directory
    for name in list(sys.modules):
        if name.startswith(directory.name):
            sys.modules.pop(name, None)


def engine():
    return diagnostics.DiagnosticsEngine(bus=None, memory=None)


def test_empty_source_tree_is_not_a_successful_compile_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "PACKAGE_DIR", tmp_path)
    result = engine()._check_compile()
    assert result.status == "WARN" and "0" in result.detail


def test_frozen_loose_source_compilation_never_claims_the_complete_package(package, monkeypatch):
    (package / "loose.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "pyimod02_importers", SimpleNamespace(
        pyz_archive=SimpleNamespace(toc={"orion_core": (), "orion_core.hidden": ()})))
    result = engine()._check_compile()
    assert result.status == "WARN"
    assert "2" in result.detail and "loose" in result.detail and "every module" not in result.detail


def test_import_initializers_never_enter_the_live_interpreter(package, tmp_path, monkeypatch):
    marker = tmp_path / "initialized.json"
    monkeypatch.delenv("SYNTHETIC_DIAGNOSTIC_EFFECT", raising=False)
    (package / "cold.py").write_text(
        "import os, json\nfrom pathlib import Path\n"
        "os.environ['SYNTHETIC_DIAGNOSTIC_EFFECT']='child-only'\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid()}}))\n", encoding="utf-8")
    result = engine()._check_imports()
    assert result.status == "PASS", result.detail
    assert marker.exists() and json.loads(marker.read_text())["pid"] != os.getpid()
    assert f"{package.name}.cold" not in sys.modules
    assert "SYNTHETIC_DIAGNOSTIC_EFFECT" not in os.environ


def test_child_does_not_inherit_secrets_or_personal_configuration(package, tmp_path, monkeypatch):
    marker = tmp_path / "child_environment.json"
    monkeypatch.setenv("SYNTHETIC_API_KEY", "synthetic-do-not-forward")
    monkeypatch.setenv("ORION_CONFIG_DIR", str(tmp_path / "personal"))
    old_appdata = os.environ.get("APPDATA")
    (package / "inspect_env.py").write_text(
        "import os,json\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{k:os.environ.get(k) for k in "
        "['SYNTHETIC_API_KEY','ORION_CONFIG_DIR','APPDATA','USERPROFILE']}))\n", encoding="utf-8")
    result = engine()._check_imports()
    assert result.status == "PASS", result.detail
    observed = json.loads(marker.read_text())
    assert observed["SYNTHETIC_API_KEY"] is None
    assert observed["ORION_CONFIG_DIR"] != os.environ["ORION_CONFIG_DIR"]
    assert observed["APPDATA"] != old_appdata and os.environ.get("APPDATA") == old_appdata
    assert not Path(observed["ORION_CONFIG_DIR"]).exists()  # its owned temp work was retired


def test_frozen_inventory_reports_actual_pyz_membership_without_loose_source(package, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "pyimod02_importers", SimpleNamespace(
        pyz_archive=SimpleNamespace(toc={"orion_core": (), "orion_core.hidden": (), "other": ()})))
    result = isolated.inventory(package)
    assert result["kind"] == "frozen" and result["modules"] == ["orion_core", "orion_core.hidden"]


def test_unavailable_frozen_inventory_is_warning_instead_of_vacuous_success(package, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "pyimod02_importers", SimpleNamespace())
    result = engine()._check_imports()
    assert result.status == "WARN" and "inventory" in result.detail.lower()


def test_worker_only_dependency_presence_does_not_count_as_an_import(package):
    (package / "training.py").write_text("import torch\nraise RuntimeError('must not import training')\n")
    (package / "indirect.py").write_text("from . import training\n")
    result = isolated.check(package)
    assert result["status"] == "WARN", result
    assert result["code_checked"] == 3 and result["imported"] == 1
    assert {row["module"] for row in result["dependency_only"]} == {
        f"{package.name}.training", f"{package.name}.indirect"}


def test_large_module_extended_arguments_preserve_relative_worker_dependencies(package):
    (package / "training.py").write_text("import torch\nraise RuntimeError('do not import training')\n")
    large = "\n".join(f"value_{index} = 'synthetic constant {index}'" for index in range(800))
    (package / "indirect.py").write_text(large + "\nfrom . import training\n")
    result = isolated.check(package)
    assert result["status"] == "WARN", result
    assert result["imported"] == 1 and len(result["dependency_only"]) == 2


def test_inactive_invalid_relative_import_is_not_a_scanner_crash(package):
    (package / "conditional.py").write_text(
        "import sys\nif sys.platform == 'orion-nonexistent-platform':\n    from ......missing import value\n")
    result = isolated.check(package)
    assert result["status"] == "PASS", result
    assert result["imported"] == result["code_checked"] == 2


def test_real_child_import_failure_is_named_and_fails_the_check(package):
    (package / "broken.py").write_text("raise ValueError('synthetic import fault')\n")
    result = isolated.check(package)
    assert result["status"] == "FAIL"
    assert "broken" in result["detail"] and "synthetic import fault" in result["detail"]


def test_child_network_admission_is_disabled(package):
    (package / "network.py").write_text(
        "import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=.1)\n")
    result = isolated.check(package)
    assert result["status"] == "FAIL" and "Network access is disabled" in result["detail"]


def test_hung_child_has_a_bounded_failure_instead_of_blocking_the_parent(package):
    (package / "hang.py").write_text("import time\ntime.sleep(30)\n")
    result = isolated.check(package, timeout=0.3)
    assert result["status"] == "FAIL" and "timeout" in result["detail"]


def test_output_flood_is_a_failure_instead_of_truncated_success(package):
    (package / "flood.py").write_text("print('x'*(4*1024*1024+1))\n")
    result = isolated.check(package)
    assert result["status"] == "FAIL" and "output limit" in result["detail"].lower()


@pytest.mark.parametrize("change", ["extra_status", "negative_count", "duplicate_exclusion", "unknown_worker"])
def test_malformed_child_coverage_cannot_override_the_parent_verdict(package, monkeypatch, change):
    from orion_core.sandbox import SandboxVerificationHarness
    result = {"kind": "source", "inventory_count": 1, "code_checked": 1,
              "imported": 1, "failures": [], "excluded": [], "dependency_only": []}
    if change == "extra_status":
        result["status"] = "PASS"
    elif change == "negative_count":
        result["code_checked"] = -1
    elif change == "duplicate_exclusion":
        result["excluded"] = ["orion_core.app", "orion_core.app"]
    else:
        result["dependency_only"] = [{"module": "unknown", "missing": []}]
    monkeypatch.setattr(SandboxVerificationHarness, "_run_script", lambda *_:
        (isolated._MARKER + json.dumps(result), "", 0, False, False))
    observed = isolated.check(package)
    assert observed["status"] == "FAIL" and "valid complete report" in observed["detail"]


def test_cancelled_diagnostic_does_not_publish_late_success(monkeypatch):
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    events = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None),
                          dashboard_event=SimpleNamespace(emit=lambda *args: events.append(args)))
    diagnostic = diagnostics.DiagnosticsEngine(bus, memory=None)

    def blocked():
        entered.set()
        assert release.wait(5)
        try:
            assert isolated.cancelled()
            return [diagnostics.Check("imports", "PASS", "late result")]
        finally:
            settled.set()

    monkeypatch.setattr(diagnostic, "_run_all", blocked)

    async def run():
        task = asyncio.create_task(diagnostic.run_full())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        assert await asyncio.to_thread(settled.wait, 3)
        assert not events

    asyncio.run(run())


def test_real_import_child_is_reaped_after_cooperative_cancellation(package, tmp_path):
    import psutil
    marker = tmp_path / "owned_child.json"
    (package / "hang.py").write_text(
        "import os,json,time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid()}}))\n"
        "time.sleep(30)\n")
    stop = threading.Event()

    async def run():
        with isolated.cancellation_scope(stop):
            work = asyncio.create_task(asyncio.to_thread(isolated.check, package, timeout=10))
        try:
            async def admitted():
                while not marker.exists():
                    await asyncio.sleep(.01)
            await asyncio.wait_for(admitted(), 5)
        finally:
            stop.set()
        outcome = await asyncio.wait_for(work, 8)
        assert outcome["status"] == "WARN" and "cancelled" in outcome["detail"]
        assert not psutil.pid_exists(json.loads(marker.read_text())["pid"])

    asyncio.run(run())
