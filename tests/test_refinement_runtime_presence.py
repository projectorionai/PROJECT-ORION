"""Shallow dependency checks must not execute optional package initializers."""

import importlib
import importlib.machinery
import sys
import types

import pytest


@pytest.fixture(params=["diagnostics", "capability_health"])
def presence(request, monkeypatch):
    module = importlib.import_module(f"orion_core.{request.param}")
    if request.param == "capability_health":
        monkeypatch.setattr(module, "_IMPORTABLE", {})
        probe = module._importable
    else:
        probe = module._module_present
    yield probe
    for name in list(sys.modules):
        if name.startswith("refinement_"):
            sys.modules.pop(name, None)


def test_dotted_dependency_does_not_execute_its_parent(tmp_path, monkeypatch, presence):
    package = tmp_path / "refinement_optional_native"
    package.mkdir()
    marker = tmp_path / "initializer-ran"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('native initialization must not run in a presence probe')\n",
        encoding="utf-8",
    )
    (package / "backend.py").write_text("raise RuntimeError('do not import me')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert presence("refinement_optional_native.backend") is True
    assert not marker.exists()
    assert "refinement_optional_native" not in sys.modules
    assert "refinement_optional_native.backend" not in sys.modules


def test_namespace_children_and_missing_children(tmp_path, monkeypatch, presence):
    package = tmp_path / "refinement_namespace"
    package.mkdir()
    (package / "backend.py").write_text("raise RuntimeError('do not import me')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert presence("refinement_namespace.backend") is True
    assert presence("refinement_namespace.absent") is False
    assert "refinement_namespace" not in sys.modules


def test_loaded_specless_modules_and_blocked_entries(monkeypatch, presence):
    monkeypatch.setitem(sys.modules, "refinement_loaded", types.ModuleType("refinement_loaded"))
    monkeypatch.setitem(sys.modules, "refinement_blocked", None)
    assert presence("refinement_loaded") is True
    assert presence("refinement_blocked") is False
    assert presence("refinement_loaded.child") is False


def test_loaded_package_actual_path_is_used_without_module_getattr(tmp_path, monkeypatch, presence):
    (tmp_path / "backend.py").write_text("raise RuntimeError('do not import me')\n", encoding="utf-8")
    package = types.ModuleType("refinement_loaded_package")
    package.__path__ = [str(tmp_path)]
    package.__getattr__ = lambda *_: pytest.fail("discovery must not invoke package lazy attributes")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    assert presence("refinement_loaded_package.backend") is True
    assert "refinement_loaded_package.backend" not in sys.modules


def test_already_loaded_child_does_not_require_reimporting_a_parent(monkeypatch, presence):
    monkeypatch.setitem(sys.modules, "refinement_detached", None)
    monkeypatch.setitem(sys.modules, "refinement_detached.child",
                        types.ModuleType("refinement_detached.child"))
    assert presence("refinement_detached.child") is True


def test_meta_path_package_discovery_preserves_frozen_importer_support(monkeypatch, presence):
    calls = []

    class Finder:
        def find_spec(self, name, path=None, target=None):
            if name == "refinement_frozen":
                calls.append((name, path))
                spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                spec.submodule_search_locations = ["frozen://refinement"]
                return spec
            if name == "refinement_frozen.backend":
                calls.append((name, path))
                assert path == ["frozen://refinement"]
                return importlib.machinery.ModuleSpec(name, loader=None)
            return None

    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    assert presence("refinement_frozen.backend") is True
    assert [name for name, _ in calls] == ["refinement_frozen", "refinement_frozen.backend"]
    assert "refinement_frozen" not in sys.modules
