"""
Files that ship with ORION must be findable once he is packaged.

This is a class of bug that never raises. A bundled asset that cannot be
located is simply absent, whatever depended on it falls back to its second
choice, and everything carries on looking fine — so it is only ever noticed
by someone comparing what they expected to see with what is on screen.

It has now happened twice in one afternoon, both times to the same file:

  * ``.gitignore`` had an unanchored ``build/``, which matches ANY directory
    of that name at any depth. ``assets/three/build/`` is one, so the vendored
    WebGL library was silently never committed.
  * The staging step of the standalone build skipped any path with a part
    named ``build``. Same file, same reason, and again the only symptom was
    that it was not there afterwards.

And a third waiting to happen: ``BASE_DIR`` is the executable's own folder in
a frozen build, but PyInstaller unpacks bundled data into ``_internal/``. Any
lookup of ``BASE_DIR / "assets" / …`` finds nothing once packaged, and ORION's
avatar would have streamed three.js from a CDN in every standalone build while
the vendored copy sat unused inside the bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.constants import RESOURCE_DIR, resource_path

#: Everything ORION needs at runtime that is not code. Named individually
#: rather than globbed, so deleting one fails here rather than at launch.
SHIPPED = (
    ("assets", "three", "build", "three.module.js"),
    ("assets", "three", "examples", "jsm", "postprocessing",
     "EffectComposer.js"),
    ("assets", "three", "examples", "jsm", "postprocessing", "RenderPass.js"),
    ("assets", "three", "examples", "jsm", "postprocessing",
     "UnrealBloomPass.js"),
)


@pytest.mark.parametrize("parts", SHIPPED, ids=lambda p: "/".join(p[1:]))
def test_the_asset_is_present_in_the_source_tree(parts):
    assert resource_path(*parts).is_file(), (
        f"{'/'.join(parts)} is missing — the avatar will fall back to a CDN")


def test_resource_path_is_not_the_writable_directory():
    """Two different roots that are the same in a checkout and diverge once
    packaged. Conflating them is the whole bug."""
    from orion_core.constants import BASE_DIR, CONFIG_DIR

    assert CONFIG_DIR == BASE_DIR / "config"
    # In a checkout these coincide; the point is that the resource lookup
    # goes through its own function rather than assuming they always will.
    assert RESOURCE_DIR.exists()


def test_a_missing_resource_returns_a_path_rather_than_raising():
    """Callers check `.exists()`; raising here would turn an absent optional
    asset into a crash at import time."""
    missing = resource_path("assets", "definitely-not-here.bin")
    assert isinstance(missing, Path)
    assert not missing.exists()


def test_the_avatar_finds_three_js():
    from orion_core.gui.face3d import local_three

    assert local_three() is not None, (
        "the vendored three.js is not discoverable; the avatar would stream "
        "it from a CDN and have no face offline")


def test_the_avatar_looks_through_resource_path_not_base_dir():
    """The frozen-build trap, asserted in the source.

    BASE_DIR is the executable's folder; PyInstaller puts bundled data in
    _internal/. A lookup through BASE_DIR finds nothing once packaged, and
    says nothing about it.
    """
    import ast

    source = (ROOT / "orion_core" / "gui" / "face3d.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "local_three")
    # Names actually USED, from the AST — a comment saying "not BASE_DIR" is
    # documentation, and a text search flags it as the very thing it warns
    # against.
    used = {node.id for node in ast.walk(function)
            if isinstance(node, ast.Name)}
    used |= {alias.name for node in ast.walk(function)
             if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "resource_path" in used
    assert "BASE_DIR" not in used, (
        "local_three uses BASE_DIR, which is wrong in a frozen build")


def test_the_page_prefers_the_local_copy(monkeypatch):
    from orion_core.gui import face3d

    assert face3d.local_three() is not None
    # Rewriting requires a registered loader as well as complete files.
    # Another GUI test may already have created QApplication before this
    # module is imported. Actual early registration is checked fresh below.
    monkeypatch.setattr(face3d, "_SCHEME_REGISTERED", True)
    page = face3d.three_sourced(face3d.FACE_HTML)
    assert "__THREE_BASE__" not in page, "the placeholder was left unresolved"
    assert face3d.CDN_BASE not in page, (
        "the page is still pointing at the CDN even though a local copy "
        "exists")


def test_local_files_without_a_registered_loader_use_the_cdn(monkeypatch):
    from orion_core.gui import face3d

    assert face3d.local_three() is not None
    monkeypatch.setattr(face3d, "_SCHEME_REGISTERED", False)
    page = face3d.three_sourced(face3d.FACE_HTML)
    assert "__THREE_BASE__" not in page
    assert face3d.CDN_BASE in page


@pytest.mark.parametrize("application_first", [False, True], ids=["early", "late"])
def test_scheme_registration_obeys_fresh_process_startup_order(application_first):
    """Keep the real Qt import-order assertion out of prior GUI test state.

    No WebEngineView is constructed and no Chromium process is started.
    """
    import json
    import os
    import subprocess

    code = "\n".join((
        "import importlib.util, json, sys",
        "from PyQt6.QtWidgets import QApplication",
        "present = importlib.util.find_spec('PyQt6.QtWebEngineWidgets') is not None",
        f"sys.path.insert(0, {str(ROOT)!r})",
        f"app = QApplication([]) if {application_first!r} else None",
        "from orion_core.gui import face3d",
        "print(json.dumps({'present': present, 'webengine': face3d.WEBENGINE_OK,",
        "                  'registered': face3d._SCHEME_REGISTERED,",
        "                  'local': face3d.local_three() is not None,",
        "                  'cdn': face3d.CDN_BASE in face3d.three_sourced(face3d.FACE_HTML)}))",
    ))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if not observed["present"]:
        pytest.skip("Qt WebEngine is not installed")
    assert observed["local"]
    assert observed["webengine"] is (not application_first)
    assert observed["registered"] is (not application_first)
    assert observed["cdn"] is application_first


def test_the_cdn_remains_the_fallback():
    """Vendoring is the preference, not the only route: a build that shipped
    without the assets should still render rather than showing nothing."""
    from orion_core.gui import face3d

    original = face3d.local_three
    try:
        face3d.local_three = lambda: None
        page = face3d.three_sourced(face3d.FACE_HTML)
        assert face3d.CDN_BASE in page
    finally:
        face3d.local_three = original


# ── what the build is told to carry ───────────────────────────────────────────

def test_the_standalone_build_bundles_the_assets_directory():
    source = (ROOT / "build_standalone.py").read_text(encoding="utf-8")
    assert '("assets", "assets")' in source, (
        "the build no longer carries assets/, so a packaged ORION has no "
        "vendored three.js")


def test_the_assets_are_not_excluded_from_the_repository(tmp_path):
    """The first of the two times this happened."""
    import os
    import subprocess

    # A frozen audit copy can itself live beneath an ignored wrapper. Test
    # this copy's shipping rules in a fresh repo, not its parent's rules or
    # the real index (tracked files are otherwise omitted by check-ignore).
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    initialized = subprocess.run(["git", "init", "--quiet"], cwd=tmp_path,
                                 capture_output=True, text=True, timeout=10, check=False)
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    command = ["git", "-c", f"core.excludesFile={os.devnull}", "check-ignore",
               "--no-index", "-v", "assets/three/build/three.module.js"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 1, (
        "assets/three is gitignored — it would not ship with ORION at all: "
        + result.stdout + result.stderr)
    # Control for the original regression: an unanchored build/ rule MUST
    # still flag the nested vendored asset, independent of capture location.
    with (tmp_path / ".gitignore").open("a", encoding="utf-8") as ignore:
        ignore.write("\nbuild/\n")
    control = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False)
    assert control.returncode == 0 and "build/" in control.stdout


def test_offline_whisper_ships_with_its_voice_detector():
    """faster-whisper's code reached the 8 Oct exe but its assets did not, and
    every transcription asks for vad_filter, which loads assets/silero_vad_*.onnx."""
    import build_standalone
    assert "faster_whisper" in build_standalone.COLLECT_ALL


def test_the_lichess_client_ships_with_its_package_metadata():
    """berserk reads its own version from its dist-info on import; the 8 Oct
    exe carried the code without it, so the Lichess bot failed frozen with
    "No package metadata was found for berserk" (parity check, 11 Oct)."""
    import build_standalone
    assert "berserk" in build_standalone.COPY_METADATA
    assert "berserk" in build_standalone.HIDDEN_IMPORTS
