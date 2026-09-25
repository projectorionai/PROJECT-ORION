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

from orion_core.constants import RESOURCE_DIR, resource_path  # noqa: E402

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


def test_the_page_prefers_the_local_copy():
    from orion_core.gui.face3d import CDN_BASE, FACE_HTML, three_sourced

    page = three_sourced(FACE_HTML)
    assert "__THREE_BASE__" not in page, "the placeholder was left unresolved"
    assert CDN_BASE not in page, (
        "the page is still pointing at the CDN even though a local copy "
        "exists")


def test_the_cdn_remains_the_fallback():
    """Vendoring is the preference, not the only route: a build that shipped
    without the assets should still render rather than showing nothing."""
    import orion_core.gui.face3d as face3d

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


def test_the_assets_are_not_excluded_from_the_repository():
    """The first of the two times this happened."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "assets/three/build/three.module.js"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0, (
        "assets/three is gitignored — it would not ship with ORION at all")
