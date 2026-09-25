"""ORION's face must materialise without a network.

He reported "ORION's 3-D face needs an internet connection" on a machine with
working WiFi, and had done ever since the library was vendored. Three separate
faults stacked up, each of which alone was enough to stop the face appearing,
and none of which produced a usable error message:

1. The custom URL scheme was registered on first use. Qt freezes the set of
   schemes when the web engine initialises with QApplication and ignores later
   registrations, so the page was handed a base URL that resolved to nothing.

2. The import map's values were bare — ``build/three.module.js`` rather than
   ``./build/three.module.js``. An import map entry must be a URL or begin
   with /, ./ or ../; a bare value is discarded and the specifier it was meant
   to define then fails as "blocked by a null value".

3. ``examples/jsm/postprocessing/Pass.js`` had never been vendored.
   EffectComposer imports it, so the static import graph failed to resolve
   before a line of the page executed. A failed module graph reports a SCRIPT
   error with no message, no filename and no line — which is why the console
   was empty while the face showed a loading message for ever.

The completeness check that was supposed to catch (3) listed four files by
hand and Pass.js was not one of them, so a demonstrably broken copy passed.
These tests hold the shape that actually matters: the page is self-contained,
its import map is well-formed, and the vendored copy is judged by walking its
graph rather than by a list someone maintained by hand.

No QWebEngineView is constructed here. Importing WebEngine into the test
process wedges later Qt tests, so everything below works on the HTML and the
file tree, which is where all three faults lived anyway.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui import face3d  # noqa: E402


# ── the vendored copy is complete, judged the way a browser judges it ─────────

def test_every_module_the_page_imports_is_present():
    """The check that let Pass.js through.

    Walking the graph is the only honest test: the page imports four modules,
    those import others, and the browser fails on the first one missing —
    whatever a hand-written list says.
    """
    root = face3d.local_three()
    assert root is not None, (
        "the vendored three.js is incomplete or missing; ORION's face would "
        "fall back to streaming from a CDN and would not work offline")


def test_the_module_that_was_missing_is_vendored():
    """Named explicitly, because it is the one that broke it and the one a
    future re-vendoring is most likely to drop again."""
    root = face3d.local_three()
    assert root is not None
    assert (root / "examples/jsm/postprocessing/Pass.js").is_file(), (
        "EffectComposer imports Pass.js; without it the whole import graph "
        "fails before the page runs, with no error message of any kind")


def test_a_missing_dependency_is_refused_rather_than_trusted(tmp_path,
                                                             monkeypatch):
    """A half-copied vendor folder must not present itself as usable.

    It is worse than no copy at all: the import map then points at local paths
    that do not resolve, instead of falling back to the network.
    """
    real = face3d.local_three()
    assert real is not None
    vendor = tmp_path / "three"
    for source in real.rglob("*"):
        if source.is_file():
            target = vendor / source.relative_to(real)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

    assert face3d._graph_resolves(vendor) is True, "the copy itself is broken"
    (vendor / "examples/jsm/postprocessing/Pass.js").unlink()
    assert face3d._graph_resolves(vendor) is False, (
        "a missing transitive import was accepted; the face would fail with "
        "an empty console rather than falling back to the CDN")


def test_the_completeness_check_is_not_a_hand_written_list():
    """The old one was, and it is why this bug shipped.

    Four files were named; Pass.js was not among them, so a copy that could
    not possibly work was declared good.
    """
    source = (ROOT / "orion_core" / "gui" / "face3d.py").read_text(
        encoding="utf-8")
    assert "_graph_resolves" in source
    assert "_IMPORT_RE" in source, (
        "nothing parses the imports, so the check is back to trusting a list")


# ── the page is self-contained ───────────────────────────────────────────────

def test_the_page_never_mentions_the_cdn_when_the_copy_is_local():
    """The whole point of vendoring. A CDN URL left in the page is a page that
    needs the internet, whatever the local files say."""
    if face3d.local_three() is None:
        pytest.skip("nothing vendored on this machine")
    html = face3d.three_sourced(face3d.FACE_HTML)
    assert face3d.CDN_BASE not in html


def test_import_map_values_are_resolvable_urls():
    """A bare value is silently discarded by the loader.

    ``build/three.module.js`` is read as another bare specifier and dropped,
    and the import it was meant to satisfy then fails with "blocked by a null
    value" — the second of the three faults.
    """
    if face3d.local_three() is None:
        pytest.skip("nothing vendored on this machine")
    html = face3d.three_sourced(face3d.FACE_HTML)
    values = re.findall(r'"(?:three|three/addons/)"\s*:\s*"([^"]*)"', html)
    assert values, "the import map has no entries"
    for value in values:
        assert value.startswith(("./", "../", "/", "http://", "https://")), (
            f"{value!r} is a bare specifier; the loader will discard it")


def test_the_scheme_is_registered_at_import_not_on_first_use():
    """Qt freezes the scheme list when the web engine starts with QApplication.

    Registering later is ignored with a warning, and the page is then given a
    base URL that resolves to nothing — the first of the three faults. app.py
    imports this module before constructing QApplication precisely so that
    import-time registration lands in the window Qt allows.
    """
    source = (ROOT / "orion_core" / "gui" / "face3d.py").read_text(
        encoding="utf-8")
    call = source.index("\nif WEBENGINE_OK:\n    _register_scheme()")
    definition = source.index("def _register_scheme()")
    assert definition < call, "the call precedes the definition"
    assert call < source.index("class QuantumFace3D"), (
        "registration must happen at import, before any widget is built")

    app = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert app.index("from .gui.face3d import") < app.index(
        "app = QApplication(sys.argv)"), (
        "face3d must be imported before QApplication or the scheme "
        "registration it performs at import is too late to take effect")


def test_the_page_is_served_rather_than_set_as_html():
    """setHtml documents have an opaque origin.

    Fetching the library beside them is therefore cross-origin, and a
    CORS-enabled scheme that sends no allow-origin header fails that check
    without resolving, rejecting or logging anything. Navigating to the page
    over the same scheme gives it a real origin and the fetch becomes
    same-origin.
    """
    source = (ROOT / "orion_core" / "gui" / "face3d.py").read_text(
        encoding="utf-8")
    build = source[source.index("def _ensure_built"):]
    build = build[:build.index("def showEvent")]
    # The CALLS, not the words: a comment above the navigation explains why
    # setHtml is wrong here, and matching prose put "setHtml" first and failed
    # a correct implementation.
    assert "self.view.setUrl(" in build, "the page is not navigated to"
    assert build.index("self.view.setUrl(") < build.index(
        "self.view.setHtml("), (
        "setHtml is the fallback for when the scheme is unavailable, not the "
        "primary path")


def test_the_face_rests_when_nothing_is_moving():
    """A silent orb was composited 60 times a second with full bloom.

    Measured on the installed build once the face finally rendered: the two
    Chromium processes cost 83% of one core between them while ORION sat
    saying nothing. There was already an adaptive QUALITY governor, which
    lowers point density when frames get slow; there was no frame-RATE
    governor at all.
    """
    page = face3d.FACE_HTML
    assert "targetFrameMs" in page, "the render loop has no frame-rate governor"
    assert "IDLE_FPS" in page and "ACTIVE_FPS" in page

    loop = page[page.index("function animate()"):]
    loop = loop[:loop.index("composer.render()")]
    assert "targetFrameMs()" in loop, (
        "the governor is defined but the loop does not consult it")


def test_anything_moving_restores_the_full_frame_rate():
    """The saving must not become a stutter at the one moment the face is
    being looked at. Checked per frame rather than on a timer, so speech,
    amplitude or a morph brings it back on the very next frame."""
    page = face3d.FACE_HTML
    governor = page[page.index("function targetFrameMs()"):]
    governor = governor[:governor.index("function animate()")]
    for signal in ("morphTarget", "morphT", "st.amp", "speakEnv"):
        assert signal in governor, (
            f"{signal} is not consulted, so the face would stay at the idle "
            "rate while it is actually moving")


def test_the_failure_message_does_not_blame_the_network():
    """It sent people to fix the one thing that was already fine.

    The library is vendored, so "needs an internet connection" was wrong in
    every case it appeared — and it appeared on a machine with working WiFi
    for three separate reasons, none of them the network.
    """
    assert "needs an internet connection" not in face3d.FACE_HTML
    assert "streams from a CDN" not in face3d.FACE_HTML
