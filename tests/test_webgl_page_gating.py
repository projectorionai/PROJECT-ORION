"""
Only the page you are looking at may render.

Reported: "there's an issue with loading up the SWARM then the GLOBE because
now Globe no longer works again."

Both are QWebEngineView pages with their own Chromium render process and their
own WebGL context. The globe paused itself when hidden; the swarm never did, so
its Three.js loop kept running at full frame rate behind the deck and Cesium —
much the heavier scene — lost its context or had its render process killed.

These tests pin the contract for every heavy page: pause on hide, resume on
show, and never assume the JS bridge exists yet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from orion_core.gui import swarm_view as swarm_mod  # noqa: E402


class _Sig:
    def __init__(self):
        self.messages = []

    def emit(self, *a):
        self.messages.append(a[0] if len(a) == 1 else a)

    def connect(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


class _Page:
    def __init__(self):
        self.js: list[str] = []

    def runJavaScript(self, code, *a):  # noqa: N802 (Qt name)
        self.js.append(code)


class _View:
    def __init__(self):
        self._page = _Page()

    def page(self):
        return self._page


def _view_stub():
    """A SwarmDeckView with just enough attached to exercise _set_active."""
    view = swarm_mod.SwarmDeckView.__new__(swarm_mod.SwarmDeckView)
    view.bus = _Bus()
    view._active = False
    # The page only accepts JS once it has loaded — calling
    # runJavaScript on a freshly-built view crashes Qt natively.
    view._page_loaded = True
    view.view = _View()
    view.gl_view = None
    return view


# ── the JS side of the contract ───────────────────────────────────────────────

def test_the_swarm_page_exposes_a_render_loop_switch():
    assert "window.orionSetActive" in swarm_mod.SWARM_HTML, (
        "the swarm page must be pausable, or it renders for the whole session")


def test_the_render_loop_can_actually_be_cancelled():
    html = swarm_mod.SWARM_HTML
    assert "cancelAnimationFrame" in html
    assert "requestAnimationFrame" in html


def test_the_page_also_pauses_itself_when_the_browser_hides_it():
    assert "visibilitychange" in swarm_mod.SWARM_HTML


# ── the Python side ───────────────────────────────────────────────────────────

def test_hiding_the_swarm_stops_its_render_loop():
    view = _view_stub()
    view._set_active(True)
    view._set_active(False)
    js = " ".join(view.view.page().js)
    assert "orionSetActive(false)" in js


def test_showing_the_swarm_starts_it_again():
    view = _view_stub()
    view._set_active(True)
    assert "orionSetActive(true)" in " ".join(view.view.page().js)
    assert view._active is True


def test_the_call_is_guarded_so_it_cannot_raise_before_the_page_loads():
    """Before load, orionSetActive is undefined — the guard makes it a no-op."""
    view = _view_stub()
    view._set_active(False)
    assert all("window.orionSetActive&&" in code for code in view.view.page().js)


def test_set_active_survives_a_missing_web_view():
    view = _view_stub()
    view.view = None
    view._set_active(False)          # must not raise
    assert view._active is False


def test_set_active_drives_the_native_gl_renderer_too():
    class GL:
        def __init__(self):
            self.calls = []

        def set_active(self, on):
            self.calls.append(on)

    view = _view_stub()
    view.gl_view = GL()
    view._set_active(False)
    assert view.gl_view.calls == [False]


def test_a_gl_renderer_with_no_hook_is_ignored_quietly():
    view = _view_stub()
    view.gl_view = object()
    view._set_active(False)          # must not raise


# ── both heavy pages honour the same contract ────────────────────────────────

def test_every_heavy_page_defines_show_hide_and_set_active():
    """Read the SOURCE rather than importing.

    globe.py imports QtWebEngineWidgets at module level, and Qt requires that
    to happen before the process's QApplication exists. Importing it from a
    test — where an ad-hoc QApplication already exists — destabilises OpenGL
    for every real-widget test that runs afterwards and eventually crashes the
    interpreter with no traceback. swarm_view.py's own docstring documents this
    exact hazard; this test must not re-introduce it to check for it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "orion_core" / "gui"
    for filename in ("globe.py", "swarm_view.py"):
        source = (root / filename).read_text(encoding="utf-8")
        for hook in ("def showEvent", "def hideEvent", "def _set_active"):
            assert hook in source, (
                f"{filename} is missing {hook} — it will render while hidden "
                "and starve the other WebGL page on the deck")
