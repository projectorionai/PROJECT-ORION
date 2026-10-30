"""Exercise search ownership, exact positions and the native location chooser."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core.globe_locations import GlobeLocation
from orion_core.gui.globe import GlobeView


def _stub():
    events, logs, js = [], [], []
    stub = SimpleNamespace(_navigation_seq=1, _page_generation=1, _closed=False,
                           js=js, events=events, logs=logs,
                           bus=SimpleNamespace(log=SimpleNamespace(emit=logs.append),
                               dashboard_event=SimpleNamespace(emit=lambda *args: events.append(args))))
    stub._js = js.append
    stub._show_location = lambda *args: GlobeView._show_location(stub, *args)
    stub._location_error = lambda message: GlobeView._location_error(stub, message)
    return stub


def test_exact_coordinates_do_not_wait_for_geocoding_or_news():
    stub = _stub()
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("An exact coordinate must not access a place or news provider")
    stub._resolve_locations = forbidden
    stub._news = forbidden
    asyncio.run(GlobeView._go(stub, "51.507400123,-0.127800987"))
    assert stub.js[0].startswith("flyTo(51.507400123,-0.127800987,")
    assert "Supplied WGS84 coordinate" in stub.js[0]
    assert stub.events[0][1]["lat"] == 51.507400123
    assert stub.events[0][1]["source"] == "User"


@pytest.mark.parametrize("query", ["91,0", "51.5,London", "geo:51.5,-0.1,10"])
def test_invalid_coordinate_is_an_error_not_a_guessed_place(query):
    stub = _stub()
    async def forbidden(_query):
        raise AssertionError("Malformed coordinates reached the place provider")
    stub._resolve_locations = forbidden
    asyncio.run(GlobeView._go(stub, query))
    assert stub.logs and stub.js[0].startswith("showNews(")
    assert not stub.events


def test_new_selection_during_news_cannot_publish_old_articles():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        stub = _stub()
        async def news(_query):
            entered.set()
            await release.wait()
            return [{"title": "Old article"}]
        stub._news = news
        task = asyncio.create_task(GlobeView._show_location(
            stub, GlobeLocation(51.5, -0.1, "Old place", "Test", "town"), 1, 1))
        await entered.wait()
        assert stub.events  # Location is published before slow news completes.
        stub._navigation_seq = 2
        release.set()
        await task
        assert all("Old article" not in code for code in stub.js)
    asyncio.run(scenario())


def test_pending_old_navigation_is_removed_without_losing_feed_updates():
    stub = SimpleNamespace(_pending_js=["flyTo(1,2,'Old')", "showMap('Old',1,2,'Old')",
                                        "showNews([], '')", "resetView()", "clearLocation()",
                                        "window.orionWorldLayer('weather',true)"])
    GlobeView._discard_pending_navigation(stub)
    assert stub._pending_js == ["window.orionWorldLayer('weather',true)"]


@pytest.mark.parametrize("query", ["SW1A 1AA", "10 Downing Street, London", "Example Road, London"])
def test_precise_address_cannot_fall_back_to_city_gazetteer(query):
    async def scenario():
        async def geocode(*_args, **_kwargs):
            return []
        stub = SimpleNamespace(geo=SimpleNamespace(geocode=geocode))
        assert await GlobeView._resolve_locations(stub, query) == []
    asyncio.run(scenario())


def test_native_chooser_requires_selection_and_reset_rejects_old_choices():
    # QtWebEngine must be imported before QApplication in a separate process;
    # constructing Qt widgets here would contaminate the suite's GUI lifecycle.
    script = r'''
import asyncio
from types import SimpleNamespace
from orion_core.gui.globe import GlobeView
from orion_core.globe_locations import GlobeLocation
from orion_core.bus import OrionBus
from orion_core import background
from PyQt6.QtWidgets import QApplication

app = QApplication([])
view = GlobeView(OrionBus())
calls = []
view._js = calls.append
tasks = []
def spawn(coro, **kwargs):
    task = asyncio.create_task(coro)
    tasks.append(task)
    return task
background.spawn = spawn
async def resolve(query):
    return [GlobeLocation(51, -1, 'Same name, First county', 'Test', 'town'),
            GlobeLocation(52, -2, 'Same name, Second county', 'Test', 'town')]
async def news(query):
    return []
view._resolve_locations = resolve
view._news = news
async def check():
    await view._go('Same name')
    assert view.location_choices.count() == 3
    assert not view.location_choices.isHidden()
    assert not any(code.startswith('flyTo(') for code in calls)
    view.location_choices.activated.emit(2)
    await asyncio.gather(*tasks)
    assert any(code.startswith('flyTo(52,-2,') for code in calls)
    view.reset_view()
    assert view.location_choices.isHidden()
    count = len(tasks)
    view.location_choices.activated.emit(2)
    assert len(tasks) == count
    await view._go('Same name')
    view._page_generation += 1
    view.location_choices.activated.emit(1)
    assert len(tasks) == count
asyncio.run(check())
view.shutdown()
print('Native chooser acceptance passed')
'''
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native chooser acceptance passed" in result.stdout


# ── Mark XXXV W1.2: Cesium in its own process ────────────────────────────────

def test_the_globe_runs_in_its_host_and_is_rebuilt_here_if_the_host_fails(monkeypatch):
    """MEASURED 11 Oct, ORION's thread blocked 40%: in-process 4 stutters in
    6 s (longest 183 ms); in its own process 0 caused by ORION (60 fps)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QApplication, QWidget
    from orion_core.bus import OrionBus
    from orion_core.gui import globe as globe_module
    from orion_core.gui import page_host
    if not globe_module.WEBENGINE_OK:
        pytest.skip("PyQt6-WebEngine is not installed")
    app = QApplication.instance() or QApplication([])
    built: list = []

    class FakeRemote(QWidget):
        loadFinished = pyqtSignal(bool)
        renderProcessTerminated = pyqtSignal(object, int)

        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.html = None
            self.on_failed = None
            self.closed = False
            self._containers: dict = {}
            built.append(self)

        def setHtml(self, html, base=None):
            self.html = (html, base)

        def page(self):
            return self

        def runJavaScript(self, code, callback=None):
            pass

        def shutdown(self):
            self.closed = True

    class FakeView(QWidget):
        loadFinished = pyqtSignal(bool)
        renderProcessTerminated = pyqtSignal(object, int)

        def setPage(self, page): self._page = page
        def page(self): return self
        def setWebChannel(self, channel): pass
        def setHtml(self, html, base=None): self.html = html
        def runJavaScript(self, code, callback=None): pass

    monkeypatch.setattr(page_host, "RemoteWebView", FakeRemote)
    monkeypatch.setattr(globe_module, "QWebEngineView", FakeView)
    monkeypatch.setattr(globe_module, "_GlobePage", lambda view, on_js_error=None: None)
    monkeypatch.setattr(globe_module, "QWebChannel",
                        lambda page: type("C", (), {"registerObject": lambda *a: None})())
    monkeypatch.setenv("ORION_GLOBE_PROCESS", "1")
    monkeypatch.delenv("ORION_PAGE_PROCESS", raising=False)
    view = globe_module.GlobeView(OrionBus(), remote=True)
    view._ensure_built()
    remote = built[0]
    assert view.view is remote
    assert remote.kwargs["bridges"] == {"globeBridge": view._world_bridge}
    assert remote.kwargs["allow"] == (globe_module.GLOBE_BASE_URL,)
    assert remote.html[1].toString() == globe_module.GLOBE_BASE_URL
    remote.on_failed()                                   # the host gave up
    app.processEvents()
    assert isinstance(view.view, FakeView), "rebuilt in ORION's own process"
    assert remote.closed and not view._remote_wanted
    view.shutdown()
    # Built anywhere but the composition root, it never starts a host.
    built.clear()
    plain = globe_module.GlobeView(OrionBus())
    plain._ensure_built()
    assert not built and isinstance(plain.view, FakeView)
    plain.shutdown()


def test_mark40_globe_moves_smoothly_and_reads_out_where_it_points():
    # 27 Oct 2026: "make it so that you can actually ... drag and zoom smoothly".
    from orion_core.gui.globe import GLOBE_HTML
    assert "viewer.targetFrameRate=60" in GLOBE_HTML
    assert "ssc.inertiaZoom=0.45" in GLOBE_HTML
    # A resolution change clears the canvas, so it must redraw at once.
    low = GLOBE_HTML[GLOBE_HTML.index("function _lowRes()"):]
    assert "scene.requestRender()" in low[:low.index("}") + 40]
    assert "listen(viewer.canvas,'pointerdown',()=>{ _userDown=true; _lowRes()" not in GLOBE_HTML
    # Readout, double-click to fly in, keys that steer, the HUD frame.
    for piece in ('<div id="readout"', "'dblclick'", "'keydown'", "ArrowUp", '<div id="hud"',
                  "--crimson:#c8102e"):
        assert piece in GLOBE_HTML, piece
    # Typing in the route boxes never steers the globe.
    keys = GLOBE_HTML[GLOBE_HTML.index("listen(window,'keydown'"):]
    assert "t.tagName==='INPUT'" in keys[:600]



def test_a_map_tile_that_fails_to_download_is_not_a_crash(monkeypatch):
    """27 Oct 2026: 71 expired radar tiles became crash reports and started
    the Reliability Agent on the owner's quota."""
    from orion_core import crash_reporter
    from orion_core.gui.globe import GlobeView
    reported, lines = [], []
    monkeypatch.setattr(crash_reporter, "report", lambda kind, detail: reported.append(kind))
    view = SimpleNamespace(bus=SimpleNamespace(log=SimpleNamespace(emit=lines.append)),
                           _NETWORK_ERROR=GlobeView._NETWORK_ERROR)
    blocked = ("Access to XMLHttpRequest at 'https://tilecache.rainviewer.com/v2/radar/x/256/3/4/5/2/"
               "1_1.png' from origin 'https://cdn.jsdelivr.net' has been blocked by CORS policy")
    GlobeView._on_js_error(view, blocked)
    GlobeView._on_js_error(view, blocked)
    assert reported == [] and len(lines) == 1 and "could not be downloaded" in lines[0]
    GlobeView._on_js_error(view, "TypeError: viewer.scene is undefined")
    assert reported == ["globe-js"]


def test_the_rain_radar_replaces_an_expired_frame():
    from orion_core.gui.globe import GLOBE_HTML
    assert "provider.errorEvent.addEventListener" in GLOBE_HTML
    assert "OVERLAY.radarAt=Date.now()" in GLOBE_HTML
    assert "Date.now()-(OVERLAY.radarAt||0)>600000" in GLOBE_HTML
