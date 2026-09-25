"""
Tests for the ``max_zoom_in_globe`` controller (Section 1).

Covers, with fakes for both a desktop (Cesium-altitude) and a browser/web
(zoom-level) backend: explicit maximum, unknown maximum detected by state
stability, already-at-maximum idempotency, transient zoom failure + recovery,
cancellation, timeout and backend-unavailable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.globe_zoom import (
    GlobeZoomConfig,
    GlobeZoomController,
    ZoomTermination,
)


async def _instant_sleep(_seconds: float) -> None:
    return None


def _run(backend, config=None, cancel=None, clock=None):
    controller = GlobeZoomController(
        backend, config=config, sleep=_instant_sleep, clock=clock
    )
    return asyncio.run(controller.run(cancel=cancel))


class DesktopAltitudeGlobe:
    """Fake Cesium desktop globe: altitude falls towards a 180 m floor; the
    backend reports 'at maximum' once the floor is reached (explicit max)."""

    name = "cesium-desktop"

    def __init__(self, start=24_000_000.0, floor=180.0, factor=0.45,
                 ready=True, fail_steps=()):
        self.altitude = start
        self.floor = floor
        self.factor = factor
        self.ready = ready
        self.fail_steps = set(fail_steps)
        self.steps = 0

    def available(self):
        return self.ready

    def read_state(self):
        return self.altitude

    def zoom_in_step(self):
        self.steps += 1
        if self.steps in self.fail_steps:
            return None  # transient failure — altitude unchanged this step
        self.altitude = max(self.floor, self.altitude * self.factor)
        return self.altitude

    def at_maximum(self, state):
        if state is None:
            return None
        return state <= self.floor * 1.0001


class WebZoomLevelGlobe:
    """Fake browser globe: an integer zoom level climbs to a cap, but the
    backend does NOT expose whether it is at max — the controller must infer it
    from the level ceasing to change (unknown-maximum path)."""

    name = "web-browser"

    def __init__(self, start=2.0, cap=20.0, ready=True):
        self.level = start
        self.cap = cap
        self.ready = ready

    def available(self):
        return self.ready

    def read_state(self):
        return self.level

    def zoom_in_step(self):
        self.level = min(self.cap, self.level + 1.0)
        return self.level

    def at_maximum(self, state):
        return None  # web backend can't say; rely on stability detection


def test_desktop_explicit_maximum_reached():
    globe = DesktopAltitudeGlobe()
    res = _run(globe)
    assert res.success
    assert res.termination is ZoomTermination.MAX_ZOOM_REACHED
    assert res.final_state == globe.floor
    assert res.backend == "cesium-desktop"
    assert res.attempts >= 1


def test_web_unknown_maximum_detected_by_stability():
    globe = WebZoomLevelGlobe(start=17.0, cap=20.0)
    res = _run(globe, config=GlobeZoomConfig(unchanged_threshold=2))
    assert res.success
    assert res.termination is ZoomTermination.STATE_STABLE
    assert res.final_state == globe.cap


def test_already_at_maximum_is_idempotent():
    globe = DesktopAltitudeGlobe(start=180.0)   # already on the floor
    res = _run(globe)
    assert res.success
    assert res.termination is ZoomTermination.MAX_ZOOM_REACHED
    assert res.attempts == 0                    # no zoom issued
    assert res.start_state == res.final_state == 180.0


def test_transient_failure_then_recovery():
    # Step 2 returns no state, then zooming recovers and still reaches the floor.
    globe = DesktopAltitudeGlobe(fail_steps=(2,))
    res = _run(globe)
    assert res.success
    assert res.termination is ZoomTermination.MAX_ZOOM_REACHED
    assert res.final_state == globe.floor


def test_cancellation_between_attempts():
    globe = DesktopAltitudeGlobe()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2          # cancel after a couple of checks

    res = _run(globe, cancel=cancel)
    assert not res.success
    assert res.termination is ZoomTermination.CANCELLED
    assert res.final_state is not None


def test_timeout_is_bounded():
    # A globe that never reaches max; a fake clock forces the time budget out.
    class Endless:
        name = "endless"

        def available(self):
            return True

        def read_state(self):
            return 1000.0

        def zoom_in_step(self):
            return 999.0  # meaningful change every step, so only time can stop it

        def at_maximum(self, state):
            return None

    ticks = iter([0.0] + [i * 5.0 for i in range(1, 50)])
    res = _run(Endless(), config=GlobeZoomConfig(max_seconds=8.0, unchanged_threshold=99),
               clock=lambda: next(ticks))
    assert res.termination is ZoomTermination.TIMEOUT
    assert res.success


def test_backend_unavailable():
    globe = DesktopAltitudeGlobe(ready=False)
    res = _run(globe)
    assert not res.success
    assert res.termination is ZoomTermination.BACKEND_UNAVAILABLE
    assert res.attempts == 0


def test_state_unobservable_falls_back_to_bounded_blind_attempts():
    class Blind:
        name = "blind"

        def available(self):
            return True

        def read_state(self):
            return None

        def zoom_in_step(self):
            return None

        def at_maximum(self, state):
            return None

    res = _run(Blind(), config=GlobeZoomConfig(blind_attempts=4))
    assert res.termination is ZoomTermination.STATE_UNOBSERVABLE
    assert res.success
    assert res.attempts == 4
    assert any("blind" in w for w in res.warnings)


def test_max_attempts_bound():
    class Crawler:
        name = "crawler"

        def __init__(self):
            self.v = 1_000_000.0

        def available(self):
            return True

        def read_state(self):
            return self.v

        def zoom_in_step(self):
            self.v *= 0.5   # always a meaningful change → never "stable"
            return self.v

        def at_maximum(self, state):
            return None

    res = _run(Crawler(), config=GlobeZoomConfig(max_attempts=5, max_seconds=999,
                                                 unchanged_threshold=99))
    assert res.termination is ZoomTermination.MAX_ATTEMPTS
    assert res.attempts == 5


def test_result_serialisation_is_structured():
    res = _run(DesktopAltitudeGlobe())
    d = res.as_dict()
    for key in ("success", "termination", "start_state", "final_state",
                "attempts", "duration_s", "backend", "warnings"):
        assert key in d
    assert isinstance(res.summary(), str)


# ── the Globe's live aircraft layer ──────────────────────────────────────────
#
# The measured page behaviour (300 aircraft: +4.7 MB JS heap, 0.19 ms per
# render; 600 updates plateau near 55 MB) comes from a real window. These pin
# the parts that made it cheap and the rule that the feed is polled only while
# the Globe is on screen AND the layer is on.

class _FakeTimer:
    def __init__(self):
        self.active = False

    def isActive(self):
        return self.active

    def start(self):
        self.active = True

    def stop(self):
        self.active = False


def _globe_stub(active=True):
    from types import SimpleNamespace
    stub = SimpleNamespace(_aviation_timer=_FakeTimer(), _active=active, view=object(),
                           _aviation_seq=0, _aviation_reset=False, _aviation_drawn=False,
                           js=[], polls=[])
    stub._js = stub.js.append
    stub._poll_aviation = lambda: stub.polls.append(1)
    return stub


def _fresh_aviation_service(monkeypatch):
    from orion_core import aviation
    service = aviation.AviationService(providers=[])
    monkeypatch.setattr(aviation, "_SERVICE", service)
    return service


def test_globe_aviation_polls_only_while_visible_and_layer_on(monkeypatch):
    from orion_core.gui.globe import GlobeView
    service = _fresh_aviation_service(monkeypatch)
    stub = _globe_stub(active=True)

    GlobeView._sync_aviation(stub)                        # layer off: nothing runs
    assert not stub._aviation_timer.active and stub.polls == [] and stub.js == []

    service.show_layer(50.5, -3.5, 40, "Testville")
    GlobeView._sync_aviation(stub)
    assert stub._aviation_timer.active and stub.polls == [1]
    assert [c for c in stub.js if "orionAircraftFocus" in c] == [
        'window.orionAircraftFocus&&orionAircraftFocus(50.5,-3.5,40.0,"Testville")']
    assert stub._aviation_reset                           # next batch replaces the old region

    GlobeView._sync_aviation(stub)                        # same focus, timer running: no re-fly
    assert stub.polls == [1] and len(stub.js) == 1

    stub._active = False                                  # tab hidden: polling stops
    GlobeView._sync_aviation(stub)
    assert not stub._aviation_timer.active

    service.hide_layer()                                  # layer off: the page is cleared
    GlobeView._sync_aviation(stub)
    assert stub.js[-1] == "window.orionAircraftLayer&&orionAircraftLayer(false)"
    assert not stub._aviation_timer.active


def test_globe_aviation_page_layer_is_instanced_bounded_and_pausable():
    from orion_core.gui.globe import GLOBE_HTML
    script = GLOBE_HTML.split("// ── live aviation layer", 1)[1].split("window.flyTo", 1)[0]
    # One collection for every aircraft, never an Entity each (the only entity
    # is the single search-circle outline).
    assert "new Cesium.BillboardCollection()" in script
    assert "new Cesium.LabelCollection()" in script
    assert script.count("viewer.entities.add(") == 1 and "AV.ring=viewer.entities.add(" in script
    assert "AV_MAX=500" in script and "AV_STALE_S=60" in script
    # Hidden tab: the aircraft pump and the detail card's clock both stop.
    gate = script.split("window.orionSetActive=function(on){", 1)[1]
    assert "clearTimeout(AV.timer)" in gate and "clearInterval(AV.panelTimer)" in gate
    # Public-feed strings are written as text, never parsed as markup.
    panel = script.split("function avPanel(){", 1)[1].split("window.orionAircraft=", 1)[0]
    assert "innerHTML" not in panel and "textContent" in panel
