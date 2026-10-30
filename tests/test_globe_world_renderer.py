"""Exercise the real embedded page and asynchronous request ownership."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from orion_core.gui.globe import GLOBE_HTML, GlobeView


@pytest.fixture
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(headless=True)
        except playwright.Error as exc:
            pytest.skip(f"Chromium unavailable: {str(exc).splitlines()[0]}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    page = browser.new_page(viewport={"width": 1024, "height": 768})
    # The source under test is unchanged. Offline rendering tests must neither
    # fetch public data nor depend on a CDN. A tiny Cesium primitive double is
    # injected only after the actual page has loaded without its engine.
    page.route("**/*", lambda route: route.abort())
    errors = []
    page.on("pageerror", lambda error: errors.append(error))
    page.set_content(GLOBE_HTML)
    page.evaluate("""() => {
      class Points {
        constructor(){this.items=[];this.show=true;this.dead=false;}
        add(item){this.items.push(item);return item;}
        removeAll(){this.items=[];}
        get length(){return this.items.length;}
        isDestroyed(){return this.dead;}
      }
      class Handler {
        constructor(){this.dead=false;}
        setInputAction(fn){this.fn=fn;}
        isDestroyed(){return this.dead;}
        destroy(){this.dead=true;}
      }
      window.Cesium={PointPrimitiveCollection:Points,ScreenSpaceEventHandler:Handler,
        ScreenSpaceEventType:{LEFT_CLICK:0},Color:{BLACK:'black',fromCssColorString:x=>x},
        Math:{toRadians:x=>x*Math.PI/180},
        BoundingSphere:class {constructor(center,radius){Object.assign(this,{center,radius});}},
        HeadingPitchRange:class {constructor(heading,pitch,range){Object.assign(this,{heading,pitch,range});}},
        Cartesian2:class {constructor(x,y){Object.assign(this,{x,y});}},
        HeightReference:{CLAMP_TO_GROUND:1},LabelStyle:{FILL_AND_OUTLINE:1},
        Cartesian3:{fromDegrees:(lon,lat,height)=>height===undefined?({lon,lat}):({lon,lat,height})}};
      window.cameraCalls=[];window.entities=[];
      viewer={dead:false,canvas:document.createElement('canvas'),
        isDestroyed(){return this.dead;},destroy(){this.dead=true;},resize(){},
        entities:{add(item){entities.push(item);return item;},
          remove(item){entities.splice(entities.indexOf(item),1);}},
        camera:{flyTo(item){cameraCalls.push({method:'flyTo',item});},
          cancelFlight(){cameraCalls.push({method:'cancel'});},
          flyToBoundingSphere(sphere,options){cameraCalls.push({method:'sphere',sphere,options});}},
        scene:{requestRender(){},pick(){return null;},
          primitives:{add:item=>item}}};
      WORLD.bridge={setWorldLayer(){},refreshWorld(){}};
    }""")
    yield page
    assert not errors, errors
    page.close()


def _report(items, **kwargs):
    return {"items": items, "status": "healthy", "stale": False,
            "source": "Official test source", "source_url": "https://example.org/feed",
            "fetched_at": "2026-09-30T12:00:00Z", **kwargs}


def test_repeated_batches_are_bounded_and_switching_hides_points(page):
    items = [{"id": str(i), "title": f"Earthquake {i}", "latitude": 10,
              "longitude": 20, "magnitude": 3} for i in range(500)]
    page.evaluate("orionWorldLayer('earthquakes',true)")
    for _ in range(3):
        page.evaluate("report=>orionWorldSnapshot({layers:{earthquakes:report}})", _report(items))
    assert page.evaluate("orionWorldStats().points") == 250
    assert page.evaluate("WORLD.layers.get('earthquakes').byId.size") == 250
    assert page.locator(".worlditem").count() == 8
    page.locator("#layer-earthquakes").uncheck()
    assert page.evaluate("WORLD.layers.get('earthquakes').points.show") is False
    assert page.locator(".worlditem").count() == 0


def test_reports_use_text_and_reject_unsafe_source_links(page):
    item = {"id": "bad", "title": '<img src=x onerror="window.pwned=true">',
            "source_url": "javascript:window.pwned=true", "details": "<script>unsafe</script>"}
    page.evaluate("orionWorldLayer('tsunami',true)")
    page.evaluate("r=>orionWorldSnapshot({layers:{tsunami:r}})", _report([item]))
    page.locator(".worlditem").click()
    assert page.locator("#eventtitle").inner_text() == item["title"]
    assert page.locator("#eventtitle img").count() == 0
    assert "<script>unsafe</script>" in page.locator("#eventdetails").inner_text()
    assert page.locator("#eventsource").get_attribute("href") is None
    assert page.evaluate("window.pwned") is None
    assert "does not supply map coordinates" in page.locator("#eventdetails").inner_text()
    assert page.evaluate("orionWorldStats().points") == 0


def test_no_renderer_still_displays_saved_bulletins_and_prediction_time(page):
    page.evaluate("viewer=null;orionWorldLayer('satellites',true)")
    item = {"id": "25544", "title": "ISS", "event_at": "2026-09-29T10:00:00Z",
            "predicted_at": "2026-09-30T12:00:00Z", "epoch": "2026-09-29T10:00:00Z"}
    page.evaluate("r=>orionWorldSnapshot({layers:{satellites:r}})",
                  _report([item], stale=True, status="stale", error="Network unavailable"))
    assert "30/09/2026" in page.locator(".worlditem small").inner_text()
    assert page.locator("#state-satellites").get_attribute("data-stale") == "true"
    page.locator(".worlditem").click()
    assert "Saved report" in page.locator("#eventdetails").inner_text()


def test_duplicate_and_invalid_positions_do_not_create_duplicate_points(page):
    items = [{"id": "same", "title": "One", "latitude": 5, "longitude": 10},
             {"id": "same", "title": "Duplicate", "latitude": 5, "longitude": 10},
             {"id": "invalid", "title": "Invalid", "latitude": 500, "longitude": 10},
             {"id": "x" * 300 + "A", "title": "Long GUID A", "latitude": 5, "longitude": 10},
             {"id": "x" * 300 + "B", "title": "Long GUID B", "latitude": 5, "longitude": 10}]
    page.evaluate("orionWorldLayer('weather',true)")
    page.evaluate("r=>orionWorldSnapshot({layers:{weather:r}})", _report(items))
    assert page.evaluate("orionWorldStats().points") == 3
    assert page.locator(".worlditem").count() == 4


def test_hidden_page_and_teardown_release_loops_and_handlers(page):
    page.evaluate("""() => {
      window.calls=0;listen(window,'test-orion',()=>window.calls++);
      AV.timer=setTimeout(()=>{},100000);AV.panelTimer=setInterval(()=>{},100000);
      orionSetActive(false);
    }""")
    assert page.evaluate("viewer.useDefaultRenderLoop") is False
    assert page.evaluate("AV.timer===null&&AV.panelTimer===null")
    page.evaluate("window.dispatchEvent(new Event('test-orion'));orionDestroy();window.dispatchEvent(new Event('test-orion'))")
    assert page.evaluate("window.calls") == 1
    assert page.evaluate("viewer===null&&WORLD.handler===null&&rendererCleanup.length===0")
    page.evaluate("orionWorldSnapshot({layers:{weather:{items:[]}}})")
    assert page.evaluate("orionWorldStats().reports") == 0


def test_news_links_and_footage_are_bounded_and_https_only(page):
    page.evaluate("""showNews([{title:'<script>bad</script>',url:'javascript:alert(1)'},
      {title:'Good',url:'https://example.org/report'}], 'data:text/html,unsafe')""")
    assert page.locator("#news a").nth(0).get_attribute("href") is None
    assert page.locator("#news a").nth(1).get_attribute("href") == "https://example.org/report"
    assert page.locator("#news a").nth(1).get_attribute("target") is None
    assert not page.locator("#footage").is_visible()
    assert page.locator("#news script").count() == 0


def test_page_reset_uses_owner_to_cancel_in_flight_navigation(page):
    page.evaluate("() => {window.resetCalls=0;WORLD.bridge.resetView=()=>window.resetCalls++;}")
    page.locator("#reset").click()
    assert page.evaluate("window.resetCalls") == 1


def test_source_links_use_the_navigation_interceptor(page):
    page.evaluate("orionWorldLayer('weather',true)")
    page.evaluate("r=>orionWorldSnapshot({layers:{weather:r}})",
                  _report([{"id": "1", "title": "Warning", "source_url": "https://example.org/warning"}]))
    page.locator(".worlditem").click()
    assert page.locator("#eventsource").get_attribute("href") == "https://example.org/warning"
    assert page.locator("#eventsource").get_attribute("target") is None


def test_every_unlocated_bulletin_is_accessible_by_bounded_pagination(page):
    items = [{"id": str(i), "title": f"Bulletin {i}", "details": f"Details {i}"}
             for i in range(25)]
    page.evaluate("orionWorldLayer('tsunami',true)")
    page.evaluate("r=>orionWorldSnapshot({layers:{tsunami:r}})", _report(items))
    found = []
    while True:
        buttons = page.locator(".worlditem")
        assert buttons.count() <= 8
        found.extend(buttons.all_text_contents())
        for i in range(buttons.count()):
            buttons.nth(i).click()
            assert "This source does not supply map coordinates" in page.locator("#eventdetails").inner_text()
            page.locator("#eventclose").click()
        next_button = page.locator(".worldpager[data-layer=tsunami]").get_by_role("button", name="Next")
        if next_button.is_disabled():
            break
        next_button.click()
    assert len(found) == 25
    assert len(set(found)) == 25
    assert page.evaluate("orionWorldStats().points") == 0


def test_world_response_cannot_publish_after_hidden_or_page_replaced(monkeypatch):
    async def scenario():
        ready, release = asyncio.Event(), asyncio.Event()
        async def snapshot(**_kwargs):
            ready.set()
            await release.wait()
            return {"layers": {"weather": _report([])}}
        monkeypatch.setitem(sys.modules, "orion_core.world_feeds",
                            SimpleNamespace(get_world_feeds=lambda: SimpleNamespace(snapshot=snapshot)))
        stub = SimpleNamespace(_page_generation=1, _world_layers={"weather"}, _active=True,
                               _closed=False, _world_error="", _world_snapshot=None, js=[])
        stub._js = stub.js.append
        task = asyncio.create_task(GlobeView._refresh_world(stub))
        await ready.wait()
        stub._page_generation = 2
        release.set()
        await task
        assert stub._world_snapshot is None and stub.js == []
    asyncio.run(scenario())


def test_old_navigation_cannot_override_a_newer_place():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def geocode(_place):
            entered.set()
            await release.wait()
            from orion_core.globe_locations import GlobeLocation
            return [GlobeLocation(1, 2, "Old place", "Test", "town")]
        stub = SimpleNamespace(_navigation_seq=1, _page_generation=1, _closed=False, js=[])
        stub._resolve_locations = geocode
        stub._js = stub.js.append
        task = asyncio.create_task(GlobeView._go(stub, "Old place"))
        await entered.wait()
        stub._navigation_seq = 2
        release.set()
        await task
        assert stub.js == []
    asyncio.run(scenario())


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181), (float("nan"), 0),
                                        (0, float("inf")), (None, 0), ("x", "y"), (True, 0)])
def test_invalid_geocoder_positions_are_refused(lat, lon):
    assert not GlobeView._valid_coordinates(lat, lon)


def test_world_selection_whitelist_and_cancellation():
    task = SimpleNamespace(done=lambda: False, cancelled=False)
    task.cancel = lambda: setattr(task, "cancelled", True)
    stub = SimpleNamespace(WORLD_LAYERS=GlobeView.WORLD_LAYERS, _closed=False,
                           _world_layers={"weather"}, _world_task=task, js=[], polls=[])
    stub._js = stub.js.append
    stub._sync_world = lambda: stub.polls.append(1)
    GlobeView.set_world_layer(stub, "https://evil.example/feed", True)
    assert not task.cancelled and not stub.polls
    GlobeView.set_world_layer(stub, "weather", False)
    assert task.cancelled and not stub._world_layers and stub._world_task is None
    assert stub.polls == [1]
    # A queued timeout delivered after a tab hide must not start another
    # aircraft request after _cancel_feed_tasks has already stopped it.
    stub._active = False
    stub.view = object()
    GlobeView._poll_aviation(stub)


@pytest.mark.parametrize("lat,lon", [(51.507400123, -0.127800987), (90, 180), (-90, -180)])
def test_camera_and_marker_share_exact_target_without_latitude_offset(page, lat, lon):
    page.evaluate("([lat,lon])=>flyTo(lat,lon,'Target','Supplied WGS84 coordinate',3000)", [lat, lon])
    calls = page.evaluate("cameraCalls")
    assert calls[0]["method"] == "cancel"
    assert calls[1]["sphere"]["center"] == {"lat": lat, "lon": lon}
    assert calls[1]["options"]["offset"]["range"] == 3000
    assert calls[1]["options"]["offset"]["pitch"] == pytest.approx(-1.5707963267948966)
    assert page.evaluate("entities[0].position") == {"lat": lat, "lon": lon}
    assert page.locator("#coords").inner_text() == f"{lat:.6f}, {lon:.6f} · Supplied WGS84 coordinate"


def test_new_target_cancels_previous_flight_and_replaces_marker(page):
    page.evaluate("flyTo(1,2,'First');flyTo(3,4,'Second')")
    assert page.evaluate("entities.length") == 1
    assert page.evaluate("entities[0].position.lat") == 3
    assert page.evaluate("cameraCalls.map(x=>x.method)") == ["cancel", "sphere", "cancel", "sphere"]


def test_map_card_uses_resolved_point_instead_of_ambiguous_query(page):
    page.evaluate("showMap('Springfield',51.507400123,-0.127800987,'Selected place')")
    frame = page.locator("#mapframe").get_attribute("src")
    assert "51.507400123%2C-0.127800987" in frame
    assert "Springfield" not in frame
    assert page.evaluate("_mapLL.q") == "51.507400123,-0.127800987"
    page.evaluate("hideMap()")
    assert page.evaluate("_mapLL") is None


def test_world_card_focus_uses_the_same_centred_navigation(page):
    page.evaluate("orionWorldLayer('earthquakes',true)")
    page.evaluate("r=>orionWorldSnapshot({layers:{earthquakes:r}})",
                  _report([{"id": "quake", "title": "Quake", "latitude": 51.5, "longitude": -0.1}]))
    page.locator(".worlditem").click()
    assert page.evaluate("cameraCalls.filter(x=>x.method==='sphere')[0].sphere.center") == {"lat": 51.5, "lon": -0.1}
    assert page.locator("#coords").inner_text().startswith("51.500000, -0.100000")
    assert page.evaluate("_mapLL.q") == "51.5,-0.1"


def test_unlocated_bulletin_never_inherits_previous_place_marker(page):
    page.evaluate("flyTo(1,2,'Place');showMap('Place',1,2,'Place');orionWorldLayer('tsunami',true)")
    page.evaluate("r=>orionWorldSnapshot({layers:{tsunami:r}})",
                  _report([{"id": "1", "title": "Unlocated bulletin"}]))
    page.locator(".worlditem").click()
    assert page.evaluate("entities.length") == 0
    assert page.evaluate("_mapLL") is None
    assert page.evaluate("cameraCalls.filter(x=>x.method==='sphere').length") == 1


def test_reset_removes_marker_map_link_and_cancels_flight(page):
    page.evaluate("flyTo(1,2,'Place');showMap('Place',1,2,'Place');resetView()")
    assert page.evaluate("entities.length") == 0
    assert page.evaluate("_mapLL") is None
    assert page.locator("#coords").inner_text() == "Earth view"
    assert page.evaluate("cameraCalls.slice(-1)[0].method") == "flyTo"
