"""
GlobeView — a detailed, interactive Three.js globe embedded in the GUI.

The globe is a properly textured model of the Earth (NASA "blue marble" colour
map, topology bump map and an ocean specular mask), wrapped in an atmosphere
glow and a starfield.  ORION or the user can fly it to any place; ORION then
fetches that region's news and offers footage.

Two things this version fixes:

  • EXTERNAL LINKS — news and footage links open in the *system browser*, never
    inside the globe's own web view.  Previously a click navigated the web view
    away from the globe (so you could go into a country but never back out) and
    loaded third-party video players that tripped the page's media Content
    Security Policy (the VIDEOJS / MEDIA_ERR_SRC_NOT_SUPPORTED console errors).
    A custom QWebEnginePage intercepts navigations and hands external URLs to
    the OS browser, leaving the globe untouched.

  • RESET / EXIT — a "⌂ Reset view" control zooms back out to the whole Earth
    and resumes the idle spin, so you are never stuck focused on one country.

Textures load from unpkg (CORS-open) so the live globe needs internet — as does
news and footage — but it degrades to a plain shaded Earth when offline.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import webbrowser
from typing import Any
from urllib.parse import quote_plus

from aiohttp import ClientSession, ClientTimeout
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus

try:
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_OK = True
except Exception:  # pragma: no cover - optional PyQt6-WebEngine
    WEBENGINE_OK = False
    QWebEnginePage = object  # type: ignore


GLOBE_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;height:100%;background:#03040a;overflow:hidden;font-family:'Segoe UI',Arial}
  #panel{position:absolute;top:12px;right:12px;width:330px;max-height:80%;overflow:auto;
    background:rgba(8,8,14,.84);border:1px solid #7c0d1e;border-radius:10px;color:#fff;padding:12px}
  #panel h2{margin:0 0 4px;font-size:16px;color:#ff5c73}
  #panel .sub{color:#a9a9b2;font-size:11px;margin-bottom:8px}
  #news a{display:block;color:#fff;text-decoration:none;font-size:12px;line-height:1.35;
    padding:6px 0;border-bottom:1px solid #241018;cursor:pointer}
  #news a:hover{color:#ff5c73}
  .btnrow{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
  .btn{background:#991024;color:#fff;padding:6px 12px;border-radius:8px;font-size:12px;cursor:pointer;border:1px solid #ff1a3c}
  .btn.ghost{background:transparent}
  #hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);color:#5b5b66;font-size:11px;text-align:center}
  #loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#5b6b7a;font-size:13px}
  #mapcard{position:absolute;bottom:12px;left:12px;width:380px;max-width:46%;
    background:rgba(8,8,14,.92);border:1px solid #7c0d1e;border-radius:10px;color:#fff;padding:8px;display:none}
  #mapcard .maphead{display:flex;justify-content:space-between;align-items:center;
    font-size:13px;color:#ff5c73;font-weight:700;margin-bottom:6px}
  #mapcard .maphead .mx{cursor:pointer;color:#a9a9b2;padding:0 4px}
  #mapaddr{color:#a9a9b2;font-size:11px;margin:0 2px 6px}
  #mapframe{width:100%;height:230px;border:0;border-radius:6px;background:#0b0f14}
</style></head><body>
<div id="loading">Rendering Earth…</div>
<div id="panel"><h2 id="place">O.R.I.O.N. Globe</h2>
  <div class="sub" id="coords">Ask me to take you somewhere, sir — or type a place.</div>
  <div id="news"></div>
  <div class="btnrow">
    <span class="btn" id="footage" style="display:none">▶ Footage</span>
    <span class="btn ghost" id="reset">⌂ Reset view</span>
  </div>
</div>
<div id="mapcard">
  <div class="maphead"><span id="maptitle">Street map</span><span class="mx" onclick="hideMap()">✕</span></div>
  <div id="mapaddr"></div>
  <iframe id="mapframe" src="about:blank" loading="lazy" referrerpolicy="no-referrer-when-downgrade"
    allowfullscreen></iframe>
  <div class="btnrow">
    <span class="btn" onclick="streetView()">🚶 Street View</span>
    <span class="btn ghost" onclick="openMaps()">↗ Open in Google Maps</span>
  </div>
</div>
<div id="hint">Drag to rotate · scroll to zoom · Reset to return to the whole Earth</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const R=100, IMG='https://unpkg.com/three-globe/example/img/';
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,0.1,6000);
cam.position.set(0,60,340);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
document.body.appendChild(renderer.domElement);
const controls=new OrbitControls(cam,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=0.08;
controls.minDistance=150;controls.maxDistance=900;controls.enablePan=false;

const world=new THREE.Group();scene.add(world);
const loader=new THREE.TextureLoader();loader.crossOrigin='anonymous';
let loaded=0;const need=2;
function done(){loaded++;if(loaded>=need){const l=document.getElementById('loading');if(l)l.remove();}}
function srgb(t){if(t){t.colorSpace=THREE.SRGBColorSpace;t.anisotropy=8;}return t;}
// Brighter, more detailed Earth: high-res day map + topology bump + ocean
// specular + NIGHT-LIGHTS emissive map so cities glow on the dark side.
const earthMat=new THREE.MeshPhongMaterial({
  color:0xffffff, specular:0x2a4a66, shininess:22,
  map:srgb(loader.load(IMG+'earth-blue-marble.jpg',done,undefined,done)),
  bumpMap:loader.load(IMG+'earth-topology.png',done,undefined,done),
  bumpScale:1.1,
  specularMap:loader.load(IMG+'earth-water.png'),
  emissive:0xffd9a0,
  emissiveIntensity:0.9,
  emissiveMap:srgb(loader.load(IMG+'earth-night.jpg'))
});
const globe=new THREE.Mesh(new THREE.SphereGeometry(R,160,160),earthMat);
world.add(globe);
// Cloud shell — a faint animated layer for depth (uses the topology alpha).
const cloudMat=new THREE.MeshPhongMaterial({
  map:loader.load(IMG+'earth-topology.png'),
  transparent:true, opacity:0.16, depthWrite:false, color:0xffffff
});
const clouds=new THREE.Mesh(new THREE.SphereGeometry(R*1.015,96,96),cloudMat);
world.add(clouds);
// Faint graticule for a "command" look, kept subtle so the map reads.
const grat=new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(R*1.004,36,24)),
  new THREE.LineBasicMaterial({color:0x3a8fc9,transparent:true,opacity:0.08}));
world.add(grat);
// Atmosphere — brighter blue halo + inner glow.
const atmo=new THREE.Mesh(new THREE.SphereGeometry(R*1.10,96,96),
  new THREE.MeshBasicMaterial({color:0x4ab4ff,transparent:true,opacity:0.18,side:THREE.BackSide}));
scene.add(atmo);
const rim=new THREE.Mesh(new THREE.SphereGeometry(R*1.03,96,96),
  new THREE.MeshBasicMaterial({color:0x8fd0ff,transparent:true,opacity:0.10,side:THREE.BackSide}));
scene.add(rim);
// Lights — much brighter, with a warm key, cool fill and a rim light.
scene.add(new THREE.AmbientLight(0xb8c8dd,1.15));
const sun=new THREE.DirectionalLight(0xfff2e0,1.7);sun.position.set(300,150,260);scene.add(sun);
const fill=new THREE.DirectionalLight(0x89b7ff,0.7);fill.position.set(-260,-60,-160);scene.add(fill);
const rimLight=new THREE.DirectionalLight(0xffffff,0.5);rimLight.position.set(0,220,-260);scene.add(rimLight);
// Stars — brighter and denser.
const sg=new THREE.BufferGeometry();const sv=[];
for(let i=0;i<3200;i++){const r=2600,t=Math.random()*Math.PI*2,p=Math.acos(2*Math.random()-1);
  sv.push(r*Math.sin(p)*Math.cos(t),r*Math.sin(p)*Math.sin(t),r*Math.cos(p));}
sg.setAttribute('position',new THREE.Float32BufferAttribute(sv,3));
scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0xcdd8e6,size:1.9})));
// A glowing sun off to the light side.
const sunSprite=new THREE.Mesh(new THREE.SphereGeometry(26,24,24),
  new THREE.MeshBasicMaterial({color:0xfff0c0}));
sunSprite.position.set(760,380,640);scene.add(sunSprite);
const sunGlow=new THREE.Mesh(new THREE.SphereGeometry(60,24,24),
  new THREE.MeshBasicMaterial({color:0xffd27a,transparent:true,opacity:0.22}));
sunGlow.position.copy(sunSprite.position);scene.add(sunGlow);
function ll(lat,lon,rad){const phi=(90-lat)*Math.PI/180,th=(lon+180)*Math.PI/180;
  return new THREE.Vector3(-rad*Math.sin(phi)*Math.cos(th),rad*Math.cos(phi),rad*Math.sin(phi)*Math.sin(th));}
// Fine graticule — meridians every 20°, parallels every 15° (great circles).
const gmat=new THREE.LineBasicMaterial({color:0x3f9ad6,transparent:true,opacity:0.16});
for(let lon=-180;lon<180;lon+=20){const pts=[];for(let lat=-90;lat<=90;lat+=3)pts.push(ll(lat,lon,R*1.002));
  world.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gmat));}
for(let lat=-75;lat<=75;lat+=15){const pts=[];for(let lon=-180;lon<=180;lon+=3)pts.push(ll(lat,lon,R*1.002));
  world.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gmat));}
// Major-city glowing markers (always available, offline).
const CITIES=[['London',51.5,-0.1],['New York',40.7,-74.0],['Los Angeles',34.0,-118.2],
  ['Tokyo',35.7,139.7],['Shanghai',31.2,121.5],['Delhi',28.6,77.2],['Dubai',25.2,55.3],
  ['Paris',48.9,2.4],['Berlin',52.5,13.4],['Moscow',55.8,37.6],['Sydney',-33.9,151.2],
  ['São Paulo',-23.5,-46.6],['Lagos',6.5,3.4],['Cairo',30.0,31.2],['Singapore',1.35,103.8],
  ['Hong Kong',22.3,114.2],['Mexico City',19.4,-99.1],['Toronto',43.7,-79.4],
  ['Johannesburg',-26.2,28.0],['Istanbul',41.0,29.0],['Mumbai',19.1,72.9],['Seoul',37.6,127.0],
  ['Chicago',41.9,-87.6],['Nairobi',-1.3,36.8],['Bangkok',13.8,100.5],
  ['Madrid',40.4,-3.7],['Rome',41.9,12.5],['Amsterdam',52.4,4.9],['Jakarta',-6.2,106.8],
  ['Karachi',24.9,67.0],['Manila',14.6,121.0],['Buenos Aires',-34.6,-58.4],['Lima',-12.0,-77.0],
  ['Bogotá',4.7,-74.1],['Riyadh',24.7,46.7],['Tehran',35.7,51.4],['Baghdad',33.3,44.4],
  ['Kyiv',50.5,30.5],['Warsaw',52.2,21.0],['Stockholm',59.3,18.1],['Vienna',48.2,16.4],
  ['Athens',38.0,23.7],['Kuala Lumpur',3.1,101.7],['Ho Chi Minh City',10.8,106.7],
  ['Dhaka',23.8,90.4],['Bengaluru',13.0,77.6],['Chennai',13.1,80.3],['Kolkata',22.6,88.4],
  ['Guangzhou',23.1,113.3],['Chengdu',30.6,104.1],['Osaka',34.7,135.5],['Tehran',35.7,51.4],
  ['Casablanca',33.6,-7.6],['Accra',5.6,-0.2],['Addis Ababa',9.0,38.7],['Kinshasa',-4.3,15.3],
  ['Vancouver',49.3,-123.1],['San Francisco',37.8,-122.4],['Seattle',47.6,-122.3],
  ['Miami',25.8,-80.2],['Boston',42.4,-71.1],['Washington',38.9,-77.0],['Houston',29.8,-95.4],
  ['Melbourne',-37.8,144.9],['Auckland',-36.8,174.8],['Santiago',-33.5,-70.7],['Doha',25.3,51.5]];
const cityGeo=new THREE.BufferGeometry();const cv=[];
CITIES.forEach(c=>{const p=ll(c[1],c[2],R*1.012);cv.push(p.x,p.y,p.z);});
cityGeo.setAttribute('position',new THREE.Float32BufferAttribute(cv,3));
world.add(new THREE.Points(cityGeo,new THREE.PointsMaterial({color:0xffe08a,size:3.4,transparent:true,opacity:0.95})));
// Country borders from a GeoJSON — best-effort, silently skipped when offline.
fetch('https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/ne_110m_admin_0_countries.geojson')
 .then(r=>r.ok?r.json():null).then(gj=>{if(!gj)return;const bmat=new THREE.LineBasicMaterial({color:0x6fb7e8,transparent:true,opacity:0.34});
  const seg=[];function ring(coords){for(let i=0;i<coords.length-1;i++){const a=ll(coords[i][1],coords[i][0],R*1.006),b=ll(coords[i+1][1],coords[i+1][0],R*1.006);seg.push(a.x,a.y,a.z,b.x,b.y,b.z);}}
  (gj.features||[]).forEach(f=>{const g=f.geometry;if(!g)return;if(g.type==='Polygon')g.coordinates.forEach(ring);
    else if(g.type==='MultiPolygon')g.coordinates.forEach(poly=>poly.forEach(ring));});
  const bg=new THREE.BufferGeometry();bg.setAttribute('position',new THREE.Float32BufferAttribute(seg,3));
  world.add(new THREE.LineSegments(bg,bmat));}).catch(()=>{});
// Marker + pulse
const marker=new THREE.Mesh(new THREE.SphereGeometry(2.2,16,16),
  new THREE.MeshBasicMaterial({color:0xff2a44}));marker.visible=false;world.add(marker);
const ring=new THREE.Mesh(new THREE.RingGeometry(3,4.4,32),
  new THREE.MeshBasicMaterial({color:0xff5c73,side:THREE.DoubleSide,transparent:true,opacity:0.8}));
ring.visible=false;world.add(ring);
function place(lat,lon,rad){const phi=(90-lat)*Math.PI/180,th=(lon+180)*Math.PI/180;
  return new THREE.Vector3(-rad*Math.sin(phi)*Math.cos(th),rad*Math.cos(phi),rad*Math.sin(phi)*Math.sin(th));}
let targetRot=null,spin=true,pulse=0;
window.flyTo=function(lat,lon,label){
  const p=place(lat,lon,R*1.02);marker.position.copy(p);marker.visible=true;
  ring.position.copy(place(lat,lon,R*1.03));ring.lookAt(0,0,0);ring.visible=true;
  targetRot={y:-(lon+180)*Math.PI/180-Math.PI/2, x:Math.max(-0.9,Math.min(0.9,lat*Math.PI/180*0.6))};
  spin=false;controls.autoRotate=false;
  document.getElementById('place').textContent=label||'Location';
  document.getElementById('coords').textContent=lat.toFixed(3)+', '+lon.toFixed(3);
};
window.resetView=function(){
  targetRot={y:world.rotation.y, x:0};spin=true;marker.visible=false;ring.visible=false;
  document.getElementById('place').textContent='O.R.I.O.N. Globe';
  document.getElementById('coords').textContent='Whole Earth — drag to rotate, scroll to zoom.';
  document.getElementById('news').innerHTML='';
  document.getElementById('footage').style.display='none';
  hideMap();
  cam.position.set(0,60,340);controls.target.set(0,0,0);controls.update();
};
// Google Maps street-level card — correlates the 3-D globe with a real map.
// Driven by the raw query so Google resolves an exact street/apartment, not
// just the city centre.  Keyless (output=embed); Street View / open-in-Maps
// hand off to the system browser via the page's navigation interceptor.
window._mapLL=null;
window.showMap=function(query,lat,lon,label){
  const q=(query&&query.length)?query:(lat+','+lon);
  document.getElementById('mapframe').src=
    'https://www.google.com/maps?q='+encodeURIComponent(q)+'&z=17&output=embed';
  document.getElementById('maptitle').textContent=label||q;
  document.getElementById('mapaddr').textContent=q;
  window._mapLL={lat:lat,lon:lon,q:q};
  document.getElementById('mapcard').style.display='block';
};
window.hideMap=function(){
  document.getElementById('mapcard').style.display='none';
  document.getElementById('mapframe').src='about:blank';
};
window.streetView=function(){if(!window._mapLL)return;
  window.location.href='https://www.google.com/maps/@?api=1&map_action=pano&viewpoint='
    +window._mapLL.lat+','+window._mapLL.lon;};
window.openMaps=function(){if(!window._mapLL)return;
  window.location.href='https://www.google.com/maps/search/?api=1&query='
    +encodeURIComponent(window._mapLL.q);};
window.showNews=function(items,footageUrl){
  const n=document.getElementById('news');n.innerHTML='';
  (items||[]).forEach(it=>{const a=document.createElement('a');a.textContent='• '+it.title;
    a.href=it.url;a.target='_blank';n.appendChild(a);});
  const f=document.getElementById('footage');
  if(footageUrl){f.style.display='inline-block';f.onclick=()=>{window.location.href=footageUrl;};}
  else f.style.display='none';
};
document.getElementById('reset').onclick=()=>window.resetView();
function animate(){requestAnimationFrame(animate);
  pulse+=0.05;const s=1+Math.sin(pulse)*0.18;ring.scale.set(s,s,s);
  clouds.rotation.y+=0.0003;  // gentle cloud drift for depth
  if(spin)world.rotation.y+=0.0007;
  if(targetRot){world.rotation.y+=(targetRot.y-world.rotation.y)*0.05;
    world.rotation.x+=(targetRot.x-world.rotation.x)*0.05;}
  controls.update();renderer.render(scene,cam);}
animate();
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);});
</script></body></html>"""


class _GlobePage(QWebEnginePage):
    """Opens external link clicks in the system browser, keeping the globe up."""

    def acceptNavigationRequest(self, url: Any, nav_type: Any, is_main_frame: bool) -> bool:  # type: ignore[override]
        try:
            u = url.toString()
        except Exception:
            u = str(url)
        # Only intercept real navigations away to web pages; the globe itself
        # lives on an about:blank document, so its own load is never http(s).
        if is_main_frame and u.startswith(("http://", "https://")):
            webbrowser.open(u)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class GlobeView(QWidget):
    """The globe page: search bar + textured Three.js Earth + news overlay."""

    def __init__(self, bus: OrionBus, parent: QWidget | None = None,
                 geo: Any | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.geo = geo          # GeoIntelligenceEngine — town-level accuracy
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        heading = QLabel("GLOBAL INTELLIGENCE GLOBE")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)

        row = QHBoxLayout()
        self.place_input = QLineEdit()
        self.place_input.setPlaceholderText("Fly to a place — e.g. Tokyo, Kyiv, São Paulo…")
        self.place_input.returnPressed.connect(self._on_go)
        go = QPushButton("FLY THERE")
        go.clicked.connect(self._on_go)
        reset = QPushButton("⌂ RESET")
        reset.setToolTip("Zoom back out to the whole Earth")
        reset.clicked.connect(self.reset_view)
        row.addWidget(self.place_input, 1)
        row.addWidget(go)
        row.addWidget(reset)
        layout.addLayout(row)

        self.view: Any = None
        self._layout = layout
        self._built = False
        if WEBENGINE_OK:
            # LAZY: the QWebEngineView (Chromium — the heaviest RAM consumer) is
            # NOT created until this tab is first shown, so ORION starts lighter
            # and never pays for the globe if you never open it.
            self._placeholder = QLabel("🌍  The intelligence globe loads when you open this tab.")
            self._placeholder.setObjectName("mutedLabel")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._placeholder, 1)
        else:
            self._placeholder = None
            fallback = QLabel(
                "The 3-D globe needs PyQt6-WebEngine.\n"
                "Install it with:  pip install PyQt6-WebEngine\n"
                "News and footage still work through voice commands."
            )
            fallback.setObjectName("mutedLabel")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setWordWrap(True)
            layout.addWidget(fallback, 1)

        self.bus.globe_request.connect(self.go_to_place)

    def _ensure_built(self) -> None:
        """Create the WebEngine view on first use (lazy — saves RAM at startup)."""
        if self._built or not WEBENGINE_OK:
            return
        self._built = True
        if self._placeholder is not None:
            self._placeholder.hide()
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None
        self.view = QWebEngineView()
        self.view.setPage(_GlobePage(self.view))
        self.view.setHtml(GLOBE_HTML)
        self._layout.addWidget(self.view, 1)
        self.bus.log.emit("GLOBE: renderer initialised on first view (lazy-loaded).")

    def showEvent(self, event: Any) -> None:
        self._ensure_built()
        super().showEvent(event)

    # ── control ───────────────────────────────────────────────────────────────

    def _on_go(self) -> None:
        place = self.place_input.text().strip()
        if place:
            self.go_to_place(place)

    def reset_view(self) -> None:
        self._js("resetView()")
        self.bus.log.emit("GLOBE: view reset to the whole Earth.")

    def go_to_place(self, place: str) -> None:
        place = str(place or "").strip()
        if not place:
            return
        self._ensure_built()   # build the renderer if ORION was asked by voice first
        self.bus.log.emit(f"GLOBE: travelling to {place}.")
        try:
            asyncio.create_task(self._go(place))
        except RuntimeError:
            pass

    async def _go(self, place: str) -> None:
        geo = await self._geocode(place)
        if geo is None:
            self.bus.log.emit(f"GLOBE: couldn't locate '{place}' (offline or unknown).")
            self._js("showNews([{'title':'Could not locate that place — offline or unknown.',"
                     "'url':'about:blank'}], '')")
            return
        lat, lon, label = geo
        self._js(f"flyTo({lat},{lon},{json.dumps(label)})")
        # Correlate the globe with a real street-level Google map. The raw query
        # (not just the city centroid) lets Google resolve an exact street or
        # address — useful for research on a specific place.
        self._js(f"showMap({json.dumps(place)}, {lat}, {lon}, {json.dumps(label)})")
        items = await self._news(label)
        footage = f"https://www.youtube.com/results?search_query={quote_plus(label + ' news today')}"
        self._js(f"showNews({json.dumps(items)}, {json.dumps(footage)})")
        self.bus.dashboard_event.emit("globe", {"place": label, "lat": lat, "lon": lon})

    # ── data ──────────────────────────────────────────────────────────────────

    async def _geocode(self, place: str) -> tuple[float, float, str] | None:
        # Prefer the GeoIntelligenceEngine (OSM/Nominatim) — it resolves towns,
        # villages and districts worldwide, not merely major cities.
        if self.geo is not None:
            try:
                results = await self.geo.geocode(place, limit=1)
                if results:
                    p = results[0]
                    label = ", ".join(x for x in (p.name, p.country) if x) or place
                    return p.lat, p.lon, label
            except Exception:
                pass  # fall through to the lightweight open-meteo geocoder
        try:
            url = ("https://geocoding-api.open-meteo.com/v1/search?count=1&name="
                   + quote_plus(place))
            timeout = ClientTimeout(total=8.0, connect=3.0)
            async with ClientSession(timeout=timeout) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        return None
                    data = await r.json()
            results = data.get("results") or []
            if not results:
                return None
            top = results[0]
            label = ", ".join(x for x in (top.get("name"), top.get("country")) if x)
            return float(top["latitude"]), float(top["longitude"]), label or place
        except Exception:
            return None

    async def _news(self, place: str) -> list[dict[str, str]]:
        try:
            url = ("https://news.google.com/rss/search?q="
                   + quote_plus(f"{place} news") + "&hl=en-GB&gl=GB&ceid=GB:en")
            timeout = ClientTimeout(total=10.0, connect=3.0)
            async with ClientSession(timeout=timeout) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        return []
                    raw = await r.text()
        except Exception:
            return []
        items: list[dict[str, str]] = []
        for block in re.findall(r"<item>(.*?)</item>", raw, re.S)[:8]:
            t = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.S)
            link = re.search(r"<link>(.*?)</link>", block, re.S)
            if t:
                items.append({"title": html.unescape(re.sub(r"\s+", " ", t.group(1))).strip()[:130],
                              "url": link.group(1).strip() if link else "about:blank"})
        return items

    # ── helpers ───────────────────────────────────────────────────────────────

    def _js(self, code: str) -> None:
        if self.view is not None:
            self.view.page().runJavaScript(code)
