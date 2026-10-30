"""
GlobeView — Mark XXXIII world reports on the existing CesiumJS globe.

Esri imagery, place search, map cards, voice camera controls and live aviation
remain available. Read-only world layers add CelesTrak satellite predictions,
USGS earthquakes, NOAA tsunami bulletins and Met Office UK warning summaries.
Reports retain their source, retrieval time, coverage and freshness. Bulletins
without official coordinates appear in the report list rather than on the map.

The renderer uses bounded primitive collections, explicit render requests and
owned task cancellation. Hiding the page stops feed polling and its render
loop. Rebuilding or closing it removes handlers and rejects late responses.
The globe imagery and Cesium engine require internet; saved reports can still
appear when the renderer cannot load. This is not an offline navigation map.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
import webbrowser  # noqa: F401  (kept: see ..browser.open_url)
from typing import Any
from urllib.parse import quote_plus

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from ..lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")
from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..geo import looks_like_uk_postcode
from ..globe_coordinates import parse_coordinates
from ..globe_locations import GlobeLocation, valid_coordinates
from ..browser import open_url
from ..utils import first_line
from .. import background

try:
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebChannel import QWebChannel
    WEBENGINE_OK = True
except Exception:  # pragma: no cover - optional PyQt6-WebEngine
    WEBENGINE_OK = False
    QWebEnginePage = object  # type: ignore


# The page is given a document URL ON the Cesium CDN origin, so Cesium's web
# workers (Workers/*.js) are same-origin and start normally.  Loaded from
# about:blank (plain setHtml) the worker bootstrapper receives module names it
# cannot resolve — "importScripts … 'createVerticesFromHeightmap' is invalid"
# — no surface geometry is ever built, and the globe renders as empty space.
GLOBE_BASE_URL = "https://cdn.jsdelivr.net/npm/cesium@1.115.0/Build/Cesium/orion-globe.html"

GLOBE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;height:100%;background:#03040a;overflow:hidden;font-family:'Segoe UI',Arial}
  #cesiumContainer{position:absolute;inset:0}
  #panel{position:absolute;top:12px;right:12px;width:330px;max-height:80%;overflow:auto;z-index:10;
    background:rgba(8,8,14,.84);border:1px solid #7c0d1e;border-radius:10px;color:#fff;padding:12px}
  #panel h2{margin:0 0 4px;font-size:16px;color:#ff5c73}
  #panel .sub{color:#a9a9b2;font-size:11px;margin-bottom:8px}
  #news a{display:block;color:#fff;text-decoration:none;font-size:12px;line-height:1.35;
    padding:6px 0;border-bottom:1px solid #241018;cursor:pointer}
  #news a:hover{color:#ff5c73}
  .btnrow{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
  .btn{background:#991024;color:#fff;padding:6px 12px;border-radius:8px;font-size:12px;cursor:pointer;border:1px solid #ff1a3c}
  .btn.ghost{background:transparent}
  #hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);color:#8a8a96;font-size:11px;
    text-align:center;z-index:10;text-shadow:0 1px 3px #000}
  /* Holographic zoom rail — direct camera control, HUD-style. */
  #zoomrail{position:absolute;left:14px;top:50%;transform:translateY(-50%);z-index:10;
    display:flex;flex-direction:column;gap:8px;align-items:center}
  .zbtn{width:40px;height:40px;display:flex;align-items:center;justify-content:center;
    font-size:22px;font-weight:700;color:#7fe9ff;cursor:pointer;user-select:none;
    background:linear-gradient(180deg,rgba(10,26,32,.86),rgba(6,14,20,.86));
    border:1px solid #0a7d8c;border-radius:11px;box-shadow:0 0 14px rgba(0,229,255,.16),inset 0 0 8px rgba(0,229,255,.08);
    transition:all .12s ease}
  .zbtn:hover{color:#eafcff;border-color:#00e5ff;box-shadow:0 0 20px rgba(0,229,255,.42),inset 0 0 12px rgba(0,229,255,.18)}
  .zbtn:active{transform:scale(.92)}
  #zoomrail .ztick{width:2px;height:44px;border-radius:2px;
    background:linear-gradient(180deg,rgba(0,229,255,.05),rgba(0,229,255,.5),rgba(0,229,255,.05))}
  #loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#5b6b7a;font-size:13px;z-index:5}
  #mapcard{position:absolute;bottom:12px;left:12px;width:380px;max-width:46%;z-index:10;
    background:rgba(8,8,14,.92);border:1px solid #7c0d1e;border-radius:10px;color:#fff;padding:8px;display:none}
  #mapcard .maphead{display:flex;justify-content:space-between;align-items:center;
    font-size:13px;color:#ff5c73;font-weight:700;margin-bottom:6px}
  #mapcard .maphead .mx{cursor:pointer;color:#a9a9b2;padding:0 4px}
  #mapaddr{color:#a9a9b2;font-size:11px;margin:0 2px 6px}
  #mapframe{width:100%;height:230px;border:0;border-radius:6px;background:#0b0f14}
  /* Aircraft detail card — opened by clicking an aircraft on the live layer. */
  #acpanel{position:absolute;top:12px;left:70px;width:250px;z-index:11;display:none;
    background:rgba(8,8,14,.9);border:1px solid #0a7d8c;border-radius:10px;color:#fff;
    padding:10px 12px;box-shadow:0 0 14px rgba(0,229,255,.16)}
  #acpanel .achead{display:flex;justify-content:space-between;align-items:center;
    font-size:15px;font-weight:700;color:#7fe9ff;margin-bottom:6px}
  #acpanel .mx{cursor:pointer;color:#a9a9b2;padding:0 4px}
  #acrows{width:100%;border-collapse:collapse;font-size:12px}
  #acrows td{padding:2px 0;vertical-align:top}
  #acrows td:first-child{color:#8a8a96;padding-right:10px;white-space:nowrap}
  /* Provider credit display remains visible, including its full attribution popup. */
  #credit{position:absolute;bottom:0;left:8px;right:8px;z-index:12;color:#b9b9bd;
    font-size:10px;text-shadow:0 1px 2px #000;pointer-events:auto}
  html,body{background:#090909;color:#f0f0f2}
  button{font:inherit;cursor:pointer}
  button.mx{background:transparent;border:0;font:inherit}
  #panel,#mapcard,#acpanel,#worldpanel,#eventpanel{background:rgba(17,17,19,.94);
    border:1px solid #39393d;border-radius:14px;box-shadow:0 12px 34px #0005}
  #panel{top:14px;right:14px;width:288px;padding:16px;max-height:34%}
  #panel h2,#mapcard .maphead,#acpanel .achead{color:#f2f2f3}
  #coords{line-height:1.5}
  #news a:hover{color:#f06b82}
  .btn{background:#aa2442;border:1px solid #bc3653;padding:8px 12px;border-radius:9px}
  .btn.ghost{border-color:#45454a;color:#e6e6e8}
  .btn:hover{filter:brightness(1.18)}
  .zbtn{color:#f5f5f5;background:#202023;border:1px solid #45454a;box-shadow:none;border-radius:10px}
  .zbtn:hover{color:#fff;border-color:#bb3653;box-shadow:0 0 0 2px #a5213c33}
  #zoomrail .ztick{height:16px;background:#4a4a50}
  #acpanel{top:14px;left:68px;box-shadow:0 12px 34px #0005}
  #mapcard{bottom:38px;max-width:46%}
  #worldpanel{position:absolute;right:14px;bottom:38px;width:288px;padding:16px;z-index:10;
    max-height:49%;overflow:auto;font-size:12px}
  .worldhead{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}
  .worldhead h3{margin:0;font-size:13px;font-weight:600;letter-spacing:.04em}
  #refreshWorld{color:#dcdce0;background:#242427;border:1px solid #45454a;border-radius:8px;padding:6px 8px}
  #refreshWorld:disabled{opacity:.45;cursor:default}
  .layerrow{display:grid;grid-template-columns:1fr auto;align-items:center;gap:10px;padding:9px 0;border-top:1px solid #303034}
  .layerrow label{display:flex;gap:8px;align-items:center;cursor:pointer}
  .layerrow input{accent-color:#b32e4c;width:15px;height:15px;margin:0}
  .layerstate{color:#a8a8ad;font-size:10px;max-width:108px;text-align:right}
  .layerstate[data-stale=true]{color:#e8b580}
  #worldsummary{color:#a8a8ad;margin-top:9px;font-size:11px;line-height:1.45}
  #worldlist{display:flex;flex-direction:column;gap:5px;margin-top:10px}
  .worlditem{text-align:left;width:100%;color:#eee;background:#222225;border:1px solid #38383d;
    padding:9px;border-radius:8px;font-size:11px;line-height:1.35}
  .worlditem:hover{border-color:#a12c45}
  .worlditem small{display:block;color:#aaaab0;margin-top:3px}
  .worldgroup{font-size:11px;font-weight:600;color:#c1c1c6;margin:8px 0 3px}
  .worldpager{display:flex;justify-content:space-between;align-items:center;gap:6px;color:#a8a8ad;font-size:10px}
  .worldpager button{background:#242427;color:#ddd;border:1px solid #45454a;padding:5px 7px;border-radius:7px;font-size:10px}
  .worldpager button:disabled{opacity:.4;cursor:default}
  #eventpanel{position:absolute;left:68px;top:14px;z-index:13;width:300px;padding:16px;display:none}
  #eventtitle{font-size:15px;line-height:1.4;margin:0;padding-right:20px}
  #eventclose{position:absolute;right:10px;top:10px;border:0;background:transparent;color:#bbb;font-size:18px}
  #eventdetails{white-space:pre-line;color:#c1c1c6;font-size:12px;line-height:1.6;max-height:280px;overflow:auto;margin:12px 0}
  #eventsource{color:#eee;font-size:12px}
  /* Overlays: day and night, rain radar, distance - and the point card. */
  #overlays{display:flex;flex-direction:column;gap:2px;margin-bottom:10px}
  #overlaynote{color:#a8a8ad;font-size:11px;line-height:1.45;margin:2px 0 8px;min-height:0}
  #routeform{display:flex;flex-direction:column;gap:4px;margin:0 0 12px}
  #routeform input{background:#0d0d0e;border:1px solid #2c2c2c;border-radius:6px;color:#ededed;
    font-size:11px;padding:5px 7px;outline:none}
  #routeform input:focus{border-color:#8a8a8a}
  #routeform button{align-self:flex-start;background:#212121;border:1px solid #2c2c2c;border-radius:6px;
    color:#ededed;font-size:11px;padding:4px 10px;cursor:pointer}
  #routeform button:disabled{opacity:.5;cursor:default}
  #pointpanel{position:absolute;left:68px;bottom:38px;z-index:13;width:300px;padding:16px;display:none;
    background:rgba(17,17,19,.94);border:1px solid #39393d;border-radius:14px;box-shadow:0 12px 34px #0005}
  #pointtitle{font-size:15px;line-height:1.4;margin:0;padding-right:20px}
  #pointclose{position:absolute;right:10px;top:10px;border:0;background:transparent;color:#bbb;font-size:18px}
  #pointdetails{white-space:pre-line;color:#c1c1c6;font-size:12px;line-height:1.6;margin:10px 0}
  .btn:focus-visible,.zbtn:focus-visible,.worlditem:focus-visible,button:focus-visible{outline:2px solid #ed8395;outline-offset:2px}
  /* Mark XXXX: ORION's crimson-and-silver HUD. Glass panels with a crimson
     edge, silver type, a corner frame and a live position readout. */
  :root{--crimson:#c8102e;--crimson-soft:#c8102e55;--silver:#c9ccd2;--dim:#8e9198;
    --glass:rgba(10,10,14,.78);--line:#3a3b41}
  #panel,#mapcard,#acpanel,#worldpanel,#eventpanel,#pointpanel{background:var(--glass);
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid var(--line);
    border-top:2px solid var(--crimson);border-radius:12px;box-shadow:0 14px 38px #000a,inset 0 1px 0 #ffffff0d}
  #panel h2,#pointtitle,#eventtitle{color:#f4f4f6;letter-spacing:.06em;font-weight:600}
  #panel h2{font-size:14px;text-transform:uppercase}
  .worldhead h3{color:var(--silver);text-transform:uppercase;font-size:11px;letter-spacing:.14em}
  #panel .sub,#worldsummary,.layerstate,#pointdetails,#eventdetails{color:var(--dim)}
  .btn{background:linear-gradient(180deg,#d0213e,#9c0d24);border:1px solid #e0425c}
  .btn.ghost{background:#ffffff08;border-color:var(--line);color:var(--silver)}
  .zbtn{background:var(--glass);border:1px solid var(--line);color:var(--silver);border-radius:50%;
    backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
  .zbtn:hover{border-color:var(--crimson);color:#fff;box-shadow:0 0 0 3px var(--crimson-soft)}
  #zoomrail .ztick{background:linear-gradient(180deg,transparent,var(--crimson),transparent);width:2px;height:22px}
  #acpanel{border-top-color:var(--crimson)}
  #acpanel .achead{color:#f4f4f6}
  .layerrow input{accent-color:var(--crimson)}
  .worlditem{background:#ffffff07;border-color:var(--line)}
  .worlditem:hover,#routeform input:focus{border-color:var(--crimson)}
  #hud{position:absolute;inset:10px 10px 30px;pointer-events:none;z-index:4}
  #hud i{position:absolute;width:22px;height:22px;border:solid #c9ccd255}
  #hud .tl{left:0;top:0;border-width:1px 0 0 1px}#hud .tr{right:0;top:0;border-width:1px 1px 0 0}
  #hud .bl{left:0;bottom:0;border-width:0 0 1px 1px}#hud .br{right:0;bottom:0;border-width:0 1px 1px 0}
  #readout{position:absolute;left:50%;transform:translateX(-50%);bottom:30px;z-index:9;pointer-events:none;
    font:11px/1 'Cascadia Mono',Consolas,monospace;letter-spacing:.08em;color:var(--silver);
    background:var(--glass);border:1px solid var(--line);border-left:2px solid var(--crimson);
    border-radius:6px;padding:6px 10px;white-space:pre;text-shadow:0 1px 2px #000}
  #readout:empty{display:none}
  @media(max-width:700px){#panel{width:220px;right:10px}#worldpanel{width:220px;right:10px}
    #mapcard{max-width:50%;width:300px}#eventpanel{width:250px}}
  @media(max-width:480px){#panel{left:60px;right:8px;width:auto;max-height:24%;padding:12px}
    #worldpanel{left:60px;right:8px;width:auto;max-height:38%;padding:12px}
    #mapcard{left:8px;right:8px;max-width:none;width:auto;z-index:14}
    #eventpanel{left:60px;right:8px;width:auto;padding:12px}#zoomrail{left:10px}}
</style>
<link href="https://cdn.jsdelivr.net/npm/cesium@1.115.0/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
</head><body>
<div id="cesiumContainer"></div>
<div id="loading">Rendering Earth…</div>
<div id="panel"><h2 id="place">O.R.I.O.N. Globe</h2>
  <div class="sub" id="coords">Earth view</div>
  <div id="news"></div>
  <div class="btnrow">
    <button class="btn" id="footage" style="display:none">Footage</button>
    <button class="btn ghost" id="reset">Reset View</button>
  </div>
</div>
<div id="mapcard">
  <div class="maphead"><span id="maptitle">Street Map</span><button class="mx" aria-label="Close map" onclick="hideMap()">×</button></div>
  <div id="mapaddr"></div>
  <iframe id="mapframe" src="about:blank" loading="lazy" referrerpolicy="no-referrer-when-downgrade"
    allowfullscreen></iframe>
  <div class="btnrow">
    <button class="btn" onclick="streetView()">Street View</button>
    <button class="btn ghost" onclick="openMaps()">Google Maps</button>
  </div>
</div>
<div id="acpanel">
  <div class="achead"><span id="accs">—</span><button class="mx" aria-label="Close aircraft details" onclick="avSelect(null)">×</button></div>
  <table id="acrows"></table>
</div>
<div id="zoomrail">
  <button class="zbtn" id="zin" title="Zoom in" aria-label="Zoom in">+</button>
  <div class="ztick"></div>
  <button class="zbtn" id="zout" title="Zoom out" aria-label="Zoom out">−</button>
</div>
<div id="worldpanel" aria-label="World layers">
  <div class="worldhead"><h3>Overlays</h3></div>
  <div id="overlays"></div><div id="overlaynote"></div>
  <form id="routeform" autocomplete="off" aria-label="Driving route">
    <input id="routefrom" maxlength="80" placeholder="From (blank = where you are)" aria-label="Route from">
    <input id="routeto" maxlength="80" placeholder="Route to…" aria-label="Route to">
    <button id="routego" type="submit">Show Route</button>
  </form>
  <div class="worldhead"><h3>World Layers</h3><button id="refreshWorld" disabled>Refresh</button></div>
  <div id="worldlayers"></div><div id="worldsummary">Choose a layer to view its latest reports.</div>
  <div id="worldlist"></div>
</div>
<div id="pointpanel" aria-label="Place details"><button id="pointclose" aria-label="Close place details">×</button>
  <h3 id="pointtitle"></h3><div id="pointdetails"></div>
  <div class="btnrow"><button class="btn" id="pointmap">Street Map</button>
  <button class="btn ghost" id="pointmeasure">Measure from Here</button></div></div>
<div id="eventpanel"><button id="eventclose" aria-label="Close report">×</button>
  <h3 id="eventtitle"></h3><div id="eventdetails"></div><a id="eventsource" rel="noopener noreferrer">Official Source</a></div>
<div id="credit"></div>
<div id="hud" aria-hidden="true"><i class="tl"></i><i class="tr"></i><i class="bl"></i><i class="br"></i></div>
<div id="readout" aria-live="off"></div>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
// The page is injected via setHtml() (origin about:blank), so Cesium cannot
// infer where its web workers live.  Without this, the surface-geometry
// workers (createVerticesFromHeightmap) die with importScripts errors and the
// globe never renders — stars only.  Declare the base explicitly, BEFORE
// Cesium.js loads, so every worker resolves to an absolute CDN URL.
window.CESIUM_BASE_URL='https://cdn.jsdelivr.net/npm/cesium@1.115.0/Build/Cesium/';
</script>
<script src="https://cdn.jsdelivr.net/npm/cesium@1.115.0/Build/Cesium/Cesium.js"
  onerror="document.getElementById('loading').textContent='The globe needs an internet connection.'"></script>
<script>
// ═══════════════════════════════════════════════════════════════════════════
// O.R.I.O.N. Global Intelligence Globe — Mark X.6.1.
// CesiumJS virtual globe (the engine class Google Earth belongs to): real
// satellite imagery, smooth orbit-to-street zoom, tilt, atmosphere, stars.
//
// X.6 built the base layer with ArcGisMapServerImageryProvider.fromUrl — an
// ASYNC metadata request to Esri.  When that one request failed inside the
// embedded Chromium, the layer never became ready and Cesium never rendered
// the globe surface at all: stars, no Earth.  X.6.1 constructs the tile
// providers SYNCHRONOUSLY from URL templates (no metadata round-trip), so the
// Earth always renders; if the satellite tiles themselves error out, a
// watchdog swaps in OpenStreetMap so the globe is never blank.
// The Python bridge API is unchanged: flyTo / resetView / showNews / showMap.
// ═══════════════════════════════════════════════════════════════════════════
let viewer=null, spin=false, marker=null, baseLayer=null, labelLayer=null, fellBack=false;
let globeActive=true, globeDisposed=false, rendererGeneration=0;
const rendererCleanup=[];
let restoreTimer=null;
function listen(target,event,fn,options){
  target.addEventListener(event,fn,options);
  rendererCleanup.push(()=>target.removeEventListener(event,fn,options));
}
function watch(event,fn){ const remove=event.addEventListener(fn); rendererCleanup.push(remove); }
function safeUrl(value){
  try{const url=new URL(String(value||''));
    return url.protocol==='https:'&&!url.username&&!url.password?url.href:null;
  }catch(_){return null;}
}

// Keyless tile sources, all constructed synchronously (never blocks the globe).
const IMAGERY={
  esri:()=>new Cesium.UrlTemplateImageryProvider({
    url:'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maximumLevel:19, credit:new Cesium.Credit('Imagery: Esri, Vantor, Earthstar Geographics, and the GIS User Community',true)}),
  esriLabels:()=>new Cesium.UrlTemplateImageryProvider({
    url:'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    maximumLevel:19, credit:new Cesium.Credit('Labels: Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community',true)}),
  osm:()=>new Cesium.OpenStreetMapImageryProvider({url:'https://tile.openstreetmap.org/'}),
};

function setSub(text){const e=document.getElementById('coords'); if(e)e.textContent=text;}

function initGlobe(){
  if(globeDisposed||typeof Cesium==='undefined') return;    // offline — message already shown
  rendererGeneration++;
  Cesium.Ion.defaultAccessToken=undefined;   // keyless providers only
  const esri=IMAGERY.esri();
  viewer=new Cesium.Viewer('cesiumContainer',{
    baseLayer:new Cesium.ImageryLayer(esri),
    baseLayerPicker:false, geocoder:false, timeline:false, animation:false,
    homeButton:false, sceneModePicker:false, navigationHelpButton:false,
    fullscreenButton:false, selectionIndicator:false, infoBox:false,
    creditContainer:document.getElementById('credit'),
    requestRenderMode:true,maximumRenderTimeChange:Infinity,
    contextOptions:{webgl:{preserveDrawingBuffer:true}},
  });
  baseLayer=viewer.imageryLayers.get(0);
  // Place names + borders reference layer (city labels as you zoom — the
  // Google-Earth feel).  Synchronous too; a failure only costs the labels.
  try{ labelLayer=viewer.imageryLayers.addImageryProvider(IMAGERY.esriLabels()); }catch(e){}
  // Watchdog: if the satellite tiles error out repeatedly in the first half
  // minute, swap the whole base layer to OpenStreetMap — never a black globe.
  let tileErrors=0; const started=Date.now();
  watch(esri.errorEvent,()=>{
    tileErrors++;
    if(!fellBack && tileErrors>=8 && (Date.now()-started)<45000){
      fellBack=true;
      try{
        viewer.imageryLayers.remove(baseLayer,true);
        if(labelLayer) viewer.imageryLayers.remove(labelLayer,true);
        baseLayer=viewer.imageryLayers.addImageryProvider(IMAGERY.osm());
        setSub('Satellite imagery unreachable — street-map fallback engaged.');
      }catch(e){}
    }
  });
  const scene=viewer.scene;
  scene.globe.enableLighting=false;
  // ── zoom-out crash guard (Mark X.7.1) ───────────────────────────────────
  // Zooming OUT put the WHOLE Earth back on screen, so Cesium's per-fragment
  // sky- and ground-atmosphere shaders suddenly had to cover almost every
  // pixel of the viewport in one frame.  On this GPU that sustained shader
  // spike tripped a driver timeout (TDR) that surfaced — in the *main*
  // process, via the shared-GL composite path — as a native access violation
  // (see config/crash_reports/faulthandler.log): no Python traceback, and it
  // killed the whole app before the render-process rebuild could run.
  // Turning the two whole-globe atmosphere passes off removes that spike.
  // The satellite imagery + starfield still give the Earth-from-orbit look,
  // and zoom now scales smoothly from orbit to street level without crashing.
  scene.globe.showGroundAtmosphere=false;
  scene.skyAtmosphere.show=false;
  scene.fog.enabled=false;
  // Render only when the view actually changes (drag, zoom, flyTo, idle spin)
  // instead of a constant 60 fps.  When nothing moves the globe now costs the
  // GPU nothing, which removes the steady heat/driver-reset crash class too.
  scene.requestRenderMode=true;
  scene.maximumRenderTimeChange=Infinity;
  // Visible even before the first tile arrives — never a black void.
  scene.globe.baseColor=Cesium.Color.fromCssColorString('#25282e');
  scene.screenSpaceCameraController.minimumZoomDistance=180;      // street level
  scene.screenSpaceCameraController.maximumZoomDistance=40e6;     // ~orbit cap
  // ── fast-navigation crash hardening (Mark X.7.2) ────────────────────────
  // Zooming / panning FAST used to flood the GPU with high-detail tiles and
  // full-resolution frames quicker than the driver could drain them, which
  // re-tripped the access violation.  The rule here is simple: the harder you
  // push the camera, the LIGHTER each frame becomes, so a burst of input can
  // never outrun the renderer.
  // Mark XXXX: the globe draws in its own process now (page_host), so a driver
  // reset costs the globe a restart, not ORION. Motion is smoother - 60 frames
  // a second while it moves (requestRenderMode still draws nothing at rest), a
  // short glide after a zoom or drag - and every frame in motion stays light.
  viewer.targetFrameRate=60;
  try{ scene.postProcessStages.fxaa.enabled=false; }catch(e){}   // drop full-screen AA pass
  const ssc=scene.screenSpaceCameraController;
  ssc.inertiaZoom=0.45;                               // a short glide, not a coast
  ssc.inertiaSpin=0.78; ssc.inertiaTranslate=0.78;
  scene.globe.preloadAncestors=false;                 // stop eager tile-prefetch storms
  scene.globe.preloadSiblings=false;
  scene.globe.tileCacheSize=100;
  try{ Cesium.RequestScheduler.maximumRequests=12;    // fewer simultaneous tile uploads
       Cesium.RequestScheduler.maximumRequestsPerServer=6;
       Cesium.RequestScheduler.throttleRequests=true; }catch(e){}
  // While the camera is in motion, render at 60% resolution and coarser tiles
  // (a fraction of the fragments + geometry per frame), then snap back to
  // crisp shortly after it settles.  This is the core fast-zoom / fast-drag
  // guard: the renderer stays light for exactly as long as you keep moving,
  // so a burst of input can never outrun the GPU.  Driven off BOTH the wheel
  // and pointer directly (not only Cesium's moveStart/moveEnd) so a dropped
  // moveEnd can never leave the globe stuck soft — and the idle auto-spin,
  // which never touches wheel/pointer, always stays crisp.
  let _userDown=false;
  // A new resolution resizes (and so clears) the canvas: draw straight away,
  // or a click that never moved the camera left the globe black until the
  // crisp timer fired (Mark XXXX).
  function _lowRes(){ if(viewer.resolutionScale===0.75)return;
      viewer.resolutionScale=0.75; scene.globe.maximumScreenSpaceError=4; scene.requestRender(); }
  function _crisp(){ if(viewer&&!viewer.isDestroyed()){ viewer.resolutionScale=1.0;
      scene.globe.maximumScreenSpaceError=2; scene.requestRender(); } }
  function _scheduleCrisp(ms){ clearTimeout(restoreTimer);
      restoreTimer=setTimeout(()=>{ if(!_userDown&&globeActive) _crisp(); }, ms); }
  watch(scene.camera.moveStart,()=>{ _lowRes(); clearTimeout(restoreTimer); });
  watch(scene.camera.moveEnd,()=>{ _scheduleCrisp(200); });
  listen(viewer.canvas,'pointerdown',()=>{ _userDown=true; clearTimeout(restoreTimer); });   // moveStart lightens
  listen(window,'pointerup',()=>{ _userDown=false; _scheduleCrisp(200); });
  listen(viewer.canvas,'wheel',()=>{ _lowRes(); _scheduleCrisp(300); },{passive:true});
  // Idle orbit spin until the user (or a flyTo) takes over.  Moving the camera
  // implicitly requests a render, so the spin still animates under
  // requestRenderMode; a change here also keeps requestRender explicit.
  watch(viewer.clock.onTick,()=>{
    if(spin&&globeActive){ viewer.scene.camera.rotate(Cesium.Cartesian3.UNIT_Z,-0.0006);
              viewer.scene.requestRender(); }
  });
  listen(viewer.canvas,'pointerdown',()=>{spin=false;});
  // Mark XXXX: where the cursor points, how high the camera is and which way it
  // faces - text only, coalesced to one update a frame, never a render.
  const readout=document.getElementById('readout');
  let roPending=false, roPos=null;
  function roDraw(){
    roPending=false;
    if(!viewer||viewer.isDestroyed()||!readout)return;
    const cam=viewer.camera, h=cam.positionCartographic.height;
    const alt=h>=1e4?Math.round(h/1000).toLocaleString('en-GB')+' km':Math.round(h).toLocaleString('en-GB')+' m';
    const hdg=(Math.round(Cesium.Math.toDegrees(cam.heading))+360)%360;
    let at='—';
    if(roPos){
      const c=cam.pickEllipsoid(roPos,viewer.scene.globe.ellipsoid);
      if(c){const g=Cesium.Cartographic.fromCartesian(c);
        const la=Cesium.Math.toDegrees(g.latitude),lo=Cesium.Math.toDegrees(g.longitude);
        at=Math.abs(la).toFixed(4)+'° '+(la>=0?'N':'S')+'  '+Math.abs(lo).toFixed(4)+'° '+(lo>=0?'E':'W');}
    }
    readout.textContent=at+'   ALT '+alt+'   HDG '+hdg+'°';
  }
  function roQueue(){ if(!roPending){roPending=true;requestAnimationFrame(roDraw);} }
  function canvasPoint(e){ const r=viewer.canvas.getBoundingClientRect();
    return new Cesium.Cartesian2(e.clientX-r.left,e.clientY-r.top); }
  listen(viewer.canvas,'mousemove',e=>{roPos=canvasPoint(e);roQueue();},{passive:true});
  listen(viewer.canvas,'mouseleave',()=>{roPos=null;roQueue();});
  scene.camera.percentageChanged=0.005;
  watch(scene.camera.changed,roQueue);
  roQueue();
  // Double-click a spot to fly in towards it (a third of the height each time).
  try{ viewer.screenSpaceEventHandler.removeInputAction(Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK); }catch(e){}
  listen(viewer.canvas,'dblclick',e=>{
    const c=viewer.camera.pickEllipsoid(canvasPoint(e),viewer.scene.globe.ellipsoid);
    if(!c)return;
    spin=false;
    const g=Cesium.Cartographic.fromCartesian(c), h=viewer.camera.positionCartographic.height;
    viewer.camera.flyTo({destination:Cesium.Cartesian3.fromRadians(g.longitude,g.latitude,Math.max(400,h/3)),
                         duration:1.1});
  });
  // Arrow keys steer, + and - zoom, Home resets - whenever a text box is not typing.
  listen(window,'keydown',e=>{
    if(!viewer||viewer.isDestroyed()||!globeActive||e.ctrlKey||e.altKey||e.metaKey)return;
    const t=e.target; if(t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.isContentEditable))return;
    const cam=viewer.camera, h=cam.positionCartographic.height;
    const step=Math.min(0.12,Math.max(0.00002,h/4e7));
    const k=e.key;
    if(k==='ArrowLeft')cam.rotateLeft(step);
    else if(k==='ArrowRight')cam.rotateRight(step);
    else if(k==='ArrowUp')cam.rotateDown(step);      // Cesium's "down" carries the view north
    else if(k==='ArrowDown')cam.rotateUp(step);
    else if(k==='+'||k==='=')cam.zoomIn(Math.max(0,Math.min(h*0.3,h-ssc.minimumZoomDistance)));
    else if(k==='-'||k==='_')cam.zoomOut(Math.max(0,Math.min(h*0.45,ssc.maximumZoomDistance-h)));
    else if(k==='Home'){const r=document.getElementById('reset');if(r)r.click();}
    else return;
    spin=false; e.preventDefault(); viewer.scene.requestRender();
  });
  // WebGL context loss is the classic "globe went blank / crashed" cause on
  // flaky GPU drivers.  Report it and rebuild the viewer when the context is
  // restored, instead of leaving a dead canvas.
  listen(viewer.canvas,'webglcontextlost',(e)=>{
    e.preventDefault(); console.error('WEBGL_CONTEXT_LOST');
    const l=document.getElementById('loading');
    if(l){l.style.display='';l.textContent='Restoring the globe…';}
  },false);
  listen(viewer.canvas,'webglcontextrestored',()=>{
    console.error('WEBGL_CONTEXT_RESTORED');
    destroyRenderer(); fellBack=false;
    initGlobe();
  },false);
  viewer.camera.setView({destination:Cesium.Cartesian3.fromDegrees(0.0,18.0,24e6)});
  // Drop the loading veil on the first rendered frame (the globe is already
  // visible then, tiles stream in behind it).
  const l=document.getElementById('loading');
  if(l){ const once=()=>{ l.style.display='none'; viewer.scene.postRender.removeEventListener(once); };
         watch(viewer.scene.postRender,once); }
  viewer.useDefaultRenderLoop=globeActive;
  if(typeof worldRebuild==='function') worldRebuild();
  if(typeof overlayRebuild==='function') overlayRebuild();
  if(AV.on&&AV.byId.size){avEnsure();AV.byId.forEach(avSync);avLabels();avPump();}
  window.__globeReady=true;
}

// Global JS error trap → routed to the crash recorder via the console hook.
window.onerror=function(msg,src,line){console.error('JS_ERROR: '+msg+' @'+src+':'+line); return false;};
window.addEventListener('unhandledrejection',function(e){
  try{console.error('JS_ERROR: unhandled promise '+(e.reason&&e.reason.message||e.reason));}catch(_){}
});

// ── live aviation layer ─────────────────────────────────────────────────────
// Every aircraft is ONE billboard in ONE BillboardCollection, and only the
// nearest few (plus the selected one) get a label from ONE LabelCollection.
// An Entity per aircraft would carry a property graph Cesium re-evaluates
// every frame, and the layer has to stay cheap enough that render-on-demand
// still means something.  Between feed updates each position is dead-reckoned
// from ground speed + track; a new report snaps it back to the truth.
const AV={bb:null,lb:null,scene:null,click:null,ring:null,byId:new Map(),
  selected:null,active:true,on:false,timer:null,panelTimer:null,fps:0,
  tickMs:0,lastPrune:0,col:null,pos:null};
const AV_MAX=500, AV_STALE_S=60, AV_LABELS=8, AV_MIN_H=60;
let AV_ICON='';
function avIcon(){
  if(AV_ICON) return AV_ICON;
  // Drawn once and shared BY URL: billboards given the same image string share
  // one texture-atlas slot, where a canvas object would get a slot EACH.
  const c=document.createElement('canvas'); c.width=c.height=48;
  const g=c.getContext('2d'); g.translate(24,24); g.beginPath();
  [[0,-21],[3,-14],[3,-5],[20,5],[20,9],[3,4],[2.5,14],[8,18],[8,21],[0,19],
   [-8,21],[-8,18],[-2.5,14],[-3,4],[-20,9],[-20,5],[-3,-5],[-3,-14]]
    .forEach((p,i)=>i?g.lineTo(p[0],p[1]):g.moveTo(p[0],p[1]));
  g.closePath(); g.lineWidth=3; g.strokeStyle='rgba(0,0,0,.8)'; g.stroke();
  g.fillStyle='#ffffff'; g.fill();
  AV_ICON=c.toDataURL('image/png'); return AV_ICON;
}
function avEnsure(){
  if(!viewer) return false;
  if(AV.scene===viewer.scene && AV.bb && !AV.bb.isDestroyed()) return true;
  // First use, or the viewer was rebuilt after a WebGL context loss and the
  // old collections died with it: keep the aircraft, rebuild the primitives.
  AV.scene=viewer.scene;
  AV.pos=new Cesium.Cartesian3();
  AV.col={air:Cesium.Color.fromCssColorString('#ededf0'),
          gnd:Cesium.Color.fromCssColorString('#9aa3ad'),
          sel:Cesium.Color.fromCssColorString('#ff5c73')};
  AV.bb=viewer.scene.primitives.add(new Cesium.BillboardCollection());
  AV.lb=viewer.scene.primitives.add(new Cesium.LabelCollection());
  AV.byId.forEach(a=>{a.b=null;a.l=null;});
  AV.ring=null;
  watch(viewer.scene.preUpdate,avTick);
  AV.click=new Cesium.ScreenSpaceEventHandler(viewer.canvas);
  AV.click.setInputAction(avClick,Cesium.ScreenSpaceEventType.LEFT_CLICK);
  return true;
}
function avMoving(a){ return !a.gnd && a.v>0 && a.hdg!=null; }
function avPlace(a,now){
  // Flat-earth step from the last reported fix: exact enough for the <=60 s
  // an aircraft is ever extrapolated before it is dropped as stale.
  let lat=a.lat, lon=a.lon, h=(a.alt==null?0:a.alt);
  if(avMoving(a)){
    const dt=Math.min(Math.max(now-a.t,0),AV_STALE_S);
    const d=a.v*dt/6371008.8, r=a.hdg*Math.PI/180;
    lat+=d*Math.cos(r)*57.29577951308232;
    lon+=d*Math.sin(r)*57.29577951308232/Math.max(0.01,Math.cos(a.lat*Math.PI/180));
    if(a.vr) h+=a.vr*Math.min(dt,30);
  }
  return Cesium.Cartesian3.fromDegrees(lon,lat,Math.max(AV_MIN_H,h),undefined,AV.pos);
}
function avTick(){                      // scene.preUpdate: every rendered frame
  if(!AV.on||!AV.active||!AV.byId.size) return;
  const t0=performance.now(), now=Date.now()/1000;
  AV.byId.forEach(a=>{ if(!a.b||!avMoving(a)) return;
    const p=avPlace(a,now); a.b.position=p; if(a.l) a.l.position=p; });
  AV.tickMs=AV.tickMs*0.9+(performance.now()-t0)*0.1;
}
function avSync(a){
  if(!a.b){
    a.b=AV.bb.add({image:avIcon(),id:a.id,width:24,height:24,
      alignedAxis:Cesium.Cartesian3.UNIT_Z,             // "up" = north, so
      scaleByDistance:new Cesium.NearFarScalar(2e4,1.0,5e6,0.45)});
  }
  a.b.rotation=(a.hdg==null)?0:-Cesium.Math.toRadians(a.hdg);  // ...CCW rotation = -heading
  a.b.color=(a.id===AV.selected)?AV.col.sel:(a.gnd?AV.col.gnd:AV.col.air);
  a.b.position=avPlace(a,Date.now()/1000);
}
function avRemove(a){
  if(a.b) AV.bb.remove(a.b); if(a.l) AV.lb.remove(a.l);
  AV.byId.delete(a.id);
  if(AV.selected===a.id) avSelect(null);
}
function avPrune(now){
  AV.byId.forEach(a=>{ if(now-a.t>AV_STALE_S) avRemove(a); });
}
function avLabels(){
  const want=new Set([...AV.byId.values()].filter(a=>a.d!=null)
    .sort((x,y)=>x.d-y.d).slice(0,AV_LABELS).map(a=>a.id));
  if(AV.selected) want.add(AV.selected);
  AV.byId.forEach(a=>{
    if(!want.has(a.id)){ if(a.l){AV.lb.remove(a.l); a.l=null;} return; }
    const text=(a.cs||a.id)+(a.gnd?'  ground':(a.alt!=null?'  '+Math.round(a.alt).toLocaleString()+' m':''));
    if(!a.l) a.l=AV.lb.add({id:a.id,text:text,font:'12px Segoe UI',fillColor:Cesium.Color.WHITE,
      outlineColor:Cesium.Color.BLACK,outlineWidth:3,style:Cesium.LabelStyle.FILL_AND_OUTLINE,
      horizontalOrigin:Cesium.HorizontalOrigin.LEFT,pixelOffset:new Cesium.Cartesian2(14,-4),
      distanceDisplayCondition:new Cesium.DistanceDisplayCondition(0,2.5e6)});
    else if(a.l.text!==text) a.l.text=text;
    a.l.position=a.b?a.b.position:avPlace(a,Date.now()/1000);
  });
}
function avClear(){
  if(AV.bb&&!AV.bb.isDestroyed()) AV.bb.removeAll();
  if(AV.lb&&!AV.lb.isDestroyed()) AV.lb.removeAll();
  AV.byId.clear(); avSelect(null);
}
function avPump(){
  clearTimeout(AV.timer); AV.timer=null;
  if(!viewer||!AV.on||!AV.active||!AV.byId.size) return;
  const now=Date.now()/1000;
  if(now-AV.lastPrune>5){ AV.lastPrune=now; avPrune(now); avLabels(); }
  let vmax=0; AV.byId.forEach(a=>{ if(avMoving(a)&&a.v>vmax) vmax=a.v; });
  if(vmax<=0){ AV.fps=0; return; }
  // Frames only as fast as the fastest aircraft crosses the screen at this
  // zoom: about one pixel of motion per frame, clamped to 0.5–15 fps.  From
  // orbit a jet covers a fraction of a pixel a second and the globe stays all
  // but idle; zoomed onto a city it gets the frames it needs to glide.
  const cam=viewer.camera, h=Math.max(1,cam.positionCartographic.height);
  const fovy=(cam.frustum&&cam.frustum.fovy)||1.0;
  const mpp=2*h*Math.tan(fovy/2)/Math.max(1,viewer.canvas.clientHeight);
  AV.fps=Math.min(15,Math.max(0.5,vmax/mpp));
  viewer.scene.requestRender();
  AV.timer=setTimeout(avPump,1000/AV.fps);
}
function avClick(e){
  if(!AV.on) return;
  const p=viewer.scene.pick(e.position);
  if(p&&(p.collection===AV.bb||p.collection===AV.lb)&&AV.byId.has(p.id)) avSelect(p.id);
  else if(AV.selected) avSelect(null);
}
function avSelect(id){
  const prev=AV.selected; AV.selected=(id&&AV.byId.has(id))?id:null;
  [prev,AV.selected].forEach(k=>{ const a=k&&AV.byId.get(k); if(a&&a.b) avSync(a); });
  if(AV.lb) avLabels();
  clearInterval(AV.panelTimer); AV.panelTimer=null;
  const el=document.getElementById('acpanel');
  if(!AV.selected){ el.style.display='none'; }
  else { el.style.display='block'; avPanel();
         if(AV.active) AV.panelTimer=setInterval(avPanel,1000); }
  if(viewer) viewer.scene.requestRender();
}
window.avSelect=avSelect;
function avPanel(){
  const a=AV.selected&&AV.byId.get(AV.selected);
  if(!a) return;
  const age=Math.max(0,Date.now()/1000-a.t);
  const rows=[['ICAO24',a.id]];
  if(a.reg||a.typ) rows.push(['Aircraft',[a.reg,a.typ].filter(Boolean).join(' · ')]);
  if(a.ctry) rows.push(['Registered in',a.ctry]);
  rows.push(['Altitude',a.gnd?'On the ground':(a.alt==null?'not reported':
    Math.round(a.alt).toLocaleString()+' m ('+Math.round(a.alt/0.3048).toLocaleString()+' ft)')]);
  rows.push(['Speed',a.v==null?'not reported':Math.round(a.v*3.6)+' km/h ('+Math.round(a.v/0.514444)+' kt)']);
  rows.push(['Heading',a.hdg==null?'not reported':Math.round(a.hdg)+'°']);
  if(a.vr!=null&&!a.gnd) rows.push(['Vertical',(a.vr>0?'+':'')+a.vr.toFixed(1)+' m/s']);
  if(a.d!=null) rows.push(['Distance',a.d.toFixed(1)+' km from centre at last report']);
  rows.push(['Last update',Math.round(age)+' s ago'+(a.src?' · '+a.src:'')]);
  // textContent throughout: callsigns and registrations come from a public
  // feed and are never parsed as markup.
  document.getElementById('accs').textContent=a.cs||'Unidentified';
  const t=document.getElementById('acrows'); t.textContent='';
  rows.forEach(r=>{ const tr=document.createElement('tr');
    r.forEach(v=>{ const td=document.createElement('td'); td.textContent=v; tr.appendChild(td); });
    t.appendChild(tr); });
}
// Python → page: a batch from aviation.globe_batch().  Upserts by ICAO24; an
// aircraft missing from a batch keeps gliding until its last fix is 60 s old,
// which is how one that has left the circle fades out rather than blinking.
window.orionAircraft=function(batch){
  if(!batch||!avEnsure()) return 0;
  AV.on=true;
  if(batch.reset) avClear();
  const now=Date.now()/1000, list=batch.aircraft||[];
  for(let i=0;i<Math.min(list.length,AV_MAX*2);i++){
    const n=list[i]; if(!n||!n.id||n.lat==null||n.lon==null) continue;
    let a=AV.byId.get(n.id);
    if(!a){ if(AV.byId.size>=AV_MAX) continue; a={id:n.id,b:null,l:null,t:0}; AV.byId.set(n.id,a); }
    else if(n.t&&n.t<a.t) continue;                 // never step back to an older fix
    a.cs=n.cs||''; a.lat=n.lat; a.lon=n.lon; a.alt=n.alt; a.v=n.v; a.hdg=n.hdg;
    a.vr=n.vr; a.gnd=!!n.gnd; a.t=n.t||now; a.reg=n.reg||''; a.typ=n.typ||'';
    a.ctry=n.ctry||''; a.d=n.d; a.src=batch.source||'';
    avSync(a);
  }
  avPrune(now); avLabels(); avPanel(); avPump();
  viewer.scene.requestRender();
  return AV.byId.size;
};
window.orionAircraftFocus=function(lat,lon,radiusKm,label){
  if(!avEnsure()) return;
  spin=false; AV.on=true; avClear();
  const r=Math.max(1,radiusKm||25)*1000, c=Cesium.Cartesian3.fromDegrees(lon,lat,0);
  // The search circle: ONE static outline entity, not one per aircraft.
  if(AV.ring) viewer.entities.remove(AV.ring);
  AV.ring=viewer.entities.add({position:c,ellipse:{semiMajorAxis:r,semiMinorAxis:r,height:0,
    fill:false,outline:true,outlineColor:Cesium.Color.fromCssColorString('#c8c8ce').withAlpha(0.55)}});
  viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(c,r),
    {offset:new Cesium.HeadingPitchRange(0,Cesium.Math.toRadians(-62),r*2.6),duration:3.0});
  document.getElementById('place').textContent='Air Traffic'+(label?' — '+label:'');
  setSub('Live aircraft within '+Math.round(radiusKm)+' km — click one for its details.');
};
window.orionAircraftLayer=function(on){
  AV.on=!!on;
  if(on) return;
  avClear(); clearTimeout(AV.timer); AV.timer=null;
  if(AV.ring&&viewer){ viewer.entities.remove(AV.ring); } AV.ring=null;
  if(viewer) viewer.scene.requestRender();
};
window.orionAviationStats=()=>({n:AV.byId.size,labels:AV.lb?AV.lb.length:0,
  fps:AV.fps,tickMs:AV.tickMs,on:AV.on,active:AV.active,
  heap:(performance.memory?performance.memory.usedJSHeapSize:null)});

// World reports are static batches, not a per-frame entity graph. Coordinates
// are drawn only when supplied by the official feed or its SGP4 prediction.
const WORLD_SPEC={satellites:{label:'Satellites',cap:64,colour:'#f2f2f3'},
  earthquakes:{label:'Earthquakes',cap:250,colour:'#d44766'},
  tsunami:{label:'Tsunami Bulletins',cap:200,colour:'#e2e2e5'},
  weather:{label:'UK Weather Warnings',cap:200,colour:'#b5b5bd'},
  events:{label:'Natural Events',cap:300,colour:'#e8743b'}};
// Natural events are coloured by what they are, so a glance separates the
// fires from the storms from the ice.
const EVENT_COLOURS={'Wildfires':'#e8743b','Severe Storms':'#9fc3e6','Volcanoes':'#d44766',
  'Sea and Lake Ice':'#eef3f7','Floods':'#5b8fd6','Landslides':'#a8875a','Drought':'#c9a227',
  'Dust and Haze':'#c2b280','Snow':'#ffffff','Temperature Extremes':'#ff5c73',
  'Water Color':'#3ab0a2','Manmade':'#9a9aa3','Earthquakes':'#d44766'};
function worldColour(key,item){
  if(key==='events'&&item&&EVENT_COLOURS[item.category])return EVENT_COLOURS[item.category];
  return WORLD_SPEC[key].colour;
}
const WORLD={enabled:new Set(),layers:new Map(),selected:null,handler:null,
  busy:false,bridge:null,active:true,pages:new Map()};
function worldControlState(){
  document.getElementById('refreshWorld').disabled=WORLD.busy||!WORLD.enabled.size||!WORLD.bridge;
}
function worldControls(){
  const host=document.getElementById('worldlayers'); host.textContent='';
  Object.entries(WORLD_SPEC).forEach(([key,spec])=>{
    const row=document.createElement('div'); row.className='layerrow';
    const label=document.createElement('label'),box=document.createElement('input');
    box.type='checkbox';box.id='layer-'+key;
    box.onchange=()=>{
      window.orionWorldLayer(key,box.checked);
      if(WORLD.bridge) WORLD.bridge.setWorldLayer(key,box.checked);
    };
    label.append(box,document.createTextNode(spec.label));
    const state=document.createElement('span');state.className='layerstate';state.id='state-'+key;
    state.textContent='Off';row.append(label,state);host.appendChild(row);
  });
  document.getElementById('refreshWorld').onclick=()=>{
    if(!WORLD.bridge||WORLD.busy||!WORLD.enabled.size)return;
    window.orionWorldBusy(true);WORLD.bridge.refreshWorld();
  };
  document.getElementById('eventclose').onclick=()=>worldSelect(null);
}
function worldEnsure(key){
  if(!WORLD_SPEC[key])return null;
  let layer=WORLD.layers.get(key);
  if(!layer){layer={points:null,byId:new Map(),report:null};WORLD.layers.set(key,layer);}
  if(!viewer||viewer.isDestroyed())return layer;
  if(!layer.points||layer.points.isDestroyed()){
    layer.points=viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    layer.points.show=WORLD.enabled.has(key);
  }
  // A moving event's path (a storm's track): one polyline per event, in the
  // same batch-and-render model as the points.
  if(key==='events'&&(!layer.lines||layer.lines.isDestroyed())){
    layer.lines=viewer.scene.primitives.add(new Cesium.PolylineCollection());
    layer.lines.show=WORLD.enabled.has(key);
  }
  if(!WORLD.handler){
    WORLD.handler=new Cesium.ScreenSpaceEventHandler(viewer.canvas);
    WORLD.handler.setInputAction(e=>{
      const picked=viewer.scene.pick(e.position);
      if(picked&&typeof picked.id==='string'&&picked.id.startsWith('world:'))worldSelect(picked.id);
    },Cesium.ScreenSpaceEventType.LEFT_CLICK);
  }
  return layer;
}
function worldDraw(key){
  const layer=worldEnsure(key);if(!layer)return;
  if(layer.points)layer.points.removeAll();if(layer.lines)layer.lines.removeAll();layer.byId.clear();
  const items=(layer.report&&Array.isArray(layer.report.items)?layer.report.items:[]).slice(0,WORLD_SPEC[key].cap);
  items.forEach((item,index)=>{
    if(!item||typeof item!=='object')return;
    const id='world:'+key+':'+String(item.id||index).slice(0,400);
    if(layer.byId.has(id))return;
    layer.byId.set(id,item);
    const lat=item.latitude,lon=item.longitude;
    if(!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180)return;
    const height=key==='satellites'?Math.max(0,Math.min(1e8,(Number(item.altitude_km)||0)*1000)):2000;
    if(layer.points)layer.points.add({id:id,position:Cesium.Cartesian3.fromDegrees(lon,lat,height),
      pixelSize:key==='earthquakes'?Math.max(5,Math.min(13,(Number(item.magnitude)||2)*1.5))
        :key==='events'?(item.category==='Severe Storms'?10:7):6,
      color:Cesium.Color.fromCssColorString(worldColour(key,item)),
      outlineColor:Cesium.Color.BLACK,outlineWidth:1});
    const track=Array.isArray(item.track)?item.track.slice(-60):[];
    if(layer.lines&&track.length>1){
      const flat=[];
      track.forEach(pt=>{if(Array.isArray(pt)&&Number.isFinite(pt[0])&&Number.isFinite(pt[1])
        &&Math.abs(pt[0])<=90&&Math.abs(pt[1])<=180)flat.push(pt[1],pt[0]);});
      if(flat.length>=4)layer.lines.add({id:id,width:2,
        positions:Cesium.Cartesian3.fromDegreesArray(flat),
        material:Cesium.Material.fromType('Color',{color:Cesium.Color.fromCssColorString(
          worldColour(key,item)).withAlpha(0.75)})});
    }
  });
  if(layer.points)layer.points.show=WORLD.enabled.has(key);
  if(layer.lines)layer.lines.show=WORLD.enabled.has(key);
}
function worldTime(value){
  const date=new Date(value||'');return Number.isNaN(date.getTime())?'Not supplied':date.toLocaleString('en-GB');
}
function worldRenderList(){
  const list=document.getElementById('worldlist');list.textContent='';let count=0;
  WORLD.enabled.forEach(key=>{
    const layer=WORLD.layers.get(key),report=layer&&layer.report;
    const state=document.getElementById('state-'+key);
    if(!report){state.textContent=WORLD.busy?'Loading…':'Waiting';return;}
    const items=Array.isArray(report.items)?report.items:[];
    const stale=!!report.stale;
    state.dataset.stale=String(stale);
    const entries=layer?[...layer.byId.entries()]:[];
    state.textContent=report.status==='unavailable'?'Unavailable':(stale?'Saved · ':'')
      +entries.length+(entries.length<items.length?' / '+items.length:'')+' reports';
    state.title=[report.source,report.coverage,'Updated '+worldTime(report.fetched_at),report.error].filter(Boolean).join('\n');
    count+=entries.length;
    if(!items.length){
      const note=document.createElement('div');note.className='layerstate';note.style.maxWidth='none';note.style.textAlign='left';
      note.textContent=report.error||('No current '+WORLD_SPEC[key].label.toLowerCase()+'.');list.appendChild(note);
    }
    const totalPages=Math.max(1,Math.ceil(entries.length/8));
    const page=Math.min(WORLD.pages.get(key)||0,totalPages-1);WORLD.pages.set(key,page);
    if(entries.length){const heading=document.createElement('div');heading.className='worldgroup';
      heading.textContent=WORLD_SPEC[key].label;list.appendChild(heading);}
    entries.slice(page*8,page*8+8).forEach(([id,item])=>{
      const button=document.createElement('button');button.className='worlditem';
      button.textContent=String(item.title||WORLD_SPEC[key].label).slice(0,180);
      const note=document.createElement('small');
      note.textContent=(key==='satellites'?'Predicted position':WORLD_SPEC[key].label)+' · '
        +worldTime(key==='satellites'?item.predicted_at:item.event_at);
      button.appendChild(note);button.onclick=()=>worldSelect(id,true);list.appendChild(button);
    });
    if(totalPages>1){
      const pager=document.createElement('div');pager.className='worldpager';pager.dataset.layer=key;
      const prev=document.createElement('button'),next=document.createElement('button'),label=document.createElement('span');
      prev.textContent='Previous';next.textContent='Next';prev.disabled=page===0;next.disabled=page===totalPages-1;
      label.textContent=(page*8+1)+'–'+Math.min(entries.length,page*8+8)+' of '+entries.length;
      prev.onclick=()=>{WORLD.pages.set(key,page-1);worldRenderList();};
      next.onclick=()=>{WORLD.pages.set(key,page+1);worldRenderList();};
      pager.append(prev,label,next);list.appendChild(pager);
    }
  });
  document.getElementById('worldsummary').textContent=WORLD.enabled.size
    ?(WORLD.busy?'Refreshing reports…':count+' reports · select a report for its source and time.')
    :'Choose a layer to view its latest reports.';
  worldControlState();
}
function worldSelect(id,focus=false){
  WORLD.selected=null;let found=null,report=null,key=null;
  if(id)WORLD.layers.forEach((layer,name)=>{if(layer.byId.has(id)){found=layer.byId.get(id);report=layer.report;key=name;}});
  const panel=document.getElementById('eventpanel');
  if(!found||!WORLD.enabled.has(key)){panel.style.display='none';return;}
  WORLD.selected=id;panel.style.display='block';
  document.getElementById('eventtitle').textContent=String(found.title||WORLD_SPEC[key].label);
  const details=[report.source,report.coverage,'Reported: '+worldTime(found.event_at),
    'Retrieved: '+worldTime(report.fetched_at)];
  if(key==='satellites')details.push('SGP4 predicted position: '+worldTime(found.predicted_at),
    'Orbital epoch: '+worldTime(found.epoch),'Altitude: '+Number(found.altitude_km||0).toFixed(1)+' km');
  if(key==='earthquakes')details.push('Magnitude: '+(found.magnitude??'Not supplied'),'Depth: '+(found.depth_km??'Not supplied')+' km');
  if(key==='events')details.push('Kind: '+(found.category||'Natural event'),
    Array.isArray(found.track)&&found.track.length>1?'Track: '+found.track.length+' reported positions (drawn on the globe)':'');
  if(found.details)details.push(typeof found.details==='string'?found.details:JSON.stringify(found.details));
  if(report.stale)details.push(report.error?'Saved report — the source could not provide a fresh update.'
    :'Saved report — an update is due.');
  if(report.error)details.push(report.error);
  if(!Number.isFinite(found.latitude)||!Number.isFinite(found.longitude))details.push('This source does not supply map coordinates.');
  document.getElementById('eventdetails').textContent=details.filter(Boolean).join('\n');
  const link=document.getElementById('eventsource'),url=safeUrl(found.source_url||report.source_url);
  link.style.display=url?'':'none';if(url)link.href=url;else link.removeAttribute('href');
  if(focus&&viewer){
    clearLocation();
    if(focusCoordinate(found.latitude,found.longitude,
        key==='satellites'?Math.max(2e6,(Number(found.altitude_km)||0)*1000+2e6):150000)){
      document.getElementById('place').textContent=found.title||WORLD_SPEC[key].label;
      document.getElementById('coords').textContent=found.latitude.toFixed(6)+', '
        +found.longitude.toFixed(6)+' · '+(key==='satellites'?'Predicted ground position':'Source reference point');
      showMap('',found.latitude,found.longitude,found.title);
    }
  }
}
window.orionWorldLayer=function(key,on){
  if(!WORLD_SPEC[key]||globeDisposed)return;
  on?WORLD.enabled.add(key):WORLD.enabled.delete(key);
  document.getElementById('layer-'+key).checked=!!on;
  const layer=WORLD.layers.get(key);if(layer&&layer.points)layer.points.show=!!on;
  if(layer&&layer.lines)layer.lines.show=!!on;
  if(!on){document.getElementById('state-'+key).textContent='Off';worldSelect(null);}
  worldRenderList();if(viewer)viewer.scene.requestRender();
};
window.orionWorldBusy=function(on){WORLD.busy=!!on;worldRenderList();};
window.orionWorldSnapshot=function(snapshot){
  if(globeDisposed||!snapshot||!snapshot.layers)return;
  Object.entries(snapshot.layers).forEach(([key,report])=>{
    if(!WORLD_SPEC[key]||!report||typeof report!=='object')return;
    const layer=worldEnsure(key);if(!layer)return;
    if(layer.report&&Date.parse(layer.report.fetched_at)>Date.parse(report.fetched_at))return;
    layer.report=report;worldDraw(key);
  });
  WORLD.busy=false;worldRenderList();if(WORLD.selected)worldSelect(WORLD.selected);
  if(viewer)viewer.scene.requestRender();
};
function worldRebuild(){WORLD.layers.forEach((layer,key)=>{layer.points=null;layer.lines=null;worldDraw(key);});}
window.orionWorldStats=()=>({enabled:[...WORLD.enabled],busy:WORLD.busy,
  points:[...WORLD.layers.values()].reduce((n,layer)=>n+(layer.points?layer.points.length:0),0),
  reports:[...WORLD.layers.values()].reduce((n,layer)=>n+layer.byId.size,0),generation:rendererGeneration});
worldControls();

// ── overlays: day and night, rain radar, distance; the point card ─────────
// Day and night lights the globe from the real sun (the terminator moves)
// and shows NASA's night-lights imagery on the dark side only. Rain radar is
// RainViewer's latest frame. Clicking open ground asks ORION (through the
// bridge, as two numbers - never a URL) what is there; measuring adds
// great-circle legs between clicks.
const OVERLAY={lighting:false,night:null,clock:null,radarOn:false,radar:null,radarReq:0,
  radarTimer:null,measure:false,pts:[],lines:null,click:null,point:null};
const OVERLAY_SPEC={daynight:'Day and Night',radar:'Rain Radar',measure:'Measure Distance'};
function overlayNote(text){const e=document.getElementById('overlaynote');if(e)e.textContent=text||'';}
function overlayControls(){
  const host=document.getElementById('overlays');host.textContent='';
  Object.entries(OVERLAY_SPEC).forEach(([key,label])=>{
    const row=document.createElement('div');row.className='layerrow';
    const lab=document.createElement('label'),box=document.createElement('input');
    box.type='checkbox';box.id='overlay-'+key;
    box.onchange=()=>{ if(key==='daynight')window.orionDayNight(box.checked);
      else if(key==='radar')window.orionRadar(box.checked); else window.orionMeasure(box.checked); };
    lab.append(box,document.createTextNode(label));row.append(lab);host.appendChild(row);
  });
  document.getElementById('pointclose').onclick=()=>{document.getElementById('pointpanel').style.display='none';};
  document.getElementById('pointmap').onclick=()=>{const p=OVERLAY.point;if(p)showMap('',p.lat,p.lon,p.title||'');};
  document.getElementById('pointmeasure').onclick=()=>{const p=OVERLAY.point;if(!p)return;
    window.orionMeasure(true);OVERLAY.pts=[[p.lat,p.lon]];measureDraw();};
}
function setOverlayBox(key,on){const b=document.getElementById('overlay-'+key);if(b)b.checked=!!on;}
window.orionDayNight=function(on){
  OVERLAY.lighting=!!on;setOverlayBox('daynight',on);
  if(!viewer||viewer.isDestroyed())return;
  try{
    viewer.scene.globe.enableLighting=!!on;
    viewer.clock.shouldAnimate=false;viewer.clock.currentTime=Cesium.JulianDate.now();
    if(on){
      if(!OVERLAY.night){
        OVERLAY.night=viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
          url:'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/2016-01-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png',
          maximumLevel:8,credit:new Cesium.Credit('Night lights: NASA Earth Observatory, VIIRS Black Marble (GIBS)',true)}));
        OVERLAY.night.dayAlpha=0.0;OVERLAY.night.nightAlpha=1.0;
      }
      OVERLAY.night.show=true;
      clearInterval(OVERLAY.clock);
      OVERLAY.clock=setInterval(()=>{if(!viewer||viewer.isDestroyed())return;
        viewer.clock.currentTime=Cesium.JulianDate.now();viewer.scene.requestRender();},60000);
      overlayNote('Lit by the sun as it is now; city lights show on the night side.');
    }else{
      if(OVERLAY.night)OVERLAY.night.show=false;
      clearInterval(OVERLAY.clock);OVERLAY.clock=null;overlayNote('');
    }
  }catch(e){overlayNote('Day and night is unavailable on this graphics driver.');}
  viewer.scene.requestRender();
};
function radarRemove(){
  if(OVERLAY.radar&&viewer&&!viewer.isDestroyed()){try{viewer.imageryLayers.remove(OVERLAY.radar,true);}catch(_){}}
  OVERLAY.radar=null;
}
window.orionRadar=async function(on){
  OVERLAY.radarOn=!!on;setOverlayBox('radar',on);const req=++OVERLAY.radarReq;
  clearInterval(OVERLAY.radarTimer);OVERLAY.radarTimer=null;
  if(!on){radarRemove();overlayNote('');if(viewer)viewer.scene.requestRender();return;}
  overlayNote('Loading the rain radar…');
  try{
    const res=await fetch('https://api.rainviewer.com/public/weather-maps.json',{credentials:'omit'});
    const data=await res.json();
    if(req!==OVERLAY.radarReq||!OVERLAY.radarOn||!viewer||viewer.isDestroyed())return;
    const host=String(data&&data.host||'');
    const frames=data&&data.radar&&Array.isArray(data.radar.past)?data.radar.past:[];
    const last=frames[frames.length-1];
    if(host!=='https://tilecache.rainviewer.com'||!last||!/^\/v2\/radar\/[A-Za-z0-9]{1,40}$/.test(String(last.path)))
      throw new Error('unexpected radar index');
    radarRemove();
    const provider=new Cesium.UrlTemplateImageryProvider({
      url:host+last.path+'/256/{z}/{x}/{y}/2/1_1.png',maximumLevel:7,
      credit:new Cesium.Credit('Rain radar: RainViewer',true)});
    // RainViewer keeps about two hours of frames; tiles of an expired frame
    // fail as cross-origin errors (38 in one evening, 27 Oct). A failing
    // frame is replaced by the newest one, at most once a minute.
    provider.errorEvent.addEventListener(()=>{
      if(OVERLAY.radarOn&&Date.now()-(OVERLAY.radarRetry||0)>60000){
        OVERLAY.radarRetry=Date.now();window.orionRadar(true);}
    });
    OVERLAY.radar=viewer.imageryLayers.addImageryProvider(provider);
    OVERLAY.radarAt=Date.now();
    OVERLAY.radar.alpha=0.72;
    const at=new Date(Number(last.time)*1000);
    overlayNote('Rain radar as of '+(Number.isNaN(at.getTime())?'the latest frame'
      :at.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'}))+' (updates every ten minutes).');
    OVERLAY.radarTimer=setInterval(()=>{if(OVERLAY.radarOn&&globeActive)window.orionRadar(true);},600000);
    viewer.scene.requestRender();
  }catch(e){if(req===OVERLAY.radarReq)overlayNote('Rain radar is unavailable right now.');}
};
function greatCircleKm(a,b){
  const r=Math.PI/180,dl=(b[0]-a[0])*r,dn=(b[1]-a[1])*r;
  const h=Math.sin(dl/2)**2+Math.cos(a[0]*r)*Math.cos(b[0]*r)*Math.sin(dn/2)**2;
  return 2*6371.0088*Math.asin(Math.min(1,Math.sqrt(h)));
}
function measureDraw(){
  if(!viewer||viewer.isDestroyed())return;
  if(!OVERLAY.lines||OVERLAY.lines.isDestroyed())
    OVERLAY.lines=viewer.scene.primitives.add(new Cesium.PolylineCollection());
  OVERLAY.lines.removeAll();
  let km=0;
  for(let i=1;i<OVERLAY.pts.length;i++)km+=greatCircleKm(OVERLAY.pts[i-1],OVERLAY.pts[i]);
  if(OVERLAY.pts.length>1){
    const flat=[];OVERLAY.pts.forEach(p=>flat.push(p[1],p[0]));
    OVERLAY.lines.add({positions:Cesium.Cartesian3.fromDegreesArray(flat),width:3,
      arcType:Cesium.ArcType?Cesium.ArcType.GEODESIC:undefined,
      material:Cesium.Material.fromType('Color',{color:Cesium.Color.fromCssColorString('#f2f2f3')})});
  }
  const legs=Math.max(0,OVERLAY.pts.length-1);
  overlayNote(OVERLAY.pts.length<2?'Click two or more points on the globe to measure.'
    :'Distance: '+km.toLocaleString('en-GB',{maximumFractionDigits:1})+' km ('
      +(km*0.621371).toLocaleString('en-GB',{maximumFractionDigits:1})+' miles) over '
      +legs+(legs===1?' leg':' legs')+', along the Earth\'s surface.');
  viewer.scene.requestRender();
}
window.orionMeasure=function(on){
  OVERLAY.measure=!!on;setOverlayBox('measure',on);OVERLAY.pts=[];
  if(!on){if(OVERLAY.lines&&!OVERLAY.lines.isDestroyed())OVERLAY.lines.removeAll();
    overlayNote('');if(viewer)viewer.scene.requestRender();return;}
  measureDraw();
};
window.orionMeasureStats=()=>({points:OVERLAY.pts.length,
  km:OVERLAY.pts.slice(1).reduce((n,p,i)=>n+greatCircleKm(OVERLAY.pts[i],p),0)});
// Road route (geo action=route): ONE polyline and two end points, camera
// fitted to the whole journey. Python sends numbers and plain text only.
function routeDraw(fly){
  const r=OVERLAY.route; if(!r||!viewer||viewer.isDestroyed())return;
  if(!OVERLAY.routeLines||OVERLAY.routeLines.isDestroyed())
    OVERLAY.routeLines=viewer.scene.primitives.add(new Cesium.PolylineCollection());
  if(!OVERLAY.routePts||OVERLAY.routePts.isDestroyed())
    OVERLAY.routePts=viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
  OVERLAY.routeLines.removeAll();OVERLAY.routePts.removeAll();
  const flat=[];let w=180,e=-180,s=90,n=-90;
  r.coords.forEach(c=>{const lon=Number(c[0]),lat=Number(c[1]);
    if(!Number.isFinite(lon)||!Number.isFinite(lat))return;
    flat.push(lon,lat);w=Math.min(w,lon);e=Math.max(e,lon);s=Math.min(s,lat);n=Math.max(n,lat);});
  if(flat.length<4)return;
  OVERLAY.routeLines.add({positions:Cesium.Cartesian3.fromDegreesArray(flat),width:4,
    material:Cesium.Material.fromType('Color',{color:Cesium.Color.fromCssColorString('#f2f2f3')})});
  [[flat[0],flat[1]],[flat[flat.length-2],flat[flat.length-1]]].forEach((p,i)=>
    OVERLAY.routePts.add({position:Cesium.Cartesian3.fromDegrees(p[0],p[1]),pixelSize:i?12:9,
      color:Cesium.Color.WHITE,outlineColor:Cesium.Color.BLACK,outlineWidth:2}));
  overlayNote(String(r.summary||'').slice(0,300)+' (Esc clears the route.)');
  if(fly){
    const padLon=Math.max(0.05,(e-w)*0.25),padLat=Math.max(0.05,(n-s)*0.25);
    try{viewer.camera.flyTo({destination:Cesium.Rectangle.fromDegrees(
      Math.max(-180,w-padLon),Math.max(-90,s-padLat),Math.min(180,e+padLon),Math.min(90,n+padLat)),
      duration:1.6});}catch(err){}
  }
  viewer.scene.requestRender();
}
(function routeForm(){
  const form=document.getElementById('routeform');if(!form)return;
  form.addEventListener('submit',e=>{
    e.preventDefault();
    const from=document.getElementById('routefrom').value.trim().slice(0,80);
    const to=document.getElementById('routeto').value.trim().slice(0,80);
    if(!to){overlayNote('Type where the route should go to.');return;}
    if(!WORLD.bridge||!WORLD.bridge.routeTo){overlayNote('Routes need ORION running.');return;}
    document.getElementById('routego').disabled=true;
    overlayNote('Finding the route…');
    WORLD.bridge.routeTo(from,to);
  });
})();
window.orionRouteDone=function(message){
  const b=document.getElementById('routego');if(b)b.disabled=false;
  if(message)overlayNote(String(message).slice(0,300));
};
window.orionRoute=function(r){
  const b=document.getElementById('routego');if(b)b.disabled=false;
  if(!r||typeof r!=='object'||!Array.isArray(r.coords)||r.coords.length<2)return;
  OVERLAY.route={coords:r.coords.slice(0,2000),summary:String(r.summary||'')};
  routeDraw(true);
};
window.orionRouteClear=function(){
  OVERLAY.route=null;
  if(OVERLAY.routeLines&&!OVERLAY.routeLines.isDestroyed())OVERLAY.routeLines.removeAll();
  if(OVERLAY.routePts&&!OVERLAY.routePts.isDestroyed())OVERLAY.routePts.removeAll();
  overlayNote('');if(viewer&&!viewer.isDestroyed())viewer.scene.requestRender();
};
window.orionRouteStats=()=>({on:!!OVERLAY.route,points:OVERLAY.route?OVERLAY.route.coords.length:0,
  drawn:!!(OVERLAY.routeLines&&!OVERLAY.routeLines.isDestroyed()&&OVERLAY.routeLines.length)});
function overlayClick(e){
  if(!viewer||viewer.isDestroyed())return;
  const picked=viewer.scene.pick(e.position);
  if(picked&&picked.id!==undefined&&picked.id!==null)return;     // a report or aircraft: theirs
  const cart=viewer.camera.pickEllipsoid(e.position,viewer.scene.globe.ellipsoid);
  if(!cart)return;
  const c=Cesium.Cartographic.fromCartesian(cart);
  const lat=Cesium.Math.toDegrees(c.latitude),lon=Cesium.Math.toDegrees(c.longitude);
  if(!Number.isFinite(lat)||!Number.isFinite(lon))return;
  if(OVERLAY.measure){OVERLAY.pts.push([lat,lon]);if(OVERLAY.pts.length>50)OVERLAY.pts.shift();measureDraw();return;}
  window.orionPointPending(lat,lon);
  if(WORLD.bridge&&WORLD.bridge.inspectPoint)WORLD.bridge.inspectPoint(lat,lon);
}
window.orionPointPending=function(lat,lon){
  OVERLAY.point={lat:lat,lon:lon,title:''};
  document.getElementById('pointpanel').style.display='block';
  document.getElementById('pointtitle').textContent=lat.toFixed(4)+', '+lon.toFixed(4);
  document.getElementById('pointdetails').textContent='Looking up this place…';
};
window.orionPointInfo=function(info){
  if(!info||typeof info!=='object'||!OVERLAY.point)return;
  if(Math.abs(Number(info.lat)-OVERLAY.point.lat)>1e-6||Math.abs(Number(info.lon)-OVERLAY.point.lon)>1e-6)return;
  OVERLAY.point.title=String(info.title||'').slice(0,160);
  document.getElementById('pointtitle').textContent=OVERLAY.point.title
    ||(OVERLAY.point.lat.toFixed(4)+', '+OVERLAY.point.lon.toFixed(4));
  const lines=Array.isArray(info.lines)?info.lines.map(x=>String(x).slice(0,200)).slice(0,8):[];
  document.getElementById('pointdetails').textContent=lines.join('\n')||'No details for this spot.';
};
function overlayEnsure(){
  if(!viewer||viewer.isDestroyed()||OVERLAY.click)return;
  OVERLAY.click=new Cesium.ScreenSpaceEventHandler(viewer.canvas);
  OVERLAY.click.setInputAction(overlayClick,Cesium.ScreenSpaceEventType.LEFT_CLICK);
}
function overlayRebuild(){
  OVERLAY.night=null;OVERLAY.radar=null;OVERLAY.lines=null;OVERLAY.click=null;
  OVERLAY.routeLines=null;OVERLAY.routePts=null;
  overlayEnsure();
  if(OVERLAY.lighting)window.orionDayNight(true);
  if(OVERLAY.radarOn)window.orionRadar(true);
  if(OVERLAY.measure&&OVERLAY.pts.length)measureDraw();
  if(OVERLAY.route)routeDraw(false);
}
window.orionOverlayStats=()=>({lighting:OVERLAY.lighting,radar:OVERLAY.radarOn,measure:OVERLAY.measure,
  night:!!OVERLAY.night,click:!!OVERLAY.click});
overlayControls();
document.addEventListener('keydown',e=>{if(e.key!=='Escape')return;
  if(OVERLAY.measure)window.orionMeasure(false);else if(OVERLAY.route)window.orionRouteClear();});
if(typeof qt!=='undefined'&&typeof QWebChannel!=='undefined'){
  new QWebChannel(qt.webChannelTransport,channel=>{
    if(globeDisposed)return;WORLD.bridge=channel.objects.globeBridge;
    WORLD.enabled.forEach(key=>WORLD.bridge.setWorldLayer(key,true));worldControlState();
  });
}

// Render-loop gate: Python pauses the loop when the globe tab is hidden, so the
// GPU is not driven continuously behind another view (heat, power, driver
// resets — a frequent crash trigger).  Resumed when the tab is shown again.
// The aircraft pump and the detail card's clock stop with it.
window.orionSetActive=function(on){
  globeActive=!!on;WORLD.active=!!on;
  AV.active=!!on;
  if(on){ avPump(); if(AV.selected&&!AV.panelTimer) AV.panelTimer=setInterval(avPanel,1000); }
  else { clearTimeout(AV.timer); AV.timer=null; clearInterval(AV.panelTimer); AV.panelTimer=null;
    clearTimeout(restoreTimer);restoreTimer=null; }
  if(!viewer) return;
  try{ viewer.useDefaultRenderLoop=!!on; }catch(e){}
  if(on){viewer.resize();viewer.scene.requestRender();}
  // Off screen the ten-minute refresh is skipped; back on screen a stale
  // radar frame is replaced at once rather than asked for expired tiles.
  if(on&&typeof OVERLAY!=='undefined'&&OVERLAY.radarOn&&Date.now()-(OVERLAY.radarAt||0)>600000)
    window.orionRadar(true);
};
function destroyRenderer(){
  clearTimeout(restoreTimer);restoreTimer=null;clearTimeout(AV.timer);AV.timer=null;
  clearInterval(AV.panelTimer);AV.panelTimer=null;
  rendererCleanup.splice(0).forEach(remove=>{try{remove();}catch(_){}});
  if(AV.click&&!AV.click.isDestroyed())AV.click.destroy();AV.click=null;
  if(WORLD.handler&&!WORLD.handler.isDestroyed())WORLD.handler.destroy();WORLD.handler=null;
  if(typeof OVERLAY!=='undefined'){
    if(OVERLAY.click&&!OVERLAY.click.isDestroyed())OVERLAY.click.destroy();OVERLAY.click=null;
    clearInterval(OVERLAY.clock);OVERLAY.clock=null;clearInterval(OVERLAY.radarTimer);OVERLAY.radarTimer=null;
    OVERLAY.night=null;OVERLAY.radar=null;OVERLAY.lines=null;
  }
  AV.bb=null;AV.lb=null;AV.scene=null;AV.ring=null;
  WORLD.layers.forEach(layer=>{layer.points=null;});
  if(viewer&&!viewer.isDestroyed())viewer.destroy();viewer=null;marker=null;
  window.__globeReady=false;
}
window.orionDestroy=function(){globeDisposed=true;globeActive=false;WORLD.bridge=null;destroyRenderer();};
window.addEventListener('pagehide',window.orionDestroy,{once:true});

function focusCoordinate(lat,lon,range){
  if(!viewer||!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180) return false;
  spin=false;
  viewer.camera.cancelFlight();
  // Centre the actual WGS84 target in its local frame, including at the poles.
  // A fixed latitude offset cannot centre a tilted camera at every location.
  viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(
    Cesium.Cartesian3.fromDegrees(lon,lat),1),{
    offset:new Cesium.HeadingPitchRange(0,Cesium.Math.toRadians(-90),
      Math.max(180,Math.min(40e6,Number.isFinite(range)?range:16000))),duration:1.8});
  return true;
}
window.flyTo=function(lat,lon,label,detail,range){
  if(!viewer||!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180) return;
  spin=false;
  if(marker){viewer.entities.remove(marker);}
  marker=viewer.entities.add({
    position:Cesium.Cartesian3.fromDegrees(lon,lat),
    point:{pixelSize:11,color:Cesium.Color.fromCssColorString('#ff2a44'),
      outlineColor:Cesium.Color.fromCssColorString('#ffffff'),outlineWidth:2,
      heightReference:Cesium.HeightReference.CLAMP_TO_GROUND},
    label:{text:label||'',font:'13px Segoe UI',fillColor:Cesium.Color.WHITE,
      outlineColor:Cesium.Color.BLACK,outlineWidth:3,style:Cesium.LabelStyle.FILL_AND_OUTLINE,
      pixelOffset:new Cesium.Cartesian2(0,-18),
      heightReference:Cesium.HeightReference.CLAMP_TO_GROUND},
  });
  focusCoordinate(lat,lon,range);
  document.getElementById('place').textContent=label||'Location';
  document.getElementById('coords').textContent=lat.toFixed(6)+', '+lon.toFixed(6)
    +(detail?' · '+detail:'');
  showNews([], '');
};
// ── camera zoom (voice + on-screen rail) ────────────────────────────────────
// A zoom step flies straight up/down the current column to a clamped altitude,
// preserving heading and tilt.  Routing it through camera.flyTo means the
// existing moveStart/moveEnd low-res guard fires automatically, so a burst of
// zoom commands stays as crash-safe as a mouse-wheel zoom.  Altitude is clamped
// to the same street-level → orbit band as the wheel.
window.zoomStep=function(dir,factorIn,factorOut){
  if(!viewer) return;
  spin=false;
  const carto=viewer.camera.positionCartographic;
  const h=carto.height;
  const factor=(dir<0)?(factorIn||0.45):(factorOut||2.2);
  const target=Math.min(40e6, Math.max(180, h*factor));
  if(Math.abs(target-h) < 1) return;   // already at a rail limit
  viewer.camera.flyTo({
    destination:Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, target),
    orientation:{heading:viewer.camera.heading, pitch:viewer.camera.pitch, roll:0.0},
    duration:0.85,
  });
};
window.zoomIn =function(){ window.zoomStep(-1); };
window.zoomOut=function(){ window.zoomStep( 1); };
// ── programmatic max-zoom support (max_zoom_in_globe) ───────────────────────
// The street-level floor: the closest the camera may approach the surface.
// This is the SAME clamp the wheel/rail zoom uses, so "maximum zoom in" is a
// documented, observable altitude rather than a guess.
window.ORION_GLOBE_MIN_ALTITUDE=180;
window.orionZoomMinAltitude=function(){ return window.ORION_GLOBE_MIN_ALTITUDE; };
// Current camera altitude (metres) — the observable zoom state the controller
// reads between steps.  Returns null before the viewer exists.
window.orionZoomState=function(){ if(!viewer) return null;
  return viewer.camera.positionCartographic.height; };
// One INSTANT zoom-in step (no flyTo animation) so the returned altitude is the
// true post-step value the controller can compare — animated flyTo would report
// a stale height mid-flight.  Clamped to the same floor as the rail.
window.orionZoomInStep=function(factor){ if(!viewer) return null;
  spin=false;
  const carto=viewer.camera.positionCartographic;
  const h=carto.height;
  const f=(factor&&factor>0&&factor<1)?factor:0.45;
  const target=Math.max(window.ORION_GLOBE_MIN_ALTITUDE, h*f);
  viewer.camera.setView({destination:Cesium.Cartesian3.fromRadians(
      carto.longitude, carto.latitude, target),
    orientation:{heading:viewer.camera.heading, pitch:viewer.camera.pitch, roll:0.0}});
  return viewer.camera.positionCartographic.height; };
window.resetView=function(){
  if(!viewer) return;
  clearLocation();
  viewer.camera.cancelFlight();
  viewer.camera.flyTo({destination:Cesium.Cartesian3.fromDegrees(0.0,18.0,24e6),duration:3.0});
  spin=false;
  document.getElementById('place').textContent='O.R.I.O.N. Globe';
  document.getElementById('coords').textContent='Earth view';
  document.getElementById('news').textContent='';
  document.getElementById('footage').style.display='none';
  hideMap();
};
window.clearLocation=function(){
  if(viewer)viewer.camera.cancelFlight();
  if(marker&&viewer){viewer.entities.remove(marker);marker=null;}
  document.getElementById('place').textContent='O.R.I.O.N. Globe';
  document.getElementById('coords').textContent='';
  document.getElementById('news').textContent='';
  document.getElementById('footage').style.display='none';
  hideMap();
};
// Google Maps street-level card — correlates the globe with a real map.
window._mapLL=null;
window.showMap=function(query,lat,lon,label){
  if(!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180) return;
  // Use the resolved point in every map. Re-geocoding the name independently
  // can open another city, postcode or building in the companion map card.
  const q=lat+','+lon;
  document.getElementById('mapframe').src=
    'https://www.google.com/maps?q='+encodeURIComponent(q)+'&z=17&output=embed';
  document.getElementById('maptitle').textContent=label||q;
  document.getElementById('mapaddr').textContent=lat.toFixed(6)+', '+lon.toFixed(6);
  window._mapLL={lat:lat,lon:lon,q:q};
  document.getElementById('mapcard').style.display='block';
};
window.hideMap=function(){
  window._mapLL=null;
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
  const n=document.getElementById('news');n.textContent='';
  (Array.isArray(items)?items:[]).slice(0,8).forEach(it=>{if(!it||typeof it!=='object')return;const a=document.createElement('a');
    a.textContent=String(it.title||'Report').slice(0,180);
    const url=safeUrl(it.url);if(url){a.href=url;a.rel='noopener noreferrer';}
    n.appendChild(a);});
  const f=document.getElementById('footage');
  const url=safeUrl(footageUrl);
  if(url){f.style.display='inline-block';f.onclick=()=>{window.location.href=url;};}
  else f.style.display='none';
};
document.getElementById('reset').onclick=()=>{
  if(WORLD.bridge&&WORLD.bridge.resetView)WORLD.bridge.resetView();else window.resetView();
};
document.getElementById('zin').onclick=()=>window.zoomIn();
document.getElementById('zout').onclick=()=>window.zoomOut();
// Diagnostic snapshot (same convention as the face) — renders a frame and
// returns a small JPEG data URL.
window.orionSnap=(w=700,q=0.7)=>{ if(!viewer) return '';
  viewer.render();
  const c=document.createElement('canvas'); c.width=w;
  c.height=Math.round(w*viewer.canvas.height/viewer.canvas.width);
  c.getContext('2d').drawImage(viewer.canvas,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',q);};
window.orionPump=async(n,ms)=>{ if(!viewer) return 0;   // manual tile pump
  const generation=rendererGeneration;
  for(let i=0;i<Math.min(100,Math.max(0,n));i++){
    if(!viewer||globeDisposed||generation!==rendererGeneration)return false;
    viewer.render();await new Promise(r=>setTimeout(r,Math.max(20,Math.min(200,ms||60))));
  }
  return !!viewer&&viewer.scene.globe.tilesLoaded; };
initGlobe();
</script></body></html>"""


def describe_point(lat: float, lon: float, place: Any, weather: dict[str, Any] | None) -> dict[str, Any]:
    """The click card's content: a title and short plain lines. Pure."""
    from datetime import datetime, timedelta, timezone
    from ..utils import weather_code_label
    title = ""
    lines: list[str] = []
    if place is not None:
        admin = getattr(place, "admin", {}) or {}
        name = str(getattr(place, "name", "") or "")
        area = next((admin.get(k) for k in ("city", "town", "village", "county", "state", "region")
                     if admin.get(k) and admin.get(k) != name), "")
        country = str(getattr(place, "country", "") or admin.get("country") or "")
        title = ", ".join(bit for bit in (name, area, country) if bit)[:160]
    if not title:
        title = "Open water or remote ground" if weather else f"{lat:.4f}, {lon:.4f}"
    if weather:
        offset = weather.get("utc_offset_seconds")
        zone = str(weather.get("timezone_abbreviation") or weather.get("timezone") or "")
        if isinstance(offset, (int, float)):
            local = datetime.now(timezone.utc) + timedelta(seconds=float(offset))
            hours = float(offset) / 3600.0
            sign = "+" if hours >= 0 else "-"
            utc = f"UTC{sign}{abs(hours):g}" if hours else "UTC"
            named = zone if zone and not zone.startswith(("GMT+", "GMT-", "UTC")) else ""
            lines.append(f"Local time {local.strftime('%H:%M %a')} ({named + ', ' if named else ''}{utc})")
        current = weather.get("current") or {}
        if current.get("temperature_2m") is not None:
            label = weather_code_label(int(current.get("weather_code") or 0))
            wind = current.get("wind_speed_10m")
            lines.append(f"Now {float(current['temperature_2m']):.0f}°C, {label}"
                         + (f", wind {float(wind):.0f} km/h" if wind is not None else ""))
        daily = weather.get("daily") or {}

        def first(key: str) -> Any:
            values = daily.get(key) or []
            return values[0] if values else None

        high, low, rain = first("temperature_2m_max"), first("temperature_2m_min"), \
            first("precipitation_probability_max")
        if high is not None and low is not None:
            lines.append(f"Today {float(low):.0f}° to {float(high):.0f}°"
                         + (f", {float(rain):.0f}% chance of rain" if rain is not None else ""))
        rise, set_ = first("sunrise"), first("sunset")
        if rise and set_:
            lines.append(f"Sunrise {str(rise)[-5:]}, sunset {str(set_)[-5:]} (local)")
        elif current.get("is_day") is not None:
            lines.append("Daytime now" if current.get("is_day") else "Night-time now")
        if weather.get("elevation") is not None:
            lines.append(f"Elevation {float(weather['elevation']):.0f} m")
    lines.append(f"{lat:.5f}, {lon:.5f}")
    return {"lat": lat, "lon": lon, "title": title, "lines": lines[:8]}


class _GlobeBridge(QObject):
    """The page may select named, read-only layers; it cannot supply a URL."""

    def __init__(self, owner: Any) -> None:
        super().__init__(owner)
        self.owner = owner

    @pyqtSlot(str, bool)
    def setWorldLayer(self, layer: str, enabled: bool) -> None:
        self.owner.set_world_layer(layer, enabled)

    @pyqtSlot()
    def refreshWorld(self) -> None:
        self.owner._poll_world()

    @pyqtSlot(float, float)
    def inspectPoint(self, lat: float, lon: float) -> None:
        """What is at a clicked spot: two numbers in, nothing else."""
        self.owner.inspect_point(lat, lon)

    @pyqtSlot()
    def resetView(self) -> None:
        self.owner.reset_view()

    @pyqtSlot(str, str)
    def routeTo(self, origin: str, destination: str) -> None:
        """The route form: two short place names, looked up on the Python side."""
        self.owner.route_to(origin, destination)


class _GlobePage(QWebEnginePage):
    """Opens external link clicks in the system browser, keeping the globe up,
    and forwards JavaScript errors (incl. WebGL context loss) to the crash
    recorder so a silently-broken globe is diagnosable."""

    def __init__(self, parent: Any = None, on_js_error: Any = None) -> None:
        super().__init__(parent)
        self._on_js_error = on_js_error

    def javaScriptConsoleMessage(self, level: Any, message: str, line: int, source: str) -> None:  # type: ignore[override]
        try:
            text = str(message or "")
            name = getattr(level, "name", str(level))
            severe = ("Error" in name) or ("WEBGL" in text) or ("JS_ERROR" in text)
            if severe and self._on_js_error is not None:
                self._on_js_error(f"{text} @ {source}:{line}")
        except Exception:
            pass

    def acceptNavigationRequest(self, url: Any, nav_type: Any, is_main_frame: bool) -> bool:  # type: ignore[override]
        try:
            u = url.toString()
        except Exception:
            u = str(url)
        # The globe document itself is anchored to the Cesium CDN origin (so
        # its web workers are same-origin) — always let that load through.
        if u == GLOBE_BASE_URL or u.startswith("data:"):
            return super().acceptNavigationRequest(url, nav_type, is_main_frame)
        # Intercept real navigations away to web pages: open them externally.
        if is_main_frame and u.startswith(("http://", "https://")):
            open_url(u)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class GlobeView(QWidget):
    """The globe page: search bar + CesiumJS virtual Earth + news overlay."""

    def __init__(self, bus: OrionBus, parent: QWidget | None = None,
                 geo: Any | None = None, *, remote: bool = False) -> None:
        super().__init__(parent)
        self.bus = bus
        self.geo = geo          # GeoIntelligenceEngine with source provenance
        # Mark XXXV W1.2: the composition root asks for Cesium in its own
        # process (gui/page_host.py) so a stall in ORION's thread no longer
        # freezes the globe; built anywhere else, it stays in this process.
        self._remote_wanted = bool(remote)
        # The route form's handler, set by the composition root: (origin,
        # destination) -> awaitable ToolResult - the same geo action=route a
        # spoken request runs, which draws through dashboard_event.
        self.route_handler: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        heading = QLabel("Globe")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)

        row = QHBoxLayout()
        self.place_input = QLineEdit()
        self.place_input.setPlaceholderText(
            "Place, address, postcode or latitude, longitude")
        self.place_input.setMaxLength(256)
        self.place_input.returnPressed.connect(self._on_go)
        go = QPushButton("Fly There")
        go.clicked.connect(self._on_go)
        zoom_out = QPushButton("−")
        zoom_out.setObjectName("ghostButton")
        zoom_out.setToolTip("Zoom out")
        zoom_out.setAccessibleName("Zoom out")
        zoom_out.setFixedWidth(38)
        zoom_out.clicked.connect(lambda: self.zoom("out"))
        zoom_in = QPushButton("+")
        zoom_in.setObjectName("ghostButton")
        zoom_in.setToolTip("Zoom in")
        zoom_in.setAccessibleName("Zoom in")
        zoom_in.setFixedWidth(38)
        zoom_in.clicked.connect(lambda: self.zoom("in"))
        reset = QPushButton("⌂ Reset")
        reset.setToolTip("Zoom back out to the whole Earth")
        reset.clicked.connect(self.reset_view)
        row.addWidget(self.place_input, 1)
        row.addWidget(go)
        row.addWidget(zoom_out)
        row.addWidget(zoom_in)
        row.addWidget(reset)
        layout.addLayout(row)

        self.location_choices = QComboBox()
        self.location_choices.setAccessibleName("Matching locations")
        self.location_choices.setToolTip("Choose the full address before moving the camera")
        self.location_choices.activated.connect(self._on_choose_location)
        self.location_choices.hide()
        layout.addWidget(self.location_choices)
        self._candidates: list[GlobeLocation] = []
        self._candidate_generation = 0

        self.view: Any = None
        self._layout = layout
        self._built = False
        self._rebuilds = 0          # crash-recovery rebuilds this session (throttled)
        self._active = False        # is this tab currently visible?
        self._closed = False
        self._page_generation = 0
        self._navigation_seq = 0
        self._navigation_task: Any = None
        # Click-anywhere place cards: the newest click wins.
        self._inspect_seq = 0
        self._inspect_task: Any = None
        self._aviation_task: Any = None
        self._world_task: Any = None
        self._world_timer: Any = None
        self._world_layers: set[str] = set()
        self._world_snapshot: dict[str, Any] | None = None
        self._world_error = ""
        if WEBENGINE_OK:
            # LAZY: the QWebEngineView (Chromium — the heaviest RAM consumer) is
            # NOT created until this tab is first shown, so ORION starts lighter
            # and never pays for the globe if you never open it.
            self._placeholder = QLabel("")
            self._placeholder.setObjectName("mutedLabel")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._placeholder, 1)
        else:
            self._placeholder = None
            fallback = QLabel("Globe unavailable (PyQt6-WebEngine is not installed)")
            fallback.setObjectName("mutedLabel")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setWordWrap(True)
            layout.addWidget(fallback, 1)

        # Live aviation layer. Its state lives in aviation's shared service, so
        # the 'show' action works before this lazily-built page exists; the
        # feed is polled only while the page is visible AND the layer is on.
        self._aviation_timer: Any = None
        self._aviation_seq = 0          # the LayerState.focus_seq last drawn
        self._aviation_reset = False    # next batch replaces the old region
        self._aviation_busy = False
        self._aviation_repoll = False
        self._aviation_drawn = False
        self._aviation_error = ""

        self.bus.globe_request.connect(self.go_to_place)
        self.bus.globe_zoom.connect(self.zoom)
        self.bus.globe_max_zoom.connect(self.on_max_zoom_request)
        self.bus.dashboard_event.connect(self._on_dashboard_event)
        # max_zoom_in_globe state: the last structured outcome and a cooperative
        # cancel flag the bounded controller polls between attempts.
        self.last_zoom_result: Any = None
        self._max_zoom_cancel = False
        self._max_zoom_running = False

    def _ensure_built(self) -> None:
        """Create the WebEngine view on first use (lazy — saves RAM at startup)."""
        if self._built or self._closed or not WEBENGINE_OK:
            return
        self._built = True
        if self._placeholder is not None:
            self._placeholder.hide()
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None
        self._page_generation += 1
        self._pending_js = []
        self._world_bridge = _GlobeBridge(self)
        self.view = self._remote_view() if self._remote_wanted else None
        if self.view is None:
            self.view = QWebEngineView()
            self.view.setPage(_GlobePage(self.view, on_js_error=self._on_js_error))
            self._web_channel = QWebChannel(self.view.page())
            self._web_channel.registerObject("globeBridge", self._world_bridge)
            self.view.page().setWebChannel(self._web_channel)
        # Recover from a Chromium render-process crash (GPU driver reset, WebGL
        # loss, OOM) instead of leaving a dead grey rectangle.
        self._page_loaded = False
        try:
            self.view.renderProcessTerminated.connect(self._on_render_crashed)
            self.view.loadFinished.connect(self._on_load_finished)
        except Exception:
            pass
        self.view.setHtml(GLOBE_HTML, QUrl(GLOBE_BASE_URL))
        self._layout.addWidget(self.view, 1)
        # A new page (first build or a crash rebuild) has no aircraft layer
        # yet: the next sync re-applies the current focus to it.
        self._aviation_seq = 0
        self.bus.log.emit("GLOBE: renderer initialised on first view (lazy-loaded).")

    def _remote_view(self) -> Any:
        """The globe page in its own process, or None to draw it here."""
        from . import page_host
        if not page_host.enabled("globe"):
            return None
        try:
            from ..constants import CONFIG_DIR
            view = page_host.RemoteWebView(
                page="globe", bridges={"globeBridge": self._world_bridge},
                allow=(GLOBE_BASE_URL,), on_js_error=self._on_js_error,
                on_open_url=open_url, log_tag="GLOBE",
                cache=str(CONFIG_DIR / "web_cache" / "globe"))
        except Exception as exc:
            self.bus.log.emit(f"GLOBE: own process unavailable ({exc}) - drawing it here.")
            return None
        # Given up for this run (it kept stopping, or never showed a window):
        # rebuild the page in this process, exactly as after a crash.
        view.on_failed = lambda: QTimer.singleShot(0, self._remote_failed)
        return view

    def _remote_failed(self) -> None:
        if self._closed or not self._remote_wanted:
            return
        self._remote_wanted = False
        self.bus.log.emit("GLOBE: its own process could not run - drawing it in ORION's.")
        self._drop_view()
        self._built = False
        self._ensure_built()
        if self._active:
            self._set_active(True)

    def _drop_view(self) -> None:
        """Take the page off screen; a page host is asked to quit."""
        view, self.view = self.view, None
        if view is None:
            return
        try:
            self._layout.removeWidget(view)
            if isinstance(view, QWidget) and hasattr(view, "_containers"):
                view.shutdown()
            view.deleteLater()
        except Exception:
            pass

    # ── crash resilience ────────────────────────────────────────────────────────

    #: A tile or feed that failed to download: the network's, not the page's.
    #: Logged at most once a minute and never filed as a crash - 27 Oct, 71
    #: expired radar tiles became crash reports and started the Reliability
    #: Agent on the owner's quota.
    _NETWORK_ERROR = re.compile(r"XMLHttpRequest|CORS|Failed to fetch|net::ERR_|NetworkError|"
                                r"Failed to load resource|status code [45]\d\d", re.I)

    def _on_js_error(self, message: str) -> None:
        """A severe JS / WebGL error inside the globe page."""
        if self._NETWORK_ERROR.search(message or ""):
            now = time.monotonic()
            if now - getattr(self, "_network_error_at", -1e9) >= 60.0:
                self._network_error_at = now
                self.bus.log.emit(f"GLOBE: a map tile or feed could not be downloaded - "
                                  f"{message[:120]}")
            return
        self.bus.log.emit(f"GLOBE: page error - {message[:160]}")
        try:
            from .. import crash_reporter
            crash_reporter.report("globe-js", message[:200])
        except Exception:
            pass

    def _on_render_crashed(self, status: Any, exit_code: int) -> None:
        detail = f"QWebEngine render process terminated: status={status}, exit_code={exit_code}"
        self.bus.log.emit(f"GLOBE: {detail}")
        try:
            from .. import crash_reporter
            crash_reporter.report("globe-render", detail)
        except Exception:
            pass
        # Rebuild once or twice, then stop so a hard crash loop can't spin the
        # GPU forever — leave a clear message instead.
        if self._rebuilds >= 3:
            self.bus.log.emit("GLOBE: repeated render crashes — leaving the globe closed this session.")
            return
        self._rebuilds += 1
        self._cancel_feed_tasks()
        self._page_generation += 1
        self._page_loaded = False
        self._pending_js = []
        self._clear_location_choices()
        self._built = False
        self._drop_view()
        # Small defer so the dead process is fully reaped before we respawn.
        QTimer.singleShot(800, self._rebuild_after_crash)

    def _rebuild_after_crash(self) -> None:
        if self._closed:
            return
        self._ensure_built()
        if self._active:
            self._set_active(True)
        self.bus.log.emit(f"GLOBE: renderer rebuilt after crash (attempt {self._rebuilds}).")

    def _set_active(self, on: bool) -> None:
        """Drive the Cesium render loop only while the tab is visible."""
        self._active = on
        if not on:
            self._cancel_feed_tasks()
        if self.view is not None:
            # Guard the call: before the page finishes loading orionSetActive is
            # undefined, and the default render loop is already on, so a no-op is
            # correct (and avoids a spurious ReferenceError on first show).
            self._js(f"window.orionSetActive&&orionSetActive({'true' if on else 'false'})")
        self._sync_aviation()
        self._sync_world()

    # ── official world layers ──────────────────────────────────────────────

    WORLD_LAYERS = frozenset({"satellites", "earthquakes", "tsunami", "weather", "events"})
    WORLD_POLL_MS = 15_000

    def set_world_layer(self, layer: str, enabled: bool) -> None:
        """A whitelisted layer selection; no page-supplied network address."""
        if layer not in self.WORLD_LAYERS or self._closed:
            return
        if enabled:
            self._world_layers.add(layer)
        else:
            self._world_layers.discard(layer)
        self._js("window.orionWorldLayer&&orionWorldLayer("
                 + json.dumps(layer) + "," + str(bool(enabled)).lower() + ")")
        # A selection change cancels the previous snapshot before scheduling
        # another, so a late response cannot re-enable a removed layer.
        task = self._world_task
        if task is not None and not task.done():
            task.cancel()
        self._world_task = None
        self._sync_world()

    def _sync_world(self) -> None:
        timer = self._world_timer
        if not (self._world_layers and self._active and self.view is not None):
            if timer is not None:
                timer.stop()
            return
        if timer is None:
            from PyQt6.QtCore import QTimer
            timer = self._world_timer = QTimer(self)
            timer.setInterval(self.WORLD_POLL_MS)
            timer.timeout.connect(self._poll_world)
        if not timer.isActive():
            timer.start()
        self._poll_world()

    def _poll_world(self) -> None:
        if (self._closed or not self._active or not self._world_layers
                or self.view is None):
            return
        if self._world_task is not None and not self._world_task.done():
            return
        self._js("window.orionWorldBusy&&orionWorldBusy(true)")
        self._world_task = background.spawn(self._refresh_world(), name="globe-world")
        if self._world_task is None:
            self._js("window.orionWorldBusy&&orionWorldBusy(false)")

    async def _refresh_world(self) -> None:
        from ..world_feeds import get_world_feeds
        generation = self._page_generation
        layers = tuple(sorted(self._world_layers))
        try:
            snapshot = await get_world_feeds().snapshot(layers=layers, refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = first_line(exc, 160)
            if message != self._world_error:
                self.bus.log.emit("GLOBE: world reports - " + message)
                self._world_error = message
            if generation == self._page_generation and self._active:
                self._js("window.orionWorldBusy&&orionWorldBusy(false)")
            return
        if (self._closed or not self._active or generation != self._page_generation
                or layers != tuple(sorted(self._world_layers))):
            return
        self._world_error = ""
        self._world_snapshot = snapshot
        self._js("window.orionWorldSnapshot&&orionWorldSnapshot("
                 + json.dumps(snapshot, separators=(",", ":"), allow_nan=False) + ")")

    def _cancel_feed_tasks(self) -> None:
        for name in ("_world_task", "_aviation_task"):
            task = getattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()
            setattr(self, name, None)
        self._aviation_busy = False
        self._aviation_repoll = False
        for name in ("_world_timer", "_aviation_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()

    # ── live aviation layer ─────────────────────────────────────────────────

    AVIATION_POLL_MS = 10_000       # = aviation.CACHE_TTL_S; OpenSky data lasts 30 s

    def _on_dashboard_event(self, channel: str, payload: Any) -> None:
        if channel == "aviation":       # the aviation tool's show / hide
            self._sync_aviation()
        elif channel == "world":
            self._sync_world()
        elif channel == "globe_overlay":
            self.apply_overlay(payload)
        elif channel == "globe_route":
            self.show_route(payload)

    def route_to(self, origin: str, destination: str) -> None:
        """The page's route form. Two names in (bounded, control characters
        removed); the lookup and the drawing are the geo tool's."""
        def clean(text: Any) -> str:
            return re.sub(r"[\x00-\x1f\x7f]", " ", str(text or ""))[:80].strip()

        if self._closed:
            return
        start, end = clean(origin), clean(destination)
        handler = self.route_handler

        def done(message: str) -> None:
            if not self._closed:
                self._js("window.orionRouteDone&&orionRouteDone(" + json.dumps(message) + ")")

        if handler is None:
            done("Routes aren't available here.")
            return
        if not end:
            done("Type where the route should go to.")
            return

        async def run() -> None:
            try:
                result = await handler(start, end)
                ok, text = bool(getattr(result, "ok", False)), str(getattr(result, "text", ""))
            except Exception as exc:
                ok, text = False, f"The route couldn't be found: {exc}"
            done("" if ok else text)

        try:
            background.spawn(run(), name="globe-route")
        except RuntimeError:
            done("Routes aren't available right now.")

    def show_route(self, payload: Any) -> None:
        """Draw a road route (geo action=route): {"coords": [[lon, lat], ...],
        "summary": str, ...}. Only numbers and plain text reach the page."""
        if not isinstance(payload, dict) or self._closed:
            return
        coords = []
        for point in payload.get("coords") or []:
            try:
                lon, lat = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                coords.append([round(lon, 5), round(lat, 5)])
        if len(coords) < 2:
            return
        clean = {"coords": coords[:2000], "summary": str(payload.get("summary") or "")[:300]}
        self._ensure_built()
        self._js("window.orionRoute&&orionRoute(" + json.dumps(clean) + ")")

    #: Overlays the page draws itself (no feed behind them).
    OVERLAYS = {"daynight": "orionDayNight", "radar": "orionRadar", "measure": "orionMeasure"}

    def apply_overlay(self, payload: Any) -> None:
        """Voice switching an overlay or a world layer on or off:
        {"name": "radar" | "daynight" | "measure" | <world layer>, "on": bool}."""
        if not isinstance(payload, dict) or self._closed:
            return
        name = str(payload.get("name") or "").strip().lower()
        on = bool(payload.get("on", True))
        if name == "route" and not on:
            self._js("window.orionRouteClear&&orionRouteClear()")
        elif name in self.OVERLAYS:
            self._ensure_built()
            self._js(f"window.{self.OVERLAYS[name]}&&{self.OVERLAYS[name]}({str(on).lower()})")
        elif name in self.WORLD_LAYERS:
            self._ensure_built()
            self.set_world_layer(name, on)

    def _sync_aviation(self) -> None:
        """Poll the aircraft feed only while this page is on screen AND the
        layer is on, and stop the moment either stops being true."""
        try:
            from ..aviation import get_service
            layer = get_service().layer
        except Exception:
            return
        timer = self._aviation_timer
        if not (layer.on and self._active and self.view is not None):
            if timer is not None:
                timer.stop()
            if not layer.on and self._aviation_drawn:
                self._aviation_drawn = False
                self._js("window.orionAircraftLayer&&orionAircraftLayer(false)")
            return
        poll_now = False
        if layer.focus_seq != self._aviation_seq:
            # A new 'show': clear the old region, fly to the new one — once.
            self._aviation_seq = layer.focus_seq
            self._aviation_reset = self._aviation_drawn = True
            self._js(f"window.orionAircraftFocus&&orionAircraftFocus({layer.lat},"
                     f"{layer.lon},{layer.radius_km},{json.dumps(layer.label)})")
            poll_now = True
        if timer is None:
            from PyQt6.QtCore import QTimer
            timer = self._aviation_timer = QTimer(self)
            timer.setInterval(self.AVIATION_POLL_MS)
            timer.timeout.connect(self._poll_aviation)
        if not timer.isActive():
            timer.start()
            poll_now = True
        if poll_now:
            self._poll_aviation()

    def _poll_aviation(self) -> None:
        if self._closed or not self._active or self.view is None:
            return
        from ..aviation import get_service
        if not get_service().layer.on:
            return
        if self._aviation_busy:
            self._aviation_repoll = True     # e.g. a new focus mid-request
            return
        self._aviation_busy = True
        self._aviation_task = background.spawn(self._aviation_refresh(), name="globe-aviation")
        if self._aviation_task is None:
            self._aviation_busy = False      # no loop (shutting down)

    async def _aviation_refresh(self) -> None:
        from ..aviation import get_service, globe_batch
        service = get_service()
        layer = service.layer
        seq = layer.focus_seq
        generation = self._page_generation
        report = None
        owner = asyncio.current_task()
        try:
            report = await service.nearby(layer.lat, layer.lon, layer.radius_km)
        except Exception as exc:             # a feed fault must never reach the loop
            self._aviation_note(f"refresh failed - {first_line(exc, 120)}")
        finally:
            if self._aviation_task is owner:
                self._aviation_busy = False
        current = service.layer
        # Hidden, switched off or re-focused while the request was in flight:
        # drawing this now would undo that.
        if (report is not None and current.on and self._active and current.focus_seq == seq
                and generation == self._page_generation and not self._closed):
            if report.error:
                self._aviation_note(report.error)
            else:
                self._aviation_error = ""
            if report.source:
                batch = globe_batch(report, reset=self._aviation_reset)
                self._aviation_reset = False
                self._js("window.orionAircraft&&orionAircraft("
                         + json.dumps(batch, separators=(",", ":")) + ")")
        if (self._aviation_repoll and self._active and not self._closed
                and self._aviation_task is owner):
            self._aviation_repoll = False
            self._poll_aviation()

    def _aviation_note(self, message: str) -> None:
        """Log a feed problem once, not every ten seconds while it lasts."""
        if message != self._aviation_error:
            self._aviation_error = message
            self.bus.log.emit(f"GLOBE: aircraft layer - {message[:180]}")

    #: Off screen this long, the renderer is given back (Mark XXXX.4).
    RELEASE_AFTER_S = 120.0

    def showEvent(self, event: Any) -> None:
        timer = getattr(self, "_release_timer", None)
        if timer is not None:
            timer.stop()
        self._ensure_built()
        self._set_active(True)
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:
        # Stop rendering the globe when it's not on screen — no GPU churn behind
        # another tab (a common overheating / driver-reset crash trigger).
        self._set_active(False)
        if not self._closed:
            timer = getattr(self, "_release_timer", None)
            if timer is None:
                timer = self._release_timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self.release)
            timer.start(int(self.RELEASE_AFTER_S * 1000))
        super().hideEvent(event)

    def release(self) -> bool:
        """Give the renderer back while the globe is off screen; opening it
        builds it again, exactly as after a crash (Mark XXXX.4).

        Measured 27 Oct: the globe's page host held 323 MB, plus its renderer,
        for the rest of the session once opened - memory went from 87% to 92%
        within a minute - and the deck's release-when-hidden step found no
        release() here to call."""
        if self._closed or self.isVisible() or not self._built:
            return False
        self._cancel_feed_tasks()
        self._page_generation += 1
        self._page_loaded = False
        self._pending_js = []
        self._clear_location_choices()
        self._built = False
        self._drop_view()
        self.bus.log.emit("GLOBE: renderer released while off screen - rebuilt when opened.")
        return True

    def shutdown(self) -> None:
        """Release GUI-owned tasks, feeds and renderer resources on exit."""
        if self._closed:
            return
        self._closed = True
        self._active = False
        self._page_generation += 1
        self._cancel_feed_tasks()
        self.cancel_max_zoom()
        if self._navigation_task is not None and not self._navigation_task.done():
            self._navigation_task.cancel()
        self._navigation_task = None
        if self._inspect_task is not None and not self._inspect_task.done():
            self._inspect_task.cancel()
        self._inspect_task = None
        if self.view is not None and getattr(self, "_page_loaded", False):
            try:
                self.view.page().runJavaScript("window.orionDestroy&&orionDestroy()")
            except RuntimeError:
                pass
        if self.view is not None and hasattr(self.view, "_containers"):
            self.view.shutdown()          # the page host quits after orionDestroy
        self._pending_js = []
        for signal, slot in ((self.bus.globe_request, self.go_to_place),
                             (self.bus.globe_zoom, self.zoom),
                             (self.bus.globe_max_zoom, self.on_max_zoom_request),
                             (self.bus.dashboard_event, self._on_dashboard_event)):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def closeEvent(self, event: Any) -> None:
        self.shutdown()
        super().closeEvent(event)

    # ── control ───────────────────────────────────────────────────────────────

    def _on_go(self) -> None:
        place = self.place_input.text().strip()
        if place:
            self.go_to_place(place)

    def reset_view(self) -> None:
        self._navigation_seq += 1
        if self._navigation_task is not None and not self._navigation_task.done():
            self._navigation_task.cancel()
        self._clear_location_choices()
        self._discard_pending_navigation()
        self._js("resetView()")
        self.bus.log.emit("GLOBE: view reset to the whole Earth.")

    def zoom(self, direction: str) -> None:
        """Zoom the globe camera without changing location.  ``direction`` is
        'in', 'out' or 'reset' (spoken 'zoom in on the globe' routes here)."""
        d = str(direction or "").strip().lower()
        self._ensure_built()   # build the renderer if ORION was asked by voice first
        if d in {"reset", "orbit", "out to space", "whole earth"}:
            self.reset_view()
            return
        if d in {"in", "closer", "nearer", "+"}:
            self._js("window.zoomIn&&zoomIn()")
            self.bus.log.emit("GLOBE: zooming in.")
        elif d in {"out", "back", "away", "-", "−"}:
            self._js("window.zoomOut&&zoomOut()")
            self.bus.log.emit("GLOBE: zooming out.")

    def go_to_place(self, place: str) -> None:
        place = str(place or "").strip()
        if not place or self._closed:
            return
        self._ensure_built()   # build the renderer if ORION was asked by voice first
        self.bus.log.emit(f"GLOBE: travelling to {place}.")
        try:
            if self._navigation_task is not None and not self._navigation_task.done():
                self._navigation_task.cancel()
            self._clear_location_choices()
            self._discard_pending_navigation()
            self._js("clearLocation()")
            self._navigation_seq += 1
            self._navigation_task = background.spawn(self._go(place), name="globe-navigation")
        except RuntimeError:
            pass

    async def _go(self, place: str) -> None:
        sequence = self._navigation_seq
        generation = self._page_generation
        try:
            coordinates = parse_coordinates(place)
        except ValueError as exc:
            self._location_error(str(exc))
            return
        if coordinates is not None:
            lat, lon = coordinates
            target = GlobeLocation(lat, lon, f"{lat:.6f}, {lon:.6f}", "User", "coordinate")
            await self._show_location(target, sequence, generation)
            return
        targets = await self._resolve_locations(place)
        if self._closed or sequence != self._navigation_seq or generation != self._page_generation:
            return
        if not targets:
            self._location_error("Could not locate that place — offline or unknown.")
            return
        if len(targets) > 1:
            self._candidates = targets
            self._candidate_generation = generation
            self.location_choices.clear()
            self.location_choices.addItem("Choose a matching location…")
            for target in targets:
                self.location_choices.addItem(
                    f"{target.label} · {target.lat:.6f}, {target.lon:.6f}")
            self.location_choices.setCurrentIndex(0)
            self.location_choices.show()
            self._js("showNews([{title:'Choose a matching location above.'}], '')")
            return
        await self._show_location(targets[0], sequence, generation)

    async def _show_location(self, target: GlobeLocation, sequence: int, generation: int) -> None:
        if self._closed or sequence != self._navigation_seq or generation != self._page_generation:
            return
        lat, lon, label = target.lat, target.lon, target.label
        self._js(f"flyTo({lat!r},{lon!r},{json.dumps(label)},"
                 f"{json.dumps(target.detail)},{target.camera_range})")
        self._js(f"showMap('',{lat!r},{lon!r},{json.dumps(label)})")
        # Publish the actual selection promptly; news is unrelated and may
        # take several seconds. Supplied coordinates need no place/news lookup.
        self.bus.dashboard_event.emit("globe", {"place": label, "lat": lat, "lon": lon,
                                               "source": target.source, "kind": target.kind})
        if target.kind == "coordinate":
            self._js("showNews([], '')")
            return
        items = await self._news(label)
        if self._closed or sequence != self._navigation_seq or generation != self._page_generation:
            return
        footage = f"https://www.youtube.com/results?search_query={quote_plus(label + ' news today')}"
        self._js(f"showNews({json.dumps(items)}, {json.dumps(footage)})")

    # ── click anywhere: what is here? ─────────────────────────────────────────

    def inspect_point(self, lat: float, lon: float) -> None:
        """Place, local time, weather and daylight for a clicked spot. Two
        numbers from the page; the requests are fixed and made from here."""
        if self._closed or not self._valid_coordinates(lat, lon):
            return
        self._inspect_seq += 1
        if self._inspect_task is not None and not self._inspect_task.done():
            self._inspect_task.cancel()
        try:
            self._inspect_task = background.spawn(
                self._inspect(float(lat), float(lon), self._inspect_seq, self._page_generation),
                name="globe-inspect")
        except RuntimeError:
            pass

    async def _inspect(self, lat: float, lon: float, sequence: int, generation: int) -> None:
        async def nothing() -> None:
            return None

        place_job = self.geo.reverse(lat, lon) if self.geo is not None else nothing()
        place, weather = await asyncio.gather(place_job, self._point_weather(lat, lon),
                                              return_exceptions=True)
        if self._closed or sequence != self._inspect_seq or generation != self._page_generation:
            return
        payload = describe_point(lat, lon, place if not isinstance(place, BaseException) else None,
                                 weather if isinstance(weather, dict) else None)
        self._js(f"window.orionPointInfo&&orionPointInfo({json.dumps(payload)})")
        self.bus.dashboard_event.emit("globe", {"place": payload["title"], "lat": lat, "lon": lon,
                                               "source": "click", "kind": "point"})

    @staticmethod
    async def _point_weather(lat: float, lon: float) -> dict[str, Any] | None:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat:.4f}&longitude={lon:.4f}"
               "&current=temperature_2m,weather_code,wind_speed_10m,is_day"
               "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
               "sunrise,sunset&forecast_days=1&timezone=auto")
        try:
            async with ClientSession(timeout=ClientTimeout(total=8.0, connect=4.0)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    data = await response.json(content_type=None)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _location_error(self, message: str) -> None:
        self.bus.log.emit(f"GLOBE: {message}")
        self._js(f"showNews({json.dumps([{'title': message}])}, '')")

    def _clear_location_choices(self) -> None:
        self._candidates = []
        self.location_choices.clear()
        self.location_choices.hide()

    def _discard_pending_navigation(self) -> None:
        # A page still loading can hold already-completed old requests. Task
        # cancellation alone does not remove their queued JavaScript.
        self._pending_js = [code for code in getattr(self, "_pending_js", [])
                            if not code.startswith(("flyTo(", "showMap(", "showNews(",
                                                    "resetView(", "clearLocation("))]

    def _on_choose_location(self, index: int) -> None:
        if (self._closed or self._candidate_generation != self._page_generation
                or not 1 <= index <= len(self._candidates)):
            return
        if self._navigation_task is not None and not self._navigation_task.done():
            self._navigation_task.cancel()
        self._navigation_seq += 1
        self._discard_pending_navigation()
        self._navigation_task = background.spawn(
            self._show_location(self._candidates[index - 1], self._navigation_seq,
                                self._page_generation), name="globe-location-selection")

    # ── data ──────────────────────────────────────────────────────────────────

    async def _resolve_locations(self, place: str) -> list[GlobeLocation]:
        # Prefer the GeoIntelligenceEngine (OSM/Nominatim) — it resolves towns,
        # villages and districts worldwide, not merely major cities.
        if self.geo is not None:
            try:
                results = await self.geo.geocode(place, limit=6)
                targets = []
                seen = set()
                for p in results[:6]:
                    target = GlobeLocation.from_place(p, place)
                    if target is not None and (target.lat, target.lon, target.label) not in seen:
                        seen.add((target.lat, target.lon, target.label))
                        targets.append(target)
                if targets:
                    return targets
            except Exception:
                pass  # fall through to the lightweight open-meteo geocoder
        # Open-Meteo is a city/town gazetteer — it cannot resolve a postcode and
        # returns a wrong or empty hit (the "flies to the sea" bug).  For a
        # postcode-shaped query, don't guess: report not-found so the user sees
        # the miss rather than a random point.
        if looks_like_uk_postcode(place) or re.search(
                r"\d|\b(?:street|road|lane|avenue|close|drive|flat|apartment|house)\b", place, re.I):
            # A settlement gazetteer cannot establish a requested street or
            # premises position. Report the failed precise lookup instead.
            return []
        try:
            url = ("https://geocoding-api.open-meteo.com/v1/search?count=6&name="
                   + quote_plus(place))
            timeout = ClientTimeout(total=8.0, connect=3.0)
            async with ClientSession(timeout=timeout) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
            results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(results, list):
                return []
            targets = []
            for top in results[:6]:
                if not isinstance(top, dict) or not self._valid_coordinates(
                        top.get("latitude"), top.get("longitude")):
                    continue
                label = ", ".join(dict.fromkeys(str(x) for x in (
                    top.get("name"), top.get("admin2"), top.get("admin1"), top.get("country")) if x))
                targets.append(GlobeLocation(float(top["latitude"]), float(top["longitude"]),
                                             label or place, "Open-Meteo", "settlement"))
            return targets
        except Exception:
            return []

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

    @staticmethod
    def _valid_coordinates(lat: Any, lon: Any) -> bool:
        return valid_coordinates(lat, lon)

    def _on_load_finished(self, ok: bool) -> None:
        self._page_loaded = bool(ok)
        if ok:
            # Whatever was asked before the page existed lands now: the latest
            # visibility, then any fly-to/news requests in the order given.
            self._set_active(self._active)
            for layer in self._world_layers:
                self._js("window.orionWorldLayer&&orionWorldLayer("
                         + json.dumps(layer) + ",true)")
            if self._world_snapshot is not None:
                self._js("window.orionWorldSnapshot&&orionWorldSnapshot("
                         + json.dumps(self._world_snapshot, separators=(",", ":"), allow_nan=False) + ")")
            pending, self._pending_js = list(getattr(self, "_pending_js", [])), []
            for code in pending:
                self._js(code)

    def _js(self, code: str) -> None:
        """Run JS only once the page has LOADED. runJavaScript on a view that
        was only just built kills the process natively — no traceback — which
        follows the same deferred renderer pattern; showEvent did exactly that here,
        calling straight after setHtml."""
        if self.view is None or self._closed:
            return
        if not getattr(self, "_page_loaded", False):
            # Held, not dropped: "fly to Paris" on the globe's first opening
            # arrives while Cesium is still loading. Visibility is re-applied
            # from _active on load, so it need not queue.
            if not code.startswith("window.orionSetActive"):
                queue = self.__dict__.setdefault("_pending_js", [])
                queue.append(code)
                del queue[:-12]
            return
        try:
            self.view.page().runJavaScript(code)
        except RuntimeError:
            pass      # page torn down mid-call (shutdown / crash rebuild)

    async def _eval_js(self, code: str, timeout: float = 1.5) -> Any:
        """Evaluate JavaScript and return its value.  QWebEngine delivers the
        result through a callback on the GUI thread; we bridge it to an awaitable
        with a hard timeout so a wedged renderer can never hang the caller."""
        if self.view is None or not getattr(self, "_page_loaded", False):
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def _settle(value: Any) -> None:
            # Checked when it RUNS, not only when scheduled: a callback that
            # arrives after the timeout finds the future already cancelled, and
            # set_result on it raises InvalidStateError inside the event loop.
            if not future.done():
                future.set_result(value)

        def _callback(value: Any) -> None:
            if not future.done():
                loop.call_soon_threadsafe(_settle, value)

        try:
            self.view.page().runJavaScript(code, _callback)
        except Exception:
            return None
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        # CancelledError is deliberately NOT caught. It used to be, which
        # turned "stop" into "the JS returned nothing" — so cancelling the
        # bounded zoom loop in globe_zoom left it running.

    # ── max_zoom_in_globe (bounded, cancellable) ─────────────────────────────

    def on_max_zoom_request(self, payload: Any = None) -> None:
        """Bus slot: schedule a bounded max-zoom run on the GUI event loop."""
        self._ensure_built()
        self._max_zoom_cancel = False
        args = payload if isinstance(payload, dict) else {}
        try:
            background.spawn(self.run_max_zoom_in(args))
        except RuntimeError:
            # No running loop (headless / test) — nothing to drive.
            self.bus.log.emit("GLOBE: max-zoom requested with no event loop running.")

    def cancel_max_zoom(self) -> None:
        self._max_zoom_cancel = True

    async def run_max_zoom_in(self, args: dict[str, Any] | None = None) -> Any:
        """Drive the desktop Cesium backend to maximum zoom and return the
        structured :class:`GlobeZoomResult`.  Idempotent when already maxed."""
        from ..globe_zoom import GlobeZoomController, GlobeZoomConfig, CesiumGlobeBackend
        if self._max_zoom_running:
            self.bus.log.emit("GLOBE: max-zoom already in progress; ignoring duplicate request.")
            return self.last_zoom_result
        self._ensure_built()
        args = args or {}
        base = GlobeZoomConfig()
        config = GlobeZoomConfig(
            max_attempts=int(args.get("max_attempts", base.max_attempts)),
            max_seconds=float(args.get("max_seconds", base.max_seconds)),
            delay_between_s=float(args.get("delay_between_s", base.delay_between_s)),
            unchanged_threshold=int(args.get("unchanged_threshold", base.unchanged_threshold)),
            tolerance=float(args.get("tolerance", base.tolerance)),
            blind_attempts=int(args.get("blind_attempts", base.blind_attempts)),
        )
        backend = CesiumGlobeBackend(self)
        controller = GlobeZoomController(backend, config=config)
        self._max_zoom_running = True
        self.bus.log.emit("GLOBE: max_zoom_in_globe engaged — driving to closest zoom.")
        try:
            result = await controller.run(cancel=lambda: self._max_zoom_cancel)
        finally:
            self._max_zoom_running = False
        self.last_zoom_result = result
        self.bus.log.emit("GLOBE: " + result.summary())
        try:
            self.bus.dashboard_event.emit("globe_zoom", result.as_dict())
        except Exception:
            pass
        return result
