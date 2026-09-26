"""
GlobeView — the Global Intelligence Globe (Mark X.6: CesiumJS virtual globe).

Mark X.5 rendered a textured Three.js sphere — pretty, but a painted ball.
Mark X.6 swaps the renderer for CesiumJS, the open-source virtual-globe engine
in the same class as Google Earth: real Esri World Imagery satellite tiles,
place-name labels that appear as you zoom, smooth flight from orbit down to
street level (~180 m), tilt with right-drag, ground atmosphere and stars.
No API key is required.

Everything around the renderer is unchanged:

  • the Python bridge API (``flyTo`` / ``resetView`` / ``showNews`` /
    ``showMap``) is identical, so voice commands and the search bar work
    exactly as before;
  • EXTERNAL LINKS still open in the system browser via the custom
    QWebEnginePage navigation interceptor;
  • the news panel and the Google Maps street-level card overlay the globe.

The engine and imagery stream from CDNs, so the live globe needs internet —
offline it shows a clear message instead of a broken view.

Mark X.6.1: the imagery providers are now constructed synchronously from tile
URL templates instead of Esri's async ``fromUrl`` metadata handshake — the
X.6 build blanked to a starfield whenever that single request failed, because
Cesium never renders the globe surface while its base layer is un-ready.  A
watchdog now also swaps to OpenStreetMap tiles if the satellite source errors
out, so the Earth is always visible.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import webbrowser  # noqa: F401  (kept: see ..browser.open_url)
from typing import Any
from urllib.parse import quote_plus

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from ..lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..geo import looks_like_uk_postcode
from ..browser import open_url
from ..utils import first_line
from .. import background

try:
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView
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

GLOBE_HTML = """<!doctype html><html><head><meta charset="utf-8">
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
  .cesium-viewer-bottom{display:none!important}  /* hide default credit clutter; attribution below */
  #credit{position:absolute;bottom:2px;right:8px;color:#6a6a74;font-size:9px;z-index:10}
</style>
<link href="https://cdn.jsdelivr.net/npm/cesium@1.115.0/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
</head><body>
<div id="cesiumContainer"></div>
<div id="loading">Rendering Earth…</div>
<div id="panel"><h2 id="place">O.R.I.O.N. Globe</h2>
  <div class="sub" id="coords">Ask me to take you somewhere — or type a place.</div>
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
<div id="acpanel">
  <div class="achead"><span id="accs">—</span><span class="mx" onclick="avSelect(null)">✕</span></div>
  <table id="acrows"></table>
</div>
<div id="zoomrail">
  <div class="zbtn" id="zin" title="Zoom in">+</div>
  <div class="ztick"></div>
  <div class="zbtn" id="zout" title="Zoom out">−</div>
</div>
<div id="hint">Drag to rotate · scroll or +/− to zoom · right-drag to tilt · Reset to return to orbit</div>
<div id="credit">Imagery © Esri · Cesium</div>
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
let viewer=null, spin=true, marker=null, baseLayer=null, labelLayer=null, fellBack=false;

// Keyless tile sources, all constructed synchronously (never blocks the globe).
const IMAGERY={
  esri:()=>new Cesium.UrlTemplateImageryProvider({
    url:'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maximumLevel:19, credit:'Esri World Imagery'}),
  esriLabels:()=>new Cesium.UrlTemplateImageryProvider({
    url:'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    maximumLevel:19, credit:'Esri'}),
  osm:()=>new Cesium.OpenStreetMapImageryProvider({url:'https://tile.openstreetmap.org/'}),
};

function setSub(text){const e=document.getElementById('coords'); if(e)e.textContent=text;}

function initGlobe(){
  if(typeof Cesium==='undefined') return;    // offline — message already shown
  Cesium.Ion.defaultAccessToken=undefined;   // keyless providers only
  const esri=IMAGERY.esri();
  viewer=new Cesium.Viewer('cesiumContainer',{
    baseLayer:new Cesium.ImageryLayer(esri),
    baseLayerPicker:false, geocoder:false, timeline:false, animation:false,
    homeButton:false, sceneModePicker:false, navigationHelpButton:false,
    fullscreenButton:false, selectionIndicator:false, infoBox:false,
    contextOptions:{webgl:{preserveDrawingBuffer:true}},
  });
  baseLayer=viewer.imageryLayers.get(0);
  // Place names + borders reference layer (city labels as you zoom — the
  // Google-Earth feel).  Synchronous too; a failure only costs the labels.
  try{ labelLayer=viewer.imageryLayers.addImageryProvider(IMAGERY.esriLabels()); }catch(e){}
  // Watchdog: if the satellite tiles error out repeatedly in the first half
  // minute, swap the whole base layer to OpenStreetMap — never a black globe.
  let tileErrors=0; const started=Date.now();
  esri.errorEvent.addEventListener(()=>{
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
  scene.globe.baseColor=Cesium.Color.fromCssColorString('#16324f');
  scene.screenSpaceCameraController.minimumZoomDistance=180;      // street level
  scene.screenSpaceCameraController.maximumZoomDistance=40e6;     // ~orbit cap
  // ── fast-navigation crash hardening (Mark X.7.2) ────────────────────────
  // Zooming / panning FAST used to flood the GPU with high-detail tiles and
  // full-resolution frames quicker than the driver could drain them, which
  // re-tripped the access violation.  The rule here is simple: the harder you
  // push the camera, the LIGHTER each frame becomes, so a burst of input can
  // never outrun the renderer.
  viewer.targetFrameRate=30;                          // hard cap on GPU frames/s
  try{ scene.postProcessStages.fxaa.enabled=false; }catch(e){}   // drop full-screen AA pass
  const ssc=scene.screenSpaceCameraController;
  ssc.inertiaZoom=0.0;                                // no coasting after a fast flick
  ssc.inertiaSpin=0.5; ssc.inertiaTranslate=0.5;
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
  let _userDown=false, _restoreTimer=null;
  function _lowRes(){ viewer.resolutionScale=0.6; scene.globe.maximumScreenSpaceError=6; }
  function _crisp(){ if(viewer&&!viewer.isDestroyed()){ viewer.resolutionScale=1.0;
      scene.globe.maximumScreenSpaceError=2; scene.requestRender(); } }
  function _scheduleCrisp(ms){ clearTimeout(_restoreTimer);
      _restoreTimer=setTimeout(()=>{ if(!_userDown) _crisp(); }, ms); }
  scene.camera.moveStart.addEventListener(()=>{ _lowRes(); clearTimeout(_restoreTimer); });
  scene.camera.moveEnd.addEventListener(()=>{ _scheduleCrisp(200); });
  viewer.canvas.addEventListener('pointerdown',()=>{ _userDown=true; _lowRes(); clearTimeout(_restoreTimer); });
  window.addEventListener('pointerup',()=>{ _userDown=false; _scheduleCrisp(200); });
  viewer.canvas.addEventListener('wheel',()=>{ _lowRes(); _scheduleCrisp(300); },{passive:true});
  // Idle orbit spin until the user (or a flyTo) takes over.  Moving the camera
  // implicitly requests a render, so the spin still animates under
  // requestRenderMode; a change here also keeps requestRender explicit.
  viewer.clock.onTick.addEventListener(()=>{
    if(spin){ viewer.scene.camera.rotate(Cesium.Cartesian3.UNIT_Z,-0.0006);
              viewer.scene.requestRender(); }
  });
  viewer.canvas.addEventListener('pointerdown',()=>{spin=false;});
  // WebGL context loss is the classic "globe went blank / crashed" cause on
  // flaky GPU drivers.  Report it and rebuild the viewer when the context is
  // restored, instead of leaving a dead canvas.
  viewer.canvas.addEventListener('webglcontextlost',(e)=>{
    e.preventDefault(); console.error('WEBGL_CONTEXT_LOST');
    const l=document.getElementById('loading');
    if(l){l.style.display='';l.textContent='Restoring the globe…';}
  },false);
  viewer.canvas.addEventListener('webglcontextrestored',()=>{
    console.error('WEBGL_CONTEXT_RESTORED');
    try{ if(viewer){ viewer.destroy(); viewer=null; fellBack=false; } }catch(e){}
    initGlobe();
  },false);
  viewer.camera.setView({destination:Cesium.Cartesian3.fromDegrees(0.0,18.0,24e6)});
  // Drop the loading veil on the first rendered frame (the globe is already
  // visible then, tiles stream in behind it).
  const l=document.getElementById('loading');
  if(l){ const once=()=>{ l.remove(); viewer.scene.postRender.removeEventListener(once); };
         viewer.scene.postRender.addEventListener(once); }
  window.__globeReady=true;
}
initGlobe();

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
  AV.col={air:Cesium.Color.fromCssColorString('#7fe9ff'),
          gnd:Cesium.Color.fromCssColorString('#9aa3ad'),
          sel:Cesium.Color.fromCssColorString('#ff5c73')};
  AV.bb=viewer.scene.primitives.add(new Cesium.BillboardCollection());
  AV.lb=viewer.scene.primitives.add(new Cesium.LabelCollection());
  AV.byId.forEach(a=>{a.b=null;a.l=null;});
  AV.ring=null;
  viewer.scene.preUpdate.addEventListener(avTick);
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
  if(!AV.on||!AV.byId.size) return;
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
  for(let i=0;i<list.length;i++){
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
    fill:false,outline:true,outlineColor:Cesium.Color.fromCssColorString('#7fe9ff').withAlpha(0.55)}});
  viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(c,r),
    {offset:new Cesium.HeadingPitchRange(0,Cesium.Math.toRadians(-62),r*2.6),duration:3.0});
  document.getElementById('place').textContent='Air traffic'+(label?' — '+label:'');
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

// Render-loop gate: Python pauses the loop when the globe tab is hidden, so the
// GPU is not driven continuously behind another view (heat, power, driver
// resets — a frequent crash trigger).  Resumed when the tab is shown again.
// The aircraft pump and the detail card's clock stop with it.
window.orionSetActive=function(on){
  AV.active=!!on;
  if(on){ avPump(); if(AV.selected&&!AV.panelTimer) AV.panelTimer=setInterval(avPanel,1000); }
  else { clearTimeout(AV.timer); AV.timer=null; clearInterval(AV.panelTimer); AV.panelTimer=null; }
  if(!viewer) return;
  try{ viewer.useDefaultRenderLoop=!!on; }catch(e){}
};

window.flyTo=function(lat,lon,label){
  if(!viewer) return;
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
  // Google-Earth style approach: high orbit → swoop to a tilted city view.
  viewer.camera.flyTo({
    destination:Cesium.Cartesian3.fromDegrees(lon,lat-0.12,16000),
    orientation:{heading:0.0,pitch:Cesium.Math.toRadians(-38),roll:0.0},
    duration:4.2,
  });
  document.getElementById('place').textContent=label||'Location';
  document.getElementById('coords').textContent=lat.toFixed(3)+', '+lon.toFixed(3)
    +' — scroll to street level, right-drag to tilt.';
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
  if(marker){viewer.entities.remove(marker); marker=null;}
  viewer.camera.flyTo({destination:Cesium.Cartesian3.fromDegrees(0.0,18.0,24e6),duration:3.0});
  setTimeout(()=>{spin=true;},3200);
  document.getElementById('place').textContent='O.R.I.O.N. Globe';
  document.getElementById('coords').textContent='Whole Earth — drag to rotate, scroll to zoom.';
  document.getElementById('news').innerHTML='';
  document.getElementById('footage').style.display='none';
  hideMap();
};
// Google Maps street-level card — correlates the globe with a real map.
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
  for(let i=0;i<n;i++){ viewer.render(); await new Promise(r=>setTimeout(r,ms||60)); }
  return viewer.scene.globe.tilesLoaded; };
</script></body></html>"""


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
        self.place_input.setPlaceholderText(
            "Fly to a place or postcode — e.g. Tokyo, Kyiv, São Paulo, SW1A 1AA…")
        self.place_input.returnPressed.connect(self._on_go)
        go = QPushButton("FLY THERE")
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
        reset = QPushButton("⌂ RESET")
        reset.setToolTip("Zoom back out to the whole Earth")
        reset.clicked.connect(self.reset_view)
        row.addWidget(self.place_input, 1)
        row.addWidget(go)
        row.addWidget(zoom_out)
        row.addWidget(zoom_in)
        row.addWidget(reset)
        layout.addLayout(row)

        self.view: Any = None
        self._layout = layout
        self._built = False
        self._rebuilds = 0          # crash-recovery rebuilds this session (throttled)
        self._active = False        # is this tab currently visible?
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
        if self._built or not WEBENGINE_OK:
            return
        self._built = True
        if self._placeholder is not None:
            self._placeholder.hide()
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None
        self.view = QWebEngineView()
        self.view.setPage(_GlobePage(self.view, on_js_error=self._on_js_error))
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

    # ── crash resilience ────────────────────────────────────────────────────────

    def _on_js_error(self, message: str) -> None:
        """A severe JS / WebGL error inside the globe page."""
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
        self._built = False
        try:
            if self.view is not None:
                self._layout.removeWidget(self.view)
                self.view.deleteLater()
        except Exception:
            pass
        self.view = None
        # Small defer so the dead process is fully reaped before we respawn.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(800, self._rebuild_after_crash)

    def _rebuild_after_crash(self) -> None:
        self._ensure_built()
        if self._active:
            self._set_active(True)
        self.bus.log.emit(f"GLOBE: renderer rebuilt after crash (attempt {self._rebuilds}).")

    def _set_active(self, on: bool) -> None:
        """Drive the Cesium render loop only while the tab is visible."""
        self._active = on
        if self.view is not None:
            # Guard the call: before the page finishes loading orionSetActive is
            # undefined, and the default render loop is already on, so a no-op is
            # correct (and avoids a spurious ReferenceError on first show).
            self._js(f"window.orionSetActive&&orionSetActive({'true' if on else 'false'})")
        self._sync_aviation()

    # ── live aviation layer ─────────────────────────────────────────────────

    AVIATION_POLL_MS = 10_000       # = aviation.CACHE_TTL_S; OpenSky data lasts 30 s

    def _on_dashboard_event(self, channel: str, payload: Any) -> None:
        if channel == "aviation":       # the aviation tool's show / hide
            self._sync_aviation()

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
        if self._aviation_busy:
            self._aviation_repoll = True     # e.g. a new focus mid-request
            return
        self._aviation_busy = True
        if background.spawn(self._aviation_refresh(), name="globe-aviation") is None:
            self._aviation_busy = False      # no loop (shutting down)

    async def _aviation_refresh(self) -> None:
        from ..aviation import get_service, globe_batch
        service = get_service()
        layer = service.layer
        seq = layer.focus_seq
        report = None
        try:
            report = await service.nearby(layer.lat, layer.lon, layer.radius_km)
        except Exception as exc:             # a feed fault must never reach the loop
            self._aviation_note(f"refresh failed - {first_line(exc, 120)}")
        finally:
            self._aviation_busy = False
        current = service.layer
        # Hidden, switched off or re-focused while the request was in flight:
        # drawing this now would undo that.
        if report is not None and current.on and self._active and current.focus_seq == seq:
            if report.error:
                self._aviation_note(report.error)
            else:
                self._aviation_error = ""
            if report.source:
                batch = globe_batch(report, reset=self._aviation_reset)
                self._aviation_reset = False
                self._js("window.orionAircraft&&orionAircraft("
                         + json.dumps(batch, separators=(",", ":")) + ")")
        if self._aviation_repoll:
            self._aviation_repoll = False
            self._poll_aviation()

    def _aviation_note(self, message: str) -> None:
        """Log a feed problem once, not every ten seconds while it lasts."""
        if message != self._aviation_error:
            self._aviation_error = message
            self.bus.log.emit(f"GLOBE: aircraft layer - {message[:180]}")

    def showEvent(self, event: Any) -> None:
        self._ensure_built()
        self._set_active(True)
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:
        # Stop rendering the globe when it's not on screen — no GPU churn behind
        # another tab (a common overheating / driver-reset crash trigger).
        self._set_active(False)
        super().hideEvent(event)

    # ── control ───────────────────────────────────────────────────────────────

    def _on_go(self) -> None:
        place = self.place_input.text().strip()
        if place:
            self.go_to_place(place)

    def reset_view(self) -> None:
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
        if not place:
            return
        self._ensure_built()   # build the renderer if ORION was asked by voice first
        self.bus.log.emit(f"GLOBE: travelling to {place}.")
        try:
            background.spawn(self._go(place))
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
        # Open-Meteo is a city/town gazetteer — it cannot resolve a postcode and
        # returns a wrong or empty hit (the "flies to the sea" bug).  For a
        # postcode-shaped query, don't guess: report not-found so the user sees
        # the miss rather than a random point.
        if looks_like_uk_postcode(place):
            return None
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

    def _on_load_finished(self, ok: bool) -> None:
        self._page_loaded = bool(ok)
        if ok:
            # Whatever was asked before the page existed lands now: the latest
            # visibility, then any fly-to/news requests in the order given.
            self._set_active(self._active)
            pending, self._pending_js = list(getattr(self, "_pending_js", [])), []
            for code in pending:
                self._js(code)

    def _js(self, code: str) -> None:
        """Run JS only once the page has LOADED. runJavaScript on a view that
        was only just built kills the process natively — no traceback — which
        follows the same deferred renderer pattern; showEvent did exactly that here,
        calling straight after setHtml."""
        if self.view is None:
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
