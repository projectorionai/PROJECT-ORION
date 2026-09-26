"""
QuantumFace3D — ORION's Quantum Human Visual System (Mark XIV).

ORION exists as a quantum intelligence.  No solid flesh.  No mask.  No
cartoon.  Data, light, geometry, and intelligence — red holographic structure
on black space, with subtle white highlights — sheathed in a TRANSLUCENT
QUANTUM SKIN: a semi-physical membrane with data streams flowing beneath it,
emissive data veins, and a subsurface glow (per the Mark XIV spec poster:
"skin is translucent, not solid — data moves beneath the surface").
Two states, one being:

    QUANTUM ORB (idle) — a floating spherical energy form: a turbulent
      fbm-shader core pulsing inside the instanced cube shell, rotating
      quantum rings and an icosahedral lattice, internal particle weather.
      Constant internal movement — alive, observant, waiting.

    QUANTUM HUMAN (active) — the orb destabilises and the being materialises
      as a fully holographic human head (and upper torso) built entirely
      from red quantum data: an assembling wireframe skull lattice, a dense
      surface point cloud, the signature cube swarm, glowing feature lines
      (brows, lips), digital eye constructs that wake last, and — new in
      Mark XIV — a translucent quantum-skin membrane that knits itself over
      the lattice: fresnel-rimmed, subsurface-lit by the baked form shading,
      with flowing data streams, pulsing vein filaments, and a living
      micro-structure animating beneath the surface.  Semi-physical,
      semi-digital; the anatomy stays readable and human-like.

TRANSFORMATION (≈2.4 s, scrubbed through smoothstep stage windows so it is
reversible and interruption-safe):
    1 destabilise   orb core surges, quantum geometry spins up
    2 explosion     cubes + particles erupt outward
    3 structure     feature cubes migrate to the facial anchor lines first
    4 framework     the wireframe skull lattice draws itself together
    5 detailing     dense data-point layers fill in, the skin membrane
                    knits crown-to-neck over the lattice, a scan ring sweeps
    6 eyes          the eye constructs boot with an ignition flicker
    7 torso         a wireframe upper torso rises (with its own membrane),
                    dissolving below the chest

The anatomy comes from ONE radial field function (headPoint): brow ridge,
orbital sockets, nose with tip and alae, philtrum, cupid-bow lips, chin,
zygomatic cheekbones, squared mandible, neck.  Every layer — cubes, points,
wires, feature lines — samples the same field, so the face reads as a single
being rather than stacked effects.

BEHAVIOUR STATES (window.orionState): IDLE subtle eye movement + micro data
shifts · SPEAKING amplitude-driven jaw/lips/cheeks + orbiting thought
streams · LISTENING eye tracking + attentive micro head movement ·
PROCESSING/THINKING cube activity intensifies, pupils contract, scan sweeps.
Volume reactivity throughout: more amplitude = brighter and more active.

HARD-WON RENDERING NOTES (do not regress):
  • Every holographic layer is ADDITIVE with depthWrite:false — additive
    blending is order-independent, so the transparent head never sorts wrong.
    The Mark XIV skin membrane follows the same rule (FrontSide only, so the
    inside of the shell never doubles the brightness).
  • The skin clears the eye sockets (aSock alpha cut) exactly like the point
    cloud does — the eye constructs must stay the brightest thing in the
    sockets or the gaze dies.
  • Each eye keeps an opaque BLACK backing disk behind the iris shader: it
    depth-occludes the back-of-skull wires so the pupil reads deep black
    (additive layers alone cannot darken anything).
  • Back-of-head data dims via model-space z (rig yaw stays within ±0.25 rad)
    — that single factor is what keeps the transparent face readable.
  • Torso layers use their own material instances with uJaw pinned to 0 —
    the GLSL jaw weight function would otherwise rotate the whole chest.
    This includes the torso skin membrane.
  • No shadow mapping, no scene lighting for holo layers; only the cube
    swarm uses lit material (emissive-heavy) for depth.

Rendered via Three.js/WebGL in a ``QWebEngineView`` — one InstancedMesh for
8 500 shell cubes, one Points draw for 22 000 data points, a handful of
LineSegments, adaptive frame governor, hidden-tab pause.  Interface parity
preserved: ``set_amplitude`` / ``set_speaking`` / ``set_state`` /
``apply_emotion`` / ``timer``; the page keeps
``window.orion{Amplitude,Morph,Emotion,Label,Pose,Snap,Step}`` and
``orionState`` plus a postMessage bridge (same contract the phone PWA uses).
Three.js loads from cdn.jsdelivr.net — the /face route CSP pins that exact
host, so the import map must not move.

Three.js is served from the complete vendored ``assets/three`` graph when it is
available, with the pinned jsDelivr build as a fallback. Diagnostics:
``window.orionSnap()`` (JPEG data URL of the current frame),
``window.orionStep(n)`` (manual frame advance), ``window.__dbg`` (timeline
stages and fps snapshot).
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..bus import OrionBus

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_OK = True
except Exception:  # pragma: no cover
    WEBENGINE_OK = False


FACE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;height:100%;background:#020207;overflow:hidden;font-family:'Segoe UI',Arial}
  #label{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
    color:#ff2438;font-size:12px;letter-spacing:4px;text-transform:uppercase;opacity:.8;
    text-shadow:0 0 12px rgba(255,36,56,.6)}
  #loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#5b2430;font-size:13px;
    text-align:center;line-height:1.7;max-width:80%}
</style></head><body>
<div id="loading">Materialising ORION…</div>
<div id="label">ORION</div>
<script>
// ── offline / CDN-failure guard ────────────────────────────────────────────
// Three.js is fetched from a CDN, so with no internet (ORION's own MODE B
// offline operation, a blocked CDN, a captive portal) the module below never
// executes at all — a failed `import` aborts the whole module script silently.
// The face then sat on "Materialising ORION…" forever with no error and no
// explanation, which reads as a hang rather than a missing dependency.
// A classic (non-module) script still runs in that case, so the honest
// message and the Python-side signal are wired here.
(function(){
  var settled=false;
  window.__orionFaceBooted=function(){ settled=true; };
  function fail(reason){
    if(settled) return;
    settled=true;
    // Recorded so the Qt side can tell a page that FAILED from one that is
    // merely still loading. Inferring failure from the absence of a frame
    // condemned pages that only needed another second, and the fallback it
    // triggered was permanent.
    window.__orionFailed=String(reason||'unknown');
    var l=document.getElementById('loading');
    if(l){
      // Not "no internet": the library is vendored, so the usual cause is the
      // renderer itself — no WebGL, a driver that refuses, a corrupt file.
      // Saying "connect to the internet" sent people to fix the one thing
      // that was already fine.
      l.innerHTML='ORION’s 3-D renderer did not start'
        + '<br><span style="opacity:.7;font-size:11px">'
        + 'falling back to his software-rendered face — nothing else is affected</span>';
    }
    // Let the Qt side know, so it can fall back to the 2-D face if it wants to.
    try{ if(window.orionFaceFailed) window.orionFaceFailed(String(reason||'')); }catch(_){}
  }
  window.addEventListener('error', function(e){
    var t=e&&e.target;
    if(t&&(t.tagName==='SCRIPT'||t.tagName==='LINK')) fail('asset load failed');
  }, true);
  // Belt and braces: a module that never runs fires no error event in some
  // engines, so a watchdog covers the silent case too.
  setTimeout(function(){ fail('timeout'); }, 12000);
})();
</script>
<script type="importmap">{"imports":{
  "three":"__THREE_BASE__build/three.module.js",
  "three/addons/":"__THREE_BASE__examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ═══════════════════════════════════════════════════════════════════════════
// ORION — QUANTUM HUMAN VISUAL SYSTEM (Mark XIV).
//
// Only data, light, geometry, and intelligence.  Idle: a QUANTUM ORB — a
// contained star breathing inside the voxel shell.  Active: a QUANTUM HUMAN —
// head and upper torso built from red holographic data: wireframe skull
// lattice, dense point cloud, instanced cube swarm, glowing feature lines,
// digital eye constructs — wrapped in a TRANSLUCENT QUANTUM SKIN membrane
// with data streams, vein filaments and micro-structure flowing beneath it.
// No texture maps, no solid opaque surface: semi-physical, semi-digital.
//
// Architecture (all procedural, zero external assets):
//   FIELD      headPoint(): one radial anatomy function feeds every layer
//   WIRE       lat/long skull lattice, draw-in ordered (LineSegments shader)
//   POINTS     22k surface + volumetric data points, feature-weighted
//   SKIN       translucent membrane over the same field: fresnel rim,
//              subsurface glow, under-skin data flow, emissive veins
//   SHELL      InstancedMesh cube swarm (orb ⇄ face homes, features first)
//   EYES       shader iris constructs (rings, fibres, ticks), halo rings,
//              digital shutter blink, saccades + camera-converged contact
//   MOUTH      feature-line lips, amplitude jaw + pseudo-visemes
//   TORSO      parametric wireframe bust + membrane, bottom-dissolve (st. 7)
//   SPACE      grid floor, drifting data cubes, volumetric haze billboard
//   TIMELINE   morph 0→1 scrubbed through 7 smoothstep stage windows
//   PERF       instancing, strided colour writes, adaptive quality governor
// ═══════════════════════════════════════════════════════════════════════════

const COUNT=8500;             // shell voxels (one instanced draw call)
const PTN=22000;              // holographic data points (one draw call)
const ORB_R=52;               // orb radius for the morph
const HEAD={a:39,b:57,c:47};  // half-axes — human head ratio
const FLAT=0.88;              // faceplate flattening of the front hemisphere
const EYE={x:16.5,y:13,r:9.5};// socket centres + depression radius
const MOUTH_Y=-23.2;          // lip seam line

// ── renderer / composer ─────────────────────────────────────────────────────
const scene=new THREE.Scene();
scene.fog=new THREE.FogExp2(0x050208,0.0015);
const BASE_FOV=32, FRAME_ASPECT=16/9;
const cam=new THREE.PerspectiveCamera(BASE_FOV,innerWidth/innerHeight,0.1,5000);
cam.position.set(0,6,268); cam.lookAt(0,4,0);
// A PerspectiveCamera's fov is VERTICAL, so a tall narrow viewport silently
// crops horizontally: docked beside the Command Deck at 300x893 the effective
// horizontal field collapsed to ~11 degrees and the face fell almost entirely
// outside it — the pane rendered black and looked like the face had broken.
// Widening the vertical fov on narrow viewports preserves the horizontal
// framing instead, so the face stays whole at any pane width.
function fitCamera(){
  const a=Math.max(1e-4,innerWidth/innerHeight);
  cam.aspect=a;
  cam.fov = a < FRAME_ASPECT
    ? 2*Math.atan(Math.tan(BASE_FOV*Math.PI/360)*(FRAME_ASPECT/a))*180/Math.PI
    : BASE_FOV;
  cam.updateProjectionMatrix();
}
fitCamera();
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));
renderer.outputColorSpace=THREE.SRGBColorSpace; renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=0.95; document.body.appendChild(renderer.domElement);
const composer=new EffectComposer(renderer);
composer.addPass(new RenderPass(scene,cam));
const bloom=new UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),0.42,0.55,0.55);
composer.addPass(bloom);

const rig=new THREE.Group(); scene.add(rig);
const torsoRig=new THREE.Group(); scene.add(torsoRig);
// Lighting exists ONLY for the cube swarm (emissive-heavy standard material);
// every holographic layer is unlit additive shader work.
scene.add(new THREE.AmbientLight(0x1a1014,0.9));
const key=new THREE.DirectionalLight(0xfff2ee,0.8);  key.position.set(-120,140,220); scene.add(key);
const rim=new THREE.DirectionalLight(0xff2038,1.2);  rim.position.set(200,20,-160);  scene.add(rim);

// ── reactive state (eased towards targets set through the Python bridge) ───
const st={density:1.0,glow:0.5,pulse:0.0,speed:1.0,pvel:1.0,turb:0.1,brow:0.0,
  eyeH:1.0,eyeW:1.0,mouthCurve:0.05,tension:0.0,amp:0.0,dir:'drift',
  specLow:0.0,specMid:0.0,specHigh:0.0,
  dark:new THREE.Color(0x1e0509),mid:new THREE.Color(0x96162a),bright:new THREE.Color(0xff6a80),accent:new THREE.Color(0xff2a44)};
const tg=Object.assign({},st,{dark:st.dark.clone(),mid:st.mid.clone(),bright:st.bright.clone(),accent:st.accent.clone()});
const lerp=(a,b,t)=>a+(b-a)*t;
function sstep(a,b,x){const t=Math.max(0,Math.min(1,(x-a)/(b-a)));return t*t*(3-2*t);}
function bump(v,c,w){const d=(v-c)/w;return Math.exp(-d*d);}
let morphT=0, morphTarget=1;              // 0 = orb, 1 = quantum human
let mode='STANDBY', processing=0, listen=0;
window.orionAmplitude=a=>{tg.amp=Math.max(0,Math.min(1,a));};
// Spectral mouth shape (Mark X.14): band-energy ratios from the real audio,
// replacing the old sine-wave "pseudo-visemes" that moved on a fixed rhythm
// regardless of what was being said. clamp01 keeps a malformed bridge value
// from ever producing a NaN mouth shape.
const clamp01=v=>Math.max(0,Math.min(1,Number(v)||0));
window.orionSpectrum=(l,m,h)=>{tg.specLow=clamp01(l);tg.specMid=clamp01(m);tg.specHigh=clamp01(h);};
window.orionMorph=v=>{morphTarget=Math.max(0,Math.min(1,v));};
window.orionState=s=>{mode=String(s||'').toUpperCase();};
window.orionEmotion=p=>{ if(!p)return;
  const m={voxel_density:'density',glow:'glow',glow_pulse:'pulse',speed:'speed',particle_velocity:'pvel',
    turbulence:'turb',brow:'brow',eye_height:'eyeH',eye_width:'eyeW',mouth_curve:'mouthCurve',mouth_tension:'tension'};
  for(const k in m){ if(k in p) tg[m[k]]=p[k]; }
  if('particle_direction' in p) tg.dir=p.particle_direction;
  if(p.name) emoName=String(p.name).toLowerCase();
  const rgb=a=>new THREE.Color(a[0]/255,a[1]/255,a[2]/255);
  if(p.palette_dark)tg.dark=rgb(p.palette_dark); if(p.palette_mid)tg.mid=rgb(p.palette_mid);
  if(p.palette_bright)tg.bright=rgb(p.palette_bright); if(p.accent)tg.accent=rgb(p.accent);
};
// ── EXPRESSION RIG — named-emotion extras layered over the numeric profile ──
// The scalar profile (brow/eyes/mouth/palette) sets the base face; these
// give each named emotion the SHAPE language a face needs to actually read:
//   knit   inner-brow pull (anger/concentration)  sorrow inner-brow raise
//   headX  pitch (droop + / lean-in −)            headZ  curious head tilt
//   cheek  smile cheek-lift (data + membrane)     part   resting lip parting
//   pupil  dilation + / contraction −             gazeY  gaze bias (up +)
//   blink  blink-interval multiplier (alert fast, sad slow)
let emoName='neutral';
const EXPR={
  neutral:   {blink:1},
  happy:     {cheek:0.85,part:0.10,pupil:0.5,headX:-0.012,blink:1},
  excited:   {cheek:1.0,part:0.18,pupil:0.7,headX:-0.022,gazeY:0.12,blink:0.8},
  concerned: {knit:0.35,sorrow:0.5,headX:0.02,headZ:0.02,pupil:0.1,blink:1.25},
  thinking:  {knit:0.5,headZ:0.035,pupil:-0.4,gazeY:0.5,blink:1.4},
  frustrated:{knit:1.0,headX:0.012,pupil:-0.5,blink:0.7},
  critical:  {knit:0.8,part:0.10,pupil:-0.7,headX:-0.02,blink:0.55},
  sad:       {sorrow:1.0,knit:0.15,headX:0.06,headZ:0.015,pupil:0.1,gazeY:-0.45,blink:1.8},
  listening: {headZ:0.02,cheek:0.15,part:0.05,pupil:0.15,blink:1},
  speaking:  {cheek:0.25,part:0.04,blink:1},
  alert:     {headX:-0.03,part:0.10,pupil:-0.3,gazeY:0.1,blink:0.6},
  // Mark XXVI — the nine new expressions get their eye/posture body-language too
  // (they already receive the scalar geometry via orionEmotion); micro scale.
  concentrating:{knit:0.55,headZ:0.02,pupil:-0.45,gazeY:0.2,blink:1.5},
  curious:   {headZ:0.045,gazeY:0.15,pupil:0.3,part:0.05,cheek:0.15,blink:0.95},
  amused:    {cheek:0.6,part:0.08,pupil:0.35,headX:-0.008,blink:1},
  confused:  {knit:0.25,headZ:0.05,pupil:0.05,gazeY:-0.05,blink:1.1},
  uncertain: {knit:0.15,sorrow:0.2,headZ:0.02,gazeY:-0.1,pupil:0.05,blink:1.3},
  disappointed:{sorrow:0.6,knit:0.1,headX:0.03,gazeY:-0.25,pupil:0.1,blink:1.4},
  proud:     {cheek:0.7,part:0.06,headX:-0.02,pupil:0.2,blink:0.95},
  reassuring:{cheek:0.5,part:0.05,pupil:0.25,headX:0.006,blink:1.05},
  empathetic:{sorrow:0.4,knit:0.1,headZ:0.02,headX:0.015,pupil:0.15,gazeY:-0.05,blink:1.2},
};
const exState={knit:0,sorrow:0,headX:0,headZ:0,cheek:0,part:0,pupil:0,gazeY:0,blink:1};
window.orionLabel=t=>{const e=document.getElementById('label'); if(e)e.textContent=t;};
// postMessage bridge — the phone PWA (and the dev harness) drive the avatar
// with {amp, morph, emotion, label, state} messages.
addEventListener('message',e=>{const d=e.data||{};try{
  if(d.amp!==undefined)window.orionAmplitude(d.amp);
  if(d.spectrum!==undefined)window.orionSpectrum(d.spectrum.low,d.spectrum.mid,d.spectrum.high);
  if(d.morph!==undefined)window.orionMorph(d.morph);
  if(d.emotion)window.orionEmotion(d.emotion);
  if(d.label)window.orionLabel(d.label);
  if(d.state)window.orionState(d.state);
}catch(_){}});
window.orionPose=(y,x)=>{rig.rotation.y=y||0; rig.rotation.x=x||0; poseHold=1.0;};
window.orionSnap=(w=420,q=0.6)=>{composer.render();
  const c=document.createElement('canvas'); c.width=w; c.height=Math.round(w*innerHeight/innerWidth);
  c.getContext('2d').drawImage(renderer.domElement,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',q);};
let poseHold=0;

// ═══════════════════════════════════════════════════════════════════════════
// FIELD — the anatomy.  One radial function shapes every layer of the being.
// ═══════════════════════════════════════════════════════════════════════════
function headPoint(dx,dy,dz){
  const r=1/Math.sqrt((dx/HEAD.a)**2+(dy/HEAD.b)**2+(dz/HEAD.c)**2);
  let x=dx*r, y=dy*r, z=dz*r;
  const front=z>0;
  // Cranium: broad, shallow crown; slight parietal width; blunt base.
  if(y>28) y=28+(y-28)*0.58;
  x*=1+0.03*bump(y,24,18);
  if(y<-46) y=-46+(y+46)*0.60;
  // Temples flatten between brow and cheekbone.
  const temple=sstep(8,22,Math.abs(x))*bump(y,13,24)*(front?0.5:0.32);
  x*=1-0.08*temple;
  // Ears: a subtle lateral swell where the pinna sits.
  x*=1+0.09*bump(y,2,8)*bump(z,-6,8)*sstep(26,36,Math.abs(x));
  if(!front) z*=1.06;                       // occipital fullness
  const ax=Math.abs(x);
  if(front){
    const fz=z/HEAD.c;
    // Forehead plane: gentle slope back above the brow shelf.
    z-=2.2*sstep(24,40,y)*sstep(0.2,0.6,fz);
    // Brow ridge: bony shelf over the sockets, softened toward the temples.
    z+=4.2*bump(y,19.5,6.5)*sstep(32,4,ax)*sstep(0.15,0.5,fz);
    // Orbital sockets: deep smooth recessions the eye constructs sit inside.
    const socket=Math.exp(-(((ax-EYE.x)/7.8)**2+((y-EYE.y)/6.2)**2));
    z-=6.6*socket*sstep(0.1,0.4,fz);
    // Nose: bridge descending from glabella into a defined tip + alae,
    // tapering off below the tip so it never smears into the lip.
    const noseCol=bump(x,0,3.4);
    const noseTaper=1-sstep(-9.5,-15,y);
    if(y<15&&y>-16) z+=noseCol*(2.4+7.0*sstep(15,-9,y))*noseTaper;
    z+=noseCol*5.4*bump(y,-9.5,4.2);                       // tip
    z+=2.6*bump(ax,4.4,2.2)*bump(y,-11.5,2.8);             // alae
    z-=1.4*bump(ax,3.0,1.3)*bump(y,-13.2,1.5);             // nostril shadow
    // Zygomatic cheekbones + a soft diagonal hollow beneath.
    z+=4.6*bump(y,1,7)*bump(ax,25,7);
    z-=1.1*bump(y,-8-0.25*ax,8)*bump(ax,18,9);
    // Philtrum groove between nose and upper lip.
    z-=0.55*bump(x,0,2.2)*bump(y,-17.5,2.6);
    // LIPS — cupid-bow upper, fuller lower, seam crease between.
    const lipW=sstep(17,4,ax);
    const lipU=bump(y,-20.6,2.3)*lipW*(0.75+0.5*bump(ax,2.6,2.2));
    const seam=bump(y,MOUTH_Y,1.2)*sstep(15,3,ax);
    const lipL=bump(y,-26.2,2.7)*sstep(14,3,ax);
    z+=3.0*lipU-2.0*seam+3.4*lipL;
    z-=1.4*bump(y,-31.5,2.6)*sstep(12,3,ax);               // mentolabial groove
    z+=2.6*bump(y,-40,5)*sstep(13,0,ax);                   // chin boss
    z*=FLAT;
  }
  // Mandible: the face narrows along a squared jawline to a decisive chin.
  // The gonial-corner term fades in and out with t — a hard gate here once
  // printed a visible ledge across both cheeks.
  if(y<-14){
    const t=Math.min(1,(-14-y)/34);
    x*=(1-0.48*t*t)*(1+0.12*bump(ax,24,9)*t*(1-t)*4);
    if(z>0) z*=(1-0.15*t); else z*=(1-0.34*t*t);
  }
  // Neck: below the jaw the form blends into a columnar neck.
  if(y<-44){
    const t=sstep(-44,-62,y);
    x=lerp(x,x*0.62,t);
    z=lerp(z,z*0.55-3.5,t);
    y=Math.max(y,-66);
  }
  return {x,y,z};
}
function fieldRadius(theta,phi){ // spherical → surface point
  const dx=Math.sin(phi)*Math.cos(theta), dy=Math.cos(phi), dz=Math.sin(phi)*Math.sin(theta);
  return headPoint(dx,dy,dz);
}
function plateZ(x,y){
  const q=1-(x/HEAD.a)**2-(y/HEAD.b)**2;
  return q>0?FLAT*HEAD.c*Math.sqrt(q):0;
}
// Featureness: how strongly a surface point sits on an anatomical frame line
// (brow, nose, cheekbones, orbital rims, jawline, lips).  Feature data lands
// first in the assembly and stays brightest — this is what keeps the face
// readable when everything is transparent.
function featureWeight(p){
  const ax=Math.abs(p.x); let f=0;
  f=Math.max(f,bump(p.y,19.5,4)*sstep(30,4,ax));                  // brow
  f=Math.max(f,bump(p.x,0,4)*sstep(16,-14,p.y)*(p.z>0?1:0));      // nose line
  f=Math.max(f,bump(p.y,1,5)*bump(ax,25,6));                      // zygomatic
  f=Math.max(f,bump(Math.hypot(ax-EYE.x,p.y-EYE.y),EYE.r,2.5));   // orbit rims
  f=Math.max(f,sstep(-20,-34,p.y)*bump(ax,17,5)*(p.z>0?1:0));     // jawline
  f=Math.max(f,bump(p.y,-20.6,2.6)*sstep(15,3,ax)*(p.z>0?1:0));   // upper lip
  f=Math.max(f,bump(p.y,-26.2,2.8)*sstep(14,3,ax)*(p.z>0?1:0));   // lower lip
  return Math.min(1,f);
}
// Socket mask: the eye constructs replace surface data inside the sockets.
function socketWeight(p){
  if(p.z<=6) return 0;
  const d=Math.hypot(Math.abs(p.x)-EYE.x,p.y-EYE.y);
  return Math.exp(-((d/6.4)**2));
}
// Baked form shading: numerical field normal against a fixed front-top light.
// This is what makes the transparent data head read as a 3-D face — bright
// planes, shadowed sockets and jaw underside — without any scene lighting.
// (Baking is safe: the rig yaw stays within ±0.25 rad.)
const LIGHT={x:0.20,y:0.36,z:0.91};
function fieldNormal(theta,phi){
  const e=0.012, p0=fieldRadius(theta,phi), pt=fieldRadius(theta+e,phi), pp=fieldRadius(theta,phi+e);
  const t1x=pt.x-p0.x,t1y=pt.y-p0.y,t1z=pt.z-p0.z;
  const t2x=pp.x-p0.x,t2y=pp.y-p0.y,t2z=pp.z-p0.z;
  let nx=t1y*t2z-t1z*t2y, ny=t1z*t2x-t1x*t2z, nz=t1x*t2y-t1y*t2x;
  const l=Math.hypot(nx,ny,nz)||1;
  const s=(nx*p0.x+ny*p0.y+nz*p0.z)<0?-1:1;
  return {x:s*nx/l,y:s*ny/l,z:s*nz/l};
}
function shadeAt(theta,phi){
  phi=Math.max(0.03,Math.min(Math.PI-0.03,phi));
  const n=fieldNormal(theta,phi);
  return Math.max(0,Math.min(1,0.24+0.76*Math.max(0,n.x*LIGHT.x+n.y*LIGHT.y+n.z*LIGHT.z)));
}

// Jaw deform shared by every layer: weight(y,z) then rotate about the condyle
// axis.  The GLSL mirror lives in JAW_GLSL below.
const JAW={py:-6,pz:-16,angle:0.15};
function jawWeight(y,z){ return sstep(-16,-30,y)*sstep(-20,8,z); }
function jawApply(p,open){
  const w=jawWeight(p.y,p.z)*open; if(w<=0)return p;
  const a=JAW.angle*w, ca=Math.cos(a), sa=Math.sin(a);
  const y=p.y-JAW.py, z=p.z-JAW.pz;
  return {x:p.x, y:JAW.py+y*ca-z*sa, z:JAW.pz+y*sa+z*ca};
}
const JAW_GLSL=`
  float sstepf(float a,float b,float x){float t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
  vec3 jawDeform(vec3 p,float open){
    float w=sstepf(-16.,-30.,p.y)*sstepf(-20.,8.,p.z)*open;
    if(w>0.001){
      float a=0.15*w; float ca=cos(a), sa=sin(a);
      float yy=p.y+6.0, zz=p.z+16.0;
      p.y=-6.0+yy*ca-zz*sa; p.z=-16.0+yy*sa+zz*ca;
    }
    return p;
  }`;

// ═══════════════════════════════════════════════════════════════════════════
// WIRE — the skull lattice.  Lat/long contour lines over the field, each line
// drawing itself in along its length as uBuild sweeps (stage 4 framework /
// stage 5 fine detail / stage 7 torso).  Additive, depthWrite off.
// ═══════════════════════════════════════════════════════════════════════════
const WIRE_VERT=`
  uniform float uJaw;
  attribute float aOrder,aFeat,aAux,aShade;
  varying float vO,vF,vZ,vY,vAux,vSh;
  `+JAW_GLSL+`
  void main(){
    vec3 p=jawDeform(position,uJaw);
    vO=aOrder; vF=aFeat; vZ=position.z; vY=position.y; vAux=aAux; vSh=aShade;
    gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
  }`;
const WIRE_FRAG=`
  uniform float uBuild,uTime,uOpacity,uEnergy,uFadeA,uFadeB;
  uniform vec3 uMid,uBright,uAccent;
  varying float vO,vF,vZ,vY,vAux,vSh;
  float sstepf(float a,float b,float x){float t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
  void main(){
    float vis=sstepf(vO-0.05,vO,uBuild);
    if(vis<0.004) discard;
    float tip=exp(-pow((uBuild-vO)*26.,2.))*step(0.02,uBuild)*(1.-step(0.985,uBuild));
    float scan=0.85+0.15*sin(vY*0.32-uTime*2.2);
    float front=mix(0.30,1.0,smoothstep(-24.,12.,vZ));
    float fade=1.0-smoothstep(uFadeA,uFadeB,-vY);
    float a=uOpacity*vis*front*scan*fade*(0.35+0.65*vSh)*(1.+0.4*vF)*(1.0-vAux);
    vec3 col=(mix(uMid,uBright,vSh*0.7+vF*0.3)+uAccent*(tip*2.5+uEnergy*0.35))*1.6;
    gl_FragColor=vec4(col,a);
  }`;
function makeWireMat(opacity,fadeA,fadeB){
  return new THREE.ShaderMaterial({
    uniforms:{uJaw:{value:0},uBuild:{value:0},uTime:{value:0},uOpacity:{value:opacity},
      uEnergy:{value:0.4},uFadeA:{value:fadeA},uFadeB:{value:fadeB},
      uMid:{value:st.mid.clone()},uBright:{value:st.bright.clone()},uAccent:{value:st.accent.clone()}},
    vertexShader:WIRE_VERT,fragmentShader:WIRE_FRAG,
    transparent:true,blending:THREE.AdditiveBlending,depthWrite:false});
}
function buildHeadWire(latStep,lonStep,phase){
  const LAT=44,LON=88, pos=[],ord=[],fea=[],aux=[],shd=[];
  const ANG=(i,j)=>[ (j%LON)/LON*Math.PI*2, (i/LAT)*Math.PI ];
  const push=(i,j,o)=>{
    const [th,phi]=ANG(i,j), p=fieldRadius(th,phi);
    pos.push(p.x,p.y,p.z); ord.push(Math.min(0.93,o));
    fea.push(featureWeight(p)); aux.push(socketWeight(p)*0.7); shd.push(shadeAt(th,phi));
  };
  const wrap=t=>{let d=Math.abs(t-Math.PI/2); if(d>Math.PI)d=2*Math.PI-d; return d/Math.PI;};
  // meridians first (0.05–0.55): they sweep outward from the face centre
  for(let j=phase;j<LON;j+=lonStep){
    const th=(j/LON)*Math.PI*2, base=0.05+0.42*wrap(th);
    for(let i=3;i<LAT-1;i++){
      push(i,j,base+0.10*(i/LAT));
      push(i+1,j,base+0.10*((i+1)/LAT));
    }
  }
  // rings second (0.45–0.93): they close the lattice crown-to-neck
  for(let i=3+phase;i<LAT-1;i+=latStep){
    const base=0.45+0.40*(i/LAT);
    for(let j=0;j<LON;j++){
      push(i,j,base+0.08*(j/LON));
      push(i,j+1,base+0.08*((j+1)/LON));
    }
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('aOrder',new THREE.Float32BufferAttribute(ord,1));
  g.setAttribute('aFeat',new THREE.Float32BufferAttribute(fea,1));
  g.setAttribute('aAux',new THREE.Float32BufferAttribute(aux,1));
  g.setAttribute('aShade',new THREE.Float32BufferAttribute(shd,1));
  return g;
}
const wireMain=new THREE.LineSegments(buildHeadWire(4,4,0),makeWireMat(0.85,900,1000));
const wireFine=new THREE.LineSegments(buildHeadWire(4,4,2),makeWireMat(0.42,900,1000));
wireMain.frustumCulled=false; wireFine.frustumCulled=false;
rig.add(wireMain); rig.add(wireFine);

// ═══════════════════════════════════════════════════════════════════════════
// POINTS — 22k holographic data points: surface + volumetric interior, denser
// and brighter along the feature lines, twinkling, density-controllable.
// ═══════════════════════════════════════════════════════════════════════════
const PTS_VERT=`
  uniform float uTime,uFill,uStruct,uDensity,uJaw,uAmp,uPx,uSpeak;
  attribute float aSeed,aFeat,aOrder,aVol,aAux,aShade;
  varying float vA,vF,vTw,vSh;
  `+JAW_GLSL+`
  void main(){
    float fillV=sstepf(aOrder,aOrder+0.16,uFill);
    float guide=uStruct*step(0.55,aFeat);
    float vis=max(fillV,guide);
    float show=step(aSeed,uDensity);
    vec3 p=jawDeform(position,uJaw);
    vec3 dir=normalize(position+vec3(0.,0.,0.001));
    p+=dir*(1.0-vis)*10.0;                       // data falls into place
    p+=dir*sin(uTime*0.9+aSeed*40.)*0.35;        // breathing micro-drift
    float tw=0.62+0.38*sin(uTime*(1.2+aSeed*1.8)+aSeed*61.7);
    float cheek=uSpeak*exp(-pow((position.y+8.)/12.,2.))*step(0.,position.z);
    vec4 mv=modelViewMatrix*vec4(p,1.);
    float front=smoothstep(-24.,12.,position.z);
    vA=vis*show*mix(0.30,1.0,front)*mix(1.0,0.22,aVol)*(1.-aAux*0.75);
    vF=aFeat; vTw=tw; vSh=aShade;
    float size=mix(2.2,3.2,aFeat*0.4+aShade*0.6)*(0.8+0.2*tw)*(1.+uAmp*0.35+cheek*0.5);
    gl_PointSize=size*uPx*vis*show*(340.0/max(40.0,-mv.z));
    gl_Position=projectionMatrix*mv;
  }`;
const PTS_FRAG=`
  uniform vec3 uMid,uBright,uAccent; uniform float uEnergy;
  varying float vA,vF,vTw,vSh;
  void main(){
    vec2 q=gl_PointCoord-0.5; float d=length(q);
    float a=smoothstep(0.5,0.12,d)*vA*(0.25+0.75*vSh)*(1.+0.30*vF);
    if(a<0.004) discard;
    vec3 col=mix(uMid,uBright,min(1.,vSh*0.75+vF*0.25));
    col+=uAccent*uEnergy*0.6;
    gl_FragColor=vec4(col*(0.60+vTw*0.4+vSh*0.4),a);
  }`;
function makePtsMat(){
  return new THREE.ShaderMaterial({
    uniforms:{uTime:{value:0},uFill:{value:0},uStruct:{value:0},uDensity:{value:0.8},
      uJaw:{value:0},uAmp:{value:0},uPx:{value:renderer.getPixelRatio()},uSpeak:{value:0},
      uEnergy:{value:0.4},uMid:{value:st.mid.clone()},uBright:{value:st.bright.clone()},
      uAccent:{value:st.accent.clone()}},
    vertexShader:PTS_VERT,fragmentShader:PTS_FRAG,
    transparent:true,blending:THREE.AdditiveBlending,depthWrite:false});
}
let pts=null;
{
  const pos=[],seed=[],fea=[],ord=[],vol=[],aux=[],shd=[];
  let n=0, guard=0;
  while(n<PTN&&guard++<PTN*4){
    // Marsaglia sphere direction → field surface
    let u=Math.random()*2-1, v=Math.random()*2-1, s=u*u+v*v;
    if(s>=1||s<1e-9) continue;
    const f=2*Math.sqrt(1-s);
    const dx=u*f, dy=v*f, dz=1-2*s;
    const p=headPoint(dx,dy,dz);
    const feat=featureWeight(p), sock=socketWeight(p);
    if(sock>0.55&&Math.random()<0.75) continue;      // the eye constructs own the sockets
    if(feat<0.3&&Math.random()<0.32) continue;       // shape density toward features
    // The animated LIP LINES own the mouth: static data points inside the
    // mouth corridor dim (like the sockets) so an emotional smile/frown —
    // which moves the lines away from the field's static lip bumps — reads
    // as one mouth, not a washed-out double one.
    const mouthW=(p.z>0?1:0)*bump(p.y,-23.2,4.4)*sstep(15,4,Math.abs(p.x));
    const interior=Math.random()<0.18;               // volumetric depth data
    const sc=interior?0.84+0.13*Math.random():1;
    pos.push(p.x*sc,p.y*sc,p.z*sc);
    seed.push(Math.random()); fea.push(feat);
    ord.push(feat>0.5?Math.random()*0.4:0.2+Math.random()*0.6);
    vol.push(interior?1:0); aux.push(Math.max(sock*0.7,mouthW*0.8));
    shd.push(interior?0.3:shadeAt(Math.atan2(dz,dx),Math.acos(Math.max(-1,Math.min(1,dy)))));
    n++;
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('aSeed',new THREE.Float32BufferAttribute(seed,1));
  g.setAttribute('aFeat',new THREE.Float32BufferAttribute(fea,1));
  g.setAttribute('aOrder',new THREE.Float32BufferAttribute(ord,1));
  g.setAttribute('aVol',new THREE.Float32BufferAttribute(vol,1));
  g.setAttribute('aAux',new THREE.Float32BufferAttribute(aux,1));
  g.setAttribute('aShade',new THREE.Float32BufferAttribute(shd,1));
  pts=new THREE.Points(g,makePtsMat()); pts.frustumCulled=false; rig.add(pts);
}

// ═══════════════════════════════════════════════════════════════════════════
// SKIN — the translucent quantum membrane (Mark XIV).  A single displaced
// surface over the same anatomy field: fresnel-rimmed so it reads as a
// membrane not a shell, subsurface-lit by the baked form shading, with data
// streams, pulsing vein filaments and a living micro-structure flowing
// beneath it.  Knits crown-to-neck during stage 5; clears the eye sockets;
// additive + FrontSide + depthWrite:false like every holographic layer.
// ═══════════════════════════════════════════════════════════════════════════
const SKIN_VERT=`
  uniform float uJaw,uTime,uSpeak,uBreathe;
  attribute float aShade,aSock,aFeat,aOrder;
  varying vec3 vN,vPos,vView;
  varying float vSh,vSock,vFeat,vOrd;
  `+JAW_GLSL+`
  void main(){
    vec3 p=jawDeform(position,uJaw);
    p.z*=uBreathe;
    // micro-muscle simulation: a slow living ripple, cheeks lift in speech
    float cheek=exp(-pow((position.y+8.)/14.,2.))*step(0.,position.z);
    float mm=sin(position.x*1.7+uTime*2.1)*sin(position.y*1.9-uTime*1.7)
            *sin(position.z*2.3+uTime*1.3);
    p+=normal*mm*(0.10+0.55*uSpeak*cheek);
    vN=normalize(normalMatrix*normal);
    vPos=position; vSh=aShade; vSock=aSock; vFeat=aFeat; vOrd=aOrder;
    vec4 mv=modelViewMatrix*vec4(p,1.);
    vView=-mv.xyz;
    gl_Position=projectionMatrix*mv;
  }`;
const SKIN_FRAG=`
  uniform float uTime,uBuild,uOpacity,uEnergy,uAmp,uFadeA,uFadeB;
  uniform vec3 uMid,uBright,uAccent;
  varying vec3 vN,vPos,vView;
  varying float vSh,vSock,vFeat,vOrd;
  float hash3(vec3 p){return fract(sin(dot(p,vec3(12.9898,78.233,45.164)))*43758.5453);}
  float vnoise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
    float a=hash3(i),b=hash3(i+vec3(1,0,0)),c=hash3(i+vec3(0,1,0)),d=hash3(i+vec3(1,1,0));
    float e=hash3(i+vec3(0,0,1)),g=hash3(i+vec3(1,0,1)),h=hash3(i+vec3(0,1,1)),k=hash3(i+vec3(1,1,1));
    return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),mix(mix(e,g,f.x),mix(h,k,f.x),f.y),f.z);}
  float fbm(vec3 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*vnoise(p);p*=2.13;a*=.55;}return v;}
  void main(){
    // crown-to-neck materialisation with a glowing knit front
    float vis=smoothstep(vOrd,vOrd+0.07,uBuild);
    if(vis<0.004) discard;
    float front=mix(0.22,1.0,smoothstep(-20.,14.,vPos.z));   // back-of-head dim
    float fade=1.0-smoothstep(uFadeA,uFadeB,-vPos.y);        // bottom dissolve
    vec3 V=normalize(vView), N=normalize(vN);
    float fres=pow(1.-abs(dot(N,V)),2.2);                    // membrane rim
    float t=uTime;
    vec3 q=vPos*0.055;
    // data flowing beneath the surface: two depths, two speeds — the deeper
    // layer is offset along the normal so the streams parallax under the rim
    float flowA=fbm(q*1.15+vec3(0.,-t*0.32,t*0.05));
    float flowB=fbm(q*2.3+N*0.35+vec3(t*0.05,-t*0.55,0.));
    float stream=smoothstep(0.55,0.9,flowA)*0.7+smoothstep(0.6,0.95,flowB)*0.5;
    // emissive data veins: thin ridged filaments pulsing with energy
    float vr=fbm(q*2.0+vec3(0.,t*0.06,0.));
    float vein=smoothstep(0.035,0.0,abs(vr-0.5))*(0.55+0.45*sin(t*2.4+vr*24.));
    // living micro-structure: faint fractal cell grid in the membrane itself
    float cells=smoothstep(0.92,1.0,sin(vPos.x*2.3)*sin(vPos.y*2.1)*sin(vPos.z*2.5)*0.5+0.5)*0.5;
    // subsurface glow: baked form shading stands in for scattered light
    float sss=vSh*vSh*(0.55+0.45*flowA);
    vec3 col=mix(uMid*0.55,uBright,sss)*0.85;
    col+=uAccent*(stream*(0.5+uAmp*0.6)+vein*(0.8+uEnergy)+cells*0.25);
    col+=uBright*fres*(0.75+0.35*sin(t*1.1));
    // the knit front glows as the membrane assembles
    float tip=exp(-pow((uBuild-vOrd)*22.,2.))*(1.-step(0.985,uBuild));
    col+=uAccent*tip*2.2;
    // TRANSLUCENT, not solid: fresnel dominates, the facing view stays a thin
    // veil (the data layers beneath must keep carrying the anatomy — a higher
    // floor here washes the face into a featureless blob; see module notes)
    float alpha=uOpacity*vis*front*fade*(1.-vSock*0.92)
      *(fres*0.95+0.06+stream*0.07+vein*0.09)*(0.22+0.78*vSh*vSh)*(1.+0.20*vFeat)
      +uOpacity*tip*0.8*vis*front*fade;
    if(alpha<0.004) discard;
    gl_FragColor=vec4(col,alpha);
  }`;
function makeSkinMat(opacity,fadeA,fadeB){
  return new THREE.ShaderMaterial({
    uniforms:{uJaw:{value:0},uTime:{value:0},uSpeak:{value:0},uBreathe:{value:1},
      uBuild:{value:0},uOpacity:{value:opacity},uEnergy:{value:0.4},uAmp:{value:0},
      uFadeA:{value:fadeA},uFadeB:{value:fadeB},
      uMid:{value:st.mid.clone()},uBright:{value:st.bright.clone()},uAccent:{value:st.accent.clone()}},
    vertexShader:SKIN_VERT,fragmentShader:SKIN_FRAG,
    transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,side:THREE.FrontSide});
}
// Generic membrane grid: sample(u01,v01) → {p,n,shade,sock,feat,ord}.
function buildSkinGeometry(SU,SV,sample){
  const pos=[],nor=[],shd=[],sock=[],fea=[],ord=[],idx=[];
  for(let i=0;i<=SV;i++)for(let j=0;j<=SU;j++){
    const s=sample(j/SU,i/SV);
    pos.push(s.p.x,s.p.y,s.p.z); nor.push(s.n.x,s.n.y,s.n.z);
    shd.push(s.shade); sock.push(s.sock); fea.push(s.feat); ord.push(s.ord);
  }
  for(let i=0;i<SV;i++)for(let j=0;j<SU;j++){
    const a=i*(SU+1)+j, b=a+SU+1;
    idx.push(a,b,a+1,b,b+1,a+1);
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('normal',new THREE.Float32BufferAttribute(nor,3));
  g.setAttribute('aShade',new THREE.Float32BufferAttribute(shd,1));
  g.setAttribute('aSock',new THREE.Float32BufferAttribute(sock,1));
  g.setAttribute('aFeat',new THREE.Float32BufferAttribute(fea,1));
  g.setAttribute('aOrder',new THREE.Float32BufferAttribute(ord,1));
  g.setIndex(idx);
  return g;
}
let skinHead=null;
{
  const geo=buildSkinGeometry(128,88,(u,v)=>{
    const th=u*Math.PI*2, phi=0.04+v*(Math.PI-0.08);
    const p=fieldRadius(th,phi), n=fieldNormal(th,phi);
    // knit order: crown-to-neck sweep with an organic ragged front
    const rag=hashOrd(u*97.13+v*57.77);
    return {p,n,shade:shadeAt(th,phi),sock:socketWeight(p),
      feat:featureWeight(p),ord:Math.min(0.93,0.03+0.82*v+0.06*rag)};
  });
  skinHead=new THREE.Mesh(geo,makeSkinMat(0.34,900,1000));
  skinHead.frustumCulled=false; rig.add(skinHead);
}
function hashOrd(x){return (Math.sin(x)*43758.5453)%1*0.5+0.5;}

// ═══════════════════════════════════════════════════════════════════════════
// EYES — digital constructs: shader iris (concentric data rings, radial
// fibres, rotating tick segments, glowing pupil rim), halo rings, digital
// shutter blink.  A black backing disk depth-occludes the skull behind each
// eye so the construct reads deep and the pupil stays truly dark.
// ═══════════════════════════════════════════════════════════════════════════
const eyes=new THREE.Group(); rig.add(eyes);
function makeEye(sign){
  const g=new THREE.Group();
  g.position.set(sign*EYE.x,EYE.y,plateZ(EYE.x,EYE.y)-4.2);
  const backing=new THREE.Mesh(new THREE.CircleGeometry(6.6,32),
    new THREE.MeshBasicMaterial({color:0x000000}));
  backing.position.z=-0.35; g.add(backing);
  const uni={uTime:{value:0},uPupil:{value:0.34},uData:{value:0},uLid:{value:0},uBoot:{value:0},
    uAccent:{value:st.accent.clone()},uBright:{value:st.bright.clone()}};
  const iris=new THREE.Mesh(new THREE.CircleGeometry(5.2,48),new THREE.ShaderMaterial({
    uniforms:uni,transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,
    vertexShader:`varying vec2 vUv;void main(){vUv=uv*2.-1.;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader:`
      uniform float uTime,uPupil,uData,uLid,uBoot; uniform vec3 uAccent,uBright; varying vec2 vUv;
      float hash(float n){return fract(sin(n)*43758.5453);}
      void main(){
        float r=length(vUv); if(r>1.0) discard;
        float lidTop=1.0-1.9*uLid;                     // digital shutter blink
        float above=vUv.y-lidTop;
        if(above>0.07) discard;
        float a=atan(vUv.y,vUv.x);
        vec3 col=vec3(0.);
        // radial data fibres
        float fib=0.55+0.45*sin(a*64.+hash(floor(a*64.))*3.14);
        float fibMask=smoothstep(0.30,0.42,r)*smoothstep(0.98,0.80,r);
        col+=mix(uAccent*0.45,uBright,fib*0.6)*fibMask*(0.5+0.5*fib);
        // concentric construct rings
        col+=uAccent*smoothstep(0.014,0.0,abs(r-0.985))*1.2;
        col+=uAccent*smoothstep(0.011,0.0,abs(r-0.70))*0.8;
        col+=uAccent*smoothstep(0.009,0.0,abs(r-0.46))*0.7;
        // rotating tick segments — visible data activity
        float segs=step(0.86,fract(a*2.55+uTime*0.55))*smoothstep(0.05,0.015,abs(r-0.78));
        float seg2=step(0.90,fract(-a*3.8+uTime*0.9))*smoothstep(0.04,0.012,abs(r-0.58));
        col+=uBright*(segs+seg2)*(0.6+uData*1.6);
        // pupil: dark core, glowing rim, bright centre spark
        col*=smoothstep(uPupil-0.02,uPupil+0.06,r);
        col+=uAccent*smoothstep(0.035,0.0,abs(r-uPupil))*(1.2+uData*0.8);
        col+=uBright*smoothstep(0.10,0.0,r)*(0.9+0.4*sin(uTime*2.2));
        // the shutter itself is a glowing data line — a blink never goes dark
        if(above>0.0) col=uAccent*1.8*(1.-above/0.07)*step(0.02,uLid);
        else if(uLid>0.02) col+=uAccent*smoothstep(0.05,0.0,-above)*1.2;
        col*=uBoot*1.35;
        float alpha=clamp(max(max(col.r,col.g),col.b),0.,1.);
        gl_FragColor=vec4(col,alpha);
      }`}));
  g.add(iris);
  const haloMat=new THREE.MeshBasicMaterial({color:0xff2a44,transparent:true,opacity:0,
    blending:THREE.AdditiveBlending,depthWrite:false});
  const halo1=new THREE.Mesh(new THREE.TorusGeometry(6.4,0.10,6,48),haloMat);
  const halo2=new THREE.Mesh(new THREE.TorusGeometry(7.2,0.07,6,48),haloMat.clone());
  halo2.rotation.x=0.35; g.add(halo1); g.add(halo2);
  // subtle white highlight — the only non-red element of the being
  const catchlight=new THREE.Mesh(new THREE.CircleGeometry(0.4,10),
    new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0,depthTest:false}));
  catchlight.position.set(-0.85,0.95,0.2); g.add(catchlight);
  g.userData={uni,halo1,halo2,catchlight,sign};
  eyes.add(g); return g;
}
const eyeL=makeEye(1), eyeR=makeEye(-1);
// Brows — holographic light arcs hugging the brow ridge, emotion-rigged.
const browMat=new THREE.MeshBasicMaterial({color:0xff3550,transparent:true,opacity:0,
  blending:THREE.AdditiveBlending,depthWrite:false});
function makeBrow(sign){
  const y0=EYE.y+7.6, pts2=[];
  for(let i=0;i<=8;i++){
    const t=i/8;
    const x=sign*(5.5+18.0*t);
    const y=y0+2.4*Math.sin(Math.PI*Math.min(1,t/0.72))-1.4*t;
    pts2.push(new THREE.Vector3(x,y,plateZ(x,y)+3.0-1.6*t));
  }
  const curve=new THREE.CatmullRomCurve3(pts2,false,'catmullrom',0.3);
  const brow=new THREE.Mesh(new THREE.TubeGeometry(curve,20,0.7,6),browMat.clone());
  brow.userData={y0,sign}; eyes.add(brow); return brow;
}
const browL=makeBrow(1), browR=makeBrow(-1);

// ═══════════════════════════════════════════════════════════════════════════
// MOUTH — feature-line lips: upper contour, seam, lower contour.  CPU-updated
// every frame from the same field constants; the jaw and pseudo-visemes shape
// the opening, mouth_curve lifts or drops the corners.
// ═══════════════════════════════════════════════════════════════════════════
const LIPN=33;
const lipMats={
  up:new THREE.LineBasicMaterial({color:0xff5468,transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false}),
  seam:new THREE.LineBasicMaterial({color:0xff2a44,transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false}),
  low:new THREE.LineBasicMaterial({color:0xff5468,transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false})};
function makeLipLine(mat){
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(LIPN*3),3));
  const l=new THREE.Line(g,mat); l.frustumCulled=false; rig.add(l); return l;
}
const lipUp=makeLipLine(lipMats.up), lipSeam=makeLipLine(lipMats.seam), lipLow=makeLipLine(lipMats.low);
// Lip glow points — 1-px additive lines vanish against the point cloud, so
// the same three curves also carry sized glow points: THIS is what makes a
// smile or a pressed frown readable at a glance.
const lipPtsMat=new THREE.PointsMaterial({color:0xff2a44,size:2.4,transparent:true,opacity:0,
  blending:THREE.AdditiveBlending,depthWrite:false,sizeAttenuation:true});
const lipPts=(()=>{const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(LIPN*3*3),3));
  const o=new THREE.Points(g,lipPtsMat); o.frustumCulled=false; rig.add(o); return o;})();
function updateLips(open,curve,tension,round){
  const au=lipUp.geometry.attributes.position.array,
        as=lipSeam.geometry.attributes.position.array,
        al=lipLow.geometry.attributes.position.array;
  // smiles widen the mouth a touch, tension and rounding narrow it
  const w=12.8*(1+0.10*curve)*(1-0.18*tension)*(1-0.12*round*open);
  for(let i=0;i<LIPN;i++){
    const t=i/(LIPN-1)*2-1, ax=Math.abs(t);
    const x=t*w;
    // The emotional curve sweeps the WHOLE mouth — gentle at the philtrum,
    // strongest at the corners — so a smile or a frown reads at a glance
    // (the old rig moved only the outer 20% by ~2 units: invisible).
    const lift=curve*(0.9*(1-ax)+5.2*sstep(0.30,1,ax));
    const cornerY=MOUTH_Y+curve*5.6*sstep(0.55,1,ax);
    const cornerMix=sstep(0.72,1,ax);
    // pressed / compressed lips under tension: both lips close on the seam
    const press=tension*(1-sstep(0.7,1,ax));
    // cupid-bow upper lip
    let yu=lerp(-20.5-0.7*bump(ax,0.16,0.13)+lift*0.75-press*0.55,cornerY,cornerMix);
    let ys=lerp(MOUTH_Y+lift,cornerY,cornerMix);
    let yl=lerp(-26.0+lift*0.9+press*1.15,cornerY,cornerMix);
    const drop=open*7.5*(1-sstep(0.6,1,ax));
    ys-=drop*0.55; yl-=drop;
    au[i*3]=x; au[i*3+1]=yu; au[i*3+2]=plateZ(x,yu)+3.6;
    as[i*3]=x; as[i*3+1]=ys; as[i*3+2]=plateZ(x,Math.max(ys,-40))+3.2;
    al[i*3]=x; al[i*3+1]=yl; al[i*3+2]=plateZ(x,Math.max(yl,-40))+3.8;
  }
  lipUp.geometry.attributes.position.needsUpdate=true;
  lipSeam.geometry.attributes.position.needsUpdate=true;
  lipLow.geometry.attributes.position.needsUpdate=true;
  // mirror the three curves into the sized glow points
  const lp=lipPts.geometry.attributes.position.array;
  lp.set(au,0); lp.set(as,LIPN*3); lp.set(al,LIPN*6);
  lipPts.geometry.attributes.position.needsUpdate=true;
}
updateLips(0,0.05,0,0);

// ═══════════════════════════════════════════════════════════════════════════
// SHELL — the signature quantum-cube swarm (single instanced draw call).
// Orb sphere ⇄ head surface; feature cubes assemble first (stage 3).
// ═══════════════════════════════════════════════════════════════════════════
let homes=[], mesh=null, N=0;
const dummy=new THREE.Object3D(), col=new THREE.Color(), Zc=new THREE.Vector3(0,0,1);
const qA=new THREE.Quaternion();
{
  const GRs=Math.PI*(3-Math.sqrt(5));
  for(let i=0;i<COUNT*2 && homes.length<COUNT;i++){
    const yy=1-((i%COUNT)/(COUNT-1))*2, rr=Math.sqrt(Math.max(0,1-yy*yy)), th=GRs*i;
    let dx=Math.cos(th)*rr, dy=yy, dz=Math.sin(th)*rr;
    const p=headPoint(dx,dy,dz);
    const feat=featureWeight(p), sock=socketWeight(p);
    const mouthW=(p.z>0?1:0)*bump(p.y,-23.2,4.4)*sstep(15,4,Math.abs(p.x));
    let nx=p.x/(HEAD.a*HEAD.a), ny=p.y/(HEAD.b*HEAD.b), nz=p.z/(HEAD.c*HEAD.c);
    const nl=Math.hypot(nx,ny,nz)||1; nx/=nl; ny/=nl; nz/=nl;
    const shade=Math.max(0,Math.min(1,0.16+0.84*Math.max(0,nx*LIGHT.x+ny*LIGHT.y+nz*LIGHT.z)));
    const yn=(p.y+HEAD.b)/(2*HEAD.b);
    const j=homes.length;
    const oyy=1-(j/(COUNT-1))*2, orr=Math.sqrt(Math.max(0,1-oyy*oyy)), oth=GRs*j;
    homes.push({x:p.x,y:p.y,z:p.z,nx,ny,nz,yn,feat,sock,mouth:mouthW,shade,
      phase:Math.random()*6.283, mote:Math.random()<0.03, dis:0, disT:0,
      arrive:feat>0.45?Math.random()*0.35:0.3+Math.random()*0.65,  // features first
      swirl:Math.random()*2-1,
      quat:new THREE.Quaternion().setFromUnitVectors(Zc,new THREE.Vector3(nx,ny,nz)),
      ox:Math.cos(oth)*orr*ORB_R, oy:oyy*ORB_R, oz:Math.sin(oth)*orr*ORB_R,
      oquat:new THREE.Quaternion().setFromUnitVectors(Zc,new THREE.Vector3(Math.cos(oth)*orr,oyy,Math.sin(oth)*orr))});
  }
  N=homes.length;
  const cube=new THREE.BoxGeometry(1.45,1.45,1.45);
  const mat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xaa1c34,
    emissiveIntensity:0.5,metalness:0.1,roughness:0.65,transparent:true,opacity:0.95});
  mesh=new THREE.InstancedMesh(cube,mat,N); mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.setColorAt(0,new THREE.Color(1,1,1)); mesh.userData.mat=mat; mesh.frustumCulled=false;
  rig.add(mesh);
}

// Orbiters — thought streams circling the head while he speaks/thinks.
const ORBN=96;
let orbiters=null;
{
  const mat=new THREE.MeshBasicMaterial({color:0xff5468,transparent:true,opacity:0.8,
    blending:THREE.AdditiveBlending,depthWrite:false});
  orbiters=new THREE.InstancedMesh(new THREE.BoxGeometry(1.1,1.1,1.1),mat,ORBN);
  orbiters.instanceMatrix.setUsage(THREE.DynamicDrawUsage); orbiters.frustumCulled=false;
  orbiters.userData.bands=[...Array(ORBN)].map((_,i)=>({
    band:i%3, a0:Math.random()*6.283, speed:0.5+Math.random()*0.7,
    rx:66+8*(i%3), rz:52+7*((i+1)%3), tilt:(i%3-1)*0.5+(Math.random()-0.5)*0.2,
    scale:0.6+Math.random()*0.9}));
  rig.add(orbiters);
}

// ═══════════════════════════════════════════════════════════════════════════
// ORB CORE — pulsating contained star: turbulent shader sphere + geometry.
// ═══════════════════════════════════════════════════════════════════════════
const coreUni={uTime:{value:0},uHeat:{value:0.5},uAccent:{value:st.accent.clone()},
  uBright:{value:st.bright.clone()}};
const core=new THREE.Mesh(new THREE.SphereGeometry(30,48,36),new THREE.ShaderMaterial({
  uniforms:coreUni,transparent:true,
  vertexShader:`varying vec3 vPos;varying vec3 vN;void main(){vPos=position;vN=normalize(normalMatrix*normal);
    gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader:`
    uniform float uTime,uHeat; uniform vec3 uAccent,uBright; varying vec3 vPos; varying vec3 vN;
    float hash3(vec3 p){return fract(sin(dot(p,vec3(12.9898,78.233,45.164)))*43758.5453);}
    float vnoise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
      float a=hash3(i),b=hash3(i+vec3(1,0,0)),c=hash3(i+vec3(0,1,0)),d=hash3(i+vec3(1,1,0));
      float e=hash3(i+vec3(0,0,1)),g=hash3(i+vec3(1,0,1)),h=hash3(i+vec3(0,1,1)),k=hash3(i+vec3(1,1,1));
      return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),mix(mix(e,g,f.x),mix(h,k,f.x),f.y),f.z);}
    float fbm(vec3 p){float v=0.,a=.5;for(int i=0;i<5;i++){v+=a*vnoise(p);p*=2.07;a*=.55;}return v;}
    void main(){
      vec3 p=vPos*0.06;
      float t=uTime*0.22;
      float n=fbm(p+vec3(t*0.6,t*0.3,-t*0.4)+fbm(p*1.7+vec3(-t,t*0.5,t*0.2)));
      float fres=pow(1.-abs(vN.z),1.8);
      vec3 col=mix(uAccent*0.35,uBright*1.6,smoothstep(0.35,0.85,n));
      col+=uBright*fres*0.9+uAccent*pow(n,3.)*2.2*uHeat;
      float alpha=0.85*smoothstep(0.15,0.55,n)+fres*0.5;
      gl_FragColor=vec4(col*(0.7+uHeat),min(1.,alpha));
    }`}));
core.frustumCulled=false; rig.add(core);
const ringMat=new THREE.MeshBasicMaterial({color:0xff2a44,transparent:true,opacity:0.5,
  blending:THREE.AdditiveBlending,depthWrite:false});
const ring1=new THREE.Mesh(new THREE.TorusGeometry(60,0.5,8,72),ringMat);
const ring2=new THREE.Mesh(new THREE.TorusGeometry(67,0.35,8,72),ringMat.clone());
const latt=new THREE.Mesh(new THREE.IcosahedronGeometry(57,1),
  new THREE.MeshBasicMaterial({color:0xff4d68,wireframe:true,transparent:true,opacity:0.28,
    blending:THREE.AdditiveBlending,depthWrite:false}));
rig.add(ring1); rig.add(ring2); rig.add(latt);

// Scan ring — a data sweep across the head during detailing + processing.
const scanRing=new THREE.Mesh(new THREE.TorusGeometry(1,0.16,6,72),
  new THREE.MeshBasicMaterial({color:0xff2a44,transparent:true,opacity:0,
    blending:THREE.AdditiveBlending,depthWrite:false}));
scanRing.rotation.x=Math.PI/2; scanRing.scale.set(46,52,1); scanRing.frustumCulled=false;
rig.add(scanRing);

// ═══════════════════════════════════════════════════════════════════════════
// TORSO — parametric wireframe bust (stage 7), dissolving below the chest.
// Own material instances: uJaw stays 0 and the bottom fade is active.
// ═══════════════════════════════════════════════════════════════════════════
function torsoPoint(u,v){
  const sh=sstep(0.05,0.32,v), tapr=sstep(0.62,1.0,v);
  const ax=15+56*sh-14*tapr, az=13+16*sh-4*tapr;
  const cu=Math.cos(u), su=Math.sin(u), n=2.7;
  let x=ax*Math.sign(cu)*Math.abs(cu)**(2/n);
  let z=az*Math.sign(su)*Math.abs(su)**(2/n);
  if(z>0) z+=3.5*sstep(0.35,0.75,v);                 // chest plates
  const nb=1-sstep(0.0,0.12,v);
  if(nb>0){x=lerp(x,14*cu,nb); z=lerp(z,12*su,nb);}  // blend into the neck
  const y=-56-v*92-7*sh*Math.abs(cu)**1.6;           // deltoid slope
  return {x,y,z};
}
function torsoNormal(u,v){
  const e=0.01, p0=torsoPoint(u,v), pu=torsoPoint(u+e,v), pv=torsoPoint(u,Math.min(0.99,v+e));
  const t1x=pu.x-p0.x,t1y=pu.y-p0.y,t1z=pu.z-p0.z;
  const t2x=pv.x-p0.x,t2y=pv.y-p0.y,t2z=pv.z-p0.z;
  let nx=t1y*t2z-t1z*t2y, ny=t1z*t2x-t1x*t2z, nz=t1x*t2y-t1y*t2x;
  const l=Math.hypot(nx,ny,nz)||1;
  const s=(nx*p0.x+nz*p0.z)<0?-1:1;
  return {x:s*nx/l,y:s*ny/l,z:s*nz/l};
}
function torsoShade(u,v){
  const n=torsoNormal(u,v);
  return Math.max(0,Math.min(1,0.16+0.84*Math.max(0,n.x*LIGHT.x+n.y*LIGHT.y+n.z*LIGHT.z)));
}
let torsoWire=null, torsoPts=null;
{
  const pos=[],ord=[],fea=[],aux=[],shd=[];
  const push=(u,v,o,f)=>{const p=torsoPoint(u,v);
    pos.push(p.x,p.y,p.z);ord.push(Math.min(0.93,o));fea.push(f);aux.push(0);shd.push(torsoShade(u,v));};
  for(let ri=0;ri<9;ri++){                           // rings, top-down
    const v=0.04+ri/8*0.92, base=0.05+0.55*(ri/8);
    const clav=bump(v,0.28,0.1)*0.8;
    for(let j=0;j<72;j++){
      push(j/72*Math.PI*2,v,base+0.06*(j/72),0.25+clav);
      push((j+1)/72*Math.PI*2,v,base+0.06*((j+1)/72),0.25+clav);
    }
  }
  for(let mi=0;mi<14;mi++){                          // meridians
    const u=mi/14*Math.PI*2, base=0.3+0.5*(mi/14);
    for(let i=0;i<26;i++){
      push(u,0.04+i/26*0.92,base+0.08*(i/26),0.3);
      push(u,0.04+(i+1)/26*0.92,base+0.08*((i+1)/26),0.3);
    }
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('aOrder',new THREE.Float32BufferAttribute(ord,1));
  g.setAttribute('aFeat',new THREE.Float32BufferAttribute(fea,1));
  g.setAttribute('aAux',new THREE.Float32BufferAttribute(aux,1));
  g.setAttribute('aShade',new THREE.Float32BufferAttribute(shd,1));
  torsoWire=new THREE.LineSegments(g,makeWireMat(0.22,118,146));
  torsoWire.frustumCulled=false; torsoRig.add(torsoWire);
  // sparse torso data points
  const p2=[],s2=[],f2=[],o2=[],v2=[],a2=[],sh2=[];
  for(let i=0;i<1800;i++){
    const u=Math.random()*Math.PI*2, v=0.04+Math.random()*0.92;
    const p=torsoPoint(u,v);
    p2.push(p.x,p.y,p.z); s2.push(Math.random()); f2.push(0.3);
    o2.push(Math.random()*0.8); v2.push(0); a2.push(0); sh2.push(torsoShade(u,v));
  }
  const g2=new THREE.BufferGeometry();
  g2.setAttribute('position',new THREE.Float32BufferAttribute(p2,3));
  g2.setAttribute('aSeed',new THREE.Float32BufferAttribute(s2,1));
  g2.setAttribute('aFeat',new THREE.Float32BufferAttribute(f2,1));
  g2.setAttribute('aOrder',new THREE.Float32BufferAttribute(o2,1));
  g2.setAttribute('aVol',new THREE.Float32BufferAttribute(v2,1));
  g2.setAttribute('aAux',new THREE.Float32BufferAttribute(a2,1));
  g2.setAttribute('aShade',new THREE.Float32BufferAttribute(sh2,1));
  torsoPts=new THREE.Points(g2,makePtsMat()); torsoPts.frustumCulled=false; torsoRig.add(torsoPts);
}
// Torso membrane — same skin shader, own material instance: uJaw stays 0
// (the jaw deform must never rotate the chest) and the bottom fade matches
// the torso wires so the being dissolves below the chest.
let skinTorso=null;
{
  const geo=buildSkinGeometry(96,40,(u,v)=>{
    const uu=u*Math.PI*2, vv=0.04+v*0.92;
    const p=torsoPoint(uu,vv), n=torsoNormal(uu,vv);
    const rag=hashOrd(u*83.7+v*41.3);
    return {p,n,shade:torsoShade(uu,vv),sock:0,feat:0.25,
      ord:Math.min(0.93,0.04+0.80*v+0.06*rag)};
  });
  skinTorso=new THREE.Mesh(geo,makeSkinMat(0.20,118,146));
  skinTorso.frustumCulled=false; torsoRig.add(skinTorso);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPACE — the quantum environment: grid floor, drifting data cubes, haze.
// ═══════════════════════════════════════════════════════════════════════════
{
  const pos=[];
  for(let x=-600;x<=600;x+=40){pos.push(x,-160,-80,x,-160,-1400);}
  for(let z=-80;z>=-1400;z-=40){pos.push(-600,-160,z,600,-160,z);}
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  const grid=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0x8a1220,
    transparent:true,opacity:0.07,blending:THREE.AdditiveBlending,depthWrite:false}));
  grid.frustumCulled=false; scene.add(grid);
}
let bgCubes=null;
{
  const BG=110;
  bgCubes=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),
    new THREE.MeshBasicMaterial({color:0xff2a44,transparent:true,opacity:0.10,
      blending:THREE.AdditiveBlending,depthWrite:false}),BG);
  const d=new THREE.Object3D();
  for(let i=0;i<BG;i++){
    d.position.set((Math.random()-0.5)*840,(Math.random()-0.5)*440+40,-180-Math.random()*720);
    d.scale.setScalar(1.5+Math.random()*5.5);
    d.rotation.set(Math.random()*3,Math.random()*3,0);
    d.updateMatrix(); bgCubes.setMatrixAt(i,d.matrix);
  }
  bgCubes.frustumCulled=false; scene.add(bgCubes);
}
const hazeUni={uHeat:{value:0.5},uAccent:{value:st.accent.clone()}};
const haze=new THREE.Mesh(new THREE.PlaneGeometry(1000,760),new THREE.ShaderMaterial({
  uniforms:hazeUni,transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,
  vertexShader:`varying vec2 vUv;void main(){vUv=uv*2.-1.;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader:`uniform float uHeat;uniform vec3 uAccent;varying vec2 vUv;
    void main(){float r=length(vUv*vec2(1.,1.25));
      gl_FragColor=vec4(uAccent*exp(-r*3.0)*(0.055+0.055*uHeat),1.);}`}));
haze.position.set(0,10,-330); haze.frustumCulled=false; scene.add(haze);

// Ambient particles — quantum weather around the being.
let parts=[], pgeo=null, ppos=null, pmat=null;
{
  const PC=520; pgeo=new THREE.BufferGeometry(); ppos=new Float32Array(PC*3);
  for(let i=0;i<PC;i++){const h=homes[(Math.random()*N)|0];
    parts.push({x:h.x,y:h.y,z:h.z,vx:0,vy:0,vz:0,life:Math.random()});
    ppos[i*3]=h.x;ppos[i*3+1]=h.y;ppos[i*3+2]=h.z;}
  pgeo.setAttribute('position',new THREE.BufferAttribute(ppos,3));
  pmat=new THREE.PointsMaterial({color:0xff8fa0,size:1.2,transparent:true,opacity:0.5,
    blending:THREE.AdditiveBlending,depthWrite:false});
  scene.add(new THREE.Points(pgeo,pmat));
}
// Reaching here means Three.js loaded and the scene is built — disarm the
// offline watchdog before it can fire a false "no internet" message.
{try{window.__orionFaceBooted&&window.__orionFaceBooted();}catch(_){}
 const l=document.getElementById('loading'); if(l)l.remove();}

// ═══════════════════════════════════════════════════════════════════════════
// ANIMATION — 7-stage materialisation timeline, speech, gaze, behaviour.
// ═══════════════════════════════════════════════════════════════════════════
let clock=0, blinkT=2.5, blinking=0, blinkP=0, blinkAmt=0;
let sacT=1.5, sacX=0, sacY=0, curX=0, curY=0, dataT=1.0, dataV=0;
let twitchT=7, twitch=0;
let speakEnv=0, visW=0, visR=0, nodP=0, quality=1;
let frameEMA=16, colorTick=0, lastT=performance.now();
window.orionStep=n=>{for(let i=0;i<(n|0);i++)step(0.016); composer.render();};

// Stage windows over morphT (scrubbed, reversible, interruption-safe):
//  1 destab 0–.12 · 2 explosion .10–.30 · 3 structure .26–.52 ·
//  4 framework .45–.70 · 5 detailing .62–.88 · 6 eyes .86–.97 · 7 torso .93–1
const stage=(a,b)=>sstep(a,b,morphT);
// eye ignition: irregular flicker while booting, steady once conscious
const ignite=x=>x<=0?0:(x>=1?1:x*(0.45+0.55*Math.abs(Math.sin(x*25+Math.sin(x*47)))));

function step(dt){
  clock+=dt*st.speed;
  // Materialisation pace: full transform ≈ 2.4 s either direction.
  morphT+=Math.sign(morphTarget-morphT)*Math.min(Math.abs(morphTarget-morphT),dt/2.4);
  for(const k of ['density','glow','pulse','speed','pvel','turb','brow','eyeH','eyeW','mouthCurve','tension'])
    st[k]=lerp(st[k],tg[k],0.06);
  st.amp=lerp(st.amp,tg.amp,0.35);
  st.specLow=lerp(st.specLow,tg.specLow,0.35); st.specMid=lerp(st.specMid,tg.specMid,0.35); st.specHigh=lerp(st.specHigh,tg.specHigh,0.35);
  st.dark.lerp(tg.dark,0.05); st.mid.lerp(tg.mid,0.05); st.bright.lerp(tg.bright,0.05); st.accent.lerp(tg.accent,0.05);
  processing=lerp(processing,(mode==='PROCESSING')?1:0,0.08);
  listen=lerp(listen,(mode==='LISTENING')?1:0,0.08);
  // named-emotion expression extras ease like everything else on this face
  const exT=EXPR[emoName]||EXPR.neutral;
  for(const k in exState) exState[k]=lerp(exState[k],(k in exT)?exT[k]:(k==='blink'?1:0),0.05);

  const sDestab=stage(0.00,0.12), sBurst=stage(0.10,0.30), sStruct=stage(0.26,0.52);
  const sFrame=stage(0.45,0.70), sDetail=stage(0.62,0.88), sSkin=stage(0.66,0.92);
  const sEyes=stage(0.86,0.97), sTorso=stage(0.93,1.0);
  const face=morphT;
  window.__dbg={morphT,fps:Math.round(1000/Math.max(1,frameEMA)),
    stages:{destab:sDestab,burst:sBurst,struct:sStruct,frame:sFrame,detail:sDetail,skin:sSkin,eyes:sEyes,torso:sTorso}};

  // ── speech envelope → jaw, spectral visemes, energy ───────────────────────
  const rise=st.amp>speakEnv?0.55:0.10;
  speakEnv=lerp(speakEnv,st.amp,rise);
  // Real spectral visemes (Mark X.14): low+mid band energy (open vowels —
  // "ah"/"oh") widens the jaw; high band alone (sibilants/fricatives —
  // "s"/"f"/"sh") pulls it back toward narrow regardless of loudness. This
  // replaces a fixed sine-wave rhythm that moved the same way no matter what
  // was being said.
  const openness=Math.min(1,st.specLow*0.7+st.specMid*0.5);
  const sibilance=Math.min(1,st.specHigh*1.2);
  visW=lerp(visW,Math.max(0,openness-sibilance*0.6),0.3);   // wide/narrow
  visR=lerp(visR,st.specLow,0.3);                             // round — low band = rounded back vowels
  // resting lip parting (happy/excited/alert) underlays the speech jaw
  const jawOpen=Math.min(1,Math.max(speakEnv*(0.55+0.45*visW),exState.part))*sEyes;
  // Debug/verification surface only — window.__dbg is the documented
  // inspection contract (see orionStep/orionSnap), extended additively.
  Object.assign(window.__dbg,{visW,visR,jawOpen,
    spec:{low:st.specLow,mid:st.specMid,high:st.specHigh}});
  const energy=0.14+st.glow*0.34+speakEnv*0.32+processing*0.28;

  // ── adaptive quality governor ─────────────────────────────────────────────
  // Judged against the rate we ASKED for, not a fixed 26 ms. That constant
  // was calibrated for a 60 fps target (16.7*1.55), and once the frame-rate
  // governor below settles the face to 30 fps on purpose, a fixed threshold
  // reads the deliberate 33.3 ms as "we are struggling" and halves the point
  // density for the whole idle - so the face visibly thinned every time
  // ORION stopped talking and thickened again when he started. Scaling with
  // the target keeps the original 60 fps behaviour exactly and stops idling
  // from being mistaken for overload.
  quality=lerp(quality,frameEMA>frameTargetMs*1.55?0.55:1,0.02);

  // ── holographic layer uniforms ────────────────────────────────────────────
  const uDensity=Math.max(0.3,Math.min(1,0.75*st.density))*quality;
  const pu=pts.material.uniforms;
  pu.uTime.value=clock; pu.uFill.value=sDetail; pu.uStruct.value=sStruct;
  pu.uDensity.value=uDensity; pu.uJaw.value=jawOpen; pu.uAmp.value=st.amp;
  pu.uSpeak.value=Math.max(speakEnv,exState.cheek*0.5)*sEyes; pu.uEnergy.value=energy;
  pu.uMid.value.copy(st.mid); pu.uBright.value.copy(st.bright); pu.uAccent.value.copy(st.accent);
  pts.visible=(sStruct>0.02||sDetail>0.02);
  for(const [w,build,op] of [[wireMain,sFrame,0.85],[wireFine,sDetail,0.42]]){
    const u=w.material.uniforms;
    u.uBuild.value=build; u.uTime.value=clock; u.uJaw.value=jawOpen;
    u.uEnergy.value=energy; u.uOpacity.value=op*(1+st.amp*0.25);
    u.uMid.value.copy(st.mid); u.uBright.value.copy(st.bright); u.uAccent.value.copy(st.accent);
    w.visible=build>0.01;
  }
  {
    const u=torsoWire.material.uniforms;
    u.uBuild.value=sTorso; u.uTime.value=clock; u.uEnergy.value=energy*0.8;
    u.uOpacity.value=0.38;
    u.uMid.value.copy(st.mid); u.uBright.value.copy(st.bright); u.uAccent.value.copy(st.accent);
    torsoWire.visible=sTorso>0.01;
    const tp=torsoPts.material.uniforms;
    tp.uTime.value=clock; tp.uFill.value=sTorso; tp.uDensity.value=uDensity*0.8;
    tp.uAmp.value=st.amp; tp.uEnergy.value=energy*0.8; tp.uPx.value=renderer.getPixelRatio();
    tp.uMid.value.copy(st.mid); tp.uBright.value.copy(st.bright); tp.uAccent.value.copy(st.accent);
    torsoPts.visible=sTorso>0.01;
  }

  // ── orb core + quantum geometry ───────────────────────────────────────────
  const orbLife=1-stage(0.10,0.40);
  const breathe=1+0.035*Math.sin(clock*1.05)+0.012*Math.sin(clock*2.7);
  core.visible=orbLife>0.02;
  if(core.visible){
    core.scale.setScalar(orbLife*breathe*(1+sDestab*0.22));
    coreUni.uTime.value=clock*(1+sDestab*2.5);
    coreUni.uHeat.value=0.45+st.glow*0.5+sDestab*1.4+st.amp*0.4;
    coreUni.uAccent.value.copy(st.accent); coreUni.uBright.value.copy(st.bright);
  }
  ring1.visible=ring2.visible=latt.visible=core.visible;
  if(latt.visible){
    const spin=1+sDestab*5;
    ring1.rotation.x=1.1+clock*0.5*spin; ring1.rotation.y=clock*0.33*spin;
    ring2.rotation.x=-0.6-clock*0.41*spin; ring2.rotation.z=clock*0.27*spin;
    latt.rotation.y=clock*0.21*spin; latt.rotation.x=Math.sin(clock*0.3)*0.4;
    const ro=orbLife*(0.35+sDestab*0.5+st.amp*0.2);
    ring1.material.opacity=ro; ring2.material.opacity=ro*0.8; latt.material.opacity=orbLife*0.26;
    ring1.scale.setScalar(orbLife*breathe); ring2.scale.setScalar(orbLife*breathe);
    latt.scale.setScalar(orbLife*breathe);
    ring1.material.color.copy(st.accent); ring2.material.color.copy(st.accent);
  }

  // ── skin membranes: knit-in, speech-reactive under-skin data flow ─────────
  // (after the orb block: `breathe` is const-declared there)
  for(const [s,build,jaw] of [[skinHead,sSkin,jawOpen],[skinTorso,sTorso,0]]){
    const u=s.material.uniforms;
    u.uTime.value=clock; u.uBuild.value=build; u.uJaw.value=jaw;
    u.uSpeak.value=Math.max(speakEnv,exState.cheek*0.5)*sEyes; u.uBreathe.value=1+(breathe-1)*face;
    u.uEnergy.value=energy; u.uAmp.value=st.amp;
    u.uMid.value.copy(st.mid); u.uBright.value.copy(st.bright); u.uAccent.value.copy(st.accent);
    s.visible=build>0.01;
  }

  // ── scan ring: detailing sweep + processing sweeps at full presence ───────
  {
    const sweepP=(clock*0.35)%1;
    const act=sDetail*(1-sEyes)+processing*face*0.6;
    scanRing.visible=act>0.03;
    if(scanRing.visible){
      scanRing.position.y=-60+120*sweepP;
      scanRing.material.opacity=0.45*act*Math.sin(Math.PI*sweepP);
      scanRing.material.color.copy(st.accent);
    }
  }

  // ── blink + gaze + eye constructs ─────────────────────────────────────────
  blinkT-=dt; if(blinkT<=0){blinking=1;blinkT=(2.6+Math.random()*4.2)*Math.max(0.4,exState.blink);}
  if(blinking){blinkP+=0.22;blinkAmt=Math.sin(Math.min(Math.PI,blinkP));if(blinkP>=Math.PI){blinking=0;blinkP=0;}} else blinkAmt*=0.5;
  const sacPace=listen>0.5?0.7:1.0;                  // listening: livelier tracking
  sacT-=dt/sacPace; if(sacT<=0){sacT=0.9+Math.random()*2.6; sacX=(Math.random()-0.5)*2.0; sacY=(Math.random()-0.5)*1.2;
    if(Math.random()<0.55){sacX*=0.25;sacY*=0.25;}}                    // mostly eye contact
  curX=lerp(curX,sacX,0.14); curY=lerp(curY,sacY,0.14);
  dataT-=dt; if(dataT<=0){dataT=3+Math.random()*7; dataV=1;}
  dataV*=0.94;
  twitchT-=dt; if(twitchT<=0){twitchT=6+Math.random()*9; twitch=1;}    // micro-expression
  twitch*=0.92;
  // emotional pupil: joy dilates, threat/focus contracts (over the base)
  const pupil=Math.max(0.14,0.34-speakEnv*0.04-processing*0.09+(1-st.eyeH)*0.05
    +exState.pupil*0.12);
  const open=Math.max(0.05,st.eyeH-blinkAmt);
  const boot=ignite(sEyes);
  eyes.visible=sEyes>0.02;
  for(const e of [eyeL,eyeR]){
    const u=e.userData;
    u.uni.uTime.value=clock; u.uni.uPupil.value=pupil;
    u.uni.uData.value=Math.min(1.5,dataV*(0.5+processing)+listen*0.3);
    u.uni.uLid.value=Math.min(1,1-open);
    u.uni.uBoot.value=boot;
    u.uni.uAccent.value.copy(st.accent); u.uni.uBright.value.copy(st.bright);
    // gaze: converge on the camera + saccade offsets
    e.rotation.y=curX*0.12+(e===eyeL?-0.04:0.04);
    e.rotation.x=-curY*0.08-exState.gazeY*0.10;    // thinking drifts up, sad drops
    // eye_height narrows the whole aperture (squint), not just the shutter
    e.scale.set(st.eyeW,0.55+0.45*Math.min(1.2,st.eyeH),1).multiplyScalar(0.9+0.1*boot);
    u.halo1.rotation.z=clock*0.7*u.sign; u.halo2.rotation.z=-clock*0.45*u.sign;
    u.halo1.material.opacity=0.55*boot*(0.7+0.3*Math.sin(clock*1.3+u.sign));
    u.halo2.material.opacity=0.25*boot;
    u.halo1.material.color.copy(st.accent); u.halo2.material.color.copy(st.bright);
    u.catchlight.material.opacity=0.8*boot*Math.min(1,open+0.2)*(1+exState.cheek*0.5);
  }
  // brows — the two shapes a face cannot express itself without:
  //   KNIT   (anger/concentration): inner ends pull down and inward, V-shape
  //   SORROW (sadness/concern): inner ends lift, outer ends fall, ramp shape
  // rotation.z pivots near the glabella, so position.y compensates the
  // centroid drift and the *tilt* is what reads.
  const raise=Math.max(0,st.brow)+twitch*0.12;
  const furrow=Math.max(0,-st.brow)+processing*0.35;
  const knit=Math.min(1.4,exState.knit+furrow*0.5), sorrow=exState.sorrow;
  for(const b of [browL,browR]){
    const s=b.userData.sign;
    b.position.y=raise*3.8-furrow*2.2-knit*3.2+sorrow*4.2;
    b.position.x=-s*knit*2.0;                          // knit pulls inward
    b.rotation.z=s*(knit*0.26-sorrow*0.34+raise*0.06);
    b.visible=sEyes>0.05;
    b.material.opacity=(0.75+0.2*(knit+sorrow)*0.5)*boot;
    b.material.color.copy(st.bright);
  }
  // lips — feature-line mouth, voice-synced
  updateLips(jawOpen,st.mouthCurve*(1-st.tension*0.7),st.tension,visR);
  // expressive mouths glow brighter: speech AND emotional curve/tension both
  // lift the lip lines above the point-cloud wash so the expression reads
  const lipGain=1+speakEnv*0.8+(Math.abs(st.mouthCurve)+st.tension)*1.1;
  lipMats.up.opacity=Math.min(1,0.62*boot*lipGain);
  lipMats.low.opacity=Math.min(1,0.62*boot*lipGain);
  lipMats.seam.opacity=Math.min(1,0.95*boot*lipGain);
  lipMats.up.color.copy(st.bright); lipMats.low.color.copy(st.bright); lipMats.seam.color.copy(st.accent);
  lipPtsMat.opacity=Math.min(1,0.85*boot*lipGain);
  lipPtsMat.color.copy(st.accent).lerp(st.bright,0.35);
  lipPtsMat.size=2.4+speakEnv*0.8+Math.abs(st.mouthCurve)*1.2;
  lipUp.visible=lipSeam.visible=lipLow.visible=lipPts.visible=sEyes>0.02;

  // ── shell cubes ───────────────────────────────────────────────────────────
  const mat=mesh.userData.mat;
  const jitter=0.10+st.turb*1.2+st.amp*0.4+(1-face)*0.9+sDestab*2.2;
  mat.emissiveIntensity=(0.35+st.glow*0.45)*(1+st.pulse*0.6*Math.sin(clock*6.0));
  mat.emissive.copy(st.accent).multiplyScalar(0.55);
  bloom.strength=0.30+st.glow*0.26+st.amp*0.18+(1-face)*0.15+sDestab*0.6+Math.sin(Math.min(1,sBurst)*Math.PI)*0.4;
  const hide=st.density<1?(1-st.density)*1.3:0, sweep=clock*1.3;
  const actP=(0.0018+processing*0.004+st.pulse*0.003+speakEnv*0.002);  // dissolve chance
  const doColors=(colorTick++%(frameEMA>22?4:2))===0;
  const P=stage(0.26,0.70);                        // structure + framework window
  // The burst rises and STAYS — the scattered cloud contracts into the head
  // cube-by-cube via each arrival, never re-forming the tight orb shell.
  const explode=Math.sin(Math.min(1,sBurst)*Math.PI*0.5);

  for(let i=0;i<N;i++){
    const h=homes[i];
    // staged arrival: feature cubes migrate to the anchor lines first
    const am=Math.min(1,sstep(h.arrive*0.75,h.arrive*0.75+0.25,P)+(morphT>=0.99?1:0));
    // per-cube dissolve/reform pulses (thought ripples)
    if(h.dis>0){h.disT+=dt*2.2; if(h.disT>=1){h.dis=0;h.disT=0;}}
    else if(((i+colorTick)&1023)===0 && Math.random()<actP*1024){h.dis=1;h.disT=0;}
    const disS=h.dis?Math.sin(h.disT*Math.PI):0;
    const nn=Math.sin(clock*1.7+h.phase);
    // orb-side position + stage-2 explosion outward
    const ex=explode*(20+12*Math.sin(h.phase*3));
    const ox=h.ox*(1+ex/ORB_R), oy=h.oy*(1+ex/ORB_R), oz=h.oz*(1+ex/ORB_R);
    // head-side position: persistent hover motes drift just off the lattice
    const lift=h.mote?face*(1.2+0.7*Math.sin(clock*0.9+h.phase*3)):0;
    const hx=h.x+h.nx*lift, hy=h.y+h.ny*lift, hz=h.z+h.nz*lift;
    let x=lerp(ox,hx,am), y=lerp(oy,hy,am), z=lerp(oz,hz*(am<1?1:breathe),am);
    // orb-state slow swirl
    if(am<1){const sw=(1-am)*0.35*h.swirl;const cx=Math.cos(sw),sx=Math.sin(sw);
      const rx=x*cx-z*sx, rz=x*sx+z*cx; x=rx; z=rz;}
    x+=nn*jitter*0.30; y+=Math.cos(clock*1.5+h.phase)*jitter*0.30; z+=nn*jitter*0.26;
    x+=h.nx*disS*5; y+=h.ny*disS*5; z+=h.nz*disS*5;
    // jaw-follow for the lower-face cubes
    // Scalar maths in place, NOT jawApply({x,y,z}): that built two objects
    // per cube per frame — 8,500 cubes at 60 fps, ~33 MB/s of garbage while
    // he speaks (MEASURED with V8's sampling heap profiler) — and the heap
    // swelled to ~300 MB between collections: renderer RAM this machine did
    // not have, and a long collection pause mid-sentence.
    if(jawOpen>0.01&&am>0.9){const jw=jawWeight(y,z)*jawOpen;
      if(jw>0){const ja=JAW.angle*jw, jc=Math.cos(ja), js=Math.sin(ja);
        const jy=y-JAW.py, jz=z-JAW.pz; y=JAW.py+jy*jc-jz*js; z=JAW.pz+jy*js+jz*jc;}}
    dummy.position.set(x,y,z);
    // as the data layers fill in, plain cubes recede so the lattice reads;
    // feature cubes and hover motes stay prominent; the eye constructs clear
    // their sockets.
    const settle=1-sDetail*(h.feat>0.45?0.10:0.65)*face;
    const sockClear=(1-0.75*h.sock*sEyes)*(1-0.60*h.mouth*sEyes);
    const qs=(1+0.10*Math.sin(clock*3.2+h.phase*2.0))*settle*sockClear*(1-disS*0.9);
    let show=1; if(hide>0 && ((i*2654435761)%1000)/1000<hide*0.5) show=0;
    dummy.scale.setScalar(show*qs*(1+st.amp*0.08));
    qA.copy(h.oquat).slerp(h.quat,am); dummy.quaternion.copy(qA);
    dummy.rotateZ(disS*3*h.swirl);
    dummy.updateMatrix(); mesh.setMatrixAt(i,dummy.matrix);
    if(doColors){
      const wave=0.72+0.30*Math.sin(sweep-h.yn*6.0+h.phase*0.4);
      const form=lerp(0.3+0.26*h.yn,0.20+0.50*h.shade,face);   // orb wave → face form shading
      let bb=Math.max(0,Math.min(1,form*wave+st.amp*0.10+disS*0.5+h.feat*(sStruct*0.7+sFrame*0.4)*(1-sDetail)));
      if(bb<=0.5) col.copy(st.dark).lerp(st.mid,bb*2); else col.copy(st.mid).lerp(st.bright,(bb-0.5)*2);
      if(h.mote&&face>0.9) col.copy(st.bright).multiplyScalar(1.2+0.5*Math.sin(clock*4+h.phase));
      mesh.setColorAt(i,col);
    }
  }
  mesh.instanceMatrix.needsUpdate=true;
  if(doColors&&mesh.instanceColor)mesh.instanceColor.needsUpdate=true;

  // ── orbiters: thought streams while speaking / thinking ──────────────────
  const orbAct=Math.min(1,(speakEnv*1.3+processing)*face);
  orbiters.visible=orbAct>0.03;
  if(orbiters.visible){
    const bands=orbiters.userData.bands;
    for(let i=0;i<ORBN;i++){
      const b=bands[i];
      const a=b.a0+clock*b.speed*(1+processing*0.8);
      const px=Math.cos(a)*b.rx, pz=Math.sin(a)*b.rz;
      const py=Math.sin(a*1.3+b.a0)*10+px*Math.sin(b.tilt)*0.4;
      dummy.position.set(px,py+4,pz);
      dummy.scale.setScalar(b.scale*orbAct);
      dummy.rotation.set(a,a*1.7,0);
      dummy.updateMatrix(); orbiters.setMatrixAt(i,dummy.matrix);
    }
    orbiters.instanceMatrix.needsUpdate=true;
    orbiters.material.opacity=0.75*orbAct;
    orbiters.material.color.copy(st.bright);
  }

  // ── ambient particles ─────────────────────────────────────────────────────
  const PC=parts.length, pv=st.pvel, dir=st.dir;
  for(let i=0;i<PC;i++){const p=parts[i]; p.life-=0.011+(dir==='down'?0.008:0);
    if(p.life<=0){const h=homes[(Math.random()*N)|0]; p.x=lerp(h.ox,h.x,face);p.y=lerp(h.oy,h.y,face);p.z=lerp(h.oz,h.z,face);p.life=1;
      if(dir==='up'){p.vx=(Math.random()-0.5)*1.0*pv;p.vy=(0.7+Math.random())*pv;p.vz=(Math.random()-0.5)*1.0*pv;}
      else if(dir==='down'){p.vx=(Math.random()-0.5)*0.5*pv;p.vy=-(0.4+Math.random())*pv;p.vz=(Math.random()-0.5)*0.5*pv;}
      else if(dir==='burst'){const d=Math.hypot(p.x,p.y,p.z)||1;p.vx=p.x/d*1.7*pv;p.vy=p.y/d*1.7*pv;p.vz=(p.z/d+0.4)*1.7*pv;}
      else{p.vx=(Math.random()-0.5)*0.8*pv;p.vy=(Math.random()-0.5)*0.8*pv;p.vz=(0.4+Math.random())*0.8*pv;}}
    p.x+=p.vx+(Math.random()-0.5)*st.turb*1.2; p.y+=p.vy+(Math.random()-0.5)*st.turb*1.2; p.z+=p.vz;
    ppos[i*3]=p.x;ppos[i*3+1]=p.y;ppos[i*3+2]=p.z;}
  pgeo.attributes.position.needsUpdate=true; pmat.color.copy(st.bright); pmat.opacity=0.28+st.glow*0.35;

  // ── space: haze pulse + drifting data cubes ───────────────────────────────
  hazeUni.uHeat.value=st.glow+st.amp*0.5; hazeUni.uAccent.value.copy(st.accent);
  bgCubes.rotation.y=Math.sin(clock*0.05)*0.05;
  bgCubes.position.y=Math.sin(clock*0.11)*4;

  // ── presence: living head motion (speech nods, idle drift, listening) ─────
  if(poseHold>0){poseHold-=dt;}
  else{
    nodP=lerp(nodP,speakEnv,0.2);
    // Idle drift uses two slow incommensurate frequencies so it reads as LIVING
    // presence, not a metronome (§10: "he is present", not "an animation is
    // running"), at a deliberately small amplitude (was a ±0.22rad sine swing).
    // It SETTLES while LISTENING so attention reads as stillness, not drift.
    const calm=1-listen*0.6;
    rig.rotation.y=(Math.sin(clock*0.19)*0.6+Math.sin(clock*0.37)*0.4)*0.09*calm
      +Math.sin(clock*1.9)*0.015*nodP;
    // emotional posture: sadness droops the head, alertness leans in,
    // curiosity tilts — the body language half of the expression.  A faint
    // attentive lean-in while listening.
    rig.rotation.x=-0.02+(Math.sin(clock*0.16)*0.6+Math.sin(clock*0.29)*0.4)*0.02*calm
      -nodP*0.02+Math.sin(clock*2.3)*0.012*nodP-listen*0.02+exState.headX*face;
    rig.rotation.z=Math.sin(clock*0.13)*0.010*calm+listen*0.04*Math.sin(clock*0.5)
      +exState.headZ*face;
  }
  rig.position.y=Math.sin(clock*0.9)*0.6*face;                 // breathing
  torsoRig.rotation.y=rig.rotation.y*0.35;                     // torso lags the head
  torsoRig.rotation.z=rig.rotation.z*0.5;
}

let stepFault='';

// ── frame-rate governor ─────────────────────────────────────────────────────
// Separate from the quality governor above, which lowers point density when
// frames get slow. This one lowers the frame RATE when nothing is moving.
//
// Measured on the installed build: the two Chromium render processes cost 83%
// of one core between them while ORION sat silent, because a collapsed orb
// was still composited 60 times a second with full bloom. Idle is all three
// of settled-at-target, silent, and no speech envelope left — and then the
// only motion is a slow breath, which 30 fps carries as well as 60. Anything
// that moves restores full rate on the very next frame, since this is checked
// per frame rather than on a timer.
const ACTIVE_FPS=60, IDLE_FPS=__IDLE_FPS__;
// Hysteresis, and a dwell before dropping. The first version compared each
// signal to one threshold every frame, and every signal here is an
// exponential lerp that approaches its threshold asymptotically and sits on
// it - st.amp is lerped toward live mic amplitude, so ambient room noise
// alone parks it on 0.01. The rate then alternated 60/30/60/30 from one frame
// to the next, and alternating 16.7 and 33.3 ms deltas reads as a far worse
// stutter than a steady 30 ever would. That is what "his face is stuttering"
// turned out to be: not a stall (the Qt thread showed zero pauses over 50 ms
// in 60 s at 100 Hz) and not the GPU (3d engine ~11% mean), but the governor
// meant to make him cheaper.
// Leaving idle is instant and any one signal can do it; entering it needs
// every signal calm for IDLE_AFTER consecutive frames.
const QUIET_IN=0.010, QUIET_OUT=0.030;
const SETTLE_IN=0.002, SETTLE_OUT=0.008;
const IDLE_AFTER=60;
let idleRate=false, calmFor=0;
// The frame target is published as well as returned, because the quality
// governor in step() needs it too and targetFrameMs() is NOT idempotent -
// it advances the dwell counter. Calling it a second time per frame would
// count the same calm frame twice.
let frameTargetMs=1000/ACTIVE_FPS;
function targetFrameMs(){
  if(idleRate){
    if(Math.abs(morphTarget-morphT)>SETTLE_OUT||st.amp>QUIET_OUT||speakEnv>QUIET_OUT){
      idleRate=false; calmFor=0;
    }
  }else{
    const calm=Math.abs(morphTarget-morphT)<SETTLE_IN&&st.amp<QUIET_IN&&speakEnv<QUIET_IN;
    calmFor=calm?calmFor+1:0;
    if(calmFor>=IDLE_AFTER){ idleRate=true; }
  }
  frameTargetMs=1000/(idleRate?IDLE_FPS:ACTIVE_FPS);
  return frameTargetMs;
}

function animate(){
  requestAnimationFrame(animate);
  if(document.hidden)return;                 // never burn GPU unwatched
  const now=performance.now();
  // Half a millisecond of slack: without it a frame that arrives a hair early
  // is dropped and the effective rate halves.
  if(now-lastT<targetFrameMs()-0.5)return;
  const dt=Math.min(0.05,(now-lastT)/1000); lastT=now;
  frameEMA=frameEMA*0.95+dt*1000*0.05;
  try{step(dt);}catch(e){if(stepFault!==String(e)){stepFault=String(e);console.error('ORION avatar step fault:',e);}}
  composer.render();
}
lastT=performance.now();
animate();
addEventListener('resize',()=>{fitCamera();
  renderer.setSize(innerWidth,innerHeight);composer.setSize(innerWidth,innerHeight);});
</script></body></html>"""


#: Where a vendored copy of three.js lives, if one was shipped.
THREE_DIR = "assets/three"

#: The version pinned everywhere — the local copy and the CDN fallback must
#: agree, or an addon compiled against one runs against the other.
THREE_VERSION = "0.160.0"

CDN_BASE = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/"


def local_three() -> "Path | None":
    """The vendored three.js directory, if it is complete.

    Checked file by file rather than by the directory existing: a half-copied
    vendor folder is worse than none, because the importmap then points at
    local paths and the missing addon fails with a module error instead of
    falling back to the network.
    """
    global _LOCAL_THREE
    if _LOCAL_THREE is not _UNCHECKED:
        return _LOCAL_THREE

    from ..constants import resource_path

    # resource_path, not BASE_DIR: in a frozen build the bundled assets are
    # under sys._MEIPASS, and BASE_DIR points at the executable's folder.
    root = resource_path(*THREE_DIR.split("/"))
    _LOCAL_THREE = root if _graph_resolves(root) else None
    return _LOCAL_THREE


#: Sentinel distinct from None, which is a real answer meaning "not vendored".
_UNCHECKED = object()
_LOCAL_THREE: "Path | None | object" = _UNCHECKED

#: The imports the page makes itself. Everything else is discovered.
_ENTRY_MODULES = (
    "build/three.module.js",
    "examples/jsm/postprocessing/EffectComposer.js",
    "examples/jsm/postprocessing/RenderPass.js",
    "examples/jsm/postprocessing/UnrealBloomPass.js",
)

_IMPORT_RE = re.compile(
    r"""(?:^|\n)\s*(?:import|export)[^;'"]*?from\s*['"]([^'"]+)['"]""")


def _resolve_specifier(importer: str, spec: str) -> "str | None":
    """Where an import in *importer* points, as a path under the vendor root.

    Mirrors the page's import map: "three" and "three/addons/" are the two
    bare specifiers it defines, and anything else bare is left to the browser.
    """
    if spec == "three":
        return "build/three.module.js"
    if spec.startswith("three/addons/"):
        return "examples/jsm/" + spec[len("three/addons/"):]
    if not spec.startswith("."):
        return None
    parts: list[str] = []
    for part in (PurePosixPath(importer).parent / spec).as_posix().split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _graph_resolves(root: "Path") -> bool:
    """Whether every module the page imports, directly or not, is present.

    A half-copied vendor folder is worse than none: the import map then points
    at local paths, the graph fails on the missing file, and the page reports
    an error with no message rather than falling back to the network.
    """
    seen: set[str] = set()
    queue = list(_ENTRY_MODULES)
    while queue:
        relative = queue.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False        # missing, unreadable, or a directory
        for spec in set(_IMPORT_RE.findall(text)):
            target = _resolve_specifier(relative, spec)
            if target is not None:
                queue.append(target)
    return True


def _three_base_url():
    """The base URL the page resolves its imports against.

    The custom scheme rather than a file path: an ES module imported from a
    file:// base is blocked by Chromium's module CORS rules, so the vendored
    copy would be present and unusable and the page would quietly stream from
    the CDN instead.
    """
    from PyQt6.QtCore import QUrl

    if local_three() is not None and _register_scheme():
        return QUrl(f"{ORION_SCHEME}://three/")
    return QUrl(CDN_BASE)


def idle_fps() -> int:
    """The frame rate the face settles to when nothing is moving.

    Dropping to 30 while ORION sits silent was measured to save most of a
    core, and with the governor's hysteresis fixed it is genuinely invisible:
    30 is an exact divisor of a 60 Hz display, so the cadence is even.

    It is still an override rather than a constant, because "even 30" and
    "smooth" are not the same judgement for everybody, and somebody who can
    see it should be able to buy it back for the CPU rather than be told the
    stutter is gone. Set ORION_FACE_IDLE_FPS=60 to disable the drop entirely.
    """
    import os

    try:
        wanted = int(str(os.getenv("ORION_FACE_IDLE_FPS", "")).strip() or 30)
    except ValueError:
        return 30
    return max(10, min(60, wanted))


def three_sourced(html: str) -> str:
    """The page with its import map pointed at a real source.

    Local when ORION shipped with three.js, the CDN when he did not. The
    difference matters more than it looks: streaming a megabyte of WebGL
    library from the internet means ORION has no face on a train, behind a
    corporate proxy, on a VPS with restricted egress, or during a CDN
    incident — and the failure is a blank rectangle where he should be.
    """
    # Relative to the orion:// base URL when the scheme is live; otherwise the
    # CDN, because a local path the loader cannot read is worse than a remote
    # one it can. The two must agree: an importmap pointing at files the base
    # URL does not serve fails silently and looks like a missing network.
    #
    # "./" and not "": an import map value is required to be a URL or a path
    # beginning with /, ./ or ../. Left bare, "build/three.module.js" is read
    # as another bare specifier, the entry is discarded, and the page fails
    # with "blocked by a null value" — which presents as the face never
    # materialising. Found by loading the real page and reading its console;
    # nothing about the code looked wrong.
    if local_three() is not None and _SCHEME_REGISTERED:
        base = "./"
    else:
        base = CDN_BASE
    return (html.replace("__THREE_BASE__", base)
                .replace("__IDLE_FPS__", str(idle_fps())))



# ── serving the vendored library ─────────────────────────────────────────────
#
# Vendoring three.js beside ORION is not enough on its own. The page is
# delivered with setHtml, and an ES module imported from a file:// base is
# blocked by Chromium's module CORS rules whatever the WebEngine settings say
# — so the import fails and the page falls back to streaming a megabyte from a
# CDN, which is exactly the offline problem the vendoring was meant to solve.
#
# A custom URL scheme fixes it properly. It is a real origin as far as the
# module loader is concerned, it opens no network port, and it serves only the
# files under assets/three — nothing else on the disk is reachable from the
# page.

ORION_SCHEME = "orion"

_SCHEME_REGISTERED = False


def _register_scheme() -> bool:
    """Declare the scheme. MUST happen before QApplication exists.

    Qt requires scheme registration before the web engine initialises, which
    is the same constraint that already governs importing this module early.
    """
    global _SCHEME_REGISTERED
    if _SCHEME_REGISTERED:
        return True
    try:
        from PyQt6.QtWebEngineCore import QWebEngineUrlScheme
    except Exception:
        return False
    try:
        name = ORION_SCHEME.encode("ascii")
        if QWebEngineUrlScheme.schemeByName(name).name():
            _SCHEME_REGISTERED = True
            return True
        scheme = QWebEngineUrlScheme(name)
        scheme.setFlags(
            QWebEngineUrlScheme.Flag.SecureScheme
            | QWebEngineUrlScheme.Flag.LocalAccessAllowed
            | QWebEngineUrlScheme.Flag.CorsEnabled)
        scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
        QWebEngineUrlScheme.registerScheme(scheme)
        _SCHEME_REGISTERED = True
        return True
    except Exception:
        return False


class _AssetHandler:
    """Serves files from assets/three, and refuses everything else.

    Built lazily and only when WebEngine is present, so importing this module
    on a machine without it stays free.
    """

    MIME = {
        ".js": b"text/javascript",
        ".mjs": b"text/javascript",
        ".json": b"application/json",
        ".wasm": b"application/wasm",
    }

    #: The page's own address. Serving it from the scheme rather than with
    #: setHtml is what gives the document a real origin, and therefore what
    #: lets it fetch the library beside it at all.
    PAGE = "face.html"

    def __new__(cls, root, page: bytes = b""):
        from PyQt6.QtCore import QBuffer, QByteArray
        from PyQt6.QtWebEngineCore import QWebEngineUrlSchemeHandler

        class Handler(QWebEngineUrlSchemeHandler):
            def __init__(self, root, page):
                super().__init__()
                self.root = root
                self.page = page
                self._alive = []

            def _serve(self, job, data: bytes, mime: bytes) -> None:
                buffer = QBuffer(job)
                buffer.setData(QByteArray(data))
                buffer.open(QBuffer.OpenModeFlag.ReadOnly)
                # Kept referenced: a buffer collected before the reply is read
                # serves an empty file, which looks like a corrupt library.
                self._alive.append(buffer)
                del self._alive[:-32]
                job.reply(mime, buffer)

            def requestStarted(self, job):
                from pathlib import Path

                relative = job.requestUrl().path().lstrip("/")
                if relative in ("", _AssetHandler.PAGE):
                    self._serve(job, self.page, b"text/html")
                    return
                if relative in _EXTRA_PAGES:
                    # Other pages that want the vendored library (the brain).
                    self._serve(job, _EXTRA_PAGES[relative], b"text/html")
                    return
                target = (self.root / relative).resolve()
                try:
                    # Refuse anything outside assets/three. A page that can
                    # read arbitrary files is a page that can exfiltrate them.
                    target.relative_to(self.root.resolve())
                except ValueError:
                    job.fail(job.Error.RequestDenied)
                    return
                if not target.is_file():
                    job.fail(job.Error.UrlNotFound)
                    return
                try:
                    data = target.read_bytes()
                except OSError:
                    job.fail(job.Error.RequestFailed)
                    return
                self._serve(job, data,
                            _AssetHandler.MIME.get(target.suffix.lower(),
                                                   b"application/octet-stream"))

        return Handler(root, page)


# Registered at import, which is the only moment Qt will accept it: the set of
# custom schemes is frozen when the web engine initialises with QApplication,
# and a later call is ignored with a warning. app.py already imports this
# module before constructing QApplication — WebEngine decides GPU compositor
# availability at import as well — so this runs inside that window.
#
# Registering it here rather than on first use is the whole fix: done late, the
# page received an orion:// base URL that resolved to nothing, so the vendored
# three.js was present, complete and unreachable, and the face reported a
# missing internet connection on a machine that had one.
if WEBENGINE_OK:
    _register_scheme()


#: Further pages served by the one orion:// handler (name -> HTML bytes).
_EXTRA_PAGES: dict[str, bytes] = {}
#: The one handler, held for the life of the process. A per-widget handler was
#: collected when its face was rebuilt (overlay toggle), and the profile will
#: not take a second handler for a scheme that already has one — so a rebuilt
#: face could fall back to the CDN, and a second page could never share it.
_SHARED_HANDLER: Any = None
_HANDLED_PROFILES: set[int] = set()


def register_page(name: str, html: str) -> None:
    """Serve *html* at orion://three/<name>, import map pointed at the
    vendored three.js (so it works offline, like the face)."""
    _EXTRA_PAGES[str(name)] = three_sourced(html).encode("utf-8")


def ensure_asset_handler(profile: Any) -> bool:
    """Install the shared orion:// handler on *profile* once. True when the
    vendored library (and registered pages) can be served from it."""
    global _SHARED_HANDLER
    root = local_three()
    if root is None or not _register_scheme():
        return False
    try:
        from PyQt6 import sip
        key = int(sip.unwrapinstance(profile))
    except Exception:
        key = id(profile)
    if key in _HANDLED_PROFILES:
        return True
    if _SHARED_HANDLER is None:
        _SHARED_HANDLER = _AssetHandler(root, three_sourced(FACE_HTML).encode("utf-8"))
    profile.installUrlSchemeHandler(ORION_SCHEME.encode("ascii"), _SHARED_HANDLER)
    _HANDLED_PROFILES.add(key)
    return True


class QuantumFace3D(QWidget):
    """Real-time GPU quantum human — holographic data head with orb morph."""

    available = WEBENGINE_OK
    # States where ORION collapses back to the glowing orb (dormant/idle).
    _ORB_STATES = {"STANDBY", "INITIALISING", "OFFLINE", "SHUTTING DOWN"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state_name = "STANDBY"
        # "face": the state decides (orb when dormant, him when active).
        # "orb": held in the orb whatever he is doing — the form somebody asks
        # for when a human face is not what the room wants. Pinned HERE rather
        # than by swapping widgets, so switching is the page's own morph (a
        # second and a half, no reload) instead of a new Chromium page.
        self._form = "face"
        self._built = False
        self._pending_emotion: tuple[str, dict] | None = None
        # AvatarController updates these values at 20 Hz.  Keep the visual
        # stream smooth while avoiding a QtWebEngine IPC call for every tiny
        # tracking jitter; the values are reset whenever the page is rebuilt.
        self._last_js_amplitude: float | None = None
        self._last_js_spectrum: tuple[float, float, float] | None = None
        self._last_js_pose: tuple[float, float] | None = None
        # The last emotion sent, so a reloaded page gets it back; and when the
        # renderer died recently, so a crash loop is not reloaded forever.
        self._last_emotion: dict | None = None
        self._render_crashes: list[float] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout
        self.view: Any = None
        self.timer = QTimer(self)      # interface parity
        if WEBENGINE_OK:
            self._placeholder = QLabel("◉  ORION materialises when you open this view.")
            self._placeholder.setObjectName("mutedLabel")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._placeholder)
        else:
            self._placeholder = None

    def _ensure_built(self) -> None:
        if self._built or not WEBENGINE_OK:
            return
        self._built = True
        if self._placeholder is not None:
            self._placeholder.hide()
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None
        self.view = QWebEngineView()
        self._reset_js_stream_cache()
        try:
            # A call made while Chromium is still loading can be accepted by
            # Qt but never reach the page.  Re-arm the first live frame once
            # the document is ready so the dedupe cache cannot hide it.
            self.view.loadFinished.connect(self._on_page_loaded)
            # When the machine runs out of memory Windows (or Chromium) kills
            # a renderer. The globe recovered from that; the face did not, so
            # the main window stayed black until ORION was restarted.
            self.view.renderProcessTerminated.connect(self._on_render_crashed)
        except Exception:
            pass
        self._install_asset_handler()
        if getattr(self, "_asset_handler", None) is not None:
            # A real navigation, so the document has a real origin and may
            # fetch the library beside it. setHtml cannot: its documents are
            # opaque-origin, so the fetch is cross-origin and fails the CORS
            # check without resolving, rejecting or logging anything.
            from PyQt6.QtCore import QUrl

            self.view.setUrl(
                QUrl(f"{ORION_SCHEME}://three/{_AssetHandler.PAGE}"))
        else:
            self.view.setHtml(three_sourced(FACE_HTML), _three_base_url())
        self._layout.addWidget(self.view)
        if self._pending_emotion is not None:
            name, params = self._pending_emotion
            self.apply_emotion(name, params)
        self.set_state(self.state_name)

    def showEvent(self, event: Any) -> None:
        self._ensure_built()
        super().showEvent(event)

    # ── public interface (mirrors HologramFace exactly) ───────────────────────

    def set_amplitude(self, value: float) -> None:
        amplitude = max(0.0, min(1.0, float(value)))
        if self.view is None:
            return
        if (self._last_js_amplitude is not None
                and abs(amplitude - self._last_js_amplitude) < 0.008):
            return
        if self._js(f"window.orionAmplitude&&orionAmplitude({amplitude:.3f})"):
            self._last_js_amplitude = amplitude

    def set_spectrum(self, low: float, mid: float, high: float) -> None:
        clamp = lambda v: max(0.0, min(1.0, float(v)))
        spectrum = (clamp(low), clamp(mid), clamp(high))
        if self.view is None:
            return
        if (self._last_js_spectrum is not None
                and max(abs(a - b) for a, b in zip(spectrum, self._last_js_spectrum)) < 0.012):
            return
        if self._js(
            f"window.orionSpectrum&&orionSpectrum("
            f"{spectrum[0]:.3f},{spectrum[1]:.3f},{spectrum[2]:.3f})"
        ):
            self._last_js_spectrum = spectrum

    def set_speaking(self, active: bool) -> None:
        pass

    def set_state(self, state: str) -> None:
        self.state_name = str(state or "STANDBY").upper()
        self._js(f"window.orionLabel&&orionLabel({json.dumps('ORION · ' + self.state_name)})")
        # Materialise the quantum human when present; collapse to the orb when
        # dormant — or always, while the orb form is pinned.
        morph = 0.0 if (self._form == "orb" or self.state_name in self._ORB_STATES) else 1.0
        self._js(f"window.orionMorph&&orionMorph({morph})")
        # The page reacts to the state itself too (PROCESSING data surge,
        # LISTENING eye tracking, pupil contraction).
        self._js(f"window.orionState&&orionState({json.dumps(self.state_name)})")

    def apply_emotion(self, name: str, params: Any) -> None:
        if not isinstance(params, dict):
            return
        if not self._built:
            self._pending_emotion = (name, params)
            return
        self._last_emotion = params
        self._js(f"window.orionEmotion&&orionEmotion({json.dumps(params)})")

    def set_viseme(self, viseme_id: Any, weight: Any = 1.0,
                   closure: Any = None) -> None:
        """Shape the mouth toward a phoneme posture (Mark XXVI) through the
        EXISTING spectral bridge — NO renderer change, so no risk to the WebGL
        face. On speech paths that already supply a real audio spectrum this
        reinforces it; on paths that do not (offline TTS / text) it is what gives
        the 3-D mouth phoneme shape rather than a flat jaw.

        Accepts BOTH shapes the window sends, because there are two:

            set_viseme("PP", 1.0)          an identifier and a weight
            set_viseme(open, wide, close)  a raw posture, three floats

        Only the first was implemented here, so every raw-posture call raised
        TypeError — 140 times in a 55-second run, once per viseme while
        speaking, each one a full traceback into the self-repair journal. It
        was invisible until this became the default face, because the other
        renderer had always accepted both.

        A raw posture is mapped rather than rejected: openness drives the jaw,
        and a closure at all means the lips are pressed shut, which must win
        over loudness — /m/, /b/ and /p/ are made with the mouth closed.
        """
        from ..viseme import viseme_to_spectral

        if not isinstance(viseme_id, str):
            openness = max(0.0, min(1.0, float(viseme_id or 0.0)))
            shut = max(0.0, min(1.0, float(closure if closure is not None
                                           else 0.0)))
            amp = 0.0 if shut > 0.5 else openness
            wide = max(-1.0, min(1.0, float(weight or 0.0)))
            # Wide vowels carry more high band, rounded ones more low. The
            # same relationship viseme_to_spectral encodes, applied directly.
            self.set_amplitude(amp)
            self.set_spectrum(
                max(0.0, 0.5 - wide * 0.4),
                0.4 + amp * 0.3,
                max(0.0, 0.3 + wide * 0.4),
            )
            return

        amp, low, mid, high = viseme_to_spectral(viseme_id, weight)
        self.set_amplitude(amp)
        self.set_spectrum(low, mid, high)

    # ── avatar-system extensions (AvatarController drives these) ──────────────

    def set_pose(self, yaw: float, pitch: float) -> None:
        """Rotate the head rig — the AvatarController's face-tracking follow."""
        pose = (float(yaw), float(pitch))
        if self.view is None:
            return
        if (self._last_js_pose is not None
                and max(abs(a - b) for a, b in zip(pose, self._last_js_pose)) < 0.003):
            return
        if self._js(f"window.orionPose&&orionPose({pose[0]:.4f},{pose[1]:.4f})"):
            self._last_js_pose = pose

    def set_morph(self, value: float) -> None:
        """Scrub the orb↔human materialisation directly (0 orb … 1 human)."""
        self._js(f"window.orionMorph&&orionMorph({max(0.0, min(1.0, float(value))):.3f})")

    def set_label(self, text: str) -> None:
        self._js(f"window.orionLabel&&orionLabel({json.dumps(str(text)[:60])})")

    @property
    def form(self) -> str:
        return self._form

    def set_form(self, form: str) -> str:
        """Hold ORION in his orb ("orb") or let his state decide ("face").

        Applied at once through the page's own morph; the orb still breathes,
        pulses with his voice and reacts to his state.
        """
        self._form = "orb" if str(form or "").strip().lower() == "orb" else "face"
        self.set_state(self.state_name)
        return self._form

    def check_state(self, callback) -> None:
        """Report "alive", "failed" or "loading" to *callback*.

        The three are genuinely different and were previously collapsed into
        two. A page that has not drawn yet is not a page that cannot draw, and
        treating them alike meant a slow start looked exactly like a machine
        without WebGL — which then latched a permanent fallback.
        """
        if self.view is None:
            callback("failed")
            return
        try:
            self.view.page().runJavaScript(
                "(function(){try{"
                " if(window.__dbg && typeof window.__dbg.fps==='number')"
                "  return 'alive';"
                " return window.__orionFailed ? 'failed' : 'loading';"
                "}catch(e){return 'loading';}})()",
                lambda result: callback(str(result or "loading")))
        except Exception:
            callback("loading")

    def check_alive(self, callback) -> None:
        """Ask the page whether its render loop is actually running.

        ``window.__dbg`` is assigned inside the animation frame, so it exists
        only once Three.js has loaded, compiled its shaders and drawn at least
        once. Absent, something failed — WebGL unavailable, a GPU driver that
        refuses, a corrupt vendored file — and the page is showing a message
        where ORION should be.

        The page has always called ``window.orionFaceFailed`` in that case and
        nothing on this side ever listened, so the failure was invisible: no
        face, no fallback, no log line. This is the listener.

        Asynchronous because runJavaScript is: *callback* receives True or
        False, on the GUI thread, whenever the page answers.
        """
        if self.view is None:
            callback(False)
            return
        try:
            self.view.page().runJavaScript(
                "(function(){try{return !!(window.__dbg &&"
                " typeof window.__dbg.fps === 'number');}"
                "catch(e){return false;}})()",
                lambda result: callback(bool(result)))
        except Exception:
            callback(False)

    def _install_asset_handler(self) -> None:
        """Let the page fetch the vendored library, and nothing else."""
        root = local_three()
        if root is None or not _register_scheme():
            return
        try:
            # One process-wide handler (see _SHARED_HANDLER), installed once
            # per profile and never collected. Tracked by profile rather than
            # asked of it: the profile returns a non-None placeholder even
            # when nothing is installed, so asking silently did nothing.
            if ensure_asset_handler(self.view.page().profile()):
                self._asset_handler = _SHARED_HANDLER
            else:
                self._asset_handler = None
        except Exception as exc:
            # Reported, not swallowed: falling back to the CDN is survivable,
            # doing it silently is how the vendored copy went unused.
            self._asset_handler = None
            try:
                print(f"[ORION] FACE: serving three.js locally failed ({exc}); "
                      f"falling back to the CDN.", flush=True)
            except Exception:
                pass

    def _reset_js_stream_cache(self) -> None:
        self._last_js_amplitude = None
        self._last_js_spectrum = None
        self._last_js_pose = None

    def _on_page_loaded(self, _ok: bool) -> None:
        """The document is ready: resend what a still-loading page dropped.

        _ensure_built sets the state and any pending emotion straight after
        navigating, before the page can run them, so the face used to start in
        its default state until the next change arrived.
        """
        self._reset_js_stream_cache()
        self.set_state(self.state_name)
        if self._last_emotion is not None:
            self._js(f"window.orionEmotion&&orionEmotion({json.dumps(self._last_emotion)})")

    #: Reloads allowed inside RENDER_CRASH_WINDOW_S before the face is left
    #: alone, so a renderer that dies on every load cannot spin forever.
    RENDER_CRASH_LIMIT = 5
    RENDER_CRASH_WINDOW_S = 600.0

    def _on_render_crashed(self, status: Any, exit_code: int) -> None:
        import time
        try:
            from PyQt6.QtWebEngineCore import QWebEnginePage
            if status == QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus:
                return                      # shutdown, not a crash
        except Exception:
            pass
        now = time.monotonic()
        self._render_crashes = [t for t in self._render_crashes
                                if now - t < self.RENDER_CRASH_WINDOW_S] + [now]
        detail = f"face render process terminated: status={status}, exit_code={exit_code}"
        try:
            from .. import crash_reporter
            crash_reporter.report("face-render", detail)
        except Exception:
            pass
        if len(self._render_crashes) > self.RENDER_CRASH_LIMIT:
            print(f"[FACE3D] {detail} - repeated crashes, not reloading again.", flush=True)
            return
        print(f"[FACE3D] {detail} - reloading the face.", flush=True)
        QTimer.singleShot(500, self._reload_after_crash)

    def _reload_after_crash(self) -> None:
        if self.view is None:
            return
        self._reset_js_stream_cache()
        self.view.reload()

    def _js(self, code: str) -> bool:
        if self.view is None:
            return False
        try:
            self.view.page().runJavaScript(code)
            return True
        except Exception:
            return False
