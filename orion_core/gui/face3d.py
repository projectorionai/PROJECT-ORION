"""
QuantumFace3D — ORION's true 3-D quantum voxel avatar (Phase 8, Section A).

ORION's head is built by VOXELISING A REAL ANATOMICAL HEAD MESH (the Lee
Perry-Smith scan bundled with Three.js): thousands of points are sampled
across the model's surface and rendered as glowing cubes, each oriented to the
surface normal so a real key light shades the brow, sockets, nose, cheeks,
lips and jaw like a sculpt.

This build adds what the avatar was missing:

    • SYMMETRY & DENSITY — only one half of the head is sampled and mirrored,
      so the face is perfectly left/right symmetric and evenly detailed.
    • REAL EYES — the sockets are darkened into recesses and two clean,
      blinking irises are placed in them, instead of a clump of bright cubes.
    • ORB ↔ FACE MORPH — every voxel also has a home on a sphere, so ORION can
      seamlessly form out of a glowing orb into the face when he is present and
      collapse back to the orb when dormant (driven by his state).
    • A FACIAL-ACTION SYSTEM — per-voxel muscle weights (brow, eyelids, cheeks,
      lip corners, jaw, forehead) driven by the emotion parameters, plus
      always-on idle micro-movements and blinking, so he is never static.

Rendered on the GPU via Three.js/WebGL in a ``QWebEngineView`` (the same
pipeline the globe uses, with ``AA_ShareOpenGLContexts`` set in ``app.py``) —
one ``InstancedMesh`` draw call, cinematic bloom, 60 FPS.  Offline it falls
back to a procedural voxel head.  Interface parity (``set_amplitude`` /
``set_speaking`` / ``set_state`` / ``apply_emotion`` / ``timer``) is preserved.
"""

from __future__ import annotations

import json
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
  html,body{margin:0;height:100%;background:#03040a;overflow:hidden;font-family:'Segoe UI',Arial}
  #label{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
    color:#8fe0ff;font-size:12px;letter-spacing:4px;text-transform:uppercase;opacity:.72}
  #loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#5b6b7a;font-size:13px}
</style></head><body>
<div id="loading">Materialising ORION…</div>
<div id="label">ORION</div>
<script type="importmap">{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js';

const HEAD_URL='https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/LeePerrySmith/LeePerrySmith.glb';
const COUNT=16000;            // voxels (symmetric pairs)
const TARGET=150;             // head height in world units
const ORB_R=52;               // orb radius for the morph
// Anatomical eye placement as fractions of the head bounding box (tunable):
// EYE_X = half the interpupillary distance, EYE_Y = height (0 chin → 1 crown),
// EYE_Z = how far forward from the head centre the eyeball sits.
const EYE_X=0.135, EYE_Y=0.655, EYE_Z=0.30;

const scene=new THREE.Scene();
scene.fog=new THREE.FogExp2(0x03040a,0.0016);
const cam=new THREE.PerspectiveCamera(34,innerWidth/innerHeight,0.1,5000);
cam.position.set(0,6,300); cam.lookAt(0,4,0);
const COOL=new THREE.Color(0x2f8fd0);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace; renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=0.82; document.body.appendChild(renderer.domElement);
const composer=new EffectComposer(renderer);
composer.addPass(new RenderPass(scene,cam));
const bloom=new UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),0.32,0.5,0.6);
composer.addPass(bloom);

const rig=new THREE.Group(); scene.add(rig);
scene.add(new THREE.AmbientLight(0x141f33,0.32));
const key=new THREE.DirectionalLight(0xe6f0ff,1.55);  key.position.set(-150,170,190); scene.add(key);
const rimL=new THREE.DirectionalLight(0x2f6bd0,1.05);  rimL.position.set(190,-20,-150); scene.add(rimL);
const fill=new THREE.DirectionalLight(0x5a8fd8,0.30);  fill.position.set(120,40,170);  scene.add(fill);

// ── emotion / reactive state (eased) ──────────────────────────────────────
const st={density:1.0,glow:0.5,pulse:0.0,speed:1.0,pvel:1.0,turb:0.1,brow:0.0,
  eyeH:1.0,eyeW:1.0,mouthCurve:0.05,tension:0.0,amp:0.0,dir:'drift',
  dark:new THREE.Color(0x0a1830),mid:new THREE.Color(0x3aa0e0),bright:new THREE.Color(0xbfe6ff),accent:new THREE.Color(0x66e6ff)};
const tg=Object.assign({},st);
const lerp=(a,b,t)=>a+(b-a)*t;
let morphT=0, morphTarget=1;          // 0 = orb, 1 = face
window.orionAmplitude=a=>{tg.amp=Math.max(0,Math.min(1,a));};
window.orionMorph=v=>{morphTarget=Math.max(0,Math.min(1,v));};
window.orionEmotion=p=>{ if(!p)return;
  const m={voxel_density:'density',glow:'glow',glow_pulse:'pulse',speed:'speed',particle_velocity:'pvel',
    turbulence:'turb',brow:'brow',eye_height:'eyeH',eye_width:'eyeW',mouth_curve:'mouthCurve',mouth_tension:'tension'};
  for(const k in m){ if(k in p) tg[m[k]]=p[k]; }
  if('particle_direction' in p) tg.dir=p.particle_direction;
  const rgb=a=>new THREE.Color(a[0]/255,a[1]/255,a[2]/255);
  if(p.palette_dark)tg.dark=rgb(p.palette_dark); if(p.palette_mid)tg.mid=rgb(p.palette_mid);
  if(p.palette_bright)tg.bright=rgb(p.palette_bright); if(p.accent)tg.accent=rgb(p.accent);
};
window.orionLabel=t=>{const e=document.getElementById('label'); if(e)e.textContent=t;};

// ── facial action units (FACS-lite) — per-voxel muscle weights ────────────
function auW(yn,xn,nz){
  // Face mask: zero beyond |xn|>0.25 (ears/temples) and on non-front voxels,
  // so a blink or brow move never drags the EARS with it (the "multiple eyes").
  const mask=Math.max(0,Math.min(1,(0.25-Math.abs(xn))/0.05));
  const f=Math.max(0,nz*1.3)*mask;
  const G=(cy,cx,sy,sx)=>Math.exp(-((yn-cy)*(yn-cy))/(sy*sy)-((xn-cx)*(xn-cx))/(sx*sx));
  return {
    browIn:(G(0.735,0.075,0.028,0.065)+G(0.735,-0.075,0.028,0.065))*f,
    browOut:(G(0.735,0.19,0.028,0.075)+G(0.735,-0.19,0.028,0.075))*f,
    lid:(G(0.665,0.165,0.026,0.085)+G(0.665,-0.165,0.026,0.085))*f,
    cheek:(G(0.585,0.20,0.045,0.10)+G(0.585,-0.20,0.045,0.10))*f,
    lipUp:G(0.505,0,0.018,0.10)*f, lipLo:G(0.455,0,0.022,0.11)*f,
    lipC:(G(0.48,0.12,0.02,0.05)+G(0.48,-0.12,0.02,0.05))*f,
    jaw:G(0.43,0,0.05,0.17)*f, fore:G(0.80,0,0.05,0.24)*f, nose:G(0.57,0,0.04,0.055)*f,
    side:(xn>=0?1:-1)
  };
}

// ── build the voxel head (fallback: procedural) ───────────────────────────
let homes=[], mesh=null, N=0, parts=[], pgeo=null, ppos=null, pmat=null, irisL=null, irisR=null, eyeC=null;
const dummy=new THREE.Object3D(), col=new THREE.Color(), Zc=new THREE.Vector3(0,0,1);
const qA=new THREE.Quaternion(), qB=new THREE.Quaternion(), vTmp=new THREE.Vector3();

new GLTFLoader().load(HEAD_URL, gltf=>{
  let src=null; gltf.scene.traverse(o=>{ if(o.isMesh && o.geometry && o.geometry.attributes.position && !src) src=o; });
  if(!src){ buildProcedural(); return; }
  try{ buildFromMesh(src); }catch(e){ buildProcedural(); }
}, undefined, ()=>buildProcedural());

function buildFromMesh(src){
  src.updateWorldMatrix(true,false);
  const geo=src.geometry.clone().applyMatrix4(src.matrixWorld);
  geo.computeBoundingBox(); geo.computeVertexNormals();
  const bb=geo.boundingBox, size=new THREE.Vector3(), center=new THREE.Vector3();
  bb.getSize(size); bb.getCenter(center);
  const scale=TARGET/size.y;
  // Place the eyes geometrically from the head proportions — reliable, unlike
  // averaging a fuzzy region (which caught the ears and threw them sideways).
  eyeC={x:size.x*EYE_X*scale, y:(bb.min.y+size.y*EYE_Y-center.y)*scale, z:size.z*EYE_Z*scale};
  const sampler=new MeshSurfaceSampler(new THREE.Mesh(geo)).build();
  const P=new THREE.Vector3(), Nr=new THREE.Vector3();
  // Sample only the RIGHT half; mirror each point → perfect symmetry.
  for(let guard=0; homes.length<COUNT && guard<COUNT*4; guard++){
    sampler.sample(P,Nr);
    if(P.x<center.x) continue;
    const x=(P.x-center.x)*scale, y=(P.y-center.y)*scale, z=(P.z-center.z)*scale;
    const yn=(P.y-bb.min.y)/size.y, xn=(P.x-center.x)/size.x;
    let region='skin';
    // Tight, front-facing, central bands so the ears/temples are excluded.
    if(yn>0.615&&yn<0.695&&xn>0.06&&xn<0.20&&Nr.z>0.45) region='eye';
    else if(yn>0.70&&yn<0.79&&xn<0.24&&Nr.z>0.4) region='brow';
    else if(yn>0.45&&yn<0.55&&xn<0.20&&Nr.z>0.45) region='mouth';
    const b=0.28+0.30*yn + (region==='mouth'||yn>0.55&&yn<0.60&&xn<0.06?0.10:0);  // lift lips + nose ridge
    const base={x:x,y:y,z:z,nx:Nr.x,ny:Nr.y,nz:Nr.z,b:b,yn:yn,xn:xn,region:region,phase:Math.random()*6.283};
    homes.push(base);
    homes.push({x:-x,y:y,z:z,nx:-Nr.x,ny:Nr.y,nz:Nr.z,b:b,yn:yn,xn:-xn,region:region,phase:Math.random()*6.283});
  }
  build();
}

function buildProcedural(){
  const PR=[[-1.18,0.05],[-1.02,0.34],[-0.82,0.60],[-0.56,0.75],[-0.26,0.84],[0.02,0.87],
    [0.30,0.83],[0.56,0.73],[0.80,0.58],[1.00,0.40],[1.16,0.20],[1.28,0.05]];
  const fw=v=>{if(v<=PR[0][0])return PR[0][1];if(v>=PR[11][0])return PR[11][1];
    for(let i=0;i<11;i++){const[a,c]=PR[i],[b,d]=PR[i+1];if(v>=a&&v<=b){const t=(v-a)/(b-a+1e-9);return c+(d-c)*t;}}return 0;};
  const g=(u,v,cu,cv,su,sv)=>{const du=(u-cu)/su,dv=(v-cv)/sv,e=du*du+dv*dv;return e<9?Math.exp(-e):0;};
  const S=78,step=0.024;
  for(let v=-1.2;v<=1.28;v+=step){const wv=fw(v);
    for(let u=0;u<=wv;u+=step){        // right half only → mirror
      let z=0.5*Math.exp(-(u*u)/0.82-(v*v)/1.55)+0.55*g(u,v,0,0.13,0.075,0.36)+0.26*g(u,v,0,0.42,0.075,0.08)+0.18*g(u,v,0,-0.27,0.5,0.075);
      const es=g(u,v,0.32,-0.06,0.145,0.115); z-=0.4*es; const ms=g(u,v,0,0.6,0.21,0.04); z-=0.15*ms;
      if(es>0.72||(ms>0.8&&u<0.15)||z<-0.03)continue;
      const yn=(1.28-v)/2.48, b=0.3+0.28*(0.5+0.4*u);
      const region=(v>0.53&&v<0.7&&u<0.24)?'mouth':(v>-0.34&&v<-0.16?'brow':(v>-0.13&&v<0.02&&Math.abs(u-0.32)<0.14?'eye':'skin'));
      homes.push({x:u*S,y:-v*S,z:z*S,nx:u,ny:-v,nz:0.9,b:b,yn:yn,xn:u*0.5,region:region,phase:Math.random()*6.283});
      if(u>0.01) homes.push({x:-u*S,y:-v*S,z:z*S,nx:-u,ny:-v,nz:0.9,b:b,yn:yn,xn:-u*0.5,region:region,phase:Math.random()*6.283});
    }
  }
  eyeC={x:0.30*S,y:0.06*S,z:0.42*S};
  build();
}

function build(){
  N=homes.length;
  const L=document.getElementById('loading'); if(L)L.remove();
  const GR=Math.PI*(3-Math.sqrt(5));
  for(let i=0;i<N;i++){ const h=homes[i];
    // Orientation to the surface normal (sculpt shading).
    const n=new THREE.Vector3(h.nx,h.ny,h.nz); if(n.lengthSq()<1e-6)n.set(0,0,1); n.normalize();
    h.quat=new THREE.Quaternion().setFromUnitVectors(Zc,n);
    h.au=auW(h.yn,h.xn||0,h.nz);
    // Orb home (fibonacci sphere) + its outward orientation, for the morph.
    const yy=1-(i/(N-1))*2, rr=Math.sqrt(1-yy*yy), th=GR*i;
    h.ox=Math.cos(th)*rr*ORB_R; h.oy=yy*ORB_R; h.oz=Math.sin(th)*rr*ORB_R;
    h.oquat=new THREE.Quaternion().setFromUnitVectors(Zc,new THREE.Vector3(h.ox,h.oy,h.oz).normalize());
  }
  const cube=new THREE.BoxGeometry(2.1,2.1,2.1);
  const mat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0x1c6aa8,emissiveIntensity:0.28,metalness:0.35,roughness:0.55});
  mesh=new THREE.InstancedMesh(cube,mat,N); mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.setColorAt(0,new THREE.Color(1,1,1)); mesh.userData.mat=mat; rig.add(mesh);
  // No floating irises: the head mesh already has eyeballs in the sockets, so
  // the voxels form the eyes naturally and in the right place — clean recessed
  // sockets like the reference sculpt.
  // Particle field.
  const PC=Math.min(600,(N/3)|0); pgeo=new THREE.BufferGeometry(); ppos=new Float32Array(PC*3);
  for(let i=0;i<PC;i++){const h=homes[(Math.random()*N)|0]; parts.push({x:h.x,y:h.y,z:h.z,vx:0,vy:0,vz:0,life:Math.random()});
    ppos[i*3]=h.x;ppos[i*3+1]=h.y;ppos[i*3+2]=h.z;}
  pgeo.setAttribute('position',new THREE.BufferAttribute(ppos,3));
  pmat=new THREE.PointsMaterial({color:0x8fe0ff,size:1.3,transparent:true,opacity:0.55,blending:THREE.AdditiveBlending,depthWrite:false});
  scene.add(new THREE.Points(pgeo,pmat)); animate();
}

let clock=0, blinkT=2.5, blinking=0, blinkP=0, blinkAmt=0;
function animate(){
  requestAnimationFrame(animate);
  clock+=0.016*st.speed;
  morphT=lerp(morphT,morphTarget,0.045);
  for(const k of ['density','glow','pulse','speed','pvel','turb','brow','eyeH','eyeW','mouthCurve','tension']) st[k]=lerp(st[k],tg[k],0.06);
  st.amp=lerp(st.amp,tg.amp,0.3);
  st.dark.lerp(tg.dark,0.05); st.mid.lerp(tg.mid,0.05); st.bright.lerp(tg.bright,0.05); st.accent.lerp(tg.accent,0.05);
  // expression activations + always-on idle + blink
  const idleBrow=0.12*Math.sin(clock*0.6)+0.07*Math.sin(clock*1.9), idleLid=0.09+0.07*Math.sin(clock*0.85), idleLip=0.04*Math.sin(clock*0.7+1);
  blinkT-=0.016; if(blinkT<=0){blinking=1;blinkT=2.4+Math.random()*4.5;}
  if(blinking){blinkP+=0.20;blinkAmt=Math.sin(Math.min(Math.PI,blinkP));if(blinkP>=Math.PI){blinking=0;blinkP=0;}} else blinkAmt*=0.55;
  const face=morphT;                     // expression only when face-formed
  const browRaise=(Math.max(0,st.brow)+idleBrow*0.5)*face, browFurrow=Math.max(0,-st.brow)*face;
  const smile=Math.max(0,st.mouthCurve)*face, frown=Math.max(0,-st.mouthCurve)*face;
  const squint=(Math.max(0,1-st.eyeH)+idleLid*0.35)*face, widen=Math.max(0,st.eyeH-1)*face;
  const jawOpen=st.amp*face, tension=st.tension*face;
  const mat=mesh.userData.mat;
  const breathe=1+0.02*Math.sin(clock*1.15);
  const jitter=0.12+st.turb*1.4+st.amp*0.6+(1-morphT)*0.8;   // more shimmer as an orb
  mat.emissiveIntensity=(0.06+st.glow*0.30)*(1+st.pulse*0.6*Math.sin(clock*6.0));
  mat.emissive.copy(st.accent).lerp(COOL,0.45).multiplyScalar(0.5);
  bloom.strength=0.12+st.glow*0.32+st.amp*0.15+(1-morphT)*0.25;
  const hide=st.density<1?(1-st.density)*1.3:0, sweep=clock*1.3;
  for(let i=0;i<N;i++){
    const h=homes[i];
    let show=1; if(hide>0 && ((i*2654435761)%1000)/1000<hide*(h.region==='skin'?0.6:0.2)) show=0;
    const nn=Math.sin(clock*1.7+h.phase);
    // Morph between the orb home and the face home.
    let x=lerp(h.ox,h.x,morphT), y=lerp(h.oy,h.y,morphT), z=lerp(h.oz,h.z*breathe,morphT);
    x+=nn*jitter*0.32; y+=Math.cos(clock*1.5+h.phase)*jitter*0.32; z+=nn*jitter*0.28;
    // QUANTUM FLOW — coordinated energy ripples travel across the surface, so
    // the cubes are never static: a sentient, quantum shimmer that stays on the
    // face (small amplitude) rather than dispersing into a cloud.
    const flow=Math.sin(clock*2.6+h.yn*11-h.xn*7+h.phase);
    const flow2=Math.cos(clock*1.9+(h.x+h.y)*0.045);
    const qamp=(0.7+st.amp*1.6+(1-morphT)*1.4);      // stronger when speaking / as an orb
    x+=flow2*0.35*qamp; y+=flow*0.30*qamp; z+=flow*0.55*qamp;
    // Coordinated facial muscle action (scaled by how face-formed he is).
    const a=h.au;
    y+=a.browIn*(browRaise*8-browFurrow*7)+a.browOut*(browRaise*6.5-browFurrow*1.5);
    x+=a.browIn*(-a.side*browFurrow*3.2); z+=a.browIn*(browFurrow*3.5);
    y+=a.lid*(widen*4-squint*3.2-blinkAmt*face*7.5-smile*2.2);
    y+=a.cheek*(smile*5.5);
    y+=a.lipUp*(-jawOpen*4-tension*2.5+smile*1.5+idleLip*face);
    y+=a.lipLo*(jawOpen*9-frown*4-tension*1.5); y+=a.lipC*(smile*5.5-frown*5); x+=a.lipC*(a.side*smile*3.5);
    y+=a.jaw*(-jawOpen*6.5); z+=a.fore*(browRaise*2.2)+a.nose*(0.6*Math.sin(clock*1.15)*face);
    dummy.position.set(x,y,z);
    // Per-voxel quantum shimmer — each cube pulses on its own phase.
    const qs=1+0.14*Math.sin(clock*3.2+h.phase*2.0);
    dummy.scale.setScalar(show*(1+st.amp*0.12)*qs);
    qA.copy(h.oquat).slerp(h.quat,morphT); dummy.quaternion.copy(qA);
    dummy.updateMatrix(); mesh.setMatrixAt(i,dummy.matrix);
    const wave=0.72+0.30*Math.sin(sweep-h.yn*6.0+h.phase*0.4);
    const bb=Math.max(0,Math.min(1,h.b*wave+st.amp*0.12));
    if(bb<=0.5) col.copy(st.dark).lerp(st.mid,bb*2); else col.copy(st.mid).lerp(st.bright,(bb-0.5)*2);
    if(h.region==='eye') col.multiplyScalar(0.72);   // subtle socket recess
    mesh.setColorAt(i,col);
  }
  mesh.instanceMatrix.needsUpdate=true; if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;
  // Particles.
  const PC=parts.length, pv=st.pvel, dir=st.dir;
  for(let i=0;i<PC;i++){const p=parts[i]; p.life-=0.011+(dir==='down'?0.008:0);
    if(p.life<=0){const h=homes[(Math.random()*N)|0]; p.x=lerp(h.ox,h.x,morphT);p.y=lerp(h.oy,h.y,morphT);p.z=lerp(h.oz,h.z,morphT);p.life=1;
      if(dir==='up'){p.vx=(Math.random()-0.5)*1.0*pv;p.vy=(0.7+Math.random())*pv;p.vz=(Math.random()-0.5)*1.0*pv;}
      else if(dir==='down'){p.vx=(Math.random()-0.5)*0.5*pv;p.vy=-(0.4+Math.random())*pv;p.vz=(Math.random()-0.5)*0.5*pv;}
      else if(dir==='burst'){const d=Math.hypot(p.x,p.y,p.z)||1;p.vx=p.x/d*1.7*pv;p.vy=p.y/d*1.7*pv;p.vz=(p.z/d+0.4)*1.7*pv;}
      else{p.vx=(Math.random()-0.5)*0.8*pv;p.vy=(Math.random()-0.5)*0.8*pv;p.vz=(0.4+Math.random())*0.8*pv;}}
    p.x+=p.vx+(Math.random()-0.5)*st.turb*1.4; p.y+=p.vy+(Math.random()-0.5)*st.turb*1.4; p.z+=p.vz;
    ppos[i*3]=p.x;ppos[i*3+1]=p.y;ppos[i*3+2]=p.z;}
  pgeo.attributes.position.needsUpdate=true; pmat.color.copy(st.bright); pmat.opacity=0.3+st.glow*0.4;
  rig.rotation.y=Math.sin(clock*0.4)*0.5; rig.rotation.x=-0.02+Math.sin(clock*0.28)*0.04;
  composer.render();
}
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);composer.setSize(innerWidth,innerHeight);});
</script></body></html>"""


class QuantumFace3D(QWidget):
    """Real-time GPU 3-D voxel head (real anatomy + orb morph)."""

    available = WEBENGINE_OK
    # States where ORION collapses back to the glowing orb (dormant/idle).
    _ORB_STATES = {"STANDBY", "INITIALISING", "OFFLINE", "SHUTTING DOWN"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state_name = "STANDBY"
        self._built = False
        self._pending_emotion: tuple[str, dict] | None = None
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
        self.view.setHtml(FACE_HTML)
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
        self._js(f"window.orionAmplitude&&orionAmplitude({max(0.0, min(1.0, float(value))):.3f})")

    def set_speaking(self, active: bool) -> None:
        pass

    def set_state(self, state: str) -> None:
        self.state_name = str(state or "STANDBY").upper()
        self._js(f"window.orionLabel&&orionLabel({json.dumps('ORION · ' + self.state_name)})")
        # Form into the face when present; collapse to the orb when dormant.
        morph = 0.0 if self.state_name in self._ORB_STATES else 1.0
        self._js(f"window.orionMorph&&orionMorph({morph})")

    def apply_emotion(self, name: str, params: Any) -> None:
        if not isinstance(params, dict):
            return
        if not self._built:
            self._pending_emotion = (name, params)
            return
        self._js(f"window.orionEmotion&&orionEmotion({json.dumps(params)})")

    def _js(self, code: str) -> None:
        if self.view is not None:
            try:
                self.view.page().runJavaScript(code)
            except Exception:
                pass
