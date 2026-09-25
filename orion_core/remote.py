"""
Remote gateway — pairing-authenticated uplink to ORION's brain for browser
and mobile.

This is the backbone for **Cloud ORION** and the **private Android app**. It
serves an installable Progressive Web App (add-to-home-screen → behaves like a
native app, works over mobile data, no store required) and a JSON API. Remote
turns are answered by the *full* brain — the language-model router grounded in
ORION's identity and conversation memory, with the tool-capable offline
``LocalBrain`` as the fallback — so a phone gets the real ORION, not a stub.
Every turn is logged into the same episodic memory as desktop conversations, so
context stays synchronised across devices.

Disabled by default. Enable with ``ORION_REMOTE_ACCESS=1``; it then binds
``ORION_REMOTE_HOST`` (default ``0.0.0.0``) on ``ORION_REMOTE_PORT`` (default
``8765``).

AUTHENTICATION (Cloud Roadmap Phase C1 — replaces the old static token):

    1. The desktop begins a pairing window (``gateway.begin_pairing()``; a
       window opens automatically on first run while no device is registered,
       and the code is written to the log).
    2. The phone/browser POSTs the one-time code to ``/v1/auth/pair`` and
       receives a revocable per-device **refresh token** (stored hashed in
       ``config/remote_devices.json``).
    3. The device exchanges the refresh token at ``/v1/auth/token`` for a
       15-minute HMAC-signed **access token**, which authenticates every
       ``/api/chat`` and ``/v1/agent/tasks`` call (Authorization: Bearer).

The old ``config/remote_token.txt`` static token is no longer accepted.

Remote tool execution goes through ``RemoteAgentQueue``, whose capability
manifest whitelists a handful of read-only tools; ``desktop_control``,
``file_controller``, mail sending and every other host-mutating tool are
hard-denied regardless of the whitelist contents.

SECURITY: for internet exposure put this behind an HTTPS reverse proxy (Caddy /
nginx / Cloudflare Tunnel) — see ``deploy/README_ORACLE_CLOUD.md``. The gateway
adds hardening headers, per-client rate limiting and constant-time credential
comparisons, but it does not terminate TLS itself.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent
from .remote_capability import (
    RemoteConfirmationRegistry,
    RemoteToolGate,
    describe_action,
)
from .remote_endpoints import detect_endpoints
from .security import SecuritySanitiser, SecurityViolation
from .utils import utc_stamp
from .atomic_io import atomic_write_text

# ── installable PWA shell ─────────────────────────────────────────────────────
#
# The phone gets the REAL ORION: the same Three.js voxel orb/face the desktop
# renders (served at /face, mirrored live over /api/events), voice output via
# the browser's speech synthesis, and voice input via speech recognition where
# the context is secure (HTTPS or localhost).  Chat still flows through the
# full memory-synced brain.

REMOTE_PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#050508">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="ORION">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-192.png">
<title>O.R.I.O.N.</title>
<style>
:root{--pri:#ff1a3c;--pri-dim:#991024;--cyan:#00e5ff;--bg:#050508;--panel:#0f0f14}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(120% 80% at 70% 0%,#12060b 0%,var(--bg) 60%);
 color:#fff;font-family:'Segoe UI',system-ui,Arial,sans-serif;display:flex;flex-direction:column;
 height:100dvh;overflow:hidden}
header{padding:calc(8px + env(safe-area-inset-top)) 16px 8px;border-bottom:1px solid #2a1118;
 background:linear-gradient(#12121a,#0a0a10);display:flex;align-items:center;gap:10px}
h1{margin:0;font-size:15px;letter-spacing:2px}
.sub{color:#a9a9b2;font-size:10px;letter-spacing:.5px}
#dot{margin-left:auto;font-size:10px;color:#3ddc84;white-space:nowrap}
#stage{position:relative;flex:0 0 38dvh;min-height:180px;border-bottom:1px solid #2a1118;background:#03040a}
#face{width:100%;height:100%;border:0;display:block}
#state{position:absolute;top:8px;left:12px;font-size:10px;letter-spacing:3px;color:#ff5c73;
 text-transform:uppercase;opacity:.85;pointer-events:none}
#log{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;
 -webkit-overflow-scrolling:touch}
.msg{max-width:88%;padding:10px 13px;border-radius:14px;font-size:15px;line-height:1.5;white-space:pre-wrap;
 word-wrap:break-word;animation:rise .18s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.user{align-self:flex-end;background:linear-gradient(#b3132b,#7c0d1e)}
.orion{align-self:flex-start;background:#14141d;border:1px solid #2a1118}
.orion.think{color:#a9a9b2;font-style:italic}
.meta{align-self:center;color:#5c5c68;font-size:11px;text-align:center}
form{display:flex;gap:8px;padding:10px 10px calc(10px + env(safe-area-inset-bottom));
 border-top:1px solid #2a1118;background:#0a0a10}
input{flex:1;background:#050508;color:#fff;border:1px solid var(--pri-dim);border-radius:12px;
 padding:12px;font-size:16px;outline:none;min-width:0}
input:focus{border-color:var(--pri)}
button{background:linear-gradient(#b3132b,#7c0d1e);color:#fff;border:1px solid var(--pri);
 border-radius:12px;padding:0 16px;font-weight:800;font-size:15px}
button:active{background:#66091a}
button.icon{padding:0 13px;font-size:17px}
button.off{opacity:.45;border-color:#3a1a22}
#mic.listening{animation:pulse 1s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,26,60,.6)}50%{box-shadow:0 0 0 8px rgba(255,26,60,0)}}
</style></head><body>
<header><div><h1>O.R.I.O.N.</h1>
<div class="sub">REMOTE UPLINK · MEMORY-SYNCED</div></div><div id="dot">● online</div></header>
<div id="stage"><iframe id="face" src="/face" title="ORION"></iframe><div id="state">standby</div></div>
<div id="log"></div>
<form id="f">
 <input id="m" placeholder="Message ORION…" autocomplete="off" autocapitalize="sentences">
 <button type="button" id="mic" class="icon" title="Speak to ORION">🎙</button>
 <button type="button" id="spk" class="icon" title="Voice replies">🔊</button>
 <button>SEND</button>
</form>
<script>
const log=document.getElementById('log'),dot=document.getElementById('dot');
const face=document.getElementById('face'),stateEl=document.getElementById('state');
let dev=localStorage.getItem('orion_device')||'';
let refresh=localStorage.getItem('orion_refresh')||'';
let access=localStorage.getItem('orion_access')||'';
let accessExp=+(localStorage.getItem('orion_access_exp')||0);
function add(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;
 log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
add('meta','Connected to ORION. Conversations sync into his memory.');
function faceMsg(m){try{face.contentWindow.postMessage(m,'*');}catch(e){}}
async function pair(){
 const code=(prompt('Enter the one-time pairing code shown on the ORION desktop')||'').trim();
 if(!code)return false;
 try{const r=await fetch('/v1/auth/pair',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({code,device_name:(navigator.userAgent||'device').slice(0,60)})});
  if(!r.ok)return false;const j=await r.json();
  dev=j.device_id;refresh=j.refresh_token;
  localStorage.setItem('orion_device',dev);localStorage.setItem('orion_refresh',refresh);
  learnEndpoints(j.endpoints);
  return true;}catch(e){return false;}}
// Hand the endpoint list the desktop reported to the native app, so it can
// auto-connect (Tailscale name works home + away) without an IP ever typed.
function learnEndpoints(eps){
 if(eps&&window.OrionNative&&OrionNative.saveEndpoints){
  try{OrionNative.saveEndpoints(JSON.stringify(eps));}catch(e){}}}
async function refreshAccess(){
 if(!dev||!refresh)return false;
 try{const r=await fetch('/v1/auth/token',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({device_id:dev,refresh_token:refresh})});
  if(!r.ok)return false;const j=await r.json();
  access=j.access_token;accessExp=Date.now()+Math.max(30,(j.expires_in-60))*1000;
  localStorage.setItem('orion_access',access);localStorage.setItem('orion_access_exp',accessExp);
  learnEndpoints(j.endpoints);
  return true;}catch(e){return false;}}
async function ensureAccess(){
 if(access&&Date.now()<accessExp)return true;
 if(await refreshAccess())return true;
 if(await pair())return await refreshAccess();
 return false;}
async function health(){try{const r=await fetch('/api/health');const j=await r.json();
 dot.textContent=(j.mode?('● '+j.mode.toLowerCase()):'● online');dot.style.color='#3ddc84';}
 catch(e){dot.textContent='○ offline';dot.style.color='#ffb020';}}
// Battery: this page lives on a phone, often for hours, frequently with the
// screen off. A fixed 20 s poll that never stops wakes the radio ~180 times an
// hour for nothing. Two changes: it stops entirely while the page is hidden
// (backgrounded or screen off), and it backs off while the SSE stream is
// connected -- an open EventSource already IS the liveness signal, so polling
// on top of it is asking a question we can already see the answer to.
// streamLive rather than reading `es` directly: `es` is declared with `let`
// further down this script, so touching it from here would hit the temporal
// dead zone and throw ReferenceError before the page finished loading.
var streamLive=false;
let healthTimer=null;
function healthInterval(){return streamLive?90000:20000;}
function scheduleHealth(){clearInterval(healthTimer);
 healthTimer=setInterval(health,healthInterval());}
function startHealth(){health();scheduleHealth();}
function stopHealth(){clearInterval(healthTimer);healthTimer=null;}
document.addEventListener('visibilitychange',()=>{
 if(document.hidden){stopHealth();}
 else{startHealth();}});
startHealth();

// ── voice output: ORION speaks his replies aloud (toggleable) ───────────────
const spkBtn=document.getElementById('spk');
let voiceOn=localStorage.getItem('orion_voice')!=='0';
let ttsAmpTimer=null;
function paintSpk(){spkBtn.classList.toggle('off',!voiceOn);}
paintSpk();
spkBtn.addEventListener('click',()=>{voiceOn=!voiceOn;
 localStorage.setItem('orion_voice',voiceOn?'1':'0');paintSpk();
 if(!voiceOn&&window.speechSynthesis)speechSynthesis.cancel();});
function pickVoice(){if(!window.speechSynthesis)return null;
 const vs=speechSynthesis.getVoices();
 return vs.find(v=>/en[-_]GB/i.test(v.lang)&&/male|daniel|arthur|ryan/i.test(v.name))
  ||vs.find(v=>/en[-_]GB/i.test(v.lang))||vs.find(v=>/^en/i.test(v.lang))||null;}
function speak(text){
 if(!voiceOn||!window.speechSynthesis||!text)return;
 speechSynthesis.cancel();
 const u=new SpeechSynthesisUtterance(text.slice(0,1200));
 const v=pickVoice(); if(v)u.voice=v;
 u.rate=1.02;u.pitch=0.92;
 // 90 ms is ~11 frames a second of pure decoration. Worth it while you are
 // looking at the orb; pure battery burn when the phone is in a pocket, so it
 // does not run while the page is hidden.
 u.onstart=()=>{clearInterval(ttsAmpTimer);
  if(document.hidden)return;
  ttsAmpTimer=setInterval(()=>faceMsg({amp:0.25+Math.random()*0.6}),90);};
 u.onend=u.onerror=()=>{clearInterval(ttsAmpTimer);ttsAmpTimer=null;faceMsg({amp:0});};
 speechSynthesis.speak(u);}

// ── voice input ─────────────────────────────────────────────────────────────
// Two paths, chosen by what the device actually supports:
//   A. Browser SpeechRecognition (Chrome/Edge over HTTPS) — instant, on-device.
//   B. Record raw audio and let the desktop transcribe it (/api/transcribe).
// Path B is what makes voice work INSIDE the Android app: its WebView grants the
// microphone (getUserMedia) but has no SpeechRecognition API, so path A is blank
// there. Both need a secure context, which is why the desktop must serve HTTPS.
const micBtn=document.getElementById('mic');
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
const AC=window.AudioContext||window.webkitAudioContext;
const canRecord=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia&&AC);
if(!window.isSecureContext||(!SR&&!canRecord)){micBtn.style.display='none';}
else{
 function submitText(t){t=(t||'').trim();if(!t)return;
  document.getElementById('m').value=t;
  document.getElementById('f').dispatchEvent(new Event('submit'));}
 // Path A ---------------------------------------------------------------------
 let rec=null;
 function useSR(){
  if(rec){rec.stop();return;}
  rec=new SR();rec.lang='en-GB';rec.interimResults=false;rec.maxAlternatives=1;
  micBtn.classList.add('listening');
  rec.onresult=e=>submitText(e.results[0][0].transcript||'');
  rec.onend=rec.onerror=()=>{micBtn.classList.remove('listening');rec=null;};
  try{rec.start();}catch(e){micBtn.classList.remove('listening');rec=null;}}
 // Path B ---------------------------------------------------------------------
 let recording=false,audioCtx=null,srcNode=null,procNode=null,stream=null,frames=[];
 async function startRecord(){
  try{stream=await navigator.mediaDevices.getUserMedia({audio:true});}
  catch(e){add('orion','Microphone blocked — allow it for ORION in Android settings.');return;}
  audioCtx=new AC();try{await audioCtx.resume();}catch(e){}
  srcNode=audioCtx.createMediaStreamSource(stream);
  procNode=audioCtx.createScriptProcessor(4096,1,1);
  frames=[];
  procNode.onaudioprocess=e=>{frames.push(new Float32Array(e.inputBuffer.getChannelData(0)));};
  srcNode.connect(procNode);procNode.connect(audioCtx.destination);  // output stays silent
  recording=true;micBtn.classList.add('listening');}
 async function stopRecord(){
  recording=false;micBtn.classList.remove('listening');
  const rate=audioCtx?audioCtx.sampleRate:48000;
  try{procNode.disconnect();srcNode.disconnect();
   stream.getTracks().forEach(t=>t.stop());audioCtx.close();}catch(e){}
  const wav=encodeWav(frames,rate);frames=[];
  if(!wav){add('orion','I didn’t catch anything.');return;}
  const thinking=add('orion think','… hearing you');
  if(!(await ensureAccess())){thinking.remove();add('orion','Pairing required.');return;}
  try{const r=await fetch('/api/transcribe',{method:'POST',
    headers:{'Content-Type':'audio/wav','Authorization':'Bearer '+access},body:wav});
   const j=await r.json();thinking.remove();
   if(j.ok&&j.text){submitText(j.text);}
   else{add('orion',j.error?('Voice: '+j.error):'I didn’t catch that.');}}
  catch(e){thinking.remove();add('orion','Voice link fault: '+e);}}
 // Down-sample the captured audio to the 16 kHz mono 16-bit PCM WAV the desktop
 // transcriber expects, built by hand so it needs no library and runs anywhere.
 function encodeWav(buffers,inRate){
  let len=0;for(const b of buffers)len+=b.length;if(!len)return null;
  const flat=new Float32Array(len);let o=0;for(const b of buffers){flat.set(b,o);o+=b.length;}
  const outRate=16000,ratio=inRate/outRate,outLen=Math.max(0,Math.floor(flat.length/ratio));
  if(outLen<1600)return null;                    // < 0.1 s — a stray tap, not speech
  const pcm=new Int16Array(outLen);
  for(let i=0;i<outLen;i++){const s=Math.max(-1,Math.min(1,flat[Math.floor(i*ratio)]||0));
   pcm[i]=s<0?s*0x8000:s*0x7fff;}
  const buf=new ArrayBuffer(44+pcm.length*2),v=new DataView(buf);
  const ws=(off,str)=>{for(let i=0;i<str.length;i++)v.setUint8(off+i,str.charCodeAt(i));};
  ws(0,'RIFF');v.setUint32(4,36+pcm.length*2,true);ws(8,'WAVE');ws(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,outRate,true);v.setUint32(28,outRate*2,true);v.setUint16(32,2,true);
  v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,pcm.length*2,true);
  let p=44;for(let i=0;i<pcm.length;i++,p+=2)v.setInt16(p,pcm[i],true);
  return buf;}
 micBtn.addEventListener('click',()=>{
  if(SR){useSR();return;}
  if(recording)stopRecord();else startRecord();});}

// ── live state mirror: the phone orb tracks the desktop orb in real time ────
let es=null,esBackoff=2000;
function stateToMorph(s){return /STANDBY|OFFLINE|SHUTTING/i.test(s)?0:1;}
async function openEvents(){
 if(es){es.close();es=null;}
 if(!(await ensureAccess()))return;
 try{es=new EventSource('/api/events?token='+encodeURIComponent(access));}catch(e){return;}
 es.onopen=()=>{esBackoff=2000;streamLive=true;scheduleHealth();};
 es.addEventListener('error',()=>{streamLive=false;scheduleHealth();});
 es.onmessage=ev=>{let d;try{d=JSON.parse(ev.data);}catch(e){return;}
  if(d.type==='state'){stateEl.textContent=(d.value||'').toLowerCase();
   faceMsg({morph:stateToMorph(d.value||'')});}
  else if(d.type==='amplitude'){if(!ttsAmpTimer)faceMsg({amp:d.value});}
  else if(d.type==='speaking'){if(!d.value&&!ttsAmpTimer)faceMsg({amp:0});}
  else if(d.type==='emotion'&&d.params){faceMsg({emotion:d.params});}
  else if(d.type==='thought'&&d.text){add('orion think','◈ '+d.text);}
  else if(d.type==='action'&&d.action){doAction(d.action);}
  else if(d.type==='confirm'&&d.id){showConfirm(d);}
  else if(d.type==='confirm_result'&&d.id&&d.reply){add('orion',d.reply);}};
 es.onerror=()=>{if(es){es.close();es=null;}
  access='';setTimeout(openEvents,esBackoff);esBackoff=Math.min(30000,esBackoff*1.7);};}
refreshAccess().then(openEvents);

// ── native phone hand-off ──────────────────────────────────────────────────
// When ORION asks the phone to call / text / navigate, use the app's native
// bridge (window.OrionNative) if we're inside the ORION app; otherwise render a
// tappable link so it still works in a plain browser. Nothing is dialled or
// sent without the user's own final tap.
function actLabel(a){
 if(a.kind==='call')return 'Call '+(a.number||'');
 if(a.kind==='sms')return 'Text '+(a.number||'');
 if(a.kind==='email')return 'Email '+(a.to||'');
 if(a.kind==='navigate'||a.kind==='map')return 'Navigate to '+(a.query||'');
 if(a.kind==='openUrl')return 'Open '+(a.url||'');
 if(a.kind==='share')return 'Share';
 return a.kind;}
function actHref(a){
 if(a.kind==='call')return 'tel:'+(a.number||'');
 if(a.kind==='sms')return 'smsto:'+(a.number||'')+(a.body?('?body='+encodeURIComponent(a.body)):'');
 if(a.kind==='email')return 'mailto:'+encodeURIComponent(a.to||'')+'?subject='+encodeURIComponent(a.subject||'')+'&body='+encodeURIComponent(a.body||'');
 if(a.kind==='navigate'||a.kind==='map')return 'geo:0,0?q='+encodeURIComponent(a.query||'');
 if(a.kind==='openUrl')return a.url||'';
 return '';}
function doAction(a){
 if(!a||!a.kind)return;
 var N=window.OrionNative;
 if(N){try{
   if(a.kind==='call')N.call(a.number||'');
   else if(a.kind==='sms')N.sms(a.number||'',a.body||'');
   else if(a.kind==='email')N.email(a.to||'',a.subject||'',a.body||'');
   else if(a.kind==='navigate')N.navigate(a.query||'');
   else if(a.kind==='map')N.map(a.query||'');
   else if(a.kind==='openUrl')N.openUrl(a.url||'');
   else if(a.kind==='share')N.share(a.text||'');
   add('orion think','⇢ '+actLabel(a));
   return;
  }catch(e){}}
 var href=actHref(a);
 var m=add('orion','');
 if(href){var link=document.createElement('a');link.href=href;
  link.textContent=actLabel(a);link.rel='noopener';m.textContent='';m.appendChild(link);}
 else{m.textContent=actLabel(a);}}

// ── on-phone approval for irreversible/external actions (full-parity gate) ──
// ORION runs read/benign actions at once; anything that sends, deletes, buys or
// powers something off arrives here as a card you tap to approve — the desktop's
// confirmation gate, where you actually are.
const seenConfirms={};
async function sendConfirm(id,token,approve,card){
 if(card)card.remove();
 if(!(await ensureAccess())){add('orion','I lost the link before I could act.');return;}
 try{const r=await fetch('/api/confirm',{method:'POST',
   headers:{'Content-Type':'application/json','Authorization':'Bearer '+access},
   body:JSON.stringify({id:id,token:token,decision:approve?'approve':'deny'})});
  const j=await r.json();
  if(j&&j.reply)add('orion',j.reply);
  else if(!approve)add('orion','Left it.');}
 catch(e){add('orion','Approval failed: '+e);}}
function showConfirm(d){
 if(seenConfirms[d.id])return;seenConfirms[d.id]=1;
 const card=add('orion confirm','Approve: '+(d.summary||d.tool||'this action')+'?');
 const row=document.createElement('div');row.style.cssText='margin-top:8px';
 const yes=document.createElement('button');yes.textContent='Approve';
 yes.style.cssText='margin-right:10px;padding:9px 18px;border:0;border-radius:9px;background:#3ddc84;color:#04120a;font-weight:700';
 const no=document.createElement('button');no.textContent='Deny';
 no.style.cssText='padding:9px 18px;border:0;border-radius:9px;background:#3a2a2e;color:#ffb0bd;font-weight:700';
 yes.onclick=()=>sendConfirm(d.id,d.token,true,card);
 no.onclick=()=>sendConfirm(d.id,d.token,false,card);
 row.appendChild(yes);row.appendChild(no);card.appendChild(row);}

document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const inp=document.getElementById('m');const text=inp.value.trim();if(!text)return;
 inp.value='';add('user',text);const thinking=add('orion think','…');
 faceMsg({morph:1});stateEl.textContent='processing';
 if(!(await ensureAccess())){thinking.remove();
  add('orion','Pairing required. Start a pairing window on the desktop and try again.');return;}
 try{let res=await fetch('/api/chat',{method:'POST',
   headers:{'Content-Type':'application/json','Authorization':'Bearer '+access},
   body:JSON.stringify({message:text})});
  if(res.status===401){access='';localStorage.removeItem('orion_access');
   if(await ensureAccess()){res=await fetch('/api/chat',{method:'POST',
    headers:{'Content-Type':'application/json','Authorization':'Bearer '+access},
    body:JSON.stringify({message:text})});}}
  const data=await res.json();thinking.remove();
  if(res.status===401){add('orion','This device is no longer authorised. Pair it again from the desktop.');return;}
  if(res.status===429){add('orion','Easy — too many requests at once. One moment.');return;}
  const reply=data.ok?data.reply:('Fault: '+(data.error||'unknown'));
  add('orion',reply);
  if(data.ok)speak(reply);
  stateEl.textContent='listening';}
 catch(err){thinking.remove();add('orion','Link fault: '+err);stateEl.textContent='standby';}
});
if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
</script></body></html>"""


# Served when the desktop's Three.js face module is unavailable (a barebones
# cloud node): a lightweight 2-D canvas orb with the same crimson identity.
ORB_FALLBACK_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{margin:0;height:100%;background:#03040a;overflow:hidden}canvas{display:block}</style>
</head><body><canvas id="c"></canvas><script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let amp=0,tgAmp=0,t=0;
function fit(){cv.width=innerWidth;cv.height=innerHeight;}
fit();addEventListener('resize',fit);
addEventListener('message',e=>{const d=e.data||{};
 if(d.amp!==undefined)tgAmp=Math.max(0,Math.min(1,d.amp));});
const P=[...Array(90)].map(()=>({a:Math.random()*6.283,r:0.55+Math.random()*0.45,
 s:0.002+Math.random()*0.006,ph:Math.random()*6.283}));
(function draw(){requestAnimationFrame(draw);t+=0.016;amp+=(tgAmp-amp)*0.3;
 const w=cv.width,h=cv.height,cx=w/2,cy=h/2,R=Math.min(w,h)*0.26*(1+amp*0.12+0.02*Math.sin(t*1.2));
 ctx.fillStyle='#03040a';ctx.fillRect(0,0,w,h);
 let g=ctx.createRadialGradient(cx,cy,R*0.1,cx,cy,R*2.2);
 g.addColorStop(0,'rgba(255,26,60,'+(0.30+amp*0.35)+')');g.addColorStop(1,'rgba(255,26,60,0)');
 ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R*2.2,0,6.283);ctx.fill();
 g=ctx.createRadialGradient(cx-R*0.3,cy-R*0.35,R*0.05,cx,cy,R);
 g.addColorStop(0,'#fff');g.addColorStop(0.35,'#ff4d68');g.addColorStop(1,'#4a0812');
 ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R,0,6.283);ctx.fill();
 ctx.strokeStyle='rgba(0,229,255,'+(0.5+amp*0.3)+')';ctx.lineWidth=2;
 ctx.beginPath();ctx.arc(cx,cy,R*0.82,-1.2+t*0.7,0.4+t*0.7);ctx.stroke();
 ctx.fillStyle='rgba(255,143,160,0.8)';
 for(const p of P){p.a+=p.s*(1+amp*2);
  const rr=R*(1.1+p.r*0.5+0.05*Math.sin(t*2+p.ph));
  ctx.beginPath();ctx.arc(cx+Math.cos(p.a)*rr,cy+Math.sin(p.a)*rr*0.92,1.4,0,6.283);ctx.fill();}
})();
</script></body></html>"""


# postMessage bridge injected into the desktop face page when it is served to
# a phone: maps {amp, morph, emotion, label} messages onto the page's
# window.orion* control functions so the parent PWA can drive it.
_FACE_BRIDGE = r"""<script>
addEventListener('message',e=>{const d=e.data||{};try{
 if(d.amp!==undefined&&window.orionAmplitude)window.orionAmplitude(d.amp);
 if(d.spectrum!==undefined&&window.orionSpectrum)window.orionSpectrum(d.spectrum.low,d.spectrum.mid,d.spectrum.high);
 if(d.morph!==undefined&&window.orionMorph)window.orionMorph(d.morph);
 if(d.emotion&&window.orionEmotion)window.orionEmotion(d.emotion);
 if(d.label&&window.orionLabel)window.orionLabel(d.label);
}catch(_){}});
</script>"""

_MANIFEST = {
    "name": "O.R.I.O.N.",
    "short_name": "ORION",
    "description": "Private uplink to ORION — Open Resolution Intelligence Overt Network.",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#050508",
    "theme_color": "#050508",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
    ],
}

# Network-first for the API, cache-first shell so the app opens offline.
_SERVICE_WORKER = r"""
const SHELL='orion-shell-v3';
const ASSETS=['/','/manifest.webmanifest','/icon-192.png','/icon-512.png'];
self.addEventListener('install',e=>{self.skipWaiting();
 e.waitUntil(caches.open(SHELL).then(c=>c.addAll(ASSETS)).catch(()=>{}));});
self.addEventListener('activate',e=>{e.waitUntil(Promise.all([
 self.clients.claim(),
 caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==SHELL).map(k=>caches.delete(k)))),
]));});
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);
 if(u.pathname.startsWith('/api/')){return;}  // always live
 // Page shell: NETWORK-FIRST so desktop-side updates (e.g. new voice code) land
 // immediately; fall back to cache only when the phone is actually offline.
 // Everything else stays cache-first for speed on cellular.
 if(e.request.mode==='navigate'||u.pathname==='/'){
  e.respondWith(fetch(e.request).then(r=>{
   if(r&&r.ok){const cp=r.clone();caches.open(SHELL).then(c=>c.put('/',cp)).catch(()=>{});}
   return r;}).catch(()=>caches.match('/')));
  return;}
 e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)).catch(()=>caches.match('/')));});
"""


# ── authentication: pairing → refresh token → short-lived access token ───────

class RemoteAuthManager:
    """
    Device pairing and token issue/verify (Cloud Roadmap Phase C1).

    • ``begin_pairing()`` creates a one-time code (shown on the desktop) with
      a short expiry window.
    • ``complete_pairing(code)`` consumes the code and registers a device:
      a random refresh token is returned once and stored only as a SHA-256
      hash in ``config/remote_devices.json`` (revocable per device).
    • ``issue_access()`` exchanges a valid refresh token for a 15-minute
      HMAC-SHA256-signed access token; ``verify_access()`` checks signature,
      expiry and revocation. All credential comparisons are constant-time.
    """

    ACCESS_TTL_S = 15 * 60
    PAIR_TTL_S = 10 * 60

    def __init__(self, config_dir: Path | None = None, log: Any = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self._log = log or (lambda _msg: None)
        self._devices_path = self.config_dir / "remote_devices.json"
        self._secret = self._load_secret()
        self._pairings: dict[str, float] = {}      # sha256(code) → expiry epoch
        self._devices = self._load_devices()

    # ── secret + device registry persistence ─────────────────────────────────

    def _load_secret(self) -> bytes:
        secret_path = self.config_dir / "remote_secret.key"
        try:
            if secret_path.exists():
                existing = secret_path.read_text(encoding="utf-8").strip()
                if len(existing) >= 32:
                    return bytes.fromhex(existing)
        except Exception:
            pass
        secret = secrets.token_bytes(32)
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            secret_path.write_text(secret.hex(), encoding="utf-8")
        except Exception:
            pass
        return secret

    def _load_devices(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._devices_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass
        return {}

    def _save_devices(self) -> None:
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._devices_path,
                json.dumps(self._devices, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── pairing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def begin_pairing(self) -> str:
        """Open a pairing window; returns the one-time code to display."""
        raw = secrets.token_hex(4).upper()
        code = f"{raw[:4]}-{raw[4:]}"
        now = time.time()
        # Drop expired windows; keep at most a couple of concurrent codes.
        self._pairings = {h: exp for h, exp in self._pairings.items() if exp > now}
        self._pairings[self._hash(code)] = now + self.PAIR_TTL_S
        self._log(f"REMOTE: pairing window open for {self.PAIR_TTL_S // 60} min — code {code}")
        return code

    def pairing_active(self) -> bool:
        now = time.time()
        return any(exp > now for exp in self._pairings.values())

    def complete_pairing(self, code: str, device_name: str = "") -> Optional[tuple[str, str]]:
        """Consume a one-time code → (device_id, refresh_token) or None."""
        supplied = self._hash(str(code or "").strip().upper())
        now = time.time()
        matched = ""
        for stored, expiry in self._pairings.items():
            # Constant-time comparison for every candidate — no early exit.
            if hmac.compare_digest(supplied, stored) and expiry > now:
                matched = stored
        if not matched:
            return None
        del self._pairings[matched]                 # single-use
        device_id = uuid.uuid4().hex
        refresh_token = secrets.token_urlsafe(32)
        self._devices[device_id] = {
            "name": str(device_name or "device")[:80],
            "refresh_sha256": self._hash(refresh_token),
            "created": utc_stamp(),
            "revoked": False,
        }
        self._save_devices()
        self._log(f"REMOTE: paired device '{self._devices[device_id]['name']}' ({device_id[:8]}…)")
        return device_id, refresh_token

    # ── access tokens ─────────────────────────────────────────────────────────

    def issue_access(self, device_id: str, refresh_token: str) -> Optional[tuple[str, int]]:
        """Refresh → (access_token, expires_in_s), or None when unauthorised."""
        device = self._devices.get(str(device_id or ""))
        if not device or device.get("revoked"):
            return None
        expected = str(device.get("refresh_sha256") or "")
        if not hmac.compare_digest(self._hash(str(refresh_token or "")), expected):
            return None
        payload = f"{device_id}:{int(time.time()) + self.ACCESS_TTL_S}"
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
        signature = hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}", self.ACCESS_TTL_S

    def verify_access(self, token: str) -> Optional[str]:
        """Validate signature, expiry and revocation → device_id or None."""
        token = str(token or "")
        if "." not in token:
            return None
        encoded, _, signature = token.rpartition(".")
        try:
            payload = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        except Exception:
            return None
        expected = hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        device_id, _, expiry_text = payload.rpartition(":")
        try:
            if time.time() > int(expiry_text):
                return None
        except ValueError:
            return None
        device = self._devices.get(device_id)
        if not device or device.get("revoked"):
            return None
        return device_id

    # ── device administration ─────────────────────────────────────────────────

    def list_devices(self) -> list[dict[str, Any]]:
        return [
            {"device_id": device_id, "name": info.get("name", ""),
             "created": info.get("created", ""), "revoked": bool(info.get("revoked"))}
            for device_id, info in self._devices.items()
        ]

    def has_devices(self) -> bool:
        return any(not info.get("revoked") for info in self._devices.values())

    def revoke_device(self, device_id: str) -> bool:
        device = self._devices.get(str(device_id or ""))
        if not device:
            return False
        device["revoked"] = True
        self._save_devices()
        self._log(f"REMOTE: device {device_id[:8]}… revoked.")
        return True


# ── remote tool execution: capability-manifested queue ───────────────────────

# Tools a remote-origin request may execute — read-only status/recall surface.
REMOTE_TOOL_WHITELIST = frozenset({
    "query_intelligence",
    "recall_conversation",
    "resource_status",
    "token_usage",
    "diagnostics",
    "patch_notes",
})

# Tools that must NEVER run from a remote origin, regardless of what the
# whitelist says (defence in depth: checked first, so a future whitelist edit
# cannot accidentally expose them). External/mutating actions always require
# the desktop's own approval gates.
REMOTE_HARD_DENY = frozenset({
    "desktop_control", "file_controller", "outlook_mail", "notion_workspace",
    "web_control", "web_automation", "self_repair", "shutdown_orion",
    "restart_orion", "process_governor", "dev_workbench", "execute_plan",
    "forge", "mcp", "peripherals", "messaging", "clipboard_operate",
    "organise_files", "backup", "security_watch", "interface_control",
    "gaming", "entertainment", "audio_studio",
})

# A real check, not an assert: `python -O` strips assertions, and PyInstaller
# can be told to build that way for size. A hard-denied tool that is also
# whitelisted would then ship silently permitted, which is the one direction
# this must never fail in.
if REMOTE_TOOL_WHITELIST & REMOTE_HARD_DENY:
    raise RuntimeError(
        "remote whitelist and hard-deny overlap: "
        + ", ".join(sorted(REMOTE_TOOL_WHITELIST & REMOTE_HARD_DENY))
        + ". A hard-denied tool must never also be whitelisted.")


class RemoteAgentQueue:
    """
    Runs whitelisted dispatcher tools for authenticated remote devices
    (Cloud Roadmap Phase C4, minimal seed).

    Hard-denied tools are refused before the whitelist is even consulted, so
    a remote request cannot reach ``desktop_control``, ``file_controller``
    writes or mail sending even by asking for them by name.
    """

    def __init__(self, dispatcher: Any = None, log: Any = None) -> None:
        self.dispatcher = dispatcher
        self._log = log or (lambda _msg: None)
        self.tasks: deque[dict[str, Any]] = deque(maxlen=50)
        self._owners: dict[str, str] = {}

    def _remember(self, task: dict[str, Any], device_id: str) -> None:
        self.tasks.append(task)
        self._owners[task["id"]] = device_id
        self._owners = {item["id"]: self._owners[item["id"]]
                        for item in self.tasks if item["id"] in self._owners}

    @staticmethod
    def allowed(tool: str) -> bool:
        tool = str(tool or "")
        return tool not in REMOTE_HARD_DENY and tool in REMOTE_TOOL_WHITELIST

    async def run(self, tool: str, args: dict[str, Any] | None,
                  device_id: str = "") -> dict[str, Any]:
        tool = str(tool or "").strip()
        task: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12], "tool": tool,
            "device": device_id[:8], "at": utc_stamp(),
        }
        if tool in REMOTE_HARD_DENY:
            task.update(status="denied",
                        error="this capability is desktop-only and cannot run remotely")
            self._remember(task, device_id)
            self._log(f"REMOTE: hard-denied remote tool request '{tool}'")
            return task
        if tool not in REMOTE_TOOL_WHITELIST:
            task.update(status="denied",
                        error="tool is not on the remote capability whitelist")
            self._remember(task, device_id)
            return task
        if self.dispatcher is None:
            task.update(status="unavailable", error="no dispatcher wired on this node")
            self._remember(task, device_id)
            return task
        try:
            result = await self.dispatcher.dispatch(tool, dict(args or {}))
            task.update(status="done", ok=bool(result.ok), result=str(result.text)[:2000])
        except SecurityViolation as exc:
            task.update(status="denied", error=str(exc))
        except Exception as exc:
            task.update(status="failed", error=str(exc).splitlines()[0][:200])
        self._remember(task, device_id)
        return task

    def get(self, task_id: str, device_id: str) -> Optional[dict[str, Any]]:
        if not device_id or self._owners.get(task_id) != device_id:
            return None
        for task in self.tasks:
            if task.get("id") == task_id:
                return task
        return None


# The exact face iframe in REMOTE_PAGE_HTML, and its lightweight lite-mode
# stand-in: a self-contained CSS orb (no CDN modules, no WebGL). It keeps
# id="face" so the page's faceMsg() postMessage calls no-op harmlessly.
_FACE_IFRAME = '<iframe id="face" src="/face" title="ORION"></iframe>'
_LITE_ORB = (
    '<div id="face" style="display:grid;place-items:center;height:100%;width:100%">'
    '<div style="width:46vw;max-width:220px;aspect-ratio:1;border-radius:50%;'
    'background:radial-gradient(circle at 50% 40%,#ff5a6e 0%,#c2121f 55%,#2a0509 100%);'
    'box-shadow:0 0 55px rgba(255,42,70,.5);animation:orionlite 3.2s ease-in-out infinite">'
    '</div><style>@keyframes orionlite{0%,100%{transform:scale(.94);opacity:.85}'
    '50%{transform:scale(1.02);opacity:1}}</style></div>'
)


class _BodyTooLarge(ValueError):
    """A request body over the endpoint's size limit (answered with 413)."""


class RemoteGateway:
    """Pairing-authenticated aiohttp uplink answering through ORION's full brain."""

    # Per-client sliding-window rate limit.
    _RATE_MAX     = 30      # requests …
    _RATE_WINDOW  = 60.0    # … per this many seconds
    # Voice-input upload ceiling: ~4 min of 16 kHz mono 16-bit PCM WAV. Guards
    # the desktop against a phone streaming a huge blob at the transcriber.
    _MAX_AUDIO_BYTES = 8 * 1024 * 1024
    # JSON request ceiling. The app-wide limit above is sized for audio, so
    # without this every JSON endpoint — including the unauthenticated pairing
    # and token doors — would parse an 8 MB body. The largest real payload is
    # a 4,000-character chat message.
    _MAX_JSON_BYTES = 64 * 1024
    _UNSET = object()       # "transcriber not built yet" sentinel
    # How long teardown lets an in-flight request (a phone's chat turn, a
    # transcription) finish before aiohttp cancels it. aiohttp's own default is
    # 60 s per phase, and an open /api/events stream never finishes by itself,
    # so one connected phone held shutdown past the app's 25 s watchdog.
    _SHUTDOWN_GRACE_S = 2.0

    def __init__(
        self,
        router: Any,
        memory: MemoryAgent,
        bus: OrionBus,
        *,
        identity: Any | None = None,
        local_brain: Any | None = None,
        conversation: Any | None = None,
        telemetry: Any | None = None,
        dispatcher: Any | None = None,
        knowledge: Any | None = None,
        config_dir: Path | None = None,
    ) -> None:
        self.router       = router
        self.memory       = memory
        self.bus          = bus
        self.identity     = identity
        self.local_brain  = local_brain
        self.conversation = conversation
        self.telemetry    = telemetry
        self.dispatcher   = dispatcher
        self.host   = os.getenv("ORION_REMOTE_HOST", "0.0.0.0").strip() or "0.0.0.0"
        self.port   = int(os.getenv("ORION_REMOTE_PORT", "8765") or 8765)
        self.auth   = RemoteAuthManager(config_dir=config_dir, log=bus.log.emit)
        self.agent_queue = RemoteAgentQueue(dispatcher=dispatcher, log=bus.log.emit)
        # ── Full-parity remote (#12): a capability gate + on-phone confirmation.
        # Remote tool calls run through ``tool_gate`` — read/benign run at once,
        # irreversible/outward actions park as a confirmation pushed to the phone
        # (tap to approve), and code-execution tools are refused outright.
        self.confirmations = RemoteConfirmationRegistry()
        self.tool_gate = RemoteToolGate(
            dispatcher, self.confirmations,
            notify=self._push_confirm, log=bus.log.emit)
        # Give the conversational path a GATED brain so a spoken command on the
        # move actually executes (through the gate), and — crucially — so the
        # offline fallback can no longer reach the dispatcher ungated.
        if dispatcher is not None and knowledge is not None:
            try:
                from .local_brain import LocalBrain
                self.local_brain = LocalBrain(
                    bus, memory, self.tool_gate, knowledge, router, telemetry)
            except Exception as exc:
                bus.log.emit(f"REMOTE: gated brain unavailable ({exc}); "
                             "using the supplied brain.")
        self._web: Any    = None
        self._runner: Any = None
        self._hits: dict[str, deque[float]] = {}
        self._auth_hits: dict[str, deque[float]] = {}
        self._icons: dict[int, bytes] = {}
        self._started_at = time.time()
        # Live-state mirror: connected phones subscribe over SSE and receive
        # the same state/amplitude/emotion stream that drives the desktop orb.
        self._event_subs: set[Any] = set()
        self._event_devices: dict[Any, str] = {}
        self._closing = False               # set by stop(): SSE streams end
        self._amp_pending: float | None = None
        self._amp_task: Any = None
        self._last_state = "STANDBY"
        self._last_emotion: dict[str, Any] = {}
        self._face_html: str | None = None
        self._three: tuple[dict[str, Path], str] | None = None
        # The paired device that last reached the uplink: where a phone action
        # raised on the desktop (not by a phone's own request) is sent.
        self._last_device = ""
        # Set true once TLS is actually serving, so reported endpoint URLs carry
        # the right scheme. Cached endpoints avoid re-shelling to tailscale.
        self._https_enabled = False
        self._endpoints_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._endpoints_lock = asyncio.Lock()
        # Offline speech-to-text for phone voice input, built on first use (a
        # Whisper model is slow to load, so it is deferred until actually asked).
        self._transcriber: Any = self._UNSET

    # ── pairing front door (desktop side) ──────────────────────────────────────

    def begin_pairing(self) -> str:
        """Open a pairing window and return the one-time code to display."""
        return self.auth.begin_pairing()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def _build_app(self) -> Any:
        from aiohttp import web
        self._web = web

        @web.middleware
        async def security_mw(request: Any, handler: Any) -> Any:
            response = await handler(request)
            # Mobile-data saver: gzip text/JSON/HTML when the client accepts it.
            # The live SSE stream is already prepared (and must not be buffered),
            # so ``not prepared`` skips it cleanly.
            try:
                ctype = response.headers.get("Content-Type", "")
                if ("gzip" in request.headers.get("Accept-Encoding", "").lower()
                        and "event-stream" not in ctype
                        and not getattr(response, "prepared", False)
                        and hasattr(response, "enable_compression")):
                    response.enable_compression()
            except Exception:
                pass
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            # Default: refuse framing everywhere.  The single exception is /face,
            # which the uplink page frames from the SAME origin — that handler
            # sets its own SAMEORIGIN policy, so setdefault here leaves it intact.
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:",
            )
            return response

        # Raise the body ceiling above the default 1 MB so a phone's recorded
        # voice clip (up to _MAX_AUDIO_BYTES) reaches /api/transcribe and is
        # size-checked there, rather than being cut off by aiohttp first.
        app = web.Application(middlewares=[security_mw],
                              client_max_size=self._MAX_AUDIO_BYTES + 65536)
        app.router.add_get("/",                    self._handle_page)
        app.router.add_get("/face",                self._handle_face)
        app.router.add_get("/three/{version}/{path:.+}", self._handle_three)
        app.router.add_get("/manifest.webmanifest", self._handle_manifest)
        app.router.add_get("/sw.js",               self._handle_sw)
        app.router.add_get("/icon-192.png",        self._handle_icon)
        app.router.add_get("/icon-512.png",        self._handle_icon)
        app.router.add_get("/api/health",          self._handle_health)
        app.router.add_get("/api/events",          self._handle_events)
        app.router.add_post("/api/chat",           self._handle_chat)
        app.router.add_post("/api/transcribe",     self._handle_transcribe)
        app.router.add_post("/api/confirm",        self._handle_confirm)
        app.router.add_get("/api/confirmations",   self._handle_confirmations)
        app.router.add_get("/api/endpoints",       self._handle_endpoints)
        app.router.add_post("/v1/auth/pair",       self._handle_pair)
        app.router.add_post("/v1/auth/token",      self._handle_token)
        app.router.add_post("/v1/agent/tasks",     self._handle_agent_task)
        app.router.add_get("/v1/agent/tasks/{id}", self._handle_agent_task_status)
        return app

    @staticmethod
    def _lan_ip() -> str:
        """Best-effort LAN address for the 'open this on your phone' hint."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("10.255.255.255", 1))     # no packets are sent
                return probe.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    async def _ensure_firewall_access(self) -> None:
        """Best-effort: make sure Windows Firewall allows inbound LAN traffic to
        the uplink port.  A blocked inbound rule is the usual cause of a phone on
        the same Wi-Fi seeing 'refused to connect' even though the server is up.
        Scoped to private/domain networks only (never public) for safety.
        Idempotent, admin-aware, and never raises."""
        if os.name != "nt" or os.getenv("ORION_REMOTE_FIREWALL", "1").strip().lower() in {
                "0", "false", "no", "off"}:
            return
        rule = "ORION Remote Uplink"
        try:
            def _run(args: list[str]) -> Any:
                return subprocess.run(
                    args, capture_output=True, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            show = await asyncio.to_thread(
                _run, ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule}"])
            if show.returncode == 0 and rule in (show.stdout or ""):
                return  # already allowed — nothing to do

            add = await asyncio.to_thread(_run, [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule}", "dir=in", "action=allow", "protocol=TCP",
                f"localport={self.port}", "profile=private,domain",
                "description=Allow the ORION phone uplink on the local network only",
            ])
            if add.returncode == 0:
                self.bus.log.emit(
                    f"REMOTE: firewall opened for TCP {self.port} on private/domain "
                    "networks — phones on your Wi-Fi can now reach ORION.")
            else:
                self.bus.log.emit(
                    "REMOTE: could not open the firewall automatically (needs "
                    "Administrator). If your phone says 'refused to connect', run this "
                    'once in an elevated PowerShell:  netsh advfirewall firewall add '
                    f'rule name="{rule}" dir=in action=allow protocol=TCP '
                    f"localport={self.port} profile=private,domain")
        except Exception:
            pass

    def _maybe_ssl_context(self) -> Any | None:
        """Self-signed TLS when ORION_REMOTE_TLS=1: phones then run in a secure
        context, which unlocks the browser microphone for voice input.  The
        certificate persists under config/remote_tls/ so the phone's one-time
        'proceed anyway' acceptance survives restarts."""
        if os.getenv("ORION_REMOTE_TLS", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return None
        import ssl
        tls_dir = self.auth.config_dir / "remote_tls"
        cert_path, key_path = tls_dir / "cert.pem", tls_dir / "key.pem"
        try:
            if not (cert_path.exists() and key_path.exists()):
                from datetime import datetime, timedelta, timezone

                from cryptography import x509
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import ec
                from cryptography.x509.oid import NameOID
                key = ec.generate_private_key(ec.SECP256R1())
                name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "orion.local")])
                import ipaddress
                sans = x509.SubjectAlternativeName([
                    x509.DNSName("orion.local"), x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address(self._lan_ip())),
                ])
                now = datetime.now(timezone.utc)
                cert = (x509.CertificateBuilder()
                        .subject_name(name).issuer_name(name)
                        .public_key(key.public_key())
                        .serial_number(x509.random_serial_number())
                        .not_valid_before(now - timedelta(days=1))
                        .not_valid_after(now + timedelta(days=825))
                        .add_extension(sans, critical=False)
                        .sign(key, hashes.SHA256()))
                tls_dir.mkdir(parents=True, exist_ok=True)
                key_path.write_bytes(key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption()))
                cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(cert_path), str(key_path))
            return context
        except Exception as exc:
            self.bus.log.emit(f"REMOTE: TLS unavailable ({exc}); serving plain HTTP.")
            return None

    def _push_event(self, payload: dict[str, Any]) -> None:
        """Fan a state event out to every connected phone; never blocks."""
        for queue in list(self._event_subs):
            try:
                queue.put_nowait(payload)
            except Exception:
                pass   # a full queue means a stalled client — drop, don't block

    def _push_device_event(self, device_id: str, payload: dict[str, Any]) -> None:
        """Send private approval details only to the owning paired device."""
        for queue, owner in list(self._event_devices.items()):
            if owner == device_id:
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    async def _push_confirm(self, device_id: str, conf: Any) -> None:
        """Push a pending CONFIRM action to the phone for an approval tap.

        The single-use approval token rides the SSE stream, which is itself
        access-token authenticated — only a paired device receives it, and it
        must be echoed back to /api/confirm to fire the action."""
        self._push_device_event(device_id, {
            "type": "confirm", "id": conf.id, "tool": conf.tool,
            "summary": conf.summary, "token": conf.token,
            "created_at": conf.created_at,
        })

    def _wire_bus_events(self) -> None:
        """Mirror the desktop orb's driving signals to remote subscribers.
        Amplitude arrives at audio rate, so it is coalesced and shipped by a
        ~8 Hz ticker instead of per-sample."""
        def _on_state(value: str) -> None:
            self._last_state = str(value)
            self._push_event({"type": "state", "value": self._last_state})

        def _on_speaking(active: bool) -> None:
            self._push_event({"type": "speaking", "value": bool(active)})

        def _on_emotion(name: str, params: Any) -> None:
            if isinstance(params, dict):
                try:
                    self._last_emotion = json.loads(json.dumps(params, default=str))
                except Exception:
                    return
                self._push_event({"type": "emotion", "name": str(name),
                                  "params": self._last_emotion})

        def _on_amplitude(value: float) -> None:
            self._amp_pending = float(value)

        def _on_thought(payload: Any) -> None:
            if isinstance(payload, dict):
                self._push_event({"type": "thought",
                                  "text": str(payload.get("text", ""))[:600],
                                  "kind": str(payload.get("kind", ""))})

        def _on_phone_action(payload: Any) -> None:
            # Hand a native action (call / text / navigate …) to connected
            # phones.  Only meaningful for a device running the app bridge; a
            # plain browser renders it as a tappable link fallback instead.
            if isinstance(payload, dict) and payload.get("kind"):
                self._push_event({"type": "action", "action": payload})

        for signal, handler in (("state", _on_state), ("speaking", _on_speaking),
                                ("emotion_changed", _on_emotion), ("amplitude", _on_amplitude),
                                ("thought", _on_thought), ("phone_action", _on_phone_action)):
            try:
                getattr(self.bus, signal).connect(handler)
            except Exception:
                pass

    async def _amplitude_ticker(self) -> None:
        while True:
            await asyncio.sleep(0.125)
            value = self._amp_pending
            if value is None or not self._event_subs:
                continue
            self._amp_pending = None
            self._push_event({"type": "amplitude", "value": round(value, 3)})

    async def start(self) -> None:
        app = self._build_app()
        self._closing = False
        self._runner = self._web.AppRunner(
            app, access_log=None, shutdown_timeout=self._SHUTDOWN_GRACE_S)
        await self._runner.setup()
        ssl_context = self._maybe_ssl_context()
        self._https_enabled = ssl_context is not None
        self._endpoints_cache = None                 # re-detect after (re)start
        site = self._web.TCPSite(self._runner, self.host, self.port, ssl_context=ssl_context)
        await site.start()
        # Open the local firewall for the port so a phone on the same Wi-Fi is
        # not met with 'refused to connect' despite the server being up.
        await self._ensure_firewall_access()
        self._wire_bus_events()
        if self._amp_task is None:
            self._amp_task = asyncio.create_task(
                self._amplitude_ticker(), name="orion-remote-amplitude")
        scheme = "https" if ssl_context is not None else "http"
        self.bus.log.emit(
            f"REMOTE: uplink active on {self.host}:{self.port} (installable PWA); "
            "pairing-based auth. Expose via HTTPS proxy for internet use."
        )
        self.bus.log.emit(
            f"REMOTE: on your phone (same Wi-Fi) open {scheme}://{self._lan_ip()}:{self.port} "
            "— add to home screen for the full-screen ORION app."
        )
        self.bus.log.emit(
            "REMOTE: if the phone still shows 'refused to connect', make sure this "
            "Wi-Fi is set to a Private network in Windows (Settings ▸ Network) and "
            "that the phone is on the same network — not a guest/5GHz-isolated SSID."
        )
        if ssl_context is None:
            self.bus.log.emit(
                "REMOTE: phone voice INPUT needs a secure context — set "
                "ORION_REMOTE_TLS=1 to serve HTTPS (accept the certificate once "
                "on the phone). Voice replies work either way.")
        legacy = self.auth.config_dir / "remote_token.txt"
        if legacy.exists():
            self.bus.log.emit(
                "REMOTE: config/remote_token.txt is obsolete — static tokens are "
                "no longer accepted; you may delete the file.")
        # First-run bootstrap: with no paired device the uplink is unreachable,
        # so open a pairing window and surface the code in the log.
        if not self.auth.has_devices():
            code = self.auth.begin_pairing()
            self.bus.log.emit(
                f"REMOTE: no paired devices yet — enter pairing code {code} "
                "in the PWA within 10 minutes to link your first device.")
        if self.telemetry is not None:
            try:
                self.telemetry.health.register("remote")
                self.telemetry.health.beat("remote", "OK", f"port {self.port}")
            except Exception:
                pass

    def _close_event_streams(self) -> None:
        """Wake every open /api/events stream so its handler returns now.

        A stream otherwise waits on its queue for up to the access-token
        lifetime, and aiohttp's graceful shutdown waits for it."""
        for queue in list(self._event_subs):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()          # a stalled client: make room
                    queue.put_nowait(None)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def stop(self) -> None:
        """Bounded and idempotent: phone streams are told to end, in-flight
        requests get ``_SHUTDOWN_GRACE_S`` before aiohttp cancels them, and the
        whole cleanup is capped in case a connection still will not let go."""
        self._closing = True
        self._close_event_streams()
        amp_task, self._amp_task = self._amp_task, None
        if amp_task is not None:
            amp_task.cancel()
            try:
                await amp_task
            except (asyncio.CancelledError, Exception):
                pass
        runner, self._runner = self._runner, None
        if runner is not None:
            try:
                await asyncio.wait_for(runner.cleanup(),
                                       timeout=self._SHUTDOWN_GRACE_S * 2 + 2.0)
            except asyncio.TimeoutError:
                self.bus.log.emit("REMOTE: uplink teardown timed out - "
                                  "abandoning the connections still open.")

    # ── rate limiting ──────────────────────────────────────────────────────────

    def _client_key(self, request: Any) -> str:
        # A direct client controls X-Forwarded-For; it cannot define its own
        # rate-limit bucket. Use the authenticated socket peer instead.
        peer = request.remote or "unknown"
        return str(peer)

    def _rate_ok(self, key: str) -> bool:
        now = time.time()
        # Opportunistically evict clients idle beyond the window so the map can
        # never grow without bound on a long-running public node.
        if len(self._hits) > 512:
            for k in [k for k, b in self._hits.items()
                      if not b or now - b[-1] > self._RATE_WINDOW]:
                self._hits.pop(k, None)
        bucket = self._hits.setdefault(key, deque())
        while bucket and now - bucket[0] > self._RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= self._RATE_MAX:
            return False
        bucket.append(now)
        return True

    # ── static routes ──────────────────────────────────────────────────────────

    def _is_lite(self, request: Any) -> bool:
        """Lite (mobile-data) mode: an explicit ?lite=1, else the browser's
        Save-Data hint that metered connections set automatically."""
        q = str(request.query.get("lite") or "").lower()
        if q in {"1", "true", "yes", "on"}:
            return True
        if q in {"0", "false", "no", "off"}:
            return False
        return request.headers.get("Save-Data", "").lower() == "on"

    async def _handle_page(self, request: Any) -> Any:
        html = REMOTE_PAGE_HTML
        if self._is_lite(request):
            # Drop the Three.js voxel face (its modules are pulled from a CDN —
            # the single biggest cellular cost) for a self-contained CSS orb.
            html = html.replace(_FACE_IFRAME, _LITE_ORB)
        return self._web.Response(text=html, content_type="text/html")

    def _face_page(self) -> str:
        """The desktop's exact Three.js voxel orb/face, with the postMessage
        bridge injected so the PWA can drive amplitude/morph/emotion.  Falls
        back to a self-contained 2-D canvas orb where the module is absent.

        FACE_HTML is a template: the desktop fills its import-map base and idle
        frame rate (face3d.three_sourced) before showing it. Served raw, the
        import map held bare "__THREE_BASE__build/…" values, which Chromium
        discards, and the phone sat on "Materialising ORION…" for good. Filled
        the same way here, with the base pointed at the vendored library this
        uplink serves (/three/…), so the face works on a LAN with no internet;
        the pinned CDN build only when three.js was not shipped."""
        if self._face_html is None:
            try:
                from .gui import face3d
                files, version = self._three_files()
                base = f"/three/{version}/" if files else face3d.CDN_BASE
                html = (face3d.FACE_HTML.replace("__THREE_BASE__", base)
                        .replace("__IDLE_FPS__", str(face3d.idle_fps())))
                self._face_html = html.replace("</body>", _FACE_BRIDGE + "</body>")
            except Exception:
                self._face_html = ORB_FALLBACK_HTML
        return self._face_html

    def _three_files(self) -> tuple[dict[str, Path], str]:
        """The vendored three.js modules the face may load, keyed by their path
        under the vendor root, and the pinned version. Empty when three.js was
        not shipped complete (face3d.local_three checks the whole import graph).
        Built once: this set IS the /three/ route's allow-list."""
        if self._three is None:
            files: dict[str, Path] = {}
            version = ""
            try:
                from .gui import face3d
                version = str(face3d.THREE_VERSION)
                root = face3d.local_three()
                if root is not None:
                    base = Path(root).resolve()
                    for path in base.rglob("*"):
                        if path.suffix.lower() not in {".js", ".mjs"}:
                            continue
                        try:
                            resolved = path.resolve()
                            relative = resolved.relative_to(base).as_posix()
                        except (OSError, ValueError):
                            continue        # a link out of the vendor root
                        if resolved.is_file():
                            files[relative] = resolved
            except Exception:
                files = {}
            self._three = (files, version)
        return self._three

    async def _handle_three(self, request: Any) -> Any:
        """The vendored three.js modules, for the phone's face. Unauthenticated
        like /face itself (it is a published library), read-only, and limited
        to exactly the files in the vendor graph: any other path — including
        every traversal attempt — is a 404, never a disk read."""
        files, version = self._three_files()
        path = files.get(str(request.match_info.get("path") or ""))
        if path is None or str(request.match_info.get("version") or "") != version:
            return self._web.Response(status=404, text="not found")
        # The version is in the URL, so a cached copy can never be stale.
        return self._web.FileResponse(path, headers={
            "Content-Type": "text/javascript; charset=utf-8",
            "Cache-Control": "public, max-age=604800, immutable",
        })

    async def _handle_face(self, request: Any) -> Any:
        response = self._web.Response(text=self._face_page(), content_type="text/html")
        # The face pulls Three.js modules from jsdelivr; everything else stays
        # locked down.  It is deliberately framed by the uplink page from the
        # SAME origin, so we must (a) allow same-origin framing and (b) set these
        # explicitly *before* the middleware runs setdefault, so ours win.
        #   frame-ancestors 'self'  → modern browsers honour this over XFO
        #   X-Frame-Options SAMEORIGIN → legacy fallback for older engines
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "connect-src 'self' https://cdn.jsdelivr.net; img-src 'self' data: blob:; "
            "frame-ancestors 'self'"
        )
        return response

    # ── live state stream (SSE) ────────────────────────────────────────────────

    async def _handle_events(self, request: Any) -> Any:
        """Server-sent events mirroring the desktop orb's state to the phone.
        EventSource cannot set headers, so the access token rides the query
        string; it is verified exactly like a Bearer token."""
        device_id = self.auth.verify_access(str(request.query.get("token") or ""))
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        response = self._web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._event_subs.add(queue)
        self._event_devices[queue] = device_id

        async def _send(payload: dict[str, Any]) -> None:
            await response.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

        # The stream lives no longer than the access token that opened it, so
        # a revoked device loses its mirror within the normal token window.
        deadline = time.time() + RemoteAuthManager.ACCESS_TTL_S
        try:
            await _send({"type": "state", "value": self._last_state})
            if self._last_emotion:
                await _send({"type": "emotion", "params": self._last_emotion})
            while time.time() < deadline and not self._closing:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if item is None:                  # stop(): the gateway is closing
                    break
                await _send(item)
        except (asyncio.CancelledError, ConnectionResetError, Exception):
            pass
        finally:
            self._event_subs.discard(queue)
            self._event_devices.pop(queue, None)
        return response

    async def _handle_manifest(self, request: Any) -> Any:
        return self._web.json_response(_MANIFEST, content_type="application/manifest+json")

    async def _handle_sw(self, request: Any) -> Any:
        return self._web.Response(text=_SERVICE_WORKER, content_type="application/javascript")

    async def _handle_icon(self, request: Any) -> Any:
        size = 512 if "512" in request.path else 192
        try:
            return self._web.Response(body=self._icon_png(size), content_type="image/png")
        except Exception:
            # PIL absent — fall back to a tiny inline SVG the browser accepts.
            svg = (
                f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'>"
                f"<rect width='100%' height='100%' fill='#050508'/>"
                f"<circle cx='{size//2}' cy='{size//2}' r='{size//3}' fill='#ff1a3c'/></svg>"
            )
            return self._web.Response(text=svg, content_type="image/svg+xml")

    def _icon_png(self, size: int) -> bytes:
        if size in self._icons:
            return self._icons[size]
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (size, size), (5, 5, 8, 255))
        draw = ImageDraw.Draw(img)
        cx = cy = size / 2
        # Crimson orb with concentric glow + cyan rim → matches the desktop orb.
        for r, col in (
            (size * 0.44, (255, 26, 60, 40)),
            (size * 0.36, (255, 26, 60, 90)),
            (size * 0.30, (120, 8, 20, 255)),
            (size * 0.22, (255, 100, 120, 255)),
            (size * 0.12, (255, 240, 245, 255)),
        ):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        rr = size * 0.31
        draw.arc([cx - rr, cy - rr, cx + rr, cy + rr], -70, 80, fill=(0, 229, 255, 220), width=max(2, size // 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        self._icons[size] = data
        return data

    async def _endpoints(self) -> list[dict[str, Any]]:
        """Reachable endpoints for the phone to auto-connect through (Tailscale
        first, then LAN). Cached ~60 s so we don't re-shell to tailscale per
        request; detection failures degrade to an empty list."""
        async with self._endpoints_lock:
            now = time.time()
            if self._endpoints_cache and now - self._endpoints_cache[0] < 60.0:
                return self._endpoints_cache[1]
            try:
                eps = await asyncio.to_thread(
                    detect_endpoints, self.port, https=self._https_enabled)
            except Exception as exc:
                self.bus.log.emit(f"REMOTE: endpoint detection failed - {exc}")
                eps = []
            self._endpoints_cache = (time.time(), eps)
            return eps

    async def _handle_endpoints(self, request: Any) -> Any:
        """Authenticated: the paired device refreshes its connection list here so
        it can hop between Tailscale (away) and LAN (home) with no IP typing."""
        header = str(request.headers.get("Authorization") or "")
        token = (header[7:].strip() if header.lower().startswith("bearer ")
                 else str(request.query.get("token") or ""))
        if self.auth.verify_access(token) if token else None:
            return self._web.json_response({
                "ok": True, "port": self.port, "https": self._https_enabled,
                "endpoints": await self._endpoints(),
            })
        return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)

    async def _handle_health(self, request: Any) -> Any:
        mode = "ONLINE"
        try:
            if self.router is not None and hasattr(self.router, "current_mode"):
                mode = str(self.router.current_mode())
        except Exception:
            pass
        return self._web.json_response({
            "ok": True,
            "state": "online",
            "mode": mode,
            "uptime_s": int(time.time() - self._started_at),
            "brain": bool(self.router is not None or self.local_brain is not None),
        })

    # ── authentication endpoints ───────────────────────────────────────────────

    @classmethod
    async def _json_object(cls, request: Any) -> dict[str, Any]:
        """Parse a JSON object body no larger than ``_MAX_JSON_BYTES``.

        Raises ``_BodyTooLarge`` for an oversized body (declared or streamed)
        and ``ValueError`` for anything that is not a JSON object."""
        limit = cls._MAX_JSON_BYTES
        declared = request.content_length
        if declared is not None and declared > limit:
            raise _BodyTooLarge(declared)
        # read(n) returns whatever is buffered, so collect until EOF — and stop
        # as soon as the body passes the limit rather than buffering the rest.
        raw = bytearray()
        while chunk := await request.content.read(limit + 1 - len(raw)):
            raw += chunk
            if len(raw) > limit:
                raise _BodyTooLarge(len(raw))
        payload = json.loads(raw.decode("utf-8")) if raw else None
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _bad_body(self, exc: Exception) -> Any:
        """The error response for a body ``_json_object`` refused."""
        if isinstance(exc, _BodyTooLarge):
            return self._web.json_response(
                {"ok": False, "error": "request body too large"}, status=413)
        return self._web.json_response({"ok": False, "error": "invalid JSON"}, status=400)

    def _auth_rate_ok(self, key: str) -> bool:
        """Tighter limiter for credential endpoints (brute-force resistance)."""
        now = time.time()
        if len(self._auth_hits) > 512:
            for old_key, hits in list(self._auth_hits.items()):
                if not hits or now - hits[-1] > 60.0:
                    self._auth_hits.pop(old_key, None)
        bucket = self._auth_hits.setdefault(key, deque())
        while bucket and now - bucket[0] > 60.0:
            bucket.popleft()
        if len(bucket) >= 10:
            return False
        bucket.append(now)
        return True

    async def _handle_pair(self, request: Any) -> Any:
        if not self._auth_rate_ok(self._client_key(request)):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request)
        except Exception as exc:
            return self._bad_body(exc)
        paired = self.auth.complete_pairing(
            str(payload.get("code") or ""), str(payload.get("device_name") or ""))
        if paired is None:
            return self._web.json_response(
                {"ok": False, "error": "invalid or expired pairing code"}, status=401)
        device_id, refresh_token = paired
        # Hand the phone its connection list at pairing time — this is what lets
        # it auto-connect forever after (Tailscale name works home + away) with
        # no IP ever typed again.
        return self._web.json_response(
            {"ok": True, "device_id": device_id, "refresh_token": refresh_token,
             "endpoints": await self._endpoints(), "port": self.port,
             "https": self._https_enabled})

    async def _handle_token(self, request: Any) -> Any:
        if not self._auth_rate_ok(self._client_key(request)):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request)
        except Exception as exc:
            return self._bad_body(exc)
        issued = self.auth.issue_access(
            str(payload.get("device_id") or ""), str(payload.get("refresh_token") or ""))
        if issued is None:
            return self._web.json_response(
                {"ok": False, "error": "invalid or revoked device credentials"}, status=401)
        access_token, expires_in = issued
        # Refresh the phone's endpoint list on every launch so a changed
        # Tailscale name or LAN IP is picked up without re-pairing.
        return self._web.json_response(
            {"ok": True, "access_token": access_token, "expires_in": expires_in,
             "endpoints": await self._endpoints()})

    def _authenticated_device(self, request: Any, payload: dict[str, Any]) -> Optional[str]:
        """Device id for a valid Bearer (or payload) access token, else None."""
        header = str(request.headers.get("Authorization") or "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not token:
            token = str(payload.get("access_token") or "")
        return self.auth.verify_access(token) if token else None

    # ── remote tool execution (capability-whitelisted) ────────────────────────

    async def _handle_agent_task(self, request: Any) -> Any:
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request)
        except Exception as exc:
            return self._bad_body(exc)
        device_id = self._authenticated_device(request, payload)
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised"}, status=401)
        tool = str(payload.get("tool") or "")
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        task = await self.agent_queue.run(tool, args, device_id=device_id)
        status = 200 if task.get("status") == "done" else 403 \
            if task.get("status") == "denied" else 503
        return self._web.json_response({"ok": task.get("status") == "done", **task},
                                       status=status)

    async def _handle_agent_task_status(self, request: Any) -> Any:
        payload: dict[str, Any] = {}
        device_id = self._authenticated_device(request, payload)
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised"}, status=401)
        task = self.agent_queue.get(str(request.match_info.get("id") or ""), device_id)
        if task is None:
            return self._web.json_response({"ok": False, "error": "unknown task"}, status=404)
        return self._web.json_response({"ok": True, **task})

    # ── chat: the full brain ───────────────────────────────────────────────────

    async def _handle_chat(self, request: Any) -> Any:
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request)
        except Exception as exc:
            return self._bad_body(exc)
        device_id = self._authenticated_device(request, payload)
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised — pair this device first"}, status=401)
        message = str(payload.get("message") or "").strip()[:4000]
        if not message:
            return self._web.json_response({"ok": False, "error": "empty message"}, status=400)
        try:
            SecuritySanitiser.guard_text(message, "remote.message")
        except SecurityViolation as exc:
            return self._web.json_response({"ok": False, "error": str(exc)}, status=403)

        self.bus.log.emit(f"REMOTE: {message[:120]}")
        try:
            await asyncio.to_thread(self.memory.log_episode, "user (remote)", message)
        except Exception:
            pass
        try:
            with self.tool_gate.for_device(device_id):
                reply, provider = await self._answer(message, device_id=device_id)
        except Exception as exc:
            return self._web.json_response(
                {"ok": False, "error": str(exc).splitlines()[0][:200]}, status=502)
        try:
            await asyncio.to_thread(self.memory.log_episode, "orion (remote)", reply)
        except Exception:
            pass
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("remote.turns")
            except Exception:
                pass
        return self._web.json_response({"ok": True, "reply": reply, "provider": provider})

    # ── voice input: speech-to-text for the phone ──────────────────────────────

    async def _handle_transcribe(self, request: Any) -> Any:
        """Turn a phone's recorded audio into text, fully offline.

        This is the voice-INPUT path for the Android app: its WebView can reach
        the microphone (getUserMedia) but has no in-browser SpeechRecognition,
        so it records a small 16 kHz mono PCM WAV and posts it here.  The clip is
        transcribed on the desktop with ORION's local Whisper/Vosk engine and the
        text is returned for the phone to send as an ordinary message."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        device_id = self._authenticated_device(request, {})
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised — pair this device first"}, status=401)
        try:
            body = await request.read()
        except Exception:
            return self._web.json_response({"ok": False, "error": "no audio"}, status=400)
        if not body:
            return self._web.json_response({"ok": False, "error": "no audio"}, status=400)
        if len(body) > self._MAX_AUDIO_BYTES:
            return self._web.json_response(
                {"ok": False, "error": "audio too large"}, status=413)
        try:
            text, err = await asyncio.get_running_loop().run_in_executor(
                None, self._transcribe_blocking, body)
        except Exception as exc:
            return self._web.json_response(
                {"ok": False, "error": str(exc).splitlines()[0][:160]}, status=400)
        if err:
            status = 503 if "engine" in err else 400
            return self._web.json_response({"ok": False, "error": err}, status=status)
        if text and self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("remote.voice_transcriptions")
            except Exception:
                pass
        self.bus.log.emit(f"REMOTE: voice ▸ {text[:120]}" if text
                          else "REMOTE: voice ▸ (nothing heard)")
        return self._web.json_response({"ok": True, "text": text})

    def _get_transcriber(self) -> Any | None:
        """Build (once) and return the offline transcriber, or None if unavailable.
        Called only from an executor thread — model loading blocks."""
        if self._transcriber is self._UNSET:
            try:
                from . import speech_offline
                # The process-wide instance: a second one here meant a second
                # copy of the Whisper model in RAM.
                self._transcriber = speech_offline.shared(bus=self.bus, telemetry=self.telemetry)
            except Exception as exc:
                self.bus.log.emit(f"REMOTE: voice transcriber unavailable ({exc}).")
                self._transcriber = None
        return self._transcriber

    def _transcribe_blocking(self, body: bytes) -> tuple[str, str]:
        """Parse a WAV upload and transcribe it — all the blocking work, for an
        executor.  Returns (text, error); error is non-empty on failure."""
        transcriber = self._get_transcriber()
        if transcriber is None or not transcriber.available:
            return "", "no speech engine on the desktop"
        import io as _io
        import wave
        try:
            with wave.open(_io.BytesIO(body), "rb") as wf:
                rate = wf.getframerate()
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                pcm = wf.readframes(wf.getnframes())
        except Exception:
            return "", "unreadable audio (expected a WAV)"
        if width != 2:
            return "", "expected 16-bit PCM audio"
        if channels > 1:
            pcm = self._downmix_mono(pcm, channels)
        return (transcriber.transcribe_pcm(pcm, rate) or "").strip(), ""

    @staticmethod
    def _downmix_mono(pcm: bytes, channels: int) -> bytes:
        """Average interleaved 16-bit channels down to mono (audioop is gone in
        3.13, so this is done by hand).  Rarely needed — the phone sends mono."""
        import array
        samples = array.array("h")
        samples.frombytes(pcm[: (len(pcm) // (2 * channels)) * 2 * channels])
        mono = array.array("h", bytes(len(samples) // channels * 2))
        for i in range(len(mono)):
            base = i * channels
            mono[i] = int(sum(samples[base:base + channels]) / channels)
        return mono.tobytes()

    async def _handle_confirmations(self, request: Any) -> Any:
        """List the actions this device has awaiting an approval tap (so a phone
        that reconnected mid-flight can catch up)."""
        header = str(request.headers.get("Authorization") or "")
        token = (header[7:].strip() if header.lower().startswith("bearer ")
                 else str(request.query.get("token") or ""))
        device_id = self.auth.verify_access(token) if token else None
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        pending = [c.public() for c in self.confirmations.pending_for(device_id)]
        return self._web.json_response({"ok": True, "pending": pending})

    async def _handle_confirm(self, request: Any) -> Any:
        """Approve or deny a CONFIRM-tier action from the phone. On approval the
        tool finally runs through the dispatcher and the result is streamed back
        over SSE (and returned inline)."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response({"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request)
        except Exception as exc:
            return self._bad_body(exc)
        device_id = self._authenticated_device(request, payload)
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised — pair this device first"}, status=401)
        conf_id = str(payload.get("id") or "")
        token = str(payload.get("token") or "")
        decision = payload.get("decision", payload.get("approve"))
        approve = (decision is True or str(decision).lower() in
                   {"approve", "approved", "yes", "true", "1", "confirm", "ok"})
        conf = self.confirmations.resolve(conf_id, token, approve, device_id=device_id)
        if conf is None:
            return self._web.json_response(
                {"ok": False, "error": "no such pending action, or it expired"}, status=409)
        if not approve:
            self._push_device_event(device_id, {"type": "confirm_result", "id": conf_id, "status": "denied"})
            return self._web.json_response({"ok": True, "status": "denied"})
        self.bus.log.emit(f"REMOTE: approved '{conf.tool}' from {device_id[:8]}")
        try:
            result = await self.tool_gate.run_approved(conf)
        except Exception as exc:
            return self._web.json_response(
                {"ok": False, "error": str(exc).splitlines()[0][:200]}, status=502)
        text = (getattr(result, "text", "") or "Done.").splitlines()[0][:400]
        ok = bool(getattr(result, "ok", True))
        self._push_device_event(device_id, {"type": "confirm_result", "id": conf_id,
                          "status": "done", "ok": ok, "reply": text})
        return self._web.json_response({"ok": ok, "status": "done", "reply": text})

    async def _answer(self, message: str, device_id: str = "") -> tuple[str, str]:
        """Answer *message* with the full brain: identity- and memory-grounded
        language model when reachable, tool-capable LocalBrain otherwise.

        Full-parity remote (#12): an actionable command (open, search, read your
        mail, text someone, drive the browser…) is executed first through the
        capability gate — so ORION actually *does* it on the move, with an
        on-phone confirmation for anything irreversible. Non-command chat falls
        through to the model so conversation stays rich."""
        if self.local_brain is not None and hasattr(self.local_brain, "try_task"):
            try:
                task_reply = await self.local_brain.try_task(message)
            except Exception as exc:
                self.bus.log.emit(f"REMOTE: task path failed - {exc}")
                task_reply = None
            if task_reply is not None:
                return task_reply, "orion-remote"
        system_extra = await asyncio.to_thread(self._grounding, message)
        if self.router is not None:
            try:
                if self.router.has_text_fallback():
                    profile, reply = await self.router.generate_text(
                        message, system_extra=system_extra)
                    if reply and reply.strip():
                        return reply.strip(), getattr(profile, "name", "model")
            except Exception as exc:
                self.bus.log.emit(f"REMOTE: model path failed, using local brain - {exc}")
        # Offline or the model failed → the tool-capable rule-based brain.
        if self.local_brain is not None:
            reply = await self.local_brain.respond(message)
            return (reply or "I'm here."), "local-brain"
        raise RuntimeError("no language model or local brain is reachable")

    def _grounding(self, message: str) -> str:
        """Assemble the identity persona plus any relevant conversation context
        so remote answers sound like ORION and remember prior turns."""
        parts: list[str] = []
        if self.identity is not None:
            try:
                parts.append(self.identity.persona_text())
            except Exception:
                pass
        if self.conversation is not None:
            for attr in ("context_for", "prompt_context", "retrieve"):
                fn = getattr(self.conversation, attr, None)
                if fn is None:
                    continue
                try:
                    ctx = fn(message) if attr != "prompt_context" else fn()
                    if hasattr(ctx, "__await__"):
                        continue  # skip coroutine-only variants here
                    if ctx:
                        parts.append(str(ctx)[:1500])
                        break
                except Exception:
                    continue
        elif self.memory is not None:
            try:
                ctx = self.memory.prompt_context()
                if ctx:
                    parts.append(str(ctx)[:1500])
            except Exception:
                pass
        return "\n\n".join(p for p in parts if p)
