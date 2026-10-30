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
``ORION_REMOTE_HOST`` (default ``127.0.0.1``) on ``ORION_REMOTE_PORT`` (default
``8000``). Tailscale Serve can terminate HTTPS and forward to loopback. When
Serve is unavailable, ORION can terminate TLS while bound to its tailnet IP.

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

Remote tool execution goes through ``RemoteToolGate`` and ``RemoteAgentQueue``.
Read-only tools may run directly; supported actions require a reviewed approval
owned by the paired device. Unsupported tools remain unavailable remotely.

SECURITY: for public internet exposure put this behind an HTTPS reverse proxy.
Direct TLS is intended only for the private Tailscale adapter. The gateway adds
hardening headers, per-client rate limiting and constant-time credential
comparisons.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import io
import json
import os
import re
import secrets
import subprocess
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Optional

from . import key_seal, request_trace
from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent
from .request_trace import TRACES, Stage
from .remote_capability import (
    RemoteConfirmationRegistry,
    RemoteToolGate,
    describe_action,
)
from .remote_endpoints import detect_endpoints, lan_ipv4_addresses, public_https_endpoint
from .remote_lan import LanLink
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, utc_stamp
from .atomic_io import atomic_write_text

# ── installable PWA shell ─────────────────────────────────────────────────────
#
# The phone gets the REAL ORION: a compact vector signal mirrored live over
# /api/events, voice output via
# the browser's speech synthesis, and voice input via speech recognition where
# the context is secure (HTTPS or localhost).  Chat still flows through the
# full memory-synced brain.

REMOTE_PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#08080A">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="ORION">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-192.png">
<title>O.R.I.O.N.</title>
<style>
:root{--pri:#D32F2F;--pri-dim:#801818;--slate:#8A99AD;--bg:#08080A;--panel:#121216}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);
 color:#fff;font-family:'Segoe UI',system-ui,Arial,sans-serif;display:flex;flex-direction:column;
 height:100dvh;overflow:hidden}
header{height:calc(24px + env(safe-area-inset-top));padding:env(safe-area-inset-top) 8px 0;
 border-bottom:1px solid var(--pri-dim);background:var(--panel);display:flex;align-items:center;
 gap:9px;white-space:nowrap;overflow:hidden;font:10px ui-monospace,monospace;color:var(--slate)}
header strong{color:var(--pri);letter-spacing:1px}
header .lit{color:#F2E6E8}
#dot{margin-left:auto;color:var(--pri);white-space:nowrap}
#stage{position:relative;flex:0 0 142px;margin:12px 14px 0;border:1px solid #35272d;
 border-radius:18px;background:linear-gradient(135deg,#181218,#101015);overflow:hidden;padding:16px}
#vector{position:absolute;width:100%;height:46px;display:block;left:0;bottom:7px;opacity:.6}
#state{position:absolute;top:17px;right:16px;font:10px ui-monospace,monospace;
 color:var(--slate);text-transform:uppercase;pointer-events:none}
#voice-title{font-size:15px;font-weight:600;margin:0 70px 6px 0}
#voice-hint{font-size:12px;color:var(--slate);line-height:1.4;min-height:34px;max-width:85%}
#voice-heard{position:relative;z-index:1;font-size:12px;color:#f4dfe5;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap;height:18px;margin-top:3px}
#voice-meter{position:absolute;bottom:0;left:0;height:3px;max-width:100%;width:0;
 background:#e86170;transition:width .16s ease-out;box-shadow:0 0 12px #d32f2f80}
.native header{display:none}
#options{margin-left:auto;position:relative}
#options summary{cursor:pointer;padding:3px 8px;color:#eee;list-style:none}
#options nav{position:fixed;right:12px;top:30px;z-index:25;padding:8px;border-radius:12px;
 background:#202026;border:1px solid #4b333d;box-shadow:0 14px 36px #0008;display:grid;gap:6px}
#options button{background:transparent;border:none;text-align:left;padding:12px;font-size:13px}
.sheet{position:fixed;inset:0;z-index:22;background:var(--bg);overflow-y:auto;padding:16px;
 padding-top:calc(16px + env(safe-area-inset-top));padding-bottom:calc(16px + env(safe-area-inset-bottom))}
.sheet-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px}
.sheet h2{font-size:18px;margin:0}.sheet-head button{padding:10px;font-size:12px}
.sheet pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,monospace;color:#c5cbd6}
#log{flex:1;overflow-y:auto;padding:18px clamp(14px,5vw,28px);display:flex;flex-direction:column;gap:14px;
 -webkit-overflow-scrolling:touch;position:relative;overflow-anchor:none}
.msg{max-width:min(88%,680px);padding:14px 16px;border-radius:18px;font-size:15px;line-height:1.55;white-space:pre-wrap;
 word-wrap:break-word;animation:rise .18s ease-out}
.msg.hist,.msg.settled{animation:none}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
#earlier{align-self:center;flex:none;background:transparent;border:1px solid var(--pri-dim);
 color:var(--slate);padding:6px 12px;font:11px ui-monospace,monospace;border-radius:10px}
.sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;border:0;overflow:hidden;
 clip:rect(0 0 0 0);white-space:nowrap}
[hidden]{display:none!important}
.user{align-self:flex-end;background:#801818}
.orion{align-self:flex-start;background:var(--panel);border:1px solid var(--pri-dim)}
.orion.think{color:var(--slate)}
.skeleton{width:min(72%,340px);min-height:72px;background:linear-gradient(105deg,#18181e 20%,#29252b 42%,#18181e 65%);
 background-size:220% 100%;animation:shimmer 1.5s linear infinite;border-color:#32252c}
@keyframes shimmer{to{background-position-x:-220%}}
.dashboard{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:12px clamp(14px,5vw,28px);
 border-bottom:1px solid #30242a;background:#0e0e12}
.sensor-card{background:#17171d;border:1px solid #35272d;border-radius:16px;padding:12px;min-width:0;
 box-shadow:0 8px 24px #0003}
.sensor-card strong{display:block;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sensor-card .reading{font-size:24px;font-weight:700;margin:8px 0 3px;font-variant-numeric:tabular-nums}
.sensor-card small{font-size:11px;color:var(--slate)}
.sensor-card canvas{width:100%;height:34px;display:block;margin-top:5px}
.sensor-card progress{width:100%;height:8px;accent-color:var(--pri)}
.widget{display:grid;gap:8px;white-space:normal}.widget h3{font-size:14px;margin:0;color:#f5d9df}
.widget-row{display:flex;justify-content:space-between;gap:16px;font-size:13px}
.widget-row span:first-child{color:var(--slate)}.widget details{overflow-wrap:anywhere}
.widget pre{max-height:220px;overflow:auto;white-space:pre-wrap;word-break:break-word;
 font-size:12px;color:#d5d5dc;background:#0b0b10;padding:10px;border-radius:10px}
@media(max-width:360px){.dashboard{gap:6px;padding:8px}.sensor-card{padding:9px}.sensor-card .reading{font-size:20px}}
.meta{align-self:center;color:var(--slate);font-size:11px;text-align:center}
details.trace{align-self:stretch;background:var(--panel);border-left:2px solid var(--pri-dim);
 padding:6px 10px;color:var(--slate);font:11px ui-monospace,monospace}
details.trace summary{cursor:pointer;color:#F2E6E8}
.confirm{align-self:stretch;background:var(--panel);border:1px solid var(--pri-dim)}
.diff{font:11px ui-monospace,monospace;color:#F2E6E8;white-space:pre-wrap;
 margin:8px 0;max-height:35vh;overflow:auto}
.slide{position:relative;height:46px;background:#301216;border:1px solid var(--pri-dim);
 border-radius:23px;touch-action:none;user-select:none;overflow:hidden}
.slide .fill{position:absolute;inset:0 auto 0 0;width:0;background:var(--pri-dim);border-radius:23px}
.slide .thumb{position:absolute;top:2px;left:2px;width:40px;height:40px;border-radius:50%;
 background:var(--pri);display:grid;place-items:center;font-size:22px}
.slide .hint{position:absolute;top:15px;left:58px;font:10px ui-monospace,monospace;letter-spacing:1px}
.deny{margin-top:8px;background:transparent;border:1px solid var(--pri-dim);color:var(--slate);padding:7px 12px}
form{display:flex;gap:8px;padding:10px 10px calc(10px + env(safe-area-inset-bottom));
 border-top:1px solid var(--pri-dim);background:var(--panel)}
input{flex:1;background:var(--bg);color:#fff;border:1px solid var(--pri-dim);border-radius:12px;
 padding:12px;font-size:16px;outline:none;min-width:0}
input:focus{border-color:var(--pri)}
button{background:var(--pri-dim);color:#fff;border:1px solid var(--pri);
 border-radius:12px;padding:0 16px;font-weight:800;font-size:15px}
button:active{background:var(--pri)}
button.icon{padding:0 13px;font-size:17px}
button.off{opacity:.45;border-color:#3a1a22}
#mic.listening{animation:pulse 1s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(211,47,47,.6)}50%{box-shadow:0 0 0 8px rgba(211,47,47,0)}}
#tasks{position:fixed;inset:0;z-index:20;background:var(--bg);display:flex;flex-direction:column;
 padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom)}
#tasks .head{display:flex;align-items:center;gap:10px;padding:12px;border-bottom:1px solid var(--pri-dim);
 background:var(--panel)}
#tasks .head strong{color:var(--pri);letter-spacing:2px;font:12px ui-monospace,monospace}
#tasks-sum{flex:1;color:var(--slate);font-size:12px}
#tasks-close{height:36px}
#tasks-appr{margin:10px 12px 0;padding:10px;border:1px solid var(--pri-dim);border-radius:10px;
 color:#F2E6E8;font-size:13px;line-height:1.45}
#tasks-list{list-style:none;margin:0;padding:10px 12px;overflow-y:auto;flex:1;display:flex;
 flex-direction:column;gap:8px}
#tasks-list li{background:var(--panel);border:1px solid var(--pri-dim);border-radius:10px;padding:10px 12px;
 display:grid;gap:3px;font-size:14px;min-width:0}
#tasks-list .t-state{font:11px ui-monospace,monospace;letter-spacing:1px;color:var(--slate)}
#tasks-list .t-note{color:var(--slate);font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}
</style></head><body>
<header><strong>ORION</strong><span id="dot">●</span>
<details id="options"><summary>Options ▾</summary><nav aria-label="Options">
 <button type="button" id="hardwarebtn">PC statistics</button>
 <button type="button" id="tasksbtn" aria-haspopup="dialog" aria-controls="tasks">Tasks</button>
 <button type="button" id="diagnosticsbtn">Connection &amp; microphone</button>
</nav></details></header>
<section id="stage" aria-label="ORION listening box">
 <div id="voice-title">Ready when you are</div><div id="state">standby</div>
 <div id="voice-hint">Tap the microphone to talk to ORION.</div><div id="voice-heard"></div>
 <canvas id="vector" aria-label="ORION signal waveform"></canvas><div id="voice-meter"></div>
</section>
<section id="hardware" class="sheet" role="dialog" aria-modal="true" aria-labelledby="hardware-title" hidden>
 <div class="sheet-head"><h2 id="hardware-title">PC statistics</h2>
 <button type="button" id="hardware-close">Back to terminal</button></div>
 <section class="dashboard" aria-label="Host hardware">
 <div class="sensor-card"><strong id="cpu-name">CPU</strong><div class="reading" id="cpu-load">—</div>
  <small id="cpu-temp">Temperature unavailable</small><canvas id="cpu-chart" aria-label="CPU load history"></canvas></div>
 <div class="sensor-card"><strong id="gpu-name">GPU</strong><div class="reading" id="gpu-load">—</div>
  <small id="gpu-temp">Temperature unavailable</small><canvas id="gpu-chart" aria-label="GPU load history"></canvas></div>
 </section>
</section>
<section id="diagnostics" class="sheet" role="dialog" aria-modal="true" aria-labelledby="diagnostics-title" hidden>
 <div class="sheet-head"><h2 id="diagnostics-title">Connection &amp; microphone</h2>
 <button type="button" id="diagnostics-close">Back to terminal</button></div>
 <p><span id="ping">-- ms</span> · <span id="model">ROUTE --</span> · <span id="tls">TLS --</span></p>
 <p id="sensors">MIC ○ CAM ○ ACCESS ○</p><pre id="audio-diagnostics">Loading…</pre>
</section>
<div id="log" role="log" aria-live="off" aria-label="Conversation with ORION"><button type="button" id="earlier" hidden>Show earlier messages</button></div>
<div id="say" class="sr" aria-live="polite" aria-atomic="true"></div>
<form id="f">
 <input id="m" placeholder="Message ORION…" autocomplete="off" autocapitalize="sentences" aria-label="Message ORION">
 <button type="button" id="mic" class="icon" title="Speak to ORION" aria-label="Speak to ORION">🎙</button>
 <button type="button" id="spk" class="icon" title="Voice replies" aria-label="Voice replies">🔊</button>
 <button type="button" id="stop" class="icon" title="Stop the reply" aria-label="Stop the reply" hidden>■</button>
 <button id="send">SEND</button>
</form>
<div id="tasks" role="dialog" aria-modal="true" aria-labelledby="tasks-title" hidden>
 <div class="head"><strong id="tasks-title">TASKS</strong><span id="tasks-sum">Loading…</span>
 <button type="button" id="tasks-close" aria-label="Close tasks">Close</button></div>
 <div id="tasks-appr" hidden></div>
 <ul id="tasks-list" aria-label="ORION's tasks"></ul>
</div>
<script>
const QUICK_TUNNEL=false;
document.body.classList.toggle('native',!!window.OrionNative);
const log=document.getElementById('log'),dot=document.getElementById('dot');
const earlier=document.getElementById('earlier'),sayEl=document.getElementById('say');
const stopBtn=document.getElementById('stop');
async function fetchLimited(url,options={},timeoutMs=8000,retries=0){
 for(let attempt=0;;attempt++){
  const ctl=new AbortController(),timer=setTimeout(()=>ctl.abort(),timeoutMs);
  try{const response=await fetch(url,{...options,signal:ctl.signal});
   if(attempt>=retries||!['GET',undefined].includes(options.method)||
      ![429,502,503,504].includes(response.status))return response;}
  catch(error){if(attempt>=retries)throw error;}
  finally{clearTimeout(timer);}
  await new Promise(resolve=>setTimeout(resolve,
   Math.min(4000,400*(2**attempt))+Math.random()*200));
 }}
const vector=document.getElementById('vector'),stateEl=document.getElementById('state');
const vx=vector.getContext('2d');
let signalAmp=0,signalTarget=0,signalMorph=0,signalClock=0,voiceLevel=0;
const voiceTitle=document.getElementById('voice-title'),voiceHint=document.getElementById('voice-hint');
const voiceHeard=document.getElementById('voice-heard'),voiceMeter=document.getElementById('voice-meter');
function voiceStatus(title,hint){voiceTitle.textContent=title;voiceHint.textContent=hint||'';}
function microphoneLevel(level){voiceLevel=Math.max(0,Math.min(1,(Number(level)||0)*18));
 voiceMeter.style.width=(voiceLevel*100)+'%';}
function fitVector(){const d=Math.min(devicePixelRatio||1,2);vector.width=vector.clientWidth*d;
 vector.height=vector.clientHeight*d;}
addEventListener('resize',fitVector);fitVector();
function paintVector(){requestAnimationFrame(paintVector);if(document.hidden)return;
 signalClock+=.035;signalAmp+=(Math.max(signalTarget,voiceLevel)-signalAmp)*.18;
 const w=vector.width,h=vector.height,cx=w/2,cy=h/2,d=Math.min(devicePixelRatio||1,2);
 vx.clearRect(0,0,w,h);vx.strokeStyle='#D32F2F';vx.lineWidth=2*d;
 const r=12*d*(1+.08*Math.sin(signalClock));vx.shadowColor='#D32F2F';vx.shadowBlur=12*d;
 vx.beginPath();vx.arc(cx,cy,r,0,Math.PI*2);vx.stroke();vx.shadowBlur=0;
 if(signalAmp>.015||signalMorph){for(let line=0;line<3;line++){vx.globalAlpha=1-line*.22;
  vx.beginPath();for(let i=0;i<=64;i++){const x=cx-110*d+i*220*d/64;
   const taper=Math.max(0,1-Math.abs(x-cx)/(110*d));
   const y=cy+Math.sin(i*.5-signalClock*(2+line*.2)+line)*signalAmp*22*d*taper;
   if(i===0)vx.moveTo(x,y);else vx.lineTo(x,y);}vx.stroke();}vx.globalAlpha=1;}}
paintVector();
let dev=localStorage.getItem('orion_device')||'';
let refresh=localStorage.getItem('orion_refresh')||'';
let access=localStorage.getItem('orion_access')||'';
let accessExp=+(localStorage.getItem('orion_access_exp')||0);
// ── the conversation log: bounded, batched, and never pulled from the reader ─
// A phone session runs for hours. Every message and every desktop thought
// used to stay in the DOM for good, and each insert jumped the view to the
// bottom even while you were reading further up. The log now keeps the latest
// MAX_MSGS messages and MAX_TRACES activity lines while you are at the bottom
// (older conversation comes back from the desktop with "Show earlier"). While
// you are scrolled up it holds up to HARD_CAP nodes, shedding off-screen
// activity lines first and then only messages above the view, and keeps the
// top visible line exactly where it was. Approval cards are never evicted,
// and new text is followed only when the view was already at the bottom.
// DOM work is gathered per animation frame.
const MAX_MSGS=100,MAX_TRACES=40,HARD_CAP=400;
const frameJobs=new Set();let frameReq=0;
function onFrame(job){frameJobs.add(job);if(!frameReq)frameReq=requestAnimationFrame(runFrame);}
function runFrame(){frameReq=0;const jobs=[...frameJobs];frameJobs.clear();
 for(const job of jobs){try{job();}catch(e){}}}
function nearBottom(){return log.scrollHeight-log.scrollTop-log.clientHeight<64;}
function toBottom(){log.scrollTop=log.scrollHeight;}
function place(node,force){const stick=force||nearBottom();log.appendChild(node);
 if(stick)toBottom();onFrame(trimLog);return node;}
function add(cls,text,force){const d=document.createElement('div');d.className='msg '+cls;
 d.textContent=text;return place(d,force);}
function renderReply(node,reply){let value;try{value=JSON.parse(reply);}catch(e){return false;}
 if(!value||Array.isArray(value)||typeof value!=='object')return false;
 const kind=String(value.type||value.kind||'');
 if(!['system_metrics','script_result','api_data'].includes(kind))return false;
 node.textContent='';node.classList.add('widget');
 const title=document.createElement('h3');title.textContent=String(value.title||kind.replace('_',' '));node.append(title);
 const data=value.data&&typeof value.data==='object'&&!Array.isArray(value.data)?value.data:value;
 for(const [key,val] of Object.entries(data).slice(0,12)){
  if(['type','kind','title','data'].includes(key))continue;
  if(kind==='script_result'&&key==='output'){
   const detail=document.createElement('details'),summary=document.createElement('summary');
   summary.textContent='View script output';const output=document.createElement('pre');
   output.textContent=String(val).slice(0,4000);detail.append(summary,output);node.append(detail);continue;}
  const row=document.createElement('div');row.className='widget-row';
  const label=document.createElement('span');label.textContent=key.replaceAll('_',' ');
  const content=document.createElement('span');content.textContent=typeof val==='object'?JSON.stringify(val):String(val);
  row.append(label,content);node.append(row);
  if(kind==='system_metrics'&&typeof val==='number'&&val>=0&&val<=100){
   const bar=document.createElement('progress');bar.max=100;bar.value=val;node.append(bar);}
 }return true;}
let evicted=false;
function pinned(n){return n===earlier||n.classList.contains('confirm')||n.classList.contains('streaming');}
function trimLog(){const stick=nearBottom(),kids=[...log.children],drop=new Set();
 if(stick){let msgs=0,traces=0;
  for(let i=kids.length-1;i>=0;i--){const n=kids[i];if(pinned(n))continue;
   if(n.classList.contains('trace')?++traces>MAX_TRACES:++msgs>MAX_MSGS)drop.add(n);}}
 else{let excess=kids.length-HARD_CAP;if(excess<=0)return;
  const top=log.scrollTop,bottom=top+log.clientHeight;
  const above=n=>n.offsetTop+n.offsetHeight<=top;
  const offscreen=n=>above(n)||n.offsetTop>=bottom;
  for(const n of kids){if(excess<=0)break;
   if(!pinned(n)&&n.classList.contains('trace')&&offscreen(n)){drop.add(n);excess--;}}
  for(const n of kids){if(excess<=0)break;
   if(!pinned(n)&&!drop.has(n)&&above(n)){drop.add(n);excess--;}}}
 if(!drop.size)return;
 // Scrolled up: keep the top visible node where it is on screen, whatever was
 // removed above or below it.
 let ref=null,refTop=0;
 if(!stick){const top=log.scrollTop;
  ref=kids.find(n=>!drop.has(n)&&n.offsetTop+n.offsetHeight>top)||null;if(ref)refTop=ref.offsetTop;}
 for(const n of drop){if(n.dataset.hid)evicted=true;n.remove();}
 if(stick)toBottom();else if(ref)log.scrollTop+=ref.offsetTop-refTop;
 paintEarlier();}
log.addEventListener('scroll',()=>{if(nearBottom())onFrame(trimLog);},{passive:true});
// Screen readers hear each finished reply once, from here, rather than every
// repaint of a streaming answer.
function announce(text){sayEl.textContent='';
 setTimeout(()=>{sayEl.textContent=String(text||'').slice(0,800);},30);}
let pendingState=null;
function setState(s){pendingState=s;onFrame(applyState);}
function applyState(){if(pendingState===null)return;stateEl.textContent=pendingState;pendingState=null;}
add('meta','Connected to ORION. Conversations sync into his memory.');
function faceMsg(m){if(m.amp!==undefined)signalTarget=Math.max(0,Math.min(1,+m.amp||0));
 if(m.morph!==undefined)signalMorph=+m.morph||0;}
async function pair(given){
 const code=(given||prompt('Enter the one-time pairing code shown on the ORION desktop')||'').trim();
 if(!code)return false;
 try{const r=await fetchLimited('/v1/auth/pair',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({code,device_name:(navigator.userAgent||'device').slice(0,60)})},10000);
  if(!r.ok)return false;const j=await r.json();
  dev=j.device_id;refresh=j.refresh_token;
  localStorage.setItem('orion_device',dev);localStorage.setItem('orion_refresh',refresh);
  learnEndpoints(j.endpoints);
  return true;}catch(e){return false;}}
// Hand the endpoint list the desktop reported to the native app, so it can
// auto-connect (Tailscale name works home + away) without an IP ever typed.
function learnEndpoints(eps){
 if(eps&&window.OrionNative&&OrionNative.endpointsChanged){
  try{OrionNative.endpointsChanged();}catch(e){}}}
// 'ok', 'denied' (the desktop refused this device: pair again) or 'failed'
// (busy, rate-limited, offline: try again later). Every failure used to pop the
// pairing prompt, including a 429 from reconnecting too often.
async function refreshAccess(){
 if(!dev||!refresh)return 'denied';
 try{const r=await fetchLimited('/v1/auth/token',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({device_id:dev,refresh_token:refresh})},10000);
  if(r.status===401||r.status===403)return 'denied';
  if(!r.ok)return 'failed';const j=await r.json();
  access=j.access_token;accessExp=Date.now()+Math.max(30,(j.expires_in-60))*1000;
  localStorage.setItem('orion_access',access);localStorage.setItem('orion_access_exp',accessExp);
  learnEndpoints(j.endpoints);
  return 'ok';}catch(e){return 'failed';}}
async function ensureAccess(){
 if(access&&Date.now()<accessExp)return true;
 const got=await refreshAccess();
 if(got==='ok')return true;
 if(got==='denied'&&await pair())return (await refreshAccess())==='ok';
 return false;}
// A QR code from the desktop's Security Centre carries ?pair=CODE.
(async()=>{const q=new URLSearchParams(location.search).get('pair');
 if(q){history.replaceState(null,'',location.pathname);
  if(await pair(q))add('meta','Paired with ORION.');}
 await refreshAccess();openEvents();if(QUICK_TUNNEL)catchUpConfirms();
 probeHistory();loadHardware();})();
const hardwareHistory={cpu:[],gpu:[]};
function drawHardware(which){const canvas=document.getElementById(which+'-chart');
 const width=Math.max(80,canvas.clientWidth),height=34,d=Math.min(devicePixelRatio||1,2);
 canvas.width=width*d;canvas.height=height*d;const ctx=canvas.getContext('2d');ctx.scale(d,d);
 const points=hardwareHistory[which];if(points.length<2)return;
 ctx.strokeStyle=which==='cpu'?'#f06a75':'#8dbaff';ctx.lineWidth=2;ctx.beginPath();
 points.forEach((value,index)=>{const x=index*width/39,y=height-3-(Math.max(0,Math.min(100,value))/100)*(height-6);
  if(index===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();}
function showHardware(which,data,stale){const load=data.load_percent==null?NaN:Number(data.load_percent);
 document.getElementById(which+'-name').textContent=String(data.name||which.toUpperCase());
 document.getElementById(which+'-load').textContent=Number.isFinite(load)?Math.round(load)+'%':'—';
 const temp=data.temperature_c;
 const extra=which==='gpu'&&data.vram_total_mb?
  ' · VRAM '+data.vram_used_mb+'/'+data.vram_total_mb+' MB':'';
 document.getElementById(which+'-temp').textContent=
  (temp===null||temp===undefined?'Temperature unavailable':temp+' °C')+extra+
  (stale?' · Reading delayed':'');
 if(Number.isFinite(load)&&!stale){const values=hardwareHistory[which];values.push(load);
  if(values.length>40)values.shift();drawHardware(which);}}
async function loadHardware(){if(document.getElementById('hardware').hidden||document.hidden||!access||Date.now()>=accessExp)return;
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),4500);
 try{const response=await fetch('/api/hardware',{headers:{Authorization:'Bearer '+access},
  signal:controller.signal,cache:'no-store'});
  if(!response.ok)return;const data=await response.json();
  if(data.ok){showHardware('cpu',data.cpu,data.stale);showHardware('gpu',data.gpu,data.stale);}}
 catch(e){}finally{clearTimeout(timer);}}
setInterval(loadHardware,5000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)loadHardware();});
function updateSensors(){let status={};try{if(window.OrionNative&&OrionNative.deviceStatus)
 status=JSON.parse(OrionNative.deviceStatus());}catch(e){}
 document.getElementById('sensors').textContent='MIC '+(document.getElementById('mic').classList.contains('listening')?'●':'○')+
  ' CAM ○ ACCESS '+(status.accessibility?'●':'○')+(status.watching?' SCREEN ●':'');}
async function health(){const begun=performance.now();try{const r=await fetchLimited('/api/health',{},5000,1);
 if(!r.ok)throw new Error('health unavailable');
 const status=await r.json();if(!status.ok)throw new Error('health unavailable');
 document.getElementById('ping').textContent=Math.round(performance.now()-begun)+' ms';
 document.getElementById('tls').textContent=location.protocol==='https:'?'TLS ✓':'TLS ○';
 dot.textContent='●';dot.style.color='#D32F2F';updateSensors();}
 catch(e){dot.textContent='○';dot.style.color='#8A99AD';}
 if(QUICK_TUNNEL&&access){catchUpConfirms();catchUpPhoneActions();}}
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
function healthInterval(){return QUICK_TUNNEL?15000:(streamLive?90000:20000);}
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
let ttsAmpTimer=null,speechGeneration=0,nativeSpeechActive=false;
function paintSpk(){spkBtn.classList.toggle('off',!voiceOn);}
paintSpk();
spkBtn.addEventListener('click',()=>{voiceOn=!voiceOn;
 localStorage.setItem('orion_voice',voiceOn?'1':'0');paintSpk();
 if(!voiceOn){speechGeneration++;if(window.speechSynthesis)speechSynthesis.cancel();
  window.orionNativeSpeechState(false);
  if(window.OrionNative&&OrionNative.stopSpeaking)OrionNative.stopSpeaking();}});
window.orionNativeSpeechState=active=>{
 nativeSpeechActive=!!active;paintStop();
 clearInterval(ttsAmpTimer);ttsAmpTimer=null;
 if(active&&!document.hidden)ttsAmpTimer=setInterval(()=>faceMsg({amp:0.25+Math.random()*0.6}),90);
 else faceMsg({amp:0});};
function pickVoice(){if(!window.speechSynthesis)return null;
 const vs=speechSynthesis.getVoices();
 return vs.find(v=>/en[-_]GB/i.test(v.lang)&&/male|daniel|arthur|ryan/i.test(v.name))
  ||vs.find(v=>/en[-_]GB/i.test(v.lang))||vs.find(v=>/^en/i.test(v.lang))||null;}
function speak(text){
 if(!voiceOn||!text)return;
 if(window.OrionNative&&OrionNative.speak){OrionNative.speak(text);return;}
 if(!window.speechSynthesis)return;
 const generation=++speechGeneration;
 speechSynthesis.cancel();
 const parts=[];let remaining=String(text).trim();
 while(remaining.length>900){let cut=-1;
  for(let i=899;i>=0;i--)if(/[.!?\n]/.test(remaining[i])&&/\s/.test(remaining[i+1])){cut=i+1;break;}
  if(cut<1)cut=remaining.lastIndexOf(' ',900);
  if(cut<1)cut=900;
  if(/[\uD800-\uDBFF]/.test(remaining[cut-1])&&/[\uDC00-\uDFFF]/.test(remaining[cut]))cut--;
  parts.push(remaining.slice(0,cut).trim());remaining=remaining.slice(cut).trimStart();}
 if(remaining)parts.push(remaining);
 const finished=()=>{window.orionNativeSpeechState(false);};
 const next=()=>{if(generation!==speechGeneration||!voiceOn)return;
  if(!parts.length){finished();return;}
  const u=new SpeechSynthesisUtterance(parts.shift());const v=pickVoice();if(v)u.voice=v;
  u.rate=1.10;u.pitch=0.92;
  u.onstart=()=>{if(generation===speechGeneration)window.orionNativeSpeechState(true);};
  u.onend=next;u.onerror=()=>{if(generation===speechGeneration){finished();
   add('meta','Phone speech stopped before the reply finished. The complete answer is in the terminal.');}};
  speechSynthesis.speak(u);};next();}

// ── voice input ─────────────────────────────────────────────────────────────
// Two paths, chosen by what the device actually supports:
//   A. Browser SpeechRecognition (Chrome/Edge over HTTPS) — instant, on-device.
//   B. Record raw audio and let the desktop transcribe it (/api/transcribe).
// Path B is what makes voice work INSIDE the Android app: its WebView grants the
// microphone (getUserMedia) but has no SpeechRecognition API, so path A is blank
// there. Both need a secure context, which is why the desktop must serve HTTPS.
const micBtn=document.getElementById('mic');
// Android WebView can expose SpeechRecognition even when its service cannot
// recognise anything. The native app always uses the host transcription path.
const SR=window.OrionNative?null:(window.SpeechRecognition||window.webkitSpeechRecognition);
const AC=window.AudioContext||window.webkitAudioContext;
const canRecord=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia&&AC);
const nativeDictation=!!(window.OrionNative&&OrionNative.startDictation);
let backgroundVoiceTrace=null;
window.onBackgroundAudio=e=>{
 if(e.state==='listening'||e.state==='off')window.orionNativeSpeechState(false);
 if(e.state==='level'){microphoneLevel(e.level);return;}
 if(e.state==='reply'){
  if(e.user_text&&!log.querySelector('[data-hid="'+(+e.user_hid)+'"]')){
   const user=add('user',e.user_text);if(e.user_hid)user.dataset.hid=e.user_hid;}
  if(!e.hid||!log.querySelector('[data-hid="'+(+e.hid)+'"]')){
   const reply=add('orion',e.text);if(e.hid)reply.dataset.hid=e.hid;announce(e.text);
   for(const widget of Array.isArray(e.widgets)?e.widgets:[])renderPresentation(widget);}
  voiceStatus('Listening for ORION','Say “ORION” followed by your next request.');}
 else if(e.state==='transcript'){voiceHeard.textContent='Heard: '+e.text;}
 else if(e.state==='connecting')voiceStatus('Connecting the microphone',e.text);
 else if(e.state==='speaking'){window.orionNativeSpeechState(true);voiceStatus('ORION is speaking',e.text);}
 else if(e.state==='listening'||e.state==='status')voiceStatus(nativeSpeechActive?'ORION is speaking':'Listening for ORION',e.text);
 else if(e.state==='transcribing'){microphoneLevel(0);voiceStatus('Understanding your voice',e.text);}
 else if(e.state==='error'){microphoneLevel(0);voiceStatus('Microphone needs attention',e.text);}
 else if(e.state==='off'){microphoneLevel(0);
  if(voiceTitle.textContent!=='Microphone needs attention')voiceStatus('Ready when you are',e.text);}
};
if(nativeDictation){
 let listening=false,lastHeard=0;
 window.onNativeDictation=e=>{
  if(e.state==='listening'){
   listening=true;micBtn.disabled=false;micBtn.classList.add('listening');
   voiceHeard.textContent='';voiceStatus('Listening to you','Speak naturally. I’ll send your request after a pause.');
  }else if(e.state==='level'){
   microphoneLevel(e.level);if(e.level>0.006)lastHeard=Date.now();
   if(listening&&Date.now()-lastHeard<1400)voiceTitle.textContent='Hearing your voice';
  }else if(e.state==='transcribing'){
   listening=false;micBtn.classList.remove('listening');micBtn.disabled=true;
   microphoneLevel(0);voiceStatus('Understanding your voice','Your words are being transcribed on the PC.');
  }else if(e.state==='transcript'){
   voiceHeard.textContent='Heard: '+e.text;voiceStatus('Request sent','ORION is working on your request.');
   micBtn.disabled=false;document.getElementById('m').value=e.text;
   document.getElementById('f').dispatchEvent(new Event('submit'));
  }else if(e.state==='error'){
   listening=false;micBtn.classList.remove('listening');micBtn.disabled=false;
   microphoneLevel(0);voiceStatus('Microphone needs attention',e.text);
  }else if(e.state==='finished'){
   listening=false;micBtn.classList.remove('listening');micBtn.disabled=false;
  }
 };
 micBtn.addEventListener('click',()=>{
  if(listening)OrionNative.finishDictation();
  else{micBtn.disabled=true;voiceStatus('Opening your microphone','One moment…');OrionNative.startDictation();}
 });
}else if(!window.isSecureContext||(!SR&&!canRecord)){micBtn.style.display='none';}
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
  if(window.OrionNative&&OrionNative.reportMic)OrionNative.reportMic(true);
  rec.onresult=e=>submitText(e.results[0][0].transcript||'');
  rec.onerror=e=>{add('orion','Voice input failed: '+(e.error||'recogniser unavailable'));};
  rec.onend=()=>{micBtn.classList.remove('listening');rec=null;
   if(window.OrionNative&&OrionNative.reportMic)OrionNative.reportMic(false);};
  try{rec.start();}catch(e){micBtn.classList.remove('listening');rec=null;}}
 // Path B ---------------------------------------------------------------------
 let recording=false,starting=false,audioCtx=null,srcNode=null,procNode=null,stream=null,frames=[];
 let recordTimer=null,listenTrace=null,startedAt=0,lastVoice=0;
 async function startRecord(){
  if(starting||recording)return;starting=true;
  try{
   if(window.OrionNative&&OrionNative.stopSpeaking)OrionNative.stopSpeaking();
   if(window.OrionNative&&OrionNative.prepareDictation){OrionNative.prepareDictation();
    await new Promise(resolve=>setTimeout(resolve,300));}
   stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
   audioCtx=new AC();await audioCtx.resume();
   srcNode=audioCtx.createMediaStreamSource(stream);
   procNode=audioCtx.createScriptProcessor(4096,1,1);
   frames=[];startedAt=Date.now();lastVoice=0;
   procNode.onaudioprocess=e=>{
    if(!recording)return;
    const data=new Float32Array(e.inputBuffer.getChannelData(0));frames.push(data);
    let sum=0;for(const sample of data)sum+=sample*sample;
    if(Math.sqrt(sum/data.length)>0.012)lastVoice=Date.now();
    if(lastVoice&&Date.now()-lastVoice>1100&&Date.now()-startedAt>700)stopRecord();};
   srcNode.connect(procNode);procNode.connect(audioCtx.destination);
   recording=true;micBtn.classList.add('listening');
   micBtn.setAttribute('aria-label','Listening. Tap to finish.');
   listenTrace=addTrace('Listening… speak now. I’ll send your request after a pause.');
   recordTimer=setTimeout(()=>{if(recording)stopRecord();},15000);
   if(window.OrionNative&&OrionNative.reportMic)OrionNative.reportMic(true);
  }catch(e){
   if(stream)stream.getTracks().forEach(t=>t.stop());
   if(audioCtx)audioCtx.close();
   add('orion','Microphone could not start. Check ORION’s microphone permission and try again.');
  }finally{starting=false;}}
 async function stopRecord(){
  if(!recording)return;
  recording=false;micBtn.classList.remove('listening');
  clearTimeout(recordTimer);recordTimer=null;
  if(listenTrace){listenTrace.remove();listenTrace=null;}
  micBtn.setAttribute('aria-label','Speak to ORION');
  if(window.OrionNative&&OrionNative.reportMic)OrionNative.reportMic(false);
  const rate=audioCtx?audioCtx.sampleRate:48000;
  try{procNode.disconnect();srcNode.disconnect();
   stream.getTracks().forEach(t=>t.stop());audioCtx.close();}catch(e){}
  const wav=encodeWav(frames,rate);frames=[];
  if(!wav){add('orion','I didn’t catch anything.');return;}
  const thinking=addTrace('Transcribing microphone input…');
  micBtn.disabled=true;
  if(!(await ensureAccess())){thinking.remove();micBtn.disabled=false;add('orion','Pairing required.');return;}
  try{const r=await fetchLimited('/api/transcribe',{method:'POST',
    headers:{'Content-Type':'audio/wav','Authorization':'Bearer '+access},body:wav},90000);
   const j=await r.json();thinking.remove();
   if(j.ok&&j.text){submitText(j.text);}
   else{add('orion',j.error?('Voice: '+j.error):'I didn’t catch that.');}}
  catch(e){thinking.remove();add('orion','Voice link fault. Please try again.');}
  finally{micBtn.disabled=false;}}
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
  if(recording)stopRecord();else startRecord();});
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&recording)stopRecord();});}

// ── live state mirror: the phone orb tracks the desktop orb in real time ────
let es=null,esBackoff=2000,esFails=0;
function stateToMorph(s){return /STANDBY|OFFLINE|SHUTTING/i.test(s)?0:1;}
async function openEvents(){
 if(es){es.close();es=null;}
 if(QUICK_TUNNEL)return;
 if(!(await ensureAccess()))return;
 try{es=new EventSource('/api/events?token='+encodeURIComponent(access));}catch(e){return;}
 es.onopen=()=>{esBackoff=2000;esFails=0;streamLive=true;scheduleHealth();catchUpConfirms();};
 es.addEventListener('error',()=>{streamLive=false;scheduleHealth();});
 es.onmessage=ev=>{let d;try{d=JSON.parse(ev.data);}catch(e){return;}
  if(d.type==='state'){setState((d.value||'').toLowerCase());
   faceMsg({morph:stateToMorph(d.value||'')});}
  else if(d.type==='amplitude'){if(!ttsAmpTimer)faceMsg({amp:d.value});}
  else if(d.type==='speaking'){if(!d.value&&!ttsAmpTimer)faceMsg({amp:0});}
  else if(d.type==='emotion'&&d.params){faceMsg({emotion:d.params});}
  else if(d.type==='thought'&&d.text){queueThought(d.text);}
  else if(d.type==='action'&&d.action){doAction(d.action);}
  else if(d.type==='confirm'&&d.id){showConfirm(d);}
  else if(d.type==='confirm_result'&&d.id&&d.reply&&!resolvedHere[d.id]&&!submittingHere[d.id]){add('orion',d.reply);
   for(const widget of Array.isArray(d.widgets)?d.widgets:[])renderPresentation(widget);}};
 // The desktop ends every stream on a timer and networks drop; neither makes
 // the token invalid. Only repeated failures without a successful open do.
 es.onerror=()=>{if(es){es.close();es=null;}
  esFails++;if(esFails>=2){access='';}
  setTimeout(openEvents,esBackoff+Math.random()*500);
  esBackoff=Math.min(30000,esBackoff*1.7);};}
addEventListener('online',()=>{esBackoff=1000;openEvents();loadHardware();});
// Approval cards are sent only to this phone's live stream, so one raised
// while it was reconnecting would be lost without this catch-up.
async function catchUpConfirms(){
 if(!access)return;
 try{const r=await fetchLimited('/api/confirmations',{headers:{'Authorization':'Bearer '+access}},8000,1);
  if(!r.ok)return;const j=await r.json();
  (j.pending||[]).forEach(c=>{if(c&&c.id&&c.token)showConfirm(c);});}catch(e){}}
let phoneActionsBusy=false;
async function catchUpPhoneActions(){
 if(!access||phoneActionsBusy||document.hidden)return;phoneActionsBusy=true;
 try{const r=await fetchLimited('/api/phone-actions',{headers:{'Authorization':'Bearer '+access}},8000,0);
  if(r.ok){const j=await r.json();(j.events||[]).forEach(e=>{if(e.type==='action')doAction(e.action);});}}
 catch(e){}finally{phoneActionsBusy=false;}}
setInterval(()=>{if(QUICK_TUNNEL&&!document.hidden){catchUpConfirms();catchUpPhoneActions();}},3000);

// ── native phone hand-off ──────────────────────────────────────────────────
// When ORION asks the phone to call / text / navigate, use the app's native
// bridge (window.OrionNative) if we're inside the ORION app; otherwise render a
// tappable link so it still works in a plain browser. Nothing is dialled or
// sent without the user's own final tap.
function actLabel(a){
 if(a.kind==='open_app')return 'Open '+(a.package||'app');
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
 if(a.kind==='openUrl'){
  const url=String(a.url||'');return /^https?:\/\/[^\s/]+/i.test(url)?url:'';}
 return '';}
function doAction(a){
 if(!a||!a.kind)return;
 const card=add('orion confirm','DEVICE ACTION');
 const diff=document.createElement('div');diff.className='diff';
 diff.textContent='TARGET: '+actLabel(a)+(a.body?'\nPAYLOAD: '+JSON.stringify(a.body):'');
 card.appendChild(diff);
 mountSlide(card,()=>{card.remove();const N=window.OrionNative;
  if(N){try{
    if(a.kind==='call')N.call(a.number||'');
    else if(a.kind==='sms')N.sms(a.number||'',a.body||'');
    else if(a.kind==='open_app')N.openApp(a.package||'');
    else if(a.kind==='email')N.email(a.to||'',a.subject||'',a.body||'');
    else if(a.kind==='navigate')N.navigate(a.query||'');
    else if(a.kind==='map')N.map(a.query||'');
    else if(a.kind==='openUrl')N.openUrl(a.url||'');
    else if(a.kind==='share')N.share(a.text||'');
    addTrace('Handed '+actLabel(a)+' to Android');return;
   }catch(e){add('orion','Device action failed: '+e);return;}}
  const href=actHref(a);if(href){const link=document.createElement('a');link.href=href;
   link.rel='noopener';link.textContent=actLabel(a);const m=add('orion','');m.appendChild(link);}
  else add('orion','This action needs the ORION Android app.');},'SLIDE TO REVIEW');
 addDeny(card,()=>card.remove());}

function traceNode(label,detail){const d=document.createElement('details');d.className='trace';
 const s=document.createElement('summary');s.textContent='├─◆ '+String(label).slice(0,140);d.appendChild(s);
 if(detail){const p=document.createElement('pre');p.textContent=String(detail).slice(0,2000);d.appendChild(p);}
 return d;}
function addTrace(label,detail){return place(traceNode(label,detail));}
// Desktop thoughts can arrive several times a second: they are gathered and
// inserted once per frame, and only the newest MAX_TRACES of a burst survive.
const pendingThoughts=[];
function queueThought(text){pendingThoughts.push(text);
 if(pendingThoughts.length>MAX_TRACES)pendingThoughts.shift();onFrame(flushThoughts);}
function flushThoughts(){if(!pendingThoughts.length)return;const stick=nearBottom();
 const frag=document.createDocumentFragment();
 for(const t of pendingThoughts.splice(0))frag.appendChild(traceNode(t));
 log.appendChild(frag);if(stick)toBottom();trimLog();}
function mountSlide(card,onDone,hint){const slide=document.createElement('div');slide.className='slide';
 slide.setAttribute('role','slider');slide.setAttribute('aria-label',hint||'Slide to authorize');
 const fill=document.createElement('div');fill.className='fill';const thumb=document.createElement('div');
 thumb.className='thumb';thumb.textContent='››';const label=document.createElement('span');
 label.className='hint';label.textContent=hint||'SLIDE TO AUTHORIZE';
 slide.append(fill,label,thumb);card.appendChild(slide);
 let active=false,start=0,completed=false;
 const reset=()=>{fill.style.width='0';thumb.style.left='2px';};
 slide.addEventListener('pointerdown',e=>{if(completed||e.offsetX>48)return;
  active=true;start=e.clientX;slide.setPointerCapture(e.pointerId);});
 slide.addEventListener('pointermove',e=>{if(!active)return;const max=slide.clientWidth-44;
  const x=Math.max(0,Math.min(max,e.clientX-start));fill.style.width=(x+42)+'px';thumb.style.left=(x+2)+'px';});
 slide.addEventListener('pointerup',e=>{if(!active)return;active=false;
  const ratio=(e.clientX-start)/(slide.clientWidth-44);
  if(ratio>=.92){completed=true;onDone();}else reset();});
 slide.addEventListener('pointercancel',()=>{active=false;reset();});}
function addDeny(card,onDeny){const no=document.createElement('button');no.className='deny';
 no.textContent='DENY';no.onclick=onDeny;card.appendChild(no);}

// ── on-phone approval for irreversible/external actions (full-parity gate) ──
// ORION runs read/benign actions at once; anything that sends, deletes, buys or
// powers something off arrives here as a card you tap to approve — the desktop's
// confirmation gate, where you actually are.
const seenConfirms={},resolvedHere={},submittingHere={};
// Approval ids are remembered so a card is never shown twice; the desktop
// expires a pending approval after minutes, so only the latest few hundred
// ids can still matter and a long session need not keep every one.
function remember(ids,id){ids[id]=1;const keys=Object.keys(ids);
 for(let i=0;i<keys.length-256;i++)delete ids[keys[i]];}
async function sendConfirm(d,approve,card){
 const id=d.id;submittingHere[id]=1;card.style.opacity='.55';card.style.pointerEvents='none';
 const restore=()=>{delete submittingHere[id];delete seenConfirms[id];card.remove();showConfirm(d);};
 if(!(await ensureAccess())){restore();add('orion','I lost the link before I could act.');return;}
 try{const r=await fetch('/api/confirm',{method:'POST',
   headers:{'Content-Type':'application/json','Authorization':'Bearer '+access},
   body:JSON.stringify({id:id,token:d.token,decision:approve?'approve':'deny'})});
  const j=await r.json();
  if(r.ok&&(j.status==='done'||j.status==='denied')){
   delete submittingHere[id];remember(resolvedHere,id);card.remove();
   if(j.reply)add('orion',j.reply);else if(!approve)add('orion','Left it.');
   for(const widget of Array.isArray(j.widgets)?j.widgets:[])renderPresentation(widget);
  }else if(r.status===409){delete submittingHere[id];remember(resolvedHere,id);card.remove();
   add('orion','That authorization expired. Ask ORION to try again.');}
  else{restore();add('orion','Approval could not be sent: '+(j.error||r.status));}}
 catch(e){restore();add('orion','Approval failed: '+e);}}
function showConfirm(d){
 if(seenConfirms[d.id])return;remember(seenConfirms,d.id);
 const card=add('orion confirm','AUTHORIZATION REQUIRED');card.dataset.cid=d.id;
 const diff=document.createElement('div');diff.className='diff';
 diff.textContent='TARGET: '+(d.tool||'agent action')+'\nPARAMS: '+
  (d.parameterDiff||d.summary||'Review on desktop');
 card.appendChild(diff);
 mountSlide(card,()=>sendConfirm(d,true,card));
 addDeny(card,()=>sendConfirm(d,false,card));}

// ── streamed replies ─────────────────────────────────────────────────────────
// The reply is read as server-sent events from the POST itself, so the first
// words show while ORION is still writing; tokens are painted once per frame.
// Every event names its request and carries a rising sequence number: an
// event from another request, or one already applied (a duplicate from a
// retried or abandoned stream), is dropped rather than spliced into this
// answer. `restart` clears text that belonged to an abandoned attempt.
const inflight=new Set();
function paintStop(){stopBtn.hidden=!inflight.size&&!nativeSpeechActive;}
function newTurn(){const st={rid:'',lastSeq:0,text:'',node:null,userNode:null,thinking:null,
  done:false,stopped:false,ctl:window.AbortController?new AbortController():null};
 st.paint=()=>{if(st.done||!st.node)return;const stick=nearBottom();
  st.node.textContent=st.text;if(stick)toBottom();};
 return st;}
function clearThinking(st){if(st.thinking){st.thinking.remove();st.thinking=null;}}
function settle(st){st.done=true;clearThinking(st);inflight.delete(st);paintStop();
 if(st.node){st.node.classList.remove('streaming');st.node.classList.add('settled');
  st.node.removeAttribute('aria-busy');}}
function streamEvent(st,d){
 if(!d||st.done)return;
 if(d.type==='start'){if(st.rid)return;st.rid=String(d.request_id||'');
  if(d.user_hid&&st.userNode)st.userNode.dataset.hid=d.user_hid;return;}
 if(!st.rid||d.request_id!==st.rid)return;              // another request's event
 if(!(+d.seq>st.lastSeq))return;st.lastSeq=+d.seq;      // already applied
 if(d.type==='delta'){if(!d.text)return;st.text+=String(d.text);
  if(!st.node){clearThinking(st);st.node=add('orion streaming','');
   st.node.setAttribute('aria-busy','true');}
  onFrame(st.paint);}
 else if(d.type==='restart'){st.text='';if(st.node)st.node.textContent='';
  addTrace('Answer restarted'+(d.reason?' — '+d.reason:''));}
 else if(d.type==='done')finishTurn(st,String(d.reply||''),d.provider,d.hid,d.widgets);
 else if(d.type==='error'){if(d.cancelled)stoppedTurn(st);
  else failTurn(st,'Fault: '+(d.message||'unknown'));}}
function renderPresentation(p){if(!p||typeof p!=='object')return;
 if(p.operation==='display'&&typeof p.html==='string'){
  const card=add('orion widget',''),heading=document.createElement('strong');heading.textContent=p.title||'ORION display';
  const frame=document.createElement('iframe');frame.title=heading.textContent;
  frame.setAttribute('sandbox','allow-popups allow-popups-to-escape-sandbox');
  frame.style.cssText='width:100%;height:270px;border:0;background:#111;border-radius:12px';
  frame.srcdoc='<meta http-equiv="Content-Security-Policy" content="default-src '+"'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"+'">'+
   '<style>body{color:#eee;background:#111;font:15px system-ui;margin:16px;line-height:1.5}img{max-width:100%;height:auto}td,th{padding:8px}a{color:#67d4f3}pre{white-space:pre-wrap}</style>'+p.html;
  card.append(heading,frame);
 }else if(p.operation==='ui_theme'&&/^#[0-9a-f]{6}$/i.test(p.accent))document.documentElement.style.setProperty('--pri',p.accent);
 else if(p.operation==='ui_reset'){document.documentElement.style.removeProperty('--pri');faceMsg({presentation:p});}
 else if(p.operation==='ui_screen'&&(p.target==='transcript'||p.target==='all')){log.replaceChildren();}
 else if(p.operation==='ui_screen'){log.querySelectorAll('.widget').forEach(node=>node.remove());}
 else if(['ui_reactor','ui_orbit','ui_effect'].includes(p.operation))faceMsg({presentation:p});}
function finishTurn(st,reply,provider,hid,widgets){if(st.done)return;
 const stick=nearBottom();settle(st);
 const node=st.node||add('orion','');if(!renderReply(node,reply))node.textContent=reply;
 if(hid)node.dataset.hid=hid;
 for(const widget of Array.isArray(widgets)?widgets:[])renderPresentation(widget);
 if(stick)toBottom();onFrame(trimLog);
 if(provider){document.getElementById('model').textContent='ROUTE '+provider;
  if(window.OrionNative&&OrionNative.reportModel)OrionNative.reportModel(provider);}
 announce(reply);speak(reply);setState('listening');}
function failTurn(st,message,state){if(st.done)return;settle(st);
 add('orion',message);announce(message);setState(state||'listening');}
function stoppedTurn(st){if(st.done)return;settle(st);add('meta','Stopped.');setState('listening');}
function cancelTurn(st){if(st.done||st.stopped)return;st.stopped=true;
 if(st.rid&&access)fetch('/api/chat/cancel',{method:'POST',headers:{'Content-Type':'application/json',
  'Authorization':'Bearer '+access},body:JSON.stringify({request_id:st.rid})}).catch(()=>{});
 if(st.ctl){try{st.ctl.abort();}catch(e){}}
 stoppedTurn(st);}
stopBtn.addEventListener('click',()=>{for(const st of [...inflight])cancelTurn(st);
 speechGeneration++;if(window.speechSynthesis)speechSynthesis.cancel();
 if(window.OrionNative&&OrionNative.stopSpeaking)OrionNative.stopSpeaking();
 window.orionNativeSpeechState(false);});
async function readStream(st,res){const dec=new TextDecoder();let buf='';
 const ndjson=(res.headers.get('Content-Type')||'').includes('application/x-ndjson');
 const feed=chunk=>{buf+=chunk;let cut;
  if(buf.length>8000000)throw new Error('Reply frame exceeds the limit');
  if(ndjson){while((cut=buf.indexOf('\n'))>=0){const line=buf.slice(0,cut).trim();buf=buf.slice(cut+1);
   if(!line)continue;streamEvent(st,JSON.parse(line));}return;}
  while((cut=buf.indexOf('\n\n'))>=0){const frame=buf.slice(0,cut);buf=buf.slice(cut+2);
   let data='';for(const line of frame.split('\n'))if(line.startsWith('data:'))data+=line.slice(5).trim();
   if(!data)continue;let d;try{d=JSON.parse(data);}catch(e){continue;}streamEvent(st,d);}};
 if(!res.body||!res.body.getReader){feed(await res.text()+'\n\n');return;}
 // Read to the end even after `done` (it follows within milliseconds):
 // abandoning the body closes the connection, and over TLS the next message
 // would pay a fresh handshake instead of reusing it.
 const reader=res.body.getReader();
 for(;;){const {value,done}=await reader.read();if(done)break;
  feed(dec.decode(value,{stream:true}));}}
async function legacyReply(st,res){let data={};try{data=await res.json();}catch(e){}
 if(res.status===401){failTurn(st,'This device is no longer authorised. Pair it again from the desktop.');return;}
 if(res.status===429){failTurn(st,'Easy — too many requests at once. One moment.');return;}
 if(data.ok)finishTurn(st,String(data.reply||''),data.provider,0,data.widgets);
 else failTurn(st,'Fault: '+(data.error||'unknown'));}
async function runTurn(st,text){
 const body=JSON.stringify({message:text});
 const route=QUICK_TUNNEL?'/api/chat/frames':'/api/chat/stream';
 const post=async url=>{const timer=setTimeout(()=>{if(st.ctl)st.ctl.abort();},
   url==='/api/chat'?210000:30000);
  try{return await fetch(url,{method:'POST',body,signal:st.ctl?st.ctl.signal:undefined,
   headers:{'Content-Type':'application/json','Authorization':'Bearer '+access}});}
  finally{clearTimeout(timer);}};
 let res=await post(route);
 if(res.status===401){access='';localStorage.removeItem('orion_access');
  if(await ensureAccess())res=await post(route);}
 // A desktop from before streamed replies has no such route: ask the old way.
 if(res.status===404||res.status===405)res=await post('/api/chat');
 const contentType=res.headers.get('Content-Type')||'';
 if(!contentType.includes('text/event-stream')&&!contentType.includes('application/x-ndjson'))return legacyReply(st,res);
 await readStream(st,res);
 if(!st.done)failTurn(st,'Link fault: the reply was cut off.','standby');}
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const inp=document.getElementById('m');const text=inp.value.trim();if(!text)return;
 inp.value='';const st=newTurn();st.userNode=add('user',text,true);
 st.thinking=add('orion skeleton','');st.thinking.setAttribute('aria-label','Preparing reply');
 faceMsg({morph:1});setState('processing');
 inflight.add(st);paintStop();
 try{if(!(await ensureAccess())){clearThinking(st);st.done=true;
   add('orion','Pairing required. Start a pairing window on the desktop and try again.');return;}
  await runTurn(st,text);}
 catch(err){if(st.stopped)stoppedTurn(st);else failTurn(st,'Link fault: '+err,'standby');}
 finally{inflight.delete(st);paintStop();}
});

// ── older conversation, paged back from the desktop on request ──────────────
let earlierMore=false,historyFloor=0,loadingEarlier=false,noHistory=false;
function paintEarlier(){const show=!noHistory&&(earlierMore||evicted);
 if(show){earlier.hidden=false;earlier.removeAttribute('aria-disabled');
  earlier.textContent='Show earlier messages';}
 else if(document.activeElement===earlier){earlier.setAttribute('aria-disabled','true');
  earlier.textContent='No earlier messages';}
 else earlier.hidden=true;}
async function probeHistory(){if(!access)return;
 try{const r=await fetchLimited('/api/history?limit=0',{headers:{'Authorization':'Bearer '+access}},8000,1);
  if(r.status===404){noHistory=true;return;}if(!r.ok)return;const j=await r.json();
  historyFloor=+j.next||0;earlierMore=!!j.more;paintEarlier();}catch(e){}}
function oldestHid(){let low=0;for(const n of log.querySelectorAll('[data-hid]')){
 const h=+n.dataset.hid;if(h>0&&(!low||h<low))low=h;}return low;}
async function loadEarlier(){
 if(loadingEarlier||earlier.getAttribute('aria-disabled')==='true')return;
 loadingEarlier=true;earlier.setAttribute('aria-busy','true');
 try{if(!(await ensureAccess()))return;const before=oldestHid()||historyFloor;
  const r=await fetchLimited('/api/history?limit=50'+(before?'&before='+before:''),
   {headers:{'Authorization':'Bearer '+access}},8000,1);
  if(r.status===404){noHistory=true;return;}if(!r.ok)return;
  const j=await r.json(),frag=document.createDocumentFragment();
  for(const it of (j.items||[])){const hid=+(it&&it.id)||0;
   if(!hid||log.querySelector('[data-hid="'+hid+'"]'))continue;
   const n=document.createElement('div');n.className='msg hist '+(it.role==='user'?'user':'orion');
   const text=String(it.text||'');if(it.role==='user'||!renderReply(n,text))n.textContent=text;
   n.dataset.hid=hid;frag.appendChild(n);}
  // Older messages go in above what is on screen; the offset grows by their
  // height so the view does not move.
  const height=log.scrollHeight,top=log.scrollTop;earlier.after(frag);
  log.scrollTop=top+(log.scrollHeight-height);
  earlierMore=!!j.more;evicted=false;}
 catch(e){}
 finally{loadingEarlier=false;earlier.removeAttribute('aria-busy');paintEarlier();}}
earlier.addEventListener('click',loadEarlier);
// Tasks: what ORION is doing, read from the desktop's job list. Polled only
// while the sheet is open and the page is visible; closed, it costs nothing.
const tasksEl=document.getElementById('tasks'),tasksBtn=document.getElementById('tasksbtn');
const tasksList=document.getElementById('tasks-list'),tasksSum=document.getElementById('tasks-sum');
const tasksAppr=document.getElementById('tasks-appr');
const TASK_WORDS={queued:'Queued',running:'Running',waiting_for_approval:'Waiting for approval',
 paused:'Paused',completed:'Completed',failed:'Failed',cancelled:'Cancelled',
 interrupted:'Interrupted — can be resumed'};
let tasksTimer=0;
function taskProgress(t){if(['completed','failed','cancelled'].includes(t.state))return '';
 const step=+t.step||0,total=+t.total||0;
 if(typeof t.fraction==='number')return Math.round(Math.max(0,Math.min(1,t.fraction))*100)+'%'+(total?' · step '+step+' of '+total:'');
 if(total)return 'step '+step+' of '+total;return t.state==='running'?'in progress':'';}
function renderTasks(j){const jobs=j.jobs||[],counts=j.counts||{};
 const live=['running','queued','waiting_for_approval','paused','interrupted'].filter(s=>counts[s]);
 tasksSum.textContent=live.length?live.map(s=>counts[s]+' '+TASK_WORDS[s].split(' —')[0].toLowerCase()).join(', '):'Nothing in progress.';
 const appr=j.approvals||[];tasksAppr.hidden=!appr.length;
 tasksAppr.textContent=appr.length?(appr.length+' step'+(appr.length>1?'s are':' is')+
  ' waiting for your approval ('+appr.map(a=>a.tool).join(', ')+'). Approve '+(appr.length>1?'them':'it')+
  ' on the desktop Tasks page.'):'';
 const frag=document.createDocumentFragment();
 for(const t of jobs){const li=document.createElement('li');
  const name=document.createElement('strong');name.textContent=String(t.label||t.name||t.id);
  const state=document.createElement('span');state.className='t-state';
  let words=TASK_WORDS[t.state]||String(t.state||'');
  if(t.state==='queued'&&t.position)words='Queued — number '+t.position+' in line';
  const prog=taskProgress(t);state.textContent=words+(prog?' · '+prog:'');li.append(name,state);
  const detail=String(t.result||t.note||'').trim();
  if(detail){const n=document.createElement('span');n.className='t-note';n.textContent=detail;li.append(n);}
  frag.appendChild(li);}
 if(!jobs.length){const li=document.createElement('li');li.textContent='No tasks yet.';frag.appendChild(li);}
 tasksList.replaceChildren(frag);}
async function loadTasks(){if(tasksEl.hidden||document.hidden)return;
 try{if(!(await ensureAccess()))return;
  const r=await fetchLimited('/api/tasks?limit=30',{headers:{'Authorization':'Bearer '+access}},8000,1);
  if(r.status===404){tasksSum.textContent='This desktop does not share tasks yet.';return;}
  if(!r.ok){tasksSum.textContent='Tasks are unavailable right now.';return;}
  renderTasks(await r.json());}
 catch(e){tasksSum.textContent='Tasks are unavailable right now.';}}
function openTasks(){tasksEl.hidden=false;tasksSum.textContent='Loading…';loadTasks();
 document.getElementById('options').open=false;
 clearInterval(tasksTimer);tasksTimer=setInterval(loadTasks,5000);
 document.getElementById('tasks-close').focus();}
function closeTasks(){tasksEl.hidden=true;clearInterval(tasksTimer);tasksTimer=0;tasksBtn.focus();}
tasksBtn.addEventListener('click',openTasks);
document.getElementById('tasks-close').addEventListener('click',closeTasks);
tasksEl.addEventListener('keydown',e=>{if(e.key==='Escape')closeTasks();});
window.openOrionTasks=openTasks;
window.openOrionHardware=()=>{document.getElementById('options').open=false;
 document.getElementById('hardware').hidden=false;loadHardware();
 document.getElementById('hardware-close').focus();};
document.getElementById('hardwarebtn').onclick=window.openOrionHardware;
document.getElementById('hardware-close').onclick=()=>{document.getElementById('hardware').hidden=true;};
window.openOrionDiagnostics=async()=>{document.getElementById('options').open=false;
 document.getElementById('diagnostics').hidden=false;
 const view=document.getElementById('audio-diagnostics');view.textContent='Checking microphone connection…';
 try{if(!await ensureAccess()){view.textContent='Pair this phone to check its audio connection.';return;}
  const response=await fetchLimited('/api/audio/status',{headers:{Authorization:'Bearer '+access}},5000,1);
  if(!response.ok)throw new Error();const data=await response.json();
  let local={};try{if(window.OrionNative)local=JSON.parse(OrionNative.deviceStatus());}catch(e){}
  view.textContent='Phone microphone permission: '+(local.microphone?'allowed':'not granted')+'\n'+
   'Background LISTEN: '+(local.background_audio?'on':'off')+'\n'+
   'Audio socket on PC: '+(data.connected?'connected':'not connected')+'\n'+
   'Audio received: '+data.pcm_bytes+' bytes\nUtterances received: '+data.utterances+
   '\nTranscriptions: '+data.transcriptions+'\nWake requests: '+data.requests+
   '\nLast result: '+data.state;
 }catch(e){view.textContent='Audio diagnostics are unavailable. Check the PC connection.';}};
document.getElementById('diagnosticsbtn').onclick=window.openOrionDiagnostics;
document.getElementById('diagnostics-close').onclick=()=>{document.getElementById('diagnostics').hidden=true;};
for(const id of ['hardware','diagnostics'])document.getElementById(id).addEventListener('keydown',e=>{
 if(e.key==='Escape')document.getElementById(id).hidden=true;});
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
        # Uncertain retained pairing state must not create replacement key
        # material before we have established that the registry is usable.
        self._devices = self._load_devices()
        self._secret = self._load_secret()
        self._pairings: dict[str, float] = {}      # sha256(code) → expiry epoch

    # ── secret + device registry persistence ─────────────────────────────────

    def _load_secret(self) -> bytes:
        secret_path = self.config_dir / "remote_secret.key"
        if secret_path.exists():
            # Do not silently replace a temporarily unreadable or corrupt key:
            # that would invalidate every previously paired device.
            existing = secret_path.read_text(encoding="utf-8").strip()
            # Mark XXXX.3: the key is sealed to this Windows account
            # (key_seal.py); a plain one is sealed once it has been read.
            sealed = key_seal.is_sealed(existing)
            plain = key_seal.open_value(existing) if sealed else existing
            if sealed and not plain:
                raise ValueError("Remote signing key was sealed by another Windows account")
            try:
                key = bytes.fromhex(plain)
            except ValueError as exc:
                raise ValueError("Remote signing key is invalid") from exc
            if len(key) != 32:
                raise ValueError("Remote signing key must be 32 bytes")
            if not sealed and key_seal.available():
                try:
                    atomic_write_text(secret_path, key_seal.seal(plain), encoding="utf-8")
                except Exception:
                    pass                      # still plain; read again next start
            return key
        secret = secrets.token_bytes(32)
        # An ephemeral signing key would invalidate every issued access token
        # after restart, even if the device registry was saved successfully.
        self.config_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(secret_path, key_seal.seal(secret.hex()), encoding="utf-8")
        return secret

    def _load_devices(self) -> dict[str, dict[str, Any]]:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate registry field")
                result[key] = value
            return result

        try:
            data = json.loads(self._devices_path.read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object)
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            # Treating an unreadable registry as a new installation allowed
            # the next pairing to overwrite all retained device identities.
            # Keep the file untouched and surface a local repair requirement.
            raise ValueError(
                "Remote device registry could not be read. Restore or repair "
                "remote_devices.json before enabling phone access.") from exc
        if not isinstance(data, dict) or any(
                not isinstance(value, dict)
                or type(value.get("revoked")) is not bool
                or not isinstance(value.get("refresh_sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", value["refresh_sha256"]) is None
                for value in data.values()):
            raise ValueError(
                "Remote device registry has invalid authentication fields or structure. Restore or repair "
                "remote_devices.json before enabling phone access.")
        return {str(key): dict(value) for key, value in data.items()}

    def _save_devices(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._devices_path,
            json.dumps(self._devices, indent=2), encoding="utf-8")

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
        try:
            self._save_devices()
        except Exception:
            # Never issue credentials for a device absent from the durable
            # registry. The pairing code has already been consumed.
            self._devices.pop(device_id, None)
            raise
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
        # compare_digest rejects non-ASCII str operands with TypeError. Tokens
        # arrive from untrusted HTTP/WebSocket clients: reject, never return 500.
        if not signature.isascii() or not hmac.compare_digest(signature, expected):
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
        from . import phone_control
        if not phone_control.enabled():
            # Mark XXXX.4: this route called the dispatcher directly, so phone
            # control off still let a lost phone read private history.
            phone_control.record(tool, "-", "off", device=device_id)
            task.update(status="denied", error=phone_control.OFF_REPLY)
            self._remember(task, device_id)
            return task
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




def _on_windows() -> bool:
    """Whether Windows-only facilities (the firewall rule) apply. A function so
    tests can say "Windows" without rewriting os.name for the whole process,
    which makes pathlib build WindowsPath objects on Linux."""
    return os.name == "nt"


def _tls_requested() -> bool:
    """ORION_REMOTE_TLS asks for HTTPS; plain HTTP is only ever the default
    when it is NOT set, never a fallback after a TLS fault."""
    return os.getenv("ORION_REMOTE_TLS", "").strip().lower() in {"1", "true", "yes", "on"}


class RemoteTLSError(RuntimeError):
    """TLS was requested but could not be set up, so no listener is opened.

    Returning ``None`` here used to mean "serve plain HTTP", the same value as
    a deliberately unencrypted uplink: a corrupt key or an unwritable folder
    quietly turned a phone that had been set up for HTTPS back into clear
    text. The failure is now typed, names the stage and carries the repair."""

    def __init__(self, stage: str, cause: Any, repair: str) -> None:
        self.stage = stage
        self.repair = repair
        self.cause = first_line(cause, 160) if cause is not None else ""
        super().__init__(f"TLS was requested but {stage} failed ({self.cause}). {repair}")


class _BodyTooLarge(ValueError):
    """A request body over the endpoint's size limit (answered with 413)."""


class _BadField(ValueError):
    """A request field of the wrong type or size (answered with 400)."""


#: Accepted JSON fields per endpoint: name -> (allowed types, max string length
#: or 0 for none). Unknown fields are ignored so an older or newer phone app
#: keeps working; a known field of the wrong type is refused rather than
#: str()-ed into something the handler never expected. Limits are several
#: times the real sizes (pairing code 9 chars, device id 32, refresh token 43).
_ACCESS = ((str,), 1024)
_REQUEST_SCHEMAS: dict[str, dict[str, tuple[tuple[type, ...], int]]] = {
    "pair": {"code": ((str,), 32), "device_name": ((str,), 200)},
    "token": {"device_id": ((str,), 64), "refresh_token": ((str,), 256)},
    "task": {"tool": ((str,), 128), "args": ((dict,), 0), "access_token": _ACCESS},
    "chat": {"message": ((str,), 0), "access_token": _ACCESS},
    "speech": {"text": ((str,), 1200)},
    "chat_cancel": {"request_id": ((str,), 64), "access_token": _ACCESS},
    "confirm": {"id": ((str,), 64), "token": ((str,), 128), "decision": ((str, bool), 16),
                "approve": ((str, bool), 16), "access_token": _ACCESS},
}
_TYPE_NAMES = {str: "a string", bool: "true or false", dict: "an object"}


def _check_fields(payload: dict[str, Any], schema: dict[str, tuple[tuple[type, ...], int]]) -> None:
    """Raise _BadField for the first known field of the wrong type or size."""
    for name, (kinds, limit) in schema.items():
        value = payload.get(name)
        if value is None:
            continue
        # bool is an int subclass, and no field here is numeric, so an exact
        # check is right: {"code": true} is not a pairing code.
        if type(value) not in kinds:
            wanted = " or ".join(_TYPE_NAMES.get(k, k.__name__) for k in kinds)
            raise _BadField(f"'{name}' must be {wanted}")
        if limit and isinstance(value, str) and len(value) > limit:
            raise _BadField(f"'{name}' is too long")


# ── streamed phone replies ────────────────────────────────────────────────────
#
# /api/chat returns nothing until the whole answer exists, so on a phone the
# first word of a slow model reply arrived with the last one. The streamed
# route writes the answer as it is generated: `start`, then `delta`s, an
# explicit `restart` if the router abandons one provider mid-answer for
# another (so the phone clears text that belonged to the failed attempt
# instead of splicing two answers together), and one `done` or `error`.

def _stream_support(generate: Any) -> Optional[bool]:
    """Whether ``generate_text`` takes the ``on_delta``/``on_restart`` pair.

    True when both are named parameters, False when the signature plainly
    lacks them, None when it cannot tell (a ``**kwargs`` wrapper, or no
    readable signature) and a call has to find out."""
    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        return None
    if "on_delta" in params and "on_restart" in params:
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return False


async def _generate_text(router: Any, prompt: str, system_extra: str,
                         on_delta: Optional[Callable[[str], None]],
                         on_restart: Optional[Callable[[str], None]]) -> tuple[Any, str]:
    """``router.generate_text``, streamed when both the caller and the router
    support it; otherwise the plain call, which ends in a single ``done``."""
    if on_delta is None or on_restart is None:
        return await router.generate_text(prompt, system_extra=system_extra)
    support = _stream_support(router.generate_text)
    if support is False:
        return await router.generate_text(prompt, system_extra=system_extra)
    try:
        return await router.generate_text(prompt, system_extra=system_extra,
                                          on_delta=on_delta, on_restart=on_restart)
    except TypeError as exc:
        # Only a router that could not say what it accepts gets a second call,
        # and only when the refusal names the new keywords: a TypeError from
        # inside a generation that had started must not run it twice.
        text = str(exc)
        if support is None and ("on_delta" in text or "on_restart" in text):
            return await router.generate_text(prompt, system_extra=system_extra)
        raise


class _ChatStream:
    """One streamed phone answer: its unsent events, in order, and the
    generation it can cancel.

    Deltas that arrive while a write is in flight are merged into one event,
    so a fast local model costs the phone a frame per flush rather than a
    frame per token, and the unsent backlog can never outgrow the answer."""

    #: What the phone is told when its answer stops for each reason.
    ENDINGS = {
        "phone": "Cancelled.",
        "disconnect": "The phone disconnected.",
        "shutdown": "ORION is shutting down.",
    }

    def __init__(self, request_id: str, device_id: str,
                 trace: request_trace.RequestTrace, *, ndjson: bool = False) -> None:
        self.request_id = request_id
        self.device_id = device_id
        self.trace = trace
        self.ndjson = ndjson
        self.widgets: list[dict] = []
        self.task: Optional[asyncio.Task] = None
        self.outbox: deque[tuple[str, dict[str, Any]]] = deque()
        self.seq = 0
        self.deltas = 0
        self.cancel_reason = ""
        self.cancel_at = 0.0
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()

    # ── producer side: called by the router while it generates ─────────────

    def _on_loop(self, fn: Callable[..., None], *args: Any) -> None:
        """Run *fn* on the gateway's loop. A provider may report tokens from a
        worker thread; the outbox is only ever touched from the loop."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            fn(*args)
        else:
            try:
                self._loop.call_soon_threadsafe(fn, *args)
            except RuntimeError:
                pass                                  # the loop has closed

    def delta(self, text: str) -> None:
        self._on_loop(self._delta, text)

    def restart(self, reason: str = "") -> None:
        self._on_loop(self._restart, reason)

    def _delta(self, text: Any) -> None:
        text = str(text or "")
        if not text or self.cancel_reason:
            return
        if self.deltas == 0:
            self.trace.mark("first_token")
        self.deltas += 1
        if self.outbox and self.outbox[-1][0] == "delta":
            self.outbox[-1][1]["text"] += text
        else:
            self.outbox.append(("delta", {"text": text}))
        self._wake.set()

    def _restart(self, reason: Any) -> None:
        if self.cancel_reason:
            return
        # Unsent text belongs to the attempt being abandoned; text already
        # sent is cleared by the phone when this event reaches it.
        self.outbox = deque(item for item in self.outbox if item[0] != "delta")
        self.outbox.append(("restart", {"reason": str(reason or "")[:160]}))
        self.trace.mark("stream_restart")
        self._wake.set()

    # ── consumer side: the handler writing to the phone ─────────────────────

    def wake(self) -> None:
        self._wake.set()

    async def wait(self, timeout: float) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    def cancel(self, reason: str) -> None:
        if self.cancel_reason:
            return
        self.cancel_reason = reason
        self.cancel_at = time.monotonic()
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self._wake.set()

    async def send(self, response: Any, kind: str, **data: Any) -> None:
        payload = {"type": kind, "request_id": self.request_id, "seq": self.seq, **data}
        self.seq += 1
        # ensure_ascii=False keeps non-English replies compact; json escapes
        # every newline, so a reply can never break the event framing.
        encoded = json.dumps(payload, ensure_ascii=False)
        frame = encoded + "\n" if self.ndjson else f"event: {kind}\ndata: {encoded}\n\n"
        await response.write(frame.encode("utf-8"))


class _PhoneHistory:
    """This desktop session's phone conversation, per paired device.

    The page keeps only its latest messages in the DOM; "Show earlier" pages
    back through this. It is bounded per device and in the number of devices,
    lives only as long as the gateway, and is never shown to another device.
    Long-term recall stays where it always was: the episodic memory."""

    PER_DEVICE = 300
    DEVICES = 16
    TEXT_LIMIT = 65536
    CHAR_BUDGET = 1_048_576
    PAGE_MAX = 100

    def __init__(self) -> None:
        # device id -> [last id issued, entries oldest first]
        self._by_device: OrderedDict[str, list[Any]] = OrderedDict()

    def add(self, device_id: str, role: str, text: str, request_id: str = "") -> int:
        slot = self._by_device.get(device_id)
        if slot is None:
            slot = self._by_device[device_id] = [0, deque(), 0]
            while len(self._by_device) > self.DEVICES:
                self._by_device.popitem(last=False)
        self._by_device.move_to_end(device_id)
        slot[0] += 1
        full_text = str(text)
        stored = full_text[:self.TEXT_LIMIT]
        slot[1].append({"id": slot[0], "role": role, "text": stored,
                        "truncated": len(full_text) > self.TEXT_LIMIT,
                        "at": round(time.time(), 3), "request_id": request_id})
        slot[2] += len(stored)
        while len(slot[1]) > self.PER_DEVICE or slot[2] > self.CHAR_BUDGET:
            slot[2] -= len(slot[1].popleft()["text"])
        return slot[0]

    def page(self, device_id: str, before: int | None, limit: int
             ) -> tuple[list[dict[str, Any]], bool, int]:
        """Up to *limit* entries older than *before* (all when None), oldest
        first; whether older ones remain; and the id the next entry gets."""
        slot = self._by_device.get(device_id)
        if slot is None:
            return [], False, 1
        older = [e for e in slot[1] if before is None or e["id"] < before]
        chosen = older[max(0, len(older) - limit):] if limit > 0 else []
        more = len(older) > len(chosen)
        return [dict(e) for e in chosen], more, slot[0] + 1


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
    # transcription) finish before aiohttp cancels it; aiohttp then waits as
    # long again for the cancelled handler. aiohttp's own default is 60 s per
    # phase, and an open /api/events stream never finishes by itself, so one
    # connected phone held shutdown past the app's 25 s watchdog. At 2.0 s a
    # connected companion app (its /v1/device/uplink socket was never closed
    # by stop) still cost 4.0 s, MEASURED 11 Oct 2026; the streams stop()
    # ends itself no longer depend on this at all.
    _SHUTDOWN_GRACE_S = 0.5
    # Includes power, websocket and runner cleanup together, never per phone.
    # wait_for waits for cancellation to finish, so it is not a hard bound on
    # a third-party coroutine that swallows CancelledError.
    _STOP_TIMEOUT_S = 2.0
    _STOP_CANCEL_GRACE_S = 0.1
    # One websocket's close handshake. aiohttp waits 10 s for the phone's
    # reply, and an asleep or out-of-range phone never sends one.
    _SOCKET_CLOSE_S = 0.5
    # How long a connection may still hold unsent bytes at shutdown before its
    # phone is taken to have stopped reading and the connection is dropped.
    _STALL_SETTLE_S = 0.25
    # Live /api/events streams. aiohttp no longer cancels a handler when its
    # client goes away, so a phone that dropped off the network left its
    # stream (and queue, still collecting events) waiting on a 20 s keepalive
    # before the dead socket was noticed; every reconnect in that window added
    # another. Streams now look for a closed socket every tick, a device keeps
    # at most a few (a new one retires its oldest), and the total is capped.
    _EVENT_STREAMS_PER_DEVICE = 3
    _EVENT_STREAMS_MAX = 32
    _EVENTS_TICK_S = 5.0
    _EVENTS_KEEPALIVE_S = 20.0
    # Streamed chat replies (/api/chat/stream).
    _MAX_STREAMS_PER_DEVICE = 3     # answers one phone may have in progress
    _STREAM_TICK_S = 0.5            # how often a silent answer checks its socket
    _STREAM_KEEPALIVE_S = 15.0      # comment line while a slow model thinks
    _STREAM_FLUSH_S = 0.04          # tokens gathered between writes after the first
    _CANCEL_GRACE_S = 2.0           # wait for a cancelled generation to settle

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
        self.host   = os.getenv("ORION_REMOTE_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.port   = int(os.getenv("ORION_REMOTE_PORT", "8000") or 8000)
        exposure = os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower()
        # A host-local proxy needs a loopback upstream. An isolated managed
        # container instead needs a listener reachable by the platform's TLS
        # proxy; its application port must never be published directly.
        self._platform_proxy_mode = exposure == "platform-proxy"
        self._reverse_proxy_mode = exposure in {"reverse-proxy", "platform-proxy"}
        self._public_endpoint = (public_https_endpoint(
            os.getenv("ORION_REMOTE_PUBLIC_URL", ""))
            if self._reverse_proxy_mode else None)
        if exposure == "reverse-proxy" and self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("A reverse-proxy uplink must bind to loopback only.")
        if self._platform_proxy_mode and self.host != "0.0.0.0":
            raise ValueError("A platform-proxy uplink must bind to 0.0.0.0 inside its isolated container.")
        if self._reverse_proxy_mode and self._public_endpoint is None:
            raise ValueError("ORION_REMOTE_PUBLIC_URL must be a named HTTPS origin on port 443.")
        self._serve_mode = exposure in {"serve", "funnel"} or (
            not exposure and self.host in {"127.0.0.1", "::1", "localhost"}
            and self.port == 8000)
        self._public_https = self._serve_mode or self._reverse_proxy_mode
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
        if dispatcher is not None:
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
        #: The phone that most recently spoke to ORION: where an action he
        #: starts himself (a reminder's call) goes, instead of every phone.
        self._last_active_device = ""
        self._closing = False               # set by stop(): SSE streams end
        self._stop_task: asyncio.Task | None = None
        self._shutdown_tasks: set[asyncio.Task] = set()
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
        #: The home Wi-Fi listener (remote_lan.py): the phone reaches the PC
        #: directly on the LAN, its certificate pinned, private peers only.
        self.lan = LanLink(self)
        self._last_emotion_name = ""
        # Offline speech-to-text for phone voice input, built on first use (a
        # Whisper model is slow to load, so it is deferred until actually asked).
        self._transcriber: Any = self._UNSET
        #: Streamed answers in progress, by request id (for /api/chat/cancel).
        self._chat_streams: dict[str, _ChatStream] = {}
        self._audio_sessions: dict[str, Any] = {}
        self._audio_diagnostics: dict[str, dict[str, Any]] = {}
        self._speech_devices: set[str] = set()
        self._phone_actions: dict[str, deque[tuple[float, dict[str, Any]]]] = {}
        self._history = _PhoneHistory()

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
        app = web.Application(middlewares=[security_mw, self.lan.middleware()],
                              client_max_size=self._MAX_AUDIO_BYTES + 65536)
        app.router.add_get("/",                    self._handle_page)
        app.router.add_get("/face",                self._handle_face)
        app.router.add_get("/three/{version}/{path:.+}", self._handle_three)
        app.router.add_get("/manifest.webmanifest", self._handle_manifest)
        app.router.add_get("/sw.js",               self._handle_sw)
        app.router.add_get("/icon-192.png",        self._handle_icon)
        app.router.add_get("/icon-512.png",        self._handle_icon)
        app.router.add_get("/download/orion-companion.apk", self._handle_companion_apk)
        app.router.add_get("/api/health",          self._handle_health)
        app.router.add_get("/api/hardware",        self._handle_hardware)
        app.router.add_get("/api/audio/status",    self._handle_audio_status)
        app.router.add_get("/api/events",          self._handle_events)
        app.router.add_post("/api/chat",           self._handle_chat)
        app.router.add_post("/api/chat/stream",    self._handle_chat_stream)
        app.router.add_post("/api/chat/frames",    self._handle_chat_stream)
        app.router.add_post("/api/chat/cancel",    self._handle_chat_cancel)
        app.router.add_get("/api/history",         self._handle_history)
        app.router.add_get("/api/tasks",           self._handle_tasks)
        app.router.add_post("/api/transcribe",     self._handle_transcribe)
        app.router.add_post("/api/speech",         self._handle_speech)
        app.router.add_post("/api/speech/stream",  self._handle_speech)
        app.router.add_post("/v1/device/ai-profile", self._handle_mobile_ai_profile)
        app.router.add_get("/v1/audio/stream",      self._handle_audio_stream)
        app.router.add_post("/api/confirm",        self._handle_confirm)
        app.router.add_get("/api/confirmations",   self._handle_confirmations)
        app.router.add_get("/api/phone-actions",   self._handle_phone_actions)
        app.router.add_get("/api/endpoints",       self._handle_endpoints)
        app.router.add_post("/v1/auth/pair",       self._handle_pair)
        app.router.add_post("/v1/auth/token",      self._handle_token)
        app.router.add_post("/v1/agent/tasks",     self._handle_agent_task)
        app.router.add_get("/v1/agent/tasks/{id}", self._handle_agent_task_status)
        from .remote_mobile_ws import register_mobile_ws_routes
        register_mobile_ws_routes(app, self)
        from .remote_power import register_pc_power_routes
        register_pc_power_routes(app, self)
        self.tool_gate.mobile_viewport = self.mobile_viewport
        return app

    @staticmethod
    def _lan_ip() -> str:
        """Best-effort LAN address for the 'open this on your phone' hint."""
        try:
            return next(iter(lan_ipv4_addresses()), "127.0.0.1")
        except Exception:
            return "127.0.0.1"

    async def _ensure_firewall_access(self) -> None:
        """Best-effort: make sure Windows Firewall allows inbound LAN traffic to
        the uplink port.  A blocked inbound rule is the usual cause of a phone on
        the same Wi-Fi seeing 'refused to connect' even though the server is up.
        LAN access is limited to private/domain networks; public-profile access
        is limited to Tailscale peer ranges. Idempotent, admin-aware, and never
        raises."""
        if self.host in {"127.0.0.1", "::1", "localhost"}:
            return
        if not _on_windows() or os.getenv("ORION_REMOTE_FIREWALL", "1").strip().lower() in {
                "0", "false", "no", "off"}:
            return
        rule = "ORION Remote Uplink"
        if os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() == "tailnet-direct":
            rules = ((f"{rule} (Direct Tailscale HTTPS)", "any",
                      "100.64.0.0/10", self.host),)
        else:
            rules = (
                (rule, "private,domain", None, None),
                (f"{rule} (Tailscale)", "public",
                 "100.64.0.0/10,fd7a:115c:a1e0::/48", None),
            )
        try:
            def _run(args: list[str]) -> Any:
                return subprocess.run(
                    args, capture_output=True, text=True, errors="replace", timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            for name, profile, remote_ips, local_ip in rules:
                show = await asyncio.to_thread(
                    _run, ["netsh", "advfirewall", "firewall", "show", "rule",
                           f"name={name}"])
                if show.returncode == 0 and name in (show.stdout or ""):
                    continue

                args = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={name}", "dir=in", "action=allow", "protocol=TCP",
                    f"localport={self.port}", f"profile={profile}",
                ]
                if remote_ips:
                    args.append(f"remoteip={remote_ips}")
                if local_ip:
                    args.append(f"localip={local_ip}")
                args.append("description=Allow the ORION phone uplink")
                add = await asyncio.to_thread(_run, args)
                if add.returncode == 0:
                    self.bus.log.emit(
                        f"REMOTE: firewall rule added for TCP {self.port} "
                        f"({profile}{', Tailscale peers only' if remote_ips else ''}).")
                else:
                    remote_arg = f" remoteip={remote_ips}" if remote_ips else ""
                    local_arg = f" localip={local_ip}" if local_ip else ""
                    self.bus.log.emit(
                        "REMOTE: could not open the firewall automatically (needs "
                        "Administrator). Run this once in an elevated PowerShell: "
                        'netsh advfirewall firewall add rule '
                        f'name="{name}" dir=in action=allow protocol=TCP '
                        f"localport={self.port} profile={profile}{remote_arg}{local_arg}")
        except Exception:
            pass

    def _maybe_ssl_context(self) -> Any | None:
        """Self-signed TLS when ORION_REMOTE_TLS=1: phones then run in a secure
        context, which unlocks the browser microphone for voice input.  The
        certificate persists under config/remote_tls/ so the phone's one-time
        'proceed anyway' acceptance survives restarts.

        Returns ``None`` ONLY when TLS was not requested. When it was, any
        failure raises :class:`RemoteTLSError` and no listener is opened — the
        certificate is never silently replaced either, because the phone app
        pins its fingerprint and a new one would look like interception."""
        if not _tls_requested():
            return None
        context, der = self._tls_material()
        self.bus.log.emit(
            "REMOTE: TLS certificate SHA-256 for first phone pairing: "
            + hashlib.sha256(der).hexdigest())
        return context

    def _tls_material(self) -> tuple[Any, bytes]:
        """ORION's own certificate as (server SSLContext, DER bytes), issued
        on first use. Shared by the direct TLS listener and the home Wi-Fi
        link, so a phone pins ONE certificate whichever way it connects.
        Raises :class:`RemoteTLSError`; never replaces an existing certificate."""
        import ssl
        tls_dir = self.auth.config_dir / "remote_tls"
        cert_path, key_path = tls_dir / "cert.pem", tls_dir / "key.pem"
        repair_files = (f"Check that {tls_dir} is readable, or delete that folder to "
                        "issue a new certificate (the phone then asks you to accept "
                        "it again). Unset ORION_REMOTE_TLS only if you mean to serve "
                        "unencrypted HTTP.")
        if not (cert_path.exists() and key_path.exists()):
            try:
                self._generate_tls_files(tls_dir, cert_path, key_path)
            except Exception as exc:
                raise RemoteTLSError("certificate generation", exc, repair_files) from exc
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(cert_path), str(key_path))
            der = ssl.PEM_cert_to_DER_cert(cert_path.read_text(encoding="ascii"))
        except Exception as exc:
            raise RemoteTLSError("certificate loading", exc, repair_files) from exc
        self._warn_if_certificate_expiring(der, tls_dir)
        return context, der

    def _warn_if_certificate_expiring(self, der: bytes, tls_dir: Path) -> None:
        """Say so when the certificate is near its end date. It is not renewed
        automatically: the phone app trusts exactly this fingerprint, so a new
        certificate needs the owner to re-accept it on each phone."""
        try:
            from datetime import datetime, timezone

            from cryptography import x509
            cert = x509.load_der_x509_certificate(der)
            ends = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(
                tzinfo=timezone.utc)
            days = (ends - datetime.now(timezone.utc)).days
        except Exception:
            return
        if days <= 30:
            when = f"expired {-days} days ago" if days < 0 else f"expires in {days} days"
            self.bus.log.emit(
                f"REMOTE: the TLS certificate {when}. Delete {tls_dir} and restart "
                "to issue a new one, then accept it on each phone.")

    def _generate_tls_files(self, tls_dir: Path, cert_path: Path, key_path: Path) -> None:
        """Issue the self-signed certificate with the configured tailnet name."""
        import ipaddress
        from datetime import datetime, timedelta, timezone

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "orion.local")])
        san_entries: list[Any] = [
            x509.DNSName("orion.local"), x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            x509.IPAddress(ipaddress.ip_address(self._lan_ip())),
        ]
        tailnet_name = os.getenv("ORION_REMOTE_DNS_NAME", "").strip().rstrip(".").lower()
        if tailnet_name:
            if not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+\.ts\.net", tailnet_name):
                raise ValueError("ORION_REMOTE_DNS_NAME must be a full ts.net MagicDNS name")
            san_entries.append(x509.DNSName(tailnet_name))
        try:
            bind_ip = ipaddress.ip_address(self.host)
            if bind_ip not in [entry.value for entry in san_entries
                               if isinstance(entry, x509.IPAddress)]:
                san_entries.append(x509.IPAddress(bind_ip))
        except ValueError:
            pass
        sans = x509.SubjectAlternativeName(san_entries)
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

    def _push_event(self, payload: dict[str, Any]) -> None:
        """Fan a state event out to every connected phone; never blocks."""
        for queue in list(self._event_subs):
            try:
                queue.put_nowait(payload)
            except Exception:
                pass   # a full queue means a stalled client — drop, don't block

    def _push_device_event(self, device_id: str, payload: dict[str, Any]) -> None:
        """Send private approval details only to the owning paired device."""
        delivered = False
        for queue, owner in list(self._event_devices.items()):
            if owner == device_id:
                try:
                    queue.put_nowait(payload)
                    delivered = True
                except asyncio.QueueFull:
                    pass
        if payload.get("type") == "action" and not delivered:
            # Quick Tunnels do not support SSE. Retain a bounded, short-lived
            # proposal for this phone to fetch; no action runs on receipt.
            queue = self._phone_actions.setdefault(device_id, deque(maxlen=20))
            queue.append((time.monotonic(), payload))

    async def _push_confirm(self, device_id: str, conf: Any) -> None:
        """Push a pending CONFIRM action to the phone for an approval tap.

        The single-use approval token rides the SSE stream, which is itself
        access-token authenticated — only a paired device receives it, and it
        must be echoed back to /api/confirm to fire the action."""
        self._push_device_event(device_id, {
            "type": "confirm", **conf.public(), "token": conf.token,
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
                self._last_emotion_name = str(name)
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
            # Hand a native action (call / text / navigate …) to a phone. Only
            # meaningful for a device running the app bridge; a plain browser
            # renders it as a tappable link fallback instead.
            #
            # It went to EVERY connected phone, so two paired phones both
            # opened the dialer. The phone whose request produced it gets it
            # (the gate's per-request scope is still set here: the signal is
            # delivered synchronously); failing that, the last one used.
            if not (isinstance(payload, dict) and payload.get("kind")):
                return
            event = {"type": "action", "action": payload}
            requester = str(payload.get("device_id") or "") or (
                self.tool_gate.request_device() or "")
            if requester:
                self._push_device_event(requester, event)
                return
            last = self._last_active_device
            if last and last in self._event_devices.values():
                self._push_device_event(last, event)
            else:
                self.bus.log.emit("REMOTE: phone action held; no target device is active")

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
        if (self._stop_task is not None and not self._stop_task.done()
                or any(not task.done() for task in self._shutdown_tasks)):
            raise RuntimeError("The previous phone uplink is still closing.")
        self._stop_task = None                 # a new lifecycle may stop once
        # Loading (or generating) the certificate and building the context is
        # blocking work; on the event loop it was part of a 582 ms boot stall.
        # It is settled BEFORE the runner exists, so a requested-TLS failure
        # leaves nothing listening and nothing half-built to tear down.
        try:
            ssl_context = await asyncio.to_thread(self._maybe_ssl_context)
        except RemoteTLSError as exc:
            self._https_enabled = False
            self.bus.log.emit(f"REMOTE: phone uplink NOT started. {exc}")
            if self.telemetry is not None:
                try:
                    self.telemetry.health.register("remote")
                    self.telemetry.health.beat("remote", "DOWN", f"TLS {exc.stage} failed")
                except Exception:
                    pass
            raise
        app = self._build_app()
        self._closing = False
        self._runner = self._web.AppRunner(
            app, access_log=None, shutdown_timeout=self._SHUTDOWN_GRACE_S)
        await self._runner.setup()
        self._https_enabled = ssl_context is not None
        self._endpoints_cache = None                 # re-detect after (re)start
        site = self._web.TCPSite(self._runner, self.host, self.port, ssl_context=ssl_context)
        try:
            await site.start()
        except BaseException:
            # A taken port (or a cancelled boot) left the runner set up; a
            # later "turn on phone access" would then build a second one.
            runner, self._runner = self._runner, None
            try:
                await runner.cleanup()
            except Exception:
                pass
            raise
        # Direct access still needs firewall rules. Serve mode listens on
        # loopback only, so no inbound application port must be opened.
        await self._ensure_firewall_access()
        # The home Wi-Fi link: the same app on the PC's LAN address, TLS with
        # ORION's pinned certificate, private peers only. Never fatal.
        await self.lan.start(self._runner, self._web, self._tls_material)
        self._wire_bus_events()
        if self._amp_task is None:
            self._amp_task = asyncio.create_task(
                self._amplitude_ticker(), name="orion-remote-amplitude")
        scheme = "https" if ssl_context is not None else "http"
        self.bus.log.emit(
            f"REMOTE: uplink active on {self.host}:{self.port} (installable PWA); "
            "pairing-based auth."
        )
        if self._serve_mode:
            self.bus.log.emit(
                f"REMOTE: loopback uplink at http://127.0.0.1:{self.port}; "
                f"run tailscale serve --bg {self.port} and use the HTTPS "
                "MagicDNS address shown by tailscale serve status on the phone.")
        elif self._platform_proxy_mode:
            self.bus.log.emit(
                f"REMOTE: managed container upstream on {self.host}:{self.port}; "
                f"paired phones use {self._public_endpoint['url']}")
        elif self._reverse_proxy_mode:
            self.bus.log.emit(
                f"REMOTE: reverse proxy upstream at http://127.0.0.1:{self.port}; "
                f"paired phones use {self._public_endpoint['url']}")
        elif os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() == "tailnet-direct":
            name = os.getenv("ORION_REMOTE_DNS_NAME", "").strip()
            self.bus.log.emit(
                f"REMOTE: direct tailnet HTTPS at https://{name}/; "
                "keep Tailscale connected on the phone and verify the certificate fingerprint.")
        elif os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() == "quick-tunnel":
            self.bus.log.emit(
                f"REMOTE: quick tunnel upstream ready at http://127.0.0.1:{self.port}; "
                "run deploy/start_quick_https_tunnel.ps1 for a temporary public HTTPS URL.")
        else:
            # Off the loop: the address lookup can reach the DNS resolver.
            lan_ip = await asyncio.to_thread(self._lan_ip)
            self.bus.log.emit(
                f"REMOTE: on your phone (same Wi-Fi) open "
                f"{scheme}://{lan_ip}:{self.port}.")
        if (_on_windows() and not self._public_https and
                os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() not in
                {"tailnet-direct", "quick-tunnel"}):
            self.bus.log.emit(
                "REMOTE: if the phone still shows 'refused to connect', make sure this "
                "Wi-Fi is set to a Private network in Windows (Settings ▸ Network) and "
                "that the phone is on the same network — not a guest/5GHz-isolated SSID."
            )
        if (ssl_context is None and not self._public_https and
                os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() != "quick-tunnel"):
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
        pair_on_start = os.getenv("ORION_REMOTE_PAIR_ON_START", "0").strip().lower() in {
            "1", "true", "yes", "on"}
        if not self.auth.has_devices() or pair_on_start:
            code = self.auth.begin_pairing()
            self.bus.log.emit(
                f"REMOTE: enter pairing code {code} "
                "in the PWA within 10 minutes to link this device.")
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
            self._signal_end(queue)

    @staticmethod
    def _signal_end(queue: Any) -> None:
        """Put the end-of-stream marker on *queue*, making room if it is full."""
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()          # a stalled client: make room
                queue.put_nowait(None)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def _retire_event_stream(self, queue: Any) -> None:
        """Stop publishing to *queue* now and tell its handler to return."""
        self._event_subs.discard(queue)
        self._event_devices.pop(queue, None)
        self._signal_end(queue)

    @staticmethod
    def _peer_gone(request: Any) -> bool:
        """Whether the client's socket has closed. aiohttp (3.9+) no longer
        cancels a handler when that happens, so a long-lived response has to
        look for itself."""
        try:
            transport = request.transport
        except AttributeError:
            return False                    # not a live request (a test double)
        return transport is None or transport.is_closing()

    async def stop(self) -> None:
        """One owned teardown, even if callers overlap or a caller is cancelled.

        Independent closes run together. An uncooperative close is cancelled
        at the deadline, remains strongly owned until it finishes, and is
        named in the log rather than silently counted as clean."""
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(
                self._finish_stop(), name="orion-remote-stop")
            self._stop_task.add_done_callback(self._observe_shutdown_task)
        # A voice toggle can be cancelled independently of the gateway's
        # shutdown. The single bounded teardown still has an owner.
        await asyncio.shield(self._stop_task)

    def _observe_shutdown_task(self, task: asyncio.Task) -> None:
        self._shutdown_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self.bus.log.emit(
                    f"REMOTE: {task.get_name()} shutdown reported - "
                    f"{type(error).__name__}: {error}")

    async def _finish_stop(self) -> None:
        self._closing = True
        self._close_event_streams()
        for stream in list(self._chat_streams.values()):
            stream.cancel("shutdown")

        tasks = set()
        def own(task):
            tasks.add(task)
            self._shutdown_tasks.add(task)
            task.add_done_callback(self._observe_shutdown_task)

        async def close_resource(factory):
            await factory()

        def start(label, factory):
            own(asyncio.create_task(close_resource(factory),
                                    name=f"orion-remote-stop-{label}"))

        power = getattr(self, "power_controller", None)
        if power is not None:
            start("power", power.close)
        for index, socket in enumerate(list(self._audio_sessions.values())):
            start(f"audio-{index}", lambda socket=socket: self._close_socket(socket))
        # The companion app's /v1/device/uplink socket is open whenever the app
        # runs. Nothing closed it here, so aiohttp's grace ran out on it twice
        # over (4.0 s at a 2.0 s grace, MEASURED) before cancelling its handler.
        viewport = getattr(self, "mobile_viewport", None)
        uplinks = list(viewport.open_sockets()) if hasattr(viewport, "open_sockets") else []
        for index, socket in enumerate(uplinks):
            start(f"uplink-{index}", lambda socket=socket: self._close_socket(socket))
        for stream in list(self._chat_streams.values()):
            if stream.task is not None:
                own(stream.task)
        amp_task, self._amp_task = self._amp_task, None
        if amp_task is not None:
            amp_task.cancel()
            own(amp_task)
        # The runner's cleanup stops the LAN site itself; only its address
        # watcher is ours to end. A gateway that never finished starting has
        # no LAN link, and a shutdown must never raise over that.
        lan = getattr(self, "lan", None)
        lan_watch = lan.detach() if lan is not None else None
        if lan_watch is not None:
            own(lan_watch)
        runner, self._runner = self._runner, None
        # Taken before the runner forgets its connections: a phone that stopped
        # reading keeps its socket after aiohttp closes it (a close waits to
        # flush what is queued), and its handler parked in drain() with it.
        transports = self._connection_transports(runner)
        if transports:
            start("stalled", lambda: self._release_stalled_after(transports))
        if runner is not None:
            start("runner", runner.cleanup)
        if not tasks:
            return
        try:
            _done, pending = await asyncio.wait(
                tasks, timeout=max(0.0, self._STOP_TIMEOUT_S))
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        for task in pending:
            task.cancel()
        # Past the deadline every connection goes; otherwise only the ones
        # still holding bytes their phone will not take.
        self._release_stalled(transports, everything=bool(pending))
        if pending:
            names = ", ".join(sorted(task.get_name() for task in pending))
            self.bus.log.emit(f"REMOTE: shutdown deadline reached; cancelling: {names}.")
            _done, pending = await asyncio.wait(
                pending, timeout=max(0.0, self._STOP_CANCEL_GRACE_S))
            if pending:
                names = ", ".join(sorted(task.get_name() for task in pending))
                self.bus.log.emit(
                    f"REMOTE: shutdown deadline reached; still running: {names}.")

    async def _close_socket(self, socket: Any) -> None:
        """Send one phone websocket its close frame, waiting at most
        ``_SOCKET_CLOSE_S`` for the reply. Giving up cancels aiohttp's close,
        which drops the connection itself."""
        try:
            await asyncio.wait_for(
                socket.close(code=1001, message=b"Server shutting down"),
                timeout=self._SOCKET_CLOSE_S)
        except asyncio.TimeoutError:
            pass

    @staticmethod
    def _connection_transports(runner: Any) -> list[Any]:
        """The transports of every connection the runner is serving now."""
        server = getattr(runner, "server", None)
        try:
            handlers = list(getattr(server, "connections", None) or ())
        except Exception:
            return []
        return [t for t in (getattr(h, "transport", None) for h in handlers) if t is not None]

    async def _release_stalled_after(self, transports: list[Any]) -> None:
        """A healthy phone takes what it was sent in milliseconds; one still
        holding unsent bytes after ``_STALL_SETTLE_S`` has stopped reading.
        Without this its handler waited out both of aiohttp's graces in
        drain() (1.0 s at a 0.5 s grace, 4.0 s at the old 2.0 s, MEASURED).
        Returns as soon as every connection has closed with nothing unsent."""
        deadline = time.monotonic() + self._STALL_SETTLE_S
        while time.monotonic() < deadline:
            try:
                if all(t.is_closing() and not t.get_write_buffer_size() for t in transports):
                    return
            except Exception:
                pass
            await asyncio.sleep(0.02)
        self._release_stalled(transports)

    @staticmethod
    def _release_stalled(transports: list[Any], *, everything: bool = False) -> int:
        """Abort connections that would otherwise outlive stop(): any still
        holding unsent bytes (or all of them, past the deadline). Aborting
        releases the socket at once and wakes a handler parked in drain() with
        ConnectionResetError. Returns how many were aborted."""
        aborted = 0
        for transport in transports:
            try:
                if everything or transport.get_write_buffer_size() > 0:
                    transport.abort()
                    aborted += 1
            except Exception:
                pass
        return aborted

    # ── rate limiting ──────────────────────────────────────────────────────────

    def _client_key(self, request: Any) -> str:
        # Quick Tunnel forwards every phone through the same loopback peer.
        # Give a *verified* paired device its own bucket so one phone cannot
        # exhaust another's allowance. Never key on X-Forwarded-For, or on an
        # unverified bearer string: either can be invented by a caller.
        header = str(request.headers.get("Authorization") or "")
        if header.lower().startswith("bearer "):
            device_id = self.auth.verify_access(header[7:].strip())
            if device_id:
                return f"device:{device_id}"
        peer = request.remote or "unknown"
        return f"peer:{peer}"

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

    async def _handle_companion_apk(self, request: Any) -> Any:
        """Publish only the deliberately staged companion build, never host files."""
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "releases" / "android" / "orion-companion.apk"
        if not path.is_file():
            raise self._web.HTTPNotFound(text="The companion update has not been staged.")
        return self._web.FileResponse(path, headers={
            "Content-Type": "application/vnd.android.package-archive",
            "Content-Disposition": 'attachment; filename="orion-companion.apk"',
            "Cache-Control": "no-store",
        })

    async def _handle_page(self, request: Any) -> Any:
        # The page's stage is a 2-D canvas signal with no WebGL and nothing
        # fetched from a CDN, so a metered connection gets the same page.
        page = REMOTE_PAGE_HTML
        if os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() == "quick-tunnel":
            page = page.replace("const QUICK_TUNNEL=false;", "const QUICK_TUNNEL=true;", 1)
        return self._web.Response(text=page, content_type="text/html")

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
        files, version = await asyncio.to_thread(self._three_files)
        path = files.get(str(request.match_info.get("path") or ""))
        if path is None or str(request.match_info.get("version") or "") != version:
            return self._web.Response(status=404, text="not found")
        # The version is in the URL, so a cached copy can never be stale.
        return self._web.FileResponse(path, headers={
            "Content-Type": "text/javascript; charset=utf-8",
            "Cache-Control": "public, max-age=604800, immutable",
        })

    async def _handle_face(self, request: Any) -> Any:
        response = self._web.Response(
            text=await asyncio.to_thread(self._face_page), content_type="text/html")
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

    @staticmethod
    def _face_style() -> dict[str, Any]:
        """The desktop Screen Face's design and palette, for the phone's copy
        of it (the app caches this, so the face matches with the PC off)."""
        design = "sentinel"
        try:
            from . import appearance
            design = str(appearance.face_design() or design)
        except Exception:
            pass
        from .constants import C
        colours = {}
        for key, name in (("primary", "PRI"), ("accent", "ACCENT"), ("background", "BG")):
            value = str(getattr(C, name, "") or "")
            if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                colours[key] = value.lower()
        return {"type": "face", "design": design, **colours}

    async def _handle_events(self, request: Any) -> Any:
        """Server-sent events mirroring the desktop orb's state to the phone.
        EventSource cannot set headers, so the browser's access token rides
        the query string; the native app sends it as a Bearer header instead,
        which keeps it out of every URL. Both are verified the same way."""
        headers = getattr(request, "headers", None) or {}
        header = str(headers.get("Authorization") or "")
        token = (header[7:].strip() if header.lower().startswith("bearer ")
                 else str(request.query.get("token") or ""))
        device_id = self.auth.verify_access(token) if token else None
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        # A reconnecting phone opens its new stream before the old socket is
        # known to be dead: retire this device's oldest beyond the allowance
        # rather than let dead ones pile up. Other devices are never evicted.
        mine = [q for q, owner in self._event_devices.items() if owner == device_id]
        for stale in mine[:max(0, len(mine) - self._EVENT_STREAMS_PER_DEVICE + 1)]:
            self._retire_event_stream(stale)
        if len(self._event_subs) >= self._EVENT_STREAMS_MAX:
            return self._web.json_response(
                {"ok": False, "error": "too many live streams"}, status=503)
        response = self._web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._event_subs.add(queue)
        self._event_devices[queue] = device_id
        self._last_active_device = device_id

        async def _send(payload: dict[str, Any]) -> None:
            await response.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

        # The stream lives no longer than its access token. Recheck before
        # every queued event as well: a revoked phone must not receive a
        # private confirmation token through a stream it opened earlier.
        deadline = time.time() + RemoteAuthManager.ACCESS_TTL_S
        last_write = time.monotonic()
        try:
            await _send({"type": "state", "value": self._last_state})
            if self._last_emotion:
                await _send({"type": "emotion", "name": self._last_emotion_name,
                             "params": self._last_emotion})
            while time.time() < deadline and not self._closing:
                if self.auth.verify_access(token) != device_id:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._EVENTS_TICK_S)
                except asyncio.TimeoutError:
                    if self._peer_gone(request) or queue not in self._event_subs:
                        break
                    if time.monotonic() - last_write < self._EVENTS_KEEPALIVE_S:
                        continue
                    if self.auth.verify_access(token) != device_id:
                        break
                    await response.write(b": keepalive\n\n")
                    last_write = time.monotonic()
                    continue
                if item is None:                  # stop() or retired: end now
                    break
                if self.auth.verify_access(token) != device_id:
                    break
                await _send(item)
                last_write = time.monotonic()
        except asyncio.CancelledError:
            # Not swallowed: aiohttp cancels this handler at shutdown, and a
            # stream that carried on from here went straight back into a
            # write its stalled phone would never take.
            raise
        except Exception:                 # includes ConnectionResetError
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
        """Reachable endpoints for the phone: the home Wi-Fi link first (with
        its certificate pin), then HTTPS MagicDNS for tailnet modes."""
        link = getattr(self, "lan", None)        # absent on a bare test double
        lan = link.endpoint() if link is not None else None
        others = await self._remote_endpoints()
        if lan is None:
            return others
        return [lan] + [e for e in others if e.get("host") != lan["host"]]

    def lan_pairing_hint(self) -> str:
        """Address and certificate code for pairing on the home Wi-Fi, or ''."""
        lan = getattr(self, "lan", None)
        return lan.pairing_hint() if lan is not None else ""

    async def _remote_endpoints(self) -> list[dict[str, Any]]:
        """Endpoints that work away from home. Cached for about 60 seconds to
        avoid repeated Tailscale subprocesses."""
        if os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower() == "quick-tunnel":
            # The random public URL is known only to cloudflared after launch.
            # Never teach the phone an internal tailnet address in its place.
            return []
        if self._reverse_proxy_mode:
            return [dict(self._public_endpoint)]
        async with self._endpoints_lock:
            now = time.time()
            if self._endpoints_cache and now - self._endpoints_cache[0] < 60.0:
                return self._endpoints_cache[1]
            try:
                eps = await asyncio.to_thread(
                    detect_endpoints, self.port, https=self._https_enabled,
                    serve=self._serve_mode,
                    tailnet_only=(os.getenv("ORION_REMOTE_EXPOSURE", "").strip().lower()
                                  == "tailnet-direct"))
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
                "ok": True, "port": 443 if self._public_https else self.port,
                "https": self._https_enabled or self._public_https,
                "endpoints": await self._endpoints(),
            })
        return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)

    async def _handle_hardware(self, request: Any) -> Any:
        """Read-only host sensors for a paired device; sample off the loop."""
        if self._authenticated_device(request, {}) is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised"}, status=401)
        try:
            from .hardware_telemetry import sample
            reading = await asyncio.wait_for(asyncio.to_thread(sample), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            return self._web.json_response(
                {"ok": False, "error": "hardware sensors unavailable"}, status=503)
        return self._web.json_response({"ok": True, **reading})

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

    def _audio_stats(self, device_id: str) -> dict[str, Any]:
        """Per-device counters only: never retain audio, words or credentials."""
        if device_id not in self._audio_diagnostics:
            if len(self._audio_diagnostics) >= 64:
                idle = next((key for key in self._audio_diagnostics if key not in self._audio_sessions), None)
                if idle is not None:
                    self._audio_diagnostics.pop(idle, None)
            self._audio_diagnostics[device_id] = {
                "pcm_bytes": 0, "utterances": 0, "transcriptions": 0,
                "requests": 0, "errors": 0, "state": "not started",
            }
        return self._audio_diagnostics[device_id]

    async def _handle_audio_status(self, request: Any) -> Any:
        device_id = self._authenticated_device(request, {})
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        return self._web.json_response({"ok": True, **self._audio_stats(device_id),
            "connected": device_id in self._audio_sessions}, headers={"Cache-Control": "no-store"})

    # ── authentication endpoints ───────────────────────────────────────────────

    @classmethod
    async def _json_object(cls, request: Any, schema: str = "") -> dict[str, Any]:
        """Parse a JSON object body no larger than ``_MAX_JSON_BYTES``.

        Raises ``_BodyTooLarge`` for an oversized body (declared or streamed),
        ``_BadField`` for a field that breaks the endpoint's *schema*, and
        ``ValueError`` for anything that is not a JSON object."""
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
        if schema:
            _check_fields(payload, _REQUEST_SCHEMAS[schema])
        return payload

    def _bad_body(self, exc: Exception) -> Any:
        """The error response for a body ``_json_object`` refused."""
        if isinstance(exc, _BodyTooLarge):
            return self._web.json_response(
                {"ok": False, "error": "request body too large"}, status=413)
        if isinstance(exc, _BadField):
            return self._web.json_response({"ok": False, "error": str(exc)}, status=400)
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
            payload = await self._json_object(request, "pair")
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
             "endpoints": await self._endpoints(),
             "port": 443 if self._public_https else self.port,
             "https": self._https_enabled or self._public_https})

    async def _handle_token(self, request: Any) -> Any:
        if not self._auth_rate_ok(self._client_key(request)):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request, "token")
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
        # "face": the desktop Screen Face's design and colours, so the app's
        # copy of the face matches (and keeps matching with the PC off).
        return self._web.json_response(
            {"ok": True, "access_token": access_token, "expires_in": expires_in,
             "endpoints": await self._endpoints(), "face": self._face_style()})

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
            payload = await self._json_object(request, "task")
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

    async def _chat_admission(self, request: Any) -> tuple[str, str] | Any:
        """Rate limit, body limit and schema, authentication and the text
        guard, shared by the plain and streamed chat routes so a check added
        to one cannot be missed by the other. ``(device_id, message)``, or the
        refusal to return."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request, "chat")
        except Exception as exc:
            return self._bad_body(exc)
        device_id = self._authenticated_device(request, payload)
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised — pair this device first"}, status=401)
        message = str(payload.get("message") or "").strip()[:4000]
        if not message:
            return self._web.json_response({"ok": False, "error": "empty message"}, status=400)
        from . import phone_control
        if not phone_control.enabled():
            # Mark XXXX.4: refused HERE, before the message is logged, added to
            # the history or written to memory.
            phone_control.record("chat", "-", "off", device=device_id)
            return self._web.json_response(
                {"ok": True, "reply": phone_control.OFF_REPLY, "provider": "phone-control-off"})
        try:
            SecuritySanitiser.guard_text(message, "remote.message")
        except SecurityViolation as exc:
            return self._web.json_response({"ok": False, "error": str(exc)}, status=403)
        return device_id, message

    @staticmethod
    def _begin_trace(device_id: str) -> request_trace.RequestTrace:
        # The detail is a device prefix, never the message: stage lines go to
        # the ORION log, and the message is already logged once, truncated.
        trace = TRACES.begin(origin="remote")
        trace.stage(Stage.COMMAND_RECEIVED, f"phone {device_id[:8]}")
        return trace

    def _note_user_turn(self, device_id: str, message: str, request_id: str) -> int:
        """Log the phone's message and add it to the page history; its id."""
        # PHONE: is the words; REMOTE: lines are what the gateway did, so the
        # private log can drop one and keep the other (Mark XXXX.4).
        self.bus.log.emit(f"PHONE: {message[:120]}")
        return self._history.add(device_id, "user", message, request_id)

    async def _remember_turn(self, role: str, text: str) -> None:
        try:
            await asyncio.to_thread(self.memory.log_episode, role, text)
        except Exception:
            pass

    async def _persist_reply(self, reply: str) -> None:
        """The final answer goes into episodic memory exactly as before
        streaming existed: whole, once, and only when it is final."""
        await self._remember_turn("orion (remote)", reply)
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("remote.turns")
            except Exception:
                pass

    @staticmethod
    def _finish_trace(trace: request_trace.RequestTrace, detail: str) -> None:
        # A phone turn has no microphone to restore: its end is the reply
        # reaching the phone (or the phone giving up on it). Anything short of
        # a completing stage counts as stalled after 45 s.
        trace.stage(Stage.REPLY_DELIVERED, detail)

    async def answer_text(self, message: str, device_id: str) -> str:
        """One remote turn from a channel other than the phone app (the
        Telegram remote, telegram_remote.py): the same capability gate,
        history and memory as /api/chat. The phone-only branches of _answer
        (the phone's screen, phone actions) are not offered: *device_id*
        scopes tool ownership but is not a paired phone."""
        message = str(message or "").strip()[:4000]
        if not message:
            return ""
        from . import phone_control
        if not phone_control.enabled():
            phone_control.record("chat", "-", "off", device=device_id)
            return phone_control.OFF_REPLY
        trace = self._begin_trace(device_id)
        rid = trace.request_id
        self._note_user_turn(device_id, message, rid)
        await self._remember_turn("user (remote)", message)
        try:
            with request_trace.bound(trace), self.tool_gate.for_device(device_id):
                trace.stage(Stage.API_REQUEST_START, "remote reply")
                reply, provider = await self._answer(message, device_id="")
        except Exception as exc:
            error = str(exc).splitlines()[0][:200]
            trace.failed(Stage.API_REQUEST_START, error, recovery="reported to the sender")
            raise
        trace.stage(Stage.API_RESPONSE_RECEIVED, provider)
        self._history.add(device_id, "orion", reply, rid)
        await self._persist_reply(reply)
        self._finish_trace(trace, "remote reply delivered")
        return reply

    def pending_for_channel(self, device_id: str) -> list[dict[str, Any]]:
        """Confirmations a non-app channel (Telegram) is waiting on, with the
        single-use token its buttons carry. Never all devices' at once."""
        device_id = str(device_id or "")
        if not device_id:
            return []
        return [{**c.public(), "token": c.token}
                for c in self.confirmations.pending_for(device_id)]

    async def decide_for_channel(self, device_id: str, conf_id: str, token: str,
                                 approve: bool) -> str:
        """Approve or decline one of that channel's confirmations (Mark XXXX.3)."""
        device_id = str(device_id or "")
        conf = self.confirmations.resolve(conf_id, token, approve, device_id=device_id)
        if conf is None:
            return "That request has expired or was already answered - nothing ran."
        channel = device_id.split(":")[0]
        if not approve:
            self.bus.log.emit(f"REMOTE: declined '{conf.tool}' from {channel}")
            return "Declined - nothing ran."
        self.bus.log.emit(f"REMOTE: approved '{conf.tool}' from {channel}")
        result = await self.tool_gate.run_approved(conf)
        text = str(getattr(result, "text", "") or "Done.")
        return text if getattr(result, "ok", True) else f"That didn't work: {text}"

    async def _handle_chat(self, request: Any) -> Any:
        admitted = await self._chat_admission(request)
        if not isinstance(admitted, tuple):
            return admitted
        device_id, message = admitted
        trace = self._begin_trace(device_id)
        rid = trace.request_id
        headers = {"X-Request-Id": rid}
        self._note_user_turn(device_id, message, rid)
        await self._remember_turn("user (remote)", message)
        widgets = []
        try:
            self._last_active_device = device_id
            with request_trace.bound(trace), self.tool_gate.for_device(device_id):
                trace.stage(Stage.API_REQUEST_START, "remote reply")
                reply, provider = await self._answer(message, device_id=device_id, on_widget=widgets.append)
        except Exception as exc:
            error = str(exc).splitlines()[0][:200]
            trace.failed(Stage.API_REQUEST_START, error, recovery="reported to the phone")
            return self._web.json_response(
                {"ok": False, "error": error, "request_id": rid}, status=502, headers=headers)
        trace.stage(Stage.API_RESPONSE_RECEIVED, provider)
        self._history.add(device_id, "orion", reply, rid)
        await self._persist_reply(reply)
        self._finish_trace(trace, "remote reply delivered")
        return self._web.json_response(
            {"ok": True, "reply": reply, "provider": provider, "request_id": rid, "widgets": widgets},
            headers=headers)

    async def _handle_chat_stream(self, request: Any) -> Any:
        """The same turn as /api/chat, answered as server-sent events on this
        one response: ``start {request_id, user_hid}``, ``delta {seq, text}``,
        ``restart {reason}``, then ``done {reply, provider, hid}`` or
        ``error {message}``. Every event carries ``request_id`` and a rising
        ``seq`` so the phone can drop duplicates and strays.

        A separate route rather than a flag on /api/chat: that route's JSON
        contract stays byte-for-byte what existing clients parse, and a new
        page talking to an older desktop gets a clean 404 it can fall back
        from instead of a JSON body where it expected a stream. Refusals
        before the stream opens (rate, size, auth, guard) are the same JSON
        responses /api/chat gives."""
        admitted = await self._chat_admission(request)
        if not isinstance(admitted, tuple):
            return admitted
        device_id, message = admitted
        if sum(1 for s in self._chat_streams.values()
               if s.device_id == device_id) >= self._MAX_STREAMS_PER_DEVICE:
            return self._web.json_response(
                {"ok": False, "error": "too many replies in progress"}, status=429)
        trace = self._begin_trace(device_id)
        rid = trace.request_id
        ndjson = request.path.endswith("/frames")
        response = self._web.StreamResponse(headers={
            "Content-Type": ("application/x-ndjson" if ndjson else "text/event-stream") + "; charset=utf-8",
            "Cache-Control": "no-cache, no-store, no-transform",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "X-Request-Id": rid,
        })
        await response.prepare(request)
        stream = _ChatStream(rid, device_id, trace, ndjson=ndjson)
        self._chat_streams[rid] = stream
        try:
            # `start` goes out before the memory write so the phone holds the
            # request id (and can cancel) at once; the message is still in
            # memory before generation begins, as on /api/chat.
            user_hid = self._note_user_turn(device_id, message, rid)
            await stream.send(response, "start", user_hid=user_hid)
            await self._remember_turn("user (remote)", message)
            self._last_active_device = device_id
            # create_task copies the context: the request trace and the tool
            # gate's device scope follow the generation into its own task.
            with request_trace.bound(trace), self.tool_gate.for_device(device_id):
                trace.stage(Stage.API_REQUEST_START, "remote reply (streamed)")
                stream.task = asyncio.create_task(
                    self._answer(message, device_id=device_id,
                                 on_delta=stream.delta, on_restart=stream.restart, on_widget=stream.widgets.append),
                    name=f"orion-remote-{rid}")
            stream.task.add_done_callback(lambda _task: stream.wake())
            await self._pump_chat_stream(request, response, stream)
            await self._end_chat_stream(response, stream)
        except (Exception, asyncio.CancelledError) as exc:
            # The phone went away: a write met a reset socket, or aiohttp
            # cancelled this handler (it does when built with
            # handler_cancellation, as its test server is; the production
            # runner is not, and the pump notices the closed socket instead).
            # Stop the work if it is still running; a finished answer is kept.
            if stream.task is None or not stream.task.done():
                stream.cancel("disconnect")
                await self._settle_cancelled(stream)
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._chat_streams.pop(rid, None)
            if stream.task is not None and not stream.task.done():
                stream.task.cancel()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _pump_chat_stream(self, request: Any, response: Any,
                                stream: _ChatStream) -> None:
        """Write the answer's events until it finishes or is cancelled; cancel
        it when the phone's socket closes."""
        last_write = time.monotonic()
        while not stream.task.done() and not stream.cancel_reason:
            if not stream.outbox:
                await stream.wait(self._STREAM_TICK_S)
                if stream.outbox or stream.task.done() or stream.cancel_reason:
                    continue
                if self._peer_gone(request):
                    stream.cancel("disconnect")
                    break
                if time.monotonic() - last_write >= self._STREAM_KEEPALIVE_S:
                    await response.write(b"\n" if stream.ndjson else b": keepalive\n\n")
                    last_write = time.monotonic()
                continue
            kind, data = stream.outbox.popleft()
            await stream.send(response, kind, **data)
            last_write = time.monotonic()
            if kind == "delta":
                # The first token goes out at once; the next few are merged
                # while this pause lasts, so the phone repaints per flush.
                await asyncio.sleep(self._STREAM_FLUSH_S)

    async def _settle_cancelled(self, stream: _ChatStream) -> None:
        """Wait briefly for a cancelled generation to unwind, and record how
        long cancelling took and why. Cancelling a task does not stop a native
        call already running in a thread, so this never waits for long."""
        task = stream.task
        settled = True
        if task is not None and not task.done():
            done, _pending = await asyncio.wait({task}, timeout=self._CANCEL_GRACE_S)
            settled = bool(done)
        if stream.trace.marks_named("cancelled"):
            return
        stream.trace.mark("cancelled", reason=stream.cancel_reason or "error",
                          settled=settled,
                          ms=(time.monotonic() - stream.cancel_at) * 1000.0
                          if stream.cancel_at else None)
        detail = {"phone": "remote reply cancelled by the phone",
                  "disconnect": "phone disconnected before the reply finished",
                  "shutdown": "remote reply abandoned for shutdown"}.get(
                      stream.cancel_reason, "remote reply abandoned")
        self._finish_trace(stream.trace, detail)
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("remote.stream_cancelled")
            except Exception:
                pass

    @staticmethod
    async def _send_restarts(response: Any, stream: _ChatStream) -> None:
        """Send any restart still queued when the answer ended, and drop the
        queued deltas (the final reply supersedes them). A provider that fails
        part-way and is followed at once by the offline brain's answer would
        otherwise finish before its restart was written: text from the failed
        attempt has to be withdrawn explicitly, not quietly overwritten."""
        pending = [data for kind, data in stream.outbox if kind == "restart"]
        stream.outbox.clear()
        for data in pending:
            await stream.send(response, "restart", **data)

    async def _end_chat_stream(self, response: Any, stream: _ChatStream) -> None:
        """The last event: ``done`` with the final reply, or ``error``."""
        trace = stream.trace
        if stream.cancel_reason:
            await self._settle_cancelled(stream)
            if stream.cancel_reason != "disconnect":
                await stream.send(response, "error", cancelled=True,
                                  message=_ChatStream.ENDINGS.get(stream.cancel_reason, "Stopped."))
            return
        task = stream.task
        try:
            reply, provider = task.result()
        except asyncio.CancelledError:
            # Cancelled by something other than this gateway (the loop
            # tearing down): nothing was delivered.
            stream.cancel_reason = stream.cancel_reason or "shutdown"
            await self._settle_cancelled(stream)
            return
        except Exception as exc:
            error = str(exc).splitlines()[0][:200] or type(exc).__name__
            trace.failed(Stage.API_REQUEST_START, error, recovery="reported to the phone")
            await self._send_restarts(response, stream)
            await stream.send(response, "error", message=error)
            return
        trace.stage(Stage.API_RESPONSE_RECEIVED, provider)
        hid = self._history.add(stream.device_id, "orion", reply, stream.request_id)
        delivered = True
        try:
            await self._send_restarts(response, stream)
            await stream.send(response, "done", reply=reply, provider=provider, hid=hid, widgets=stream.widgets)
        except Exception:
            delivered = False
        trace.mark("reply", streamed=stream.deltas > 0, deltas=stream.deltas,
                   delivered=delivered)
        # Persisted whether or not the phone was still there to read it: the
        # answer was produced and is part of the conversation.
        await self._persist_reply(reply)
        self._finish_trace(trace, "remote reply delivered" if delivered
                           else "remote reply ready; the phone had gone")

    async def _handle_chat_cancel(self, request: Any) -> Any:
        """Stop a streamed answer in progress. Owner-only: another paired
        device, or an unknown id, gets the same 404."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request, "chat_cancel")
        except Exception as exc:
            return self._bad_body(exc)
        device_id = self._authenticated_device(request, payload)
        if device_id is None:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised — pair this device first"}, status=401)
        rid = str(payload.get("request_id") or "")
        stream = self._chat_streams.get(rid)
        if stream is None or stream.device_id != device_id:
            return self._web.json_response(
                {"ok": False, "error": "no such reply in progress"}, status=404)
        stream.cancel("phone")
        return self._web.json_response({"ok": True, "status": "cancelled", "request_id": rid})

    async def _handle_history(self, request: Any) -> Any:
        """This device's phone conversation from this session, a page at a
        time, oldest first: ``?before=<id>&limit=<0..100>``. ``limit=0`` only
        reports whether there is any and the id the next message will get."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        device_id = self._authenticated_device(request, {})
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        try:
            raw_before = request.query.get("before")
            before = int(raw_before) if raw_before not in (None, "") else None
            limit = int(request.query.get("limit") or 50)
        except (TypeError, ValueError):
            return self._web.json_response(
                {"ok": False, "error": "'before' and 'limit' must be whole numbers"}, status=400)
        if limit < 0 or limit > _PhoneHistory.PAGE_MAX or (before is not None and before < 1):
            return self._web.json_response(
                {"ok": False, "error": "'before' or 'limit' is out of range"}, status=400)
        items, more, next_id = self._history.page(device_id, before, limit)
        return self._web.json_response(
            {"ok": True, "items": items, "more": more, "next": next_id})

    # ── tasks: what ORION is doing, from the durable job store ─────────────────

    #: What the phone is shown of a job. Arguments and checkpoints stay on the
    #: desktop: a task list is for status, and a tool's arguments can carry
    #: anything (paths, message text) the phone has no need to hold.
    _TASK_FIELDS = ("id", "kind", "name", "label", "state", "cls", "fraction", "step",
                    "total", "position", "created_at", "updated_at", "finished_at", "ok")
    _TASK_TERMINAL = frozenset({"completed", "failed", "cancelled"})

    async def _handle_tasks(self, request: Any) -> Any:
        """Plans, workflows and background jobs, newest change first:
        ``?limit=<1..60>``. Read-only. Steps waiting for approval are listed
        with their reason; they are approved on the desktop's TASKS page (tool
        actions raised from the phone keep their on-phone approval cards)."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        if self._authenticated_device(request, {}) is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        try:
            limit = int(request.query.get("limit") or 30)
        except (TypeError, ValueError):
            return self._web.json_response(
                {"ok": False, "error": "'limit' must be a whole number"}, status=400)
        if not 1 <= limit <= 60:
            return self._web.json_response(
                {"ok": False, "error": "'limit' must be between 1 and 60"}, status=400)
        manager = getattr(self.dispatcher, "jobs", None)
        try:
            if manager is not None and hasattr(manager, "snapshot"):
                snap = await asyncio.to_thread(manager.snapshot, limit=limit)
            else:
                from . import job_store
                snap = await asyncio.to_thread(
                    job_store.default_store().snapshot, limit=limit)
        except Exception as exc:
            return self._web.json_response(
                {"ok": False, "error": f"tasks are unavailable: {first_line(exc)[:120]}"},
                status=503)
        jobs = []
        for job in snap.get("jobs") or []:
            item = {field: job.get(field) for field in self._TASK_FIELDS}
            item["note"] = str(job.get("note") or "")[:200]
            if job.get("state") in self._TASK_TERMINAL:
                item["result"] = str(job.get("result") or "")[:400]
            jobs.append(item)
        approvals = [{"tool": str(a.get("tool") or ""), "reason": str(a.get("reason") or "")[:200],
                      "expires_at": a.get("expires_at")}
                     for a in (snap.get("approvals") or [])]
        return self._web.json_response({
            "ok": True, "jobs": jobs, "counts": dict(snap.get("counts") or {}),
            "approvals": approvals, "approve_on": "desktop"})

    # ── voice input: speech-to-text for the phone ──────────────────────────────

    async def _handle_mobile_ai_profile(self, request: Any) -> Any:
        """Provision only the owner's explicitly approved paired phone."""
        from .mobile_ai import approved, profile
        headers = {"Cache-Control": "no-store, private", "Pragma": "no-cache"}
        # The response contains a reusable provider credential. TLS may end
        # here or at a local Serve/reverse-proxy/tunnel process, but never send
        # the key over a cleartext connection from another LAN host.
        if not getattr(request, "secure", False) and str(getattr(request, "remote", "")) not in {
                "127.0.0.1", "::1"}:
            return self._web.json_response(
                {"ok": False, "error": "AI provisioning requires HTTPS."},
                status=403, headers=headers)
        header = str(request.headers.get("Authorization") or "")
        device_id = self.auth.verify_access(header[7:].strip()) if header.lower().startswith("bearer ") else None
        if not device_id:
            return self._web.json_response({"ok": False, "error": "Pair this phone first."}, status=401, headers=headers)
        if not self._auth_rate_ok("mobile-ai:" + device_id):
            return self._web.json_response({"ok": False, "error": "Try again shortly."}, status=429, headers=headers)
        permission = await asyncio.to_thread(approved, self.auth.config_dir, device_id)
        if not permission:
            return self._web.json_response({"ok": False, "error": "The PC owner has not authorised AI provisioning for this pairing."}, status=403, headers=headers)
        if self.auth.verify_access(header[7:].strip()) != device_id:
            return self._web.json_response({"ok": False, "error": "Pair this phone first."}, status=401, headers=headers)
        try:
            value = profile(self.router)
        except ValueError as exc:
            return self._web.json_response({"ok": False, "error": str(exc)}, status=503, headers=headers)
        # Never log, trace, persist or send this payload to a WebView event.
        return self._web.json_response(value, headers=headers)

    async def _handle_speech(self, request: Any) -> Any:
        """Return selected Gemini audio only to an authenticated, paired phone."""
        from .remote_speech import MAX_SPEECH_CHARS, SpeechUnavailable, speech_voice, synthesise_charon
        device_id = self._authenticated_device(request, {})
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        if not self._rate_ok(f"speech:{device_id}"):
            return self._web.json_response({"ok": False, "error": "Rate limited"}, status=429)
        try:
            payload = await self._json_object(request, "speech")
        except Exception as exc:
            return self._bad_body(exc)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPEECH_CHARS:
            return self._web.json_response(
                {"ok": False, "error": "Speech needs 1–1200 characters"}, status=400)
        try:
            voice = speech_voice(payload.get("voice"))
        except ValueError as exc:
            return self._web.json_response({"ok": False, "error": str(exc)}, status=400)
        if device_id in self._speech_devices or len(self._speech_devices) >= 2:
            return self._web.json_response({"ok": False, "error": "Speech is busy"}, status=429)
        self._speech_devices.add(device_id)
        try:
            if request.path.endswith("/stream"):
                return await self._stream_speech(request, device_id, text.strip(), voice)
            audio = await synthesise_charon(self.router, text.strip(), voice) if "voice" in payload else \
                await synthesise_charon(self.router, text.strip())
            # Revocation during synthesis must also prevent audio delivery.
            if self._authenticated_device(request, {}) != device_id:
                return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
            return self._web.Response(body=audio, content_type="audio/wav", headers={
                "Cache-Control": "no-store", "X-Orion-Voice": voice})
        except SpeechUnavailable as exc:
            self.bus.log.emit(f"REMOTE: {exc}")
            return self._web.json_response({"ok": False, "error": str(exc)}, status=503)
        finally:
            self._speech_devices.discard(device_id)

    async def _stream_speech(self, request: Any, device_id: str, text: str, voice: str = "Charon") -> Any:
        """Send length-prefixed PCM over HTTPS, including a verified end marker.

        This binary response works through Quick Tunnels, which reject SSE.
        A missing end marker tells the phone a connection or provider failed.
        """
        import struct
        from .remote_speech import PCM_RATE, SpeechUnavailable, stream_charon

        started = time.monotonic()
        audio = stream_charon(self.router, text) if voice == "Charon" else stream_charon(self.router, text, voice)
        response = None
        next_chunk = None
        try:
            while True:
                next_chunk = asyncio.create_task(anext(audio))
                while not next_chunk.done():
                    await asyncio.wait({next_chunk}, timeout=0.5)
                    if request.transport is None or request.transport.is_closing():
                        raise ConnectionResetError("Speech client disconnected")
                    if self._authenticated_device(request, {}) != device_id:
                        raise SpeechUnavailable("Speech authorisation expired")
                try:
                    pcm = next_chunk.result()
                except StopAsyncIteration:
                    break
                if response is None:
                    response = self._web.StreamResponse(headers={
                        "Content-Type": "application/vnd.orion.pcm",
                        "Cache-Control": "no-store, no-transform",
                        "X-Accel-Buffering": "no",
                        "X-Orion-Voice": voice,
                        "X-Orion-PCM-Format": "s16le-mono-v1",
                        "X-Orion-Sample-Rate": str(PCM_RATE),
                        "X-Orion-First-Audio-Ms": str(round(
                            (time.monotonic() - started) * 1000)),
                    })
                    await response.prepare(request)
                await response.write(struct.pack("!I", len(pcm)) + pcm)
            if response is None:
                raise SpeechUnavailable("Charon returned no speech audio")
            await response.write(b"\x00\x00\x00\x00")
            await response.write_eof()
            return response
        except (SpeechUnavailable, ConnectionError) as exc:
            if response is None:
                if isinstance(exc, SpeechUnavailable):
                    raise
                return self._web.Response(status=499)
            response.force_close()
            # Do not mark partial audio as complete or send provider text.
            if request.transport is not None:
                request.transport.close()
            self.bus.log.emit("REMOTE: Charon speech stream was interrupted")
            return response
        finally:
            if next_chunk is not None:
                if not next_chunk.done():
                    next_chunk.cancel()
                await asyncio.gather(next_chunk, return_exceptions=True)
            await audio.aclose()

    async def _handle_audio_stream(self, request: Any) -> Any:
        """Receive 16 kHz mono PCM and return transcript and speech instructions.

        A visible user action starts the Android microphone service. The phone
        commits a bounded utterance after a pause; only a transcript beginning
        with the wake word is routed into the paired conversational brain.
        """
        from aiohttp import WSMsgType
        header = str(request.headers.get("Authorization") or "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        device_id = self.auth.verify_access(token) if token else None
        if not device_id:
            return self._web.json_response(
                {"ok": False, "error": "unauthorised"}, status=401)
        if device_id not in self._audio_sessions and len(self._audio_sessions) >= 4:
            return self._web.json_response(
                {"ok": False, "error": "too many audio sessions"}, status=503)
        socket = self._web.WebSocketResponse(
            heartbeat=15, max_msg_size=32768, autoclose=True)
        await socket.prepare(request)
        previous = self._audio_sessions.get(device_id)
        self._audio_sessions[device_id] = socket
        stats = self._audio_stats(device_id)
        stats["state"] = "connected; waiting for speech"
        if previous is not None and previous is not socket:
            await previous.close(code=1000, message=b"New audio session")
        pcm = bytearray()
        max_pcm = 16000 * 2 * 12
        reply_task: asyncio.Task[None] | None = None
        wake_until = 0.0

        async def process_utterance(wav_bytes: bytes, interrupt_only: bool = False) -> None:
            nonlocal wake_until
            # Keep the WebSocket receive loop free to process protocol PONGs
            # while transcription or a provider reply is in progress. aiohttp
            # closes a heartbeat socket if its receive loop is held by a slow
            # provider, even when the phone has sent PONG on time.
            try:
                stats["state"] = "transcribing"
                transcript, error = await asyncio.wait_for(asyncio.to_thread(
                    self._transcribe_blocking, wav_bytes), timeout=75)
                if error:
                    stats["errors"] += 1
                    stats["state"] = "transcription unavailable"
                    await socket.send_json({"type": "error", "message": error})
                    return
                if self.auth.verify_access(token) != device_id:
                    await socket.close(code=1008, message=b"Authorisation expired")
                    return
                stats["transcriptions"] += 1
                if interrupt_only:
                    if re.fullmatch(r"\s*(?:(?:hey|okay|ok)\s+)?o\.?r\.?i\.?o\.?n\.?[,!\s]+"
                                    r"(?:stop|pause|quiet|stop speaking|be quiet)[.!?\s]*", transcript, re.I):
                        await socket.send_json({"type": "interrupt"})
                        stats["state"] = "speech interrupted by wake request"
                    else:
                        await socket.send_json({"type": "status", "message": "Say ORION stop to interrupt speech."})
                    return
                await socket.send_json({"type": "transcript", "text": transcript})
                spoken = re.sub(r"^\s*(?:(?:hey|hello|okay|ok)\s+)?"
                                r"(?:o\.?r\.?i\.?o\.?n\.?)\b[,.:;!\s]*", "",
                                transcript, flags=re.IGNORECASE)
                has_wake = spoken != transcript
                if has_wake and not spoken.strip():
                    stats["state"] = "wake word heard; waiting for request"
                    wake_until = time.monotonic() + 8
                    await socket.send_json({"type": "status", "message": "I'm listening. Say your request now."})
                    return
                if not has_wake and time.monotonic() > wake_until:
                    stats["state"] = "speech heard; say ORION before the request"
                    await socket.send_json({"type": "status", "message": "Say ORION before your request, or tap the microphone to dictate."})
                    return
                wake_until = 0.0
                if not spoken.strip():
                    return
                if not self._rate_ok(f"audio:{device_id}"):
                    await socket.send_json({"type": "error", "message": "Rate limited"})
                    return
                try:
                    SecuritySanitiser.guard_text(spoken, "remote.audio")
                except SecurityViolation:
                    await socket.send_json({"type": "error", "message": "Request refused"})
                    return
                self._last_active_device = device_id
                stats["requests"] += 1
                stats["state"] = "answering"
                user_hid = self._history.add(device_id, "user", spoken)
                await self._remember_turn("user", spoken)
                widgets = []
                with self.tool_gate.for_device(device_id):
                    reply, provider = await self._answer(spoken, device_id, on_widget=widgets.append)
                hid = self._history.add(device_id, "orion", reply)
                await self._persist_reply(reply)
                await socket.send_json({"type": "tts_text", "text": reply,
                                        "provider": provider, "widgets": widgets,
                                        "user_text": spoken, "user_hid": user_hid, "hid": hid})
                stats["state"] = "reply delivered; listening"
            except asyncio.CancelledError:
                raise
            except Exception:
                stats["errors"] += 1
                stats["state"] = "voice request failed"
                if not socket.closed:
                    try:
                        await socket.send_json(
                            {"type": "error", "message": "Reply unavailable"})
                    except Exception:
                        pass

        async def watch_access() -> None:
            while not socket.closed:
                await asyncio.sleep(15)
                if self.auth.verify_access(token) != device_id:
                    await socket.close(code=1008, message=b"Authorisation expired")
                    break

        watcher = asyncio.create_task(watch_access())
        try:
            async for message in socket:
                if self.auth.verify_access(token) != device_id:
                    await socket.close(code=1008, message=b"Authorisation expired")
                    break
                if message.type == WSMsgType.BINARY:
                    chunk = message.data
                    if not isinstance(chunk, bytes) or len(chunk) % 2 or len(chunk) > 32768:
                        await socket.close(code=1003, message=b"Invalid PCM")
                        break
                    if len(pcm) + len(chunk) > max_pcm:
                        pcm.clear()
                        await socket.send_json({"type": "error", "message": "Audio window too long"})
                    else:
                        pcm.extend(chunk)
                        stats["pcm_bytes"] += len(chunk)
                elif message.type == WSMsgType.TEXT:
                    try:
                        command = json.loads(message.data)
                    except ValueError:
                        continue
                    if not isinstance(command, dict):
                        continue
                    if command.get("type") == "ping":
                        await socket.send_json({"type": "pong"})
                    elif command.get("type") == "capture_status" and command.get("capturing") is True:
                        if reply_task is None or reply_task.done():
                            stats["state"] = "microphone capturing; waiting for speech"
                    elif command.get("type") == "commit" and len(pcm) >= 3200:
                        if not isinstance(command.get("interrupt_only", False), bool):
                            await socket.close(code=1003, message=b"Invalid capture mode")
                            break
                        if not self._rate_ok(f"audio-clips:{device_id}"):
                            pcm.clear()
                            await socket.send_json({"type": "error", "message": "Rate limited"})
                            continue
                        if reply_task is not None and not reply_task.done():
                            pcm.clear()
                            await socket.send_json({
                                "type": "error", "message": "Reply already in progress"})
                            continue
                        import wave
                        sound = io.BytesIO()
                        with wave.open(sound, "wb") as output:
                            output.setnchannels(1)
                            output.setsampwidth(2)
                            output.setframerate(16000)
                            output.writeframes(pcm)
                        pcm.clear()
                        reply_task = asyncio.create_task(
                            process_utterance(sound.getvalue(), command.get("interrupt_only", False)))
                        stats["utterances"] += 1
                elif message.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            watcher.cancel()
            if reply_task is not None and not reply_task.done():
                reply_task.cancel()
                await asyncio.wait({reply_task}, timeout=2)
            if self._audio_sessions.get(device_id) is socket:
                self._audio_sessions.pop(device_id, None)
                stats["state"] = "disconnected"
        return socket

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
        self.bus.log.emit(f"PHONE (voice): {text[:120]}" if text
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
        if rate < 8000 or rate > 48000:
            return "", "unsupported audio sample rate"
        if channels not in (1, 2):
            return "", "unsupported audio channels"
        if len(pcm) % (channels * width):
            return "", "unreadable audio (incomplete PCM frame)"
        if len(pcm) > rate * channels * width * 240:
            return "", "audio too long"
        # Validate cheap container metadata before the first request constructs
        # a potentially large offline speech model.
        transcriber = self._get_transcriber()
        if transcriber is None or not transcriber.available:
            return "", "no speech engine on the desktop"
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

    async def _handle_phone_actions(self, request: Any) -> Any:
        """Deliver pending native proposals over HTTPS when SSE is unavailable."""
        device_id = self._authenticated_device(request, {})
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        queue = self._phone_actions.pop(device_id, deque())
        now = time.monotonic()
        events = [payload for created, payload in queue if now - created < 180]
        return self._web.json_response({"ok": True, "events": events}, headers={
            "Cache-Control": "no-store"})

    async def _handle_confirmations(self, request: Any) -> Any:
        """List the actions this device has awaiting an approval tap (so a phone
        that reconnected mid-flight can catch up)."""
        header = str(request.headers.get("Authorization") or "")
        token = (header[7:].strip() if header.lower().startswith("bearer ")
                 else str(request.query.get("token") or ""))
        device_id = self.auth.verify_access(token) if token else None
        if device_id is None:
            return self._web.json_response({"ok": False, "error": "unauthorised"}, status=401)
        # Owner-only list, so it may carry the single-use token exactly as
        # the owner's live stream does; without it the phone could see a
        # pending approval it had missed but never act on it.
        pending = [{**c.public(), "token": c.token}
                   for c in self.confirmations.pending_for(device_id)]
        return self._web.json_response({"ok": True, "pending": pending})

    async def _handle_confirm(self, request: Any) -> Any:
        """Approve or deny a CONFIRM-tier action from the phone. On approval the
        tool finally runs through the dispatcher and the result is streamed back
        over SSE (and returned inline)."""
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response({"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await self._json_object(request, "confirm")
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
        text = (getattr(result, "text", "") or "Done.")
        widgets = self._widgets_from_result(result)
        ok = bool(getattr(result, "ok", True))
        self._push_device_event(device_id, {"type": "confirm_result", "id": conf_id,
                          "status": "done", "ok": ok, "reply": text, "widgets": widgets})
        return self._web.json_response({"ok": ok, "status": "done", "reply": text, "widgets": widgets})

    @staticmethod
    def _widgets_from_result(result: Any) -> list[dict]:
        from .presentation import TOOLS
        return [item["presentation"] for item in (getattr(result, "evidence", None) or [])
                if isinstance(item, dict) and isinstance(item.get("presentation"), dict)
                and item["presentation"].get("operation") in TOOLS][:6]

    async def _answer(self, message: str, device_id: str = "", *,
                      on_delta: Optional[Callable[[str], None]] = None,
                      on_restart: Optional[Callable[[str], None]] = None,
                      on_widget: Optional[Callable[[dict], None]] = None) -> tuple[str, str]:
        """Answer *message* with the full brain: identity- and memory-grounded
        language model when reachable, tool-capable LocalBrain otherwise.

        Full-parity remote (#12): an actionable command (open, search, read your
        mail, text someone, drive the browser…) is executed first through the
        capability gate — so ORION actually *does* it on the move, with an
        on-phone confirmation for anything irreversible. Non-command chat falls
        through to the model so conversation stays rich.

        *on_delta*/*on_restart* stream the model's text as it is written. Only
        the model path streams: the phone-vision and task paths produce one
        finished answer, and so does the offline brain."""
        # Phone control switched off (Mark XXXX.1): nothing from the phone reaches
        # this computer - its conversation included, not only its tools.
        from . import phone_control
        if not phone_control.enabled():
            phone_control.record("chat", "-", "off",
                                 device=device_id or (self.tool_gate.request_device() or ""))
            return phone_control.OFF_REPLY, "phone-control-off"
        # A requested look at the paired phone must send its fresh JPEG to a
        # vision-capable model. A text-only tool result cannot describe pixels.
        if device_id and re.search(r"\b(phone|mobile|android)\b", message, re.IGNORECASE):
            if re.search(r"\bwhat\s+(?:can|could)\s+(?:you|i|we)\s+(?:do|use)\b|"
                         r"^\s*(?:what|which|why|do|can|does|have)\b.*"
                         r"\b(?:capabilities|features|access|permissions)\b", message, re.IGNORECASE):
                return self._phone_capabilities(device_id), "phone-capabilities"
        if device_id and self.mobile_viewport is not None:
            low = message.lower()
            if (not re.search(r"\b(pc|desktop|computer|host|monitor)\b", low)
                    and re.search(r"\b(screen|screenshot|display|viewport)\b", low)
                    and re.search(r"\b(what|look|see|show|take|describe|read|analyse|analyze|identify|check)\b", low)):
                frame = self.mobile_viewport.latest_frame(device_id)
                if frame is not None:
                    from .vision_describe import describe_image
                    _ok, description = await describe_image(
                        self.router, frame.jpeg, message, max_tokens=450)
                    return description.replace("The camera worked", "The screen uplink worked"), "phone-vision"
        from .remote_tasks import is_action_request, plan_request, requested_target
        action_request = is_action_request(message)
        if device_id and action_request:
            try:
                target = requested_target(message)
            except ValueError as exc:
                return str(exc), "remote-task"
            if target == "phone":
                try:
                    system_extra = await asyncio.to_thread(self._grounding, message)
                    system_extra += "\n\nPAIRED PHONE CONTEXT:\n" + self._phone_capabilities(device_id)
                    steps = await plan_request(self.router, self.tool_gate, message, system_extra,
                                               target="phone")
                    result = await self.tool_gate.dispatch_plan(steps)
                    if on_widget:
                        for widget in self._widgets_from_result(result):
                            on_widget(widget)
                    return result.text, "remote-task"
                except ValueError as exc:
                    return str(exc), "remote-task"
                except Exception:
                    return ("The phone action could not return a verified receipt. "
                            "Check the phone connection and try again."), "remote-task"
        compound = bool(re.search(r"\b(?:and|then)\b", message, re.IGNORECASE))
        if self.local_brain is not None and hasattr(self.local_brain, "try_task") and not compound:
            try:
                task_reply = await self.local_brain.try_task(message)
            except Exception as exc:
                self.bus.log.emit(f"REMOTE: task path failed - {exc}")
                task_reply = None
            if task_reply is not None:
                return task_reply, "orion-remote"
        system_extra = await asyncio.to_thread(self._grounding, message)
        if device_id:
            system_extra += "\n\nPAIRED PHONE CONTEXT:\n" + self._phone_capabilities(device_id)
            system_extra += ("\nDescribe these implemented features accurately. Explain missing "
                             "permissions specifically; claim success only after a tool result.")
        if action_request and device_id:
            try:
                steps = await plan_request(self.router, self.tool_gate, message, system_extra,
                                           target="desktop")
                result = await self.tool_gate.dispatch_plan(steps)
                if on_widget:
                    for widget in self._widgets_from_result(result):
                        on_widget(widget)
                return result.text, "remote-task"
            except ValueError as exc:
                return str(exc), "remote-task"
            except Exception:
                return "I could not route that instruction to a tool. Nothing has run. Please check the PC diagnostics.", "remote-task"
        system_extra += ("\nThis is a conversation-only reply. Do not claim to control the PC or "
                         "phone, start a task, or perform an action without a tool result.")
        streamed = False

        def delta(text: str) -> None:
            nonlocal streamed
            streamed = True
            on_delta(text)

        if self.router is not None:
            try:
                if self.router.has_text_fallback():
                    profile, reply = await _generate_text(
                        self.router, message, system_extra,
                        delta if on_delta is not None else None, on_restart)
                    if reply and reply.strip():
                        return reply.strip(), getattr(profile, "name", "model")
            except Exception as exc:
                self.bus.log.emit(f"REMOTE: model path failed, using local brain - {exc}")
        # Text the model already streamed is not this answer's: tell the
        # phone to clear it before the offline brain's reply arrives.
        if streamed and on_restart is not None:
            on_restart("the model did not finish; answering offline")
        # Offline or the model failed → the tool-capable rule-based brain.
        if self.local_brain is not None:
            reply = await self.local_brain.respond(message)
            return (reply or "I'm here."), "local-brain"
        raise RuntimeError("no language model or local brain is reachable")

    def _phone_capabilities(self, device_id: str) -> str:
        """Describe implemented phone features using fresh permission telemetry."""
        bridge = getattr(self, "mobile_viewport", None)
        status = bridge.device_status(device_id) if bridge is not None else {"connected": False}
        parts = [
            "The native phone mode runs Gemini, Charon voice, wake listening and reviewed "
            "phone actions directly on Android, including when the PC is switched off, "
            "once its key has been provisioned. The phone needs mobile data or Wi-Fi. "
            "Desktop mode is optional and its PC tools require the PC to be online.",
            "Phone features include opening apps, reviewed calls and texts, verified e-mail "
            "sending where the supported mail app exposes a receipt, drafts, maps, navigation, "
            "sharing, contacts, calendar and alarms. Phone actions require local approval. "
            "Screen reading and gestures require ACCESS; screen images require VISION consent. "
            "Only shared screen images or approved UI snapshots are available, not private app data.",
        ]
        if status.get("connected"):
            permissions = []
            for field, label in (("accessibility", "ACCESS"), ("screen_capture", "VISION"),
                                 ("background_audio", "LISTEN"), ("microphone", "microphone permission"),
                                 ("calls", "call permission"), ("sms", "SMS permission")):
                if field in status:
                    permissions.append(f"{label}: {'on' if status[field] else 'off'}")
            parts.append("The native phone uplink is connected. " + "; ".join(permissions))
        else:
            parts.append("The desktop uplink is disconnected. Native phone mode can still run "
                         "directly with Gemini when its key and an internet connection are available.")
        return "\n\n".join(parts)

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
