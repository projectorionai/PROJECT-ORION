"""
Remote gateway — token-protected uplink to ORION's brain for browser and mobile.

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
``8765``). Access requires the token stored in ``config/remote_token.txt``.

SECURITY: for internet exposure put this behind an HTTPS reverse proxy (Caddy /
nginx / Cloudflare Tunnel) — see ``deploy/README_ORACLE_CLOUD.md``. The gateway
adds hardening headers, per-client rate limiting and a constant-time token
check, but it does not terminate TLS itself.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import time
from collections import deque
from typing import Any, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent
from .security import SecuritySanitiser, SecurityViolation

# ── installable PWA shell ─────────────────────────────────────────────────────

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
header{padding:calc(12px + env(safe-area-inset-top)) 18px 12px;border-bottom:1px solid #2a1118;
 background:linear-gradient(#12121a,#0a0a10);display:flex;align-items:center;gap:12px}
.orb{width:30px;height:30px;border-radius:50%;flex:0 0 auto;
 background:radial-gradient(circle at 35% 32%,#fff 0%,var(--pri) 38%,#4a0812 100%);
 box-shadow:0 0 14px 2px rgba(255,26,60,.7),0 0 4px 1px var(--cyan) inset}
h1{margin:0;font-size:17px;letter-spacing:2px}
.sub{color:#a9a9b2;font-size:10.5px;letter-spacing:.5px}
#dot{margin-left:auto;font-size:10px;color:#3ddc84}
#log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;
 -webkit-overflow-scrolling:touch}
.msg{max-width:88%;padding:11px 14px;border-radius:14px;font-size:15px;line-height:1.5;white-space:pre-wrap;
 word-wrap:break-word;animation:rise .18s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.user{align-self:flex-end;background:linear-gradient(#b3132b,#7c0d1e)}
.orion{align-self:flex-start;background:#14141d;border:1px solid #2a1118}
.orion.think{color:#a9a9b2;font-style:italic}
.meta{align-self:center;color:#5c5c68;font-size:11px}
form{display:flex;gap:8px;padding:12px 12px calc(12px + env(safe-area-inset-bottom));
 border-top:1px solid #2a1118;background:#0a0a10}
input{flex:1;background:#050508;color:#fff;border:1px solid var(--pri-dim);border-radius:12px;
 padding:13px;font-size:16px;outline:none}
input:focus{border-color:var(--pri)}
button{background:linear-gradient(#b3132b,#7c0d1e);color:#fff;border:1px solid var(--pri);
 border-radius:12px;padding:0 20px;font-weight:800;font-size:15px}
button:active{background:#66091a}
</style></head><body>
<header><div class="orb"></div><div><h1>O.R.I.O.N.</h1>
<div class="sub">REMOTE UPLINK · MEMORY-SYNCED</div></div><div id="dot">● online</div></header>
<div id="log"></div>
<form id="f"><input id="m" placeholder="Message ORION…" autocomplete="off" autocapitalize="sentences"><button>SEND</button></form>
<script>
const log=document.getElementById('log'),dot=document.getElementById('dot');
let token=localStorage.getItem('orion_token')||'';
if(!token){token=(prompt('Enter the ORION remote access token (config/remote_token.txt)')||'').trim();
 if(token)localStorage.setItem('orion_token',token);}
function add(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;
 log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
add('meta','Connected to ORION. Conversations sync into his memory.');
async function health(){try{const r=await fetch('/api/health');const j=await r.json();
 dot.textContent=(j.mode?('● '+j.mode.toLowerCase()):'● online');dot.style.color='#3ddc84';}
 catch(e){dot.textContent='○ offline';dot.style.color='#ffb020';}}
health();setInterval(health,20000);
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const inp=document.getElementById('m');const text=inp.value.trim();if(!text)return;
 inp.value='';add('user',text);const thinking=add('orion think','…');
 try{const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({token,message:text})});
  const data=await res.json();thinking.remove();
  if(res.status===401){localStorage.removeItem('orion_token');add('orion','Invalid token. Reload and re-enter it.');return;}
  if(res.status===429){add('orion','Easy, sir — too many requests at once. One moment.');return;}
  add('orion',data.ok?data.reply:('Fault: '+(data.error||'unknown')));}
 catch(err){thinking.remove();add('orion','Link fault: '+err);}
});
if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
</script></body></html>"""

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
const SHELL='orion-shell-v1';
const ASSETS=['/','/manifest.webmanifest','/icon-192.png','/icon-512.png'];
self.addEventListener('install',e=>{self.skipWaiting();
 e.waitUntil(caches.open(SHELL).then(c=>c.addAll(ASSETS)).catch(()=>{}));});
self.addEventListener('activate',e=>{e.waitUntil(self.clients.claim());});
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);
 if(u.pathname.startsWith('/api/')){return;}  // always live
 e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)).catch(()=>caches.match('/')));});
"""


class RemoteGateway:
    """Token-protected aiohttp uplink answering through ORION's full brain."""

    # Per-client sliding-window rate limit.
    _RATE_MAX     = 30      # requests …
    _RATE_WINDOW  = 60.0    # … per this many seconds

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
    ) -> None:
        self.router       = router
        self.memory       = memory
        self.bus          = bus
        self.identity     = identity
        self.local_brain  = local_brain
        self.conversation = conversation
        self.telemetry    = telemetry
        self.host   = os.getenv("ORION_REMOTE_HOST", "0.0.0.0").strip() or "0.0.0.0"
        self.port   = int(os.getenv("ORION_REMOTE_PORT", "8765") or 8765)
        self.token  = self._load_token()
        self._web: Any    = None
        self._runner: Any = None
        self._hits: dict[str, deque[float]] = {}
        self._icons: dict[int, bytes] = {}
        self._started_at = time.time()

    # ── token (persisted, 24-byte url-safe) ───────────────────────────────────

    def _load_token(self) -> str:
        token_path = CONFIG_DIR / "remote_token.txt"
        try:
            if token_path.exists():
                existing = token_path.read_text(encoding="utf-8").strip()
                if existing:
                    return existing
        except Exception:
            pass
        import secrets
        token = secrets.token_urlsafe(24)
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token, encoding="utf-8")
        except Exception:
            pass
        return token

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        from aiohttp import web
        self._web = web

        @web.middleware
        async def security_mw(request: Any, handler: Any) -> Any:
            response = await handler(request)
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:",
            )
            return response

        app = web.Application(middlewares=[security_mw])
        app.router.add_get("/",                    self._handle_page)
        app.router.add_get("/manifest.webmanifest", self._handle_manifest)
        app.router.add_get("/sw.js",               self._handle_sw)
        app.router.add_get("/icon-192.png",        self._handle_icon)
        app.router.add_get("/icon-512.png",        self._handle_icon)
        app.router.add_get("/api/health",          self._handle_health)
        app.router.add_post("/api/chat",           self._handle_chat)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.bus.log.emit(
            f"REMOTE: uplink active on {self.host}:{self.port} (installable PWA); "
            "token in config/remote_token.txt. Expose via HTTPS proxy for internet use."
        )
        if self.telemetry is not None:
            try:
                self.telemetry.health.register("remote")
                self.telemetry.health.beat("remote", "OK", f"port {self.port}")
            except Exception:
                pass

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ── rate limiting ──────────────────────────────────────────────────────────

    def _client_key(self, request: Any) -> str:
        # Honour a reverse proxy's forwarded client IP, else the socket peer.
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
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

    async def _handle_page(self, request: Any) -> Any:
        return self._web.Response(text=REMOTE_PAGE_HTML, content_type="text/html")

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

    # ── chat: the full brain ───────────────────────────────────────────────────

    async def _handle_chat(self, request: Any) -> Any:
        key = self._client_key(request)
        if not self._rate_ok(key):
            return self._web.json_response(
                {"ok": False, "error": "rate limited"}, status=429)
        try:
            payload = await request.json()
        except Exception:
            return self._web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
        # Constant-time token comparison — no early-exit timing leak.
        supplied = str(payload.get("token") or "")
        if not hmac.compare_digest(supplied, self.token):
            return self._web.json_response({"ok": False, "error": "invalid token"}, status=401)
        message = str(payload.get("message") or "").strip()[:4000]
        if not message:
            return self._web.json_response({"ok": False, "error": "empty message"}, status=400)
        try:
            SecuritySanitiser.guard_text(message, "remote.message")
        except SecurityViolation as exc:
            return self._web.json_response({"ok": False, "error": str(exc)}, status=403)

        self.bus.log.emit(f"REMOTE: {message[:120]}")
        try:
            self.memory.log_episode("user (remote)", message)
        except Exception:
            pass
        try:
            reply, provider = await self._answer(message)
        except Exception as exc:
            return self._web.json_response(
                {"ok": False, "error": str(exc).splitlines()[0][:200]}, status=502)
        try:
            self.memory.log_episode("orion (remote)", reply)
        except Exception:
            pass
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("remote.turns")
            except Exception:
                pass
        return self._web.json_response({"ok": True, "reply": reply, "provider": provider})

    async def _answer(self, message: str) -> tuple[str, str]:
        """Answer *message* with the full brain: identity- and memory-grounded
        language model when reachable, tool-capable LocalBrain otherwise."""
        system_extra = self._grounding(message)
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
            return (reply or "I'm here, sir."), "local-brain"
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
