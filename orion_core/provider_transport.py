"""
Provider transport — how a chat request actually reaches a model server.

``providers.ProviderRouter`` decides WHICH provider answers: order, cooldowns,
healing, payload budgets. This module owns HOW a request travels, one
transport per kind of server:

    OpenAIChatTransport   ``/v1/chat/completions`` — every cloud provider,
                          LM Studio, llama.cpp's server, vLLM, and Ollama
                          when native use is switched off.
    OllamaChatTransport   Ollama's native ``/api/chat``.

Why Ollama has a transport of its own
-------------------------------------
ORION sent ``keep_alive: 0`` and ``options.num_ctx`` to Ollama through
``/v1/chat/completions``. Ollama 0.32.6 converts a compatible request into
its own from a fixed list of OpenAI fields and drops the rest, so neither
setting ever reached the model: "unload the moment the call finishes" and
"hold the context to N tokens" existed only in the outgoing JSON, and in a
test that asserted that JSON. The native endpoint documents both, reports
model-load time, and counts tokens exactly.

Nothing here assumes a local server is Ollama. Locality says nothing about
which fields a server understands, so another server on 127.0.0.1 keeps the
``/v1`` path and never receives an Ollama-only field. Ollama is identified by
asking it (``GET /api/version``) — see ``probe_ollama``.

Sessions
--------
Every call used to open and close its own ``aiohttp.ClientSession``, so no
connection was ever reused: each cloud turn paid DNS, TCP and TLS again, and a
research run's six parallel sections paid it six times. ``SessionPool`` keeps
one session per event loop with a bounded connector. Credentials and timeouts
stay per request; the shared session carries no Authorization header and no
cookie jar, so two profiles on one host cannot see each other's state.

A session belongs to the loop that created it. The suite runs hundreds of
``asyncio.run`` loops, and reusing a session across them raises "attached to
a different loop" or, worse, sends on a dead transport — so the pool is keyed
by the running loop and drops sessions whose loop has closed.

Admission
---------
``EndpointGate`` bounds concurrent requests per provider (per server for local
models: one GPU, one generation at a time by default). Interactive callers are
served before background ones, and background work can be held below the
limit so a conversation always finds a free slot; a background waiter that has
queued for long enough is promoted so it cannot starve.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable


def _aiohttp() -> Any:
    # aiohttp costs ~178 ms to import; nothing here runs before the first
    # provider call, so it stays off ORION's startup path.
    import aiohttp

    return aiohttp


# ──────────────────────────────────────────────────────────────────────────────
# REQUESTS AND RESULTS
# ──────────────────────────────────────────────────────────────────────────────

#: Placeholder "keys" that mean "no credential" for local servers.
_NO_KEY = {"", "local", "none", "no-key"}


def auth_headers(api_key: str, *, accept: str | None = None) -> dict[str, str]:
    """Headers for ONE request. Built per call, never stored on a session."""
    headers = {"Content-Type": "application/json"}
    if accept:
        headers["Accept"] = accept
    key = str(api_key or "").strip()
    if key.lower() not in _NO_KEY:
        headers["Authorization"] = f"Bearer {key}"
    return headers


@dataclass
class ChatRequest:
    """One chat turn, independent of the wire format that carries it."""

    model: str
    system: str
    prompt: str
    max_tokens: int
    api_key: str = ""
    temperature: float = 0.35
    image_jpeg: bytes | None = None
    stream: bool = False
    on_delta: Callable[[str], None] | None = None
    #: OpenAI-compatible only (``reasoning_effort`` and the like).
    extra: dict[str, Any] = field(default_factory=dict)
    #: Ollama only. ``0`` unloads as soon as the reply is done.
    keep_alive: int | str | None = None
    #: Ollama only. The context the model is actually loaded with.
    num_ctx: int | None = None
    #: Bound on the reply's size in bytes (vision replies are untrusted JSON).
    max_response_bytes: int | None = None
    #: Ollama only. A JSON schema the reply is decoded against (``format`` on
    #: /api/chat). It makes well-formed output likely, never guaranteed: the
    #: caller still validates the reply as untrusted text.
    response_format: dict[str, Any] | None = None
    #: Previous assistant output when resuming a reply stopped by a token limit.
    continuation: str = ""


@dataclass
class ChatResult:
    """What came back, plus where the time went."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    response_id: str = ""
    model: str = ""
    streamed: bool = False
    #: True when the request rode an already-open connection.
    reused: bool | None = None
    #: Socket set-up (0.0 on a reused connection; None if not observed).
    connect_ms: float | None = None
    #: Request start to response headers.
    headers_ms: float | None = None
    #: Request start to the first text fragment.
    first_token_ms: float | None = None
    total_ms: float = 0.0
    #: Ollama: time spent loading the model into memory for this request.
    load_ms: float | None = None
    #: Ollama: the server's own measure of the whole request.
    server_ms: float | None = None
    done_reason: str = ""
    #: Native model function calls are proposals, never executed by transport.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolChatRequest:
    """A multi-message chat turn that offers the model functions (Ollama only).

    The local agent's loop: the whole conversation so far (system, user,
    assistant tool calls, tool results) travels each time, with the shortlist
    of tools as JSON-schema functions. The reply's ``tool_calls`` are
    PROPOSALS; nothing here executes one.
    """

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 512
    api_key: str = ""
    temperature: float = 0.2
    num_ctx: int | None = None
    keep_alive: int | str | None = None
    on_delta: Callable[[str], None] | None = None
    #: Bound on the reply's text in bytes (model output is untrusted).
    max_response_bytes: int | None = 262_144
    #: Always streamed: first-token time is observable and a cancelled call
    #: closes the connection, which stops generation.
    stream: bool = True
    #: Never an image; present so the shared error path can read it.
    image_jpeg: bytes | None = None


#: Bound on the JSON of the function calls in one reply.
MAX_TOOL_CALL_BYTES = 65_536
#: More proposals than this in one reply is not a plan, it is noise.
MAX_TOOL_CALLS = 8


def _tool_call(raw: Any) -> dict[str, Any] | None:
    """Ollama's ``{"function": {"name", "arguments"}}`` -> ``{"name", "args", "id"}``.

    Arguments normally arrive as an object; some model templates emit a JSON
    string, which is decoded. Anything else is kept as unparseable so the
    caller can refuse it rather than guess.
    """
    if not isinstance(raw, dict):
        return None
    function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args: Any = function.get("arguments", function.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args.strip() or "{}")
        except ValueError:
            args = None
    call: dict[str, Any] = {"name": name.strip(), "args": args if isinstance(args, dict) else {},
                            "id": str(raw.get("id") or function.get("index") or "")}
    if not isinstance(args, dict):
        call["invalid_args"] = True
    return call


CONTINUE_REPLY = (
    "Continue the preceding answer exactly where it stopped. Do not repeat its "
    "opening or any completed sections. If it stopped inside a word or code "
    "block, finish that first. Complete the answer to the original request, "
    "including its remaining steps and conclusion. Use British English."
)


def _continuation_messages(req: ChatRequest) -> list[dict[str, str]]:
    if not req.continuation:
        return []
    return [{"role": "assistant", "content": req.continuation},
            {"role": "user", "content": CONTINUE_REPLY}]


def _emit(on_delta: Callable[[str], None] | None, piece: str) -> None:
    # A display callback must never break the answer it is displaying.
    if on_delta is None:
        return
    try:
        on_delta(piece)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# SESSIONS
# ──────────────────────────────────────────────────────────────────────────────

def _timing_trace() -> Any:
    """aiohttp trace hooks that time connection set-up separately from the
    model's own latency, and record whether a pooled connection was reused.

    Each request passes a plain dict as ``trace_request_ctx``; the hooks write
    into it. Without this, "the provider was slow" and "the TLS handshake was
    slow" were the same number.
    """
    aiohttp = _aiohttp()
    trace = aiohttp.TraceConfig()

    def _ctx(config_ctx: Any) -> dict[str, Any] | None:
        found = getattr(config_ctx, "trace_request_ctx", None)
        return found if isinstance(found, dict) else None

    async def queued_start(_session: Any, config_ctx: Any, _params: Any) -> None:
        ctx = _ctx(config_ctx)
        if ctx is not None:
            ctx["pool_wait_start"] = time.monotonic()

    async def queued_end(_session: Any, config_ctx: Any, _params: Any) -> None:
        ctx = _ctx(config_ctx)
        if ctx is not None and "pool_wait_start" in ctx:
            ctx["pool_wait_ms"] = (time.monotonic() - ctx.pop("pool_wait_start")) * 1000.0

    async def create_start(_session: Any, config_ctx: Any, _params: Any) -> None:
        ctx = _ctx(config_ctx)
        if ctx is not None:
            ctx["connect_start"] = time.monotonic()

    async def create_end(_session: Any, config_ctx: Any, _params: Any) -> None:
        ctx = _ctx(config_ctx)
        if ctx is not None and "connect_start" in ctx:
            ctx["connect_ms"] = (time.monotonic() - ctx.pop("connect_start")) * 1000.0
            ctx["reused"] = False

    async def reused(_session: Any, config_ctx: Any, _params: Any) -> None:
        ctx = _ctx(config_ctx)
        if ctx is not None:
            ctx["connect_ms"] = 0.0
            ctx["reused"] = True

    trace.on_connection_queued_start.append(queued_start)
    trace.on_connection_queued_end.append(queued_end)
    trace.on_connection_create_start.append(create_start)
    trace.on_connection_create_end.append(create_end)
    trace.on_connection_reuseconn.append(reused)
    return trace


@dataclass
class _PoolEntry:
    loop_ref: Any                    # weakref to the owning event loop
    session: Any
    connector: Any
    keeper: Any = None               # async generator closed with the loop


async def _close_with_loop(entry: _PoolEntry) -> Any:
    """Close the session when its loop shuts down.

    ``asyncio.run`` (and ``asyncio.Runner``) finalise every live async
    generator before closing the loop, so a generator parked at ``yield`` is a
    reliable "this loop is ending" hook. Without it, every test that routes a
    real request would leave an open session behind, and aiohttp reports each
    one as "Unclosed client session" when it is collected. The application
    closes the pool explicitly (``ProviderRouter.aclose``); this covers every
    loop nobody remembered to close.
    """
    try:
        yield
    finally:
        session = entry.session
        if not getattr(session, "closed", True):
            await session.close()


class SessionPool:
    """One reusable aiohttp session per running event loop."""

    #: Idle connections are dropped after this long. Shorter than the idle
    #: timeout of any provider front end seen so far, so ORION closes an idle
    #: connection before the server does and rarely writes into one the server
    #: has just dropped (the stale-connection retry below covers the rest).
    KEEPALIVE_S = 30.0

    def __init__(self, *, limit: int = 32, limit_per_host: int = 8,
                 keepalive_s: float | None = None) -> None:
        self.limit = max(1, int(limit))
        self.limit_per_host = max(1, int(limit_per_host))
        self.keepalive_s = self.KEEPALIVE_S if keepalive_s is None else float(keepalive_s)
        self._entries: dict[int, _PoolEntry] = {}
        self._lock = threading.Lock()
        self.closed = False
        #: Sessions opened over this pool's life (tests assert reuse with it).
        self.created = 0

    async def get(self) -> Any:
        """The session for the running loop, created on first use."""
        if self.closed:
            raise RuntimeError("the provider session pool has been closed")
        loop = asyncio.get_running_loop()
        self._purge()
        with self._lock:
            entry = self._entries.get(id(loop))
            if entry is not None and entry.loop_ref() is loop \
                    and not getattr(entry.session, "closed", False):
                return entry.session
            aiohttp = _aiohttp()
            connector = aiohttp.TCPConnector(
                limit=self.limit, limit_per_host=self.limit_per_host,
                keepalive_timeout=self.keepalive_s, ttl_dns_cache=300)
            session = aiohttp.ClientSession(
                connector=connector,
                # No cookie jar: two profiles on one host (a general profile
                # and the 'thoughts' profile on OpenRouter, say) must never
                # carry each other's session state.
                cookie_jar=aiohttp.DummyCookieJar(),
                trace_configs=[_timing_trace()],
                # Every request passes its own timeout.
                timeout=aiohttp.ClientTimeout(total=None))
            entry = _PoolEntry(weakref.ref(loop), session, connector)
            self._entries[id(loop)] = entry
            self.created += 1
        keeper = _close_with_loop(entry)
        try:
            await keeper.__anext__()
            entry.keeper = keeper
        except Exception:
            pass
        return session

    def _purge(self) -> None:
        """Forget sessions whose loop has gone. They cannot be closed properly
        any more (closing needs their loop), so they are detached quietly."""
        with self._lock:
            dead = [key for key, entry in self._entries.items()
                    if entry.loop_ref() is None or entry.loop_ref().is_closed()]
            entries = [self._entries.pop(key) for key in dead]
        for entry in entries:
            self._discard(entry)

    @staticmethod
    def _discard(entry: _PoolEntry) -> None:
        try:
            entry.connector._close()
        except Exception:
            pass
        try:
            entry.session.detach()      # marks it closed without a loop
        except Exception:
            pass
        keeper = entry.keeper
        if keeper is not None:
            # The session reads as closed now, so the keeper's finally has
            # nothing to await and finishes in one synchronous step — which
            # keeps its finaliser from trying to schedule on a dead loop.
            closer = keeper.aclose()
            try:
                closer.send(None)
            except (StopIteration, RuntimeError):
                pass
            except Exception:
                pass
            finally:
                try:
                    closer.close()
                except Exception:
                    pass

    async def aclose(self) -> None:
        """Close this loop's session; detach any left by closed loops.
        Idempotent. Nothing can be opened afterwards."""
        if self.closed:
            return
        self.closed = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            if loop is not None and entry.loop_ref() is loop:
                try:
                    if entry.keeper is not None:
                        await entry.keeper.aclose()
                    if not getattr(entry.session, "closed", True):
                        await entry.session.close()
                except Exception:
                    pass
            else:
                self._discard(entry)

    def live_sessions(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values()
                       if not getattr(e.session, "closed", True))


# ──────────────────────────────────────────────────────────────────────────────
# SENDING
# ──────────────────────────────────────────────────────────────────────────────

async def _send(session: Any, method: str, url: str, *, trace: dict[str, Any],
                **kwargs: Any) -> Any:
    """Send one request and return the response once its headers arrive.

    One retry, and only for a connection taken from the pool that turned out
    to be dead before any response arrived — the server had closed it while it
    sat idle. That request was never read, so sending it again cannot do the
    work twice; it is the retry browsers make for the same reason, and without
    it pooling would have turned "the server tidied up an idle socket" into a
    provider failure and a 45-second cooldown. A fresh connection that fails,
    or any failure after the response began, is not retried here.
    """
    aiohttp = _aiohttp()
    for attempt in (1, 2):
        trace.clear()
        trace["started"] = time.monotonic()
        try:
            # Provider requests carry API keys. A redirect may point at a
            # different host, so leave any change of destination to the
            # configured provider profile rather than forwarding credentials.
            response = await session.request(
                method, url, trace_request_ctx=trace, allow_redirects=False, **kwargs)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
            if attempt == 1 and trace.get("reused"):
                continue
            raise
        trace["headers_at"] = time.monotonic()
        if 300 <= response.status < 400:
            response.close()
            raise RuntimeError(f"Provider redirected the request (HTTP {response.status})")
        return response
    raise RuntimeError("unreachable")  # pragma: no cover


def _timings(result: ChatResult, trace: dict[str, Any], started: float) -> ChatResult:
    result.total_ms = (time.monotonic() - started) * 1000.0
    result.connect_ms = trace.get("connect_ms")
    result.reused = trace.get("reused")
    if "headers_at" in trace:
        result.headers_ms = (trace["headers_at"] - started) * 1000.0
    return result


async def _drain(response: Any) -> None:
    """Read what follows an end-of-stream marker so the connection can go back
    to the pool. Normal EOF is already buffered or follows immediately. A
    server that retains the response after its completion marker gets one
    brief scheduling window, then its socket is closed. Waiting half a second
    here delayed every continuation and occupied an interactive provider slot.

    Ignore trailing bytes after a COMPLETE marker, with a size bound; reading
    an unbounded response after [DONE] could otherwise grow memory indefinitely.
    """
    async def discard() -> None:
        remaining = 65_536
        while remaining > 0:
            chunk = await response.content.read(min(16_384, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
        response.close()

    try:
        await asyncio.wait_for(discard(), timeout=0.02)
    except Exception:
        response.close()


async def _read_bounded(response: Any, limit: int | None) -> str:
    if limit is None:
        return await response.text()
    chunks = bytearray()
    async for chunk in response.content.iter_chunked(16_384):
        chunks.extend(chunk)
        if len(chunks) > limit:
            raise RuntimeError("Vision provider response exceeded the size limit")
    return chunks.decode("utf-8")


def _error_text(raw: str) -> str:
    """The server's own error message when the body is JSON, else the body."""
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    # Google's compatible API sometimes wraps its error in a JSON array.
    # Returning that pretty-printed body made the persisted diagnosis just
    # "HTTP 503: [{", losing the actual reason and any retry/quota hints.
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if not isinstance(row, dict):
            continue
        err = row.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or json.dumps(err))
            details = err.get("details")
            if details:
                message += " " + json.dumps(details, separators=(",", ":"))
            err = message
        if err:
            return str(err)
    return raw


def chat_endpoint(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def ollama_root(base_url: str) -> str:
    """``http://127.0.0.1:11434/v1`` -> ``http://127.0.0.1:11434``."""
    base = str(base_url or "").rstrip("/")
    for suffix in ("/chat/completions", "/v1", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


# ──────────────────────────────────────────────────────────────────────────────
# TRANSPORTS
# ──────────────────────────────────────────────────────────────────────────────

class OpenAIChatTransport:
    """``/v1/chat/completions`` — JSON, or server-sent events when streaming."""

    kind = "openai"

    def __init__(self, base_url: str) -> None:
        self.endpoint = chat_endpoint(base_url)

    def payload(self, req: ChatRequest) -> dict[str, Any]:
        user: Any = req.prompt
        if req.image_jpeg is not None:
            user = [
                {"type": "text", "text": req.prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64,"
                           + base64.b64encode(req.image_jpeg).decode("ascii")}},
            ]
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": user},
            ],
            "temperature": req.temperature,
            "stream": bool(req.stream),
            # ALWAYS bounded: a provider not told a ceiling reserves the
            # model's whole completion budget against the balance (HTTP 402).
            "max_tokens": int(req.max_tokens),
        }
        payload["messages"].extend(_continuation_messages(req))
        payload.update(req.extra)
        return payload

    async def chat(self, session: Any, req: ChatRequest, *, timeout: Any,
                   trace: dict[str, Any]) -> ChatResult:
        headers = auth_headers(req.api_key, accept="text/event-stream" if req.stream else None)
        started = time.monotonic()
        response = await _send(session, "POST", self.endpoint, trace=trace,
                               json=self.payload(req), headers=headers, timeout=timeout)
        try:
            ctype = str(response.headers.get("Content-Type", "")).lower()
            if req.stream and response.status < 400 and "json" not in ctype:
                result = await self._read_events(response, req, started)
            else:
                # Also the answer to a stream request from a server that does
                # not stream: take the whole reply and hand it over as one piece.
                result = await self._read_json(response, req)
                if req.stream and result.text:
                    result.first_token_ms = (time.monotonic() - started) * 1000.0
                    _emit(req.on_delta, result.text)
        except BaseException:
            response.close()
            raise
        response.release()
        return _timings(result, trace, started)

    async def _read_json(self, response: Any, req: ChatRequest) -> ChatResult:
        raw = await _read_bounded(response, req.max_response_bytes)
        if response.status >= 400:
            if req.image_jpeg is not None:
                raise RuntimeError(f"Vision provider HTTP {response.status}")
            raise RuntimeError(f"HTTP {response.status}: {_error_text(raw)[:500]}")
        data = json.loads(raw)
        choices = (data.get("choices") if isinstance(data, dict) else None) or []
        if not choices:
            raise RuntimeError("the provider returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text") or ""
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatResult(text=str(content), usage=dict(usage),
                          response_id=str(data.get("id") or ""),
                          model=str(data.get("model") or ""),
                          done_reason=str(choices[0].get("finish_reason") or ""))

    async def _read_events(self, response: Any, req: ChatRequest,
                           started: float) -> ChatResult:
        parts: list[str] = []
        usage: dict[str, Any] = {}
        first: float | None = None
        size = 0
        finished = False
        done_reason = ""
        async for raw_line in response.content:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                finished = True
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("error"), (dict, str)):
                # OpenRouter and others report a mid-stream failure as an event;
                # its text carries the class (quota, credit...) the router reads.
                raise RuntimeError(f"stream error: {_error_text(data_str)[:300]}")
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            done_reason = str(choices[0].get("finish_reason") or done_reason)
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece is None:  # some servers echo the whole message once
                piece = (choices[0].get("message") or {}).get("content")
            if piece:
                piece = str(piece)
                size += len(piece.encode("utf-8"))
                if req.max_response_bytes is not None and size > req.max_response_bytes:
                    raise RuntimeError("Vision provider response exceeded the size limit")
                if first is None:
                    first = time.monotonic()
                parts.append(piece)
                _emit(req.on_delta, piece)
        if finished:
            await _drain(response)
        elif not done_reason:
            raise RuntimeError("Provider stream ended before the reply was complete")
        result = ChatResult(text="".join(parts), usage=dict(usage), streamed=True,
                            done_reason=done_reason)
        if first is not None:
            result.first_token_ms = (first - started) * 1000.0
        return result


class OllamaChatTransport:
    """Ollama's native ``/api/chat``, always streamed as NDJSON.

    Streaming even when the caller wants the whole answer costs nothing on
    loopback and buys two things: the first-token time is observable, and a
    cancelled call closes the connection mid-reply, which is what makes Ollama
    stop generating instead of finishing a reply nobody will read.
    """

    kind = "ollama"

    def __init__(self, root: str) -> None:
        self.root = ollama_root(root)
        self.endpoint = f"{self.root}/api/chat"

    def payload(self, req: ChatRequest) -> dict[str, Any]:
        user: dict[str, Any] = {"role": "user", "content": req.prompt}
        if req.image_jpeg is not None:
            user["images"] = [base64.b64encode(req.image_jpeg).decode("ascii")]
        options: dict[str, Any] = {"temperature": req.temperature,
                                   "num_predict": int(req.max_tokens)}
        if req.num_ctx:
            options["num_ctx"] = int(req.num_ctx)
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": "system", "content": req.system}, user],
            "stream": True,
            "options": options,
        }
        payload["messages"].extend(_continuation_messages(req))
        if req.keep_alive is not None:
            payload["keep_alive"] = req.keep_alive
        if req.response_format is not None:
            payload["format"] = req.response_format
        return payload

    async def chat(self, session: Any, req: ChatRequest, *, timeout: Any,
                   trace: dict[str, Any]) -> ChatResult:
        headers = auth_headers(req.api_key, accept="application/x-ndjson")
        started = time.monotonic()
        response = await _send(session, "POST", self.endpoint, trace=trace,
                               json=self.payload(req), headers=headers, timeout=timeout)
        try:
            if response.status >= 400:
                raw = await _read_bounded(response, 65_536)
                if req.image_jpeg is not None:
                    raise RuntimeError(f"Vision provider HTTP {response.status}")
                raise RuntimeError(f"HTTP {response.status}: {_error_text(raw)[:500]}")
            result = await self._read_stream(response, req, started)
        except BaseException:
            response.close()
            raise
        response.release()
        return _timings(result, trace, started)

    async def _read_stream(self, response: Any, req: ChatRequest,
                           started: float) -> ChatResult:
        parts: list[str] = []
        first: float | None = None
        final: dict[str, Any] | None = None
        size = 0
        calls: list[dict[str, Any]] = []
        call_bytes = 0
        async for raw_line in response.content:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                # Status is already 200 by now; the text carries the class
                # (out of memory, model not found) the router acts on.
                raise RuntimeError(f"Ollama error: {str(chunk['error'])[:400]}")
            message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
            # Function calls arrive whole, in one chunk or spread over
            # several (newer servers stream them); every one is collected.
            for raw_call in message.get("tool_calls") or ():
                call_bytes += len(json.dumps(raw_call, default=str))
                if call_bytes > MAX_TOOL_CALL_BYTES or len(calls) >= MAX_TOOL_CALLS:
                    raise RuntimeError("Ollama tool calls exceeded the size limit")
                parsed = _tool_call(raw_call)
                if parsed is not None:
                    calls.append(parsed)
                if first is None:
                    first = time.monotonic()
            piece = str(message.get("content") or "")
            if piece:
                size += len(piece.encode("utf-8"))
                if req.max_response_bytes is not None and size > req.max_response_bytes:
                    raise RuntimeError("Vision provider response exceeded the size limit")
                if first is None:
                    first = time.monotonic()
                parts.append(piece)
                _emit(req.on_delta, piece)
            if chunk.get("done"):
                final = chunk
                break
        if final is None:
            raise RuntimeError("Ollama stream ended before the reply was complete")
        await _drain(response)
        prompt_tokens = int(final.get("prompt_eval_count") or 0)
        output_tokens = int(final.get("eval_count") or 0)
        result = ChatResult(
            text="".join(parts),
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens,
                   "total_tokens": prompt_tokens + output_tokens},
            model=str(final.get("model") or req.model),
            streamed=True,
            done_reason=str(final.get("done_reason") or ""),
            tool_calls=calls,
        )
        if first is not None:
            result.first_token_ms = (first - started) * 1000.0
        if final.get("load_duration") is not None:
            result.load_ms = float(final["load_duration"]) / 1e6
        if final.get("total_duration") is not None:
            result.server_ms = float(final["total_duration"]) / 1e6
        return result

    # ── residency ─────────────────────────────────────────────────────────────

    async def loaded_models(self, session: Any, *, timeout: float = 2.0) -> list[dict[str, Any]]:
        """``/api/ps``: what is resident right now, with its context length."""
        aiohttp = _aiohttp()
        async with session.get(f"{self.root}/api/ps",
                               timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status} from /api/ps")
            data = json.loads(await response.text())
        rows = data.get("models") if isinstance(data, dict) else None
        return [row for row in (rows or []) if isinstance(row, dict)]

    async def unload(self, session: Any, model: str, *, timeout: float = 5.0) -> None:
        """Ask Ollama to drop *model* from memory now.

        Idempotent (unloading an unloaded model is a no-op), so one retry on a
        dropped connection is safe. Ollama itself will not tear a model out
        from under a request that is still running on it.
        """
        aiohttp = _aiohttp()
        for attempt in (1, 2):
            try:
                async with session.post(
                        f"{self.root}/api/generate",
                        json={"model": model, "keep_alive": 0},
                        timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    body = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status}: {_error_text(body)[:200]}")
                    return
            except aiohttp.ClientConnectionError:
                if attempt == 2:
                    raise


class OllamaToolTransport(OllamaChatTransport):
    """Ollama's native ``/api/chat`` with a ``tools`` list (the local agent).

    The same streamed NDJSON exchange as OllamaChatTransport — the same
    cancellation, error and size handling — carrying a whole message history
    (a ToolChatRequest) instead of one system+user pair. ``message.tool_calls``
    in the stream become ``ChatResult.tool_calls``: proposals for the caller to
    check and dispatch, never executed here.
    """

    kind = "ollama"

    def payload(self, req: Any) -> dict[str, Any]:
        if not isinstance(req, ToolChatRequest):
            return super().payload(req)
        options: dict[str, Any] = {"temperature": float(req.temperature),
                                   "num_predict": int(req.max_tokens)}
        if req.num_ctx:
            options["num_ctx"] = int(req.num_ctx)
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [dict(message) for message in req.messages],
            "stream": True,
            "options": options,
        }
        if req.tools:
            payload["tools"] = list(req.tools)
        if req.keep_alive is not None:
            payload["keep_alive"] = req.keep_alive
        return payload


async def probe_ollama(session: Any, root: str, *, timeout: float) -> str | None:
    """Ollama's version if *root* is an Ollama server; None if something else
    answered. Raises ``ConnectionError`` when nothing answered at all.

    On this machine a connect to a closed loopback port hangs rather than
    being refused (see local_models.STARTUP_PROBE_TIMEOUT), so the timeout is
    short, and a server that is down fails here in a second or so instead of
    holding the chat request for its full two-minute allowance.
    """
    aiohttp = _aiohttp()
    try:
        async with session.get(f"{ollama_root(root)}/api/version",
                               timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                return None
            raw = await response.text()
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise ConnectionError(
            f"local server {ollama_root(root)} did not answer within {timeout:.1f}s") from exc
    except aiohttp.ClientError as exc:
        raise ConnectionError(
            f"local server {ollama_root(root)} is not reachable ({type(exc).__name__})") from exc
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return version if isinstance(version, str) and version.strip() else None


# ──────────────────────────────────────────────────────────────────────────────
# ADMISSION
# ──────────────────────────────────────────────────────────────────────────────

INTERACTIVE = "interactive"
BACKGROUND = "background"


@dataclass
class _Waiter:
    background: bool
    since: float
    seq: int
    future: Any


class EndpointGate:
    """A priority semaphore for one provider (or one local server).

    ``limit`` bounds everything; ``background_limit`` bounds background work
    below it, so a conversation turn always finds a slot free even while a
    research run is using the provider hard. Among waiters, interactive ones
    go first; a background waiter older than ``PROMOTE_AFTER_S`` counts as
    interactive, so a stream of conversation cannot starve it forever.

    Thread-aware: the router is shared, and a caller on another thread's loop
    is woken through that loop rather than from this one.
    """

    PROMOTE_AFTER_S = 30.0

    def __init__(self, limit: int = 1, background_limit: int | None = None) -> None:
        self.limit = max(1, int(limit))
        cap = self.limit if background_limit is None else int(background_limit)
        self.background_limit = max(1, min(self.limit, cap))
        self.active = 0
        self.active_background = 0
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()
        self._lock = threading.Lock()

    def _can_take(self, background: bool) -> bool:
        if self.active >= self.limit:
            return False
        return not (background and self.active_background >= self.background_limit)

    def _take(self, background: bool) -> None:
        self.active += 1
        if background:
            self.active_background += 1

    async def acquire(self, cls: str = INTERACTIVE, timeout: float | None = None) -> float:
        """Wait for a slot; returns seconds spent waiting. Raises
        ``TimeoutError`` if none came free within *timeout*."""
        background = cls != INTERACTIVE
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._can_take(background):
                self._take(background)
                return 0.0
            waiter = _Waiter(background, started, next(self._seq), loop.create_future())
            self._waiters.append(waiter)
        try:
            if timeout is None:
                await asyncio.shield(waiter.future)
            else:
                await asyncio.wait_for(asyncio.shield(waiter.future), max(0.0, timeout))
        except BaseException:
            with self._lock:
                granted = waiter.future.done() and not waiter.future.cancelled()
                if not granted:
                    waiter.future.cancel()
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
            if granted:
                self.release(cls)       # the slot arrived as we gave up
            raise
        return time.monotonic() - started

    def release(self, cls: str = INTERACTIVE) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            if cls != INTERACTIVE:
                self.active_background = max(0, self.active_background - 1)
            grants = self._grant_locked()
        for waiter in grants:
            self._resolve(waiter)

    def _grant_locked(self) -> list[_Waiter]:
        now = time.monotonic()
        granted: list[_Waiter] = []
        while True:
            live = [w for w in self._waiters
                    if not w.future.done() and not w.future.get_loop().is_closed()]
            self._waiters = live
            ranked = sorted(live, key=lambda w: (
                w.background and now - w.since < self.PROMOTE_AFTER_S, w.seq))
            chosen = next((w for w in ranked if self._can_take(w.background)), None)
            if chosen is None:
                return granted
            self._waiters.remove(chosen)
            self._take(chosen.background)
            granted.append(chosen)

    def _resolve(self, waiter: _Waiter) -> None:
        future = waiter.future
        cls = BACKGROUND if waiter.background else INTERACTIVE

        def _set() -> None:
            if future.done():
                # The waiter gave up between the grant and this callback.
                self.release(cls)
            else:
                future.set_result(True)

        loop = future.get_loop()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            _set()
        else:
            try:
                loop.call_soon_threadsafe(_set)
            except RuntimeError:
                self.release(cls)       # its loop has closed

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            waiting = [w for w in self._waiters if not w.future.done()]
            return {
                "limit": self.limit,
                "background_limit": self.background_limit,
                "active": self.active,
                "active_background": self.active_background,
                "waiting": len(waiting),
                "waiting_background": sum(1 for w in waiting if w.background),
            }


__all__ = [
    "BACKGROUND", "INTERACTIVE", "ChatRequest", "ChatResult", "EndpointGate",
    "MAX_TOOL_CALLS", "OllamaChatTransport", "OllamaToolTransport", "OpenAIChatTransport",
    "SessionPool", "ToolChatRequest", "auth_headers", "chat_endpoint", "ollama_root",
    "probe_ollama",
]
