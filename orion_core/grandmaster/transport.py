"""Outbound HTTPS only. aiohttp streams; bounded berserk chat on an I/O thread."""
from __future__ import annotations

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from ..chess_lichess import (
    LICHESS_URL,
    LichessAuthError,
    LichessClient,
    LichessError,
    LichessRateLimited,
)
from .lifetime import finish


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9]{8,12}", value):
        raise ValueError("Invalid Lichess identifier")
    return value


class SecureClient(LichessClient):
    def __init__(self, token: str):
        super().__init__(token, LICHESS_URL)
        self._cooldown = 0.0
        self._writes = asyncio.Lock()
        self._chat_lock = asyncio.Lock()
        self._chat_pool = ThreadPoolExecutor(1, thread_name_prefix="lichess-chat")
        self._closed = False

    async def _get_session(self):
        if self._closed:
            raise RuntimeError("Lichess client closed")
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._token}", "User-Agent": "ORION-Grandmaster/1"},
                trust_env=False, read_bufsize=32768,
                connector=aiohttp.TCPConnector(limit=5, limit_per_host=5))
        return self._session

    async def _ready(self):
        if self._closed:
            raise RuntimeError("Lichess client closed")
        await asyncio.sleep(max(0.0, self._cooldown - time.monotonic()))

    async def _check(self, response):
        if response.status in (401, 403):
            raise LichessAuthError(response.status, "Bot authorisation rejected")
        if response.status == 429:
            try:
                retry = min(3600, max(65, int(response.headers.get("Retry-After", "65"))))
            except ValueError:
                retry = 65
            self._cooldown = max(self._cooldown, time.monotonic() + retry)
            raise LichessRateLimited(429, "Request cooldown")
        if response.status >= 300:
            # Do not log remote bodies: they may echo credentials or chat text.
            raise LichessError(response.status, "Request rejected")

    @staticmethod
    def _path(path):
        if not re.fullmatch(r"/api/[A-Za-z0-9_/-]+", path) or "//" in path:
            raise ValueError("Invalid API path")
        return LICHESS_URL + path

    async def _request(self, method, path, data=None):
        import aiohttp
        async with self._writes:
            await self._ready()
            session = await self._get_session()
            async with session.request(method, self._path(path), data=data, allow_redirects=False,
                                       timeout=aiohttp.ClientTimeout(total=12)) as response:
                await self._check(response)
                try:
                    body = await response.content.readexactly(65537)
                except asyncio.IncompleteReadError as exc:
                    body = exc.partial
                if len(body) > 65536:
                    raise LichessError(502, "Oversized response")
                return json.loads(body) if body else {}

    async def _stream(self, path, on_open=None):
        import aiohttp
        await self._ready()
        session = await self._get_session()
        async with session.get(self._path(path), allow_redirects=False,
                               timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=45)) as response:
            await self._check(response)
            if on_open:
                on_open()
            async for line in response.content:
                if len(line) > 65536:
                    raise LichessError(502, "Oversized event")
                if not line.strip():
                    continue
                event = json.loads(line)
                if isinstance(event, dict):
                    yield event

    def _chat_sync(self, game_id, message, room):
        try:
            import berserk
        except ModuleNotFoundError as exc:
            if exc.name != "berserk":
                raise
            raise LichessError(
                503, "Grandmaster chat requires the optional berserk package; "
                "install requirements/chess-grandmaster.txt in ORION's environment") from None

        class FixedSession(berserk.TokenSession):
            def request(self, method, url, **kwargs):
                if url != f"{LICHESS_URL}/api/bot/game/{game_id}/chat":
                    raise ValueError("Unexpected chat endpoint")
                kwargs.update(timeout=(3, 5), allow_redirects=False)
                response = super().request(method, url, **kwargs)
                if 300 <= response.status_code < 400:
                    response.close()
                    raise LichessError(response.status_code, "Chat redirect refused")
                return response

        with FixedSession(self._token) as session:
            session.trust_env = False
            try:
                berserk.Client(session=session).bots.post_message(game_id, message, spectator=room == "spectator")
            except berserk.exceptions.ResponseError as exc:
                status = getattr(exc, "status_code", 500)
                if status == 429:
                    return 429
                raise LichessError(status, "Chat rejected") from None
        return 200

    async def chat(self, game_id, text, room="player"):
        identifier(game_id)
        message = " ".join(str(text).split())[:140]
        async with self._chat_lock:
            await self._ready()
            future = asyncio.get_running_loop().run_in_executor(self._chat_pool, self._chat_sync,
                                                                game_id, message, room)
            try:
                status = await asyncio.shield(future)
            finally:
                # Cancellation cannot release this lock while requests is using the worker.
                if not future.done():
                    _result, interrupted = await finish(future)
                    if interrupted:
                        raise asyncio.CancelledError
            if status == 429:
                self._cooldown = max(self._cooldown, time.monotonic() + 65)
                raise LichessRateLimited(429, "Chat cooldown")

    async def close(self):
        self._closed = True
        await super().close()
        await asyncio.to_thread(self._chat_pool.shutdown, wait=True, cancel_futures=True)
