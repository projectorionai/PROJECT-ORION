"""Authenticated backend calls stop at redirects without forwarding credentials."""

from __future__ import annotations

import aiohttp
import pytest

from orion_core import breach_monitor, web_search_backends
from orion_core.chess_lichess import LichessClient, LichessError


class _Redirect:
    status = 302
    headers = {"Location": "https://attacker.example/collect"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self, **_kwargs):
        return "redirect"


async def test_brave_request_never_forwards_subscription_token(monkeypatch):
    calls = []

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return _Redirect()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    monkeypatch.setattr(web_search_backends, "brave_key", lambda: "private-key")
    assert await web_search_backends.search_brave("query", 2) == []
    assert len(calls) == 1
    assert calls[0][1]["headers"]["X-Subscription-Token"] == "private-key"
    assert calls[0][1]["allow_redirects"] is False


async def test_grounded_google_request_never_forwards_api_key(monkeypatch):
    calls = []

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return _Redirect()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    monkeypatch.setattr(web_search_backends, "gemini_key", lambda: "private-key")
    assert await web_search_backends.search_google("query", 2) == []
    assert len(calls) == 1
    assert calls[0][1]["headers"]["x-goog-api-key"] == "private-key"
    assert calls[0][1]["allow_redirects"] is False


async def test_grounding_link_resolution_only_fetches_google_wrappers(monkeypatch):
    calls = []

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return _Redirect()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    private = "http://127.0.0.1/private"
    ordinary = "https://example.org/article"
    assert await web_search_backends._resolve_redirect(private) == private
    assert await web_search_backends._resolve_redirect(ordinary) == ordinary
    assert calls == []
    wrapper = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
    assert await web_search_backends._resolve_redirect(wrapper) == _Redirect.headers["Location"]
    assert calls == [(wrapper, {"allow_redirects": False,
                                "headers": {"User-Agent": web_search_backends._USER_AGENT}})]


class _Log:
    def __init__(self):
        self.messages = []

    def emit(self, message):
        self.messages.append(message)


class _Bus:
    def __init__(self):
        self.log = _Log()


async def test_breach_requests_do_not_redirect_or_log_request_secrets(monkeypatch):
    calls = []

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return _Redirect()

    monkeypatch.setattr(breach_monitor, "ClientSession", Session)
    bus = _Bus()
    monitor = breach_monitor.BreachMonitor(bus)
    status, data = await monitor._get_json_with_status(
        "https://haveibeenpwned.com/api/v3/breachedaccount/private@example.org",
        headers={"hibp-api-key": "private-key"})
    assert (status, data) == (302, None)
    assert await monitor._get_text("https://api.pwnedpasswords.com/range/ABCDE") is None
    assert len(calls) == 2
    assert all(kwargs["allow_redirects"] is False for _url, kwargs in calls)

    def fail_with_secret(_url, **_kwargs):
        raise RuntimeError("private@example.org private-key")

    monkeypatch.setattr(Session, "get", fail_with_secret)
    assert await monitor._get_json_with_status("https://example.invalid") == (0, None)
    assert "private-key" not in repr(bus.log.messages)
    assert "private@example.org" not in repr(bus.log.messages)


async def test_breach_account_is_encoded_as_one_path_segment(monkeypatch):
    monitor = breach_monitor.BreachMonitor(_Bus())
    monkeypatch.setattr(monitor, "_hibp_api_key", lambda: "private-key")
    urls = []

    async def record(url, **_kwargs):
        urls.append(url)
        return 404, None

    monkeypatch.setattr(monitor, "_get_json_with_status", record)
    await monitor.check_account("name+tag@example.org?x=1")
    assert urls == ["https://haveibeenpwned.com/api/v3/breachedaccount/"
                    "name%2Btag%40example.org%3Fx%3D1?truncateResponse=false"]


async def test_lichess_bot_rejects_redirects_on_request_and_stream(monkeypatch):
    calls = []

    class Session:
        closed = False

        def __init__(self, *, headers):
            assert headers["Authorization"] == "Bearer private-token"

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return _Redirect()

        def get(self, url, **kwargs):
            calls.append(("GET", url, kwargs))
            return _Redirect()

        async def close(self):
            self.closed = True

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    client = LichessClient("private-token")
    try:
        with pytest.raises(LichessError, match="redirect refused"):
            await client.account()
        with pytest.raises(LichessError, match="redirect refused"):
            await anext(client.stream_events())
    finally:
        await client.close()
    assert len(calls) == 2
    assert all(kwargs["allow_redirects"] is False for _method, _url, kwargs in calls)
