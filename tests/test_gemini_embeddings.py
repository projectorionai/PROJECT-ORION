"""Cloud text comparison is explicit, bounded and safe to fall back locally."""

from __future__ import annotations

import asyncio
import math

import pytest

from orion_core import gemini_embeddings as embeddings


async def test_local_is_the_default_and_never_calls_the_provider():
    async def forbidden(*_args):
        raise AssertionError("The cloud must not be called")

    result = await embeddings.rank_texts(
        "red apple", ["blue train", "red apple"], embed=forbidden)
    assert result.source == "local"
    assert result.rows[0].position == 2


async def test_cloud_needs_per_call_consent():
    with pytest.raises(PermissionError, match="explicit consent"):
        await embeddings.rank_texts("alpha", ["beta"], cloud=True, api_key="fake")


async def test_embedding_two_uses_symmetric_prefix_and_one_model():
    calls = []

    async def fake_embed(model, text, key):
        calls.append((model, text, key))
        return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

    result = await embeddings.rank_texts(
        "cat", ["dog", "cat"], cloud=True, consent=True,
        api_key="secret", embed=fake_embed)
    assert result.source == "gemini-embedding-2"
    assert [row.position for row in result.rows] == [2, 1]
    assert len(calls) == 3
    assert all(model == "gemini-embedding-2" for model, _, _ in calls)
    assert [text for _, text, _ in calls] == [
        "task: sentence similarity | query: cat",
        "task: sentence similarity | query: dog",
        "task: sentence similarity | query: cat",
    ]


async def test_embedding_one_uses_unprefixed_text():
    async def fake_embed(model, text, key):
        assert model == "gemini-embedding-001"
        assert text in {"question", "answer"}
        return [3.0, 4.0]

    result = await embeddings.rank_texts(
        "question", ["answer"], cloud=True, consent=True,
        model="gemini-embedding-001", api_key="secret", embed=fake_embed)
    assert result.rows[0].score == 1.0


async def test_default_fallback_restarts_the_whole_set_on_embedding_one():
    calls = []

    async def fake_embed(model, text, _key):
        calls.append((model, text))
        if model == "gemini-embedding-2" and "second" in text:
            raise RuntimeError("Google embedding HTTP 429")
        return [1.0, 0.0] if "first" in text else [0.0, 1.0]

    result = await embeddings.rank_texts(
        "query", ["first", "second"], cloud=True, consent=True,
        api_key="key", embed=fake_embed)
    assert result.source == "gemini-embedding-001"
    assert "Embedding 1" in result.detail
    assert [model for model, _ in calls] == [
        "gemini-embedding-2", "gemini-embedding-2", "gemini-embedding-2",
        "gemini-embedding-001", "gemini-embedding-001", "gemini-embedding-001",
    ]
    assert [text for model, text in calls if model == "gemini-embedding-001"] == [
        "query", "first", "second"]


async def test_explicit_embedding_two_never_switches_models():
    calls = []

    async def unavailable(model, _text, _key):
        calls.append(model)
        raise RuntimeError("Google embedding HTTP 503")

    result = await embeddings.rank_texts(
        "query", ["candidate"], cloud=True, consent=True,
        model="gemini-embedding-2", api_key="key", embed=unavailable)
    assert result.source == "local"
    assert calls == ["gemini-embedding-2"]


async def test_overall_deadline_bounds_both_model_attempts(monkeypatch):
    monkeypatch.setattr(embeddings, "_TOTAL_TIMEOUT_S", 0.02)
    calls = []

    async def very_slow(model, _text, _key):
        calls.append(model)
        await asyncio.sleep(1)
        return [1.0, 0.0]

    start = asyncio.get_running_loop().time()
    result = await embeddings.rank_texts(
        "query", ["candidate"], cloud=True, consent=True,
        api_key="key", embed=very_slow)
    elapsed = asyncio.get_running_loop().time() - start
    assert result.source == "local"
    assert "timed out" in result.detail
    assert elapsed < 0.3


async def test_quota_failure_falls_back_without_echoing_secrets():
    async def fake_embed(_model, _text, _key):
        raise RuntimeError("Google embedding HTTP 429")

    result = await embeddings.rank_texts(
        "private query", ["private item"], cloud=True, consent=True,
        api_key="secret", embed=fake_embed)
    assert result.source == "local"
    assert "429" in result.detail
    assert "secret" not in result.detail
    assert "private" not in result.detail


async def test_input_bounds_and_model_whitelist():
    with pytest.raises(ValueError, match="1 to 6"):
        await embeddings.rank_texts("query", [])
    with pytest.raises(ValueError, match="supported"):
        await embeddings.rank_texts("query", ["candidate"], model="gemini-2.5-flash")
    with pytest.raises(ValueError, match="4,000"):
        await embeddings.rank_texts("x" * 4001, ["candidate"])


def test_normalisation_rejects_bad_vectors():
    assert math.isclose(sum(value ** 2 for value in embeddings._normalise([3, 4])), 1.0)
    with pytest.raises(ValueError):
        embeddings._normalise([0.0, 0.0])


async def test_rest_request_uses_embedding_endpoint_and_one_text(monkeypatch):
    import aiohttp

    captured = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {"embedding": {"values": [1.0] + [0.0] * 767}}

    class Session:
        def __init__(self, *, timeout):
            assert timeout.total == 12

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, *, json, headers, allow_redirects):
            captured.append((url, json, headers, allow_redirects))
            return Response()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    for model in embeddings.EMBEDDING_MODELS:
        result = await embeddings._post_embedding(model, "supplied", "key")
        assert len(result) == 768
    assert captured[0][0].endswith("/gemini-embedding-2:embedContent")
    assert captured[0][1]["content"] == {"parts": [{"text": "supplied"}]}
    assert "taskType" not in captured[0][1]
    assert captured[1][1]["taskType"] == "SEMANTIC_SIMILARITY"
    assert all(item[2] == {"x-goog-api-key": "key"} for item in captured)
    assert all(item[3] is False for item in captured)


async def test_cloud_ranking_reuses_one_https_session(monkeypatch):
    import aiohttp

    sessions = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {"embedding": {"values": [1.0] + [0.0] * 767}}

    class Session:
        def __init__(self, *, timeout):
            assert timeout.total == 12
            self.calls = 0
            sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, _url, *, json, headers, allow_redirects):
            assert headers == {"x-goog-api-key": "key"}
            assert allow_redirects is False
            assert len(json["content"]["parts"]) == 1
            self.calls += 1
            return Response()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    result = await embeddings.rank_texts(
        "query", ["first", "second"], cloud=True, consent=True,
        api_key="key")
    assert result.source == "gemini-embedding-2"
    assert len(sessions) == 1
    assert sessions[0].calls == 3


async def test_dispatcher_tool_has_a_schema_and_honours_local_default(monkeypatch):
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    assert sum(decl["name"] == "semantic_compare" for decl in TOOL_DECLARATIONS) == 1
    result = await KnowledgeDispatchMixin().semantic_compare(
        {"query": "red apple", "candidates": ["red apple", "blue sky"]})
    assert result.ok
    assert result.text.startswith("Compared with local embeddings.")
    refused = await KnowledgeDispatchMixin().semantic_compare(
        {"query": "red apple", "candidates": ["red apple"], "cloud": True})
    assert not refused.ok
    assert "explicit consent" in refused.text
