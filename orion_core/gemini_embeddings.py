"""Explicit text comparison with Gemini embeddings and a local fallback.

The cloud path accepts only the text passed to :func:`rank_texts`. It does not
read ORION's memory, files, clipboard or conversation history. A caller must
opt in for each request, so ordinary recall stays on the existing local index.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Awaitable, Callable

from .ingestion import _hash_embed


EMBEDDING_MODELS = ("gemini-embedding-2", "gemini-embedding-001")
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
_MAX_TEXT = 4_000
_MAX_CANDIDATES = 6
_TOTAL_TIMEOUT_S = 30


@dataclass(frozen=True)
class RankedText:
    position: int
    text: str
    score: float


@dataclass(frozen=True)
class Ranking:
    rows: tuple[RankedText, ...]
    source: str
    detail: str = ""


def _normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if not norm or not math.isfinite(norm):
        raise ValueError("The embedding was empty or invalid.")
    return [value / norm for value in values]


def _rank(query: list[float], candidates: list[list[float]], texts: list[str],
          source: str, detail: str = "") -> Ranking:
    if any(len(vector) != len(query) for vector in candidates):
        raise ValueError("The embedding dimensions do not match.")
    rows = tuple(sorted(
        (RankedText(index + 1, text,
                    round(sum(a * b for a, b in zip(query, vector)), 4))
         for index, (text, vector) in enumerate(zip(texts, candidates))),
        key=lambda row: (-row.score, row.position),
    ))
    return Ranking(rows, source, detail)


def _local(query: str, candidates: list[str], detail: str = "") -> Ranking:
    return _rank(_hash_embed(query), [_hash_embed(text) for text in candidates],
                 candidates, "local", detail)


async def _post_embedding(model: str, text: str, key: str, *, session=None) -> list[float]:
    import aiohttp

    if session is None:
        timeout = aiohttp.ClientTimeout(total=12, connect=5)
        async with aiohttp.ClientSession(timeout=timeout) as own_session:
            return await _post_embedding(model, text, key, session=own_session)
    payload: dict[str, object] = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768,
    }
    if model == "gemini-embedding-001":
        payload["taskType"] = "SEMANTIC_SIMILARITY"
    async with session.post(f"{_ENDPOINT}/{model}:embedContent", json=payload,
                            headers={"x-goog-api-key": key},
                            allow_redirects=False) as response:
        if response.status != 200:
            # Provider error bodies can echo request data. Never return or log them.
            raise RuntimeError(f"Google embedding HTTP {response.status}")
        body = await response.json()
    values = body.get("embedding", {}).get("values")
    if not isinstance(values, list) or len(values) != 768:
        raise ValueError("Google returned an unexpected embedding shape.")
    return _normalise([float(value) for value in values])


async def rank_texts(
    query: str,
    candidates: list[str],
    *,
    cloud: bool = False,
    consent: bool = False,
    model: str | None = None,
    api_key: str | None = None,
    embed: Callable[[str, str, str], Awaitable[list[float]]] | None = None,
) -> Ranking:
    """Rank caller-supplied text; cloud use needs explicit consent per call.

    On a missing key, timeout, quota error or model error, the request remains
    useful through ORION's local text vectors. Cloud embeddings are never mixed
    across model versions because their vector spaces are incompatible.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty query is required.")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= _MAX_CANDIDATES:
        raise ValueError(f"Provide 1 to {_MAX_CANDIDATES} candidate texts.")
    if any(not isinstance(text, str) or not text.strip() for text in candidates):
        raise ValueError("Every candidate must contain text.")
    if any(len(text) > _MAX_TEXT for text in [query, *candidates]):
        raise ValueError(f"Each text must have at most {_MAX_TEXT:,} characters.")
    if model is not None and model not in EMBEDDING_MODELS:
        raise ValueError("Choose a supported Gemini embedding model.")
    if not cloud:
        return _local(query, candidates)
    if not consent:
        raise PermissionError("Cloud comparison needs explicit consent for this text.")
    if api_key is None:
        from .web_search_backends import gemini_key

        api_key = gemini_key()
    if not api_key:
        return _local(query, candidates, "No Gemini key; compared locally.")
    async def compare(sender: Callable[[str, str, str], Awaitable[list[float]]]) -> Ranking:
        # A default request can try Embedding 1 after Embedding 2 fails. A
        # caller who selects a model explicitly stays pinned to that model.
        models = (model,) if model else EMBEDDING_MODELS
        last_error: Exception | None = None
        for selected_model in models:
            try:
                # Embedding 2 aggregates multiple parts in one request. Keep
                # each text separate; discard the entire partial set on error.
                vectors = []
                for text in [query, *candidates]:
                    prepared = (f"task: sentence similarity | query: {text}"
                                if selected_model == "gemini-embedding-2" else text)
                    vectors.append(_normalise(await sender(selected_model, prepared, api_key)))
                detail = ("Gemini Embedding 2 was unavailable; used Gemini Embedding 1."
                          if selected_model == "gemini-embedding-001" and model is None
                          else "")
                return _rank(vectors[0], vectors[1:], candidates, selected_model, detail)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    try:
        async with asyncio.timeout(_TOTAL_TIMEOUT_S):
            if embed is not None:
                return await compare(embed)
            import aiohttp

            # One connector/TLS session for the entire comparison, including
            # the consistent whole-set retry on Embedding 1.
            timeout = aiohttp.ClientTimeout(total=12, connect=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async def send(selected_model: str, text: str, key: str) -> list[float]:
                    return await _post_embedding(selected_model, text, key, session=session)

                return await compare(send)
    except Exception as exc:
        # No key, input text, URL or provider response body is reflected into
        # the result. The caller can still see that the cloud route failed.
        if isinstance(exc, TimeoutError):
            reason = "Gemini embedding timed out; compared locally."
        elif isinstance(exc, RuntimeError) and str(exc).startswith("Google embedding HTTP "):
            reason = f"{exc}; compared locally."
        else:
            reason = "Gemini embedding was unavailable; compared locally."
        return _local(query, candidates, reason)
