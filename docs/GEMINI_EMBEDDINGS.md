# Gemini text embeddings in ORION

ORION has a `semantic_compare` tool for ranking short, explicitly supplied
texts by meaning. It is separate from chat generation and from the existing
local memory and document indexes. The default comparison runs entirely on
the PC using ORION's local text vectors.

For an explicit cloud comparison, request Gemini embeddings for the exact
texts in question. The tool requires both `cloud=true` and
`confirm_cloud=true` in that call. It never fetches files, memory, clipboard
contents or earlier conversation content. Keep private material local unless
you deliberately supply it for this comparison. Each text is limited to
4,000 characters and each call accepts at most six candidates.

The cloud model may be `gemini-embedding-2` (default) or
`gemini-embedding-001`. Embedding 2 uses Google's sentence similarity prompt
prefix; Embedding 1 uses `SEMANTIC_SIMILARITY` in its request. One model embeds
the query and every candidate in a comparison. ORION does not persist these
vectors or compare vectors across the two model spaces, which are
incompatible. A default cloud request tries Embedding 2 first; if it fails,
ORION discards that partial set and retries every text with Embedding 1. An
explicitly selected model stays pinned to that model. Both attempts share one
HTTPS session and have a 30-second overall deadline. A missing key, timeout,
quota refusal or model error returns a local comparison with a clear reason.
Local scores are less semantically rich
than the cloud models, so the source is always shown in the result.

Example via the Python API:

```python
from orion_core.gemini_embeddings import rank_texts

result = await rank_texts(
    "a quick way to travel by rail",
    ["a train journey", "a recipe for bread"],
    cloud=True,
    consent=True,
    model="gemini-embedding-2",
)
for row in result.rows:
    print(row.position, row.score, row.text)
```

The key is read from ORION's existing Gemini configuration. API responses are
not logged, and provider error bodies are never shown. The capability is for
similarity ranking; it does not answer factual questions or verify claims.

Google's [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings)
documents the `embedContent` endpoint, both model identifiers, task formats,
dimensions and the incompatibility between embedding spaces.
