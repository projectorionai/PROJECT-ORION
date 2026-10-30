from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from pathlib import Path

from .async_utils import complete_before_cancel


class SemanticMemory:
    """One local Chroma owner; synchronous embedding runs off the event loop.

    encode(texts) must return normalised vectors. Supply ORION's existing
    semantic.ENCODER.encode after explicitly warming that encoder.
    """

    def __init__(self, directory: Path, encode, embedding_version: str):
        import chromadb
        from chromadb.config import Settings

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", embedding_version):
            raise ValueError(
                "Embedding version must be a short collection-safe identifier"
            )

        self.encode = encode
        self.lock = threading.Lock()
        self.client = chromadb.PersistentClient(
            path=str(directory),
            settings=Settings(anonymized_telemetry=False),
        )
        # Change the collection name when the encoder or pooling changes.
        self.collection = self.client.get_or_create_collection(
            name=f"orion_v2_{embedding_version}",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def _add(self, memory_id, owner, text, source, kind):
        if not memory_id or not owner or not source or not kind:
            raise ValueError("Memory identity, owner and provenance are required")
        if not text.strip() or len(text) > 2000:
            raise ValueError(
                "Store a short, curated fact or episode, not a raw transcript"
            )
        with self.lock:
            vectors = self._vectors([text])
            # Collection IDs are global: metadata filtering alone cannot stop
            # another principal upserting the same ID. Namespace the identity.
            identity = hashlib.sha256(
                json.dumps(
                    [owner, memory_id],
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            self.collection.upsert(
                ids=[identity],
                documents=[text],
                embeddings=vectors.tolist(),
                metadatas=[
                    {
                        "owner": owner,
                        "source": source,
                        "kind": kind,
                        "memory_id": memory_id,
                        "created_at": time.time(),
                        "active": True,
                    }
                ],
            )

    async def remember(
        self, memory_id: str, owner: str, text: str, source: str, kind: str = "episode"
    ):
        # owner/source/kind come from trusted ingestion, never from model output.
        await complete_before_cancel(
            asyncio.to_thread(self._add, memory_id, owner, text, source, kind)
        )

    def _vectors(self, texts):
        import numpy as np

        vectors = self.encode(texts)
        if vectors is None:
            raise RuntimeError("Embedding model is not ready")
        vectors = np.asarray(vectors, dtype=np.float32)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(texts)
            or not vectors.shape[1]
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid embedding shape or non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Embeddings must have finite, non-zero length")
        return vectors / norms

    def _search(self, owner, query, k):
        if not owner or not query.strip():
            raise ValueError("Owner and a non-empty query are required")
        with self.lock:
            count = self.collection.count()
            if not count:
                return []
            vectors = self._vectors([query])
            result = self.collection.query(
                query_embeddings=vectors.tolist(),
                n_results=min(k, count),
                where={"$and": [{"owner": owner}, {"active": True}]},
                include=["documents", "metadatas", "distances"],
            )
            return [
                {
                    "id": meta["memory_id"],
                    "text": doc,
                    "metadata": meta,
                    "distance": distance,
                }
                for mid, doc, meta, distance in zip(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0],
                )
            ]

    async def recall(self, owner: str, query: str, k: int = 4):
        return await complete_before_cancel(
            asyncio.to_thread(self._search, owner, query, max(1, min(k, 8)))
        )
