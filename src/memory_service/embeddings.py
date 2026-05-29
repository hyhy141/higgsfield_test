"""Local embeddings + cross-encoder reranking via fastembed (ONNX, CPU).

Running these locally — instead of calling an embedding API — means the retrieval
path needs no API key and no network at request time. The model weights are baked
into the Docker image at build time (see scripts/prefetch_models.py).

bge-small-en-v1.5: 384-dim, strong quality/size tradeoff for short memory facts.
ms-marco-MiniLM-L-6-v2 cross-encoder: cheap, accurate reranking over ~tens of
candidates; its raw logits are squashed to (0,1) with a sigmoid so they can be
compared against a single relevance threshold.
"""

from __future__ import annotations

import logging
import math
import threading

from .config import get_settings

log = logging.getLogger("memory.embeddings")


class Embedder:
    def __init__(self, embed_model: str, rerank_model: str) -> None:
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        log.info("loading embedding model %s", embed_model)
        self._embed = TextEmbedding(model_name=embed_model)
        log.info("loading reranker model %s", rerank_model)
        self._rerank = TextCrossEncoder(model_name=rerank_model)
        self._has_query_embed = hasattr(self._embed, "query_embed")

    def embed_documents(self, texts: list[str]):
        """Return one numpy float32 vector per input. We hand numpy arrays (not
        lists) to psycopg so pgvector's registered ndarray dumper sends them as
        the `vector` type — a plain list would adapt to double precision[] and
        break the `<=>` operator."""
        if not texts:
            return []
        return list(self._embed.embed(texts))

    def embed_query(self, text: str):
        # bge models prepend a retrieval instruction to queries via query_embed,
        # which measurably improves asymmetric (query→passage) search.
        if self._has_query_embed:
            try:
                return next(iter(self._embed.query_embed([text])))
            except Exception:  # pragma: no cover - fall back to symmetric embed
                pass
        return next(iter(self._embed.embed([text])))

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return a relevance score in (0, 1) for each doc, aligned to input order."""
        if not docs:
            return []
        raw = list(self._rerank.rerank(query, docs))
        return [_sigmoid(x) for x in raw]


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:  # pragma: no cover
        return 0.0 if x < 0 else 1.0


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                s = get_settings()
                _embedder = Embedder(s.embed_model, s.rerank_model)
    return _embedder
