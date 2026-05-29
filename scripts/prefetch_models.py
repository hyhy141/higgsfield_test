"""Download and cache the fastembed models at image-build time.

Running this during `docker build` bakes the embedding + reranker ONNX weights
into an image layer, so the running container needs no network access for
retrieval. The model names must match those in `memory_service.config`.
"""

from __future__ import annotations

import os
import sys

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


def main() -> int:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    print(f"[prefetch] embedding model: {EMBED_MODEL}", flush=True)
    emb = TextEmbedding(model_name=EMBED_MODEL)
    # Force a real forward pass so the weights are materialised and cached.
    vec = next(iter(emb.embed(["warmup sentence"])))
    print(f"[prefetch]   embedding dim = {len(vec)}", flush=True)

    print(f"[prefetch] reranker model: {RERANK_MODEL}", flush=True)
    rr = TextCrossEncoder(model_name=RERANK_MODEL)
    scores = list(rr.rerank("warmup query", ["candidate a", "candidate b"]))
    print(f"[prefetch]   reranker scores = {scores}", flush=True)

    print("[prefetch] done — models cached.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
