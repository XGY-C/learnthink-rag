from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path

from app.settings import settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _load_model():
    """Load the BGE-M3 model with thread-safe double-checked locking.

    Prevents race conditions in multi-threaded environments (e.g. uvicorn
    thread pool) where two threads could simultaneously trigger model loading,
    resulting in duplicate 2 GB model copies in memory.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        from FlagEmbedding import BGEM3FlagModel

        # Use local model path to avoid network issues
        model_path = str(Path(settings.embedding_cache_dir).resolve() / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181")
        
        logger.info("Loading embedding model from local path: %s (device=%s)...", model_path, settings.embedding_device)
        _model = BGEM3FlagModel(
            model_path,
            use_fp16=settings.embedding_device != "cpu",
            devices=settings.embedding_device,
        )
        logger.info("Embedding model loaded.")
        return _model


def warmup() -> None:
    try:
        model = _load_model()
        _ = model.encode(["warmup"], return_dense=True, return_sparse=True)
        dim = model.model.config.hidden_size
        logger.info("Embedding model ready, dense_dim=%d", dim)
    except Exception as e:
        logger.warning("Embedding model warmup failed: %s", e)


# ---------------------------------------------------------------------------
# Query encoding with LRU cache
# ---------------------------------------------------------------------------
# BGE-M3 encode on CPU ≈ 200ms/query.  For high-frequency identical queries
# (e.g. students repeatedly asking the same question), caching avoids
# redundant computation.  maxsize=256 balances memory vs. hit rate.
@lru_cache(maxsize=256)
def encode_query(query: str) -> list[float]:
    """Encode a single query string into a dense vector (1024-d for BGE-M3)."""
    model = _load_model()
    embeddings = model.encode([query], return_dense=True, return_sparse=False)
    return embeddings["dense_vecs"][0].tolist()


@lru_cache(maxsize=256)
def encode_query_sparse(query: str) -> dict[int, float]:
    """Encode a single query string into a sparse vector (lexical weights)."""
    model = _load_model()
    embeddings = model.encode([query], return_dense=False, return_sparse=True)
    sparse_vec = embeddings["lexical_weights"][0]
    return {int(k): float(v) for k, v in sparse_vec.items()}


@lru_cache(maxsize=256)
def encode_query_hybrid(query: str) -> tuple[list[float], dict[int, float]]:
    """Encode query into both dense and sparse vectors in one pass."""
    model = _load_model()
    embeddings = model.encode([query], return_dense=True, return_sparse=True)
    dense = embeddings["dense_vecs"][0].tolist()
    sparse = {int(k): float(v) for k, v in embeddings["lexical_weights"][0].items()}
    return dense, sparse


def encode_documents(texts: list[str]) -> list[list[float]]:
    """Encode document texts into dense vectors."""
    model = _load_model()
    embeddings = model.encode(texts, return_dense=True, return_sparse=False)
    return [v.tolist() for v in embeddings["dense_vecs"]]


def encode_documents_with_sparse(
    texts: list[str],
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Encode document texts into dense + sparse vectors in one pass.

    Returns (dense_vecs, sparse_vecs) where:
      dense_vecs[i]  = [f32, ...]  (1024-d)
      sparse_vecs[i] = {token_id: weight}
    """
    logger.info("Encoding %d texts...", len(texts))
    model = _load_model()
    logger.info("Model loaded, starting encoding...")
    
    # Use batch_size parameter to optimize encoding speed
    # This helps the tokenizer use __call__ method internally
    embeddings = model.encode(
        texts, 
        return_dense=True, 
        return_sparse=True,
        batch_size=len(texts)  # Process all texts in one batch
    )
    
    logger.info("Encoding complete, converting to lists...")
    dense = [v.tolist() for v in embeddings["dense_vecs"]]
    sparse = [
        {int(k): float(v) for k, v in sv.items()}
        for sv in embeddings["lexical_weights"]
    ]
    logger.info("Conversion complete, returning results...")
    return dense, sparse


def get_embedding_dim() -> int:
    model = _load_model()
    return model.model.config.hidden_size


def clear_query_cache() -> None:
    """Clear all LRU caches for query encoding functions.

    Useful for testing or when memory pressure is high.
    """
    encode_query.cache_clear()
    encode_query_sparse.cache_clear()
    encode_query_hybrid.cache_clear()
    logger.info("Query encoding caches cleared.")
