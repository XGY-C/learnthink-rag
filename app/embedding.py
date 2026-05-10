from __future__ import annotations

import logging
from pathlib import Path

from app.settings import settings

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    from FlagEmbedding import BGEM3FlagModel

    model_path = settings.embedding_model
    cache_dir = str(Path(settings.embedding_cache_dir).resolve())

    logger.info("Loading embedding model %s (device=%s)...", model_path, settings.embedding_device)
    _model = BGEM3FlagModel(
        model_path,
        use_fp16=settings.embedding_device != "cpu",
        devices=settings.embedding_device,
        cache_dir=cache_dir,
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


def encode_query(query: str) -> list[float]:
    """Encode a single query string into a dense vector (1024-d for BGE-M3)."""
    model = _load_model()
    embeddings = model.encode([query], return_dense=True, return_sparse=False)
    return embeddings["dense_vecs"][0].tolist()


def encode_query_sparse(query: str) -> dict[int, float]:
    """Encode a single query string into a sparse vector (lexical weights)."""
    model = _load_model()
    embeddings = model.encode([query], return_dense=False, return_sparse=True)
    sparse_vec = embeddings["lexical_weights"][0]
    return {int(k): float(v) for k, v in sparse_vec.items()}


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
    model = _load_model()
    embeddings = model.encode(texts, return_dense=True, return_sparse=True)
    dense = [v.tolist() for v in embeddings["dense_vecs"]]
    sparse = [
        {int(k): float(v) for k, v in sv.items()}
        for sv in embeddings["lexical_weights"]
    ]
    return dense, sparse


def get_embedding_dim() -> int:
    model = _load_model()
    return model.model.config.hidden_size
