from __future__ import annotations

import logging
from collections import defaultdict

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
)
from pymilvus.exceptions import CollectionNotExistException, MilvusException

from app.embedding import encode_query, encode_query_sparse, encode_query_hybrid
from app.settings import settings

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "kb_"
EXCERPT_MAX_CHARS = settings.excerpt_max_chars
DIM = 1024  # BGE-M3 dense dimension
_connected: bool = False


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def _collection_name(course_id: str) -> str:
    return f"{COLLECTION_PREFIX}{course_id}".replace("-", "_")


def connect() -> None:
    global _connected
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            if settings.milvus_uri:
                # Milvus Lite embedded mode (zero-dependency local dev)
                connections.connect(alias="default", uri=settings.milvus_uri)
                _connected = True
                logger.info("Milvus Lite connected: %s", settings.milvus_uri)
                return
            else:
                # Connect to Milvus Standalone
                # Note: Database will be auto-created on first use in Milvus 2.4+
                connections.connect(
                    alias="default",
                    host=settings.milvus_host,
                    port=settings.milvus_port,
                    db_name=settings.milvus_db if settings.milvus_db else "default",
                )
                _connected = True
                logger.info(
                    "Milvus connected: %s:%s/%s",
                    settings.milvus_host, settings.milvus_port, 
                    settings.milvus_db if settings.milvus_db else "default"
                )
                return
        except Exception as e:
            _connected = False
            logger.warning(
                "Milvus connection attempt %d/%d failed: %s", 
                attempt, max_retries, e
            )
            if attempt < max_retries:
                import time
                time.sleep(retry_delay)
            else:
                logger.error(
                    "Milvus connection failed after %d attempts: %s. Service will return 503 on requests.", 
                    max_retries, e
                )


def disconnect() -> None:
    global _connected
    if _connected:
        connections.disconnect("default")
        _connected = False
        logger.info("Milvus disconnected.")


def _ensure_connected():
    if not _connected:
        logger.warning("Milvus connection lost or not initialized. Attempting to reconnect...")
        connect()
        if not _connected:
            raise ConnectionError("Milvus not connected after retry")


# ---------------------------------------------------------------------------
# Schema & collection management
# ---------------------------------------------------------------------------
def _make_schema() -> CollectionSchema:
    """Build the collection schema with both dense and sparse vectors."""
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="heading_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="locator", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    return CollectionSchema(fields, description="Collection for course knowledge base")


def collection_exists(course_id: str) -> bool:
    _ensure_connected()
    client_uri = settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}"
    # Add db_name to URI if using Milvus Standalone
    if not settings.milvus_uri and settings.milvus_db:
        client_uri += f"/{settings.milvus_db}"
    client = MilvusClient(uri=client_uri)
    return client.has_collection(_collection_name(course_id))


def get_or_create_collection(course_id: str) -> Collection:
    _ensure_connected()
    name = _collection_name(course_id)
    try:
        col = Collection(name)
        col.load()
        return col
    except (CollectionNotExistException, Exception) as e:
        # Collection doesn't exist or schema not ready, create it
        if "not exist" in str(e).lower() or "schema" in str(e).lower():
            schema = _make_schema()
            col = Collection(name, schema)

            # Dense index
            col.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                },
            )

            # Sparse index
            col.create_index(
                field_name="sparse_vector",
                index_params={
                    "metric_type": "IP",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {"drop_ratio_build": 0.2},
                },
            )

            col.load()
            logger.info("Created collection %s with dense + sparse indexes", name)
            return col
        else:
            raise


# ---------------------------------------------------------------------------
# Excerpt helper
# ---------------------------------------------------------------------------
def _make_excerpt(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Search modes
# ---------------------------------------------------------------------------
def _dense_search(
    col: Collection, query_vec: list[float], k: int, expr: str | None
) -> dict[str, tuple[float, dict]]:
    """Dense-only search. Returns {chunk_id: (score, entity_fields)}."""
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    results = col.search(
        data=[query_vec],
        anns_field="embedding",
        param=search_params,
        limit=k,
        expr=expr,
        output_fields=["doc_id", "doc_title", "source_type", "chunk_id",
                       "text", "locator", "heading_path"],
        timeout=3,
    )
    hits: dict[str, tuple[float, dict]] = {}
    for hits_list in results:
        for hit in hits_list:
            cid = str(hit.entity.get("chunk_id", ""))
            if cid not in hits:
                hits[cid] = (hit.distance, hit.entity)
    return hits


def _sparse_search(
    col: Collection, query_sparse: dict[int, float], k: int, expr: str | None
) -> dict[str, tuple[float, dict]]:
    """Sparse-only search. Returns {chunk_id: (score, entity_fields)}."""
    search_params = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
    results = col.search(
        data=[query_sparse],
        anns_field="sparse_vector",
        param=search_params,
        limit=k,
        expr=expr,
        output_fields=["doc_id", "doc_title", "source_type", "chunk_id",
                       "text", "locator", "heading_path"],
        timeout=3,
    )
    hits: dict[str, tuple[float, dict]] = {}
    for hits_list in results:
        for hit in hits_list:
            cid = str(hit.entity.get("chunk_id", ""))
            if cid not in hits:
                hits[cid] = (hit.distance, hit.entity)
    return hits


def _hybrid_search(
    col: Collection,
    query_dense: list[float],
    query_sparse: dict[int, float],
    k: int,
    expr: str | None,
    alpha: float,
) -> dict[str, tuple[float, dict]]:
    """Hybrid search: weighted combination of dense + sparse.

    alpha = dense weight (0-1); sparse weight = 1 - alpha.
    Each set returned as k*2 to increase candidate pool before fusion.
    """
    fusion_k = max(k * 2, 20)

    dense_hits = _dense_search(col, query_dense, fusion_k, expr)
    sparse_hits = _sparse_search(col, query_sparse, fusion_k, expr)

    # Normalise each score set to [0, 1] via min-max
    def _normalise(scores: list[float]) -> list[float]:
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    all_ids = set(dense_hits.keys()) | set(sparse_hits.keys())
    combined: dict[str, float] = {}

    for cid in all_ids:
        d_score = dense_hits[cid][0] if cid in dense_hits else 0.0
        s_score = sparse_hits[cid][0] if cid in sparse_hits else 0.0
        combined[cid] = alpha * d_score + (1.0 - alpha) * s_score

    # Re-normalise combined scores
    ids, scores = zip(*combined.items()) if combined else ((), ())
    norm_scores = _normalise(list(scores))

    # Sort descending, pick top-k
    ranked = sorted(zip(ids, norm_scores), key=lambda x: x[1], reverse=True)[:k]

    # Reconstruct result dict: pick entity from whichever side had it
    result: dict[str, tuple[float, dict]] = {}
    for cid, score in ranked:
        entity = dense_hits.get(cid)
        result[cid] = (score, entity[1] if entity else sparse_hits[cid][1])

    return result


# ---------------------------------------------------------------------------
# Public retrieve API
# ---------------------------------------------------------------------------
def retrieve(
    course_id: str,
    query: str,
    k: int,
    topic: str | None = None,
    search_mode: str | None = None,
    alpha: float | None = None,
) -> list[dict]:
    """Retrieve top-k sources from Milvus.

    Args:
        course_id: 课程 ID
        query: 自然语言查询
        k: 返回数量
        topic: 可选 topic 过滤
        search_mode: "dense" | "sparse" | "hybrid" (默认 settings.search_mode)
        alpha: dense weight, only for hybrid (默认 settings.sparse_weight → 1-sparse_weight)
    """
    _ensure_connected()

    mode = search_mode or settings.search_mode
    if mode not in ("dense", "sparse", "hybrid"):
        raise ValueError(f"Invalid search_mode: {mode}")

    if alpha is None:
        alpha = 1.0 - settings.sparse_weight  # 默认 dense=0.7
    alpha = max(0.0, min(1.0, alpha))

    col = Collection(_collection_name(course_id))
    col.load()

    # Topic filter expression
    expr = None
    if topic:
        escaped = topic.replace('"', '\\"')
        expr = f'topic == "{escaped}"'

    try:
        logger.info("Starting %s search for course=%s query='%s' k=%d", mode, course_id, query[:80], k)
        
        if mode == "dense":
            query_vec = encode_query(query)
            hits = _dense_search(col, query_vec, k, expr)
            logger.info("Dense search returned %d hits for course=%s", len(hits), course_id)

        elif mode == "sparse":
            query_sparse = encode_query_sparse(query)
            hits = _sparse_search(col, query_sparse, k, expr)
            logger.info("Sparse search returned %d hits for course=%s", len(hits), course_id)

        else:  # hybrid
            query_dense, query_sparse = encode_query_hybrid(query)
            logger.debug("Hybrid search params: alpha=%.2f, dense_dim=%d, sparse_nonzero=%d", 
                        alpha, len(query_dense), len(query_sparse))
            hits = _hybrid_search(col, query_dense, query_sparse, k, expr, alpha)
            logger.info("Hybrid search returned %d hits for course=%s", len(hits), course_id)

    except MilvusException as e:
        if "timeout" in str(e).lower():
            raise TimeoutError(f"Milvus search timeout: {e}")
        raise

    # Dedup & build output
    seen: set[str] = set()
    sources: list[dict] = []

    for chunk_id, (score, entity) in sorted(
        hits.items(), key=lambda x: x[1][0], reverse=True
    ):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        text = str(entity.get("text", ""))
        sources.append({
            "doc_id": str(entity.get("doc_id", "")),
            "doc_title": str(entity.get("doc_title", "")),
            "source_type": entity.get("source_type"),
            "chunk_id": chunk_id,
            "excerpt": _make_excerpt(text, EXCERPT_MAX_CHARS),
            "locator": entity.get("locator"),
            "heading_path": entity.get("heading_path"),
            "relevance": round(float(score), 4),
        })

    logger.info(
        "retrieve course=%s k=%d returned=%d mode=%s alpha=%.2f",
        course_id, k, len(sources), mode, alpha,
    )
    return sources
