from __future__ import annotations

import logging
import threading
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
from app.reranker import rerank_candidates
from app.settings import settings

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "kb_"
EXCERPT_MAX_CHARS = settings.excerpt_max_chars
DIM = 1024  # BGE-M3 dense dimension
_connected: bool = False
_connection_lock = threading.Lock()  # Protect _connected from race conditions


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def _collection_name(course_id: str) -> str:
    return f"{COLLECTION_PREFIX}{course_id}".replace("-", "_")


def connect() -> None:
    global _connected
    max_retries = 3
    retry_delay = 2  # seconds
    
    with _connection_lock:
        # Check again inside lock to avoid redundant connections
        if _connected:
            return
        
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
    with _connection_lock:
        if _connected:
            connections.disconnect("default")
            _connected = False
            logger.info("Milvus disconnected.")


def _ensure_connected():
    with _connection_lock:
        if not _connected:
            logger.warning("Milvus connection lost or not initialized. Attempting to reconnect...")
    
    # Reconnect outside the lock to avoid holding it during network I/O
    if not _connected:
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


def _normalise_scores(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _hybrid_search(
    col: Collection,
    query_dense: list[float],
    query_sparse: dict[int, float],
    k: int,
    expr: str | None,
    alpha: float,
) -> dict[str, tuple[float, dict]]:
    """Hybrid search: normalise-then-fuse with RRF fallback.

    Two fusion strategies are available:
    - **Linear** (default): min-max normalise each score set to [0,1] *before*
      fusion, then apply alpha weighting.  Missing-hit penalty = -0.3 so that
      chunks appearing in only one recall set are demoted.
    - **RRF**: Reciprocal Rank Fusion.  Rank-based, score-space agnostic,
      robust when score distributions differ wildly.

    Candidate pool is expanded to k*5 (min 40) to ensure sufficient overlap.
    """
    # Expand candidate pool — k*5 ensures enough candidates after fusion
    fusion_k = max(k * 5, 40)

    dense_hits = _dense_search(col, query_dense, fusion_k, expr)
    sparse_hits = _sparse_search(col, query_sparse, fusion_k, expr)

    all_ids = set(dense_hits.keys()) | set(sparse_hits.keys())
    if not all_ids:
        return {}

    # ---- Strategy 1: Normalise-then-fuse (linear combination) ----
    # Step 1: Collect raw scores per hit set
    dense_scores_raw = {cid: hits[0] for cid, hits in dense_hits.items()}
    sparse_scores_raw = {cid: hits[0] for cid, hits in sparse_hits.items()}

    # Step 2: Normalise each score set independently to [0, 1]
    d_norm_map: dict[str, float] = {}
    if dense_scores_raw:
        d_ids, d_vals = zip(*dense_scores_raw.items())
        d_norm_vals = _normalise_scores(list(d_vals))
        d_norm_map = dict(zip(d_ids, d_norm_vals))

    s_norm_map: dict[str, float] = {}
    if sparse_scores_raw:
        s_ids, s_vals = zip(*sparse_scores_raw.items())
        s_norm_vals = _normalise_scores(list(s_vals))
        s_norm_map = dict(zip(s_ids, s_norm_vals))

    # Step 3: Fused score with missing-hit penalty
    MISSING_PENALTY = -0.3
    combined: dict[str, float] = {}
    for cid in all_ids:
        d_score = d_norm_map.get(cid, MISSING_PENALTY)
        s_score = s_norm_map.get(cid, MISSING_PENALTY)
        combined[cid] = alpha * d_score + (1.0 - alpha) * s_score

    # Step 4: Re-normalise final scores to [0, 1]
    ids_list, scores_list = zip(*combined.items())
    norm_scores = _normalise_scores(list(scores_list))
    final_scores = dict(zip(ids_list, norm_scores))

    # ---- Strategy 2: RRF (Reciprocal Rank Fusion) as validation ----
    # RRF is rank-based and immune to score-scale differences.
    RRF_K_CONST = 60  # standard constant from original RRF paper
    dense_ranked = sorted(dense_scores_raw.keys(), key=lambda x: dense_scores_raw[x], reverse=True)
    sparse_ranked = sorted(sparse_scores_raw.keys(), key=lambda x: sparse_scores_raw[x], reverse=True)
    rrf_scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ranked, start=1):
        rrf_scores[cid] += alpha / (RRF_K_CONST + rank)
    for rank, cid in enumerate(sparse_ranked, start=1):
        rrf_scores[cid] += (1.0 - alpha) / (RRF_K_CONST + rank)

    # Normalise RRF scores to [0, 1]
    if rrf_scores:
        rrf_ids, rrf_vals = zip(*rrf_scores.items())
        rrf_norm = _normalise_scores(list(rrf_vals))
        rrf_final = dict(zip(rrf_ids, rrf_norm))
    else:
        rrf_final = {}

    # ---- Blend: average linear + RRF for robustness ----
    BLEND_RRF_WEIGHT = 0.3  # 30% RRF, 70% linear
    blended: dict[str, float] = {}
    for cid in all_ids:
        linear_score = final_scores.get(cid, 0.0)
        rrf_score = rrf_final.get(cid, 0.0)
        blended[cid] = (1.0 - BLEND_RRF_WEIGHT) * linear_score + BLEND_RRF_WEIGHT * rrf_score

    # Sort descending, pick top-k
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:k]

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
    use_reranker: bool = True,
    rerank_top_k_multiplier: int = 3,
) -> list[dict]:
    """Retrieve top-k sources from Milvus with optional re-ranking.

    Args:
        course_id: 课程 ID
        query: 自然语言查询
        k: 返回数量
        topic: 可选 topic 过滤
        search_mode: "dense" | "sparse" | "hybrid" (默认 settings.search_mode)
        alpha: dense weight, only for hybrid (默认 settings.sparse_weight → 1-sparse_weight)
        use_reranker: 是否启用重排序 (默认 True)
        rerank_top_k_multiplier: 召回候选集倍数，用于reranker精排 (默认 3倍)
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

    # Topic filter expression - use exact match to avoid SQL injection and granularity issues
    expr = None
    if topic:
        # Escape special characters for Milvus expression
        # Only allow alphanumeric, Chinese characters, and common punctuation
        import re as _re
        if not _re.match(r'^[\w\u4e00-\u9fff\s\-_.]+$', topic):
            logger.warning("Invalid topic format: %s. Skipping topic filter.", topic)
        else:
            # Use exact match instead of LIKE to prevent injection and ensure precision
            escaped = topic.replace('\\', '\\\\').replace('"', '\\"')
            expr = f'topic == "{escaped}"'
            logger.debug("Topic filter applied: %s", expr)

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

    # Apply Cross-Encoder re-ranking if enabled and we have candidates
    if use_reranker and sources:
        # Retrieve more candidates for re-ranking
        rerank_k = min(k * rerank_top_k_multiplier, len(sources) + 10)
        if rerank_k > k:
            logger.info(
                "Re-ranking enabled: retrieving %d candidates for top-%d final results",
                rerank_k, k
            )
            # Re-run retrieval with larger k for reranking
            if mode == "dense":
                query_vec = encode_query(query)
                hits_rr = _dense_search(col, query_vec, rerank_k, expr)
            elif mode == "sparse":
                query_sparse = encode_query_sparse(query)
                hits_rr = _sparse_search(col, query_sparse, rerank_k, expr)
            else:  # hybrid
                query_dense, query_sparse = encode_query_hybrid(query)
                hits_rr = _hybrid_search(col, query_dense, query_sparse, rerank_k, expr, alpha)
            
            # Rebuild sources with expanded candidate set
            sources_rr = []
            seen_rr: set[str] = set()
            for chunk_id, (score, entity) in sorted(
                hits_rr.items(), key=lambda x: x[1][0], reverse=True
            ):
                if chunk_id in seen_rr:
                    continue
                seen_rr.add(chunk_id)
                text = str(entity.get("text", ""))
                sources_rr.append({
                    "doc_id": str(entity.get("doc_id", "")),
                    "doc_title": str(entity.get("doc_title", "")),
                    "source_type": entity.get("source_type"),
                    "chunk_id": chunk_id,
                    "excerpt": _make_excerpt(text, EXCERPT_MAX_CHARS),
                    "locator": entity.get("locator"),
                    "heading_path": entity.get("heading_path"),
                    "relevance": round(float(score), 4),
                })
            
            # Re-rank the expanded candidate set
            sources = rerank_candidates(query, sources_rr, top_k=k)
        else:
            # Not enough extra candidates, just re-rank what we have
            sources = rerank_candidates(query, sources, top_k=k)
    
    # Apply metadata-aware post-ranking to boost lecture materials
    if sources:
        from app.metadata_reranker import apply_metadata_reranking
        sources = apply_metadata_reranking(sources, query, top_k=k, blend_weight=0.7)

    logger.info(
        "retrieve course=%s k=%d returned=%d mode=%s alpha=%.2f reranker=%s",
        course_id, k, len(sources), mode, alpha, use_reranker,
    )
    return sources
