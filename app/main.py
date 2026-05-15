from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.retrievers.milvus_retriever import (
    collection_exists,
    connect,
    disconnect,
    retrieve as milvus_retrieve,
)
from app.schemas import RetrieveRequest, RetrieveResponse, RetrieveStats
from app.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("learnthink-rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    try:
        connect()
        from app.embedding import warmup
        warmup()
        # Warmup reranker model
        from app.reranker import warmup_reranker
        warmup_reranker()
        logger.info("learnthink-rag started, Milvus mode enabled with reranker")
    except Exception as e:
        logger.error("learnthink-rag startup failed: %s", e)
        raise
    yield
    # === SHUTDOWN ===
    disconnect()
    logger.info("learnthink-rag stopped")


app = FastAPI(title="learnthink-rag", version="0.2.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "mode": "milvus"}


@app.post("/internal/rag/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    t0 = time.time()

    if not collection_exists(req.course_id):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "KB_NOT_READY",
                "message": "知识库未就绪：Milvus Collection 不存在或索引未建立",
                "course_id": req.course_id,
                "extra": {},
            },
        )
    try:
        sources = milvus_retrieve(
            course_id=req.course_id,
            query=req.query,
            k=req.k,
            topic=req.topic,
            search_mode=req.search_mode,
            alpha=1.0 - req.sparse_weight,
        )
        
        # Apply min_relevance filter as per API contract
        if req.min_relevance > 0:
            filtered_sources = [s for s in sources if s['relevance'] >= req.min_relevance]
            if len(filtered_sources) < len(sources):
                logger.info(
                    "Filtered %d sources by min_relevance=%.2f, %d remaining",
                    len(sources), req.min_relevance, len(filtered_sources)
                )
            sources = filtered_sources
    except TimeoutError as e:
        elapsed_ms = (time.time() - t0) * 1000
        logger.error("retrieval timeout course=%s elapsed=%.0fms", req.course_id, elapsed_ms)
        raise HTTPException(
            status_code=504,
            detail={
                "code": "RETRIEVAL_TIMEOUT",
                "message": "检索超时，请稍后重试",
                "course_id": req.course_id,
                "extra": {"timeout_ms": 2000},
            },
        )
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        logger.error("milvus error course=%s: %s", req.course_id, e)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MILVUS_ERROR",
                "message": f"Milvus 检索异常: {str(e)[:200]}",
                "course_id": req.course_id,
                "extra": {"error": str(e)[:200]},
            },
        )

    stats = RetrieveStats(
        k=req.k,
        returned=len(sources),
        min_relevance=req.min_relevance,
        min_sources=req.min_sources,
        query_mode=req.query_mode,
        search_mode=req.search_mode,
        deduped=True,
    )

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "retrieve done course=%s k=%d returned=%d search_mode=%s query_mode=%s elapsed=%.0fms",
        req.course_id, req.k, len(sources), req.search_mode, req.query_mode, elapsed_ms,
    )

    return RetrieveResponse(sources=sources, stats=stats)
