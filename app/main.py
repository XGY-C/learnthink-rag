from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.retrievers.file_retriever import FileIndexRetriever
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
    if settings.retriever_mode == "milvus":
        try:
            connect()
            from app.embedding import warmup
            warmup()
            logger.info("learnthink-rag started, mode=milvus, dim=1024")
        except Exception as e:
            logger.warning("learnthink-rag started with warnings, mode=milvus: %s", e)
    else:
        logger.info("learnthink-rag started, mode=jsonl (fallback)")
    yield
    # === SHUTDOWN ===
    if settings.retriever_mode == "milvus":
        disconnect()
    logger.info("learnthink-rag stopped")


app = FastAPI(title="learnthink-rag", version="0.2.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "retriever_mode": settings.retriever_mode}


def _resolve_index_file(course_id: str) -> Path:
    base = settings.kb_base_dir
    index_dir = (base / course_id / "index").resolve()
    return index_dir / settings.index_filename


@app.post("/internal/rag/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    t0 = time.time()

    if settings.retriever_mode == "milvus":
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
    else:
        index_file = _resolve_index_file(req.course_id)
        if not index_file.exists():
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "KB_NOT_READY",
                    "message": "知识库未就绪：索引文件不存在",
                    "course_id": req.course_id,
                    "extra": {"expected_index": str(index_file)},
                },
            )
        retriever = FileIndexRetriever(index_file=index_file, excerpt_max_chars=settings.excerpt_max_chars)
        sources = retriever.retrieve(query=req.query, k=req.k)

    stats = RetrieveStats(
        k=req.k,
        returned=len(sources),
        min_relevance=req.min_relevance,
        min_sources=req.min_sources,
        query_mode=req.query_mode,
        deduped=True,
    )

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "retrieve done course=%s k=%d returned=%d mode=%s elapsed=%.0fms",
        req.course_id, req.k, len(sources), req.query_mode, elapsed_ms,
    )

    return RetrieveResponse(sources=sources, stats=stats)
