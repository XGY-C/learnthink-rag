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
from app.schemas import RetrieveRequest, RetrieveResponse, RetrieveStats, IngestRequest, IngestResponse, TaskProgress
from app.settings import settings

from app.log_config import setup_logger

logger = setup_logger("learnthink-rag")


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
    logger.info("Shutdown initiated, waiting for running tasks to complete...")
    
    # 检查是否有正在运行的任务
    from app.task_manager import task_manager, TaskStatus
    running_tasks = [
        t for t in task_manager.list_tasks(limit=100)
        if t['status'] == TaskStatus.RUNNING
    ]
    
    if running_tasks:
        logger.warning(f"There are {len(running_tasks)} running tasks. They will continue in background.")
        for task in running_tasks:
            logger.info(f"  - Task {task['task_id']}: {task['current_step']} ({task['progress']}%)")
        logger.info("Note: Non-daemon threads will complete their work even after server stops.")
    else:
        logger.info("No running tasks. Safe to shutdown.")
    
    disconnect()
    logger.info("learnthink-rag stopped")


app = FastAPI(title="learnthink-rag", version="0.3.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "mode": "milvus"}


@app.post("/admin/ingest-document", response_model=IngestResponse)
def ingest_document(req: IngestRequest) -> IngestResponse:
    """
    Java端调用此接口：上传MD到OSS后，通知RAG服务下载并构建索引
    
    支持同步和异步两种模式：
    - async_mode=true（推荐）：立即返回task_id，通过 /admin/task/{task_id} 查询进度
    - async_mode=false：阻塞等待完成，直接返回结果（适合小文件）
    """
    logger.info(
        "[Ingest] Received request: course_id=%s, doc_ids=%s, full_rebuild=%s, async=%s",
        req.course_id, req.doc_ids, req.full_rebuild, req.async_mode
    )
    
    if req.async_mode:
        # 异步模式：立即返回task_id
        from app.task_executor import start_ingest_task
        
        task_id = start_ingest_task(
            course_id=req.course_id,
            doc_ids=req.doc_ids,
            full_rebuild=req.full_rebuild
        )
        
        return IngestResponse(
            status="accepted",
            task_id=task_id,
            course_id=req.course_id,
            message=f"任务已提交，使用 task_id={task_id} 查询进度"
        )
    else:
        # 同步模式：阻塞等待完成
        import time
        from app.task_executor import execute_ingest_task
        from app.task_manager import task_manager, TaskStatus
        
        task_id = task_manager.create_task(req.course_id)
        
        try:
            execute_ingest_task(
                task_id=task_id,
                course_id=req.course_id,
                doc_ids=req.doc_ids,
                full_rebuild=req.full_rebuild
            )
            
            # 获取最终结果
            result = task_manager.get_task(task_id)
            
            if result["status"] == TaskStatus.COMPLETED:
                return IngestResponse(
                    status="completed",
                    task_id=task_id,
                    course_id=req.course_id,
                    message=result["message"]
                )
            else:
                raise Exception(result.get("error", "Unknown error"))
        
        except Exception as e:
            logger.error(f"[Ingest] Sync mode failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Ingest failed: {str(e)[:200]}"
            )


@app.get("/admin/task/{task_id}", response_model=TaskProgress)
def get_task_progress(task_id: str) -> dict:
    """
    查询任务进度（Java端轮询此接口获取实时进度）
    
    返回示例：
    {
        "task_id": "a1b2c3d4",
        "status": "running",
        "progress": 45.5,
        "current_step": "正在构建向量索引...",
        "total_files": 10,
        "processed_files": 4,
        "total_chunks": 0,
        "message": "已成功下载4个文件，正在处理第5个",
        "error": null,
        "created_at": 1234567890.0,
        "updated_at": 1234567895.0
    }
    """
    from app.task_manager import task_manager
    
    result = task_manager.get_task(task_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return result


@app.get("/admin/tasks")
def list_tasks(course_id: str | None = None, limit: int = 10) -> list:
    """
    列出任务历史（可选按课程过滤）
    """
    from app.task_manager import task_manager
    return task_manager.list_tasks(course_id=course_id, limit=limit)


@app.get("/admin/stats")
def get_stats() -> dict:
    """
    获取任务统计信息（轻量级监控）
    
    返回示例：
    {
        "total": 50,
        "pending": 2,
        "running": 3,
        "completed": 43,
        "failed": 2,
        "avg_duration_seconds": 45.5
    }
    """
    from app.task_manager import task_manager
    return task_manager.get_stats()


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
