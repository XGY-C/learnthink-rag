from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    course_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    k: int = Field(default=200, ge=1, le=200)
    topic: str | None = None

    # ---- 以下阈值字段由调用方（Java RetrieverAgent）消费，RAG 服务不做过滤 ----
    # 保留在 Request 中是为了透传给 Response.stats，方便上游做降级决策。
    # 如果上游不需要透传，可移除这两个字段。
    min_relevance: float = Field(default=0.60, ge=0.0, le=1.0)
    min_sources: int = Field(default=1, ge=0, le=20)

    query_mode: Literal["raw", "templated"] = "raw"

    # 检索模式（v2 新增）
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class SourceItem(BaseModel):
    doc_id: str
    book_title: str = ""
    book_type: str = ""
    chapter_index: int | None = None
    chapter_title: str = ""
    source_type: str | None = None

    chunk_id: str
    excerpt: str

    locator: str | None = None
    heading_path: str | None = None

    # relevance 语义：dense=cosine[0,1], sparse=IP[0,+inf), hybrid=normalised[0,1]
    # 放宽上限以兼容 sparse IP 分数；上游应根据 search_mode 解释分数含义
    relevance: float = Field(ge=0.0)


class RetrieveStats(BaseModel):
    k: int
    returned: int
    min_relevance: float
    min_sources: int
    query_mode: Literal["raw", "templated"]
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    deduped: bool


class RetrieveResponse(BaseModel):
    sources: list[SourceItem]
    stats: RetrieveStats


class IngestRequest(BaseModel):
    """文档摄入请求 - Java端调用"""
    course_id: str = Field(min_length=1, description="课程ID")
    doc_ids: list[str] = Field(default_factory=list, description="需要更新的文档ID列表（不含扩展名），空则同步整个课程")
    oss_prefix: str = Field(default="", description="OSS前缀，为空则使用默认路径")
    full_rebuild: bool = Field(default=False, description="是否全量重建索引")
    async_mode: bool = Field(default=True, description="是否异步执行（推荐true）")
    file_urls: list[str] = Field(default_factory=list, description="文档OSS URL列表，RAG通过HTTP直接下载（无需OSS SDK）")


class IngestResponse(BaseModel):
    """文档摄入响应"""
    status: str
    task_id: str
    course_id: str
    message: str


class TaskProgress(BaseModel):
    """任务进度信息"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0-100
    current_step: str = ""  # 当前步骤描述
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    message: str = ""
    error: str | None = None
    created_at: float = 0
    updated_at: float = 0
