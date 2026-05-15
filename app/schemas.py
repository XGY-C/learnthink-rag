from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    course_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=50)
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
    doc_title: str
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
