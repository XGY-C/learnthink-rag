from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    course_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=50)
    topic: str | None = None

    # 阈值字段由调用方（Java RetrieverAgent）消费，RAG 服务不做过滤
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

    relevance: float = Field(ge=0.0, le=1.0)


class RetrieveStats(BaseModel):
    k: int
    returned: int
    min_relevance: float
    min_sources: int
    query_mode: Literal["raw", "templated"]
    deduped: bool


class RetrieveResponse(BaseModel):
    sources: list[SourceItem]
    stats: RetrieveStats
