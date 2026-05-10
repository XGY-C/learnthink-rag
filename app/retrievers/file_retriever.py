from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def _tokenize(query: str) -> list[str]:
    # MVP：用空白与常见标点切分；调用方示例 query 通常带空格
    raw = re.split(r"[\s,，。;；:：\n\t]+", query.strip())
    return [t for t in raw if t]


def _score_text(text: str, tokens: list[str]) -> int:
    # 简易占位评分：统计 token 子串命中次数（上限裁剪，避免超长文本过分加分）
    score = 0
    for token in tokens:
        if not token:
            continue
        occurrences = text.count(token)
        score += min(3, occurrences)
    return score


def _make_excerpt(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_type: str | None
    text: str
    locator: str | None
    heading_path: str | None


class FileIndexRetriever:
    def __init__(self, index_file: Path, excerpt_max_chars: int) -> None:
        self._index_file = index_file
        self._excerpt_max_chars = excerpt_max_chars

    @property
    def index_file(self) -> Path:
        return self._index_file

    def load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        with self._index_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=str(obj.get("chunk_id", "")),
                        doc_id=str(obj.get("doc_id", "")),
                        doc_title=str(obj.get("doc_title", "")),
                        source_type=obj.get("source_type"),
                        text=str(obj.get("text", "")),
                        locator=obj.get("locator"),
                        heading_path=obj.get("heading_path") or obj.get("section_title"),
                    )
                )
        return [c for c in chunks if c.chunk_id and c.doc_id and c.doc_title and c.text]

    def retrieve(self, query: str, k: int) -> list[dict]:
        tokens = _tokenize(query)
        chunks = self.load_chunks()

        scored: list[tuple[int, Chunk]] = []
        for chunk in chunks:
            score = _score_text(chunk.text, tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        max_score = top[0][0] if top else 0
        results: list[dict] = []
        seen_chunk_ids: set[str] = set()

        for score, chunk in top:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)

            relevance = 0.0
            if max_score > 0:
                relevance = min(1.0, score / max_score)

            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "source_type": chunk.source_type,
                    "chunk_id": chunk.chunk_id,
                    "excerpt": _make_excerpt(chunk.text, self._excerpt_max_chars),
                    "locator": chunk.locator,
                    "heading_path": chunk.heading_path,
                    "relevance": relevance,
                }
            )

        return results
