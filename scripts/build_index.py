from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunking: prefer splitting at paragraph boundaries."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            if len(para) > chunk_size:
                # Long paragraph: fall back to sliding window
                step = max(1, chunk_size - overlap)
                for start in range(0, len(para), step):
                    end = min(len(para), start + chunk_size)
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if len(c) >= 80]


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
def _hash_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _detect_source_type(file_path: Path) -> str:
    name = file_path.name.lower()
    if any(kw in name for kw in ["讲义", "lecture", "chapter", "章"]):
        return "讲义"
    if any(kw in name for kw in ["题", "exercise", "quiz", "练习"]):
        return "题库"
    if any(kw in name for kw in ["术语", "glossary", "词汇"]):
        return "术语"
    if any(kw in name for kw in ["阅读", "reading", "paper", "论文", "文献"]):
        return "阅读"
    return "讲义"


def _extract_heading_path(text: str) -> str:
    """Extract a heading path from the first few Markdown headings."""
    headings: list[str] = []
    for line in text.split("\n")[:30]:
        m = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Keep only top 3 levels, build breadcrumb
            while len(headings) >= level:
                headings.pop()
            headings.append(title)
            if len(headings) >= 3:
                break
    return "/".join(headings) if headings else ""


def _extract_topics(text: str) -> list[str]:
    """Extract potential topic keywords from early content + headings.

    Strategy: look for bold terms, code spans, and heading titles in the first
    2000 chars.  Returns a deduped list (max 8 topics).
    """
    topics: list[str] = []
    head = text[:2000]

    # Bold terms: **深度学习**, **CNN**
    for m in re.finditer(r"\*\*(.{2,30}?)\*\*", head):
        t = m.group(1).strip()
        if t and not re.search(r"[\u4e00-\u9fff]", t):
            continue  # require at least one CJK character to filter noise
        if t and len(t) <= 20:
            topics.append(t)

    # Markdown headings
    for m in re.finditer(r"^#{1,4}\s+(.+)$", head, re.MULTILINE):
        t = m.group(1).strip()
        if t and len(t) <= 30:
            topics.append(t)

    # Dedup preserving order
    seen = set()
    unique: list[str] = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:8]


def _assign_chunk_topic(chunk: str, doc_topics: list[str]) -> str:
    """Assign a single best topic to a chunk based on keyword overlap."""
    chunk_lower = chunk.lower()
    for topic in doc_topics:
        if topic.lower() in chunk_lower:
            return topic
    return doc_topics[0] if doc_topics else ""


# ---------------------------------------------------------------------------
# JSONL builder (legacy / dry-run)
# ---------------------------------------------------------------------------
def build_jsonl(kb_base_dir: Path, course_id: str, chunk_size: int, overlap: int) -> Path:
    """Build JSONL index (legacy / dry-run mode). Returns output file path."""
    course_dir = kb_base_dir / course_id
    processed_dir = course_dir / "processed"
    index_dir = course_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    files = list(processed_dir.rglob("*.md")) + list(processed_dir.rglob("*.txt"))
    out_file = index_dir / "chunks.jsonl"

    with out_file.open("w", encoding="utf-8") as out:
        if not files:
            placeholder = {
                "chunk_id": "placeholder#0",
                "doc_id": "placeholder",
                "doc_title": "占位材料（请在 kb/<course_id>/processed/ 放入文档后重建索引）",
                "source_type": "讲义",
                "text": "这是一个占位 chunk，用于联调 /internal/rag/retrieve。",
                "locator": "",
                "heading_path": "",
                "topic": "",
            }
            out.write(json.dumps(placeholder, ensure_ascii=False) + "\n")
            logger.info("Wrote placeholder index: %s", out_file)
            return out_file

        chunk_id_counter = 0
        for file_path in sorted(files):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            source_type = _detect_source_type(file_path)
            doc_id = file_path.stem
            heading_path = _extract_heading_path(text)
            doc_topics = _extract_topics(text)
            parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

            for i, part in enumerate(parts):
                topic = _assign_chunk_topic(part, doc_topics)
                obj = {
                    "chunk_id": f"{doc_id}#{i}",
                    "doc_id": doc_id,
                    "doc_title": doc_id,
                    "source_type": source_type,
                    "text": part,
                    "locator": "",
                    "heading_path": heading_path,
                    "topic": topic,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                chunk_id_counter += 1

    logger.info("Wrote %d chunks to %s", chunk_id_counter, out_file)
    return out_file


# ---------------------------------------------------------------------------
# Milvus builder
# ---------------------------------------------------------------------------
def build_milvus(kb_base_dir: Path, course_id: str, chunk_size: int, overlap: int) -> None:
    """Build Milvus collection from processed documents.

    Encodes both dense (1024-d) and sparse (BGE-M3 lexical weights) vectors,
    and writes topic metadata for content-aware filtering.
    """
    from app.embedding import encode_documents_with_sparse
    from app.retrievers.milvus_retriever import get_or_create_collection

    course_dir = kb_base_dir / course_id
    processed_dir = course_dir / "processed"
    index_dir = course_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    files = list(processed_dir.rglob("*.md")) + list(processed_dir.rglob("*.txt"))

    from pymilvus import utility

    if not files:
        logger.warning("No documents found in %s. Creating empty collection.", processed_dir)
        get_or_create_collection(course_id)
        return

    # Full rebuild: drop existing → recreate → insert
    col_name = f"kb_{course_id}".replace("-", "_")
    
    # Use MilvusClient to check collection existence (avoids database context issues)
    from app.settings import settings
    client_uri = settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}"
    from pymilvus import MilvusClient
    client = MilvusClient(uri=client_uri)
    
    if client.has_collection(col_name):
        client.drop_collection(col_name)
        logger.info("Dropped existing collection: %s", col_name)
    col = get_or_create_collection(course_id)

    all_entities: list[dict] = []
    chunk_id_counter = 0
    doc_hashes: dict[str, str] = {}
    batch_size = 32  # encode in batches to manage memory

    for file_path in sorted(files):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        source_type = _detect_source_type(file_path)
        doc_id = file_path.stem
        doc_hash = _hash_file(file_path)
        doc_hashes[doc_id] = doc_hash

        heading_path = _extract_heading_path(text)
        doc_topics = _extract_topics(text)

        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not parts:
            continue

        # Batch-encode for efficiency
        dense_list: list[list[float]] = []
        sparse_list: list[dict[int, float]] = []
        for batch_start in range(0, len(parts), batch_size):
            batch = parts[batch_start : batch_start + batch_size]
            d, s = encode_documents_with_sparse(batch)
            dense_list.extend(d)
            sparse_list.extend(s)

        for i, part in enumerate(parts):
            topic = _assign_chunk_topic(part, doc_topics)
            all_entities.append({
                "chunk_id": f"{doc_id}#{i}",
                "doc_id": doc_id,
                "doc_title": doc_id,
                "source_type": source_type,
                "text": part,
                "heading_path": heading_path,
                "locator": "",
                "topic": topic,
                "embedding": dense_list[i],
                "sparse_vector": sparse_list[i],
            })
            chunk_id_counter += 1

    if all_entities:
        # Insert in batches to avoid gRPC message size limits
        INSERT_BATCH = 500
        for batch_start in range(0, len(all_entities), INSERT_BATCH):
            batch = all_entities[batch_start : batch_start + INSERT_BATCH]
            col.insert(batch)
        col.flush()
        logger.info(
            "Inserted %d chunks into Milvus collection (dense=%dd + sparse).",
            chunk_id_counter, DIM,
        )

    # Persist build manifest
    manifest = {
        "course_id": course_id,
        "chunk_count": chunk_id_counter,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": "BAAI/bge-m3",
        "dense_dim": DIM,
        "sparse": True,
        "documents": doc_hashes,
    }
    manifest_file = index_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Build manifest written to %s", manifest_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DIM = 1024  # BGE-M3 dense dimension


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index for learnthink-rag.")
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--kb-base-dir", default="kb")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=125)
    parser.add_argument(
        "--mode", choices=["jsonl", "milvus"], default="milvus",
        help="Index backend: milvus (default) or jsonl (legacy)",
    )

    args = parser.parse_args()

    kb_base_dir = Path(args.kb_base_dir).resolve()
    logger.info(
        "Building index: course=%s, mode=%s, chunk_size=%d, overlap=%d",
        args.course_id, args.mode, args.chunk_size, args.overlap,
    )

    if args.mode == "jsonl":
        build_jsonl(kb_base_dir, args.course_id, args.chunk_size, args.overlap)
    else:
        build_milvus(kb_base_dir, args.course_id, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
