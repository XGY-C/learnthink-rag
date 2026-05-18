from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

# Add project root to sys.path for module imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.log_config import setup_logger

logger = setup_logger("build_index")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunking with semantic boundary detection.

    Improvements over naive chunking:
    - Raises minimum length from 80 → 150 chars (80 was too low for Chinese;
      a chunk of just 80 chars often carries too little context).
    - Tries to split at paragraph boundaries first, then sentence boundaries
      within long paragraphs.
    """
    text = text.strip()
    if not text:
        return []

    MIN_CHUNK_LEN = 150  # was 80 — too many low-information chunks

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
                # Long paragraph: try sentence-boundary-aware splitting first
                sentence_chunks = _split_long_paragraph(para, chunk_size, overlap)
                chunks.extend(sentence_chunks)
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


def _split_long_paragraph(para: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long paragraph at sentence boundaries when possible.

    Falls back to character-level sliding window only when a single
    sentence exceeds chunk_size.
    """
    import re as _re
    # Split on Chinese/English sentence endings (expanded coverage)
    # Added support for semicolons and colons which are common in technical writing
    sentences = _re.split(r"(?<=[。！？；:;])\s*", para)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        # Fallback: character-level sliding window
        return _sliding_window_split(para, chunk_size, overlap)

    chunks: list[str] = []
    buffer = ""

    for sent in sentences:
        if len(buffer) + len(sent) <= chunk_size:
            buffer = (buffer + sent).strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(sent) > chunk_size:
                # Single sentence too long — fall back to sliding window
                sub_chunks = _sliding_window_split(sent, chunk_size, overlap)
                chunks.extend(sub_chunks)
                buffer = ""
            else:
                buffer = sent

    if buffer:
        chunks.append(buffer)

    return chunks


def _sliding_window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-level sliding window split for text with no natural boundaries."""
    step = max(1, chunk_size - overlap)
    results: list[str] = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# Course metadata & chapter info extraction (v3.5)
# ---------------------------------------------------------------------------
def _load_course_meta(course_dir: Path) -> dict[str, tuple[str, str]]:
    """Load course_meta.json and return a {prefix: (book_title, book_type)} map.

    Expected JSON structure:
      { "name": "...", "books": [{"prefix": "ch", "title": "...", "type": "..."}, ...] }

    Returns empty dict if file missing or unparseable (build continues with defaults).
    """
    meta_file = course_dir / "course_meta.json"
    if not meta_file.exists():
        logger.info("No course_meta.json found at %s", meta_file)
        return {}
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        books = meta.get("books", [])
        result: dict[str, tuple[str, str]] = {}
        for b in books:
            prefix = b.get("prefix", "")
            title = b.get("title", "")
            btype = b.get("type", "unknown")
            if prefix:
                result[prefix] = (title, btype)
        logger.info("Loaded course meta with %d book mappings", len(result))
        return result
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load course_meta.json: %s", e)
        return {}


def _parse_chapter_info(doc_id: str) -> tuple[int | None, str]:
    """Extract chapter_index and chapter_title from a doc_id (filename stem).

    Chapter files match pattern:  ch(\d+)_(.+)     → e.g. ch03_贝叶斯分类器
    Non-chapter files (exercises, glossary, etc.):  prefix_descriptive_part

    Returns: (chapter_index or None, chapter_title or "")
    """
    # Chapter files: ch03_贝叶斯分类器 → index=3, title="贝叶斯分类器"
    m = re.match(r"^ch(\d+)_(.+)$", doc_id)
    if m:
        return int(m.group(1)), m.group(2).strip()

    # Non-chapter files: extract descriptive part after first underscore
    if "_" in doc_id:
        parts = doc_id.split("_", 1)
        desc = parts[1].strip() if len(parts) > 1 else ""
        return None, desc

    return None, ""


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
        return "习题"
    if any(kw in name for kw in ["术语", "glossary", "词汇"]):
        return "术语"
    if any(kw in name for kw in ["代码", "code", "程序", "示例"]):
        return "代码"
    if any(kw in name for kw in ["项目", "project", "实践"]):
        return "项目"
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


def _build_heading_map(text: str) -> list[tuple[int, str, int]]:
    """Parse all Markdown headings in a document with their byte offsets.

    Returns a list of (char_offset, title, level) sorted by offset.
    Used to determine which heading each chunk belongs to.
    """
    heading_map: list[tuple[int, str, int]] = []
    for m in re.finditer(r"^(#{1,4})\s+(.+)$", text, re.MULTILINE):
        level = len(m.group(1))
        title = m.group(2).strip()
        heading_map.append((m.start(), title, level))
    return heading_map


def _heading_path_for_chunk(
    chunk_start: int,
    heading_map: list[tuple[int, str, int]],
) -> str:
    """Determine the heading breadcrumb for a chunk at `chunk_start` offset.

    Walks the heading map to find the active heading stack at the chunk's
    position, building a proper breadcrumb like:
      "人工智能导论 / 第3章 贝叶斯分类器 / 3.2 实战代码"
    instead of using the same static heading for all chunks.
    """
    if not heading_map:
        return ""

    # Find all headings that appear before chunk_start
    active: list[tuple[int, str]] = []  # (level, title)
    for offset, title, level in heading_map:
        if offset > chunk_start:
            break
        # Pop headings at same or deeper level
        while active and active[-1][0] >= level:
            active.pop()
        active.append((level, title))

    if not active:
        return ""
    return "/".join(title for _, title in active)


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
    """Assign a single best topic to a chunk based on keyword overlap.

    Fixes from original implementation:
    - Uses jieba word segmentation for Chinese text to avoid substring false positives.
      ("人工智能" no longer matches "弱人工智能", "人工智能导论", etc.)
    - Returns empty string instead of doc_topics[0] as fallback, because
      forcing the first topic on unrelated chunks is metadata pollution.
    - Picks the longest matching topic to prefer more specific matches.
    - Requires at least 2 overlapping words for a match to reduce noise.
    """
    try:
        import jieba
    except ImportError:
        # Fallback to simple substring matching if jieba not installed
        logger.warning("jieba not installed, falling back to substring matching")
        return _assign_chunk_topic_fallback(chunk, doc_topics)

    chunk_words = set(jieba.cut(chunk))
    
    best_topic = ""
    max_overlap = 0

    for topic in doc_topics:
        topic_words = set(jieba.cut(topic))
        # Calculate word-level overlap
        overlap = len(chunk_words & topic_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_topic = topic

    # Require at least 2 overlapping words to consider it a valid match
    # This reduces false positives from common single-character words
    if max_overlap >= 2:
        return best_topic
    return ""  # No forced fallback — empty is better than wrong


def _assign_chunk_topic_fallback(chunk: str, doc_topics: list[str]) -> str:
    """Fallback topic assignment using substring matching (when jieba unavailable).
    
    This is less accurate but ensures the system works without jieba dependency.
    """
    import re as _re
    chunk_lower = chunk.lower()

    # Find all matching topics, prefer the longest (most specific) match
    matches: list[str] = []
    for topic in doc_topics:
        topic_lower = topic.lower()
        pattern = _re.escape(topic_lower)
        if _re.search(pattern, chunk_lower):
            matches.append(topic)

    if matches:
        # Return the longest matching topic (most specific)
        return max(matches, key=len)
    return ""

# ---------------------------------------------------------------------------
# Milvus builder
# ---------------------------------------------------------------------------
def _load_manifest(index_dir: Path) -> dict | None:
    """Load the previous build manifest if it exists."""
    manifest_file = index_dir / "manifest.json"
    if manifest_file.exists():
        try:
            return json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load manifest: %s", e)
    return None


def _compute_chunk_offsets(text: str, chunks: list[str]) -> list[int]:
    """Compute the character offset of each chunk within the original text.

    Used to determine which heading section each chunk belongs to.
    Returns a list of offsets (one per chunk).
    """
    offsets: list[int] = []
    search_start = 0
    for chunk in chunks:
        key = chunk[:50].strip()
        pos = text.find(key, search_start)
        if pos >= 0:
            offsets.append(pos)
            search_start = pos + len(key)
        else:
            offsets.append(search_start)
    return offsets


DIM = 1024  # BGE-M3 dense dimension (moved up for use in build_milvus)


def build_milvus(
    kb_base_dir: Path,
    course_id: str,
    chunk_size: int,
    overlap: int,
    full_rebuild: bool = False,
) -> None:
    """Build Milvus collection from processed documents.

    Supports incremental indexing:
    - Reads the previous manifest to get document hashes.
    - Only re-encodes documents whose hash has changed.
    - Deletes stale chunks for changed/deleted documents before re-inserting.
    - Use --full-rebuild to force a complete rebuild.

    Encodes both dense (1024-d) and sparse (BGE-M3 lexical weights) vectors,
    and writes topic metadata for content-aware filtering.
    """
    from app.embedding import encode_documents_with_sparse
    from app.retrievers.milvus_retriever import get_or_create_collection, _collection_name
    from app.settings import settings
    from pymilvus import MilvusClient

    course_dir = kb_base_dir / course_id
    processed_dir = course_dir / "processed"
    index_dir = course_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    files = list(processed_dir.rglob("*.md")) + list(processed_dir.rglob("*.txt"))

    if not files:
        logger.warning("No documents found in %s. Creating empty collection.", processed_dir)
        get_or_create_collection(course_id)
        return

    # Load course metadata for book_title / book_type mapping
    book_map = _load_course_meta(course_dir)

    # ---- Incremental indexing logic ----
    prev_manifest = _load_manifest(index_dir) if not full_rebuild else None
    prev_hashes: dict[str, str] = prev_manifest.get("documents", {}) if prev_manifest else {}

    # Compute current hashes and determine changed/unchanged/deleted docs
    current_hashes: dict[str, str] = {}
    files_to_process: list[Path] = []
    skipped_count = 0

    for file_path in sorted(files):
        doc_id = file_path.stem
        doc_hash = _hash_file(file_path)
        current_hashes[doc_id] = doc_hash
        if doc_id in prev_hashes and prev_hashes[doc_id] == doc_hash:
            skipped_count += 1
            continue  # Document unchanged -- skip re-encoding
        files_to_process.append(file_path)

    # Detect deleted documents (in prev but not in current)
    deleted_doc_ids = set(prev_hashes.keys()) - set(current_hashes.keys())

    # Create MilvusClient for collection management (fixes undefined 'client' bug)
    client_uri = settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}"
    if not settings.milvus_uri and settings.milvus_db:
        client_uri += f"/{settings.milvus_db}"
    client = MilvusClient(uri=client_uri)
    col_name = _collection_name(course_id)

    is_incremental = (
        prev_manifest is not None
        and client.has_collection(col_name)
        and (skipped_count > 0 or deleted_doc_ids)
    )

    if is_incremental and files_to_process:
        logger.info(
            "Incremental build: %d unchanged, %d changed/new, %d deleted",
            skipped_count, len(files_to_process), len(deleted_doc_ids),
        )
    elif is_incremental and not files_to_process and not deleted_doc_ids:
        logger.info("No document changes detected. Skipping build.")
        return
    else:
        # Full rebuild path
        is_incremental = False
        if client.has_collection(col_name):
            client.drop_collection(col_name)
            logger.info("Dropped existing collection: %s", col_name)

    col = get_or_create_collection(course_id)

    # ---- Delete stale chunks for changed/deleted documents ----
    if is_incremental and (files_to_process or deleted_doc_ids):
        docs_to_delete = set()
        for f in files_to_process:
            docs_to_delete.add(f.stem)
        docs_to_delete.update(deleted_doc_ids)
        doc_list = sorted(docs_to_delete)
        DEL_BATCH = 50
        for i in range(0, len(doc_list), DEL_BATCH):
            batch = doc_list[i:i + DEL_BATCH]
            expr_parts = [f'doc_id == "{d}"' for d in batch]
            col.delete(expr=" || ".join(expr_parts))
            logger.info("Deleted stale chunks for docs: %s", batch)
        col.flush()

    # ---- Process changed/new documents ----

    all_entities: list[dict] = []
    chunk_id_counter = 0
    batch_size = 5  # encode in smaller batches to see progress faster
    total_files = len(files_to_process)
    logger.info("Starting to process %d documents...", total_files)

    for file_idx, file_path in enumerate(files_to_process, 1):
        logger.info("Processing document %d/%d: %s", file_idx, total_files, file_path.name)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        source_type = _detect_source_type(file_path)
        doc_id = file_path.stem

        # v3.5: Parse chapter info and book metadata
        chapter_index, chapter_title = _parse_chapter_info(doc_id)
        # Match filename prefix to course_meta book mapping
        book_title = ""
        book_type = "unknown"
        if book_map:
            for prefix, (bt, btype) in sorted(book_map.items(), key=lambda x: -len(x[0])):
                if doc_id.startswith(prefix):
                    book_title = bt
                    book_type = btype
                    break

        # Build heading map for per-chunk heading_path (fixes static heading bug)
        heading_map = _build_heading_map(text)
        doc_topics = _extract_topics(text)

        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not parts:
            logger.info("  No chunks generated for %s", file_path.name)
            continue

        logger.info("  Generated %d chunks, encoding...", len(parts))

        # Batch-encode for efficiency
        dense_list: list[list[float]] = []
        sparse_list: list[dict[int, float]] = []
        for batch_start in range(0, len(parts), batch_size):
            batch = parts[batch_start : batch_start + batch_size]
            d, s = encode_documents_with_sparse(batch)
            dense_list.extend(d)
            sparse_list.extend(s)

        logger.info("  Encoding complete, building entities...")

        # Compute chunk offsets for heading lookup
        chunk_offsets = _compute_chunk_offsets(text, parts)

        for i, part in enumerate(parts):
            topic = _assign_chunk_topic(part, doc_topics)
            # Per-chunk heading path instead of static document-level heading
            chunk_heading = _heading_path_for_chunk(
                chunk_offsets[i] if i < len(chunk_offsets) else 0, heading_map
            )
            all_entities.append({
                "chunk_id": f"{doc_id}#{i}",
                "doc_id": doc_id,
                "book_title": book_title,
                "book_type": book_type,
                "chapter_index": chapter_index if chapter_index is not None else -1,  # Use -1 for non-chapter documents
                "chapter_title": chapter_title if chapter_title else "",
                "source_type": source_type,
                "text": part,
                "heading_path": chunk_heading,
                "locator": "",
                "topic": topic,
                "embedding": dense_list[i],
                "sparse_vector": sparse_list[i],
            })
            chunk_id_counter += 1

        logger.info("  Document %d/%d completed (%d chunks so far)", file_idx, total_files, chunk_id_counter)

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
        "documents": current_hashes,
    }
    manifest_file = index_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Build manifest written to %s", manifest_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index for learnthink-rag.")
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--kb-base-dir", default="kb")
    parser.add_argument("--chunk-size", type=int, default=800, 
                        help="Chunk size in characters (default: 800, increased from 500 for better context)")
    parser.add_argument("--overlap", type=int, default=200,
                        help="Overlap between chunks in characters (default: 200, 25% of chunk_size)")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Force full rebuild instead of incremental indexing")

    args = parser.parse_args()

    kb_base_dir = Path(args.kb_base_dir).resolve()
    logger.info(
        "Building Milvus index: course=%s, chunk_size=%d, overlap=%d, full_rebuild=%s",
        args.course_id, args.chunk_size, args.overlap, args.full_rebuild,
    )

    build_milvus(kb_base_dir, args.course_id, args.chunk_size, args.overlap,
                 full_rebuild=args.full_rebuild)


if __name__ == "__main__":
    main()
