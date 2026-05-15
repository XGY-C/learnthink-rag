#!/usr/bin/env python3
"""一键搭建 RAG 向量数据库

用法：
  python setup_rag.py              # 默认: course-ai-001, milvus 模式
  python setup_rag.py --course-id demo
  python setup_rag.py --help

前置条件:
  pip install -r requirements.txt  (或 pip install pymilvus FlagEmbedding)

流程:
  1. 验证 kb/{course_id}/processed/ 中有文档
  2. 加载 BGE-M3 嵌入模型
  3. 创建/重建 Milvus Collection（dense + sparse 双索引）
  4. 切分 → 编码 → 写入
  5. 验证性检索测试
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("setup-rag")

# 确保项目在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.settings import settings


def check_docs(kb_base: Path, course_id: str) -> list[Path]:
    """检查是否存在知识库文档。"""
    processed = kb_base / course_id / "processed"
    if not processed.exists():
        logger.error("知识库目录不存在: %s", processed)
        logger.info("请创建目录并放入 .md / .txt 文档: mkdir -p %s", processed)
        sys.exit(1)

    files = sorted(processed.rglob("*.md")) + sorted(processed.rglob("*.txt"))
    if not files:
        logger.error("目录为空: %s", processed)
        sys.exit(1)

    logger.info("找到 %d 个文档:", len(files))
    for f in files:
        size_kb = f.stat().st_size / 1024
        logger.info("  · %s (%.1f KB)", f.name, size_kb)
    return files


def build_index(course_id: str):
    """启动构建流程。"""
    from scripts.build_index import build_milvus

    kb_base = Path(settings.kb_base_dir).resolve()
    logger.info("知识库根目录: %s", kb_base)

    check_docs(kb_base, course_id)

    # 连接 Milvus (warming up 模型)
    from app.retrievers.milvus_retriever import connect
    connect()
    from app.embedding import warmup
    warmup()

    # 构建索引 (chunk_size=500, overlap=125)
    t0 = time.time()
    build_milvus(
        kb_base_dir=kb_base,
        course_id=course_id,
        chunk_size=500,
        overlap=125,
    )
    elapsed = time.time() - t0
    logger.info("✅ 向量数据库构建完成 (%.1f s)", elapsed)


def test_retrieve(course_id: str):
    """验证性检索测试。"""
    from app.retrievers.milvus_retriever import retrieve

    queries = [
        ("贝叶斯分类器 定义", "hybrid"),
        ("A*搜索算法 启发函数", "hybrid"),
        ("机器学习 监督学习", "dense"),
        ("垃圾邮件 朴素贝叶斯", "sparse"),
    ]

    logger.info("--- 检索验证 ---")
    for query, mode in queries:
        sources = retrieve(
            course_id=course_id,
            query=query,
            k=3,
            search_mode=mode,
        )
        logger.info("query='%s' mode=%s → %d 条结果", query, mode, len(sources))
        for i, s in enumerate(sources[:2]):
            logger.info("  [%d] %s (%.4f)", i + 1, s["doc_title"], s["relevance"])


def main():
    parser = argparse.ArgumentParser(description="一键搭建 RAG 向量数据库")
    parser.add_argument("--course-id", default="course-ai-001",
                        help="课程 ID，对应 kb/{course_id}/processed/")
    parser.add_argument("--skip-test", action="store_true",
                        help="跳过检索验证")
    args = parser.parse_args()

    build_index(args.course_id)

    if not args.skip_test:
        test_retrieve(args.course_id)

    logger.info("---")
    logger.info("下一步: uvicorn app.main:app --host 0.0.0.0 --port 19531 --reload")


if __name__ == "__main__":
    main()
