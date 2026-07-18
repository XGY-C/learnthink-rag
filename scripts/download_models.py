#!/usr/bin/env python3
"""预下载 RAG 所需的 HuggingFace 模型到 ./models/

模型体积较大（BGE-M3 ≈ 2.3GB，BGE-Reranker-v2-m3 ≈ 2.2GB），建议在构建/部署
阶段提前下载，避免服务首次启动时长时间等待。

用法:
  python scripts/download_models.py                # 下载全部
  python scripts/download_models.py --embedding-only
  python scripts/download_models.py --reranker-only

国内网络推荐先配置镜像（二选一）:
  方式 A: 在 .env 中设置  HF_ENDPOINT=https://hf-mirror.com
  方式 B: 设置环境变量后运行
          set HF_ENDPOINT=https://hf-mirror.com        (Windows)
          export HF_ENDPOINT=https://hf-mirror.com     (Linux/Mac)

已下载的模型会被自动复用，不会重复下载。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.log_config import setup_logger
from app.model_loader import resolve_model_path
from app.settings import settings

logger = setup_logger("download_models")


def main() -> None:
    parser = argparse.ArgumentParser(description="预下载 RAG 所需的 HuggingFace 模型")
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="只下载嵌入模型 (BGE-M3)",
    )
    parser.add_argument(
        "--reranker-only",
        action="store_true",
        help="只下载重排序模型 (BGE-Reranker-v2-m3)",
    )
    args = parser.parse_args()

    do_all = not args.embedding_only and not args.reranker_only

    logger.info("缓存目录: %s", Path(settings.embedding_cache_dir).resolve())
    if settings.hf_endpoint:
        logger.info("HF 镜像: %s", settings.hf_endpoint)

    if do_all or args.embedding_only:
        logger.info("--- 下载嵌入模型: %s ---", settings.embedding_model)
        resolve_model_path(settings.embedding_model)

    if do_all or args.reranker_only:
        logger.info("--- 下载重排序模型: %s ---", settings.reranker_model)
        resolve_model_path(settings.reranker_model)

    logger.info("✅ 模型准备完成，可启动 RAG 服务")


if __name__ == "__main__":
    main()
