from __future__ import annotations

import logging
import os
from pathlib import Path

from app.settings import settings

logger = logging.getLogger(__name__)


def resolve_model_path(
    repo_id: str,
    cache_dir: str | Path | None = None,
) -> str:
    """Resolve a HuggingFace model to a local directory, downloading if missing.

    优先复用本地缓存（无需联网）；若缓存不存在则从 HuggingFace Hub 下载到
    ``cache_dir``（默认 ``./models``，即 ``settings.embedding_cache_dir``）。

    国内网络可在 ``.env`` 中配置镜像::

        HF_ENDPOINT=https://hf-mirror.com

    Args:
        repo_id: HuggingFace 仓库 ID，如 ``"BAAI/bge-m3"``。
        cache_dir: 本地缓存目录，默认使用 ``settings.embedding_cache_dir``。

    Returns:
        模型快照的本地绝对路径。
    """
    cache_dir_path = Path(cache_dir or settings.embedding_cache_dir).resolve()

    local = _find_cached_snapshot(repo_id, cache_dir_path)
    if local:
        logger.info("Using cached model '%s': %s", repo_id, local)
        return str(local)

    if settings.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    logger.info(
        "Model '%s' not found in %s, downloading from HuggingFace Hub...",
        repo_id, cache_dir_path,
    )
    if os.environ.get("HF_ENDPOINT"):
        logger.info("Using HF endpoint: %s", os.environ["HF_ENDPOINT"])

    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir_path))
    logger.info("Model '%s' ready at: %s", repo_id, local)
    return local


def _find_cached_snapshot(repo_id: str, cache_dir: Path) -> Path | None:
    """Scan the HF cache layout for an existing complete snapshot of ``repo_id``.

    Handles caches that lack a ``refs/main`` pointer (e.g. reranker models
    downloaded via ``from_pretrained``), falling back to any non-empty snapshot.
    """
    org, _, name = repo_id.partition("/")
    repo_dir = cache_dir / f"models--{org}--{name}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    # Prefer the commit referenced by refs/main
    main_ref_file = repo_dir / "refs" / "main"
    if main_ref_file.exists():
        try:
            main_ref = main_ref_file.read_text(encoding="utf-8").strip()
            for snap in candidates:
                if snap.name == main_ref and _snapshot_has_files(snap):
                    return snap
        except Exception:
            pass

    # Fall back to the most recently modified snapshot with files
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in candidates:
        if _snapshot_has_files(snap):
            return snap
    return None


def _snapshot_has_files(snap_path: Path) -> bool:
    """Check that a snapshot directory contains real model files."""
    try:
        return any(entry.is_file() for entry in snap_path.rglob("*"))
    except Exception:
        return False
