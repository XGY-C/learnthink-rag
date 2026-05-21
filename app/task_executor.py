"""异步任务执行器 - 在后台线程中执行索引构建任务"""
import logging
import time
import threading
import requests
from pathlib import Path
from typing import Callable, Optional

from app.task_manager import task_manager, TaskStatus
from app.settings import settings

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path) -> None:
    """通过HTTP下载文件到本地"""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info(f"[Download] {url} -> {dest.name} ({len(resp.content)} bytes)")


def execute_ingest_task(
    task_id: str,
    course_id: str,
    doc_ids: list,
    full_rebuild: bool,
    file_urls: Optional[list] = None,
    progress_callback: Callable = None
):
    """
    在后台线程中执行文档摄入任务

    Args:
        task_id: 任务ID
        course_id: 课程ID
        doc_ids: 文档ID列表
        full_rebuild: 是否全量重建
        file_urls: 文件URL列表，通过HTTP直接下载（无需OSS SDK）
        progress_callback: 进度回调函数（可选）
    """
    from scripts.build_index import build_milvus

    t0 = time.time()
    total_files = 0

    try:
        # === Step 1: 下载文件 ===
        logger.info(f"[Task {task_id}] Step 1: Downloading files...")
        task_manager.update_progress(
            task_id,
            status=TaskStatus.RUNNING,
            progress=5,
            current_step="正在下载文件...",
            message="准备从OSS同步Markdown文件"
        )

        kb_base = Path(settings.kb_base_dir).resolve()
        local_dir = kb_base / course_id / "processed"
        local_dir.mkdir(parents=True, exist_ok=True)

        if file_urls:
            # 通过HTTP直接下载（无需OSS SDK）
            downloaded_files = []
            for idx, url in enumerate(file_urls):
                filename = Path(url).name
                if not filename.endswith(('.md', '.txt')):
                    continue
                if doc_ids:
                    doc_id = filename.rsplit('.', 1)[0]
                    if doc_id not in doc_ids:
                        continue
                dest = local_dir / filename
                download_file(url, dest)
                downloaded_files.append(filename)

            total_files = len(downloaded_files)
            logger.info(f"[Task {task_id}] Downloaded {total_files} files via HTTP")
        else:
            # 兼容旧模式：通过OSS SDK下载
            from app.oss_client import OSSClient
            client = OSSClient()
            downloaded_files = client.download_files(
                course_id=course_id,
                local_dir=local_dir,
                doc_ids=doc_ids if doc_ids else None
            )
            total_files = len(downloaded_files)
            logger.info(f"[Task {task_id}] Downloaded {total_files} files via OSS SDK")

        if total_files == 0:
            raise Exception(f"No files found for course={course_id}")

        task_manager.update_progress(
            task_id,
            progress=20,
            current_step=f"已下载 {total_files} 个文件",
            total_files=total_files,
            processed_files=0,
            message=f"成功同步 {total_files} 个Markdown文件"
        )

        # === Step 2: 构建索引 ===
        logger.info(f"[Task {task_id}] Step 2: Building Milvus index...")
        task_manager.update_progress(
            task_id,
            progress=30,
            current_step="正在构建向量索引...",
            message="开始对文档进行分块、编码和索引构建"
        )

        build_milvus(
            kb_base_dir=kb_base,
            course_id=course_id,
            chunk_size=800,
            overlap=200,
            full_rebuild=full_rebuild,
            progress_callback=lambda pct, msg: task_manager.update_progress(
                task_id,
                progress=30 + pct * 0.55,  # local 0-100 → overall 30-85
                current_step=msg,
                processed_files=int(total_files * min(pct, 100) / 100) if total_files > 0 else 0,
                message=msg
            )
        )

        # === Step 3: 读取结果 ===
        logger.info(f"[Task {task_id}] Step 3: Reading build results...")
        task_manager.update_progress(
            task_id,
            progress=90,
            current_step="正在读取构建结果...",
            message="索引构建完成，正在统计结果"
        )

        manifest_file = kb_base / course_id / "index" / "manifest.json"
        chunk_count = 0
        if manifest_file.exists():
            import json
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            chunk_count = manifest.get("chunk_count", 0)

        elapsed_ms = (time.time() - t0) * 1000

        # === 完成任务 ===
        task_manager.update_progress(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            current_step="完成",
            processed_files=total_files,
            total_chunks=chunk_count,
            message=f"索引构建成功！处理了 {total_files} 个文件，生成 {chunk_count} 个文本块，耗时 {elapsed_ms/1000:.1f}秒"
        )

        logger.info(
            f"[Task {task_id}] Completed: course={course_id}, files={total_files}, "
            f"chunks={chunk_count}, elapsed={elapsed_ms:.0f}ms"
        )

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        logger.error(f"[Task {task_id}] Failed: {str(e)}", exc_info=True)

        import traceback
        full_traceback = traceback.format_exc()
        logger.debug(f"[Task {task_id}] Full traceback:\n{full_traceback}")

        task_manager.update_progress(
            task_id,
            status=TaskStatus.FAILED,
            progress=0,
            current_step="失败",
            error=str(e)[:500],
            message=f"任务执行失败: {str(e)[:200]}"
        )


def start_ingest_task(
    course_id: str,
    doc_ids: list,
    full_rebuild: bool,
    file_urls: Optional[list] = None
) -> str:
    """
    启动异步摄入任务

    Args:
        course_id: 课程ID
        doc_ids: 文档ID列表
        full_rebuild: 是否全量重建
        file_urls: 文件URL列表，通过HTTP直接下载

    Returns:
        task_id
    """
    # 检查并发任务数（简单限流）
    running_tasks = [
        t for t in task_manager.list_tasks(limit=100)
        if t['status'] == TaskStatus.RUNNING
    ]

    if len(running_tasks) >= 10:
        logger.warning(
            f"High concurrency detected: {len(running_tasks)} running tasks. "
            f"Consider limiting concurrent submissions."
        )

    # 创建任务
    task_id = task_manager.create_task(course_id)

    # 初始化任务状态
    task_manager.update_progress(
        task_id,
        status=TaskStatus.PENDING,
        progress=0,
        current_step="任务已创建，等待执行...",
        message="文档摄入任务已提交，即将开始执行"
    )

    # 在后台线程中执行（非daemon，确保任务能完整执行）
    thread = threading.Thread(
        target=execute_ingest_task,
        args=(task_id, course_id, doc_ids, full_rebuild, file_urls),
        daemon=False
    )
    thread.start()

    logger.info(f"[Task {task_id}] Started in background thread (non-daemon)")

    return task_id
