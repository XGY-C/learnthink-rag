"""阿里云OSS客户端 - 用于从OSS同步知识库文件"""
import logging
from pathlib import Path
from typing import List, Optional

import oss2

from app.settings import settings

logger = logging.getLogger(__name__)


class OSSClient:
    """阿里云OSS客户端封装"""

    def __init__(self):
        if settings.oss_access_key:
            self.auth = oss2.Auth(settings.oss_access_key, settings.oss_secret_key)
            logger.info("OSS client initialized with access key")
        else:
            self.auth = oss2.AnonymousAuth()
            logger.info("OSS client initialized with anonymous auth (public bucket)")

        self.bucket = oss2.Bucket(
            self.auth,
            settings.oss_endpoint,
            settings.oss_bucket_name
        )
        logger.info(f"OSS bucket: {settings.oss_bucket_name}")

    def download_files(
        self,
        course_id: str,
        local_dir: Path,
        doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        从OSS下载指定课程的MD文件到本地

        Args:
            course_id: 课程ID，如 "course-ai-001"
            local_dir: 本地目标目录，如 "kb/course-ai-001/processed"
            doc_ids: 指定文档ID列表，None则下载全部

        Returns:
            下载的文件名列表
        """
        prefix = f"kb/{course_id}/processed/"
        downloaded_files = []

        logger.info(f"[OSS Sync] Starting download from oss://{settings.oss_bucket_name}/{prefix}")

        # 确保本地目录存在
        local_dir.mkdir(parents=True, exist_ok=True)

        # 遍历OSS目录
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            if not obj.key.endswith('.md'):
                continue

            # 提取文件名（不含扩展名）
            filename = Path(obj.key).name
            doc_id = filename[:-3]  # 去掉 .md

            # 如果指定了doc_ids，只下载匹配的
            if doc_ids and doc_id not in doc_ids:
                logger.debug(f"[OSS Sync] Skipping {filename} (not in doc_ids)")
                continue

            local_file = local_dir / filename

            # 下载文件
            try:
                self.bucket.get_object_to_file(obj.key, str(local_file))
                downloaded_files.append(filename)
                logger.info(f"[OSS Sync] Downloaded: {filename} ({obj.content_length} bytes)")
            except Exception as e:
                logger.error(f"[OSS Sync] Failed to download {filename}: {e}")
                raise

        logger.info(f"[OSS Sync] Completed: {len(downloaded_files)} files downloaded")
        return downloaded_files

    def verify_file_exists(self, course_id: str, doc_id: str) -> bool:
        """验证OSS中是否存在指定文件"""
        key = f"kb/{course_id}/processed/{doc_id}.md"
        return self.bucket.object_exists(key)
