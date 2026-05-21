"""任务进度管理器 - 跟踪索引构建任务的实时进度"""
import logging
import time
import uuid
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class TaskStatus:
    """任务状态常量"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskProgressInfo:
    """任务进度信息"""
    
    def __init__(self, task_id: str, course_id: str):
        self.task_id = task_id
        self.course_id = course_id
        self.status = TaskStatus.PENDING
        self.progress = 0.0  # 0-100
        self.current_step = ""
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.message = ""
        self.error = None
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "course_id": self.course_id,
            "status": self.status,
            "progress": round(self.progress, 2),
            "current_step": self.current_step,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "total_chunks": self.total_chunks,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskManager:
    """全局任务管理器（单例）"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tasks: Dict[str, TaskProgressInfo] = {}
        self._lock = Lock()
        self._initialized = True
        logger.info("TaskManager initialized")
    
    def create_task(self, course_id: str) -> str:
        """创建新任务，返回task_id"""
        task_id = str(uuid.uuid4())[:8]  # 短ID便于展示
        
        with self._lock:
            task = TaskProgressInfo(task_id, course_id)
            self._tasks[task_id] = task
        
        logger.info(f"[TaskManager] Created task {task_id} for course {course_id}")
        return task_id
    
    def update_progress(
        self, 
        task_id: str, 
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        total_files: Optional[int] = None,
        processed_files: Optional[int] = None,
        total_chunks: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ):
        """更新任务进度"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"[TaskManager] Task {task_id} not found")
                return
            
            task = self._tasks[task_id]
            
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = progress
            if current_step is not None:
                task.current_step = current_step
            if total_files is not None:
                task.total_files = total_files
            if processed_files is not None:
                task.processed_files = processed_files
            if total_chunks is not None:
                task.total_chunks = total_chunks
            if message is not None:
                task.message = message
            if error is not None:
                task.error = error
            
            task.updated_at = time.time()
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务进度"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.to_dict()
            return None
    
    def list_tasks(self, course_id: Optional[str] = None, limit: int = 10) -> list:
        """列出任务（可选按课程过滤）"""
        with self._lock:
            tasks = list(self._tasks.values())
            
            if course_id:
                tasks = [t for t in tasks if t.course_id == course_id]
            
            # 按创建时间倒序
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            
            return [t.to_dict() for t in tasks[:limit]]
    
    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """清理过期任务（默认1小时）"""
        now = time.time()
        with self._lock:
            to_delete = [
                task_id for task_id, task in self._tasks.items()
                if now - task.created_at > max_age_seconds
            ]
            for task_id in to_delete:
                del self._tasks[task_id]
            if to_delete:
                logger.info(f"[TaskManager] Cleaned up {len(to_delete)} old tasks")
    
    def get_stats(self) -> dict:
        """获取任务统计信息（轻量级监控）"""
        with self._lock:
            all_tasks = list(self._tasks.values())
            
            stats = {
                "total": len(all_tasks),
                "pending": sum(1 for t in all_tasks if t.status == TaskStatus.PENDING),
                "running": sum(1 for t in all_tasks if t.status == TaskStatus.RUNNING),
                "completed": sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in all_tasks if t.status == TaskStatus.FAILED),
            }
            
            # 计算平均耗时（仅已完成的任务）
            completed_tasks = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
            if completed_tasks:
                avg_duration = sum(
                    t.updated_at - t.created_at 
                    for t in completed_tasks
                ) / len(completed_tasks)
                stats["avg_duration_seconds"] = round(avg_duration, 2)
            else:
                stats["avg_duration_seconds"] = 0
            
            return stats


# 全局实例
task_manager = TaskManager()
