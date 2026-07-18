"""
演示实时进度追踪功能（无需实际OSS连接）
模拟任务执行过程，展示进度更新机制
"""
import time
from app.task_manager import task_manager, TaskStatus


def simulate_task_execution():
    """模拟任务执行过程"""
    
    print("=" * 70)
    print("实时进度追踪功能演示")
    print("=" * 70)
    print()
    
    # 1. 创建任务
    task_id = task_manager.create_task("course-demo-001")
    print(f"✅ 任务已创建: {task_id}")
    print()
    
    # 2. 模拟执行过程
    steps = [
        (TaskStatus.RUNNING, 5, "正在初始化...", "任务开始执行"),
        (None, 10, "正在从OSS下载文件...", "准备从阿里云OSS同步Markdown文件"),
        (None, 20, "已下载 3 个文件", "成功从OSS同步 3 个Markdown文件"),
        (None, 30, "正在构建向量索引...", "开始对文档进行分块、编码和索引构建"),
        (None, 45, "正在处理文档 2/5...", "已完成第1个文档，正在处理第2个"),
        (None, 60, "正在处理文档 4/5...", "已完成第3个文档，正在处理第4个"),
        (None, 75, "正在插入Milvus数据库...", "所有文档编码完成，正在写入向量数据库"),
        (None, 90, "正在统计结果...", "索引构建完成，正在统计结果"),
        (TaskStatus.COMPLETED, 100, "完成", "索引构建成功！处理了 5 个文件，生成 85 个文本块，耗时 12.3秒"),
    ]
    
    print("开始模拟任务执行...\n")
    print("-" * 70)
    
    for status, progress, step, message in steps:
        # 更新进度
        task_manager.update_progress(
            task_id,
            status=status,
            progress=progress,
            current_step=step,
            message=message
        )
        
        # 获取当前进度
        current = task_manager.get_task(task_id)
        
        # 显示进度
        print(f"\r⏳ [{current['progress']:5.1f}%] {current['current_step']}")
        if current['message']:
            print(f"   💬 {current['message']}")
        
        # 模拟处理时间
        time.sleep(0.8)
    
    print("-" * 70)
    print()
    
    # 3. 显示最终结果
    final = task_manager.get_task(task_id)
    print("📊 最终任务状态:")
    print(f"   任务ID:     {final['task_id']}")
    print(f"   课程ID:     {final['course_id']}")
    print(f"   状态:       {final['status']}")
    print(f"   进度:       {final['progress']}%")
    print(f"   总文件数:   {final['total_files']}")
    print(f"   处理文件数: {final['processed_files']}")
    print(f"   文本块数:   {final['total_chunks']}")
    print(f"   消息:       {final['message']}")
    print()
    
    # 4. 列出所有任务
    print("📋 任务历史:")
    tasks = task_manager.list_tasks(limit=5)
    for i, task in enumerate(tasks, 1):
        status_icon = "✅" if task['status'] == 'completed' else "⏳"
        print(f"   {i}. {status_icon} Task {task['task_id']} - {task['status']} ({task['progress']}%)")
    
    print()
    print("=" * 70)
    print("演示完成！")
    print("=" * 70)
    print()
    print("💡 提示:")
    print("   - Java端可以通过 GET /admin/task/{task_id} 查询实时进度")
    print("   - 建议每2秒轮询一次")
    print("   - 前端可以根据 progress 字段显示进度条")


if __name__ == "__main__":
    simulate_task_execution()
