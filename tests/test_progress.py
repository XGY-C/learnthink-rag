"""测试任务进度追踪功能"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"


def test_ingest_async():
    """测试异步摄入任务"""
    print("=" * 60)
    print("测试1: 异步模式提交任务")
    print("=" * 60)
    
    # 提交任务
    payload = {
        "course_id": "course-ai-001",
        "doc_ids": ["ch03_贝叶斯分类器"],
        "full_rebuild": False,
        "async_mode": True
    }
    
    response = requests.post(f"{BASE_URL}/admin/ingest-document", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    task_id = response.json()["task_id"]
    print(f"\nTask ID: {task_id}\n")
    
    # 轮询查询进度
    print("开始轮询任务进度...")
    print("-" * 60)
    
    for i in range(30):  # 最多查询30次
        time.sleep(2)  # 每2秒查询一次
        
        progress_response = requests.get(f"{BASE_URL}/admin/task/{task_id}")
        
        if progress_response.status_code == 200:
            progress = progress_response.json()
            
            print(f"[{i+1}] 进度: {progress['progress']}% | "
                  f"状态: {progress['status']} | "
                  f"步骤: {progress['current_step']}")
            
            if progress['message']:
                print(f"    消息: {progress['message']}")
            
            # 任务完成或失败则退出
            if progress['status'] in ['completed', 'failed']:
                print("\n" + "=" * 60)
                print(f"最终结果:")
                print(json.dumps(progress, indent=2, ensure_ascii=False))
                break
        else:
            print(f"查询失败: {progress_response.status_code}")
            break
    
    print("-" * 60)


def test_list_tasks():
    """列出所有任务"""
    print("\n" + "=" * 60)
    print("测试2: 列出任务历史")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/admin/tasks?limit=5")
    
    if response.status_code == 200:
        tasks = response.json()
        print(f"共 {len(tasks)} 个任务:\n")
        
        for task in tasks:
            print(f"Task ID: {task['task_id']}")
            print(f"  课程: {task['course_id']}")
            print(f"  状态: {task['status']}")
            print(f"  进度: {task['progress']}%")
            print(f"  消息: {task['message']}")
            print()
    else:
        print(f"查询失败: {response.status_code}")


if __name__ == "__main__":
    try:
        test_ingest_async()
        test_list_tasks()
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到RAG服务，请先启动服务:")
        print("python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
