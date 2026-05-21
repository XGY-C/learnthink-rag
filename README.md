# learnthink-rag

RAG 检索微服务（FastAPI），为上层 Java 服务提供文档检索能力。

## 核心功能

- **文档摄入**：从 OSS 同步 Markdown 文件，自动构建向量索引
- **混合检索**：BGE-M3 稠密 + 稀疏双向量检索
- **实时进度**：异步任务追踪，支持进度查询
- **重排序**：BGE-Reranker 提升检索精度

## 项目结构

```
learnthink-rag/
├── app/                    # 应用代码
│   ├── main.py            # FastAPI 主程序
│   ├── schemas.py         # 数据模型
│   ├── task_manager.py    # 任务管理器
│   ├── task_executor.py   # 异步任务执行器
│   ├── oss_client.py      # OSS 客户端
│   └── retrievers/        # 检索器实现
│
├── scripts/                # 工具脚本
│   ├── build_index.py     # 索引构建
│   └── *.py               # 调试/检查脚本
│
├── tests/                  # 测试脚本
├── examples/               # 示例代码
├── docs/                   # 详细文档
│
├── kb/                     # 知识库数据
├── models/                 # 模型缓存
├── milvus_data/           # Milvus 数据
└── eval/                   # 评估数据
```

## 目录约定

```
kb/
├── {course_id}/
│   ├── raw/          # 原始文件
│   ├── processed/    # 清洗后的纯文本/markdown
│   └── index/        # 构建元数据（manifest.json）
```

## 快速启动

### 1) 安装依赖

> **⚠️ 重要提示**：如果项目路径包含空格（如 `LearnThink Companion`），在 CMD 中必须使用引号包裹路径。

#### PowerShell
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

#### CMD (命令提示符)
```cmd
:: 创建虚拟环境
python -m venv .venv

:: 激活虚拟环境（如果路径有空格，先 cd 到项目目录）
cd /d "D:\project_xgy\LearnThink Companion\learnthink-rag"
.venv\Scripts\activate.bat

:: 安装依赖
pip install -r requirements.txt
```

### 2) 准备知识库

#### PowerShell
```powershell
# 把文档放到 kb/{course_id}/processed/ 下
# 然后构建索引（Milvus 模式需先启动 Milvus）
python scripts/build_index.py --course-id demo --mode milvus
```

#### CMD
```cmd
:: 把文档放到 kb/{course_id}/processed/ 下
:: 然后构建索引（Milvus 模式需先启动 Milvus）
:: 如果路径有空格，确保已使用 cd /d 进入项目目录
python scripts\build_index.py --course-id demo --mode milvus
```

### 3) 启动服务

#### PowerShell
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 19531 --reload
```

#### CMD
```cmd
:: 确保已在项目目录中，然后启动服务
uvicorn app.main:app --host 0.0.0.0 --port 19531 --reload
```

健康检查：`GET http://localhost:19531/healthz`
检索接口：`POST http://localhost:19531/internal/rag/retrieve`

#### 示例请求（PowerShell）
```powershell
Invoke-RestMethod -Uri "http://localhost:19531/internal/rag/retrieve" -Method Post -ContentType "application/json" -Body '{"course_id":"demo","query":"贝叶斯 分类器 定义 例子","k":8}'
```

#### 示例请求（CMD）
```cmd
curl -X POST http://localhost:19531/internal/rag/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"course_id\":\"demo\",\"query\":\"贝叶斯 分类器 定义 例子\",\"k\":8}"
```

## 索引格式

### Milvus Collection: kb_{course_id}

每条记录包含：`chunk_id`, `doc_id`, `doc_title`, `source_type`, `text`, `heading_path`, `locator`, `topic`, `embedding(1024d)`, `sparse_vector`（v2 新增）

### JSONL（legacy fallback）

文件：`kb/<course_id>/index/chunks.jsonl`

```json
{"chunk_id":"doc1#0","doc_id":"doc1","doc_title":"讲义-第1章","source_type":"讲义","text":"...","locator":"","heading_path":""}
```

## 检索模式

### 向量检索模式（`search_mode` 参数）

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `hybrid`（默认） | 稠密 + 稀疏加权融合（0.7/0.3） | 通用，推荐 |
| `dense` | 纯稠密（余弦相似度） | 纯语义理解 |
| `sparse` | 纯稀疏（关键词内积） | 术语/公式精确匹配 |

```powershell
# 混合检索（默认）
Invoke-RestMethod -Uri "http://localhost:19531/internal/rag/retrieve" -Method Post -ContentType "application/json" -Body '{"course_id":"demo","query":"贝叶斯分类器","k":8,"search_mode":"hybrid","sparse_weight":0.3}'
```

```cmd
:: 混合检索（默认）
curl -X POST http://localhost:19531/internal/rag/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"course_id\":\"demo\",\"query\":\"贝叶斯分类器\",\"k\":8,\"search_mode\":\"hybrid\",\"sparse_weight\":0.3}"
```

### 检索模式对比

| | Milvus (hybrid) | JSONL |
|------|--------|-------|
| 检索方式 | BGE-M3 稠密+稀疏双向量 | 关键词子串计数 |
| 中文理解 | 强（语义 + 关键词） | 弱（依赖字面匹配） |
| 外部依赖 | Milvus + GPU/CPU | 无 |
| 适用场景 | 正常使用 | 紧急降级 / 联调占位 |
