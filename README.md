# learnthink-rag

RAG 检索服务（FastAPI）。支持两种模式：

- **Milvus**（默认）：向量检索，使用 BGE-M3 embedding + Milvus 2.4+
- **JSONL**（fallback）：简单关键词打分，零外部依赖，仅用于紧急降级

切换方式：`.env` 中 `RETRIEVER_MODE=milvus|jsonl`

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

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### 2) 准备知识库

```bash
# 把文档放到 kb/{course_id}/processed/ 下
# 然后构建索引（Milvus 模式需先启动 Milvus）
python scripts/build_index.py --course-id demo --mode milvus
```

### 3) 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 19531 --reload
```

健康检查：`GET http://localhost:19531/healthz`
检索接口：`POST http://localhost:19531/internal/rag/retrieve`

示例请求：

```bash
curl -X POST http://localhost:19531/internal/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"course_id":"demo","query":"贝叶斯 分类器 定义 例子","k":8}'
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

```bash
# 混合检索（默认）
curl -X POST http://localhost:19531/internal/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"course_id":"demo","query":"贝叶斯分类器","k":8,"search_mode":"hybrid","sparse_weight":0.3}'
```

### 检索模式对比

| | Milvus (hybrid) | JSONL |
|------|--------|-------|
| 检索方式 | BGE-M3 稠密+稀疏双向量 | 关键词子串计数 |
| 中文理解 | 强（语义 + 关键词） | 弱（依赖字面匹配） |
| 外部依赖 | Milvus + GPU/CPU | 无 |
| 适用场景 | 正常使用 | 紧急降级 / 联调占位 |
