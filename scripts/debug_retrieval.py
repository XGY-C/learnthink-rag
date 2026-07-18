#!/usr/bin/env python3
"""Debug script to check actual retrieval results."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.retrievers.milvus_retriever import retrieve as milvus_retrieve, connect

# Connect to Milvus
print("Connecting to Milvus...")
connect()

# Test queries from test_data.jsonl
test_queries = [
    "什么是贝叶斯分类器",
    "朴素贝叶斯算法的原理",
    "搜索算法有哪些类型",
    "A*算法的实现",
    "人工智能绪论内容"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    results = milvus_retrieve(
        course_id="course-ai-001",
        query=query,
        k=5,
        search_mode="hybrid",
        use_reranker=True,
    )
    
    print(f"\nTop 5 Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['chunk_id']}] (score: {result['relevance']:.4f})")
        text = result.get('text', result.get('content', 'N/A'))
        print(f"   Text: {str(text)[:100]}...")
        print()
