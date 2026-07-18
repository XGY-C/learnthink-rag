#!/usr/bin/env python3
"""Debug script to compare retrieval modes BEFORE reranking."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.retrievers.milvus_retriever import connect
from app.retrievers.milvus_retriever import _dense_search, _sparse_search, _hybrid_search
from app.retrievers.milvus_retriever import encode_query, encode_query_sparse, encode_query_hybrid
from pymilvus import Collection

def test_modes_without_reranker():
    """Test three modes without reranker to see the difference."""
    
    print("Connecting to Milvus...")
    connect()
    
    course_id = "course-ai-001"
    query = "什么是贝叶斯分类器"
    k = 5
    
    col_name = f"kb_{course_id}".replace("-", "_")
    col = Collection(col_name)
    col.load()
    
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    # Test DENSE mode (without reranker)
    print("【DENSE Mode - Before Reranker】")
    print("-" * 70)
    query_vec = encode_query(query)
    dense_hits = _dense_search(col, query_vec, k, expr=None)
    
    for rank, (chunk_id, (score, entity)) in enumerate(
        sorted(dense_hits.items(), key=lambda x: x[1][0], reverse=True), 1
    ):
        source_type = entity.get('source_type', 'N/A')
        doc_title = entity.get('doc_title', 'N/A')[:30]
        print(f"{rank}. [{chunk_id}] score={score:.4f} | type={source_type:4s} | {doc_title}")
    
    # Test SPARSE mode (without reranker)
    print(f"\n【SPARSE Mode - Before Reranker】")
    print("-" * 70)
    query_sparse = encode_query_sparse(query)
    sparse_hits = _sparse_search(col, query_sparse, k, expr=None)
    
    for rank, (chunk_id, (score, entity)) in enumerate(
        sorted(sparse_hits.items(), key=lambda x: x[1][0], reverse=True), 1
    ):
        source_type = entity.get('source_type', 'N/A')
        doc_title = entity.get('doc_title', 'N/A')[:30]
        print(f"{rank}. [{chunk_id}] score={score:.4f} | type={source_type:4s} | {doc_title}")
    
    # Test HYBRID mode (without reranker)
    print(f"\n【HYBRID Mode - Before Reranker】")
    print("-" * 70)
    query_dense, query_sparse = encode_query_hybrid(query)
    hybrid_hits = _hybrid_search(col, query_dense, query_sparse, k, expr=None, alpha=0.7)
    
    for rank, (chunk_id, (score, entity)) in enumerate(
        sorted(hybrid_hits.items(), key=lambda x: x[1][0], reverse=True), 1
    ):
        source_type = entity.get('source_type', 'N/A')
        doc_title = entity.get('doc_title', 'N/A')[:30]
        print(f"{rank}. [{chunk_id}] score={score:.4f} | type={source_type:4s} | {doc_title}")
    
    # Compare top-1 results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print("="*70)
    
    dense_top1 = list(dense_hits.keys())[0] if dense_hits else None
    sparse_top1 = list(sparse_hits.keys())[0] if sparse_hits else None
    hybrid_top1 = list(hybrid_hits.keys())[0] if hybrid_hits else None
    
    print(f"Dense Top-1:  {dense_top1}")
    print(f"Sparse Top-1: {sparse_top1}")
    print(f"Hybrid Top-1: {hybrid_top1}")
    
    if dense_top1 == sparse_top1 == hybrid_top1:
        print("\n⚠️  WARNING: All three modes return the same top-1 result!")
        print("   This suggests the dataset is too small or queries are too simple.")
    else:
        print("\n✅ GOOD: Different modes produce different results.")
    
    # Check overlap
    dense_set = set(list(dense_hits.keys())[:k])
    sparse_set = set(list(sparse_hits.keys())[:k])
    hybrid_set = set(list(hybrid_hits.keys())[:k])
    
    overlap_ds = len(dense_set & sparse_set)
    overlap_dh = len(dense_set & hybrid_set)
    overlap_sh = len(sparse_set & hybrid_set)
    
    print(f"\nTop-{k} Overlap:")
    print(f"  Dense ∩ Sparse:  {overlap_ds}/{k}")
    print(f"  Dense ∩ Hybrid:  {overlap_dh}/{k}")
    print(f"  Sparse ∩ Hybrid: {overlap_sh}/{k}")

if __name__ == "__main__":
    test_modes_without_reranker()
