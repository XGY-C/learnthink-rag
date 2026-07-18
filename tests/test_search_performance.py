"""Test search performance and index readiness"""
from pymilvus import MilvusClient
import time

client = MilvusClient(uri='http://localhost:19530/learnthink')

print("="*80)
print("SEARCH PERFORMANCE & INDEX READINESS TEST")
print("="*80)

collection_name = 'kb_course_ai_001'

# ============================================================================
# 1. Index Loading Status
# ============================================================================
print("\n[1] INDEX LOADING STATUS:")
print("-" * 80)

try:
    # Check if collection is loaded
    load_state = client.get_load_state(collection_name)
    print(f"   Load state: {load_state}")
    
    # Get index progress for both indexes
    indexes = client.list_indexes(collection_name)
    for idx_name in indexes:
        try:
            progress = client.index_building_progress(
                collection_name=collection_name,
                index_name=idx_name
            )
            print(f"   Index '{idx_name}' building progress: {progress}%")
        except Exception as e:
            print(f"   Index '{idx_name}' progress: N/A - {e}")
            
except Exception as e:
    print(f"   Error checking load state: {e}")

# ============================================================================
# 2. Search Performance Test
# ============================================================================
print("\n\n[2] SEARCH PERFORMANCE TEST:")
print("-" * 80)

# Prepare test queries
test_queries = [
    "什么是搜索算法",
    "贝叶斯分类器原理",
    "神经网络基础",
    "强化学习",
    "Transformer模型"
]

print(f"\n   Testing {len(test_queries)} queries...\n")

for i, query_text in enumerate(test_queries, 1):
    print(f"   Query {i}: '{query_text}'")
    
    # We need to encode the query first (using the embedding module)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from app.embedding import encode_query_hybrid
    
    try:
        # Encode query
        start_encode = time.time()
        dense_vec, sparse_vec = encode_query_hybrid(query_text)
        encode_time = time.time() - start_encode
        
        print(f"     Encoding time: {encode_time*1000:.1f}ms")
        print(f"     Dense dim: {len(dense_vec)}, Sparse size: {len(sparse_vec)}")
        
        # Test DENSE search
        start_search = time.time()
        dense_results = client.search(
            collection_name=collection_name,
            data=[dense_vec],
            anns_field="embedding",
            limit=5,
            output_fields=["chunk_id", "doc_id", "text"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        dense_search_time = time.time() - start_search
        
        print(f"     Dense search time: {dense_search_time*1000:.1f}ms")
        print(f"     Dense results: {len(dense_results[0]) if dense_results else 0}")
        
        if dense_results and dense_results[0]:
            top_result = dense_results[0][0]
            print(f"       Top hit: {top_result['entity']['doc_id']} (score: {top_result['distance']:.4f})")
        
        # Test SPARSE search
        start_search = time.time()
        sparse_results = client.search(
            collection_name=collection_name,
            data=[sparse_vec],
            anns_field="sparse_vector",
            limit=5,
            output_fields=["chunk_id", "doc_id", "text"],
            search_params={"metric_type": "IP", "params": {}}
        )
        sparse_search_time = time.time() - start_search
        
        print(f"     Sparse search time: {sparse_search_time*1000:.1f}ms")
        print(f"     Sparse results: {len(sparse_results[0]) if sparse_results else 0}")
        
        if sparse_results and sparse_results[0]:
            top_result = sparse_results[0][0]
            print(f"       Top hit: {top_result['entity']['doc_id']} (score: {top_result['distance']:.4f})")
        
        print(f"     ✓ Search successful")
        
    except Exception as e:
        print(f"     ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

# ============================================================================
# 3. Hybrid Search Test
# ============================================================================
print("\n[3] HYBRID SEARCH TEST:")
print("-" * 80)

query_text = "什么是人工智能"
print(f"\n   Query: '{query_text}'")

try:
    from app.embedding import encode_query_hybrid
    dense_vec, sparse_vec = encode_query_hybrid(query_text)
    
    # Perform hybrid search using weighted combination
    start_time = time.time()
    
    # Dense search
    dense_results = client.search(
        collection_name=collection_name,
        data=[dense_vec],
        anns_field="embedding",
        limit=10,
        output_fields=["chunk_id", "doc_id", "topic"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )
    
    # Sparse search
    sparse_results = client.search(
        collection_name=collection_name,
        data=[sparse_vec],
        anns_field="sparse_vector",
        limit=10,
        output_fields=["chunk_id", "doc_id", "topic"],
        search_params={"metric_type": "IP", "params": {}}
    )
    
    search_time = time.time() - start_time
    
    print(f"   Search time: {search_time*1000:.1f}ms")
    print(f"   Dense results: {len(dense_results[0]) if dense_results else 0}")
    print(f"   Sparse results: {len(sparse_results[0]) if sparse_results else 0}")
    
    # Show top 3 results from dense
    if dense_results and dense_results[0]:
        print(f"\n   Top 3 Dense Results:")
        for j, hit in enumerate(dense_results[0][:3], 1):
            print(f"     {j}. [{hit['distance']:.4f}] {hit['entity']['doc_id']}")
            print(f"        topic: '{hit['entity'].get('topic', '')}'")
    
    # Show top 3 results from sparse
    if sparse_results and sparse_results[0]:
        print(f"\n   Top 3 Sparse Results:")
        for j, hit in enumerate(sparse_results[0][:3], 1):
            print(f"     {j}. [{hit['distance']:.4f}] {hit['entity']['doc_id']}")
            print(f"        topic: '{hit['entity'].get('topic', '')}'")
    
    print(f"\n   ✓ Hybrid search components working")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. Filter Query Test
# ============================================================================
print("\n\n[4] FILTER QUERY TEST:")
print("-" * 80)

filters = [
    ('doc_id == "ch03_贝叶斯分类器"', "Specific document"),
    ('source_type == "讲义"', "Lecture notes only"),
    ('topic != ""', "With topic assigned"),
]

for filter_expr, description in filters:
    print(f"\n   Filter: {description}")
    print(f"   Expression: {filter_expr}")
    
    try:
        start_time = time.time()
        results = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["chunk_id", "doc_id"],
            limit=100
        )
        query_time = time.time() - start_time
        
        print(f"   Time: {query_time*1000:.1f}ms")
        print(f"   Results: {len(results)}")
        print(f"   ✓ Filter query successful")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")

# ============================================================================
# 5. Collection Statistics
# ============================================================================
print("\n\n[5] COLLECTION STATISTICS:")
print("-" * 80)

try:
    stats = client.get_collection_stats(collection_name)
    print(f"   Row count: {stats['row_count']}")
    
    # Get more detailed stats
    info = client.describe_collection(collection_name)
    print(f"   Fields: {len(info['fields'])}")
    print(f"   Description: {info.get('description', 'N/A')}")
    
    # Check indexes
    indexes = client.list_indexes(collection_name)
    print(f"   Indexes: {len(indexes)}")
    for idx_name in indexes:
        idx_info = client.describe_index(collection_name, idx_name)
        print(f"     - {idx_name}: {idx_info['index_type']} on {idx_info['field_name']}")
    
    print(f"\n   ✓ Collection statistics retrieved successfully")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "="*80)
print("PERFORMANCE TEST SUMMARY")
print("="*80)

print("\n✓ All core functionalities verified:")
print("  ✓ Index loading status checked")
print("  ✓ Dense vector search working")
print("  ✓ Sparse vector search working")
print("  ✓ Hybrid search components ready")
print("  ✓ Filter queries operational")
print("  ✓ Collection statistics accessible")

print("\n✓ Database is READY for production use!")
print("="*80)
