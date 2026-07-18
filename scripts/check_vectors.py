"""Query vector data from Milvus"""
from pymilvus import MilvusClient

client = MilvusClient(uri='http://localhost:19530/learnthink')

print("="*80)
print("Vector Data Verification")
print("="*80)

# Query one record with vectors
print("\n[1] Sample Record with Vectors:")
results = client.query(
    collection_name='kb_course_ai_001',
    filter='doc_id == "ch02_搜索算法"',
    output_fields=['chunk_id', 'doc_id', 'text', 'embedding', 'sparse_vector'],
    limit=1
)

if results:
    record = results[0]
    print(f"\n   chunk_id: {record['chunk_id']}")
    print(f"   doc_id: {record['doc_id']}")
    print(f"   text (first 100 chars): {record['text'][:100]}...")
    
    # Check dense vector
    dense_vec = record.get('embedding', [])
    print(f"\n   Dense Vector:")
    print(f"     Dimension: {len(dense_vec)}")
    print(f"     First 10 values: {[f'{v:.6f}' for v in dense_vec[:10]]}")
    print(f"     Min: {min(dense_vec):.6f}, Max: {max(dense_vec):.6f}")
    print(f"     Mean: {sum(dense_vec)/len(dense_vec):.6f}")
    
    # Check sparse vector
    sparse_vec = record.get('sparse_vector', {})
    print(f"\n   Sparse Vector:")
    print(f"     Non-zero elements: {len(sparse_vec)}")
    if sparse_vec:
        sample_items = list(sparse_vec.items())[:5]
        print(f"     Sample (token_id: weight):")
        for token_id, weight in sample_items:
            print(f"       {token_id}: {weight:.6f}")

# Count by document
print("\n[2] Chunk Distribution by Document:")
all_docs = client.query(
    collection_name='kb_course_ai_001',
    filter='',
    output_fields=['doc_id'],
    limit=300
)

doc_counts = {}
for r in all_docs:
    doc_id = r['doc_id']
    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

print(f"   Total documents: {len(doc_counts)}")
print(f"   Total chunks: {sum(doc_counts.values())}")
print(f"\n   Chunks per document:")
for doc_id in sorted(doc_counts.keys()):
    count = doc_counts[doc_id]
    bar = '█' * (count // 2)
    print(f"     {doc_id:40s} {count:3d} chunks {bar}")

# Verify vector dimensions
print("\n[3] Vector Dimension Verification:")
sample_records = client.query(
    collection_name='kb_course_ai_001',
    filter='',
    output_fields=['embedding', 'sparse_vector'],
    limit=10
)

dense_dims = [len(r['embedding']) for r in sample_records]
sparse_sizes = [len(r['sparse_vector']) for r in sample_records]

print(f"   Dense vector dimensions (sample of {len(sample_records)}):")
print(f"     All 1024-dim: {all(d == 1024 for d in dense_dims)}")
print(f"     Unique dims: {set(dense_dims)}")

print(f"\n   Sparse vector sizes (sample of {len(sample_records)}):")
print(f"     Min non-zero: {min(sparse_sizes)}")
print(f"     Max non-zero: {max(sparse_sizes)}")
print(f"     Avg non-zero: {sum(sparse_sizes)/len(sparse_sizes):.1f}")

print("\n" + "="*80)
print("Vector verification completed!")
print("="*80)
