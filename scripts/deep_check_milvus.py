"""Deep inspection of Milvus vector database"""
from pymilvus import MilvusClient
import json

client = MilvusClient(uri='http://localhost:19530/learnthink')

print("="*80)
print("DEEP MILVUS DATABASE INSPECTION")
print("="*80)

# ============================================================================
# 1. Collection Index Details
# ============================================================================
print("\n[1] INDEX DETAILS:")
print("-" * 80)

collection_name = 'kb_course_ai_001'
indexes = client.list_indexes(collection_name)
print(f"   Total indexes: {len(indexes)}")

for index_name in indexes:
    index_info = client.describe_index(collection_name, index_name)
    print(f"\n   Index: {index_name}")
    print(f"     Field name: {index_info['field_name']}")
    print(f"     Index type: {index_info['index_type']}")
    print(f"     Metric type: {index_info['metric_type']}")
    
    # Get index params
    params = index_info.get('params', {})
    if params:
        print(f"     Parameters:")
        for k, v in params.items():
            print(f"       {k}: {v}")
    
    # Check index status
    try:
        index_progress = client.index_building_progress(
            collection_name=collection_name,
            index_name=index_name
        )
        print(f"     Building progress: {index_progress}")
    except Exception as e:
        print(f"     Progress check: N/A ({e})")

# ============================================================================
# 2. Data Integrity Checks
# ============================================================================
print("\n\n[2] DATA INTEGRITY CHECKS:")
print("-" * 80)

# Check all records
all_records = client.query(
    collection_name=collection_name,
    filter='',
    output_fields=['chunk_id', 'doc_id', 'text', 'embedding', 'sparse_vector', 
                   'topic', 'source_type', 'heading_path'],
    limit=300
)

total_count = len(all_records)
print(f"   Total records queried: {total_count}")

# Check for NULL/empty values
null_checks = {
    'chunk_id': 0,
    'doc_id': 0,
    'text': 0,
    'embedding': 0,
    'sparse_vector': 0,
    'topic': 0,
    'source_type': 0,
}

for record in all_records:
    for field in null_checks.keys():
        value = record.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ''):
            if field not in ['topic']:  # topic can be empty
                null_checks[field] += 1

print(f"\n   NULL/Empty checks:")
for field, count in null_checks.items():
    status = "⚠ WARNING" if count > 0 else "✓ OK"
    print(f"     {status} {field:20s}: {count} empty/null")

# Check embedding validity
print(f"\n   Embedding validation:")
invalid_embeddings = 0
wrong_dims = 0
nan_values = 0
inf_values = 0

for record in all_records:
    emb = record.get('embedding', [])
    if not emb:
        invalid_embeddings += 1
        continue
    
    if len(emb) != 1024:
        wrong_dims += 1
    
    # Check for NaN or Inf
    import math
    for val in emb:
        if math.isnan(val):
            nan_values += 1
            break
        if math.isinf(val):
            inf_values += 1
            break

print(f"     ✓ Valid embeddings: {total_count - invalid_embeddings}/{total_count}")
if invalid_embeddings > 0:
    print(f"     ⚠ Invalid (empty) embeddings: {invalid_embeddings}")
if wrong_dims > 0:
    print(f"     ✗ Wrong dimensions: {wrong_dims}")
if nan_values > 0:
    print(f"     ✗ NaN values found: {nan_values}")
if inf_values > 0:
    print(f"     ✗ Inf values found: {inf_values}")

# Check sparse vector validity
print(f"\n   Sparse vector validation:")
invalid_sparse = 0
empty_sparse = 0

for record in all_records:
    sv = record.get('sparse_vector', {})
    if sv is None:
        invalid_sparse += 1
    elif len(sv) == 0:
        empty_sparse += 1

print(f"     ✓ Valid sparse vectors: {total_count - invalid_sparse}/{total_count}")
if empty_sparse > 0:
    print(f"     ℹ Empty sparse vectors: {empty_sparse}")
if invalid_sparse > 0:
    print(f"     ✗ Invalid sparse vectors: {invalid_sparse}")

# ============================================================================
# 3. Vector Quality Analysis
# ============================================================================
print("\n\n[3] VECTOR QUALITY ANALYSIS:")
print("-" * 80)

import statistics

dense_vectors = [r['embedding'] for r in all_records if r.get('embedding')]
sparse_vectors = [r['sparse_vector'] for r in all_records if r.get('sparse_vector')]

# Dense vector statistics
print(f"\n   Dense Vector Statistics (sample of {len(dense_vectors)}):")
dims = [len(v) for v in dense_vectors]
print(f"     Dimension: min={min(dims)}, max={max(dims)}, all_same={len(set(dims))==1}")

# Calculate norms
norms = []
for vec in dense_vectors[:50]:  # Sample first 50
    norm = sum(x**2 for x in vec) ** 0.5
    norms.append(norm)

if norms:
    print(f"     L2 Norm (sample 50):")
    print(f"       Mean: {statistics.mean(norms):.6f}")
    print(f"       Std:  {statistics.stdev(norms):.6f}")
    print(f"       Min:  {min(norms):.6f}")
    print(f"       Max:  {max(norms):.6f}")
    print(f"       {'✓ Normalized' if abs(statistics.mean(norms) - 1.0) < 0.1 else '✗ Not normalized'}")

# Value distribution
all_values = []
for vec in dense_vectors[:20]:
    all_values.extend(vec)

print(f"\n     Value Distribution (sample):")
print(f"       Mean:   {statistics.mean(all_values):.6f}")
print(f"       Std:    {statistics.stdev(all_values):.6f}")
print(f"       Median: {statistics.median(all_values):.6f}")
print(f"       Min:    {min(all_values):.6f}")
print(f"       Max:    {max(all_values):.6f}")

# Sparse vector statistics
print(f"\n   Sparse Vector Statistics:")
sparse_sizes = [len(sv) for sv in sparse_vectors]
print(f"     Non-zero elements per vector:")
print(f"       Mean: {statistics.mean(sparse_sizes):.1f}")
print(f"       Median: {statistics.median(sparse_sizes):.1f}")
print(f"       Min: {min(sparse_sizes)}")
print(f"       Max: {max(sparse_sizes)}")
print(f"       Std: {statistics.stdev(sparse_sizes):.1f}")

# Weight distribution
all_weights = []
for sv in sparse_vectors[:20]:
    all_weights.extend(sv.values())

if all_weights:
    print(f"\n     Weight Distribution (sample):")
    print(f"       Mean:   {statistics.mean(all_weights):.6f}")
    print(f"       Median: {statistics.median(all_weights):.6f}")
    print(f"       Min:    {min(all_weights):.6f}")
    print(f"       Max:    {max(all_weights):.6f}")

# ============================================================================
# 4. Document Coverage Analysis
# ============================================================================
print("\n\n[4] DOCUMENT COVERAGE ANALYSIS:")
print("-" * 80)

from collections import Counter

doc_chunks = Counter([r['doc_id'] for r in all_records])
print(f"   Total unique documents: {len(doc_chunks)}")
print(f"   Total chunks: {sum(doc_chunks.values())}")

print(f"\n   Chunks per document:")
for doc_id in sorted(doc_chunks.keys()):
    count = doc_chunks[doc_id]
    percentage = (count / total_count) * 100
    bar_len = int(percentage / 2)
    bar = '█' * bar_len
    print(f"     {doc_id:45s} {count:3d} ({percentage:5.1f}%) {bar}")

# Source type distribution
source_types = Counter([r.get('source_type', 'unknown') for r in all_records])
print(f"\n   Source type distribution:")
for stype, count in source_types.most_common():
    percentage = (count / total_count) * 100
    print(f"     {stype:20s}: {count:3d} ({percentage:5.1f}%)")

# Topic coverage
topics = [r.get('topic', '') for r in all_records]
topics_with_value = [t for t in topics if t]
topics_empty = len(topics) - len(topics_with_value)

print(f"\n   Topic coverage:")
print(f"     With topic: {len(topics_with_value)} ({len(topics_with_value)/len(topics)*100:.1f}%)")
print(f"     Empty topic: {topics_empty} ({topics_empty/len(topics)*100:.1f}%)")

unique_topics = set(topics_with_value)
print(f"     Unique topics: {len(unique_topics)}")

# ============================================================================
# 5. Text Content Analysis
# ============================================================================
print("\n\n[5] TEXT CONTENT ANALYSIS:")
print("-" * 80)

text_lengths = [len(r.get('text', '')) for r in all_records]
print(f"   Text length statistics:")
print(f"     Mean: {statistics.mean(text_lengths):.0f} chars")
print(f"     Median: {statistics.median(text_lengths):.0f} chars")
print(f"     Min: {min(text_lengths)} chars")
print(f"     Max: {max(text_lengths)} chars")
print(f"     Std: {statistics.stdev(text_lengths):.0f} chars")

# Check for very short or very long texts
short_texts = sum(1 for l in text_lengths if l < 50)
long_texts = sum(1 for l in text_lengths if l > 2000)
print(f"\n     Very short (<50 chars): {short_texts}")
print(f"     Very long (>2000 chars): {long_texts}")

# ============================================================================
# 6. Chunk ID Format Validation
# ============================================================================
print("\n\n[6] CHUNK ID FORMAT VALIDATION:")
print("-" * 80)

chunk_ids = [r['chunk_id'] for r in all_records]
valid_format = 0
invalid_format = 0

for cid in chunk_ids:
    if '#' in cid and len(cid.split('#')) == 2:
        parts = cid.split('#')
        if parts[0] and parts[1].isdigit():
            valid_format += 1
        else:
            invalid_format += 1
    else:
        invalid_format += 1

print(f"   Valid format (doc_id#index): {valid_format}/{len(chunk_ids)}")
if invalid_format > 0:
    print(f"   Invalid format: {invalid_format}")
    # Show examples
    invalid_examples = [cid for cid in chunk_ids if '#' not in cid or len(cid.split('#')) != 2]
    for ex in invalid_examples[:5]:
        print(f"     Example: '{ex}'")

# Check for duplicates
if len(chunk_ids) != len(set(chunk_ids)):
    duplicates = [cid for cid in chunk_ids if chunk_ids.count(cid) > 1]
    print(f"   ⚠ DUPLICATE chunk_ids found: {len(set(duplicates))}")
    for dup in set(duplicates)[:5]:
        print(f"     '{dup}' appears {chunk_ids.count(dup)} times")
else:
    print(f"   ✓ No duplicate chunk_ids")

# ============================================================================
# 7. Sample Records Display
# ============================================================================
print("\n\n[7] SAMPLE RECORDS (3 random):")
print("-" * 80)

import random
sample_indices = random.sample(range(len(all_records)), min(3, len(all_records)))

for idx in sample_indices:
    record = all_records[idx]
    print(f"\n   Record #{idx+1}:")
    print(f"     chunk_id:    {record['chunk_id']}")
    print(f"     doc_id:      {record['doc_id']}")
    print(f"     source_type: {record.get('source_type', 'N/A')}")
    print(f"     topic:       '{record.get('topic', '')}'")
    print(f"     text_len:    {len(record.get('text', ''))} chars")
    print(f"     dense_dim:   {len(record.get('embedding', []))}")
    print(f"     sparse_size: {len(record.get('sparse_vector', {}))}")
    print(f"     text_preview: {record.get('text', '')[:80]}...")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "="*80)
print("INSPECTION SUMMARY")
print("="*80)

issues = []
if invalid_embeddings > 0:
    issues.append(f"⚠ {invalid_embeddings} invalid embeddings")
if wrong_dims > 0:
    issues.append(f"✗ {wrong_dims} embeddings with wrong dimensions")
if nan_values > 0:
    issues.append(f"✗ {nan_values} embeddings contain NaN")
if inf_values > 0:
    issues.append(f"✗ {inf_values} embeddings contain Inf")
if invalid_format > 0:
    issues.append(f"⚠ {invalid_format} chunk_ids with invalid format")
if short_texts > 0:
    issues.append(f"ℹ {short_texts} very short texts (<50 chars)")

if issues:
    print("\nIssues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ No critical issues found!")

print(f"\nDatabase Status: {'READY' if not any('✗' in i for i in issues) else 'NEEDS ATTENTION'}")
print("="*80)
