"""Direct Milvus database query script"""
from pymilvus import MilvusClient

# Connect to Milvus
client = MilvusClient(uri='http://localhost:19530/learnthink')

print("="*80)
print("Direct Milvus Database Query")
print("="*80)

# List all collections
print("\n[1] Collection List:")
collections = client.list_collections()
print(f"   Total collections: {len(collections)}")
for col in collections:
    print(f"   - {col}")

# Check if our collection exists
if 'kb_course_ai_001' not in collections:
    print("\nERROR: Collection 'kb_course_ai_001' not found!")
    exit(1)

# Describe collection
print("\n[2] Collection Details (kb_course_ai_001):")
info = client.describe_collection('kb_course_ai_001')
print(f"   Collection name: {info['collection_name']}")
print(f"   Description: {info.get('description', 'N/A')}")
print(f"   Fields count: {len(info['fields'])}")

print("\n   Schema Fields:")
for field in info['fields']:
    field_name = field['name']
    field_type = field['type']
    dim = field.get('params', {}).get('dim', 'N/A')
    is_primary = field.get('is_primary', False)
    print(f"   - {field_name:20s} type={field_type:3d} dim={str(dim):6s} primary={is_primary}")

# Get entity count
stats = client.get_collection_stats('kb_course_ai_001')
print(f"\n[3] Entity Count: {stats['row_count']}")

# Query sample data
print("\n[4] Sample Data (first 3 records):")
results = client.query(
    collection_name='kb_course_ai_001',
    filter='',
    output_fields=['chunk_id', 'doc_id', 'doc_title', 'topic', 'source_type'],
    limit=3
)

for i, record in enumerate(results, 1):
    print(f"\n   Record {i}:")
    print(f"     chunk_id:    {record.get('chunk_id', 'N/A')}")
    print(f"     doc_id:      {record.get('doc_id', 'N/A')}")
    print(f"     doc_title:   {record.get('doc_title', 'N/A')}")
    print(f"     topic:       '{record.get('topic', '')}'")
    print(f"     source_type: {record.get('source_type', 'N/A')}")

# Query by specific doc_id
print("\n[5] Query chunks for 'ch03_贝叶斯分类器':")
results = client.query(
    collection_name='kb_course_ai_001',
    filter='doc_id == "ch03_贝叶斯分类器"',
    output_fields=['chunk_id', 'doc_id', 'topic'],
    limit=5
)
print(f"   Found {len(results)} chunks")
for i, record in enumerate(results, 1):
    print(f"   {i}. chunk_id={record['chunk_id']}, topic='{record.get('topic', '')}'")

# Query by topic containing "贝叶斯"
print("\n[6] Query chunks with topic containing '贝叶斯':")
results = client.query(
    collection_name='kb_course_ai_001',
    filter='topic like "%贝叶斯%"',
    output_fields=['chunk_id', 'doc_id', 'topic'],
    limit=10
)
print(f"   Found {len(results)} chunks")
for i, record in enumerate(results, 1):
    print(f"   {i}. chunk_id={record['chunk_id'][:40]:40s} doc={record['doc_id']:30s} topic='{record.get('topic', '')}'")

# Show unique topics
print("\n[7] Unique Topics (sample):")
results = client.query(
    collection_name='kb_course_ai_001',
    filter='',
    output_fields=['topic'],
    limit=50
)
topics = set([r.get('topic', '') for r in results])
print(f"   Total unique topics in sample: {len(topics)}")
sorted_topics = sorted([t for t in topics if t], key=len, reverse=True)[:15]
for i, topic in enumerate(sorted_topics, 1):
    print(f"   {i:2d}. '{topic}'")

print("\n" + "="*80)
print("Query completed successfully!")
print("="*80)
