"""Import exported Milvus data into a target collection."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

SOURCE_URI = "http://localhost:19530/learnthink"
EXPORT_FILE = Path(__file__).resolve().parent / "export_kb_course_ai_001.json"
BATCH_SIZE = 50  # Milvus insert batch size


def create_collection(client: MilvusClient, name: str):
    """Create collection with the same schema as source."""
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="book_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="chapter_index", dtype=DataType.INT64),
        FieldSchema(name="chapter_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="heading_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="locator", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields, description="Migrated collection")
    client.create_collection(
        collection_name=name,
        schema=schema,
    )
    # Create indexes
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )
    client.create_index(name, index_params)
    print(f"Created collection '{name}' with indexes")


def import_data(target_collection: str, drop_existing: bool = True):
    client = MilvusClient(uri=SOURCE_URI)

    # Load export data
    with open(EXPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    print(f"Loaded {len(records)} records from export file")

    # Handle target collection
    if client.has_collection(target_collection):
        if drop_existing:
            client.drop_collection(target_collection)
            print(f"Dropped existing collection '{target_collection}'")
        else:
            print(f"Collection '{target_collection}' already exists. Use --drop to recreate.")
            return

    create_collection(client, target_collection)

    # Convert sparse vectors back
    for rec in records:
        if "sparse_vector" in rec and rec["sparse_vector"] is not None:
            sv = rec["sparse_vector"]
            # Convert string keys back to int
            rec["sparse_vector"] = {int(k): v for k, v in sv.items()}

    # Insert in batches
    total = len(records)
    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        client.insert(collection_name=target_collection, data=batch)
        inserted += len(batch)
        print(f"  Inserted {inserted}/{total} records...")

    print(f"\nDone! {inserted} records imported into '{target_collection}'")

    # Verify
    stats = client.get_collection_stats(target_collection)
    print(f"Verification: collection has {stats['row_count']} records")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import Milvus export data")
    parser.add_argument(
        "-c", "--collection",
        default="kb_course_ai_002",
        help="Target collection name (default: kb_course_ai_002)",
    )
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Don't drop existing collection",
    )
    args = parser.parse_args()

    import_data(args.collection, drop_existing=not args.no_drop)
