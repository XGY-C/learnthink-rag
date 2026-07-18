"""Export all records from a Milvus collection to JSON for migration/copy.

Milvus import expects: a JSON array of objects, each object = one record.
sparse_vector must use {"indices": [...], "values": [...]} format.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient

SOURCE_URI = "http://localhost:19530/learnthink"
SOURCE_COLLECTION = "kb_course_ai_001"
OUTPUT_FILE = Path(__file__).resolve().parent / "export_kb_course_ai_001.json"

BATCH_SIZE = 100


def _convert_sparse(sv) -> dict | None:
    """Convert sparse vector to Milvus import format: {indices: [...], values: [...]}"""
    if sv is None:
        return None
    if isinstance(sv, dict):
        indices = []
        values = []
        for k, v in sv.items():
            indices.append(int(k))
            values.append(float(v))
        return {"indices": indices, "values": values}
    # If it's already in the right format
    if isinstance(sv, dict) and "indices" in sv and "values" in sv:
        return sv
    return None


def export():
    client = MilvusClient(uri=SOURCE_URI)

    if not client.has_collection(SOURCE_COLLECTION):
        print(f"ERROR: Collection '{SOURCE_COLLECTION}' not found!")
        return

    info = client.describe_collection(SOURCE_COLLECTION)
    field_names = [f["name"] for f in info["fields"]]
    print(f"Collection: {SOURCE_COLLECTION}")
    print(f"Fields: {field_names}")

    stats = client.get_collection_stats(SOURCE_COLLECTION)
    total = int(stats["row_count"])
    print(f"Total records: {total}")

    # Query all fields including sparse_vector
    output_fields = [f for f in field_names]

    all_records = []
    offset = 0
    while offset < total:
        batch = client.query(
            collection_name=SOURCE_COLLECTION,
            filter="",
            output_fields=output_fields,
            limit=BATCH_SIZE,
            offset=offset,
        )
        all_records.extend(batch)
        print(f"  Fetched {len(all_records)}/{total} records...")
        offset += BATCH_SIZE

    print(f"\nTotal exported: {len(all_records)} records")

    # Convert data for Milvus import format
    export_records = []
    for rec in all_records:
        row = {}
        for k, v in rec.items():
            if k == "embedding":
                # Dense vector -> list of floats
                row[k] = v.tolist() if hasattr(v, "tolist") else list(v)
            elif k == "sparse_vector":
                # Sparse vector -> {indices: [...], values: [...]}
                row[k] = _convert_sparse(v)
            else:
                row[k] = v
        export_records.append(row)

    # Write as plain JSON array (Milvus import format)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(export_records, f, ensure_ascii=False)

    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"Exported to: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"Format: JSON array of {len(export_records)} records (Milvus import compatible)")


if __name__ == "__main__":
    export()
