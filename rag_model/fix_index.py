from rag_model.rag_utils import s3vectors_client, S3_VECTORS_BUCKET_NAME, S3_VECTORS_INDEX_NAME
from botocore.exceptions import ClientError

print(f"Clearing ALL old vectors from index '{S3_VECTORS_INDEX_NAME}' (keeping the correct index structure)...")

try:
    keys_to_delete = []
    next_token = None

    while True:
        query_kwargs = {
            "vectorBucketName": S3_VECTORS_BUCKET_NAME,
            "indexName": S3_VECTORS_INDEX_NAME,
            "queryVector": {"float32": [0.0] * 1024},
            "topK": 100,
        }
        if next_token:
            query_kwargs["nextToken"] = next_token

        resp = s3vectors_client.query_vectors(**query_kwargs)
        batch = [item["key"] for item in resp.get("vectors", [])]
        keys_to_delete.extend(batch)

        next_token = resp.get("nextToken")
        if not next_token:
            break

    if keys_to_delete:
        for i in range(0, len(keys_to_delete), 100):
            batch = keys_to_delete[i:i+100]
            s3vectors_client.delete_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                keys=batch
            )
        print(f"Deleted {len(keys_to_delete)} old vectors — index is now clean!")
    else:
        print("No old vectors found — already clean.")

except ClientError as e:
    if e.response['Error']['Code'] in ['NotFoundException', 'ValidationException']:
        print("Index is empty or still initializing — nothing to delete.")
    else:
        print("Unexpected error:", e)
        raise

print("Ready for fresh reindexing with correct filterable tenant_id!")