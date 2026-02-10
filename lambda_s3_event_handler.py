"""
AWS Lambda function to handle automatic indexing via SQS messages
This function is deployed to AWS Lambda and triggered by SQS queue messages
"""
import json
import os
import sys
import boto3
import tempfile
import shutil
import uuid
from typing import Optional, List
from botocore.exceptions import ClientError

# AWS Configuration
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = os.getenv("S3_VECTORS_BUCKET_NAME", "rag-vectordb-bucket")
S3_VECTORS_INDEX_NAME = os.getenv("S3_VECTORS_INDEX_NAME", "tenant-knowledge-index")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
S3_PREFIX_KNOWLEDGE = "knowledge_base"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION_NAME = "ap-south-1"

# Initialize AWS clients
print("Initializing AWS clients...")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
s3_client = boto3.client("s3", region_name=REGION_NAME)
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)

# Import embeddings
try:
    from langchain_aws import BedrockEmbeddings
    from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader, JSONLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)
    print("✅ LangChain imports successful")
except Exception as e:
    print(f"❌ Error importing LangChain: {e}")
    embeddings = None


def ensure_vector_index():
    """Create vector index if it doesn't exist"""
    try:
        # Test if index exists
        s3vectors_client.put_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            vectors=[{
                "key": "ping",
                "data": {"float32": [0.0] * 1024},
                "metadata": {"tenant_id": "0"}
            }]
        )
        s3vectors_client.delete_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            keys=["ping"]
        )
        print(f"✅ Index '{S3_VECTORS_INDEX_NAME}' exists")
        return
    except ClientError as e:
        if e.response['Error']['Code'] not in ['NotFoundException', 'ValidationException']:
            raise
    
    # Create index
    print(f"Creating index '{S3_VECTORS_INDEX_NAME}'...")
    s3vectors_client.create_index(
        vectorBucketName=S3_VECTORS_BUCKET_NAME,
        indexName=S3_VECTORS_INDEX_NAME,
        dataType="float32",
        dimension=1024,
        distanceMetric="cosine",
        metadataConfiguration={"nonFilterableMetadataKeys": ["internal_id"]}
    )
    print("✅ Index created")


def delete_tenant_vectors(tenant_id: int):
    """Delete all vectors for a tenant"""
    try:
        keys_to_delete = []
        next_token = None
        
        while True:
            query_kwargs = {
                "vectorBucketName": S3_VECTORS_BUCKET_NAME,
                "indexName": S3_VECTORS_INDEX_NAME,
                "queryVector": {"float32": [0.0] * 1024},
                "topK": 30,
                "filter": {"tenant_id": {"eq": str(tenant_id)}},
                "returnMetadata": False
            }
            if next_token:
                query_kwargs["nextToken"] = next_token
            
            resp = s3vectors_client.query_vectors(**query_kwargs)
            batch_keys = [item["key"] for item in resp.get("vectors", [])]
            keys_to_delete.extend(batch_keys)
            
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
            print(f"✅ Deleted {len(keys_to_delete)} vectors for tenant {tenant_id}")
    except Exception as e:
        print(f"⚠️ Delete vectors error: {e}")


def s3_list_tenant_files(tenant_id: int) -> List[str]:
    """List all files for a tenant in S3"""
    prefix = f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/"
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/urls.txt'):
                keys.append(obj['Key'])
    return keys


def index_tenant_files(tenant_id: int):
    """Index all files for a tenant"""
    print(f"\n🚀 Starting indexing for tenant {tenant_id}")
    
    if not embeddings:
        print("❌ Embeddings not available")
        return 0
    
    ensure_vector_index()
    delete_tenant_vectors(tenant_id)
    
    all_docs = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get files from S3
        files = s3_list_tenant_files(tenant_id)
        print(f"📁 Found {len(files)} files in S3")
        
        for key in files:
            filename = os.path.basename(key)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in {".pdf", ".csv", ".txt", ".docx", ".json"}:
                continue
            
            local_path = os.path.join(temp_dir, filename)
            s3_client.download_file(S3_BUCKET_NAME, key, local_path)
            
            loader_map = {
                ".pdf": PyPDFLoader,
                ".csv": CSVLoader,
                ".txt": lambda p: TextLoader(p, encoding="utf-8"),
                ".docx": Docx2txtLoader,
                ".json": lambda p: JSONLoader(p, jq_schema=".")
            }
            loader = loader_map.get(ext)
            if loader:
                docs = loader(local_path).load()
                for doc in docs:
                    doc.metadata.update({
                        "source": f"s3://{S3_BUCKET_NAME}/{key}",
                        "tenant_id": str(tenant_id)
                    })
                all_docs.extend(docs)
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)
        print(f"📝 Created {len(chunks)} chunks")
        
        if not chunks:
            print("⚠️ No content to index")
            return 0
        
        # Generate embeddings
        vectors = embeddings.embed_documents([c.page_content for c in chunks])
        print(f"🔢 Generated {len(vectors)} embeddings")
        
        # Upload to S3 Vectors
        batch_size = 500
        payload = []
        total = 0
        
        for vec, chunk in zip(vectors, chunks):
            payload.append({
                "key": str(uuid.uuid4()),
                "data": {"float32": vec},
                "metadata": {
                    "tenant_id": str(tenant_id),
                    "source": chunk.metadata.get("source", "unknown"),
                    "content_preview": chunk.page_content[:500]
                }
            })
            
            if len(payload) >= batch_size:
                s3vectors_client.put_vectors(
                    vectorBucketName=S3_VECTORS_BUCKET_NAME,
                    indexName=S3_VECTORS_INDEX_NAME,
                    vectors=payload
                )
                total += len(payload)
                payload = []
        
        if payload:
            s3vectors_client.put_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                vectors=payload
            )
            total += len(payload)
        
        print(f"✅ INDEXING COMPLETE: {total} vectors added for tenant {tenant_id}")
        return total
        
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def lambda_handler(event, context):
    """
    Lambda handler for SQS-triggered indexing
    """
    print(f"📨 Received event: {json.dumps(event)}")
    
    tenant_ids_to_process = set()
    
    # Parse SQS messages
    if 'Records' in event:
        for record in event['Records']:
            if record.get('eventSource') == 'aws:sqs':
                try:
                    body = json.loads(record['body'])
                    if 'tenant_id' in body:
                        tenant_id = int(body['tenant_id'])
                        tenant_ids_to_process.add(tenant_id)
                        print(f"✅ Extracted tenant_id: {tenant_id}")
                except Exception as e:
                    print(f"❌ Error parsing SQS message: {e}")
    
    if not tenant_ids_to_process:
        print("⚠️ No tenant IDs found in event")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No tenant_id found in SQS message'})
        }
    
    # Process each tenant
    results = {}
    for tenant_id in tenant_ids_to_process:
        try:
            vector_count = index_tenant_files(tenant_id)
            results[tenant_id] = {
                'success': True,
                'vectors': vector_count
            }
        except Exception as e:
            print(f"❌ Error processing tenant {tenant_id}: {e}")
            results[tenant_id] = {
                'success': False,
                'error': str(e)
            }
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Processed {len(tenant_ids_to_process)} tenants',
            'results': results
        })
    }

# Example usage and testing
if __name__ == "__main__":
    # Test with manual SQS message
    manual_event = {
        "Records": [
            {
                "eventSource": "aws:sqs",
                "body": '{"tenant_id": 25}',
                "messageId": "test-message-1"
            }
        ]
    }
    
    # Test with S3 event
    s3_event = {
        "Records": [
            {
                "eventSource": "aws:s3",
                "s3": {
                    "object": {
                        "key": "knowledge_base/25/test-file.pdf"
                    }
                }
            }
        ]
    }
    
    print("Testing manual trigger:")
    lambda_handler(manual_event, None)
    
    print("\nTesting S3 event:")
    lambda_handler(s3_event, None)