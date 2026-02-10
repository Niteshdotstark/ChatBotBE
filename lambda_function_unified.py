import os
import tempfile
import shutil
import uuid
import json
import boto3
import traceback
from typing import List

# LangChain imports (from your Lambda layers)
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# CONFIGURATION - MATCHES SERVER
# ==============================================================================
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = os.getenv("S3_VECTORS_BUCKET_NAME", "rag-vectordb-bucket")
S3_VECTORS_INDEX_NAME = os.getenv("S3_VECTORS_INDEX_NAME", "tenant-knowledge-index")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
S3_PREFIX_KNOWLEDGE = "knowledge_base"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION_NAME = "ap-south-1"
TENANT_ID_KEY = "tenant_id"
SOURCE_KEY = "source"
CONTENT_PREVIEW_KEY = "content_preview"

# Initialize clients
print("🔧 Initializing AWS clients...")
s3_client = boto3.client("s3", region_name=REGION_NAME)
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)
print("✅ Clients initialized")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def ensure_vector_index():
    """Create vector index if it doesn't exist"""
    try:
        s3vectors_client.put_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            vectors=[{"key": "ping", "data": {"float32": [0.0] * 1024}, "metadata": {"tenant_id": "0"}}]
        )
        s3vectors_client.delete_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            keys=["ping"]
        )
        print(f"✅ Index '{S3_VECTORS_INDEX_NAME}' exists")
        return
    except Exception as e:
        if 'NotFoundException' not in str(e) and 'ValidationException' not in str(e):
            raise
    
    print(f"📝 Creating index '{S3_VECTORS_INDEX_NAME}'...")
    try:
        try:
            s3vectors_client.create_vector_bucket(vectorBucketName=S3_VECTORS_BUCKET_NAME)
        except:
            pass
        s3vectors_client.create_index(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": ["internal_id"]}
        )
        print("✅ Index created")
    except Exception as e:
        if 'ConflictException' in str(e):
            print("✅ Index already exists")
        else:
            raise


def s3_list_tenant_files(tenant_id: int) -> List[str]:
    """List all files for a tenant in S3"""
    prefix = f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/"
    print(f"📂 Listing files: {prefix}")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/') and not key.endswith('/urls.txt'):
                keys.append(key)
    print(f"✅ Found {len(keys)} files")
    return keys


def index_tenant_files(tenant_id: int):
    """Main indexing function - MATCHES SERVER LOGIC"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting indexing for tenant {tenant_id}")
    print(f"{'='*70}\n")
    
    temp_dir = tempfile.mkdtemp()
    all_docs = []
    
    try:
        ensure_vector_index()
        files = s3_list_tenant_files(tenant_id)
        
        if not files:
            print(f"⚠️ No files found for tenant {tenant_id}")
            return {"status": "success", "tenant_id": tenant_id, "total_chunks_added": 0}
        
        # Load all documents
        for s3_key in files:
            filename = os.path.basename(s3_key)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in {".pdf", ".csv", ".txt", ".docx", ".json"}:
                print(f"⏭️ Skipping: {filename}")
                continue
            
            print(f"\n📄 Processing: {filename}")
            local_file_path = os.path.join(temp_dir, f"doc_{uuid.uuid4()}{ext}")
            
            try:
                s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
                print(f"   ✅ Downloaded")
                
                # Loader mapping - same as server
                loader_map = {
                    ".pdf": PyPDFLoader,
                    ".csv": CSVLoader,
                    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
                    ".docx": Docx2txtLoader,
                    ".json": lambda p: JSONLoader(p, jq_schema=".")
                }
                
                loader_class = loader_map.get(ext)
                if not loader_class:
                    continue
                
                loader = loader_class(local_file_path)
                documents = loader.load()
                
                if not documents:
                    print(f"   ⚠️ No text extracted")
                    continue
                
                print(f"   ✅ Loaded {len(documents)} pages/rows")
                
                # Add metadata - same as server
                for doc in documents:
                    doc.metadata.update({
                        "source": f"s3://{S3_BUCKET_NAME}/{s3_key}",
                        "tenant_id": str(tenant_id)
                    })
                all_docs.extend(documents)
                
            except Exception as file_error:
                print(f"   ❌ Error: {str(file_error)}")
                traceback.print_exc()
            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
        
        if not all_docs:
            print("⚠️ No documents loaded")
            return {"status": "success", "tenant_id": tenant_id, "total_chunks_added": 0}
        
        # Split & Embed - EXACTLY like server
        print(f"\n📝 Splitting {len(all_docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)
        
        if not chunks:
            print("⚠️ No chunks created")
            return {"status": "success", "tenant_id": tenant_id, "total_chunks_added": 0}
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"🔄 Generating embeddings...")
        vectors = embeddings.embed_documents([c.page_content for c in chunks])
        print(f"✅ Generated {len(vectors)} embeddings")
        
        # Upload in batches - same as server
        print(f"📤 Uploading vectors...")
        batch_size = 500
        payload = []
        total = 0
        
        for vec, chunk in zip(vectors, chunks):
            payload.append({
                "key": str(uuid.uuid4()),
                "data": {"float32": vec},
                "metadata": {
                    TENANT_ID_KEY: str(tenant_id),
                    SOURCE_KEY: chunk.metadata.get("source", "unknown"),
                    CONTENT_PREVIEW_KEY: chunk.page_content[:500]
                }
            })
            
            if len(payload) >= batch_size:
                s3vectors_client.put_vectors(
                    vectorBucketName=S3_VECTORS_BUCKET_NAME,
                    indexName=S3_VECTORS_INDEX_NAME,
                    vectors=payload
                )
                total += len(payload)
                print(f"   ✅ Uploaded batch: {total} vectors so far")
                payload = []
        
        # Upload remaining
        if payload:
            s3vectors_client.put_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                vectors=payload
            )
            total += len(payload)
        
        print(f"\n{'='*70}")
        print(f"✅ INDEXING COMPLETE")
        print(f"   Tenant: {tenant_id}")
        print(f"   Files: {len(files)}")
        print(f"   Chunks: {total}")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "files_processed": len(files),
            "total_chunks_added": total
        }
        
    except Exception as e:
        print(f"\n❌ INDEXING FAILED: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# LAMBDA HANDLER - Entry point for AWS Lambda
# ==============================================================================
def lambda_handler(event, context):
    """Lambda handler for SQS-triggered indexing"""
    print(f"\n{'#'*70}")
    print(f"📨 Lambda invoked")
    print(f"{'#'*70}")
    print(f"Event: {json.dumps(event)}\n")
    
    try:
        if "Records" in event:
            results = []
            for record in event["Records"]:
                print(f"📬 Processing message: {record.get('messageId')}")
                try:
                    body = json.loads(record["body"])
                    tenant_id = body.get("tenant_id")
                    if tenant_id is None:
                        print("❌ Missing tenant_id")
                        continue
                    result = index_tenant_files(int(tenant_id))
                    results.append(result)
                except Exception as msg_error:
                    print(f"❌ Message error: {str(msg_error)}")
                    traceback.print_exc()
                    results.append({"status": "error", "error": str(msg_error)})
            
            return {
                "statusCode": 200,
                "body": json.dumps({"message": f"Processed {len(results)} messages", "results": results})
            }
        else:
            tenant_id = event.get("tenant_id")
            if tenant_id is None:
                return {"statusCode": 400, "body": json.dumps({"error": "Missing tenant_id"})}
            result = index_tenant_files(int(tenant_id))
            return {"statusCode": 200, "body": json.dumps(result)}
    
    except Exception as e:
        print(f"\n❌ HANDLER FAILED: {str(e)}")
        traceback.print_exc()
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "traceback": traceback.format_exc()})}
