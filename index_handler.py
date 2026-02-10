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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# CONFIGURATION
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
PERSON_KEY = "person"  # ✅ FIXED: Changed from "subject" to match API expectations
CHUNK_TYPE_KEY = "chunk_type"
CONTENT_PREVIEW_KEY = "content_preview"  # CRITICAL: Required for API retrieval

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


# Note: delete_tenant_vectors function removed
# S3 Vectors metadata filtering has limitations in current API
# Vectors will accumulate over time - this is acceptable for now
# Each reindex adds new vectors with updated content


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


class MultiTenantProcessor:
    """Document processor with subject detection"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased from 1000 to keep more context together
            chunk_overlap=300,  # Increased from 150 for better continuity
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def detect_subject(self, first_page_text: str, filename: str) -> str:
        clean_filename = filename.split(".")[0].replace("_", " ").replace("-", " ")
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()][:3]
        if lines and 3 < len(lines[0]) < 40:
            potential = lines[0]
            if not any(word in potential.lower() for word in ["resume", "cv", "bio", "curriculum"]):
                return potential
        return clean_filename
    
    def detect_chunk_type(self, text: str) -> str:
        text_l = text.lower()
        mapping = {
            "skills": ["skills", "expertise", "proficient", "tools", "stack"],
            "projects": ["projects", "developed", "built", "implemented"],
            "experience": ["experience", "employment", "worked", "internship"]
        }
        for category, keywords in mapping.items():
            if any(kw in text_l for kw in keywords):
                return category
        return "general"


def upload_to_vector_db(processed_chunks: List[dict], tenant_id: int, source: str):
    """Upload chunks to vector database"""
    if not processed_chunks:
        return
    
    batch_size = 50
    total_uploaded = 0
    
    for i in range(0, len(processed_chunks), batch_size):
        batch = processed_chunks[i:i + batch_size]
        texts = [item["text"] for item in batch]
        vector_embeddings = embeddings.embed_documents(texts)
        
        payload = []
        for vec, item in zip(vector_embeddings, batch):
            # Extract original content (without the "Subject: X | Section: Y |" prefix)
            full_text = item["text"]
            content_start = full_text.find("Content: ")
            if content_start != -1:
                original_content = full_text[content_start + 9:]  # Skip "Content: "
            else:
                original_content = full_text
            
            payload.append({
                "key": str(uuid.uuid4()),
                "data": {"float32": vec},
                "metadata": {
                    TENANT_ID_KEY: str(tenant_id),
                    SOURCE_KEY: source,
                    PERSON_KEY: item["subject"],  # ✅ FIXED: Use PERSON_KEY to match API
                    CHUNK_TYPE_KEY: item["type"],
                    "content_preview": original_content[:500]  # CRITICAL: Add this for API retrieval
                }
            })
        
        s3vectors_client.put_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            vectors=payload
        )
        total_uploaded += len(payload)
        print(f"📤 Uploaded batch {i//batch_size + 1}: {len(payload)} vectors")
    
    print(f"✅ Total uploaded: {total_uploaded} vectors")


def index_tenant_files(tenant_id: int):
    """Main indexing function"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting indexing for tenant {tenant_id}")
    print(f"{'='*70}\n")
    
    processor = MultiTenantProcessor()
    temp_dir = tempfile.mkdtemp()
    total_chunks_created = 0
    
    try:
        ensure_vector_index()
        # Skipping delete - S3 Vectors filtering limitations
        # New vectors will be added alongside existing ones
        files = s3_list_tenant_files(tenant_id)
        
        if not files:
            print(f"⚠️ No files found for tenant {tenant_id}")
            return {"status": "success", "tenant_id": tenant_id, "total_chunks_added": 0}
        
        for s3_key in files:
            filename = os.path.basename(s3_key)
            ext = s3_key.lower().split('.')[-1]
            
            if ext not in ["pdf", "docx", "txt"]:
                print(f"⏭️ Skipping: {filename}")
                continue
            
            print(f"\n📄 Processing: {filename}")
            local_file_path = os.path.join(temp_dir, f"doc_{uuid.uuid4()}")
            
            try:
                s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
                print(f"   ✅ Downloaded")
                
                LoaderClass = {"pdf": PyPDFLoader, "docx": Docx2txtLoader, "txt": TextLoader}.get(ext)
                if not LoaderClass:
                    continue
                
                loader = LoaderClass(local_file_path)
                documents = loader.load()
                
                if not documents:
                    print(f"   ⚠️ No text extracted")
                    continue
                
                print(f"   ✅ Loaded {len(documents)} pages")
                subject_name = processor.detect_subject(documents[0].page_content, filename)
                print(f"   📌 Subject: {subject_name}")
                
                for doc in documents:
                    raw_chunks = processor.text_splitter.split_text(doc.page_content)
                    batch_to_upload = []
                    
                    for text in raw_chunks:
                        c_type = processor.detect_chunk_type(text)
                        contextualized_text = f"Subject: {subject_name} | Section: {c_type} | Content: {text}"
                        batch_to_upload.append({
                            "text": contextualized_text,
                            "subject": subject_name,
                            "type": c_type
                        })
                    
                    total_chunks_created += len(batch_to_upload)
                    print(f"   📝 Created {len(batch_to_upload)} chunks")
                    upload_to_vector_db(batch_to_upload, tenant_id, f"s3://{S3_BUCKET_NAME}/{s3_key}")
                
            except Exception as file_error:
                print(f"   ❌ Error: {str(file_error)}")
                traceback.print_exc()
            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
        
        print(f"\n{'='*70}")
        print(f"✅ INDEXING COMPLETE")
        print(f"   Tenant: {tenant_id}")
        print(f"   Files: {len(files)}")
        print(f"   Chunks: {total_chunks_created}")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "files_processed": len(files),
            "total_chunks_added": total_chunks_created
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
