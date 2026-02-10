# Document Indexing Workflow

Complete guide to how documents are processed, chunked, embedded, and stored in isolated tenant indexes.

## Overview

```
User Upload → S3 → SQS → Lambda → Chunking → Embeddings → S3 Vectors (Isolated Index)
```

## Step-by-Step Process

### Step 1: File Upload to S3

**User Action**: Upload document via API or directly to S3

**S3 Structure**:
```
s3://rag-chat-uploads/
└── knowledge_base/
    ├── 12/                    # Tenant 12
    │   ├── policy.pdf
    │   ├── handbook.docx
    │   └── faq.txt
    ├── 26/                    # Tenant 26
    │   └── guidelines.pdf
    └── ...
```

**Supported Formats**:
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text Files (`.txt`)
- CSV Files (`.csv`)
- JSON Files (`.json`)

### Step 2: SQS Message Trigger

**Automatic Trigger**: S3 event notification sends message to SQS

**SQS Message Format**:
```json
{
  "tenant_id": 12
}
```

**Queue Configuration**:
- Type: FIFO (First-In-First-Out)
- URL: `https://sqs.ap-south-1.amazonaws.com/.../RagLambdaIndexing.fifo`
- Message Group ID: `reindexing`
- Deduplication: UUID-based

**Manual Trigger** (for testing):
```python
import boto3
import json
import time

sqs = boto3.client('sqs', region_name='ap-south-1')
sqs.send_message(
    QueueUrl="https://sqs.ap-south-1.amazonaws.com/.../RagLambdaIndexing.fifo",
    MessageBody=json.dumps({"tenant_id": 12}),
    MessageGroupId="manual-reindex",
    MessageDeduplicationId=f"manual-{int(time.time())}"
)
```

### Step 3: Lambda Function Processing

**Function**: `rag-indexer-lambda`  
**Handler**: `index_handler.lambda_handler`  
**Runtime**: Python 3.10  
**Timeout**: 5 minutes  
**Memory**: 1024 MB

**Lambda Workflow**:

```python
def lambda_handler(event, context):
    # 1. Parse SQS message
    tenant_id = extract_tenant_id(event)
    
    # 2. Create isolated index for tenant
    index_name = f"tenant-{tenant_id}-index"
    ensure_vector_index(tenant_id)
    
    # 3. List files from S3
    files = s3_list_tenant_files(tenant_id)
    
    # 4. Process each file
    for file in files:
        documents = load_document(file)
        chunks = chunk_documents(documents)
        embeddings = generate_embeddings(chunks)
        upload_to_isolated_index(embeddings, tenant_id)
    
    return {"status": "success", "vectors": total_count}
```

### Step 4: Document Loading

**Loaders by File Type**:

```python
loader_map = {
    ".pdf": PyPDFLoader,           # PDF documents
    ".docx": Docx2txtLoader,       # Word documents
    ".txt": TextLoader,            # Plain text
    ".csv": CSVLoader,             # CSV files
    ".json": JSONLoader            # JSON files
}
```

**Example - PDF Loading**:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("policy.pdf")
documents = loader.load()  # Returns list of Document objects

# Each document has:
# - page_content: str (text content)
# - metadata: dict (source, page number, etc.)
```

### Step 5: Chunking Strategy

**Why Chunking?**
- LLMs have token limits
- Better semantic search with focused chunks
- Improved retrieval accuracy

**Chunking Configuration**:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # Characters per chunk
    chunk_overlap=300,      # Overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Why These Values?**
- **1500 chars**: Keeps related information together (e.g., "Level 2 escalation" + "4 hours")
- **300 overlap**: Ensures context continuity across chunks
- **Separators**: Splits at natural boundaries (paragraphs, sentences)

**Example Chunking**:

**Original Document** (3000 chars):
```
Leave Policy

Employees are entitled to:
- 18 days of Earned Leave
- 12 days of Sick Leave
- 3 days of Casual Leave

To apply for leave:
1. Submit request 7 days in advance
2. Get manager approval
3. Update leave calendar
...
```

**After Chunking** (2 chunks):

**Chunk 1** (1500 chars):
```
Leave Policy

Employees are entitled to:
- 18 days of Earned Leave
- 12 days of Sick Leave
- 3 days of Casual Leave

To apply for leave:
1. Submit request 7 days in advance
...
```

**Chunk 2** (1500 chars with 300 overlap):
```
...7 days in advance
2. Get manager approval
3. Update leave calendar

Emergency Leave:
- Notify manager immediately
- Submit documentation within 48 hours
...
```

### Step 6: Subject Detection (Optional)

**For Resume/Profile Documents**:

```python
def detect_subject(first_page_text: str, filename: str) -> str:
    """Detect person name from document"""
    lines = first_page_text.split('\n')[:3]
    
    # Check first line for name
    if lines and 3 < len(lines[0]) < 40:
        return lines[0]  # "John Doe"
    
    # Fallback to filename
    return filename.replace("_", " ")  # "john_doe_resume" → "john doe resume"
```

**Contextualized Chunks**:
```python
contextualized_text = f"Subject: {subject_name} | Section: {chunk_type} | Content: {text}"
```

Example:
```
Subject: John Doe | Section: skills | Content: Python, AWS, Docker, FastAPI...
```

### Step 7: Embedding Generation

**Model**: Amazon Titan Embed Text v2  
**Dimension**: 1024  
**Region**: ap-south-1

```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v2:0"
)

# Generate embeddings for chunks
vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
# Returns: List[List[float]] - each vector is 1024 dimensions
```

**Example**:
```python
text = "Employees are entitled to 18 days of Earned Leave"
vector = embeddings.embed_query(text)
# vector = [0.123, -0.456, 0.789, ..., 0.321]  # 1024 floats
```

### Step 8: Vector Storage in Isolated Index

**AWS Best Practice**: Separate index per tenant

**Index Naming**:
```python
def get_tenant_index_name(tenant_id: int) -> str:
    return f"tenant-{tenant_id}-index"

# Examples:
# Tenant 12 → "tenant-12-index"
# Tenant 26 → "tenant-26-index"
```

**Index Creation**:
```python
s3vectors_client.create_index(
    vectorBucketName="rag-vectordb-bucket",
    indexName="tenant-12-index",
    dataType="float32",
    dimension=1024,
    distanceMetric="cosine",
    metadataConfiguration={
        "nonFilterableMetadataKeys": ["internal_id"]
    }
)
```

**Vector Upload**:
```python
payload = []
for vec, chunk in zip(vectors, chunks):
    payload.append({
        "key": str(uuid.uuid4()),           # Unique ID
        "data": {"float32": vec},           # 1024-dim vector
        "metadata": {
            "tenant_id": str(tenant_id),    # "12"
            "source": "s3://bucket/file",   # Source file
            "content_preview": chunk[:500]  # First 500 chars
        }
    })

# Upload in batches of 500
s3vectors_client.put_vectors(
    vectorBucketName="rag-vectordb-bucket",
    indexName="tenant-12-index",
    vectors=payload
)
```

**Batch Processing**:
- Batch size: 500 vectors per request
- Rate limit: 1000 requests/sec or 2500 vectors/sec
- Optimal: 5 batches/sec with 500 vectors each

### Step 9: Completion & Logging

**Success Response**:
```json
{
  "status": "success",
  "tenant_id": 12,
  "index_name": "tenant-12-index",
  "files_processed": 7,
  "total_chunks_added": 44
}
```

**Lambda Logs**:
```
🔒 ISOLATED INDEXING FOR TENANT 12
📦 Using dedicated index: tenant-12-index
======================================================================
📄 Processing: policy.pdf
   ✅ Downloaded
   ✅ Loaded 5 pages
   📝 Created 8 chunks
📤 Uploaded batch 1: 8 vectors to tenant-12-index
======================================================================
✅ ISOLATED INDEXING COMPLETE
Tenant: 12
Index: tenant-12-index
Files: 7
Chunks: 44
======================================================================
```

## Complete Code Flow

### Lambda Function (`index_handler.py`)

```python
def index_tenant_files(tenant_id: int):
    """Main indexing function"""
    index_name = get_tenant_index_name(tenant_id)
    
    # 1. Ensure isolated index exists
    ensure_vector_index(tenant_id)
    
    # 2. List files from S3
    files = s3_list_tenant_files(tenant_id)
    
    # 3. Process each file
    for s3_key in files:
        # Download file
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        
        # Load document
        loader = get_loader_for_file(s3_key)
        documents = loader.load()
        
        # Chunk document
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings
        vectors = embeddings.embed_documents([c.page_content for c in chunks])
        
        # Upload to isolated index
        upload_to_vector_db(vectors, chunks, tenant_id, index_name)
    
    return {"status": "success", "vectors": total_count}
```

## Testing Indexing

### Test 1: Manual Trigger
```bash
python3 << 'EOF'
import boto3, json, time
sqs = boto3.client('sqs', region_name='ap-south-1')
sqs.send_message(
    QueueUrl="https://sqs.ap-south-1.amazonaws.com/.../RagLambdaIndexing.fifo",
    MessageBody=json.dumps({"tenant_id": 12}),
    MessageGroupId="test",
    MessageDeduplicationId=f"test-{int(time.time())}"
)
print("✅ Message sent")
EOF
```

### Test 2: Check Lambda Logs
```bash
aws logs tail /aws/lambda/rag-indexer-lambda --follow --region ap-south-1
```

### Test 3: Verify Vectors
```python
from rag_model.rag_utils import retrieve_s3_vectors

docs = retrieve_s3_vectors("test query", tenant_id=12, top_k=5)
print(f"Found {len(docs)} documents in tenant-12-index")
```

## Troubleshooting

### Issue: Lambda Timeout
**Cause**: Too many files or large files  
**Solution**: Increase timeout to 5 minutes
```bash
aws lambda update-function-configuration \
  --function-name rag-indexer-lambda \
  --timeout 300
```

### Issue: "No vectors found"
**Cause**: Files not in correct S3 location  
**Solution**: Check S3 path
```bash
aws s3 ls s3://rag-chat-uploads/knowledge_base/12/
```

### Issue: Embedding errors
**Cause**: Bedrock permissions  
**Solution**: Add Bedrock policy to Lambda role
```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel"],
  "Resource": "*"
}
```

## Performance Optimization

### Chunking Optimization
- **Small chunks** (500 chars): More precise but may lose context
- **Large chunks** (2000 chars): More context but less precise
- **Recommended**: 1500 chars with 300 overlap

### Batch Size Optimization
- **Small batches** (50 vectors): Slower but more reliable
- **Large batches** (500 vectors): Faster but may hit rate limits
- **Recommended**: 500 vectors per batch

### Parallel Processing
```python
# Process multiple files in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_file, file) for file in files]
    results = [f.result() for f in futures]
```

## Best Practices

1. ✅ **Use isolated indexes** - One index per tenant
2. ✅ **Optimize chunk size** - 1500 chars with 300 overlap
3. ✅ **Batch uploads** - 500 vectors per request
4. ✅ **Add metadata** - Include source, tenant_id, content_preview
5. ✅ **Handle errors** - Retry failed uploads
6. ✅ **Monitor logs** - Check Lambda CloudWatch logs
7. ✅ **Test thoroughly** - Verify vectors are searchable

## Next Steps

- [2_RETRIEVAL_WORKFLOW.md](2_RETRIEVAL_WORKFLOW.md) - Learn how queries retrieve documents
- [3_DEPLOYMENT_GUIDE.md](3_DEPLOYMENT_GUIDE.md) - Deploy to production
- [4_TENANT_ISOLATION.md](4_TENANT_ISOLATION.md) - Understand security architecture
