#!/usr/bin/env python3
"""
Enhanced Lambda indexing handler with person-aware chunking for better accuracy.
This prevents cross-contamination between different people's information.
"""

import os
import tempfile
import shutil
import uuid
import json
import boto3
import re
from typing import List, Optional, Dict, Tuple
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
import httpx
from bs4 import BeautifulSoup

# LangChain imports (all available in your layer)
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ==============================================================================
# CONFIG
# ==============================================================================
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = os.getenv("S3_VECTORS_BUCKET_NAME", "rag-vectordb-bucket")
S3_VECTORS_INDEX_NAME = os.getenv("S3_VECTORS_INDEX_NAME", "tenant-knowledge-index")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
S3_PREFIX_KNOWLEDGE = "knowledge_base"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

# Metadata keys
TENANT_ID_KEY = "tenant_id"
SOURCE_KEY = "source"
CONTENT_PREVIEW_KEY = "content_preview"
PERSON_KEY = "person"  # NEW: Person identification
CHUNK_TYPE_KEY = "chunk_type"  # NEW: Type of content (skills, projects, etc.)

# Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
s3_client = boto3.client("s3", region_name=REGION_NAME)
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)

# ==============================================================================
# Person-Aware Document Processing
# ==============================================================================

class PersonAwareProcessor:
    """Enhanced document processor that identifies and tags content by person"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Common name patterns to detect
        self.name_patterns = {
            "mahi": ["mahi", "maheshwari", "mahi maheshwari"],
            "nitesh": ["nitesh"],
            # Add more names as needed
        }
        
        # Content type indicators
        self.content_types = {
            "skills": ["skills", "technical skills", "expertise", "proficient", "experienced"],
            "projects": ["projects", "developed", "built", "created", "implemented"],
            "experience": ["experience", "worked", "employment", "internship"],
            "education": ["education", "university", "college", "degree", "bachelor"]
        }
    
    def extract_person_names(self, content: str) -> List[str]:
        """Extract person names from document content"""
        content_lower = content.lower()
        found_names = []
        
        for person, patterns in self.name_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                found_names.append(person.title())
        
        return found_names
    
    def identify_chunk_person(self, chunk: str, document_persons: List[str], source: str) -> Optional[str]:
        """Identify which person a chunk belongs to"""
        chunk_lower = chunk.lower()
        
        # Method 1: Direct name mention in chunk
        for person, patterns in self.name_patterns.items():
            if any(pattern in chunk_lower for pattern in patterns):
                return person.title()
        
        # Method 2: Check source filename
        source_lower = source.lower()
        for person, patterns in self.name_patterns.items():
            if any(pattern in source_lower for pattern in patterns):
                return person.title()
        
        # Method 3: If document has only one person, assign to that person
        if len(document_persons) == 1:
            return document_persons[0]
        
        # Method 4: Context-based identification
        # Look for personal indicators that suggest this chunk belongs to someone
        personal_indicators = [
            "my", "i am", "i have", "i worked", "i developed", "i built",
            "experience:", "skills:", "projects:", "education:"
        ]
        
        if any(indicator in chunk_lower for indicator in personal_indicators):
            # If chunk has personal content but no clear name, try to infer
            # This could be enhanced with more sophisticated NLP
            if document_persons:
                return document_persons[0]  # Default to first person found in document
        
        return None
    
    def identify_content_type(self, chunk: str) -> Optional[str]:
        """Identify the type of content in the chunk"""
        chunk_lower = chunk.lower()
        
        for content_type, indicators in self.content_types.items():
            if any(indicator in chunk_lower for indicator in indicators):
                return content_type
        
        return "general"
    
    def process_document(self, document: Document) -> List[Document]:
        """Process document with person-aware chunking"""
        
        # Extract person names from the full document
        document_persons = self.extract_person_names(document.page_content)
        source = document.metadata.get("source", "unknown")
        
        print(f"Processing document: {source}")
        print(f"Found persons: {document_persons}")
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([document])
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Identify which person this chunk belongs to
            chunk_person = self.identify_chunk_person(
                chunk.page_content, 
                document_persons, 
                source
            )
            
            # Identify content type
            content_type = self.identify_content_type(chunk.page_content)
            
            # Create enhanced metadata
            enhanced_metadata = {
                **chunk.metadata,
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                PERSON_KEY: chunk_person or "unknown",
                CHUNK_TYPE_KEY: content_type
            }
            
            # Add person prefix to chunk content for better retrieval
            if chunk_person:
                enhanced_content = f"[PERSON: {chunk_person}] [TYPE: {content_type}] {chunk.page_content}"
            else:
                enhanced_content = f"[TYPE: {content_type}] {chunk.page_content}"
            
            enhanced_chunk = Document(
                page_content=enhanced_content,
                metadata=enhanced_metadata
            )
            
            enhanced_chunks.append(enhanced_chunk)
        
        print(f"Created {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks

# ==============================================================================
# S3 & Vector DB Helpers (Updated)
# ==============================================================================

def ensure_vector_index():
    """Ensure vector index exists with updated metadata configuration"""
    try:
        buckets = s3vectors_client.list_vector_buckets().get("vectorBuckets", [])
        if not any(b["vectorBucketName"] == S3_VECTORS_BUCKET_NAME for b in buckets):
            s3vectors_client.create_vector_bucket(vectorBucketName=S3_VECTORS_BUCKET_NAME)
        
        try:
            # Test if index exists
            s3vectors_client.put_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                vectors=[{
                    "key": "ping", 
                    "data": {"float32": [0.0]*1024}, 
                    "metadata": {
                        TENANT_ID_KEY: "0",
                        PERSON_KEY: "test",
                        CHUNK_TYPE_KEY: "test"
                    }
                }]
            )
            s3vectors_client.delete_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME, 
                indexName=S3_VECTORS_INDEX_NAME, 
                keys=["ping"]
            )
            return
        except ClientError as e:
            if e.response['Error']['Code'] not in ['NotFoundException', 'ValidationException']:
                raise
        
        # Create index with enhanced metadata configuration
        s3vectors_client.create_index(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
            metadataConfiguration={
                "nonFilterableMetadataKeys": ["internal_id", CONTENT_PREVIEW_KEY]
            }
        )
        print("Created vector index with enhanced metadata support")
        
    except ClientError as e:
        if e.response['Error']['Code'] in ['ConflictException', 'ResourceInUseException']:
            pass

def delete_tenant_vectors(tenant_id: int):
    """Delete all vectors for a tenant"""
    try:
        keys = []
        next_token = None
        
        while True:
            kwargs = {
                "vectorBucketName": S3_VECTORS_BUCKET_NAME,
                "indexName": S3_VECTORS_INDEX_NAME,
                "queryVector": {"float32": [0.0]*1024},
                "topK": 30,
                "filter": {TENANT_ID_KEY: {"eq": str(tenant_id)}},
                "returnMetadata": False
            }
            
            if next_token:
                kwargs["nextToken"] = next_token
            
            resp = s3vectors_client.query_vectors(**kwargs)
            keys.extend([item["key"] for item in resp.get("vectors", [])])
            next_token = resp.get("nextToken")
            
            if not next_token:
                break
        
        # Delete in batches
        for i in range(0, len(keys), 100):
            s3vectors_client.delete_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                keys=keys[i:i+100]
            )
        
        print(f"Deleted {len(keys)} existing vectors for tenant {tenant_id}")
        
    except ClientError:
        pass  # safe to ignore

def s3_list_tenant_files(tenant_id: int) -> List[str]:
    """List all files for a tenant"""
    prefix = f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/"
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []
    
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/urls.txt'):
                keys.append(obj['Key'])
    
    return keys

def s3_load_urls(tenant_id: int) -> List[str]:
    """Load URLs from tenant's urls.txt file"""
    try:
        obj = s3_client.get_object(
            Bucket=S3_BUCKET_NAME, 
            Key=f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/urls.txt"
        )
        return [
            line.strip() 
            for line in obj['Body'].read().decode('utf-8').splitlines() 
            if line.strip()
        ]
    except ClientError:
        return []

# ==============================================================================
# Crawler (Enhanced)
# ==============================================================================

def fetch_url(url: str) -> Optional[Document]:
    """Fetch and process URL content"""
    try:
        resp = httpx.get(
            url, 
            headers={'User-Agent': 'RAGIndexer/1.0'}, 
            timeout=20, 
            follow_redirects=True
        )
        
        if resp.status_code != 200:
            return None
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = re.sub(r'\s+', ' ', soup.get_text()).strip()
        
        if len(text) > 150:
            return Document(
                page_content=text[:15000], 
                metadata={"source": url}
            )
        
        return None
        
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def crawl_urls(urls: List[str]) -> List[Document]:
    """Crawl multiple URLs concurrently"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        return [doc for doc in executor.map(fetch_url, urls) if doc]

# ==============================================================================
# Main Enhanced Indexing Function
# ==============================================================================

def index_tenant(tenant_id: int):
    """Enhanced indexing with person-aware chunking"""
    print(f"🚀 Starting enhanced indexing for tenant {tenant_id}")
    
    # Initialize
    ensure_vector_index()
    delete_tenant_vectors(tenant_id)
    
    processor = PersonAwareProcessor()
    all_docs = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process files
        file_keys = s3_list_tenant_files(tenant_id)
        print(f"📄 Processing {len(file_keys)} files")
        
        for key in file_keys:
            ext = os.path.splitext(key)[1].lower()
            if ext not in {".pdf", ".csv", ".txt", ".docx", ".json"}:
                continue
            
            local_path = os.path.join(temp_dir, os.path.basename(key))
            s3_client.download_file(S3_BUCKET_NAME, key, local_path)
            
            # Load document
            loader_map = {
                ".pdf": PyPDFLoader,
                ".csv": CSVLoader,
                ".txt": lambda p: TextLoader(p, encoding="utf-8"),
                ".docx": Docx2txtLoader,
                ".json": lambda p: JSONLoader(p, jq_schema="."),
            }
            
            loader = loader_map.get(ext)
            if loader:
                try:
                    docs = loader(local_path).load()
                    for doc in docs:
                        doc.metadata.update({
                            "source": f"s3://{S3_BUCKET_NAME}/{key}",
                            TENANT_ID_KEY: str(tenant_id)
                        })
                    all_docs.extend(docs)
                    print(f"✅ Loaded {len(docs)} documents from {os.path.basename(key)}")
                except Exception as e:
                    print(f"❌ Error loading {key}: {e}")
        
        # Crawl URLs
        urls = s3_load_urls(tenant_id)
        if urls:
            print(f"🌐 Crawling {len(urls)} URLs...")
            url_docs = crawl_urls(urls)
            for doc in url_docs:
                doc.metadata[TENANT_ID_KEY] = str(tenant_id)
            all_docs.extend(url_docs)
            print(f"✅ Crawled {len(url_docs)} URL documents")
        
        if not all_docs:
            print("⚠️  No documents found to process")
            return {"status": "success", "vectors_added": 0}
        
        # Enhanced processing with person-aware chunking
        print(f"🧠 Processing {len(all_docs)} documents with person-aware chunking...")
        all_chunks = []
        
        for doc in all_docs:
            enhanced_chunks = processor.process_document(doc)
            all_chunks.extend(enhanced_chunks)
        
        print(f"📦 Created {len(all_chunks)} enhanced chunks")
        
        if not all_chunks:
            return {"status": "success", "vectors_added": 0}
        
        # Generate embeddings
        print("🔗 Generating embeddings...")
        chunk_texts = [chunk.page_content for chunk in all_chunks]
        vectors = embeddings.embed_documents(chunk_texts)
        
        # Upload vectors in batches
        print("⬆️  Uploading vectors to S3 Vectors...")
        batch_size = 500
        total_uploaded = 0
        payload = []
        
        for vec, chunk in zip(vectors, all_chunks):
            vector_data = {
                "key": str(uuid.uuid4()),
                "data": {"float32": vec},
                "metadata": {
                    TENANT_ID_KEY: str(tenant_id),
                    SOURCE_KEY: chunk.metadata.get("source", "unknown"),
                    CONTENT_PREVIEW_KEY: chunk.page_content[:500],
                    PERSON_KEY: chunk.metadata.get(PERSON_KEY, "unknown"),
                    CHUNK_TYPE_KEY: chunk.metadata.get(CHUNK_TYPE_KEY, "general")
                }
            }
            payload.append(vector_data)
            
            if len(payload) >= batch_size:
                s3vectors_client.put_vectors(
                    vectorBucketName=S3_VECTORS_BUCKET_NAME,
                    indexName=S3_VECTORS_INDEX_NAME,
                    vectors=payload
                )
                total_uploaded += len(payload)
                print(f"📤 Uploaded batch: {total_uploaded} vectors")
                payload = []
        
        # Upload remaining vectors
        if payload:
            s3vectors_client.put_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                vectors=payload
            )
            total_uploaded += len(payload)
        
        print(f"✅ Successfully indexed tenant {tenant_id}: {total_uploaded} vectors")
        return {"status": "success", "vectors_added": total_uploaded}
        
    except Exception as e:
        print(f"❌ Error indexing tenant {tenant_id}: {e}")
        raise
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ==============================================================================
# Lambda Handler
# ==============================================================================

def lambda_handler(event, context):
    """Enhanced Lambda handler with better error handling"""
    print("🚀 Enhanced Lambda Indexing Handler Started")
    print("Raw event:", json.dumps(event))
    
    try:
        # Handle both direct invoke AND SQS trigger
        if "Records" in event:
            # Records are from SQS
            results = []
            for record in event["Records"]:
                try:
                    body = json.loads(record["body"])
                    tenant_id = int(body["tenant_id"])
                    print(f"📨 Processing tenant {tenant_id} from SQS message {record['messageId']}")
                    
                    result = index_tenant(tenant_id)
                    results.append({
                        "tenant_id": tenant_id,
                        "message_id": record['messageId'],
                        "result": result
                    })
                    
                except Exception as e:
                    print(f"❌ Failed to process message {record['messageId']}: {e}")
                    raise  # Let it go to DLQ
            
            return {"status": "success", "processed": results}
            
        else:
            # Direct invoke (e.g. from console test)
            tenant_id = int(event["tenant_id"])
            print(f"🎯 Direct invocation for tenant {tenant_id}")
            
            result = index_tenant(tenant_id)
            return {
                "status": "success", 
                "tenant_id": tenant_id,
                "result": result
            }
            
    except Exception as e:
        print(f"❌ Lambda handler error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# ==============================================================================
# Testing Function (for local development)
# ==============================================================================

if __name__ == "__main__":
    # Test locally
    test_event = {"tenant_id": 25}
    result = lambda_handler(test_event, None)
    print("Test result:", json.dumps(result, indent=2))