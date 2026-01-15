import os
import boto3
import random
import time
import json
import uuid
import re
from typing import List, Optional, Dict, Any
from botocore.exceptions import ClientError
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor
import httpx
from bs4 import BeautifulSoup
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ==============================================================================
# CONFIGURATION
# ==============================================================================
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = os.getenv("S3_VECTORS_BUCKET_NAME", "rag-vectordb-bucket")
S3_VECTORS_INDEX_NAME = os.getenv("S3_VECTORS_INDEX_NAME", "tenant-knowledge-index")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
S3_PREFIX_KNOWLEDGE = "knowledge_base"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "meta.llama3-8b-instruct-v1:0"
REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
TENANT_ID_KEY = "tenant_id"
SOURCE_KEY = "source"
CONTENT_PREVIEW_KEY = "content_preview"
sqs = boto3.client('sqs')
INDEXING_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo"

# Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
s3_client = boto3.client("s3", region_name=REGION_NAME)
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)

# Fixed responses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_RESPONSES_FILE = os.path.join(BASE_DIR, "fixed_responses.json")
FIXED_RESPONSES = {}
if os.path.exists(FIXED_RESPONSES_FILE):
    with open(FIXED_RESPONSES_FILE, "r", encoding="utf-8") as f:
        FIXED_RESPONSES = json.load(f)


def trigger_reindexing(tenant_id: int):
    try:
        sqs.send_message(
            QueueUrl=INDEXING_QUEUE_URL,
            MessageBody=json.dumps({"tenant_id": tenant_id}),
            MessageGroupId="reindexing",
            MessageDeduplicationId=str(uuid.uuid4())
        )
        print(f"Reindexing queued for tenant {tenant_id}")
    except Exception as e:
        print(f"SQS failed: {e}")

def s3_append_url(tenant_id: int, url: str):
    """Appends a URL to the tenant's urls.txt in S3"""
    key = f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/urls.txt"
    try:
        # Try to download existing file
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        content = response['Body'].read().decode('utf-8')
        lines = [line.strip() for line in content.splitlines() if line.strip()]
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchKey':
            raise
        lines = []

    # Avoid duplicates
    if url.strip() not in lines:
        lines.append(url.strip())

    # Upload back
    new_content = "\n".join(lines) + "\n"
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=new_content,
        ContentType="text/plain"
    )
    print(f"Appended URL to s3://{S3_BUCKET_NAME}/{key}")

# ==============================================================================
# CRAWLERS — FULLY RESTORED
# ==============================================================================
# def is_valid_url(url: str) -> bool:
#     try:
#         parsed = urlparse(url)
#         return bool(parsed.netloc) and bool(parsed.scheme)
#     except:
#         return False

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def fetch_url_content(url: str) -> Optional[Document]:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0)'}
        resp = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = clean_text(soup.get_text(separator=' '))
        if len(text) < 150:
            return None
        return Document(page_content=text[:15000], metadata={"source": url})
    except Exception as e:
        print(f"[Crawl] Failed {url}: {e}")
        return None

def crawl_urls_lightweight(urls: List[str]) -> List[Document]:
    docs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for doc in executor.map(fetch_url_content, urls):
            if doc:
                docs.append(doc)
    return docs

# ==============================================================================
# S3 HELPERS
# ==============================================================================
def s3_list_tenant_files(tenant_id: int) -> List[str]:
    prefix = f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/"
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/urls.txt'):
                keys.append(obj['Key'])
    return keys

def s3_load_urls_from_file(tenant_id: int) -> List[str]:
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/urls.txt")
        content = obj['Body'].read().decode('utf-8')
        return [line.strip() for line in content.splitlines() if line.strip() and is_valid_url(line.strip())]
    except ClientError:
        return []

# ==============================================================================
# S3 VECTORS CORE
# ==============================================================================
def ensure_vector_index():
    """Idempotent index creation — safe to call on every request"""
    try:
        # 1. Bucket
        buckets = s3vectors_client.list_vector_buckets().get("vectorBuckets", [])
        if not any(b["vectorBucketName"] == S3_VECTORS_BUCKET_NAME for b in buckets):
            print(f"Creating vector bucket '{S3_VECTORS_BUCKET_NAME}'...")
            s3vectors_client.create_vector_bucket(vectorBucketName=S3_VECTORS_BUCKET_NAME)

        # 2. Test if index exists by trying a tiny put_vectors
        try:
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
            print(f"Index '{S3_VECTORS_INDEX_NAME}' exists and ready")
            return  # ← INDEX EXISTS → EXIT EARLY
        except ClientError as e:
            if e.response['Error']['Code'] not in ['NotFoundException', 'ValidationException']:
                raise

        # 3. Index does NOT exist → create it ONCE
        print(f"Creating vector index '{S3_VECTORS_INDEX_NAME}' (first time only)...")
        s3vectors_client.create_index(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
            metadataConfiguration={
                "nonFilterableMetadataKeys": ["internal_id"]  # satisfies AWS rule
            }
        )
        print("INDEX CREATED SUCCESSFULLY — tenant_id filtering ENABLED")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['ConflictException', 'ResourceInUseException']:
            print("Index already exists (created by another process) — continuing safely")
            return
        else:
            print(f"Fatal error creating index: {e}")
            raise

def delete_tenant_vectors(tenant_id: int):
    """Delete all vectors for a tenant — handles topK=30 limit with pagination"""
    try:
        keys_to_delete = []
        next_token = None

        print(f"Scanning and deleting vectors for tenant {tenant_id} (max 30 per page)...")

        while True:
            query_kwargs = {
                "vectorBucketName": S3_VECTORS_BUCKET_NAME,
                "indexName": S3_VECTORS_INDEX_NAME,
                "queryVector": {"float32": [0.0] * 1024},
                "topK": 30,
                "filter": {TENANT_ID_KEY: {"eq": str(tenant_id)}},
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
            # Delete in batches of 100 (max allowed by delete_vectors)
            for i in range(0, len(keys_to_delete), 100):
                batch = keys_to_delete[i:i+100]
                s3vectors_client.delete_vectors(
                    vectorBucketName=S3_VECTORS_BUCKET_NAME,
                    indexName=S3_VECTORS_INDEX_NAME,
                    keys=batch
                )
            print(f"Deleted {len(keys_to_delete)} vectors for tenant {tenant_id}")
        else:
            print(f"No vectors found for tenant {tenant_id}")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ["ResourceNotFoundException", "ValidationException"]:
            print("Index not ready or no vectors — skipping delete")
        else:
            print(f"Delete failed: {e}")
            # Don't crash indexing — continue

def index_tenant_files(tenant_id: int, additional_urls: List[str] = None):
    print(f"\nStarting indexing for tenant {tenant_id} (S3 Vectors + BackgroundTasks)")
    ensure_vector_index()
    delete_tenant_vectors(tenant_id)

    all_docs = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Files
        for key in s3_list_tenant_files(tenant_id):
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

        # URLs
        urls = s3_load_urls_from_file(tenant_id)
        if additional_urls:
            urls.extend(additional_urls)
        if urls:
            print(f"Crawling {len(urls)} URLs...")
            all_docs.extend(crawl_urls_lightweight(urls))

        # Split & Embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)
        if not chunks:
            print("No content found")
            return 0

        vectors = embeddings.embed_documents([c.page_content for c in chunks])

        # Upload in batches
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
                payload = []

        if payload:
            s3vectors_client.put_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                vectors=payload
            )
            total += len(payload)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"INDEXING COMPLETE: {total} vectors added for tenant {tenant_id}\n")
    return total

def retrieve_s3_vectors(query: str, tenant_id: int, top_k: int = 8) -> List[Document]:
    ensure_vector_index()
    q_vec = embeddings.embed_query(query)

    try:
        resp = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": q_vec},
            topK=top_k,
            returnMetadata=True,
            returnDistance=True,
            filter={
                TENANT_ID_KEY: str(tenant_id)
            }
        )
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == "ResourceNotFoundException":
            return []
        print(f"Query failed: {e}")
        return []

    docs = []
    for v in resp.get("vectors", []):
        meta = v.get("metadata", {})
        content = meta.get(CONTENT_PREVIEW_KEY, "").strip()
        if not content:
            continue
        doc_meta = {"source": meta.get(SOURCE_KEY, "S3 Vectors")}
        # Optional: Add distance score to metadata
        if "distance" in v:
            doc_meta["similarity_score"] = v["distance"]
        docs.append(Document(page_content=content, metadata=doc_meta))
    return docs
# ==============================================================================
# LangChain RAG Chain
# ==============================================================================
class S3VectorRetriever(BaseRetriever):
    tenant_id: int
    top_k: int = 8
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return retrieve_s3_vectors(query, self.tenant_id, self.top_k)
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a friendly, human-like chatbot designed to chat naturally and helpfully.
You speak like a thoughtful friend — casual, clear, and kind — not like a formal AI assistant.

**Response Guidelines:**
1. Be warm and conversational, but keep focus on the user's question.
2. Never use phrases like “I'd be happy to help”, “Based on the text you provided”, or “I think I can help you with that.”
3. When information is available, state it directly and confidently.
4. If something isn't found in your knowledge base, say it simply:
   - “Hmm, I don't have info about that in my knowledge base.”
   - “Looks like the context doesn't mention that yet.”
   - “I couldn't find details about that, want me to check somewhere else?”
5. If a similar or related person/topic exists, you can suggest it:
   - “I didn't find anyone named Anil, but Sunil Sharma is mentioned as the CEO of DotStark.”
6. Keep responses short (under 700 characters), natural, and friendly.
7. Use contractions (it's, don't, that's) to sound human.
8. Never invent information that isn't in the context or knowledge base.
9. Avoid robotic or apologetic language — speak casually, like chatting with a real person.
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(
        """Context:\n{context}\n\nQuestion: {input}\n\nAnswer (under 1000 characters):"""
    )
])

def get_rag_chain(tenant_id: int, user_id: str = "default", initial_history: list = None):
    retriever = S3VectorRetriever(tenant_id=tenant_id)
    llm = ChatBedrock(model_id=LLM_MODEL, region_name=REGION_NAME, model_kwargs={"temperature": 0.3})
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # If an initial history (list of HumanMessage/AIMessage) is provided, pre-load it into memory
    if initial_history:
        try:
            memory.chat_memory.messages = list(initial_history)
        except Exception:
            # Fallback: ignore if messages can't be set
            pass

    qa_chain = create_stuff_documents_chain(llm, CONVERSATIONAL_RAG_PROMPT)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    def invoke(text: str):
        result = rag_chain.invoke({"input": text, "chat_history": memory.chat_memory.messages})
        memory.save_context({"input": text}, {"output": result.get("answer", "")})
        return result

    return invoke

def answer_question_modern(question: str, tenant_id: int, user_id: str = "default", context_messages: list = None):
    cleaned = question.strip().lower()
    if cleaned in FIXED_RESPONSES:
        fixed_resp = FIXED_RESPONSES[cleaned]
        if isinstance(fixed_resp, dict):
            answer = fixed_resp.get("answer", str(fixed_resp))  # Extract inner 'answer' or fallback to str
        else:
            answer = str(fixed_resp)
        return {"answer": answer.strip(), "sources": ["Fixed Response"]}

    chain = get_rag_chain(tenant_id, user_id, initial_history=context_messages)
    result = chain(question)
    sources = list(set(d.metadata.get("source", "unknown") for d in result.get("context", [])))
    return {"answer": result["answer"], "sources": sources}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "index":
        print("Usage: python -m rag_model.rag_utils index <tenant_id>")
        sys.exit(1)
    tenant_id = int(sys.argv[2])
    index_tenant_files(tenant_id)