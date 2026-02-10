import os
import boto3
import random
import time
import json
import uuid
import re
import tempfile
import shutil
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

# Safe imports (no circular dependencies)
from rag_model.config_models import ContextConfig, RetrievalConfig

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
def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    except:
        return False

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

def retrieve_s3_vectors(query: str, tenant_id: int, top_k: int = 12) -> List[Document]:
    """
    Retrieve documents using S3 Vectors, with fallback to simple text search.
    Increased top_k from 8 to 12 for better coverage of policy documents.
    """
    try:
        ensure_vector_index()
        q_vec = embeddings.embed_query(query)

        # Query without filter first (since filtering is broken)
        resp = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": q_vec},
            topK=top_k * 5,  # Increased multiplier to get more results for filtering
            returnMetadata=True,
            returnDistance=True
            # NOTE: Removed filter due to S3 Vectors filtering issues
        )
        
        docs = []
        for v in resp.get("vectors", []):
            meta = v.get("metadata", {})
            
            # Manual tenant filtering (since S3 Vectors filter is broken)
            vector_tenant_id = meta.get(TENANT_ID_KEY, "")
            if str(vector_tenant_id) != str(tenant_id):
                continue
            
            content = meta.get(CONTENT_PREVIEW_KEY, "").strip()
            if not content:
                continue
                
            doc_meta = {
                "source": meta.get(SOURCE_KEY, "S3 Vectors"),
                "tenant_id": vector_tenant_id,
                "person": meta.get("person", "unknown"),
                "chunk_type": meta.get("chunk_type", "general")
            }
            
            # Optional: Add distance score to metadata
            if "distance" in v:
                doc_meta["similarity_score"] = v["distance"]
                
            docs.append(Document(page_content=content, metadata=doc_meta))
            
            # Stop when we have enough docs for this tenant
            if len(docs) >= top_k:
                break
        
        print(f"🔍 Retrieved {len(docs)} documents for tenant {tenant_id}")
        return docs
        
    except Exception as e:
        print(f"S3 Vectors not available ({e}), falling back to simple text search...")
        
        # Fallback to simple text-based retrieval
        try:
            from rag_model.simple_text_retriever import get_simple_text_retriever
            retriever = get_simple_text_retriever()
            docs = retriever.retrieve_documents(query, tenant_id, top_k)
            print(f"Simple text retriever found {len(docs)} documents")
            return docs
        except Exception as fallback_error:
            print(f"Fallback retrieval also failed: {fallback_error}")
            return []
# ==============================================================================
# LangChain RAG Chain
# ==============================================================================
def filter_documents_by_person(documents: List[Document], person_name: str) -> List[Document]:
    """
    Filter documents to only include those that contain information about the specific person.
    This prevents cross-contamination between different people's information.
    """
    if not person_name or not documents:
        return documents
    
    person_name_lower = person_name.lower()
    filtered_docs = []
    
    for doc in documents:
        content = doc.page_content.lower()
        source = doc.metadata.get('source', '').lower()
        
        # Check if the document source contains the person's name
        if person_name_lower in source:
            filtered_docs.append(doc)
            continue
            
        # Check if the document content contains the person's name
        if person_name_lower in content:
            filtered_docs.append(doc)
            continue
            
        # Special handling for common name variations
        if person_name_lower == "nitesh":
            if "nitesh" in content or "nitesh" in source:
                filtered_docs.append(doc)
        elif person_name_lower == "mahi":
            if "mahi" in content or "mahi" in source or "maheshwari" in content:
                filtered_docs.append(doc)
    
    print(f"🔍 Filtered {len(documents)} → {len(filtered_docs)} documents for {person_name}")
    return filtered_docs

class S3VectorRetriever(BaseRetriever):
    tenant_id: int
    top_k: int = 8
    person_filter: str = None
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get all relevant documents
        docs = retrieve_s3_vectors(query, self.tenant_id, self.top_k)
        
        # If we have a person filter, apply it
        if self.person_filter:
            docs = filter_documents_by_person(docs, self.person_filter)
        
        return docs
        
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
6. For instructional content (how-to guides, procedures, reporting steps), provide comprehensive details with all available options and steps.
7. For general conversation, keep responses natural and friendly.
8. Use contractions (it's, don't, that's) to sound human.
9. Never invent information that isn't in the context or knowledge base.
10. Avoid robotic or apologetic language — speak casually, like chatting with a real person.
11. When multiple options or methods are available, present them all clearly with proper structure.
12. CRITICAL: When asked about a specific person, ONLY provide information about that exact person. DO NOT mix information from different people's documents.
13. DOCUMENT ATTRIBUTION: Before mentioning any project, skill, or experience, verify it belongs to the person being asked about. If a document mentions multiple people, only extract information that clearly belongs to the specific person in question.
14. CROSS-CONTAMINATION PREVENTION: If you're unsure whether information belongs to the person being asked about, do not include it. It's better to say "I don't have that information" than to provide incorrect attribution.
15. PRONOUN RESOLUTION: When users use pronouns like "he", "she", "it", "they", "his", "her", etc., look at the conversation history to understand who or what they're referring to. For example:
    - If previous message mentioned "Nitesh" and user asks "How many years of experience he has?", understand that "he" = "Nitesh"
    - If previous message mentioned "the project" and user asks "What technologies does it use?", understand that "it" = "the project"
16. CONTEXT CONTINUITY: Always consider the full conversation when answering follow-up questions.
17. VERIFICATION RULE: Before stating any fact about a person, ask yourself: "Does this information clearly and explicitly belong to [person's name] in the source documents?"
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(
        """Retrieved Context (PRIMARY SOURCE - USE THIS FIRST):\n{context}\n\nConversation History (for pronoun resolution only - DO NOT use for facts):\n{chat_history}\n\nQuestion: {input}\n\nCRITICAL INSTRUCTIONS:\n1. Base your answer ONLY on the Retrieved Context above\n2. If the Retrieved Context contains information about the person asked, use ONLY that information\n3. IGNORE any facts from Conversation History that contradict the Retrieved Context\n4. If the Retrieved Context doesn't contain information about the specific person, say "I don't have information about [person's name] in my knowledge base"\n5. DO NOT mix information from different people under any circumstances\n\nAnswer:"""
    )
])

def extract_person_name_from_query(query: str, chat_history: list = None) -> str:
    """
    Extract the person's name from the query or chat history.
    """
    query_lower = query.lower()
    
    # Direct name mentions in current query
    if "nitesh" in query_lower:
        return "Nitesh"
    elif "mahi" in query_lower:
        return "Mahi"
    
    # Check for pronouns and resolve from chat history
    pronouns = ["he", "his", "him", "she", "her", "they", "their"]
    if any(pronoun in query_lower for pronoun in pronouns) and chat_history:
        # Look at recent chat history to find the person being referenced
        for message in reversed(chat_history[-6:]):  # Last 3 exchanges
            if hasattr(message, 'content'):
                content = message.content.lower()
                if "nitesh" in content:
                    return "Nitesh"
                elif "mahi" in content:
                    return "Mahi"
    
    return None

def get_rag_chain(tenant_id: int, user_id: str = "default", initial_history: list = None):
    retriever = S3VectorRetriever(tenant_id=tenant_id)
    llm = ChatBedrock(model_id=LLM_MODEL, region_name=REGION_NAME, model_kwargs={"temperature": 0.1})
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
        # Extract person name from query and chat history
        person_name = extract_person_name_from_query(text, memory.chat_memory.messages)
        
        # Update retriever with person filter
        if person_name:
            retriever.person_filter = person_name
            print(f"🎯 Filtering documents for: {person_name}")
        
        result = rag_chain.invoke({"input": text, "chat_history": memory.chat_memory.messages})
        memory.save_context({"input": text}, {"output": result.get("answer", "")})
        return result

    return invoke

def answer_question_modern(question: str, tenant_id: int, user_id: str = "default", context_messages: list = None, response_mode: str = "detailed"):
    """
    Enhanced answer function with intelligent query processing and support for different response modes
    
    Args:
        question: User's question
        tenant_id: Tenant ID
        user_id: User ID
        context_messages: Conversation context
        response_mode: "detailed", "summary", or "both"
    """
    cleaned = question.strip().lower()
    if cleaned in FIXED_RESPONSES:
        fixed_resp = FIXED_RESPONSES[cleaned]
        if isinstance(fixed_resp, dict):
            answer = fixed_resp.get("answer", str(fixed_resp))  # Extract inner 'answer' or fallback to str
        else:
            answer = str(fixed_resp)
        
        # Apply response mode to fixed responses too
        if response_mode == "summary":
            from rag_model.response_summarizer import summarize_for_social_media
            summary = summarize_for_social_media(answer)
            return {"answer": summary, "sources": ["Fixed Response"], "response_type": "summary"}
        elif response_mode == "both":
            from rag_model.response_summarizer import summarize_for_social_media
            summary = summarize_for_social_media(answer)
            return {
                "detailed_answer": answer, 
                "summary_answer": summary,
                "sources": ["Fixed Response"], 
                "response_type": "both"
            }
        else:
            return {"answer": answer.strip(), "sources": ["Fixed Response"], "response_type": "detailed"}

    # Extract person name from query and context
    person_filter = extract_person_name_from_query(question, context_messages)
    
    # Try intelligent query processing first for better semantic understanding
    try:
        from rag_model.intelligent_query_processor import answer_question_with_intelligent_processing
        print(f"🧠 Using Intelligent Query Processing with person filter: {person_filter}")
        
        result = answer_question_with_intelligent_processing(
            question, tenant_id, user_id, context_messages, response_mode
        )
        
        # If intelligent processing found good results, return them
        if result and result.get("answer") and "I don't have information" not in result["answer"]:
            return result
        else:
            print("🔄 Intelligent processing didn't find good results, trying advanced RAG...")
            
    except Exception as e:
        print(f"Intelligent query processing failed: {e}")
    
    # Try advanced RAG system as fallback
    try:
        from rag_model.advanced_aws_rag import answer_question_advanced
        print(f"🚀 Using Advanced RAG System with person filter: {person_filter}")
        
        result = answer_question_advanced(question, tenant_id, person_filter)
        
        # Apply intelligent fallback if the response is too generic
        from rag_model.intelligent_fallback import enhance_response_with_fallback
        enhanced_answer = enhance_response_with_fallback(
            question, 
            result["answer"], 
            []  # Advanced system handles document context internally
        )
        result["answer"] = enhanced_answer
        
        # Apply response mode
        if response_mode == "summary":
            from rag_model.response_summarizer import summarize_for_social_media
            summary = summarize_for_social_media(result["answer"])
            return {
                "answer": summary, 
                "sources": result.get("sources", [])[:2],  # Limit sources for summary
                "response_type": "summary"
            }
        elif response_mode == "both":
            from rag_model.response_summarizer import summarize_for_social_media
            summary = summarize_for_social_media(result["answer"])
            return {
                "detailed_answer": result["answer"],
                "summary_answer": summary,
                "sources": result.get("sources", []),
                "response_type": "both"
            }
        else:
            result["response_type"] = "detailed"
            return result
            
    except Exception as e:
        print(f"Advanced RAG failed, falling back to enhanced: {e}")
        
        # Try enhanced context-aware response generation
        try:
            result = answer_question_enhanced(question, tenant_id, user_id, context_messages)
            
            # Apply intelligent fallback if the response is too generic
            from rag_model.intelligent_fallback import enhance_response_with_fallback
            enhanced_answer = enhance_response_with_fallback(
                question, 
                result["answer"], 
                result.get("retrieved_docs", [])
            )
            result["answer"] = enhanced_answer
            
            # Apply response mode
            if response_mode == "summary":
                from rag_model.response_summarizer import summarize_for_social_media
                summary = summarize_for_social_media(result["answer"])
                return {
                    "answer": summary, 
                    "sources": result.get("sources", [])[:2],  # Limit sources for summary
                    "response_type": "summary"
                }
            elif response_mode == "both":
                from rag_model.response_summarizer import summarize_for_social_media
                summary = summarize_for_social_media(result["answer"])
                return {
                    "detailed_answer": result["answer"],
                    "summary_answer": summary,
                    "sources": result.get("sources", []),
                    "response_type": "both"
                }
            else:
                result["response_type"] = "detailed"
                return result
                
        except Exception as e:
            print(f"Enhanced response generation failed, falling back to standard: {e}")
            # Fallback to original implementation
            chain = get_rag_chain(tenant_id, user_id, initial_history=context_messages)
            result = chain(question)
            sources = list(set(d.metadata.get("source", "unknown") for d in result.get("context", [])))
            
            # Apply intelligent fallback to standard response too
            from rag_model.intelligent_fallback import enhance_response_with_fallback
            enhanced_answer = enhance_response_with_fallback(
                question, 
                result["answer"], 
                result.get("context", [])
            )
            
            # Apply response mode to fallback response
            if response_mode == "summary":
                from rag_model.response_summarizer import summarize_for_social_media
                summary = summarize_for_social_media(enhanced_answer)
                return {
                    "answer": summary, 
                    "sources": sources[:2],
                    "response_type": "summary"
                }
            elif response_mode == "both":
                from rag_model.response_summarizer import summarize_for_social_media
                summary = summarize_for_social_media(enhanced_answer)
                return {
                    "detailed_answer": enhanced_answer,
                    "summary_answer": summary,
                    "sources": sources,
                    "response_type": "both"
                }
            else:
                return {
                    "answer": enhanced_answer, 
                    "sources": sources,
                    "response_type": "detailed"
                }


def answer_question_enhanced(question: str, tenant_id: int, user_id: str = "default", context_messages: list = None):
    """
    Enhanced question answering with context-aware response generation.
    Implements Requirements 6.1, 6.2, 6.3, 6.4 for context integration, conflict detection, 
    topic shift handling, and clarification support.
    """
    try:
        # Import advanced RAG components
        from rag_model.context_manager import EnhancedContextManager
        from rag_model.dynamic_retriever import DynamicRetriever
        from rag_model.context_aware_response import ContextAwareResponseGenerator
        from rag_model.topic_clarification_handler import (
            detect_clarification_request, detect_topic_shift_with_adaptation, adapt_retrieval_strategy
        )
        from rag_model.config_manager import get_tenant_config
        
        # Get tenant configuration
        try:
            config = get_tenant_config(tenant_id)
            context_config = config.context
            retrieval_config = config.retrieval
        except Exception as e:
            print(f"Failed to load tenant config, using defaults: {e}")
            context_config = ContextConfig()
            retrieval_config = RetrievalConfig()
        
        # Initialize enhanced components
        context_manager = EnhancedContextManager(context_config)
        dynamic_retriever = DynamicRetriever(retrieval_config)
        response_generator = ContextAwareResponseGenerator()
        
        # Get enhanced conversation context
        conversation_context = context_manager.get_conversation_context(
            tenant_id, user_id, question, max_tokens=context_config.max_context_tokens
        )
        
        # Detect topic shifts and clarification requests
        topic_shift = detect_topic_shift_with_adaptation(question, conversation_context)
        clarification_request = detect_clarification_request(question, conversation_context)
        
        # Adapt retrieval strategy based on topic shifts and clarifications
        adapted_retrieval_config = adapt_retrieval_strategy(
            topic_shift, clarification_request, retrieval_config
        )
        
        # Update dynamic retriever with adapted configuration
        dynamic_retriever.base_config = adapted_retrieval_config
        
        # Perform dynamic retrieval with context awareness
        retrieved_docs, retrieval_metadata = dynamic_retriever.analyze_and_retrieve(
            question, tenant_id, conversation_context, retrieve_s3_vectors
        )
        
        # Generate context-aware response
        response_data = response_generator.generate_enhanced_response(
            question, retrieved_docs, conversation_context, 
            retrieval_metadata.get("query_analysis")
        )
        
        # Update conversation context with the new exchange
        context_manager.update_context(tenant_id, user_id, question, response_data["response"])
        
        return {
            "answer": response_data["response"],
            "sources": response_data["sources"],
            "metadata": {
                "retrieval": retrieval_metadata,
                "response": response_data["metadata"],
                "topic_shift": {
                    "detected": topic_shift.shift_type.value != "no_shift",
                    "type": topic_shift.shift_type.value,
                    "adaptation_applied": topic_shift.retrieval_adaptation_needed
                },
                "clarification": {
                    "detected": clarification_request.is_clarification,
                    "type": clarification_request.clarification_type.value if clarification_request.is_clarification else None,
                    "handled": clarification_request.is_clarification
                }
            }
        }
    except Exception as e:
        print(f"Enhanced response generation failed: {e}")
        # Fallback to original implementation
        raise e

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "index":
        print("Usage: python -m rag_model.rag_utils index <tenant_id>")
        sys.exit(1)
    tenant_id = int(sys.argv[2])
    index_tenant_files(tenant_id)