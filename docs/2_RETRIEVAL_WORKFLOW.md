# Query Retrieval Workflow

Complete guide to how user queries are processed, documents are retrieved from isolated indexes, and responses are generated.

## Overview

```
User Query → API → Embedding → S3 Vectors Query (Isolated Index) → Retrieved Docs → LLM → Response
```

## Step-by-Step Process

### Step 1: User Submits Query

**API Endpoint**:
```
POST /chatbot/ask?tenant_id=12
Content-Type: application/json

{
  "question": "What is the leave policy?"
}
```

**FastAPI Handler** (`main.py`):
```python
@app.post("/chatbot/ask")
async def ask_chatbot(
    question: str,
    tenant_id: int,
    user_id: str = "anonymous"
):
    # Process query
    result = answer_question_modern(
        question=question,
        tenant_id=tenant_id,
        user_id=user_id
    )
    
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result.get("confidence", 1.0)
    }
```

### Step 2: Query Embedding Generation

**Convert text query to vector**:

```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0"
)

query = "What is the leave policy?"
query_vector = embeddings.embed_query(query)
# Returns: [0.123, -0.456, ..., 0.321]  # 1024 dimensions
```

**Why Embedding?**
- Enables semantic search (meaning-based, not keyword-based)
- Finds similar content even with different wording
- Example: "leave policy" matches "vacation entitlement"

### Step 3: Retrieve from Isolated Index

**Key Security Feature**: Each tenant has their own index

```python
def retrieve_s3_vectors(query: str, tenant_id: int, top_k: int = 12):
    """Retrieve from tenant's ISOLATED index"""
    
    # 1. Get tenant's isolated index name
    index_name = get_tenant_index_name(tenant_id)
    # tenant_id=12 → "tenant-12-index"
    
    # 2. Ensure index exists
    ensure_vector_index(tenant_id)
    
    # 3. Generate query embedding
    q_vec = embeddings.embed_query(query)
    
    # 4. Query ONLY this tenant's index
    resp = s3vectors_client.query_vectors(
        vectorBucketName="rag-vectordb-bucket",
        indexName=index_name,  # ← ISOLATED: tenant-12-index
        queryVector={"float32": q_vec},
        topK=top_k,  # Get top 12 most similar vectors
        returnMetadata=True,
        returnDistance=True
    )
    
    # 5. Convert to documents
    docs = []
    for v in resp.get("vectors", []):
        meta = v.get("metadata", {})
        docs.append(Document(
            page_content=meta.get("content_preview"),
            metadata={
                "source": meta.get("source"),
                "tenant_id": meta.get("tenant_id"),
                "similarity_score": v.get("distance")
            }
        ))
    
    print(f"🔒 Retrieved {len(docs)} documents from ISOLATED index: {index_name}")
    return docs
```

**Similarity Search**:
- Uses cosine similarity between query vector and document vectors
- Returns top K most similar documents
- Distance score: 0.0 (identical) to 2.0 (opposite)

**Example**:
```
Query: "What is the leave policy?"
Query Vector: [0.1, 0.5, -0.3, ...]

Document Vectors in tenant-12-index:
1. [0.12, 0.48, -0.28, ...] → Distance: 0.05 ✅ (very similar)
2. [0.15, 0.52, -0.32, ...] → Distance: 0.08 ✅ (similar)
3. [-0.8, 0.1, 0.9, ...]    → Distance: 1.85 ❌ (not similar)

Returns: Documents 1 and 2
```

### Step 4: Document Filtering & Ranking

**Retrieved Documents** (top 12):
```python
[
    Document(
        page_content="Employees are entitled to 18 days of Earned Leave...",
        metadata={
            "source": "s3://bucket/policy.pdf",
            "tenant_id": "12",
            "similarity_score": 0.05
        }
    ),
    Document(
        page_content="To apply for leave, submit request 7 days in advance...",
        metadata={
            "source": "s3://bucket/policy.pdf",
            "tenant_id": "12",
            "similarity_score": 0.08
        }
    ),
    # ... 10 more documents
]
```

**Optional Person Filtering**:
```python
def filter_documents_by_person(documents, person_name):
    """Filter to specific person's documents"""
    if not person_name:
        return documents
    
    filtered = []
    for doc in documents:
        if person_name.lower() in doc.page_content.lower():
            filtered.append(doc)
    
    return filtered
```

### Step 5: Context Preparation

**Combine retrieved documents into context**:

```python
context = "\n\n".join([
    f"Source: {doc.metadata['source']}\n{doc.page_content}"
    for doc in retrieved_docs
])
```

**Example Context**:
```
Source: s3://bucket/policy.pdf
Employees are entitled to:
- 18 days of Earned Leave
- 12 days of Sick Leave
- 3 days of Casual Leave

Source: s3://bucket/policy.pdf
To apply for leave:
1. Submit request 7 days in advance
2. Get manager approval
3. Update leave calendar
```

### Step 6: Prompt Construction

**System Prompt**:
```python
system_prompt = """
You are a friendly, human-like chatbot.
Answer questions based ONLY on the provided context.
If information is not in the context, say so clearly.
Be conversational and helpful.
"""
```

**User Prompt Template**:
```python
prompt_template = """
Retrieved Context:
{context}

Question: {question}

Answer based ONLY on the context above:
"""
```

**Final Prompt**:
```
Retrieved Context:
Source: s3://bucket/policy.pdf
Employees are entitled to:
- 18 days of Earned Leave
- 12 days of Sick Leave
- 3 days of Casual Leave

Question: What is the leave policy?

Answer based ONLY on the context above:
```

### Step 7: LLM Response Generation

**Model**: Meta Llama 3 8B Instruct  
**Provider**: AWS Bedrock  
**Region**: ap-south-1

```python
from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="meta.llama3-8b-instruct-v1:0",
    region_name="ap-south-1",
    model_kwargs={"temperature": 0.1}  # Low temp for factual responses
)

# Generate response
response = llm.invoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=final_prompt)
])

answer = response.content
```

**Example Response**:
```
Based on the leave policy, employees are entitled to:
- 18 days of Earned Leave
- 12 days of Sick Leave  
- 3 days of Casual Leave

To apply for leave, you need to submit your request 7 days in advance, 
get manager approval, and update the leave calendar.
```

### Step 8: Response Post-Processing

**Add Source Attribution**:
```python
sources = list(set([
    doc.metadata.get("source", "unknown")
    for doc in retrieved_docs
]))

result = {
    "answer": answer,
    "sources": sources,
    "confidence": calculate_confidence(retrieved_docs),
    "retrieved_docs": len(retrieved_docs)
}
```

**Calculate Confidence**:
```python
def calculate_confidence(docs):
    """Calculate confidence based on similarity scores"""
    if not docs:
        return 0.0
    
    avg_score = sum(
        1.0 - doc.metadata.get("similarity_score", 1.0)
        for doc in docs
    ) / len(docs)
    
    return min(avg_score, 1.0)
```

### Step 9: Return Response to User

**API Response**:
```json
{
  "answer": "Based on the leave policy, employees are entitled to:\n- 18 days of Earned Leave\n- 12 days of Sick Leave\n- 3 days of Casual Leave\n\nTo apply for leave, you need to submit your request 7 days in advance, get manager approval, and update the leave calendar.",
  "sources": [
    "s3://rag-chat-uploads/knowledge_base/12/policy.pdf"
  ],
  "confidence": 0.95,
  "retrieved_docs": 5
}
```

## Complete Code Flow

### Main Retrieval Function (`rag_model/rag_utils.py`)

```python
def answer_question_modern(
    question: str,
    tenant_id: int,
    user_id: str = "default",
    context_messages: list = None
):
    """Main RAG pipeline"""
    
    # 1. Check fixed responses
    if question.lower() in FIXED_RESPONSES:
        return {"answer": FIXED_RESPONSES[question.lower()]}
    
    # 2. Extract person filter (if any)
    person_filter = extract_person_name_from_query(question, context_messages)
    
    # 3. Try intelligent query processing
    try:
        from rag_model.intelligent_query_processor import answer_question_with_intelligent_processing
        result = answer_question_with_intelligent_processing(
            question, tenant_id, user_id, context_messages
        )
        if result and result.get("answer"):
            return result
    except Exception as e:
        print(f"Intelligent processing failed: {e}")
    
    # 4. Fallback to advanced RAG
    try:
        from rag_model.advanced_aws_rag import answer_question_advanced
        result = answer_question_advanced(question, tenant_id, person_filter)
        return result
    except Exception as e:
        print(f"Advanced RAG failed: {e}")
    
    # 5. Final fallback to basic RAG
    chain = get_rag_chain(tenant_id, user_id, context_messages)
    result = chain(question)
    
    sources = list(set([
        d.metadata.get("source", "unknown")
        for d in result.get("context", [])
    ]))
    
    return {
        "answer": result["answer"],
        "sources": sources
    }
```

### RAG Chain (`rag_model/rag_utils.py`)

```python
def get_rag_chain(tenant_id: int, user_id: str = "default"):
    """Create RAG chain with isolated retriever"""
    
    # 1. Create retriever for tenant's isolated index
    retriever = S3VectorRetriever(tenant_id=tenant_id)
    
    # 2. Create LLM
    llm = ChatBedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={"temperature": 0.1}
    )
    
    # 3. Create memory for conversation
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    # 4. Create QA chain
    qa_chain = create_stuff_documents_chain(llm, CONVERSATIONAL_RAG_PROMPT)
    
    # 5. Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    def invoke(text: str):
        result = rag_chain.invoke({
            "input": text,
            "chat_history": memory.chat_memory.messages
        })
        memory.save_context(
            {"input": text},
            {"output": result.get("answer", "")}
        )
        return result
    
    return invoke
```

## Advanced Features

### 1. Intelligent Query Processing

**Handles complex queries**:
```python
from rag_model.intelligent_query_processor import answer_question_with_intelligent_processing

result = answer_question_with_intelligent_processing(
    question="What should I do if I fall sick?",
    tenant_id=12,
    user_id="user123",
    context_messages=previous_messages
)
```

**Features**:
- Query reformulation
- Multi-step reasoning
- Context-aware responses
- Clarification handling

### 2. Advanced RAG System

**Enhanced retrieval**:
```python
from rag_model.advanced_aws_rag import answer_question_advanced

result = answer_question_advanced(
    question="What is the escalation timeline?",
    tenant_id=12,
    person_filter="John Doe"  # Optional: filter to specific person
)
```

**Features**:
- Semantic search
- Re-ranking
- Query expansion
- Confidence scoring

### 3. Conversation Memory

**Maintains context across turns**:
```python
# Turn 1
Q: "What is the leave policy?"
A: "Employees get 18 days of Earned Leave..."

# Turn 2 (uses context from Turn 1)
Q: "How do I apply for it?"
A: "To apply for leave, submit request 7 days in advance..."
```

**Implementation**:
```python
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# Stores previous Q&A pairs
memory.save_context(
    {"input": question},
    {"output": answer}
)
```

### 4. Person-Specific Filtering

**For resume/profile queries**:
```python
# Query: "What are John's skills?"
person_filter = extract_person_name_from_query(query, chat_history)
# Returns: "John"

# Filter documents to only John's resume
docs = filter_documents_by_person(retrieved_docs, "John")
```

## Performance Optimization

### Retrieval Speed

**Before (Shared Index)**:
```
Query Time: ~500ms
- Query 600 vectors (topK * 50)
- Manual filtering by tenant_id
- Return 12 documents
```

**After (Isolated Index)**:
```
Query Time: ~100ms
- Query 12 vectors (topK)
- No filtering needed
- Return 12 documents
```

**5x Faster!**

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_embedding(text: str):
    """Cache embeddings for common queries"""
    return embeddings.embed_query(text)
```

### Batch Queries

```python
# Process multiple queries in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(retrieve_s3_vectors, q, tenant_id)
        for q in queries
    ]
    results = [f.result() for f in futures]
```

## Testing Retrieval

### Test 1: Basic Query
```python
from rag_model.rag_utils import answer_question_modern

result = answer_question_modern(
    question="What is the leave policy?",
    tenant_id=12
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Confidence: {result.get('confidence', 'N/A')}")
```

### Test 2: Check Retrieved Documents
```python
from rag_model.rag_utils import retrieve_s3_vectors

docs = retrieve_s3_vectors(
    query="leave policy",
    tenant_id=12,
    top_k=5
)

for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['source']}")
    print(f"Score: {doc.metadata.get('similarity_score', 'N/A')}")
```

### Test 3: Verify Isolation
```python
# Should only return tenant 12 data
docs_12 = retrieve_s3_vectors("test", tenant_id=12, top_k=10)
assert all(d.metadata['tenant_id'] == '12' for d in docs_12)

# Should only return tenant 26 data
docs_26 = retrieve_s3_vectors("test", tenant_id=26, top_k=10)
assert all(d.metadata['tenant_id'] == '26' for d in docs_26)

print("✅ Tenant isolation verified!")
```

## Troubleshooting

### Issue: "Retrieved 0 documents"

**Causes**:
1. Index doesn't exist
2. No vectors in index
3. Query embedding failed

**Solutions**:
```bash
# 1. Check if index exists
python3 -c "from rag_model.rag_utils import ensure_vector_index; ensure_vector_index(12)"

# 2. Reindex tenant
python3 -c "from rag_model.rag_utils import index_tenant_files; index_tenant_files(12)"

# 3. Check Bedrock permissions
aws bedrock list-foundation-models --region ap-south-1
```

### Issue: Irrelevant answers

**Causes**:
1. Poor chunking strategy
2. Low similarity threshold
3. Insufficient context

**Solutions**:
```python
# 1. Adjust chunk size
chunk_size = 2000  # Increase for more context

# 2. Increase top_k
top_k = 20  # Retrieve more documents

# 3. Add query expansion
expanded_query = f"{query} {related_terms}"
```

### Issue: Slow queries

**Causes**:
1. Large top_k value
2. Network latency
3. Cold start

**Solutions**:
```python
# 1. Reduce top_k
top_k = 8  # Instead of 12

# 2. Use caching
@lru_cache(maxsize=100)
def cached_retrieve(query, tenant_id):
    return retrieve_s3_vectors(query, tenant_id)

# 3. Warm up index
ensure_vector_index(tenant_id)  # Call on startup
```

## Best Practices

1. ✅ **Use isolated indexes** - One index per tenant for security
2. ✅ **Optimize top_k** - 12 documents is usually sufficient
3. ✅ **Add conversation memory** - For multi-turn conversations
4. ✅ **Filter by person** - For resume/profile queries
5. ✅ **Cache embeddings** - For common queries
6. ✅ **Monitor performance** - Track query latency
7. ✅ **Handle errors gracefully** - Fallback to basic search

## Next Steps

- [1_INDEXING_WORKFLOW.md](1_INDEXING_WORKFLOW.md) - Learn how documents are indexed
- [3_DEPLOYMENT_GUIDE.md](3_DEPLOYMENT_GUIDE.md) - Deploy to production
- [4_TENANT_ISOLATION.md](4_TENANT_ISOLATION.md) - Understand security architecture
