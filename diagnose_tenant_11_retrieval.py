#!/usr/bin/env python3
"""
Diagnose why tenant 11 retrieval is not working
"""
import os
os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"

import boto3
from langchain_aws import BedrockEmbeddings

S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
TENANT_ID = 11

print("=" * 80)
print("DIAGNOSING TENANT 11 RETRIEVAL")
print("=" * 80)
print()

# Initialize clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1")
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)

# Test query
test_query = "Can Earned Leave be carried forward?"
print(f"Test Query: '{test_query}'")
print()

# Generate embedding
print("📊 Generating query embedding...")
query_embedding = embeddings.embed_query(test_query)
print(f"✅ Embedding generated: {len(query_embedding)} dimensions")
print()

# Query vectors with tenant filter
print(f"🔍 Searching vectors for tenant {TENANT_ID}...")
try:
    # Try without filter first to see if vectors exist at all
    response = s3vectors_client.query_vectors(
        vectorBucketName=S3_VECTORS_BUCKET_NAME,
        indexName=S3_VECTORS_INDEX_NAME,
        queryVector={"float32": query_embedding},
        topK=10,
        returnMetadata=True
    )
    
    vectors = response.get("vectors", [])
    print(f"✅ Found {len(vectors)} total vectors (no filter)")
    print()
    
    if vectors:
        # Count by tenant
        tenant_counts = {}
        tenant_11_vectors = []
        
        for vec in vectors:
            metadata = vec.get("metadata", {})
            tid = metadata.get('tenant_id', 'unknown')
            tenant_counts[tid] = tenant_counts.get(tid, 0) + 1
            if tid == str(TENANT_ID):
                tenant_11_vectors.append(vec)
        
        print("📊 Vectors by Tenant ID:")
        for tid, count in sorted(tenant_counts.items()):
            marker = " ← TARGET" if tid == str(TENANT_ID) else ""
            print(f"   Tenant {tid}: {count} vectors{marker}")
        print()
        
        if tenant_11_vectors:
            print(f"✅ Found {len(tenant_11_vectors)} vectors for tenant 11!")
            print()
            print("📄 Top Results for Tenant 11:")
            print("-" * 80)
            for i, vec in enumerate(tenant_11_vectors[:5], 1):
                metadata = vec.get("metadata", {})
                print(f"\n{i}. Score: {vec.get('score', 'N/A')}")
                print(f"   Source: {metadata.get('source', 'N/A')[-50:]}")  # Last 50 chars
                print(f"   Subject: {metadata.get('subject', 'N/A')}")
                print(f"   Chunk Type: {metadata.get('chunk_type', 'N/A')}")
        else:
            print("❌ NO VECTORS FOR TENANT 11!")
            print()
            print("Issue: Vectors exist but none have tenant_id='11'")
            print("Check Lambda indexing code")
    else:
        print("❌ NO VECTORS FOUND AT ALL!")
        print()
        print("Possible issues:")
        print("1. Index is empty")
        print("2. Wrong index name")
        print("3. Wrong bucket name")
        
except Exception as e:
    print(f"❌ Error querying vectors: {e}")
    print()

# Check total vectors for tenant
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("✅ Diagnostic complete - check results above")
print()
print("If tenant 11 vectors found:")
print("  → Issue is with retrieval code in main.py")
print("  → Check filter syntax in rag_utils.py")
print()
print("If NO tenant 11 vectors:")
print("  → Lambda indexed with wrong tenant_id")
print("  → Need to reindex tenant 11")
print()
print("=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
