#!/usr/bin/env python3
"""
Test S3 Vectors retrieval to see what's being returned
"""

import boto3
from langchain_aws import BedrockEmbeddings

# Configuration
TENANT_ID = 12
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION_NAME = "ap-south-1"

print("="*70)
print("🔍 TESTING S3 VECTORS RETRIEVAL")
print("="*70)

# Initialize clients
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)

# Create query embedding
query = "employee resignation"
print(f"\nQuery: {query}")
q_vec = embeddings.embed_query(query)
print(f"✅ Query embedding created ({len(q_vec)} dimensions)")

# Query S3 Vectors
print(f"\n🔍 Querying S3 Vectors (topK=60)...")

response = s3vectors_client.query_vectors(
    vectorBucketName=S3_VECTORS_BUCKET_NAME,
    indexName=S3_VECTORS_INDEX_NAME,
    queryVector={"float32": q_vec},
    topK=60,
    returnMetadata=True,
    returnDistance=True
)

all_vectors = response.get("vectors", [])
print(f"✅ Retrieved {len(all_vectors)} total vectors")

# Analyze by tenant
tenant_counts = {}
for v in all_vectors:
    tenant_id = v.get("metadata", {}).get("tenant_id", "unknown")
    tenant_counts[tenant_id] = tenant_counts.get(tenant_id, 0) + 1

print(f"\n📊 Vectors by tenant:")
for tenant_id, count in sorted(tenant_counts.items()):
    print(f"   Tenant {tenant_id}: {count} vectors")

# Check tenant 12 specifically
tenant_12_vectors = [v for v in all_vectors if str(v.get("metadata", {}).get("tenant_id")) == "12"]
print(f"\n🎯 Tenant 12 vectors in results: {len(tenant_12_vectors)}")

if len(tenant_12_vectors) == 0:
    print("\n❌ PROBLEM: Tenant 12 vectors not in top 60 results!")
    print("This means:")
    print("  1. Other tenants have more relevant vectors")
    print("  2. Need to increase topK multiplier")
    print("  3. OR tenant 12 vectors have low similarity scores")
    
    # Check if tenant 12 vectors exist at all
    print("\n🔍 Checking if tenant 12 vectors exist...")
    test_response = s3vectors_client.query_vectors(
        vectorBucketName=S3_VECTORS_BUCKET_NAME,
        indexName=S3_VECTORS_INDEX_NAME,
        queryVector={"float32": [0.0] * 1024},
        topK=100,
        returnMetadata=True
    )
    
    all_test_vectors = test_response.get("vectors", [])
    tenant_12_test = [v for v in all_test_vectors if str(v.get("metadata", {}).get("tenant_id")) == "12"]
    
    if len(tenant_12_test) > 0:
        print(f"✅ Found {len(tenant_12_test)} tenant 12 vectors with generic query")
        print("   Problem: They're not ranking high enough for semantic queries")
        print("\n🔧 SOLUTION: Increase topK multiplier from 5 to 10")
    else:
        print("❌ No tenant 12 vectors found at all!")
        print("   Indexing may have failed")
else:
    print("\n✅ Tenant 12 vectors found in results!")
    
    # Show sample
    sample = tenant_12_vectors[0]
    metadata = sample.get("metadata", {})
    
    print(f"\n📋 Sample metadata:")
    for key, value in metadata.items():
        if key == "content_preview":
            print(f"   {key}: {str(value)[:100]}...")
        else:
            print(f"   {key}: {value}")

print("\n" + "="*70)
