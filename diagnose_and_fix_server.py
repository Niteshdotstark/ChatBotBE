#!/usr/bin/env python3
"""
Diagnose and fix tenant 12 retrieval issue on server
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
print("🔍 DIAGNOSING TENANT 12 RETRIEVAL ISSUE")
print("="*70)

# Initialize clients
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)

print("\n✅ AWS clients initialized")

# Test 1: Check if index exists
print("\n" + "="*70)
print("TEST 1: Check if index exists")
print("="*70)

try:
    # Try to query the index
    q_vec = embeddings.embed_query("test")
    
    response = s3vectors_client.query_vectors(
        vectorBucketName=S3_VECTORS_BUCKET_NAME,
        indexName=S3_VECTORS_INDEX_NAME,
        queryVector={"float32": q_vec},
        topK=100,
        returnMetadata=True
    )
    
    all_vectors = response.get("vectors", [])
    print(f"✅ Index exists with {len(all_vectors)} total vectors")
    
    # Check which tenants have vectors
    tenants = set()
    for v in all_vectors:
        tenant_id = v.get("metadata", {}).get("tenant_id")
        if tenant_id:
            tenants.add(tenant_id)
    
    print(f"📊 Tenants with vectors: {sorted(tenants)}")
    
    # Check tenant 12 specifically
    tenant_12_vectors = [v for v in all_vectors if v.get("metadata", {}).get("tenant_id") == "12"]
    print(f"\n🎯 Tenant 12 vectors: {len(tenant_12_vectors)}")
    
    if len(tenant_12_vectors) == 0:
        print("❌ NO VECTORS FOR TENANT 12!")
        print("\nThis means Lambda indexing didn't work or vectors were deleted.")
        print("\n🔧 SOLUTION: Reindex tenant 12")
        print("\nRun this command:")
        print("  python3 -c \"from rag_model.rag_utils import index_tenant_files; index_tenant_files(12)\"")
        print("\nOR trigger via SQS:")
        print("  python3 trigger_reindex_tenant_12.py")
    else:
        print("✅ Tenant 12 has vectors!")
        
        # Check metadata structure
        sample = tenant_12_vectors[0].get("metadata", {})
        print(f"\n📋 Sample metadata keys: {list(sample.keys())}")
        
        # Check for critical keys
        has_person = "person" in sample
        has_subject = "subject" in sample
        has_content = "content_preview" in sample
        
        print(f"\n🔍 Metadata check:")
        print(f"   person key: {'✅' if has_person else '❌'}")
        print(f"   subject key: {'⚠️ OLD' if has_subject else '✅ Not present'}")
        print(f"   content_preview: {'✅' if has_content else '❌'}")
        
        if has_subject and not has_person:
            print("\n❌ PROBLEM: Using old 'subject' key instead of 'person'")
            print("Lambda needs to be redeployed and tenant reindexed")
        elif not has_content:
            print("\n❌ PROBLEM: Missing content_preview")
            print("Vectors are broken, need reindexing")
        else:
            print("\n✅ Metadata looks good!")
            print("The retrieval issue might be in the query logic")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
