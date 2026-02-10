#!/usr/bin/env python3
"""
Debug script to compare tenant 10 (working) vs tenant 11 (not working)
Run this on your server to identify the issue
"""

import sys
import json
from rag_model.rag_utils import (
    s3_list_tenant_files,
    retrieve_s3_vectors,
    ensure_vector_index,
    s3_client,
    s3vectors_client,
    S3_BUCKET_NAME,
    S3_VECTORS_BUCKET_NAME,
    S3_VECTORS_INDEX_NAME
)

def debug_tenant(tenant_id: int):
    """Comprehensive debugging for a tenant"""
    print(f"\n{'='*70}")
    print(f"DEBUGGING TENANT {tenant_id}")
    print(f"{'='*70}\n")
    
    # Test 1: Check S3 Files
    print(f"📁 Test 1: Checking S3 files...")
    try:
        files = s3_list_tenant_files(tenant_id)
        print(f"✅ Found {len(files)} files in S3:")
        for f in files:
            print(f"   - {f}")
            # Get file size
            try:
                response = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=f)
                size_mb = response['ContentLength'] / (1024 * 1024)
                print(f"     Size: {size_mb:.2f} MB")
            except Exception as e:
                print(f"     Error getting size: {e}")
    except Exception as e:
        print(f"❌ Error listing S3 files: {e}")
        return False
    
    if len(files) == 0:
        print(f"⚠️  NO FILES FOUND for tenant {tenant_id}")
        print(f"   Check if files were uploaded to: knowledge_base/{tenant_id}/")
        return False
    
    # Test 2: Check Vector Index
    print(f"\n🔍 Test 2: Checking vector index...")
    try:
        ensure_vector_index()
        print(f"✅ Vector index '{S3_VECTORS_INDEX_NAME}' exists")
    except Exception as e:
        print(f"❌ Vector index error: {e}")
        return False
    
    # Test 3: Query Vectors (Manual Filter)
    print(f"\n🔢 Test 3: Querying vectors with manual tenant filter...")
    try:
        # Query without filter first
        resp = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": [0.0] * 1024},
            topK=100,
            returnMetadata=True
        )
        
        all_vectors = resp.get("vectors", [])
        print(f"   Total vectors in index: {len(all_vectors)}")
        
        # Manual filter by tenant
        tenant_vectors = [
            v for v in all_vectors 
            if v.get("metadata", {}).get("tenant_id") == str(tenant_id)
        ]
        print(f"✅ Vectors for tenant {tenant_id}: {len(tenant_vectors)}")
        
        if len(tenant_vectors) == 0:
            print(f"⚠️  NO VECTORS FOUND for tenant {tenant_id}")
            print(f"   This means Lambda indexing didn't run or failed")
            return False
        
        # Show sample vectors
        print(f"\n   Sample vectors (first 3):")
        for i, v in enumerate(tenant_vectors[:3]):
            meta = v.get("metadata", {})
            print(f"   {i+1}. Tenant: {meta.get('tenant_id')}")
            print(f"      Source: {meta.get('source', 'unknown')}")
            print(f"      Subject: {meta.get('subject', 'N/A')}")
            print(f"      Type: {meta.get('chunk_type', 'N/A')}")
            content = meta.get('content_preview', '')
            print(f"      Content: {content[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error querying vectors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test Retrieval with Actual Query
    print(f"\n🎯 Test 4: Testing retrieval with actual queries...")
    test_queries = [
        "What is the policy?",
        "Tell me about the company",
        "What are the guidelines?",
        "Can I use USB drive?"
    ]
    
    for query in test_queries:
        try:
            docs = retrieve_s3_vectors(query, tenant_id, top_k=5)
            print(f"   Query: '{query}'")
            print(f"   ✅ Retrieved {len(docs)} documents")
            if len(docs) > 0:
                print(f"      Top result: {docs[0].page_content[:100]}...")
            else:
                print(f"      ⚠️  No documents retrieved")
            print()
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Test 5: Check Metadata Consistency
    print(f"\n🔍 Test 5: Checking metadata consistency...")
    try:
        unique_sources = set()
        unique_subjects = set()
        unique_types = set()
        
        for v in tenant_vectors:
            meta = v.get("metadata", {})
            unique_sources.add(meta.get('source', 'unknown'))
            unique_subjects.add(meta.get('subject', 'unknown'))
            unique_types.add(meta.get('chunk_type', 'unknown'))
        
        print(f"   Unique sources: {len(unique_sources)}")
        for src in list(unique_sources)[:5]:
            print(f"      - {src}")
        
        print(f"   Unique subjects: {len(unique_subjects)}")
        for subj in list(unique_subjects)[:5]:
            print(f"      - {subj}")
        
        print(f"   Unique chunk types: {unique_types}")
        
    except Exception as e:
        print(f"❌ Metadata check error: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY FOR TENANT {tenant_id}")
    print(f"{'='*70}")
    print(f"S3 Files: {len(files)}")
    print(f"Vectors: {len(tenant_vectors)}")
    print(f"Status: {'✅ WORKING' if len(tenant_vectors) > 0 else '❌ NOT WORKING'}")
    print(f"{'='*70}\n")
    
    return len(tenant_vectors) > 0


def compare_tenants(tenant_id_1: int, tenant_id_2: int):
    """Compare two tenants side by side"""
    print(f"\n{'#'*70}")
    print(f"COMPARING TENANT {tenant_id_1} vs TENANT {tenant_id_2}")
    print(f"{'#'*70}\n")
    
    result_1 = debug_tenant(tenant_id_1)
    result_2 = debug_tenant(tenant_id_2)
    
    print(f"\n{'#'*70}")
    print(f"COMPARISON RESULTS")
    print(f"{'#'*70}")
    print(f"Tenant {tenant_id_1}: {'✅ WORKING' if result_1 else '❌ NOT WORKING'}")
    print(f"Tenant {tenant_id_2}: {'✅ WORKING' if result_2 else '❌ NOT WORKING'}")
    
    if not result_2 and result_1:
        print(f"\n🔧 DIAGNOSIS:")
        print(f"Tenant {tenant_id_2} is not working because:")
        print(f"1. Files may not be uploaded to S3")
        print(f"2. Lambda indexing may not have run")
        print(f"3. Lambda indexing may have failed")
        print(f"\n💡 SOLUTION:")
        print(f"Run manual indexing for tenant {tenant_id_2}:")
        print(f"   docker exec -it fastapi-backend python -c \"")
        print(f"   from rag_model.rag_utils import index_tenant_files")
        print(f"   result = index_tenant_files({tenant_id_2})")
        print(f"   print(f'Indexed {{result}} vectors for tenant {tenant_id_2}')")
        print(f"   \"")
    
    print(f"{'#'*70}\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_tenant_comparison.py <tenant_id>")
        print("  python debug_tenant_comparison.py <tenant_id_1> <tenant_id_2>")
        print("\nExamples:")
        print("  python debug_tenant_comparison.py 11")
        print("  python debug_tenant_comparison.py 10 11")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        tenant_id = int(sys.argv[1])
        debug_tenant(tenant_id)
    else:
        tenant_id_1 = int(sys.argv[1])
        tenant_id_2 = int(sys.argv[2])
        compare_tenants(tenant_id_1, tenant_id_2)


if __name__ == "__main__":
    main()
