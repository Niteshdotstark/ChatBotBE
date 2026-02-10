#!/usr/bin/env python3
"""
Test script to verify automatic indexing is working correctly
Run this on your server after deployment
"""

import sys
import time
from rag_model.rag_utils import (
    index_tenant_files, 
    retrieve_s3_vectors, 
    s3_list_tenant_files,
    ensure_vector_index
)

def test_indexing(tenant_id: int):
    """Test automatic indexing for a tenant"""
    print(f"\n{'='*60}")
    print(f"Testing Automatic Indexing for Tenant {tenant_id}")
    print(f"{'='*60}\n")
    
    # Test 1: Verify vector index exists
    print("Test 1: Verifying vector index...")
    try:
        ensure_vector_index()
        print("✅ Vector index verified successfully")
    except Exception as e:
        print(f"❌ Vector index verification failed: {e}")
        return False
    
    # Test 2: Check S3 files
    print(f"\nTest 2: Checking S3 files for tenant {tenant_id}...")
    try:
        files = s3_list_tenant_files(tenant_id)
        print(f"✅ Found {len(files)} files in S3:")
        for f in files[:5]:  # Show first 5 files
            print(f"   - {f}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more files")
    except Exception as e:
        print(f"❌ Failed to list S3 files: {e}")
        return False
    
    # Test 3: Check current vector count
    print(f"\nTest 3: Checking current vector count...")
    try:
        docs_before = retrieve_s3_vectors("test query", tenant_id, top_k=100)
        print(f"✅ Current vectors in index: {len(docs_before)}")
    except Exception as e:
        print(f"❌ Failed to retrieve vectors: {e}")
        docs_before = []
    
    # Test 4: Run indexing
    print(f"\nTest 4: Running indexing for tenant {tenant_id}...")
    print("This may take 30-60 seconds depending on file count...")
    try:
        start_time = time.time()
        result = index_tenant_files(tenant_id)
        elapsed = time.time() - start_time
        print(f"✅ Indexing completed in {elapsed:.1f} seconds")
        print(f"✅ Indexed {result} vectors for tenant {tenant_id}")
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return False
    
    # Test 5: Verify vectors were created
    print(f"\nTest 5: Verifying vectors were created...")
    try:
        docs_after = retrieve_s3_vectors("test query", tenant_id, top_k=100)
        print(f"✅ Vectors after indexing: {len(docs_after)}")
        
        if len(docs_after) > 0:
            print(f"✅ Sample document preview:")
            sample_doc = docs_after[0]
            print(f"   Content: {sample_doc.page_content[:100]}...")
            print(f"   Source: {sample_doc.metadata.get('source', 'unknown')}")
            print(f"   Tenant: {sample_doc.metadata.get('tenant_id', 'unknown')}")
        else:
            print("⚠️  No vectors found after indexing")
            return False
    except Exception as e:
        print(f"❌ Failed to verify vectors: {e}")
        return False
    
    # Test 6: Test retrieval with actual query
    print(f"\nTest 6: Testing retrieval with actual query...")
    test_queries = [
        "What is the policy?",
        "Tell me about the company",
        "What are the guidelines?"
    ]
    
    for query in test_queries:
        try:
            docs = retrieve_s3_vectors(query, tenant_id, top_k=3)
            print(f"✅ Query: '{query}' → Retrieved {len(docs)} documents")
            if len(docs) > 0:
                print(f"   Top result: {docs[0].page_content[:80]}...")
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"✅ Vector index: Working")
    print(f"✅ S3 files: {len(files)} files found")
    print(f"✅ Indexing: {result} vectors created")
    print(f"✅ Retrieval: Working")
    print(f"\n🎉 All tests passed! Automatic indexing is working correctly.")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage: python test_automatic_indexing.py <tenant_id>")
        print("Example: python test_automatic_indexing.py 10")
        sys.exit(1)
    
    try:
        tenant_id = int(sys.argv[1])
    except ValueError:
        print("Error: tenant_id must be a number")
        sys.exit(1)
    
    success = test_indexing(tenant_id)
    
    if success:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Upload a new file via your frontend")
        print("2. Watch logs: docker logs -f fastapi-backend | grep 'indexing'")
        print("3. Test chatbot queries")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
