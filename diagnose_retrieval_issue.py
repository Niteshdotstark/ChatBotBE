#!/usr/bin/env python3
"""
Diagnostic script to understand why tenant 12 retrieval is failing.
This will show us the distribution of vectors across tenants.
"""
import boto3
import json

S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"
TENANT_ID_KEY = "tenant_id"

s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)

def diagnose_vector_distribution():
    """Check how many vectors exist for each tenant"""
    print("=" * 70)
    print("VECTOR DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Query a large number of vectors to see distribution
    try:
        resp = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": [0.0] * 1024},  # Dummy query
            topK=500,  # Get many vectors
            returnMetadata=True
        )
        
        vectors = resp.get("vectors", [])
        print(f"\n📊 Retrieved {len(vectors)} vectors from S3 Vectors")
        
        # Count by tenant
        tenant_counts = {}
        for v in vectors:
            meta = v.get("metadata", {})
            tenant_id = meta.get(TENANT_ID_KEY, "unknown")
            tenant_counts[tenant_id] = tenant_counts.get(tenant_id, 0) + 1
        
        print("\n🔢 Vector count by tenant:")
        for tenant_id in sorted(tenant_counts.keys(), key=lambda x: tenant_counts[x], reverse=True):
            count = tenant_counts[tenant_id]
            percentage = (count / len(vectors)) * 100
            print(f"  Tenant {tenant_id}: {count} vectors ({percentage:.1f}%)")
        
        # Check tenant 12 specifically
        tenant_12_count = tenant_counts.get("12", 0)
        print(f"\n🎯 Tenant 12 Analysis:")
        print(f"  - Vectors in top 500: {tenant_12_count}")
        print(f"  - Position in ranking: {list(tenant_counts.keys()).index('12') + 1 if '12' in tenant_counts else 'Not in top 500'}")
        
        if tenant_12_count == 0:
            print("\n❌ PROBLEM IDENTIFIED:")
            print("  Tenant 12 has NO vectors in the top 500 results!")
            print("  This means tenant 12's vectors are being crowded out by other tenants.")
            print("\n💡 SOLUTION:")
            print("  1. Increase topK multiplier to 50x or 100x")
            print("  2. OR enable S3 Vectors native filtering (if AWS fixed it)")
            print("  3. OR reindex tenant 12 with better embeddings")
        elif tenant_12_count < 12:
            print(f"\n⚠️ PROBLEM IDENTIFIED:")
            print(f"  Tenant 12 only has {tenant_12_count} vectors in top 500")
            print(f"  Need at least 12 for good retrieval (top_k=12)")
            print("\n💡 SOLUTION:")
            print(f"  Increase topK multiplier from 15x to at least {500 // tenant_12_count}x")
        else:
            print(f"\n✅ Tenant 12 has {tenant_12_count} vectors in top 500 - should work!")
        
    except Exception as e:
        print(f"\n❌ Error querying vectors: {e}")
        return

def test_native_filtering():
    """Test if S3 Vectors native filtering works"""
    print("\n" + "=" * 70)
    print("TESTING S3 VECTORS NATIVE FILTERING")
    print("=" * 70)
    
    try:
        resp = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": [0.0] * 1024},
            topK=12,
            filter={TENANT_ID_KEY: {"eq": "12"}},
            returnMetadata=True
        )
        
        vectors = resp.get("vectors", [])
        print(f"\n✅ Native filtering WORKS! Retrieved {len(vectors)} vectors for tenant 12")
        
        # Verify all are tenant 12
        all_tenant_12 = all(
            v.get("metadata", {}).get(TENANT_ID_KEY) == "12" 
            for v in vectors
        )
        
        if all_tenant_12:
            print("✅ All vectors are correctly filtered to tenant 12")
            print("\n💡 RECOMMENDATION: Use native filtering instead of manual filtering!")
        else:
            print("⚠️ Some vectors are NOT tenant 12 - filtering is buggy")
            
    except Exception as e:
        print(f"\n❌ Native filtering FAILED: {e}")
        print("💡 Must use manual filtering with large topK multiplier")

if __name__ == "__main__":
    diagnose_vector_distribution()
    test_native_filtering()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
