"""
Direct S3 Vectors Inspection Tool
Bypasses API to directly inspect what's stored in S3 Vectors
"""

import boto3
import json
from typing import List, Dict

# Configuration
TENANT_ID = 12
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"

# Initialize client
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)


def inspect_tenant_vectors(tenant_id: int, max_vectors: int = 30):
    """Directly inspect vectors for a tenant"""
    
    print("="*70)
    print(f"🔍 DIRECT S3 VECTORS INSPECTION")
    print(f"   Tenant ID: {tenant_id}")
    print(f"   Bucket: {S3_VECTORS_BUCKET_NAME}")
    print(f"   Index: {S3_VECTORS_INDEX_NAME}")
    print("="*70)
    
    try:
        # Query vectors for this tenant
        response = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": [0.0] * 1024},
            topK=max_vectors,
            filter={"tenant_id": {"eq": str(tenant_id)}},
            returnMetadata=True,
            returnDistance=True
        )
        
        vectors = response.get("vectors", [])
        
        if not vectors:
            print(f"\n❌ NO VECTORS FOUND for tenant {tenant_id}")
            print("\nPossible reasons:")
            print("1. Lambda indexing hasn't run yet")
            print("2. Tenant ID mismatch")
            print("3. Index doesn't exist")
            return
        
        print(f"\n✅ Found {len(vectors)} vectors for tenant {tenant_id}\n")
        
        # Analyze metadata structure
        print("="*70)
        print("📋 METADATA ANALYSIS")
        print("="*70)
        
        # Collect all unique metadata keys
        all_keys = set()
        for vec in vectors:
            metadata = vec.get("metadata", {})
            all_keys.update(metadata.keys())
        
        print(f"\n🔑 Unique Metadata Keys Found: {sorted(all_keys)}")
        
        # Check for critical keys
        critical_keys = {
            "tenant_id": "✅ Required for filtering",
            "person": "✅ Required for person-specific queries (NEW)",
            "subject": "❌ OLD KEY - should be 'person' instead",
            "content_preview": "✅ Required for retrieval",
            "source": "✅ Required for source attribution",
            "chunk_type": "✅ Optional metadata"
        }
        
        print(f"\n🔍 Critical Key Check:")
        for key, description in critical_keys.items():
            status = "PRESENT" if key in all_keys else "MISSING"
            symbol = "✅" if key in all_keys else "❌"
            
            if key == "subject":
                # Special case: subject should NOT be present
                if key in all_keys:
                    print(f"   {symbol} {key}: {status} - ⚠️ OLD KEY DETECTED! Lambda needs update")
                else:
                    print(f"   ✅ {key}: {status} - Good (using 'person' instead)")
            else:
                print(f"   {symbol} {key}: {status} - {description}")
        
        # Show sample vectors
        print("\n" + "="*70)
        print("📄 SAMPLE VECTORS (First 5)")
        print("="*70)
        
        for i, vec in enumerate(vectors[:5]):
            metadata = vec.get("metadata", {})
            distance = vec.get("distance", "N/A")
            
            print(f"\n--- Vector #{i+1} ---")
            print(f"Distance: {distance}")
            print(f"Metadata:")
            
            for key, value in sorted(metadata.items()):
                if key == "content_preview":
                    # Truncate long content
                    preview = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                    print(f"  {key}: {preview}")
                else:
                    print(f"  {key}: {value}")
        
        # Analyze person distribution
        print("\n" + "="*70)
        print("👥 PERSON DISTRIBUTION")
        print("="*70)
        
        person_counts = {}
        for vec in vectors:
            person = vec.get("metadata", {}).get("person", "unknown")
            person_counts[person] = person_counts.get(person, 0) + 1
        
        print(f"\nFound {len(person_counts)} unique persons:")
        for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {person}: {count} vectors")
        
        if "unknown" in person_counts and len(person_counts) == 1:
            print("\n⚠️ WARNING: All vectors have 'unknown' person!")
            print("This means the 'person' metadata is not being set correctly.")
            print("Check if Lambda is using 'person' key instead of 'subject'.")
        
        # Analyze chunk types
        print("\n" + "="*70)
        print("📦 CHUNK TYPE DISTRIBUTION")
        print("="*70)
        
        chunk_counts = {}
        for vec in vectors:
            chunk_type = vec.get("metadata", {}).get("chunk_type", "unknown")
            chunk_counts[chunk_type] = chunk_counts.get(chunk_type, 0) + 1
        
        print(f"\nFound {len(chunk_counts)} unique chunk types:")
        for chunk_type, count in sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {chunk_type}: {count} vectors")
        
        # Content quality check
        print("\n" + "="*70)
        print("📝 CONTENT QUALITY CHECK")
        print("="*70)
        
        content_lengths = []
        empty_content = 0
        
        for vec in vectors:
            content = vec.get("metadata", {}).get("content_preview", "")
            length = len(content)
            content_lengths.append(length)
            if length == 0:
                empty_content += 1
        
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            min_length = min(content_lengths)
            max_length = max(content_lengths)
            
            print(f"\nContent Preview Statistics:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Min length: {min_length} characters")
            print(f"  Max length: {max_length} characters")
            print(f"  Empty content: {empty_content} vectors")
            
            if empty_content > 0:
                print(f"\n⚠️ WARNING: {empty_content} vectors have empty content!")
            
            if avg_length < 100:
                print(f"\n⚠️ WARNING: Average content length is very short ({avg_length:.0f} chars)")
        
        # Final diagnosis
        print("\n" + "="*70)
        print("🏥 DIAGNOSIS")
        print("="*70)
        
        issues = []
        
        if "subject" in all_keys:
            issues.append("❌ Using old 'subject' key instead of 'person'")
        
        if "person" not in all_keys:
            issues.append("❌ Missing 'person' metadata key")
        
        if "content_preview" not in all_keys:
            issues.append("❌ Missing 'content_preview' metadata key")
        
        if "unknown" in person_counts and len(person_counts) == 1:
            issues.append("❌ All vectors have 'unknown' person")
        
        if empty_content > 0:
            issues.append(f"⚠️ {empty_content} vectors have empty content")
        
        if issues:
            print("\n🔴 ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
            
            print("\n💡 RECOMMENDED ACTIONS:")
            if "subject" in all_keys or "person" not in all_keys:
                print("  1. Update lambda_function_for_console.py to use 'person' instead of 'subject'")
                print("  2. Redeploy Lambda function")
                print("  3. Trigger reindexing for tenant 12")
            
            if empty_content > 0:
                print("  4. Check content extraction logic in Lambda")
        else:
            print("\n✅ NO ISSUES FOUND - Metadata structure looks good!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def compare_before_after(tenant_id: int):
    """Compare metadata before and after fix"""
    
    print("="*70)
    print("🔄 BEFORE/AFTER COMPARISON")
    print("="*70)
    
    try:
        response = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": [0.0] * 1024},
            topK=30,
            filter={"tenant_id": {"eq": str(tenant_id)}},
            returnMetadata=True
        )
        
        vectors = response.get("vectors", [])
        
        if not vectors:
            print("No vectors found")
            return
        
        # Check if we have both old and new format
        has_subject = any("subject" in v.get("metadata", {}) for v in vectors)
        has_person = any("person" in v.get("metadata", {}) for v in vectors)
        
        print(f"\nMetadata Format Detection:")
        print(f"  Old format (subject): {'✅ Found' if has_subject else '❌ Not found'}")
        print(f"  New format (person): {'✅ Found' if has_person else '❌ Not found'}")
        
        if has_subject and has_person:
            print("\n⚠️ MIXED FORMAT DETECTED!")
            print("Some vectors use 'subject', others use 'person'")
            print("This happens when reindexing after the fix.")
            print("\nRecommendation: Delete all old vectors and reindex completely")
        elif has_subject and not has_person:
            print("\n❌ STILL USING OLD FORMAT")
            print("Lambda function hasn't been updated yet")
        elif has_person and not has_subject:
            print("\n✅ USING NEW FORMAT")
            print("Lambda function has been updated correctly")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n🚀 Starting S3 Vectors Inspection...\n")
    
    # Run inspection
    inspect_tenant_vectors(TENANT_ID, max_vectors=30)
    
    # Compare formats
    print("\n")
    compare_before_after(TENANT_ID)
    
    print("\n✅ Inspection complete!\n")
