"""
Quick Diagnosis Script
Run this to quickly check if the metadata fix is working
"""

import boto3
import json
import sys

# Configuration
TENANT_ID = 12
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"

def quick_check():
    """Quick diagnostic check"""
    
    print("🔍 QUICK DIAGNOSIS FOR TENANT 12")
    print("="*60)
    
    try:
        s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
        
        # Query a few vectors (without filter due to S3 Vectors API limitations)
        response = s3vectors_client.query_vectors(
            vectorBucketName=S3_VECTORS_BUCKET_NAME,
            indexName=S3_VECTORS_INDEX_NAME,
            queryVector={"float32": [0.0] * 1024},
            topK=30,
            returnMetadata=True
        )
        
        vectors = response.get("vectors", [])
        
        vectors = response.get("vectors", [])
        
        if not vectors:
            print("❌ PROBLEM: No vectors found for tenant 12")
            print("\nPossible causes:")
            print("  1. Lambda hasn't run yet")
            print("  2. Indexing failed")
            print("  3. Tenant ID mismatch")
            return False
        
        print(f"✅ Found {len(vectors)} vectors\n")
        
        # Check first vector's metadata
        sample = vectors[0].get("metadata", {})
        keys = list(sample.keys())
        
        print("Metadata Keys Present:")
        for key in sorted(keys):
            print(f"  ✓ {key}")
        
        print("\nCritical Checks:")
        
        # Check 1: Has 'person' key?
        if "person" in keys:
            person_value = sample.get("person", "")
            print(f"  ✅ 'person' key: PRESENT (value: '{person_value}')")
            status_person = True
        else:
            print(f"  ❌ 'person' key: MISSING")
            status_person = False
        
        # Check 2: Has old 'subject' key?
        if "subject" in keys:
            print(f"  ❌ 'subject' key: PRESENT (should be removed!)")
            status_subject = False
        else:
            print(f"  ✅ 'subject' key: ABSENT (good)")
            status_subject = True
        
        # Check 3: Has content_preview?
        if "content_preview" in keys:
            content = sample.get("content_preview", "")
            print(f"  ✅ 'content_preview': PRESENT ({len(content)} chars)")
            status_content = True
        else:
            print(f"  ❌ 'content_preview': MISSING")
            status_content = False
        
        # Overall status
        print("\n" + "="*60)
        
        if status_person and status_subject and status_content:
            print("✅ DIAGNOSIS: METADATA IS CORRECT")
            print("\nThe fix is working! Chatbot should retrieve documents.")
            return True
        else:
            print("❌ DIAGNOSIS: METADATA HAS ISSUES")
            
            if not status_person:
                print("\n🔧 ACTION REQUIRED:")
                print("  1. Update lambda_function_for_console.py")
                print("  2. Change SUBJECT_KEY to PERSON_KEY")
                print("  3. Redeploy Lambda function")
                print("  4. Delete old vectors")
                print("  5. Trigger reindexing")
            
            if not status_subject:
                print("\n🔧 ACTION REQUIRED:")
                print("  1. Delete all vectors for tenant 12")
                print("  2. Trigger reindexing with updated Lambda")
            
            if not status_content:
                print("\n🔧 ACTION REQUIRED:")
                print("  1. Check Lambda content extraction logic")
                print("  2. Ensure content_preview is being set")
            
            return False
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_check()
    sys.exit(0 if success else 1)
