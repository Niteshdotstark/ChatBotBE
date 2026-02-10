"""
Simple verification script to check if Lambda indexing completed successfully
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
print("🔍 VERIFYING LAMBDA INDEXING FOR TENANT 12")
print("="*70)

try:
    # Initialize clients
    s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)
    embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=EMBEDDING_MODEL_ID)
    
    print("\n✅ AWS clients initialized")
    
    # Create a test query embedding
    print("📝 Creating test query embedding...")
    test_query = "projects"
    q_vec = embeddings.embed_query(test_query)
    print(f"✅ Query embedding created ({len(q_vec)} dimensions)")
    
    # Query S3 Vectors (without filter, then manually filter)
    print(f"\n🔍 Querying S3 Vectors for tenant {TENANT_ID}...")
    
    response = s3vectors_client.query_vectors(
        vectorBucketName=S3_VECTORS_BUCKET_NAME,
        indexName=S3_VECTORS_INDEX_NAME,
        queryVector={"float32": q_vec},
        topK=50,  # Get more to ensure we find tenant 12
        returnMetadata=True,
        returnDistance=True
    )
    
    all_vectors = response.get("vectors", [])
    print(f"✅ Retrieved {len(all_vectors)} total vectors from S3 Vectors")
    
    # Manually filter by tenant_id
    tenant_vectors = []
    for v in all_vectors:
        meta = v.get("metadata", {})
        if str(meta.get("tenant_id")) == str(TENANT_ID):
            tenant_vectors.append(v)
    
    print(f"\n📊 RESULTS FOR TENANT {TENANT_ID}:")
    print(f"   Total vectors found: {len(tenant_vectors)}")
    
    if not tenant_vectors:
        print("\n❌ NO VECTORS FOUND FOR TENANT 12")
        print("\nPossible reasons:")
        print("  1. Lambda hasn't finished indexing yet (wait longer)")
        print("  2. Lambda indexing failed (check CloudWatch logs)")
        print("  3. Lambda code wasn't deployed to AWS")
        print("\nNext steps:")
        print("  1. Check Lambda logs in AWS Console")
        print("  2. Verify Lambda function was deployed")
        print("  3. Try triggering reindexing again")
    else:
        print("\n✅ VECTORS FOUND! Checking metadata...")
        
        # Check first few vectors
        sample_size = min(3, len(tenant_vectors))
        
        has_person_key = False
        has_subject_key = False
        has_content = False
        
        for i, vec in enumerate(tenant_vectors[:sample_size]):
            meta = vec.get("metadata", {})
            
            print(f"\n--- Vector {i+1} ---")
            print(f"Keys: {list(meta.keys())}")
            
            if "person" in meta:
                has_person_key = True
                print(f"✅ 'person' key: {meta.get('person')}")
            
            if "subject" in meta:
                has_subject_key = True
                print(f"⚠️ 'subject' key: {meta.get('subject')} (OLD KEY!)")
            
            if "content_preview" in meta:
                has_content = True
                content = meta.get("content_preview", "")
                print(f"✅ 'content_preview': {len(content)} chars")
                print(f"   Preview: {content[:100]}...")
        
        # Final diagnosis
        print("\n" + "="*70)
        print("🏥 DIAGNOSIS")
        print("="*70)
        
        if has_person_key and not has_subject_key and has_content:
            print("\n✅ SUCCESS! Metadata is correct!")
            print("   - Using 'person' key (not 'subject')")
            print("   - Content preview present")
            print("   - Lambda fix is working!")
            print("\n🎉 Your chatbot should now work correctly!")
        elif has_subject_key:
            print("\n❌ PROBLEM: Still using old 'subject' key")
            print("   - Lambda code wasn't deployed to AWS")
            print("   - Or old vectors weren't deleted")
            print("\n🔧 ACTION REQUIRED:")
            print("   1. Deploy lambda_function_for_console.py to AWS Lambda")
            print("   2. Delete all vectors for tenant 12")
            print("   3. Trigger reindexing again")
        elif not has_person_key:
            print("\n❌ PROBLEM: Missing 'person' key")
            print("   - Lambda code may not be updated")
            print("\n🔧 ACTION REQUIRED:")
            print("   1. Verify lambda_function_for_console.py has PERSON_KEY")
            print("   2. Deploy to AWS Lambda")
            print("   3. Trigger reindexing")
        elif not has_content:
            print("\n⚠️ WARNING: Missing content_preview")
            print("   - Content extraction may have failed")
        
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
