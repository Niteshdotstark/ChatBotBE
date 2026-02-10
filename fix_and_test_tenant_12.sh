#!/bin/bash

# Complete Fix and Test Script for Tenant 12 Metadata Issue
# This script fixes the Lambda function, redeploys it, and tests the fix

set -e  # Exit on error

echo "========================================================================"
echo "🔧 METADATA FIX AND TEST SCRIPT"
echo "   Tenant ID: 12"
echo "   Issue: Lambda uses 'subject' but API expects 'person'"
echo "========================================================================"

# Configuration
TENANT_ID=12
LAMBDA_FUNCTION_NAME="RagLambdaIndexing"
REGION="ap-south-1"

echo ""
echo "Step 1: Verify Lambda function code has been updated"
echo "--------------------------------------------------------------------"
echo "Checking lambda_function_for_console.py for 'PERSON_KEY'..."

if grep -q "PERSON_KEY = \"person\"" lambda_function_for_console.py; then
    echo "✅ Lambda code has been updated to use 'person' key"
else
    echo "❌ Lambda code still uses old 'subject' key"
    echo "Please update lambda_function_for_console.py first!"
    exit 1
fi

echo ""
echo "Step 2: Deploy updated Lambda function"
echo "--------------------------------------------------------------------"
echo "⚠️ MANUAL STEP REQUIRED:"
echo ""
echo "Option A: Deploy via AWS Console"
echo "  1. Open AWS Lambda console"
echo "  2. Find function: $LAMBDA_FUNCTION_NAME"
echo "  3. Upload lambda_function_for_console.py"
echo "  4. Click 'Deploy'"
echo ""
echo "Option B: Deploy via AWS CLI"
echo "  1. Create deployment package:"
echo "     zip lambda_function.zip lambda_function_for_console.py"
echo "  2. Update function:"
echo "     aws lambda update-function-code \\"
echo "       --function-name $LAMBDA_FUNCTION_NAME \\"
echo "       --zip-file fileb://lambda_function.zip \\"
echo "       --region $REGION"
echo ""
read -p "Press Enter after deploying Lambda function..."

echo ""
echo "Step 3: Clear old vectors for tenant $TENANT_ID"
echo "--------------------------------------------------------------------"
echo "Running Python script to delete old vectors..."

python3 << 'PYTHON_SCRIPT'
import boto3
import sys

TENANT_ID = 12
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"

s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)

try:
    print(f"🗑️ Deleting vectors for tenant {TENANT_ID}...")
    
    keys_to_delete = []
    next_token = None
    
    while True:
        query_kwargs = {
            "vectorBucketName": S3_VECTORS_BUCKET_NAME,
            "indexName": S3_VECTORS_INDEX_NAME,
            "queryVector": {"float32": [0.0] * 1024},
            "topK": 30,
            "filter": {"tenant_id": {"eq": str(TENANT_ID)}},
            "returnMetadata": False
        }
        if next_token:
            query_kwargs["nextToken"] = next_token
        
        resp = s3vectors_client.query_vectors(**query_kwargs)
        batch_keys = [item["key"] for item in resp.get("vectors", [])]
        keys_to_delete.extend(batch_keys)
        
        next_token = resp.get("nextToken")
        if not next_token:
            break
    
    if keys_to_delete:
        for i in range(0, len(keys_to_delete), 100):
            batch = keys_to_delete[i:i+100]
            s3vectors_client.delete_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                keys=batch
            )
        print(f"✅ Deleted {len(keys_to_delete)} old vectors")
    else:
        print("ℹ️ No vectors found to delete")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

PYTHON_SCRIPT

echo ""
echo "Step 4: Trigger reindexing via SQS"
echo "--------------------------------------------------------------------"
echo "Sending message to SQS queue..."

python3 << 'PYTHON_SCRIPT'
import boto3
import json
import time

TENANT_ID = 12
SQS_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo"

sqs_client = boto3.client('sqs', region_name='ap-south-1')

try:
    response = sqs_client.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps({"tenant_id": TENANT_ID}),
        MessageGroupId="fix-test",
        MessageDeduplicationId=f"fix-{int(time.time())}"
    )
    
    print(f"✅ SQS message sent: {response['MessageId']}")
    print(f"⏳ Waiting 45 seconds for Lambda to process...")
    time.sleep(45)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import sys
    sys.exit(1)

PYTHON_SCRIPT

echo ""
echo "Step 5: Inspect S3 Vectors metadata"
echo "--------------------------------------------------------------------"
echo "Running direct inspection..."

python3 test_direct_s3_vectors.py

echo ""
echo "Step 6: Test API retrieval"
echo "--------------------------------------------------------------------"
echo "Testing chatbot query..."

python3 << 'PYTHON_SCRIPT'
import requests
import json

TENANT_ID = 12
API_ENDPOINT = "http://localhost:8000"

try:
    response = requests.post(
        f"{API_ENDPOINT}/rag/ask/{TENANT_ID}",
        json={"message": "Tell me about the projects", "response_mode": "detailed"},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        
        print(f"✅ API Response ({len(answer)} chars):")
        print(f"\n{answer[:500]}...")
        
        # Check if it's a "no information" response
        no_info_phrases = ["don't have information", "no information", "couldn't find"]
        if any(phrase in answer.lower() for phrase in no_info_phrases):
            print("\n❌ Still getting 'no information' response!")
            print("The fix may not be working correctly.")
        else:
            print("\n✅ Got meaningful response - fix is working!")
    else:
        print(f"❌ API Error: HTTP {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")

PYTHON_SCRIPT

echo ""
echo "========================================================================"
echo "✅ FIX AND TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  1. Lambda function updated to use 'person' instead of 'subject'"
echo "  2. Old vectors deleted"
echo "  3. Reindexing triggered via SQS"
echo "  4. Metadata inspected"
echo "  5. API retrieval tested"
echo ""
echo "If tests passed, the fix is working correctly!"
echo "If tests failed, check the output above for details."
echo ""
