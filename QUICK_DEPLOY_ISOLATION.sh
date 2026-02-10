#!/bin/bash
# Quick deployment script for tenant isolation fix

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         TENANT ISOLATION DEPLOYMENT - CRITICAL SECURITY FIX          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will:"
echo "  1. Update Lambda function with separate indexes"
echo "  2. Reindex tenant 12 to isolated index"
echo "  3. Update API server with separate indexes"
echo "  4. Verify deployment"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

# Step 1: Update Lambda
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 1: Updating Lambda Function"
echo "═══════════════════════════════════════════════════════════════════════"

cd ~/ChatBotBE
zip -r lambda_function.zip index_handler.py

aws lambda update-function-code \
  --function-name rag-indexer-lambda \
  --zip-file fileb://lambda_function.zip \
  --region ap-south-1

if [ $? -eq 0 ]; then
    echo "✅ Lambda function updated"
else
    echo "❌ Lambda update failed"
    exit 1
fi

# Step 2: Reindex tenant 12
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 2: Reindexing Tenant 12 to Isolated Index"
echo "═══════════════════════════════════════════════════════════════════════"

python3 << 'EOF'
import boto3
import json
import time

sqs = boto3.client('sqs', region_name='ap-south-1')
response = sqs.send_message(
    QueueUrl="https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo",
    MessageBody=json.dumps({"tenant_id": 12}),
    MessageGroupId="isolated-reindex",
    MessageDeduplicationId=f"isolated-12-{int(time.time())}"
)
print(f"✅ Reindex queued for tenant 12: {response['MessageId']}")
print("⏳ Waiting 60 seconds for Lambda to process...")
time.sleep(60)
print("✅ Wait complete")
EOF

# Check Lambda logs
echo ""
echo "📋 Checking Lambda logs..."
aws logs tail /aws/lambda/rag-indexer-lambda --since 2m --region ap-south-1 | grep -E "(ISOLATED|tenant-12-index|COMPLETE)"

# Step 3: Update API server
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 3: Updating API Server"
echo "═══════════════════════════════════════════════════════════════════════"

# Note: Update SERVER_IP with your actual server IP
SERVER_IP="ec2-13-233-113-155.ap-south-1.compute.amazonaws.com"
SERVER_USER="ubuntu"

echo "Connecting to server: $SERVER_USER@$SERVER_IP"

ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
cd ~/ChatBotBE
echo "📥 Pulling latest code..."
git pull origin main

echo "📦 Copying updated file to container..."
docker cp ~/ChatBotBE/rag_model/rag_utils.py fastapi-backend:/app/rag_model/rag_utils.py

echo "🔄 Restarting container..."
docker restart fastapi-backend

echo "⏳ Waiting 10 seconds for container to restart..."
sleep 10

echo "✅ API server updated"
ENDSSH

if [ $? -eq 0 ]; then
    echo "✅ API server deployment complete"
else
    echo "❌ API server deployment failed"
    exit 1
fi

# Step 4: Verify
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 4: Verification"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "Checking API server logs..."
ssh $SERVER_USER@$SERVER_IP "docker logs fastapi-backend --tail 20" | grep -E "(ISOLATED|tenant-12-index)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ DEPLOYMENT COMPLETE                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next Steps:"
echo "  1. Test a query in your chatbot"
echo "  2. Check logs for 'ISOLATED index' messages"
echo "  3. Verify no cross-tenant data leakage"
echo ""
echo "Expected Log Messages:"
echo "  ✅ Tenant 12 isolated index 'tenant-12-index' exists and ready"
echo "  🔒 Retrieved X documents from ISOLATED index: tenant-12-index"
echo ""
echo "To monitor logs:"
echo "  ssh $SERVER_USER@$SERVER_IP 'docker logs -f fastapi-backend'"
echo ""
echo "To reindex other tenants:"
echo "  python3 reindex_all_tenants.py"
echo ""
