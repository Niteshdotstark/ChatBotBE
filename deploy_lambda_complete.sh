#!/bin/bash

# Deploy complete Lambda function with all dependencies

set -e

echo "=========================================="
echo "Lambda Function Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
FUNCTION_NAME="RagIndexingFunction"
REGION="ap-south-1"
DEPLOYMENT_DIR="lambda_deployment_complete"

# Step 1: Create deployment directory
echo -e "${YELLOW}Step 1: Creating deployment directory...${NC}"
rm -rf $DEPLOYMENT_DIR
mkdir -p $DEPLOYMENT_DIR
cd $DEPLOYMENT_DIR
echo -e "${GREEN}✓ Directory created${NC}"
echo ""

# Step 2: Copy Lambda function
echo -e "${YELLOW}Step 2: Copying Lambda function...${NC}"
cp ../lambda_function_complete.py lambda_function.py
echo -e "${GREEN}✓ Function copied${NC}"
echo ""

# Step 3: Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
echo "This may take 5-10 minutes..."

pip install \
    langchain-aws \
    langchain-community \
    langchain-text-splitters \
    langchain-core \
    boto3 \
    pypdf \
    python-docx \
    -t . \
    --upgrade \
    --quiet

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 4: Create deployment package
echo -e "${YELLOW}Step 4: Creating deployment package...${NC}"
zip -r lambda_deployment.zip . -q
PACKAGE_SIZE=$(du -h lambda_deployment.zip | cut -f1)
echo -e "${GREEN}✓ Package created (Size: $PACKAGE_SIZE)${NC}"
echo ""

# Step 5: Check package size
PACKAGE_SIZE_MB=$(du -m lambda_deployment.zip | cut -f1)
if [ $PACKAGE_SIZE_MB -gt 50 ]; then
    echo -e "${YELLOW}⚠️  Package is larger than 50MB, uploading to S3...${NC}"
    
    # Upload to S3
    S3_BUCKET="your-deployment-bucket"  # Change this to your bucket
    S3_KEY="lambda_deployment.zip"
    
    read -p "Enter S3 bucket name for deployment: " S3_BUCKET
    
    aws s3 cp lambda_deployment.zip s3://$S3_BUCKET/$S3_KEY --region $REGION
    echo -e "${GREEN}✓ Uploaded to S3${NC}"
    
    # Update Lambda from S3
    echo -e "${YELLOW}Step 6: Updating Lambda function from S3...${NC}"
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --s3-bucket $S3_BUCKET \
        --s3-key $S3_KEY \
        --region $REGION
else
    # Update Lambda directly
    echo -e "${YELLOW}Step 6: Updating Lambda function...${NC}"
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda_deployment.zip \
        --region $REGION
fi

echo -e "${GREEN}✓ Lambda function updated${NC}"
echo ""

# Step 7: Update Lambda configuration
echo -e "${YELLOW}Step 7: Updating Lambda configuration...${NC}"
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --timeout 300 \
    --memory-size 3008 \
    --environment Variables="{
        S3_BUCKET_NAME=rag-chat-uploads,
        S3_VECTORS_BUCKET_NAME=rag-vectordb-bucket,
        S3_VECTORS_INDEX_NAME=tenant-knowledge-index,
        AWS_DEFAULT_REGION=ap-south-1
    }" \
    --region $REGION \
    --no-cli-pager

echo -e "${GREEN}✓ Configuration updated${NC}"
echo ""

# Step 8: Test Lambda function
echo -e "${YELLOW}Step 8: Testing Lambda function...${NC}"
read -p "Enter tenant ID to test (or press Enter to skip): " TEST_TENANT_ID

if [ ! -z "$TEST_TENANT_ID" ]; then
    echo "Testing with tenant $TEST_TENANT_ID..."
    
    # Create test event
    cat > test_event.json <<EOF
{
  "Records": [
    {
      "eventSource": "aws:sqs",
      "body": "{\"tenant_id\": $TEST_TENANT_ID}",
      "messageId": "test-message-1"
    }
  ]
}
EOF
    
    # Invoke Lambda
    aws lambda invoke \
        --function-name $FUNCTION_NAME \
        --payload file://test_event.json \
        --region $REGION \
        response.json
    
    echo ""
    echo "Lambda response:"
    cat response.json | jq .
    echo ""
    
    # Check logs
    echo "Checking Lambda logs..."
    sleep 5
    aws logs tail /aws/lambda/$FUNCTION_NAME --since 1m --region $REGION
    
    echo -e "${GREEN}✓ Test completed${NC}"
else
    echo -e "${YELLOW}⊘ Skipped test${NC}"
fi
echo ""

# Cleanup
cd ..
echo -e "${YELLOW}Cleaning up...${NC}"
# Keep deployment directory for debugging
echo -e "${GREEN}✓ Deployment directory kept at: $DEPLOYMENT_DIR${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Lambda Function: $FUNCTION_NAME"
echo "Region: $REGION"
echo "Timeout: 300 seconds (5 minutes)"
echo "Memory: 3008 MB"
echo ""
echo "Next steps:"
echo "1. Upload a file for tenant 11 via frontend"
echo "2. Check Lambda logs:"
echo "   aws logs tail /aws/lambda/$FUNCTION_NAME --follow --region $REGION"
echo "3. Verify vectors were created"
echo ""
echo -e "${GREEN}Lambda is ready! 🚀${NC}"
