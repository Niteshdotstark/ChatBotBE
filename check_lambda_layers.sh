#!/bin/bash

# Check Lambda layers and packages via AWS CLI

set -e

echo "=========================================="
echo "Lambda Layer Checker"
echo "=========================================="
echo ""

FUNCTION_NAME="RagIndexingFunction"
REGION="ap-south-1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Get Lambda function configuration
echo -e "${YELLOW}Step 1: Getting Lambda function configuration...${NC}"
aws lambda get-function-configuration \
    --function-name $FUNCTION_NAME \
    --region $REGION \
    --query '{Runtime:Runtime,Timeout:Timeout,Memory:MemorySize,Layers:Layers}' \
    --output json > lambda_config.json

echo -e "${GREEN}✓ Configuration retrieved${NC}"
echo ""

# Step 2: Show layers
echo -e "${YELLOW}Step 2: Checking attached layers...${NC}"
LAYERS=$(cat lambda_config.json | jq -r '.Layers[]?.Arn' 2>/dev/null)

if [ -z "$LAYERS" ]; then
    echo -e "${RED}❌ No layers attached to Lambda function${NC}"
    echo ""
    echo "You need to create and attach Lambda layers with these packages:"
    echo "  - langchain-aws"
    echo "  - langchain-community"
    echo "  - langchain-text-splitters"
    echo "  - boto3"
    echo "  - pypdf"
    echo "  - python-docx"
    echo ""
    echo "See instructions below to create layers."
    exit 1
else
    echo -e "${GREEN}✓ Found attached layers:${NC}"
    echo "$LAYERS" | while read -r layer; do
        echo "  - $layer"
    done
fi
echo ""

# Step 3: Get layer details
echo -e "${YELLOW}Step 3: Getting layer details...${NC}"
echo "$LAYERS" | while read -r layer_arn; do
    if [ ! -z "$layer_arn" ]; then
        LAYER_VERSION=$(echo $layer_arn | rev | cut -d: -f1 | rev)
        LAYER_NAME=$(echo $layer_arn | rev | cut -d: -f2 | rev | cut -d/ -f2)
        
        echo ""
        echo "Layer: $LAYER_NAME (version $LAYER_VERSION)"
        echo "ARN: $layer_arn"
        
        # Try to get layer version details
        aws lambda get-layer-version \
            --layer-name $LAYER_NAME \
            --version-number $LAYER_VERSION \
            --region $REGION \
            --query '{Description:Description,CreatedDate:CreatedDate,CompatibleRuntimes:CompatibleRuntimes}' \
            --output json 2>/dev/null || echo "  (Could not retrieve layer details)"
    fi
done
echo ""

# Step 4: Deploy checker function
echo -e "${YELLOW}Step 4: Testing packages in Lambda environment...${NC}"
read -p "Deploy package checker to Lambda? This will temporarily replace your code (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deploying checker..."
    
    # Create temporary deployment package
    mkdir -p temp_checker
    cp lambda_layer_checker.py temp_checker/lambda_function.py
    cd temp_checker
    zip -q lambda_checker.zip lambda_function.py
    
    # Update Lambda function
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda_checker.zip \
        --region $REGION \
        --no-cli-pager
    
    echo "Waiting for function to update..."
    sleep 5
    
    # Invoke function
    echo "Running package checker..."
    aws lambda invoke \
        --function-name $FUNCTION_NAME \
        --payload '{}' \
        --region $REGION \
        response.json \
        --no-cli-pager
    
    echo ""
    echo "Checker results:"
    cat response.json | jq .
    
    echo ""
    echo "Checking CloudWatch logs..."
    sleep 3
    aws logs tail /aws/lambda/$FUNCTION_NAME --since 1m --region $REGION
    
    # Cleanup
    cd ..
    rm -rf temp_checker
    
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: Your Lambda function code has been replaced with the checker.${NC}"
    echo -e "${YELLOW}   You need to redeploy your actual function code now.${NC}"
else
    echo -e "${YELLOW}⊘ Skipped package testing${NC}"
fi
echo ""

# Step 5: Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "To check packages manually:"
echo "1. Go to AWS Lambda Console"
echo "2. Open your function: $FUNCTION_NAME"
echo "3. Copy-paste lambda_layer_checker.py"
echo "4. Click Deploy and Test"
echo "5. Check logs for package availability"
echo ""
echo "To create missing layers:"
echo "1. mkdir python"
echo "2. pip install langchain-aws langchain-community boto3 pypdf python-docx -t python/"
echo "3. zip -r lambda-layer.zip python/"
echo "4. Upload to Lambda Layers in AWS Console"
echo "5. Attach layer to your function"
echo ""

# Cleanup
rm -f lambda_config.json response.json

echo -e "${GREEN}Check complete!${NC}"
