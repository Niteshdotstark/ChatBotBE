#!/bin/bash

# Script to create Lambda layer for rag-indexer-lambda

set -e

echo "=========================================="
echo "Lambda Layer Creation Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
LAYER_NAME="rag-indexing-layer-v6"
LAYER_DIR="lambda-layer-creation"
ZIP_FILE="rag-layer-v6.zip"

# Step 1: Create directory
echo -e "${YELLOW}Step 1: Creating directory structure...${NC}"
rm -rf $LAYER_DIR
mkdir -p $LAYER_DIR
cd $LAYER_DIR
mkdir python
echo -e "${GREEN}✓ Directory created${NC}"
echo ""

# Step 2: Install packages
echo -e "${YELLOW}Step 2: Installing packages...${NC}"
echo "This may take 5-10 minutes..."
echo ""

pip install \
    langchain-aws \
    langchain-community \
    langchain-text-splitters \
    langchain-core \
    boto3 \
    pypdf \
    python-docx \
    beautifulsoup4 \
    lxml \
    -t python/ \
    --upgrade \
    --quiet

echo ""
echo -e "${GREEN}✓ Packages installed${NC}"
echo ""

# Step 3: Create ZIP
echo -e "${YELLOW}Step 3: Creating ZIP file...${NC}"
zip -r $ZIP_FILE python/ -q
PACKAGE_SIZE=$(du -h $ZIP_FILE | cut -f1)
echo -e "${GREEN}✓ ZIP created: $ZIP_FILE (Size: $PACKAGE_SIZE)${NC}"
echo ""

# Step 4: Move to parent directory
mv $ZIP_FILE ..
cd ..
echo -e "${GREEN}✓ Layer package ready: $ZIP_FILE${NC}"
echo ""

# Step 5: Instructions
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "Your Lambda layer is ready: $ZIP_FILE"
echo ""
echo "To upload to AWS Console:"
echo "1. Go to: https://console.aws.amazon.com/lambda/"
echo "2. Region: ap-south-1"
echo "3. Click 'Layers' in left sidebar"
echo "4. Click 'Create layer'"
echo "5. Name: $LAYER_NAME"
echo "6. Upload: $ZIP_FILE"
echo "7. Compatible runtimes: Python 3.11, Python 3.12"
echo "8. Click 'Create'"
echo ""
echo "To upload via AWS CLI:"
echo "aws lambda publish-layer-version \\"
echo "    --layer-name $LAYER_NAME \\"
echo "    --zip-file fileb://$ZIP_FILE \\"
echo "    --compatible-runtimes python3.11 python3.12 \\"
echo "    --region ap-south-1"
echo ""
echo "After creating layer, attach it to your function:"
echo "1. Go to Lambda function: rag-indexer-lambda"
echo "2. Scroll to 'Layers' section"
echo "3. Click 'Add a layer'"
echo "4. Select 'Custom layers'"
echo "5. Choose: $LAYER_NAME"
echo "6. Click 'Add'"
echo ""
echo -e "${GREEN}Layer creation complete! 🚀${NC}"
