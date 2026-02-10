#!/bin/bash

# Automatic Indexing Fix - Deployment Script
# This script deploys the automatic indexing fix to your server

set -e  # Exit on any error

echo "=========================================="
echo "Automatic Indexing Fix - Deployment"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if we're in the right directory
echo -e "${YELLOW}Step 1: Checking directory...${NC}"
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: main.py not found. Please run this script from the ChatBotBE directory.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Directory verified${NC}"
echo ""

# Step 2: Show current git status
echo -e "${YELLOW}Step 2: Checking git status...${NC}"
git status --short
echo ""

# Step 3: Pull latest changes (if using git)
echo -e "${YELLOW}Step 3: Pulling latest changes...${NC}"
read -p "Do you want to pull from git? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git pull origin main
    echo -e "${GREEN}✓ Git pull completed${NC}"
else
    echo -e "${YELLOW}⊘ Skipped git pull${NC}"
fi
echo ""

# Step 4: Stop containers
echo -e "${YELLOW}Step 4: Stopping Docker containers...${NC}"
docker-compose down
echo -e "${GREEN}✓ Containers stopped${NC}"
echo ""

# Step 5: Rebuild containers
echo -e "${YELLOW}Step 5: Rebuilding Docker containers...${NC}"
docker-compose build --no-cache
echo -e "${GREEN}✓ Containers rebuilt${NC}"
echo ""

# Step 6: Start containers
echo -e "${YELLOW}Step 6: Starting Docker containers...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ Containers started${NC}"
echo ""

# Step 7: Wait for containers to be ready
echo -e "${YELLOW}Step 7: Waiting for containers to be ready...${NC}"
sleep 10
echo -e "${GREEN}✓ Containers should be ready${NC}"
echo ""

# Step 8: Check container status
echo -e "${YELLOW}Step 8: Checking container status...${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Step 9: Check logs for errors
echo -e "${YELLOW}Step 9: Checking logs for errors...${NC}"
echo "Last 20 lines of fastapi-backend logs:"
docker logs --tail 20 fastapi-backend
echo ""

# Step 10: Test automatic indexing (optional)
echo -e "${YELLOW}Step 10: Test automatic indexing?${NC}"
read -p "Do you want to test indexing for a tenant? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter tenant ID to test: " tenant_id
    echo "Testing indexing for tenant $tenant_id..."
    docker exec -it fastapi-backend python -c "
from rag_model.rag_utils import index_tenant_files
result = index_tenant_files($tenant_id)
print(f'✓ Indexed {result} vectors for tenant $tenant_id')
"
    echo -e "${GREEN}✓ Test completed${NC}"
else
    echo -e "${YELLOW}⊘ Skipped indexing test${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload a file via your frontend"
echo "2. Watch logs: docker logs -f fastapi-backend | grep 'indexing'"
echo "3. Test chatbot queries"
echo ""
echo "To monitor logs in real-time:"
echo "  docker logs -f fastapi-backend"
echo ""
echo "To check vector count for a tenant:"
echo "  docker exec -it fastapi-backend python -c \"from rag_model.rag_utils import retrieve_s3_vectors; docs = retrieve_s3_vectors('test', TENANT_ID, 50); print(f'Vectors: {len(docs)}')\""
echo ""
echo -e "${GREEN}Automatic indexing is now active! 🚀${NC}"
