#!/bin/bash

# Quick fix script for tenant 11 indexing issue

set -e

echo "=========================================="
echo "Fixing Tenant 11 Indexing Issue"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Debug tenant 11
echo -e "${YELLOW}Step 1: Debugging tenant 11...${NC}"
docker exec -it fastapi-backend python debug_tenant_comparison.py 10 11
echo ""

# Step 2: Check if files exist in S3
echo -e "${YELLOW}Step 2: Checking S3 files for tenant 11...${NC}"
docker exec -it fastapi-backend python -c "
from rag_model.rag_utils import s3_list_tenant_files
files = s3_list_tenant_files(11)
print(f'Files in S3 for tenant 11: {len(files)}')
for f in files:
    print(f'  - {f}')
"
echo ""

# Step 3: Check current vector count
echo -e "${YELLOW}Step 3: Checking current vector count for tenant 11...${NC}"
docker exec -it fastapi-backend python -c "
from rag_model.rag_utils import retrieve_s3_vectors
docs = retrieve_s3_vectors('test query', 11, top_k=100)
print(f'Current vectors for tenant 11: {len(docs)}')
"
echo ""

# Step 4: Ask user if they want to reindex
echo -e "${YELLOW}Step 4: Reindexing tenant 11...${NC}"
read -p "Do you want to reindex tenant 11 now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting indexing for tenant 11..."
    docker exec -it fastapi-backend python -c "
from rag_model.rag_utils import index_tenant_files
result = index_tenant_files(11)
print(f'✅ Indexed {result} vectors for tenant 11')
"
    echo -e "${GREEN}✓ Indexing completed${NC}"
else
    echo -e "${YELLOW}⊘ Skipped indexing${NC}"
fi
echo ""

# Step 5: Verify vectors were created
echo -e "${YELLOW}Step 5: Verifying vectors after indexing...${NC}"
docker exec -it fastapi-backend python -c "
from rag_model.rag_utils import retrieve_s3_vectors
docs = retrieve_s3_vectors('test query', 11, top_k=100)
print(f'Vectors after indexing: {len(docs)}')
if len(docs) > 0:
    print(f'Sample document:')
    print(f'  Content: {docs[0].page_content[:100]}...')
    print(f'  Source: {docs[0].metadata.get(\"source\", \"unknown\")}')
    print(f'  Tenant: {docs[0].metadata.get(\"tenant_id\", \"unknown\")}')
"
echo ""

# Step 6: Test chatbot query
echo -e "${YELLOW}Step 6: Testing chatbot query for tenant 11...${NC}"
read -p "Enter a test query (or press Enter to skip): " test_query
if [ ! -z "$test_query" ]; then
    echo "Testing query: $test_query"
    curl -X POST "http://localhost:8000/rag/ask/11" \
      -H "Content-Type: application/json" \
      -d "{\"message\": \"$test_query\"}" | jq .
else
    echo -e "${YELLOW}⊘ Skipped query test${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Fix Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload a new file for tenant 11 via frontend"
echo "2. Check Lambda logs: aws logs tail /aws/lambda/RagIndexingFunction --follow"
echo "3. Test chatbot queries"
echo ""
echo -e "${GREEN}Tenant 11 should now be working! 🚀${NC}"
