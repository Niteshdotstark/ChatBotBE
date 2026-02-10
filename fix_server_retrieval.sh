#!/bin/bash
# Quick fix for tenant 12 retrieval issue on production server

echo "========================================================================"
echo "FIXING TENANT 12 RETRIEVAL ISSUE"
echo "========================================================================"

# Step 1: Copy updated rag_utils.py to server container
echo ""
echo "📦 Step 1: Updating rag_utils.py in container..."
docker cp ~/ChatBotBE/rag_model/rag_utils.py fastapi-backend:/app/rag_model/rag_utils.py

if [ $? -eq 0 ]; then
    echo "✅ File copied successfully"
else
    echo "❌ Failed to copy file"
    exit 1
fi

# Step 2: Restart the container to apply changes
echo ""
echo "🔄 Step 2: Restarting container..."
docker restart fastapi-backend

echo ""
echo "⏳ Waiting 10 seconds for container to restart..."
sleep 10

# Step 3: Test retrieval
echo ""
echo "🧪 Step 3: Testing retrieval..."
docker logs fastapi-backend --tail 20

echo ""
echo "========================================================================"
echo "✅ FIX APPLIED"
echo "========================================================================"
echo ""
echo "The updated code now:"
echo "  1. Tries S3 Vectors native filtering first (in case AWS fixed it)"
echo "  2. Falls back to manual filtering with topK * 50 (600 vectors)"
echo "  3. Shows diagnostic info: 'checked X vectors' to help debug"
echo ""
echo "Next steps:"
echo "  1. Test a query in your chatbot"
echo "  2. Check logs: docker logs -f fastapi-backend"
echo "  3. Look for: 'Retrieved X documents for tenant 12 (checked Y vectors)'"
echo ""
