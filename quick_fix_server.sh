#!/bin/bash
# Quick fix script for server deployment issues
# Run this on your server to diagnose and fix the chatbot

echo "🔧 RAG Chatbot Quick Fix Script"
echo "================================"
echo ""

# Check if running inside Docker container
if [ -f /.dockerenv ]; then
    echo "✅ Running inside Docker container"
    INSIDE_DOCKER=true
else
    echo "📦 Running on host - will enter Docker container"
    INSIDE_DOCKER=false
fi

# Function to run commands inside Docker if needed
run_command() {
    if [ "$INSIDE_DOCKER" = true ]; then
        eval "$1"
    else
        docker exec -it fastapi-backend bash -c "$1"
    fi
}

echo ""
echo "🔍 Step 1: Checking system health..."
run_command "python system_health_check.py"

echo ""
echo "🔍 Step 2: Running diagnostics..."
run_command "python debug_server_deployment.py"

echo ""
echo "🔄 Step 3: Reindexing tenant 26..."
run_command "python reindex_tenant_on_server.py 26"

echo ""
echo "🧪 Step 4: Testing the fix..."
run_command "python -c \"
from rag_model.rag_utils import answer_question_modern
result = answer_question_modern('Can Earned Leave be carried forward?', 26)
print('Sources:', len(result.get('sources', [])))
print('Answer:', result.get('answer', '')[:200])
\""

echo ""
echo "✅ Quick fix complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Test via web interface"
echo "2. If still not working, check SERVER_DEPLOYMENT_GUIDE.md"
echo "3. Check logs: docker logs fastapi-backend"