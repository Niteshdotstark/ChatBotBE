#!/usr/bin/env python3
"""
System health check for RAG chatbot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_system_health():
    """Check system health and functionality"""
    
    print("🏥 RAG System Health Check")
    print("=" * 40)
    
    checks = []
    
    # Check 1: Import core modules
    try:
        from rag_model.rag_utils import answer_question_modern
        checks.append(("Core RAG Module", True, "✅"))
    except Exception as e:
        checks.append(("Core RAG Module", False, f"❌ {e}"))
    
    # Check 2: Import intelligent processor
    try:
        from rag_model.intelligent_query_processor import IntelligentQueryProcessor
        checks.append(("Intelligent Query Processor", True, "✅"))
    except Exception as e:
        checks.append(("Intelligent Query Processor", False, f"❌ {e}"))
    
    # Check 3: Import advanced RAG
    try:
        from rag_model.advanced_aws_rag import AdvancedRAGSystem
        checks.append(("Advanced AWS RAG", True, "✅"))
    except Exception as e:
        checks.append(("Advanced AWS RAG", False, f"❌ {e}"))
    
    # Check 4: Import response summarizer
    try:
        from rag_model.response_summarizer import summarize_for_social_media
        checks.append(("Response Summarizer", True, "✅"))
    except Exception as e:
        checks.append(("Response Summarizer", False, f"❌ {e}"))
    
    # Check 5: Import intelligent fallback
    try:
        from rag_model.intelligent_fallback import enhance_response_with_fallback
        checks.append(("Intelligent Fallback", True, "✅"))
    except Exception as e:
        checks.append(("Intelligent Fallback", False, f"❌ {e}"))
    
    # Display results
    for component, status, message in checks:
        print(f"{message} {component}")
    
    # Summary
    passed = sum(1 for _, status, _ in checks if status)
    total = len(checks)
    
    print(f"\n📊 Health Check Results: {passed}/{total} components healthy")
    
    if passed == total:
        print("🎉 System is fully operational!")
        return True
    else:
        print("⚠️  Some components need attention")
        return False

if __name__ == "__main__":
    check_system_health()
