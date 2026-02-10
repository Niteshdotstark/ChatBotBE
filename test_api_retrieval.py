import requests
import json

TENANT_ID = 12
API_ENDPOINT = "http://localhost:8000"

print("="*70)
print("🧪 TESTING API RETRIEVAL FOR TENANT 12")
print("="*70)

test_queries = [
    "Tell me about the projects",
    "What skills are mentioned?",
    "What is the experience?"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n--- Test {i}: {query} ---")
    
    try:
        response = requests.post(
            f"{API_ENDPOINT}/rag/ask/{TENANT_ID}",
            json={"message": query, "response_mode": "detailed"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
            
            # Check if it's a "no information" response
            no_info_phrases = [
                "don't have information",
                "no information",
                "couldn't find",
                "not available",
                "no data"
            ]
            
            has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
            
            if has_no_info:
                print(f"❌ FAIL: Got 'no information' response")
                print(f"   Response: {answer[:200]}...")
            else:
                print(f"✅ PASS: Got meaningful response ({len(answer)} chars)")
                print(f"   Preview: {answer[:200]}...")
        else:
            print(f"❌ FAIL: HTTP {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
