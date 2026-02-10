"""
Test actual chatbot query to verify it's working
"""

import requests

TENANT_ID = 12
API_ENDPOINT = "http://localhost:8000"

print("="*70)
print("🧪 TESTING ACTUAL CHATBOT QUERIES FOR TENANT 12")
print("="*70)

test_queries = [
    "How many leaves do I have?",
    "What is the leave policy?",
    "Tell me about employee onboarding",
    "What are the HR policies?",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}: {query}")
    print('='*70)
    
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
                "no data",
                "no context"
            ]
            
            has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
            
            if has_no_info:
                print(f"❌ Got 'no information' response")
                print(f"\nResponse: {answer[:300]}...")
            else:
                print(f"✅ Got meaningful response!")
                print(f"\nResponse ({len(answer)} chars):")
                print(f"{answer[:400]}...")
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"Error: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to API at {API_ENDPOINT}")
        print("Make sure your FastAPI server is running:")
        print("  docker-compose up -d")
        print("  OR")
        print("  uvicorn main:app --reload")
        break
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nIf you got meaningful responses above, your system is working!")
print("The Lambda → S3 Vectors → API → Chatbot flow is complete.")
