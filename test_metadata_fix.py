"""
Comprehensive Test Suite for Metadata Mismatch Fix
Tests Lambda indexing → S3 Vectors storage → API retrieval flow
"""

import boto3
import json
import time
from typing import Dict, List

# Configuration
TENANT_ID = 12
S3_VECTORS_REGION = "ap-south-1"
S3_VECTORS_BUCKET_NAME = "rag-vectordb-bucket"
S3_VECTORS_INDEX_NAME = "tenant-knowledge-index"
SQS_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo"
API_ENDPOINT = "http://localhost:8000"  # Update with your API endpoint

# Initialize clients
s3vectors_client = boto3.client('s3vectors', region_name=S3_VECTORS_REGION)
sqs_client = boto3.client('sqs', region_name=S3_VECTORS_REGION)


class MetadataTestSuite:
    """Test suite for metadata consistency"""
    
    def __init__(self, tenant_id: int):
        self.tenant_id = tenant_id
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, message: str):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = f"{status} | {test_name}: {message}"
        print(result)
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
    
    def test_1_trigger_lambda_indexing(self) -> bool:
        """Test 1: Trigger Lambda indexing via SQS"""
        print("\n" + "="*70)
        print("TEST 1: Trigger Lambda Indexing via SQS")
        print("="*70)
        
        try:
            response = sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps({"tenant_id": self.tenant_id}),
                MessageGroupId="test-indexing",
                MessageDeduplicationId=f"test-{int(time.time())}"
            )
            
            message_id = response.get('MessageId')
            self.log_test(
                "SQS Message Sent",
                True,
                f"Message ID: {message_id}"
            )
            
            print(f"\n⏳ Waiting 30 seconds for Lambda to process...")
            time.sleep(30)
            
            return True
            
        except Exception as e:
            self.log_test("SQS Message Sent", False, str(e))
            return False
    
    def test_2_verify_vectors_exist(self) -> bool:
        """Test 2: Verify vectors were created in S3 Vectors"""
        print("\n" + "="*70)
        print("TEST 2: Verify Vectors Exist in S3 Vectors")
        print("="*70)
        
        try:
            # Query for any vectors for this tenant
            response = s3vectors_client.query_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                queryVector={"float32": [0.0] * 1024},
                topK=10,
                filter={"tenant_id": {"eq": str(self.tenant_id)}},
                returnMetadata=True
            )
            
            vectors = response.get("vectors", [])
            count = len(vectors)
            
            if count > 0:
                self.log_test(
                    "Vectors Exist",
                    True,
                    f"Found {count} vectors for tenant {self.tenant_id}"
                )
                return True
            else:
                self.log_test(
                    "Vectors Exist",
                    False,
                    f"No vectors found for tenant {self.tenant_id}"
                )
                return False
                
        except Exception as e:
            self.log_test("Vectors Exist", False, str(e))
            return False
    
    def test_3_verify_metadata_keys(self) -> Dict[str, bool]:
        """Test 3: Verify metadata contains correct keys"""
        print("\n" + "="*70)
        print("TEST 3: Verify Metadata Keys")
        print("="*70)
        
        required_keys = ["tenant_id", "source", "person", "chunk_type", "content_preview"]
        results = {}
        
        try:
            response = s3vectors_client.query_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                queryVector={"float32": [0.0] * 1024},
                topK=5,
                filter={"tenant_id": {"eq": str(self.tenant_id)}},
                returnMetadata=True
            )
            
            vectors = response.get("vectors", [])
            
            if not vectors:
                self.log_test("Metadata Keys", False, "No vectors to check")
                return results
            
            # Check first vector's metadata
            sample_metadata = vectors[0].get("metadata", {})
            
            print(f"\n📋 Sample Metadata Keys: {list(sample_metadata.keys())}")
            print(f"📋 Sample Metadata Values:")
            for key, value in sample_metadata.items():
                preview = str(value)[:100] if len(str(value)) > 100 else str(value)
                print(f"   {key}: {preview}")
            
            # Check each required key
            for key in required_keys:
                exists = key in sample_metadata
                results[key] = exists
                
                if exists:
                    value_preview = str(sample_metadata[key])[:50]
                    self.log_test(
                        f"Metadata Key: {key}",
                        True,
                        f"Present with value: {value_preview}"
                    )
                else:
                    self.log_test(
                        f"Metadata Key: {key}",
                        False,
                        "Missing from metadata"
                    )
            
            # CRITICAL: Check for old "subject" key (should NOT exist)
            if "subject" in sample_metadata:
                self.log_test(
                    "Old 'subject' Key",
                    False,
                    "⚠️ OLD KEY STILL PRESENT - Lambda not updated!"
                )
            else:
                self.log_test(
                    "Old 'subject' Key",
                    True,
                    "Correctly removed (using 'person' instead)"
                )
            
            return results
            
        except Exception as e:
            self.log_test("Metadata Keys", False, str(e))
            return results
    
    def test_4_verify_content_preview(self) -> bool:
        """Test 4: Verify content_preview contains actual text"""
        print("\n" + "="*70)
        print("TEST 4: Verify Content Preview Quality")
        print("="*70)
        
        try:
            response = s3vectors_client.query_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                queryVector={"float32": [0.0] * 1024},
                topK=3,
                filter={"tenant_id": {"eq": str(self.tenant_id)}},
                returnMetadata=True
            )
            
            vectors = response.get("vectors", [])
            
            if not vectors:
                self.log_test("Content Preview", False, "No vectors to check")
                return False
            
            all_valid = True
            for i, vec in enumerate(vectors):
                metadata = vec.get("metadata", {})
                content = metadata.get("content_preview", "")
                
                # Check content is not empty and has reasonable length
                if len(content) < 50:
                    self.log_test(
                        f"Content Preview #{i+1}",
                        False,
                        f"Too short: {len(content)} chars"
                    )
                    all_valid = False
                else:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    self.log_test(
                        f"Content Preview #{i+1}",
                        True,
                        f"Valid ({len(content)} chars): {preview}"
                    )
            
            return all_valid
            
        except Exception as e:
            self.log_test("Content Preview", False, str(e))
            return False
    
    def test_5_api_retrieval(self, test_query: str = "Tell me about the projects") -> bool:
        """Test 5: Test API retrieval with actual query"""
        print("\n" + "="*70)
        print("TEST 5: API Retrieval Test")
        print("="*70)
        
        try:
            import requests
            
            response = requests.post(
                f"{API_ENDPOINT}/rag/ask/{self.tenant_id}",
                json={"message": test_query, "response_mode": "detailed"}
            )
            
            if response.status_code != 200:
                self.log_test(
                    "API Request",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                return False
            
            data = response.json()
            answer = data.get("response", "")
            
            # Check if answer is meaningful (not a "no information" response)
            no_info_phrases = [
                "don't have information",
                "no information",
                "couldn't find",
                "not available",
                "no data"
            ]
            
            has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
            
            if has_no_info:
                self.log_test(
                    "API Response Quality",
                    False,
                    f"Got 'no information' response: {answer[:200]}"
                )
                return False
            else:
                self.log_test(
                    "API Response Quality",
                    True,
                    f"Got meaningful response ({len(answer)} chars)"
                )
                print(f"\n📝 Sample Response:\n{answer[:300]}...")
                return True
                
        except Exception as e:
            self.log_test("API Retrieval", False, str(e))
            return False
    
    def test_6_person_filtering(self) -> bool:
        """Test 6: Verify person-specific filtering works"""
        print("\n" + "="*70)
        print("TEST 6: Person-Specific Filtering")
        print("="*70)
        
        try:
            # Get all vectors and check person metadata
            response = s3vectors_client.query_vectors(
                vectorBucketName=S3_VECTORS_BUCKET_NAME,
                indexName=S3_VECTORS_INDEX_NAME,
                queryVector={"float32": [0.0] * 1024},
                topK=20,
                filter={"tenant_id": {"eq": str(self.tenant_id)}},
                returnMetadata=True
            )
            
            vectors = response.get("vectors", [])
            
            if not vectors:
                self.log_test("Person Filtering", False, "No vectors to check")
                return False
            
            # Count unique persons
            persons = set()
            for vec in vectors:
                person = vec.get("metadata", {}).get("person", "unknown")
                persons.add(person)
            
            print(f"\n👥 Found {len(persons)} unique persons: {persons}")
            
            if "unknown" in persons and len(persons) == 1:
                self.log_test(
                    "Person Metadata",
                    False,
                    "All vectors have 'unknown' person - metadata not set correctly"
                )
                return False
            else:
                self.log_test(
                    "Person Metadata",
                    True,
                    f"Found {len(persons)} distinct persons"
                )
                return True
                
        except Exception as e:
            self.log_test("Person Filtering", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🧪 METADATA FIX TEST SUITE")
        print(f"   Tenant ID: {self.tenant_id}")
        print("="*70)
        
        # Run tests in sequence
        self.test_1_trigger_lambda_indexing()
        self.test_2_verify_vectors_exist()
        self.test_3_verify_metadata_keys()
        self.test_4_verify_content_preview()
        self.test_5_api_retrieval()
        self.test_6_person_filtering()
        
        # Print summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Metadata fix is working correctly.")
        else:
            print("\n⚠️ SOME TESTS FAILED. Review the output above for details.")
        
        return passed == total


if __name__ == "__main__":
    # Run test suite
    suite = MetadataTestSuite(TENANT_ID)
    success = suite.run_all_tests()
    
    exit(0 if success else 1)
