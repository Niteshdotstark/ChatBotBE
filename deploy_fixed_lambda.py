"""
Deploy the fixed Lambda function to rag-indexer-lambda
"""

import boto3
import zipfile
import os
import time

LAMBDA_FUNCTION_NAME = "rag-indexer-lambda"
REGION = "ap-south-1"
LAMBDA_FILE = "index_handler.py"  # Must match the handler name
ZIP_FILE = "lambda_deployment.zip"

print("="*70)
print("🚀 DEPLOYING FIXED LAMBDA FUNCTION")
print(f"   Function: {LAMBDA_FUNCTION_NAME}")
print(f"   File: {LAMBDA_FILE}")
print("="*70)

# Step 1: Verify file exists
if not os.path.exists(LAMBDA_FILE):
    print(f"\n❌ Error: {LAMBDA_FILE} not found")
    print("Creating it from lambda_function_for_console.py...")
    import shutil
    shutil.copy("lambda_function_for_console.py", LAMBDA_FILE)
    print(f"✅ Created {LAMBDA_FILE}")

# Step 2: Verify the fix
print("\n🔍 Verifying code fix...")
with open(LAMBDA_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

if 'PERSON_KEY = "person"' in content:
    print("✅ Code has correct PERSON_KEY")
elif 'SUBJECT_KEY = "subject"' in content:
    print("❌ ERROR: Code still has old SUBJECT_KEY!")
    print("Please update the code first!")
    exit(1)
else:
    print("⚠️ Warning: Could not verify metadata key")

# Step 3: Create ZIP
print("\n📦 Creating deployment package...")
if os.path.exists(ZIP_FILE):
    os.remove(ZIP_FILE)

with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(LAMBDA_FILE)

size = os.path.getsize(ZIP_FILE)
print(f"✅ Package created: {size} bytes")

# Step 4: Deploy
print(f"\n🚀 Deploying to {LAMBDA_FUNCTION_NAME}...")

try:
    lambda_client = boto3.client('lambda', region_name=REGION)
    
    with open(ZIP_FILE, 'rb') as f:
        zip_content = f.read()
    
    response = lambda_client.update_function_code(
        FunctionName=LAMBDA_FUNCTION_NAME,
        ZipFile=zip_content
    )
    
    print(f"✅ Deployment initiated")
    print(f"   Last Modified: {response['LastModified']}")
    
    # Wait for update
    print("\n⏳ Waiting for deployment to complete...")
    waiter = lambda_client.get_waiter('function_updated')
    waiter.wait(
        FunctionName=LAMBDA_FUNCTION_NAME,
        WaiterConfig={'Delay': 2, 'MaxAttempts': 30}
    )
    
    print("✅ Deployment completed!")
    
    # Cleanup
    os.remove(ZIP_FILE)
    print(f"🧹 Cleaned up {ZIP_FILE}")
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Lambda function updated with fixed code")
    print("="*70)
    print("\nNext steps:")
    print("1. Trigger reindexing: python trigger_reindex_tenant_12.py")
    print("2. Verify: python verify_indexing_complete.py")
    
except Exception as e:
    print(f"\n❌ Deployment failed: {e}")
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
    exit(1)
