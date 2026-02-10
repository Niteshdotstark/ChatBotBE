"""
Automated Lambda Deployment Script
Deploys the fixed lambda_function_for_console.py to AWS
"""

import boto3
import zipfile
import os
import time
import sys

# Configuration
LAMBDA_FUNCTION_NAME = "RagLambdaIndexing"
REGION = "ap-south-1"
LAMBDA_FILE = "lambda_function_for_console.py"
ZIP_FILE = "lambda_deployment.zip"

def verify_code_fix():
    """Verify the code has been fixed"""
    print("🔍 Step 1: Verifying code fix...")
    
    if not os.path.exists(LAMBDA_FILE):
        print(f"❌ Error: {LAMBDA_FILE} not found")
        return False
    
    with open(LAMBDA_FILE, 'r') as f:
        content = f.read()
    
    if 'PERSON_KEY = "person"' in content:
        print("✅ Code fix verified: Using PERSON_KEY")
        return True
    elif 'SUBJECT_KEY = "subject"' in content:
        print("❌ Code still uses old SUBJECT_KEY")
        print("Please update the code first!")
        return False
    else:
        print("⚠️ Warning: Could not verify metadata key")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'

def create_deployment_package():
    """Create ZIP file for Lambda deployment"""
    print("\n📦 Step 2: Creating deployment package...")
    
    try:
        # Remove old zip if exists
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
        
        # Create new zip
        with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Rename to lambda_function.py (Lambda expects this name)
            zipf.write(LAMBDA_FILE, 'lambda_function.py')
        
        size = os.path.getsize(ZIP_FILE)
        print(f"✅ Deployment package created: {ZIP_FILE} ({size} bytes)")
        return True
        
    except Exception as e:
        print(f"❌ Error creating deployment package: {e}")
        return False

def deploy_to_lambda():
    """Deploy to AWS Lambda"""
    print("\n🚀 Step 3: Deploying to AWS Lambda...")
    
    try:
        lambda_client = boto3.client('lambda', region_name=REGION)
        
        # Read zip file
        with open(ZIP_FILE, 'rb') as f:
            zip_content = f.read()
        
        # Update function code
        print(f"   Uploading to function: {LAMBDA_FUNCTION_NAME}...")
        response = lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME,
            ZipFile=zip_content
        )
        
        print(f"✅ Deployment initiated")
        print(f"   Function ARN: {response['FunctionArn']}")
        print(f"   Last Modified: {response['LastModified']}")
        
        # Wait for update to complete
        print("\n⏳ Waiting for deployment to complete...")
        waiter = lambda_client.get_waiter('function_updated')
        waiter.wait(
            FunctionName=LAMBDA_FUNCTION_NAME,
            WaiterConfig={'Delay': 2, 'MaxAttempts': 30}
        )
        
        print("✅ Deployment completed successfully!")
        return True
        
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"❌ Error: Lambda function '{LAMBDA_FUNCTION_NAME}' not found")
        print("Please check the function name and region")
        return False
        
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

def verify_deployment():
    """Verify the deployment was successful"""
    print("\n✅ Step 4: Verifying deployment...")
    
    try:
        lambda_client = boto3.client('lambda', region_name=REGION)
        
        response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        
        config = response['Configuration']
        print(f"   Function: {config['FunctionName']}")
        print(f"   Runtime: {config['Runtime']}")
        print(f"   Last Modified: {config['LastModified']}")
        print(f"   Code Size: {config['CodeSize']} bytes")
        
        # Check if modification time is recent (within last 5 minutes)
        from datetime import datetime, timezone
        last_modified = datetime.fromisoformat(config['LastModified'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        age_seconds = (now - last_modified).total_seconds()
        
        if age_seconds < 300:  # 5 minutes
            print(f"✅ Deployment verified (modified {int(age_seconds)} seconds ago)")
            return True
        else:
            print(f"⚠️ Warning: Function was last modified {int(age_seconds/60)} minutes ago")
            print("The deployment may not have been applied")
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def cleanup():
    """Clean up temporary files"""
    print("\n🧹 Cleaning up...")
    try:
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
            print(f"✅ Removed {ZIP_FILE}")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main deployment flow"""
    print("="*70)
    print("🚀 LAMBDA DEPLOYMENT SCRIPT")
    print(f"   Function: {LAMBDA_FUNCTION_NAME}")
    print(f"   Region: {REGION}")
    print("="*70)
    
    # Step 1: Verify code fix
    if not verify_code_fix():
        print("\n❌ Deployment aborted: Code verification failed")
        return False
    
    # Step 2: Create deployment package
    if not create_deployment_package():
        print("\n❌ Deployment aborted: Package creation failed")
        return False
    
    # Step 3: Deploy to Lambda
    if not deploy_to_lambda():
        print("\n❌ Deployment failed")
        cleanup()
        return False
    
    # Step 4: Verify deployment
    if not verify_deployment():
        print("\n⚠️ Deployment may not have been applied correctly")
        cleanup()
        return False
    
    # Cleanup
    cleanup()
    
    # Success!
    print("\n" + "="*70)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("="*70)
    print("\nNext steps:")
    print("1. Delete old vectors for tenant 12")
    print("2. Trigger reindexing via SQS")
    print("3. Run: python3 quick_diagnosis.py")
    print("4. Test chatbot queries")
    print("\nOr run the automated script:")
    print("   ./fix_and_test_tenant_12.sh")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Deployment cancelled by user")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)
