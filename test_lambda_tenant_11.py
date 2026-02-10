#!/usr/bin/env python3
"""
Test script to run Lambda function for tenant 11
This simulates what Lambda would do when triggered by SQS
"""
import os
os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"

from lambda_function_for_console import index_tenant_files

print("=" * 80)
print("TESTING FIXED LAMBDA FUNCTION FOR TENANT 11")
print("=" * 80)
print()

try:
    result = index_tenant_files(11)
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Result: {result}")
    print(f"\n✅ Indexed {result.get('total_chunks_added', 0)} vectors for tenant 11")
    print(f"✅ Files processed: {result.get('files_processed', 0)}")
    print()
    print("Next steps:")
    print("1. Update Lambda function code in AWS Console")
    print("2. Send test SQS message to trigger reindexing")
    print("3. Check CloudWatch logs")
    print("4. Run diagnostic again to verify all 63 vectors exist")
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED")
    print("=" * 80)
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
