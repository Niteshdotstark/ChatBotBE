#!/usr/bin/env python3
"""
Send test SQS message from local machine to trigger Lambda
"""
import boto3
import json
import time
from datetime import datetime

QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo"
REGION = "ap-south-1"
TENANT_ID = 11

print("=" * 80)
print("SENDING TEST MESSAGE TO SQS")
print("=" * 80)
print(f"Queue: {QUEUE_URL}")
print(f"Region: {REGION}")
print(f"Tenant ID: {TENANT_ID}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    sqs = boto3.client('sqs', region_name=REGION)
    
    message_body = json.dumps({"tenant_id": TENANT_ID})
    dedup_id = f"test-local-{int(time.time())}"
    
    print("📤 Sending message...")
    response = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=message_body,
        MessageGroupId="reindexing",
        MessageDeduplicationId=dedup_id
    )
    
    print("✅ Message sent successfully!")
    print(f"   Message ID: {response['MessageId']}")
    print(f"   Deduplication ID: {dedup_id}")
    print()
    
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Wait 1-2 minutes for Lambda to process")
    print("2. Check CloudWatch logs:")
    print("   - Go to CloudWatch → Log groups")
    print("   - Find: /aws/lambda/YOUR_FUNCTION_NAME")
    print("   - Look for logs with timestamp after:", datetime.now().strftime('%H:%M:%S'))
    print()
    print("3. If you see logs → Lambda trigger is working! ✅")
    print("   If no logs → Lambda trigger NOT configured ❌")
    print()
    print("4. To check queue status:")
    print("   aws sqs get-queue-attributes \\")
    print(f"     --queue-url {QUEUE_URL} \\")
    print("     --attribute-names ApproximateNumberOfMessages \\")
    print(f"     --region {REGION}")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Possible issues:")
    print("- AWS credentials not configured locally")
    print("- No permission to send to SQS queue")
    print("- Queue URL incorrect")
    print()
    print("To configure AWS credentials:")
    print("  aws configure")
    print()

print("=" * 80)
