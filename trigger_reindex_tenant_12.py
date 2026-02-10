import boto3
import json
import time

sqs = boto3.client('sqs', region_name='ap-south-1')

print("📤 Sending SQS message to trigger reindexing for tenant 12...")

response = sqs.send_message(
    QueueUrl='https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo',
    MessageBody=json.dumps({'tenant_id': 12}),
    MessageGroupId='fix-reindex',
    MessageDeduplicationId=f'fix-{int(time.time())}'
)

print(f"✅ SQS message sent successfully!")
print(f"   Message ID: {response['MessageId']}")
print(f"\n⏳ Waiting 45 seconds for Lambda to process...")

time.sleep(45)

print("✅ Wait complete. Lambda should have finished indexing.")
print("\nNext step: Run 'python quick_diagnosis.py' to verify the fix")
