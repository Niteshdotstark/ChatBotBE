import boto3

sqs = boto3.client('sqs', region_name='ap-south-1')

print("🔍 Checking SQS Queue Status...")

r = sqs.get_queue_attributes(
    QueueUrl='https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo',
    AttributeNames=['All']
)

attrs = r['Attributes']

print(f"\n📊 Queue Statistics:")
print(f"   Messages Available: {attrs.get('ApproximateNumberOfMessages', '0')}")
print(f"   Messages In Flight: {attrs.get('ApproximateNumberOfMessagesNotVisible', '0')}")
print(f"   Messages Delayed: {attrs.get('ApproximateNumberOfMessagesDelayed', '0')}")

available = int(attrs.get('ApproximateNumberOfMessages', 0))
in_flight = int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0))

if available > 0:
    print(f"\n⚠️ {available} messages waiting to be processed!")
    print("Lambda may not be triggered or is failing")
elif in_flight > 0:
    print(f"\n⏳ {in_flight} messages currently being processed")
else:
    print("\n✅ Queue is empty - messages were processed")
    print("But no vectors found for tenant 12, so indexing likely failed")
