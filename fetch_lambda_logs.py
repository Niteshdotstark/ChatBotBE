import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='ap-south-1')

# Get logs from last hour
start_time = int((datetime.utcnow() - timedelta(hours=1)).timestamp() * 1000)

print("Fetching Lambda logs for rag-indexer-lambda...")
print("="*70)

response = logs_client.filter_log_events(
    logGroupName='/aws/lambda/rag-indexer-lambda',
    startTime=start_time,
    limit=200
)

events = response.get('events', [])

print(f"\nFound {len(events)} log events\n")

# Print all messages
for event in events:
    message = event['message'].strip()
    if message:
        print(message)

print("\n" + "="*70)
