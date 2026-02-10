import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='ap-south-1')

# Get logs from last 5 minutes
start_time = int((datetime.utcnow() - timedelta(minutes=5)).timestamp() * 1000)

response = logs_client.filter_log_events(
    logGroupName='/aws/lambda/rag-indexer-lambda',
    startTime=start_time,
    limit=100
)

events = response.get('events', [])

print(f"Found {len(events)} events in last 5 minutes\n")

if events:
    print("Latest logs:")
    print("="*70)
    for event in events[-30:]:
        print(event['message'].strip())
else:
    print("No recent Lambda executions found")
    print("Lambda may not have been triggered yet")
