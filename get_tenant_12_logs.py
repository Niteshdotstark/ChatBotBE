import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='ap-south-1')

start_time = int((datetime.utcnow() - timedelta(hours=1)).timestamp() * 1000)

response = logs_client.filter_log_events(
    logGroupName='/aws/lambda/rag-indexer-lambda',
    startTime=start_time,
    limit=500
)

events = response.get('events', [])

# Find tenant 12 processing
tenant_12_start = None
tenant_12_logs = []

for i, event in enumerate(events):
    message = event['message'].strip()
    
    if 'Starting indexing for tenant 12' in message:
        tenant_12_start = i
    
    if tenant_12_start is not None:
        tenant_12_logs.append(message)
        
        # Stop at END RequestId or next tenant
        if 'END RequestId' in message or 'Starting indexing for tenant' in message and i > tenant_12_start:
            break

print("="*70)
print("TENANT 12 INDEXING LOGS")
print("="*70)

for log in tenant_12_logs:
    print(log)

print("\n" + "="*70)
