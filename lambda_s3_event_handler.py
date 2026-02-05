"""
Modified Lambda function to handle both manual triggers and S3 events
"""
import json
import boto3
from typing import Dict, Any, Optional

def extract_tenant_id_from_s3_path(s3_key: str) -> Optional[int]:
    """
    Extract tenant ID from S3 key path
    Example: knowledge_base/25/file.pdf -> 25
    """
    try:
        parts = s3_key.split('/')
        if len(parts) >= 2 and parts[0] == 'knowledge_base':
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return None

def lambda_handler(event, context):
    """
    Handle both SQS messages (manual triggers) and S3 events (automatic triggers)
    """
    
    print(f"Raw event: {json.dumps(event)}")
    
    tenant_ids_to_process = set()
    
    # Check if this is an SQS event
    if 'Records' in event:
        for record in event['Records']:
            
            # Handle SQS messages (manual triggers)
            if record.get('eventSource') == 'aws:sqs':
                try:
                    body = json.loads(record['body'])
                    
                    # Check if it's a direct tenant_id message
                    if 'tenant_id' in body:
                        tenant_id = int(body['tenant_id'])
                        tenant_ids_to_process.add(tenant_id)
                        print(f"Processing tenant {tenant_id} from SQS message {record['messageId']}")
                    
                    # Check if it's an S3 event wrapped in SQS
                    elif 'Records' in body:
                        for s3_record in body['Records']:
                            if s3_record.get('eventSource') == 'aws:s3':
                                s3_key = s3_record['s3']['object']['key']
                                tenant_id = extract_tenant_id_from_s3_path(s3_key)
                                if tenant_id:
                                    tenant_ids_to_process.add(tenant_id)
                                    print(f"Processing tenant {tenant_id} from S3 event for file: {s3_key}")
                                
                except Exception as e:
                    print(f"Error processing SQS record: {e}")
                    continue
            
            # Handle direct S3 events (if Lambda is triggered directly by S3)
            elif record.get('eventSource') == 'aws:s3':
                s3_key = record['s3']['object']['key']
                tenant_id = extract_tenant_id_from_s3_path(s3_key)
                if tenant_id:
                    tenant_ids_to_process.add(tenant_id)
                    print(f"Processing tenant {tenant_id} from direct S3 event for file: {s3_key}")
    
    # Process each unique tenant
    for tenant_id in tenant_ids_to_process:
        try:
            print(f"Starting indexing for tenant {tenant_id}")
            
            # Your existing indexing logic here
            # index_tenant_files(tenant_id)
            
            print(f"Completed indexing for tenant {tenant_id}")
            
        except Exception as e:
            print(f"Error indexing tenant {tenant_id}: {e}")
            # Continue processing other tenants
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Processed {len(tenant_ids_to_process)} tenants',
            'tenants': list(tenant_ids_to_process)
        })
    }

# Example usage and testing
if __name__ == "__main__":
    # Test with manual SQS message
    manual_event = {
        "Records": [
            {
                "eventSource": "aws:sqs",
                "body": '{"tenant_id": 25}',
                "messageId": "test-message-1"
            }
        ]
    }
    
    # Test with S3 event
    s3_event = {
        "Records": [
            {
                "eventSource": "aws:s3",
                "s3": {
                    "object": {
                        "key": "knowledge_base/25/test-file.pdf"
                    }
                }
            }
        ]
    }
    
    print("Testing manual trigger:")
    lambda_handler(manual_event, None)
    
    print("\nTesting S3 event:")
    lambda_handler(s3_event, None)