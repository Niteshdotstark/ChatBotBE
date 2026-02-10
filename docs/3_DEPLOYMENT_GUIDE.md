# Production Deployment Guide

Complete guide to deploying the multi-tenant RAG chatbot system to production.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   User      │────▶│  FastAPI    │────▶│  S3 Vectors  │
│  (Browser)  │     │   (Docker)  │     │  (Isolated)  │
└─────────────┘     └─────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   AWS S3     │
                    │  (Uploads)   │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   AWS SQS    │
                    │   (FIFO)     │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ AWS Lambda   │
                    │  (Indexing)  │
                    └──────────────┘
```

## Prerequisites

### AWS Services Required
- ✅ S3 (file storage)
- ✅ S3 Vectors (vector database)
- ✅ SQS (message queue)
- ✅ Lambda (serverless processing)
- ✅ Bedrock (LLM & embeddings)
- ✅ EC2 (API server hosting)
- ✅ IAM (permissions)

### Local Requirements
- Python 3.10+
- Docker & Docker Compose
- AWS CLI configured
- Git

## Part 1: AWS Infrastructure Setup

### Step 1: Create S3 Buckets

```bash
# File uploads bucket
aws s3 mb s3://rag-chat-uploads --region ap-south-1

# Vector database bucket (created automatically by S3 Vectors)
# No manual creation needed
```

### Step 2: Create SQS Queue

```bash
# Create FIFO queue
aws sqs create-queue \
  --queue-name RagLambdaIndexing.fifo \
  --attributes FifoQueue=true,ContentBasedDeduplication=false \
  --region ap-south-1

# Get queue URL
aws sqs get-queue-url \
  --queue-name RagLambdaIndexing.fifo \
  --region ap-south-1
```

**Output**:
```json
{
  "QueueUrl": "https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo"
}
```

### Step 3: Configure S3 Event Notification

```bash
# Create event notification configuration
cat > s3-event-config.json << 'EOF'
{
  "QueueConfigurations": [
    {
      "QueueArn": "arn:aws:sqs:ap-south-1:068733247141:RagLambdaIndexing.fifo",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "knowledge_base/"
            }
          ]
        }
      }
    }
  ]
}
EOF

# Apply configuration
aws s3api put-bucket-notification-configuration \
  --bucket rag-chat-uploads \
  --notification-configuration file://s3-event-config.json
```

### Step 4: Create IAM Role for Lambda

```bash
# Create trust policy
cat > lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name RagLambdaIndexingRole \
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name RagLambdaIndexingRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name RagLambdaIndexingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name RagLambdaIndexingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess

# Create custom policy for Bedrock and S3 Vectors
cat > lambda-custom-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3vectors:*"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name RagLambdaIndexingRole \
  --policy-name BedrockS3VectorsPolicy \
  --policy-document file://lambda-custom-policy.json
```

## Part 2: Lambda Function Deployment

### Step 1: Create Lambda Layer (Dependencies)

```bash
# Create layer directory
mkdir -p lambda-layer/python

# Install dependencies
pip install \
  langchain-aws \
  langchain-community \
  langchain-text-splitters \
  boto3 \
  pypdf \
  python-docx \
  beautifulsoup4 \
  -t lambda-layer/python/

# Create layer zip
cd lambda-layer
zip -r ../lambda-layer.zip python/
cd ..

# Upload layer
aws lambda publish-layer-version \
  --layer-name rag-dependencies \
  --zip-file fileb://lambda-layer.zip \
  --compatible-runtimes python3.10 \
  --region ap-south-1
```

**Note Layer ARN** from output:
```
arn:aws:lambda:ap-south-1:068733247141:layer:rag-dependencies:1
```

### Step 2: Create Lambda Function

```bash
# Create function zip
zip lambda_function.zip index_handler.py

# Create Lambda function
aws lambda create-function \
  --function-name rag-indexer-lambda \
  --runtime python3.10 \
  --role arn:aws:iam::068733247141:role/RagLambdaIndexingRole \
  --handler index_handler.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 300 \
  --memory-size 1024 \
  --layers arn:aws:lambda:ap-south-1:068733247141:layer:rag-dependencies:1 \
  --region ap-south-1
```

### Step 3: Configure SQS Trigger

```bash
# Add SQS as event source
aws lambda create-event-source-mapping \
  --function-name rag-indexer-lambda \
  --event-source-arn arn:aws:sqs:ap-south-1:068733247141:RagLambdaIndexing.fifo \
  --batch-size 1 \
  --region ap-south-1
```

### Step 4: Test Lambda Function

```bash
# Send test message to SQS
python3 << 'EOF'
import boto3, json, time
sqs = boto3.client('sqs', region_name='ap-south-1')
sqs.send_message(
    QueueUrl="https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo",
    MessageBody=json.dumps({"tenant_id": 12}),
    MessageGroupId="test",
    MessageDeduplicationId=f"test-{int(time.time())}"
)
print("✅ Test message sent")
EOF

# Check Lambda logs
aws logs tail /aws/lambda/rag-indexer-lambda --follow --region ap-south-1
```

**Expected Output**:
```
🔒 ISOLATED INDEXING FOR TENANT 12
📦 Using dedicated index: tenant-12-index
✅ ISOLATED INDEXING COMPLETE
```

## Part 3: API Server Deployment

### Step 1: Launch EC2 Instance

```bash
# Launch Ubuntu instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --region ap-south-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=RAG-API-Server}]'
```

### Step 2: Connect and Setup Server

```bash
# SSH to server
ssh -i your-key.pem ubuntu@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for docker group to take effect
exit
ssh -i your-key.pem ubuntu@your-server-ip
```

### Step 3: Clone Repository

```bash
# Clone your repository
git clone https://github.com/your-repo/ChatBotBE.git
cd ChatBotBE
```

### Step 4: Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET_NAME=rag-chat-uploads
S3_VECTORS_BUCKET_NAME=rag-vectordb-bucket
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
LLM_MODEL=meta.llama3-8b-instruct-v1:0
INDEXING_QUEUE_URL=https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo
EOF

# Set AWS credentials (use IAM role or credentials)
aws configure
```

### Step 5: Build and Start Docker Containers

```bash
# Build Docker image
docker-compose build

# Start containers
docker-compose up -d

# Check logs
docker-compose logs -f backend
```

**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Configure Nginx (Optional)

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx config
sudo cat > /etc/nginx/sites-available/rag-api << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 7: Setup SSL (Optional)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## Part 4: Testing Production Deployment

### Test 1: Upload Document

```bash
# Upload test document
aws s3 cp test-document.pdf s3://rag-chat-uploads/knowledge_base/12/

# Check SQS queue
aws sqs get-queue-attributes \
  --queue-url https://sqs.ap-south-1.amazonaws.com/068733247141/RagLambdaIndexing.fifo \
  --attribute-names ApproximateNumberOfMessages

# Check Lambda logs
aws logs tail /aws/lambda/rag-indexer-lambda --follow
```

### Test 2: Query API

```bash
# Test query
curl -X POST "http://your-server-ip:8000/chatbot/ask?tenant_id=12" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?"}'
```

**Expected Response**:
```json
{
  "answer": "Based on the leave policy, employees are entitled to...",
  "sources": ["s3://rag-chat-uploads/knowledge_base/12/policy.pdf"],
  "confidence": 0.95
}
```

### Test 3: Verify Tenant Isolation

```bash
# Test tenant 12
curl -X POST "http://your-server-ip:8000/chatbot/ask?tenant_id=12" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Test tenant 26
curl -X POST "http://your-server-ip:8000/chatbot/ask?tenant_id=26" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Verify logs show different indexes
docker logs fastapi-backend | grep "tenant-.*-index"
```

## Part 5: Monitoring & Maintenance

### CloudWatch Monitoring

```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name RAG-System \
  --dashboard-body file://dashboard-config.json
```

**dashboard-config.json**:
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Invocations", {"stat": "Sum"}],
          [".", "Errors", {"stat": "Sum"}],
          [".", "Duration", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "ap-south-1",
        "title": "Lambda Metrics"
      }
    }
  ]
}
```

### Log Monitoring

```bash
# Lambda logs
aws logs tail /aws/lambda/rag-indexer-lambda --follow

# API logs
ssh ubuntu@your-server "docker logs -f fastapi-backend"

# Nginx logs (if using)
ssh ubuntu@your-server "sudo tail -f /var/log/nginx/access.log"
```

### Backup Strategy

```bash
# Backup S3 data
aws s3 sync s3://rag-chat-uploads s3://rag-chat-uploads-backup

# Backup S3 Vectors (export metadata)
python3 << 'EOF'
import boto3, json
s3vectors = boto3.client('s3vectors', region_name='ap-south-1')

# List all indexes
indexes = []
for tenant_id in [12, 26, 11, 10]:
    index_name = f"tenant-{tenant_id}-index"
    indexes.append(index_name)

with open('vector-indexes-backup.json', 'w') as f:
    json.dump({"indexes": indexes}, f)
print("✅ Backup complete")
EOF
```

## Part 6: Scaling & Optimization

### Lambda Optimization

```bash
# Increase memory for faster processing
aws lambda update-function-configuration \
  --function-name rag-indexer-lambda \
  --memory-size 2048

# Increase timeout for large files
aws lambda update-function-configuration \
  --function-name rag-indexer-lambda \
  --timeout 600

# Enable provisioned concurrency (for consistent performance)
aws lambda put-provisioned-concurrency-config \
  --function-name rag-indexer-lambda \
  --provisioned-concurrent-executions 2
```

### API Server Scaling

```bash
# Increase Docker container resources
# Edit docker-compose.yml:
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

# Restart containers
docker-compose down
docker-compose up -d
```

### Database Optimization

```python
# Add caching for embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed_query(text: str):
    return embeddings.embed_query(text)
```

## Part 7: Security Hardening

### IAM Policies

```bash
# Principle of least privilege
# Create separate roles for:
# 1. Lambda execution
# 2. API server
# 3. Admin access

# Example: API server role (read-only)
cat > api-server-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3vectors:QueryVectors",
        "s3vectors:GetVectors",
        "bedrock:InvokeModel"
      ],
      "Resource": "*"
    }
  ]
}
EOF
```

### Network Security

```bash
# Configure security groups
# Allow only necessary ports:
# - 22 (SSH) from your IP only
# - 80/443 (HTTP/HTTPS) from anywhere
# - 8000 (API) from Nginx only

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 22 \
  --cidr your-ip/32

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0
```

### Secrets Management

```bash
# Use AWS Secrets Manager
aws secretsmanager create-secret \
  --name rag-system-secrets \
  --secret-string '{"api_key":"xxx","db_password":"yyy"}'

# Retrieve in application
import boto3
secrets = boto3.client('secretsmanager')
secret = secrets.get_secret_value(SecretId='rag-system-secrets')
```

## Troubleshooting

### Issue: Lambda timeout
**Solution**: Increase timeout and memory
```bash
aws lambda update-function-configuration \
  --function-name rag-indexer-lambda \
  --timeout 600 \
  --memory-size 2048
```

### Issue: API server not responding
**Solution**: Check Docker logs
```bash
docker logs fastapi-backend
docker restart fastapi-backend
```

### Issue: High costs
**Solution**: Optimize queries and caching
```python
# Reduce top_k
top_k = 8  # Instead of 12

# Add caching
@lru_cache(maxsize=100)
def cached_retrieve(query, tenant_id):
    return retrieve_s3_vectors(query, tenant_id)
```

## Cost Estimation

### Monthly Costs (Approximate)

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 100 GB | $2.30 |
| S3 Vectors | 1M queries | $10.00 |
| Lambda | 1M invocations | $0.20 |
| Bedrock | 1M tokens | $15.00 |
| EC2 (t3.medium) | 730 hours | $30.00 |
| **Total** | | **~$60/month** |

## Next Steps

- [1_INDEXING_WORKFLOW.md](1_INDEXING_WORKFLOW.md) - Understand indexing process
- [2_RETRIEVAL_WORKFLOW.md](2_RETRIEVAL_WORKFLOW.md) - Understand retrieval process
- [4_TENANT_ISOLATION.md](4_TENANT_ISOLATION.md) - Security architecture
