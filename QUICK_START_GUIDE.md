# 🚀 RAG Chatbot - Quick Start Guide

## 📋 Prerequisites
- Python 3.8+
- AWS credentials configured
- Required dependencies installed

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy and configure environment file
cp env.example env
# Edit env file with your AWS credentials and settings
```

### 3. Run Health Check
```bash
python system_health_check.py
```

### 4. Start the Application
```bash
python main.py
```

## 🧪 Test the System

### Basic API Test
```bash
curl -X POST "http://localhost:8000/chatbot/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens on Day 1?", "tenant_id": 26}'
```

### Webhook Test (1000 char limit)
```bash
curl -X POST "http://localhost:8000/webhook/26" \
  -H "Content-Type: application/json" \
  -d '{"question": "USB drive policy"}'
```

## 🎯 Demo Features

### Run Live Demo
```bash
python demo_advanced_features.py
```

### Test Queries
- **Semantic Understanding**: "What happens on Day 1?"
- **Person Filtering**: "Tell me about Mahi's projects"
- **USB Policy**: "Can I use USB drive in laptop?"
- **Conversation Context**: "Who is Nitesh?" → "How many years of experience he has?"

## 📊 Key Endpoints

### Main Endpoints
- `POST /chatbot/ask` - Detailed responses
- `POST /webhook/{tenant_id}` - Summary responses (≤1000 chars)
- `POST /webhook/telegram/{tenant_id}` - Telegram integration

### Admin Endpoints
- `POST /upload/{tenant_id}` - Upload documents
- `POST /reindex/{tenant_id}` - Trigger reindexing

## 🔧 Configuration

### Response Modes
- **detailed**: Full responses for API
- **summary**: 1000 char limit for webhooks
- **both**: Returns both versions

### Tenant Configuration
- Each tenant has isolated data
- Tenant-specific document processing
- Independent vector storage

## 📈 Monitoring

### Quality Metrics
- Response accuracy tracking
- Performance monitoring
- Cross-contamination detection

### Health Checks
- System component validation
- AWS service connectivity
- Database health monitoring

## 🎉 Advanced Features

### 1. Intelligent Query Processing
- Automatic query expansion
- Semantic variations generation
- Intent analysis and strategy selection

### 2. Person-Specific Filtering
- Zero cross-contamination
- Multi-layer validation
- Conversation context awareness

### 3. Multi-Tier Processing
- 4-tier fallback system
- Intelligent degradation
- Policy-aware responses

### 4. AWS Bedrock Integration
- Claude 3 Sonnet LLM
- Titan v2 embeddings
- ap-south-1 region optimization

## 🆘 Troubleshooting

### Common Issues
1. **AWS Credentials**: Ensure proper AWS configuration
2. **Dependencies**: Run `pip install -r requirements.txt`
3. **Region**: Verify ap-south-1 region access
4. **Permissions**: Check S3 and Bedrock permissions

### Support
- Check `FINAL_PROJECT_SUMMARY.md` for complete documentation
- Run `system_health_check.py` for diagnostics
- Review logs in `test_logs/` directory

## 🚀 Production Deployment

### Environment Setup
1. Configure production AWS credentials
2. Set up proper security groups
3. Configure load balancing
4. Set up monitoring and alerting

### Scaling Considerations
- Multi-tenant architecture supports unlimited tenants
- Vector database scales automatically
- Consider API rate limiting for production

**🎯 Your advanced RAG system is ready for enterprise deployment!**
