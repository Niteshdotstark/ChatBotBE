#!/bin/bash

echo "=========================================="
echo "Testing Lambda Function for Tenant 11"
echo "=========================================="
echo ""

# Run inside Docker container
docker exec -it fastapi-backend python test_lambda_tenant_11.py

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
