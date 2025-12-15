from dotenv import load_dotenv
from .rag_utils import index_tenant_files, answer_question_modern
import os

# Load environment variables
load_dotenv()

# Test indexing files for a tenant
tenant_id = 10  # Replace with your actual tenant ID
index_tenant_files(tenant_id)
user_id = 1

# Test answering a question
question = "Are you provide BCA course as well in jecrc?"
answer = answer_question_modern(question, tenant_id, user_id)
print(f"Answer: {answer}")