import os
import json
import re
import glob
from collections import Counter
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.orm import Session # Required for the DB session
from datetime import date
import boto3
from botocore.exceptions import ClientError

# --- Configuration ---
# For fallback/local usage only
HISTORY_DIR = "./uploads/conversation_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
S3_REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_CONVERSATION_PREFIX = "conversation_history"

# S3 Client
try:
    s3_client = boto3.client("s3", region_name=S3_REGION_NAME)
except Exception as e:
    print(f"⚠️ Warning: Failed to initialize S3 client: {e}. Falling back to local storage.")
    s3_client = None

def clean_query(question: str) -> str:
    """Normalize the user question for dictionary lookup."""
    if not question:
        return ""
    # Convert to lowercase, remove leading/trailing spaces
    query = question.lower().strip()
    # Remove all punctuation
    query = re.sub(r'[^\w\s]', '', query)
    # Replace multiple spaces with a single space
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def load_conversation_history(tenant_id: int, user_id: str) -> list:
    """Load conversation history for a user from S3 or local fallback."""
    s3_key = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/{user_id}.json"
    
    # Try S3 first
    if s3_client:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            history = json.loads(response['Body'].read().decode('utf-8'))
            # Convert JSON messages to LangChain message objects
            return [
                HumanMessage(content=msg["human"]) if msg["type"] == "human" else AIMessage(content=msg["ai"])
                for msg in history
            ]
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                print(f"Error loading conversation history from S3 for tenant {tenant_id}, user {user_id}: {e}")
        except Exception as e:
            print(f"Error loading conversation history from S3 for tenant {tenant_id}, user {user_id}: {e}")
    
    # Fallback to local filesystem
    history_file = os.path.join(HISTORY_DIR, str(tenant_id), f"{user_id}.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                # Convert JSON messages to LangChain message objects
                return [
                    HumanMessage(content=msg["human"]) if msg["type"] == "human" else AIMessage(content=msg["ai"])
                    for msg in history
                ]
        except Exception as e:
            print(f"Error loading conversation history from local disk for tenant {tenant_id}, user {user_id}: {e}")
    
    return []

def save_conversation_history(tenant_id: int, user_id: str, history: list):
    """Save conversation history for a user to S3 (or local fallback)."""
    # Convert LangChain messages to JSON-serializable format
    history_data = [
        {"type": "human", "human": msg.content} if isinstance(msg, HumanMessage) else {"type": "ai", "ai": msg.content}
        for msg in history
    ]
    json_content = json.dumps(history_data, ensure_ascii=False, indent=2)
    
    s3_key = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/{user_id}.json"
    
    # Try S3 first
    if s3_client:
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✅ Conversation history saved to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
            return
        except Exception as e:
            print(f"⚠️ Error saving conversation history to S3 for tenant {tenant_id}, user {user_id}: {e}")
    
    # Fallback to local filesystem
    try:
        history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"{user_id}.json")
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Conversation history saved locally: {history_file}")
    except Exception as e:
        print(f"Error saving conversation history locally for tenant {tenant_id}, user {user_id}: {e}")


def append_conversation_message(tenant_id: int, user_id: str, sender: str, text: str, max_entries: int = 500):
    """Append a single message to a user's conversation history (S3 or local fallback).

    Args:
        tenant_id: tenant id (int or str)
        user_id: user id (str)
        sender: 'human' or 'ai'
        text: message content
        max_entries: maximum number of message entries to keep (oldest dropped)
    """
    s3_key = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/{user_id}.json"
    history = []
    
    # Try to load from S3 first
    if s3_client:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            history = json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                print(f"⚠️ Error loading from S3: {e}")
        except Exception:
            pass
    
    # Fallback to local filesystem
    if not history:
        history_file = os.path.join(HISTORY_DIR, str(tenant_id), f"{user_id}.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

    # Append new entry
    entry = {"type": "human", "human": text} if sender == "human" else {"type": "ai", "ai": text}
    history.append(entry)

    # Keep only the most recent `max_entries` entries to avoid unbounded growth
    if len(history) > max_entries:
        history = history[-max_entries:]

    # Save to S3 and fallback to local
    json_content = json.dumps(history, ensure_ascii=False, indent=2)
    
    if s3_client:
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✅ Message appended to S3: {s3_key}")
            return
        except Exception as e:
            print(f"⚠️ Error appending to S3: {e}")
    
    # Fallback to local filesystem
    try:
        history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"{user_id}.json")
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error appending conversation history locally for tenant {tenant_id}, user {user_id}: {e}")


def get_recent_conversation_context(tenant_id: int, user_id: str, last_n_questions: int = 10) -> list:
    """Return a list of LangChain messages covering the most recent `last_n_questions` user questions
    and the surrounding AI replies (so continuity is preserved). Loads from S3 or local fallback.

    The result is in chronological order (oldest -> newest).
    """
    s3_key = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/{user_id}.json"
    raw = None
    
    # Try S3 first
    if s3_client:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            raw = json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                print(f"⚠️ Error loading from S3: {e}")
        except Exception:
            pass
    
    # Fallback to local filesystem
    if raw is None:
        history_file = os.path.join(HISTORY_DIR, str(tenant_id), f"{user_id}.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                return []
        else:
            return []

    if not raw:
        return []

    # Walk backwards until we've collected last_n_questions human messages
    collected = []
    human_count = 0
    # iterate reversed to pick most recent entries
    for entry in reversed(raw):
        collected.append(entry)
        if entry.get("type") == "human":
            human_count += 1
            if human_count >= last_n_questions:
                break

    if not collected:
        return []

    # Now reverse to chronological order and convert to LangChain message objects
    collected.reverse()
    messages = []
    for e in collected:
        if e.get("type") == "human":
            messages.append(HumanMessage(content=e.get("human", "")))
        else:
            messages.append(AIMessage(content=e.get("ai", "")))

    return messages

def analyze_all_tenants_daily(db: Session, TenantModel, top_n: int = 10):
    """
    Runs the conversation analysis for ALL tenants and saves the report.
    
    Args:
        db: SQLAlchemy Session dependency.
        TenantModel: The Tenant class model from models.py (needs to be passed from main.py).
        top_n: The number of top questions to report.
    """
    print(f"--- Starting Conversation Analysis on {date.today()} ---")
    
    # 1. Fetch all tenant IDs
    try:
        # Assuming TenantModel is the SQLAlchemy class and has an 'id' column
        tenants = db.query(TenantModel.id).all()
        tenant_ids = [t[0] for t in tenants]
        print(f"Found {len(tenant_ids)} tenants to analyze: {tenant_ids}")
    except Exception as e:
        print(f"❌ Error fetching tenant list: {e}")
        return

    # 2. Run analysis for each tenant
    for tenant_id in tenant_ids:
        try:
            # We call the existing analyze_user_questions function
            analyze_user_questions(tenant_id, top_n=top_n)
        except Exception as e:
            print(f"❌ Critical error running analysis for tenant {tenant_id}: {e}")
    
    print(f"--- Finished Conversation Analysis ---")
# --- Analysis Function ---

def analyze_user_questions(tenant_id: int, top_n: int = 10):
    """
    Analyzes conversation history for a tenant to find the top N most asked questions.
    Reads from S3 (or local fallback) and saves the results to S3.

    Args:
        tenant_id: The ID of the tenant to analyze.
        top_n: The number of top questions to report.
    """
    all_user_questions = []
    
    # Try to list files from S3 first
    user_ids_to_analyze = []
    
    if s3_client:
        try:
            prefix = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/"
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
            
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Extract user_id from key: conversation_history/{tenant_id}/{user_id}.json
                    if key.endswith('.json') and not key.endswith('_analysis.json'):
                        user_id = key.split('/')[-1].replace('.json', '')
                        user_ids_to_analyze.append(user_id)
            
            print(f"Found {len(user_ids_to_analyze)} users in S3 for tenant {tenant_id}")
        except Exception as e:
            print(f"⚠️ Error listing S3 files: {e}")
    
    # Fallback to local filesystem
    if not user_ids_to_analyze:
        tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
        if not os.path.exists(tenant_history_dir):
            print(f"❌ History directory for tenant {tenant_id} not found at: {tenant_history_dir}")
            return
        
        for user_file in glob.glob(os.path.join(tenant_history_dir, "*.json")):
            if not os.path.basename(user_file).startswith("top_questions_"):
                user_id = os.path.basename(user_file).replace('.json', '')
                user_ids_to_analyze.append(user_id)
    
    # Analyze conversations for each user
    for user_id in user_ids_to_analyze:
        try:
            history = load_conversation_history(tenant_id, user_id) 
            
            for message in history:
                if isinstance(message, HumanMessage):
                    cleaned_q = clean_query(message.content)
                    if cleaned_q:
                        all_user_questions.append({
                            "original": message.content,
                            "cleaned": cleaned_q
                        })
        except Exception as e:
            print(f"Error processing history for user {user_id}: {e}")

    if not all_user_questions:
        print(f"No conversation history found for tenant {tenant_id}.")
        return

    cleaned_counts = Counter(item['cleaned'] for item in all_user_questions)
    top_cleaned_questions = cleaned_counts.most_common(top_n)

    analysis_results = []
    for cleaned_q, count in top_cleaned_questions:
        original_example = next((item['original'] for item in all_user_questions if item['cleaned'] == cleaned_q), cleaned_q)
        
        analysis_results.append({
            "count": count,
            "cleaned_question": cleaned_q,
            "example_original_question": original_example,
        })

    # --- Save Results to S3 ---
    json_content = json.dumps(analysis_results, ensure_ascii=False, indent=2)
    s3_key = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/top_questions_for_review_tenant_{tenant_id}_{top_n}.json"
    
    if s3_client:
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"\n✅ Analysis Complete for Tenant {tenant_id}.")
            print(f"Found {len(all_user_questions)} total user questions.")
            print(f"Saved top {len(analysis_results)} questions to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
            return
        except Exception as e:
            print(f"⚠️ Error saving to S3: {e}")
    
    # Fallback to local filesystem
    try:
        tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
        os.makedirs(tenant_history_dir, exist_ok=True)
        output_file = os.path.join(tenant_history_dir, f"top_questions_for_review_tenant_{tenant_id}_{top_n}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Analysis Complete for Tenant {tenant_id}.")
        print(f"Found {len(all_user_questions)} total user questions.")
        print(f"Saved top {len(analysis_results)} questions locally to: {output_file}")
        
    except Exception as e:
        print(f"Error saving analysis file locally: {e}")