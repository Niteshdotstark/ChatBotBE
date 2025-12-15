import os
import json
import re
import glob
from collections import Counter
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.orm import Session # Required for the DB session
from datetime import date

# --- Configuration ---
HISTORY_DIR = "./uploads/conversation_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

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
    """Load conversation history for a user from a JSON file."""
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
            print(f"Error loading conversation history for tenant {tenant_id}, user {user_id}: {e}")
    return []

def save_conversation_history(tenant_id: int, user_id: str, history: list):
    """Save conversation history for a user to a JSON file."""
    history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f"{user_id}.json")
    try:
        # Convert LangChain messages to JSON-serializable format
        history_data = [
            {"type": "human", "human": msg.content} if isinstance(msg, HumanMessage) else {"type": "ai", "ai": msg.content}
            for msg in history
        ]
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving conversation history for tenant {tenant_id}, user {user_id}: {e}")

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
    Saves the results to a JSON file in the tenant's history directory.

    Args:
        tenant_id: The ID of the tenant to analyze.
        top_n: The number of top questions to report.
    """
    tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
    
    if not os.path.exists(tenant_history_dir):
        print(f"❌ History directory for tenant {tenant_id} not found at: {tenant_history_dir}")
        return

    all_user_questions = []
    
    # Iterate through all user history files in the tenant's directory
    for user_file in glob.glob(os.path.join(tenant_history_dir, "*.json")):
        try:
            user_id = os.path.basename(user_file).replace('.json', '')
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
            print(f"Error reading or processing history file {user_file}: {e}")

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

    # --- Save Results ---
    output_file = os.path.join(tenant_history_dir, f"top_questions_for_review_tenant_{tenant_id}_{top_n}.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Analysis Complete for Tenant {tenant_id}.")
        print(f"Found {len(all_user_questions)} total user questions.")
        print(f"Saved top {len(analysis_results)} questions to: {output_file}")
        
    except Exception as e:
        print(f"Error saving analysis file: {e}")