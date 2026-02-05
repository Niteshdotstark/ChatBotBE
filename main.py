from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, Request, Response, BackgroundTasks
from urllib.parse import unquote
import httpx
import boto3
from botocore.exceptions import ClientError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, engine, Base
from models import User, Tenant, KnowledgeBaseFile, TenantValues
from rag_model.rag_utils import index_tenant_files, s3_client, S3_BUCKET_NAME, S3_PREFIX_KNOWLEDGE, s3_append_url, index_tenant_files, s3_list_tenant_files, trigger_reindexing
from rag_model.config_manager import get_config_manager, get_tenant_config, update_tenant_config
from rag_model.config_models import RAGSystemConfig, DEFAULT_CONFIGS
from rag_model.config_schemas import (
    RAGConfigSchema, ConfigUpdateRequest, ConfigResetRequest, ConfigResponse,
    ConfigValidationResponse, TenantConfigListResponse, ConfigPresetListResponse
)
from schemas import (
    UserCreate, UserResponse, TenantCreate, TenantBase,TenantUpdate, ActivityLogResponse,
    KnowledgeBaseFileResponse, DatabaseCreate, DatabaseResponse, TokenResponse, 
    ChatRequest, ChatResponse, DualChatResponse, TenantValuesUpdate, TopQuestionAnalysis, 
    ConversationCountResponse, TenantResponse
)
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from rag_model.rag_utils import answer_question_modern
from typing import Optional
import asyncio
import re
from uuid import UUID
import glob
import os, uuid, boto3, shutil
from rag_model.chat_data_utils import HISTORY_DIR, append_conversation_message, get_recent_conversation_context
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
SECRET_KEY = "your_secret_key"  # Replace with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
VERIFY_TOKEN = "chatbottoken@3420"
FACEBOOK_ACCESS_TOKEN = "EAAKgyth24vEBPCOP16Fw4cnnGW0t9N6qoeSCtp5VlWMzSXlnsZCEUM5YWtFzQqCq2BZChkF3FKD6tszoybJ21KbpDecvga0Xr0WGlXMChSICVB9KDfHFmnT0rrUVs3DkOJlKtk5OZCq55zkls1FfSpJ0vRnnHGVAln5Y1bRqnNX3u5ZCqISAeil3X4Yc6N2XnmuJAwZDZD"  # Replace with your actual access token
# TENANT_ID = 1
INSTAGRAM_ACCESS_TOKEN = "IGAAbUoqM5z9pBZAFEyMVhubGtfNWhYUWpTMXVnS2MxNWJOeUw4amVTUlpacEs2SVJsM1p5UXRfekJpU09QNGtyQ3dRbG40UENWV3BhMW1wRjU3UldWY1U4V2JONXpiZA00yMUsxSGEyZA1V0OUxnZAkdDOVlsbGhRV21wZAWszc2dfNAZDZD"

TENANT_VALUES_DB = {
    "tenant_id": 1,
}

TENANT_TELEGRAM_BOTS = {
    "10": "8"
}

 # UPLOAD_DIR = "./uploads/knowledge_base"
UPLOAD_DIR = S3_PREFIX_KNOWLEDGE
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_tenant_values(db: Session = Depends(get_db)):
    tenant_values = db.query(TenantValues).first()
    if not tenant_values:
        raise HTTPException(status_code=500, detail="Tenant values not configured.")
    return tenant_values

@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if email already registered
    db_user_email = db.query(User).filter(User.email == user.email).first()
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    db_user_username = db.query(User).filter(User.username == user.username).first()
    if db_user_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_password = pwd_context.hash(user.password)
    new_user = User(
        email=user.email,
        hashed_password=hashed_password,
        username=user.username,
        phone_number=user.phone_number,
        address=user.address,
        is_active=user.is_active
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login/", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    print(f"DEBUG: Login attempt for username: {form_data.username}")
    email_to_search = form_data.username.lower().strip()
    user = db.query(User).filter(User.email == email_to_search).first() # form_data.username is the email
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    try:
        # This is where the crash was happening
        if not pwd_context.verify(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
    except Exception as e:
        # This catches the UnknownHashError and prevents a 500 error
        print(f"CRITICAL: Password verification failed for {user.email}")
        print(f"REASON: {str(e)}")
        print(f"DATABASE HASH WAS: {user.hashed_password}")
        raise HTTPException(
            status_code=401, 
            detail="Your account security format is outdated. Please register again or reset your password."
        )

    tenant = db.query(Tenant).filter(Tenant.creator_id == user.id).first()
    tenant_id = tenant.id if tenant else None


    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        phone_number=user.phone_number,
        address=user.address,
        is_active=user.is_active,
        tenant_id=tenant_id
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.get("/users/me/recent_activity/", response_model=list[ActivityLogResponse])
async def get_recent_activity(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Gathers a sorted list of the most recent activities for the user,
    including new tenants, new knowledge sources, and new chat conversations.
    """
    all_activities = []
    
    # 1. Get all tenants for the user
    tenants = db.query(Tenant).filter(Tenant.creator_id == current_user.id).all()
    tenant_map = {tenant.id: tenant.name for tenant in tenants}

    # 2. Add Tenant Creation events
    for tenant in tenants:
        all_activities.append(
            ActivityLogResponse(
                id=f"tenant-{tenant.id}",
                title="Organization created",
                subtitle=tenant.name,
                created_at=tenant.created_at,
                type="warning" # Matches your mock data for 'Organization created'
            )
        )

    # 3. Add Knowledge Source events
    kb_items = db.query(KnowledgeBaseFile).filter(KnowledgeBaseFile.tenant_id.in_(tenant_map.keys())).all()
    for item in kb_items:
        subtitle = item.filename or item.url or "Untitled Source"
        # Truncate long URLs/filenames
        if len(subtitle) > 50:
            subtitle = subtitle[:47] + "..."
            
        all_activities.append(
            ActivityLogResponse(
                id=f"kb-{item.id}",
                title="Knowledge source added",
                subtitle=subtitle,
                created_at=item.created_at,
                type="success" # Matches 'Knowledge source synced'
            )
        )

    # 4. Add Chat History events (from files)
    for tenant_id, tenant_name in tenant_map.items():
        tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant_id))
        
        if not os.path.isdir(tenant_history_dir):
            continue
            
        try:
            # Find all .json files
            all_json_files = glob.glob(os.path.join(tenant_history_dir, "*.json"))
            
            # Filter out the analysis reports
            conversation_files = [
                f for f in all_json_files 
                if not os.path.basename(f).startswith("top_questions_for_review_")
            ]
            
            for file_path in conversation_files:
                try:
                    # Get the file's modification time as the event time
                    mtime = os.path.getmtime(file_path)
                    all_activities.append(
                        ActivityLogResponse(
                            id=f"chat-{tenant_id}-{os.path.basename(file_path)}",
                            title="New conversation started",
                            subtitle=tenant_name,
                            created_at=datetime.utcfromtimestamp(mtime),
                            type="info" # Matches 'New conversation started'
                        )
                    )
                except Exception:
                    continue # Skip file if it's unreadable, etc.
                    
        except Exception:
            continue # Skip tenant if glob fails

    # 5. Sort all collected activities by date, descending
    all_activities.sort(key=lambda x: x.created_at, reverse=True)

    # 6. Return the 10 most recent
    return all_activities[:10]

@app.post("/tenants/", response_model=TenantBase, status_code=status.HTTP_201_CREATED)
async def create_tenant(tenant: TenantCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Check if the user already has a tenant
    # if current_user.created_tenants:
    #     raise HTTPException(status_code=400, detail="User can only create one tenant.")
    
    existing_tenant = db.query(Tenant).filter(Tenant.name == tenant.name).first()
    if existing_tenant:
        raise HTTPException(status_code=400, detail="Tenant name already exists")
        
    new_tenant = Tenant(
        name=tenant.name,
        fb_url=tenant.fb_url,
        insta_url=tenant.insta_url,
        fb_verify_token=tenant.fb_verify_token,
        fb_access_token=tenant.fb_access_token,
        insta_access_token=tenant.insta_access_token,
        telegram_bot_token=tenant.telegram_bot_token,
        telegram_chat_id=tenant.telegram_chat_id,
        creator_id=current_user.id
    )
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)
    return new_tenant


@app.put("/tenants/{tenant_id}/", response_model=TenantBase)
async def update_tenant(
      tenant_id: int,
      tenant_update: TenantUpdate,
      db: Session = Depends(get_db),
      current_user: User = Depends(get_current_user)
  ):
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this tenant")
        
    update_data = tenant_update.dict(exclude_unset=True)
    
    # Check for duplicate name if name is being updated
    if 'name' in update_data and update_data['name'] != tenant.name:
        existing_tenant = db.query(Tenant).filter(Tenant.name == update_data['name'], Tenant.id != tenant_id).first()
        if existing_tenant:
            raise HTTPException(status_code=400, detail="Tenant name already exists")
    
    for key, value in update_data.items():
        setattr(tenant, key, value)
        
    db.commit()
    db.refresh(tenant)
    return tenant

@app.get("/tenants/{tenant_id}/analysis_report_data/", response_model=list[TopQuestionAnalysis])
async def get_latest_analysis_report_data(
    tenant_id: int,
    db: Session = Depends(get_db),
):
    """
    Retrieves the data from the latest Top Questions Analysis Report 
    for a given tenant as a structured JSON list (PUBLIC ENDPOINT).
    Reads from S3 first, falls back to local filesystem.
    """
    # S3 Configuration
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
    S3_REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
    S3_CONVERSATION_PREFIX = "conversation_history"
    
    try:
        s3_client = boto3.client("s3", region_name=S3_REGION_NAME)
        
        # Try to find the latest report in S3
        try:
            prefix = f"{S3_CONVERSATION_PREFIX}/{tenant_id}/top_questions_for_review_"
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)
            
            report_keys = []
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        report_keys.append(obj['Key'])
            
            if report_keys:
                # Get the latest one by modified time
                latest_key = max(report_keys, key=lambda k: s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=k)['LastModified'])
                
                response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=latest_key)
                report_data = json.loads(response['Body'].read().decode('utf-8'))
                print(f"Loaded report data from S3: {latest_key}")
                return report_data
        except ClientError:
            pass  # Fall through to local filesystem
        except Exception as e:
            print(f"⚠️ Error reading from S3: {e}")
    
    except Exception:
        pass
    
    # Fallback to local filesystem
    from chat_data_utils import HISTORY_DIR 
    tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant_id))

    # Find the latest report file
    file_pattern = os.path.join(tenant_history_dir, f"top_questions_for_review_tenant_{tenant_id}_*_*.json")
    all_reports = glob.glob(file_pattern)

    if not all_reports:
        raise HTTPException(status_code=404, detail="No analysis report data found for this tenant.")

    # Find the latest file (based on last modified time)
    latest_report_path = max(all_reports, key=os.path.getmtime)
    
    # Read the file content
    try:
        with open(latest_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            print(f"Loaded report data from local filesystem: {latest_report_path}")
            
            # Return the structured data (FastAPI automatically validates against TopQuestionAnalysis list)
            return report_data
            
    except Exception as e:
        print(f"Error loading or parsing report file {latest_report_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load analysis report data.")

@app.post("/rag/ask/{tenant_id}")
async def ask_rag_with_path_tenant_id(
    tenant_id: int, # <--- Taken directly from the URL path
    request: ChatRequest, # <--- The body includes message and optional response_mode
    db: Session = Depends(get_db)
):
    """
    Public-facing RAG endpoint - returns SUMMARY responses by default (for social media platforms)
    
    Response modes:
    - "detailed": Full comprehensive response (for admin/direct API calls when explicitly requested)
    - "summary": Concise summary with 1000 character limit (default, for social media platforms)
    - "both": Returns both detailed and summary responses
    """
    # Optional: Basic check to ensure the tenant exists (good practice)
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant ID {tenant_id} not found.")

    try:
        user_key = "api_anonymous"
        append_conversation_message(tenant_id, user_key, 'human', request.message)
        context_msgs = get_recent_conversation_context(tenant_id, user_key, last_n_questions=10)

        # Default to summary mode for social media platforms, allow override via request
        response_mode = getattr(request, 'response_mode', 'summary')
        
        response_data = answer_question_modern(
            request.message, 
            tenant_id, 
            user_key, 
            context_messages=context_msgs,
            response_mode=response_mode
        )

        # Handle different response types
        if response_data.get("response_type") == "both":
            # Save detailed response to history
            append_conversation_message(tenant_id, user_key, 'ai', response_data.get("detailed_answer", ""))
            
            return DualChatResponse(
                detailed_response=response_data.get("detailed_answer"),
                summary_response=response_data.get("summary_answer"),
                sources=[],  # Remove sources from response
                response_type="both",
                summary_metadata=response_data.get("summary_metadata")
            )
        else:
            # Single response (detailed or summary)
            answer_text = response_data.get("answer", "No answer found.")
            
            # Save AI reply to history
            append_conversation_message(tenant_id, user_key, 'ai', answer_text)

            return ChatResponse(
                response=answer_text,
                sources=[],  # Remove sources from response
                response_type=response_data.get("response_type", "summary")
            )
            
    except Exception as e:
        # Handle potential errors from the RAG model
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")


@app.get("/users/me/conversation_count/", response_model=ConversationCountResponse)
async def get_total_conversation_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Counts all conversation files for all tenants owned by the current user.
    It excludes the 'top_questions_for_review' analysis reports.
    """
    tenants = db.query(Tenant).filter(Tenant.creator_id == current_user.id).all()
    total_count = 0
    
    for tenant in tenants:
        tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant.id))
        
        if not os.path.isdir(tenant_history_dir):
            continue
            
        try:
            # Get all JSON files (assuming conversations are .json)
            all_json_files = glob.glob(os.path.join(tenant_history_dir, "*.json"))
            
            # Filter out the analysis reports
            conversation_files = [
                f for f in all_json_files 
                if not os.path.basename(f).startswith("top_questions_for_review_")
            ]
            print(f" -> Filtered count (non-analysis files): {len(all_json_files)}")
            total_count += len(all_json_files)
        except Exception as e:
            print(f"Error counting files for tenant {tenant.id}: {e}")
            # Continue to next tenant even if one fails
            
    return {"count": total_count}


@app.get("/users/me/knowledge_item_count/", response_model=ConversationCountResponse)
async def get_total_knowledge_item_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Counts all knowledge base items for all tenants owned by the current user.
    """
    # This query joins Tenant and KnowledgeBaseFile to efficiently count
    # only the items belonging to the current user's tenants.
    count = db.query(KnowledgeBaseFile).join(Tenant).filter(
        Tenant.creator_id == current_user.id
    ).count()
    
    return {"count": count}


@app.post("/admin/run_analysis/", status_code=status.HTTP_202_ACCEPTED)
async def run_analysis_manually(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Manually triggers the conversation analysis job for all tenants in the background.
    """
    # ⚠️ WARNING: Add proper admin authentication here before deployment!
    
    from rag_model.chat_data_utils import analyze_all_tenants_daily
    from models import Tenant
    
    # We use BackgroundTasks so the API call returns immediately
    background_tasks.add_task(analyze_all_tenants_daily, db, Tenant, top_n=25)
    
    return {"message": "Conversation analysis job started in the background."}
# @app.put("/tenants/{tenant_id}/", response_model=TenantBase)
# async def update_tenant(
#       tenant_id: int,
#       tenant_update: TenantUpdate,
#       db: Session = Depends(get_db),
#       current_user: User = Depends(get_current_user)
#   ):
#       print(f"Updating tenant ID: {tenant_id}")
#       tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
#       if not tenant:
#           print("Tenant not found")
#           raise HTTPException(status_code=404, detail="Tenant not found")
#       if tenant.creator_id != current_user.id:
#           print("User not authorized to update tenant")
#           raise HTTPException(status_code=403, detail="Not authorized to update this tenant")
#       if tenant_update.name:
#           existing_tenant = db.query(Tenant).filter(Tenant.name == tenant_update.name, Tenant.id != tenant_id).first()
#           if existing_tenant:
#               print("Tenant name already exists")
#               raise HTTPException(status_code=400, detail="Tenant name already exists")
#           tenant.name = tenant_update.name
#       if tenant_update.fb_url is not None:
#           tenant.fb_url = tenant_update.fb_url
#       if tenant_update.insta_url is not None:
#           tenant.insta_url = tenant_update.insta_url
#       db.commit()
#       db.refresh(tenant)
#       print(f"Tenant updated: {tenant.name}")
#       return tenant

@app.get("/tenants/", response_model=list[TenantResponse])  # <-- UPDATED response_model
async def read_tenants(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    tenants = db.query(Tenant).filter(Tenant.creator_id == current_user.id).all()
    
    response_data = []
    
    for tenant in tenants:
        # 1. Get knowledge item count for THIS tenant
        knowledge_count = db.query(KnowledgeBaseFile).filter(
            KnowledgeBaseFile.tenant_id == tenant.id
        ).count()
        
        # 2. Get conversation count for THIS tenant
        # (This is the same logic from your global count, but tenant-specific)
        convo_count = 0
        tenant_history_dir = os.path.join(HISTORY_DIR, str(tenant.id))
        
        if os.path.isdir(tenant_history_dir):
            try:
                # Find all .json files in the tenant's history directory
                all_json_files = glob.glob(os.path.join(tenant_history_dir, "*.json"))
                
                # Filter out the analysis reports
                conversation_files = [
                    f for f in all_json_files 
                    if not os.path.basename(f).startswith("top_questions_for_review_")
                ]
                convo_count = len(conversation_files)
            except Exception as e:
                print(f"Error counting files for tenant {tenant.id}: {e}")
                convo_count = 0 # Default to 0 on error
        
        # 3. Build the new response object
        tenant_data = TenantResponse(
            id=tenant.id,
            name=tenant.name,
            created_at=tenant.created_at,
            fb_url=tenant.fb_url,
            insta_url=tenant.insta_url,
            fb_verify_token=tenant.fb_verify_token,
            fb_access_token=tenant.fb_access_token,
            insta_access_token=tenant.insta_access_token,
            telegram_bot_token=tenant.telegram_bot_token,
            telegram_chat_id=tenant.telegram_chat_id,
            knowledge_item_count=knowledge_count,
            conversation_count=convo_count
        )
        response_data.append(tenant_data)
        
    return response_data

@app.post("/tenants/{tenant_id}/knowledge_base_items/", response_model=KnowledgeBaseFileResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base_item(
    tenant_id: int,
    category: str = Form(...),
    file: UploadFile = File(None),
    url: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # --- Authorization Check (Using Mock/Placeholder) ---
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to add items to this tenant's knowledge base")

    if category == "url":
        if not url:
            raise HTTPException(status_code=400, detail="URL must be provided for URL category")
        
        # 1. Save URL to S3 urls.txt
        try:
            s3_append_url(tenant_id, url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save URL for indexing: {e}")
            
        # 2. Create DB record (Placeholder)
        kb_file = KnowledgeBaseFile(
            filename=None, stored_filename=None, file_path=None, file_type="url", 
            category="url", url=url, tenant_id=tenant_id, uploaded_by=current_user.id
        )

    elif category in ["file", "database"]:
        if not file:
            raise HTTPException(status_code=400, detail="File must be provided for File or Database category")
        
        # Generate S3 Key
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        s3_key = f"{S3_PREFIX_KNOWLEDGE}/{tenant_id}/{unique_filename}"
        
        # 1. CRITICAL S3 UPLOAD FIX: Stream upload_fileobj
        try:
            # Ensure the file pointer is at the beginning
            file.file.seek(0)
            s3_client.upload_fileobj(
                Fileobj=file.file,
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
            )
            print(f"✅ S3 Upload successful: s3://{S3_BUCKET_NAME}/{s3_key}")
        except ClientError as e:
            print(f"❌ S3 Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"S3 Upload failed: {e}")
        finally:
            # Always close the underlying file handle
            await file.close()

        # 2. Create DB record (Placeholder)
        kb_file = KnowledgeBaseFile(
            filename=file.filename,
            stored_filename=unique_filename,
            file_path=s3_key, # Store S3 key/path
            file_type=file.content_type,
            category=category,
            url=None,
            tenant_id=tenant_id,
            uploaded_by=current_user.id
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid category")

    # DB Operations (uncomment these in your actual code)
    db.add(kb_file)
    db.commit()
    db.refresh(kb_file)

    # NO INDEXING CALL HERE - IT'S DECOUPLED!
    trigger_reindexing(tenant_id)
    return kb_file

@app.post("/tenants/{tenant_id}/knowledge_base_items/add_url/", status_code=status.HTTP_201_CREATED)
async def add_url_to_file_and_db(
    tenant_id: int,
    url: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # --- Authorization Check (Using Mock/Placeholder) ---
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to add items to this tenant's knowledge base")

    # 1. Update the S3 urls.txt file (using helper)
    try:
        s3_append_url(tenant_id, url)
        print(f"✅ URL appended to S3 urls.txt for tenant {tenant_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save URL for indexing: {e}")

    # 2. Save to database (Placeholder)
    kb_file = KnowledgeBaseFile(
        filename=None, stored_filename=None, file_path=None, file_type="url",
        category="url", url=url, tenant_id=tenant_id, uploaded_by=current_user.id
    )
    db.add(kb_file)
    db.commit()
    db.refresh(kb_file)

    trigger_reindexing(tenant_id)
    # NO INDEXING CALL HERE - IT'S DECOUPLED!

    return kb_file

@app.get("/tenants/{tenant_id}/knowledge_base_items/", response_model=list[KnowledgeBaseFileResponse])
async def list_knowledge_base_items(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # This endpoint remains largely logical/DB-focused.
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first() if db else type('Tenant', (object,), {'id': tenant_id, 'creator_id': current_user.id})
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # In a real S3 setup, this would query your DB for KB items.
    items = db.query(KnowledgeBaseFile).filter(KnowledgeBaseFile.tenant_id == tenant_id).all() if db else []
    return items
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    items = db.query(KnowledgeBaseFile).filter(KnowledgeBaseFile.tenant_id == tenant_id).all()
    return items


@app.put("/tenants/{tenant_id}/knowledge_base_items/{item_id}", response_model=KnowledgeBaseFileResponse)
async def update_knowledge_base_item(
    tenant_id: int,
    item_id: str,
    new_url: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)   
):
    """
    Updates an existing knowledge base item.
    Currently, this only supports changing the URL of an item with category 'url'.
    """
    # 1. Check if the tenant belongs to the current user
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    # 2. Find the specific item
    item = db.query(KnowledgeBaseFile).filter(
        KnowledgeBaseFile.id == item_id, 
        KnowledgeBaseFile.tenant_id == tenant_id
    ).first()

    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base item not found")

    # 3. Check if it's a 'url' item (we don't support file replacement via this endpoint)
    if item.category != "url":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="This endpoint only supports updating items of category 'url'. To replace a file, please delete and re-upload."
        )

    item.url = new_url
    db.commit()

    try:
        tenant_dir = os.path.join(UPLOAD_DIR, str(tenant.id))
        os.makedirs(tenant_dir, exist_ok=True)
        file_location = os.path.join(tenant_dir, "urls.txt")
        
        all_urls = db.query(KnowledgeBaseFile.url).filter(
            KnowledgeBaseFile.tenant_id == tenant_id, 
            KnowledgeBaseFile.category == "url", 
            KnowledgeBaseFile.url.isnot(None)
        ).all()
        
        with open(file_location, "w") as url_file:
            for (url_val,) in all_urls:
                url_file.write(url_val + "\n")
    except IOError as e:
        print(f"Failed to rebuild url.txt: {e}")

    trigger_reindexing(tenant_id)
    
    db.refresh(item)
    return item


@app.delete("/tenants/{tenant_id}/knowledge_base_items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base_item(
    tenant_id: int,
    item_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Deletes a knowledge base item from the database and removes its associated file from disk.
    """
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    item = db.query(KnowledgeBaseFile).filter(
        KnowledgeBaseFile.id == item_id, 
        KnowledgeBaseFile.tenant_id == tenant_id
    ).first()

    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base item not found")

    item_was_url = (item.category == "url")
    item_file_path = item.file_path

    # 3. Delete the item from the database
    db.delete(item)
    db.commit()

    # 4. If it was a file, delete it from the physical disk
    if item_file_path and os.path.exists(item_file_path):
        try:
            os.remove(item_file_path)
        except OSError as e:
            # Log the error but don't fail the request, as the DB entry is gone
            print(f"Error deleting file {item_file_path}: {e}")

    # 5. If a URL was deleted, rebuild 'urls.txt' to keep it in sync
    if item_was_url:
        try:
            tenant_dir = os.path.join(UPLOAD_DIR, str(tenant.id))
            os.makedirs(tenant_dir, exist_ok=True)
            file_location = os.path.join(tenant_dir, "urls.txt")
            
            all_urls = db.query(KnowledgeBaseFile.url).filter(
                KnowledgeBaseFile.tenant_id == tenant_id, 
                KnowledgeBaseFile.category == "url", 
                KnowledgeBaseFile.url.isnot(None)
            ).all()
            
            with open(file_location, "w") as url_file:
                for (url_val,) in all_urls:
                    url_file.write(url_val + "\n")
        except IOError as e:
            print(f"Failed to rebuild url.txt after delete: {e}")

    # 6. Trigger a background task to re-index the tenant's files
    trigger_reindexing(tenant_id)

    # Return 204 No Content, which is standard for a successful DELETE
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.put("/tenant_values/", status_code=status.HTTP_200_OK)
async def update_tenant_values(tenant_values: TenantValuesUpdate, db: Session = Depends(get_db)):
    """
    Allows an admin to update the single row of tenant-related values.
    Since there is only one row, the update is global.
    """
    db_values = db.query(TenantValues).first()
    if not db_values:
        db_values = TenantValues(**tenant_values.dict())
        db.add(db_values)
    else:
        db_values.tenant_id = tenant_values.tenant_id
    db.commit()
    db.refresh(db_values)
    return db_values

@app.post("/tenants/{tenant_id}/chat", response_model=ChatResponse)
async def chat_with_tenant_kb(
    tenant_id: int,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Admin/authenticated user chat endpoint - always returns detailed responses
    """
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this tenant's knowledge base")

    # Call your RAG model with the provided message and tenant_id
    try:
        user_key = str(current_user.id)
        append_conversation_message(tenant_id, user_key, 'human', chat_request.message)
        context_msgs = get_recent_conversation_context(tenant_id, user_key, last_n_questions=10)

        # Admin users always get detailed responses
        response_data = answer_question_modern(
            chat_request.message, 
            tenant_id, 
            user_key, 
            context_messages=context_msgs,
            response_mode="detailed"
        )
        answer_text = response_data.get("answer", "No answer found.")

        append_conversation_message(tenant_id, user_key, 'ai', answer_text)

        # Ensure the response_data has 'answer' and 'sources' keys
        return ChatResponse(
            response=answer_text,
            sources=[],  # Remove sources from response
            response_type="detailed"
        )
    except Exception as e:
        # Handle potential errors from the RAG model
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while processing the request: {str(e)}")

@app.post("/chatbot/ask", response_model=ChatResponse)
async def ask_chatbot(
    tenant_id: int,
    request: ChatRequest,
    # api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """
    Direct API chatbot endpoint - returns DETAILED responses without limitations (for admin/direct API calls)
    """
    user_key = "api_anonymous"
    append_conversation_message(tenant_id, user_key, 'human', request.message)
    context_msgs = get_recent_conversation_context(tenant_id, user_key, last_n_questions=10)

    # Direct API calls always get detailed responses without limitations
    response_data = answer_question_modern(
        request.message, 
        tenant_id, 
        user_key, 
        context_messages=context_msgs,
        response_mode="detailed"
    )
    
    print(f"DEBUG: Type of response_data: {type(response_data)}")
    print(f"DEBUG: Content of response_data: {response_data}")
    
    answer_value = response_data.get("answer", "No answer found.")

    # Check if the answer_value is a dictionary (the nested case)
    if isinstance(answer_value, dict):
        # It's the fixed response: extract the inner 'answer' string
        final_answer_string = answer_value.get("answer", "No answer found.")
    else:
        # It's the RAG response: the answer_value is already the string
        final_answer_string = str(answer_value) # Ensure it's a string

    append_conversation_message(tenant_id, user_key, 'ai', final_answer_string)

    return ChatResponse(
        response=final_answer_string,
        sources=[],  # Remove sources from response
        response_type="detailed"
    )

@app.get("/webhook/{tenant_id}")
async def verify_webhook(tenant_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Verifies the webhook for Facebook/Instagram using the tenant_id provided in the URL path.
    """
    query_params = dict(request.query_params)
    mode = query_params.get("hub.mode")
    token = unquote(query_params.get("hub.verify_token", ""))
    challenge = query_params.get("hub.challenge")
    
    print(f"Received webhook verification request for tenant_id={tenant_id}")
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            print(f"❌ Verification failed: Tenant {tenant_id} not found")
            raise HTTPException(status_code=404, detail="Tenant not found")
            
        VERIFY_TOKEN = tenant.fb_verify_token

        if not VERIFY_TOKEN:
            print(f"❌ Verification failed: Verify token not set for Tenant {tenant_id}")
            return Response(content="Verification token not configured", status_code=403)
            
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print(f"✅ Webhook verified for challenge={challenge}, tenant={tenant_id}")
            return Response(
                content=challenge.encode("utf-8"),
                media_type="text/plain; charset=utf-8",
                status_code=200
            )
        else:
            print(f"❌ Verification failed: mode={mode}, token={token}, expected={VERIFY_TOKEN}")
            return Response(content="Invalid verification", status_code=403)
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return Response(content="Internal error during tenant lookup", status_code=500)


# --- MODIFIED: Uses {tenant_id} as a path parameter ---
@app.post("/webhook/{tenant_id}")
async def handle_webhook(tenant_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Handles incoming messages for Facebook and Instagram using the tenant_id provided in the URL path.
    """
    try:
        data = await request.json()
        print(f"Received payload: {data}")

        TENANT_ID = tenant_id
        tenant = db.query(Tenant).filter(Tenant.id == TENANT_ID).first()
        
        if not tenant:
            print(f"⚠️ Tenant {TENANT_ID} not found for webhook processing.")
            return {"status": "error", "message": f"Tenant {TENANT_ID} not configured."}

        # Dynamically fetch tokens from the tenant object
        FACEBOOK_ACCESS_TOKEN = tenant.fb_access_token
        INSTAGRAM_ACCESS_TOKEN = tenant.insta_access_token
        
        if not FACEBOOK_ACCESS_TOKEN and not INSTAGRAM_ACCESS_TOKEN:
            print(f"⚠️ Access tokens for tenant {TENANT_ID} are missing.")
            return {"status": "error", "message": "Access tokens missing."}

        for entry in data.get("entry", []):
            is_instagram = data.get("object") == "instagram"
            access_token = INSTAGRAM_ACCESS_TOKEN if is_instagram else FACEBOOK_ACCESS_TOKEN
            platform = "Instagram" if is_instagram else "Facebook"

            for messaging in entry.get("messaging", []):
                if messaging.get("message") and messaging.get("message").get("text"):
                    if messaging.get("message").get("is_self", False) or messaging.get("message").get("is_echo", False):
                        print(f"Skipping echo message from {platform} {messaging.get('sender').get('id')}")
                        continue
                        
                    sender_id = messaging.get("sender").get("id")
                    text = messaging.get("message").get("text")
                    print(f"📩 {platform} {sender_id}: {text}")
                    
                    if not access_token:
                        print(f"⚠️ Skipping message: {platform} access token is missing for tenant {TENANT_ID}.")
                        continue

                    try:
                        # Save incoming human message
                        append_conversation_message(TENANT_ID, str(sender_id), 'human', text)

                        # Load recent context (last 10 human questions + surrounding AI replies)
                        context_msgs = get_recent_conversation_context(TENANT_ID, str(sender_id), last_n_questions=10)

                        # Pass TENANT_ID and context to RAG model with SUMMARY mode for social media
                        response_data = answer_question_modern(
                            text, 
                            TENANT_ID, 
                            str(sender_id), 
                            context_messages=context_msgs,
                            response_mode="summary"  # Force summary mode for social media platforms
                        )

                        # SAFELY extract the answer string, no matter what
                        if isinstance(response_data, dict):
                            response_text = response_data.get("answer", "No answer found.")
                        elif isinstance(response_data, str):
                            response_text = response_data
                        else:
                            response_text = "Sorry, I couldn't generate a response."

                        # Now 100% guaranteed to be a string
                        response_text = str(response_text).strip()
                        print(f"Final Reply Text: {response_text[:100]}...")

                        # Append AI reply to history
                        append_conversation_message(TENANT_ID, str(sender_id), 'ai', response_text)

                    except HTTPException as e:
                        if e.status_code == 402:
                            print(f"⚠️ Inference provider credit limit exceeded for {platform} {sender_id}")
                            response_text = "Sorry, I've reached my query limit for now. Please try again later."
                        else:
                            print(f"Error in RAG call: {e}")
                            response_text = "An error occurred. Please try again."
                    
                    await send_reply(sender_id, response_text, access_token, platform.lower())
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}

def get_tenant_bot_token(tenant_id: str):
    token = TENANT_TELEGRAM_BOTS.get(str(tenant_id))
    if not token:
        raise ValueError(f"No Telegram bot token found for tenant {tenant_id}")
    return token

@app.post("/webhook/telegram/{tenant_id}") # Added tenant_id to the path for flexibility
async def telegram_webhook(
    tenant_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        data = await request.json()
        print(f"📨 Telegram payload for tenant {tenant_id}: {data}")
        
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            print(f"⚠️ Telegram webhook failed: Tenant {tenant_id} not found.")
            raise HTTPException(status_code=404, detail="Tenant not found")
            
        bot_token = tenant.telegram_bot_token
        if not bot_token:
            print(f"⚠️ Telegram webhook failed: Bot token for tenant {tenant_id} is missing.")
            return {"status": "ignored", "message": "Bot token missing for tenant."}

        if "message" not in data:
            return {"status": "ignored"}

        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")

        try:
            # Save incoming message
            append_conversation_message(tenant_id, str(chat_id), 'human', text)
            context_msgs = get_recent_conversation_context(tenant_id, str(chat_id), last_n_questions=10)

            response_data = answer_question_modern(
                text, 
                tenant_id, 
                str(chat_id), 
                context_messages=context_msgs,
                response_mode="summary"  # Force summary mode for Telegram
            )
            response_text = response_data.get("answer", "No answer found.")
            response_text = format_response(response_text)

            # Append AI reply
            append_conversation_message(tenant_id, str(chat_id), 'ai', response_text)
        except HTTPException as e:
            if e.status_code == 402:
                print(f"⚠️ RAG provider limit exceeded for Telegram {chat_id}")
                response_text = "Sorry, I've reached my query limit for now. Please try again later."
            else:
                print(f"Error in RAG call: {e}")
                response_text = "An error occurred. Please try again."

        # Send reply
        await send_telegram_reply(chat_id, response_text, bot_token)
        return {"status": "success"}

    except Exception as e:
        print(f"❌ Telegram webhook error: {e}")
        # Telegram expects a 200 OK, so we should return an appropriate status even on error
        return {"status": "error", "message": str(e)}

async def send_telegram_reply(chat_id, text, bot_token):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )

def format_response(text: str) -> str:
    """Format the response with structured bullet points, spacing, and breaks."""
    # Split the text into lines for processing
    lines = text.strip().split('\n')
    formatted_lines = []
    current_course = None
    intro = True

    for line in lines:
        line = line.strip()
        # Handle introductory text (before the course list)
        if intro and not re.match(r'^\d+\.\s*\*\*', line):
            formatted_lines.append(line)
            continue
        intro = False

        # Detect course entries (e.g., "1. **Engineering Mathematics-I & II**:")
        course_match = re.match(r'^\d+\.\s*\*\*(.*?)\*\*:(.*)', line)
        if course_match:
            course_name = course_match.group(1).strip()
            description = course_match.group(2).strip()
            if current_course:
                formatted_lines.append("")  # Add spacing between courses
            formatted_lines.append(f"• {course_name}")
            if description:
                formatted_lines.append(f"  - {description}")
            current_course = course_name
        else:
            # Handle continuation of description or notes
            if line and current_course:
                formatted_lines.append(f"  - {line}")
            elif line:
                formatted_lines.append("")  # Add spacing for notes
                formatted_lines.append(line)

    # Join with newlines for proper spacing
    return "\n".join(formatted_lines)

def split_message(text: str, max_length: int) -> list:
    """Split a message into chunks while preserving formatting."""
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1
        if current_length + line_length <= max_length:
            current_chunk.append(line)
            current_length += line_length
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

async def send_reply(recipient_id: str, reply_text: str, access_token: str, platform: str):
    if platform == "instagram":
        base_url = "https://graph.instagram.com/v23.0"
        max_length = 1000 
    else:
        base_url = "https://graph.facebook.com/v19.0"
        max_length = 2000 

    max_length -= 15 
    messages = split_message(reply_text, max_length)
    print(f"Sending {len(messages)} message(s) to {platform} user {recipient_id}")

    async with httpx.AsyncClient() as client:
        for i, message in enumerate(messages, 1):
            # Add "..." to non-final messages
            if len(messages) > 1:
                suffix = "..." if i < len(messages) else ""
                message_text = f"{i}/{len(messages)}: {message}{suffix}"
            else:
                message_text = message  # No numbering for single message
            
            payload = {
                "recipient": {"id": recipient_id},
                "message": {"text": message_text}
            }
            
            response = await client.post(
                f"{base_url}/me/messages?access_token={access_token}",
                json=payload
            )

            print(f"Send reply {i}/{len(messages)} response: {response.status_code}, {response.text}")
            if not response.is_success:
                print(f"❌ {platform.capitalize()} send failed for message {i}: {response.status_code}, {response.text}")
            else:
                print(f"✅ {platform.capitalize()} reply {i}/{len(messages)} sent")
            await asyncio.sleep(1)


# ==============================================================================
# RAG CONFIGURATION ENDPOINTS
# ==============================================================================

@app.get("/tenants/{tenant_id}/config/", response_model=ConfigResponse)
async def get_tenant_config_endpoint(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get RAG configuration for a specific tenant"""
    # Verify tenant ownership
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this tenant's configuration")
    
    try:
        config = get_tenant_config(tenant_id)
        config_schema = RAGConfigSchema(
            chunking=config.chunking.__dict__,
            retrieval=config.retrieval.__dict__,
            context=config.context.__dict__,
            quality=config.quality.__dict__,
            tenant_id=config.tenant_id
        )
        
        # Validate configuration
        warnings = config.validate_compatibility()
        validation = ConfigValidationResponse(
            valid=True,
            warnings=warnings,
            errors=[]
        )
        
        return ConfigResponse(
            success=True,
            message=f"Configuration retrieved for tenant {tenant_id}",
            config=config_schema,
            validation=validation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving configuration: {str(e)}")


@app.put("/tenants/{tenant_id}/config/", response_model=ConfigResponse)
async def update_tenant_config_endpoint(
    tenant_id: int,
    config_data: RAGConfigSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update RAG configuration for a specific tenant"""
    # Verify tenant ownership
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this tenant's configuration")
    
    try:
        # Convert schema to config model
        config_dict = config_data.dict()
        config_dict['tenant_id'] = tenant_id
        config = RAGSystemConfig.from_dict(config_dict)
        
        # Update configuration
        success = update_tenant_config(tenant_id, config)
        
        if success:
            # Validate configuration
            warnings = config.validate_compatibility()
            validation = ConfigValidationResponse(
                valid=True,
                warnings=warnings,
                errors=[]
            )
            
            return ConfigResponse(
                success=True,
                message=f"Configuration updated for tenant {tenant_id}",
                config=config_data,
                validation=validation
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")


@app.patch("/tenants/{tenant_id}/config/", response_model=ConfigResponse)
async def partial_update_tenant_config(
    tenant_id: int,
    update_request: ConfigUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Partially update RAG configuration for a specific tenant"""
    # Verify tenant ownership
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this tenant's configuration")
    
    try:
        config_manager = get_config_manager()
        success = config_manager.update_config_partial(tenant_id, update_request.updates)
        
        if success:
            # Get updated configuration
            updated_config = get_tenant_config(tenant_id)
            config_schema = RAGConfigSchema(
                chunking=updated_config.chunking.__dict__,
                retrieval=updated_config.retrieval.__dict__,
                context=updated_config.context.__dict__,
                quality=updated_config.quality.__dict__,
                tenant_id=updated_config.tenant_id
            )
            
            # Validate configuration
            warnings = updated_config.validate_compatibility()
            validation = ConfigValidationResponse(
                valid=True,
                warnings=warnings,
                errors=[]
            )
            
            return ConfigResponse(
                success=True,
                message=f"Configuration partially updated for tenant {tenant_id}",
                config=config_schema,
                validation=validation
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration update: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")


@app.post("/tenants/{tenant_id}/config/reset/", response_model=ConfigResponse)
async def reset_tenant_config(
    tenant_id: int,
    reset_request: ConfigResetRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Reset tenant configuration to a default preset"""
    # Verify tenant ownership
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this tenant's configuration")
    
    try:
        config_manager = get_config_manager()
        success = config_manager.reset_to_default(tenant_id, reset_request.preset.value)
        
        if success:
            # Get reset configuration
            reset_config = get_tenant_config(tenant_id)
            config_schema = RAGConfigSchema(
                chunking=reset_config.chunking.__dict__,
                retrieval=reset_config.retrieval.__dict__,
                context=reset_config.context.__dict__,
                quality=reset_config.quality.__dict__,
                tenant_id=reset_config.tenant_id
            )
            
            return ConfigResponse(
                success=True,
                message=f"Configuration reset to '{reset_request.preset.value}' preset for tenant {tenant_id}",
                config=config_schema,
                validation=ConfigValidationResponse(valid=True, warnings=[], errors=[])
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to reset configuration")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting configuration: {str(e)}")


@app.post("/tenants/{tenant_id}/config/invalidate-cache/")
async def invalidate_tenant_config_cache(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Invalidate configuration cache for a tenant (force reload from storage)"""
    # Verify tenant ownership
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant or tenant.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to manage this tenant's configuration")
    
    try:
        config_manager = get_config_manager()
        config_manager.invalidate_cache(tenant_id)
        
        return {"success": True, "message": f"Configuration cache invalidated for tenant {tenant_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invalidating cache: {str(e)}")


@app.get("/config/presets/", response_model=ConfigPresetListResponse)
async def get_config_presets():
    """Get available configuration presets"""
    try:
        presets = {}
        descriptions = {
            "balanced": "Balanced configuration suitable for most use cases",
            "high_precision": "High precision configuration with strict similarity thresholds",
            "high_recall": "High recall configuration for comprehensive document retrieval",
            "fast_response": "Fast response configuration optimized for speed"
        }
        
        for preset_name, preset_config in DEFAULT_CONFIGS.items():
            presets[preset_name] = RAGConfigSchema(
                chunking=preset_config.chunking.__dict__,
                retrieval=preset_config.retrieval.__dict__,
                context=preset_config.context.__dict__,
                quality=preset_config.quality.__dict__,
                tenant_id=None
            )
        
        return ConfigPresetListResponse(
            presets=presets,
            descriptions=descriptions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving presets: {str(e)}")


@app.get("/admin/config/tenants/", response_model=TenantConfigListResponse)
async def list_all_tenant_configs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List configurations for all tenants owned by the current user"""
    try:
        # Get all tenants owned by the current user
        user_tenants = db.query(Tenant).filter(Tenant.creator_id == current_user.id).all()
        tenant_ids = [tenant.id for tenant in user_tenants]
        
        config_manager = get_config_manager()
        tenant_configs = {}
        
        for tenant_id in tenant_ids:
            try:
                config = get_tenant_config(tenant_id)
                tenant_configs[tenant_id] = RAGConfigSchema(
                    chunking=config.chunking.__dict__,
                    retrieval=config.retrieval.__dict__,
                    context=config.context.__dict__,
                    quality=config.quality.__dict__,
                    tenant_id=config.tenant_id
                )
            except Exception as e:
                print(f"⚠️ Error loading config for tenant {tenant_id}: {e}")
                continue
        
        return TenantConfigListResponse(
            tenant_configs=tenant_configs,
            total_count=len(tenant_configs)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tenant configurations: {str(e)}")