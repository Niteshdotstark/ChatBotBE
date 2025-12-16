from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from fastapi import UploadFile
import re

class UserBase(BaseModel):
    email: str = Field(..., description="User email") # Use EmailStr for email validation
    username: Optional[str] # Username is now a required field for UserBase
    phone_number: Optional[str] = None # Optional phone number
    address: Optional[str] = None # Optional address
    is_active: Optional[bool] = True # Include is_active for consistency, default to True

class UserCreate(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8)
    username: str
    phone_number: Optional[str] = None
    address: Optional[str] = None
    is_active: Optional[bool] = True

    @validator('email')
    def validate_email(cls, v):
        if not re.match(r"^[\w\.\-]*@[\w\.\-]+\.\w+$", v):
            raise ValueError('Invalid email format')
        return v.lower()

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r"[0-9]", v):
            raise ValueError('Password must contain at least one number')
        return v

class UserResponse(UserBase):
    id: int
    tenant_id: Optional[int] = None
    class Config:
        orm_mode = True

class TenantBase(BaseModel):
    id: int
    name: str
    created_at: datetime
    fb_url: Optional[str]
    insta_url: Optional[str]
    fb_verify_token: Optional[str] = None
    fb_access_token: Optional[str] = None
    insta_access_token: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    class Config:
        orm_mode = True

class TenantUpdate(BaseModel):
    name: Optional[str] = None  # Make name optional for updates
    fb_url: Optional[str] = None
    insta_url: Optional[str] = None
    fb_verify_token: Optional[str] = None
    fb_access_token: Optional[str] = None
    insta_access_token: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

class TenantCreate(BaseModel):
    name: str
    fb_url: Optional[str]
    insta_url: Optional[str]
    fb_verify_token: Optional[str] = None
    fb_access_token: Optional[str] = None
    insta_access_token: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class KnowledgeBaseFileCreate(BaseModel):
    pass

class KnowledgeBaseFileResponse(BaseModel):
    id: str
    filename: Optional[str]
    stored_filename: Optional[str]
    file_path: Optional[str]
    file_type: str
    url: Optional[str]
    tenant_id: int
    uploaded_by: Optional[int]
    created_at: datetime
    class Config:
        orm_mode = True

class DatabaseBase(BaseModel):
    name: str
    description: Optional[str] = None

class TopQuestionAnalysis(BaseModel):
    count: int
    cleaned_question: str
    example_original_question: str

class DatabaseCreate(DatabaseBase):
    pass

class DatabaseResponse(DatabaseBase):
    id: int
    tenant_id: int
    created_at: datetime
    class Config:
        orm_mode = True


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = Field(default_factory=list)

class TenantValuesUpdate(BaseModel):
    tenant_id: int

class ConversationCountResponse(BaseModel):
    count: int

class TenantResponse(TenantBase):
    id: int
    created_at: datetime
    knowledge_item_count: int
    conversation_count: int

    class Config:
        orm_mode = True

class ActivityLogResponse(BaseModel):
    id: str  # A unique ID for the key prop in React
    title: str
    subtitle: str
    created_at: datetime
    type: str  # 'info', 'success', or 'warning'

    class Config:
        from_attributes = True # or orm_mode = True