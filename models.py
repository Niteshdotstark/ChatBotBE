from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False) # Make email non-nullable
    hashed_password = Column(String, nullable=False) # Make password non-nullable
    # New fields
    username = Column(String, unique=True, index=True, nullable=False) # Username should typically be unique and required
    phone_number = Column(String, unique=True, index=True, nullable=True) # Phone number can be optional and unique
    address = Column(String, nullable=True) # Address can be optional
    # Add is_active if you want to manage user activation status
    is_active = Column(Boolean, default=True)

    created_tenants = relationship("Tenant", back_populates="creator")


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    fb_url = Column(String, nullable=True)
    insta_url = Column(String, nullable=True)
    fb_verify_token = Column(String, nullable=True)
    fb_access_token = Column(String, nullable=True)
    insta_access_token = Column(String, nullable=True)
    telegram_bot_token = Column(String, nullable=True)
    telegram_chat_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    creator = relationship("User", back_populates="created_tenants")
    files = relationship("KnowledgeBaseFile", back_populates="tenant")

class KnowledgeBaseFile(Base):
    __tablename__ = "knowledge_base_files"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, index=True, nullable=True)
    stored_filename = Column(String, index=True, nullable=True)
    file_path = Column(String, nullable=True)
    file_type = Column(String)  # MIME type or "url"
    category = Column(String)  # "file", "url", "database"
    url = Column(String, nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    tenant = relationship("Tenant", back_populates="files")

class TenantValues(Base):
    __tablename__ = "tenant_values"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, nullable=False, default=1)