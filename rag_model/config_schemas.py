"""
Pydantic schemas for RAG configuration API endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from enum import Enum


class ConfigPreset(str, Enum):
    """Available configuration presets"""
    BALANCED = "balanced"
    HIGH_PRECISION = "high_precision"
    HIGH_RECALL = "high_recall"
    FAST_RESPONSE = "fast_response"


class ChunkingConfigSchema(BaseModel):
    """Schema for chunking configuration"""
    base_chunk_size: int = Field(default=1000, ge=100, le=5000, description="Base chunk size in characters")
    overlap_ratio: float = Field(default=0.15, ge=0.0, le=0.5, description="Overlap ratio between chunks")
    max_chunk_size: int = Field(default=2000, ge=500, le=10000, description="Maximum chunk size")
    min_chunk_size: int = Field(default=200, ge=50, le=1000, description="Minimum chunk size")
    preserve_sentences: bool = Field(default=True, description="Preserve sentence boundaries")
    preserve_paragraphs: bool = Field(default=True, description="Preserve paragraph boundaries")
    document_type_specific: Dict[str, Any] = Field(default_factory=dict, description="Document type specific settings")
    
    @validator('max_chunk_size')
    def max_chunk_size_validation(cls, v, values):
        if 'base_chunk_size' in values and v <= values['base_chunk_size']:
            raise ValueError('max_chunk_size must be greater than base_chunk_size')
        return v
    
    @validator('min_chunk_size')
    def min_chunk_size_validation(cls, v, values):
        if 'base_chunk_size' in values and v >= values['base_chunk_size']:
            raise ValueError('min_chunk_size must be less than base_chunk_size')
        return v


class RetrievalConfigSchema(BaseModel):
    """Schema for retrieval configuration"""
    base_top_k: int = Field(default=8, ge=1, le=50, description="Base number of documents to retrieve")
    max_top_k: int = Field(default=20, ge=5, le=100, description="Maximum number of documents to retrieve")
    min_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    diversity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Diversity threshold for results")
    rerank_enabled: bool = Field(default=True, description="Enable result re-ranking")
    hybrid_search_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Vector vs keyword search balance")
    
    @validator('max_top_k')
    def max_top_k_validation(cls, v, values):
        if 'base_top_k' in values and v <= values['base_top_k']:
            raise ValueError('max_top_k must be greater than base_top_k')
        return v


class ContextConfigSchema(BaseModel):
    """Schema for context configuration"""
    base_context_turns: int = Field(default=10, ge=1, le=100, description="Base number of conversation turns")
    max_context_turns: int = Field(default=25, ge=5, le=200, description="Maximum number of conversation turns")
    max_context_tokens: int = Field(default=4000, ge=500, le=10000, description="Maximum context tokens")
    summarization_threshold: int = Field(default=15, ge=5, le=500, description="Threshold for conversation summarization")
    context_relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Context relevance threshold")
    
    @validator('max_context_turns')
    def max_context_turns_validation(cls, v, values):
        if 'base_context_turns' in values and v <= values['base_context_turns']:
            raise ValueError('max_context_turns must be greater than base_context_turns')
        return v
    
    @validator('summarization_threshold')
    def summarization_threshold_validation(cls, v, values):
        if 'base_context_turns' in values and v <= values['base_context_turns']:
            raise ValueError('summarization_threshold must be greater than base_context_turns')
        return v


class QualityConfigSchema(BaseModel):
    """Schema for quality monitoring configuration"""
    min_retrieval_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum retrieval confidence")
    min_response_relevance: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum response relevance")
    min_context_utilization: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum context utilization")
    alert_threshold_violations: int = Field(default=5, ge=1, le=100, description="Alert threshold violations")
    performance_window_minutes: int = Field(default=60, ge=1, le=1440, description="Performance monitoring window")


class RAGConfigSchema(BaseModel):
    """Complete RAG system configuration schema"""
    chunking: ChunkingConfigSchema = Field(default_factory=ChunkingConfigSchema)
    retrieval: RetrievalConfigSchema = Field(default_factory=RetrievalConfigSchema)
    context: ContextConfigSchema = Field(default_factory=ContextConfigSchema)
    quality: QualityConfigSchema = Field(default_factory=QualityConfigSchema)
    tenant_id: Optional[int] = Field(None, description="Tenant ID")


class ConfigUpdateRequest(BaseModel):
    """Request schema for partial configuration updates"""
    updates: Dict[str, Any] = Field(..., description="Configuration updates to apply")


class ConfigResetRequest(BaseModel):
    """Request schema for configuration reset"""
    preset: ConfigPreset = Field(default=ConfigPreset.BALANCED, description="Configuration preset to use")


class ConfigValidationResponse(BaseModel):
    """Response schema for configuration validation"""
    valid: bool = Field(..., description="Whether configuration is valid")
    warnings: List[str] = Field(default_factory=list, description="Configuration warnings")
    errors: List[str] = Field(default_factory=list, description="Configuration errors")


class ConfigResponse(BaseModel):
    """Response schema for configuration operations"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Operation result message")
    config: Optional[RAGConfigSchema] = Field(None, description="Current configuration")
    validation: Optional[ConfigValidationResponse] = Field(None, description="Validation results")


class TenantConfigListResponse(BaseModel):
    """Response schema for listing tenant configurations"""
    tenant_configs: Dict[int, RAGConfigSchema] = Field(..., description="Tenant configurations")
    total_count: int = Field(..., description="Total number of tenant configurations")


class ConfigPresetListResponse(BaseModel):
    """Response schema for listing available presets"""
    presets: Dict[str, RAGConfigSchema] = Field(..., description="Available configuration presets")
    descriptions: Dict[str, str] = Field(..., description="Preset descriptions")