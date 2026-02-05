"""
Enhanced Configuration System for RAG Chatbot
Implements configuration data models for chunking, retrieval, and context parameters
with validation and tenant-specific management capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
import os
from pathlib import Path


class ChunkType(Enum):
    """Types of document chunks"""
    SEMANTIC = "semantic"
    FIXED = "fixed"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategies"""
    base_chunk_size: int = 1000
    overlap_ratio: float = 0.15
    max_chunk_size: int = 2000
    min_chunk_size: int = 200
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    document_type_specific: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (100 <= self.base_chunk_size <= 5000):
            raise ValueError("base_chunk_size must be between 100 and 5000")
        if not (0.0 <= self.overlap_ratio <= 0.5):
            raise ValueError("overlap_ratio must be between 0.0 and 0.5")
        if self.max_chunk_size <= self.base_chunk_size:
            raise ValueError("max_chunk_size must be greater than base_chunk_size")
        if self.min_chunk_size >= self.base_chunk_size:
            raise ValueError("min_chunk_size must be less than base_chunk_size")


@dataclass
class RetrievalConfig:
    """Configuration for dynamic retrieval parameters"""
    base_top_k: int = 8
    max_top_k: int = 20
    min_similarity_threshold: float = 0.7
    diversity_threshold: float = 0.8
    rerank_enabled: bool = True
    hybrid_search_weight: float = 0.7  # Vector vs keyword search balance
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (1 <= self.base_top_k <= 50):
            raise ValueError("base_top_k must be between 1 and 50")
        if self.max_top_k <= self.base_top_k:
            raise ValueError("max_top_k must be greater than base_top_k")
        if not (0.0 <= self.min_similarity_threshold <= 1.0):
            raise ValueError("min_similarity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.diversity_threshold <= 1.0):
            raise ValueError("diversity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.hybrid_search_weight <= 1.0):
            raise ValueError("hybrid_search_weight must be between 0.0 and 1.0")


@dataclass
class ContextConfig:
    """Configuration for conversation context management"""
    base_context_turns: int = 10
    max_context_turns: int = 25
    max_context_tokens: int = 4000
    summarization_threshold: int = 15
    context_relevance_threshold: float = 0.6
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (1 <= self.base_context_turns <= 100):
            raise ValueError("base_context_turns must be between 1 and 100")
        if self.max_context_turns <= self.base_context_turns:
            raise ValueError("max_context_turns must be greater than base_context_turns")
        if not (500 <= self.max_context_tokens <= 10000):
            raise ValueError("max_context_tokens must be between 500 and 10000")
        if self.summarization_threshold <= self.base_context_turns:
            raise ValueError("summarization_threshold must be greater than base_context_turns")
        if not (0.0 <= self.context_relevance_threshold <= 1.0):
            raise ValueError("context_relevance_threshold must be between 0.0 and 1.0")


@dataclass
class QualityConfig:
    """Configuration for quality monitoring and alerting"""
    min_retrieval_confidence: float = 0.5
    min_response_relevance: float = 0.6
    min_context_utilization: float = 0.4
    alert_threshold_violations: int = 5
    performance_window_minutes: int = 60
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (0.0 <= self.min_retrieval_confidence <= 1.0):
            raise ValueError("min_retrieval_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.min_response_relevance <= 1.0):
            raise ValueError("min_response_relevance must be between 0.0 and 1.0")
        if not (0.0 <= self.min_context_utilization <= 1.0):
            raise ValueError("min_context_utilization must be between 0.0 and 1.0")
        if self.alert_threshold_violations < 1:
            raise ValueError("alert_threshold_violations must be at least 1")
        if self.performance_window_minutes < 1:
            raise ValueError("performance_window_minutes must be at least 1")


@dataclass
class RAGSystemConfig:
    """Complete RAG system configuration"""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    tenant_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "chunking": {
                "base_chunk_size": self.chunking.base_chunk_size,
                "overlap_ratio": self.chunking.overlap_ratio,
                "max_chunk_size": self.chunking.max_chunk_size,
                "min_chunk_size": self.chunking.min_chunk_size,
                "preserve_sentences": self.chunking.preserve_sentences,
                "preserve_paragraphs": self.chunking.preserve_paragraphs,
                "document_type_specific": self.chunking.document_type_specific
            },
            "retrieval": {
                "base_top_k": self.retrieval.base_top_k,
                "max_top_k": self.retrieval.max_top_k,
                "min_similarity_threshold": self.retrieval.min_similarity_threshold,
                "diversity_threshold": self.retrieval.diversity_threshold,
                "rerank_enabled": self.retrieval.rerank_enabled,
                "hybrid_search_weight": self.retrieval.hybrid_search_weight
            },
            "context": {
                "base_context_turns": self.context.base_context_turns,
                "max_context_turns": self.context.max_context_turns,
                "max_context_tokens": self.context.max_context_tokens,
                "summarization_threshold": self.context.summarization_threshold,
                "context_relevance_threshold": self.context.context_relevance_threshold
            },
            "quality": {
                "min_retrieval_confidence": self.quality.min_retrieval_confidence,
                "min_response_relevance": self.quality.min_response_relevance,
                "min_context_utilization": self.quality.min_context_utilization,
                "alert_threshold_violations": self.quality.alert_threshold_violations,
                "performance_window_minutes": self.quality.performance_window_minutes
            },
            "tenant_id": self.tenant_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGSystemConfig':
        """Create configuration from dictionary"""
        chunking_data = data.get("chunking", {})
        retrieval_data = data.get("retrieval", {})
        context_data = data.get("context", {})
        quality_data = data.get("quality", {})
        
        return cls(
            chunking=ChunkingConfig(**chunking_data),
            retrieval=RetrievalConfig(**retrieval_data),
            context=ContextConfig(**context_data),
            quality=QualityConfig(**quality_data),
            tenant_id=data.get("tenant_id")
        )
    
    def validate_compatibility(self) -> List[str]:
        """Validate parameter combinations and return any warnings"""
        warnings = []
        
        # Check if context tokens can accommodate chunk sizes
        avg_chunk_size = (self.chunking.base_chunk_size + self.chunking.max_chunk_size) / 2
        estimated_tokens = avg_chunk_size * self.retrieval.base_top_k * 1.3  # rough token estimation
        
        if estimated_tokens > self.context.max_context_tokens * 0.7:  # Leave 30% for conversation
            warnings.append(
                f"Retrieved content may exceed context window. "
                f"Estimated tokens: {int(estimated_tokens)}, "
                f"Available: {int(self.context.max_context_tokens * 0.7)}"
            )
        
        # Check overlap vs chunk size ratio
        overlap_size = int(self.chunking.base_chunk_size * self.chunking.overlap_ratio)
        if overlap_size < 50:
            warnings.append(
                f"Overlap size ({overlap_size}) may be too small for context preservation"
            )
        
        # Check retrieval thresholds
        if self.retrieval.min_similarity_threshold > 0.9:
            warnings.append(
                "Very high similarity threshold may result in too few retrieved documents"
            )
        
        return warnings


# Default configurations for different use cases
DEFAULT_CONFIGS = {
    "balanced": RAGSystemConfig(),
    "high_precision": RAGSystemConfig(
        retrieval=RetrievalConfig(
            base_top_k=5,
            min_similarity_threshold=0.8,
            diversity_threshold=0.9
        ),
        quality=QualityConfig(
            min_retrieval_confidence=0.7,
            min_response_relevance=0.8
        )
    ),
    "high_recall": RAGSystemConfig(
        retrieval=RetrievalConfig(
            base_top_k=15,
            max_top_k=30,
            min_similarity_threshold=0.6,
            diversity_threshold=0.7
        ),
        context=ContextConfig(
            max_context_turns=30,
            max_context_tokens=6000
        )
    ),
    "fast_response": RAGSystemConfig(
        chunking=ChunkingConfig(
            base_chunk_size=800,
            max_chunk_size=1200
        ),
        retrieval=RetrievalConfig(
            base_top_k=5,
            max_top_k=10,
            rerank_enabled=False
        ),
        context=ContextConfig(
            base_context_turns=5,
            max_context_turns=10
        )
    )
}