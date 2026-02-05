"""
Adaptive Document Chunker for RAG System
Implements semantic boundary detection, hierarchical chunking, and intelligent overlap management.
"""

import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import nltk
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config_models import ChunkingConfig, ChunkType


# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class DocumentType(Enum):
    """Document types for adaptive chunking"""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    CODE = "code"
    CSV = "csv"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class OverlapInfo:
    """Information about chunk overlap"""
    start_overlap: int
    end_overlap: int
    overlap_content: str
    semantic_boundary: bool


@dataclass
class OverlapStrategy:
    """Strategy for managing overlaps between chunks"""
    overlap_type: str  # 'semantic', 'fixed', 'adaptive'
    overlap_size: int
    preserve_sentences: bool
    preserve_context: bool
    quality_score: float = 0.0


class IntelligentOverlapManager:
    """Manages intelligent overlap between document chunks"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.boundary_detector = SemanticBoundaryDetector()
    
    def calculate_optimal_overlap(self, prev_chunk: str, current_chunk: str, 
                                 doc_type: DocumentType) -> OverlapStrategy:
        """Calculate optimal overlap strategy between two chunks"""
        base_overlap_size = int(len(prev_chunk) * self.config.overlap_ratio)
        
        # Adjust overlap based on document type
        type_adjustments = {
            DocumentType.CODE: 1.5,  # More overlap for code to preserve context
            DocumentType.MARKDOWN: 1.2,  # More overlap for structured content
            DocumentType.HTML: 1.1,  # Slightly more for HTML
            DocumentType.JSON: 0.5,  # Less overlap for structured data
            DocumentType.CSV: 0.3,   # Minimal overlap for tabular data
            DocumentType.TEXT: 1.0,  # Standard overlap
            DocumentType.PDF: 1.0,   # Standard overlap
        }
        
        adjustment = type_adjustments.get(doc_type, 1.0)
        adjusted_overlap_size = int(base_overlap_size * adjustment)
        
        # Ensure overlap doesn't exceed reasonable bounds
        max_overlap = min(len(prev_chunk) // 2, self.config.max_chunk_size // 4)
        min_overlap = max(50, len(prev_chunk) // 10)
        
        optimal_size = max(min_overlap, min(adjusted_overlap_size, max_overlap))
        
        # Determine overlap strategy based on content analysis
        if self._has_strong_semantic_boundaries(prev_chunk, current_chunk):
            strategy_type = 'semantic'
            preserve_sentences = True
            preserve_context = True
        elif doc_type in [DocumentType.CODE, DocumentType.JSON]:
            strategy_type = 'adaptive'
            preserve_sentences = False
            preserve_context = True
        else:
            strategy_type = 'fixed'
            preserve_sentences = self.config.preserve_sentences
            preserve_context = True
        
        return OverlapStrategy(
            overlap_type=strategy_type,
            overlap_size=optimal_size,
            preserve_sentences=preserve_sentences,
            preserve_context=preserve_context
        )
    
    def create_context_preserving_overlap(self, prev_chunk: str, current_chunk: str,
                                        strategy: OverlapStrategy) -> OverlapInfo:
        """Create overlap that preserves context between chunks"""
        if strategy.overlap_size <= 0:
            return OverlapInfo(0, 0, "", False)
        
        if strategy.overlap_type == 'semantic':
            return self._create_semantic_overlap(prev_chunk, current_chunk, strategy)
        elif strategy.overlap_type == 'adaptive':
            return self._create_adaptive_overlap(prev_chunk, current_chunk, strategy)
        else:
            return self._create_fixed_overlap(prev_chunk, current_chunk, strategy)
    
    def _create_semantic_overlap(self, prev_chunk: str, current_chunk: str,
                               strategy: OverlapStrategy) -> OverlapInfo:
        """Create overlap based on semantic boundaries"""
        target_size = strategy.overlap_size
        
        # Find sentence boundaries in the previous chunk
        sentence_boundaries = self.boundary_detector.detect_sentence_boundaries(prev_chunk)
        
        if not sentence_boundaries:
            return self._create_fixed_overlap(prev_chunk, current_chunk, strategy)
        
        # Find the best sentence boundary for overlap
        best_boundary = None
        best_score = -1
        
        for boundary in sentence_boundaries:
            overlap_start = len(prev_chunk) - boundary
            if overlap_start <= 0:
                continue
            
            # Score based on how close to target size and semantic quality
            size_score = 1.0 - abs(overlap_start - target_size) / max(target_size, overlap_start)
            
            # Check if this creates a good semantic boundary
            overlap_content = prev_chunk[-overlap_start:]
            semantic_score = self._score_semantic_quality(overlap_content, current_chunk[:200])
            
            total_score = (size_score * 0.6) + (semantic_score * 0.4)
            
            if total_score > best_score:
                best_score = total_score
                best_boundary = boundary
        
        if best_boundary:
            overlap_start = len(prev_chunk) - best_boundary
            overlap_content = prev_chunk[-overlap_start:]
            
            return OverlapInfo(
                start_overlap=0,
                end_overlap=overlap_start,
                overlap_content=overlap_content,
                semantic_boundary=True
            )
        
        # Fall back to fixed overlap if no good semantic boundary found
        return self._create_fixed_overlap(prev_chunk, current_chunk, strategy)
    
    def _create_adaptive_overlap(self, prev_chunk: str, current_chunk: str,
                               strategy: OverlapStrategy) -> OverlapInfo:
        """Create adaptive overlap based on content characteristics"""
        target_size = strategy.overlap_size
        
        # Analyze content characteristics
        prev_lines = prev_chunk.split('\n')
        current_lines = current_chunk.split('\n')
        
        # For code, try to preserve complete logical units
        if self._looks_like_code(prev_chunk):
            return self._create_code_aware_overlap(prev_chunk, current_chunk, strategy)
        
        # For structured content, try to preserve structure
        if self._has_list_structure(prev_chunk):
            return self._create_structure_aware_overlap(prev_chunk, current_chunk, strategy)
        
        # Default to semantic overlap
        return self._create_semantic_overlap(prev_chunk, current_chunk, strategy)
    
    def _create_fixed_overlap(self, prev_chunk: str, current_chunk: str,
                            strategy: OverlapStrategy) -> OverlapInfo:
        """Create fixed-size overlap"""
        overlap_size = min(strategy.overlap_size, len(prev_chunk))
        
        if overlap_size <= 0:
            return OverlapInfo(0, 0, "", False)
        
        overlap_content = prev_chunk[-overlap_size:]
        
        # If preserving sentences, adjust to sentence boundary
        if strategy.preserve_sentences and overlap_size > 100:
            sentences = self.boundary_detector.detect_sentence_boundaries(overlap_content)
            if sentences:
                # Use the last complete sentence boundary
                last_sentence_end = sentences[-1]
                if last_sentence_end < len(overlap_content):
                    adjusted_content = overlap_content[:last_sentence_end]
                    if len(adjusted_content) > 50:  # Ensure minimum overlap
                        overlap_content = adjusted_content
                        overlap_size = len(adjusted_content)
        
        return OverlapInfo(
            start_overlap=0,
            end_overlap=overlap_size,
            overlap_content=overlap_content,
            semantic_boundary=strategy.preserve_sentences
        )
    
    def _create_code_aware_overlap(self, prev_chunk: str, current_chunk: str,
                                 strategy: OverlapStrategy) -> OverlapInfo:
        """Create overlap that preserves code structure"""
        target_size = strategy.overlap_size
        lines = prev_chunk.split('\n')
        
        # Find logical code boundaries (function ends, class ends, etc.)
        code_boundaries = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.endswith(':') or stripped.endswith('{') or 
                stripped.endswith('}') or stripped.startswith('def ') or
                stripped.startswith('class ') or stripped.startswith('function ')):
                code_boundaries.append(sum(len(l) + 1 for l in lines[:i+1]))
        
        if code_boundaries:
            # Find the boundary closest to our target
            best_boundary = min(code_boundaries, 
                              key=lambda x: abs(len(prev_chunk) - x - target_size))
            overlap_start = len(prev_chunk) - best_boundary
            
            if overlap_start > 0:
                overlap_content = prev_chunk[-overlap_start:]
                return OverlapInfo(
                    start_overlap=0,
                    end_overlap=overlap_start,
                    overlap_content=overlap_content,
                    semantic_boundary=True
                )
        
        # Fall back to fixed overlap
        return self._create_fixed_overlap(prev_chunk, current_chunk, strategy)
    
    def _create_structure_aware_overlap(self, prev_chunk: str, current_chunk: str,
                                      strategy: OverlapStrategy) -> OverlapInfo:
        """Create overlap that preserves list/structured content"""
        target_size = strategy.overlap_size
        lines = prev_chunk.split('\n')
        
        # Find list item boundaries
        list_boundaries = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith(('- ', '* ', '+ ')) or 
                re.match(r'^\d+\.', stripped) or
                stripped.startswith(('• ', '◦ '))):
                list_boundaries.append(sum(len(l) + 1 for l in lines[:i]))
        
        if list_boundaries:
            # Find the boundary closest to our target
            best_boundary = min(list_boundaries,
                              key=lambda x: abs(len(prev_chunk) - x - target_size))
            overlap_start = len(prev_chunk) - best_boundary
            
            if overlap_start > 0:
                overlap_content = prev_chunk[-overlap_start:]
                return OverlapInfo(
                    start_overlap=0,
                    end_overlap=overlap_start,
                    overlap_content=overlap_content,
                    semantic_boundary=True
                )
        
        # Fall back to fixed overlap
        return self._create_fixed_overlap(prev_chunk, current_chunk, strategy)
    
    def _has_strong_semantic_boundaries(self, prev_chunk: str, current_chunk: str) -> bool:
        """Check if chunks have strong semantic boundaries"""
        # Check for paragraph breaks
        if prev_chunk.endswith('\n\n') or current_chunk.startswith('\n\n'):
            return True
        
        # Check for sentence endings
        if prev_chunk.rstrip().endswith(('.', '!', '?')):
            return True
        
        # Check for section breaks
        if re.search(r'\n\s*[A-Z][A-Z\s]{5,}\s*\n', prev_chunk[-100:] + current_chunk[:100]):
            return True
        
        return False
    
    def _score_semantic_quality(self, overlap_content: str, next_content: str) -> float:
        """Score the semantic quality of an overlap"""
        score = 0.0
        
        # Bonus for complete sentences
        if overlap_content.strip().endswith(('.', '!', '?')):
            score += 0.3
        
        # Bonus for paragraph boundaries
        if overlap_content.endswith('\n\n') or next_content.startswith('\n\n'):
            score += 0.2
        
        # Penalty for cutting mid-sentence
        if not overlap_content.strip().endswith(('.', '!', '?', ':', ';')):
            score -= 0.2
        
        # Bonus for preserving context words
        overlap_words = set(overlap_content.lower().split())
        next_words = set(next_content.lower().split())
        common_words = overlap_words.intersection(next_words)
        
        if len(overlap_words) > 0:
            context_score = len(common_words) / len(overlap_words)
            score += context_score * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+\s*[:\(]',
            r'function\s+\w+\s*\(',
            r'import\s+\w+',
            r'#include\s*<',
            r'{\s*$',
            r'}\s*$',
        ]
        
        return any(re.search(pattern, text) for pattern in code_indicators)
    
    def _has_list_structure(self, text: str) -> bool:
        """Check if text has list structure"""
        lines = text.split('\n')
        list_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith(('- ', '* ', '+ ')) or 
                re.match(r'^\d+\.', stripped) or
                stripped.startswith(('• ', '◦ '))):
                list_lines += 1
        
        return list_lines > len(lines) * 0.3  # 30% of lines are list items


@dataclass
class DocumentChunk:
    """Enhanced document chunk with semantic information"""
    content: str
    metadata: Dict[str, Any]
    chunk_type: ChunkType
    semantic_boundaries: List[int]
    parent_document_id: str
    chunk_index: int
    overlap_info: Optional[OverlapInfo] = None
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format"""
        enhanced_metadata = self.metadata.copy()
        enhanced_metadata.update({
            'chunk_type': self.chunk_type.value,
            'chunk_index': self.chunk_index,
            'parent_document_id': self.parent_document_id,
            'has_semantic_boundaries': len(self.semantic_boundaries) > 0
        })
        return Document(page_content=self.content, metadata=enhanced_metadata)


class SemanticBoundaryDetector:
    """Detects semantic boundaries in text for intelligent chunking"""
    
    def __init__(self):
        self.sentence_tokenizer = nltk.sent_tokenize
        
    def detect_sentence_boundaries(self, text: str) -> List[int]:
        """Detect sentence boundaries in text"""
        sentences = self.sentence_tokenizer(text)
        boundaries = []
        current_pos = 0
        
        for sentence in sentences:
            # Find the sentence in the text starting from current position
            sentence_start = text.find(sentence, current_pos)
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)
                boundaries.append(sentence_end)
                current_pos = sentence_end
        
        return boundaries
    
    def detect_paragraph_boundaries(self, text: str) -> List[int]:
        """Detect paragraph boundaries (double newlines or similar patterns)"""
        boundaries = []
        
        # Pattern for paragraph breaks: double newlines, possibly with whitespace
        paragraph_pattern = r'\n\s*\n'
        
        for match in re.finditer(paragraph_pattern, text):
            boundaries.append(match.end())
        
        return boundaries
    
    def detect_section_boundaries(self, text: str, doc_type: DocumentType) -> List[int]:
        """Detect section boundaries based on document type"""
        boundaries = []
        
        if doc_type == DocumentType.MARKDOWN:
            # Markdown headers
            header_pattern = r'\n#{1,6}\s+.+\n'
            for match in re.finditer(header_pattern, text):
                boundaries.append(match.start())
                
        elif doc_type == DocumentType.HTML:
            # HTML headers and section tags
            section_pattern = r'<(?:h[1-6]|section|article|div class="section")[^>]*>'
            for match in re.finditer(section_pattern, text, re.IGNORECASE):
                boundaries.append(match.start())
                
        elif doc_type == DocumentType.CODE:
            # Function/class definitions
            code_patterns = [
                r'\ndef\s+\w+',  # Python functions
                r'\nclass\s+\w+',  # Python classes
                r'\nfunction\s+\w+',  # JavaScript functions
                r'\n\w+\s+\w+\s*\([^)]*\)\s*{',  # C-style functions
            ]
            for pattern in code_patterns:
                for match in re.finditer(pattern, text):
                    boundaries.append(match.start())
        
        return boundaries
    
    def get_optimal_split_points(self, text: str, target_size: int, 
                                doc_type: DocumentType = DocumentType.TEXT) -> List[int]:
        """Find optimal split points considering semantic boundaries"""
        sentence_boundaries = self.detect_sentence_boundaries(text)
        paragraph_boundaries = self.detect_paragraph_boundaries(text)
        section_boundaries = self.detect_section_boundaries(text, doc_type)
        
        # Combine and sort all boundaries
        all_boundaries = sorted(set(sentence_boundaries + paragraph_boundaries + section_boundaries))
        
        split_points = []
        current_pos = 0
        
        while current_pos < len(text):
            target_end = current_pos + target_size
            
            if target_end >= len(text):
                # Last chunk
                break
            
            # Find the best boundary near the target size
            best_boundary = self._find_best_boundary(
                all_boundaries, current_pos, target_end, 
                paragraph_boundaries, section_boundaries
            )
            
            if best_boundary and best_boundary > current_pos:
                split_points.append(best_boundary)
                current_pos = best_boundary
            else:
                # No good boundary found, use target size
                split_points.append(target_end)
                current_pos = target_end
        
        return split_points
    
    def _find_best_boundary(self, all_boundaries: List[int], start: int, target_end: int,
                           paragraph_boundaries: List[int], section_boundaries: List[int]) -> Optional[int]:
        """Find the best boundary point near the target end position"""
        # Look for boundaries within a reasonable range of the target
        search_range = min(200, target_end - start // 4)  # 25% of chunk size or 200 chars
        min_pos = max(start + 100, target_end - search_range)  # Don't make chunks too small
        max_pos = target_end + search_range
        
        candidates = [b for b in all_boundaries if min_pos <= b <= max_pos]
        
        if not candidates:
            return None
        
        # Prioritize different boundary types
        for boundary in candidates:
            if boundary in section_boundaries:
                return boundary  # Highest priority: section boundaries
        
        for boundary in candidates:
            if boundary in paragraph_boundaries:
                return boundary  # Medium priority: paragraph boundaries
        
        # Return the closest sentence boundary to target
        return min(candidates, key=lambda x: abs(x - target_end))


class DocumentTypeDetector:
    """Detects document type for adaptive chunking strategies"""
    
    @staticmethod
    def detect_type(content: str, metadata: Dict[str, Any]) -> DocumentType:
        """Detect document type from content and metadata"""
        # Check metadata first
        source = metadata.get('source', '').lower()
        
        if any(ext in source for ext in ['.pdf']):
            return DocumentType.PDF
        elif any(ext in source for ext in ['.html', '.htm']):
            return DocumentType.HTML
        elif any(ext in source for ext in ['.py', '.js', '.java', '.cpp', '.c']):
            return DocumentType.CODE
        elif any(ext in source for ext in ['.csv']):
            return DocumentType.CSV
        elif any(ext in source for ext in ['.json']):
            return DocumentType.JSON
        elif any(ext in source for ext in ['.md', '.markdown']):
            return DocumentType.MARKDOWN
        
        # Analyze content patterns
        content_lower = content.lower()
        
        # HTML detection
        if re.search(r'<[^>]+>', content) and any(tag in content_lower for tag in ['<html', '<body', '<div']):
            return DocumentType.HTML
        
        # Code detection
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+\s*[:{]',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'#include\s*<',  # C/C++ includes
        ]
        if any(re.search(pattern, content) for pattern in code_indicators):
            return DocumentType.CODE
        
        # Markdown detection
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE) or '```' in content:
            return DocumentType.MARKDOWN
        
        # JSON detection
        if content.strip().startswith(('{', '[')) and content.strip().endswith(('}', ']')):
            try:
                import json
                json.loads(content)
                return DocumentType.JSON
            except:
                pass
        
        return DocumentType.TEXT


class AdaptiveChunker:
    """Main adaptive chunking class with semantic boundary detection"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.boundary_detector = SemanticBoundaryDetector()
        self.type_detector = DocumentTypeDetector()
        self.overlap_manager = IntelligentOverlapManager(config)
        
    def chunk_documents(self, documents: List[Document], 
                       strategy: ChunkType = ChunkType.ADAPTIVE) -> List[DocumentChunk]:
        """Chunk documents using the specified strategy"""
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            doc_type = self.type_detector.detect_type(document.page_content, document.metadata)
            
            if strategy == ChunkType.ADAPTIVE:
                chunks = self._adaptive_chunk(document, doc_type, doc_idx)
            elif strategy == ChunkType.SEMANTIC:
                chunks = self._semantic_chunk(document, doc_type, doc_idx)
            elif strategy == ChunkType.HIERARCHICAL:
                chunks = self._hierarchical_chunk(document, doc_type, doc_idx)
            else:  # FIXED
                chunks = self._fixed_chunk(document, doc_type, doc_idx)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _adaptive_chunk(self, document: Document, doc_type: DocumentType, doc_idx: int) -> List[DocumentChunk]:
        """Adaptive chunking that adjusts based on document type and content"""
        content = document.page_content
        
        # Adjust chunk size based on document type
        chunk_size = self._get_adaptive_chunk_size(doc_type, len(content))
        
        # Get optimal split points
        split_points = self.boundary_detector.get_optimal_split_points(
            content, chunk_size, doc_type
        )
        
        return self._create_chunks_from_splits(
            document, split_points, ChunkType.ADAPTIVE, doc_idx
        )
    
    def _semantic_chunk(self, document: Document, doc_type: DocumentType, doc_idx: int) -> List[DocumentChunk]:
        """Semantic chunking that prioritizes meaning preservation"""
        content = document.page_content
        
        # Use smaller chunks for better semantic coherence
        chunk_size = min(self.config.base_chunk_size, 800)
        
        split_points = self.boundary_detector.get_optimal_split_points(
            content, chunk_size, doc_type
        )
        
        return self._create_chunks_from_splits(
            document, split_points, ChunkType.SEMANTIC, doc_idx
        )
    
    def _hierarchical_chunk(self, document: Document, doc_type: DocumentType, doc_idx: int) -> List[DocumentChunk]:
        """Hierarchical chunking for structured documents"""
        content = document.page_content
        
        if doc_type == DocumentType.HTML:
            return self._hierarchical_chunk_html(document, doc_idx)
        elif doc_type == DocumentType.PDF:
            return self._hierarchical_chunk_pdf(document, doc_idx)
        elif doc_type == DocumentType.MARKDOWN:
            return self._hierarchical_chunk_markdown(document, doc_idx)
        elif doc_type == DocumentType.CODE:
            return self._hierarchical_chunk_code(document, doc_idx)
        else:
            # Fall back to adaptive chunking for unsupported types
            return self._adaptive_chunk(document, doc_type, doc_idx)
    
    def _fixed_chunk(self, document: Document, doc_type: DocumentType, doc_idx: int) -> List[DocumentChunk]:
        """Fixed-size chunking (fallback method)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.base_chunk_size,
            chunk_overlap=int(self.config.base_chunk_size * self.config.overlap_ratio)
        )
        
        langchain_chunks = splitter.split_documents([document])
        
        chunks = []
        for i, lc_chunk in enumerate(langchain_chunks):
            chunk = DocumentChunk(
                content=lc_chunk.page_content,
                metadata=lc_chunk.metadata,
                chunk_type=ChunkType.FIXED,
                semantic_boundaries=[],
                parent_document_id=f"doc_{doc_idx}",
                chunk_index=i
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_adaptive_chunk_size(self, doc_type: DocumentType, content_length: int) -> int:
        """Calculate adaptive chunk size based on document type and content"""
        base_size = self.config.base_chunk_size
        
        # Type-specific adjustments
        type_multipliers = {
            DocumentType.CODE: 1.2,  # Larger chunks for code to preserve context
            DocumentType.JSON: 0.8,  # Smaller chunks for structured data
            DocumentType.CSV: 0.6,   # Very small chunks for tabular data
            DocumentType.HTML: 0.9,  # Slightly smaller for HTML
            DocumentType.MARKDOWN: 1.1,  # Slightly larger for markdown
            DocumentType.PDF: 1.0,   # Standard size
            DocumentType.TEXT: 1.0,  # Standard size
        }
        
        multiplier = type_multipliers.get(doc_type, 1.0)
        adjusted_size = int(base_size * multiplier)
        
        # Apply document-specific settings if available
        if doc_type.value in self.config.document_type_specific:
            type_config = self.config.document_type_specific[doc_type.value]
            if 'chunk_size' in type_config:
                adjusted_size = type_config['chunk_size']
        
        # Ensure within bounds
        return max(self.config.min_chunk_size, 
                  min(adjusted_size, self.config.max_chunk_size))
    
    def _create_chunks_from_splits(self, document: Document, split_points: List[int], 
                                  chunk_type: ChunkType, doc_idx: int) -> List[DocumentChunk]:
        """Create DocumentChunk objects from split points with intelligent overlap"""
        content = document.page_content
        chunks = []
        doc_type = self.type_detector.detect_type(content, document.metadata)
        
        start = 0
        for i, end in enumerate(split_points + [len(content)]):
            if start >= len(content):
                break
            
            chunk_content = content[start:end].strip()
            if not chunk_content:
                start = end
                continue
            
            # Calculate intelligent overlap for next chunk
            overlap_info = None
            if i < len(split_points):  # Not the last chunk
                next_chunk_start = end
                next_chunk_end = split_points[i + 1] if i + 1 < len(split_points) else len(content)
                next_chunk_content = content[next_chunk_start:next_chunk_end]
                
                # Use intelligent overlap manager
                overlap_strategy = self.overlap_manager.calculate_optimal_overlap(
                    chunk_content, next_chunk_content, doc_type
                )
                
                overlap_info = self.overlap_manager.create_context_preserving_overlap(
                    chunk_content, next_chunk_content, overlap_strategy
                )
            
            # Detect semantic boundaries within the chunk
            chunk_boundaries = []
            if self.config.preserve_sentences:
                sentence_boundaries = self.boundary_detector.detect_sentence_boundaries(chunk_content)
                chunk_boundaries.extend(sentence_boundaries)
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=document.metadata.copy(),
                chunk_type=chunk_type,
                semantic_boundaries=chunk_boundaries,
                parent_document_id=f"doc_{doc_idx}",
                chunk_index=i,
                overlap_info=overlap_info
            )
            
            chunks.append(chunk)
            
            # Adjust start position for overlap
            if overlap_info and overlap_info.end_overlap > 0:
                start = end - overlap_info.end_overlap
            else:
                start = end
        
        return chunks
    
    def _is_semantic_boundary(self, content: str, position: int) -> bool:
        """Check if a position represents a semantic boundary"""
        if position <= 0 or position >= len(content):
            return False
        
        # Check for sentence endings
        before_char = content[position - 1]
        after_char = content[position] if position < len(content) else ' '
        
        # Sentence boundary indicators
        if before_char in '.!?' and after_char.isspace():
            return True
        
        # Paragraph boundary indicators
        if before_char == '\n' and after_char == '\n':
            return True
        
        return False
    
    def optimize_chunk_parameters(self, document_type: str, content_analysis: Dict[str, Any]) -> ChunkingConfig:
        """Optimize chunking parameters based on document analysis"""
        # Create a copy of current config
        optimized_config = ChunkingConfig(
            base_chunk_size=self.config.base_chunk_size,
            overlap_ratio=self.config.overlap_ratio,
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            preserve_sentences=self.config.preserve_sentences,
            preserve_paragraphs=self.config.preserve_paragraphs,
            document_type_specific=self.config.document_type_specific.copy()
        )
        
        # Analyze content characteristics
        avg_sentence_length = content_analysis.get('avg_sentence_length', 100)
        avg_paragraph_length = content_analysis.get('avg_paragraph_length', 500)
        content_density = content_analysis.get('content_density', 1.0)  # chars per semantic unit
        
        # Adjust based on content characteristics
        if avg_sentence_length > 150:  # Long sentences
            optimized_config.base_chunk_size = min(optimized_config.base_chunk_size * 1.2, 
                                                  optimized_config.max_chunk_size)
        elif avg_sentence_length < 50:  # Short sentences
            optimized_config.base_chunk_size = max(optimized_config.base_chunk_size * 0.8, 
                                                  optimized_config.min_chunk_size)
        
        # Adjust overlap based on content density
        if content_density > 1.5:  # Dense content
            optimized_config.overlap_ratio = min(optimized_config.overlap_ratio * 1.2, 0.3)
        elif content_density < 0.5:  # Sparse content
            optimized_config.overlap_ratio = max(optimized_config.overlap_ratio * 0.8, 0.05)
        
        return optimized_config
    
    def optimize_overlap_for_content_type(self, doc_type: DocumentType) -> float:
        """Optimize overlap ratio based on document type"""
        base_ratio = self.config.overlap_ratio
        
        # Content type specific optimizations
        if doc_type == DocumentType.CODE:
            # Code needs more overlap to preserve function/class context
            return min(base_ratio * 1.5, 0.3)
        elif doc_type == DocumentType.MARKDOWN:
            # Markdown benefits from more overlap to preserve section context
            return min(base_ratio * 1.2, 0.25)
        elif doc_type == DocumentType.HTML:
            # HTML needs moderate overlap to preserve element context
            return min(base_ratio * 1.1, 0.2)
        elif doc_type in [DocumentType.JSON, DocumentType.CSV]:
            # Structured data needs minimal overlap
            return max(base_ratio * 0.5, 0.05)
        elif doc_type == DocumentType.PDF:
            # PDFs often have complex formatting, use standard overlap
            return base_ratio
        else:  # TEXT and others
            return base_ratio
    
    def _hierarchical_chunk_html(self, document: Document, doc_idx: int) -> List[DocumentChunk]:
        """Hierarchical chunking for HTML documents"""
        content = document.page_content
        chunks = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Find structural elements in order of importance
            structural_elements = [
                ('article', 'article'),
                ('section', 'section'),
                ('div', 'content-div'),
                ('h1', 'heading-1'),
                ('h2', 'heading-2'),
                ('h3', 'heading-3'),
                ('h4', 'heading-4'),
                ('h5', 'heading-5'),
                ('h6', 'heading-6'),
                ('p', 'paragraph'),
            ]
            
            chunk_index = 0
            for tag_name, chunk_category in structural_elements:
                elements = soup.find_all(tag_name)
                
                for element in elements:
                    element_text = element.get_text(separator=' ', strip=True)
                    
                    if len(element_text) < 50:  # Skip very short elements
                        continue
                    
                    # If element is too large, split it further
                    if len(element_text) > self.config.max_chunk_size:
                        sub_chunks = self._split_large_element(element_text, chunk_category, doc_idx, chunk_index)
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                    else:
                        # Create chunk with hierarchical metadata
                        chunk_metadata = document.metadata.copy()
                        chunk_metadata.update({
                            'html_tag': tag_name,
                            'chunk_category': chunk_category,
                            'element_id': element.get('id', ''),
                            'element_class': ' '.join(element.get('class', [])),
                        })
                        
                        chunk = DocumentChunk(
                            content=element_text,
                            metadata=chunk_metadata,
                            chunk_type=ChunkType.HIERARCHICAL,
                            semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(element_text),
                            parent_document_id=f"doc_{doc_idx}",
                            chunk_index=chunk_index
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Remove processed element to avoid duplication
                    element.decompose()
            
            # Process any remaining text
            remaining_text = soup.get_text(separator=' ', strip=True)
            if len(remaining_text) > 100:
                remaining_chunks = self._create_chunks_from_text(
                    remaining_text, document.metadata, ChunkType.HIERARCHICAL, doc_idx, chunk_index
                )
                chunks.extend(remaining_chunks)
            
        except Exception as e:
            # Fall back to adaptive chunking if HTML parsing fails
            print(f"HTML parsing failed: {e}. Falling back to adaptive chunking.")
            return self._adaptive_chunk(document, DocumentType.HTML, doc_idx)
        
        return chunks
    
    def _hierarchical_chunk_pdf(self, document: Document, doc_idx: int) -> List[DocumentChunk]:
        """Hierarchical chunking for PDF documents"""
        content = document.page_content
        chunks = []
        
        # PDF structure detection patterns
        patterns = {
            'chapter': r'^(Chapter\s+\d+|CHAPTER\s+\d+).*$',
            'section': r'^(\d+\.?\s+[A-Z][^.]*|[A-Z][A-Z\s]{10,})$',
            'subsection': r'^(\d+\.\d+\.?\s+[A-Z][^.]*|\s*[A-Z][a-z\s]{5,}:)$',
            'paragraph': r'^[A-Z][^.]*\.$',
        }
        
        lines = content.split('\n')
        current_chunk = []
        chunk_index = 0
        current_category = 'content'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any structural pattern
            matched_category = None
            for category, pattern in patterns.items():
                if re.match(pattern, line, re.MULTILINE):
                    matched_category = category
                    break
            
            # If we found a new structural element and have accumulated content
            if matched_category and current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if len(chunk_text) > 50:
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        'pdf_structure': current_category,
                        'chunk_category': current_category,
                    })
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        chunk_type=ChunkType.HIERARCHICAL,
                        semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(chunk_text),
                        parent_document_id=f"doc_{doc_idx}",
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = [line]
                current_category = matched_category
            else:
                current_chunk.append(line)
            
            # If chunk gets too large, split it
            if len('\n'.join(current_chunk)) > self.config.max_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                sub_chunks = self._split_large_element(chunk_text, current_category, doc_idx, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                current_chunk = []
        
        # Process remaining content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) > 50:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'pdf_structure': current_category,
                    'chunk_category': current_category,
                })
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_type=ChunkType.HIERARCHICAL,
                    semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(chunk_text),
                    parent_document_id=f"doc_{doc_idx}",
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
        
        return chunks if chunks else self._adaptive_chunk(document, DocumentType.PDF, doc_idx)
    
    def _hierarchical_chunk_markdown(self, document: Document, doc_idx: int) -> List[DocumentChunk]:
        """Hierarchical chunking for Markdown documents"""
        content = document.page_content
        chunks = []
        
        # Split by headers while preserving hierarchy
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = []
        current_header_level = 0
        current_header_text = ""
        chunk_index = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Process previous section if it exists
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if len(section_text) > 50:
                        chunk_metadata = document.metadata.copy()
                        chunk_metadata.update({
                            'markdown_header_level': current_header_level,
                            'markdown_header_text': current_header_text,
                            'chunk_category': f'header-{current_header_level}',
                        })
                        
                        if len(section_text) > self.config.max_chunk_size:
                            sub_chunks = self._split_large_element(
                                section_text, f'header-{current_header_level}', doc_idx, chunk_index
                            )
                            chunks.extend(sub_chunks)
                            chunk_index += len(sub_chunks)
                        else:
                            chunk = DocumentChunk(
                                content=section_text,
                                metadata=chunk_metadata,
                                chunk_type=ChunkType.HIERARCHICAL,
                                semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(section_text),
                                parent_document_id=f"doc_{doc_idx}",
                                chunk_index=chunk_index
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                
                # Start new section
                current_header_level = len(header_match.group(1))
                current_header_text = header_match.group(2)
                current_section = [line]
            else:
                current_section.append(line)
        
        # Process final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if len(section_text) > 50:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'markdown_header_level': current_header_level,
                    'markdown_header_text': current_header_text,
                    'chunk_category': f'header-{current_header_level}',
                })
                
                if len(section_text) > self.config.max_chunk_size:
                    sub_chunks = self._split_large_element(
                        section_text, f'header-{current_header_level}', doc_idx, chunk_index
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunk = DocumentChunk(
                        content=section_text,
                        metadata=chunk_metadata,
                        chunk_type=ChunkType.HIERARCHICAL,
                        semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(section_text),
                        parent_document_id=f"doc_{doc_idx}",
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
        
        return chunks if chunks else self._adaptive_chunk(document, DocumentType.MARKDOWN, doc_idx)
    
    def _hierarchical_chunk_code(self, document: Document, doc_idx: int) -> List[DocumentChunk]:
        """Hierarchical chunking for code documents"""
        content = document.page_content
        chunks = []
        
        # Code structure patterns
        patterns = {
            'class': r'^class\s+\w+.*?:',
            'function': r'^def\s+\w+.*?:',
            'method': r'^\s+def\s+\w+.*?:',
            'js_function': r'^function\s+\w+.*?\{',
            'js_class': r'^class\s+\w+.*?\{',
            'comment_block': r'^(\/\*[\s\S]*?\*\/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
        }
        
        lines = content.split('\n')
        current_block = []
        current_category = 'code'
        chunk_index = 0
        indent_level = 0
        
        for i, line in enumerate(lines):
            # Detect indentation level
            line_indent = len(line) - len(line.lstrip())
            
            # Check for structural patterns
            matched_category = None
            for category, pattern in patterns.items():
                if re.match(pattern, line.strip()):
                    matched_category = category
                    break
            
            # If we found a new structure at same or lower indentation level
            if matched_category and (line_indent <= indent_level or not current_block):
                # Process previous block
                if current_block:
                    block_text = '\n'.join(current_block).strip()
                    if len(block_text) > 30:  # Smaller threshold for code
                        chunk_metadata = document.metadata.copy()
                        chunk_metadata.update({
                            'code_structure': current_category,
                            'chunk_category': current_category,
                            'indent_level': indent_level,
                        })
                        
                        chunk = DocumentChunk(
                            content=block_text,
                            metadata=chunk_metadata,
                            chunk_type=ChunkType.HIERARCHICAL,
                            semantic_boundaries=[],  # Code doesn't have sentence boundaries
                            parent_document_id=f"doc_{doc_idx}",
                            chunk_index=chunk_index
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                
                current_block = [line]
                current_category = matched_category
                indent_level = line_indent
            else:
                current_block.append(line)
            
            # If block gets too large, split it
            if len('\n'.join(current_block)) > self.config.max_chunk_size:
                block_text = '\n'.join(current_block).strip()
                if len(block_text) > 30:
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        'code_structure': current_category,
                        'chunk_category': current_category,
                        'indent_level': indent_level,
                    })
                    
                    chunk = DocumentChunk(
                        content=block_text,
                        metadata=chunk_metadata,
                        chunk_type=ChunkType.HIERARCHICAL,
                        semantic_boundaries=[],
                        parent_document_id=f"doc_{doc_idx}",
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_block = []
        
        # Process remaining block
        if current_block:
            block_text = '\n'.join(current_block).strip()
            if len(block_text) > 30:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'code_structure': current_category,
                    'chunk_category': current_category,
                    'indent_level': indent_level,
                })
                
                chunk = DocumentChunk(
                    content=block_text,
                    metadata=chunk_metadata,
                    chunk_type=ChunkType.HIERARCHICAL,
                    semantic_boundaries=[],
                    parent_document_id=f"doc_{doc_idx}",
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
        
        return chunks if chunks else self._adaptive_chunk(document, DocumentType.CODE, doc_idx)
    
    def _split_large_element(self, text: str, category: str, doc_idx: int, start_index: int) -> List[DocumentChunk]:
        """Split large elements that exceed max chunk size"""
        chunks = []
        
        # Use semantic splitting for large elements
        split_points = self.boundary_detector.get_optimal_split_points(
            text, self.config.base_chunk_size, DocumentType.TEXT
        )
        
        start = 0
        for i, end in enumerate(split_points + [len(text)]):
            if start >= len(text):
                break
            
            chunk_text = text[start:end].strip()
            if not chunk_text:
                start = end
                continue
            
            chunk_metadata = {
                'chunk_category': category,
                'is_split_element': True,
                'split_part': i + 1,
            }
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_type=ChunkType.HIERARCHICAL,
                semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(chunk_text),
                parent_document_id=f"doc_{doc_idx}",
                chunk_index=start_index + i
            )
            chunks.append(chunk)
            start = end
        
        return chunks
    
    def _create_chunks_from_text(self, text: str, base_metadata: Dict[str, Any], 
                                chunk_type: ChunkType, doc_idx: int, start_index: int) -> List[DocumentChunk]:
        """Create chunks from plain text with base metadata"""
        chunks = []
        
        split_points = self.boundary_detector.get_optimal_split_points(
            text, self.config.base_chunk_size, DocumentType.TEXT
        )
        
        start = 0
        for i, end in enumerate(split_points + [len(text)]):
            if start >= len(text):
                break
            
            chunk_text = text[start:end].strip()
            if not chunk_text:
                start = end
                continue
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_category': 'remaining_content',
            })
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_type=chunk_type,
                semantic_boundaries=self.boundary_detector.detect_sentence_boundaries(chunk_text),
                parent_document_id=f"doc_{doc_idx}",
                chunk_index=start_index + i
            )
            chunks.append(chunk)
            start = end
        
        return chunks